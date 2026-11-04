"""RL 环境与特征的测试。

核心要证明两件事：
1. 向量化掩码（cqf.rl.features）与内核 feasible_actions 在每一步都完全一致（玩具拓扑 + 导师数据 TC2）；
2. 环境里按同一随机数发生器选动作，与 RandomScheduler 的结果逐流相同，说明环境和基线走的是同一条路。
再加 MaskablePPO 冒烟：学几十步、保存、用 PpoScheduler 走 run_cqf 的流程并通过两道验证。
"""

import json
import os
import random
import tempfile
import unittest

try:
    import numpy as np
    import gymnasium
    from cqf.rl.features import ActionFeatures
    from cqf.rl.env import CqfEnv, make_models, default_n_offsets
    HAVE_RL = True
except ImportError:               # 没装 numpy / gymnasium 时跳过本文件
    HAVE_RL = False

try:
    import sb3_contrib
    HAVE_SB3 = True
except ImportError:
    HAVE_SB3 = False

from cqf.algorithms import RandomScheduler, GreedyScheduler, order_streams
from cqf.validate import simulate_and_validate
from tests.test_model import make_model, flow, T

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'TCs_for_TSN', 'SA_TCs'))
HAVE_DATA = os.path.exists(os.path.join(DATA_DIR, 'TC2_flows.txt'))

# 玩具流：周期 1000/2500/5000 -> 10/25/50 个 cycle，大帧长逼出容量冲突，小 deadline 逼出 deadline 冲突
TOY_FLOWS = [
    flow(0, 'esA', 'esB', 1000, 800, 3000),
    flow(1, 'esC', 'esB', 1000, 600, 3000),
    flow(2, 'esA', 'esB', 2500, 2500, 5000),
    flow(3, 'esC', 'esB', 5000, 400, 5000),
    flow(4, 'esA', 'esB', 1000, 1000, 6000),
    flow(5, 'esC', 'esB', 2500, 300, 1000),   # deadline 300us = 3 cycle，最短路要 1+2=3 -> 只有 offset 0
    flow(6, 'esA', 'esB', 5000, 5000, 11000),
    flow(7, 'esC', 'esB', 1000, 900, 11000),  # T=100 时容量 11250 字节：单独能放，与 6 号同槽就放不下
]


def kernel_mask(model, stream, n_offsets):
    feasible, _ = model.feasible_actions(stream)
    return {(p, o) for p, o in feasible if o < n_offsets}


def features_mask(feat, stream):
    mask, _ = feat.mask_and_obs(stream)
    return {divmod(int(a), feat.n_offsets) for a in np.flatnonzero(mask)}


@unittest.skipUnless(HAVE_RL, 'numpy / gymnasium not installed')
class TestFeaturesAgainstKernel(unittest.TestCase):

    def _walk(self, **model_kwargs):
        """用贪心一步步提交，每一步前核对掩码与观测维度。"""
        model, streams = make_model(TOY_FLOWS, **model_kwargs)
        n_offsets = max(model.period_cycles.values())
        feat = ActionFeatures(model, n_offsets)
        greedy = GreedyScheduler()
        decided = 0
        for s in streams:
            self.assertEqual(kernel_mask(model, s, n_offsets), features_mask(feat, s), s.name)
            mask, obs = feat.mask_and_obs(s)
            self.assertEqual(obs.shape, (feat.obs_dim,))
            self.assertTrue(np.all(obs >= 0) and np.all(obs <= 2))
            feasible, _ = model.feasible_actions(s)
            if feasible:
                p, o = greedy.choose(model, s, feasible)
                model.commit(s, p, o)
                feat.commit(s, p, o)
                decided += 1
        self.assertGreater(decided, 0)
        self.assertGreater(len(model.streams) - decided, 0, '玩具用例应当有被拒绝的流才有意义')
        # 镜像与账本一致
        rebuilt = ActionFeatures(model, n_offsets)
        self.assertTrue(np.array_equal(rebuilt.load, feat.load))
        return model

    def test_mask_matches_kernel_default(self):
        self._walk()

    def test_mask_matches_kernel_with_source_link(self):
        self._walk(count_source_link=True)

    def test_mask_matches_kernel_with_max_offset(self):
        self._walk(max_offset_cycles=3)

    def test_rejection_reason_matches_kernel(self):
        from cqf.algorithms.base import classify_rejection
        model, streams = make_model(TOY_FLOWS)
        feat = ActionFeatures(model, max(model.period_cycles.values()))
        for s in streams:
            feasible, reasons = model.feasible_actions(s)
            if not feasible:
                self.assertEqual(classify_rejection(model, s, reasons), feat.rejection_reason(s))

    @unittest.skipUnless(HAVE_DATA, 'teacher data not available')
    def test_mask_matches_kernel_on_tc2(self):
        (model,) = make_models(DATA_DIR, [2], cycle_time_us=50)
        env = CqfEnv([model], order='input', seed=0, strict=True)       # strict: 每步都和内核核对
        env.reset(seed=0)
        rng = random.Random(1)
        done = False
        while not done:
            a = rng.choice(list(np.flatnonzero(env.action_masks())))
            _, _, done, _, _ = env.step(a)
        self.assertEqual(model.verify()['ok'], True)
        self.assertTrue(simulate_and_validate(model)['ok'])
        self.assertEqual(len(model.assignments), len(env.result.scheduled))


@unittest.skipUnless(HAVE_RL, 'numpy / gymnasium not installed')
class TestEnv(unittest.TestCase):

    def test_random_policy_equals_random_scheduler(self):
        # 基线：RandomScheduler(seed=7) 在 feasible 列表上 rng.choice
        model_a, streams = make_model(TOY_FLOWS)
        res_a = RandomScheduler(seed=7).run(model_a, streams)
        # 环境：同一个 rng，在掩码为 True 的动作列表（同样按 path 主序、offset 次序排列）上 rng.choice
        model_b, _ = make_model(TOY_FLOWS)
        env = CqfEnv([model_b], order='input')
        _, info = env.reset()
        rejected_at_reset = info['rejected']
        rng = random.Random(7)
        done = env.stream is None
        total = 0.0
        while not done:
            a = rng.choice(list(np.flatnonzero(env.action_masks())))
            rejected_before = env._info()['rejected']
            _, r, done, _, info = env.step(a)
            # 奖励 = +1 − 本步之后新增的自动拒绝数
            self.assertEqual(r, 1.0 - (info['rejected'] - rejected_before))
            total += r
        self.assertEqual(total, len(env.result.scheduled) - (len(env.result.rejected) - rejected_at_reset))
        self.assertEqual(res_a.scheduled, env.result.scheduled)
        self.assertEqual(res_a.rejected, env.result.rejected)
        self.assertEqual(model_a.assignments, model_b.assignments)

    def test_episode_structure(self):
        model, _ = make_model(TOY_FLOWS)
        env = CqfEnv([model], order='input', reject_penalty=0.0)
        self.assertEqual(env.action_space.n, model.k_paths * default_n_offsets([model]))
        obs, info = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(env.action_masks().any())
        with self.assertRaises(ValueError):                 # 选被屏蔽的动作是程序错误
            env.step(int(np.flatnonzero(~env.action_masks())[0]))
        total = 0.0
        done = False
        while not done:
            a = int(np.flatnonzero(env.action_masks())[0])
            _, r, done, truncated, info = env.step(a)
            total += r
            self.assertFalse(truncated)
        self.assertEqual(total, len(model.assignments))                 # reject_penalty=0：回报 = 接纳数
        self.assertEqual(info['remaining'], 0)
        # 结束后再 step 不改变任何东西
        _, r, done, _, _ = env.step(0)
        self.assertEqual((r, done), (0.0, True))
        # 第二个 episode 从头开始
        env.reset()
        self.assertEqual(model.assignments, {})
        self.assertEqual(env.feat.load.sum(), 0)

    def test_progress_feature(self):
        # 观测第 4 维 = 当前流在处理顺序里的下标 / 总流数
        model, _ = make_model(TOY_FLOWS)
        env = CqfEnv([model], order='input')
        obs, _ = env.reset()
        n = len(model.streams)
        done = False
        while not done:
            self.assertAlmostEqual(float(obs[3]), env.pos / n, places=6)
            obs, _, done, _, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))

    def test_episode_offset_spreads_parallel_envs_over_test_cases(self):
        model_a, _ = make_model(TOY_FLOWS)
        model_b, _ = make_model(TOY_FLOWS[:4])
        env0 = CqfEnv([model_a, model_b], episode_offset=0)
        env1 = CqfEnv([model_a, model_b], episode_offset=1)
        self.assertEqual(env0.reset()[1]['tc_index'], 0)
        self.assertEqual(env1.reset()[1]['tc_index'], 1)
        self.assertEqual(env1.reset()[1]['tc_index'], 0)

    def test_n_offsets_too_small_is_rejected(self):
        model, _ = make_model(TOY_FLOWS)
        with self.assertRaises(ValueError):
            CqfEnv([model], n_offsets=10)

    def test_spaces(self):
        # 不用 sb3 的 check_env：它会不看掩码随机采样动作，而本环境对被屏蔽的动作故意报错
        model, _ = make_model(TOY_FLOWS)
        env = CqfEnv([model])
        obs, _ = env.reset(seed=0)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(env.action_masks().shape, (env.action_space.n,))
        self.assertEqual(env.action_masks().dtype, bool)
        obs, _, _, _, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        self.assertTrue(env.observation_space.contains(obs))


@unittest.skipUnless(HAVE_RL and HAVE_SB3, 'sb3_contrib not installed')
class TestGraphObsAndGat(unittest.TestCase):

    GRAPH = dict(max_nodes=8, max_links=16, max_path_links=6, link_bins=5)

    def test_graph_observation_shapes_and_contents(self):
        from cqf.rl.graph_obs import GraphObs
        model, streams = make_model(TOY_FLOWS)
        env = CqfEnv([model], order='input', obs_mode='graph', graph_kwargs=self.GRAPH)
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs), {k: (v.shape, v.dtype) for k, v in obs.items()})
        gobs = env.gobs
        self.assertEqual((gobs.n_nodes, gobs.n_links), (6, 12))                 # 6 个节点、6 条无向链路 -> 12 条有向
        self.assertEqual(obs['node_valid'].sum(), 6)
        self.assertEqual(obs['link_valid'].sum(), 12)
        s = env.stream                                                           # 第一条流 esA -> esB
        src = model.graph.vertex_by_id[s.src].device_idx
        self.assertEqual(obs['node_x'][src, 3], 1.0)
        # 路径 0 = esA,sw1,sw2,esB 有 3 条链路；路径 1 有 4 条
        self.assertEqual(obs['path_mask'].sum(axis=1).tolist(), [3.0, 4.0, 0.0, 0.0, 0.0])
        # 两条路径都经过 esA->sw1，这条链路的"候选路径数"特征 = 2/5
        first = obs['path_links'][0, 0]
        self.assertAlmostEqual(float(obs['link_x'][first, -1]), 2 / 5, places=6)
        self.assertEqual(obs['link_x'][:, :-1].sum(), 0.0)                       # 空网络：利用率全 0
        # 提交一步后，被占用链路的利用率特征 > 0，且与账本一致
        a = int(np.flatnonzero(env.action_masks())[0])
        obs, _, _, _, _ = env.step(a)
        util = env.feat.load / env.feat.cap
        self.assertAlmostEqual(float(obs['link_x'][:, gobs.link_bins].sum()), float(util.mean(axis=0).sum()), places=5)
        # 另一个 (不同大小) 的 GraphObs 形状不够时报错
        with self.assertRaises(ValueError):
            GraphObs(env.feat, max_nodes=4, max_links=16, max_path_links=6)

    def test_gat_extractor_is_permutation_invariant(self):
        import torch
        from cqf.rl.gat import GatExtractor
        model, _ = make_model(TOY_FLOWS)
        env = CqfEnv([model], order='input', obs_mode='graph', graph_kwargs=self.GRAPH)
        obs, _ = env.reset()
        env.step(int(np.flatnonzero(env.action_masks())[0]))
        obs = env.obs
        torch.manual_seed(0)
        net = GatExtractor(env.observation_space, dim=8, heads=2, layers=2).eval()

        def run(o):
            with torch.no_grad():
                return net({k: torch.as_tensor(v, dtype=torch.float32)[None] for k, v in o.items()})

        base = run(obs)
        self.assertEqual(base.shape, (1, net.features_dim))
        # 1) 打乱链路的编号顺序（path_links 相应重映射）-> 输出不变
        rng = np.random.default_rng(0)
        perm = rng.permutation(self.GRAPH['max_links'])              # 新位置 i 放原来的链路 perm[i]
        inv = np.argsort(perm)                                       # 原链路 j 现在在 inv[j]
        o2 = dict(obs)
        for key in ('link_x', 'link_src', 'link_dst', 'link_valid'):
            o2[key] = obs[key][perm]
        o2['path_links'] = inv[obs['path_links']]
        self.assertTrue(torch.allclose(base, run(o2), atol=1e-5))
        # 2) 重新编号节点 -> 输出不变
        nperm = rng.permutation(self.GRAPH['max_nodes'])
        ninv = np.argsort(nperm)
        o3 = dict(obs)
        o3['node_x'] = obs['node_x'][nperm]
        o3['node_valid'] = obs['node_valid'][nperm]
        o3['link_src'] = ninv[obs['link_src']]
        o3['link_dst'] = ninv[obs['link_dst']]
        self.assertTrue(torch.allclose(base, run(o3), atol=1e-5))
        # 3) 改一条有效链路的特征 -> 输出应当变化（说明图信息真的进入了输出）
        o4 = dict(obs)
        o4['link_x'] = obs['link_x'].copy()
        o4['link_x'][int(obs['path_links'][0, 0]), 0] = 1.0
        self.assertFalse(torch.allclose(base, run(o4), atol=1e-5))

    def test_gat_learn_save_and_schedule(self):
        from sb3_contrib import MaskablePPO
        from cqf.rl.gat import GatExtractor
        from cqf.algorithms.ppo_scheduler import PpoScheduler
        model, streams = make_model(TOY_FLOWS)
        env = CqfEnv([model], strict=True, obs_mode='graph', graph_kwargs=self.GRAPH)
        algo = MaskablePPO('MultiInputPolicy', env, n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0, device='cpu',
                           policy_kwargs=dict(net_arch=[16], features_extractor_class=GatExtractor,
                                              features_extractor_kwargs=dict(dim=8, heads=2, layers=1)))
        algo.learn(total_timesteps=64)
        with tempfile.TemporaryDirectory() as d:
            algo.save(os.path.join(d, 'model.zip'))
            with open(os.path.join(d, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump({'cycle_time_us': T, 'k_paths': model.k_paths, 'n_offsets': env.n_offsets,
                           'count_source_link': False, 'bytes_per_cycle': model.bytes_per_cycle,
                           'obs_dim': env.feat.obs_dim, 'policy': 'gat', 'graph': self.GRAPH}, f)
            fresh, streams = make_model(TOY_FLOWS)
            result = PpoScheduler(model_path=d).run(fresh, order_streams(streams, 'input'))
            self.assertEqual(len(result.scheduled), len(fresh.assignments))
            self.assertTrue(fresh.verify()['ok'])
            self.assertTrue(simulate_and_validate(fresh)['ok'])


@unittest.skipUnless(HAVE_RL and HAVE_SB3, 'sb3_contrib not installed')
class TestTrainScript(unittest.TestCase):
    """整条训练脚本：玩具数据写成 TC1 文件，2 个并行环境（dummy 模式，不开子进程），几十步，然后用 PpoScheduler 读回来。"""

    def test_train_main_mlp_and_gat_with_parallel_envs(self):
        from cqf.rl import train
        from cqf.algorithms.ppo_scheduler import PpoScheduler
        from tests.test_model import TOPO_LINES
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, 'data')
            os.makedirs(data)
            with open(os.path.join(data, 'TC1_topo.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(TOPO_LINES) + '\n')
            with open(os.path.join(data, 'TC1_flows.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(TOY_FLOWS) + '\n')
            for policy in ('mlp', 'gat'):
                out = os.path.join(d, policy)
                rc = train.main(['--cycle', str(T), '--data-dir', data, '--train-tc', '1', '--val-tc', '1', '--out', out,
                                 '--policy', policy, '--n-envs', '2', '--vec', 'dummy', '--timesteps', '64', '--n-steps', '32',
                                 '--batch-size', '16', '--n-epochs', '1', '--eval-freq', '32', '--device', 'cpu',
                                 '--max-nodes', '8', '--max-links', '16', '--max-path-links', '6', '--link-bins', '5'])
                self.assertEqual(rc, 0)
                with open(os.path.join(out, 'config.json'), encoding='utf-8') as f:
                    config = json.load(f)
                self.assertEqual((config['n_envs'], config['n_steps_per_env'], config['policy']), (2, 16, policy))
                self.assertTrue(os.path.exists(os.path.join(out, 'model.zip')))
                self.assertTrue(os.path.exists(os.path.join(out, 'best_model.zip')))
                fresh, streams = make_model(TOY_FLOWS)
                result = PpoScheduler(model_path=out).run(fresh, order_streams(streams, 'input'))
                self.assertEqual(len(result.scheduled), len(fresh.assignments))
                self.assertTrue(simulate_and_validate(fresh)['ok'])


@unittest.skipUnless(HAVE_RL and HAVE_SB3, 'sb3_contrib not installed')
class TestMaskablePpoSmoke(unittest.TestCase):

    def test_learn_save_and_schedule(self):
        from sb3_contrib import MaskablePPO
        from cqf.algorithms.ppo_scheduler import PpoScheduler
        model, streams = make_model(TOY_FLOWS)
        env = CqfEnv([model], strict=True)
        algo = MaskablePPO('MlpPolicy', env, n_steps=32, batch_size=16, n_epochs=1, seed=0, verbose=0, device='cpu')
        algo.learn(total_timesteps=64)                     # strict=True：采样过程中选了被屏蔽的动作会直接报错
        with tempfile.TemporaryDirectory() as d:
            algo.save(os.path.join(d, 'model.zip'))
            with open(os.path.join(d, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump({'cycle_time_us': T, 'k_paths': model.k_paths, 'n_offsets': env.n_offsets,
                           'count_source_link': False, 'bytes_per_cycle': model.bytes_per_cycle,
                           'obs_dim': env.observation_space.shape[0]}, f)
            fresh, streams = make_model(TOY_FLOWS)
            result = PpoScheduler(model_path=d).run(fresh, order_streams(streams, 'input'))
            self.assertEqual(len(result.scheduled), len(fresh.assignments))
            self.assertTrue(fresh.verify()['ok'])
            self.assertTrue(simulate_and_validate(fresh)['ok'])
            # 配置不一致必须报错
            wrong, streams = make_model(TOY_FLOWS, k_paths=2)
            with self.assertRaises(ValueError):
                PpoScheduler(model_path=d).run(wrong, streams)
            # 观测维度不一致（例如用旧版特征训练的模型）也必须报错
            with open(os.path.join(d, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump({'cycle_time_us': T, 'k_paths': model.k_paths, 'n_offsets': env.n_offsets,
                           'count_source_link': False, 'bytes_per_cycle': model.bytes_per_cycle,
                           'obs_dim': env.observation_space.shape[0] - 1}, f)
            fresh, streams = make_model(TOY_FLOWS)
            with self.assertRaises(ValueError):
                PpoScheduler(model_path=d).run(fresh, streams)


if __name__ == '__main__':
    unittest.main()
