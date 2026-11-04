"""用 MaskablePPO 训练调度策略。

用法（在项目根目录）：
    python -m cqf.rl.train --cycle 50 --out results/ppo_T50
    python -m cqf.rl.train --cycle 50 --train-tc 2 --val-tc 2 --timesteps 2000 --out results/ppo_smoke   # 冒烟

数据划分：训练 TC5、TC9、TC13；验证 TC11（早停 / 选 checkpoint，不参与梯度）；
测试 TC18、TC25、TC26 只在最终评估时用 run_cqf.py --algo ppo 跑；TC1、TC2、TC3 用于冒烟。

输出目录：
    model.zip        训练结束时的策略
    best_model.zip   验证用例上回报最高的策略（MaskableEvalCallback 保存）
    config.json      建模参数（T、k_paths、n_offsets、bytes_per_cycle 等），run_cqf.py --algo ppo 加载时逐项核对
    evaluations.npz  每次验证的回报
"""

import argparse
import functools
import json
import os
import time

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from .env import CqfEnv, make_models, default_n_offsets
from .gat import GatExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'TCs_for_TSN', 'SA_TCs'))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='train a MaskablePPO policy for CQF routing + offset scheduling')
    p.add_argument('--train-tc', type=int, nargs='+', default=[5, 9, 13], help='training test cases')
    p.add_argument('--val-tc', type=int, nargs='+', default=[11], help='validation test cases (checkpoint selection)')
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--cycle', type=int, required=True, help='cycle length T in microseconds')
    p.add_argument('--k-paths', type=int, default=5)
    p.add_argument('--max-offset', type=int, default=None)
    p.add_argument('--bw-fraction', type=float, default=0.9)
    p.add_argument('--count-source-link', action='store_true')
    p.add_argument('--order', default='input', choices=['input', 'deadline', 'random'], help='flow order in every episode')
    p.add_argument('--tc-sampling', default='cycle', choices=['cycle', 'random'])
    p.add_argument('--timesteps', type=int, default=200000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-envs', type=int, default=8, help='parallel training environments (1 = no parallelism)')
    p.add_argument('--vec', default='subproc', choices=['subproc', 'dummy'],
                   help='subproc: one process per environment; dummy: all in this process (tests / debugging)')
    p.add_argument('--n-steps', type=int, default=2048, help='rollout length per update, summed over all environments')
    p.add_argument('--n-epochs', type=int, default=10, help='gradient epochs per update')
    p.add_argument('--batch-size', type=int, default=64, help='sb3 default 64; the comparison runs so far used 256')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-linear', action='store_true', help='decay the learning rate linearly from --lr to 0')
    p.add_argument('--ent-coef', type=float, default=0.0, help='entropy bonus; sb3 default 0, the comparison runs so far used 0.001')
    p.add_argument('--gamma', type=float, default=0.99,
                   help='discount; sb3 default 0.99, the comparison runs so far used 0.999 (episodes are thousands of steps long)')
    p.add_argument('--reject-penalty', type=float, default=1.0,
                   help='reward penalty per flow auto-rejected after a decision (0 = plain +1 per accepted flow)')
    p.add_argument('--net', type=int, nargs='+', default=[64, 64],
                   help='hidden layer sizes of the MLP head; sb3 default 64 64, the comparison runs so far used 256 256')
    p.add_argument('--policy', default='mlp', choices=['mlp', 'gat'],
                   help='mlp: vector observation; gat: graph observation + GAT encoder in front of the MLP head')
    p.add_argument('--gat-dim', type=int, default=32, help='GAT embedding size')
    p.add_argument('--gat-heads', type=int, default=4)
    p.add_argument('--gat-layers', type=int, default=2)
    p.add_argument('--max-nodes', type=int, default=128, help='graph observation: node slots (TC25/26 need 116)')
    p.add_argument('--max-links', type=int, default=256, help='graph observation: directed link slots (TC25/26 need 244)')
    p.add_argument('--max-path-links', type=int, default=20, help='graph observation: links per candidate path (data: <= 17)')
    p.add_argument('--link-bins', type=int, default=10, help='graph observation: bins of the per-link utilisation profile')
    p.add_argument('--eval-freq', type=int, default=20000, help='validate every N steps')
    p.add_argument('--device', default='auto', help='cpu / cuda / auto')
    return p.parse_args(argv)


def make_train_env(index, data_dir, tc_numbers, model_kw, n_offsets, order, seed, tc_sampling, reject_penalty,
                   obs_mode, graph_kwargs):
    """构造第 index 个训练环境。SubprocVecEnv 下它在子进程里运行，所以模型在这里（而不是主进程）读取。
    每个环境自己的 seed 与起始用例都不同，8 个环境轮流覆盖 3 个训练用例。"""
    models = make_models(data_dir, tc_numbers, **model_kw)
    return Monitor(CqfEnv(models, n_offsets, order=order, seed=seed + index, tc_sampling=tc_sampling,
                          reject_penalty=reject_penalty, obs_mode=obs_mode, graph_kwargs=graph_kwargs,
                          episode_offset=index))


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    kw = dict(cycle_time_us=args.cycle, k_paths=args.k_paths, max_offset=args.max_offset,
              bw_fraction=args.bw_fraction, count_source_link=args.count_source_link)
    train_models = make_models(args.data_dir, args.train_tc, **kw)
    val_models = make_models(args.data_dir, args.val_tc, **kw)
    n_offsets = int(default_n_offsets(train_models + val_models, args.max_offset))

    graph_kwargs = dict(max_nodes=args.max_nodes, max_links=args.max_links, max_path_links=args.max_path_links,
                        link_bins=args.link_bins)
    obs_mode = 'graph' if args.policy == 'gat' else 'flat'
    env_fns = [functools.partial(make_train_env, i, args.data_dir, args.train_tc, kw, n_offsets, args.order, args.seed,
                                 args.tc_sampling, args.reject_penalty, obs_mode, graph_kwargs)
               for i in range(args.n_envs)]
    env = (SubprocVecEnv if args.vec == 'subproc' and args.n_envs > 1 else DummyVecEnv)(env_fns)
    val_env = Monitor(CqfEnv(val_models, n_offsets, order=args.order, seed=args.seed + 1000,
                             reject_penalty=args.reject_penalty, obs_mode=obs_mode, graph_kwargs=graph_kwargs))
    n_steps_per_env = max(1, args.n_steps // args.n_envs)

    config = dict(cycle_time_us=args.cycle, k_paths=args.k_paths, max_offset=args.max_offset,
                  n_offsets=n_offsets, bw_fraction=args.bw_fraction, count_source_link=args.count_source_link,
                  bytes_per_cycle=train_models[0].bytes_per_cycle, order=args.order,
                  train_tc=args.train_tc, val_tc=args.val_tc, timesteps=args.timesteps, seed=args.seed,
                  n_steps=args.n_steps, n_envs=args.n_envs, n_steps_per_env=n_steps_per_env, n_epochs=args.n_epochs,
                  batch_size=args.batch_size, lr=args.lr, ent_coef=args.ent_coef, net=args.net,
                  gamma=args.gamma, reject_penalty=args.reject_penalty, lr_linear=args.lr_linear,
                  policy=args.policy, gat=dict(dim=args.gat_dim, heads=args.gat_heads, layers=args.gat_layers),
                  graph=graph_kwargs, n_actions=int(env.action_space.n),
                  obs_dim=int(val_env.get_wrapper_attr('features')[0].obs_dim))
    with open(os.path.join(args.out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # sb3 的 learning_rate 可以是函数：参数是"剩余训练进度"（1 -> 0）
    learning_rate = (lambda remaining: args.lr * remaining) if args.lr_linear else args.lr
    policy_kwargs = dict(net_arch=list(args.net))
    if args.policy == 'gat':
        policy_kwargs.update(features_extractor_class=GatExtractor,
                             features_extractor_kwargs=dict(dim=args.gat_dim, heads=args.gat_heads, layers=args.gat_layers))
    algo = MaskablePPO('MultiInputPolicy' if args.policy == 'gat' else 'MlpPolicy', env, seed=args.seed,
                       n_steps=n_steps_per_env, batch_size=args.batch_size, n_epochs=args.n_epochs,
                       learning_rate=learning_rate, ent_coef=args.ent_coef, gamma=args.gamma,
                       policy_kwargs=policy_kwargs, verbose=1, device=args.device)
    print('device: %s, %d environment(s), %d steps per environment per update' % (algo.device, args.n_envs, n_steps_per_env))
    # 回调的 eval_freq 按"每个环境走一步"计数，n 个环境并行时一次计 n 步，所以除以 n_envs 才是按总步数验证
    callback = MaskableEvalCallback(val_env, eval_freq=max(1, args.eval_freq // args.n_envs), n_eval_episodes=len(val_models),
                                    deterministic=True, best_model_save_path=args.out, log_path=args.out, verbose=1)
    t0 = time.time()
    algo.learn(total_timesteps=args.timesteps, callback=callback)
    algo.save(os.path.join(args.out, 'model.zip'))
    env.close()
    print('training finished in %.1fs; files in %s' % (time.time() - t0, args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
