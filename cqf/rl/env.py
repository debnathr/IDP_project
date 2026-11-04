"""Gymnasium 环境：把 CqfModel 包成 MaskablePPO 能用的 Env。

一个 episode：
    reset  -> 选一个测试用例（轮流或随机），清空内核资源表，按 order 排好流的顺序，停在第一条"有可行动作"的流上
    step   -> 对当前流执行动作 (route, offset)：内核 commit，前进到下一条有可行动作的流
    没有任何可行动作的流不问智能体，直接记为拒绝并跳过（与 SchedulerBase.run 的处理一致）
    所有流处理完 -> terminated

动作空间 Discrete(k_paths * n_offsets)，编码 a = path_idx * n_offsets + offset。
action_masks() 返回当前流的布尔掩码，sb3_contrib 的 MaskablePPO 通过这个方法名自动取掩码。

奖励：接纳本流 +1，再减去 reject_penalty × (本步之后、下一个决策之前被自动拒绝的流数)。
    reject_penalty=1 时 episode 总回报 = 接纳数 − 拒绝数 = 2 × 接纳数 − 总流数（reset 时就被拒绝的流不算，
    它们与策略无关），优化目标仍是"最大化接纳流数"，但每个决策立刻收到它造成的拒绝的惩罚。
    reject_penalty=0 退回到第一版"每接纳一条 +1"：那版在正式训练里价值网络的 explained_variance ≈ 0，
    因为每步奖励恒定、坏决策的后果只是 episode 早结束，被折扣抹平。

本文件不做导出：评估时用 cqf.algorithms.ppo_scheduler 走与基线相同的 run_cqf.py 流程，
这样 PPO 与随机 / 贪心的结果文件和两道验证完全同口径。
"""

import os
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..algorithms.base import ScheduleResult, order_streams
from ..data_io import load_test_case
from ..network import build_graph
from ..model import CqfModel
from .features import ActionFeatures
from .graph_obs import GraphObs


def make_models(data_dir, tc_numbers, cycle_time_us, k_paths=5, max_offset=None, bw_fraction=0.9,
                count_source_link=False):
    """为每个测试用例建一个 CqfModel（与 run_cqf.run_one 的建模参数一一对应）。"""
    models = []
    for tc in tc_numbers:
        topology, streams = load_test_case(os.path.join(data_dir, 'TC%d_topo.txt' % tc),
                                           os.path.join(data_dir, 'TC%d_flows.txt' % tc))
        if not streams:
            raise ValueError('TC%d has no flows' % tc)
        models.append(CqfModel(build_graph(topology), streams, cycle_time_us=cycle_time_us,
                               bw_fraction=bw_fraction, k_paths=k_paths, max_offset_cycles=max_offset,
                               count_source_link=count_source_link))
    return models


def default_n_offsets(models, max_offset=None):
    """动作空间里 offset 的个数：所有流里最大的周期 cycle 数（导师数据 T=50 时为 5000/50 = 100），
    有 max_offset 时再压到 max_offset + 1。"""
    n = max(max(m.period_cycles.values()) for m in models)
    if max_offset is not None:
        n = min(n, max_offset + 1)
    return n


class CqfEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, models, n_offsets=None, order='input', seed=None, tc_sampling='cycle', strict=False,
                 reject_penalty=1.0, obs_mode='flat', graph_kwargs=None, episode_offset=0):
        """
        models          CqfModel 列表，每个 episode 用其中一个
        n_offsets       动作空间 offset 个数；默认按 models 推出。必须 >= 所有流的周期 cycle 数
        order           流的处理顺序：input / deadline / random（与 run_cqf.py --order 相同）
        tc_sampling     'cycle' 轮流用每个用例；'random' 每个 episode 随机抽一个
        strict          每步都用内核 feasible_actions 核对掩码（慢，只用于测试）
        reject_penalty  每条被自动拒绝的流扣多少奖励（见文件头）
        obs_mode        'flat'：MLP 阶段的向量观测；'graph'：graph_obs.GraphObs 的字典观测（GAT 阶段）
        graph_kwargs    obs_mode='graph' 时传给 GraphObs 的形状参数（max_nodes 等）
        episode_offset  tc_sampling='cycle' 时第一个 episode 用第几个用例（并行环境各给一个不同的值，免得都从同一个用例开始）
        """
        super().__init__()
        self.models = list(models)
        if not self.models:
            raise ValueError('CqfEnv needs at least one model')
        if len({m.k_paths for m in self.models}) != 1:
            raise ValueError('all models must use the same k_paths')
        self.k_paths = self.models[0].k_paths
        self.n_offsets = int(n_offsets) if n_offsets is not None else default_n_offsets(self.models)
        max_period = max(max(m.period_cycles.values()) for m in self.models)
        if self.n_offsets < max_period and all(m.max_offset_cycles is None or m.max_offset_cycles >= self.n_offsets
                                                for m in self.models):
            raise ValueError('n_offsets=%d is smaller than the largest period (%d cycles): some kernel-feasible '
                             'offsets could not be encoded' % (self.n_offsets, max_period))
        self.features = [ActionFeatures(m, self.n_offsets) for m in self.models]
        if obs_mode not in ('flat', 'graph'):
            raise ValueError('obs_mode must be flat or graph, got %r' % obs_mode)
        self.obs_mode = obs_mode
        self.graph_obs = [GraphObs(f, **(graph_kwargs or {})) for f in self.features] if obs_mode == 'graph' else None
        self.order_mode = order
        self.tc_sampling = tc_sampling
        self.strict = strict
        self.reject_penalty = float(reject_penalty)
        self.rng = random.Random(seed)

        self.n_actions = self.k_paths * self.n_offsets
        self.action_space = spaces.Discrete(self.n_actions)
        if obs_mode == 'graph':
            self.observation_space = self.graph_obs[0].space
        else:
            self.observation_space = spaces.Box(0.0, 2.0, shape=(self.features[0].obs_dim,), dtype=np.float32)

        self._episode = int(episode_offset) - 1
        self.idx = 0
        self.model = None
        self.feat = None
        self.gobs = None
        self.queue = []
        self.pos = 0
        self.stream = None
        self.mask = np.ones(self.n_actions, dtype=bool)
        self.obs = self._empty_obs()
        self.result = ScheduleResult()

    # ------------------------------------------------------------------ Gymnasium 接口

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        self._episode += 1
        if options and 'tc' in options:
            self.idx = int(options['tc'])
        elif self.tc_sampling == 'random':
            self.idx = self.rng.randrange(len(self.models))
        else:
            self.idx = self._episode % len(self.models)
        self.model = self.models[self.idx]
        self.feat = self.features[self.idx]
        self.gobs = self.graph_obs[self.idx] if self.graph_obs else None
        self.model.reset()
        self.feat.rebuild()
        self.queue = order_streams(self.model.streams, self.order_mode, self.rng)
        self.pos = 0
        self.result = ScheduleResult()
        self._advance()
        return self.obs, self._info()

    def step(self, action):
        if self.stream is None:                                    # episode 已结束仍被 step：不做任何事
            return self.obs, 0.0, True, False, self._info()
        a = int(action)
        if not self.mask[a]:
            raise ValueError('masked action %d chosen for flow %s' % (a, self.stream.name))
        path_idx, offset = divmod(a, self.n_offsets)
        if self.strict:
            self._check_mask_against_kernel()
        self.model.commit(self.stream, path_idx, offset)          # 内核再 check 一次；不一致会抛异常
        self.feat.commit(self.stream, path_idx, offset)
        self.result.scheduled.append(self.stream.id)
        self.pos += 1
        rejected_before = len(self.result.rejected)
        self._advance()
        reward = 1.0 - self.reject_penalty * (len(self.result.rejected) - rejected_before)
        return self.obs, reward, self.stream is None, False, self._info()

    def action_masks(self):
        return self.mask

    # ------------------------------------------------------------------ 内部

    def _advance(self):
        """停在下一条有可行动作的流上；没有可行动作的流记为拒绝并跳过。"""
        while self.pos < len(self.queue):
            s = self.queue[self.pos]
            self.result.order.append(s.id)
            progress = self.pos / len(self.queue)
            mask, obs = self.feat.mask_and_obs(s, progress)
            self.result.feasible_counts[s.id] = int(mask.sum())
            if mask.any():
                if self.gobs is not None:
                    obs = self.gobs.observe(s, progress, obs)
                self.stream, self.mask, self.obs = s, mask, obs
                return
            if self.strict:
                self.stream = s
                self.mask = mask
                self._check_mask_against_kernel()
            self.result.rejected[s.id] = self.feat.rejection_reason(s)
            self.pos += 1
        self.stream = None
        self.mask = np.ones(self.n_actions, dtype=bool)            # 结束后给全 True，避免全 False 的掩码
        self.obs = self._empty_obs()

    def _empty_obs(self):
        if self.obs_mode == 'graph':
            return self.graph_obs[0].empty()
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _check_mask_against_kernel(self):
        feasible, _ = self.model.feasible_actions(self.stream)
        kernel = {(p, o) for p, o in feasible}
        mine = {divmod(int(a), self.n_offsets) for a in np.flatnonzero(self.mask)}
        if kernel != mine:
            raise AssertionError('mask differs from kernel for flow %s: only kernel %s, only features %s' % (
                self.stream.name, sorted(kernel - mine)[:5], sorted(mine - kernel)[:5]))

    def _info(self):
        return {'tc_index': self.idx, 'scheduled': len(self.result.scheduled),
                'rejected': len(self.result.rejected), 'remaining': len(self.queue) - self.pos}
