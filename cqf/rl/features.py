"""RL 用的向量化特征：动作掩码与观测。

为什么不直接用内核的 feasible_actions：
    它对每个 (route, offset) 组合逐个 check，一条流最多 k_paths × P 次；TC5 一个 episode 要上百万次 check，
    纯 Python 的速度撑不住 PPO 的采样。这里把同一套规则（offset 范围、deadline、容量）用 numpy 一次算完。

做法：
  * 维护一个与 model.usage 内容相同的二维数组 load[cycle, 链路列号]（账本镜像）；
  * 对每条流的每条候选路径，预先算好它经过的链路列号与逐跳延后量（第 i 台交换机延后 i+1 个 cycle）；
    任意 offset 下占用的槽就是 (offset + k*P + 逐跳延后) % H，k 是帧实例序号；
  * 对所有 offset 同时取"提交后最忙槽的字节数"，与容量比较得到掩码，除以容量得到观测里的负载特征。

动作编码：a = path_idx * n_offsets + offset，n_offsets 是动作空间里 offset 的个数（须 >= 所有流的周期 cycle 数，
这样内核允许的每个 offset 都能被编码）。候选路径不足 k_paths 条、offset 超出该流周期等无效动作在掩码里为 False。

真值仍然是内核：PpoScheduler 用内核给的 feasible 列表做掩码，这里的掩码只在训练环境里用；
tests/test_rl_env.py 在玩具拓扑与导师数据上逐步核对两者完全一致。
"""

import numpy as np


class ActionFeatures:
    """一个 CqfModel 的动作掩码 / 观测计算器，内部的 load 镜像要通过 commit() / rebuild() 与内核保持同步。"""

    def __init__(self, model, n_offsets):
        self.model = model
        self.n_offsets = int(n_offsets)
        self.k_paths = model.k_paths
        self.n_actions = self.k_paths * self.n_offsets
        self.H = model.hyperperiod_cycles
        self.cap = model.bytes_per_cycle

        # 有向链路 (u, v) -> 列号；两个方向是两条资源
        self.link_col = {}
        for link in model.graph.all_links:
            u, v = link.e
            self.link_col[(u, v)] = len(self.link_col)
            self.link_col[(v, u)] = len(self.link_col)
        self.load = np.zeros((self.H, len(self.link_col)), dtype=np.float64)
        self._path_cache = {}       # stream.id -> [(链路列号数组, 逐跳延后数组, 交换机数), ...]

        # 观测 = 4 个流特征（周期、deadline、帧长、本流在处理顺序里的进度）+ 每条候选路径的交换机数 + 每个动作提交后的最忙槽负载
        self.obs_dim = 4 + self.k_paths + self.n_actions
        self.rebuild()

    # ------------------------------------------------------------------ 账本镜像

    def rebuild(self):
        """从内核的 usage 重建 load 镜像（reset 后、或不确定是否同步时调用）。"""
        self.load[:] = 0.0
        for (cycle, (u, v)), used in self.model.usage.items():
            self.load[cycle, self.link_col[(u, v)]] = used

    def commit(self, stream, path_idx, offset):
        """内核 commit 之后调用，把同一批槽加到镜像上。"""
        cols, delays, _ = self._path_arrays(stream)[path_idx]
        cycles = self._slot_cycles(stream, delays, np.array([offset]))[0]      # (H//P, 跳数)
        np.add.at(self.load, (cycles, np.broadcast_to(cols, cycles.shape)), float(stream.size_bytes))

    # ------------------------------------------------------------------ 预计算

    def _path_arrays(self, stream):
        if stream.id not in self._path_cache:
            out = []
            for path in self.model.paths[stream.id]:
                cols, delays = [], []
                if self.model.count_source_link:                      # 发送端站 -> 第一台交换机，注入 cycle 本身
                    cols.append(self.link_col[(path[0], path[1])])
                    delays.append(0)
                for i in range(self.model.hops(path)):                 # 第 i 台交换机在注入 + i + 1 发出
                    cols.append(self.link_col[(path[i + 1], path[i + 2])])
                    delays.append(i + 1)
                out.append((np.array(cols, dtype=np.int64), np.array(delays, dtype=np.int64), self.model.hops(path)))
            self._path_cache[stream.id] = out
        return self._path_cache[stream.id]

    def _slot_cycles(self, stream, delays, offsets):
        """所有帧实例、所有跳占用的 cycle 下标，shape (len(offsets), H//P, len(delays))。"""
        P = self.model.period_cycles[stream.id]
        inst = np.arange(self.H // P, dtype=np.int64) * P
        return (offsets[:, None, None] + inst[None, :, None] + delays[None, None, :]) % self.H

    def offset_upper(self, stream):
        """该流可尝试的最大 offset：与 model.offset_range 相同，再受动作空间 n_offsets 限制。"""
        upper = self.model.period_cycles[stream.id] - 1
        if self.model.max_offset_cycles is not None:
            upper = min(upper, self.model.max_offset_cycles)
        return min(upper, self.n_offsets - 1)

    # ------------------------------------------------------------------ 掩码与观测

    def mask_and_obs(self, stream, progress=0.0):
        """返回 (mask, obs)：mask 为 n_actions 个布尔值；obs 为 obs_dim 个 float32，取值裁到 [0, 2]。
        progress 是本流在处理顺序里的位置 / 总流数（0 = 第一条，接近 1 = 最后几条），
        让价值网络知道离 episode 结束还有多远；不加它时折扣回报对所有状态几乎一样，学不到东西。"""
        mask = np.zeros(self.n_actions, dtype=bool)
        loads = np.ones(self.n_actions, dtype=np.float32)          # 无效动作当作"已满"
        hops_feat = np.zeros(self.k_paths, dtype=np.float32)
        P = self.model.period_cycles[stream.id]
        D = self.model.deadline_cycles[stream.id]
        upper = self.offset_upper(stream)
        size = stream.size_bytes

        if upper >= 0:
            offsets = np.arange(upper + 1, dtype=np.int64)
            for p, (cols, delays, hops) in enumerate(self._path_arrays(stream)):
                cycles = self._slot_cycles(stream, delays, offsets)
                load_after = self.load[cycles, cols[None, None, :]].max(axis=(1, 2)) + size   # 每个 offset 一个值
                base = p * self.n_offsets
                loads[base:base + upper + 1] = load_after / self.cap
                mask[base:base + upper + 1] = (1 + hops + offsets <= D) & (load_after <= self.cap)
                hops_feat[p] = hops / max(D, 1)

        head = np.array([P / self.n_offsets, D / self.n_offsets, size / self.cap, progress], dtype=np.float32)
        obs = np.concatenate((head, hops_feat, loads))
        return mask, np.clip(obs, 0.0, 2.0).astype(np.float32)

    def rejection_reason(self, stream):
        """没有任何可行动作时的拒绝原因，与 algorithms.base.classify_rejection 的归纳规则一致：
        没有候选路径 -> no_path；连 offset=0 都超 deadline -> deadline；否则 -> capacity。"""
        from ..model import REASON_NO_PATH, REASON_DEADLINE, REASON_CAPACITY
        arrays = self._path_arrays(stream)
        if not arrays:
            return REASON_NO_PATH
        D = self.model.deadline_cycles[stream.id]
        if all(1 + hops > D for _, _, hops in arrays):
            return REASON_DEADLINE
        return REASON_CAPACITY
