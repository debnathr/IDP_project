"""GAT 阶段的图观测：把当前网络状态和当前流表示成固定形状的"图"。

MLP 阶段的观测（features.ActionFeatures.mask_and_obs）只有当前流自己 5 条候选路径上的信息，看不到别的链路有多忙、
别的流会争哪些链路。这里把整张拓扑交给策略：

    节点  = 网络节点（交换机 / 端站），特征：是否交换机、是否端站、度 / 最大度、是否当前流的源 / 目的、
            有几条候选路径经过它（/ k_paths）
    链路  = 有向链路（与 ActionFeatures.link_col 的列号一致，这样负载矩阵能直接按列取），特征：
            超周期内的利用率曲线粗分 link_bins 段的均值、整体均值、最大值、接近满（>= 0.9）的槽占比、
            有几条候选路径经过它（/ k_paths）
    路径  = 当前流每条候选路径经过的链路编号列表（供 GAT 之后做"路径池化"）

形状固定为 (max_nodes, max_links, max_path_links)，不足补零并用 node_valid / link_valid / path_mask 标记，
这样同一个网络能在不同大小的拓扑上训练和评估（训练 12 / 16 台交换机，测试 18 / 24 台）。
导师数据里最大的是 TC25/26：116 个节点、244 条有向链路、候选路径最长 17 条链路；默认 128 / 256 / 20 够用，
不够时构造时直接报错。

观测字典里同时带着 MLP 阶段的那个向量（'actions'：流特征 + 每条路径跳数 + 每个动作提交后的最忙槽负载），
GAT 的输出是在它之上追加的"全图上下文"，这样 GAT 策略与 MLP 策略的差别只有图编码器，便于做消融。
"""

import numpy as np
from gymnasium import spaces


class GraphObs:

    def __init__(self, feat, max_nodes=128, max_links=256, max_path_links=20, link_bins=10):
        """feat 是 ActionFeatures（账本镜像 load 与链路列号都从它来）。"""
        self.feat = feat
        self.model = feat.model
        graph = self.model.graph
        self.max_nodes, self.max_links, self.max_path_links, self.link_bins = max_nodes, max_links, max_path_links, link_bins
        self.k_paths = self.model.k_paths
        n_nodes = graph.num_vertices
        n_links = len(feat.link_col)
        if n_nodes > max_nodes or n_links > max_links:
            raise ValueError('topology has %d nodes / %d directed links, more than max_nodes=%d / max_links=%d'
                             % (n_nodes, n_links, max_nodes, max_links))
        self.n_nodes, self.n_links = n_nodes, n_links

        # 静态部分：节点类型、度；链路端点
        self.node_valid = np.zeros(max_nodes, dtype=np.float32)
        self.node_valid[:n_nodes] = 1.0
        self.link_valid = np.zeros(max_links, dtype=np.float32)
        self.link_valid[:n_links] = 1.0
        self.link_src = np.zeros(max_links, dtype=np.int64)
        self.link_dst = np.zeros(max_links, dtype=np.int64)
        for (u, v), col in feat.link_col.items():
            self.link_src[col], self.link_dst[col] = u, v
        degree = np.array([len(graph.neighbors[i]) for i in range(n_nodes)], dtype=np.float32)
        self.node_static = np.zeros((max_nodes, 3), dtype=np.float32)
        self.node_static[:n_nodes, 0] = np.array(graph.is_switch, dtype=np.float32)
        self.node_static[:n_nodes, 1] = 1.0 - self.node_static[:n_nodes, 0]
        self.node_static[:n_nodes, 2] = degree / max(degree.max(), 1.0)

        # 利用率曲线分段：把 H 个 cycle 均分成 link_bins 段（H 不能整除时最后一段多几个 cycle）
        H = feat.H
        self.bin_starts = np.linspace(0, H, link_bins + 1).astype(int)[:-1]
        self._path_cache = {}

        n_node_feat = 3 + 3                      # 静态 3 + 动态 3（源、目的、候选路径数）
        n_link_feat = link_bins + 3 + 1          # 曲线分段 + 均值 / 最大值 / 接近满占比 + 候选路径数
        self.space = spaces.Dict({
            'actions': spaces.Box(0.0, 2.0, shape=(feat.obs_dim,), dtype=np.float32),
            'node_x': spaces.Box(0.0, 1.0, shape=(max_nodes, n_node_feat), dtype=np.float32),
            'link_x': spaces.Box(0.0, 1.0, shape=(max_links, n_link_feat), dtype=np.float32),
            'link_src': spaces.Box(0, max_nodes - 1, shape=(max_links,), dtype=np.int64),
            'link_dst': spaces.Box(0, max_nodes - 1, shape=(max_links,), dtype=np.int64),
            'node_valid': spaces.Box(0.0, 1.0, shape=(max_nodes,), dtype=np.float32),
            'link_valid': spaces.Box(0.0, 1.0, shape=(max_links,), dtype=np.float32),
            'path_links': spaces.Box(0, max_links - 1, shape=(self.k_paths, max_path_links), dtype=np.int64),
            'path_mask': spaces.Box(0.0, 1.0, shape=(self.k_paths, max_path_links), dtype=np.float32),
        })

    def _path_links(self, stream):
        """每条候选路径经过的有向链路列号（含源端站 -> 第一台交换机、最后一台交换机 -> 目的端站）。"""
        if stream.id not in self._path_cache:
            ids = np.zeros((self.k_paths, self.max_path_links), dtype=np.int64)
            mask = np.zeros((self.k_paths, self.max_path_links), dtype=np.float32)
            for p, path in enumerate(self.model.paths[stream.id]):
                cols = [self.feat.link_col[(path[i], path[i + 1])] for i in range(len(path) - 1)]
                if len(cols) > self.max_path_links:
                    raise ValueError('flow %s has a candidate path with %d links, more than max_path_links=%d'
                                     % (stream.name, len(cols), self.max_path_links))
                ids[p, :len(cols)] = cols
                mask[p, :len(cols)] = 1.0
            self._path_cache[stream.id] = (ids, mask)
        return self._path_cache[stream.id]

    def observe(self, stream, progress, flat_obs):
        """返回观测字典。flat_obs 是 ActionFeatures.mask_and_obs 给出的向量（同一步计算，直接带上）。"""
        util = self.feat.load / self.feat.cap                                  # (H, 链路数)
        link_x = np.zeros((self.max_links, self.link_bins + 4), dtype=np.float32)
        n = self.n_links
        binned = np.add.reduceat(util, self.bin_starts, axis=0)                 # (bins, 链路数) 每段之和
        lengths = np.diff(np.append(self.bin_starts, util.shape[0]))[:, None]
        link_x[:n, :self.link_bins] = (binned / lengths).T
        link_x[:n, self.link_bins] = util.mean(axis=0)
        link_x[:n, self.link_bins + 1] = util.max(axis=0)
        link_x[:n, self.link_bins + 2] = (util >= 0.9).mean(axis=0)

        path_links, path_mask = self._path_links(stream)
        node_x = np.zeros((self.max_nodes, 6), dtype=np.float32)
        node_x[:, :3] = self.node_static
        src = self.model.graph.vertex_by_id[stream.src].device_idx
        dst = self.model.graph.vertex_by_id[stream.dst].device_idx
        node_x[src, 3] = 1.0
        node_x[dst, 4] = 1.0
        for p, path in enumerate(self.model.paths[stream.id]):
            node_x[path, 5] += 1.0 / self.k_paths
            used = path_links[p, path_mask[p] > 0]
            link_x[used, self.link_bins + 3] += 1.0 / self.k_paths

        return {
            'actions': flat_obs,
            'node_x': np.clip(node_x, 0.0, 1.0),
            'link_x': np.clip(link_x, 0.0, 1.0),
            'link_src': self.link_src,
            'link_dst': self.link_dst,
            'node_valid': self.node_valid,
            'link_valid': self.link_valid,
            'path_links': path_links,
            'path_mask': path_mask,
        }

    def empty(self):
        """episode 结束后的占位观测（全零）。"""
        return {key: np.zeros(box.shape, dtype=box.dtype) for key, box in self.space.spaces.items()}
