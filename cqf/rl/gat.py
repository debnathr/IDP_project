"""GAT 编码器：作为 MaskablePPO 的特征提取器（sb3 的 BaseFeaturesExtractor）。

网络结构（按前向顺序）：
    1. 节点 / 链路输入层：把 graph_obs 里的节点特征、链路特征各自线性映射到 dim 维；
    2. layers 层 GatLayer：每层沿有向链路 u -> v 传消息，v 对它的入边做注意力加权求和，消息里带链路自己的特征
       （带边特征的 GAT，注意力打分 = a_dst·Wh_v + a_src·Wh_u + a_edge·We_uv，LeakyReLU 后按 v 的入边做 softmax）；
    3. 读出：链路嵌入 = MLP([h_u, h_v, e_uv])；路径嵌入 = 该路径经过的链路嵌入的均值；全图嵌入 = 有效节点嵌入的均值；
    4. 输出特征 = [MLP 阶段的 'actions' 向量, 全图嵌入, k 条路径嵌入]，交给 sb3 的 MLP 头产生动作 logits 与价值。

所有操作都是 gather / scatter，与节点、链路的编号顺序无关（置换不变），tests/test_rl_env.py 里专门检查这一点。
不依赖 PyTorch Geometric，只用 torch 自带的 gather / scatter_add / scatter_reduce，便于逐行解释。
"""

import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _gather_nodes(h, idx):
    """h: (B, N, ...)，idx: (B, L) 节点编号 -> (B, L, ...)。"""
    shape = (idx.shape[0], idx.shape[1]) + h.shape[2:]
    idx = idx.reshape(idx.shape + (1,) * (h.dim() - 2)).expand(shape)
    return torch.gather(h, 1, idx)


def _scatter_to_nodes(values, dst, n_nodes, reduce='sum'):
    """values: (B, L, ...) 按 dst (B, L) 汇总到节点 -> (B, N, ...)。reduce 为 'sum' 或 'amax'。"""
    shape = (values.shape[0], n_nodes) + values.shape[2:]
    idx = dst.reshape(dst.shape + (1,) * (values.dim() - 2)).expand(values.shape)
    if reduce == 'sum':
        return torch.zeros(shape, dtype=values.dtype, device=values.device).scatter_add(1, idx, values)
    out = torch.full(shape, -1e9, dtype=values.dtype, device=values.device)
    return out.scatter_reduce(1, idx, values, reduce='amax', include_self=True)


class GatLayer(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, 'dim must be divisible by heads'
        self.heads, self.d = heads, dim // heads
        self.W = nn.Linear(dim, dim, bias=False)        # 节点变换（多头拼在一起）
        self.We = nn.Linear(dim, dim, bias=False)       # 链路特征变换
        self.att_src = nn.Parameter(torch.empty(heads, self.d))
        self.att_dst = nn.Parameter(torch.empty(heads, self.d))
        self.att_edge = nn.Parameter(torch.empty(heads, self.d))
        for p in (self.att_src, self.att_dst, self.att_edge):
            nn.init.xavier_uniform_(p)
        self.skip = nn.Linear(dim, dim)                 # 残差：节点自己的信息
        self.out = nn.Linear(dim, dim)

    def forward(self, h, e, src, dst, link_valid, node_valid):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.d)
        We = self.We(e).view(B, e.shape[1], self.heads, self.d)
        hu = _gather_nodes(Wh, src)                                        # (B, L, heads, d) 源节点
        hv = _gather_nodes(Wh, dst)                                        # 目的节点
        logits = (hu * self.att_src).sum(-1) + (hv * self.att_dst).sum(-1) + (We * self.att_edge).sum(-1)
        logits = F.leaky_relu(logits, 0.2)                                 # (B, L, heads)
        logits = logits.masked_fill(link_valid[..., None] == 0, -1e9)
        # 按目的节点分组的 softmax：先减每个节点上的最大值再取 exp，数值稳定
        logits = logits - _gather_nodes(_scatter_to_nodes(logits, dst, N, 'amax'), dst)
        w = torch.exp(logits) * link_valid[..., None]
        denom = _gather_nodes(_scatter_to_nodes(w, dst, N), dst) + 1e-9
        alpha = (w / denom)[..., None]                                     # (B, L, heads, 1)
        agg = _scatter_to_nodes(alpha * (hu + We), dst, N)                 # (B, N, heads, d)
        out = F.elu(self.out(agg.reshape(B, N, -1)) + self.skip(h))
        return out * node_valid[..., None]


class GatExtractor(BaseFeaturesExtractor):
    """observation_space 是 graph_obs.GraphObs.space；输出维度 = len(actions) + dim + k_paths * dim。"""

    def __init__(self, observation_space, dim=32, heads=4, layers=2):
        n_actions = observation_space['actions'].shape[0]
        self.k_paths = observation_space['path_links'].shape[0]
        super().__init__(observation_space, features_dim=n_actions + dim + self.k_paths * dim)
        self.node_in = nn.Linear(observation_space['node_x'].shape[1], dim)
        self.link_in = nn.Linear(observation_space['link_x'].shape[1], dim)
        self.layers = nn.ModuleList([GatLayer(dim, heads) for _ in range(layers)])
        self.link_out = nn.Linear(3 * dim, dim)

    def forward(self, obs):
        node_valid, link_valid = obs['node_valid'], obs['link_valid']
        src, dst = obs['link_src'].long(), obs['link_dst'].long()         # sb3 把整数观测转成了 float，这里转回
        h = F.elu(self.node_in(obs['node_x'])) * node_valid[..., None]
        e = F.elu(self.link_in(obs['link_x'])) * link_valid[..., None]
        for layer in self.layers:
            h = layer(h, e, src, dst, link_valid, node_valid)
        # 链路嵌入与路径池化
        g = F.elu(self.link_out(torch.cat([_gather_nodes(h, src), _gather_nodes(h, dst), e], dim=-1)))
        g = g * link_valid[..., None]                                       # (B, L, dim)
        path_links = obs['path_links'].long()                               # (B, k, P)
        path_mask = obs['path_mask'][..., None]                             # (B, k, P, 1)
        B, k, P = path_links.shape
        pe = _gather_nodes(g, path_links.reshape(B, k * P)).reshape(B, k, P, -1)
        path_emb = (pe * path_mask).sum(2) / (path_mask.sum(2) + 1e-9)      # (B, k, dim)
        graph_emb = (h * node_valid[..., None]).sum(1) / (node_valid.sum(1, keepdim=True) + 1e-9)
        return torch.cat([obs['actions'], graph_emb, path_emb.reshape(B, -1)], dim=1)
