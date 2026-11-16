"""逐动作共享打分的策略：每个动作 (路径, offset) 用同一个小网络打一个分，分数就是它的 logit。

与 'mlp' 策略的区别只在策略头：
    mlp    整个观测 -> 隐层 -> 一个线性层，一次输出全部 k*n_offsets 个动作的 logit，每个动作有自己专属的一行权重；
    score  先把观测拆成"每个动作一行特征"，所有动作过同一个小 MLP，各得到一个分数。
           参数个数与动作个数无关；"负载越低分越高"这类规律只需要学一次，对所有动作都适用。
价值网络不变：仍然是整个观测过一个 MLP 得到一个标量。动作掩码、采样、PPO 的损失都沿用 MaskablePPO。

带图的版本（'gat_score'，graph_dim > 0）：观测先过 gat.GatExtractor，得到 [扁平观测, 全图嵌入, k 条路径嵌入]。
    每个动作 (p, o) 的输入 = 它自己的 8 个特征 + 路径 p 的嵌入 + 全图嵌入，再过同一个打分网络。
    这样图信息直接参与每个动作的打分，而不是和几百个负载值混在一个长向量里。
    价值网络用同一份特征，但它的梯度不回传到 GAT：GAT 只由策略损失训练，免得被数值大得多的价值损失带着走。

观测就是 features.ActionFeatures.mask_and_obs 的向量，不需要新的观测：
    obs = [周期/n, deadline/n, 帧长/容量, 进度] + k 个 (路径交换机数 / deadline) + k*n 个 (接纳后最忙槽负载 / 容量)
这里 n = n_offsets，周期与 deadline 都以 cycle 为单位。

每个动作 (p, o) 的 8 个特征（action_features）：
    0  load      接纳后最忙槽的负载 / 容量                    直接取自观测
    1  hops      路径 p 的交换机数 / 10                      = (hops/D) * (D/n) * n / 10
    2  o / P     offset 占周期的比例                         = (o/n) / (P/n)
    3  e2e / D   端到端时延占 deadline 的比例                = hops/D + (o/n + 1/n) / (D/n)，因为 e2e = 1 + hops + o
    4..7         当前流的 4 个特征（周期/n, deadline/n, 帧长/容量, 进度），对所有动作相同

规则先验（prior_beta > 0 时）：
    logit(p, o) = -prior_beta * (交换机数 + load) + 打分网络的输出
    交换机数是整数、load 在 0 到 1 之间，所以括号里的量先比交换机数、再比负载："先选交换机数最少的路径，
    再在上面选接纳后最忙槽负载最低的 offset"这条规则的动作，括号里的值最小、logit 最高。
    打分网络最后一层初始化为 0，所以训练开始时取 logit 最高的可行动作就是这条规则；训练只学"在哪里偏离规则"。
    prior_beta 越大，采样时越集中在规则选的动作附近。
"""

import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

N_ACTION_FEATURES = 8


def action_features(obs, k_paths, n_offsets):
    """obs: (B, 4 + k + k*n) -> 逐动作特征 (B, k, n, 8)。"""
    B = obs.shape[0]
    k, n = k_paths, n_offsets
    flow = obs[:, :4]                                              # 周期/n, deadline/n, 帧长/容量, 进度
    period = flow[:, 0].clamp(min=1.0 / n)[:, None, None]          # 至少 1 个 cycle，避免除以 0（episode 结束后的全零观测）
    deadline = flow[:, 1].clamp(min=1.0 / n)[:, None, None]
    hops_rel = obs[:, 4:4 + k][:, :, None]                         # (B, k, 1) 交换机数 / deadline
    loads = obs[:, 4 + k:].reshape(B, k, n)
    offset = (torch.arange(n, device=obs.device, dtype=obs.dtype) / n)[None, None, :]     # (1, 1, n) o / n
    feats = [loads,
             (hops_rel * deadline * (n / 10.0)).expand(B, k, n),
             (offset / period).expand(B, k, n),
             hops_rel + (offset + 1.0 / n) / deadline]
    feats += [flow[:, i][:, None, None].expand(B, k, n) for i in range(4)]
    return torch.stack(feats, dim=-1)


class ScoreExtractor(nn.Module):
    """代替 sb3 的 MlpExtractor：策略一侧输出全部动作的分数，价值一侧输出一个隐向量。"""

    def __init__(self, obs_dim, k_paths, n_offsets, score_hidden, value_hidden, prior_beta=0.0, graph_dim=0):
        super().__init__()
        self.k_paths, self.n_offsets, self.prior_beta = k_paths, n_offsets, float(prior_beta)
        self.graph_dim = int(graph_dim)
        self.flat_dim = obs_dim - self.graph_dim * (1 + k_paths)       # 特征向量里扁平观测占的长度
        layers, last = [], N_ACTION_FEATURES + 2 * self.graph_dim
        for width in score_hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.score_net = nn.Sequential(*layers, nn.Linear(last, 1))      # 所有动作共用
        layers, last = [], obs_dim
        for width in value_hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.value_net = nn.Sequential(*layers)
        self.latent_dim_pi = k_paths * n_offsets                         # sb3 用这两个属性确定后面两层的输入维数
        self.latent_dim_vf = last

    def forward_actor(self, obs):
        k, n, g = self.k_paths, self.n_offsets, self.graph_dim
        x = action_features(obs[:, :self.flat_dim], k, n)
        inputs = x
        if g > 0:
            B = obs.shape[0]
            graph_emb = obs[:, self.flat_dim:self.flat_dim + g]                          # (B, g)
            path_emb = obs[:, self.flat_dim + g:].reshape(B, k, g)                       # (B, k, g)
            inputs = torch.cat([x, path_emb[:, :, None, :].expand(B, k, n, g),
                                graph_emb[:, None, None, :].expand(B, k, n, g)], dim=-1)
        scores = self.score_net(inputs).squeeze(-1)                                      # (B, k, n)
        if self.prior_beta > 0:
            scores = scores - self.prior_beta * (x[..., 1] * 10.0 + x[..., 0])           # 特征 1 是交换机数 / 10，特征 0 是 load
        return scores.reshape(obs.shape[0], -1)

    def forward_critic(self, obs):
        return self.value_net(obs.detach() if self.graph_dim > 0 else obs)

    def forward(self, obs):
        return self.forward_actor(obs), self.forward_critic(obs)


class ScorePolicy(MaskableActorCriticPolicy):
    """MaskablePPO 的策略类。用法：MaskablePPO(ScorePolicy, env, policy_kwargs=dict(k_paths=5, n_offsets=100, ...))。"""

    def __init__(self, *args, k_paths, n_offsets, score_hidden=(32, 32), prior_beta=0.0, graph_dim=0, **kwargs):
        self.k_paths, self.n_offsets, self.score_hidden = int(k_paths), int(n_offsets), tuple(score_hidden)
        self.prior_beta, self.graph_dim = float(prior_beta), int(graph_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        value_hidden = self.net_arch['vf'] if isinstance(self.net_arch, dict) else self.net_arch
        self.mlp_extractor = ScoreExtractor(self.features_dim, self.k_paths, self.n_offsets, self.score_hidden, value_hidden,
                                            self.prior_beta, self.graph_dim)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # 父类在分数后面又接了一个"隐层 -> 动作数"的线性层，这里不需要：分数本身就是 logit
        self.action_net = nn.Identity()
        # 打分网络最后一层用很小的初值，训练开始时各动作分数接近相等，策略接近"可行动作里均匀随机"；
        # 有规则先验时直接置 0，训练开始时的策略恰好由先验决定
        last = self.mlp_extractor.score_net[-1]
        if self.prior_beta > 0:
            nn.init.zeros_(last.weight)
        else:
            nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(k_paths=self.k_paths, n_offsets=self.n_offsets, score_hidden=self.score_hidden,
                    prior_beta=self.prior_beta, graph_dim=self.graph_dim)
        return data
