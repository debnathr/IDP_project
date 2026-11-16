"""用训练好的 MaskablePPO 策略做 choose()，让 PPO 走与随机 / 贪心完全相同的 run() 流程。

与基线的唯一差别是 choose() 的内容：
    观测  由 cqf.rl.features.ActionFeatures 计算（与训练环境同一份代码），进度特征 = 本流在处理顺序里的下标 / 总流数；
    掩码  由内核给的 feasible 列表生成（内核是真值），只覆盖 offset < n_offsets 的动作；
    动作  policy.predict(obs, action_masks=mask)，解码成 (path_idx, offset)，并再次确认在 feasible 里。
after_commit() 钩子把提交同步到特征表的账本镜像。

模型目录需要两个文件（由 cqf.rl.train 写出）：model.zip（或 best_model.zip）与 config.json，
config.json 记录训练时的 T、k_paths、n_offsets 等，运行前逐项与当前模型核对，不一致直接报错。
"""

import json
import os

import numpy as np

from .base import SchedulerBase
from ..rl.features import ActionFeatures
from ..rl.graph_obs import GraphObs


def resolve_model_files(model_path):
    """model_path 可以是目录（取 best_model.zip，否则 model.zip）或 .zip 文件；config.json 在同一目录。"""
    if os.path.isdir(model_path):
        for name in ('best_model.zip', 'model.zip'):
            zip_path = os.path.join(model_path, name)
            if os.path.exists(zip_path):
                break
        else:
            raise FileNotFoundError('no best_model.zip / model.zip in %s' % model_path)
    else:
        zip_path = model_path
    config_path = os.path.join(os.path.dirname(os.path.abspath(zip_path)), 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError('config.json not found next to %s' % zip_path)
    return zip_path, config_path


class PpoScheduler(SchedulerBase):
    name = 'ppo'

    def __init__(self, seed=None, model_path=None, deterministic=True):
        super().__init__(seed)
        if model_path is None:
            raise ValueError('the ppo scheduler needs a model path (--model)')
        from sb3_contrib import MaskablePPO        # 延迟导入：基线算法不依赖 torch
        self.zip_path, config_path = resolve_model_files(model_path)
        with open(config_path, encoding='utf-8') as f:
            self.config = json.load(f)
        self.policy = MaskablePPO.load(self.zip_path, device='cpu')
        self.n_offsets = int(self.config['n_offsets'])
        self.deterministic = deterministic
        self.policy_kind = self.config.get('policy', 'mlp')
        self.features = None
        self.graph_obs = None
        self._position = {}

    def run(self, model, ordered_streams):
        expected = {'cycle_time_us': model.T, 'k_paths': model.k_paths,
                    'count_source_link': model.count_source_link, 'bytes_per_cycle': model.bytes_per_cycle}
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError('model config %s=%r does not match the current run (%r)' %
                                 (key, self.config.get(key), value))
        max_period = max(model.period_cycles.values())
        if self.n_offsets < max_period and (model.max_offset_cycles is None or model.max_offset_cycles >= self.n_offsets):
            raise ValueError('policy was trained with n_offsets=%d but this test case has periods up to %d cycles'
                             % (self.n_offsets, max_period))
        self.features = ActionFeatures(model, self.n_offsets)
        if 'obs_dim' in self.config and int(self.config['obs_dim']) != self.features.obs_dim:
            raise ValueError('policy was trained with obs_dim=%d but the current features have obs_dim=%d'
                             % (int(self.config['obs_dim']), self.features.obs_dim))
        uses_graph = self.policy_kind in ('gat', 'gat_score')
        self.graph_obs = GraphObs(self.features, **self.config['graph']) if uses_graph else None
        ordered_streams = list(ordered_streams)
        self._position = {s.id: i / len(ordered_streams) for i, s in enumerate(ordered_streams)}
        return super().run(model, ordered_streams)

    def choose(self, model, stream, feasible):
        progress = self._position[stream.id]
        _, obs = self.features.mask_and_obs(stream, progress)
        if self.graph_obs is not None:
            obs = self.graph_obs.observe(stream, progress, obs)
        mask = np.zeros(self.features.n_actions, dtype=bool)
        for path_idx, offset in feasible:
            if offset < self.n_offsets:
                mask[path_idx * self.n_offsets + offset] = True
        if not mask.any():
            raise ValueError('flow %s: no feasible action can be encoded with n_offsets=%d' % (stream.name, self.n_offsets))
        action, _ = self.policy.predict(obs, action_masks=mask, deterministic=self.deterministic)
        path_idx, offset = divmod(int(action), self.n_offsets)
        if (path_idx, offset) not in set(feasible):
            raise RuntimeError('policy chose a masked action %d for flow %s' % (int(action), stream.name))
        return path_idx, offset

    def after_commit(self, model, stream, path_idx, offset):
        self.features.commit(stream, path_idx, offset)
