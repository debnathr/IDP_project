"""算法基类与流顺序工具。

这里替代了 gasa 中 GaSaAlgorithm.py 的全部优化算法（遗传算法 + 模拟退火）。
gasa 是"先给所有流随便指定方案，再用目标函数整体打分、反复变异"，最后事后剔除违约的流；
本项目改为"逐条流在线接纳"：按某个顺序一条条处理，每条流只在当前资源状态下
从可行的 (route, offset) 组合里选一个并提交，没有可行组合就拒绝。
这种形式便于后续接入逐流决策的 RL 策略。
"""

import random
import time
from dataclasses import dataclass, field

from ..model import REASON_DEADLINE, REASON_CAPACITY, REASON_NO_PATH


@dataclass
class ScheduleResult:
    """一次完整调度的结果。"""
    order: list = field(default_factory=list)       # 实际处理流的顺序（流 id）
    scheduled: list = field(default_factory=list)   # 成功接纳的流 id（按接纳顺序）
    rejected: dict = field(default_factory=dict)    # 被拒绝的流 id -> 原因：'deadline' / 'capacity' / 'no_path'
    runtime_s: float = 0.0                          # 调度耗时（秒）


def order_streams(streams, mode, rng=None):
    """决定处理流的顺序。三种方式必须明确区分：
    'input'    文件顺序（固定顺序）；
    'deadline' 按 deadline 从小到大（gasa 在分组前就是这样排序的，稳定排序保留文件内相对顺序）；
    'random'   用给定随机数发生器打乱 —— 这只是"随机顺序"，不是"让模型学习选下一条流"。
    """
    streams = list(streams)
    if mode == 'input':
        return streams
    if mode == 'deadline':
        return sorted(streams, key=lambda s: s.deadline_us)
    if mode == 'random':
        (rng or random).shuffle(streams)
        return streams
    raise ValueError('unknown flow ordering mode %r' % mode)


def classify_rejection(model, stream, reasons):
    """没有任何可行组合时，归纳一个拒绝原因：
    没有候选路径 -> no_path；所有组合都因 deadline 失败 -> deadline；否则 -> capacity。
    """
    if not model.paths[stream.id]:
        return REASON_NO_PATH
    if all(r == REASON_DEADLINE for r in reasons.values()):
        return REASON_DEADLINE
    return REASON_CAPACITY


class SchedulerBase:
    """逐条接纳式调度算法的公共流程。子类只实现 choose()。"""

    name = 'base'

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, model, stream, feasible):
        """从可行组合 feasible（[(path_idx, offset), ...]，非空）里选一个返回。"""
        raise NotImplementedError

    def run(self, model, ordered_streams):
        """按 ordered_streams 的顺序逐条处理：枚举可行组合 -> choose -> commit。"""
        result = ScheduleResult()
        t0 = time.perf_counter()
        for stream in ordered_streams:
            result.order.append(stream.id)
            feasible, reasons = model.feasible_actions(stream)
            if not feasible:
                result.rejected[stream.id] = classify_rejection(model, stream, reasons)
                continue
            path_idx, offset = self.choose(model, stream, feasible)
            model.commit(stream, path_idx, offset)
            result.scheduled.append(stream.id)
        result.runtime_s = time.perf_counter() - t0
        return result
