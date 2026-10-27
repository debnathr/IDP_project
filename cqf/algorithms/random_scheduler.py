"""随机调度：在当前流的全部可行 (route, offset) 组合中等概率随机选一个。

作用：作为随机基线，也用来检验内核——随机选出的方案必须全部通过最终 verify()。
注意它不是"随机顺序"：流的顺序由 order_streams 单独决定，这里只随机选动作。
"""

from .base import SchedulerBase


class RandomScheduler(SchedulerBase):
    name = 'random'

    def choose(self, model, stream, feasible):
        return self.rng.choice(feasible)
