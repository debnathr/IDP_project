"""可替换的调度算法。

所有算法都继承 base.SchedulerBase，只需实现 choose()：
"面对当前这条流和它的可行 (route, offset) 集合，选哪一个"。
逐条流的循环、check/commit、拒绝原因记录都在基类的 run() 里，算法不需要也不允许直接改资源表。
强化学习策略 ppo_scheduler.PpoScheduler 也通过同样的接口接入（按需导入，基线不依赖 torch）。
"""

from .base import SchedulerBase, ScheduleResult, order_streams
from .random_scheduler import RandomScheduler
from .greedy_scheduler import GreedyScheduler, LoadGreedyScheduler

def _make_ppo(seed=None, **kwargs):
    from .ppo_scheduler import PpoScheduler
    return PpoScheduler(seed=seed, **kwargs)


SCHEDULERS = {
    'random': RandomScheduler,
    'greedy': GreedyScheduler,
    'greedy_load': LoadGreedyScheduler,
    'ppo': _make_ppo,
}


def make_scheduler(name, seed=None, **kwargs):
    """按名字创建算法实例；kwargs 只给需要额外参数的算法（ppo 的 model_path）。"""
    if name not in SCHEDULERS:
        raise ValueError('unknown algorithm %r, available: %s' % (name, sorted(SCHEDULERS)))
    return SCHEDULERS[name](seed=seed, **kwargs)
