"""两个贪心基线，只差排序键的先后。

两个量（action_scores）：
    e2e   该 (route, offset) 的端到端时延（cycle 数）；
    load  提交后，该方案涉及的所有槽里"最忙那个槽"的利用率（已占字节 + 本流帧长，除以每 cycle 容量）。

greedy       键 (e2e, load, path_idx, offset)：时延最小优先。这是第一版基线（2026-10-21）。
             缺点：e2e 在前等价于"offset 越小越好"，流都堆到最早的几个 cycle，
             把早期的槽填满后后面的流就没有位置，小用例 TC2 上甚至不如随机。
greedy_load  键 (load, e2e, path_idx, offset)：最不拥挤优先，拥挤程度相同时再取时延最小。
             起到负载均衡作用；空网络上与 greedy 行为相同，只有当最早的槽比别的槽忙时才改选别的 offset / 路径。

两个都保留：greedy 是早期结果表里的基线，保留它才能复现；greedy_load 是修正后的基线。
再相同就取下标最小的 (path_idx, offset)，保证结果确定、可复现。都是建立基线的朴素规则，不代表最优。
"""

from .base import SchedulerBase


def action_scores(model, stream, action):
    """返回 (e2e_cycles, 提交后涉及槽的最大利用率)。"""
    path_idx, offset = action
    path = model.paths[stream.id][path_idx]
    slots = model.frame_slots(stream, path, offset)
    max_load_after = max(model.usage.get(slot, 0) + stream.size_bytes for slot in slots)
    return model.e2e_cycles(stream, path, offset), max_load_after / model.bytes_per_cycle


class GreedyScheduler(SchedulerBase):
    name = 'greedy'

    def choose(self, model, stream, feasible):
        def key(action):
            e2e, load = action_scores(model, stream, action)
            return (e2e, load) + action
        return min(feasible, key=key)


class LoadGreedyScheduler(SchedulerBase):
    name = 'greedy_load'

    def choose(self, model, stream, feasible):
        def key(action):
            e2e, load = action_scores(model, stream, action)
            return (load, e2e) + action
        return min(feasible, key=key)
