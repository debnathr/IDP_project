"""三个贪心基线，只差排序键。

用到的量：
    e2e   该 (route, offset) 的端到端时延（cycle 数），由 action_scores 给出；
    load  提交后，该方案涉及的所有槽里"最忙那个槽"的利用率（已占字节 + 本流帧长，除以每 cycle 容量），同上；
    hops  路径经过的交换机数（model.hops）。

greedy       键 (e2e, load, path_idx, offset)：时延最小优先。这是第一版基线（2026-10-21）。
             缺点：e2e 在前等价于"offset 越小越好"，流都堆到最早的几个 cycle，
             把早期的槽填满后后面的流就没有位置，小用例 TC2 上甚至不如随机。
greedy_load  键 (load, e2e, path_idx, offset)：最不拥挤优先，拥挤程度相同时再取时延最小。
             起到负载均衡作用；空网络上与 greedy 行为相同，只有当最早的槽比别的槽忙时才改选别的 offset / 路径。
             缺点：负载在前，所以会为了避开一个忙的槽改走更长的路径，而长路径每个帧实例多占一条链路上的槽。
hops_load    键 (hops, load, offset, path_idx)：先选交换机数最少的路径（占用的槽最少），
             再在这些路径上选接纳后最忙槽负载最低的 offset；负载相同取 offset 最小的，再相同取路径下标最小的。
             只有当交换机数最少的路径上没有可行的 offset 时，才会用更长的路径。

三个都保留：greedy 是早期结果表里的基线，保留它才能复现；greedy_load 是修正后的基线；
hops_load 把"路径"与"offset"两个选择分开，路径只看长度，负载只用来选 offset。
键相同就按键里最后的下标取最小的，保证结果确定、可复现。都是建立基线的朴素规则，不代表最优。
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


class HopsLoadScheduler(SchedulerBase):
    name = 'hops_load'

    def choose(self, model, stream, feasible):
        def key(action):
            path_idx, offset = action
            _, load = action_scores(model, stream, action)
            return (model.hops(model.paths[stream.id][path_idx]), load, offset, path_idx)
        return min(feasible, key=key)
