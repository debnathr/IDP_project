"""简单贪心：在可行组合里选"端到端时延最小、其次最不拥挤"的那个。

排序键 (e2e_cycles, 提交后涉及槽的最大利用率, path_idx, offset)：
1. 优先选择端到端时延最小的可行组合（offset 小、路径短）；
2. 时延相同时，选提交后"最忙的那个槽"最不满的组合，起到负载均衡作用；
3. 再相同就取下标最小的，保证结果确定、可复现。
这是一个用于建立基线的朴素规则，不代表最优。
"""

from .base import SchedulerBase


class GreedyScheduler(SchedulerBase):
    name = 'greedy'

    def choose(self, model, stream, feasible):
        def key(action):
            path_idx, offset = action
            path = model.paths[stream.id][path_idx]
            slots = model.frame_slots(stream, path, offset)
            max_load_after = max(model.usage.get(slot, 0) + stream.size_bytes for slot in slots)
            return (model.e2e_cycles(stream, path, offset), max_load_after / model.bytes_per_cycle,
                    path_idx, offset)
        return min(feasible, key=key)
