"""CQF 调度内核（不含任何优化算法）。

一、系统模型（从 gasa 项目 GaSaAlgorithm.py 的 E2E_delay / BU_bandwidth_utilization3 和
    PriorityGroup.py 的导出函数中核实得到，并按"普通 CQF、2 个队列"做了简化）：

  * 时间被切成等长的 cycle，长度 T（微秒）。所有周期必须是 T 的整数倍（gasa 用的周期 1000/2500/5000us
    与 T=25/50/100us 都满足），超周期 H = lcm(所有周期) / T 个 cycle。
  * 一条流的决策只有两项：route（候选路径下标）和 offset（发送端在周期内延后几个 cycle 再注入，
    取整数 cycle，与 gasa 的 "injection delay / time injection" 含义相同）。
  * 逐跳时序（普通 CQF：每台交换机恰好停留 1 个 cycle，对应 gasa 中 2 个队列时 delay 只能取 1）：
        发送端在 cycle  offset + k*P       发出第 k 个帧实例
        第 i 台交换机（i 从 0 开始）在 cycle offset+k*P+i 收到帧，在 cycle offset+k*P+i+1 从出端口发出
        队列归属：在 cycle c 收到的帧进入 queue_names[c % 2]（even/odd 队列交替）。
        因此队列由 route、offset **和帧实例序号 k** 共同决定：P 为偶数时各实例队列相同，
        P 为奇数时相邻实例的队列奇偶互换。它不是独立决策，也不是"一条流固定在一个队列"。
  * 端到端时延（cycle 数）= 1（端站到第一台交换机）+ 交换机数 + offset，与 gasa 的 E2E_delay 相同；
    约束：e2e_cycles <= floor(deadline / T)（gasa：E2E_delay > floor(delay_cycles) 判违约）。
  * 资源：每个 (cycle, 有向链路) 槽内所有帧的字节数之和 <= bytes_per_cycle，
    bytes_per_cycle = 带宽占比 * 1e9 bit/s / 8 * T / 1e6（T 为微秒；T=50us、占比 0.9 时为 5625 字节）。gasa 只统计交换机出端口的链路
    （从第一台交换机到最后一跳进入接收端站），不统计"发送端站 -> 第一台交换机"这条链路；
    这里默认沿用，并用 count_source_link 开关保留另一种口径以便确认。
  * 周期性：一条流在超周期内有 H/P 个帧实例，全部都要检查；cycle 下标按超周期取模。
    这里与 gasa 有两处不同，都属于本项目的适配：
      (1) gasa 先把每个周期的 cycle 数凑成偶数（int(2*round(P/2))）再求 lcm，T=100us 时 25 会变成 24，
          H=lcm(10,24,50)=600；本项目直接用精确的 lcm，H=50。导师数据在 T=10/50 下 P 本来就是偶数，两者一致。
      (2) gasa 对超出超周期末尾的槽直接忽略（对大 offset 的带宽检查偏松），本项目改为取模回绕。
    资源占用以 H 为周期重复；队列的 even/odd 标签以 lcm(H, 2) 为周期重复（H 为奇数时标签要 2H 才回到原样）。

二、设计原则：
  * check（检查）与 commit（提交）分离：check 只读，不改变资源占用和接纳结果（只累加 n_checks 计数器）；
  * release（撤销）精确减去当初加上的占用，不留残余；
  * verify（账本复核）从 assignments 重算一遍资源占用并与增量维护的结果对比。它与 check/commit
    共用 frame_slots 公式，所以只能证明"账本自洽"，不能证明"公式正确"；独立的逐 cycle 模拟验证在 validate.py。
  * 对非法输入（负帧长、重复 id、越界下标、非整数 offset、带宽占比超 1）立即报错，而不是算出一个"合法"的结果。
"""

import math
import operator
from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    """一条已接纳流的决策结果。"""
    path_idx: int   # 候选路径下标
    offset: int     # 注入偏移（cycle 数）


# check() / 拒绝记录可能出现的原因
REASON_DEADLINE = 'deadline'      # 端到端时延超过 deadline
REASON_CAPACITY = 'capacity'      # 某个 (cycle, link) 槽的字节数会超过 bytes_per_cycle
REASON_OFFSET = 'offset_range'    # offset 不在允许范围内
REASON_NO_PATH = 'no_path'        # 发送端到接收端没有任何候选路径（拓扑不连通）


def queue_number(queue_name):
    """队列名 -> csv 里的队列编号：去掉一个前缀 q（gasa 用 strip('q')，这里只去一个前缀，避免 'q7q' 之类被两头剥）。
    队列名约定形如 'q7'；没有前缀 q 的名字原样作为编号。"""
    return queue_name[1:] if queue_name.startswith('q') else queue_name


class CqfModel:
    """CQF 网络状态：拓扑、流集合、参数、当前资源占用与已接纳流的决策。"""

    def __init__(self, graph, streams, cycle_time_us,
                 link_bandwidth_bps=1e9, bw_fraction=0.9,
                 queue_names=('q2', 'q1'), k_paths=5,
                 max_offset_cycles=None, count_source_link=False):
        """
        graph              network.Graph
        streams            data_io.Stream 列表
        cycle_time_us      cycle 长度 T（微秒）
        link_bandwidth_bps 链路速率，固定 1 Gbps
        bw_fraction        CQF 阶段可用带宽占比，项目要求 0.9；只允许 (0, 1]
        queue_names        两个队列的名字；下标 0 对应偶数 cycle 收到的帧（even 队列），1 对应奇数（odd 队列）。
                           默认 q2/q1（沿用 gasa 第 3 优先级组的编号），csv 里写成 2/1
        k_paths            每条流的候选路径条数（gasa 用 5），至少 1
        max_offset_cycles  offset 上限；None 表示只受周期限制（offset <= P-1）
        count_source_link  是否把"发送端站 -> 第一台交换机"的链路计入带宽约束（gasa 不计入）
        """
        # ---- 参数合法性 ----
        if cycle_time_us <= 0:
            raise ValueError('cycle_time_us must be positive')
        if link_bandwidth_bps <= 0:
            raise ValueError('link_bandwidth_bps must be positive')
        if not 0 < bw_fraction <= 1:
            raise ValueError('bw_fraction must be in (0, 1], got %r' % (bw_fraction,))
        if k_paths < 1:
            raise ValueError('k_paths must be at least 1, got %r' % (k_paths,))
        if max_offset_cycles is not None and max_offset_cycles < 0:
            raise ValueError('max_offset_cycles must not be negative')
        if len(queue_names) != 2 or len(set(queue_names)) != 2 or not all(queue_names):
            raise ValueError('plain CQF needs exactly two distinct, non-empty queue names, got %r' % (queue_names,))
        if len({queue_number(name) for name in queue_names}) != 2:
            raise ValueError('queue numbers (names without the leading q) must differ because the csv only stores the number, got %r' % (queue_names,))

        self.graph = graph
        self.streams = list(streams)
        self.T = cycle_time_us
        self.queue_names = tuple(queue_names)
        self.n_queues = len(self.queue_names)
        self.k_paths = k_paths
        self.max_offset_cycles = max_offset_cycles
        self.count_source_link = count_source_link

        # 每个 cycle 内一条链路最多能发送的字节数（gasa: bandwidth*cycle_time/1e6，bandwidth 单位 B/s）
        self.bytes_per_cycle = bw_fraction * link_bandwidth_bps / 8 * cycle_time_us / 1e6

        # ---- 流数据合法性 + 以 cycle 为单位的周期与 deadline ----
        self.period_cycles = {}
        self.deadline_cycles = {}
        seen_ids = set()
        for s in self.streams:
            if s.id in seen_ids:
                raise ValueError('duplicate flow id %d (%s)' % (s.id, s.name))
            seen_ids.add(s.id)
            if s.period_us <= 0 or s.deadline_us <= 0 or s.size_bytes <= 0:
                raise ValueError('flow %s: period / deadline / frame size must be positive: %r' % (s.name, s))
            if s.period_us % cycle_time_us != 0:
                raise ValueError('flow %s: period %dus is not an integer multiple of the cycle %sus, not supported by this model' %
                                 (s.name, s.period_us, cycle_time_us))
            if s.src not in graph.vertex_by_id or s.dst not in graph.vertex_by_id:
                raise ValueError('flow %s: endpoint %s / %s is not in the topology' % (s.name, s.src, s.dst))
            self.period_cycles[s.id] = s.period_us // cycle_time_us
            self.deadline_cycles[s.id] = s.deadline_us // cycle_time_us   # floor，与 gasa 一致
        self.hyperperiod_cycles = math.lcm(*self.period_cycles.values()) if self.streams else 1
        # even/odd 标签回到原样所需的 cycle 数（H 为奇数时，cycle H 与 cycle 0 奇偶不同）
        self.queue_phase_period = math.lcm(self.hyperperiod_cycles, 2)

        # 候选路径（顶点编号列表）；没有路径的流在这里得到空列表，由算法记为 no_path
        self.paths = {s.id: graph.candidate_paths(s.src, s.dst, k_paths) for s in self.streams}
        self.stream_by_id = {s.id: s for s in self.streams}
        # 模型只描述"端站 -> 若干交换机 -> 端站"：端站直连（0 台交换机）和端站转发都不在范围内
        for s in self.streams:
            for path in self.paths[s.id]:
                if len(path) < 3:
                    raise ValueError('flow %s: source and destination are directly connected, not supported by this model' % s.name)
                if not all(graph.is_switch[v] for v in path[1:-1]):
                    raise ValueError('flow %s: candidate path forwards through an end station: %s' % (s.name, graph.vertex_ids(path)))

        # 可变状态：资源占用表与已接纳流
        self.usage = {}          # (cycle, (u, v)) -> 已占用字节数
        self.assignments = {}    # stream.id -> Assignment
        self.n_checks = 0        # check() 调用次数（用于统计动作掩码的计算成本）

    # ------------------------------------------------------------------ 基本量

    def hops(self, path):
        """路径经过的交换机数 = 顶点数 - 2（去掉两端端站）。"""
        return len(path) - 2

    def e2e_cycles(self, stream, path, offset):
        """端到端时延（cycle 数）= 1 + 交换机数 + offset（gasa 的 E2E_delay）。"""
        return 1 + self.hops(path) + offset

    def offset_range(self, stream):
        """该流允许尝试的 offset 集合：0 .. min(P-1, max_offset_cycles)。"""
        upper = self.period_cycles[stream.id] - 1
        if self.max_offset_cycles is not None:
            upper = min(upper, self.max_offset_cycles)
        return range(upper + 1)

    def queue_sequence(self, stream, path, offset, instance=0):
        """第 instance 个帧实例在每台交换机上进入的队列名。

        第 i 台交换机在 cycle offset + instance*P + i 收到帧，队列按该 cycle 的奇偶决定。
        instance=0 就是 gasa 导出所用的"第一个实例"的队列；P 为奇数时其他实例的奇偶会翻转。
        """
        base = offset + instance * self.period_cycles[stream.id]
        return [self.queue_names[(base + i) % self.n_queues] for i in range(self.hops(path))]

    def frame_slots(self, stream, path, offset):
        """该流在一个超周期内占用的全部资源槽 [(cycle, (u, v)), ...]，每个槽占 size_bytes 字节。

        对第 k 个帧实例（k = 0 .. H/P-1）：注入 cycle = offset + k*P；
        第 i 台交换机 path[i+1] 在 cycle 注入+i+1 把帧发到下一个节点 path[i+2]。
        """
        P = self.period_cycles[stream.id]
        H = self.hyperperiod_cycles
        slots = []
        for k in range(H // P):
            inject = offset + k * P
            if self.count_source_link:
                slots.append((inject % H, (path[0], path[1])))
            for i in range(self.hops(path)):
                slots.append(((inject + i + 1) % H, (path[i + 1], path[i + 2])))
        return slots

    def _resolve_action(self, stream, path_idx, offset):
        """把动作编码 (path_idx, offset) 校验成合法整数并取出路径；编码错误直接抛异常。

        这里抛异常而不是返回 (False, 原因)：越界下标、非整数 offset 不是"不可行的决策"，而是程序错误
        （例如 RL 动作解码写错了），必须立刻暴露，不能被当成"最后一条路径"之类的合法动作吞掉。
        """
        try:
            path_idx = operator.index(path_idx)     # 接受 int / numpy 整数，拒绝 float
            offset = operator.index(offset)
        except TypeError:
            raise TypeError('path_idx and offset must be integers, got %r, %r' % (path_idx, offset))
        paths = self.paths[stream.id]
        if not 0 <= path_idx < len(paths):
            raise ValueError('flow %s: candidate path index %d out of range (%d paths)' % (stream.name, path_idx, len(paths)))
        return path_idx, offset, paths[path_idx]

    # ------------------------------------------------------------------ 检查 / 提交 / 撤销

    def check(self, stream, path_idx, offset):
        """只读检查：给定 route 与 offset 是否可接纳。返回 (True, None) 或 (False, 原因)。"""
        self.n_checks += 1
        path_idx, offset, path = self._resolve_action(stream, path_idx, offset)
        if offset not in self.offset_range(stream):
            return False, REASON_OFFSET
        if self.e2e_cycles(stream, path, offset) > self.deadline_cycles[stream.id]:
            return False, REASON_DEADLINE
        size = stream.size_bytes
        for slot in self.frame_slots(stream, path, offset):
            if self.usage.get(slot, 0) + size > self.bytes_per_cycle:
                return False, REASON_CAPACITY
        return True, None

    def feasible_actions(self, stream):
        """枚举该流所有 (path_idx, offset) 组合并逐个 check。

        返回 (feasible, reasons)：
          feasible  可接纳的 (path_idx, offset) 列表 —— 这就是后续 RL 的动作掩码；
          reasons   {(path_idx, offset): 原因} 记录每个不可行组合为什么不行。
        注意必须对"组合"做检查：某条 route 单独看可行、某个 offset 单独看可行，不代表二者组合可行。
        没有候选路径的流返回 ([], {})，由调用方记为 no_path。
        """
        feasible = []
        reasons = {}
        for path_idx in range(len(self.paths[stream.id])):
            for offset in self.offset_range(stream):
                ok, reason = self.check(stream, path_idx, offset)
                if ok:
                    feasible.append((path_idx, offset))
                else:
                    reasons[(path_idx, offset)] = reason
        return feasible, reasons

    def commit(self, stream, path_idx, offset):
        """接纳一条流：先 check，通过后把它的全部资源槽加到 usage，并记录决策。"""
        if stream.id in self.assignments:
            raise ValueError('flow %s is already admitted, cannot commit twice' % stream.name)
        ok, reason = self.check(stream, path_idx, offset)
        if not ok:
            raise ValueError('flow %s: (route=%d, offset=%d) is infeasible: %s' % (stream.name, path_idx, offset, reason))
        path_idx, offset, path = self._resolve_action(stream, path_idx, offset)
        for slot in self.frame_slots(stream, path, offset):
            self.usage[slot] = self.usage.get(slot, 0) + stream.size_bytes
        self.assignments[stream.id] = Assignment(path_idx, offset)

    def release(self, stream):
        """撤销一条已接纳的流：精确减去它当初加上的占用，归零的槽删掉，不留残余。"""
        assignment = self.assignments.pop(stream.id)
        path = self.paths[stream.id][assignment.path_idx]
        for slot in self.frame_slots(stream, path, assignment.offset):
            remaining = self.usage[slot] - stream.size_bytes
            if remaining < 0:
                # 减出负数说明账本已经损坏（commit/release 不配对或资源表被外部改动），不能静默吞掉
                raise RuntimeError('ledger corrupted: slot %s holds %d bytes but flow %s releases %d'
                                   % (slot, self.usage[slot], stream.name, stream.size_bytes))
            if remaining == 0:
                del self.usage[slot]
            else:
                self.usage[slot] = remaining

    def reset(self):
        """清空所有接纳结果（用于重新开始一轮调度 / RL 的 episode 重置）。"""
        self.usage = {}
        self.assignments = {}

    # ------------------------------------------------------------------ 账本复核与统计

    def verify(self):
        """账本复核：根据 assignments 用同一套公式重算资源表，与增量维护的 usage 对比，
        并检查每条流的 deadline、每个槽的容量、每个决策的取值域。

        返回 {'ok': bool, 'problems': [描述...]}。
        注意：它与 check/commit 共用 frame_slots / e2e_cycles，只能证明增量维护没有出错，
        不能证明公式本身正确。独立于这套公式的逐 cycle 模拟验证见 validate.simulate_and_validate。
        """
        problems = []
        rebuilt = {}
        for sid, a in self.assignments.items():
            s = self.stream_by_id[sid]
            if not 0 <= a.path_idx < len(self.paths[sid]):
                problems.append('flow %s: path index %r out of range' % (s.name, a.path_idx))
                continue
            path = self.paths[sid][a.path_idx]
            if a.offset not in self.offset_range(s):
                problems.append('flow %s: offset %r out of range' % (s.name, a.offset))
            if self.e2e_cycles(s, path, a.offset) > self.deadline_cycles[sid]:
                problems.append('flow %s violates its deadline' % s.name)
            for slot in self.frame_slots(s, path, a.offset):
                rebuilt[slot] = rebuilt.get(slot, 0) + s.size_bytes
        for slot, used in rebuilt.items():
            if used > self.bytes_per_cycle:
                problems.append('slot %s: %d bytes exceed capacity %.1f' % (slot, used, self.bytes_per_cycle))
        if rebuilt != self.usage:
            problems.append('incrementally maintained ledger differs from recomputation')
        return {'ok': not problems, 'problems': problems}

    def max_utilization(self):
        """当前最繁忙槽的利用率（0~1），用于观察资源使用情况。"""
        if not self.usage:
            return 0.0
        return max(self.usage.values()) / self.bytes_per_cycle
