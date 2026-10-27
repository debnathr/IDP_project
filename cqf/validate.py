"""独立验证器：不复用 model.frame_slots / e2e_cycles / queue_sequence 的闭式公式，而是按 cycle 逐拍模拟 CQF 的转发。

为什么需要它：model.verify() 用同一套公式重算账本，若公式本身漏算了某个槽或算错了时序，
check/commit/verify 会一起错而互相"印证"。这里换一种完全不同的写法——在本项目的时隙模型里一拍一拍推进时间
（注意：这是抽象模型内的转发，不涉及真实设备的传播、串行化、处理时延与门控细节）：

    每个 cycle c：
      1. 转发：所有在 cycle c-1 被某台交换机收到的帧，在 cycle c 从该交换机的出端口发出（CQF：停留恰好 1 个 cycle）。
         发出后若还有下一台交换机，它在 cycle c 收到该帧；若已是最后一跳，帧在 cycle c 送达接收端站。
      2. 注入：对每条已接纳的流，若 c = offset + m*P（m = 0,1,2,...），发送端在 cycle c 注入一个新帧，
         第一台交换机在 cycle c 收到它。

时间窗口的选取（H 为超周期，L = lcm(H, 2) 为 even/odd 队列标签的重复周期）：
  * 预热 W 拍：W 是 L 的整数倍且 W >= 最长路径的交换机数。预热保证统计窗口里不缺"更早注入、仍在路上"的帧
    （若 W 小于路径长度，窗口开头几拍的后几跳会少算）；W 是 L 的倍数保证窗口内的 cycle 奇偶与账本的 cycle 下标一致。
  * 资源统计窗口 W .. W+H-1：每次发送记入 load[(c - W, 链路)]，与 model.usage 的槽一一对比。
  * 队列/时延统计窗口 W .. W+L-1：在这段注入的帧记录送达 cycle（查 deadline）和各交换机的收帧 cycle（逐实例队列）。
    实例序号 k = (注入 cycle - W - offset) / P，取值 0 .. L/P-1，与 model.queue_sequence(instance=k) 的 k 同义；
    H 为奇数时 L = 2H，后半段实例的 even/odd 与前半段相反，这正是需要完整覆盖的相位。
  * 末尾再多模拟"最长路径 + 1"拍，让窗口末尾注入的帧也能送达。

它只依赖模型的参数（H、容量、周期、deadline、候选路径、决策、队列名）和拓扑，不依赖 frame_slots / e2e_cycles / queue_sequence。
"""

from collections import defaultdict


class _Frame:
    """一个在网中传输的帧实例。"""
    __slots__ = ('stream', 'instance', 'path', 'offset', 'hop', 'received_cycle', 'inject_cycle', 'steady')

    def __init__(self, stream, instance, path, offset, inject_cycle, steady):
        self.stream = stream
        self.instance = instance          # 帧实例序号 k（None 表示预热期注入，不记录）
        self.path = path                  # 顶点编号列表
        self.offset = offset
        self.hop = 0                      # 下一个要转发它的交换机序号 i（即 path[i+1]）
        self.received_cycle = inject_cycle
        self.inject_cycle = inject_cycle
        self.steady = steady              # 是否在队列/时延统计窗口内注入


def simulate(model):
    """逐 cycle 模拟，返回 (load, e2e, queues)：
    load    {(cycle, (u, v)): 字节}       资源统计窗口内每个槽的占用（cycle 已减去预热 W）
    e2e     {(stream.id, k): cycle 数}    每个帧实例的端到端时延（gasa 口径：送达 cycle + 1 - 周期起点）
    queues  {(stream.id, k): [队列名...]} 每个帧实例在各交换机进入的队列，k = 0 .. L/P-1
    """
    H = model.hyperperiod_cycles
    L = model.queue_phase_period
    assigned = [(model.stream_by_id[sid], a, model.paths[sid][a.path_idx]) for sid, a in model.assignments.items()]
    max_hops = max((len(path) - 2 for _, _, path in assigned), default=0)
    W = L * max(1, -(-max_hops // L))            # L 的整数倍，且 >= max_hops
    total_cycles = W + L + max_hops + 2

    load = defaultdict(int)
    e2e = {}
    queues = defaultdict(list)

    # 预先算好每个 cycle 要注入的帧
    inject_at = defaultdict(list)
    for s, a, path in assigned:
        P = model.period_cycles[s.id]
        for c in range(a.offset, total_cycles, P):
            steady = W <= c < W + L
            k = (c - W - a.offset) // P if steady else None
            inject_at[c].append((s, k, path, a.offset, steady))

    in_flight = []
    for c in range(total_cycles):
        in_load_window = W <= c < W + H
        # 1) 转发：上一拍收到的帧这一拍发出
        still = []
        for fr in in_flight:
            if fr.received_cycle != c - 1:
                still.append(fr)
                continue
            sw, nxt = fr.path[fr.hop + 1], fr.path[fr.hop + 2]
            if in_load_window:
                load[(c - W, (sw, nxt))] += fr.stream.size_bytes
            fr.hop += 1
            fr.received_cycle = c
            if fr.hop == len(fr.path) - 2:
                # 送达接收端站。gasa 的时延口径 = 1 + 交换机数 + offset = (送达 cycle + 1) - 周期起点
                if fr.steady:
                    e2e[(fr.stream.id, fr.instance)] = c + 1 - (fr.inject_cycle - fr.offset)
            else:
                if fr.steady:
                    queues[(fr.stream.id, fr.instance)].append(model.queue_names[c % 2])
                still.append(fr)
        in_flight = still
        # 2) 注入：发送端本拍发出新帧，第一台交换机本拍收到
        for s, k, path, offset, steady in inject_at.get(c, []):
            if model.count_source_link and in_load_window:
                load[(c - W, (path[0], path[1]))] += s.size_bytes
            fr = _Frame(s, k, path, offset, c, steady)
            if steady:
                queues[(s.id, k)].append(model.queue_names[c % 2])
            in_flight.append(fr)
    return dict(load), e2e, dict(queues)


def check_path_structure(model):
    """独立复核每条已接纳流的路径：两端是端站、中间全是交换机、相邻节点之间确实有链路。"""
    problems = []
    g = model.graph
    for sid, a in model.assignments.items():
        s = model.stream_by_id[sid]
        path = model.paths[sid][a.path_idx]
        names = g.vertex_ids(path)
        if len(path) < 3:
            problems.append('flow %s: path %s contains no switch' % (s.name, names))
            continue
        if g.all_vertices[path[0]].id != s.src or g.all_vertices[path[-1]].id != s.dst:
            problems.append('flow %s: path %s does not start/end at source/destination' % (s.name, names))
        if g.all_vertices[path[0]].dt != 'es' or g.all_vertices[path[-1]].dt != 'es':
            problems.append('flow %s: path %s does not start/end at end stations' % (s.name, names))
        if any(g.all_vertices[v].dt != 'sw' for v in path[1:-1]):
            problems.append('flow %s: path %s has intermediate nodes that are not switches' % (s.name, names))
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) not in g.link_lookup:
                problems.append('flow %s: path %s has no link between %s->%s' % (s.name, names, names[path.index(u)], names[path.index(v)]))
    return problems


def simulate_and_validate(model):
    """用模拟结果独立检查当前调度：
    1. 每个决策的取值域（路径下标、offset 为范围内整数，帧长为正）与路径结构；
    2. 模拟得到的槽占用与 model.usage 完全一致；
    3. 每个槽不超过容量；
    4. 每个帧实例的时延不超过 deadline，且实例数等于 Σ L/P；
    5. 逐实例队列与 model.queue_sequence（csv 队列号的来源）一致。
    返回 {'ok', 'problems', 'queues'}。
    """
    problems = []
    for sid, a in model.assignments.items():
        s = model.stream_by_id[sid]
        if type(a.path_idx) is not int or not 0 <= a.path_idx < len(model.paths[sid]):
            problems.append('flow %s: invalid path index %r' % (s.name, a.path_idx))
        if type(a.offset) is not int or a.offset not in model.offset_range(s):
            problems.append('flow %s: invalid offset %r' % (s.name, a.offset))
        if s.size_bytes <= 0:
            problems.append('flow %s: frame size %r is not positive' % (s.name, s.size_bytes))
    if problems:
        # 取值域已经非法，后面的结构检查和模拟都会用这些下标取路径，必须在这里就返回
        return {'ok': False, 'problems': problems, 'queues': {}}
    problems += check_path_structure(model)
    if problems:
        return {'ok': False, 'problems': problems, 'queues': {}}

    load, e2e, queues = simulate(model)
    if load != model.usage:
        missing = set(model.usage) - set(load)
        extra = set(load) - set(model.usage)
        differ = [k for k in set(load) & set(model.usage) if load[k] != model.usage[k]]
        problems.append('simulated slot usage differs from ledger: %d slots only in ledger, %d only in simulation, %d with different values' %
                        (len(missing), len(extra), len(differ)))
    for slot, used in load.items():
        if used > model.bytes_per_cycle:
            problems.append('slot %s: simulated usage %d exceeds capacity %.1f' % (slot, used, model.bytes_per_cycle))
    for (sid, k), delay in e2e.items():
        if delay > model.deadline_cycles[sid]:
            problems.append('flow %s: instance %d has delay %d cycles, exceeding deadline %d' %
                            (model.stream_by_id[sid].name, k, delay, model.deadline_cycles[sid]))
    L = model.queue_phase_period
    expected = sum(L // model.period_cycles[sid] for sid in model.assignments)
    if len(e2e) != expected:
        problems.append('simulation delivered %d frame instances, expected %d' % (len(e2e), expected))
    for sid, a in model.assignments.items():
        s = model.stream_by_id[sid]
        path = model.paths[sid][a.path_idx]
        for k in range(L // model.period_cycles[sid]):
            expected_q = model.queue_sequence(s, path, a.offset, instance=k)
            if queues.get((sid, k)) != expected_q:
                problems.append('flow %s: queues of instance %d differ: formula %s, simulation %s' % (s.name, k, expected_q, queues.get((sid, k))))
    return {'ok': not problems, 'problems': problems, 'queues': queues}
