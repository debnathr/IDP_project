"""结果导出，文件格式与 gasa 项目保持一致。

gasa（PriorityGroup.export_full_routes_and_schedules / export_full_routes_and_schedules_csv /
mcqf_master.py）会写出三类文件，这里逐一对应：

1. results/export_log_<run>.txt   人读的日志：
       第 1 行  TC2 scheduled/num_streams 155/260 = 59.62%
       第 2 行  k: 1 (p, '%', 'unscheduled', n, 'total', N, 'e2e too large', [...], 'bandwidth violation', [...])
       第 3 行  " cycle_time: 50us bandwidth: 5625.0bytes " 后面直接接第一条流的记录
       其后每行一条流：Stream: id .. | k_group .. | q_n .. | src->dst | period ..| deadline .. | size ..| injection delay:<offset>[路径节点名][各交换机队列名]
   gasa 有 3 个优先级组所以有 3 段 "k: ..."，本项目普通 CQF 只有 1 组，写 "k: 1" 一段。
   gasa 对未调度成功的流照样打印路径（它是事后判定的）；本项目未接纳的流没有决策，
   在行尾写 "injection delay:None[][] # UNSCHEDULED (原因)"。

2. results/res_<run>_sorted.csv   机器可读的调度结果（只含成功调度的流，按流编号排序）：
       FlowName,maxE2E(us),Deadline(us),Path(SourceName|LinkID|Qnumber)
       VLAN_0_Flow_3,400,1088,node0_0_2_3|P1-sw_0_2|e15|2|P1-sw_0_3|e21|1|P5-sw_0_4|e30|2|node0_0_4_3
   Path 列：源端站 | 每台交换机一段 "<出端口>-<交换机名>|<出链路名>|<队列号>" | 目的端站。
   队列号是队列名去掉前缀 q（gasa: el[2].strip('q')）。
   注意：队列号是**第一个帧实例**在各交换机进入的队列（gasa 的 csv 导出也只用 offset 计算，即第一个实例）。
   gasa 自己的两种导出口径并不一致：csv（export_full_routes_and_schedules_csv，current_cycle = -1+cycle_injection）
   把 offset 算进队列；txt 日志（export_full_routes_and_schedules，current_cycle = -1）忽略 offset，
   相当于所有流都从 cycle 0 注入。本项目 txt 与 csv 都采用 csv 的口径（含 offset），因为它才与实际转发一致。
   当周期的 cycle 数 P 为奇数时，后续实例的 even/odd 会交替翻转，csv 里不体现；
   逐实例队列由 validate.simulate_and_validate 返回，需要时可另行导出。

3. results/time<i>_time_<run>.txt  运行时间（开始/结束/总秒数）。gasa 只叫 time<i>_time.txt，同一 TC
   换算法或 T 再跑就会覆盖，这里把 run 名也放进文件名。

另外按导师 TC_conditions.txt 的要求，再写一个汇总文件 results/summary_<run>.txt：
TC 编号、总流数、成功调度数、调度率、运行时间。
"""

import os

from .model import queue_number


def gasa_stream_line(model, stream, assignment):
    """生成 export_log 中一条流的记录，格式照抄 gasa 的 Stream.__str__ 与导出拼接方式。
    行尾的队列列表用含 offset 的口径（与 gasa 的 csv 一致，与 gasa 的 txt 不一致，见文件头说明）。"""
    period_cycles = stream.period_us / model.T          # gasa 存的是 float
    deadline_cycles = stream.deadline_us / model.T      # gasa 打印未取整的 float
    head = 'Stream: id {} | k_group {} | q_n {} | {}->{} | period {}| deadline {} | size {}'.format(
        stream.id, 1, model.n_queues, stream.src, stream.dst, period_cycles, deadline_cycles, stream.size_bytes)
    if assignment is None:
        return head + '| injection delay:None[][]'
    path = model.paths[stream.id][assignment.path_idx]
    return head + '| injection delay:{}{}{}'.format(
        assignment.offset, model.graph.vertex_ids(path), model.queue_sequence(stream, path, assignment.offset))


def write_export_log(results_dir, run_name, tc_label, model, result):
    """写 export_log_<run>.txt。result 为 algorithms.base.ScheduleResult。"""
    total = len(model.streams)
    scheduled = len(model.assignments)
    percent = round(100 * scheduled / total, 2) if total else 0.0
    # gasa 的 find_scheduled_percentage 返回的元组，照原样 str() 出来。
    # gasa 只有 deadline / 带宽两类；本项目新增的 no_path（没有候选路径）不属于任何一类，
    # 只计入 'unscheduled' 总数，并在该流的记录行里标出。
    deadline_ids = [sid for sid, r in result.rejected.items() if r == 'deadline']
    capacity_ids = [sid for sid, r in result.rejected.items() if r == 'capacity']
    # gasa 这里写的是 100*round(ratio, 2)，会先把比例四舍五入到百分位再乘 100（1/3 -> 33.0），
    # 与首行的 33.33% 不一致；这里统一用 percent，两处数字相同。
    stats_tuple = (percent, '%', 'unscheduled', total - scheduled,
                   'total', total, 'e2e too large', deadline_ids, 'bandwidth violation', capacity_ids)

    lines = []
    lines.append('%s scheduled/num_streams %d/%d = %s%%' % (tc_label, scheduled, total, percent))
    lines.append('k: 1 %s' % (stats_tuple,))
    prefix = ' cycle_time: %sus bandwidth: %sbytes ' % (model.T, round(model.bytes_per_cycle, 2))
    stream_lines = []
    for s in model.streams:
        a = model.assignments.get(s.id)
        line = gasa_stream_line(model, s, a)
        if a is None:
            line += ' # UNSCHEDULED (%s)' % result.rejected.get(s.id, '?')
        stream_lines.append(line)
    if stream_lines:
        stream_lines[0] = prefix + stream_lines[0]
    lines.extend(stream_lines)

    path = os.path.join(results_dir, 'export_log_%s.txt' % run_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def csv_path_string(model, stream, assignment):
    """生成 csv 的 Path 列：源端站|出端口-交换机|出链路|队列号|...|目的端站。"""
    path = model.paths[stream.id][assignment.path_idx]
    names = model.graph.vertex_ids(path)
    queues = model.queue_sequence(stream, path, assignment.offset)     # 第一个帧实例的队列，同 gasa
    parts = [names[0]]
    for i in range(model.hops(path)):
        sw = path[i + 1]
        link_name, egress_port = model.graph.egress(sw, path[i + 2])
        parts.append('%s-%s|%s|%s' % (egress_port, names[i + 1], link_name, queue_number(queues[i])))
    parts.append(names[-1])
    return '|'.join(parts)


def write_result_csv(results_dir, run_name, model):
    """写 res_<run>_sorted.csv（只含成功调度的流，按流 id 排序）。

    gasa 按流名末尾的数字排序（`int(name.split('_')[-1])`），导师数据里这个数字就是流 id（文件第 2 列），
    所以按 id 排序结果相同；但按 id 排序不要求流名必须以数字结尾，名字任意时也不会在导出阶段崩溃。
    """
    rows = []
    for s in sorted(model.streams, key=lambda s: s.id):
        a = model.assignments.get(s.id)
        if a is None:
            continue
        path = model.paths[s.id][a.path_idx]
        max_e2e_us = model.e2e_cycles(s, path, a.offset) * model.T
        rows.append((s.name, str(max_e2e_us), str(s.deadline_us), csv_path_string(model, s, a)))

    path = os.path.join(results_dir, 'res_%s_sorted.csv' % run_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('FlowName,maxE2E(us),Deadline(us),Path(SourceName|LinkID|Qnumber)\n')
        f.write('\n'.join(','.join(row) for row in rows))
    return path


def write_queue_sidecar(results_dir, run_name, model, queues):
    """写 queues_<run>.csv：每条已接纳流在队列相位周期 L = lcm(H, 2) 内每个帧实例的队列序列。

    res_*_sorted.csv 只给第一个实例的队列号（与 gasa 一致）；这个文件补全完整的周期队列配置。
    queues 是 validate.simulate_and_validate 返回的 {(stream.id, k): [队列名...]}（模拟得到，不是公式算的）。
    """
    rows = []
    for s in sorted(model.streams, key=lambda s: s.id):
        a = model.assignments.get(s.id)
        if a is None:
            continue
        P = model.period_cycles[s.id]
        for k in range(model.queue_phase_period // P):
            rows.append((s.name, str(P), str(a.offset), str(k), '|'.join(queues.get((s.id, k), []))))
    path = os.path.join(results_dir, 'queues_%s.csv' % run_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('FlowName,PeriodCycles,Offset,Instance,QueuesPerSwitch\n')
        f.write('\n'.join(','.join(row) for row in rows) + '\n')
    return path


def write_failure_report(results_dir, run_name, verification, simulation):
    """验证失败时写 FAILED_<run>.txt，保存全部问题，不再写普通结果文件。"""
    path = os.path.join(results_dir, 'FAILED_%s.txt' % run_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('ledger verify ok: %s\n' % verification['ok'])
        for p in verification['problems']:
            f.write('  ' + p + '\n')
        f.write('independent simulation ok: %s\n' % simulation['ok'])
        for p in simulation['problems']:
            f.write('  ' + p + '\n')
    return path


def write_time_file(results_dir, tc_number, t_start, t_stop, run_name=''):
    """写 time<i>_time_<run>.txt（内容照抄 gasa 的 mcqf_master.py；文件名加 run 名避免覆盖）。"""
    suffix = ('_' + run_name) if run_name else ''
    path = os.path.join(results_dir, 'time%s_time%s.txt' % (tc_number, suffix))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('start time is in secs:\n%s\n' % t_start)
        f.write('stop time is in secs:\n%s\n' % t_stop)
        f.write('total runtime is in secs:\n%s\n' % (t_stop - t_start))
    return path


def write_summary(results_dir, run_name, tc_number, model, result, runtime_s, extra=None):
    """写 summary_<run>.txt：导师 TC_conditions.txt 要求的汇总（TC 编号、总流数、成功数、调度率、时间）。"""
    total = len(model.streams)
    scheduled = len(model.assignments)
    lines = [
        'TC number: %s' % tc_number,
        'total flows: %d' % total,
        'scheduled flows: %d' % scheduled,
        'schedulability: %.2f%%' % (100 * scheduled / total if total else 0.0),
        'runtime (s): %.3f' % runtime_s,
        'cycle_time (us): %s' % model.T,
        'bytes_per_cycle: %.1f' % model.bytes_per_cycle,
        'hyperperiod (cycles): %d' % model.hyperperiod_cycles,
        'queue phase period (cycles): %d' % model.queue_phase_period,
        'rejected (deadline): %d' % sum(1 for r in result.rejected.values() if r == 'deadline'),
        'rejected (capacity): %d' % sum(1 for r in result.rejected.values() if r == 'capacity'),
        'rejected (no_path): %d' % sum(1 for r in result.rejected.values() if r == 'no_path'),
        'check() calls: %d' % model.n_checks,
    ]
    for key, value in (extra or {}).items():
        lines.append('%s: %s' % (key, value))
    path = os.path.join(results_dir, 'summary_%s.txt' % run_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path
