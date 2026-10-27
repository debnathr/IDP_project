"""主入口（对应 gasa 的 mcqf_master.py）：读取导师测试集 -> 建图建模 -> 运行算法 -> 验证 -> 导出。

用法示例：
    python run_cqf.py --tc 1 2 3 --cycle 50 --algo greedy
    python run_cqf.py --tc 1 --cycle 50 --algo random --seed 0 --order random

参数说明见 --help。

验收规则：每次运行都要通过两道验证（账本复核 verify、独立逐 cycle 模拟 simulate_and_validate）。
任一失败 -> 不写普通结果文件，只写 FAILED_<run>.txt 保存全部问题，进程最终以退出码 1 结束，
这样批量实验脚本不会把未通过验证的运行当成成功结果。
"""

import argparse
import datetime
import os
import random
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cqf.data_io import load_test_case                     # noqa: E402
from cqf.network import build_graph                        # noqa: E402
from cqf.model import CqfModel                             # noqa: E402
from cqf.algorithms import make_scheduler, order_streams   # noqa: E402
from cqf import export                                     # noqa: E402
from cqf import validate                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, '..', 'TCs_for_TSN', 'SA_TCs'))
DEFAULT_RESULTS_DIR = os.path.join(HERE, 'results')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='CQF joint routing and offset scheduling (random / greedy baselines)')
    p.add_argument('--tc', type=int, nargs='+', required=True, help='test case numbers, e.g. 1 2 3')
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='directory containing TC*_topo.txt / TC*_flows.txt')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR, help='directory for result files')
    p.add_argument('--cycle', type=int, required=True, help='cycle length T in microseconds; must divide every flow period')
    p.add_argument('--algo', default='greedy', choices=['random', 'greedy'], help='scheduling algorithm')
    p.add_argument('--order', default='input', choices=['input', 'deadline', 'random'], help='flow processing order')
    p.add_argument('--seed', type=int, default=0, help='seed for the random scheduler and random ordering')
    p.add_argument('--k-paths', type=int, default=5, help='number of candidate shortest paths per flow (gasa uses 5)')
    p.add_argument('--max-offset', type=int, default=None, help='upper bound on the offset in cycles (default: period - 1)')
    p.add_argument('--bw-fraction', type=float, default=0.9, help='fraction of the link bandwidth usable by CQF traffic')
    p.add_argument('--count-source-link', action='store_true', help='also account bandwidth on the talker -> first switch link')
    p.add_argument('--tag', default='', help='free-text tag appended to result file names')
    return p.parse_args(argv)


def unique_run_name(results_dir, base):
    """保证结果文件名不与目录中已有的运行重名：同名则追加 _1、_2 ……（同秒内连续运行也不会互相覆盖）。"""
    name = base
    n = 0
    while any(os.path.exists(os.path.join(results_dir, prefix + name + ext))
              for prefix, ext in (('summary_', '.txt'), ('FAILED_', '.txt'))):
        n += 1
        name = '%s_%d' % (base, n)
    return name


def run_one(tc, args):
    """跑一个测试用例。返回 (model, result, ok)，ok 为两道验证是否都通过。"""
    topo_path = os.path.join(args.data_dir, 'TC%d_topo.txt' % tc)
    flows_path = os.path.join(args.data_dir, 'TC%d_flows.txt' % tc)
    t_start = time.time()

    topology, streams = load_test_case(topo_path, flows_path)
    if not streams:
        raise ValueError('flow file %s is empty, nothing to schedule' % flows_path)
    graph = build_graph(topology)
    model = CqfModel(graph, streams, cycle_time_us=args.cycle, bw_fraction=args.bw_fraction,
                     k_paths=args.k_paths, max_offset_cycles=args.max_offset,
                     count_source_link=args.count_source_link)

    ordered = order_streams(streams, args.order, random.Random(args.seed))
    scheduler = make_scheduler(args.algo, seed=args.seed)
    result = scheduler.run(model, ordered)
    t_stop = time.time()

    # 两道验证：verify 是账本复核（与内核共用公式），simulate_and_validate 是独立的逐 cycle 模拟
    verification = model.verify()
    simulation = validate.simulate_and_validate(model)
    ok = verification['ok'] and simulation['ok']

    # 结果文件名：参数都写进去，再加微秒时间戳与去重后缀
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    base = 'TC%d_topo_%s_cqf_T%dus_%s_%s_seed%d_k%d_mo%s%s%s' % (
        tc, stamp, args.cycle, args.algo, args.order, args.seed, args.k_paths,
        'none' if args.max_offset is None else args.max_offset,
        '_srclink' if args.count_source_link else '',
        ('_' + args.tag) if args.tag else '')
    os.makedirs(args.results_dir, exist_ok=True)
    run_name = unique_run_name(args.results_dir, base)

    total = len(streams)
    summary_line = ('TC%d | T=%dus | %s/%s | scheduled %d/%d = %.2f%% | deadline-rej %d | capacity-rej %d | '
                    'no_path %d | max util %.2f | %.1fs | checks %d | verify %s | simulate %s' % (
                        tc, args.cycle, args.algo, args.order, len(model.assignments), total,
                        100 * len(model.assignments) / total,
                        sum(1 for r in result.rejected.values() if r == 'deadline'),
                        sum(1 for r in result.rejected.values() if r == 'capacity'),
                        sum(1 for r in result.rejected.values() if r == 'no_path'),
                        model.max_utilization(), t_stop - t_start, model.n_checks,
                        'OK' if verification['ok'] else 'FAILED: %s' % verification['problems'][:3],
                        'OK' if simulation['ok'] else 'FAILED: %s' % simulation['problems'][:3]))

    if not ok:
        path = export.write_failure_report(args.results_dir, run_name, verification, simulation)
        print(summary_line)
        print('  validation failed, no result files written; see %s' % path)
        return model, result, False

    export.write_export_log(args.results_dir, run_name, 'TC%d' % tc, model, result)
    export.write_result_csv(args.results_dir, run_name, model)
    export.write_queue_sidecar(args.results_dir, run_name, model, simulation['queues'])
    export.write_time_file(args.results_dir, tc, t_start, t_stop, run_name)
    export.write_summary(args.results_dir, run_name, tc, model, result, t_stop - t_start, extra={
        'algorithm': args.algo, 'order': args.order, 'seed': args.seed, 'k_paths': args.k_paths,
        'max_offset': args.max_offset, 'bw_fraction': args.bw_fraction,
        'count_source_link': args.count_source_link,
        'verify_ok (ledger)': verification['ok'], 'simulate_ok (independent)': simulation['ok'],
    })
    print(summary_line)
    return model, result, True


def main(argv=None):
    """逐个 TC 运行。任何一个 TC 验证失败或抛异常，都记为失败并继续处理后面的 TC，最终退出码 1。
    异常（非法输入、程序错误）会连同回溯写进 results/FAILED_TC<i>_exception_<时间>.txt，便于事后诊断。"""
    args = parse_args(argv)
    all_ok = True
    for tc in args.tc:
        try:
            _, _, ok = run_one(tc, args)
        except Exception:
            ok = False
            os.makedirs(args.results_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            path = os.path.join(args.results_dir, 'FAILED_TC%d_exception_%s.txt' % (tc, stamp))
            with open(path, 'w', encoding='utf-8') as f:
                f.write(traceback.format_exc())
            print('TC%d raised an exception, traceback written to %s' % (tc, path))
            traceback.print_exc()
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
