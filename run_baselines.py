"""多 seed 基线实验：把随机 / 贪心基线在若干测试用例上跑齐，汇总成一张表。

用法（在项目根目录）：
    python run_baselines.py --tc 1 2 3 5 9 11 13 18 25 26 --cycle 50 --seeds 5
    python run_baselines.py --tc 5 --cycle 50 --seeds 3 --out results/baselines_TC5.md

每个用例跑：
    确定性算法  greedy / greedy_load × 流顺序 input / deadline，各一次（seed 不影响结果）；
    随机算法    random × 流顺序 input / random，各 --seeds 次（seed 0..N-1；random 顺序的 order_seed = seed + 10000）。
每次运行都走 run_cqf.run_one：同一套建模、两道验证和结果文件（文件名带 --tag）。
表里给随机算法的均值 ± 样本标准差，另给 greedy/input 轨迹下的掩码密度（每条流可行组合数 / 候选组合数的均值）
与"没有任何可行动作的流"占比，作为 RL 难度的度量（流按文件顺序处理时，这两个量与 RL 环境看到的一致）。

输出：--out 指定的 Markdown 表（默认 results/baselines_T<cycle>.md）和同名 .csv（每次运行一行）。
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_cqf                                              # noqa: E402
from cqf.export import mask_density_stats                   # noqa: E402

DETERMINISTIC = [('greedy', 'input'), ('greedy', 'deadline'), ('greedy_load', 'input'), ('greedy_load', 'deadline')]
RANDOMIZED = [('random', 'input'), ('random', 'random')]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='run the random / greedy baselines with several seeds and tabulate')
    p.add_argument('--tc', type=int, nargs='+', required=True)
    p.add_argument('--cycle', type=int, required=True)
    p.add_argument('--seeds', type=int, default=5, help='number of seeds for the random scheduler')
    p.add_argument('--data-dir', default=run_cqf.DEFAULT_DATA_DIR)
    p.add_argument('--results-dir', default=run_cqf.DEFAULT_RESULTS_DIR)
    p.add_argument('--out', default=None, help='markdown table path (default: <results-dir>/baselines_T<cycle>.md)')
    p.add_argument('--tag', default='bl', help='tag for the per-run result files')
    return p.parse_args(argv)


def run_once(tc, algo, order, seed, args):
    argv = ['--tc', str(tc), '--cycle', str(args.cycle), '--algo', algo, '--order', order, '--seed', str(seed),
            '--data-dir', args.data_dir, '--results-dir', args.results_dir, '--tag', args.tag]
    model, result, ok = run_cqf.run_one(tc, run_cqf.parse_args(argv))
    if not ok:
        raise RuntimeError('TC%d %s/%s seed %d failed validation' % (tc, algo, order, seed))
    density = mask_density_stats(model, result)
    return {'tc': tc, 'algo': algo, 'order': order, 'seed': seed, 'flows': len(model.streams),
            'scheduled': len(model.assignments), 'runtime_s': result.runtime_s,
            'mask_density': density['mean_density'], 'zero_feasible': density['zero_feasible']}


def fmt_mean_std(values):
    if len(values) == 1:
        return '%d' % values[0]
    return '%.1f ± %.1f' % (statistics.fmean(values), statistics.stdev(values))


def main(argv=None):
    args = parse_args(argv)
    out = args.out or os.path.join(args.results_dir, 'baselines_T%d.md' % args.cycle)
    rows = []
    for tc in args.tc:
        for algo, order in DETERMINISTIC:
            rows.append(run_once(tc, algo, order, 0, args))
        for algo, order in RANDOMIZED:
            for seed in range(args.seeds):
                rows.append(run_once(tc, algo, order, seed, args))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(os.path.splitext(out)[0] + '.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    columns = DETERMINISTIC + RANDOMIZED
    header = ['TC', 'flows'] + ['%s/%s' % c for c in columns] + ['mask density (greedy/input)', 'no-action flows']
    lines = ['# Baselines, T = %d us, %d seed(s) for random' % (args.cycle, args.seeds), '',
             '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for tc in args.tc:
        of_tc = [r for r in rows if r['tc'] == tc]
        cells = ['%d' % tc, '%d' % of_tc[0]['flows']]
        for algo, order in columns:
            values = [r['scheduled'] for r in of_tc if (r['algo'], r['order']) == (algo, order)]
            cells.append(fmt_mean_std(values))
        ref = [r for r in of_tc if (r['algo'], r['order']) == ('greedy', 'input')][0]
        cells.append('%.4f' % ref['mask_density'])
        cells.append('%d (%.1f%%)' % (ref['zero_feasible'], 100.0 * ref['zero_feasible'] / ref['flows']))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Cells are admitted flows; "a ± b" is mean ± sample standard deviation over seeds. '
                  'Mask density = mean over flows of feasible / candidate (route, offset) pairs on the greedy/input run; '
                  '"no-action flows" never reach the policy.']
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('written: %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
