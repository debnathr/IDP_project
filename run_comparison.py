"""完整对比实验：同一个周期 T、同一种流顺序下，把随机 / 贪心 / 负载贪心 / MLP 策略 / GAT 策略在若干用例上跑齐，汇总成一张表。

用法（在项目根目录）：
    python run_comparison.py --cycle 50 --order input --mlp results/mlp_T50_input --gat results/gat_T50_input
    python run_comparison.py --cycle 50 --order deadline --mlp results/mlp_T50_deadline --gat results/gat_T50_deadline_1M_s0
    python run_comparison.py --cycle 25 --order input            # 不给 --mlp / --gat 时只跑三个基线

每个用例跑：random × --seeds 次（表里给均值 ± 样本标准差）、greedy、greedy_load 各一次（确定性，seed 不影响结果），
以及 --mlp / --gat 指定目录里的 PPO 策略（取 best_model.zip，即验证用例上回报最高的那个）。
所有运行都走 run_cqf.run_one：同一套建模、两道验证、结果文件（文件名带 --tag）。
策略只按 --order 给定的顺序处理流，要和训练时的 --order 一致（策略是在那个顺序下训练出来的）。

输出：--out 指定的 Markdown 表（默认 results/comparison_T<cycle>_<order>.md）和同名 .csv（每次运行一行）。
表的最后一行把测试用例（--test-tc）的接纳数加总，方便横向比较不同 T 与顺序。
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_cqf                                              # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='compare random / greedy / greedy_load / MLP / GAT at one cycle time and flow order')
    p.add_argument('--tc', type=int, nargs='+', default=[1, 2, 3, 5, 9, 11, 13, 18, 25, 26])
    p.add_argument('--cycle', type=int, required=True, help='cycle length T in microseconds')
    p.add_argument('--order', default='input', choices=['input', 'deadline'], help='flow order for every algorithm')
    p.add_argument('--seeds', type=int, default=5, help='number of seeds for the random scheduler')
    p.add_argument('--mlp', default=None, help='directory of the trained MLP policy (best_model.zip + config.json)')
    p.add_argument('--gat', default=None, help='directory of the trained GAT policy (best_model.zip + config.json)')
    p.add_argument('--train-tc', type=int, nargs='+', default=[5, 9, 13], help='test cases used for training (label only)')
    p.add_argument('--val-tc', type=int, nargs='+', default=[11], help='test cases used for validation (label only)')
    p.add_argument('--test-tc', type=int, nargs='+', default=[18, 25, 26], help='held-out test cases (label and total row)')
    p.add_argument('--data-dir', default=run_cqf.DEFAULT_DATA_DIR)
    p.add_argument('--results-dir', default=run_cqf.DEFAULT_RESULTS_DIR)
    p.add_argument('--out', default=None, help='markdown table path (default: <results-dir>/comparison_T<cycle>_<order>.md)')
    p.add_argument('--tag', default='cmp', help='tag for the per-run result files')
    return p.parse_args(argv)


def run_once(tc, label, algo, seed, args, model_dir=None):
    argv = ['--tc', str(tc), '--cycle', str(args.cycle), '--algo', algo, '--order', args.order, '--seed', str(seed),
            '--data-dir', args.data_dir, '--results-dir', args.results_dir, '--tag', args.tag]
    if model_dir is not None:
        argv += ['--model', model_dir]
    model, result, ok = run_cqf.run_one(tc, run_cqf.parse_args(argv))
    if not ok:
        raise RuntimeError('TC%d %s/%s seed %d failed validation' % (tc, label, args.order, seed))
    return {'tc': tc, 'label': label, 'algo': algo, 'order': args.order, 'seed': seed, 'model': model_dir or '',
            'flows': len(model.streams), 'scheduled': len(model.assignments), 'runtime_s': result.runtime_s}


def fmt_mean_std(values):
    if len(values) == 1:
        return '%d' % values[0]
    return '%.1f ± %.1f' % (statistics.fmean(values), statistics.stdev(values))


def role_of(tc, args):
    if tc in args.train_tc:
        return 'train'
    if tc in args.val_tc:
        return 'val'
    if tc in args.test_tc:
        return 'test'
    return 'smoke'


def main(argv=None):
    args = parse_args(argv)
    out = args.out or os.path.join(args.results_dir, 'comparison_T%d_%s.md' % (args.cycle, args.order))

    # 列：(表头, 算法名, 模型目录, seed 个数)
    columns = [('random', 'random', None, args.seeds), ('greedy', 'greedy', None, 1), ('greedy_load', 'greedy_load', None, 1)]
    if args.mlp:
        columns.append(('MLP (PPO)', 'ppo', args.mlp, 1))
    if args.gat:
        columns.append(('GAT (PPO)', 'ppo', args.gat, 1))

    rows = []
    for tc in args.tc:
        for label, algo, model_dir, n_seeds in columns:
            for seed in range(n_seeds):
                rows.append(run_once(tc, label, algo, seed, args, model_dir))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(os.path.splitext(out)[0] + '.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mean_of(tc, label):
        return statistics.fmean(r['scheduled'] for r in rows if (r['tc'], r['label']) == (tc, label))

    header = ['TC', 'role', 'flows'] + [c[0] for c in columns]
    if args.gat:
        header.append('GAT vs greedy_load')
    lines = ['# Comparison, T = %d us, flow order = %s, %d seed(s) for random' % (args.cycle, args.order, args.seeds), '',
             '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for tc in args.tc:
        cells = ['%d' % tc, role_of(tc, args), '%d' % [r for r in rows if r['tc'] == tc][0]['flows']]
        for label, _, _, _ in columns:
            cells.append(fmt_mean_std([r['scheduled'] for r in rows if (r['tc'], r['label']) == (tc, label)]))
        if args.gat:
            diff = mean_of(tc, 'GAT (PPO)') - mean_of(tc, 'greedy_load')
            cells.append('%+d (%+.1f%%)' % (diff, 100.0 * diff / max(mean_of(tc, 'greedy_load'), 1)))
        lines.append('| ' + ' | '.join(cells) + ' |')

    test_tcs = [tc for tc in args.tc if tc in args.test_tc]
    if test_tcs:
        cells = ['test total', '+'.join(str(tc) for tc in test_tcs),
                 '%d' % sum([r for r in rows if r['tc'] == tc][0]['flows'] for tc in test_tcs)]
        totals = {label: sum(mean_of(tc, label) for tc in test_tcs) for label, _, _, _ in columns}
        cells += ['%.1f' % totals[label] for label, _, _, _ in columns]
        if args.gat:
            diff = totals['GAT (PPO)'] - totals['greedy_load']
            cells.append('%+.1f (%+.1f%%)' % (diff, 100.0 * diff / max(totals['greedy_load'], 1)))
        lines.append('| **' + '** | **'.join(cells) + '** |')

    lines += ['', 'Cells are admitted flows; "a ± b" is mean ± sample standard deviation over seeds. '
                  'PPO columns use best_model.zip (highest validation return) of the given directory; '
                  'every algorithm processes the flows in the same order (%s).' % args.order]
    if args.mlp or args.gat:
        lines.append('Models: MLP = %s, GAT = %s.' % (args.mlp or '-', args.gat or '-'))
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('written: %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
