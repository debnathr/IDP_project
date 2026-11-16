"""完整对比实验：同一个周期 T、同一种流顺序下，把随机 / 三个贪心规则 / 训练好的策略在若干用例上跑齐，汇总成一张表。

用法（在项目根目录）：
    python run_comparison.py --cycle 50 --order input --mlp results/mlp_T50_input --gat results/gat_T50_input
    python run_comparison.py --cycle 25 --order input            # 不给任何策略时只跑四个基线
    python run_comparison.py --cycle 50 --order deadline \
        --policy "GAT-score=results/gs_T50_deadline_s100/model.zip" \
        --policy "GAT-score=results/gs_T50_deadline_s101/model.zip"      # 同一个列名给两次 = 两个训练 seed

每个用例跑：random × --seeds 次、greedy、greedy_load、hops_load 各一次（确定性，seed 不影响结果），
以及 --mlp / --gat / --policy 指定的 PPO 策略。策略的位置可以是一个 .zip 文件，也可以是目录
（目录里有 best_model.zip 就用它，即验证用例上回报最高的那个，否则用 model.zip）；config.json 要在同一目录。
同一个列名下有多次运行时，表里给均值 ± 样本标准差。
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
    p = argparse.ArgumentParser(description='compare random / greedy / greedy_load / hops_load and trained policies '
                                            'at one cycle time and flow order')
    p.add_argument('--tc', type=int, nargs='+', default=[1, 2, 3, 5, 9, 11, 13, 18, 25, 26])
    p.add_argument('--cycle', type=int, required=True, help='cycle length T in microseconds')
    p.add_argument('--order', default='input', choices=['input', 'deadline'], help='flow order for every algorithm')
    p.add_argument('--seeds', type=int, default=5, help='number of seeds for the random scheduler')
    p.add_argument('--mlp', default=None,
                   help='trained MLP policy: a model .zip file, or a directory (best_model.zip if present, else model.zip)')
    p.add_argument('--gat', default=None, help='trained GAT policy, same form as --mlp')
    p.add_argument('--policy', action='append', default=[], metavar='LABEL=PATH',
                   help='one more trained policy shown in the column LABEL; PATH as for --mlp. '
                        'Repeat with the same LABEL for several training seeds (the cell is then mean ± std)')
    p.add_argument('--train-tc', type=int, nargs='+', default=[5, 9, 13], help='test cases used for training (label only)')
    p.add_argument('--val-tc', type=int, nargs='+', default=[11], help='test cases used for validation (label only)')
    p.add_argument('--test-tc', type=int, nargs='+', default=[18, 25, 26], help='held-out test cases (label and total row)')
    p.add_argument('--data-dir', default=run_cqf.DEFAULT_DATA_DIR)
    p.add_argument('--results-dir', default=run_cqf.DEFAULT_RESULTS_DIR)
    p.add_argument('--out', default=None, help='markdown table path (default: <results-dir>/comparison_T<cycle>_<order>.md)')
    p.add_argument('--tag', default='cmp', help='tag for the per-run result files')
    return p.parse_args(argv)


def make_columns(args):
    """表的列：[(表头, 算法名, [(seed, 模型位置或 None), ...]), ...]。一列里有几次运行，单元格就是几次的均值 ± 标准差。"""
    columns = [('random', 'random', [(seed, None) for seed in range(args.seeds)]),
               ('greedy', 'greedy', [(0, None)]), ('greedy_load', 'greedy_load', [(0, None)]),
               ('hops_load', 'hops_load', [(0, None)])]
    if args.mlp:
        columns.append(('MLP (PPO)', 'ppo', [(0, args.mlp)]))
    if args.gat:
        columns.append(('GAT (PPO)', 'ppo', [(0, args.gat)]))
    extra = {}                                              # 列名 -> 模型位置列表，保持命令行里的先后
    for item in args.policy:
        label, sep, path = item.partition('=')
        if not sep or not label or not path:
            raise ValueError('--policy expects LABEL=PATH, got %r' % item)
        extra.setdefault(label, []).append(path)
    for label, paths in extra.items():
        if any(label == c[0] for c in columns):
            raise ValueError('--policy label %r is already used by another column' % label)
        columns.append((label, 'ppo', list(enumerate(paths))))
    return columns


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
    columns = make_columns(args)

    rows = []
    for tc in args.tc:
        for label, algo, runs in columns:
            for seed, model_dir in runs:
                rows.append(run_once(tc, label, algo, seed, args, model_dir))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(os.path.splitext(out)[0] + '.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def scheduled(tc, label):
        """该用例、该列的各次运行的接纳数，按 seed 的先后。"""
        return [r['scheduled'] for r in rows if (r['tc'], r['label']) == (tc, label)]

    header = ['TC', 'role', 'flows'] + [c[0] for c in columns]
    if args.gat:
        header.append('GAT vs greedy_load')
    lines = ['# Comparison, T = %d us, flow order = %s, %d seed(s) for random' % (args.cycle, args.order, args.seeds), '',
             '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for tc in args.tc:
        cells = ['%d' % tc, role_of(tc, args), '%d' % [r for r in rows if r['tc'] == tc][0]['flows']]
        cells += [fmt_mean_std(scheduled(tc, label)) for label, _, _ in columns]
        if args.gat:
            diff = statistics.fmean(scheduled(tc, 'GAT (PPO)')) - statistics.fmean(scheduled(tc, 'greedy_load'))
            cells.append('%+d (%+.1f%%)' % (diff, 100.0 * diff / max(statistics.fmean(scheduled(tc, 'greedy_load')), 1)))
        lines.append('| ' + ' | '.join(cells) + ' |')

    test_tcs = [tc for tc in args.tc if tc in args.test_tc]
    if test_tcs:
        # 加总行：每个 seed 先把测试用例的接纳数加起来，再对 seed 求均值 ± 标准差
        totals = {label: [sum(per_tc) for per_tc in zip(*(scheduled(tc, label) for tc in test_tcs))]
                  for label, _, _ in columns}
        cells = ['test total', '+'.join(str(tc) for tc in test_tcs),
                 '%d' % sum([r for r in rows if r['tc'] == tc][0]['flows'] for tc in test_tcs)]
        cells += [fmt_mean_std(totals[label]) for label, _, _ in columns]
        if args.gat:
            diff = statistics.fmean(totals['GAT (PPO)']) - statistics.fmean(totals['greedy_load'])
            cells.append('%+.1f (%+.1f%%)' % (diff, 100.0 * diff / max(statistics.fmean(totals['greedy_load']), 1)))
        lines.append('| **' + '** | **'.join(cells) + '** |')

    lines += ['', 'Cells are admitted flows; "a ± b" is mean ± sample standard deviation over seeds '
                  '(random: scheduler seeds; policies: training seeds). '
                  'Every algorithm processes the flows in the same order (%s).' % args.order]
    policies = [(label, [path for _, path in runs]) for label, algo, runs in columns if algo == 'ppo']
    if policies:
        lines.append('Policy columns: a directory means its best_model.zip (highest validation return) if present, '
                     'else model.zip; a .zip file is used as given.')
        lines += ['%s = %s.' % (label, ', '.join(paths)) for label, paths in policies]
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('written: %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
