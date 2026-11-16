"""主入口 run_cqf.py 的验收规则测试：验证失败时不得写普通结果文件，退出码必须非零；结果文件名不得互相覆盖。

运行：python -m unittest tests.test_run_cqf -v
"""

import csv
import os
import shutil
import tempfile
import unittest
from unittest import mock

import run_comparison
import run_cqf
from cqf import validate

from tests.test_model import TOPO_LINES, flow


class TestRunCqf(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, 'data')
        self.results = os.path.join(self.tmp, 'results')
        os.makedirs(self.data)
        with open(os.path.join(self.data, 'TC1_topo.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(TOPO_LINES) + '\n')
        with open(os.path.join(self.data, 'TC1_flows.txt'), 'w', encoding='utf-8') as f:
            f.write(flow(0, 'esA', 'esB', 1000, 800, 500) + '\n' + flow(1, 'esC', 'esB', 2500, 900, 600) + '\n')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _argv(self, *extra):
        return ['--tc', '1', '--cycle', '100', '--data-dir', self.data, '--results-dir', self.results] + list(extra)

    def _files(self):
        return sorted(os.listdir(self.results))

    def test_success_writes_all_files_and_exit_0(self):
        self.assertEqual(run_cqf.main(self._argv()), 0)
        names = self._files()
        for prefix in ('export_log_', 'res_', 'queues_', 'summary_', 'time1_time_'):
            self.assertTrue(any(n.startswith(prefix) for n in names), (prefix, names))
        self.assertFalse(any(n.startswith('FAILED_') for n in names))

    def test_load_greedy_and_mask_density_in_summary(self):
        self.assertEqual(run_cqf.main(self._argv('--algo', 'greedy_load')), 0)
        summary = [n for n in self._files() if n.startswith('summary_')][0]
        self.assertIn('greedy_load', summary)
        with open(os.path.join(self.results, summary), encoding='utf-8') as f:
            text = f.read()
        self.assertIn('mask density', text)
        self.assertIn('order_seed: 10000', text)          # 默认 order_seed = seed + 10000

    def test_hops_load_runs_through_the_same_checks(self):
        self.assertEqual(run_cqf.main(self._argv('--algo', 'hops_load', '--order', 'deadline')), 0)
        self.assertTrue(any(n.startswith('summary_') and '_hops_load_deadline_' in n for n in self._files()))

    def test_order_seed_is_independent_of_scheduler_seed(self):
        # 随机顺序只由 order_seed 决定：默认 seed + 10000，显式给 --order-seed 时用它
        import random
        from cqf.algorithms import order_streams
        for extra, expected_seed in ((['--seed', '5'], 10005), (['--seed', '5', '--order-seed', '3'], 3)):
            model, result, ok = run_cqf.run_one(1, run_cqf.parse_args(self._argv('--order', 'random', *extra)))
            self.assertTrue(ok)
            expected = [s.id for s in order_streams(model.streams, 'random', random.Random(expected_seed))]
            self.assertEqual(result.order, expected)

    def test_validation_failure_blocks_exports_and_exit_1(self):
        broken = {'ok': False, 'problems': ['injected failure'], 'queues': {}}
        with mock.patch.object(validate, 'simulate_and_validate', return_value=broken):
            self.assertEqual(run_cqf.main(self._argv()), 1)
        names = self._files()
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].startswith('FAILED_'))
        with open(os.path.join(self.results, names[0]), encoding='utf-8') as f:
            self.assertIn('injected failure', f.read())

    def test_empty_flow_file_is_a_clear_failure(self):
        with open(os.path.join(self.data, 'TC1_flows.txt'), 'w', encoding='utf-8') as f:
            f.write('')
        self.assertEqual(run_cqf.main(self._argv()), 1)
        names = self._files()
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].startswith('FAILED_TC1_exception_'))
        with open(os.path.join(self.results, names[0]), encoding='utf-8') as f:
            self.assertIn('is empty', f.read())

    def test_exception_in_one_tc_does_not_stop_the_others(self):
        # TC2 的文件不存在 -> 异常；TC1 正常。要求 TC1 照常产出，整体退出码 1
        self.assertEqual(run_cqf.main(['--tc', '2', '1', '--cycle', '100', '--data-dir', self.data,
                                       '--results-dir', self.results]), 1)
        names = self._files()
        self.assertTrue(any(n.startswith('FAILED_TC2_exception_') for n in names))
        self.assertTrue(any(n.startswith('summary_TC1_') for n in names))

    def test_same_second_runs_do_not_overwrite(self):
        fixed = mock.Mock()
        fixed.strftime.return_value = '2026-10-21_00-00-00-000000'
        with mock.patch('run_cqf.datetime') as dt:
            dt.datetime.now.return_value = fixed
            run_cqf.main(self._argv())
            run_cqf.main(self._argv('--max-offset', '0'))
            run_cqf.main(self._argv('--max-offset', '0'))     # 完全相同的配置再跑一次
        summaries = [n for n in self._files() if n.startswith('summary_')]
        self.assertEqual(len(summaries), 3)


class TestRunComparison(unittest.TestCase):
    """run_comparison.py：随机（多 seed）与三个贪心规则在玩具用例上跑齐，写出 Markdown 表与 csv。不给策略时没有策略列。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, 'data')
        self.results = os.path.join(self.tmp, 'results')
        os.makedirs(self.data)
        with open(os.path.join(self.data, 'TC1_topo.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(TOPO_LINES) + '\n')
        with open(os.path.join(self.data, 'TC1_flows.txt'), 'w', encoding='utf-8') as f:
            f.write(flow(0, 'esA', 'esB', 1000, 800, 500) + '\n' + flow(1, 'esC', 'esB', 2500, 900, 600) + '\n')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_table_csv_and_test_total_row(self):
        out = os.path.join(self.tmp, 'cmp.md')
        rc = run_comparison.main(['--tc', '1', '--cycle', '100', '--seeds', '2', '--test-tc', '1',
                                  '--data-dir', self.data, '--results-dir', self.results, '--out', out])
        self.assertEqual(rc, 0)
        with open(out, encoding='utf-8') as f:
            text = f.read()
        self.assertIn('| 1 | test | 2 | 2.0 ± 0.0 | 2 | 2 | 2 |', text)      # 两条玩具流全部接纳
        self.assertIn('| **test total** | **1** | **2** | **2.0 ± 0.0** | **2** | **2** | **2** |', text)   # 有加总行
        self.assertNotIn('MLP (PPO)', text)
        with open(os.path.join(self.tmp, 'cmp.csv'), encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['label'] for r in rows], ['random', 'random', 'greedy', 'greedy_load', 'hops_load'])
        self.assertEqual([r['seed'] for r in rows], ['0', '1', '0', '0', '0'])
        self.assertTrue(all(r['order'] == 'input' for r in rows))
        self.assertTrue(any(n.startswith('summary_') and '_cmp' in n for n in os.listdir(self.results)))

    def test_without_test_tc_there_is_no_total_row(self):
        out = os.path.join(self.tmp, 'cmp.md')
        rc = run_comparison.main(['--tc', '1', '--cycle', '100', '--seeds', '1',
                                  '--data-dir', self.data, '--results-dir', self.results, '--out', out])
        self.assertEqual(rc, 0)
        with open(out, encoding='utf-8') as f:
            text = f.read()
        self.assertIn('| 1 | smoke | 2 | 2 | 2 | 2 | 2 |', text)              # 单 seed 时随机列不带 ±
        self.assertNotIn('test total', text)

    def test_policy_option_needs_label_and_path(self):
        for bad in ('no_equal_sign', '=path', 'label=', 'greedy=some/path'):   # 最后一个：列名与基线列重名
            with self.assertRaises(ValueError):
                run_comparison.make_columns(run_comparison.parse_args(['--cycle', '100', '--policy', bad]))
        columns = run_comparison.make_columns(run_comparison.parse_args(
            ['--cycle', '100', '--seeds', '1', '--policy', 'A=x/model.zip', '--policy', 'B=y', '--policy', 'A=z/model.zip']))
        self.assertEqual(columns[-2:], [('A', 'ppo', [(0, 'x/model.zip'), (1, 'z/model.zip')]), ('B', 'ppo', [(0, 'y')])])


if __name__ == '__main__':
    unittest.main()
