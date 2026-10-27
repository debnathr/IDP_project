"""主入口 run_cqf.py 的验收规则测试：验证失败时不得写普通结果文件，退出码必须非零；结果文件名不得互相覆盖。

运行：python -m unittest tests.test_run_cqf -v
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

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


if __name__ == '__main__':
    unittest.main()
