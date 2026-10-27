"""可以手算的小例子，用来验证 CQF 调度内核。运行方式：

    python -m unittest tests.test_model -v

玩具拓扑（3 台交换机、3 个端站）：

    esA --- sw1 ===== sw2 --- esB
             \\        /
              \\      /
               sw3--+
    esC --- sw1

链路：e1 sw1-sw2，e2 sw1-sw3，e3 sw3-sw2，e4 sw1-esA，e5 sw2-esB，e6 sw1-esC。
esA -> esB 有两条候选路径：路径 0 = esA,sw1,sw2,esB（2 台交换机），路径 1 = esA,sw1,sw3,sw2,esB（3 台）。

统一取 T = 100us：bytes_per_cycle = 0.9 * 1e9 / 8 * 100e-6 = 11250 字节。
周期 1000us = 10 cycle，2500us = 25 cycle，5000us = 50 cycle。
"""

import random
import unittest

from cqf.data_io import parse_topology, parse_flows, Stream
from cqf.network import build_graph
from cqf.model import CqfModel, REASON_DEADLINE, REASON_CAPACITY, REASON_NO_PATH
from cqf.algorithms import GreedyScheduler, RandomScheduler, order_streams
from cqf.validate import simulate_and_validate

TOPO_LINES = [
    'vertex,SWITCH,sw1,mac,00,PortNumber,4',
    'vertex,SWITCH,sw2,mac,01,PortNumber,4',
    'vertex,SWITCH,sw3,mac,02,PortNumber,4',
    'vertex,PLC,esA,mac,10,PortNumber,1',
    'vertex,PLC,esB,mac,11,PortNumber,1',
    'vertex,PLC,esC,mac,12,PortNumber,1',
    'edge,WIRE,sw1.P0,sw2.P0,undirect,e1',
    'edge,WIRE,sw1.P1,sw3.P0,undirect,e2',
    'edge,WIRE,sw3.P1,sw2.P1,undirect,e3',
    'edge,WIRE,sw1.P2,esA,undirect,e4',
    'edge,WIRE,sw2.P2,esB,undirect,e5',
    'edge,WIRE,sw1.P3,esC,undirect,e6',
]

T = 100  # us


def flow(fid, src, dst, period_us, deadline_us, size_bytes):
    """按导师文件的 13 列格式拼一行流记录。"""
    return 'FLOW,0,%d,Flow_%d,ISOCHRONOUS_REAL_TIME,%s,%s,NO,%d,MICRO_SECOND,%d,MICRO_SECOND,%d' % (
        fid, fid, src, dst, period_us, deadline_us, size_bytes)


def make_model(flow_lines, **kwargs):
    graph = build_graph(parse_topology(TOPO_LINES))
    streams = parse_flows(flow_lines)
    return CqfModel(graph, streams, cycle_time_us=T, **kwargs), streams


class TestParsingAndPaths(unittest.TestCase):

    def test_capacity_and_paths(self):
        model, (s,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)])
        self.assertAlmostEqual(model.bytes_per_cycle, 11250.0)
        paths = model.paths[s.id]
        names = [model.graph.vertex_ids(p) for p in paths]
        self.assertEqual(names, [['esA', 'sw1', 'sw2', 'esB'], ['esA', 'sw1', 'sw3', 'sw2', 'esB']])
        self.assertEqual(model.period_cycles[s.id], 10)
        self.assertEqual(model.deadline_cycles[s.id], 8)       # floor(800/100)
        self.assertEqual(model.hyperperiod_cycles, 10)

    def test_reverse_direction_paths_are_reversed(self):
        model, (s,) = make_model([flow(0, 'esB', 'esA', 1000, 800, 500)])
        names = [model.graph.vertex_ids(p) for p in model.paths[s.id]]
        self.assertEqual(names[0], ['esB', 'sw2', 'sw1', 'esA'])


class TestTiming(unittest.TestCase):
    """逐跳时序、队列归属、端到端时延的手算验证。"""

    def test_slots_queues_e2e(self):
        model, (s,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)])
        path = model.paths[s.id][0]                      # esA, sw1, sw2, esB
        # offset 0：esA 在 cycle 0 发出；sw1 在 cycle 0 收、cycle 1 发给 sw2；sw2 在 cycle 1 收、cycle 2 发给 esB
        sw1, sw2, esB = path[1], path[2], path[3]
        self.assertEqual(model.frame_slots(s, path, 0), [(1, (sw1, sw2)), (2, (sw2, esB))])
        # 队列：sw1 在偶数 cycle 0 收到 -> q2（even）；sw2 在奇数 cycle 1 收到 -> q1（odd）
        self.assertEqual(model.queue_sequence(s, path, 0), ['q2', 'q1'])
        self.assertEqual(model.queue_sequence(s, path, 1), ['q1', 'q2'])   # offset 变 1，奇偶互换
        # e2e = 1 + 2 台交换机 + offset
        self.assertEqual(model.e2e_cycles(s, path, 0), 3)
        self.assertEqual(model.e2e_cycles(s, path, 3), 6)

    def test_queue_flips_between_instances_when_period_is_odd(self):
        # 周期 2500us = 25 cycle（奇数）：第 0 个实例在 cycle 0 注入，第 1 个在 cycle 25 注入，奇偶相反。
        # 周期 1000us = 10 cycle（偶数）：所有实例奇偶相同。
        model, (odd, even) = make_model([flow(0, 'esA', 'esB', 2500, 2500, 500),
                                         flow(1, 'esA', 'esB', 1000, 800, 500)])
        path = model.paths[odd.id][0]
        self.assertEqual(model.queue_sequence(odd, path, 0, instance=0), ['q2', 'q1'])
        self.assertEqual(model.queue_sequence(odd, path, 0, instance=1), ['q1', 'q2'])
        self.assertEqual(model.queue_sequence(even, path, 0, instance=0), ['q2', 'q1'])
        self.assertEqual(model.queue_sequence(even, path, 0, instance=1), ['q2', 'q1'])
        # 独立模拟器给出的逐实例队列必须与公式一致
        model.commit(odd, 0, 0)
        model.commit(even, 0, 0)
        report = simulate_and_validate(model)
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual(report['queues'][(odd.id, 0)], ['q2', 'q1'])
        self.assertEqual(report['queues'][(odd.id, 1)], ['q1', 'q2'])
        self.assertEqual(report['queues'][(even.id, 3)], ['q2', 'q1'])
        self.assertEqual(model.hyperperiod_cycles, 50)
        self.assertEqual(model.queue_phase_period, 50)      # H 为偶数，even/odd 标签以 H 为周期

    def test_source_link_option(self):
        model, (s,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)], count_source_link=True)
        path = model.paths[s.id][0]
        esA, sw1 = path[0], path[1]
        self.assertEqual(model.frame_slots(s, path, 0)[0], (0, (esA, sw1)))


class TestCheckCommitRelease(unittest.TestCase):

    def test_check_is_read_only(self):
        model, (s,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)])
        self.assertEqual(model.check(s, 0, 0), (True, None))
        self.assertEqual(model.usage, {})
        self.assertEqual(model.assignments, {})

    def test_capacity_conflict_then_other_offset_or_route(self):
        # 两条 6000 字节的流，同路同周期：合计 12000 > 11250，不能挤在同一 cycle
        model, (a, b) = make_model([flow(0, 'esA', 'esB', 1000, 800, 6000),
                                    flow(1, 'esA', 'esB', 1000, 800, 6000)])
        model.commit(a, 0, 0)
        self.assertEqual(model.check(b, 0, 0), (False, REASON_CAPACITY))   # 同 offset 同路：冲突
        self.assertEqual(model.check(b, 0, 1), (True, None))               # 换 offset：错开一个 cycle
        self.assertEqual(model.check(b, 1, 0), (True, None))               # 换路径（经 sw3）：链路不同
        feasible, reasons = model.feasible_actions(b)
        self.assertNotIn((0, 0), feasible)
        self.assertIn((0, 1), feasible)
        self.assertEqual(reasons[(0, 0)], REASON_CAPACITY)

    def test_frame_larger_than_capacity(self):
        model, (s,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 12000)])
        self.assertEqual(model.check(s, 0, 0), (False, REASON_CAPACITY))

    def test_deadline(self):
        # 路径 0 有 2 台交换机：e2e = 3 cycle = 300us
        model, (tight, exact) = make_model([flow(0, 'esA', 'esB', 1000, 250, 500),
                                            flow(1, 'esA', 'esB', 1000, 300, 500)])
        self.assertEqual(model.check(tight, 0, 0), (False, REASON_DEADLINE))  # floor(250/100)=2 < 3
        self.assertEqual(model.check(exact, 0, 0), (True, None))              # 3 <= 3
        self.assertEqual(model.check(exact, 0, 1), (False, REASON_DEADLINE))  # offset 1 -> e2e 4
        self.assertEqual(model.check(exact, 1, 0), (False, REASON_DEADLINE))  # 3 台交换机 -> e2e 4

    def test_periodic_repetition_conflict(self):
        # A：周期 10 cycle，offset 0，在 sw1->sw2 占 cycle 1, 11, 21, 31, 41（超周期 lcm(10,25)=50）
        # B：周期 25 cycle。offset 5 时第一帧占 cycle 6（无冲突），第二帧占 cycle 31，与 A 的第 4 个实例冲突。
        model, (a, b) = make_model([flow(0, 'esA', 'esB', 1000, 2000, 6000),
                                    flow(1, 'esA', 'esB', 2500, 2500, 6000)])
        self.assertEqual(model.hyperperiod_cycles, 50)
        model.commit(a, 0, 0)
        path = model.paths[b.id][0]
        self.assertEqual([c for c, _ in model.frame_slots(b, path, 5)], [6, 7, 31, 32])
        self.assertEqual(model.check(b, 0, 5), (False, REASON_CAPACITY))   # 只查第一帧会漏掉这个冲突
        self.assertEqual(model.check(b, 0, 1), (True, None))               # cycle 2, 27：都空

    def test_wraparound_modulo_hyperperiod(self):
        # 超周期 50 cycle。A offset 49：sw1 在 cycle 50 发 -> 取模后是 cycle 0。
        # C（esC->esB）offset 0 在 cycle 1 用 sw1->sw2，不冲突；offset 49 则同样落在 cycle 0 -> 冲突。
        model, (a, c) = make_model([flow(0, 'esA', 'esB', 5000, 6000, 6000),
                                    flow(1, 'esC', 'esB', 5000, 6000, 6000)])
        model.commit(a, 0, 49)
        path_a = model.paths[a.id][0]
        self.assertEqual([cyc for cyc, _ in model.frame_slots(a, path_a, 49)], [0, 1])
        self.assertEqual(model.check(c, 0, 0), (True, None))
        self.assertEqual(model.check(c, 0, 49), (False, REASON_CAPACITY))

    def test_release_leaves_no_residue(self):
        model, (a, b) = make_model([flow(0, 'esA', 'esB', 1000, 800, 6000),
                                    flow(1, 'esA', 'esB', 1000, 800, 6000)])
        model.commit(a, 0, 0)
        self.assertEqual(model.check(b, 0, 0), (False, REASON_CAPACITY))
        model.release(a)
        self.assertEqual(model.usage, {})
        self.assertEqual(model.assignments, {})
        self.assertEqual(model.check(b, 0, 0), (True, None))

    def test_release_detects_corrupted_ledger(self):
        # 账本被外部改小后 release 会减出负数，这是损坏信号，必须报错而不是静默删掉
        model, (a,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)])
        model.commit(a, 0, 0)
        slot = next(iter(model.usage))
        model.usage[slot] = 100
        with self.assertRaises(RuntimeError):
            model.release(a)

    def test_commit_refuses_infeasible(self):
        model, (a, b) = make_model([flow(0, 'esA', 'esB', 1000, 800, 6000),
                                    flow(1, 'esA', 'esB', 1000, 800, 6000)])
        model.commit(a, 0, 0)
        with self.assertRaises(ValueError):
            model.commit(b, 0, 0)
        self.assertNotIn(b.id, model.assignments)      # 失败的提交不留任何痕迹
        self.assertEqual(sum(model.usage.values()), 2 * 6000)

    def test_offset_range(self):
        model, (s,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)], max_offset_cycles=3)
        self.assertEqual(list(model.offset_range(s)), [0, 1, 2, 3])
        self.assertEqual(model.check(s, 0, 4)[1], 'offset_range')

    def test_invalid_action_encoding_raises(self):
        # 越界下标、非整数 offset 是程序错误，必须抛异常，不能被当作"最后一条路径"之类的合法动作
        model, (s,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)])
        with self.assertRaises(ValueError):
            model.check(s, -1, 0)
        with self.assertRaises(ValueError):
            model.check(s, 2, 0)
        with self.assertRaises(TypeError):
            model.check(s, 0, 1.5)
        with self.assertRaises(ValueError):
            model.commit(s, -1, 0)
        self.assertEqual(model.assignments, {})


class TestInputValidation(unittest.TestCase):
    """非法输入必须在解析或建模时报错，不能算出一个"合法"的结果。"""

    def test_parse_rejects_bad_records(self):
        with self.assertRaises(ValueError):
            parse_flows([flow(0, 'esA', 'esB', 1000, 800, -500)])      # 负帧长
        with self.assertRaises(ValueError):
            parse_flows([flow(0, 'esA', 'esB', 0, 800, 500)])         # 周期为 0
        with self.assertRaises(ValueError):
            parse_flows([flow(0, 'esA', 'esB', 1000, 800, 500),
                         flow(0, 'esC', 'esB', 1000, 800, 500)])      # id 重复
        with self.assertRaises(ValueError):
            parse_flows([flow(0, 'esA', 'esA', 1000, 800, 500)])      # 收发端相同

    def test_model_rejects_bad_streams_and_params(self):
        graph = build_graph(parse_topology(TOPO_LINES))
        with self.assertRaises(ValueError):                           # 绕过解析直接构造负帧长
            CqfModel(graph, [Stream(0, 'x', 'esA', 'esB', 1000, 800, -500)], cycle_time_us=T)
        with self.assertRaises(ValueError):                           # 端点不在拓扑中
            CqfModel(graph, [Stream(0, 'x', 'esA', 'esZ', 1000, 800, 500)], cycle_time_us=T)
        with self.assertRaises(ValueError):                           # 周期不被 T 整除
            CqfModel(graph, [Stream(0, 'x', 'esA', 'esB', 1050, 800, 500)], cycle_time_us=T)
        good = [Stream(0, 'x', 'esA', 'esB', 1000, 800, 500)]
        with self.assertRaises(ValueError):
            CqfModel(graph, good, cycle_time_us=T, bw_fraction=1.5)  # 带宽占比超过 1
        with self.assertRaises(ValueError):
            CqfModel(graph, good, cycle_time_us=T, k_paths=0)        # 候选路径数为 0

    def test_queue_names_must_be_two_distinct(self):
        graph = build_graph(parse_topology(TOPO_LINES))
        good = [Stream(0, 'x', 'esA', 'esB', 1000, 800, 500)]
        with self.assertRaises(ValueError):
            CqfModel(graph, good, cycle_time_us=T, queue_names=('q7', 'q7'))
        with self.assertRaises(ValueError):
            CqfModel(graph, good, cycle_time_us=T, queue_names=('q7', ''))
        with self.assertRaises(ValueError):                           # 去掉前缀 q 后编号相同
            CqfModel(graph, good, cycle_time_us=T, queue_names=('q7', '7'))

    def test_export_does_not_require_numeric_flow_names(self):
        # 流名任意（不以数字结尾）时，csv 按流 id 排序，不应在导出阶段崩溃
        import tempfile
        from cqf import export
        lines = [flow(1, 'esA', 'esB', 1000, 800, 500).replace('Flow_1', 'ControlSignal'),
                 flow(0, 'esC', 'esB', 1000, 800, 500).replace('Flow_0', 'Alarm')]
        model, streams = make_model(lines)
        GreedyScheduler().run(model, streams)
        with tempfile.TemporaryDirectory() as d:
            path = export.write_result_csv(d, 'x', model)
            with open(path, encoding='utf-8') as f:
                names = [line.split(',')[0] for line in f.read().splitlines()[1:]]
        self.assertEqual(names, ['Alarm', 'ControlSignal'])          # id 0 在前

    def test_end_stations_never_forward(self):
        # 把 sw3 改成 PLC 并去掉 sw1-sw2 直连：esA 到 esB 只剩"经 PLC 转发"的路，必须视为无路
        lines = [l.replace('vertex,SWITCH,sw3', 'vertex,PLC,sw3') for l in TOPO_LINES if not l.endswith('e1')]
        graph = build_graph(parse_topology(lines))
        self.assertEqual(graph.candidate_paths('esA', 'esB', 5), [])

    def test_direct_es_link_is_rejected(self):
        # 两个端站直连：模型明确不支持，建模时报错而不是把它当成 0 台交换机的路径
        lines = ['vertex,PLC,esA,mac,10,PortNumber,1', 'vertex,PLC,esB,mac,11,PortNumber,1',
                 'edge,WIRE,esA,esB,undirect,e1']
        graph = build_graph(parse_topology(lines))
        with self.assertRaises(ValueError):
            CqfModel(graph, [Stream(0, 'x', 'esA', 'esB', 1000, 800, 500)], cycle_time_us=T)

    def test_no_path_is_reported_separately(self):
        # 去掉 esB 的接入链路 e5 后，esA -> esB 没有任何路径，应记为 no_path 而不是 deadline
        topo = parse_topology([l for l in TOPO_LINES if not l.endswith('e5')])
        graph = build_graph(topo)
        streams = parse_flows([flow(0, 'esA', 'esB', 1000, 800, 500)])
        model = CqfModel(graph, streams, cycle_time_us=T)
        self.assertEqual(model.paths[0], [])
        res = GreedyScheduler().run(model, streams)
        self.assertEqual(res.rejected, {0: REASON_NO_PATH})


class TestIndependentSimulation(unittest.TestCase):
    """validate.simulate_and_validate 不用内核公式，逐 cycle 模拟；它既要认可正确的调度，也要抓出被篡改的账本。"""

    def _schedule(self, lines):
        model, streams = make_model(lines)
        res = GreedyScheduler().run(model, streams)
        return model, streams, res

    def test_agrees_with_kernel_on_wraparound_and_repetition(self):
        model, (a, b, c), _ = self._schedule([flow(0, 'esA', 'esB', 5000, 6000, 6000),
                                              flow(1, 'esC', 'esB', 1000, 2000, 3000),
                                              flow(2, 'esA', 'esB', 2500, 2500, 2000)])
        model.release(a)
        model.commit(a, 0, 49)                      # 跨越超周期末尾的实例
        self.assertTrue(model.verify()['ok'])
        report = simulate_and_validate(model)
        self.assertTrue(report['ok'], report['problems'])

    def test_catches_corrupted_ledger(self):
        model, (a,), _ = self._schedule([flow(0, 'esA', 'esB', 1000, 800, 500)])
        slot = next(iter(model.usage))
        model.usage[slot] += 1                       # 篡改账本：verify 和模拟都应报错
        self.assertFalse(model.verify()['ok'])
        self.assertFalse(simulate_and_validate(model)['ok'])

    def test_warmup_covers_paths_longer_than_hyperperiod(self):
        # 周期 100us = 1 cycle，H = 1，路径却有 3 台交换机：稳态下每拍三条出链路各有一帧。
        # 预热若只有 1 拍，统计窗口开头后两跳会少算；模拟器必须按最长路径加长预热。
        model, (s,) = make_model([flow(0, 'esA', 'esB', 100, 800, 500)])
        self.assertEqual(model.hyperperiod_cycles, 1)
        model.commit(s, 1, 0)                        # 路径 1：esA,sw1,sw3,sw2,esB
        self.assertEqual(len(model.usage), 3)
        report = simulate_and_validate(model)
        self.assertTrue(report['ok'], report['problems'])

    def test_odd_hyperperiod_instance_labels(self):
        # 周期 300us = 3 cycle，H = 3（奇数）：队列标签要 lcm(3,2) = 6 拍才重复。
        # 模拟器返回的实例 k=0 必须与公式的 instance=0 一致，k=1（第二个超周期）奇偶相反。
        model, (s,) = make_model([flow(0, 'esA', 'esB', 300, 800, 500)])
        self.assertEqual(model.queue_phase_period, 6)
        model.commit(s, 0, 0)
        report = simulate_and_validate(model)
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual(report['queues'][(s.id, 0)], ['q2', 'q1'])
        self.assertEqual(report['queues'][(s.id, 1)], ['q1', 'q2'])
        self.assertEqual(sorted(k for _, k in report['queues']), [0, 1])

    def test_catches_wrong_queue_formula(self):
        # 导出 csv 的队列号来自 queue_sequence；把它改坏后模拟器必须报不一致
        model, (s,), _ = self._schedule([flow(0, 'esA', 'esB', 1000, 800, 500)])
        model.queue_sequence = lambda stream, path, offset, instance=0: ['q7'] * model.hops(path)
        self.assertFalse(simulate_and_validate(model)['ok'])

    def test_invalid_assignment_returns_problems_instead_of_crashing(self):
        # 篡改成越界 / 非整数的决策：验证器必须返回 ok=False 并列出问题，而不是在取路径时抛异常
        from cqf.model import Assignment
        for bad in (Assignment(99, 0), Assignment(0.5, 0), Assignment(-1, 0), Assignment(0, 2.0)):
            model, (s,), _ = self._schedule([flow(0, 'esA', 'esB', 1000, 800, 500)])
            model.assignments[s.id] = bad
            report = simulate_and_validate(model)
            self.assertFalse(report['ok'])
            self.assertTrue(report['problems'], bad)

    def test_catches_wrong_formula(self):
        # 模拟不依赖 frame_slots：把 frame_slots 换成一个漏掉最后一跳的错误公式，verify 会"自洽"地通过，模拟则报错
        model, (a,), _ = self._schedule([flow(0, 'esA', 'esB', 1000, 800, 500)])
        good_slots = model.frame_slots
        model.frame_slots = lambda s, p, o: good_slots(s, p, o)[:-1]
        model.usage = {}
        for slot in model.frame_slots(a, model.paths[a.id][0], 0):
            model.usage[slot] = a.size_bytes
        self.assertTrue(model.verify()['ok'])                     # 共用公式，查不出来
        self.assertFalse(simulate_and_validate(model)['ok'])      # 独立模拟查得出来


class TestVerifyAndAlgorithms(unittest.TestCase):

    def _many_flows(self):
        rng = random.Random(1)
        lines = []
        for i in range(40):
            src, dst = rng.choice([('esA', 'esB'), ('esC', 'esB'), ('esB', 'esA'), ('esA', 'esC')])
            lines.append(flow(i, src, dst, rng.choice([1000, 2500, 5000]), rng.choice([500, 800, 1500]),
                              rng.choice([500, 1500, 3000, 6000])))
        return lines

    def test_verify_after_random_commit_release(self):
        model, streams = make_model(self._many_flows())
        rng = random.Random(7)
        for _ in range(200):
            s = rng.choice(streams)
            if s.id in model.assignments:
                model.release(s)
            else:
                feasible, _ = model.feasible_actions(s)
                if feasible:
                    model.commit(s, *rng.choice(feasible))
        report = model.verify()
        self.assertTrue(report['ok'], report['problems'])
        self.assertTrue(simulate_and_validate(model)['ok'])

    def test_random_and_greedy_pass_verify_and_are_reproducible(self):
        for cls in (RandomScheduler, GreedyScheduler):
            results = []
            for _ in range(2):
                model, streams = make_model(self._many_flows())
                ordered = order_streams(streams, 'random', random.Random(3))
                res = cls(seed=5).run(model, ordered)
                self.assertTrue(model.verify()['ok'])
                self.assertTrue(simulate_and_validate(model)['ok'])
                self.assertEqual(len(res.scheduled) + len(res.rejected), len(streams))
                results.append((res.scheduled, sorted(model.assignments.items())))
            self.assertEqual(results[0], results[1])    # 同种子两次结果一致

    def test_greedy_prefers_smallest_e2e(self):
        model, (a,) = make_model([flow(0, 'esA', 'esB', 1000, 800, 500)])
        res = GreedyScheduler().run(model, [a])
        self.assertEqual(model.assignments[a.id].path_idx, 0)
        self.assertEqual(model.assignments[a.id].offset, 0)
        self.assertEqual(res.rejected, {})

    def test_rejection_reasons(self):
        model, (a, b, c) = make_model([flow(0, 'esA', 'esB', 1000, 800, 6000),
                                       flow(1, 'esA', 'esB', 1000, 300, 6000),
                                       flow(2, 'esA', 'esB', 1000, 250, 500)])
        res = GreedyScheduler().run(model, [a, b, c])
        # b：deadline 只允许 (路径0, offset0)，但该槽已被 a 占满 -> 没有可行组合，且并非全部因 deadline 失败 -> capacity
        self.assertEqual(res.rejected[b.id], REASON_CAPACITY)
        # c：所有组合都因 deadline 失败
        self.assertEqual(res.rejected[c.id], REASON_DEADLINE)


if __name__ == '__main__':
    unittest.main()
