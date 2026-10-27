"""读取导师提供的拓扑文件与流文件。

文件格式与 gasa 项目完全一致（已对照 TCs_for_TSN/SA_TCs/TC*_topo.txt、TC*_flows.txt 核实）：

拓扑文件每行一条记录，逗号分隔：
    vertex,SWITCH,sw_0_0,mac,00:00:00:00:00:00,PortNumber,8      # 交换机
    vertex,PLC,node0_0_0_0,mac,00:00:00:00:00:10,PortNumber,1    # 端站（PLC = 终端设备）
    edge,WIRE,sw_0_0.P0,sw_0_1.P0,undirect,e1                    # 链路：两端为 "节点.端口"，端站没有端口号
    edge,WIRE,sw_0_0.P3,node0_0_0_0,undirect,e4

流文件每行一条流，共 13 列：
    下标: 0    1  2  3             4                      5           6           7  8    9            10  11           12
    示例: FLOW,0, 0, VLAN_0_Flow_0,ISOCHRONOUS_REAL_TIME,node0_0_4_1,node0_0_7_0,NO,2500,MICRO_SECOND,521,MICRO_SECOND,228
    含义:      vlan id 名字          流类型                 发送端(talker) 接收端      -  周期  周期单位      截止  截止单位      帧长(字节)

gasa 的 PriorityGroup.generate_rs_space 取用的列为 [2] id、[3] name、[5] talker、[6] receiver、
[8] period、[10] max_delay、[12] size，本文件沿用同样的列。

本文件对应 gasa 项目 GaSaAlgorithm.py 中的 load_graph_stream_data / process_graph_data /
process_flow_data 三个函数；区别只是把"字符串列表"整理成带字段名的 Stream 对象，便于阅读。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stream:
    """一条周期性流（只保存文件中的原始物理量，单位：微秒、字节）。

    与 cycle 相关的换算（周期有几个 cycle、deadline 有几个 cycle）放在 model.py 里做，
    因为它们依赖实验选定的 cycle 长度 T，不属于输入数据本身。
    """

    id: int            # 流编号（文件第 2 列），输出 csv 时按它排序
    name: str          # 流名字，如 VLAN_0_Flow_0
    src: str           # 发送端站 id（talker）
    dst: str           # 接收端站 id（receiver / listener）
    period_us: int     # 周期，微秒
    deadline_us: int   # 端到端最大时延（deadline），微秒
    size_bytes: int    # 每个周期发送一帧，帧长（字节）


def read_lines(path):
    """按行读取文本文件（对应 gasa 的 load_graph_stream_data 中的 readlines）。"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def parse_topology(lines):
    """把拓扑文件的行拆成 {'nodes': [...], 'edges': [...]}，与 gasa 的 process_graph_data 完全一致。

    每个元素仍是"按逗号拆开的字段列表"，交给 network.build_graph 进一步解释。
    """
    nodes = []
    edges = []
    for line in lines:
        fields = line.rstrip().split(',')
        if fields[0] == 'vertex':
            nodes.append(fields)
        elif fields[0] == 'edge':
            edges.append(fields)
    return {'nodes': nodes, 'edges': edges}


def parse_flows(lines):
    """把流文件的行解析成 Stream 列表（保持文件顺序）。

    与 gasa 的 process_flow_data 相比多做了三件事：
    1. 把字符串转成 int；
    2. 检查时间单位确实是 MICRO_SECOND，避免单位理解错误却悄悄算下去；
    3. 拒绝明显非法的记录：周期 / deadline / 帧长必须为正，流 id 不能重复，发送端与接收端不能相同。
       这些在导师数据里都不会出现，但一旦出现（例如手写测试数据写错），应当立刻报错而不是算出一个"合法"结果。
    """
    streams = []
    seen_ids = set()
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        fields = line.split(',')
        if fields[0] != 'FLOW' or len(fields) != 13:
            raise ValueError('unrecognised flow record: %r' % line)
        if fields[9] != 'MICRO_SECOND' or fields[11] != 'MICRO_SECOND':
            raise ValueError('flow %s: time unit is not MICRO_SECOND, unit must be confirmed first: %r' % (fields[3], line))
        stream = Stream(
            id=int(fields[2]),
            name=fields[3],
            src=fields[5],
            dst=fields[6],
            period_us=int(fields[8]),
            deadline_us=int(fields[10]),
            size_bytes=int(fields[12]),
        )
        if stream.period_us <= 0 or stream.deadline_us <= 0 or stream.size_bytes <= 0:
            raise ValueError('flow %s: period / deadline / frame size must be positive: %r' % (stream.name, line))
        if stream.id in seen_ids:
            raise ValueError('duplicate flow id %d: %r' % (stream.id, line))
        if stream.src == stream.dst:
            raise ValueError('flow %s: source and destination are the same node: %r' % (stream.name, line))
        seen_ids.add(stream.id)
        streams.append(stream)
    return streams


def load_test_case(topo_path, flows_path):
    """一次读入一个测试用例，返回 (topology_dict, streams)。"""
    topology = parse_topology(read_lines(topo_path))
    streams = parse_flows(read_lines(flows_path))
    return topology, streams
