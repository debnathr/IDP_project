"""网络拓扑：顶点、链路、图，以及候选路径（k 条最短路径）。

本文件沿用 gasa 项目 PriorityGroup.py 中的 Vertex / Link / Graph 三个类和
GroupModel.generate_graph 的建图逻辑，做了以下改动：

1. 去掉了 Link / Vertex 中与 Multi-CQF 优先级组有关的字段（k_priority、queue_index），
   这些字段在 gasa 中只是随手记录，没有参与路径计算。
2. gasa 用 V×V 邻接矩阵，每次扩展都要扫描全部顶点；这里改为"按顶点编号排序的邻居列表"。
   因为 gasa 扫描邻居时也是按顶点编号从小到大，所以两种写法得到的候选路径及其顺序完全相同，只是更快。
3. gasa 在建图时一次性计算所有端站对的候选路径；这里改为"用到哪一对就算哪一对并缓存"，
   结果相同，但大拓扑（如 TC25 有 92 个端站）启动更快。
4. gasa 只对端站顺序 i<j 的方向真正搜索，j->i 方向直接把正向路径倒序；这里保留同样的约定，
   保证候选路径集合与 gasa 一致（当存在等长路径时，正向搜索和反向搜索的结果可能不同）。
"""

import heapq


class Vertex:
    """一个网络节点：端站（es）或交换机（sw）。"""

    def __init__(self, device_idx, device_type, device_id, address, port_number):
        self.device_idx = device_idx      # 在图中的整数编号（顶点下标）
        self.dt = device_type             # 'es' 端站 / 'sw' 交换机
        self.id = device_id               # 文件中的名字，如 sw_0_0、node0_0_4_1
        self.address = address            # mac 地址（仅记录）
        self.port = port_number           # 端口数（仅记录）
        assert device_type in ('es', 'sw'), 'device_type must be es (end station) or sw (switch)'

    def __repr__(self):
        return 'Vertex(%d, %s, %s)' % (self.device_idx, self.dt, self.id)


class Link:
    """一条无向物理链路。e = [u, v] 为两端顶点编号，p = [port_u, port_v] 为两端端口名。"""

    def __init__(self, node_u, node_v, port_u, port_v, name=''):
        self.e = [node_u, node_v]
        self.p = [port_u, port_v]
        self.name = name                  # 文件中的链路名，如 e4（输出 csv 要用）

    def __repr__(self):
        return 'Link(%s: %d.%s <-> %d.%s)' % (self.name, self.e[0], self.p[0], self.e[1], self.p[1])


class Graph:
    """拓扑图 + 候选路径。路径用顶点编号列表表示：[源端站, 交换机, ..., 交换机, 目的端站]。"""

    def __init__(self, all_vertices, all_links):
        self.all_vertices = all_vertices
        self.all_links = all_links
        self.num_vertices = len(all_vertices)

        # 邻居列表（排序后与 gasa 的邻接矩阵扫描顺序一致）
        self.neighbors = [[] for _ in range(self.num_vertices)]
        # (u, v) -> (链路名, u 侧端口)：导出 csv 时需要知道交换机从哪个端口、哪条链路发出
        self.link_lookup = {}
        for link in all_links:
            u, v = link.e
            self.neighbors[u].append(v)
            self.neighbors[v].append(u)
            self.link_lookup[(u, v)] = (link.name, link.p[0])
            self.link_lookup[(v, u)] = (link.name, link.p[1])
        for lst in self.neighbors:
            lst.sort()

        self.vertex_by_id = {v.id: v for v in all_vertices}
        self.is_switch = [v.dt == 'sw' for v in all_vertices]
        # gasa 以"端站在顶点列表中的出现顺序"决定搜索方向（i<j 正向搜索，否则倒序）
        self.end_stations = [v for v in all_vertices if v.dt == 'es']
        self.es_order = {v.id: i for i, v in enumerate(self.end_stations)}
        self._path_cache = {}

    def n_shortest_paths(self, start, end, n):
        """求 start -> end 的 n 条最短简单路径（沿用 gasa 的 Graph.n_shortest_paths）。

        做法：小根堆上的最佳优先搜索，堆元素为 (dist, 当前顶点, 已走路径)。
        起点 dist 取 -1，每走一条链路 +1，所以到达终点时 dist = 路径上的链路数 - 1 = 经过的交换机数。
        每次弹出的若是终点就记录一条路径；记录满 n 条即停止。禁止回到已在路径中的顶点（无环）。
        返回 [(dist, path), ...]，按长度从短到长（等长时按顶点编号字典序）。

        与 gasa 的一处不同：中间节点只允许交换机。gasa 会扩展任何邻居，但导师数据里端站只有 1 个端口，
        进入端站后无路可走，所以结果相同；这里显式禁止，是为了不把"经过端站转发"的路径当成合法路径。
        """
        paths = []
        heap = [(-1, start.device_idx, [start.device_idx])]
        while heap and len(paths) < n:
            dist, node, path = heapq.heappop(heap)
            if node == end.device_idx:
                paths.append((dist, path))
                continue
            for neighbor in self.neighbors[node]:
                if neighbor in path:
                    continue
                if not self.is_switch[neighbor] and neighbor != end.device_idx:
                    continue        # 端站不能作为中间节点
                heapq.heappush(heap, (dist + 1, neighbor, path + [neighbor]))
        return paths

    def candidate_paths(self, src_id, dst_id, n):
        """返回端站 src_id -> dst_id 的 n 条候选路径（顶点编号列表的列表），结果缓存。"""
        if src_id == dst_id:
            raise ValueError('source and destination are the same node: %s' % src_id)
        if src_id not in self.es_order or dst_id not in self.es_order:
            raise ValueError('endpoints must be end stations of the topology: %s -> %s' % (src_id, dst_id))
        key = (src_id, dst_id, n)
        if key not in self._path_cache:
            if self.es_order[src_id] < self.es_order[dst_id]:
                found = self.n_shortest_paths(self.vertex_by_id[src_id], self.vertex_by_id[dst_id], n)
                self._path_cache[key] = [path for _, path in found]
            else:
                # 与 gasa 相同：反方向直接把正向路径倒序
                forward = self.candidate_paths(dst_id, src_id, n)
                self._path_cache[key] = [path[::-1] for path in forward]
        return self._path_cache[key]

    def vertex_ids(self, path):
        """把顶点编号路径转成名字路径，便于打印与导出。"""
        return [self.all_vertices[idx].id for idx in path]

    def egress(self, u, v):
        """u -> v 这一跳用的链路名和 u 侧的出端口。"""
        return self.link_lookup[(u, v)]


def build_graph(topology):
    """由 data_io.parse_topology 的结果建图（沿用 gasa 的 GroupModel.generate_graph）。

    gasa 中 SWITCH 记为 'sw'，PLC 记为 'es'；其他类型的 vertex 会被忽略。
    边的两端形如 'sw_0_0.P0'（交换机带端口）或 'node0_0_0_0'（端站不带端口，gasa 记端口为 'ES'）。
    """
    nodes = topology['nodes']
    edges = topology['edges']

    all_vertices = []
    lut_vertices = {}
    for idx, node in enumerate(nodes):
        if node[1] == 'SWITCH':
            all_vertices.append(Vertex(idx, 'sw', node[2], node[4], node[6]))
            lut_vertices[node[2]] = idx
        elif node[1] == 'PLC':
            all_vertices.append(Vertex(idx, 'es', node[2], node[4], node[6]))
            lut_vertices[node[2]] = idx
    assert len(all_vertices) == len(nodes), 'topology contains node types other than SWITCH/PLC, must be confirmed first'

    all_links = []
    for edge in edges:
        node_start, port_start = _split_endpoint(edge[2])
        node_end, port_end = _split_endpoint(edge[3])
        all_links.append(Link(lut_vertices[node_start], lut_vertices[node_end],
                              port_start, port_end, name=edge[-1]))
    return Graph(all_vertices, all_links)


def _split_endpoint(text):
    """'sw_0_0.P0' -> ('sw_0_0', 'P0')；'node0_0_0_0' -> ('node0_0_0_0', 'ES')。"""
    if '.' in text:
        node, port = text.split('.')
        return node, port
    return text, 'ES'
