"""cqf 包：本科毕业论文项目的 CQF 调度核心（第一阶段）。

包结构：

- data_io.py   读取导师提供的拓扑文件与流文件，整理成 Stream 对象（沿用 gasa 的解析逻辑）
- network.py   顶点 / 链路 / 图 以及 k 条最短候选路径（沿用 gasa 的 Graph 实现）
- model.py     CQF 调度内核：资源表、可行性检查(check)、提交(commit)、撤销(release)、整体验证(verify)
- export.py    结果导出，格式与 gasa 项目的 export_log_*.txt / res_*_sorted.csv / time*_time.txt 一致
- algorithms/  可替换的调度算法（随机、贪心；后续替换为强化学习）
"""
