"""cqf.rl：强化学习部分（第二阶段）。

- features.py  动作掩码与观测的向量化计算（numpy），维护与内核账本同步的 load 镜像
- env.py       Gymnasium 环境 CqfEnv：一个 episode = 按固定顺序处理完一个测试用例的全部流
- train.py     用 sb3_contrib.MaskablePPO 训练，按用例划分训练 / 验证，保存 model.zip + config.json

这里只依赖 numpy / gymnasium / sb3_contrib（torch）；内核 cqf.model 与基线算法不依赖它们。
"""
