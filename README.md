# audio-dsp-graph

## 题面

把效果器串成 DSP 图后，并行支路一合成就出现梳状滤波、还会卡死：拓扑排序按节点插入顺序、依赖被忽略；环检测只在同层查、跨层环漏检；缓冲大小按节点数分配、大块溢出；并行支路不补偿延迟、求和出现梳状；块处理不累加 state、跨块断音；重建每次全量、帧率掉一半；未知端点静默忽略、丢边；自环不报错、死循环；批量共用 state、图之间串；block_size=0 不校验、除零。dsp/graph.py 是核心，tests/test_graph.py 15个红。要修的是：实现遵守依赖与环检测的拓扑排序，按块大小和扇出规划独立缓冲，按采样数把并行支路对齐到最大延迟，逐块累计状态保持连续，只重建受影响子图，显式校验未知端点、自环和重复边，并在批量处理中隔离每个图的状态、任一图非法时整体回滚且拒绝零块大小。

## 运行方式

```powershell
cd D:\gsb-20261119\audio-dsp-graph
python -m pytest tests -q
```

初始状态必须恰好显示 `15 failed`，且不得出现 collection、import 或 syntax error。`dsp/graph.py` 中的 8 个公开函数目前全部 `raise NotImplementedError`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 测试约束

`tests/test_graph.py` 不可修改；只能修改 `dsp/graph.py`，使 15 个测试全部通过。
