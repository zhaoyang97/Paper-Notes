# An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

**会议**: 投稿中  
**arXiv**: [2603.10671](https://arxiv.org/abs/2603.10671)  
**代码**: 未提及  
**领域**: 模型压缩  
**关键词**: fpga, implementation, displacement, vector, search  

## 一句话总结
在本文中，我们为DV搜索模块提出了一种高效的流水线FPGA架构设计，以促进IPC的实际部署。

## 核心问题
然而，DV搜索过程计算量大，给实际硬件部署带来了挑战。

## 关键方法
1. 在本文中，我们提出了一种针对DV搜索模块的高效流水线FPGA架构设计，以促进IPC的实际部署
2. IPC 执行小波域帧内补偿预测以减少屏幕内容中的空间冗余
3. IPC的一个关键模块是位移向量（DV）搜索，其目的是求解最优预测参考偏移
4. 然而，DV搜索过程计算量大，给实际硬件部署带来了挑战

## 亮点 / 我学到了什么
- 实验结果表明，我们提出的架构实现了 38.3 Mpixels/s 的吞吐量，功耗为 277 mW，证明了其在 IPC 和其他预测编码工具中实际硬件实现的可行性，并为 ASIC 部署提供了有希望的基础。

## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `agfu_foundation_pruning`
- `attention_aware_quant`
- `task_aware_token_compression`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
