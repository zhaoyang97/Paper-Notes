# SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning

**会议**: arXiv 2026  
**arXiv**: [2603.12976](https://arxiv.org/abs/2603.12976)  
**作者**: Md Anwar Hossen, Nathan R. Tallent, Luanzheng Guo, Ali Jannesary
**代码**: 待确认  
**领域**: 模型压缩/高效推理 / 隐私/安全/公平  
**关键词**: scope, semantic, coreset, orthogonal, projection  

## 一句话总结
科学发现越来越需要学习联合数据集，这些数据集由来自高分辨率仪器的流提供，这些数据集具有极端的类别不平衡。
## 背景与动机
Scientific discovery increasingly requires learning on federated datasets, fed by streams from high-resolution instruments, that have extreme class imbalance.. Current ML approaches either require impractical data aggregation or fail due to class imbalance.

## 核心问题
为了克服这些挑战，我们引入了 SCOPE（使用正交投影嵌入进行联邦学习的语义核心集），这是一个用于联邦数据的核心集框架，可过滤异常并自适应修剪冗余数据以减轻长尾偏差。
## 方法详解

### 整体框架
- Existing coreset selection methods rely on local heuristics, making them unaware of the global data landscape and prone to sub-optimal and non-representative pruning.
- To overcome these challenges, we introduce SCOPE (Semantic Coreset using Orthogonal Projection Embeddings for Federated learning), a coreset framework for federated data that filters anomalies and adaptively prunes redundant data to mitigate long-tail skew.
- By analyzing the latent space distribution, we score each data point using a representation score that measures the reliability of core class features, a diversity score that quantifies the novelty of orthogonal residuals, and a boundary proximity score that indicates similarity to competing classes.

### 关键设计
待深读后补充。

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 大量实验表明，SCOPE 具有具有竞争力的全局精度和强大的收敛性，同时实现了卓越的效率，上行链路带宽减少了 128 倍至 512 倍，挂钟加速提高了 7.72 倍，并减少了用于本地核心集选择的 FLOP 和 VRAM 占用空间。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
