# MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction

**会议**: arXiv 2026  
**arXiv**: [2603.10688](https://arxiv.org/abs/2603.10688)  
**作者**: Jonas Merkert, Alexander Blumberg, Jan-Hendrik Pauls, Christoph Stiller
**代码**: 待确认  
**领域**: 语义分割 / 自监督/表示学习  
**关键词**: mapgclr, geospatial, contrastive, learning, representations  

## 一句话总结
自动驾驶汽车依靠地图信息来了解周围的世界。
## 背景与动机
Autonomous vehicles rely on map information to understand the world around them.. However, the creation and maintenance of offline high-definition (HD) maps remains costly.

## 核心问题
然而，离线高清（HD）地图的创建和维护成本仍然很高。
## 方法详解

### 整体框架
- This work focuses on improving the latent birds-eye-view (BEV) feature grid representation within a vectorized online HD map construction model by enforcing geospatial consistency between overlapping BEV feature grids as part of a contrastive loss function.
- To ensure geospatial overlap for contrastive pairs, we introduce an approach to analyze the overlap between traversals within a given dataset and generate subsidiary dataset splits following adjustable multi-traversal requirements.
- Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.

### 关键设计
1. **关键组件1**: Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 我们的方法全面优于监督基线，无论是在下游任务矢量化地图感知性能方面的定量方面，还是在 BEV 特征空间的主成分分析 (PCA) 可视化分割方面的定性方面。
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
