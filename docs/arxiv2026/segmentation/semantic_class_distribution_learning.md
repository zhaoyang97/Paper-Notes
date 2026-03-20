# Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation

**会议**: arXiv 2026  
**arXiv**: [2603.05202](https://arxiv.org/abs/2603.05202)  
**作者**: Yingxue Su, Yiheng Zhong, Keying Zhu, Zimu Zhang, Zhuoru Zhang
**代码**: [https://github.com/Zyh55555/SCDL](https://github.com/Zyh55555/SCDL)  
**领域**: 语义分割  
**关键词**: semantic, class, distribution, learning, debiasing  

## 一句话总结
医学图像分割对于计算机辅助诊断至关重要。
## 背景与动机
Medical image segmentation is critical for computer-aided diagnosis.. However, dense pixel-level annotation is time-consuming and expensive, and medical datasets often exhibit severe class imbalance.

## 核心问题
然而，密集的像素级注释既耗时又昂贵，并且医学数据集通常表现出严重的类别不平衡。
## 方法详解

### 整体框架
- Such imbalance causes minority structures to be overwhelmed by dominant classes in feature representations, hindering the learning of discriminative features and making reliable segmentation particularly challenging.
- To address this, we propose the Semantic Class Distribution Learning (SCDL) framework, a plug-and-play module that mitigates supervision and representation biases by learning structured class-conditional feature distributions.
- SCDL integrates Class Distribution Bidirectional Alignment (CDBA) to align embeddings with learnable class proxies and leverages Semantic Anchor Constraints (SAC) to guide proxies using labeled data.

### 关键设计
1. **关键组件1**: To address this, we propose the Semantic Class Distribution Learning (SCDL) framework, a plug-and-play module that mitigates supervision and representation biases by learning structured class-conditional feature distributions.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- Synapse 和 AMOS 数据集上的实验表明，SCDL 显着提高了整体和类别级别指标的分割性能，尤其是少数类别的收益尤其强劲，实现了最先进的结果。
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
