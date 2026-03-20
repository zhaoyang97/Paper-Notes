# Representation Learning for Spatiotemporal Physical Systems

**会议**: arXiv 2026  
**arXiv**: [2603.13227](https://arxiv.org/abs/2603.13227)  
**作者**: Helen Qu, Rudy Morel, Michael McCabe, Alberto Bietti, François Lanusse et al.
**代码**: [https://github.com/helenqu/physical-representation-learning](https://github.com/helenqu/physical-representation-learning)  
**领域**: 自监督/表示学习  
**关键词**: representation, learning, spatiotemporal, physical, systems  

## 一句话总结
时空物理系统的机器学习方法主要集中在下一帧预测，其目标是学习系统及时演化的准确模拟器。
## 背景与动机
Machine learning approaches to spatiotemporal physical systems have primarily focused on next-frame prediction, with the goal of learning an accurate emulator for the system's evolution in time.. However, these emulators are computationally expensive to train and are subject to performance pitfalls, such as compounding errors during autoregressive rollout.

## 核心问题
然而，这些模拟器的训练计算成本很高，并且容易出现性能缺陷，例如在自回归推出期间出现复合错误。
## 方法详解

### 整体框架
- Accuracy on these tasks offers a uniquely quantifiable glimpse into the physical relevance of the representations of these models.
- We evaluate the effectiveness of general-purpose self-supervised methods in learning physics-grounded representations that are useful for downstream scientific tasks.
- Surprisingly, we find that not all methods designed for physical modeling outperform generic self-supervised learning methods on these tasks, and methods that learn in the latent space (e.g., joint embedding predictive architectures, or JEPAs) outperform those optimizing pixel-level prediction objectives.
- Code is available at https://github.com/helenqu/physical-representation-learning.

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
- 令人惊讶的是，我们发现并非所有为物理建模设计的方法都优于这些任务上的通用自监督学习方法，并且在潜在空间中学习的方法（例如联合嵌入预测架构或 JEPA）优于那些优化像素级预测目标的方法。
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
- 对我的价值: ⭐⭐⭐
