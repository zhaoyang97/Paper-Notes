# MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification

**会议**: arXiv 2026  
**arXiv**: [2603.09374](https://arxiv.org/abs/2603.09374)  
**作者**: Nikola Jovišić, Milica Škipina, Nicola Dall'Asen, Dubravko Ćulibrk
**代码**: 待确认  
**领域**: 医学图像 / 模型压缩/高效推理  
**关键词**: mil-pf, multiple, instance, learning, precomputed  

## 一句话总结
现代基础模型提供了高度表现力的视觉表示，但由于注释有限和监督薄弱，使其适应高分辨率医学成像仍然具有挑战性。

## 背景与动机
Modern foundation models provide highly expressive visual representations, yet adapting them to high-resolution medical imaging remains challenging due to limited annotations and weak supervision.. Mammography, in particular, is characterized by large images, variable multi-view studies and predominantly breast-level labels, making end-to-end fine-tuning computationally expensive and often impractical.

## 核心问题
现代基础模型提供了高度表现力的视觉表示，但由于注释有限和监督薄弱，使其适应高分辨率医学成像仍然具有挑战性。

## 方法详解

### 整体框架
- Modern foundation models provide highly expressive visual representations, yet adapting them to high-resolution medical imaging remains challenging due to limited annotations and weak supervision.
- We propose Multiple Instance Learning on Precomputed Features (MIL-PF), a scalable framework that combines frozen foundation encoders with a lightweight MIL head for mammography classification.
- By precomputing the semantic representations and training only a small task-specific aggregation module (40k parameters), the method enables efficient experimentation and adaptation without retraining large backbones.

### 关键设计
1. **关键组件1**: Mammography, in particular, is characterized by large images, variable multi-view studies and predominantly breast-level labels, making end-to-end fine-tuning computationally expensive and often impractical.
2. **关键组件2**: By precomputing the semantic representations and training only a small task-specific aggregation module (40k parameters), the method enables efficient experimentation and adaptation without retraining large backbones.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 通过预先计算语义表示并仅训练一个小型特定于任务的聚合模块（40k 参数），该方法可以实现高效的实验和适应，而无需重新训练大型骨干网。
- MIL-PF 在临床规模上实现了最先进的分类性能，同时大大降低了训练复杂性。

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
