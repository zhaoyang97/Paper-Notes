---
title: >-
  [论文解读] Spectral State Space Model for Rotation-Invariant Visual Representation Learning
description: >-
  [CVPR 2025][自监督学习][状态空间模型] 提出 Spectral VMamba，用谱图拉普拉斯的特征向量排序 patch 遍历顺序（替代预定义扫描线），结合旋转特征归一化器（RFN，聚合 4 个正则旋转的特征），在 miniImageNet 上达到 87.86% 准确率且对正则旋转完全不变。
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "状态空间模型"
  - "旋转不变性"
  - "谱图遍历"
  - "VMamba"
  - "图拉普拉斯"
---

# Spectral State Space Model for Rotation-Invariant Visual Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2503.06369](https://arxiv.org/abs/2503.06369)  
**代码**: 有  
**领域**: 自监督学习 / 视觉架构  
**关键词**: 状态空间模型, 旋转不变性, 谱图遍历, VMamba, 图拉普拉斯

## 一句话总结

提出 Spectral VMamba，用谱图拉普拉斯的特征向量排序 patch 遍历顺序（替代预定义扫描线），结合旋转特征归一化器（RFN，聚合 4 个正则旋转的特征），在 miniImageNet 上达到 87.86% 准确率且对正则旋转完全不变。

## 研究背景与动机

**领域现状**：Vision Mamba（VMamba）将图像展平为序列用状态空间模型处理，但展平顺序（如光栅扫描/Z字扫描）依赖于图像的空间方向——图像旋转 90° 后扫描序列完全改变，导致特征不一致。

**现有痛点**：ViT 的自注意力是排列不变的（通过位置嵌入编码空间关系），但 SSM/Mamba 依赖序列顺序——这是固有的序列模型限制。即使 VMamba 使用多方向扫描，对旋转图像的特征仍然不稳定（90° 旋转准确率可降 30+%）。

**核心矛盾**：SSM 需要固定的序列顺序，但旋转改变了空间排列→序列顺序→提取的特征。

**切入角度**：用图的谱分解定义旋转不变的遍历顺序——将 patch 间的相似性建模为图的邻接矩阵，图拉普拉斯的特征向量排序与旋转无关（因为旋转不改变 patch 间的相对关系）。

**核心 idea**：用谱图拉普拉斯的特征向量排序 patch → 旋转不变的遍历 → SSM 对旋转鲁棒。

## 方法详解

### 关键设计

1. **谱遍历扫描（Spectral Traversal Scan, STS）**:

    - 功能：生成旋转不变的 patch 遍历顺序
    - 核心思路：对图像 patch 构建 k-NN 邻接图 $\mathbf{W}$，计算对称归一化拉普拉斯 $\mathbf{L}_{sym} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$，取前 m 个特征向量按特征值排序 patch。由于拉普拉斯的特征值只依赖图结构（而非空间方向），旋转后的图像产生相同的排序
    - 设计动机：谱聚类理论保证了特征向量排序的旋转不变性（在正则旋转下精确不变）

2. **旋转特征归一化器（RFN）**:

    - 功能：处理 STS 无法覆盖的非正则旋转角度
    - 核心思路：将图像旋转 4 个正则角度 {0°, 90°, 180°, 270°}，分别 patchify 并提取特征，逐 patch 取 max：$\mathbf{F}_{i,j} = \max_{r \in \{1,...,4\}} [\mathcal{R}_{-\theta_r}(\text{Patchify}(\mathcal{R}_{\theta_r}(\mathbf{I})))]_{i,j}$
    - 设计动机：STS 在正则旋转下精确不变，RFN 进一步消除 patchify 边界效应

### 损失函数 / 训练策略

标准监督分类训练。谱分解计算开销极小（~2MB FLOPs，patch 数仅 196）。最优超参数：m=4 特征向量，k=5 近邻。

## 实验关键数据

### 主实验

| 模型 | 0° 准确率 | 90° 准确率 | 180° 准确率 |
|------|----------|-----------|------------|
| VMamba-T | 86.25% | ~55% | ~60% |
| **Spectral VMamba-T** | **87.86%** | **~87%** | **~87%** |

### 消融实验

| 配置 | 0° | 90° |
|------|-----|------|
| VMamba + RFN (无 STS) | 86.5% | 52% |
| STS (无 RFN) | 87.5% | 85% |
| **STS + RFN** | **87.86%** | **~87%** |

### 关键发现
- **STS 是旋转不变性的核心**：从 55%→85% 在 90° 旋转下
- **RFN 补充 patchify 边界效应**：额外提升 2%
- **0° 上也有提升**：87.86% vs 86.25%，谱排序本身就比光栅扫描更好

## 亮点与洞察
- **理论优雅**——将旋转不变性问题转化为图论中的谱不变性，数学基础扎实
- **对 SSM 的根本性改进**——解决了所有序列模型（不只是 Mamba）在空间任务中的旋转敏感问题

## 局限与展望
- 仅在 4 个正则旋转下精确不变，非正则角度仍有退化（~78%）
- 谱分解对断开组件的图处理不佳
- 仅在 miniImageNet 验证，大规模数据集待探索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 谱图理论 × SSM 的跨领域创新
- 实验充分度: ⭐⭐⭐ 仅 miniImageNet，规模有限
- 写作质量: ⭐⭐⭐⭐ 理论清晰
- 价值: ⭐⭐⭐⭐ 为 SSM 在空间任务中的根本局限提供了优雅解法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Exemplar-Free Continual Learning for State Space Models](../../CVPR2026/self_supervised/exemplar-free_continual_learning_for_state_space_models.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [\[CVPR 2026\] Weight Space Representation Learning via Neural Field Adaptation](../../CVPR2026/self_supervised/weight_space_representation_learning_via_neural_field_adaptation.md)
- [\[CVPR 2026\] Spectral Mixture-of-Experts for Continual Learning](../../CVPR2026/self_supervised/spectral_mixture-of-experts_for_continual_learning.md)
- [\[CVPR 2026\] Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning](../../CVPR2026/self_supervised/franca_nested_matryoshka_clustering_for_scalable_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
