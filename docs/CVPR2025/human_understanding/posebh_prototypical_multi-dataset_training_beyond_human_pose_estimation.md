---
title: >-
  [论文解读] PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation
description: >-
  [CVPR 2025][人体理解][多数据集训练] 提出 PoseBH，通过非参数关键点原型（Sinkhorn-Knopp 在线聚类）和跨类型自监督（CSS）实现人/动物/手部等不同骨骼定义数据集的统一训练，在 APT-36K 动物视频数据集上比 ViTPose++ 提升 11.2 AP，证明跨类型知识迁移的有效性。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "多数据集训练"
  - "关键点原型"
  - "跨骨骼迁移"
  - "Sinkhorn聚类"
  - "自监督"
---

# PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2505.17475](https://arxiv.org/abs/2505.17475)  
**代码**: [https://github.com/uyoung-jeong/PoseBH](https://github.com/uyoung-jeong/PoseBH)  
**领域**: 人体理解 / 姿态估计  
**关键词**: 多数据集训练, 关键点原型, 跨骨骼迁移, Sinkhorn聚类, 自监督

## 一句话总结

提出 PoseBH，通过非参数关键点原型（Sinkhorn-Knopp 在线聚类）和跨类型自监督（CSS）实现人/动物/手部等不同骨骼定义数据集的统一训练，在 APT-36K 动物视频数据集上比 ViTPose++ 提升 11.2 AP，证明跨类型知识迁移的有效性。

## 研究背景与动机

### 领域现状

**领域现状**：姿态估计数据集各有不同的骨骼定义——COCO 有 17 个人体关键点，AP-10K 有动物关键点，InterHand 有手部关键点。标准做法是为每种骨骼独立训练模型，浪费了跨数据集的共享知识。

**现有痛点**：多数据集联合训练面临两个关键问题：（1）不同骨骼的关键点数量和语义不同，无法共享预测头；（2）一个数据集标注的关键点在另一数据集中未标注，造成标签缺失。

**核心矛盾**：不同物种的骨骼看似完全不同，但关节类型有大量共享——如"弯曲关节"、"末端关节"在人/动物/手中都存在。

**切入角度**：用原型学习（prototype learning）在嵌入空间中发现跨骨骼的共享原型——不用预定义关键点对应关系，让聚类自动发现。

**核心 idea**：非参数关键点原型 + 跨类型自监督 = 所有骨骼类型的统一姿态估计。

### 解决思路

**本文目标**：### 关键设计

1. **非参数关键点原型**:

    - 功能：在嵌入空间中学习跨数据集共享的关键点表示
    - 核心思路：为每种关键点维护 $J \times M \times F$ 的原型矩阵（$J$ 个关键点 × $M$ 个原型 × $F$ 维特征），通过在线 Sinkhorn-Knopp 聚类更新原型。


## 方法详解

### 关键设计

1. **非参数关键点原型**:

    - 功能：在嵌入空间中学习跨数据集共享的关键点表示
    - 核心思路：为每种关键点维护 $J \times M \times F$ 的原型矩阵（$J$ 个关键点 × $M$ 个原型 × $F$ 维特征），通过在线 Sinkhorn-Knopp 聚类更新原型。预测时用像素特征与原型的距离做分类
    - 设计动机：不需要手工定义"人肘 = 猫前腿膝盖"，原型在训练中自动聚类

2. **跨类型自监督（CSS）**:

    - 功能：利用未标注关键点类型的预测进行自监督学习
    - 核心思路：对每个混合 batch 中的样本，用关键点头和嵌入头分别预测。两个头中置信度高的预测作为另一个头的伪标签，加权平均
    - 设计动机：动物数据在人体数据集中没有标注，但模型可以从人体训练中学到的"关节"概念做预测，反之亦然

### 损失函数 / 训练策略

$\mathcal{L}_{MDT} = \mathcal{L}_{KPL} + \mathcal{L}_{CSS}$。关键点损失包含像素-原型对比损失（$\mathcal{L}_{PPC}$）和像素-原型距离损失（$\mathcal{L}_{PPD}$）。三阶段渐进训练。

## 实验关键数据

### 主实验

| 数据集 | PoseBH (ViT-B) | ViTPose++ | 提升 |
|--------|---------------|-----------|------|
| COCO | 77.3 AP | 76.5% | +0.8 |
| AP-10K 动物 | 75.0 AP | 74.1% | +0.9 |
| **APT-36K 视频动物** | **87.2 AP** | **76.0%** | **+11.2** |
| InterHand 手部 | 87.1 AUC | 86.2% | +0.9 |

### 关键发现
- **APT-36K 提升最大 (+11.2)**——这是视频数据集，多数据集预训练提供了更好的时序理解
- 原型聚类确实发现了跨类型的共享结构
- CSS 贡献 +0.2 平均分（总+2.4 vs 基线）

## 亮点与洞察
- **"超越人体姿态估计"的哲学**——不是更好的人体估计器，而是统一所有物种/物体的关节检测器
- **APT-36K +11.2 的巨大提升**——说明跨类型知识迁移在数据稀缺的领域（动物视频）价值巨大

## 局限与展望
- 需要少样本原型学习，完全零样本跨骨骼不可行
- CSS 需要数据集分布相似
- 3D 领域未探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 原型驱动的跨骨骼统一学习新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 人/动物/手/视频四类数据集
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为统一姿态估计提供了可扩展框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](../../ECCV2024/human_understanding/worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[CVPR 2025\] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)

</div>

<!-- RELATED:END -->
