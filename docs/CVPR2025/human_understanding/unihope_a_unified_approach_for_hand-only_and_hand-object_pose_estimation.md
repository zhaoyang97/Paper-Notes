---
title: >-
  [论文解读] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation
description: >-
  [CVPR 2025][人体理解][手部姿态估计] 提出 UniHOPE，首个统一手部姿态估计（HPE）和手-物姿态估计（HOPE）的框架，通过物体开关器动态控制输出、抓握感知特征融合消除无关物体特征干扰，以及基于扩散模型的去遮挡生成+多层特征增强学习遮挡不变特征。 从单目图像估计手部和潜在手持物体的 3D 姿态是一个长期挑…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "手部姿态估计"
  - "手-物交互"
  - "统一框架"
  - "遮挡不变特征"
  - "扩散模型去遮挡"
---

# UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.13303](https://arxiv.org/abs/2503.13303)  
**代码**: [GitHub](https://github.com/JoyboyWang/UniHOPE_Pytorch)  
**领域**: 人体理解  
**关键词**: 手部姿态估计, 手-物交互, 统一框架, 遮挡不变特征, 扩散模型去遮挡

## 一句话总结

提出 UniHOPE，首个统一手部姿态估计（HPE）和手-物姿态估计（HOPE）的框架，通过物体开关器动态控制输出、抓握感知特征融合消除无关物体特征干扰，以及基于扩散模型的去遮挡生成+多层特征增强学习遮挡不变特征。

## 研究背景与动机

从单目图像估计手部和潜在手持物体的 3D 姿态是一个长期挑战。现有方法严格分为两类：(1) HPE 方法仅预测手部姿态，不考虑物体；(2) HOPE 方法假设一定有手持物体并进行物体姿态估计。两者均无法灵活适应同时包含有/无物体的通用场景。

作者通过实验揭示了关键问题：HPE 方法在手-物场景上性能退化严重（如 HandOccNet J-PE 从 12.98 恶化至 19.60），HOPE 方法在纯手场景上同样退化（如 Keypoint Trans. 从 17.99 恶化至 25.10）。即使混合训练两种数据，原始任务性能也会下降，说明现有方法缺乏跨场景泛化的统一能力。

核心动机：需要一个统一方法，(1) 基本要求——自适应切换两种场景；(2) 进阶要求——无论物体是否存在都能鲁棒估计手部姿态。特别是手持物体造成的严重遮挡需要学习遮挡不变特征。

## 方法详解

### 整体框架

UniHOPE 包含三个核心模块：(1) 动态手-物姿态估计——通过物体开关器和抓握感知特征融合灵活适配两种场景；(2) 生成式去遮挡器——利用扩散模型生成配对的去遮挡手部图像；(3) 多层特征增强——通过自蒸馏学习遮挡不变特征提升鲁棒性。

### 关键设计1：物体开关器与抓握感知特征融合

**功能**：动态控制物体姿态估计分支的开关，并防止无物体时的特征干扰。

**核心思路**：物体开关器通过 MLP 从物体特征 $\mathbf{F}_p^o$ 预测抓握状态置信度 $s$。抓握标签通过计算物体在初始帧和当前帧的 Relative Rotation Error (RRE) 和 Relative Translation Error (RTE) 自动获取。特征融合时，物体特征按抓握置信度加权：$\mathbf{F}^H = \text{Concat}(\mathbf{F}_r^h, s \cdot \mathbf{F}_r^o + (1-s) \cdot \mathbf{F}_r^h)$，当 $s \approx 0$ 时物体特征被手部特征替代。

**设计动机**：现有 HOPE 方法中的手-物信息交互结构总是从物体向手部传递特征，在无物体时这些无关特征会损害手部姿态估计精度。通过抓握置信度软选择，避免了硬切换带来的不连贯性，同时支持端到端联合优化。

### 关键设计2：扩散模型生成式去遮挡器

**功能**：生成配对的去遮挡手部图像，为遮挡不变特征学习提供监督数据。

**核心思路**：利用 ControlNet 以深度图和手-物掩码为条件，自适应调整控制强度 $\beta$ 来生成高质量的去遮挡手部图像。控制强度通过评估遮挡区域手指弯曲度和可见区域一致性来优化选择，在遮挡严重时降低强度以允许扩散模型更多想象，在遮挡轻微时提高强度以保持姿态一致性。

**设计动机**：理想情况下，未遮挡手部的特征是遮挡手部的最优表示。但这种配对数据极度匮乏，扩散模型的生成能力可以创造逼真的去遮挡图像。手动设定固定控制强度效果不佳，自适应策略兼顾了遮挡区域的合理生成和可见区域的一致保持。

### 关键设计3：多层特征增强

**功能**：从去遮挡手部图像向遮挡手部图像进行知识蒸馏，学习遮挡不变特征。

**核心思路**：在三个层级进行特征增强——(1) Image-level：将去遮挡图像编码的特征与遮挡图像的手部区域特征进行 attention 融合；(2) Token-level：对齐两者的 token 特征应使遮挡手部被推向去遮挡手部的特征空间；(3) Output-level：通过 KL 散度使两者预测的 MANO 参数分布接近。整体在自蒸馏框架下，不需要额外的教师模型。

**设计动机**：仅在单一层级进行对齐不够充分——图像级保留低层信息，token 级对齐中层语义，输出级确保最终预测一致性。多层级联合增强使遮挡不变特征的学习更全面。

### 损失函数

总损失包含：抓握状态分类损失 $\mathcal{L}^s$（BCE）、MANO 回归损失（L1 on joints/vertices/parameters）、物体姿态损失（rotation + translation）、图像级/Token 级/输出级特征增强损失。

## 实验关键数据

### 主实验：DexYCB 数据集统一设定

| 方法 | Hand-Only J-PE ↓ | Hand-Object J-PE ↓ | Object ADD-S ↓ |
|------|-----|-----|-----|
| HandOccNet (HPE) | 13.16 | 14.58 | - |
| HFL-Net (HOPE) | 13.61 | 14.77 | 29.27 |
| **UniHOPE** | **11.39** | **12.94** | **26.76** |

### 消融实验：各组件贡献

| 配置 | Hand-Only J-PE ↓ | Hand-Object J-PE ↓ |
|------|-----|-----|
| Baseline (HOPE) | 12.52 | 13.56 |
| + Object Switcher | 12.22 | 13.42 |
| + Grasp-aware Fusion | 12.05 | 13.23 |
| + Multi-level Enhancement | **11.39** | **12.94** |

### 关键发现

- UniHOPE 在两种场景上均达到 SOTA，Hand-Only J-PE 比最优 HPE 方法降低 1.77mm，Hand-Object J-PE 比最优 HOPE 方法降低 1.83mm。
- 物体开关器的抓握状态预测精度达 97%+，证明自动标签策略的有效性。
- 扩散模型生成的去遮挡图像质量优于简单的图像修复方法，自适应控制强度比固定强度 PJPE 提升 1.2mm。
- 多层特征增强中，Image-level 和 Token-level 贡献最大，Output-level 提供补充增益。

## 亮点与洞察

1. **问题定义有见地**：首次将 HPE 和 HOPE 统一，通过详细的交叉场景退化分析揭示了统一方法的必要性。
2. **扩散模型去遮挡**：创造性地利用生成式 AI 解决数据匮乏问题，自适应控制强度策略兼顾了遮挡区域想象力和可见区域一致性。
3. **端到端统一训练**：通过抓握置信度软切换避免了硬模型切换，支持联合优化。

## 局限与展望

- 仅处理单手场景，双手交互和多人场景未涉及。
- 扩散模型去遮挡的预处理开销较大，虽然不影响推理但增加了训练准备时间。
- 物体姿态估计仍依赖模板（已知 3D 模型），模板无关的物体重建未探索。
- 抓握状态仅二分类（有/无物体），更细粒度的交互状态分类可能有益。

## 相关工作与启发

- **HandOccNet / SimpleHand**：强 HPE 基线，但跨场景退化严重，验证了统一方法的价值。
- **HFL-Net**：双分支 HOPE 方法，其手-物信息交互结构在纯手场景下反而有害，启发了抓握感知融合设计。
- **ControlNet**：条件扩散模型被创造性用于去遮挡数据生成，展示了生成模型在 pose estimation 中的潜力。

## 评分

⭐⭐⭐⭐ — 问题定义清晰且实用，技术方案覆盖基本+进阶两层需求。扩散模型去遮挡的数据增强思路新颖。两种场景上均达 SOTA，证明了统一方法的可行性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[CVPR 2025\] WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](wilor_end-to-end_3d_hand_localization_and_reconstruction_in-the-wild.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](../../AAAI2026/human_understanding/vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)

</div>

<!-- RELATED:END -->
