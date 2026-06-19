---
title: >-
  [论文解读] RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance
description: >-
  [CVPR 2025][人体理解][体积视频] 提出 RePerformer，一种基于 3DGS 的体积视频表示方法，通过分层解耦运动高斯和外观高斯、Morton 编码参数化以及语义感知对齐模块，统一实现高保真回放和基于新动作的逼真再表演。 以人为中心的体积视频允许用户自由控制虚拟相机视角，在远程呈现、教育和娱乐中有重要应用…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "体积视频"
  - "3D高斯溅射"
  - "动作迁移"
  - "Morton编码"
  - "非刚体重建"
---

# RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance

**会议**: CVPR 2025  
**arXiv**: [2503.12242](https://arxiv.org/abs/2503.12242)  
**代码**: [项目页面](https://moqiyinlun.github.io/Reperformer/)  
**领域**: 人体理解  
**关键词**: 体积视频, 3D高斯溅射, 动作迁移, Morton编码, 非刚体重建

## 一句话总结

提出 RePerformer，一种基于 3DGS 的体积视频表示方法，通过分层解耦运动高斯和外观高斯、Morton 编码参数化以及语义感知对齐模块，统一实现高保真回放和基于新动作的逼真再表演。

## 研究背景与动机

以人为中心的体积视频允许用户自由控制虚拟相机视角，在远程呈现、教育和娱乐中有重要应用。目前存在两类互补的工作流：(1) 回放型方法能高保真重建动态场景但无法泛化到新动作；(2) 可动画型方法（人体化身）可驱动新动作但严重依赖 SMPL 等参数化模型，且主要针对纯人体场景。

本文探索了一个新方向——"回放-再表演"范式：给定一段动态序列的密集多视角视频，不仅要实现精确的自由视角回放，还要能在相似但未见过的新动作下逼真地重新表演整个场景（包括人与物体交互）。这一设定要求方法兼具高保真渲染能力和对新动作的泛化能力，且需处理一般的非刚体场景而非仅限人体。

现有可动画方法依赖 SMPL 模型，无法处理人-物交互场景；而回放方法没有泛化能力。RePerformer 通过解耦运动与外观、利用 2D CNN 的泛化能力来同时满足两个需求。

## 方法详解

### 整体框架

RePerformer 是一个三阶段流水线：(1) **跟踪阶段**——将动态场景解耦为稀疏运动高斯（~50K）和稠密外观高斯（~200K），通过运动高斯驱动外观高斯的非刚体变形实现拓扑一致的跟踪；(2) **训练阶段**——通过 Morton 编码将外观高斯映射到 2D 位置图，用 U-Net 学习从位置图到属性图的可泛化映射；(3) **再表演阶段**——通过语义感知对齐模块关联新表演者的运动高斯与原始外观高斯，实现动作迁移。

### 关键设计1：分层运动-外观解耦

**功能**：将动态场景解耦为拓扑一致的运动表示和可泛化的外观表示。

**核心思路**：稀疏运动高斯仅优化位置和旋转来捕获全局非刚体运动，通过 as-rigid-as-possible (ARAP) 约束保持局部刚性。稠密外观高斯在 canonical 空间初始化并通过最近邻搜索与运动高斯关联。变形通过加权插值实现：$p_{i,t}^{\mathcal{T}} = \sum_{k \in \mathcal{N}} w(p_i, p_k)(R(\Delta q_k) p_i + \Delta p_k)$。

**设计动机**：分层解耦使运动捕获和外观渲染各司其职——运动高斯负责几何变形的泛化，外观高斯负责高保真渲染。这种设计类似于传统的 Embedded Deformation Graph + Mesh Tracking 的思想，但用 3DGS 替代。

### 关键设计2：Morton 编码参数化

**功能**：将 3D 外观高斯高效编码到 2D 位置/属性图中，保持空间邻近性以支持 2D CNN 学习。

**核心思路**：对 canonical 空间外观高斯的位置进行量化后进行 Morton 排序（Z-order curve），将三维坐标的二进制表示交错排列以保持 3D 空间连续性。每个高斯 $i$ 被分配 $(u,v)$ 坐标，形成保持空间邻近关系的 $i \to (u,v)$ 映射，在所有帧间保持一致。

**设计动机**：SMPL 的 UV atlas 无法表示人-物交互场景。Morton 编码是一种通用的 3D-to-2D 映射，不依赖任何参数化人体模型，可处理任意拓扑的非刚体场景。同时保持局部空间一致性，有利于 2D CNN 的卷积操作。

### 关键设计3：语义感知动作迁移

**功能**：将新表演者的动作转移到原始场景的外观高斯上，实现保拓扑的再表演。

**核心思路**：利用 Language-SAM + GroundingDINO + SAM2 为高斯分配语义标签（如头、手、脚等），通过 K-means 聚类建立两序列间的粗对齐。然后通过优化目标 $E_{\text{re}} = \mathcal{L}_2(\mathcal{G}_t^{s'}, f(\mathcal{G}_c^{s'}, \mathcal{G}_t^r)) + \lambda_2 E_{\text{arap}}$ 进行动作转移，同时保持外观高斯的原始拓扑。

**设计动机**：传统变形迁移需要手动指定网格对应关系，对大规模高斯点云不可行。语义感知对齐自动建立两序列之间身体部件的对应，ARAP 约束确保变形过程中的拓扑保持。

### 损失函数

跟踪阶段：$E_{\text{init}} = \lambda_{iso} E_{\text{iso}} + \lambda_{size} E_{\text{size}} + E_{\text{color}}$ 加 ARAP 约束。训练阶段：预训练用 $\mathcal{L}_2$ 监督属性回归，主训练用 $(1-\lambda_{\text{color}}) \mathcal{L}_1 + \lambda_{\text{color}} \mathcal{L}_{\text{D-SSIM}}$。再表演阶段：对齐损失 + 语义损失 + ARAP 正则化。

## 实验关键数据

### 主实验：新视角渲染（DualGS 数据集，500帧）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 训练时间(min/帧) ↓ |
|------|--------|--------|---------|------------------|
| NeuS2 | 29.59 | 0.967 | 0.056 | 3.23 |
| Spacetime Gaussian | 31.69 | 0.981 | 0.029 | 2.24 |
| DualGS | **35.51** | **0.990** | 0.019 | 12.22 |
| **RePerformer** | 34.57 | 0.986 | 0.023 | **1.68** |

### 泛化实验：新动作渲染（3000帧，训练2500/测试500）

| 方法 | 新视角PSNR | 新视角SSIM | 新动作PSNR | 新动作SSIM |
|------|-----------|-----------|-----------|-----------|
| AP-NeRF | 28.26 | 0.939 | 26.85 | 0.944 |
| TAVA | 21.57 | - | - | - |
| **RePerformer** | **33.57** | **0.979** | **32.88** | **0.973** |

### 关键发现

- RePerformer 在回放质量上仅次于逐帧优化的 DualGS（差~1 dB PSNR），但训练速度快 7.3 倍（1.68 vs 12.22 min/帧）。
- 在新动作泛化上显著超越所有基线方法，PSNR 提升超过 5 dB，证明了 Morton 编码 + U-Net 的泛化能力。
- 成功处理了复杂的人-物交互场景（小提琴演奏、气球互动等），这是依赖 SMPL 的方法无法做到的。

## 亮点与洞察

1. **新范式定义**：首次提出"回放-再表演"范式，填补了回放方法和可动画方法之间的空白，具有实际应用价值。
2. **Morton 编码替代 UV Atlas**：用空间填充曲线替代参数化模型依赖的 UV 映射，使方法可处理任意非刚体场景。
3. **CNN 泛化替代逐帧优化**：用 2D CNN 学习位置到属性的映射，训练速度快且具备泛化能力。

## 局限与展望

- 再表演仅支持"相似"的新动作，对差异较大的动作可能产生伪影。
- 语义对齐需要文本 prompt 指定身体部件，自动化程度有限。
- 依赖密集多视角视频输入（最多 81 个视角），对捕获设备要求高。
- Morton 编码虽然保持局部一致性但仍可能将空间相近的高斯映射到远距 UV 坐标。

## 相关工作与启发

- **DualGS**：回放 SOTA，Joint+Skin 双高斯设计启发了本文的运动-外观解耦思路。
- **AnimatableGaussians**：用前后图预测高斯属性的思路启发了 Morton 编码 + CNN 回归。
- **Sumner et al. (Deformation Transfer)**：经典变形迁移方法被扩展到高斯点云上。

## 评分

⭐⭐⭐⭐ — 新范式定义有价值，Morton 编码参数化是亮点，技术方案完整。回放质量接近逐帧优化 SOTA 且泛化能力强。局限在于对密集多视角输入的依赖和新动作的"相似性"约束。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[CVPR 2026\] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](../../CVPR2026/human_understanding/unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)

</div>

<!-- RELATED:END -->
