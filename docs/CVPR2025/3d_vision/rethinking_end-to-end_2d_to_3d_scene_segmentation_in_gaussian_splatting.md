---
title: >-
  [论文解读] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D场景分割] 提出 Unified-Lift，一种基于 3DGS 的端到端对象感知 2D-to-3D 分割方法，通过学习全局对象级码本与高斯级特征的关联，消除了现有方法对前/后处理的依赖，在多视角一致实例分割上显著超越 SOTA。 将 2D 基础模型（如 SAM）的分割结果提升到 3D…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D场景分割"
  - "高斯溅射"
  - "2D到3D提升"
  - "对象级码本"
  - "端到端分割"
---

# Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.14029](https://arxiv.org/abs/2503.14029)  
**代码**: [GitHub](https://github.com/Runsong123/Unified-Lift)  
**领域**: 3D视觉  
**关键词**: 3D场景分割, 高斯溅射, 2D到3D提升, 对象级码本, 端到端分割

## 一句话总结

提出 Unified-Lift，一种基于 3DGS 的端到端对象感知 2D-to-3D 分割方法，通过学习全局对象级码本与高斯级特征的关联，消除了现有方法对前/后处理的依赖，在多视角一致实例分割上显著超越 SOTA。

## 研究背景与动机

将 2D 基础模型（如 SAM）的分割结果提升到 3D 辐射场是实现 3D 场景理解的有效途径。然而，2D 实例分割在不同视角间缺乏一致性（同一物体在不同角度有不同 ID），且存在欠/过分割问题，使提升过程面临冲突监督。

现有方法分为三类：(1) 端到端方法（如 Panoptic Lifting）——使用匈牙利匹配获取伪标签但对匹配结果敏感；(2) 两阶段方法——预处理建立视角间对应，但误差累积；(3) 对比学习+后处理方法——用对比学习编码实例信息到特征场，再用 HDBSCAN 聚类提取最终分割，但聚类对超参数敏感且引入误差。

核心问题：能否设计一个端到端框架实现精确 3D 场景分割，无需任何前/后处理？Unified-Lift 通过引入可学习的全局对象级码本和专门的学习策略来回答这个问题。

## 方法详解

### 整体框架

Unified-Lift 基于 3DGS 表示，包含三个组件：(1) 为每个高斯点增加可学习特征，用对比学习优化（高斯级特征）；(2) 引入全局对象级码本，通过与高斯级特征的关联进行分割预测（对象级理解）；(3) 设计关联学习模块和噪声标签过滤模块实现有效码本学习。推理时直接渲染特征→计算与码本的相似度→取最高相似度的码本索引作为实例 ID，无需任何后处理。

### 关键设计1：对象级码本表示

**功能**：提供显式的对象级 3D 场景理解，替代基于聚类的后处理。

**核心思路**：定义可学习的码本矩阵 $\mathbf{F}_{obj} \in \mathbb{R}^{L \times d}$，每行对应一个 3D 场景中的物体。对渲染的高斯级特征 $\mathbf{F}_u$，通过 softmax 相似度计算概率分布：$\mathbf{P}_u = \text{softmax}(\text{sim}(\mathbf{F}_u, \mathbf{F}_{obj}))$，其中 $\text{sim}$ 使用点积。推理时取最大概率索引即为实例 ID。

**设计动机**：高斯级特征仅隐式编码实例信息，需要聚类后处理。码本提供了显式的对象级表示，使分割预测成为特征与码本的简单匹配，消除了聚类的超参数敏感性和误差累积。

### 关键设计2：关联学习模块

**功能**：生成多视角一致的伪标签并提供鲁棒的码本优化约束。

**核心思路**：两个关键改进——(1) **面积感知 ID 映射**：改进匈牙利匹配中的目标函数，去除 Panoptic Lifting 中的归一化项 $1/|\Omega_j|$，使大面积分割掩码主导映射过程以提升多视角一致性；(2) **集中约束**：除稀疏性的交叉熵损失 $\mathcal{L}_{\text{class}}$ 外，增加 L1 距离约束使码本特征的方向与对应高斯级特征对齐：$\mathcal{L}_{\text{concen}} = \frac{1}{|\Omega|} \sum \|\mathbf{F}_{obj}^{\Pi^*(K_u)} - \mathbf{F}_u / \|\mathbf{F}_u\|\|_1$。

**设计动机**：归一化使小面积分割对映射有不成比例的影响，导致多视角不一致。去除归一化后大物体主导映射，小物体在后续训练中通过学习逐步正确关联。集中约束确保码本特征与高斯特征的方向一致，与对比学习的点积相似度度量互补。

### 关键设计3：噪声标签过滤模块

**功能**：增强对 2D 分割噪声（欠/过分割）的鲁棒性。

**核心思路**：利用已学习的高斯级特征以自监督方式估计不确定性图。对每个像素，比较其高斯级特征与同一掩码内其他像素特征的一致性，不一致区域标记为高不确定性。高不确定性区域的分割标签在训练中被降权或过滤。

**设计动机**：SAM 等 2D 模型产生的分割掩码不完美（欠/过分割），直接作为监督会引入噪声梯度。通过特征一致性自监督估计不确定性，无需额外标注即可识别并过滤噪声标签。

### 损失函数

总损失：$\mathcal{L} = \mathcal{L}_{\text{contra}} + \lambda_1 \mathcal{L}_{\text{class}} + \lambda_2 \mathcal{L}_{\text{concen}} + \mathcal{L}_{\text{photo}}$，其中 $\mathcal{L}_{\text{contra}}$ 为 InfoNCE 对比损失，$\mathcal{L}_{\text{class}}$ 为交叉熵稀疏约束，$\mathcal{L}_{\text{concen}}$ 为集中约束，$\mathcal{L}_{\text{photo}}$ 为 3DGS 的光度损失。

## 实验关键数据

### 主实验：LERF-Masked 数据集

| 方法 | mAP ↑ | mAP50 ↑ | mAP75 ↑ | 时间(min) ↓ |
|------|-------|---------|---------|-----------|
| Panoptic Lifting | 36.7 | 60.3 | 36.2 | ~120 |
| Contrastive Lift | 52.3 | 72.6 | 56.2 | ~60 |
| SAGA (+ HDBSCAN) | 56.5 | 79.1 | 57.5 | ~30 |
| OmniSeg3D (+ HDBSCAN) | 61.3 | 82.0 | 65.2 | ~35 |
| **Unified-Lift** | **67.4** | **87.1** | **73.8** | **~20** |

### Replica 数据集

| 方法 | mPQ ↑ | mSQ ↑ | mRQ ↑ |
|------|-------|-------|-------|
| Panoptic Lifting | 33.7 | 66.8 | 47.4 |
| Contrastive Lift | 39.1 | 68.3 | 55.2 |
| OmniSeg3D | 45.7 | 71.5 | 62.9 |
| **Unified-Lift** | **52.3** | **74.5** | **69.1** |

### 消融实验：各组件贡献（LERF-Masked mAP）

| 配置 | mAP ↑ |
|------|-------|
| 对比学习 + HDBSCAN | 61.3 |
| + 基础码本策略 | 63.1 |
| + 面积感知ID映射 | 64.8 |
| + 集中约束 | 66.2 |
| + 噪声标签过滤 | **67.4** |

### 关键发现

- Unified-Lift 在 LERF-Masked 上 mAP 超越所有方法 6.1 个点，且推理速度最快（无需聚类后处理）。
- 面积感知 ID 映射显著提升了多视角一致性，小物体分割改善尤为明显。
- 在 Messy Rooms 数据集（每场景 500+ 物体）上展现了良好的扩展性。
- 消除后处理不仅提升了精度，还使推理流程更简洁、超参数更少。

## 亮点与洞察

1. **端到端设计**：首个无需任何前/后处理的 2D-to-3D 实例分割提升方法，消除了误差累积和超参数调优。
2. **码本替代聚类**：用可学习的矩阵替代 HDBSCAN 聚类，既提升精度又简化流程。
3. **面积感知映射的直觉**：去掉一个归一化就能显著提升多视角一致性，简单而有效。

## 局限与展望

- 码本大小 $L$ 需要预设为场景中最大物体数，对未知场景需要估计。
- 对比学习和码本学习的联合优化可能存在训练不稳定性。
- 依赖 SAM 的 2D 分割质量，对遮挡严重或纹理缺失的物体可能效果不佳。
- 目前仅处理实例分割，语义类别信息未涉及。

## 相关工作与启发

- **SAGA / OmniSeg3D**：SOTA 对比学习+聚类方法，本文证明端到端码本方案在精度和效率上全面超越。
- **Panoptic Lifting**：早期端到端方法，其匈牙利匹配策略被本文的面积感知版本改进。
- **VQ-VAE / 码本学习**：离散码本的思想被引入连续特征的对象级表示。

## 评分

⭐⭐⭐⭐ — 端到端设计消除了后处理的痛点，码本表示优雅，面积感知映射简单有效。在多个数据集上实现显著的性能提升和效率改善。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)
- [\[CVPR 2025\] End-to-End HOI Reconstruction Transformer with Graph-based Encoding](end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)
- [\[CVPR 2025\] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)
- [\[ICML 2026\] EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation](../../ICML2026/3d_vision/eps3d_end-to-end_feed-forward_3d_panoptic_segmentation.md)
- [\[CVPR 2025\] Functionality Understanding and Segmentation in 3D Scenes](functionality_understanding_and_segmentation_in_3d_scenes.md)

</div>

<!-- RELATED:END -->
