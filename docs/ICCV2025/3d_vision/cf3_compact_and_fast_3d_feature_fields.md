---
title: >-
  [论文解读] CF³: Compact and Fast 3D Feature Fields
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出 CF³ 管线，通过 top-down 特征提升、per-Gaussian 自编码器压缩和自适应稀疏化，仅使用原始 Gaussian 数量的 5% 即可构建紧凑高速的 3D 特征场，实现 121–245× 的存储压缩和实时渲染。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "Feature Field"
  - "Sparsification"
  - "Feature Compression"
  - "图像分割"
---

# CF³: Compact and Fast 3D Feature Fields

**会议**: ICCV 2025  
**arXiv**: [2508.05254](https://arxiv.org/abs/2508.05254)  
**代码**: 无  
**领域**: 3D Vision  
**关键词**: 3D Gaussian Splatting, Feature Field, Sparsification, Feature Compression, Open-Vocabulary Segmentation

## 一句话总结

提出 CF³ 管线，通过 top-down 特征提升、per-Gaussian 自编码器压缩和自适应稀疏化，仅使用原始 Gaussian 数量的 5% 即可构建紧凑高速的 3D 特征场，实现 121–245× 的存储压缩和实时渲染。

## 研究背景与动机

将 CLIP/SAM/LSeg 等 2D 基础模型的语义特征嵌入 3DGS 是实现开放词汇 3D 理解的主流方法。但现有方法（Feature-3DGS、LangSplat）存在两大问题：

**底层优化开销大**：将原始 2D 特征当作 ground truth 联合优化颜色与特征，训练耗时长，且为恢复颜色细节产生过多冗余 Gaussian。

**高维特征存储爆炸**：直接在每个 Gaussian 中嵌入 512 维特征带来巨大的存储和计算负担。现有压缩手段（2D 自编码器、哈希网格、向量量化）并未显式考虑"为颜色优化的 Gaussian 对特征场是冗余的"这一关键问题。

此外，2D 基础模型的特征在多视角间缺乏一致性。CF³ 从 top-down 视角出发，先利用预训练 3DGS 做特征提升获得视角一致特征，再压缩和稀疏化。

## 方法详解

### 整体框架

CF³ pipeline 分三阶段：
1. **Feature Lifting**：基于预训练 3DGS 的 alpha blending 权重，对多视角 2D 特征做加权融合，赋予每个 Gaussian 一个视角一致的参考特征。
2. **Feature Compression**：在 lifted 特征上训练 per-Gaussian 自编码器，将高维特征压缩到 3 维潜在空间（等价于 RGB 通道）。
3. **Adaptive Sparsification**：在特征场上优化 Gaussian 属性，通过剪枝和合并冗余 Gaussian，极大减少 Gaussian 数量。

### 关键设计

1. **Feature Lifting（特征提升）**：利用 3DGS 的渲染权重 $w_{i,m,p}$，对所有视角的 2D 特征做加权平均：

    $\boldsymbol{f}_i \approx \frac{\sum_{m=1}^{M}\sum_{p \in \mathcal{P}_{i,m}} w_{i,m,p} \boldsymbol{F}_{m,p}}{\sum_{m=1}^{M}\sum_{p \in \mathcal{P}_{i,m}} w_{i,m,p}}$

   同时计算特征方差，过滤掉方差最大的 0.01% Gaussian（通常位于几何不准确或物体边缘处），消除多视角不一致带来的噪声。设计动机：避免从头联合优化，直接复用预训练 3DGS 的几何信息，快速且视角一致。

2. **Per-Gaussian Autoencoder（特征压缩）**：不同于 LangSplat 先在 2D 训练自编码器再提升，CF³ 先提升再压缩。自编码器为 5 层 MLP（128→64→32→16→3），将 D 维特征压缩到 3 维。损失函数包括：

    - MSE 重建损失
    - 余弦相似度损失 $\mathcal{L}_{cos}$
    - 相似结构保持正则 $\mathcal{L}_{struc}$：保持 Gaussian 间特征的相似度关系

   压缩到 3 维的核心优势：可直接复用现有 3DGS 光栅化器，将压缩特征作为 RGB 通道渲染。

3. **Adaptive Sparsification（自适应稀疏化）**：两步操作交替进行：

    - **剪枝**：基于全局贡献度 $C(g_i) = \sum_{m,p} w_{i,m,p}$ 剪除低贡献 Gaussian
    - **合并**：对梯度小（已收敛区域）的 Gaussian，找 k 近邻；若特征余弦相似度高 $\langle \boldsymbol{c}_i, \boldsymbol{c}_j \rangle > \tau_{sim}$ 且 Mahalanobis 距离满足卡方检验 $d_M < \chi^2_\beta$，则进行 moment matching 合并。合并后新 Gaussian 的属性（位置、协方差、透明度、特征）通过加权平均计算。

### 损失函数 / 训练策略

- 自编码器训练：$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_{cos} \mathcal{L}_{cos} + \lambda_{struc} \mathcal{L}_{struc}$
- 稀疏化阶段（优化 Gaussian 属性）：$\mathcal{L} = \|\boldsymbol{F}_{ref} - \boldsymbol{F}\|_1 + \lambda_{depth} \|D_{ref} - D\|_1$
  其中参考特征场被冻结，新特征场通过 L1 特征损失和深度正则化进行优化
- 使用预训练 30k 迭代的 3DGS 场景，整体 pipeline 约 30 分钟/场景

## 实验关键数据

### 主实验 (表格)

**Replica 数据集 (LSeg 特征)**

| 方法 | 存储 ↓ | FPS ↑ | mIoU ↑ | Acc ↑ | #Gaussian ↓ |
|------|--------|-------|--------|-------|-------------|
| Feature-3DGS (512d) | 1393.9M | 7.2 | 73.0 | 91.9 | 636k |
| Feature-3DGS (128d) | 463.9M | 113.8 | 73.4 | 92.9 | 640k |
| CF³ (Ours) | **3.6M** | **328.3** | 70.8 | 91.6 | **47k** |
| CF³ + VQ | **1.7M** | 327.3 | 70.1 | 90.9 | 47k |

**LERF 数据集 (CLIP+SAM 特征)**

| 方法 | 存储 ↓ | FPS ↑ | mIoU ↑ | Acc ↑ | #Gaussian ↓ |
|------|--------|-------|--------|-------|-------------|
| LangSplat | 314.9M | 33.4 | 44.7 | 72.3 | 1270k |
| Feature-3DGS (128d) | 1031.7M | 55.6 | 53.8 | 75.8 | 1423k |
| CF³ (Ours) | **4.2M** | **145.0** | 52.4 | 76.8 | **55k** |

### 消融实验 (表格)

**各组件消融 (Replica + LERF)**

| VF | Pruning | Merging | LSeg mIoU | LSeg #G | CLIP+SAM mIoU | CLIP+SAM #G |
|----|---------|---------|-----------|---------|---------------|-------------|
| ✗ | ✗ | ✗ | 61.0 | 600k | 29.7 | 1289k |
| ✓ | ✓ | ✗ | 71.0 | 165k | 53.4 | 324k |
| ✗ | ✓ | ✓ | 69.8 | 43k | 54.5 | 56k |
| ✓ | ✓ | ✓ | **70.8** | **47k** | 52.2 | **55k** |

合并步骤贡献了额外 ~70% 的存储压缩；方差滤波对低分辨率 MaskCLIP 特征尤为有效。

### 关键发现

- CF³ 比 Feature-3DGS (128d) 紧凑 **121×**，比 LangSplat 紧凑 **74×**，同时性能具有竞争力
- 在 MaskCLIP 实验中，CF³ 的 mIoU 甚至超过 Feature-3DGS **30%+**（46.9 vs 35.9），说明自适应稀疏化对低分辨率特征有补偿作用
- 3D-OVS 数据集上 CF³ 达到 84.5 mIoU（vs LangSplat 81.9），仅用 21k Gaussian
- KITTI-360 大规模室外场景：CF³ 存储仅 6.2M（vs 3810.2M），实现实时 141.6 FPS

## 亮点与洞察

1. **Top-down 思路优于 Bottom-up**：先提升特征到 3D 再压缩，比先压缩 2D 特征再优化更符合特征分布，降低不一致性
2. **3 维压缩 = RGB 通道**：极简设计使得完全兼容现有 3DGS 渲染管线，无需额外解码器
3. **显式处理冗余**：颜色优化产生的密集 Gaussian 对特征场是冗余的，通过 moment matching 合并同语义 Gaussian 是自然且高效的思路
4. **方差滤波**：简单但有效地利用多视角信息一致性来检测和过滤噪声特征

## 局限与展望

- 整体 pipeline 约 30 分钟/场景（主要在自编码器训练和稀疏化阶段），可进一步加速
- 压缩到 3 维是否为最优维度？更高维度或许能保留更多语义细节
- 合并阈值 $\tau_{sim}$ 和 $\chi^2_\beta$ 的选择对不同场景的鲁棒性有待验证
- 未涉及动态场景或大规模实时更新的特征场

## 相关工作与启发

- **Feature-3DGS / LangSplat**：代表性的特征嵌入方法，CF³ 在此基础上解决了冗余和压缩问题
- **LightGaussian**：提供了全局贡献度剪枝的基础
- **FiT3D / CONDENSE**：3D-aware 训练思想的先驱
- 启发：3D 特征场的"紧凑性"和"速度"可能比精度更为重要，尤其在实时应用场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ top-down pipeline + per-Gaussian AE + moment matching 合并的组合具有系统性创新
- **实验充分度**: ⭐⭐⭐⭐ 多数据集、多特征类型、完整消融、3D 分割和大规模场景验证
- **写作质量**: ⭐⭐⭐⭐ 流程清晰，公式推导严谨，图表直观
- **价值**: ⭐⭐⭐⭐⭐ 100×+ 压缩率 + 实时渲染，对 3D 特征场的实用部署有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[CVPR 2026\] Learning Convex Decomposition via Feature Fields](../../CVPR2026/3d_vision/learning_convex_decomposition_via_feature_fields.md)
- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](../../CVPR2025/3d_vision/gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)
- [\[ICCV 2025\] Sparfels: Fast Reconstruction from Sparse Unposed Imagery](sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)

</div>

<!-- RELATED:END -->
