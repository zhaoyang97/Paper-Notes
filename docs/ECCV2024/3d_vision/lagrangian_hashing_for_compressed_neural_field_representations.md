---
title: >-
  [论文解读] Lagrangian Hashing for Compressed Neural Field Representations
description: >-
  [ECCV 2024][3D视觉][神经场压缩] 将InstantNGP的欧拉网格哈希表与拉格朗日点云表示相结合，在哈希桶中存储可移动的高斯特征点，实现参数量减少1.8-2.8倍：但重建质量不降的紧凑神经场表示。 领域现状： 基于特征网格的神经场方法（如InstantNGP）在速度和质量上取得了显著进展…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "神经场压缩"
  - "哈希表"
  - "拉格朗日表示"
  - "高斯混合"
  - "NeRF"
---

# Lagrangian Hashing for Compressed Neural Field Representations

**会议**: ECCV 2024  
**arXiv**: [2409.05334](https://arxiv.org/abs/2409.05334)  
**代码**: [https://theialab.github.io/laghashes/](https://theialab.github.io/laghashes/)  
**领域**: 模型压缩  
**关键词**: 神经场压缩, 哈希表, 拉格朗日表示, 高斯混合, NeRF

## 一句话总结

将InstantNGP的欧拉网格哈希表与拉格朗日点云表示相结合，在哈希桶中存储可移动的高斯特征点，实现**参数量减少1.8-2.8倍**但重建质量不降的紧凑神经场表示。

## 研究背景与动机

**领域现状**: 基于特征网格的神经场方法（如InstantNGP）在速度和质量上取得了显著进展，但通常需要较大的存储开销。同时，3D Gaussian Splatting等基于点的方法虽然灵活但需要数百万个点。

**现有痛点**: 现有特征网格方法采用欧拉（Eulerian）方式在均匀网格上分布特征，无法根据场景复杂度自适应分配表示预算——简单区域和复杂区域使用相同的特征密度。

**核心矛盾**: 欧拉网格的索引效率高但空间分配固定 vs. 拉格朗日点云的空间自适应性强但索引复杂需要加速结构。

**本文目标**: 如何在保持哈希表快速索引优势的同时，让特征点能自适应地向需要更多表示能力的区域聚集。

**切入角度**: 在InstantNGP的高分辨率哈希层中，将每个桶扩展为存储多个带有位置的高斯特征点，形成欧拉-拉格朗日混合表示。

**核心 idea**: 在哈希桶内嵌入可学习位置的高斯混合模型，让特征点在训练中自动迁移到需要更多表示能力的表面区域。

## 方法详解

### 整体框架

基于InstantNGP的多尺度哈希架构，保留浅层（低分辨率）的标准欧拉特征网格，将深层（高分辨率）的哈希桶扩展为拉格朗日高斯混合表示。查询时，对于坐标 $\mathbf{x}$，在每层提取特征后拼接送入MLP解码器：

$$\mathcal{F}(\mathbf{x}) = \text{MLP}(\mathbf{f}_1(\mathbf{x}) \oplus \mathbf{f}_2(\mathbf{x}) \oplus \ldots \oplus \mathbf{f}_L(\mathbf{x}); \boldsymbol{\theta})$$

### 关键设计

1. **多尺度哈希表示（Multi-scale Representation）**: 沿用InstantNGP的 $L=16$ 层哈希结构，分辨率按几何级数 $N_l = N_{\min} \cdot b^l$ 递增。前 $L - \tilde{L}$ 层为标准欧拉特征，后 $\tilde{L}$ 层为拉格朗日表示。设计动机是仅在高分辨率层（哈希碰撞最严重的地方）引入点云来缓解碰撞问题。

2. **桶内高斯混合模型（Per-bucket Gaussian Mixture）**: 每个哈希桶存储 $K$ 个各向同性高斯，参数包括均值 $\boldsymbol{\mu}_k$、标准差 $\sigma_k$ 和特征向量 $\mathbf{f}_k$。查询位置 $\mathbf{x}$ 时，桶内特征通过高斯加权聚合：

$$\mathbf{F}(\mathbf{x}) = \sum_k \mathcal{N}_k(\mathbf{x}) \cdot \mathbf{f}_k, \quad \mathcal{N}_k(\mathbf{x}) = \frac{1}{(2\pi)^{1/2}\sigma_k} \exp\left(-\frac{\|\mathbf{x} - \boldsymbol{\mu}_k\|_2^2}{2\sigma_k^2}\right)$$

标准差与网格分辨率正相关，训练中从 $50\times$ 网格单元衰减到 $5\times$，保证初期平滑收敛。核心思路是复用哈希表作为索引结构，避免额外的近邻搜索。

3. **引导损失（Guidance Loss）**: 受EM算法启发，设计基于KL散度的引导损失，将高斯点移向表面。对于射线上的采样点 $\mathbf{x}$，找到最近的高斯（E步），然后最小化该高斯PDF与NeRF积分权重 $W(\mathbf{x}) = T(\mathbf{x}) \cdot \tau(\mathbf{x})$ 之间的KL散度（M步）：

$$\mathcal{L}_{\text{guide}}^l(\mathbf{x}) = W(\mathbf{x}) \cdot \min_{k,v}\left(-\log(\alpha_{v,l}) + \frac{\|\mathbf{x} - \boldsymbol{\mu}_{k,v,l}\|_2^2}{2\sigma_{k,v,l}^2}\right)$$

直觉解释：若 $W(\mathbf{x}) \approx 1$（即该点在表面上），则应有一个高斯的均值靠近 $\mathbf{x}$。本质上是表面点与高斯点之间的单向Chamfer距离。

### 损失函数 / 训练策略

总损失为三项加权：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{dist}} \mathcal{L}_{\text{dist}} + \lambda_{\text{guide}} \mathcal{L}_{\text{guide}}$$

- $\mathcal{L}_{\text{recon}}$：像素重建损失（Huber Loss + 体渲染）
- $\mathcal{L}_{\text{dist}}$：蒸馏损失，促进体积中形成清晰表面
- $\mathcal{L}_{\text{guide}}$：引导损失，$\lambda_{\text{guide}} = 0.1$，带warm-up调度

训练使用Adam，lr=$10^{-2}$，高斯位置lr=$10^{-3}$，20K迭代。高斯标准差在训练中指数衰减。

## 实验关键数据

### 主实验 - NeRF Synthetic Dataset（新视角合成）

| 方法 | 参数量 | Lego | Mic | Materials | Chair | Hotdog | Ficus | Drums | Ship | 平均PSNR↑ |
|------|--------|------|-----|-----------|-------|--------|-------|-------|------|----------|
| InstantNGP (B=2¹⁹) | 12.10M | 35.67 | 36.85 | 29.60 | 35.71 | 37.37 | 33.95 | 25.44 | 30.29 | 33.11 |
| **Ours (B=2¹⁷)** | **6.68M** | 35.60 | 36.45 | 29.63 | 35.61 | 37.23 | 33.89 | 25.67 | 30.84 | **33.12** |
| Ours (B=2¹⁷·⁹) | 12.13M | 35.74 | 36.78 | 29.66 | 35.76 | 37.30 | 34.02 | 25.75 | 31.01 | 33.25 |

### 消融实验 - Tanks & Temples

| 消融项 | 平均PSNR↑ |
|--------|----------|
| Full | **27.94** |
| w/o $\mathcal{L}_{\text{dist}}$ | 27.70 |
| w/o $\mathcal{L}_{\text{guide}}$ | 27.75 |

| 每桶高斯数 K | 参数量 | PSNR↑ |
|-------------|--------|-------|
| 无混合 | 0.50M | 27.49 |
| 2 | 0.67M | 27.82 |
| **4** | **0.92M** | **27.94** |
| 8 | 1.41M | 27.99 |

### 关键发现

- 参数量减半（6.68M vs 12.10M），PSNR基本持平（33.12 vs 33.11）
- Tanks & Temples数据集上同样实现1.8×压缩且PSNR不降（28.55 vs 28.51）
- 引导损失使高斯点有效迁移到表面区域，解决了哈希碰撞造成的伪影（如卡车表面微结构）
- K=4为最佳性价比，2个拉格朗日层在最细分辨率上效果最优

## 亮点与洞察

- **欧拉-拉格朗日混合**的思路很巧妙：低分辨率层碰撞少用网格，高分辨率层碰撞多用点云
- 复用哈希表作为点云的索引结构，避免了额外的KD-tree等加速结构
- 引导损失的EM类比清晰直观，将"点应在表面附近"约束优雅地形式化为KL散度最小化
- 在紧凑表示方面与CompactNGP（专为压缩设计）持平，但方法更通用

## 局限与展望

- 目前标准差按固定调度衰减，未作为可学习参数端到端优化
- 仅在最细的2层使用拉格朗日表示，更灵活的层选策略值得探索
- 训练时间与InstantNGP相当，但实际推理时高斯加权计算增加了一定开销
- 未与3DGS进行直接的压缩比较（仅比较了12k高斯的退化版本）

## 相关工作与启发

- **InstantNGP**: 本文的基础架构，采用多分辨率哈希网格
- **3D Gaussian Splatting**: 拉格朗日点云思想的来源，但需要COLMAP初始化和百万级点
- **CompactNGP**: 同样追求紧凑NeRF表示，使用hash probing策略
- **启发**: 将Lagrangian的思想引入其他网格方法（如TensoRF分解）可能带来类似的压缩收益

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 欧拉-拉格朗日混合哈希表示是新颖的组合创新
- **实验充分度**: ⭐⭐⭐⭐ — 2D图像拟合 + NeRF合成 + 真实场景 + 压缩对比 + 详尽消融
- **写作质量**: ⭐⭐⭐⭐ — 物理类比清晰，公式推导完整
- **价值**: ⭐⭐⭐⭐ — 为神经场压缩提供了新的技术路径，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [\[CVPR 2025\] SiNR: Sparsity Driven Compressed Implicit Neural Representations](../../CVPR2025/3d_vision/sinr_sparsity_driven_compressed_implicit_neural_representations.md)
- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)

</div>

<!-- RELATED:END -->
