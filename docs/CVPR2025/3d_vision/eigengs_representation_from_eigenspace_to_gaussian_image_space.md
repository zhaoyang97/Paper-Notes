---
title: >-
  [论文解读] EigenGS: Representation from Eigenspace to Gaussian Image Space
description: >-
  [CVPR 2025][3D视觉][高斯表示] 本文提出 EigenGS，将经典 PCA 的特征空间表示与 2D 高斯 Splatting 图像表示相桥接，通过在特征基上学习统一的高斯参数实现新图像的即时初始化（无需从头优化），并引入频率感知学习机制避免高分辨率重建伪影，在收敛速度和最终质量上全面超越 GaussianImage。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "高斯表示"
  - "PCA"
  - "图像重建"
  - "2D高斯"
  - "频率感知学习"
---

# EigenGS: Representation from Eigenspace to Gaussian Image Space

**会议**: CVPR 2025  
**arXiv**: [2503.07446](https://arxiv.org/abs/2503.07446)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 高斯表示, PCA, 图像重建, 2D高斯, 频率感知学习

## 一句话总结

本文提出 EigenGS，将经典 PCA 的特征空间表示与 2D 高斯 Splatting 图像表示相桥接，通过在特征基上学习统一的高斯参数实现新图像的即时初始化（无需从头优化），并引入频率感知学习机制避免高分辨率重建伪影，在收敛速度和最终质量上全面超越 GaussianImage。

## 研究背景与动机

**领域现状**：3D 高斯 Splatting（3DGS）已被广泛应用于 3D 场景表示。GaussianImage 将此概念适配到 2D，用一组 2D 高斯拟合单张图像进行重建。PCA 作为经典降维技术在计算机视觉中有广泛应用，但其像素逐点独立假设忽略了局部空间关系。

**现有痛点**：(1) GaussianImage 需要为每张新图像从随机初始化开始独立训练，收敛缓慢（前 100 次迭代 PSNR 仅约 10 dB）；(2) PCA 假设像素独立，无法利用局部和非局部像素关系；(3) 高分辨率图像优化时，高斯易收缩为均匀小尺寸，产生"penny-round-tile"圆点伪影。

**核心矛盾**：GaussianImage 需要逐图训练，无法利用训练集的共性知识进行初始化；PCA 有良好的初始化能力但缺乏局部建模能力。

**本文目标** (1) 如何将 PCA 的特征基知识转化为高斯参数，实现新图像的即时初始化？(2) 如何让高斯适应不同空间频率，避免高分辨率下的伪影？

**切入角度**：作者观察到 PCA 特征基的线性组合系数可以直接与高斯的视觉权重合并——如果特征基用同一组高斯渲染，则新图像的高斯可通过系数加权和即时导出。

**核心 idea**：用同一组 2D 高斯拟合 PCA 特征基的所有分量，新图像通过 PCA 投影系数即时获得高斯初始化，再经少量优化快速收敛。

## 方法详解

### 整体框架

输入是训练图像集 $\{I_1, ..., I_m\}$，首先计算 PCA 特征基 $\{\Psi_j\}_{j=1}^k$。然后学习一组共享的 2D 高斯 $\mathcal{N}$ 来同时近似所有 $k$ 个特征基分量。对于新图像 $I$，计算其 PCA 系数 $\{w_j\}$，通过线性组合 $c'_n = \sum_j w_j \psi'_{n,j}$ 即时得到每个高斯的权重，获得初始重建 $\tilde{I}^{(0)}$。最后通过最小化重建损失进一步优化高斯参数。

### 关键设计

1. **特征基高斯表示（EigenGS Representation）**:

    - 功能：用同一组高斯统一表示所有 PCA 特征基分量，实现从特征空间到图像空间的无缝转换
    - 核心思路：每个特征基分量 $\Psi_j$ 在像素位置 $(x,y)$ 的渲染为 $\tilde{\Psi}_j(x,y) = \sum_{n=1}^{|\mathcal{N}|} \psi'_{n,j} \cdot \exp(-\sigma_n(x,y))$，其中高斯的空间参数（位置、协方差）在所有分量间共享，仅权重 $\psi'_{n,j}$ 不同。新图像 $I$ 的高斯权重为 $c'_n = \sum_j w_j \psi'_{n,j}$，即 PCA 系数与特征基权重的线性组合。数学上保证初始重建质量等同于标准 PCA 重建
    - 设计动机：传统 PCA 重建是像素级线性组合，将其转化为高斯权重的线性组合后，不仅保留 PCA 的初始化优势，还允许通过后续优化利用高斯的局部建模能力超越 PCA 的上限

2. **频率感知学习（Frequency-aware Learning, FL）**:

    - 功能：防止所有高斯收缩为均匀小尺寸，维持大小高斯混合以覆盖不同空间频率
    - 核心思路：将高斯集合分为 $\mathcal{N}_l$ 和 $\mathcal{N}_h$ 两组，特征基分量按特征值大小分为低频 $\{\tilde{\Psi}_l\}$ 和高频 $\{\tilde{\Psi}_h\}$。训练分两阶段：第一阶段分配约 10% 的高斯建模大特征值（低频）分量，迫使这些高斯保持较大尺寸；第二阶段用剩余高斯建模小特征值（高频）分量。最终表示是大小高斯的混合
    - 设计动机：优化过程天然偏好小高斯以最小化像素级差异，在高分辨率下导致所有高斯收缩出现"penny-round-tile"伪影。通过将高频和低频分离训练，自然形成双模态尺寸分布，替代显式正则化

3. **YCbCr 色彩空间处理**:

    - 功能：减少 PCA 重建中的值截断导致的性能退化
    - 核心思路：在 YCbCr 空间而非 RGB 空间进行分解，亮度（Y: 16-235）和色度（Cb/Cr: 16-240）通道分离处理。由于 YCbCr 的值域结构为 PCA 重建的越界值提供了自然余量，且色度通道更少使得对异常值更鲁棒
    - 设计动机：RGB 的三个通道都容易受异常值影响，PCA 重建的越界值在三通道上都会产生截断。YCbCr 将颜色信息压缩到两个通道，降低截断带来的质量损失，PSNR 提升可达 7+ dB

### 损失函数 / 训练策略

使用标准图像重建损失（像素级 L2）。训练两阶段：低频阶段约 10% 高斯对应大特征值分量；高频阶段剩余高斯对应小特征值分量。训练集 10,000 张图用于 PCA 分解，使用 300 或 500 个特征分量，默认 20,000 个高斯点，在单卡 V100 上训练。

## 实验关键数据

### 主实验

FFHQ 数据集（512×512），20,000 高斯点：

| 方法 | ITER=0 PSNR | ITER=100 PSNR | ITER=1000 PSNR | ITER=10000 PSNR |
|------|-------------|---------------|----------------|-----------------|
| GaussianImage | - | 10.4 | 29.4 | 40.1 |
| EigenGS (300 comp) | 28.0 | 34.4 | 37.5 | 41.8 |
| EigenGS (500 comp) | 28.9 | 34.8 | 37.7 | 41.8 |

1000 次迭代时 EigenGS 已有 83-84% 样本达 PSNR>35dB，GaussianImage 为 0%。

### 消融实验

| 配置 | CelebA PSNR | FFHQ PSNR | Cats PSNR | Cars PSNR |
|------|-------------|-----------|-----------|-----------|
| Ours-YCbCr | 47.2 | 41.8 | 45.7 | 44.7 |
| Ours-YCbCr (w/o FL) | 48.0 | 40.7 | 46.1 | 43.5 |
| Ours-RGB | 39.5 | 34.9 | 38.5 | 36.4 |
| Ours-RGB (w/o FL) | 39.9 | 33.3 | 38.9 | 35.1 |

### 关键发现

- EigenGS 的初始 PSNR（28-29 dB）已经远超 GaussianImage 的随机初始化，100 次迭代即可达 34+ dB
- YCbCr 色彩空间是最大贡献因素，FFHQ 上相比 RGB 提升约 7 dB
- 频率感知学习（FL）在高分辨率（FFHQ 512×512, Cars）上提升明显（+1.1/+1.2 dB），在低分辨率（CelebA 256×256）上略有干扰（-0.8 dB）
- 跨数据集泛化性强：ImageNet 训练的 EigenGS 应用到 CelebA 仍有 28.7 dB 初始 PSNR，100 次迭代达 35.4 dB
- 分量数主要影响早期收敛（300 vs 500），最终质量几乎无差（41.8 vs 41.8 dB）

## 亮点与洞察

- **PCA 与高斯 Splatting 的优雅桥接**：利用线性组合的交换律，将 PCA 系数的图像重建无缝转化为高斯权重的计算，数学推导简洁且实用。这个思路可推广到任何基于基函数的初始化策略
- **频率分离替代显式正则化**：不直接约束高斯尺寸，而是通过分频段训练自然获得多尺度高斯分布，既解决伪影又保持优化灵活性
- **跨域泛化的启示**：ImageNet 训练的通用 EigenGS 在各数据集上都能提供有效初始化，暗示可能存在通用图像高斯基

## 局限与展望

- 需要预先在训练集上做 PCA，不适用于完全无先验的场景
- 仅验证了 2D 高斯图像表示，未扩展到 3DGS 的 3D 场景
- 低分辨率图像上 FL 略有负面影响，需要根据分辨率手动选择是否启用
- PCA 的线性假设限制了在高度非线性视觉变化（如大角度姿态变化）下的初始化质量
- 训练仍需 10,000 次迭代（约 13 秒），距离"实时"还有一定距离

## 相关工作与启发

- **vs GaussianImage**: GaussianImage 每张图从随机初始化训练 10,000 次迭代；EigenGS 利用 PCA 初始化，100-1000 次迭代即可达到可比质量，加速 10-100 倍
- **vs 传统 PCA**: PCA 给出线性重建的上限（PSNR 约 28-29 dB）；EigenGS 以 PCA 为起点继续用高斯优化，最终PSNR 达 41+ dB，突破线性限制
- **vs Mini-Splatting 等高斯优化方法**: 这些方法优化高斯密度/正则化，EigenGS 正交地解决初始化问题，两者可结合

## 评分

- 新颖性: ⭐⭐⭐⭐ PCA+高斯的桥接思路新颖且数学优雅，但核心贡献偏向组合创新
- 实验充分度: ⭐⭐⭐⭐ 多数据集、跨域、消融实验充分，但缺少与更多基线的对比
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，图表直观，论文结构好
- 价值: ⭐⭐⭐ 应用场景偏窄（2D 高斯图像表示），对主流 3DGS 场景重建的启发有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks](maskgaussian_adaptive_3d_gaussian_representation_from_probabilistic_masks.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[ICLR 2026\] Weight Space Representation Learning on Diverse NeRF Architectures](../../ICLR2026/3d_vision/weight_space_representation_learning_on_diverse_nerf_architectures.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](../../ICCV2025/3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[CVPR 2025\] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)

</div>

<!-- RELATED:END -->
