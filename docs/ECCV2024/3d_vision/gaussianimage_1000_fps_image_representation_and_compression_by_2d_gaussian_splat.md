---
title: >-
  [论文解读] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][2D Gaussian Splatting] 提出GaussianImage，首次将2D Gaussian Splatting用于图像表示与压缩，通过紧凑的8参数2D高斯和累积求和光栅化算法，实现了2000 FPS的解码速度，同时与INR方法在表示质量和压缩性能上持平。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "2D Gaussian Splatting"
  - "图像表示"
  - "图像压缩"
  - "向量量化"
  - "快速渲染"
---

# GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2403.08551](https://arxiv.org/abs/2403.08551)  
**代码**: [https://github.com/Xinjie-Q/GaussianImage](https://github.com/Xinjie-Q/GaussianImage)  
**领域**: 3D视觉  
**关键词**: 2D Gaussian Splatting, 图像表示, 图像压缩, 向量量化, 快速渲染

## 一句话总结

提出GaussianImage，首次将2D Gaussian Splatting用于图像表示与压缩，通过紧凑的8参数2D高斯和累积求和光栅化算法，实现了2000 FPS的解码速度，同时与INR方法在表示质量和压缩性能上持平。

## 研究背景与动机

**领域现状**: 隐式神经表示（INR）在图像表示和压缩方面取得了很大成功，能以紧凑网络实现高质量图像重建。但存在两大类方法各有局限：MLP-based INR（SIREN, WIRE）训练慢、渲染慢；Feature grid-based INR（I-NGP, NeuRBF）加速了训练推理但需大量GPU显存。

**现有痛点**: INR方法在低端设备上部署困难——要么显存需求大（feature grid方法需1500-2900 MiB），要么渲染速度慢（WIRE仅11 FPS）。这严重限制了神经图像编解码器的实际应用。

**核心矛盾**: 高质量图像表示需要大量参数/计算，但实际部署要求低显存、快速解码——如何用显式表示打破INR的隐式瓶颈？

**本文目标**: 开发一种训练高效、显存友好、解码超快的图像表示和压缩技术。

**切入角度**: 3D Gaussian Splatting在3D场景重建中已展示了显式表示+并行光栅化带来的速度优势。能否将其适配到2D图像表示任务？直接适配面临三大挑战：3D高斯参数过多（59个/点）、α-blending需要深度排序（2D图像无深度）、提前截断导致高斯利用不足。

**核心 idea**: 用仅8个参数的2D高斯替代59参数的3D高斯，用无序的累积求和替代需排序的α-blending，实现超快图像表示。

## 方法详解

### 整体框架

GaussianImage包含两个阶段：
1. **图像表示**: 用一组2D高斯点拟合图像，每个高斯仅8个参数（位置2 + 协方差3 + 加权颜色3）
2. **图像压缩**: 对拟合好的高斯属性分别进行量化感知微调+编码，可选bits-back coding进一步降低码率

### 关键设计

1. **紧凑的2D高斯表示**: 每个2D高斯由位置$\boldsymbol{\mu} \in \mathbb{R}^2$、协方差矩阵$\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$、颜色$\boldsymbol{c} \in \mathbb{R}^3$和不透明度$o \in \mathbb{R}$组成。协方差矩阵通过Cholesky分解保证正定性：

    $\boldsymbol{\Sigma} = \boldsymbol{L}\boldsymbol{L}^T$

   用Cholesky向量$\boldsymbol{l} = \{l_1, l_2, l_3\}$表示下三角元素，基础2D高斯共9参数——相比3D高斯的59参数压缩了$6.5\times$。

   也可用旋转+缩放分解：$\boldsymbol{\Sigma} = (\boldsymbol{RS})(\boldsymbol{RS})^T$，其中$\theta$为旋转角，$s_1, s_2$为缩放因子，同样3个参数。

2. **累积求和光栅化**: 3D GS的α-blending需要按深度排序高斯并计算累积透明度$T_n$，这对于无深度信息的2D图像不适用。本文提出直接加权求和：

    $\boldsymbol{C}_i = \sum_{n \in \mathcal{N}} \boldsymbol{c}_n \cdot o_n \cdot \exp(-\sigma_n), \quad \sigma_n = \frac{1}{2}\boldsymbol{d}_n^T \boldsymbol{\Sigma}^{-1} \boldsymbol{d}_n$

   进一步将颜色$\boldsymbol{c}_n$和不透明度$o_n$合并为加权颜色系数$\boldsymbol{c}_n' \in \mathbb{R}^3$（不再限制在$[0,1]$范围）：

    $\boldsymbol{C}_i = \sum_{n \in \mathcal{N}} \boldsymbol{c}_n' \cdot \exp(-\sigma_n)$

   最终2D高斯仅需**8个参数**（位置2 + 协方差3 + 加权颜色3），压缩比达$7.375\times$。

   三大优势：(a) 对高斯顺序不敏感，无需排序；(b) 跳过累积透明度$T_n$的顺序计算，加速训练和推理；(c) 所有覆盖该像素的高斯都参与渲染，充分利用信息。

3. **图像压缩流水线**: 对不同属性采用不同量化策略：

    - **位置参数**: 16-bit float精度（对量化敏感）
    - **协方差参数**: $b$-bit（默认6-bit）非对称量化，学习缩放因子$\gamma_i$和偏移$\beta_i$：
   
    $\hat{l}_i^n = \lfloor \text{clamp}(\frac{l_i^n - \beta_i}{\gamma_i}, 0, 2^b - 1) \rceil$
   
    - **加权颜色系数**: 残差向量量化（RVQ），级联$M=2$阶段、码本大小$B=8$：
   
    $\hat{\boldsymbol{c}}_n^{\prime m} = \sum_{k=1}^{m} \mathcal{C}^k[i^k]$
   
    - **可选Partial Bits-back Coding**: 利用高斯点集合的置换不变性，先用普通熵编码编码前$K$个高斯作为初始比特，剩余$N-K$个高斯用bits-back coding编码，可节省$\log(N-K)! - \log(N-K)$比特。

### 损失函数 / 训练策略

- **图像表示**: L2 loss优化高斯参数，Adan优化器，初始学习率$1 \times 10^{-3}$，每20000步减半，共50000步
- **图像压缩**: $\mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_c$，其中$\mathcal{L}_c$为RVQ码本的commitment loss
- 无需3D GS中的自适应密度控制（2D图像空间无空白区域）
- 码本用K-means初始化（5次迭代），训练中用指数移动平均更新
- 基于gsplat库实现，自定义CUDA kernel

## 实验关键数据

### 主实验——图像表示（Kodak数据集）

| 方法 | PSNR↑ | MS-SSIM↑ | 训练时间(s)↓ | FPS↑ | 显存(MiB)↓ | 参数量(K)↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| WIRE | 41.47 | 0.9939 | 14338 | 11 | 2619 | 137 |
| SIREN | 40.83 | 0.9960 | 6582 | 29 | 1809 | 273 |
| I-NGP | 43.88 | 0.9976 | 491 | 1297 | 1525 | 300 |
| NeuRBF | 43.78 | 0.9964 | 992 | 663 | 2091 | 337 |
| 3D GS | 43.69 | 0.9991 | 340 | 859 | 557 | 3540 |
| **Ours** | **44.08** | 0.9985 | **107** | **2092** | **419** | 560 |

关键数据：训练速度比I-NGP快$4.6\times$，渲染速度$2092$ FPS（WIRE的$188\times$），显存仅$419$ MiB（最低）。

### 图像压缩——计算复杂度对比（DIV2K）

| 方法 | 低bpp PSNR↑ | 编码FPS | **解码FPS↑** |
|------|:---:|:---:|:---:|
| JPEG | 25.29 | 609 | 615 |
| JPEG2000 | 27.28 | 3.5 | 4.3 |
| Ballé17 | 27.72 | 21 | 19 |
| Ballé18 | 28.75 | 17 | 16 |
| COIN | 25.80 | 5.3e-4 | 166 |
| **Ours** | 25.66 | 4.1e-3 | **1971** |

解码速度比COIN快$12\times$，比JPEG快$3\times$，达到约**2000 FPS**。

### 消融实验

| 配置 | PSNR↑ | 训练时间(s)↓ | FPS↑ | 参数(K)↓ | 说明 |
|------|:---:|:---:|:---:|:---:|------|
| 3D GS (L1+SSIM) | 37.75 | 285 | 1067 | 1770 | 基线 |
| 3D GS (L2) | 37.41 | 198 | 1190 | 1770 | 换L2加速 |
| 2D GS（无AR、无M） | 37.89 | 105 | 2340 | 270 | 2D高斯+α-blending |
| +累积求和(AR) | **38.69** | 99 | 2555 | 270 | +0.8dB！ |
| +合并颜色和不透明度(M) | 38.57 | 91 | 2565 | **240** | 参数减10% |

### 量化策略消融（BD指标，锚定为最终方案）

| 变体 | BD-PSNR(dB) | BD-rate(%) |
|------|:---:|:---:|
| 最终方案（锚定） | 0 | 0 |
| 无$\mathcal{L}_c$ + RVQ + 6bit | -3.145 | +333% |
| 无$\mathcal{L}_c$ + 无RVQ + 6bit | -0.159 | +7.02% |
| 无$\mathcal{L}_c$ + 无RVQ + 8bit | -0.195 | +11.69% |

### 关键发现

- 累积求和替代α-blending贡献了最大的性能提升（+0.8 dB PSNR），同时还加速了训练和推理
- 2D高斯比3D高斯的参数量减少$6.5\times$（270K vs 1770K），但表示质量更好
- L2 loss最适合本方法（优于L1、SSIM及组合），这与3D GS中L1+SSIM最优不同
- RVQ对颜色属性的压缩至关重要——不同高斯的颜色向量存在相似性，适合码本编码
- Cholesky分解和旋转+缩放分解表示能力等价，但压缩鲁棒性不同（默认用Cholesky）

## 亮点与洞察

- **从3D到2D的优雅适配**: 不是简单固定相机参数，而是从根本上重新设计了高斯表示（2D化、参数合并）和光栅化算法（累积求和），每一步都有明确的动机
- **置换不变性的巧妙利用**: 累积求和对高斯顺序不敏感→可用bits-back coding利用$N!$种等价排列来节省码率，这是α-blending方案无法做到的
- **性能-效率的帕累托最优**: 在表示质量（44.08 dB）、训练速度（107s）、渲染速度（2092 FPS）、显存（419 MiB）四个维度同时达到最优或次优
- **丢弃自适应密度控制的洞察**: 2D图像空间无空白区域，3D GS中的split/clone策略是不必要的

## 局限与展望

- 压缩性能在高码率区间弱于VAE-based方法（Ballé17/18），因缺少自回归上下文模型
- Bits-back coding虽然理论性能好但处理延迟高，与"超快编解码"的目标矛盾
- 当前每张图像独立拟合高斯，无法利用图像间的冗余（如视频中的时间冗余）
- 随图像分辨率增大，所需高斯点数量增加，扩展性有待验证
- 可探索自适应高斯数量（根据图像复杂度动态调整）

## 相关工作与启发

- 3D Gaussian Splatting的成功证明了显式表示+可微光栅化的强大能力，本文将这一范式首次引入2D图像领域
- 与INR方法形成互补：INR用连续隐式函数，GaussianImage用离散显式高斯——后者更适合高速解码场景
- RVQ在音频编码（SoundStream）中已被广泛使用，本文证明其同样适用于2D高斯颜色属性的压缩
- 未来可与自回归熵模型结合提升压缩性能，或扩展到视频表示

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首创2D Gaussian Splatting图像表示范式，累积求和和参数合并设计精巧
- 实验充分度: ⭐⭐⭐⭐ Kodak和DIV2K两个标准数据集，完整的表示/压缩对比，详细消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，技术细节完整，从3D GS到2D的适配过程逻辑通顺
- 价值: ⭐⭐⭐⭐⭐ 2000 FPS解码速度开创了神经图像编解码器的新milestone，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](../../AAAI2026/3d_vision/gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression](hac_hash-grid_assisted_context_for_3d_gaussian_splatting_compression.md)

</div>

<!-- RELATED:END -->
