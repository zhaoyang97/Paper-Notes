---
title: >-
  [论文解读] Architecture-Agnostic Untrained Network Priors for Image Reconstruction with Frequency Regularization
description: >-
  [ECCV2024][医学图像][untrained network prior] 提出三种与架构无关的频率正则化技术（带宽受限输入、带宽可控上采样、Lipschitz 正则化卷积层），统一解决 untrained network prior 的架构敏感性、过拟合和运行效率问题，在 MRI 重建任务中显著缩小不同架构间的性能差距。
tags:
  - "ECCV2024"
  - "医学图像"
  - "untrained network prior"
  - "deep image prior"
  - "spectral bias"
  - "frequency regularization"
  - "MRI reconstruction"
---

# Architecture-Agnostic Untrained Network Priors for Image Reconstruction with Frequency Regularization

**会议**: ECCV2024  
**arXiv**: [2312.09988](https://arxiv.org/abs/2312.09988)  
**代码**: [GitHub](https://github.com/YilinLiu97/Untrained-Recon)  
**领域**: 医学图像  
**关键词**: untrained network prior, deep image prior, spectral bias, frequency regularization, MRI reconstruction

## 一句话总结

提出三种与架构无关的频率正则化技术（带宽受限输入、带宽可控上采样、Lipschitz 正则化卷积层），统一解决 untrained network prior 的架构敏感性、过拟合和运行效率问题，在 MRI 重建任务中显著缩小不同架构间的性能差距。

## 背景与动机

受 Deep Image Prior (DIP) 启发的 untrained network 在无需训练集的情况下，仅利用损坏/部分测量数据即可恢复高质量图像。其成功被广泛归因于合适网络架构的 spectral bias（频谱偏置）——即 CNN 倾向于先拟合低频信号再拟合高频信号（如噪声）的隐式正则化效应。

然而，现有方法面临三个互相关联的核心挑战：

1. **架构选择困难**：不同架构（深度、宽度）对重建性能影响巨大，缺乏统一的架构选择准则，搜索最优架构成本高昂
2. **过拟合风险**：网络容易过拟合到噪声或不完整测量，需要 early stopping 等策略，但这些方法不稳定且无法提升架构本身能力
3. **运行效率低**：逐图像优化的方式固有地较慢，过参数化架构进一步加剧了这一问题

现有工作大多分别处理这些问题（如神经架构搜索、early stopping、预训练微调等），而本文的核心动机是：既然 spectral bias 是 DIP 成功的根本原因，能否直接从频率角度调控任意架构的正则化效果，从而同时解决上述三个挑战？

## 核心问题

如何设计与网络架构无关的频率正则化方法，使得不同配置（深度、宽度）的 untrained network 都能获得接近最优的重建性能，同时避免过拟合和降低运行时间？

## 方法详解

作者识别了 DIP 框架中导致 spectral bias 的三个核心要素，并分别提出对应的正则化方法：

### 1. Bandwidth-Constrained Input（带宽受限输入）

传统 untrained network 使用均匀分布白噪声作为输入。从频率角度看，白噪声包含所有频率分量且幅度均匀，这促使网络快速收敛到高频成分，进而产生高频伪影。

本文提出两种限制输入带宽的方法：

- **高斯模糊**：对噪声输入施加高斯模糊滤波器 $\mathcal{G}_{s,\sigma}$，去除部分高频成分。滤波器大小 $s$ 和 $\sigma$ 为超参数
- **Fourier features**：用较低最大频率 $f_c \propto L$（如 $L=4$ 或 $L=8$）的 Fourier features 替代噪声输入，受控地引入频率多样性

实验表明，当 $L$ 增大到 16 时，Fourier features 的频率范围接近原始噪声，性能也随之恶化，验证了限制输入带宽的有效性。

### 2. Bandwidth-Controllable Upsampling（带宽可控上采样）

仅限制输入带宽对浅层网络效果显著，但随着网络深度增加效果递减——因为更多的网络层可以生成任意新的高频成分。

作者设计了基于 Kaiser-Bessel 窗的可控带宽上采样器：

1. 先将输入特征图与零交错（zero-interleaving）
2. 用可定制的低通滤波器进行卷积

Kaiser 窗提供对通带波纹和阻带衰减之间权衡的显式控制：

$$w(n) = I_0(\beta\sqrt{1-(2n/M)^2}) / I_0(\beta)$$

其中 $M$ 控制窗的空间范围，$\beta$ 控制阻带衰减程度（越大图像越平滑）。该即插即用的上采样器可在不同层使用不同的 $M$ 和 $\beta$，提供灵活精确的控制。

### 3. Lipschitz-Regularized Layers（Lipschitz 正则化层）

卷积层（带非线性）是唯一能生成新频率的操作。通过正则化其 Lipschitz 常数来控制输出对输入变化的敏感度：

- 每层设置可学习的 Lipschitz 常数 $k_\ell$
- 通过矩阵范数归一化约束权重：当矩阵范数超过学习到的约束时才进行归一化
- 用 SoftPlus 确保非负性

最终优化目标为：

$$\min_{\Theta, K} \mathcal{L}(\mathbf{y}; \mathbf{AG_\Theta(z)}) + \lambda \sum_{l=1}^{L} \text{SoftPlus}(\mathbf{k}_\ell)^2$$

其中 $\lambda$ 控制平滑粒度，$K$ 为所有层的可学习 Lipschitz 常数集合。

### 方法组合

三种方法互补：带宽受限输入主要惠及浅层架构，Kaiser 上采样主要惠及深层架构，Lipschitz 正则化在所有配置上提供额外增益。组合使用时效果最佳。

## 实验关键数据

### 数据集与设置
- fastMRI 多线圈膝关节和脑部 MRI，标准 4× 加速
- Stanford 3D FSE 膝关节数据集（域外评估）
- 基础架构：N 层编码器-解码器，全跳跃连接，3000 次迭代

### 缩小架构性能差距（fastMRI Brain）

| 方法 | A2_256 PSNR | A8_64 PSNR | A2_256 SSIM | A8_64 SSIM |
|------|------------|------------|-------------|------------|
| 无正则化 | 29.08 | 31.68 | 0.729 | 0.807 |
| 高斯模糊+Lips.+Kaiser | 32.50 | 33.85 | 0.836 | 0.885 |

原本表现最差的 A2_256 提升 +3.42 dB PSNR，最终所有架构性能趋于一致。

### 与基线方法对比（fastMRI Knee）

| 方法 | PSNR | SSIM | 运行时间 |
|------|------|------|---------|
| U-Net（有监督） | 31.15 | 0.776 | ~1.5天训练 + 0.1秒推理 |
| ZS-SSL | 32.00 | 0.773 | 26.1 分钟/切片 |
| DIP | 29.16 | 0.628 | 9.2 分钟/切片 |
| A2_64（本文） | 32.07 | 0.781 | 6.4 分钟/切片 |
| A8_64（本文） | 31.73 | 0.768 | 12.3 分钟/切片 |

紧凑的 A2_64 + 本文方法即可匹配甚至超越 ZS-SSL，运行时间快约 4 倍。

### 域外泛化（Stanford FSE Knee）
- A2_64（本文）：PSNR 31.43 / SSIM 0.790
- U-Net（有监督）：PSNR 29.16 / SSIM 0.724
- 无监督方法在域外数据上显著优于有监督方法

### 运行效率
- 相比 ZS-SSL 快最高 90 倍（从 ~1小时/切片 降至 ~5分钟/切片）
- 小模型经正则化后可达到大模型性能，进一步降低计算开销

## 亮点

1. **统一框架解决三个问题**：首次同时解决 untrained network prior 的架构敏感性、过拟合和运行效率问题，且不需要修改架构本身
2. **极简实现**：核心方法仅需几行代码即可实现（高斯模糊输入、替换上采样器、添加 Lipschitz 正则项）
3. **频率视角的洞察**：将白噪声输入与 Fourier features 类比，揭示输入带宽对过拟合的影响机制
4. **域外泛化优势**：无监督方法天然避免了有监督方法的分布偏移问题，域外 PSNR 超过 U-Net 约 2 dB
5. **与 early stopping 互补**：本文方法可与自验证 early stopping 结合，进一步缩短重建时间

## 局限与展望

1. **超参数调节**：高斯模糊的 $\{s, \sigma\}$、Kaiser 窗的 $\{M, \beta\}$ 需针对不同数据集和加速倍率调节（膝关节和脑部使用不同超参数），尚未实现完全自动化
2. **评估范围有限**：主要在 MRI 重建上验证，自然图像的去噪和修复实验较为简略
3. **仅 4× 加速**：主实验在 4× 欠采样下进行，更高加速倍率（如 8×）的实验不够充分
4. **理论分析不足**：三种方法为何互补、最优组合策略的理论基础尚不完善
5. **与最新方法对比缺失**：未与基于扩散模型的 MRI 重建方法对比

## 与相关工作的对比

| 方法类型 | 代表工作 | 本文优势 |
|---------|---------|---------|
| 架构搜索 | NAS for DIP | 无需搜索，任意架构均可提升 |
| Early stopping | ZS-SSL, Wang et al. | 提升架构能力本身而非仅防过拟合，且可互补 |
| 迁移学习 | 预训练+微调 | 无需额外训练集，避免昂贵预训练 |
| 传统正则 | Total Variation (TV) | TV 仅部分缓解过拟合，本文从频率角度根本性解决 |
| 有监督方法 | U-Net | 无需训练集，域外泛化更好 |

## 启发与关联

- **频率视角的通用性**：从频率调控角度理解和改进神经网络先验的思路可推广到其他逆问题（CT 重建、超分辨等）
- **输入设计的重要性**：白噪声输入的频率特性长期被忽视，本文表明精心设计输入带宽是一种零成本的强正则化手段
- **小模型+强正则 vs 大模型**：经频率正则化后，紧凑模型可比大模型表现更好，这一发现对资源受限的医学影像部署有实际价值
- **与 INR/NeRF 的联系**：将 untrained network 的噪声输入类比为 INR 的 Fourier features，建立了两个看似不同范式之间的桥梁

## 评分
- 新颖性: ⭐⭐⭐⭐ （频率正则化思路新颖，但单个技术并非全新）
- 实验充分度: ⭐⭐⭐⭐ （MRI 实验全面，自然图像略少）
- 写作质量: ⭐⭐⭐⭐⭐ （动机清晰，逻辑链完整，图表丰富）
- 价值: ⭐⭐⭐⭐ （解决实际痛点，实现简单，即插即用）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] D$^2$-FOSA: Dual-Diffusion Guided EEG-to-Image Reconstruction with Frequency-Oriented Semantic Alignment](../../CVPR2026/medical_imaging/d2-fosa_dual-diffusion_guided_eeg-to-image_reconstruction_with_frequency-oriente.md)
- [\[ECCV 2024\] Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging](brain-id_learning_contrast-agnostic_anatomical_representations_for_brain_imaging.md)
- [\[ECCV 2024\] Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction](domesticating_sam_for_breast_ultrasound_image_segmentation_via_spatial-frequency.md)
- [\[CVPR 2026\] Learning Diffeomorphism for Medical Image Registration with Time-Embedded Architectures Using Semigroup Regularization](../../CVPR2026/medical_imaging/learning_diffeomorphism_for_medical_image_registration_with_time-embedded_archit.md)
- [\[CVPR 2025\] Novel Architecture of RPA In Oral Cancer Lesion Detection](../../CVPR2025/medical_imaging/novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)

</div>

<!-- RELATED:END -->
