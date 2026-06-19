---
title: >-
  [论文解读] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis
description: >-
  [ICCV 2025][3D视觉][4D生成] 提出一种视频到4D生成框架，通过Direct 4DMesh-to-GS Variation Field VAE将动画数据直接编码为紧凑的高斯变化场潜在空间，再训练时序感知的扩散模型生成动态3D内容，在4.5秒内实现高保真4D合成，并展示了对真实视频输入的优越泛化能力。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "4D生成"
  - "视频到4D"
  - "高斯变化场"
  - "扩散模型"
  - "3D Gaussian Splatting"
---

# Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis

**会议**: ICCV 2025  
**arXiv**: [2507.23785](https://arxiv.org/abs/2507.23785)  
**代码**: [GVFDiffusion.github.io](https://GVFDiffusion.github.io)  
**领域**: 3D视觉  
**关键词**: 4D生成, 视频到4D, 高斯变化场, 扩散模型, 3D Gaussian Splatting

## 一句话总结

提出一种视频到4D生成框架，通过Direct 4DMesh-to-GS Variation Field VAE将动画数据直接编码为紧凑的高斯变化场潜在空间，再训练时序感知的扩散模型生成动态3D内容，在4.5秒内实现高保真4D合成，并展示了对真实视频输入的优越泛化能力。

## 研究背景与动机

4D生成——创建动态3D内容——是继图像、视频和3D生成之后的下一个前沿方向。现实世界现象固有地结合了空间和时间动态，但训练鲁棒的4D扩散模型面临两大技术挑战：

**大规模4D数据集构建成本高**：直接方法需要对每个3D动画序列拟合独立的动态高斯溅射（4DGS）表示，通常每个实例需要几十分钟（4DGaussians需6分钟，K-planes需30+分钟），计算昂贵且难以扩展

**高维表示难以直接建模**：同时表示3D形状、外观和运动通常需要超过100K tokens，使直接的扩散建模极其困难

**现有方法的不足**：
- 基于优化的方法（Consistent4D, STAG4D等）依赖SDS蒸馏，耗时1小时以上且存在时空不一致
- 前馈方法（L4GM）使用2D生成的多视角图像重建4DGS，但2D生成的多视角不一致会导致质量下降
- 缺乏原生的4D扩散模型，需要从2D/3D先验间接推导

**核心思路**：将4D生成分解为canonical 3DGS生成（利用已有3D模型）和高斯变化场（Gaussian Variation Fields）建模，通过直接编码3D动画数据绕过逐实例拟合，将高维运动信息压缩到紧凑潜在空间。

## 方法详解

### 整体框架

给定输入视频 $\mathcal{I} = \{I_t\}_{t=1}^T$，目标是生成3DGS序列 $\mathcal{G} = \{G_t\}_{t=1}^T$。分解为：
- **Canonical GS** $G_1$：使用第一帧生成静态3DGS（利用预训练3D模型）
- **高斯变化场** $\mathcal{V} = \{\Delta G_t\}_{t=1}^T$：描述每个高斯属性相对于 $G_1$ 的时序变化

框架包含两个主要组件：(1) Direct 4DMesh-to-GS Variation Field VAE；(2) Gaussian Variation Field扩散模型。

### 关键设计

1. **Direct 4DMesh-to-GS Variation Field VAE**：

   **编码过程**：
    - 将mesh动画序列转换为点云 $\mathcal{P} = \{P_t \in \mathbb{R}^{N \times 3}\}_{t=1}^T$（$N = 8192$）
    - 计算位移场 $\Delta P_t = P_t - P_1$
    - 使用预训练Mesh-to-GS编码器获取canonical GS：$G_1 = \mathcal{D}_{GS}(\mathcal{E}_{GS}(M_1))$
   
   **Mesh-guided插值机制**（关键创新）：为每个canonical高斯位置 $\bm{p}_1^i$ 找K近邻，使用自适应半径加权插值位移场：
   
    $\bm{w}_{i,k} = \exp(-\frac{\beta \bm{d}_{i,k}}{r_i^2}), \quad r_i = \sqrt{\frac{1}{K}\sum_{k=1}^K \bm{d}_{i,k}}$
   
    $\Delta \bm{p}_{t,i}^{interp} = \sum_{k=1}^K \frac{\bm{w}_{i,k}}{\sum_k \bm{w}_{i,k}} \Delta P_{t,n(i,k)}$
   
   然后对插值后的位移使用FPS采样得到运动感知query $\Delta \bm{p}_t^{fps} \in \mathbb{R}^{L \times 3}$，通过cross-attention编码为潜在表示 $\bm{z} \in \mathbb{R}^{T \times L \times C}$（$L = 512$, $C = 16$），将序列长度从 $N = 8192$ 压缩到 $L = 512$。
   
   **解码过程**：通过self-attention处理潜在表示后，使用canonical GS的所有参数作为query通过cross-attention解码出变化场 $\Delta G_t = \{\Delta \bm{p}_t, \Delta \bm{s}_t, \Delta \bm{q}_t, \Delta \bm{c}_t, \Delta \alpha_t\}$。

2. **Gaussian Variation Field扩散模型**：

   基于Diffusion Transformer（DiT）架构，核心创新：
    - **时序自注意力层**：在标准空间自注意力之外加入时序自注意力，捕捉帧间运动连贯性
    - **双条件注入**：通过cross-attention注入视频视觉特征 $\mathcal{C}^v$（DINOv2提取）和canonical GS几何特征 $\mathcal{C}^{GS}$
    - **位置先验嵌入**：基于canonical GS位置 $\bm{p}_1^{fps}$ 的位置编码，增强模型对空间位置和变化场对应关系的感知
   
   使用velocity prediction参数化，训练目标：
   
    $\mathcal{L}_{simple} = \mathbb{E}_{s, \bm{z}^0, \bm{\epsilon}} \left[ \|\hat{\bm{v}}_\theta(\alpha_s \bm{z}^0 + \sigma_s \bm{\epsilon}, s, \mathcal{C}) - \bm{v}^s\|_2^2 \right]$

3. **Mesh-guided Loss**：

   对齐预测的高斯位移与mesh插值得到的伪GT位移，是运动重建质量的关键：
   
    $\mathcal{L}_{mg} = \sum_{t=1}^T \|\Delta \mathbf{p}_t - \Delta \bm{p}_t^{interp}\|_2^2$

### 损失函数 / 训练策略

**VAE训练**：两阶段——先微调canonical GS decoder 150K iterations，再与其他模块联合训练200K iterations。总损失 $\mathcal{L}_{total} = \mathcal{L}_{img} + \lambda_{mg}\mathcal{L}_{mg} + \lambda_{kl}\mathcal{L}_{kl}$，其中 $\mathcal{L}_{img}$ 含L1 + LPIPS + SSIM。

**扩散模型训练**：在24帧序列上训练1300K iterations，使用cosine noise schedule和1000 timesteps。推理时支持32帧生成。

## 实验关键数据

### 主实验（视频到4D生成）

| 方法 | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP↑ | FVD↓ | 时间↓ |
|------|-------|--------|-------|-------|------|------|
| Consistent4D | 16.20 | 0.146 | 0.880 | 0.910 | 935.19 | ~1.5hr |
| SC4D | 15.93 | 0.164 | 0.872 | 0.870 | 833.15 | ~20min |
| STAG4D | 16.85 | 0.144 | 0.887 | 0.893 | 1008.40 | ~1hr |
| DreamGaussian4D | 15.24 | 0.162 | 0.868 | 0.904 | 799.56 | ~15min |
| L4GM | 17.03 | 0.128 | 0.891 | 0.930 | 529.10 | 3.5s |
| **本文** | **18.47** | **0.114** | **0.901** | **0.935** | **476.83** | **4.5s** |

本文在所有质量指标上均达最佳，PSNR提升1.44dB（vs L4GM），FVD降低9.9%（476.83 vs 529.10），时间仅4.5秒（3.0s canonical GS + 1.5s变化场扩散）。

### 消融实验

**VAE关键组件消融**：

| 配置 | Encoder Query | Mesh-guided Loss | Variation Attrs | PSNR↑ | LPIPS↓ | SSIM↑ |
|------|:---:|:---:|:---:|-------|--------|-------|
| A. 基线 | $\bm{p}_t^{fps}$ | ✗ | $\Delta\bm{p},\Delta\bm{s},\Delta\bm{q}$ | 23.25 | 0.0678 | 0.936 |
| B. +mesh loss | $\bm{p}_t^{fps}$ | ✓ | $\Delta\bm{p},\Delta\bm{s},\Delta\bm{q}$ | 26.17 | 0.0544 | 0.950 |
| C. +运动query | $\Delta\bm{p}_t^{fps}$ | ✓ | $\Delta\bm{p},\Delta\bm{s},\Delta\bm{q}$ | 28.58 | 0.0478 | 0.958 |
| D. 完整（本文） | $\Delta\bm{p}_t^{fps}$ | ✓ | 全部5项 | **29.28** | **0.0439** | **0.964** |

**扩散模型消融**：

| 方法 | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP↑ | FVD↓ |
|------|-------|--------|-------|-------|------|
| 无位置嵌入 | 17.86 | 0.121 | 0.897 | 0.931 | 547.20 |
| 完整模型 | **18.47** | **0.114** | **0.901** | **0.935** | **476.83** |

### 关键发现

1. **Mesh-guided loss是运动学习的关键**：Config A→B PSNR提升2.92dB，解决了缺乏GT高斯运动监督的核心困难
2. **运动感知query远优于静态位置query**：Config B→C PSNR提升2.41dB
3. **颜色和不透明度变化也很重要**：加入 $\Delta\bm{c}_t$ 和 $\Delta\alpha_t$ 额外提升0.7dB
4. **位置先验嵌入对扩散模型至关重要**：增强了空间位置与变化场的对应关系感知

## 亮点与洞察

- **绕过逐实例拟合**：直接从mesh动画编码高斯变化场，单次前向传播完成，避免了4DGS重建的高昂成本
- **4D分解策略高效**：将4D问题分解为canonical 3DGS生成 + 变化场建模，降低了扩散建模的维度
- **运动感知编码设计精妙**：mesh-guided插值架桥了mesh运动和高斯运动，运动感知query大幅提升编码质量
- **合成到真实的泛化**：仅在合成数据训练却能处理in-the-wild视频，证明了学到的运动先验的泛化性

## 局限与展望

- 训练数据仅34K动画物体，数据规模受限于高质量动画资源的稀缺
- 推理需要先通过预训练3D模型生成canonical GS，依赖第三方模型的质量
- 目前支持32帧动画生成，对长序列动画可能需要自回归或滑动窗口策略
- 对于复杂拓扑变化（如物体分裂/合并）的运动建模能力待验证

## 相关工作与启发

- 3DShape2VecSet的perceiver编码启发了本文的cross-attention编码架构
- Trellis的结构化潜在表示为canonical GS的高质量生成奠定了基础
- 4D问题的分解策略（canonical + variation）可推广到其他动态3D任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 原生4D扩散模型方向开创性工作，VAE直接编码策略绕过逐实例拟合
- **实验充分度**: ⭐⭐⭐⭐ 定量对比全面、消融清晰，但测试集仅100个物体
- **写作质量**: ⭐⭐⭐⭐ 框架描述清晰，技术细节完整，消融实验设计合理
- **价值**: ⭐⭐⭐⭐⭐ 4D生成的实用解决方案，4.5秒生成时间具有实际应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[ICCV 2025\] Not All Frame Features Are Equal: Video-to-4D Generation via Decoupling Dynamic-Static Features](not_all_frame_features_are_equal_video-to-4d_generation_via_decoupling_dynamic-s.md)
- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
