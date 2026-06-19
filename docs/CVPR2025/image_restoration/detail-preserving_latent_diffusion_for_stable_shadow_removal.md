---
title: >-
  [论文解读] Detail-Preserving Latent Diffusion for Stable Shadow Removal
description: >-
  [CVPR 2025][图像恢复][阴影去除] 本文提出两阶段Stable Diffusion微调方案用于阴影去除：第一阶段在latent空间微调去噪器完成主要阴影消除，第二阶段通过阴影感知的Detail Injection模块从VAE编码器提取特征调制解码器，恢复第一阶段丢失的高频细节，实现高质量且泛化性强的阴影去除。
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "阴影去除"
  - "潜在扩散模型"
  - "扩散模型"
  - "细节注入"
  - "跨数据集泛化"
---

# Detail-Preserving Latent Diffusion for Stable Shadow Removal

**会议**: CVPR 2025  
**arXiv**: [2412.17630](https://arxiv.org/abs/2412.17630)  
**代码**: 无  
**领域**: 图像恢复  
**关键词**: 阴影去除, 潜在扩散模型, Stable Diffusion微调, 细节注入, 跨数据集泛化

## 一句话总结
本文提出两阶段Stable Diffusion微调方案用于阴影去除：第一阶段在latent空间微调去噪器完成主要阴影消除，第二阶段通过阴影感知的Detail Injection模块从VAE编码器提取特征调制解码器，恢复第一阶段丢失的高频细节，实现高质量且泛化性强的阴影去除。

## 研究背景与动机

**领域现状**：深度学习阴影去除方法通过学习像素级映射取得显著进展（如ShadowFormer、HomoFormer），但因训练数据有限容易过拟合，泛化到未见数据时性能下降。

**现有痛点**：（1）现有方法在跨数据集评估中性能急剧下降；（2）直接用SD模型做阴影去除会丢失高频细节——SD的VAE在像素空间到latent空间的映射中存在有损压缩（W×H×3 → W/8×H/8×4）；（3）现有扩散阴影去除方法（如Refusion）端到端训练在阴影数据上，未利用预训练SD的丰富视觉先验。

**核心矛盾**：SD有强大的泛化先验但latent空间会丢失细节；直接在pixel空间做扩散计算成本高且丢失全局上下文。"彻底去阴影"和"保留非阴影区细节"两个目标可能矛盾。

**本文目标**：利用预训练SD先验实现高泛化性的阴影去除，同时保留输入图像的细粒度纹理细节。

**切入角度**：两阶段解耦——先在latent space做"粗略但全面"的阴影去除，再在pixel space做"精细但局部"的细节恢复。

**核心 idea**：Stage 1 微调LDM去阴影（固定VAE），Stage 2 用Detail Injection模块从VAE encoder提取特征注入decoder，恢复高频细节。DI模块拼接有阴影和无阴影的特征实现隐式阴影感知。

## 方法详解

### 整体框架
Stage 1：固定预训练VAE，将阴影图latent $z^x$ 作为条件，微调U-Net生成无阴影图latent $\hat{z}^y$，解码得到粗略无阴影图。Stage 2：固定VAE encoder和decoder，训练Detail Injection (DI) 模块——在decoder每一层，将encoder对阴影图的特征 $e_i$ 与decoder对粗略无阴影latent的特征 $d_i$ 拼接，经RRDB处理后加回decoder，恢复高频纹理。

### 关键设计

1. **Latent Space 阴影去除（Stage 1）**:

    - 功能：利用SD泛化先验完成阴影的主体去除
    - 核心思路：将阴影图的latent $z^x$ 通道拼接到噪声latent上作为条件输入U-Net。使用 $z_0$-prediction（直接预测干净latent）而非ε-prediction，实验显示方差更低、结果更稳定。使用DDIM快速采样
    - 设计动机：预训练VAE的latent空间虽然有损，但能有效表征无阴影图像（实验验证VAE对无阴影图的编解码质量可接受）。在低分辨率latent空间做全局self-attention，能捕获阴影/非阴影区域的远距离关系

2. **阴影感知细节注入（Stage 2, DI模块）**:

    - 功能：从原始阴影图中提取并注入"无阴影的"高频细节
    - 核心思路：DI模块接收encoder特征（包含原图细节但含阴影信息）和decoder特征（无阴影但缺少细节），通过RRDB网络学习选择性注入。额外融合DINOv2特征增强泛化性。关键：拼接两种特征使网络能隐式区分阴影区域（阴影区的encoder和decoder特征差异大，非阴影区差异小），从而只注入无阴影细节
    - 设计动机：PCA可视化显示RRDB中间特征确实对阴影区有不同响应（绿色标出），证明模块具有阴影感知能力。VAE decoder权重固定保护预训练先验

3. **$z_0$-prediction + 低方差采样**:

    - 功能：提供稳定一致的阴影去除输出
    - 核心思路：使用 $z_0$-prediction替代标准ε-prediction，输出方差从1.075（DeS3）降至0.146
    - 设计动机：阴影去除需要确定性结果（不像图像生成需要多样性），低方差采样确保每次运行结果一致

### 损失函数 / 训练策略
Stage 1: latent空间L2损失（预测 $z_0$）。Stage 2: L1损失 + LPIPS感知损失。

## 实验关键数据

### 主实验（ISTD+数据集）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| ShadowFormer | 32.90 | 0.979 | - |
| ShadowRefiner | 34.67 | 0.983 | - |
| **Ours (Stage 2)** | **35.02** | **0.985** | **最优** |
| DeS3 (扩散方法) | 31.33 | - | - |

### 消融实验

| 配置 | PSNR↑ | 方差↓ | 说明 |
|------|-------|------|------|
| Stage 1 (ε-pred) | 29.66 | 0.239 | 标准预测 |
| Stage 1 (z₀-pred) | 29.95 | **0.146** | 更稳定 |
| Stage 2 (完整) | **35.02** | 0.160 | 细节注入大幅提升 |

### 关键发现
- Stage 2 细节注入将PSNR提升约5dB（从29.95到35.02），效果显著
- $z_0$-prediction比ε-prediction方差降低39%，输出更稳定
- **跨数据集泛化**：在一个数据集训练在另一个测试时，本文方法性能下降最小，远优于其他方法
- 支持高分辨率输入（1920×1440），不需要patch-based处理

## 亮点与洞察
- 两阶段解耦设计精准——Stage 1用SD先验做"粗活"（全局阴影消除），Stage 2用轻量CNN做"细活"（局部细节恢复），各司其职
- DI模块的阴影感知机制巧妙——利用有/无阴影特征的差异隐式定位阴影区，无需显式阴影mask
- $z_0$-prediction在确定性恢复任务中的优势值得更广泛关注

## 局限与展望
- 两阶段推理比单阶段方法慢
- Stage 1 仍需多步扩散采样（DDIM），不如纯前馈方法快
- 训练需要配对的阴影/无阴影图像数据
- 可考虑将两个阶段蒸馏为单一前馈网络

## 相关工作与启发
- **vs Refusion**: Refusion在latent空间做扩散但端到端训练，未利用SD预训练先验，泛化差；本文利用预训练SD+两阶段微调显著提升泛化性
- **vs ShadowDiffusion/DeS3**: 在pixel空间做扩散，计算贵且方差大；本文在latent空间操作效率更高
- **vs ShadowFormer**: 纯前馈方法快但泛化受限；本文利用SD先验泛化性更强

## 评分
- 新颖性: ⭐⭐⭐⭐ 两阶段SD微调+细节注入的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ SOTA结果+跨数据集泛化+方差分析+完整消融
- 写作质量: ⭐⭐⭐⭐ 管线清晰，PCA可视化有说服力
- 价值: ⭐⭐⭐⭐ 实用的阴影去除方案，泛化性突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](../../CVPR2026/image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](../../CVPR2026/image_restoration/phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2025\] MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](mair_a_locality-_and_continuity-preserving_mamba_for_image_restoration.md)
- [\[CVPR 2025\] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)

</div>

<!-- RELATED:END -->
