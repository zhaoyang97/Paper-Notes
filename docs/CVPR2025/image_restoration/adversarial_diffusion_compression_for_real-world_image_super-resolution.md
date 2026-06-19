---
title: >-
  [论文解读] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution
description: >-
  [CVPR 2025][图像恢复][Real-world ISR] 提出对抗扩散压缩（ADC）框架，将一步扩散模型 OSEDiff 蒸馏为精简的扩散-GAN 混合模型，实现 73% 推理时间压缩、78% 计算量削减、74% 参数缩减，同时保持生成质量，达到 34.79 FPS 实时超分。 领域现状：基于 Stable Dif…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "Real-world ISR"
  - "扩散模型"
  - "adversarial distillation"
  - "GAN"
  - "剪枝"
---

# AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2411.13383](https://arxiv.org/abs/2411.13383)  
**代码**: [Guaishou74851/AdcSR](https://github.com/Guaishou74851/AdcSR)  
**机构**: 北京大学 / 香港理工大学 / OPPO 研究院
**领域**: 图像复原 / 超分辨率  
**关键词**: Real-world ISR, diffusion compression, adversarial distillation, one-step diffusion, GAN, model pruning

## 一句话总结
提出对抗扩散压缩（ADC）框架，将一步扩散模型 OSEDiff 蒸馏为精简的扩散-GAN 混合模型，实现 73% 推理时间压缩、78% 计算量削减、74% 参数缩减，同时保持生成质量，达到 34.79 FPS 实时超分。

## 研究背景与动机

**领域现状**：基于 Stable Diffusion (SD) 的真实世界图像超分辨率方法取得显著成功（StableSR、DiffBIR、SeeSR 等），但多步推理阻碍实际部署。近期一步网络（OSEDiff、S3Diff）缓解了时延问题，但仍因依赖大型预训练 SD 模型而计算开销巨大。

**现有痛点**：
   - 多步扩散方法（StableSR 等）推理极慢（几十步迭代）
   - 一步扩散方法（OSEDiff）仍需完整 SD 模型：VAE 编码器、提示提取器、文本编码器、完整 UNet，参数量 1311M，单帧 0.07s
   - 没有方法同时解决"保质"和"提速"

**核心矛盾**：直接裁剪或移除 SD 模块会严重退化生成能力，但保留完整模型无法满足实时需求。

**切入角度**：系统分析 OSEDiff 各模块，将其分为可移除型（VAE 编码器、提示提取器、文本编码器等）和可剪枝型（UNet、VAE 解码器），设计两阶段方案逐步压缩。

**核心 idea**：模块移除 + 通道剪枝 + 分阶段对抗蒸馏 = 实时扩散超分。

## 方法详解

### 对抗扩散压缩（ADC）框架

#### 模块分类
将 OSEDiff 的模块分为两类：
- **可移除型**：VAE 编码器、提示提取器（RAM/BLIP2）、文本编码器、交叉注意力模块、时间步嵌入
- **可剪枝型**：去噪 UNet（保留前 75% 通道）、VAE 解码器（保留前 50% 通道）

#### 第一阶段：预训练剪枝 VAE 解码器
- 在 OpenImage + LAION-Face + LAION-Aesthetic 上从头训练 50% 通道剪枝的 VAE 解码器
- 训练 250K + 250K 步
- 损失：L1 + LPIPS + patch-based 对抗损失
- 恢复剪枝 VAE 解码器的图像解码能力

#### 第二阶段：对抗蒸馏
- 联合微调 25% 通道剪枝的 UNet + 第一阶段预训练的 VAE 解码器首层块
- 用 LoRA 适配的判别器进行对抗训练
- 损失：$\mathcal{L} = \mathcal{L}_{distill} + \lambda_{adv} \mathcal{L}_{adv}$

### 关键技术细节

1. **时序方向对齐（Temporal Direction Alignment, TDA）**
    - 移除文本编码器和时间步嵌入后，固定时间步 $t=1$ 并设噪声 $\epsilon = 0$
    - 使用直通估计器（Straight-through Estimator）处理不可微分的 clipping 操作

2. **UNet-VAE 连接优化**
    - 将 UNet 输出直接连接到 VAE 解码器最小分辨率的首层块
    - PixelUnshuffle 层（缩放因子 2）对齐潜空间与 LR 图像空间大小

3. **判别器设计**
    - 基于 LoRA 的轻量判别器，LoRA rank=4
    - 学习率 1e-6

## 实验关键数据

### 效率对比

| 方法 | 步数 | 推理时间(s) | FLOPs(G) | 参数量(M) | FPS |
|------|------|-----------|----------|----------|-----|
| StableSR | 200 | 11.50 | — | — | — |
| DiffBIR | 50 | 2.72 | — | — | — |
| OSEDiff | 1 | 0.11 | 513 | 1311 | — |
| S3Diff | 1 | 0.28 | — | — | — |
| **AdcSR** | **1** | **0.03** | **111** | **340** | **34.79** |

加速比：vs. StableSR **383×**，vs. OSEDiff **3.7×**，vs. S3Diff **9.3×**。

### 恢复质量对比（DRealSR）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|------|-------|-------|--------|--------|
| StableSR | 27.63 | — | 0.3317 | — |
| OSEDiff | 28.02 | — | 0.3087 | 0.2239 |
| **AdcSR** | **28.10** | — | **0.3046** | **0.2200** |

### 消融实验

| 配置 | PSNR↑ | LPIPS↓ | 参数量(M) | 时间(s) |
|------|-------|--------|----------|---------|
| 保留 VAE 编码器 | 27.97 | 0.3077 | 490 | 0.05 |
| **移除 VAE 编码器** | **28.10** | **0.3046** | **456** | **0.03** |

| 配置 | FID↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ |
|------|------|--------|---------|----------|
| 无连接优化 | 140.09 | 65.18 | 0.5807 | 0.6756 |
| **有连接优化** | **134.05** | **66.26** | **0.5927** | **0.7049** |

## 训练细节

### 硬件与规模
- 8× NVIDIA A100 (80GB)
- batch size 96
- Stage 1：OpenImage 250K步 + LAION-Face/Aesthetic 250K步
- Stage 2：LSDIR 200K步
- Real-ESRGAN 退化管线合成 LR-HR 对

### 降级管线
使用 Real-ESRGAN 的高阶退化模型，包含模糊、降采样、噪声、JPEG 压缩等多级退化。

## 亮点与洞察
- **模块分类思路**系统化：不是盲目剪枝，而是先分析各模块角色再决定移除/剪枝策略
- 移除 VAE 编码器反而**提升了质量**（PSNR +0.13），说明冗余组件可能引入噪声
- **34.79 FPS** 是首个扩散模型达到的实时超分速度
- 两阶段设计关键：先恢复 VAE 解码能力，再端到端蒸馏，避免一步到位导致的性能塌陷
- 直通估计器解决 clipping 不可微的问题，工程细节值得关注
- 与 OSEDiff 教师相比 LPIPS/DISTS 更优，蒸馏后"青出于蓝"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing](iterative_predictor-critic_code_decoding_for_real-world_image_dehazing.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](../../NeurIPS2025/image_restoration/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[CVPR 2026\] One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution](../../CVPR2026/image_restoration/one-step_diffusion_transformer_for_controllable_real-world_image_super-resolutio.md)
- [\[NeurIPS 2025\] FIPER: Factorized Features for Robust Image Super-Resolution and Compression](../../NeurIPS2025/image_restoration/fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)
- [\[CVPR 2026\] Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution](../../CVPR2026/image_restoration/time-aware_one_step_diffusion_network_for_real-world_image_super-resolution.md)

</div>

<!-- RELATED:END -->
