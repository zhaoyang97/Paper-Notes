---
title: >-
  [论文解读] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting
description: >-
  [CVPR 2025][图像生成][图像修复] TurboFill 提出一种三步对抗训练方案，直接在少步蒸馏扩散模型 DMD2 上训练修复适配器（ControlNet 架构），仅需 4 步推理即可实现超越多步 BrushNet 的高质量图像修复效果，训练成本降低 10 倍以上。 领域现状：基于扩散模型的图像修复已取得显著进展…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像修复"
  - "少步扩散模型"
  - "对抗训练"
  - "修复适配器"
  - "快速推理"
---

# TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting

**会议**: CVPR 2025  
**arXiv**: [2504.00996](https://arxiv.org/abs/2504.00996)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 图像修复, 少步扩散模型, 对抗训练, 修复适配器, 快速推理

## 一句话总结

TurboFill 提出一种三步对抗训练方案，直接在少步蒸馏扩散模型 DMD2 上训练修复适配器（ControlNet 架构），仅需 4 步推理即可实现超越多步 BrushNet 的高质量图像修复效果，训练成本降低 10 倍以上。

## 研究背景与动机

**领域现状**：基于扩散模型的图像修复已取得显著进展。BrushNet 使用 ControlNet 架构的适配器将背景条件注入冻结的扩散 U-Net，实现了高质量修复。然而多步扩散推理（通常 20-50 步）的计算开销严重限制了实际部署。

**现有痛点**：(1) 多步 BrushNet 推理慢（需 50+ 步），不适合实时应用；(2) 将预训练好的多步 BrushNet 适配器直接迁移到少步蒸馏模型 DMD2 上会导致色彩过饱和、语义不一致等严重伪影；(3) 在少步模型上单独用扩散损失训练适配器会产生模糊的低质量输出；(4) 两阶段方案（先训练多步适配器再蒸馏）因 BrushNet 的参数量大导致内存和计算开销巨大（需 64 A100 GPU 50 小时）。

**核心矛盾**：少步蒸馏模型能快速生成，但扩散损失在少步设定下无法提供足够的质量监督信号，导致适配器训练后的修复质量远不如多步模型。

**本文目标**：设计一种高效的单阶段训练方案，直接在少步蒸馏模型上训练修复适配器，实现 4 步推理即可达到甚至超越 50 步多步方法的质量。

**切入角度**：作者观察到扩散损失擅长场景理解但缺乏纹理细节，而 GAN 损失擅长提升纹理和细节但缺乏全局场景理解。两者互补的特性启发了混合训练方案。

**核心 idea**：用三步交替训练——慢生成器（SDXL）上用真实扩散损失训练适配器学习降噪方向，快生成器（DMD2）上用 GAN 损失训练适配器提升纹理质量，同时训练扩散判别器区分真假并用假扩散损失增强判别器的场景理解。

## 方法详解

### 整体框架

TurboFill 包含三个组件：快生成器（DMD2 + 适配器）、慢生成器（SDXL + 适配器）和扩散判别器。三者共享同一个修复适配器权重。训练在三步之间交替：Step 1 在慢生成器上用扩散损失更新适配器；Step 2 在快生成器上用 GAN 损失和背景保留损失更新适配器；Step 3 用真/假扩散损失和判别损失更新判别器。推理时仅使用快生成器，4 步推理即可。

### 关键设计

1. **慢生成器 + 真实扩散损失 (Step 1)**:

    - 功能：让适配器学习将噪声 latent 引导向真实图像分布的正确方向
    - 核心思路：将修复适配器与多步扩散模型 SDXL 组合为"慢生成器"。输入是拼接的噪声 latent $x_t$、背景图像 latent $x_{bg}$ 和下采样二值 mask $m$。适配器产生的特征通过残差连接注入 SDXL 的 U-Net。用标准 DDPM 的扩散损失 $\mathcal{L}^R_{Diff}$ 在 T=1000 步的完整时间步上训练，仅更新适配器参数
    - 设计动机：SDXL 作为教师提供了丰富的语义理解信号，在完整时间步上训练让适配器学到鲁棒的降噪能力，为快生成器的质量打下基础

2. **快生成器 + GAN 损失 (Step 2)**:

    - 功能：让适配器在少步推理下生成高质量纹理和细节
    - 核心思路：将适配器与 DMD2 组合为"快生成器"，用 LCM scheduler 的 4 个时间步 {999,749,499,249} 采样。单步生成干净 latent $\hat{x}_0$，送入扩散判别器计算 GAN 损失 $\mathcal{L}_\mathcal{G}$。同时对背景区域计算重建损失 $\mathcal{L}_{BG} = \|x_0 \odot m - \hat{x}_0 \odot m\|^2_2$。仅更新适配器
    - 设计动机：GAN 损失直接在生成图像的质量上提供梯度信号，擅长提升纹理锐度和细节真实感——这恰好是单独用扩散损失训练少步模型时最缺乏的。背景保留损失防止修复区域与背景之间出现明显边界

3. **扩散判别器 + 假扩散损失 (Step 3)**:

    - 功能：训练判别器区分真实和合成 latent，同时让判别器理解场景结构
    - 核心思路：扩散判别器由 SDXL 编码器 + 辅助编码器 + 卷积分类器组成。辅助编码器接收与适配器相同的三通道输入。用标准 GAN 判别损失 $\mathcal{L}_\mathcal{D}$ 区分真实 $x_t$ 和假 $\hat{x}_t$。关键创新是引入"假扩散损失" $\mathcal{L}^F_{Diff}$——用辅助解码器在假 latent 上预测噪声，让判别器同时学习场景理解
    - 设计动机：纯 GAN 损失的判别器只关注局部纹理差异，缺乏全局场景理解，可能导致生成不相关的物体。假扩散损失强制判别器理解图像的整体结构，使 GAN 损失的梯度信号更有意义

### 损失函数 / 训练策略

- Step 1: $\mathcal{L}^R_{Diff}$ 更新适配器（慢生成器上的真实扩散损失）
- Step 2: $\lambda_1 \mathcal{L}_\mathcal{G} + \lambda_2 \mathcal{L}_{BG}$ 更新适配器（快生成器上的 GAN + 背景损失）
- Step 3: $\mathcal{L}^F_{Diff} + \lambda_3 \mathcal{L}_\mathcal{D}$ 更新判别器（假扩散 + 判别损失）
- 超参 $\lambda_1=10^{-3}$, $\lambda_2=10^{-1}$, $\lambda_3=10^{-2}$
- 8 A100 GPU, batch size 2, gradient accumulation 4, lr=1e-5, AdamW, 40K iterations
- 训练数据：120 万互联网图像 + Florence-2 区域标题 + SAM2 分割 mask（LocalCaptionData）

## 实验关键数据

### 主实验

| 方法 | 步数 | Q-Align (mask) | CLIPIQA+ (mask) | Q-Align (whole) | CLIP Sim |
|------|------|------------|-----------|------------|------|
| TurboFill | 4 | 4.570 | 0.733 | 4.719 | 25.35 |
| BrushNet* (50步) | 50 | 4.469 | 0.714 | 4.531 | 25.39 |
| BrushNet* (4步) | 4 | 4.184 | 0.657 | 4.449 | 25.34 |
| PowerPaint V2 (50步) | 50 | **4.777** | **0.777** | 4.723 | **26.26** |
| SDXL-Inpainting (50步) | 50 | 4.246 | 0.667 | 4.617 | 24.85 |

在 HumanBench 上结论一致：TurboFill (4步) 在 mask region quality 上接近 50 步 PowerPaint V2、大幅超越同步数（4步）BrushNet。

### 消融实验

| 配置 | Q-Align (mask) | CLIPIQA+ (mask) | TOPIQ (mask) | 说明 |
|------|------------|-----------|----------|------|
| TurboFill (完整) | 4.570 | 0.733 | 5.275 | 完整三步训练 |
| - $\mathcal{L}_{BG}$ | 4.367 | 0.686 | 5.026 | 背景变色、边界明显 |
| - $\mathcal{L}^F_{Diff}$ | 4.188 | 0.655 | 4.870 | 出现冲突元素 |
| - $\mathcal{L}^R_{Diff}$ | 4.066 | 0.632 | 4.850 | 纹理差、背景不一致 |

### 关键发现

- **4 步 TurboFill 在多数指标上超越 50 步 BrushNet**，证明三步对抗训练方案的有效性
- 消融实验清晰展示了三个损失的互补性：扩散损失提供场景理解，GAN 损失提供纹理细节，背景损失提供区域一致性
- 使用 LocalCaptionData 显著提升了文本对齐能力，CLIP Sim 从 21.6 提升到 25.4
- PowerPaint V2 虽然在 IQA 指标上最高，但定性分析显示其修复结果过于锐利、有时出现结构扭曲（如双头老虎），整体图像质量反而不如 TurboFill 协调
- 训练效率提升 10 倍：TurboFill 仅需 8 A100 x 50h = 400 GPU-hours，而两阶段方案需要 8 V100 x 72h + 64 A100 x 50h ≈ 3776 GPU-hours

## 亮点与洞察

- **三步对抗训练的设计思路非常精巧**：慢生成器学方向、快生成器学质量、判别器学结构，三者形成完整的训练循环。这个范式可以推广到其他需要在少步模型上训练条件适配的任务
- **假扩散损失训练判别器**是一个深刻的洞察——纯 GAN 判别器只关注低级纹理差异，加入扩散损失让它同时理解高级语义，从而为生成器提供更有意义的梯度信号
- **LocalCaptionData 的构建思路**有启发性：用 Florence-2 做 dense region caption + SAM2 分割，自动构建大规模图像-mask-局部描述三元组数据集

## 局限与展望

- 仍然需要 SDXL 和 DMD2 两个模型参与训练，虽然推理时只需 DMD2，但训练时内存占用仍较大
- 没有与最新的 FLUX 或 SD3 系列模型对比
- 评估基准虽然引入了 DilationBench 和 HumanBench，但仍缺乏对修复区域与背景一致性的有效定量指标
- 改进方向：将三步训练方案应用于其他条件生成任务（如超分辨率、风格迁移）；探索 1 步推理的极限；在更大模型（SDXL → FLUX）上验证可扩展性

## 相关工作与启发

- **vs BrushNet**: BrushNet 用 ControlNet 架构实现高质量修复但需 50 步推理。TurboFill 直接在少步模型上训练，4 步超越 50 步 BrushNet。关键区别在于训练策略而非架构
- **vs 扩散蒸馏方法 (LCM, DMD2 等)**: 传统蒸馏路径是先训练完整模型再压缩步数，TurboFill 绕过蒸馏阶段直接在蒸馏后的少步模型上训练，效率高 10 倍
- **vs PowerPaint V2**: PowerPaint 在 IQA 指标上更高但视觉上过度锐化，且因基于 SD1.5 无法迁移到少步模型。TurboFill 在协调性和自然度上更优

## 评分

- 新颖性: ⭐⭐⭐⭐ 三步对抗训练方案新颖有效，假扩散损失是有洞察力的设计
- 实验充分度: ⭐⭐⭐⭐ 两个新基准、多方法对比、逐步消融分析完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 首次实现少步扩散模型上的高质量修复，对实际应用价值大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[CVPR 2025\] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)
- [\[CVPR 2025\] RAD: Region-Aware Diffusion Models for Image Inpainting](rad_region-aware_diffusion_models_for_image_inpainting.md)
- [\[CVPR 2026\] InverFill: One-Step Inversion for Enhanced Few-Step Diffusion Inpainting](../../CVPR2026/image_generation/inverfill_one-step_inversion_for_enhanced_few-step_diffusion_inpainting.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)

</div>

<!-- RELATED:END -->
