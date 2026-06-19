---
title: >-
  [论文解读] PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation
description: >-
  [ICCV 2025][图像生成][视频定制化] 本文提出 PersonalVideo 框架，通过混合奖励监督（身份一致性奖励+语义一致性奖励）直接对生成视频施加反馈，消除了传统方法中 T2I 调优与 T2V 推理之间的分布差距，在保持高身份保真度的同时避免了运动动态和语义跟随的退化。 文本到视频（T2V）生成已经取得了显著…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "视频定制化"
  - "身份保持"
  - "奖励监督"
  - "T2V生成"
  - "语义一致性"
---

# PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation

**会议**: ICCV 2025  
**arXiv**: [2411.17048](https://arxiv.org/abs/2411.17048)  
**代码**: [https://personalvideo.github.io/](https://personalvideo.github.io/)  
**领域**: 图像生成  
**关键词**: 视频定制化, 身份保持, 奖励监督, T2V生成, 语义一致性

## 一句话总结

本文提出 PersonalVideo 框架，通过混合奖励监督（身份一致性奖励+语义一致性奖励）直接对生成视频施加反馈，消除了传统方法中 T2I 调优与 T2V 推理之间的分布差距，在保持高身份保真度的同时避免了运动动态和语义跟随的退化。

## 研究背景与动机

文本到视频（T2V）生成已经取得了显著进展，但针对特定身份的人物视频生成仍不成熟。核心目标是给定少量用户照片，生成保持高身份保真度（ID fidelity）的多样化视频，让特定人物在不同动作、场景和风格中出现。

### 现有方法的痛点

现有视频身份定制化方法（如 MagicMe、DreamBooth for video）主要沿用图像定制化的思路：在 T2I 模型上用参考图像做重建训练来注入身份，然后将定制后的先验注入 T2V 模型进行推理。然而，这种策略带来了一个核心矛盾——**调优-推理差距（tuning-inference gap）**：

**分布不一致**：T2I 模型与 T2V 模型的先验分布存在偏差。在 T2I 上用静态图像重建来学习身份，会显著偏移 T2V 模型的视频先验，导致生成的视频趋于静态（动态退化）且无法跟随提示词（语义退化）。

**身份保真度不足**：由于调优阶段是在图像上做重建而推理阶段是在视频上生成，这种差距也让身份注入的效果打折扣。人眼对面部特征高度敏感，要求更高的一致性。

**数据需求过多**：为了在保持动态的同时注入身份，传统方法通常需要更多参考图片甚至额外的视频输入，给用户带来极大不便。

### 本文的切入角度

作者提出的核心洞察是：**不在图像上做重建，而是直接对 T2V 模型生成的视频施加奖励监督**。这样做有两个好处：一是训练和推理都在视频域，从根本上消除了调优-推理差距；二是可以利用多种奖励信号同时优化身份保真度和语义/动态保持。

## 方法详解

### 整体框架

PersonalVideo 的训练流程如下：从纯噪声出发，用目标 T2V 模型生成视频，然后在生成视频上同时施加两种奖励——身份一致性奖励（ICR）和语义一致性奖励（SCR）。在优化过程中，还采用模拟提示增强（Simulated Prompt Augmentation），从 LLM 生成的多样化提示词中随机采样来训练。可学习模块采用隔离式身份适配器（Isolated Identity Adapter），仅在去噪的后期步骤注入身份。

### 关键设计

1. **身份一致性奖励（ICR）**:

    - 功能：让生成视频中的人物面部与参考图像的身份一致
    - 核心思路：使用预训练的身份识别模型 $\mathcal{R}_{id}$ 提取参考图像和生成视频中随机选取帧的面部 ID 嵌入，最小化两者的余弦相似度损失：
    $\mathcal{L}_{\text{ICR}} = \mathbb{E}_{i, c \sim p(c)} \left[ \text{CosSim}\left(\mathcal{R}_{id}(I_{ref}), \mathcal{R}_{id}(G_{\mathcal{T}}(z_T, c, i))\right) \right]$
      其中 $G_{\mathcal{T}}$ 是目标 T2V 模型（带 VAE 解码器），$c$ 是包含特定关键词的文本提示
    - 设计动机：不同于重建目标需要对应参考图像，ICR 直接在生成视频上评估身份相似度，完全对齐推理时的分布。训练时使用面部裁剪和颜色抖动等增强来提升对少量参考图的鲁棒性

2. **语义一致性奖励（SCR）**:

    - 功能：维持原始 T2V 模型的语义分布，防止动态和语义退化
    - 核心思路：利用语义奖励模型 $\mathcal{R}_{sem}$ 评估原始模型和目标模型生成视频帧的图文对应分数，将分数归一化为概率分布后用 KL 散度对齐：
    $V_c^{\mathcal{S}} = \text{Softmax}(\{\mathcal{R}_{sem}(G_{\mathcal{S}}(z_T, c, i))\}_{i=1}^{M})$
    $V_c^{\mathcal{T}} = \text{Softmax}(\{\mathcal{R}_{sem}(G_{\mathcal{T}}(z_T, c, i))\}_{i=1}^{M})$
    $\mathcal{L}_{\text{SCR}} = \mathbb{E}_{c \sim p(c)} D_{KL}(V_c^T \| V_c^S)$
      其中 $G_{\mathcal{S}}$ 是冻结的原始模型，$M$ 是采样帧数
    - 设计动机：身份注入不可避免地引入分布偏移（因为使用有限的静态图像），SCR 通过对齐语义分布而非直接约束像素，能够在不影响身份注入的前提下保持原始模型的动态和语义能力

3. **模拟提示增强（Simulated Prompt Augmentation）**:

    - 功能：用 LLM 生成 50 个与参考图无关的多样化提示词，在训练中随机采样
    - 核心思路：传统重建方法只能使用描述参考图的提示，限制了泛化能力。由于本框架不做重建，可以引入任意语义场景的提示词（如"V 弹小提琴""V 在海滩微笑"），与实际测试场景高度对齐
    - 设计动机：独立于参考图像、不受参考图数量限制，有效缓解过拟合，即使只有单张参考图也能保持强鲁棒性

4. **隔离式身份适配器（Isolated Identity Adapter）**:

    - 功能：仅在去噪后期步骤注入身份信息
    - 核心思路：观察发现视频去噪过程中，人物运动在早期步骤形成，后期步骤负责恢复外观细节。因此采用 LoRA 风格的低秩适配器，仅在最后 1/4 去噪步骤激活：
    $\tilde{W} = W + \Delta W = W + A^{\text{down}} A^{\text{up}}$
    - 设计动机：在早期步骤不干预运动生成，最大限度减少对原始视频动态的影响

### 损失函数 / 训练策略

总训练目标为两个奖励的简单相加：

$$\mathcal{L}_{\text{train}} = \mathcal{L}_{\text{ICR}} + \mathcal{L}_{\text{SCR}}$$

使用 ResNet-100 (Glint360K 预训练) 作为身份奖励模型，HPSv2 作为语义奖励模型。在 DiT 架构的 HunyuanVideo 和 UNet 架构的 AnimateDiff 上均验证了有效性。

## 实验关键数据

### 主实验

| 方法 | Face Sim.↑ | Dyna. Deg.↑ | FVD↓ | T. Cons.↑ | CLIP-T↑ | CLIP-I↑ |
|------|-----------|-------------|------|-----------|---------|---------|
| DreamBooth | 42.62 | 13.86 | 1325.89 | 0.9919 | 26.26 | 44.27 |
| MagicMe | 50.51 | 11.88 | 1336.73 | 0.9928 | 25.48 | 73.03 |
| IDAnimator | 43.88 | 14.33 | 1538.44 | 0.9912 | 24.33 | 50.23 |
| ConsisID | 53.22 | 15.22 | 1622.21 | 0.9923 | 25.39 | 74.58 |
| **PersonalVideo** | **62.35** | **17.80** | **1272.32** | **0.9935** | **26.30** | **76.48** |

PersonalVideo 在身份相似度上大幅领先（62.35 vs 53.22），动态度也最高（17.80 vs 15.22），同时 FVD 最低，说明生成视频既保真又自然。

### 消融实验

| 配置 | Face Sim.↑ | CLIP-T↑ | Dynamic↑ | 说明 |
|------|-----------|---------|----------|------|
| T2I w/o Aug | 51.56 | 22.40 | 16.30 | T2I上训练，有调优-推理差距 |
| T2V w/o Aug | 60.26 | 25.50 | 17.20 | T2V上训练，Face大幅提升 |
| T2V w/ Aug | 61.05 | 28.59 | 17.85 | +提示增强，CLIP-T提升3+ |
| w/o SCR | 61.08 | 26.38 | 13.22 | 无语义一致性奖励，动态退化严重 |
| w/ SCR | 61.05 | 28.59 | 17.85 | +SCR，动态度从13→18 |
| All steps注入 | 62.37 | 26.95 | 13.93 | 全步骤注入身份，动态退化 |
| 1/4 steps注入 | 63.90 | 27.47 | 18.00 | 仅后1/4步骤注入，动态+Face双优 |

### 关键发现

- 在 T2V 模型上直接训练比在 T2I 上训练，Face Similarity 提升约 10 个点（51→61）
- SCR 对保持动态度至关重要：无 SCR 时动态度仅 13.22，加入后提升至 17.85
- 隔离式注入（后1/4步骤）不仅提升了动态度（13.93→18.00），Face Similarity 也略有改善
- 用户研究中 PersonalVideo 在身份保真、文本对齐、动态度、整体质量四个维度均显著优于对比方法

## 亮点与洞察

- 非重建式奖励训练范式：跳出了"重建参考图→注入身份"的固有思路，直接对生成视频做奖励反馈，从根本上消除了调优-推理差距
- SCR 的设计非常优雅：不直接约束语义内容，而是对齐语义评分的分布，既保持了灵活性又避免了对身份注入的干扰
- 隔离式身份注入利用了去噪过程的时序特性（早期=运动，后期=外观），简单但有效
- 与社区 LoRA（如卡通风格、国风）的兼容性为实际应用提供了极大灵活性

## 局限与展望

- 无法生成包含多个定制身份的视频（受限于 T2V 模型本身的能力）
- 效果依赖底层 T2V 模型的质量和能力
- 奖励模型的选择可能影响最终效果，文中未深入讨论不同奖励模型的比较
- 未来可探索通过解耦注意力图来支持多身份定制

## 相关工作与启发

- 与 PuLID 等编码器方法的思路有关联，都使用 ID 损失而非重建损失，但 PersonalVideo 将其扩展到了视频域
- SCR 的分布对齐思想可以迁移到其他需要保持模型原始能力的微调场景
- 模拟提示增强的策略在其他定制化任务中也值得借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[ICCV 2025\] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)
- [\[ICCV 2025\] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers](freecus_free_lunch_subject-driven_customization_in_diffusion_transformers.md)
- [\[ICCV 2025\] Generating Multi-Image Synthetic Data for Text-to-Image Customization](generating_multi-image_synthetic_data_for_text-to-image_customization.md)
- [\[NeurIPS 2025\] From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging](../../NeurIPS2025/image_generation/from_cradle_to_cane_a_two-pass_framework_for_high-fidelity_lifespan_face_aging.md)

</div>

<!-- RELATED:END -->
