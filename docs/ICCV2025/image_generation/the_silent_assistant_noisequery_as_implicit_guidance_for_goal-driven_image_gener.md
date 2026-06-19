---
title: >-
  [论文解读] The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation
description: >-
  [ICCV 2025][图像生成][初始噪声优化] 本文提出 NoiseQuery，一种免训练的 T2I 生成增强方法，通过预构建大规模噪声库并在推理时检索与用户目标最匹配的初始噪声，实现高级语义和低级视觉属性的细粒度控制，仅需 0.002 秒/prompt 的额外开销即可提升多种 T2I 模型和增强技术的效果。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "初始噪声优化"
  - "Noise Library"
  - "跨模型一致性"
  - "低层视觉属性控制"
  - "T2I增强"
---

# The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation

**会议**: ICCV 2025  
**arXiv**: [2412.05101](https://arxiv.org/abs/2412.05101)  
**代码**: [https://github.com/wangruoyu02/NoiseQuery](https://github.com/wangruoyu02/NoiseQuery)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 初始噪声优化, Noise Library, 跨模型一致性, 低层视觉属性控制, T2I增强

## 一句话总结

本文提出 NoiseQuery，一种免训练的 T2I 生成增强方法，通过预构建大规模噪声库并在推理时检索与用户目标最匹配的初始噪声，实现高级语义和低级视觉属性的细粒度控制，仅需 0.002 秒/prompt 的额外开销即可提升多种 T2I 模型和增强技术的效果。

## 研究背景与动机

**领域现状**：扩散模型在 T2I 生成中取得了显著成功，但仍面临生成内容与文本 prompt 不一致的问题。现有增强方法包括：微调 UNet（DPO）、增强文本编码器（LaVi-Bridge）、改进推理策略（CFG++）、以及优化初始噪声（ReNO）。噪声优化方向近年受到关注，因为同一初始噪声生成的图像在单个模型内常表现出高相似性。

**现有痛点**：
   - 现有噪声优化方法（如 ReNO）依赖迭代梯度反向传播，计算成本高（23.56s vs 基准 0.072s），且容易数值不稳定（梯度爆炸）
   - 这些方法通常针对特定模型设计，无法跨模型复用
   - 文本 prompt 天然擅长表达高级语义，但对低级视觉属性（颜色、纹理、锐度等）的控制力很弱，因为 CLIP 等文本编码器主要学习高级语义

**核心矛盾**：初始噪声对生成结果有重大影响（不仅在单模型内，跨模型也是如此），但现有方法要么忽视这一影响，要么以高成本的方式利用它。如何高效、通用地利用初始噪声中蕴含的生成倾向？

**本文目标** (a) 揭示初始噪声的跨模型生成一致性现象及其理论原因；(b) 构建可复用的噪声库，实现高效的噪声检索；(c) 利用噪声控制文本难以描述的低级视觉属性。

**切入角度**：从扩散模型的前向加噪过程分析，有限步数的噪声调度器无法完全消除原始图像信息（如 Stable Diffusion 的最终步仍保留 $\sqrt{\bar\alpha_T} = 0.068265$ 的原始信号），这导致模型在训练时"学会了"利用噪声中的残留信号作为快捷方式。推理时，模型继续依赖这些隐式知识来"解读"纯高斯噪声，使得初始噪声成为一个"沉默的助手"。

**核心 idea**：利用无条件生成（空 prompt）来揭示初始噪声的隐式生成倾向（generative posterior），预建大规模噪声库并提取多粒度特征索引，推理时通过特征匹配快速检索最优噪声，实现语义和低级属性的双重控制。

## 方法详解

### 整体框架

NoiseQuery 分为两个阶段：(1) **离线阶段**——采样大量高斯噪声（如 100K），用扩散模型在空 prompt 下生成对应的 generative posterior 图像，提取多粒度特征（CLIP、颜色、纹理等）构建噪声库；(2) **在线阶段**——给定用户目标（文本 prompt 和/或低级属性需求），提取对应特征，在噪声库中检索匹配度最高的噪声作为生成起点。

### 关键设计

1. **隐式生成倾向的理论分析**:

    - 功能：从理论上解释为什么初始噪声会影响生成结果，且这种影响是跨模型一致的
    - 核心思路：扩散模型的前向过程 $x_T = \sqrt{\bar\alpha_T} x_0 + \sqrt{1-\bar\alpha_T} \epsilon$。理论上 $\bar\alpha_T \to 0$ 需要无穷步，但实际有限步调度器下 $\bar\alpha_T > 0$（如 SD 中 $\sqrt{\bar\alpha_T} = 0.068265$），导致 $x_T$ 中残留原始图像信息。训练时模型学会利用这些残留作为捷径，推理时继续"解读"纯噪声中的类似模式
    - 设计动机：这解释了为什么不同架构（UNet-based SD vs Transformer-based PixArt-α）从同一噪声生成的图像在语义和视觉属性上高度相似——因为它们都学会了利用同一种噪声调度器的非渐近特性

2. **Generative Posterior 作为噪声代理**:

    - 功能：用空 prompt 生成的图像来揭示噪声的隐式信息
    - 核心思路：当不提供文本引导时，生成过程完全由噪声驱动，模型被迫解码噪声中保留的残留信号。生成的图像（generative posterior）展现出与后续有 prompt 引导生成相似的语义、空间布局、色调等倾向
    - 设计动机：generative posterior 是噪声隐式信息的可解释代理，为噪声库中的特征提取提供了操作对象

3. **多粒度特征索引与匹配**:

    - 功能：为每个噪声样本的 generative posterior 提取多种特征，支持多目标检索
    - 核心思路：根据不同生成目标提取对应特征类型和匹配函数——语义（CLIP/BLIP + cosine similarity）、风格（Gram Matrix + MSE）、颜色（RGB/HSV/LAB + 绝对差值）、纹理（GLCM + 欧氏距离）、形状（Hu Moments + 欧氏距离）、锐度（高频能量 + 绝对差值）。检索公式：$\epsilon^* = \arg\max_{\epsilon_i \in \mathcal{N}_{set}} S(\mathcal{F}_i, \mathcal{F}_O)$
    - 设计动机：文本 prompt 主要控制高级语义，而初始噪声天然编码了低级视觉属性（与像素空间直接相关），因此通过噪声检索可以填补文本控制的空白。支持多目标组合时采用渐进式重排序策略

### 损失函数 / 训练策略

NoiseQuery 是完全免训练的方法，不修改任何模型参数。100K 噪声库的离线构建是一次性成本，之后可跨模型复用。

## 实验关键数据

### 主实验

在 DrawBench 和 MSCOCO 上评估，覆盖 6 种模型和 4 种增强技术：

| 模型 | 增强方法 | ImageReward | PickScore | HPS v2 | CLIPScore | 时间 |
|------|---------|-------------|-----------|--------|-----------|------|
| SD 1.5 | Base | 0.04 | 21.11 | 24.57 | 30.90 | 1.334s |
| SD 1.5 | + NoiseQuery | **0.08** | **21.16** | **25.02** | **31.41** | 1.336s |
| SD 1.5 | + DPO | 0.09 | 21.29 | 25.02 | 31.19 | 1.350s |
| SD 1.5 | + DPO + NoiseQuery | **0.17** | **21.33** | **25.25** | **31.41** | 1.352s |
| SD-Turbo | Base | 0.26 | 21.78 | 25.23 | 31.29 | 0.072s |
| SD-Turbo | + NoiseQuery | **0.41** | **21.87** | **25.66** | **31.58** | 0.074s |
| SD-Turbo | + ReNO | 1.67 | 23.40 | 32.48 | 32.55 | 23.56s |
| SD-Turbo | + ReNO + NoiseQuery | **1.71** | **23.52** | **32.92** | **32.78** | 23.56s |
| PixArt-α | Base | 0.70 | 22.08 | 28.27 | 30.83 | 4.327s |
| PixArt-α | + NoiseQuery | **0.82** | **22.11** | **28.45** | **31.27** | 4.328s |

### 消融实验

**噪声库大小对性能的影响**：

| 库大小 | 0.5k | 1k | 2k | 5k | 10k | 50k | 100k |
|--------|------|----|----|----|----|-----|------|
| 匹配计算(×10⁻⁴s) | 1.39 | 1.39 | 1.39 | 1.39 | 1.40 | 1.40 | 1.51 |
| Argmax 选择(×10⁻⁴s) | 0.25 | 0.25 | 0.25 | 0.25 | 0.37 | 6.16 | 13.25 |
| CLIP Score | 31.51 | 31.53 | 31.57 | 31.59 | 31.66 | 31.73 | 31.74 |

### 关键发现

- **通用增强效果**：NoiseQuery 在所有 6 种模型上均带来一致的性能提升，且可以与 DPO、CFG++、ReNO、LaVi-Bridge 等 4 种不同类型的增强方法叠加使用，额外增益不冲突
- **极低开销**：每个 prompt 仅增加 0.002s（匹配+选择），与 ReNO 的 23.56s 形成鲜明对比
- **跨模型复用**：同一噪声库可以在 UNet-based（SD 系列）和 Transformer-based（PixArt-α）模型间共享，验证了跨架构一致性
- **低 CFG scale 下也有效**：使用 NoiseQuery 可以在低 CFG scale 下获得高质量输出，避免高 scale 带来的过饱和问题
- **噪声库增大的收益递减**：从 50K 到 100K 仅提升 0.01 CLIPScore，但 10K 库已能提供大部分收益
- **多样性无损**：选取 top-20 匹配噪声的结果仍有良好多样性（DIV 指标与随机噪声可比），说明 NoiseQuery 不会限制生成多样性

## 亮点与洞察

- **对扩散模型"设计缺陷"的创造性利用**：有限步噪声调度器导致的信号残留通常被视为问题（如 common diffusion noise schedules are flawed），但本文将其转化为优势——噪声中的残留信息成为可利用的生成线索。这种"化缺陷为特性"的思维方式非常巧妙
- **低级视觉属性控制是独特贡献**：这是少数系统性地利用初始噪声来控制颜色、纹理、锐度等低级属性的工作，填补了"文本控制高级语义，噪声控制低级属性"的互补框架
- **模型无关性和可组合性**：作为一个"基础层"增强方法，可以无缝叠加到任何 T2I pipeline 上，这种设计哲学在实际部署中极具价值

## 局限与展望

- 受限于基础模型的生成能力，NoiseQuery 无法超越模型本身的上限
- 对精细结构控制（如 Canny edge）的效果有限，噪声更擅长"柔性"控制
- 噪声库的离线构建需要生成大量图像（如 100K），初始成本不低
- 未探索噪声库的更新和维护策略——当新模型出现时是否需要重新构建？
- 论文声称跨模型一致性但未深入分析一致性程度的量化指标

## 相关工作与启发

- **vs ReNO**: 通过梯度反向传播迭代优化噪声，计算量大（×300+）且仅限单步模型；NoiseQuery 通过检索实现，无梯度计算，适用于所有模型
- **vs ControlNet**: 通过额外结构信号（边缘图、深度图）控制空间布局；NoiseQuery 通过初始噪声控制不同层面的属性，两者互补（实验已验证可组合使用）
- **vs Diffusion-DPO**: 需要重新训练模型参数；NoiseQuery 完全免训练，可直接叠加到 DPO 增强后的模型上获得额外提升

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 对噪声跨模型一致性的发现和"噪声即助手"的视角非常有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 6种模型+4种增强方法的全矩阵实验，高低级属性控制均有验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰，从理论分析到方法设计到实验验证一气呵成
- 价值: ⭐⭐⭐⭐ 实用性强，但提升幅度相对温和（在强基线上的边际收益有限）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FilmComposer: LLM-Driven Music Production for Silent Film Clips](../../CVPR2025/image_generation/filmcomposer_llm-driven_music_production_for_silent_film_clips.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](../../CVPR2026/image_generation/goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[ICCV 2025\] FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation](ficgen_frequency-inspired_contextual_disentanglement_for_layout-driven_degraded_.md)
- [\[ICCV 2025\] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing](anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)
- [\[ICCV 2025\] LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding](lift_latent_implicit_functions_for_task-_and_data-agnostic_encoding.md)

</div>

<!-- RELATED:END -->
