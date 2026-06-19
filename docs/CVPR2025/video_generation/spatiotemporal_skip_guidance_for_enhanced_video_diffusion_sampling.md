---
title: >-
  [论文解读] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling
description: >-
  [CVPR 2025][视频生成][视频扩散模型] STG（Spatiotemporal Skip Guidance）提出通过选择性跳过 Transformer 的时空层来构造隐式弱模型，作为原模型的退化版本进行自扰动引导，无需额外训练即可提升视频扩散模型的生成质量，同时保持样本多样性和运动动态性，克服了 CFG 在视频生成中导致多样性和动态性下降的根本缺陷。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频扩散模型"
  - "采样引导"
  - "层跳过"
  - "无训练"
  - "CFG替代"
---

# Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling

**会议**: CVPR 2025  
**arXiv**: [2411.18664](https://arxiv.org/abs/2411.18664)  
**代码**: [https://github.com/junhahyung/STGuidance](https://github.com/junhahyung/STGuidance)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 视频扩散模型, 采样引导, 层跳过, 无训练, CFG替代

## 一句话总结
STG（Spatiotemporal Skip Guidance）提出通过选择性跳过 Transformer 的时空层来构造隐式弱模型，作为原模型的退化版本进行自扰动引导，无需额外训练即可提升视频扩散模型的生成质量，同时保持样本多样性和运动动态性，克服了 CFG 在视频生成中导致多样性和动态性下降的根本缺陷。

## 研究背景与动机

**领域现状**：扩散模型已成为生成高质量图像、视频和 3D 内容的主流工具。CFG（classifier-free guidance）是最广泛使用的采样引导技术，通过对比有条件和无条件生成来增强生成质量。Autoguidance 提出用弱模型替代无条件模型来缓解 CFG 的问题，但需要额外训练弱模型，对大规模视频扩散模型不切实际。

**现有痛点**：CFG 在视频生成中存在严重缺陷——虽然能提升单帧图像质量，但会显著降低样本多样性和运动动态程度。这是因为 CFG 的引导方向过于强烈地推向条件模式，导致生成结果趋同和运动趋于静止。Autoguidance 虽能缓解此问题，但要为每个视频模型专门训练一个弱模型，代价高昂且不实用。

**核心矛盾**：质量提升与多样性/动态性之间的 trade-off——现有引导方法要么牺牲多样性换取质量（CFG），要么需要额外的训练开销（Autoguidance）。

**本文目标**：(1) 设计一种无训练的引导方法；(2) 提升视频质量的同时保持多样性和动态性；(3) 适用于不同架构的视频扩散模型。

**切入角度**：作者受 Autoguidance 启发，思考能否不通过训练而是通过自扰动来构造弱模型。观察到 Transformer 中跳过某些层会产生一个"对齐的、退化版本"的模型，这天然就是一个弱模型。

**核心 idea**：通过选择性跳过时空注意力或残差块来构造隐式弱模型，用原模型与跳过后模型的输出差异作为引导信号，即 $\hat{x} = x + s \cdot (x_{full} - x_{skip})$，其中 $s$ 为引导强度。

## 方法详解

### 整体框架
在视频扩散模型的每个去噪步骤中：(1) 用完整模型前向传播得到正常预测 $x_{full}$；(2) 跳过指定的时空层重新前向传播得到退化预测 $x_{skip}$；(3) 用两者的差异作为引导方向，按引导强度缩放后加到最终预测上。整个过程嵌入采样循环，无需修改模型权重。

### 关键设计

1. **残差跳过 (Residual Skip)**:

    - 功能：通过跳过完整残差块来产生退化输出
    - 核心思路：对于一个残差块 $\text{Res}(z_l) = f_l(z_l) + z_l$，残差跳过将其简化为恒等映射 $\text{Res}'(z_l) = z_l$，即完全跳过该层的非线性变换。这相当于移除了该层学到的所有特征调制能力，产生一个"更笨"但结构对齐的输出
    - 设计动机：残差跳过的扰动强度较大，适合注意力层简单或模型较浅的情况，能产生更明显的引导信号

2. **注意力跳过 (Attention Skip)**:

    - 功能：仅跳过自注意力计算来产生更温和的退化输出
    - 核心思路：在自注意力 $\text{SA}(Q,K,V) = \text{Softmax}(QK^T/\sqrt{d})V = AV$ 中，跳过注意力矩阵 $A$ 的计算，直接用恒等矩阵替代（即每个 token 只看自己）。这保留了大部分网络结构但破坏了 token 间的空间和时间关系
    - 设计动机：比残差跳过更温和，保留更多模型能力的同时仍能产生有效的退化信号。对于复杂的大规模模型，这种温和扰动的引导效果更稳定

3. **时空层选择策略**:

    - 功能：确定跳过哪些层以获得最佳引导效果
    - 核心思路：不是随机或全部跳过，而是选择性地跳过特定的时空注意力层。实验发现，跳过中间层的效果通常优于跳过浅层或深层。引导强度 $s$ 控制扰动的影响程度——在不同模型上需要调整这个超参数
    - 设计动机：不同层编码不同级别的信息（浅层编码低频结构，深层编码高频细节）。选择合适的层跳过可以在不破坏核心语义的前提下产生有效的质量差异信号

### 损失函数 / 训练策略
STG 完全无需训练，仅在推理时的采样循环中引入。引导公式为 $\hat{x} = x_{full} + s \cdot (x_{full} - x_{skip})$，额外的前向传播增加了约 50% 的推理成本（与 CFG 类似）。

## 实验关键数据

### 主实验

| 模型 | 引导方式 | Quality↑ | Semantic↑ | I.Q.↑ | Dyn.Deg.↑ | T.Flicker↓ |
|------|---------|---------|----------|-------|----------|------------|
| Mochi | CFG | 0.524 | 0.507 | 0.985 | 0.87 | 0.976 |
| Mochi | **STG** | **0.628** | **0.554** | **0.988** | 0.86 | **0.978** |
| Open-Sora | CFG | 0.561 | 0.493 | 0.982 | **0.902** | 0.975 |
| Open-Sora | **STG** | **0.606** | **0.509** | **0.987** | 0.895 | **0.976** |

| 模型 | 引导方式 | FVD↓ | IS↑ | Quality↑ | Semantic↑ | T.Flicker↓ | Dyn.Deg.↑ |
|------|---------|------|-----|---------|----------|------------|----------|
| SVD | CFG | 151.3 | 38.0 | 0.687 | 0.637 | 0.966 | 0.562 |
| SVD | **STG** | **128.7** | **38.5** | **0.694** | **0.639** | **0.968** | **0.694** |

### 消融实验

| 配置 | 核心指标 | 说明 |
|------|---------|------|
| STG (Residual Skip) | 高 | 适合小模型，扰动较强 |
| STG (Attention Skip) | 更高 | 适合大模型，扰动更温和 |
| 跳过浅层 | 中等 | 影响低频结构 |
| 跳过中间层 | 最优 | 平衡语义和细节 |
| 跳过深层 | 中等偏低 | 影响高频细节 |

### 关键发现
- STG 在所有测试的视频模型（Mochi、Open-Sora、SVD）上都优于 CFG，特别是在 Quality 和 Semantic 维度提升显著
- 最大的亮点是 **Dynamic Degree** 的保持：SVD+STG 的动态度从 0.562 提升到 0.694，而 CFG 往往会降低动态性
- 注意力跳过通常比残差跳过效果更好，因为扰动更温和、更可控
- 用户研究表明 STG 在视觉质量、文本对齐、运动自然度等多个维度上都优于 CFG
- 方法对引导强度 $s$ 较鲁棒，在较宽范围内都能保持良好效果

## 亮点与洞察
- **"层跳过即弱模型"的巧妙洞察**：将 Autoguidance 中需要单独训练的弱模型替换为通过跳层构造的隐式弱模型，这一简单但深刻的观察将一个昂贵的训练问题转化为零成本的推理技巧
- **质量-多样性-动态性的三角平衡**：CFG 只能在质量-多样性之间做二选一，STG 通过更温和的引导同时维持了三者。这对视频生成尤为关键——没有动态性的视频毫无意义
- **通用性极强**：方法适用于 Mochi（DiT架构）、Open-Sora（STDiT架构）、SVD（UNet架构）等多种模型，说明层跳过引导是一个通用的原理而非针对特定架构的 trick

## 局限与展望
- 额外的前向传播带来约 50% 的推理时间开销，对于实时视频生成场景可能偏高
- 最优跳过层的选择目前需要手动调试，缺乏自动化选择策略
- 层跳过的理论分析还不够深入——为什么跳过中间层比跳过浅层或深层效果好？
- 未在最新的 DiT 类长视频模型（如 CogVideoX）上测试
- 可以探索自适应的引导策略：根据去噪步骤动态调整跳过哪些层或调整引导强度
- 将这种思想扩展到图像扩散模型和 3D 生成也是有价值的方向

## 相关工作与启发
- **vs CFG**: CFG 用无条件模型作为引导基准，本质上是条件与非条件的对比。STG 用原模型的退化版本作为基准，引导方向更温和、更对齐，因此不会过度压缩多样性
- **vs Autoguidance (Karras et al. 2024)**: Autoguidance 首先提出用弱模型替代无条件模型的思想，但需要额外训练。STG 的层跳过方案完美继承了其理论优势，同时消除了训练开销
- **vs PAG (Perturbed Attention Guidance)**: PAG 用随机扰动注意力图来引导，STG 的层跳过方案更结构化，产生的退化输出更可预测，因此引导质量更稳定

## 评分
- 新颖性: ⭐⭐⭐⭐ 层跳过构造弱模型的想法简单但非常有效，是一个被忽视的好 idea
- 实验充分度: ⭐⭐⭐⭐⭐ 在三个不同架构的视频模型上验证，定量+定性+用户研究
- 写作质量: ⭐⭐⭐⭐ 概念清晰，实验展示充分
- 价值: ⭐⭐⭐⭐⭐ 作为CFG的即插即用替代方案，对整个视频生成社区有直接影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation](hunyuanportrait_implicit_condition_control_for_enhanced_portrait_animation.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICML 2026\] Self-Refining Video Sampling](../../ICML2026/video_generation/self-refining_video_sampling.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
