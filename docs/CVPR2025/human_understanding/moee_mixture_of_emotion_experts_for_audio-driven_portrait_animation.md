---
title: >-
  [论文解读] MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation
description: >-
  [CVPR 2025][人体理解][mixture of emotion experts] 提出情绪混合专家（MoEE）模型，为 6 种基础情绪各训练一个专家网络并通过 Soft MoE 门控组合，配合 150 小时专业情绪数据集和多模态情绪条件模块，实现对单一及复合情绪的精确、自然控制。 领域现状： 音频驱动说话人头像生成…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "mixture of emotion experts"
  - "audio-driven portrait animation"
  - "compound emotion"
  - "Action Units"
  - "DH-FaceEmoVid-150"
---

# MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation

**会议**: CVPR 2025  
**arXiv**: [2501.01808](https://arxiv.org/abs/2501.01808)  
**代码**: 待确认（数据集将公开发布）  
**领域**: 人体理解  
**关键词**: mixture of emotion experts, audio-driven portrait animation, compound emotion, Action Units, DH-FaceEmoVid-150

## 一句话总结

提出情绪混合专家（MoEE）模型，为 6 种基础情绪各训练一个专家网络并通过 Soft MoE 门控组合，配合 150 小时专业情绪数据集和多模态情绪条件模块，实现对单一及复合情绪的精确、自然控制。

## 研究背景与动机

**领域现状**: 音频驱动说话人头像生成在唇同步上已取得显著进展（如 Hallo、AniPortrait），但情绪控制仍然薄弱。现有方法（如 EAT、StyleTalk）要么支持的情绪种类有限，要么需要参考视频来传递风格，缺乏灵活的情绪控制能力。

**核心问题**:
1. **缺乏基础情绪建模框架**: 没有对单一情绪进行精准建模，导致复合情绪（如"愤怒地厌恶"、"悲伤地惊讶"）无法通过组合生成
2. **缺乏高质量情绪数据集**: 现有情绪数据集（如 MEAD）规模小、情绪类别有限、缺少细粒度标注（AU labels、文本描述）

**动机**: 受 MoE 架构启发——如果每种基础情绪有对应的专家模型，那么复合情绪可以通过专家权重的软组合来合成，类似"调色板"原理。

## 方法详解

### 整体框架

MoEE 基于 Stable Diffusion 的去噪 U-Net 架构，采用两阶段训练:
1. **阶段一**: 在全部情绪数据集上微调 Reference Net + Denoising U-Net，学习丰富的表情先验
2. **阶段二**: 固定空间/交叉/音频/时序模块，仅训练 Emotion MoE 模块和 Emotion-to-Latents 模块

输入: 一张肖像图 $\mathbf{I}$、音频序列 $\mathbf{A}$、情绪条件 $\mathbf{C}$（文本/标签/音频）。输出: 带有目标情绪的说话视频。

### 关键设计 1：Mixture of Emotion Experts

- **6 个专家**: 分别对应快乐、悲伤、愤怒、厌恶、恐惧、惊讶，每个专家是一个交叉注意力模块，在单一情绪数据上训练
- **Soft MoE 门控**: 与 Hard MoE（仅选一个专家）不同，采用软分配允许多专家同时处理输入
    - **局部分配**: 学习门控层 $s = \text{sigmoid}(G(X, \phi))$，$s \in \mathbb{R}^{n \times 6}$，每个 token 独立分配权重，实现面部局部表情的精细控制
    - **全局分配**: $g = \text{softmax}(G(\text{Pool}(X), \omega))$，$g \in \mathbb{R}^6$，6 个全局标量控制整体情绪基调
- **组合公式**: $X' = X + \sum_{i=1}^6 g_i \cdot E_i(X \cdot s_i)$

**单一情绪**: 仅激活对应专家 | **复合情绪**: 多专家软组合

### 关键设计 2：Emotion-to-Latents 模块

- **多模态输入对齐**: 文本（T5 编码器）、音频（emotion2vec）、标签（自训练 MLP）三种模态分别编码后通过 FC 层映射到相同维度
- **Learnable Embeddings**: 维护一组可学习嵌入作为 attention 的 key/value，将多模态特征转化为统一的 emotion latent
- **注入方式**: Emotion latent 作为 Emotion MoE Module 中交叉注意力的 key/value，注入 U-Net

### 关键设计 3：Masked Noisy Emotion Sampling

- **问题**: 单一情绪的子数据集较小，专家容易学到目标情绪之外的知识
- **方案**: 以一定概率将其他情绪/中性表情数据混入训练（增加噪声），扩大 person ID 多样性
- **嘴部遮挡**: 不同情绪来自不同说话视频，嘴型差异大会分散模型注意力；使用 MediaPipe 定位嘴部并 mask，使模型聚焦于表情变化而非嘴型

### 损失函数

$$L = L_{latent} + \lambda L_{spatial}$$

- $L_{latent}$: 标准扩散去噪损失 $\mathbb{E}_{t,c,z_t,\epsilon}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2]$
- $L_{spatial}$: 时间步感知的像素级损失，在解码后的图像空间计算 L1 + Perceptual Loss
    - $L_{spatial} = w(t)(||I_p, I_{GT}|| + ||V(I_p), V(I_{GT})||^2)$
    - $w(t) = \cos(t \cdot \pi / 2T)$，大时间步权重小（噪声多时像素级约束意义不大）

## 实验关键数据

### 主实验表

**HDTF 数据集（Table 2）**:

| 方法 | FID↓ | FVD↓ | LPIPS↓ | Sync-C↑ |
|---|---|---|---|---|
| AniPortrait | 36.83 | 476.82 | 0.211 | 5.977 |
| Hallo | 28.61 | 343.02 | 0.167 | 6.254 |
| EAT | 81.25 | 545.27 | 0.357 | 5.012 |
| **MoEE** | **28.83** | **322.63** | **0.152** | 6.114 |

**DH-FaceEmoVid-150 数据集（Table 4）**:

| 方法 | FID↓ | FVD↓ | LPIPS↓ | AKD↓ |
|---|---|---|---|---|
| AniPortrait | 66.03 | 712.29 | 0.323 | 20.654 |
| Hallo | 72.35 | 702.84 | 0.329 | 16.444 |
| EAT | 48.01 | 467.74 | 0.260 | 14.109 |
| **MoEE** | **39.62** | **402.80** | **0.182** | **4.028** |

### 消融表（Table 5）

| 变体 | FID↓ | FVD↓ | LPIPS↓ | AKD↓ |
|---|---|---|---|---|
| w/o MoEE | 58.41 | 655.33 | 0.325 | 14.959 |
| w/o Global Soft Assignment | 46.33 | 447.81 | 0.194 | 10.652 |
| w/o Masked Noisy Sampling | 51.79 | 489.24 | 0.211 | 4.591 |
| w/o DH-FaceEmoVid-150 | 52.41 | 511.93 | 0.275 | 7.885 |
| **Full MoEE** | **39.62** | **402.80** | **0.182** | **4.028** |

### 关键发现

1. **MoEE 模块贡献最大**: 去除后 FID 从 39.62 退化到 58.41，AKD 从 4.03 退化到 14.96
2. **全局软分配不可或缺**: 去除后 AKD 从 4.03 退化到 10.65，说明全局情绪基调控制对表情准确性至关重要
3. **数据集效果显著**: 仅去除 DH-FaceEmoVid-150，FID 从 39.62 退化到 52.41
4. Masked Noisy Sampling 对 FID/FVD 提升明显（51.79→39.62），同时保持 AKD 稳定
5. 情绪潜空间可视化显示，使用 MoEE 后不同情绪的分布更加分离

## 亮点与洞察

1. **类比调色板**: 将复合情绪建模为基础情绪的软组合，概念直觉且实施有效，与心理学中 Ekman 基本情绪理论高度一致
2. **局部 + 全局门控**: 局部分配控制面部局部（眉毛、嘴角等微表情），全局分配控制整体情绪基调，双层控制比单一门控更精细
3. **Masked Noisy Sampling 巧妙**: 解决小数据集专家训练的过拟合问题，同时用嘴部遮挡避免嘴型干扰
4. **DH-FaceEmoVid-150 数据集价值高**: 150 小时、1080p、包含复合情绪和 AU 标注，填补了领域数据空缺

## 局限性

1. 仅覆盖 6 种基础情绪和 4 种复合情绪，微妙情绪（讽刺、尴尬、无奈）等未涉及
2. 中文语音训练导致 Sync-C/Sync-D 评估可能不完全客观
3. AU 标注依赖 ME-GraphAU + GPT-4V 自动生成，可能存在噪声
4. 推理速度受扩散模型限制，难以实时应用

## 相关工作与启发

- **Hallo/AniPortrait/EchoMimic**: 音频驱动说话人头像的 SOTA 方法，但缺乏情绪控制
- **EAT/StyleTalk**: 情绪控制方法的代表，但表情不自然或需参考视频
- **Soft MoE**: 从 NLP 领域借鉴的架构，本文成功将其应用于情绪解耦
- **启发**: MoE 架构不仅适用于大模型的容量扩展，也可用于语义维度的解耦（情绪、风格等），这种"一个专家管一种语义"的思路值得推广到其他可控生成任务

## 评分

⭐⭐⭐⭐ — MoE 与情绪解耦的结合自然且有效，数据集贡献实质性弥补领域空缺，在情绪控制的自然度和准确度上大幅超越现有方法。扣 1 星因为情绪种类仍有限，且推理效率不支持实时应用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation](sonic_shifting_focus_to_global_audio_perception_in_portrait_animation.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](../../AAAI2026/human_understanding/spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)
- [\[CVPR 2026\] DeX-Portrait: Disentangled and Expressive Portrait Animation via Explicit and Latent Motion Representations](../../CVPR2026/human_understanding/dex-portrait_disentangled_and_expressive_portrait_animation_via_explicit_and_lat.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)

</div>

<!-- RELATED:END -->
