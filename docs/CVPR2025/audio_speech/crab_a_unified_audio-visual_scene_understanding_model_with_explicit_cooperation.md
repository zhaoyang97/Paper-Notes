---
title: >-
  [论文解读] Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation
description: >-
  [CVPR 2025][音频/语音][audio-visual understanding] 提出统一音视频场景理解模型 Crab，通过构建带显式推理过程的 AV-UIE 数据集（200K 样本）阐明跨任务协作关系，结合交互感知 LoRA（多头 LoRA）学习不同音视频交互模式，在多个任务上超越专用模型。
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "audio-visual understanding"
  - "统一模型"
  - "interaction-aware LoRA"
  - "指令微调"
  - "多任务学习"
---

# Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation

**会议**: CVPR 2025  
**arXiv**: [2503.13068](https://arxiv.org/abs/2503.13068)  
**代码**: [GeWu-Lab/Crab](https://github.com/GeWu-Lab/Crab)  
**机构**: 人民大学 / 清华大学 / 腾讯 PCG
**领域**: 音频视觉理解 / 多模态学习  
**关键词**: audio-visual understanding, 统一模型, interaction-aware LoRA, 指令微调, 多任务学习

## 一句话总结
提出统一音视频场景理解模型 Crab，通过构建带显式推理过程的 AV-UIE 数据集（200K 样本）阐明跨任务协作关系，结合交互感知 LoRA（多头 LoRA）学习不同音视频交互模式，在多个任务上超越专用模型。

## 研究背景与动机

**领域现状**：音视频场景理解包含多类任务——时序定位（AVE、AVVP）、时空推理（AVQA）、空间定位（ARIG）、像素级理解（AVS、Ref-AVS）。人类具有统一的多任务理解能力，但现有研究大多针对单一任务设计专用模型。

**现有痛点**：
   - **简单联合训练**：多任务间因音视频数据异构性和任务间复杂关系产生干扰
   - **现有统一模型**（VideoLLaMA、GroundingGPT 等）：缺乏任务间的显式协作机制，性能有限
   - 现有数据集标签简单（单词级），无法体现任务间的推理协作关系

**核心矛盾**：如何在一个模型中同时处理时序/空间/像素级多粒度任务，且避免任务间干扰。

**切入角度**：从数据和模型两方面实现显式任务间协作。

**核心 idea**：显式推理数据集（AV-UIE） + 交互感知 LoRA（多头） = 统一音视频理解。

## 方法详解

### 整体架构
- **视觉编码器**：CLIP-ViT-L/14，提取 patch 级特征
- **音频编码器**：BEATs，提取声学特征
- **分割解码器**：SAM decoder
- **语言模型**：LLaMA-2-7b-Chat
- **多模态桥接**：Audio Q-Former + Visual Q-Former（各 32 个查询 token）

### 关键设计

1. **AV-UIE 数据集（Audio-Visual Unified Instruction-tuning with Explicit reasoning）**

    - 功能：构建 200K 样本的统一指令微调数据集，包含显式推理过程
    - 核心思路：将现有数据集的简单标签扩展为包含推理链的指令格式
    - 任务覆盖：时序定位、时空推理、空间定位、像素级分割、参考分割
    - 效果：阐明任务间协作关系，如"时序定位帮助空间定位"

2. **交互感知 LoRA（Interaction-aware LoRA）**

    - 功能：在 LLM 所有线性层中插入多头 LoRA，学习不同音视频交互模式
    - 结构：共享 $\mathbf{A}$ 矩阵 + $n=3$ 个 LoRA 头（独立 $\mathbf{B}$ 矩阵）
    - 三个头分别关注：时序交互 / 空间交互 / 像素级交互
    - rank = 8
    - 设计动机：不同任务需要关注音视频数据的不同交互维度
    - 输出：三个头的加权和作为最终适配

3. **掩码解码器设计**

    - 两组 <MASK> token 对应两个尺度的视觉特征（第 14 层和倒数第 2 层）
    - 每组 3 个 token
    - 支持语义分割（AVSS）和参考分割（Ref-AVS）

### 训练策略
- **阶段一：预训练对齐**
    - 视觉分支：Video-LLaVA 数据
    - 音频分支：AudioCaps 数据
    - 分割分支：LVIS 数据
    - 全局 batch size 256，3 epochs
- **阶段二：指令微调**
    - AV-UIE 数据集，所有任务混合
    - 可训练：三个多模态分支 + 交互感知 LoRA（冻结编码器）
    - 全局 batch size 512，5 epochs

损失函数：$\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice} + \lambda_{ce}\mathcal{L}_{ce}$

## 实验关键数据

### 与专用模型的全面对比

| 任务 | 指标 | 之前 SOTA | Crab |
|------|------|----------|------|
| AVE 时序定位 | Acc | MM-Pyramid 77.80 | **80.15** |
| AVQA 时空推理 | Avg | TSPM 76.79 | **78.94** |
| ARIG 空间定位 | cIoU | FNAC 27.15 | **41.78** |
| ARIG 空间定位 | AUC | FNAC 0.31 | **0.42** |
| AVS-MS3 像素分割 | mIoU | AVSegFormer 58.40 | **58.21** |

### AVQA 子类别对比

| 方法 | Audio | Visual | Audio-Visual | Avg |
|------|-------|--------|-------------|-----|
| LAVISH | 75.97 | 80.22 | 71.26 | 74.46 |
| TSPM | 76.91 | 83.61 | 73.51 | 76.79 |
| **Crab** | 76.58 | **90.73** | **74.13** | **78.94** |

Visual 子类别上提升显著（90.73 vs. 83.61），可能受益于显式推理的视觉理解增强。

### 关键发现
- 各 LoRA 头自动学到不同的音视频理解能力（可视化验证）
- 时序定位任务在 AV-UIE 中占比最小，但因跨任务协作仍获得显著提升
- 与 VALOR（在 VALOR-1M 百万级数据上训练）性能相当（78.94 vs. 78.90），但用更少数据

## 亮点与洞察
- **多头 LoRA** 设计简洁有效：共享 A 矩阵降低参数量，多头 B 矩阵捕获不同交互模式
- **显式推理数据**比简单标签有效得多，使模型理解"为什么不同任务需要协作"
- 在空间定位（ARIG）上大幅超越专用方法（+14.63 cIoU），体现统一模型的跨任务迁移优势
- 统一模型思路比堆砌专用模型更优雅且高效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)
- [\[CVPR 2025\] Towards Open-Vocabulary Audio-Visual Event Localization](towards_open-vocabulary_audio-visual_event_localization.md)
- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](../../ICLR2026/audio_speech/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)
- [\[CVPR 2025\] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)

</div>

<!-- RELATED:END -->
