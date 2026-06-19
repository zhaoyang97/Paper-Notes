---
title: >-
  [论文解读] Long Context Tuning for Video Generation
description: >-
  [ICCV 2025][视频生成][场景级视频生成] 本文提出Long Context Tuning（LCT），将预训练单镜头视频扩散模型的上下文窗口扩展到场景级别，通过交错3D位置嵌入和异步噪声策略实现跨镜头视觉/时序一致性，无需额外参数即支持联合和自回归多镜头生成，并展现出组合生成等涌现能力。 1. 领域现状： 基于Di…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "场景级视频生成"
  - "多镜头一致性"
  - "长上下文微调"
  - "异步时间步"
  - "因果注意力"
---

# Long Context Tuning for Video Generation

**会议**: ICCV 2025  
**arXiv**: [2503.10589](https://arxiv.org/abs/2503.10589)  
**代码**: 无（项目页面可用）  
**领域**: 视频生成  
**关键词**: 场景级视频生成, 多镜头一致性, 长上下文微调, 异步时间步, 因果注意力

## 一句话总结
本文提出Long Context Tuning（LCT），将预训练单镜头视频扩散模型的上下文窗口扩展到场景级别，通过交错3D位置嵌入和异步噪声策略实现跨镜头视觉/时序一致性，无需额外参数即支持联合和自回归多镜头生成，并展现出组合生成等涌现能力。

## 研究背景与动机

1. **领域现状**: 基于DiT的视频生成模型（SoRA, Kling, HunyuanVideo等）已能合成持续一分钟的高质量单镜头视频。但真实叙事视频由多个镜头组成，需要跨镜头的一致性。
2. **现有痛点**: 现有场景级生成方案分为两类：(1) 外观条件生成（VideoStudio等），依赖预定义条件和特定数据集，难以维持光线色调等抽象元素；(2) 关键帧生成+I2V（StoryDiffusion等），各镜头独立合成无法保证时序一致性，稀疏关键帧限制条件化效果。
3. **核心矛盾**: 场景级一致性要求角色身份、背景、光线、色调等视觉一致性，以及动作、镜头运动等时序一致性。两类现有方案在一致性维度上各有缺陷。
4. **本文目标**: 如何从数据中直接学习跨镜头一致性，而不依赖预定义条件或辅助网络？
5. **切入角度**: 扩展预训练单镜头模型的上下文窗口，让全注意力机制覆盖场景内所有镜头的所有token，直接从场景级视频数据中学习跨镜头关联。
6. **核心 idea**: 通过交错3D RoPE位置嵌入区分镜头、异步时间步统一条件和扩散样本、上下文因果注意力支持高效自回归。

## 方法详解

### 整体框架
基于3B参数的MMDiT视频扩散模型，采用Rectified Flow训练。上下文窗口最大9个镜头。数据包含全局提示（角色/环境/故事）和逐镜头提示。同时在单镜头和场景级数据上联合训练以保留预训练能力。

### 关键设计

1. **交错3D位置嵌入（Interleaved 3D RoPE）**:
    - 功能: 区分不同镜头的token，保持镜头内部文本-视频对齐
    - 核心思路: 保持单镜头内文本token在视频token前的相对位置关系（沿空间对角线），多镜头时将各镜头的文本-视频组逐个追加，形成交错的"[text]-[video]-[text]-[video]-..."序列。全局提示添加虚拟视频token，作为普通文本-视频对处理。
    - 设计动机: 保持相对位置让每个镜头继承预训练的文本-视觉对齐能力；不同绝对位置区分token和对应镜头的关系。概念上类似M-RoPE（Qwen2-VL），但首次在扩散模型中使用。

2. **异步时间步策略（Asynchronous Timestep）**:
    - 功能: 统一视觉条件输入和扩散样本，为各镜头独立采样噪声水平
    - 核心思路: 训练时为每个镜头独立从logit-normal分布采样扩散时间步，而非对所有镜头使用统一时间步。当某镜头噪声较低时，自然成为外观信息源引导更嘈杂镜头的去噪。推理时可同步所有时间步做联合生成，或设部分镜头为低噪声作为视觉条件。
    - 设计动机: 不需要辅助网络做视觉条件化，一个模型同时支持联合生成、视觉条件生成和自回归生成三种模式，设计极其简洁。

3. **上下文因果注意力微调**:
    - 功能: 将双向注意力转换为高效的因果注意力，支持KV-cache自回归生成
    - 核心思路: 在LCT双向模型基础上微调：镜头内保持双向注意力，但token只attend前面所有镜头的上下文（因果掩码）。推理时历史镜头的K/V特征被缓存，避免重复计算。仅需9K迭代微调。
    - 设计动机: 自回归生成中信息流固有地是方向性的——干净历史样本不需要后续嘈杂样本的信息，因此双向注意力是冗余的。因果注意力+KV-cache显著减少计算开销。

### 损失函数 / 训练策略
Rectified Flow损失: $\mathcal{L} = \mathbb{E}_{t,z_0,\epsilon}\|v_\Theta(z_t, t, c_{text}) - (\epsilon - z_0)\|_2^2$。每个镜头独立计算损失后平均。在128张H800上训练135K迭代（LCT阶段），因果注意力微调9K迭代。训练分辨率480×480面积。

## 实验关键数据

### 主实验

| 方法 | Aesthetic↑ | Quality↑ | Consistency(avg.)↑ | Text↑ | 用户排名(AHR)↑ |
|--------|------|------|----------|------|------|
| VideoStudio | 61.68 | 73.13 | 95.25 | 28.00 | 2.14 |
| StoryDiffusion+Kling | 60.40 | 74.04 | 96.57 | 27.33 | 2.50 |
| IC-LoRA+Kling | 57.88 | 69.07 | 96.27 | 27.90 | 1.57 |
| **LCT (本文)** | 60.79 | 67.44 | 95.65 | **30.14** | **3.79** |

用户研究中LCT以平均排名3.79显著领先（满分4分）。

### 消融实验

| 配置 | 效果 | 说明 |
|------|---------|------|
| 双向注意力 | 联合+条件生成 | 全能但计算开销大 |
| 因果注意力 | 高效自回归 | KV-cache加速 |
| 无交错RoPE | 一致性下降 | 无法区分镜头归属 |
| 同步时间步 | 仅联合生成 | 失去条件化能力 |

### 关键发现
- 文本对齐分数（30.14）显著超越所有基线，说明LCT的跨镜头语义理解能力更强
- 涌现能力：组合生成（角色+环境图→视频）、交互式镜头扩展，模型从未显式训练这些任务
- "重现"问题：基线方法在角色间隔多镜头后重新出现时一致性崩塌，LCT通过历史池策略避免
- 基线方法构图多样性差，LCT能生成远景/中景/近景丰富组合

## 亮点与洞察
- 极其简洁优雅的设计：无额外参数、无辅助网络，仅通过位置嵌入+时间步策略+注意力模式实现多模式生成
- 异步时间步是核心创新：一个机制统一了联合生成/条件生成/自回归生成三种推理模式
- 涌现能力令人印象深刻：组合生成（从未训练过）+ 交互式扩展展示了场景级理解的泛化性
- 人类选择式生成策略：不依赖严格的自回归，而是从历史池中按相关性选取条件镜头

## 局限与展望
- 训练分辨率480×480相对较低
- 上下文窗口限制9镜头，更长叙事可能需要分段处理
- Video Quality和Aesthetic指标略低于某些基线，可能因训练数据差异
- 因果注意力微调仅9K迭代，完整训练可能进一步提升

## 相关工作与启发
- **vs VideoStudio**: 后者用实体嵌入保持外观但构图单一，LCT通过全注意力学习更丰富的一致性
- **vs MoviDreamer/VGoT**: 关键帧方法受I2V独立生成限制，无法保证时序一致性
- **vs MinT/DFoT**: 长视频生成通过时间依赖提示或历史引导，但不处理多镜头结构

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 异步时间步策略极其优雅地统一了多种生成范式
- 实验充分度: ⭐⭐⭐⭐ 定性结果出色，自动指标和用户研究全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，用Titanic例子生动阐述场景概念
- 价值: ⭐⭐⭐⭐⭐ 从单镜头到场景级生成的范式转变，对视频内容创作意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Long-Context State-Space Video World Models](long-context_state-space_video_world_models.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](../../CVPR2026/video_generation/geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[CVPR 2025\] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation](../../CVPR2025/video_generation/moviebench_a_hierarchical_movie_level_dataset_for_long_video_generation.md)
- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](../../CVPR2025/video_generation/longdiff_training-free_long_video_generation_in_one_go.md)

</div>

<!-- RELATED:END -->
