---
title: >-
  [论文解读] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
description: >-
  [CVPR 2025][视频生成][多镜头视频生成] ShotAdapter 提出了一个轻量框架，通过引入可学习的"转场token"和局部注意力掩码策略，仅需约 5000 次迭代的微调即可将预训练的单镜头 T2V 模型转变为支持多镜头视频生成（T2MSV）的生成器，实现角色身份一致、各镜头独立可控的多镜头视频生成。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "多镜头视频生成"
  - "Transformer"
  - "转场控制"
  - "身份一致性"
  - "注意力掩码"
---

# ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2505.07652](https://arxiv.org/abs/2505.07652)  
**代码**: [https://shotadapter.github.io/](https://shotadapter.github.io/)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 多镜头视频生成, 扩散Transformer, 转场控制, 身份一致性, 注意力掩码

## 一句话总结

ShotAdapter 提出了一个轻量框架，通过引入可学习的"转场token"和局部注意力掩码策略，仅需约 5000 次迭代的微调即可将预训练的单镜头 T2V 模型转变为支持多镜头视频生成（T2MSV）的生成器，实现角色身份一致、各镜头独立可控的多镜头视频生成。

## 研究背景与动机

**领域现状**：当前扩散模型（如 OpenSora、MovieGen、Kling AI 等）在文本到视频生成方面取得了显著进展，但所有现有模型都只能生成单个连续镜头的短视频，缺乏在不同镜头间切换的能力。

**现有痛点**：实际应用（如电影制作）需要多镜头视频——同一角色在不同场景、不同活动之间切换。现有的拼凑方案都存在严重问题：(1) 将所有描述合并为单个提示无法创建"切镜"且无法展示不同活动；(2) 分别生成每个镜头再拼接会导致角色身份不一致；(3) 先用参考图像生成一致的关键帧再用 I2V 模型动画化，受限于现有工具的能力，身份和背景一致性仍然不足。

**核心矛盾**：多镜头视频既需要跨镜头的角色/背景一致性（要求全局信息交互），又需要各镜头内容独立可控（要求局部化控制），这两个目标天然矛盾。

**本文目标**：设计一个模型无关的框架，用最少的微调代价使 T2V 模型支持多镜头生成，同时允许用户控制镜头数量、时长和内容。

**切入角度**：作者观察到在 NLP 中 [EOS] token 被成功用于标记句子边界，由此类推，可以用一个可学习的"转场 token"来标记视频中镜头之间的切换点。结合局部注意力掩码限制每个文本提示只影响对应镜头的视觉 token，就能实现局部化控制。

**核心 idea**：引入可学习的转场 token 标记镜头边界，配合局部注意力掩码实现分镜头独立文本控制，在单一完整视频中生成多镜头内容，保持全局注意力以维持身份一致性。

## 方法详解

### 整体框架

ShotAdapter 基于 DiT 架构的 T2V 模型（类似 OpenSora），整个流程如下：输入是一组分镜头的文本提示词和期望的镜头数/时长配置。3D-VAE 将视频编码为潜表示后 patchify 为视觉 token 序列，各镜头的文本条件 token 按顺序拼接，最后附加 $n-1$ 个转场 token（$n$ 为镜头数）。整个序列送入 DiT 处理，通过局部注意力掩码控制不同 token 间的交互方式。模型输出经 3D-VAE 解码后得到包含多个镜头的完整视频。

### 关键设计

1. **转场 Token (Transition Token)**:

    - 功能：标记视频中镜头切换的位置，使模型学会在指定帧处生成"剪切"效果
    - 核心思路：初始化一组与隐藏维度 $D$ 匹配的可学习参数，对于 $n$ 镜头视频重复 $n-1$ 次并附加到输入序列末尾。在注意力层中，转场 token 只与发生转场的帧的视觉 token 交互。这样模型能学会在这些帧位置生成突变式的场景切换
    - 设计动机：类比 NLP 中的特殊 token 机制，用极少的可学习参数（仅一个 embedding 向量）就能编码"镜头切换"这个语义概念。实验证明模型能泛化到训练时未见过的镜头数（2-8 镜头），平均帧误差仅 1-2 帧

2. **局部注意力掩码 (Local Attention Masking)**:

    - 功能：实现分镜头独立的文本控制，使每个文本提示只影响对应镜头的视觉生成
    - 核心思路：构建一个结构化的注意力掩码矩阵，约束三种交互方式：(a) 转场 token 只关注转场帧的视觉 token；(b) 每个文本 token 只关注其对应镜头的视觉 token；(c) 视觉和文本 token 自身保持自注意力。这样不同镜头的文本条件可以分别施加影响
    - 设计动机：没有掩码时，所有 token 相互交互会稀释分镜头信息的影响力。局部掩码既保留了全局视觉自注意力（确保角色一致性），又限制了文本条件的作用范围

3. **多镜头视频数据集构建流水线**:

    - 功能：从现有单镜头视频数据集中构建多镜头训练数据
    - 核心思路：提出两种方法——(a) 从高动态单镜头视频中随机裁切子片段并拼接（保持背景一致但动作/视角不同）；(b) 将同一身份的多个独立视频聚类后随机组合拼接（引入不同背景的多样性）。后处理包括：用 LLaVA-NeXT 生成分镜头描述、用 YOLO 检测人数、用 DINOv2 验证身份一致性，过滤掉 38% 不合格样本
    - 设计动机：缺乏现成的多镜头训练数据是该任务的核心瓶颈。通过这两种互补策略，可以从单镜头数据中自动构造质量较高的多镜头数据，无需人工标注

### 损失函数 / 训练策略

微调采用标准扩散训练损失，冻结预训练模型大部分参数，只更新转场 token 的可学习参数以及注意力层中与掩码相关的组件。仅需约 5000 次迭代（不到预训练迭代数的 1%），使用 90% 更小的 batch size。训练数据包含 2、3、4 镜头的多镜头视频。

## 实验关键数据

### 主实验

| 方法 | 2-shot IC↑ | 3-shot IC↑ | 4-shot IC↑ | 2-shot BC↑ | TA (2-shot)↑ |
|------|-----------|-----------|-----------|-----------|-------------|
| Random Shots | 71.03/80.47 | 54.76/63.72 | 48.08/55.87 | 84.46 | 26.84 |
| Similar Shots | 73.94/82.55 | 55.15/66.17 | 49.25/58.67 | 88.85 | 26.40 |
| Shots by Ref. | 81.74/84.98 | 67.92/72.97 | 57.83/67.74 | 82.11 | 25.59 |
| **ShotAdapter** | **78.67/86.33** | **70.30/76.44** | **61.86/74.89** | **89.48** | **27.12** |

（IC = Identity Consistency，diff bg/same bg；BC = Background Consistency；TA = Text Alignment）

### 消融实验

| 配置 | 2-shot IC↑ | 3-shot IC↑ | 4-shot IC↑ | 2-shot BC↑ |
|------|-----------|-----------|-----------|-----------|
| ShotAdapter (full) | 78.67/86.33 | 70.30/76.44 | 61.86/74.89 | 89.48 |
| w/o Transition Token | 77.17/84.78 | 68.95/70.98 | 58.83/70.24 | 87.94 |
| 仅 2-shot 数据训练 | 78.05/85.46 | 70.12/71.53 | 56.99/68.37 | 89.08 |

### 关键发现

- ShotAdapter 在 3-shot 和 4-shot 上的身份一致性明显优于所有基线，但在 2-shot 上与 Shots by Reference 基线竞争力相当
- 转场 token 的加入对背景一致性和文本对齐度有明显贡献，说明它确实帮助模型学会了"切镜"
- 仅用 2-shot 数据训练的模型在 3-shot 和 4-shot 上仍有不错表现，说明转场 token 的泛化能力强
- 转场 token 可以泛化到 2-8 镜头，平均帧误差（MSDE）仅 0.83-2.00 帧
- 用户研究中 ShotAdapter 在身份一致性和背景一致性上以约 73-82% 的选择率胜出

## 亮点与洞察

- **转场 token 的设计极其简洁优雅**——只需一个可学习的 embedding 向量，通过重复 $n-1$ 次并配合注意力掩码就能控制任意数量的镜头切换，可迁移到其他需要"结构化分段"的生成任务
- **数据构建流水线解决了"无数据"难题**——将单镜头→多镜头的数据转换系统化，利用聚类和自动过滤保证质量，是一个可复用的方法论
- **局部注意力掩码的设计平衡了全局一致性和局部可控性**——视觉 token 间保持全局注意力（身份一致），文本 token 被限制在对应镜头（局部控制），这种思路可推广到其他多条件控制场景

## 局限与展望

- 仅验证了以人类为主角的场景，动物等其他主体未测试，主要受限于数据过滤策略
- 最大生成时长受底层模型限制（当前为 128 帧），作者提出可用自回归方式扩展
- 微调导致轻微的视觉质量下降（用户研究中可察觉），可能与更小 batch size 有关
- 分辨率限制在 192×320，距离实际影视制作需求还有较大差距

## 相关工作与启发

- **vs Single-Shot T2V**: 单镜头模型无法生成"切镜"效果，将多个描述合并为一个提示会导致动作混杂无法区分
- **vs StoryMaker + I2V**: 先生成一致关键帧再逐帧动画化的流水线受限于 off-the-shelf 工具质量，特别在多镜头时身份退化严重
- **vs 图像故事生成 (ConsiStory/StoryDiffusion)**: 这些方法关注图像序列的一致性但缺乏视频的时间连续性和"切镜"概念

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义 T2MSV 任务并给出端到端解决方案，转场 token 设计新颖
- 实验充分度: ⭐⭐⭐⭐ 设计了完整的评估流水线和多个基线，但缺少与更多现有方法的对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法表述直观，但数据构建部分可以更精炼
- 价值: ⭐⭐⭐⭐ 开辟了多镜头视频生成的新方向，但当前效果距实用仍有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[CVPR 2026\] MultiShotMaster: A Controllable Multi-Shot Video Generation Framework](../../CVPR2026/video_generation/multishotmaster_a_controllable_multi-shot_video_generation_framework.md)
- [\[CVPR 2025\] Multi-subject Open-set Personalization in Video Generation](multi-subject_open-set_personalization_in_video_generation.md)
- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)

</div>

<!-- RELATED:END -->
