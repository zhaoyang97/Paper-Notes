---
title: >-
  [论文解读] Generative Inbetweening through Frame-wise Conditions-Driven Video Generation
description: >-
  [CVPR 2025][视频生成][视频插帧] 提出 FCVG，通过从两个关键帧中提取匹配线段并逐帧线性插值作为帧级条件，注入 SVD 视频生成模型，显著消解了生成式中间帧合成中前向/反向路径的模糊性，实现时序稳定的视频插帧。 生成式中间帧合成（Generative Inbetweening）旨在给定起始帧和终止帧…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频插帧"
  - "生成式中间帧"
  - "扩散模型"
  - "帧级条件控制"
  - "时序稳定性"
---

# Generative Inbetweening through Frame-wise Conditions-Driven Video Generation

**会议**: CVPR 2025  
**arXiv**: [2412.11755](https://arxiv.org/abs/2412.11755)  
**代码**: [https://fcvg-inbetween.github.io/](https://fcvg-inbetween.github.io/)  
**领域**: 视频生成  
**关键词**: 视频插帧, 生成式中间帧, 扩散模型, 帧级条件控制, 时序稳定性

## 一句话总结

提出 FCVG，通过从两个关键帧中提取匹配线段并逐帧线性插值作为帧级条件，注入 SVD 视频生成模型，显著消解了生成式中间帧合成中前向/反向路径的模糊性，实现时序稳定的视频插帧。

## 研究背景与动机

生成式中间帧合成（Generative Inbetweening）旨在给定起始帧和终止帧，生成中间帧序列。与传统光流插帧不同，生成式方法能处理大运动场景，但面临**插值路径模糊性**这一核心挑战：

1. **Time Reversal 策略的局限**：TRF 等方法分别以起始帧和终止帧为条件做前向/反向去噪，然后融合。但 I2V 模型生成的运动具有多样性和随机性，导致两条路径严重不对齐，融合后产生不连贯甚至完全不同的中间内容
2. **后续改进仍不够**：GI 微调时序注意力层、VIBIDSampler 改进融合策略，都试图对齐两条路径，但在大运动（如人体快速动作）场景下仍存在明显抖动和不连贯
3. **现有方案的额外成本**：噪声重注入（noise re-injection）等缓解策略大幅增加推理时间（1.5-3×），且需要对每对输入图像手动调参

核心观点：**路径模糊性的根源是中间帧缺乏显式条件引导**。只有起始和终止帧有条件，中间帧的运动完全依赖模型的随机采样。如果为每一帧都提供显式条件，前向和反向路径自然会对齐。

## 方法详解

### 整体框架

FCVG 基于 SVD（Stable Video Diffusion），采用 Time Reversal 融合策略。核心创新是引入**帧级条件**：从两个输入帧提取匹配线段，逐帧线性插值，通过 ControlNeXt 轻量模块注入 SVD，且仅需微调极少量参数。

### 关键设计

1. **帧级条件的构建 (Frame-wise Conditions)**:
    - 功能：为每一帧提供显式的运动引导，消解插值路径的模糊性
    - 核心思路：使用预训练的 GlueStick 线匹配模型建立起始帧和终止帧之间的线段对应关系，将匹配结果可视化为彩色线段图（同色表示对应匹配）。对人体场景额外使用 DWPose 提取姿态骨架。然后对起始条件 $\mathbf{c}_1$ 和终止条件 $\mathbf{c}_N$ 进行逐帧线性插值得到 $\mathbf{c}_{1 \rightarrow N}$
    - 设计动机：线段匹配天然具有全局鲁棒性，能处理大运动和复杂场景；线段图是稀疏的结构化表征，易于线性插值；匹配的线段天然编码了运动方向和幅度。线性假设虽不完全精确，但在先验视频插帧工作中已被广泛验证为足够保证时序稳定性

2. **条件注入机制 (Condition Injection)**:
    - 功能：将帧级条件融入 SVD 而不破坏其预训练知识
    - 核心思路：采用 ControlNeXt 的轻量方案——用多个 ResNet 块编码条件，通过 cross normalization 对齐条件分支和 SVD 分支的特征分布，然后以可调权重 $\gamma$ 相加融合：$\hat{\mathbf{y}}_t = \mathbf{y}_t^{\text{SVD}} + \gamma \mathbf{y}_t^{\text{Con}}$
    - 设计动机：ControlNeXt 比 ControlNet 更轻量，不显著增加推理时间。仅需微调 SVD 中注意力层的 value/output 投影矩阵和轻量 ResNet 块，保持大部分参数冻结

3. **Time Reversal 融合策略**:
    - 功能：基于双向去噪融合生成中间帧
    - 核心思路：前向路径以 $I_{\text{start}}$ 为条件 + $\mathbf{c}_{1 \rightarrow N}$ 为帧级条件，反向路径以 $I_{\text{end}}$ 为条件 + $\mathbf{c}_{N \rightarrow 1}$（时间翻转）为帧级条件，每步用线性权重 $\lambda_i = 1 - \frac{i-1}{N-1}$ 融合
    - 设计动机：有了帧级条件后，两条路径已大致对齐，**简单的线性融合**就足够。不需要 noise re-injection，推理步数从 50 减少到 25，速度提升约 2×

### 损失函数 / 训练策略

- 使用 SVD 原有的 v-prediction 目标：$\mathcal{L} = \mathbb{E}[\|\mathbf{v} - f_\theta(\mathbf{z}_t, \mathbf{c}_{\text{image}}, t)\|_2^2]$
- 仅微调注意力层的 V/O 投影 + 轻量 ResNet 编码块
- AdamW 优化器，学习率 $1 \times 10^{-6}$，70K 迭代
- 训练分辨率 512×320，推理分辨率 1024×576
- 训练数据：524 个 25 帧视频片段（DAVIS + RealEstate10K + Pexels），4:1 划分

## 实验关键数据

### 主实验（Frame Gap=23）

| 方法 | LPIPS ↓ | FID ↓ | VBench ↑ | FVMD ↓ | FVD ↓ |
|------|---------|-------|---------|--------|-------|
| FCVG (Ours) | **0.1832** | **24.05** | **0.8619** | **5607.2** | **437.9** |
| GI | 0.2155 | 31.39 | 0.8606 | 5682.6 | 524.0 |
| TRF | 0.3687 | 42.76 | 0.8438 | 10458.0 | 823.4 |
| FILM (光流) | 0.1540 | 25.00 | 0.8615 | 8208.7 | 543.4 |

### 消融实验

| 配置 | LPIPS ↓ | FID ↓ | FVMD ↓ | FVD ↓ |
|------|---------|-------|--------|-------|
| Full Model | **0.1832** | **24.05** | **5607.2** | **437.9** |
| w/o Control | 0.2485 | 27.55 | 7217.5 | 536.5 |
| w/o Matching | 0.2124 | 24.17 | 6546.8 | 498.8 |
| w/o Pose | 0.1843 | 24.70 | 5520.9 | 446.1 |

### 推理效率对比

| 方法 | 分辨率 | 推理步数 | 时间 (s) |
|------|--------|---------|---------|
| FCVG (Ours) | 25×(1024,576) | 25 | **523** |
| GI | 25×(1024,576) | 50 | 975 |
| TRF | 25×(1024,576) | 50 | 1230 |

### 关键发现

1. **帧级条件是核心**：去掉所有控制条件后 FVMD 从 5607→7218（+28.7%），证明帧级条件对时序稳定性的决定性作用
2. **线匹配比姿态更重要**：去掉匹配线段后 FVMD 劣化更严重（6546 vs 5520），匹配线段控制全局场景运动，姿态仅改善人体细节
3. **控制权重 $\gamma$ 不敏感**：$\gamma \in [0.5, 2.0]$ 范围内模型表现稳定，默认 $\gamma=1$ 即可适用大部分场景
4. 模型可**零样本泛化**到动画和线稿视频（训练集中未出现这些数据类型）

## 亮点与洞察

- **核心思想极其简洁**：仅仅是为每帧提供一个显式条件，就从根本上解决了路径模糊性。匹配线段作为稀疏但结构化的中间表征，是一个非常优雅的选择
- **非线性插值路径**：虽然训练时使用线性插值，但推理时支持 ease-in/ease-out 等非线性运动曲线，为用户提供创作灵活性
- **实际加速**：消除了 noise re-injection 步骤，推理步数减半，实际推理时间比 GI/TRF 快约 2×

## 局限与展望

- **依赖线匹配质量**：当两帧特征高度相似时可能出现错误匹配（可通过降低 $\gamma$ 缓解）；当两帧差异过大时匹配线段稀疏，简单调控制权重也不够
- **线性假设的局限**：虽然对大多数场景有效，但对非匀速运动（加速、减速、弹性运动）仍是近似
- **推理仍然昂贵**：523 秒生成 25 帧仍距实时应用有较大差距，瓶颈在预训练 SVD 本身
- 后续可探索：结合拖拽编辑（DragDiffusion）或文本引导来生成更丰富的控制条件

## 相关工作与启发

- **与光流插帧（FILM）的互补**：FILM 在小运动场景更优（LPIPS 更低），但大运动场景有严重伪影；FCVG 在大运动场景更稳定
- **与 GI/TRF/VIBIDSampler 的关系**：都基于 Time Reversal 策略，FCVG 的核心改进是引入帧级条件来对齐双向路径
- **启示**：在扩散模型的控制问题中，"为每个生成单元提供显式条件"是一个通用有效的策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 帧级条件的想法简单但抓住了问题本质，匹配线段插值是巧妙的设计
- 实验充分度: ⭐⭐⭐⭐ 多场景对比、消融、泛化实验覆盖较全，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，method 部分图示到位
- 价值: ⭐⭐⭐⭐ 对生成式视频插帧的时序稳定性问题提出了实用有效的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)
- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](../../CVPR2026/video_generation/a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)

</div>

<!-- RELATED:END -->
