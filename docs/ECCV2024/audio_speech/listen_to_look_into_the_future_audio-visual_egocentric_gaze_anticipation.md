---
title: >-
  [论文解读] Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation
description: >-
  [ECCV 2024][音频/语音][第一人称注视预测] 提出 CSTS（Contrastive Spatial-Temporal Separable）音视频融合方法，首次将音频信号引入第一人称注视预测任务，通过空间和时间分离融合模块分别建模音视频的空间共现和时序相关性，并用后融合对比学习增强表示，在 Ego4D 和 Aria 数据集上超越 SOTA。
tags:
  - "ECCV 2024"
  - "音频/语音"
  - "第一人称注视预测"
  - "音视频融合"
  - "对比学习"
  - "时空分离融合"
  - "增强现实"
---

# Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation

**会议**: ECCV 2024  
**arXiv**: [2305.03907](https://arxiv.org/abs/2305.03907)  
**代码**: [https://github.com/bolinlai/CSTS-EgoGazeAnticipation](https://github.com/bolinlai/CSTS-EgoGazeAnticipation)  
**领域**: 音频语音 / 第一人称视觉  
**关键词**: 第一人称注视预测, 音视频融合, 对比学习, 时空分离融合, 增强现实

## 一句话总结

提出 CSTS（Contrastive Spatial-Temporal Separable）音视频融合方法，首次将音频信号引入第一人称注视预测任务，通过空间和时间分离融合模块分别建模音视频的空间共现和时序相关性，并用后融合对比学习增强表示，在 Ego4D 和 Aria 数据集上超越 SOTA。

## 研究背景与动机

**领域现状**: 第一人称注视预测（Egocentric Gaze Anticipation）旨在从第一人称视频中预测佩戴者未来的注视目标，是增强现实和可穿戴计算的关键构建模块。现有方法主要关注注视估计（当前帧的注视位置），注视预测（未来帧）研究较少。

**现有痛点**: 所有已有方法仅利用视觉模态，完全忽略了音频信号对注视行为的影响。然而在日常活动中（尤其是社交场景），声音是驱动注视转移的重要刺激——例如听到某人说话后注视会转向说话者。

**核心矛盾**: 第一人称视频中音频和视觉的时空关系与第三人称视频截然不同：(1) 佩戴者对音频刺激的反应导致镜头剧烈移动（头部转动改变视角），空间上音源位置随时间变化；(2) 音频刺激与注视反应之间存在自然时间延迟，不同步。现有的联合融合方法（同时在时空维度融合）无法有效处理这些特性。

**本文目标**: 如何在第一人称场景下有效融合音频和视觉信号，以预测佩戴者未来的注视目标。

**切入角度**: 将音视频的相关性分解为空间维度（音源在哪里）和时间维度（音频如何驱动视角变化和注视移动），用两个独立模块分别建模。

**核心 idea**: 设计空间和时间分离的音视频融合模块，分别捕获帧内音视频空间共现和跨帧时序关联，配合后融合对比学习，实现第一人称注视预测。

## 方法详解

### 整体框架

模型包含以下组件：
1. **视觉编码器** $\phi(x)$：MViT（Multi-scale Vision Transformer）提取视频特征，输出 $T \times H \times W$ 个 $D$ 维 token
2. **音频编码器** $\psi(a)$：轻量化 Transformer 处理音频频谱图，输出 $T \times M$ 个 $D$ 维 token
3. **空间融合模块**：建模每一帧内音频与视觉 token 的空间共现关系
4. **时间融合模块**：建模跨帧的音视频时序关联
5. **通道重加权合并**：将两个融合分支的输出合并
6. **后融合对比学习**：在融合后的表示上施加对比损失
7. **解码器**：预测未来注视的概率分布热图

输入为 3 秒观测段的 8 帧视频 + 对应音频，预测未来 2 秒的注视位置分布。

### 关键设计

1. **空间融合模块 (Spatial Fusion)**:

    - **功能**: 建模每一帧内音频信号与视觉区域的空间共现关系，识别哪些视觉区域与当前声音最相关（如声音源位置）。
    - **核心思路**: 对每个时间步 $i$，将音频 embedding 通过卷积池化为单个 token $z_{a,s}^{(i)} \in \mathbb{R}^{1 \times 1 \times D}$，与该帧的 $N = H \times W$ 个视觉 token 拼接为 $z_s^{(i)} \in \mathbb{R}^{1 \times (N+1) \times D}$，送入帧内自注意力层：
    $\sigma(z_s^{(i)}) = \text{Softmax}\left(\frac{Q_s^{(i)} K_s^{(i)T}}{\sqrt{D}}\right) V_s^{(i)}$
   关键是使用 **帧内 mask**，完全屏蔽跨帧连接，每帧独立计算。
    - **设计动机**: 在第一人称视频中，视角随头部运动剧烈变化，同一物体的空间位置在不同帧间变化很大。帧内独立融合避免了跨帧空间位置的混淆。

2. **时间融合模块 (Temporal Fusion)**:

    - **功能**: 建模音频和视觉信号在时间维度上的关联，捕获音频如何驱动视角变化和注视移动。
    - **核心思路**: 对每个模态每个时间步的所有 token 用卷积池化为单个 token（视觉 $z_{v,t} \in \mathbb{R}^{T \times 1 \times D}$, 音频 $z_{a,t} \in \mathbb{R}^{T \times 1 \times D}$），拼接为 $z_t \in \mathbb{R}^{2T \times 1 \times D}$，送入跨帧自注意力层：
    $\pi(z_t) = \text{Softmax}\left(\frac{Q_t K_t^T}{\sqrt{D}}\right) V_t$
   所有 $2T$ 个 token 之间允许自由交互。
    - **设计动机**: 音频刺激与注视反应之间存在时间延迟（如听到声音后经过一段反应时间才转移注视），跨帧注意力可以捕捉这种延迟效应和时序动态。

3. **通道重加权合并 + 后融合对比学习 (Post-Fusion Contrastive Learning)**:

    - **功能**: 将空间和时间融合的结果合并，并用对比损失增强音视频表示的对齐。
    - **核心思路**:
        - **合并**: 用时间融合的权重对空间融合的输出做通道重加权：$u_v = u_{v,s} \otimes u_{v,t}$（逐元素乘法 + 广播）。
        - **对比学习**: 不同于传统方法在原始 embedding 上计算对比损失，本文在融合后的重加权特征上计算：将 $u_v$ 和 $u_a$（重加权后的音频表示）分别全局平均池化并映射到低维空间，用 InfoNCE 对比损失：
    $\mathcal{L}_{cntr}^{v2a} = -\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\log\frac{\exp(w_v^{(i)T} w_a^{(i)} / \mathcal{T})}{\sum_{j \in \mathcal{B}}\exp(w_v^{(i)T} w_a^{(j)} / \mathcal{T})}$
    - **设计动机**: 在融合后特征上计算对比损失比在原始特征上更有效，因为融合后的特征已经包含了时空关联信息，对比学习可以进一步强化这种关联。

### 损失函数 / 训练策略

最终损失为两项的线性组合：
$$\mathcal{L} = \mathcal{L}_{kld} + \alpha \mathcal{L}_{cntr}$$

- $\mathcal{L}_{kld}$: KL 散度损失，监督注视热图的预测分布与真实分布的匹配
- $\mathcal{L}_{cntr}$: 双向对比损失（video-to-audio + audio-to-video）
- 视觉编码器使用 MViT（在 Kinetics-400 预训练），音频编码器为轻量级 Transformer
- 输入视频分辨率 $256 \times 256$，音频采样率 24kHz，STFT 窗口 10ms、hop 5ms，频谱图大小 $256 \times 256$
- 解码器有从视频编码器到解码器的跳跃连接

## 实验关键数据

### 主实验

| 方法 | Ego4D F1 | Ego4D Recall | Ego4D Precision | Aria F1 | Aria Recall | Aria Precision |
|------|---------|-------------|----------------|---------|------------|---------------|
| MViT (Vision only) | 37.2 | 54.1 | 28.3 | 57.5 | 62.4 | 53.3 |
| GLC (之前注视估计 SOTA) | 37.8 | 52.9 | 29.4 | 58.3 | 65.4 | 52.6 |
| DFG+ (之前注视预测 SOTA) | 37.3 | 52.3 | 29.0 | 57.6 | 65.5 | 51.3 |
| **CSTS (本文)** | **39.7** | 53.3 | **31.6** | **59.9** | 66.8 | **54.3** |

提升：比 DFG+ (注视预测 SOTA) +2.4% / +2.3% F1，比 GLC (注视估计 SOTA) +1.9% / +1.6% F1。

### 消融实验

| 配置 | Ego4D F1 | Aria F1 | 说明 |
|------|---------|---------|------|
| Vision only | 37.2 | 57.5 | 仅视觉 baseline |
| S-fusion | 38.6 | 58.6 | 仅空间融合 (+1.4/+1.1) |
| T-fusion | 38.7 | 58.6 | 仅时间融合 (+1.5/+1.1) |
| STS (S+T) | 39.2 | 59.3 | 时空分离融合 (+2.0/+1.8) |
| **CSTS (STS + 对比)** | **39.7** | **59.9** | 加对比学习 (+2.5/+2.4) |

融合策略对比：

| 融合方式 | Ego4D F1 | Aria F1 | 说明 |
|---------|---------|---------|------|
| Linear | 38.2 | 58.1 | 线性融合 |
| Bilinear | 37.6 | 57.7 | 双线性融合 |
| Concat. | 38.1 | 58.0 | 拼接融合 |
| Vanilla SA | 38.5 | 58.0 | 标准联合自注意力 |
| **STS (本文)** | **39.2** | **59.3** | 时空分离融合最优 |

后融合对比 vs 传统对比：

| 方式 | Ego4D F1 | Aria F1 |
|------|---------|---------|
| STS (无对比) | 39.2 | 59.3 |
| STS + Vanilla Contr (原始特征对比) | 39.0 | 59.1 |
| **STS + Post Contr (融合后对比)** | **39.7** | **59.9** |

### 关键发现

- **音频显著提升注视预测**: 加入音频后 F1 提升 +2.5% (Ego4D) 和 +2.4% (Aria)，验证了音频对第一人称注视行为的重要驱动作用。
- **分离融合优于联合融合**: STS 比所有联合融合策略（Linear、Bilinear、Concat、Vanilla SA）都更有效，验证了在第一人称场景中需要分别处理音视频的空间和时间相关性。
- **后融合对比学习比传统对比更有效**: 在融合后特征上施加对比损失比在原始 embedding 上更好，传统对比甚至略有负面影响。
- **模型在所有预测时间步上一致优于 baseline**: 随着预测时间步远离当前时刻，任务难度增加但模型优势保持。
- 空间融合可视化显示模型能准确定位说话者位置并跟踪声音源的变化。

## 亮点与洞察

- **首次引入音频用于第一人称注视预测**: 这是一个有道理但一直被忽视的方向，填补了重要的研究空白。
- **时空分离融合的动机非常充分**: 不是技术驱动的设计，而是从第一人称视频的本质特性（视角变化、反应延迟）出发，有很强的认知科学依据。
- **后融合对比学习 (Post-Fusion Contrastive Loss)**: 这是一个新颖的贡献——在融合后的而非原始的特征上做对比学习更有效，因为融合后特征已包含时空关联信息，对比损失可以进一步增强这些关联。
- **可视化分析有说服力**: 空间关联权重的可视化清晰展示了模型如何追踪声音源和说话者位置的变化。

## 局限与展望

- 仅在社交场景（Ego4D, Aria）中验证，更多活动场景（如烹饪、运动）的音频特性可能不同，泛化性待验证。
- 音频编码器使用轻量级 Transformer，更强的预训练音频模型（如 BEATs、AudioMAE）可能带来进一步提升。
- 空间融合和时间融合的合并方式（通道重加权）相对简单，可探索更复杂的交互机制。
- 注视热图预测的粒度受限于 $256 \times 256$ 分辨率，高分辨率场景可能需要更细粒度的预测。
- 模型未显式建模注视-动作关联，结合动作预测可能进一步提升注视预测性能。

## 相关工作与启发

- **vs DFG/DFG+**: 之前的注视预测 SOTA，使用 CNN 生成未来帧再预测注视。本文直接从当前音视频预测未来注视，不需要视频生成步骤。
- **vs GLC**: 当前注视估计的 SOTA，使用全局-局部关联建模。本文将其扩展到预测任务并加入音频模态。
- **vs 音视频显著性预测**: 传统显著性预测面向固定视角的第三人称视频，使用联合融合。本文面向第一人称视频的独特挑战（视角变化、反应延迟），采用分离融合策略。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次引入音频用于第一人称注视预测，时空分离融合和后融合对比学习都是新颖的设计
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集，极其详尽的消融（模块消融、融合策略对比、对比学习策略对比），逐帧分析和可视化
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法描述系统，融合公式推导完整
- 价值: ⭐⭐⭐⭐ 为第一人称感知和 AR 应用提供了重要的多模态基线，后融合对比学习策略有通用参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Siamese Vision Transformers are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[ICML 2026\] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](../../ICML2026/audio_speech/do_audio_llms_listen_or_read_analyzing_and_mitigating_paralinguistic_failures_wi.md)

</div>

<!-- RELATED:END -->
