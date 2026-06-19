---
title: >-
  [论文解读] DiffA: Large Language Diffusion Models Can Listen and Understand
description: >-
  [AAAI 2026][音频/语音][扩散语言模型] 提出 DIFFA——首个基于扩散语言模型的大型音频-语言模型，通过冻结 LLaDA-8B 骨干网络 + 轻量双适配器架构 + 两阶段训练管线，仅用 960 小时 ASR 数据和 127 小时合成指令数据就在 MMSU、MMAU、VoiceBench 上达到与自回归 baseline 竞争的性能。
tags:
  - "AAAI 2026"
  - "音频/语音"
  - "扩散语言模型"
  - "音频理解"
  - "大型音频-语言模型"
  - "LLaDA"
  - "参数高效适配"
---

# DiffA: Large Language Diffusion Models Can Listen and Understand

**会议**: AAAI 2026  
**arXiv**: [2507.18452](https://arxiv.org/abs/2507.18452)  
**代码**: [GitHub](https://github.com/NKU-HLT/DIFFA)  
**领域**: 图像生成  
**关键词**: 扩散语言模型, 音频理解, 大型音频-语言模型, LLaDA, 参数高效适配

## 一句话总结

提出 DIFFA——首个基于扩散语言模型的大型音频-语言模型，通过冻结 LLaDA-8B 骨干网络 + 轻量双适配器架构 + 两阶段训练管线，仅用 960 小时 ASR 数据和 127 小时合成指令数据就在 MMSU、MMAU、VoiceBench 上达到与自回归 baseline 竞争的性能。

## 研究背景与动机

大型音频-语言模型（LALMs）近年发展迅速，但现有方法几乎全部基于自回归（AR）解码范式：

**编码器+LLM 范式**（如 Qwen2-Audio、SALMONN）：通过适配器将语音编码器的输出投射到 LLM 输入空间

**语音 token 化范式**（如 SpeechGPT、Moshi）：将音频离散化为 token 后直接用 LLM 训练

这两种范式都依赖 AR 解码，存在固有缺陷：曝光偏差、生成速度慢、缺乏双向上下文建模和部分条件推理的灵活性。

扩散语言模型（如 LLaDA）已在文本域展现出与 AR 模型匹敌的能力，且已扩展到视觉-语言任务（LLaDA-V）。但音频模态在扩散语言模型中完全未被探索。音频具有独特的声学变异性、复杂时间结构和丰富的副语言信息，是否适合扩散建模是一个开放问题。

## 方法详解

### 整体框架

DIFFA 采用模块化高效设计：

- **语音编码器**：冻结的 Whisper-Small（88.2M 参数）
- **双适配器**：语义适配器（14.4M）+ 声学适配器（22.3M）
- **语言骨干**：冻结的 LLaDA-8B-Instruct（8.1B 参数）

训练全程冻结编码器和语言模型，仅训练两个轻量适配器，总可训练参数约 36.7M（< 0.5%）。

### 关键设计

#### 1. 双适配器架构

**语义适配器**：
- 2 层卷积网络（4× 降采样）+ 2 层线性投影
- 将 Whisper 的 50 Hz 输出压缩为 12.5 Hz
- 从编码器最终输出提取高级语义特征

**声学适配器**：
- 2 层 Q-Former 结构，带 64 个可训练查询向量
- 从编码器**中间层状态**提取低级声学特征（音高、情感、语速等副语言信息）

最终音频表示为两个适配器输出的拼接，作为前缀 token 追加到 LLM 输入。

#### 2. 基于 LLaDA 的扩散解码

LLaDA 定义前向掩码过程：每个 token 以概率 $t$ 被替换为特殊掩码 token $\text{M}$。掩码预测器 $p_\theta(x_0|x_t)$ 训练恢复被掩码的 token：

$$\mathcal{L}(\theta) = -\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^{L} \mathbf{1}[x_t^i = \text{M}] \log p_\theta(x_0^i|x_t)\right]$$

融合音频条件后，目标函数变为：

$$L_a = -\mathbb{E}_{t,a_0,p_0,r_0,r_t}\left[\frac{1}{t}\sum_{i=1}^{L'} \mathbf{1}[r_t^i = \text{M}] \log p_\theta(r_0^i \mid a_0, p_0, r_t)\right]$$

训练时音频和提示 token 保持未掩码状态，仅对响应 token 执行掩码-预测。

#### 3. 数据构建

受 DESTA 系列启发，通过提示 LLM 生成合成指令数据：
- 输入音频转录文本 + 声学属性（性别、口音、情感、时长等 10 种标注属性）
- 提示："What can you hear from the audio?"
- LLM 生成的回答作为监督信号

此外引入**自蒸馏重写**：先用 Qwen3-8B 生成初始描述，再让 LLaDA 重写以对齐其内部数据分布。

### 损失函数 / 训练策略

**两阶段训练**：

**Stage 1——语义对齐**：
- 数据：LibriSpeech 960 小时
- 目标：ASR 任务，训练语义适配器对齐语音编码器与语言模型
- 仅训练语义适配器

**Stage 2——模态对齐**：
- 数据：127 小时合成指令数据（VCTK、Accentdb、IEMOCAP、dailytalk、VoxCeleb1）
- 目标：音频描述任务，训练双适配器
- 同时训练语义适配器和声学适配器

推理采用**半自回归策略**：从左到右按块生成，块内并行预测 token 并对低置信度位置重新掩码。

## 实验关键数据

### 主实验

**表2：MMSU 基准（语音语言理解，5000 个样本）**

| 模型 | 感知 | 推理 | 总体 |
|------|------|------|------|
| Gemini-1.5-Pro | 46.10 | 76.16 | 60.68 |
| Qwen2.5-Omni | 42.50 | 79.83 | 60.57 |
| Qwen2-Audio-Instruct | 39.02 | 68.90 | 53.27 |
| **DIFFA** | **40.28** | **72.92** | **56.04** |

**表3：MMAU 基准（音频推理，3 领域 27 技能）**

| 模型 | Sound | Music | Speech | Avg |
|------|-------|-------|--------|-----|
| GPT-4o mini Audio | 50.75 | 39.22 | 69.07 | 53.01 |
| Qwen2-Audio-Instruct | 67.27 | 56.29 | 55.26 | 59.61 |
| **DIFFA** | **46.25** | **43.41** | **59.46** | **49.71** |

**表4：VoiceBench（语音 QA）**

| 模型 | SD-QA | OBQA | IFEval | AdvBench | Overall |
|------|-------|------|--------|----------|---------|
| Qwen2-Audio | 35.72 | 49.45 | 26.33 | 96.73 | 55.34 |
| DiVA | 57.06 | 25.49 | 39.16 | 98.27 | 55.70 |
| **DIFFA** | **34.45** | **35.60** | **26.56** | **76.54** | **48.22** |

DIFFA 在 MMSU 上超越 Qwen2-Audio-Instruct 近 3 个点，语义推理尤其突出（81.53%）。

### 消融实验

**表5：架构与适配器设计消融**

| LLM 骨干 | 适配器 | MMAU | MMSU |
|----------|--------|------|------|
| LLaMA 3.1 (AR) | Dual | 28.40 | 38.40 |
| LLaDA (Diffusion) | Single | 47.70 | 52.88 |
| **LLaDA (Diffusion)** | **Dual** | **49.71** | **56.04** |

**表6：指令数据源消融**

| 数据源 | MMAU | MMSU | VoiceBench | Avg |
|--------|------|------|------------|-----|
| LLaMA 3 | 51.71 | 54.72 | 37.17 | 47.86 |
| Qwen3 | 49.71 | 56.04 | 48.22 | 51.32 |
| rewrite-Qwen3 | 50.41 | 56.43 | 46.60 | 51.15 |

### 关键发现

1. **扩散 vs 自回归**：将 AR 骨干（LLaMA 3.1）替换为扩散骨干（LLaDA）后 MMAU 提升 23.3 个点（28.4→51.7），MMSU 提升 16.3 个点——扩散语言模型对音频理解有明显优势
2. **双适配器 > 单适配器**：声学适配器带来 +2.01（MMAU）和 +3.16（MMSU）的增益，证明低级声学特征的重要性
3. **数据效率极高**：仅 127 小时合成数据 + 960 小时 ASR 数据，对比 Qwen2-Audio 的 51 万小时训练数据
4. DIFFA 在语义推理上表现最强（81.53%），音韵和副语言感知偏弱
5. 使用 LLaDA 生成的指令数据效果优于 LLaMA-3 的，暗示扩散模型的归纳偏置可能产生更对齐的监督

## 亮点与洞察

1. **首个扩散式 LALM**：开辟了音频理解领域的新方向，证明扩散语言模型不仅适用于文本和视觉，也适用于音频
2. **极致数据效率**：用不到 Qwen2-Audio 1/400 的训练数据达到竞争性能，仅需 72 A800 GPU 小时
3. **冻结策略优雅**：冻结编码器+语言模型，仅训练 36.7M 适配器参数（< 0.5%），避免灾难性遗忘
4. **双适配器设计**：语义+声学双路径互补，分别捕获高级语义和低级副语言信息

## 局限与展望

1. **训练数据受限**：960 + 127 小时的数据量可能限制在低资源口音、噪声环境等复杂场景的泛化能力
2. **副语言感知不足**：音韵和副语言任务得分明显低于语义推理，需要更多声学数据
3. **推理速度**：扩散解码需要多步迭代，实际推理延迟可能高于 AR 模型
4. **缺乏语音生成能力**：当前仅做理解不做生成，与全双工对话系统（如 Moshi）相比功能有限
5. 可扩展更多音频类型（音乐、环境声）的训练数据

## 相关工作与启发

- **LLaDA**：扩散语言模型的基础，证明了掩码扩散过程可替代自回归
- **LLaDA-V**：将扩散 LLM 扩展到视觉-语言，本文进一步扩展到音频
- **DESTA-2**：合成指令数据的构建范式，但采用级联设计（先 Whisper 转录再生成），本文实现端到端
- **Qwen2-Audio**：主要竞争对手，性能强但数据需求巨大（51 万小时）
- 启发：扩散 LLM 的模态扩展范式——冻结骨干 + 轻量适配器——可能成为多模态扩展的通用方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首个扩散式音频语言模型）
- 技术深度: ⭐⭐⭐⭐（双适配器+两阶段训练设计合理）
- 实验完整性: ⭐⭐⭐⭐（3 个 benchmark + 全面消融）
- 实用价值: ⭐⭐⭐⭐（极高数据效率，低资源场景友好）
- 总体评分: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](ahamask_reliable_task_specification_for_large_audio_language.md)
- [\[AAAI 2026\] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](../../ACL2026/audio_speech/speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](../../ACL2025/audio_speech/who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)
- [\[ICML 2026\] Focus Then Listen: An Empirical Study of Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models](../../ICML2026/audio_speech/focus_then_listen_an_empirical_study_of_plug-and-play_audio_enhancer_for_noise-r.md)

</div>

<!-- RELATED:END -->
