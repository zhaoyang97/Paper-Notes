---
title: >-
  [论文解读] Long-Form Speech Generation with Spoken Language Models
description: >-
  [ICML 2025 Oral][音频/语音][语音语言模型] 提出 SpeechSSM，首个能在单次解码会话中学习和生成长达 16 分钟语音的 textless 语音语言模型，利用 Griffin 混合 SSM 架构实现常量内存解码和无限上下文，并引入 LibriSpeech-Long 评估基准和新的嵌入/LLM 评判指标。
tags:
  - "ICML 2025 Oral"
  - "音频/语音"
  - "语音语言模型"
  - "长程语音生成"
  - "状态空间模型"
  - "SSM"
  - "语音评估"
---

# Long-Form Speech Generation with Spoken Language Models

**会议**: ICML 2025 Oral  
**arXiv**: [2412.18603](https://arxiv.org/abs/2412.18603)  
**代码**: [https://google.github.io/tacotron/publications/speechssm/](https://google.github.io/tacotron/publications/speechssm/)  
**领域**: 图像生成  
**关键词**: 语音语言模型, 长程语音生成, 状态空间模型, SSM, 语音评估

## 一句话总结
提出 SpeechSSM，首个能在单次解码会话中学习和生成长达 16 分钟语音的 textless 语音语言模型，利用 Griffin 混合 SSM 架构实现常量内存解码和无限上下文，并引入 LibriSpeech-Long 评估基准和新的嵌入/LLM 评判指标。

## 研究背景与动机
**领域现状**：现有语音语言模型（如 GSLM、TWIST、Spirit LM）最多只能生成几十秒的语音，受限于 Transformer 的二次复杂度和语音 token 的高时间分辨率（25Hz，约 10 个语音 token 对应 1-2 个文本 token）。

**现有痛点**：(a) Transformer 推理内存线性增长，无法无限延续；(b) 语音 token 序列极长导致语义一致性丧失；(c) 现有评估指标在长程设置下噪声大且不够判别。

**核心矛盾**：真实应用（语音助手对话、播客、有声书）需要分钟级连贯语音，但模型只能可靠生成秒级语音。

**本文目标** (a) 建模层面：常量内存、无限上下文的长程语音生成；(b) 评估层面：长程语音生成的评估方法和基准。

**切入角度**：用混合 SSM（状态空间模型 + 局部注意力）替代纯 Transformer，利用 SSM 的固定状态压缩无限距离的上下文。

**核心 idea**：Griffin 混合 SSM + 高质量语义 token (USM-v2) + 窗口化策略 = 可无限生成的语音 LM。

## 方法详解

### 整体框架
SpeechSSM 分两阶段：(1) 语义阶段：用 Griffin 混合 SSM 自回归预测 USM-v2 语义 token（25Hz，32k 词表）；(2) 声学阶段：用 SoundStorm 非自回归地将语义 token 转换为 SoundStream 声学 token，再解码为波形。说话人特征通过 3 秒语音提示在声学阶段注入。

### 关键设计

1. **Griffin 混合 SSM 架构**:

    - 功能：作为语义 token 的自回归解码器
    - 核心思路：交替排列 gated LRU（线性循环单元）和局部滑窗多查询注意力（2:1 比例），局部注意力捕捞近期上下文，LRU 状态传递跨任意距离的信息
    - 设计动机：满足常量内存解码 + 无限上下文 + 生成长度外推三个要求

2. **窗口化 Tokenization 和解码**:

    - 功能：让非 SSM 组件（语义 tokenizer、声学解码器）也能处理长程语音
    - 核心思路：将长音频切分为固定长度窗口（30s）并有 4s 重叠，独立 tokenize/decode 后在重叠边界处拼接
    - 设计动机：USM-v2 tokenizer 和 SoundStorm 都有上下文限制，窗口化是工程上的必要设计

3. **避免隐式 EOS**:

    - 功能：解决早期模型无法生成超过训练时长的问题
    - 核心思路：非因果 tokenizer（如 USM-v2）会在最后一个窗口的 token 中隐式编码"剩余长度"信息。解决方案：用音频开头的语音（而非静音）填充最后一个窗口
    - 设计动机：让 token 看起来像"后面还有语音"，从而支持外推

4. **文本 LM 初始化**:

    - 功能：从 RecurrentGemma-2B/9B 初始化 SpeechSSM
    - 核心思路：保留预训练的架构权重，丢弃文本 token embedding，重新初始化音频 token embedding
    - 设计动机：TWIST 等工作表明文本 LM 初始化能提升语义一致性

### 训练策略
- 在 LibriLight unlab-60k 上训练，默认使用 4 分钟（240s）音频段
- 16 TPUs (v5p)，100k 步，768k tokens/batch
- 采样温度 1，通过 checkpoint 选择（LibriSpeech-Long dev-clean 的 transcript PPL）

## 实验关键数据

### 主实验（短程 7s 续写）

| 模型 | PPL ↓ | SBERT ↑ | SpkrSim ↑ | N-MOS ↑ |
|------|-------|---------|-----------|---------|
| TWIST-7B | 6.54 | 0.20 | 0.41 | 3.24 |
| Spirit LM 7B | 6.17 | 0.19 | 0.45 | 3.00 |
| **SpeechSSM-2B** | **5.76** | **0.23** | **0.79** | **3.87** |
| **SpeechSSM-9B** | **5.60** | **0.23** | **0.79** | **3.94** |
| Ground Truth | 5.63 | 1.00 | 0.84 | 4.02 |

### 长程评估（4 分钟续写）

| 模型 | PPL ↓ | SpkrSim ↑ | 说明 |
|------|-------|-----------|------|
| Spirit LM ⊞ | 较高 | 0.45 | 滑窗扩展，语义退化 |
| SpeechSSM-2B | **最低** | **0.79** | 保持语义一致性 |

### 关键发现
- SpeechSSM 的 SpkrSim 远高于其他模型（0.79 vs 0.41-0.45），归因于 USM-v2 的大词表(32k)和声学阶段的说话人提示
- 9B 模型在短程和长程上均优于 2B，说明 SSM 同样受益于 scaling
- 现有 sWUGGY 指标在大词表下失效，与生成质量不正相关

## 亮点与洞察
- **首个长程语音 LM**：从 10s 量级直接跳到 16min，是质的飞跃
- **LibriSpeech-Long 基准**：填补了长程语音评估的空白，开源且有标准化分割
- **LLM-as-Judge 评估**：用 LLM 做对比评估（side-by-side）解决了 ASR 指标噪声大的问题

## 局限与展望
- 仅支持朗读和自由发挥两种风格，未支持对话式语音
- 语义 tokenizer USM-v2 非开源，限制了可复现性
- 未与 Moshi 等支持对话的系统做直接对比
- 窗口化拼接可能在边界处产生不自然的过渡

## 评分
- 新颖性: ⭐⭐⭐⭐ SSM 在语音 LM 中的首次系统应用
- 实验充分度: ⭐⭐⭐⭐⭐ 短程/长程/新指标/新基准全面
- 写作质量: ⭐⭐⭐⭐⭐ 工程细节丰富，问题分析透彻
- 价值: ⭐⭐⭐⭐⭐ 对语音生成领域有开创性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ACL 2026\] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](../../ACL2026/audio_speech/comprehensive_benchmarking_of_long-form_speech_generation_in_diverse_scenarios.md)
- [\[CVPR 2026\] AudioStory: Generating Long-Form Narrative Audio with Large Language Models](../../CVPR2026/audio_speech/audiostory_generating_long-form_narrative_audio_with_large_language_models.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[ICML 2025\] Aligning Spoken Dialogue Models from User Interactions](aligning_spoken_dialogue_models_from_user_interactions.md)

</div>

<!-- RELATED:END -->
