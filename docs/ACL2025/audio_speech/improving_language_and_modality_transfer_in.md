---
title: >-
  [论文解读] Improving Language and Modality Transfer in Translation by Character-level Modeling
description: >-
  [ACL 2025][音频/语音][character-level] 提出基于字符级编码器 charSONAR 的跨语言跨模态翻译方法，通过 teacher-student 训练获得字符级文本编码器，再用轻量适配器连接 1000+ 语言的 CTC ASR 模型（MMS），在 75 语言文本翻译和 33 语言语音翻译上实现 SOTA，零资源低资源场景表现尤其突出。
tags:
  - "ACL 2025"
  - "音频/语音"
  - "character-level"
  - "multilingual translation"
  - "speech translation"
  - "SONAR"
  - "跨模态"
---

# Improving Language and Modality Transfer in Translation by Character-level Modeling

**会议**: ACL 2025  
**arXiv**: [2505.24561](https://arxiv.org/abs/2505.24561)  
**代码**: 无  
**领域**: 其他  
**关键词**: character-level, multilingual translation, speech translation, SONAR, cross-modal transfer

## 一句话总结
提出基于字符级编码器 charSONAR 的跨语言跨模态翻译方法，通过 teacher-student 训练获得字符级文本编码器，再用轻量适配器连接 1000+ 语言的 CTC ASR 模型（MMS），在 75 语言文本翻译和 33 语言语音翻译上实现 SOTA，零资源低资源场景表现尤其突出。

## 研究背景与动机

**领域现状**：翻译模型已支持 200-400 种文本语言和 100 种语音语言，但仅覆盖全球 5% 的语言。扩展到长尾低资源语言面临数据稀缺挑战。

**现有痛点**：(1) 子词分词（subword）的跨语言迁移能力有限。(2) 语音翻译中 CTC 输出（字符级）与文本编码器（子词级）长度/内容不匹配，造成模态鸿沟。(3) 音素化方法有歧义且无法扩展到 1000+ 语言。

**核心矛盾**：如何在文本和语音两个模态之间、高资源和低资源语言之间实现最大化知识迁移？

**本文目标** 用字符级建模统一文本和语音的表征输入空间，提升跨语言和跨模态迁移。

**切入角度**：基于 SONAR（多语言固定维嵌入空间）+ MMS（1000+ 语言 CTC ASR），字符级编码器天然与 CTC 输出对齐，消除了子词-字符长度不匹配问题。

**核心 idea**：字符级 SONAR 编码器 + 预训练 CTC→字符适配器 = 数据高效的跨语言跨模态翻译。

## 方法详解

### 整体框架
(1) Teacher-Student: SONAR(子词) → charSONAR(字符)，用插值 MSE 损失。(2) 适配器: MMS-CTC → charSONAR，用 MSE 损失。推理时用 SONAR decoder 从嵌入生成翻译。

### 关键设计

1. **charSONAR 训练**:

    - 仅保留 SONAR 词表中的单字符 token（256K→8K）。
    - 三种 MSE 目标：重构（$\text{MSE}(\mathbf{c}^x, \mathbf{e}^x)$）、翻译（$\text{MSE}(\mathbf{c}^x, \mathbf{e}^y)$）、**插值**（$\text{MSE}(\mathbf{c}^x, \frac{\mathbf{e}^x + \mathbf{e}^y}{2})$）。
    - 插值目标最优：语言对的平均 SONAR 嵌入比单语嵌入更适合跨语言空间。
    - 加入 ASR 式增强（去大写/去标点/字符噪声），增强跨模态鲁棒性。

2. **跨模态适配器**:

    - **预训练适配器**：利用 MMS 的 CTC 分类层 + charSONAR 的 embedding 层，做 soft prediction（softmax → embedding lookup），参数仅~200K。
    - **双适配器（Dual）**：预训练适配器 + 随机初始化适配器，用门控加权组合，总参数 2.5M。
    - CTC 压缩：平均连续相同预测、去除 blank，将音频表征压缩到字符级长度。

3. **零资源语音翻译**:

    - charSONAR 冻结 + MMS 冻结，仅训练适配器。
    - 训练数据仅需 ASR data（音频-转录对），无需平行语音翻译数据。

## 实验关键数据

### 主实验

| 方法 | 文本翻译 (FLORES+ 75 语言) | 语音翻译 (FLEURS 33 语言) |
|------|:---:|:---:|
| SONAR (子词) | xCOMET: 0.925 | - |
| **charSONAR** | **xCOMET: 0.934** | - |
| SEAMLESS (监督) | - | SOTA 基线 |
| Whisper cascade | - | 强基线 |
| **charSONAR + MMS** | **优于 SONAR** | **新 SOTA** |

### 消融实验

| 配置 | xCOMET | xSIM++ | 说明 |
|------|:---:|:---:|------|
| 重构目标 | 0.929 | 7.4 | 基础 |
| 翻译目标 | 0.924 | 6.6 | 搜索好但翻译略差 |
| **插值目标** | **0.931** | **6.6** | 两者兼顾 |
| + 预训练初始化 | 0.934 | 6.4 | 更快收敛 |
| 低资源语言提升 | 最大 | - | 字符级在低资源场景优势明显 |
| 零资源语言泛化 | 优于子词 | 优于子词 | 字符共享增强迁移 |

### 关键发现
- **字符级优于子词级**：在 75 语言上整体更优，低资源和零资源场景优势尤其显著。
- **插值嵌入空间更优**：语言对的"平均"SONAR 嵌入比单语嵌入更适合做跨语言锚点——低资源语言受益最大（与高资源语言平均后质量提升）。
- **极轻量适配器即可实现 SOTA 语音翻译**：仅 2.5M 参数的适配器就超越了完全监督的 SEAMLESS 系统。

## 亮点与洞察
- **字符级统一两种模态**的洞察深刻：CTC 输出天然是字符级的，文本编码器也用字符输入，消除了模态鸿沟——这比之前的音素共享或子词压缩方案更简洁。
- **固定维嵌入 bottleneck 的意外优势**：SONAR 的均值池化生成固定维嵌入，使字符级序列长度增加不影响解码器计算。
- **适配器的"预训练初始化"设计**：利用已有的 MMS CTC 层和 charSONAR embedding 层初始化，最大化利用预训练知识。

## 局限与展望
- 编码器端字符级化增加了序列长度（1.5-3x），编码成本更高。
- 仅测试了 X→Eng 方向的语音翻译，其他方向未验证。
- 字符级分词对中文/日文等字符语言效果可能不同（每个字符承载更多语义）。

## 相关工作与启发
- **vs ZeroSwot**: ZeroSwot 用 Wasserstein 距离对齐语音-文本表征；charSONAR 用简单 MSE + 字符级统一达到更好效果。
- **vs ByT5 (字符级LM)**: ByT5 展示了字符级在噪声鲁棒性上的优势；charSONAR 将此推广到翻译和语音。

## 评分
- 新颖性: ⭐⭐⭐⭐ 字符级统一文本和语音翻译的方案简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 75 语言文本 + 33 语言语音 + 消融充分
- 写作质量: ⭐⭐⭐⭐ 方法清晰，实验全面
- 价值: ⭐⭐⭐⭐⭐ 潜在支持 1000+ 语言的语音翻译，实际影响大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [\[ACL 2025\] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [\[ACL 2026\] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](../../ACL2026/audio_speech/sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)

</div>

<!-- RELATED:END -->
