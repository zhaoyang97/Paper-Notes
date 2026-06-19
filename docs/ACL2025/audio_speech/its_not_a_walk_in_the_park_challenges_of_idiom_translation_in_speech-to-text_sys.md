---
title: >-
  [论文解读] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems
description: >-
  [ACL 2025][音频/语音][习语翻译] 本文首次系统比较了语音到文本翻译（SLT）、文本机器翻译（MT）和大语言模型（LLM）在习语翻译任务上的表现，发现 SLT 系统在处理习语时性能大幅下降，即便在编码器高层仍倾向于字面翻译，而 MT 和 LLM 对习语的处理能力明显更优。 习语（idiom）是一类其含义无法从组成…
tags:
  - "ACL 2025"
  - "音频/语音"
  - "习语翻译"
  - "语音到文本翻译"
  - "DecoderLens"
  - "级联系统"
  - "比喻语言"
---

# It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems

**会议**: ACL 2025  
**arXiv**: [2506.02995](https://arxiv.org/abs/2506.02995)  
**代码**: [有](https://github.com/IuliiaZaitova/idiom_s2t)  
**领域**: 其他  
**关键词**: 习语翻译, 语音到文本翻译, DecoderLens, 级联系统, 比喻语言

## 一句话总结

本文首次系统比较了语音到文本翻译（SLT）、文本机器翻译（MT）和大语言模型（LLM）在习语翻译任务上的表现，发现 SLT 系统在处理习语时性能大幅下降，即便在编码器高层仍倾向于字面翻译，而 MT 和 LLM 对习语的处理能力明显更优。

## 研究背景与动机

习语（idiom）是一类其含义无法从组成单词的字面意思推导出来的固定表达，如英文的 "raining cats and dogs" 或德语的 "Es ist mir wurst"（字面意为"对我来说是香肠"，实际意为"我不在乎"）。虽然现代机器翻译系统取得了长足进步，但习语翻译仍然是一个主要挑战。

已有大量工作研究了文本MT中的习语翻译问题，但关于**语音翻译（SLT）中的习语翻译**研究极为稀少。SLT 系统面临额外的复杂性：需要同时整合声学信息、句法信息和语义信息。理解这些系统在习语上失败的原因和方式，对于进一步改进 SLT 系统至关重要。

本文的核心研究问题包括：
1. 端到端 SLT 与文本 MT、LLM、级联系统在习语翻译上的相对表现如何？
2. 这些系统如何处理习语数据与常规新闻数据？
3. 通过 DecoderLens 逐层分析，能否定位系统在哪一层"失去"了习语的比喻含义？

## 方法详解

### 整体框架

本文不提出新模型，而是设计了一套**系统性评估框架**，在德语→英语和俄语→英语两个语言对上，比较多类翻译系统对习语和新闻数据的翻译质量。

### 关键设计

1. **评估系统的全面覆盖**：

    - **MT 系统**: SeamlessM4T (文本到文本)、NLLB (200语言翻译模型)
    - **LLM**: LLaMA 3 (语言特化微调版)、DeepSeek-V3
    - **SLT 系统**: SeamlessM4T (语音到文本)、Whisper Large v3
    - **级联系统**: ASR (Whisper 或 SeamlessM4T) → MT/LLM
    - 设计动机：覆盖端到端和级联两种范式，全面诊断性能瓶颈

2. **评估数据集构建**：

    - **新闻数据**: 从 News Commentary 平行语料库随机选取 250 句/语言对，作为基线
    - **习语数据**: 从 Idioms-InContext-MT 数据集中人工筛选 250 个需要非字面翻译的习语
    - 排除了字面翻译即可保留比喻含义的习语（如 "break someone's heart"）
    - 使用 Microsoft Edge TTS 合成音频，确保声学条件一致

3. **DecoderLens 逐层分析**：

    - 用中间编码器层的激活替换最终编码器输出，观察每一层的翻译结果
    - 对 50 个样本进行人工标注，分析翻译质量随层数的变化
    - 设计动机：揭示 SLT 和 MT 系统在编码过程中处理比喻语言的机制差异

4. **人工标注方案**：

    - 设计了 7 类标注类别，从"正确（惯用表达）"到"空输出"
    - 特别区分了习语特有的类别：正确惯用翻译、释义翻译、字面翻译
    - 两名标注者通过讨论解决分歧

### 评估指标

- **自动指标**: COMET (语义等价性和流畅度，与人工判断高度相关)
- **人工标注**: 每个语言-数据集-模型组合随机抽取 50 句进行标注
- 使用 Mann-Whitney U 检验和 Bonferroni 校正验证统计显著性

## 实验关键数据

### 主实验（COMET 评分对比）

| 系统 | 德→英 新闻 | 德→英 习语 | 俄→英 新闻 | 俄→英 习语 |
|------|-----------|-----------|-----------|-----------|
| Whisper (直接 SLT) | 0.8437 | 0.6402 | 0.8318 | 0.6916 |
| Seamless (直接 SLT) | 0.8697 | 0.6483 | 0.8512 | 0.6941 |
| NLLB (文本 MT) | 0.8841 | 0.6749 | 0.8664 | 0.7214 |
| Seamless (文本 MT) | 0.8870 | 0.6784 | 0.8694 | 0.7262 |
| LLaMA | 0.8724 | 0.6971 | 0.8211 | 0.7354 |
| **DeepSeek** | **0.8940** | **0.7675** | **0.8741** | **0.7939** |
| Whisper(ASR)→DeepSeek | 0.8887 | 0.7584 | 0.8607 | 0.7873 |

### 消融/对比分析

| 对比维度 | 发现 |
|---------|------|
| SLT vs MT (习语) | SLT 在习语上 COMET 下降 24.2%（Whisper 德→英），MT 下降约 14% |
| 德语 vs 俄语 | 德语习语-新闻差距更大（平均 0.198），俄语约 0.143 |
| 级联 vs 端到端SLT | 级联系统大多优于端到端 SLT，说明问题不仅来自 ASR 错误 |
| DecoderLens 层分析 | SLT 在第 0-20 层输出空/无意义内容，21-30 层出现幻觉，仅最后几层产生相关翻译 |

### 关键发现

1. **DeepSeek 全面领先**：在所有配置下（纯文本、级联）表现最优，习语翻译 COMET 约 0.76-0.79
2. **SLT 的习语性能断崖式下降**：Whisper 在德→英习语上的 COMET 仅 0.64，较新闻下降 24%
3. **级联系统优于端到端 SLT**：暗示 SLT 的问题不仅是 ASR 转写错误，更深层是声学-语义整合的困难
4. **逐层分析揭示结构差异**：SLT 系统直到编码器高层才开始产生有意义的翻译，且在最后几层仍回退到字面翻译；MT 系统则呈现更平滑的过渡

## 亮点与洞察

- **首次系统研究 SLT 中的习语翻译**，填补了语音翻译领域在比喻语言处理上的研究空白
- **DecoderLens 的创新应用**：通过逐层分析揭示了 SLT 和 MT 在处理比喻语言时的内部表征差异
- **实用建议**：当翻译内容可能包含习语时，推荐使用级联系统而非端到端 SLT
- 人工标注方案设计精细，特别是对习语翻译类别（惯用/释义/字面）的区分有参考价值

## 局限与展望

1. 仅覆盖德→英和俄→英两个语言对，习语使用在不同语言间差异很大
2. 使用合成语音而非真实自然语音，虽然已有研究表明影响较小
3. DecoderLens 仅适用于编码器-解码器架构，无法分析纯解码器模型（如 LLaMA）
4. 人工标注具有主观性，且样本量较小（50句/组合）
5. 未探索针对 SLT 的习语增强策略，如习语感知的微调或数据增强

## 相关工作与启发

- 与 Dankers et al. (2022) 和 Baziotis et al. (2023) 的 Transformer 比喻语言研究形成互补
- DecoderLens (Langedijk et al., 2024) 方法可推广到其他语义复杂场景的分析
- 级联系统在新场景下的优势值得进一步研究，特别是结合强 ASR + 强 LLM 的组合

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统研究 SLT 中的习语翻译，DecoderLens 的应用有新意
- **实验充分度**: ⭐⭐⭐⭐ — 多系统、多语言、自动+人工评估，逐层分析深入。但语言对和样本量可更多
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，问题动机阐释充分
- **价值**: ⭐⭐⭐⭐ — 对语音翻译社区有重要的实践指导意义，揭示了 SLT 在语义理解上的根本性不足

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](../../ICML2025/audio_speech/sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](../../ICML2025/audio_speech/do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)

</div>

<!-- RELATED:END -->
