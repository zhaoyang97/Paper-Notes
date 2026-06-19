---
title: >-
  [论文解读] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR
description: >-
  [AAAI 2026][音频/语音][对话式语音识别] MARS 提出多模态检索-选择方法为对话式 LLM-ASR 挑选最相关的历史上下文（而非固定前几句或全部历史），在仅用 1.5K 小时训练数据的情况下超越了用 179K 小时数据训练的 SOTA 系统 TEA-ASLP。 领域现状：对话式语音识别需要利用历史上下文来处理…
tags:
  - "AAAI 2026"
  - "音频/语音"
  - "对话式语音识别"
  - "LLM-ASR"
  - "多模态检索"
  - "RAG"
  - "历史上下文选择"
---

# Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR

**会议**: AAAI 2026  
**arXiv**: [2508.01166](https://arxiv.org/abs/2508.01166)  
**代码**: 无  
**领域**: 语音识别 / 音频处理  
**关键词**: 对话式语音识别, LLM-ASR, 多模态检索, RAG, 历史上下文选择

## 一句话总结

MARS 提出多模态检索-选择方法为对话式 LLM-ASR 挑选最相关的历史上下文（而非固定前几句或全部历史），在仅用 1.5K 小时训练数据的情况下超越了用 179K 小时数据训练的 SOTA 系统 TEA-ASLP。

## 研究背景与动机

**领域现状**：对话式语音识别需要利用历史上下文来处理说话风格、填充词、上下文关联等挑战。近期 LLM-ASR 方法已展示出利用长上下文的潜力。

**现有痛点**：现有对话式 LLM-ASR 使用上下文的方式存在两个极端：(1) 固定前 N 句：假设最相关上下文在最近几句中，但实际中最相关的历史可能出现在更早的对话中，且最近几句可能充满填充词等无关内容；(2) 全部对话历史：提供丰富上下文但引入大量冗余信息，干扰识别且计算开销大。

**核心矛盾**：历史上下文的位置不固定，最相关的上下文可能在对话早期；而全部历史又包含太多不相关信息。需要一种精准定位最相关历史上下文的机制。

**本文目标** 如何从整个对话历史中检索并选择出对当前语句最有帮助的单条历史上下文，以增强对话式 LLM-ASR 性能。

**切入角度**：借鉴 RAG 的检索理念，但针对 ASR 场景定制——RAG 旨在生成新内容，ASR 旨在将语音映射为文本，目标不同。MARS 从语音和文本双模态进行检索，再用 near-ideal ranking 方法选取最佳单条上下文。

**核心 idea**：用语音和文本双模态检索候选历史上下文，通过 TOPSIS 风格的 near-ideal ranking 综合两种相似度选出最佳一条输入 LLM，实现"少即是多"的上下文利用。

## 方法详解

### 整体框架

MARS 的流程：(1) 用微调的 Whisper 构建数据库，存储每条语句的 ID、语音嵌入和 hypothesis 三元组；(2) 对当前语句，多模态检索模块从数据库中分别检索语音和文本 Top-K 相似的历史上下文；(3) 多模态选择模块从检索结果中确定最佳一条历史上下文；(4) 将最佳上下文的 hypothesis、当前语句的语音嵌入和 hypothesis 以及语言提示一起输入 LLM 生成转录。

### 关键设计

1. **多模态检索**:

    - 功能：从整个对话历史中分别用语音和文本模态检索 Top-K 相似的历史上下文
    - 核心思路：语音模态使用 FastDTW 计算帧级声学相似度（对齐两段语音嵌入的最小累积距离）加上 pooling 后的余弦相似度作为句级相似度，加权求和得到语音检索相似度，选 Top-K。文本模态使用嵌入模型（Qwen3-Embedding-0.6B）计算 hypothesis 间的语义相似度，选 Top-K。语音检索可捕获发音变体减少发音错误，文本检索可消除词歧义
    - 设计动机：单模态检索无法全面衡量相似性——语音相似度捕获发音和韵律，文本相似度捕获语义关联，两者互补

2. **Near-Ideal Ranking 多模态选择**:

    - 功能：从 $2K$ 条检索结果中选出综合语音和文本相似度最优的单条上下文
    - 核心思路：首先为所有 $2K$ 条候选计算两种相似度（语音/文本检索各自缺少的需要补充计算）。由于两种相似度量纲不同，不能直接求和排名。采用 TOPSIS 风格的方法：(a) 规范化两种相似度消除量纲差异 $sr_i = sw_i / \sqrt{\sum sw_j^2}$；(b) 定义理想点（两种相似度均最优）和负理想点（均最差）；(c) 计算每条候选到理想点和负理想点的欧氏距离 $d_i^+, d_i^-$；(d) 计算相对接近度 $c_i = d_i^- / (d_i^+ + d_i^-)$，取最大者为最佳上下文
    - 设计动机：两种相似度的计算方法不同，量纲不同，不能简单相加。TOPSIS 方法天然适合多维度不同量纲指标的综合排序

3. **自适应上下文解码策略**:

    - 功能：训练时随机决定是否使用检索到的历史上下文
    - 核心思路：以 50% 概率随机遮蔽最佳历史上下文，防止模型过度依赖历史而忽视当前语句本身。推理时支持三种解码：直接解码（不用历史）、MARS 解码（单遍）、两遍解码（先直接解码获取初始 hypothesis 再用 MARS 重新检索和解码）
    - 设计动机：增强泛化能力，使模型在无合适历史上下文时也能保持良好性能

### 损失函数 / 训练策略

使用 Qwen2.5-7B-Instruct 作为 LLM，LoRA（rank=64, alpha=256）微调七个投影层。Projector 为两层线性+ReLU。训练 3 epoch，使用 Adam 优化器，学习率峰值 0.0001。所有 checkpoint 平均用于推理。

## 实验关键数据

### 主实验

| 方法 | 训练数据 | MER (Dev) | MER (Test) |
|------|---------|-----------|------------|
| Vanilla Whisper-large-v3 | 预训练 | 16.82 | 17.33 |
| Fine-tuned Whisper | 1.5K hr | 11.87 | 10.15 |
| Qwen2-Audio | 预训练 | 51.90 | 53.47 |
| TEA-ASLP (前SOTA) | **179K hr** | 10.62 | 9.60 |
| **MARS** | **1.5K hr** | **8.97** | **8.35** |

MARS 仅用 1.5K 小时训练数据，MER 比用 179K 小时数据的 TEA-ASLP 低 1.25 个点（相对 13% 改进）。

### 消融实验

| 配置 | MER (Dev) | MER (Test) | 说明 |
|------|-----------|------------|------|
| LLM-ASR (无上下文) | 12.75 | 11.04 | 基线 |
| + Hypothesis | 11.15 | 9.89 | 文本 hypothesis 有帮助 |
| + Speech Retrieval | 10.24 | 9.41 | 语音检索有效 |
| + Text Retrieval | 10.33 | 9.23 | 文本检索有效 |
| + Multi-modal Selection | 9.77 | 8.96 | 综合选择进一步提升 |
| + Two-pass Decoding | **8.97** | **8.35** | 两遍解码最优 |

### 关键发现

- 固定前 N 句上下文效果有限，且随 N 增大性能反降（N=5 时 MER 13.49 vs N=1 时 9.74），证实冗余信息有害
- 即使使用 ground-truth 转录作为上下文，Bi-context 方法的改进也不如 MARS 使用检索上下文的效果好，说明"选对上下文"比"上下文质量高"更重要
- 两遍解码显著优于单遍，因为第一遍 hypothesis 更准确，构建的数据库质量更高

## 亮点与洞察

- **数据效率惊人**：1.5K 小时训练数据超越 179K 小时的系统，说明精准利用上下文比暴力堆数据更有效，这对低资源场景有巨大启示
- **Near-ideal ranking**：巧妙的多准则决策方法（TOPSIS），将不同量纲的语音/文本相似度统一比较，这种方法可迁移到任何需要综合多个异质指标的场景
- **"少即是多"的设计哲学**：只选一条最佳上下文而非多条，避免信息冗余，与 RAG 中"塞满上下文"的常见做法形成对比

## 局限与展望

- 检索依赖第一遍 Whisper 的质量，如果初始 hypothesis 错误严重，后续检索和选择可能受影响
- Near-ideal ranking 假设语音和文本相似度同等重要（等权重），未探索自适应权重
- 仅选取单条历史上下文，某些情况下可能需要多条互补上下文
- FastDTW 在大规模对话中仍有一定计算开销，可以考虑更高效的近似方法
- 仅在 MLC-SLM 数据集上验证，泛化到其他对话场景需要进一步实验

## 相关工作与启发

- **vs TEA-ASLP**：前 SOTA，依赖 179K 小时大规模数据和 MoE 架构；MARS 通过精准上下文利用以 1/100 的数据量反超
- **vs Seewo/Bi-context**：固定前 N 句的策略，即使用 ground-truth 也效果有限；MARS 证明了检索式上下文选择的优越性
- **vs RAG**：RAG 侧重外部知识检索生成新内容，MARS 将检索限定在对话内部历史、目标是辅助转录而非生成

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 RAG 思路创造性地适配到对话 ASR，near-ideal ranking 有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 多语言大规模评估，详细消融，与多个基线全面对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图表丰富，逻辑通顺
- 价值: ⭐⭐⭐⭐⭐ 超强数据效率具有极高实用价值，刷新 MLC-SLM SOTA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](../../ACL2025/audio_speech/soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](../../ACL2026/audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)

</div>

<!-- RELATED:END -->
