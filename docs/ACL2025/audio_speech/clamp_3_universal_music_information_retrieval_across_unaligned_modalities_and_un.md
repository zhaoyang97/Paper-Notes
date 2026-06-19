---
title: >-
  [论文解读] CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages
description: >-
  [ACL 2025][音频/语音][音乐信息检索] 提出 CLaMP 3 统一框架，通过对比学习将乐谱、演奏信号、音频录音与多语言文本对齐到共享表示空间，在无配对训练数据的模态间实现跨模态检索，并展现出对未见语言的强泛化能力。 领域现状： 音乐信息检索（MIR）旨在开发处理、组织和访问音乐数据的计算工具…
tags:
  - "ACL 2025"
  - "音频/语音"
  - "音乐信息检索"
  - "多模态对齐"
  - "跨语言泛化"
  - "对比学习"
  - "检索增强生成"
---

# CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages

**会议**: ACL 2025  
**arXiv**: [2502.10362](https://arxiv.org/abs/2502.10362)  
**代码**: [https://github.com/sanderwood/clamp3](https://github.com/sanderwood/clamp3)  
**领域**: 语音  
**关键词**: 音乐信息检索, 多模态对齐, 跨语言泛化, 对比学习, 检索增强生成

## 一句话总结

提出 CLaMP 3 统一框架，通过对比学习将乐谱、演奏信号、音频录音与多语言文本对齐到共享表示空间，在无配对训练数据的模态间实现跨模态检索，并展现出对未见语言的强泛化能力。

## 研究背景与动机

**领域现状**: 音乐信息检索（MIR）旨在开发处理、组织和访问音乐数据的计算工具。核心挑战是基于自然语言查询检索音乐内容（乐谱、演奏信号、音频录音）。近年来的进展包括基于文本-音频对齐的检索和自动标注系统。

**现有痛点**: (1) 音乐存在多种形式（乐谱、MIDI、音频），异构表示结构使统一处理困难；(2) 音乐被全球各语言描述，现有数据集以英语为主，多语言覆盖不足；(3) 配对数据稀缺——跨模态配对数据和高质量音乐-文本对均严重匮乏。

**核心矛盾**: MIR 需要统一处理多种音乐模态和多种语言，但现有方法大多只关注特定模态对（如文本-音频或文本-乐谱），且缺乏跨语言能力和跨模态的统一表示。

**本文目标**: 如何在一个统一框架中同时对齐所有主要音乐模态与多语言文本，实现未配对模态间的检索和未见语言的泛化。

**切入角度**: 以文本为桥梁，借鉴 ImageBind 的多阶段对齐策略，通过对比学习将所有模态映射到共享空间；利用 RAG 从网络构建大规模多语言音乐-文本数据集。

**核心 idea**: 以文本为桥梁，通过多阶段对比学习统一对齐乐谱、演奏信号、音频和多语言文本到共享表示空间，辅以 RAG 构建的大规模 M4-RAG 数据集。

## 方法详解

### 整体框架

CLaMP 3 包含三个 Transformer 编码器：多语言文本编码器、符号音乐编码器和音频音乐编码器。训练目标是通过对比学习（InfoNCE 损失）对齐文本与音乐特征。

训练损失函数：

$$L_{CL} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i^t, z_i^m) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i^t, z_j^m) / \tau)}$$

其中 $z_i^t$ 和 $z_i^m$ 分别是文本和音乐嵌入，$\tau$ 为温度参数。

### 关键设计

**模块1: 多语言文本编码器**

- **功能**: 编码多语言文本描述
- **核心思路**: 基于 XLM-R-base，在 100 种语言的 2.5TB CommonCrawl 数据上预训练，12 层 Transformer，隐藏维度 768
- **设计动机**: XLM-R 的跨语言语义能力使模型能泛化到训练未见的语言

**模块2: 符号音乐编码器**

- **功能**: 编码乐谱（ABC 记谱法）和演奏信号（MIDI）
- **核心思路**: 基于 M3 自监督模型，将 ABC 按小节分段、MIDI 按消息分段，每段作为一个 patch。12 层编码器，隐藏维度 768，最多处理 512 个 patch（32768 字符）
- **设计动机**: M3 通过自监督学习获得了对符号音乐的深层理解能力

**模块3: 音频音乐编码器**

- **功能**: 编码音频录音
- **核心思路**: 12 层 Transformer（768 维），从头训练。利用冻结的 MERT-v1-95M 作为特征提取器，每 5 秒音频片段产生一个嵌入（跨 MERT 所有层和时间步取平均）。最多处理 128 个嵌入（640 秒音频）
- **设计动机**: 利用 MERT 预训练特征初始化音频表示，同时支持超长音频建模

**模块4: 多阶段对齐策略**

- **功能**: 解决无配对音乐数据时的多模态对齐问题
- **核心思路**: 四阶段策略——Stage 1: 文本↔符号编码器对齐；Stage 2: 冻结文本编码器，对齐音频编码器；Stage 3: 解冻文本编码器微调音频对齐；Stage 4: 再次冻结文本编码器修复符号对齐漂移
- **设计动机**: 交替冻结/解冻文本编码器，最小化模态间干扰，同时将所有模态映射到共享空间

### 损失函数/训练策略

- **对比损失**: InfoNCE 损失（公式如上）
- **训练配置**: 符号对齐训练 100 epoch，8×H800 GPU，4天；音频对齐训练 100 epoch，1天。学习率分别为 5e-5 和 1e-5，batch size 1024 和 2048
- **数据划分**: M4-RAG 99% 训练 + 1% 验证
- **优化**: 混合精度训练、AdamW 优化器、1000 步 warmup

**M4-RAG 数据集构建**:
- 标题过滤 → Google 搜索（每条取 top-10 结果）→ Qwen2.5-72B 生成标注（RAG）→ 质量过滤 → 后处理 → 多语言翻译
- 包含短标注（流派、标签）和长描述（背景、分析、描述、场景），跨 27 种语言、194 个国家

## 实验关键数据

### 主实验

英文文本-音乐检索（MRR 指标）:

| 模型 | WikiMT | MidiCaps | SDD | MC-R |
|------|--------|----------|-----|------|
| CLaMP | 0.2561 | 0.1236 | - | - |
| CLaMP 2 | 0.3438 | 0.2695 | - | - |
| CLAP | - | - | 0.1310 | 0.0657 |
| TTMR++ | - | - | 0.1437 | 0.1248 |
| **CLaMP 3_sa^c2** | **0.4498** | **0.2826** | 0.1612 | 0.0959 |
| **CLaMP 3_saas** | 0.3555 | 0.1798 | **0.1985** | **0.1177** |

多语言检索（ABC Notation, WikiMT-X Background）:

| 模型 | ru | fr | es | fi* | el* | kk* |
|------|-----|-----|-----|------|------|------|
| CLaMP 2 | 0.2668 | 0.2968 | 0.2934 | 0.2795 | 0.2410 | 0.2543 |
| CLaMP 3_sa^c2 | **0.3614** | **0.3949** | **0.3921** | **0.3524** | **0.3226** | **0.3397** |

(*标记为训练未见语言)

### 消融实验

跨模态涌现检索（WikiMT-X, MRR）:

| 模型 | S→P | S→A | P→S | P→A | A→S | A→P |
|------|------|------|------|------|------|------|
| CLaMP 2 | 0.5138 | - | 0.4480 | - | - | - |
| CLaMP 3_sa^c2 | 0.4547 | 0.0543 | **0.5293** | 0.0313 | 0.0492 | 0.0383 |
| CLaMP 3_saas | 0.3262 | **0.0578** | 0.3146 | **0.0397** | 0.0410 | 0.0303 |

两个模型变体对比了不同对齐顺序的效果：CLaMP 3_sa^c2 优化符号检索，CLaMP 3_saas 优化音频检索。

### 关键发现

1. CLaMP 3 在 WikiMT 符号检索上 MRR 从 0.3438（CLaMP 2）提升到 0.4498，提升 30.8%
2. 音频检索方面，CLaMP 3_saas 在 SDD 上 MRR 达 0.1985，大幅超越 TTMR++（0.1437）和 CLAP（0.1310）
3. **跨语言泛化**: CLaMP 3_saas 在训练未见的芬兰语上音频检索 MRR 达 0.1770，超过 CLAP 在英语上的表现（0.0598）
4. **跨模态涌现**: 无需配对训练数据，CLaMP 3 实现了符号-音频之间的检索（远超随机基线 0.0075）
5. M4-RAG 数据集的丰富标注使模型在 Description 和 Scene 等语义稀疏类型上也获得显著提升

## 亮点与洞察

- **以文本为桥梁**的多阶段对齐策略非常巧妙——无需任何乐谱-音频配对数据即可实现跨模态检索
- **M4-RAG 的 RAG 构建管道**是一个亮点：利用音乐标题的唯一标识性进行 web 搜索，再用 LLM 生成多维度标注，成本低且规模大（231 万对）
- **WikiMT-X 基准**填补了音乐领域首个同时包含文本-乐谱-音频三模态的评测空白
- 跨语言泛化能力令人印象深刻——在阿姆哈拉语（Amharic）等极低资源未见语言上也表现合理

## 局限与展望

1. **时序建模缺失**: 对比学习使用单一全局表示，无法捕捉音乐中的时序动态（如贝多芬第五交响曲中主题的发展变化）
2. **多语言评测依赖翻译**: 缺乏原生多语言基准，翻译质量差异引入噪声
3. **音频-符号对齐较弱**: 跨模态涌现检索虽远超随机基线，但音频相关方向的 MRR 仍较低（0.03-0.06），需要配对数据进行监督对齐
4. 数据集以西方音乐为主，对非西方音乐传统的覆盖可能不够深入

## 相关工作与启发

- **CLaMP/CLaMP 2** 是前身工作，一路从符号-文本对齐发展到多模态多语言统一
- **ImageBind** 的以文本为桥梁对齐多模态的思想直接启发了 CLaMP 3 的多阶段策略
- **MERT** 提供了高质量音频特征提取器，作为冻结特征提取器为音频编码器提供输入
- **RAG** 技术用于数据集构建而非推理，是一个新颖的应用方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ (首次统一所有音乐模态+多语言文本, 跨模态涌现检索)
- **实验充分度**: ⭐⭐⭐⭐⭐ (多模态/多语言/跨模态全面评测, 多个基准)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰, 图表丰富, 实验设置详尽)
- **价值**: ⭐⭐⭐⭐⭐ (开源模型+数据集+基准, 对MIR社区贡献巨大)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)
- [\[ACL 2025\] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors](atri_mitigating_multilingual_audio_text_retrieval_inconsistencies_by_reducing_da.md)
- [\[ACL 2025\] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](gigaspeech2_low_resource_asr.md)
- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [\[ACL 2025\] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)

</div>

<!-- RELATED:END -->
