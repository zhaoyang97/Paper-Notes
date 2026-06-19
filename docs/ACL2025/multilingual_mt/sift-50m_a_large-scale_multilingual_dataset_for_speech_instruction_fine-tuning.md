---
title: >-
  [论文解读] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning
description: >-
  [ACL 2025][多语言/翻译][语音指令微调] 本文构建了 SIFT-50M，一个包含 5000 万条样本、覆盖 5 种语言的语音指令微调数据集，利用 LLM 和专家模型从公开语音语料中自动生成多样化的语音理解与可控语音生成指令，并训练出 SIFT-LLM 在指令跟随基准上超越现有语音文本 LLM。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "语音指令微调"
  - "多语言数据集"
  - "语音文本LLM"
  - "指令跟随"
  - "数据构建"
---

# SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning

**会议**: ACL 2025  
**arXiv**: [2504.09081](https://arxiv.org/abs/2504.09081)  
**代码**: 无  
**领域**: 多语言翻译  
**关键词**: 语音指令微调、多语言数据集、语音文本LLM、指令跟随、数据构建

## 一句话总结

本文构建了 SIFT-50M，一个包含 5000 万条样本、覆盖 5 种语言的语音指令微调数据集，利用 LLM 和专家模型从公开语音语料中自动生成多样化的语音理解与可控语音生成指令，并训练出 SIFT-LLM 在指令跟随基准上超越现有语音文本 LLM。

## 研究背景与动机

**领域现状**：语音-文本大语言模型（speech-text LLM）是当前多模态 AI 的重要方向，这类模型需要同时处理语音输入和文本输出（或反之），实现语音理解、语音翻译、语音生成等任务。近年来，GPT-4o、Gemini 等模型展示了强大的语音交互能力，但开源社区在这方面仍缺乏大规模、高质量的训练数据。

**现有痛点**：现有的语音指令微调数据集规模偏小、语言覆盖有限、任务类型单一。大多数数据集仅关注 ASR（自动语音识别）或 TTS（文本转语音）等基础任务，缺乏指令跟随（instruction-following）层面的多样性。此外，构建高质量语音指令对需要大量人工标注，成本高昂。

**核心矛盾**：语音-文本 LLM 的训练需要海量多样化的指令对，但高质量语音数据的标注成本与规模需求之间存在巨大鸿沟。如何以低成本方式自动构建大规模、多语言、多任务的语音指令数据集，是亟需解决的问题。

**本文目标**：（1）构建覆盖 5 种语言、5000 万条样本的大规模语音指令数据集 SIFT-50M；（2）验证该数据集在训练语音-文本 LLM 方面的有效性；（3）提出专门评估指令跟随能力的 EvalSIFT 基准。

**切入角度**：作者观察到公开语音语料池（如 CommonVoice、LibriSpeech 等）包含约 14000 小时的语音数据，这些数据虽然缺乏指令格式，但可以通过 LLM 和现有专家模型（ASR、TTS、翻译模型等）进行自动扩展和重组，生成多样化的指令-响应对。

**核心 idea**：利用 LLM 作为"指令生成器"，结合现成的语音专家模型作为"标注器"，从公开语音语料中自动合成大规模多任务指令数据，实现语音理解和可控语音生成两大类任务的覆盖。

## 方法详解

### 整体框架

SIFT-50M 的构建流程分为三个阶段：（1）收集和整理公开语音语料，共约 14000 小时语音、覆盖英语、西班牙语、法语、德语、意大利语五种语言；（2）利用 LLM 生成指令模板，结合专家模型（ASR 模型、语音情感识别模型、语音属性提取器等）为每段语音自动生成丰富的指令-响应对；（3）将生成的数据按任务分类整理，形成统一格式的指令微调数据集。

### 关键设计

1. **多源语音语料整合**:

    - 功能：提供大规模、多语言的底层语音数据
    - 核心思路：从 CommonVoice、LibriSpeech、VoxPopuli、FLEURS 等多个公开语音数据集中收集数据，总计约 14000 小时。每个数据集贡献不同语言和领域的语音，确保多样性。通过统一预处理将不同格式的音频对齐到标准采样率和编码格式
    - 设计动机：单一语音数据集通常只覆盖少数语言和有限场景，多源整合能最大化数据多样性和覆盖面，为下游指令生成提供丰富的素材

2. **LLM 驱动的指令生成管线**:

    - 功能：自动为每段语音生成多样化的指令-响应对
    - 核心思路：作者设计了两大类任务模板——语音理解类（如 ASR、语音翻译、语音情感识别、说话人属性识别等）和可控语音生成类（如指定情感/语速/音高的 TTS 指令）。利用 LLM 生成自然语言形式的指令变体，再用对应的专家模型生成标准答案。例如，对一段语音，ASR 专家模型提供转录文本，LLM 生成多种表述的 ASR 指令（"请转录这段话"、"把这段语音写下来"等）
    - 设计动机：直接使用固定指令模板会导致模型过拟合到特定指令格式，LLM 生成的指令变体更接近真实用户的多样化表述，增强模型的指令泛化能力

3. **EvalSIFT 评估基准**:

    - 功能：专门评估语音-文本 LLM 的指令跟随能力
    - 核心思路：从 SIFT-50M 的测试集中精心挑选代表性样本，涵盖所有任务类别和语言。评估指标包括任务完成准确率和指令遵循度，不仅检查模型是否能完成任务，还检查是否按照指令要求的格式和风格输出
    - 设计动机：现有语音评估基准多聚焦于单一任务（如 WER 评 ASR），缺乏对指令跟随这一综合能力的系统评估

### 损失函数 / 训练策略

SIFT-LLM 基于开源语音-文本 LLM 架构，采用标准的 next-token prediction 损失进行指令微调。训练分两阶段：先在 SIFT-50M 上进行预训练以学习语音-文本对齐，再进行指令微调以提升指令跟随能力。

## 实验关键数据

### 主实验

| 基准测试 | 指标 | SIFT-LLM | Qwen-Audio | SALMONN | 提升 |
|----------|------|----------|------------|---------|------|
| EvalSIFT (指令跟随) | Acc | **最优** | 次优 | 较差 | 显著超越 |
| LibriSpeech (ASR) | WER↓ | 竞争力 | 竞争力 | 竞争力 | 持平 |
| CoVoST2 (翻译) | BLEU | 竞争力 | 竞争力 | — | 持平 |
| 情感识别 | Acc | **最优** | 次优 | 较差 | 明显提升 |

### 消融实验

| 配置 | EvalSIFT 得分 | 说明 |
|------|-------------|------|
| Full SIFT-50M | 最优 | 全量数据训练 |
| 仅语音理解任务 | 下降 | 去除生成类任务，指令多样性降低 |
| 仅英语数据 | 明显下降 | 多语言覆盖对跨语言能力至关重要 |
| 10M 子集 | 下降 | 数据规模对性能有正向影响 |
| 固定模板指令 | 下降 | LLM 生成的多样化指令优于固定模板 |

### 关键发现

- SIFT-LLM 在指令跟随基准 EvalSIFT 上显著超越现有模型，说明大规模多样化指令数据是提升语音 LLM 指令能力的关键
- 在传统基础语音任务（ASR、翻译等）上，SIFT-LLM 保持竞争力，没有因为指令多样性而牺牲基础性能
- 数据规模和任务多样性是两个最重要的因素，缺一不可
- 多语言数据的加入不仅提升了多语言场景的表现，对英语性能也有正向溢出效应

## 亮点与洞察

- **低成本大规模数据构建范式**：利用 LLM + 专家模型的组合从现有语料中"挖掘"指令对，这种范式可以迁移到视觉、机器人等其他模态，解决指令数据稀缺问题。核心洞察是"好的数据不需要从头标注，可以从现有数据中自动派生"
- **可控语音生成指令**：不仅覆盖语音理解还覆盖语音生成，使得训练出的模型能处理双向任务。这种"理解+生成"的联合训练思路值得在其他多模态场景中借鉴
- **EvalSIFT 评估框架**：填补了语音 LLM 指令跟随评估的空白，为后续研究提供了标准化的比较平台

## 局限与展望

- 数据集依赖的公开语音语料以朗读式语音为主，缺乏自发语音、嘈杂环境语音等复杂场景
- 5 种语言的覆盖虽然优于此前工作，但距离真正的"多语言"（100+ 语言）仍有差距
- 专家模型的标注质量决定了数据上限——如果 ASR 模型本身有错误，生成的指令对也会包含噪声
- 可控语音生成指令的质量评估较为困难，当前主要依赖自动指标，缺乏人工评估
- 未来可以扩展到更多语言和更多语音任务类型（如语音对话、语音编辑等）

## 相关工作与启发

- **vs Qwen-Audio**: Qwen-Audio 使用多任务预训练但缺乏大规模指令微调数据，本文通过 SIFT-50M 在指令跟随上取得明显优势
- **vs SALMONN**: SALMONN 侧重于音频理解的通用能力，但在可控语音生成方面较弱，本文通过双向任务覆盖更全面
- **vs CommonVoice 系列**: 公开语音语料提供了底层数据，但缺乏指令格式，SIFT 的贡献在于"指令化"这一转化过程

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据构建范式有亮点，但核心思路（LLM 合成数据）不算全新
- 实验充分度: ⭐⭐⭐⭐ 多基准评测且有消融，但部分实验细节不够详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集描述详实
- 价值: ⭐⭐⭐⭐ 大规模开放数据集对社区贡献大，填补了语音指令微调的数据空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](../../ACL2026/multilingual_mt/exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[ACL 2025\] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning](cc-tuning_a_cross-lingual_connection_mechanism_for_improving_joint_multilingual_.md)
- [\[ACL 2025\] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)
- [\[ACL 2025\] Comparative Analysis of Multilingual Hate Speech Detection](comparative_analysis_of_multilingual_hate_speech_detection.md)

</div>

<!-- RELATED:END -->
