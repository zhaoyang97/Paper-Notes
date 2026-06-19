---
title: >-
  [论文解读] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation
description: >-
  [ACL 2025][信息检索/RAG][RAG] 本文提出 HoH，首个专门评估过时信息对 RAG 系统影响的大规模动态基准，包含 96,124 个 QA 对和 219,463 篇文档，揭示了过时信息对 RAG 性能和安全性的严重危害。 RAG（检索增强生成）被广泛用于解决 LLM 的知识过时问题，通过从外部知识库检索最新…
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "RAG"
  - "过时信息"
  - "动态基准"
  - "时间敏感QA"
  - "知识更新"
---

# HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2503.04800](https://arxiv.org/abs/2503.04800)  
**代码**: [GitHub](https://github.com/0russwest0/HoH)  
**领域**: NLP / RAG  
**关键词**: RAG, 过时信息, 动态基准, 时间敏感QA, 知识更新

## 一句话总结

本文提出 HoH，首个专门评估过时信息对 RAG 系统影响的大规模动态基准，包含 96,124 个 QA 对和 219,463 篇文档，揭示了过时信息对 RAG 性能和安全性的严重危害。

## 研究背景与动机

RAG（检索增强生成）被广泛用于解决 LLM 的知识过时问题，通过从外部知识库检索最新信息来辅助生成。然而现有研究主要关注如何获取最新信息，却忽略了一个关键挑战：**知识库中过时信息的普遍存在**。

举个直观的例子：当查询"现任美国总统是谁"时，RAG 系统可能同时检索到当前和过时的信息（如上一任总统的相关文档），导致模型困惑甚至产生错误回答。这在搜索引擎场景中尤为常见，因为历史内容会被缓存和再分发。

现有的动态 QA 基准存在几个不足：(1) 依赖人工标注、规模有限；(2) 缺乏对过时信息的专门标注；(3) 无法维持持续更新。HoH 的提出旨在弥补这些空白，首次系统性地研究过时信息对 RAG 的危害。

## 方法详解

### 整体框架

HoH 基准由两个核心组件构成：
- **HoH-QA**: 动态 QA 数据集，追踪事实随时间的变化
- **HoH-SearchEngine**: 模拟真实搜索场景的模拟搜索引擎，同时维护当前和历史文档

### 关键设计

#### 1. 事实变化提取 (Factual Change Extraction)

**功能**: 从 Wikipedia 不同时间快照中提取事实性变化。

**核心思路**: 采用两阶段方法——先用 Myers Diff 算法在句子级别检测修改，再在字符级别识别具体差异，最后用语义筛选模型过滤非事实性变化。

**设计动机**: 以往方法（如 EvolvingQA、GrowOVER）仅在句子级别提取变化，但许多句子修改并非事实性变更（如语法修正、措辞调整）。通过引入字符级 diff + 启发式过滤 + 语义模型筛选，可以大幅提高提取质量。

语义筛选模型基于 Qwen2.5-0.5B 微调，使用 2000 个人工标注的句子对训练，达到 96.8% 准确率和 95.1% F1。

#### 2. QA 生成与自动更新

**功能**: 从提取的事实变化自动生成时间敏感的 QA 对，并维持数据集的持续演进。

**核心思路**: 对于新发现的事实变化，用 LLM 生成包含时间维度的问题，并同时生成当前答案和过时答案。对于已有事实的后续变化，利用 LLM 更新现有问题的答案，形成答案演化链。

**设计动机**: 现实世界中的知识是持续变化的——同一事实可能经历多次更新。通过保留完整的答案历史，可以更准确地评估 RAG 在处理多版本信息时的表现。

#### 3. HoH-SearchEngine

**功能**: 基于 Elasticsearch 构建模拟搜索引擎，同时维护当前和历史版本的文档。

**核心思路**: 在默认 BM25 排名基础上引入高斯衰减函数，对过时信息进行时间惩罚，模拟真实搜索引擎的时间偏好。

**设计动机**: 现实搜索引擎中过时信息是不可避免的——旧文档被缓存、引用、转发。单纯的 QA 评测无法反映这种复杂性，模拟搜索引擎能更真实地体现 RAG 面临的实际挑战。

### 评估框架

将文档/段落分为三类：
- **Relevant (R)**: 包含正确答案的相关段落
- **Outdated (O)**: 曾经相关但已过时的段落
- **Distracting (D)**: 语义相似但不含正确答案的干扰段落

评分体系：Perfect (+1) / Missing (0) / Harmful (-1)，惩罚错误信息而非遗漏。

## 实验关键数据

### 主实验：RAG 端到端性能

| 模型 | 总分 | R only | R+O | O only | None |
|------|------|--------|-----|--------|------|
| Llama-70B | 51.7% | 高 | 显著下降 | 危险 | 低但安全 |
| Llama-8B | 40.0% | 中 | 大幅下降 | 极危险 | 低 |
| Qwen-7B | 29.9% | 中 | 严重下降 | 比随机更差 | 低 |

### 检索模块分析

| 检索方法 | R@5 (R) | R@5 (O) | 说明 |
|----------|---------|---------|------|
| 无时间衰减 | 0.8707 | 0.8837 | 过时信息命中率更高 |
| 高斯衰减 | 0.7023 | 0.4950 | R下降17%才换来O降低 |
| BGE-M3 | 0.7312 | 0.5507 | 仍有55%概率检索到过时信息 |

### 生成模块分析（关键发现）

| 条件 | Llama-70B得分 | Llama-8B得分 | Qwen-7B得分 |
|------|--------------|-------------|-------------|
| R×1, D×0 | 89.12 | 85.70 | 76.31 |
| R×1, D×6 | 87.98 | 80.94 | 70.24 |
| R×1, O×1 (Score↓) | 72.95 | 50.93 | 33.93 |
| R×1, O×1, D×5 (Date↓) | 75.06 | 32.46 | **-2.77** |

### 关键发现

1. **过时信息比无信息更危险**: 仅检索到过时信息时，模型会产生过度自信的错误回答；而没有检索到任何相关信息时，模型反而能表现出适当的不确定性
2. **一条过时信息的危害远超六条干扰信息**: 引入1条过时段落导致得分下降超过24%，而6条干扰段落仅导致不到2%的下降
3. **排序策略影响巨大**: Qwen-7B 在特定排序下得分可达61.54%或低至-2.77%，差距超过60个百分点
4. **时间感知能力与 RAG 性能强相关**: 同时具备 Current Awareness 和 Outdated Awareness 的模型表现最佳（93.24 vs 24.70）

## 亮点与洞察

- **首次量化过时信息的危害**: 不只是"降低准确率"，更会导致模型生成有害的错误信息
- **"识别过时≠不使用过时"**: 即使模型能识别出信息过时，仍可能被其误导（Table 6 中 A_O only 的 harmful 率仍高于 A_C only），说明需要额外的对齐训练
- **diff 算法 + LLM 的数据构建方案**: 兼顾了效率和质量，比纯 LLM 方案更可靠
- **规模优势明显**: 96K QA对 + 219K文档，远超 RealtimeQA (2.3K) 和 FreshQA (600)

## 局限与展望

1. Wikipedia 快照更新频率固定（月更），无法反映快速变化领域（如股市）的实时性
2. QA 主要基于单篇文章生成，难以评测需要多源信息综合的复杂推理
3. 模拟搜索引擎的过时信息来源单一（同一文章的历史版本），现实中过时信息来源更多样
4. 未探索如何实际改善 RAG 对过时信息的鲁棒性（本文主要是诊断问题，非解决方案）

## 相关工作与启发

- **Time-Sensitive QA**: RealtimeQA、FreshQA 等依赖人工，HoH 实现了自动化+持续更新
- **CLARK-News**: 是唯一同时记录答案历史变化的先前工作，但规模小且依赖人工
- **RAG 评测**: 现有 RAG 基准（如 CRAG）关注检索质量，HoH 首次关注过时信息的负面影响
- **启发**: 未来的 RAG 系统应在检索和生成两端都加入时间感知机制，而非仅依赖简单的时间排序

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统性研究过时信息对 RAG 的影响，填补了重要空白
- **实验充分度**: ⭐⭐⭐⭐⭐ — 实验设计全面，从检索到生成到时间感知力层层递进，分析深入
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，例子直观，Table 1 的对比表很有价值
- **价值**: ⭐⭐⭐⭐⭐ — 提供了社区急需的基准和工具，发现具有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ICCV 2025\] OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](../../ICCV2025/information_retrieval/ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)

</div>

<!-- RELATED:END -->
