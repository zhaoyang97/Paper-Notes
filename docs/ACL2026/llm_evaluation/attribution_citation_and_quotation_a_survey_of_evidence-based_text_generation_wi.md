---
title: >-
  [论文解读] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models
description: >-
  [ACL 2026][LLM评测][证据基础文本生成] 本文系统综述了 134 篇关于 LLM 证据基础文本生成的论文，首次提出统一分类学（归因方式 × 引用特征 × 任务），分析了 300 个评估指标并归纳为七大维度六种方法，为该碎片化领域提供了全景式参考框架。 领域现状：LLM 面临幻觉生成和知识局限等可信性挑战…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "证据基础文本生成"
  - "引用归因"
  - "LLM可信性"
  - "评估框架"
  - "RAG"
---

# Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models

**会议**: ACL 2026  
**arXiv**: [2508.15396](https://arxiv.org/abs/2508.15396)  
**代码**: [https://github.com/faerber-lab/AttributeCiteQuote](https://github.com/faerber-lab/AttributeCiteQuote)  
**领域**: 综述/NLP  
**关键词**: 证据基础文本生成, 引用归因, LLM可信性, 评估框架, RAG

## 一句话总结

本文系统综述了 134 篇关于 LLM 证据基础文本生成的论文，首次提出统一分类学（归因方式 × 引用特征 × 任务），分析了 300 个评估指标并归纳为七大维度六种方法，为该碎片化领域提供了全景式参考框架。

## 研究背景与动机

**领域现状**：LLM 面临幻觉生成和知识局限等可信性挑战，越来越多研究关注"证据基础文本生成"——让 LLM 输出可追溯到支撑证据。但该领域高度碎片化：有的叫"引用"（citation, 75% 论文使用），有的叫"归因"（attribution, 62%），有的叫"引述"（quotation, 13%），且各自评估实践孤立。

**现有痛点**：(1) 缺乏统一术语和分类体系，研究者难以定位自己的工作；(2) 评估标准不一致——300 个指标但仅 2 个框架（ALCE、G-Eval）被多篇论文复用；(3) RAG 虽然流行但仅是七种相关方法之一，过度聚焦 RAG 会遗漏其他重要方法。

**核心矛盾**：快速增长的研究兴趣（2024 年论文数是 2023 年的 3.4 倍，75% 论文发表在 2023 年之后）vs. 缺乏统一视角来整合和比较不同方法。

**本文目标**：提供第一个专门针对 LLM 证据基础文本生成的系统性综述，建立统一分类学，分析评估实践，识别研究趋势和未来方向。

**切入角度**：采用 PRISMA 协议进行系统映射研究，从 805 篇去重论文中筛选出 134 篇相关论文，使用多面分类方法构建分类学。

**核心 idea**：将"引用"、"归因"和"引述"统一为"证据基础文本生成"范式，通过三维分类学和七维评估框架为碎片化领域提供系统化视角。

## 方法详解

### 整体框架

本文按 PRISMA 协议做系统映射研究：从 805 篇去重论文中筛出 134 篇相关工作，再用多面（faceted）分类方法逐篇编码。核心产物是一套三维独立分类学——把"归因方式（内容如何关联到证据）× 引用特征（证据以何种形态呈现）× 任务（应用场景）"三个正交维度组合起来，任何一篇证据基础文本生成工作都能被定位到这个立方体里；在此之上再叠加一层 LLM 集成方式（训练 vs. 提示）的横切视角，回答"模型用什么手段获得归因能力"。

### 关键设计

**1. 归因方式：参数式 vs. 非参数式的二分再细分。** 这一维刻画 LLM 把生成内容关联到支撑证据的根本路径，是分类学最核心的一面。**参数式**（25 篇）让证据进入模型权重，再细分为纯 LLM（直接利用既有能力，占参数式的 72%）、模型中心（改架构/改训练）、数据中心（策划/增强数据）三支；**非参数式**（126 篇）让证据停在权重之外，按检索发生的时机切成后检索（58%，RAG 为代表）、后生成（18%，先生成再回头找证据）、生成中（4%，模型动态判断当下是否需要检索）、上下文内（20%，用户直接把证据塞进 prompt）。这套三分法比此前简单的"RAG / 非 RAG"更精细，并直接暴露出领域的失衡：参数式整体被严重忽视，而生成中归因（如 Self-RAG）虽仅占 4%，却代表检索与生成更紧密耦合的前沿方向。

**2. 引用特征：五个子面刻画证据的"长相"。** 同样的证据可以有完全不同的呈现方式，本维用五个子面把它拆开——引用模态（文本 96%，图/表/视觉几乎空白）、证据级别（文档级 43%、段落级 40%、句子级 12%、token 级 2%）、引用样式（行内引用 62%，以及引用报告、段落展示、叙述性引用、高亮梯度、引述等）、可见性（最终回复 91% vs. 中间文本）、频率（多重引用 64% vs. 单一引用）。把这些子面并列后能一眼看出两个结构性空白：非文本模态严重未开发（仅 4%），而句子级/token 级这类细粒度证据虽占比小却增长最快，提示更精细的可追溯性正在成为趋势。

**3. 任务：六类应用场景的版图。** 第三维映射工作落在什么任务上，发现 QA 与接地文本生成是两大主导任务，摘要、事实验证居中，引用文本生成与相关工作生成属新兴任务。这一面的价值在于揭示评估范式的路径依赖：现有指标几乎都是围着 QA 长出来的，搬到新兴任务（如引用文本生成更需要对"为何选这条引用"的推理做评估）时未必适用。

此外，分类学叠加了一层 **LLM 集成方式** 的横切视角：训练（45% 论文，以监督微调改善归因行为为主，预训练较少）与提示（78% 论文，以零/少样本为主，并发展出 chain-of-citation、chain-of-quote、conflict-aware 等针对引用行为的专用提示策略），回答的是"模型靠什么手段获得归因能力"这一与上述三维正交的问题。

## 实验关键数据

### 文献分析/覆盖范围

### 评估指标体系

**300 个指标按七大评估维度分类**

| 评估维度 | 何时使用 | 主要方法 | 代表指标（复用次数） |
|---------|---------|---------|-------------------|
| 归因 (Attribution) | 无标注证据时 | NLI 为主 | Citation NLI P/R/F1 (33/33/16), Auto-AIS (11), FActScore (7) |
| 引用 (Citation) | 有标注证据时 | 检索为主 | Citation Retrieval P/R/F1 (6/6/5), Citation Accuracy (2) |
| 正确性 (Correctness) | 始终需要 | 词汇重叠/NLI | Exact Match (12), BLEU-N (5), Claim Recall (17) |
| 语言质量 | 模型被修改时 | LLM-as-Judge | G-Eval Fluency (4), MAUVE (21), Perplexity (4) |
| 保留度 | 后生成归因时 | 词汇重叠 | Preservation-Levenshtein (3), F1-AP (2) |
| 相关性 | 用户场景 | LLM-as-Judge | G-Eval Relevance (3), RAGAS (2) |
| 检索 | 非参数归因时 | 检索指标 | P@k (4), R@k (4), MRR (3) |

### 评估指导原则

| 维度类别 | 何时评估 | 说明 |
|---------|---------|------|
| **核心维度** | 归因或引用 + 正确性 | 始终应评估正确性；归因和引用根据证据可用性二选一 |
| **上下文维度** | 语言质量、保留度、相关性、检索 | 取决于任务设计和系统架构 |

### 关键发现

- 仅 2 个框架（ALCE、G-Eval）和 2 个基准被多篇论文复用，评估标准化严重不足
- 134 篇论文中识别出 19 个框架、11 个基准和 231 个数据集
- 文本在引用模态中占 96%，多模态证据几乎空白
- 参数式归因虽然对理解模型内部知识和数据溯源至关重要，但严重被忽视
- 人工评估在正确性维度仍占主导，反映了自动指标在捕捉语义错误方面的局限

## 亮点与洞察

- 将"引用"、"归因"和"引述"统一为"证据基础文本生成"是重要的概念贡献，消除了长期的术语混乱
- 七维评估指南（Table 1）为实践者提供了清晰的指标选择建议——核心维度 vs. 上下文维度的区分极为实用
- 参数式归因的三分法（纯 LLM / 模型中心 / 数据中心）比此前的二分法更精细
- 识别出生成中归因（in-generation）作为有前景但被低估的方向——仅占 4% 但代表了更紧密集成检索和生成的趋势
- 指出了引用行为可能存在类似人类作者的偏差，呼吁研究 LLM 引用推理的可解释性

## 局限与展望

- 单一搜索字符串可能遗漏部分相关研究（敏感性分析显示仅 4% 额外发现）
- 仅覆盖英文论文，可能低估非英语研究
- 人工筛选和分类不可避免引入一定主观性
- **四大未来方向**：(1) 参数式和混合归因的深入研究；(2) 标准化评估框架（当前 300 指标仅 2 框架被复用）；(3) 可解释的引用推理——理解 LLM 为何选择特定来源；(4) 多模态证据支持——从 96% 文本向图表、表格、图像扩展

## 相关工作与启发

- **vs Li et al. (2023a)**: 唯一先前相关综述，但已严重过时（75%+ 论文发表在其之后），且未覆盖完整范式
- **vs Huang & Chang (2024)**: 立场论文仅强调引用重要性，未系统综述
- **vs RAG surveys**: RAG 综述仅覆盖后检索这一种方法，本文覆盖七种归因方式
- **vs 幻觉/接地综述**: 聚焦不同侧面，本文专注于证据生成而非检测

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次对证据基础文本生成进行全面统一分类，三维分类学设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 134 篇论文、300 个指标、19 框架、231 数据集、11 基准的覆盖范围极全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，多维度分析平衡，每节附有精炼的 "Takeaways" 总结
- 价值: ⭐⭐⭐⭐⭐ 对快速增长但碎片化领域的全景梳理，对研究者和实践者都有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ICML 2026\] Authority, Truth, and Citation Bias: A Large-Scale Multi-Domain Benchmark for Studying Epistemic Susceptibility in Large Language Models](../../ICML2026/llm_evaluation/authority_truth_and_citation_bias_a_large-scale_multi-domain_benchmark_for_study.md)

</div>

<!-- RELATED:END -->
