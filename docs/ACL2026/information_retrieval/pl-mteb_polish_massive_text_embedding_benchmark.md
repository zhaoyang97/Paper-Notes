---
title: >-
  [论文解读] PL-MTEB: Polish Massive Text Embedding Benchmark
description: >-
  [ACL2026 Findings][信息检索/RAG][Polish NLP] PL-MTEB 为波兰语文本嵌入构建了覆盖分类、聚类、句对分类、检索和语义相似度的 30 任务评测集，并系统评测 30 个波兰语和多语言 embedding 模型，显示大模型整体领先但任务类型、训练数据泄漏和模型规模都会显著影响结论。
tags:
  - "ACL2026 Findings"
  - "信息检索/RAG"
  - "Polish NLP"
  - "Text Embedding"
  - "MTEB"
  - "信息检索"
  - "基准评测"
---

# PL-MTEB: Polish Massive Text Embedding Benchmark

**会议**: ACL2026 Findings  
**arXiv**: [2405.10138](https://arxiv.org/abs/2405.10138)  
**代码**: https://github.com/rafalposwiata/pl-mteb  
**领域**: 信息检索/RAG  
**关键词**: Polish NLP, Text Embedding, MTEB, 信息检索, 基准评测

## 一句话总结
PL-MTEB 为波兰语文本嵌入构建了覆盖分类、聚类、句对分类、检索和语义相似度的 30 任务评测集，并系统评测 30 个波兰语和多语言 embedding 模型，显示大模型整体领先但任务类型、训练数据泄漏和模型规模都会显著影响结论。

## 研究背景与动机
**领域现状**：文本嵌入是检索、聚类、分类、问答和语义匹配系统的基础组件。MTEB 为英文和部分多语言任务提供了统一评测框架，近年来也出现了中文、法语、波斯语、荷兰语、俄语、越南语、土耳其语、阿拉伯语、非洲语言等语言特定扩展。

**现有痛点**：波兰语模型长期缺少覆盖任务类型足够广的 embedding benchmark。已有波兰语评测通常只覆盖单一任务或少数任务，例如情感分类、语义相关、BEIR-PL 检索等，无法回答一个模型在不同应用场景下是否稳定。

**核心矛盾**：多语言模型可能在波兰语上可用，但它们的表现受训练语料、任务类型和模型规模影响很大。如果没有统一、公开、任务多样且标注质量可控的基准，用户很难为实际系统选模型，也难以公平比较波兰语专用模型和通用多语言模型。

**本文目标**：作者希望构建一个波兰语版 MTEB，既复用已有公开波兰语任务，又补足缺少的任务类型，尤其是聚类；同时收集 30 个公开 embedding 模型的结果，分析任务类型、模型大小和 zero-shot 覆盖对评测结论的影响。

**切入角度**：论文不是只翻译英文任务，而是把已有波兰语数据、BEIR-PL 检索任务、KLEJ/LEPISZCZE 相关数据，以及新构造的 PLSC 和 Wikinews-PL 聚类数据整合到 MTEB 框架中，并公开代码、数据与 leaderboard。

**核心 idea**：用 MTEB 的统一评测接口为波兰语搭建一个 30 任务、多类型、可复现的 embedding benchmark，同时把训练数据相似性标成 zero-shot 列，提醒读者区分真实泛化和训练集相似性收益。

## 方法详解
PL-MTEB 的方法重点在基准构建和评测协议，而不是提出新 embedding 模型。作者做了三件事：定义任务集合，清洗并构造新数据，评测模型并按任务类型/模型规模分析。

### 整体框架
基准包含 5 类任务共 30 个子任务。分类任务用少样本 logistic regression 评估 embedding 的可线性分性；聚类任务用 mini-batch k-means 和 v-measure；句对分类用相似度阈值下的 average precision；检索任务用 nDCG@10；STS 用 cosine similarity 下的 Spearman 相关。

任务来源分三组。第一组是已有 MTEB 或多语言 MTEB 中可抽取波兰语子任务的任务，主要包括 BEIR-PL 检索任务。第二组是作者基于已有波兰语公开数据新加入的任务，多来自 KLEJ 等人工标注资源。第三组是作者新构造的两个数据集 PLSC 和 Wikinews-PL，并由它们生成四个聚类任务。

评测模型包括 30 个公开 dense embedding 模型，覆盖小模型、base、large 和 1B 以上模型；既有 multilingual E5、SBERT、Arctic-Embed、Qwen3-Embedding、BGE-Multilingual-Gemma2，也有 MMLW、Stella-PL、Silver Retriever 等波兰语相关模型。每个模型都尽量按开发者推荐配置运行，并记录它在多少比例任务上可视为 zero-shot。

### 关键设计

**1. 五类任务的统一协议：把分类、聚类、句对、检索、STS 收进同一套评测接口**

embedding 模型常常在一类任务上很强、换一类就拉胯，只看一个总平均分会掩盖这种偏科。PL-MTEB 因此对五类任务各配一个轻量、可复现的评测器：分类任务对每个类别只取 8 个样本训练 logistic regression、重复 10 次取平均，看 embedding 是否线性可分；聚类用 mini-batch k-means、令簇数 $k$ 等于标签数、同样重复 10 次，用 v-measure 打分；检索以 nDCG@10 为主指标；STS 用 cosine 相似度下的 Spearman 相关；句对分类用 cosine 相似度阈值下的 average precision。五个协议共用同一份 embedding 输出，因此结果之间可以横向对照——读者既能看 30 任务总均值 Avg(30)，也能看按任务类型先平均再汇总的 Avg(by type)，从而按"我要做检索还是聚类"来选模型，而不是被单一榜单误导。

**2. PLSC 与 Wikinews-PL 聚类补强：用两套新数据填上波兰语最缺的聚类维度**

已有波兰语评测大多围着分类、检索、STS 转，聚类任务几乎是空白，而聚类恰恰最依赖 embedding 的全局语义结构、最不吃监督分类器或检索训练数据的便宜，是检验"表示空间是否稳"的好探针。作者为此新建两套数据：PLSC 取自 Polish Library of Science 元数据，约 160K 条波兰语论文记录，按 8 个科学领域和 44 个学科形成层级标签；Wikinews-PL 取自波兰语 Wikinews，按政治、经济、灾害、文化娱乐、科学、法律犯罪、体育、社会、技术等类别标注。两套数据各自构造 S2S（标题级）和 P2P（段落级）聚类任务，每个任务截到 2,048 条以符合 MMTEB 的效率假设。补上聚类后，benchmark 才能把"监督任务刷得高但语义空间松散"的模型暴露出来。

**3. 数据质量与 zero-shot 标注：先清洗去泄漏，再把"训练数据相似性"摊到明面上**

embedding benchmark 极易被训练数据污染——检索任务和常用 STS 数据尤其容易在模型训练语料里见过，高分未必是真泛化。作者一方面做硬清洗：删掉空文本和少于 3 个词的样本、核对标签与分数、删除标签冲突或分数差超过 0.5 的近重复、在 split 层面去重并验证 test-train leakage；另一方面在评测表里给每个模型标一列 zero-shot 比例，即该模型训练数据中不含相似任务的任务占比。这一列让读者读榜单时多一道防线：同样一个高检索分，zero-shot 比例只有 80 的模型就要怀疑它是否吃了相似训练数据的红利，而不是直接当成泛化能力。

### 损失函数 / 训练策略
PL-MTEB 本身不训练新模型，没有统一训练损失。评测时只训练轻量下游评估器：分类任务训练 logistic regression；聚类任务训练 k-means；其余任务直接使用 embedding 相似度或检索排序。所有模型按原始发布方式加载，评测代码基于 MTEB 框架，结果和数据公开在 GitHub 与 Hugging Face。

## 实验关键数据

### 主实验
基准共 30 个任务，其中分类 7 个、聚类 5 个、句对分类 4 个、检索 11 个、STS 3 个。检索任务覆盖 ArguAna-PL、DBPedia-PLHardNeg、FiQA-PL、HotpotQA-PLHardNeg、MSMARCO-PLHardNeg、NFCorpus-PL、NQ-PLHardNeg、Quora-PLHardNeg、SCIDOCS-PL、SciFact-PL、TRECCOVID-PL 等。

| 任务类型 | 任务数 | 主指标 | 代表任务 / 数据来源 | 设计要点 |
|------|------|------|------|------|
| Classification | 7 | Accuracy | CBD、PolEmo2.0、AllegroReviews、PAC、MassiveIntent/Scenario | 每类 8-shot logistic regression，重复 10 次 |
| Clustering | 5 | V-measure | EightTags、PLSC、Wikinews-PL | mini-batch k-means，层级任务取层级平均 |
| Pair Classification | 4 | Cosine AP | SICK-E-PL、CDSC-E、PSC、PPC | 评估句对关系的相似度可分性 |
| Retrieval | 11 | nDCG@10 | BEIR-PL 系列任务 | 大多为 query-corpus 检索，部分 HardNeg 限制语料规模 |
| STS | 3 | Cosine Spearman | SICK-R-PL、CDSC-R、STSBenchmarkMultilingual | 测语义相似度排序相关性 |

| 模型 | 参数量 | Zero-shot 比例 | Classification | Clustering | PairClass | Retrieval | STS | Avg(30) | Avg(by type) |
|------|--------|----------------|----------------|------------|-----------|-----------|-----|---------|--------------|
| mmlw-roberta-base | 124M | 96 | 62.53 | 48.00 | 88.16 | 53.60 | 85.20 | 62.52 | 67.50 |
| multilingual-e5-base | 278M | 90 | 55.36 | 44.10 | 82.08 | 47.63 | 79.13 | 56.59 | 61.66 |
| mmlw-retrieval-roberta-large | 435M | 93 | 63.90 | 45.18 | 88.48 | 57.23 | 84.71 | 63.69 | 67.90 |
| Qwen3-Embedding-0.6B | 596M | 90 | 69.66 | 56.65 | 81.31 | 48.59 | 78.45 | 62.20 | 66.93 |
| stella-pl | 1.5B | 80 | 66.94 | 38.08 | 89.20 | 60.82 | 86.87 | 64.85 | 68.38 |
| stella-pl-retrieval-8k | 1.5B | 80 | 68.14 | 35.42 | 89.56 | 61.59 | 86.56 | 64.98 | 68.25 |
| Qwen3-Embedding-4B | 4.0B | 90 | 79.30 | 59.90 | 86.68 | 56.65 | 85.55 | 69.37 | 73.62 |
| Qwen3-Embedding-8B | 7.6B | 90 | 79.87 | 58.64 | 87.61 | 59.21 | 86.72 | 70.47 | 74.41 |
| BGE-Multilingual-Gemma2 | 9.2B | 83 | 77.77 | 58.15 | 89.75 | 58.93 | 83.97 | 69.81 | 73.71 |

### 消融实验
这篇是 benchmark 论文，没有传统模型模块消融；论文的分析维度是任务类型、模型规模和训练数据相似性。

| 分析维度 | 观察 | 启发 |
|------|------|------|
| 任务类型赢家 | Qwen3-Embedding-8B 分类最好，Qwen3-Embedding-4B 聚类最好，BGE-Multilingual-Gemma2 句对分类最好，stella-pl-retrieval-8k 检索最好，stella-pl STS 最好 | 没有一个模型统治所有任务，平均分不能替代任务级选择 |
| 模型规模 | 1B 以上模型整体最高，Qwen3-Embedding-8B Avg(30)=70.47 领先 | 大模型优势明显，但不是所有任务都随规模单调提升 |
| 小模型 | mmlw-roberta-base 在 <150M 组显著领先，Avg(30)=62.52 | 波兰语专门训练的小模型在资源受限场景很有竞争力 |
| base 模型组 | snowflake-arctic-embed-m-v2.0 Avg(30)=57.06，multilingual-e5-base Avg(by type)=61.66 | 中等规模多语言模型没有明显统治者，需看任务类型 |
| 检索任务 | stella-pl-retrieval-8k 与 stella-pl 最强，但 zero-shot 比例只有 80 | 高检索分可能受相似检索训练数据影响，要结合 zero-shot 列解读 |
| P2P vs S2S 聚类 | 新建 PLSC/Wikinews 任务中 P2P 通常优于 S2S | 更长文本包含更多聚类信息，标题级 embedding 更难 |

### 关键发现
- Qwen3-Embedding-8B 是整体最强模型，Avg(30)=70.47、Avg(by type)=74.41，但它只在分类上最突出，并非每类任务都第一。
- BGE-Multilingual-Gemma2 的句对分类平均分最高，说明超大多语言模型在语义匹配类任务上仍很强。
- 波兰语专用的 stella-pl-retrieval-8k 在检索上最好，nDCG@10 类平均 61.59，但其训练数据与检索任务相似度较高，解读时需谨慎。
- mmlw-roberta-base 只有 124M 参数，却在小模型组拿到 Avg(30)=62.52，甚至超过不少 base/large 多语言模型，说明语言专用蒸馏仍然很有价值。
- benchmark 层面最大的贡献不是某个模型排名，而是把波兰语 embedding 评测从零散任务扩展到 5 类 30 任务，并加入数据质量和 zero-shot 视角。

## 亮点与洞察
- PL-MTEB 很适合作为“实际选 embedding 模型”的工具，而不仅是论文 leaderboard。因为它同时报告任务类型均值和总体均值，用户可以按检索、聚类或分类需求选择不同模型。
- 新增 PLSC 和 Wikinews-PL 聚类任务很关键。很多 embedding 模型在监督检索或 STS 上表现好，但聚类测试更能暴露语义空间结构是否稳定。
- zero-shot 列是一个非常好的评测习惯。多语言 embedding 模型训练语料复杂，单纯高分可能来自相似数据见过；把相似训练数据比例显式列出，能减少误读。
- 结果提醒我们：多语言超大模型和语言专用小模型不是简单替代关系。若资源充足，Qwen3/BGE 这类大模型整体强；若部署受限，MMLW/Polish SBERT/Stella-PL 等语言定制模型仍有现实价值。

## 局限与展望
- PL-MTEB 虽覆盖 30 任务，但其中不少检索任务来自自动翻译的 BEIR-PL，可能带来翻译风格和原英文任务结构的偏差。
- zero-shot 判断依赖作者能收集到的训练数据说明。许多模型训练语料不完全公开，因此相似数据污染只能近似估计。
- 分类任务使用每类 8-shot logistic regression，适合评估 embedding 可分性，但不一定代表真实下游系统的 full-data 微调表现。
- benchmark 主要评估 dense embedding，未深入比较 sparse retrieval、hybrid retrieval、reranker 或 instruction embedding 在具体业务场景下的组合效果。
- 未来可以加入更多波兰语原生检索数据、长文档任务、跨语言检索、领域专用任务，以及持续更新 leaderboard 来跟上 embedding 模型迭代。

## 相关工作与启发
- **vs 原始 MTEB**: MTEB 提供通用框架，但英文和少数多语言任务占主导；PL-MTEB 把任务和数据质量控制落到波兰语上，使结论更可用于本地语言应用。
- **vs BEIR-PL / PIRB**: BEIR-PL 和 PIRB 主要关注检索；PL-MTEB 覆盖分类、聚类、句对分类、检索和 STS，更适合评估通用 embedding。
- **vs KLEJ / LEPISZCZE**: 这些 benchmark 更偏 NLU 和分类理解；PL-MTEB 关注无需任务特定深度模型的向量表示质量。
- **vs MMTEB**: MMTEB 是大规模多语言社区扩展；PL-MTEB 更像面向波兰语的精细子集，补充了数据整理、任务说明和本地模型分析。

## 评分
- 新颖性: ⭐⭐⭐☆☆ 算法新意不强，但语言特定 benchmark 的系统构建价值明确。
- 实验充分度: ⭐⭐⭐⭐⭐ 30 任务、30 模型、任务类型和模型规模分析都很扎实，并公开代码和数据。
- 写作质量: ⭐⭐⭐⭐☆ 结构清楚、表格信息量大；部分附录表格较长，读者需要结合任务类型解读。
- 价值: ⭐⭐⭐⭐⭐ 对波兰语 NLP 和多语言 embedding 选型非常实用，也为其他低资源/中资源语言构建 MTEB 扩展提供了可复用范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation](skmteb_slovak_massive_text_embedding_benchmark_and_model_adaptation.md)
- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)
- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)
- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](../../ACL2025/information_retrieval/a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)

</div>

<!-- RELATED:END -->
