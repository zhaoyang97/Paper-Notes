---
title: >-
  [论文解读] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges
description: >-
  [ACL 2026][信息检索/RAG][推理密集型检索] 本文系统梳理了"推理密集型检索 (Reasoning-Intensive Retrieval, RIR)"这一新方向，按 query/index/retriever/reranker/迭代 这条流水线给出了第一份完整的 benchmark-方法-挑战 三段式综述，并指出现有评测过度依赖 nDCG 等传统 IR 指标。
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "推理密集型检索"
  - "RIR"
  - "重排"
  - "迭代检索"
  - "LLM 嵌入"
---

# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges

**会议**: ACL 2026  
**arXiv**: [2605.00063](https://arxiv.org/abs/2605.00063)  
**代码**: 无  
**领域**: 信息检索 (综述)  
**关键词**: 推理密集型检索, RIR, 重排, 迭代检索, LLM 嵌入

## 一句话总结
本文系统梳理了"推理密集型检索 (Reasoning-Intensive Retrieval, RIR)"这一新方向，按 query/index/retriever/reranker/迭代 这条流水线给出了第一份完整的 benchmark-方法-挑战 三段式综述，并指出现有评测过度依赖 nDCG 等传统 IR 指标。

## 研究背景与动机
**领域现状**：传统稠密检索（DPR、Contriever、BGE 等）依赖 query 与 document 之间的语义/词面相似，已经在网页搜索这类语义重叠高的场景达到很强水平。

**现有痛点**：在专家领域（医学、法律、数学、代码）和 deep research 场景下，query 与正确证据之间往往只通过一条隐式多跳推理链相连——例如"被海水煮过的水能喝吗？"这样的 query 需要先想到"盐分蒸发时不会消失"才能找到正确文档，简单的相似度匹配完全失效。本文把这一类问题正式命名为 RIR。

**核心矛盾**：现有研究存在两个突出问题——其一，评测高度异构，benchmark 横跨代码、数学、医学，问题形式与数据来源各不相同，无法横向比较；其二，方法散落在流水线各阶段（query rewriting、retriever 训练、reranking、iterative RAG），缺乏统一的分类骨架，研究者难以选择起点。

**本文目标**：建立 RIR 的统一路线图——把 benchmark 按推理类型/领域/模态分桶，把方法按"在 pipeline 哪里、用什么方式注入推理"分桶，并清点未解决的挑战。

**切入角度**：作者按"推理在检索流水线中介入的位置"做组织主轴——这是比按模型架构或数据集分类更稳定的视角，因为新模型层出不穷，但 pipeline 阶段是有限且稳定的。

**核心 idea**：用一张二维分类法（pipeline 阶段 × 推理注入方式）把碎片化的 RIR 研究收编进同一框架，从而暴露真正的空白点。

## 方法详解

### 整体框架
全文是一份把碎片化 RIR 研究收编进同一框架的结构化路线图，沿"推理在检索流水线哪一步介入"这条主轴展开，覆盖评测全景（Section 3）、方法分类法（Section 4）、未解挑战（Section 5）三大块。评测侧把 17 个 benchmark 按"领域 × 模态"分成 Open-Domain（如 BESPOKE、ImpliRet，侧重从聊天历史推断隐式意图）、Expert-Domain（MIRB、R2MED、CoIR、Bar Exam QA 等科学/法律/医学/代码任务）、Multi-Domain（BRIGHT、Bright-Plus、RAR-b 的跨领域大杂烩）、Multimodal（MRMR、MR²-Bench、ARK 等图文联合）四桶，并给每个 benchmark 标注演绎/类比/因果/分析/数值五类推理；方法侧则按推理注入的 pipeline 位置切成四个互斥的桶，最后清点 nDCG 过时、multimodal 缺口等空白点。

### 关键设计

**1. 按 pipeline 位置而非模型架构分类方法：让 taxonomy 对新模型保持稳定**

综述把所有 RIR 方法按"推理在检索流水线哪一步发挥作用"分到四个互斥的桶：Pre-Retrieval Augmentation、Reasoning-Aware Retriever Training、Reasoning-Enhanced Reranking、Iterative Retrieval。Pre-Retrieval 又分 query-side（query 改写/分解，如 TongSearch-QR、ThinkQE、ReDI）与 index-side（文档端扩写，如 SPIKE、EnrichIndex、LATTICE）；Retriever Training 关注 backbone 选择（LLM-based vs Diffusion LM）、难负样本策划（ReasonIR、DIVER、RaDeR）与训练目标（multi-task SFT + RL，format/embedding 双 reward）；Reranker 沿 Prompt-Tuning → SFT/Distillation → RL（Rank1、Rank-K、Rank-R1、ReasonRank）演化；Iterative 把检索-推理交替建模成 state machine（SMR）或 RL policy。之所以选 pipeline 位置而非模型架构当骨架，是因为新模型层出不穷但 pipeline 阶段有限且稳定，研究者也能按"我想在 query 端做工作"这种朴素需求快速定位相关工作。

**2. 按五类推理类型标注 benchmark：暴露不同领域的推理需求差异与缺口**

综述把 BRIGHT 提出的演绎、类比、因果、分析、数值五种推理类型作为标签贴到每个 benchmark 上。统计后能看到清晰规律：演绎推理（rule-to-case 应用）在数学/科学/医学/法律中最普遍，类比推理在代码/数学的跨语言映射中突出，数值推理多见于日常时间运算，因果/分析推理则集中在故障排查与问题分解。这层标注让研究者一眼看出自己的方法适合哪类 benchmark，也直接暴露了多模态因果等推理类型的 benchmark 缺口。

**3. Scale-Reliability 权衡视角：用一个可量化维度总结 benchmark 构造的根本张力**

benchmark 构造存在一对此消彼长的矛盾：LLM 合成（ScIRGen、ImpliRet）能规模化但带幻觉，人类标注（BRIGHT、Bar Exam QA）可靠但成本高。综述把 17 个 benchmark 沿"规模 × 标注方式"二维定位，发现"先 LLM 生成再人工审核"的混合构造正成为主流，并据此主张未来应走"先合成后专家校验"路线。用大小与人工占比这个可量化维度做总结，能把对 benchmark 的评论从"哪个更好"提升到设计权衡层面。

### 损失函数 / 训练策略
综述在附录 E 归纳了 RIR 方法的三大损失：InfoNCE（标准对比损失，几乎所有 retriever 都用）、Generation Loss（针对带 "thought" 的 retriever，如 O1 Embedder 用 next-token prediction 学中间推理）、MSE（把 LLM 推理过的嵌入蒸馏进学生 retriever，如 Dense Reasoner）。RL 方案（LREM、UME-R1、ReasonRank）则把 generation-side reward（格式合规、长度控制）与 embedding-side reward（检索精度）加权组合，使 reasoning trajectory 本身成为可优化对象。

## 实验关键数据
本文为综述，下面整理其汇总的 RIR benchmark 与方法对比数据。

### 主实验：benchmark 全景对比

| Benchmark | 领域 | 规模 | 标注方式 |
|-----------|------|------|----------|
| BRIGHT | Multi-Domain | 1,384 | Hybrid |
| Bright-Plus | Multi-Domain | 1,384 | Hybrid |
| R2MED | Medical | 876 | Hybrid |
| MIRB | Math | 39,029 | Derived |
| CoIR | Code | ~162,000 | Derived |
| CoQuIR | Code | 42,725 | LLM-Automated |
| ScIRGen | Scientific | 61,376 | LLM-Automated |
| BESPOKE | Open Domain | 150 | Human-Curated |
| MRMR | Multi-Modal | 1,435 | Hybrid |
| MR²-Bench | Multi-Modal | 1,309 | Hybrid |

### 消融实验：四类方法的特点对比

| Pipeline 阶段 | 代表方法 | 主要收益 | 主要代价 |
|---------------|----------|----------|----------|
| Pre-Retrieval (query) | TongSearch-QR / ThinkQE | 小模型即可用 RL 重写出强 query | 多轮迭代增加 token 开销 |
| Pre-Retrieval (index) | EnrichIndex / LATTICE | 把推理离线化，在线推理便宜 | 索引膨胀、需重建 |
| Retriever Training | ReasonIR / DIVER | 端到端嵌入更强 | 需精心策划难负样本 |
| Reranking | Rank1 / ReasonRank | 在 BRIGHT 上效果最佳 | 推理时延高 |
| Iterative | SMR / Vijay et al. | 处理复杂多跳 | "overthinking" 风险 |

### 关键发现
- **多阶段叠加并非越多越好**：iterative 方法虽然在 BRIGHT 上 SOTA，但容易出现"overthinking"和漂移，反而不如精心设计的单阶段方法稳定。
- **专门化方法在通用 IR 上回退**：在 RIR benchmark 上训练的 retriever 通常在 MTEB 等通用 benchmark 上不如 Gemini Embedding、Jina-V5 等通用大模型嵌入。
- **推理类型决定方法选择**：演绎推理任务对 reranker 增益最大；数值推理任务则需要 query decomposition 把问题拆成可计算的子问题。

## 亮点与洞察
- **以 pipeline 位置而非模型架构组织方法**，是这份综述最稳的设计——把"在哪里注入推理"作为分类骨架，使 taxonomy 对未来涌现的新模型保持兼容性。
- **推理类型的五分类法**（演绎/类比/因果/分析/数值）首次把 RIR 任务的难度做了量化拆解，未来 benchmark 设计可以按缺口（如多模态因果）针对性补全。
- **明确指出 nDCG 已经过时**：作者论证 RIR 时代需要把"效率"和"细粒度相关性"纳入指标，例如 Peng et al. 的 efficiency-effectiveness FLOPs 联合评测、Weller 等人的 instruction-following 指标——这是给 IR 评测社区的一记警钟。

## 局限与展望
- **作者承认**：综述仅覆盖在公开 RIR benchmark 上跑过实验的方法，HyDE、graph-based retrieval 等其它有潜力的方向未深入；同时未涉及工业界的私有 RIR 系统。
- **额外局限**：分类法严格按 pipeline 位置切分，对"跨阶段联合训练"的方法（如端到端 retriever-reranker co-training）覆盖较弱；推理类型五分类来自 BRIGHT，可能不能涵盖未来出现的"反事实推理""程序合成推理"等新类型。
- **改进方向**：建立 RIR 专用的"reasoning-faithful"评测指标（不仅看 top-k 命中，还要看检索到的证据链是否真的支撑推理），结合 deep research/long-term memory 等真实下游场景做端到端评估。

## 相关工作与启发
- **vs RAG-Reasoning 综述**（Li 2025g）：他们把检索当作支撑生成的前置步骤，本文反过来把检索本身视作终点任务，强调 retriever 自身的推理能力。
- **vs Reasoning Agentic RAG 综述**（Liang 2025）：他们关注 agent 框架下如何调度检索，本文聚焦推理如何融入 retriever/reranker 内部，互为补充。
- **vs 经典 IR survey**（Robertson & Zaragoza 2009；Yates 2021）：经典综述以语义/词面相关为核心，本综述把"推理介入相关性建模"作为新范式，标志 IR 进入第三波（继 BM25、稠密检索之后）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一份系统的 RIR 综述，分类骨架（pipeline 位置 × 推理类型）有原创性。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 17 个 benchmark 与 30+ 方法，附带实证分析（LLM-based vs LRM-based、computation cost vs performance）。
- 写作质量: ⭐⭐⭐⭐ 二维分类法清晰，trade-off 视角到位；但章节间内容密度差异较大，appendix 信息量超过正文。
- 价值: ⭐⭐⭐⭐⭐ 对正在涌入 RIR 领域的研究者而言，是必读的入门地图，明确指出 nDCG 过时和 multimodal 缺口这两个高价值方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ICLR 2026\] Beyond Sequential Reranking: Reranker-Guided Search Improves Reasoning Intensive Retrieval](../../ICLR2026/information_retrieval/beyond_sequential_reranking_reranker-guided_search_improves_reasoning_intensive_.md)
- [\[ICML 2026\] REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](../../ICML2026/information_retrieval/real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)

</div>

<!-- RELATED:END -->
