---
title: >-
  [论文解读] GeAR: Graph-enhanced Agent for Retrieval-augmented Generation
description: >-
  [ACL 2025 (Findings)][LLM Agent][图增强检索] GeAR 通过图扩展机制（SyncGE）增强传统检索器的多跳发现能力，并结合 Gist Memory 代理框架实现多步检索推理，在 MuSiQue 等多跳QA数据集上超过现有SOTA 10%+，同时消耗更少的 token 和迭代次数。
tags:
  - "ACL 2025 (Findings)"
  - "LLM Agent"
  - "图增强检索"
  - "多跳问答"
  - "RAG代理"
  - "知识图谱扩展"
  - "Gist记忆"
---

# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation

**会议**: ACL 2025 (Findings)  
**arXiv**: [2412.18431](https://arxiv.org/abs/2412.18431)  
**代码**: [https://gear-rag.github.io/](https://gear-rag.github.io/)  
**领域**: Agent / 检索增强生成  
**关键词**: 图增强检索, 多跳问答, RAG代理, 知识图谱扩展, Gist记忆

## 一句话总结

GeAR 通过图扩展机制（SyncGE）增强传统检索器的多跳发现能力，并结合 Gist Memory 代理框架实现多步检索推理，在 MuSiQue 等多跳QA数据集上超过现有SOTA 10%+，同时消耗更少的 token 和迭代次数。

## 研究背景与动机

**领域现状**：检索增强生成（RAG）已成为解决大语言模型幻觉问题的主流方案。传统 RAG 依赖稀疏检索器（如 BM25）或稠密检索器（如 DPR）进行单步检索，然后将检索到的段落拼接到 prompt 中供 LLM 生成答案。

**现有痛点**：对于多跳推理场景（如"获得2024年诺贝尔物理学奖的英裔加拿大研究者在哪里读的博士？"），单步检索很难同时找到所有相关的桥接文档。现有多步检索方案（如 ITER-RETGEN、IRCoT）虽然可以迭代查询，但往往需要大量 token 消耗和多次迭代，效率偏低，且容易在中间步骤丢失关键信息。

**核心矛盾**：多跳检索需要在不同文档间建立推理链，但传统检索器天然缺乏文档间的关联感知能力。已有图检索方法（如基于知识图谱的检索）虽有关联能力，但构建和维护代价高昂，且与现有检索器难以兼容。

**本文目标**：设计一个可即插即用地增强任意传统检索器的图扩展机制，并在此基础上构建高效的多步检索代理框架。

**切入角度**：作者观察到，从检索到的段落中可以提取"近端三元组"（proximal triples），这些三元组能够在构建的文档图上进行扩展，从而发现仅靠关键词匹配无法到达的桥接文档。

**核心 idea**：用"提取三元组→图扩展→Gist Memory 累积"的方式，在每步检索中同时利用传统检索和图结构扩展，用 Gist Memory 模拟人类工作记忆来跨步骤保留关键信息。

## 方法详解

### 整体框架

GeAR 是一个多步迭代检索框架。每步检索包含三个阶段：(1) 基础检索器（如 BM25）根据当前查询检索段落；(2) SyncGE 图扩展机制从检索段落中提取三元组并在文档图上扩展获取更多相关段落；(3) Gist Memory 模块将扩展获得的关键信息以三元组形式累积存储。一个推理模块判断当前信息是否足以回答问题——若不够则重写查询进入下一轮，若足够则将 Gist Memory 中的三元组链回原始段落，通过 RRF 融合排序后返回最终结果。

### 关键设计

1. **SyncGE（同步图扩展）**:

    - 功能：增强基础检索器的多跳发现能力
    - 核心思路：首先用 LLM 从检索到的段落中抽取"近端三元组"（proximal triples），即与查询相关的 (主体, 关系, 客体) 结构化信息。然后通过 Triple Linking 将这些三元组映射到预构建的文档三元组索引中最近的真实三元组。最后在文档图上执行 Diverse Triple Beam Search——从链接到的三元组出发沿图的边进行扩展，使用 beam search 保留多样性最高的扩展路径，最终将扩展到的新三元组关联回其来源段落。
    - 设计动机：传统检索器只能通过文本相似度匹配，无法发现语义上相关但词汇不重叠的桥接文档。图扩展利用文档间的实体关系自然地跨越了这一鸿沟。使用 beam search 保证了扩展的多样性，避免陷入局部最优。

2. **Gist Memory（要旨记忆）**:

    - 功能：跨检索步骤累积和保留关键推理信息
    - 核心思路：模拟人类海马体的工作记忆机制。每步检索后，从扩展获得的段落中提取新的近端三元组，追加到 Gist Memory 中。在后续步骤中，Gist Memory 中存储的三元组既用于辅助新一轮的三元组提取（提供上下文），也用于推理模块判断是否已获得足够信息。最终检索时，Gist Memory 中的三元组通过 Passage Linking 映射回原始段落，与常规检索段落融合。
    - 设计动机：多步检索的核心挑战是信息流失——前几步获得的关键线索可能在后续步骤中被稀释或遗忘。Gist Memory 以结构化三元组而非原始文本的形式保留信息，既压缩了存储量，又保留了关键推理链条。

3. **推理与查询重写模块**:

    - 功能：决定何时停止检索，以及如何生成下一步查询
    - 核心思路：每步检索后，将当前 Gist Memory 中所有三元组与原始问题一起输入 LLM，判断是否已有足够证据回答。若信息不足，LLM 同时输出缺失信息的描述和重写后的查询。最大迭代步数设为 4。
    - 设计动机：相比固定步数的多步检索，自适应终止可以在简单问题上节省资源，在复杂问题上保证检索充分性。查询重写基于已知信息推导缺失部分，比简单的查询扩展更有针对性。

### 损失函数 / 训练策略

GeAR 不需要端到端训练。图扩展和 Gist Memory 构建均依赖 LLM 的 in-context learning 能力（通过精心设计的 prompt 实现三元组提取、推理判断等功能）。文档三元组索引在离线阶段通过 LLM 从语料库中预提取构建。

## 实验关键数据

### 主实验

在三个多跳 QA 数据集上评估检索性能（Recall@10/20）和问答性能（EM/F1）：

| 数据集 | 指标 | GeAR | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| MuSiQue | Recall@10 | ~65% | ~55% (IRCoT) | +10%+ |
| HotpotQA | Recall@10 | SOTA | IRCoT/ITER-RETGEN | 可比/更优 |
| 2WikiMultiHopQA | Recall@10 | SOTA | IRCoT | 显著提升 |
| MuSiQue | EM (QA) | SOTA | 之前最佳 | +10%+ |
| HotpotQA | F1 (QA) | SOTA | 之前最佳 | 可比 |

### 消融实验

| 配置 | MuSiQue Recall | 说明 |
|------|---------------|------|
| Full GeAR | 最高 | 完整系统 |
| w/o Graph Expansion | 显著下降 | 图扩展是核心贡献 |
| w/o Gist Memory | 中等下降 | 信息累积对多跳很重要 |
| w/o Query Rewriting | 下降 | 自适应查询重写有帮助 |
| 仅 BM25 基线 | 最低 | 单步检索严重不足 |

### 关键发现

- **图扩展是最关键的模块**：去掉 SyncGE 后性能大幅下降，说明文档间的图结构关联是多跳检索的核心。
- **效率优势明显**：GeAR 平均只需 2-3 步迭代，消耗的 token 数远少于 IRCoT 等方法，说明图扩展能在单步中完成传统方法需要多步才能实现的跨文档发现。
- **在最难的 MuSiQue 上优势最大**：MuSiQue 需要 2-4 跳推理，传统方法很难处理，而图扩展天然适合这种场景。在较简单的 HotpotQA（2跳）上优势相对较小。
- **可即插即用**：SyncGE 可以增强 BM25、DPR 等不同基础检索器，具有很好的通用性。

## 亮点与洞察

- **图扩展+传统检索器的即插即用设计**非常巧妙——不需要替换现有检索器，只需在其输出上追加图扩展步骤，降低了实际部署门槛。
- **Gist Memory 用三元组压缩信息**的设计思路可以迁移到其他需要多步推理的任务（如多步数学推理、长文档理解），用结构化中间表示替代原始文本能大幅减少上下文长度。
- **Diverse Triple Beam Search** 是一个可复用的 trick——在图搜索中同时保证覆盖度和多样性，避免了传统 BFS/DFS 容易出现的局部聚集问题。

## 局限与展望

- **依赖 LLM 进行三元组提取**：每步检索都需要多次 LLM 调用（提取三元组、推理判断、查询重写），延迟和成本可能较高，不适合对实时性要求极高的场景。
- **离线三元组索引的构建代价**：需要对整个语料库预提取三元组并构建图索引，对于大规模语料库可能成本不低。
- **评估仅限于多跳QA**：没有在开放域对话、事实验证等其他 RAG 场景上验证，通用性有待考察。
- **改进方向**：可以探索用更轻量的模型（如小型专用提取器）替代 LLM 做三元组提取以降低成本；也可以将图扩展与向量数据库结合，实现更灵活的混合检索。

## 相关工作与启发

- **vs IRCoT**: IRCoT 使用 Chain-of-Thought 迭代检索但纯依赖文本匹配，缺乏图结构，GeAR 通过图扩展在单步中就能发现多跳桥接文档。
- **vs ITER-RETGEN**: ITER-RETGEN 迭代生成+检索交替，计算开销大且信息容易丢失，GeAR 的 Gist Memory 更好地保留了跨步信息。
- **vs GraphRAG (微软)**: GraphRAG 需要完整构建社区图并做全局摘要，开销更大且适用于摘要类查询；GeAR 更轻量且专注于多跳精确检索。

## 评分

- 新颖性: ⭐⭐⭐⭐ 图扩展+Gist Memory的组合有新意，但多步RAG并非全新方向
- 实验充分度: ⭐⭐⭐⭐ 三个数据集+检索和QA双评估+消融实验，但缺少效率详细对比
- 写作质量: ⭐⭐⭐⭐ 项目页做得很好，方法描述清晰，伪代码辅助理解
- 价值: ⭐⭐⭐⭐ 即插即用设计实用性强，10%+提升显著，对多跳RAG研究有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](../../ICML2025/llm_agent/evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)
- [\[ICLR 2026\] Aria: an Agent for Retrieval and Iterative Auto-Formalization via Dependency Graph](../../ICLR2026/llm_agent/aria_an_agent_for_retrieval_and_iterative_auto-formalization_via_dependency_grap.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] LocAgent: Graph-Guided LLM Agents for Code Localization](locagent_graph-guided_llm_agents_for_code_localization.md)
- [\[ACL 2025\] MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation](metasynth_meta-prompting-driven_agentic_scaffolds_for_diverse_synthetic_data_gen.md)

</div>

<!-- RELATED:END -->
