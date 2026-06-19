---
title: >-
  [论文解读] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models
description: >-
  [ICML 2025][图学习][RAG] 提出 HippoRAG 2，通过将段落节点融入知识图谱、用 query-to-triple 深度上下文化链接、以及 LLM 驱动的识别记忆过滤，全面超越标准 RAG 在事实记忆、语义理解和关联推理三大维度的表现，向 LLM 的非参数化持续学习迈进一步。 持续学习（Continual…
tags:
  - "ICML 2025"
  - "图学习"
  - "RAG"
  - "知识图谱"
  - "持续学习"
  - "Personalized PageRank"
  - "长期记忆"
---

# From RAG to Memory: Non-Parametric Continual Learning for Large Language Models

**会议**: ICML 2025  
**arXiv**: [2502.14802](https://arxiv.org/abs/2502.14802)  
**代码**: [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)  
**领域**: 图学习  
**关键词**: RAG, 知识图谱, 持续学习, Personalized PageRank, 长期记忆

## 一句话总结

提出 HippoRAG 2，通过将段落节点融入知识图谱、用 query-to-triple 深度上下文化链接、以及 LLM 驱动的识别记忆过滤，全面超越标准 RAG 在事实记忆、语义理解和关联推理三大维度的表现，向 LLM 的非参数化持续学习迈进一步。

## 研究背景与动机

持续学习（Continual Learning）是人类智能的核心能力之一，然而 LLM 在吸收新知识时面临**灾难性遗忘**和**知识更新代价高**两大难题。RAG 作为一种非参数化方案已成为生产系统的主流持续学习方案，但标准 RAG 依赖简单向量检索，在两方面存在缺陷：

**语义理解（Sense-making）**：无法整合跨段落的复杂上下文信息

**关联性（Associativity）**：无法进行多跳推理，连接分散的知识片段

现有的结构增强 RAG（如 RAPTOR、GraphRAG、LightRAG、HippoRAG）虽然在各自擅长的任务上有改进，但作者通过全面实验发现一个关键问题：**这些方法在自身目标任务之外的表现会显著退化**。例如 HippoRAG 在多跳 QA 优秀但在篇章理解任务上下降，RAPTOR 在简单 QA 和多跳 QA 上明显退步。这种"顾此失彼"的现象表明，现有方法距离真正的人类长期记忆系统还有很大差距。

HippoRAG 2 的目标是：**在不牺牲任何一个维度的前提下，全面提升事实记忆、语义理解和关联推理三方面能力**。

## 方法详解

### 整体框架

HippoRAG 2 延续 HippoRAG 的神经生物学启发架构，包含两大阶段：

**离线索引阶段（Offline Indexing）**：

1. 使用 LLM（Llama-3.3-70B-Instruct）通过 OpenIE 从每个段落中抽取三元组（subject, relation, object），构建无模式知识图谱
2. 检索编码器（NV-Embed-v2）对短语节点编码，检测同义关系并添加同义边
3. **新增**：将原始段落作为段落节点加入图谱，通过 "contains" 边与其衍生的短语节点连接（Dense-Sparse Integration）

**在线检索阶段（Online Retrieval）**：

1. 使用编码器将查询与三元组和段落进行匹配，识别种子节点（Query-to-Triple）
2. **新增**：LLM 作为识别记忆过滤器，筛除不相关的三元组（Recognition Memory）
3. 使用筛选后的种子节点设置 PPR 的重启概率，执行图搜索
4. 按 PageRank 分数排序段落，取 top-k 作为 QA 上下文

### 关键设计

#### 设计一：Dense-Sparse Integration（密集-稀疏编码融合）

受人脑密集编码与稀疏编码理论启发，HippoRAG 2 同时维护两种类型的节点：

- **短语节点（Phrase Node）**：对应稀疏编码，表示从段落中抽取的概念实体，简洁但有信息损失
- **段落节点（Passage Node）**：对应密集编码，保留原始段落的完整上下文

具体做法：每个段落作为一个段落节点，通过标签为 "contains" 的上下文边与该段落产出的所有短语节点相连。相比 HippoRAG 的文档集成（简单聚合图搜索和嵌入匹配分数），这种设计让上下文信息**直接参与图搜索过程**，而非事后拼接。消融实验显示移除段落节点后平均 Recall@5 从 87.1% 降至 81.0%。

#### 设计二：Deeper Contextualization（深度上下文化链接）

原始 HippoRAG 通过 NER 从查询中提取实体，再匹配到知识图谱节点（NER-to-Node），这一过程过于以概念为中心，忽略了上下文信号。HippoRAG 2 探索了三种链接策略：

| 策略 | 描述 | 效果 |
|------|------|------|
| NER to Node | 从查询提取实体，匹配KG节点 | 基线方法，Avg Recall 74.6% |
| Query to Node | 整个查询直接匹配KG节点 | 粒度不匹配，Avg Recall 59.6% |
| **Query to Triple** | 整个查询匹配KG三元组 | **最优**，Avg Recall 87.1% |

Query-to-Triple 的核心优势在于：三元组本身封装了概念之间的基本上下文关系，能更全面地捕捉查询意图。查询和三元组在信息粒度上比查询与单个节点更为对齐。

#### 设计三：Recognition Memory（识别记忆过滤）

借鉴人类记忆中回忆（recall）和识别（recognition）的互补机制：

1. **Recall 阶段**：用嵌入模型检索 top-k 个三元组 $T$
2. **Recognition 阶段**：用 LLM 对检索到的三元组进行过滤，生成子集 $T' \subseteq T$，剔除与查询不相关的三元组

过滤 prompt 通过 DSPy 的 MIPROv2 优化器自动调优（包括指令和示例）。消融实验表明过滤机制带来约 0.7% 的平均提升（86.4% → 87.1%），虽然增量不大但在多数据集上一致有效。

### 损失函数 / 训练策略

HippoRAG 2 不涉及端到端训练。其核心算法为 **Personalized PageRank (PPR)**，关键超参数控制如下：

- **种子节点选择**：短语节点根据其在被过滤三元组中的平均排名分数选取（至多 k 个）；所有段落节点也作为种子节点，因为更广泛的激活有助于多跳推理
- **重启概率分配**：短语节点基于排名分数，段落节点基于嵌入相似度并乘以权重因子
- **权重因子**：控制短语节点与段落节点之间的影响力平衡，经验证设为 **0.05**
- **Triple 过滤的 prompt 优化**：使用 DSPy MIPROv2 + Llama-3.3-70B 自动调优

## 实验关键数据

### 主实验

评估三大类任务：简单 QA（事实记忆）、多跳 QA（关联推理）、篇章理解（语义理解）。

**QA 性能（F1 分数，Llama-3.3-70B 作为 reader）**：

| 数据集 | 类型 | HippoRAG 2 | NV-Embed-v2 (最强基线) | 提升 |
|--------|------|-------------|----------------------|------|
| NQ | 简单QA | **63.3** | 61.9 | +1.4 |
| PopQA | 简单QA | 56.2 | 55.7 | +0.5 |
| MuSiQue | 多跳QA | **48.6** | 45.7 | +2.9 |
| 2Wiki | 多跳QA | 71.0 | 61.5 | +9.5 |
| HotpotQA | 多跳QA | **75.5** | 75.3 | +0.2 |
| LV-Eval | 多跳QA | **12.9** | 9.8 | +3.1 |
| NarrativeQA | 篇章理解 | **25.9** | 25.7 | +0.2 |
| **平均** | - | **59.8** | 57.0 | **+2.8** |

**检索性能（Recall@5）**：

| 数据集 | HippoRAG 2 | NV-Embed-v2 | HippoRAG | 提升(vs NV) |
|--------|------------|-------------|----------|------------|
| NQ | **78.0** | 75.4 | 44.4 | +2.6 |
| PopQA | 51.7 | 51.0 | **53.8** | +0.7 |
| MuSiQue | **74.7** | 69.7 | 53.2 | +5.0 |
| 2Wiki | **90.4** | 76.5 | **90.4** | +13.9 |
| HotpotQA | **96.3** | 94.5 | 77.3 | +1.8 |
| **平均** | **78.2** | 73.4 | 63.8 | **+4.8** |

### 消融实验

| 配置 | MuSiQue | 2Wiki | HotpotQA | Avg | 说明 |
|------|---------|-------|----------|-----|------|
| HippoRAG 2 (完整) | **74.7** | 90.4 | **96.3** | **87.1** | - |
| w/ NER to Node | 53.8 | **91.2** | 78.8 | 74.6 | 概念级匹配丢失上下文 |
| w/ Query to Node | 44.9 | 65.5 | 68.3 | 59.6 | 查询与节点粒度不匹配 |
| w/o Passage Node | 63.7 | 90.3 | 88.9 | 81.0 | 缺少密集编码 |
| w/o Filter | 73.0 | 90.7 | 95.4 | 86.4 | 未过滤噪声三元组 |

### 关键发现

1. **全面超越最强嵌入模型**：HippoRAG 2 是唯一在事实记忆、语义理解、关联推理三个维度都超过 NV-Embed-v2 的结构增强方法
2. **Query-to-Triple 是最关键设计**：比 NER-to-Node 平均高 12.5% Recall@5，因为三元组层级的信息粒度与查询最为匹配
3. **对密集检索器的通用增强**：在 GTE-Qwen2-7B、GritLM-7B、NV-Embed-v2 三种检索器上均带来一致提升（+5.0~5.6%）
4. **持续学习鲁棒性**：在语料库持续扩展的模拟实验中，HippoRAG 2 相对 NV-Embed-v2 的优势保持稳定
5. **其他结构增强方法的致命缺陷**：LightRAG 在大多数任务上 F1 < 12%，GraphRAG 和 RAPTOR 在简单 QA 上显著退化

## 亮点与洞察

1. **神经生物学类比的实用价值**：用海马体（KG + PPR）、新皮层（LLM）、海马旁区（编码器）的三组件架构映射人类记忆机制，不仅是概念上的吸引力，确实在工程层面指导了有效的设计决策
2. **概念-上下文权衡的精妙解法**：不同于 GraphRAG/LightRAG 用 KG 扩展检索语料（引入 LLM 生成噪声），HippoRAG 2 用 KG 辅助检索过程本身，同时通过段落节点保留原始上下文
3. **DSPy 自动 prompt 优化的实际应用**：识别记忆模块的 prompt 不是手工设计，而是用 MIPROv2 自动调优，展示了 prompt 工程自动化在 RAG 系统中的落地可能
4. **权重因子 0.05 的非直觉发现**：段落节点的重启概率需要大幅缩小（×0.05），说明在 PPR 中概念级信号远比上下文级信号重要，段落节点更多起"锚定"而非"主导"作用

## 局限与展望

1. **计算开销**：离线索引需要 LLM 执行 OpenIE 抽取，在线检索需要 LLM 做三元组过滤，相比纯向量 RAG 成本显著增加
2. **关联推理的退化趋势**：持续学习实验中多跳 QA 性能随语料扩展持续下降，HippoRAG 2 和 NV-Embed-v2 下降速率相似，说明 KG 结构并未根本解决信息过载问题
3. **OpenIE 质量依赖**：三元组抽取的质量直接影响 KG 结构和检索效果，对 LLM 的 OpenIE 能力有较强依赖
4. **单一图搜索算法**：仅使用 PPR，未探索其他图搜索或图神经网络方法的潜力
5. **评估局限**：NarrativeQA 仅选了 10 篇文档 293 个查询，语义理解维度的评估规模偏小

## 相关工作与启发

- **HippoRAG**（Gutiérrez et al., 2024）：直接前身，用 OpenIE + PPR 实现关联推理，但概念为中心导致上下文丢失
- **GraphRAG**（Edge et al., 2024）：用图社区检测生成摘要增强语义理解，但在简单 QA 和多跳 QA 退化
- **RAPTOR**（Sarthi et al., 2024）：用层次化摘要组织语料，但引入噪声导致 QA 退化
- **LightRAG**（Guo et al., 2024）：双层检索但实验中表现极差（avg F1 仅 6.6%）
- **NV-Embed-v2**（Lee et al., 2025）：7B 参数的 SOTA 嵌入模型，作为最强纯向量 RAG 基线

**启发**：该工作表明，结构化知识与向量检索的融合方式至关重要——用结构扩展语料不如用结构辅助检索。未来的 RAG 系统设计应追求多维度的鲁棒性评估，而非单一任务上的指标提升。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 三个改进设计各有理论和实验支撑，整体架构自然演进 |
| 实验充分度 | 5 | 7个数据集×3类任务，多组消融，多检索器验证，持续学习测试 |
| 写作质量 | 4 | 结构清晰，神经生物学类比贯穿始终 |
| 实用价值 | 4 | 开源代码，可直接集成到现有 RAG 系统，但计算成本是实际部署的障碍 |
| 总分 | 4.25 | 在 RAG 领域做了系统性的全面提升，实验设计值得学习 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation](../../NeurIPS2025/graph_learning/elastic_weight_consolidation_for_knowledge_graph_continual_learning_an_empirical.md)
- [\[ICML 2025\] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[AAAI 2026\] Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces](../../AAAI2026/graph_learning/beyond_fact_retrieval_episodic_memory_for_rag_with_generative_semantic_workspace.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)

</div>

<!-- RELATED:END -->
