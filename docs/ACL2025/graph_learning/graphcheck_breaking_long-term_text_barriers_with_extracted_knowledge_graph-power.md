---
title: >-
  [论文解读] GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking
description: >-
  [图学习][事实核查] GraphCheck 提出一种图增强的事实核查框架，利用 LLM 从文档和声明中提取知识图谱三元组，通过 GNN 编码图结构并作为 soft prompt 注入冻结的 LLM 验证器，在单次推理调用中实现细粒度事实核查，在7个基准上平均提升 7.1%，且在医学领域表现出强泛化能力。
tags:
  - "图学习"
  - "事实核查"
  - "知识图谱"
  - "图神经网络"
  - "长文本理解"
  - "幻觉检测"
---

# GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking

| 会议 | 领域 | arXiv | 代码 |
|------|------|-------|------|
| ACL2025 | Graph Learning / Fact-Checking | [2502.16514](https://arxiv.org/abs/2502.16514) | [GitHub](https://github.com/Yingjian-Chen/GraphCheck) |

**关键词**: 事实核查, 知识图谱, 图神经网络, 长文本理解, 幻觉检测

## 一句话总结

GraphCheck 提出一种图增强的事实核查框架，利用 LLM 从文档和声明中提取知识图谱三元组，通过 GNN 编码图结构并作为 soft prompt 注入冻结的 LLM 验证器，在单次推理调用中实现细粒度事实核查，在7个基准上平均提升 7.1%，且在医学领域表现出强泛化能力。

## 研究背景与动机

LLM 广泛使用但常生成微妙的事实错误（幻觉），特别是在长文本中。在医学等专业领域，事实错误可能导致误诊和不当治疗。

现有基于 grounding document 的事实核查方法面临两大挑战：

**多跳关系理解困难**：直接将长文档和声明输入 LLM 进行核查（Naive Check），容易忽略复杂的实体关系和微妙的事实不一致

**效率低下**：专门方法（如 FactScore、MiniCheck）将长文档分解为原子事实逐一核查（Atomic Check），需要多次模型调用，计算成本高

GraphCheck 的目标：在单次推理中实现细粒度事实核查，同时捕获长文本中的复杂多跳逻辑关系。

## 方法详解

### 整体框架

GraphCheck 包含三个主要步骤：

1. **图构建**：从文档 $D$ 和声明 $C$ 中提取知识三元组，构建对应的知识图谱 $G_D$ 和 $G_C$
2. **图编码**：用可训练的 GNN 编码图结构，获得图嵌入
3. **事实核查**：将图嵌入与文本嵌入拼接，输入冻结的 LLM 进行判断

### 图构建

使用 LLM 从文本中自动提取 $\{source, relation, target\}$ 三元组，构建有向图 $G = (\mathcal{V}, \mathcal{E})$：
- $\mathcal{V} = \{\mathbf{v}_i\}_{i=1,\dots,n}$：节点（实体）特征集合
- $\mathcal{E} = \{\mathbf{e}_{ij}\}$：边（关系）特征集合

节点和边的文本属性使用 Sentence-Transformers (all-roberta-large-v1) 编码为特征向量。

### 图编码（GNN）

采用消息传递机制更新节点特征：

$$\mathbf{v}_i^{l+1} = \text{UPDATE}\left(\mathbf{v}_i^l, \sum_{j \in \mathcal{N}_i} \text{MESSAGE}(\mathbf{v}_j^l, \mathbf{e}_{ji})\right)$$

最终图嵌入通过 READOUT 函数（包含求和聚合）获得：

$$\mathbf{h}_g = \text{READOUT}(\{\mathbf{v}_i^L\}_{i=1,\dots,n})$$

### 图投影

使用投影模块 $P$ 将图特征 $\mathbf{h}_g^C$ 和 $\mathbf{h}_g^D$ 映射到 LLM 的文本嵌入空间，得到 $\tilde{\mathbf{h}}_g^C$ 和 $\tilde{\mathbf{h}}_g^D$。

### 事实核查

将投影后的图嵌入与文本嵌入拼接，输入 LLM 的 self-attention 层：

$$y = \text{LLM}(\tilde{\mathbf{h}}_g^C, \tilde{\mathbf{h}}_g^D, \mathbf{h}_t)$$

输出 $y \in \{\text{"support"}, \text{"unsupport"}\}$。

### 训练策略

- **训练数据**：基于 MiniCheck 数据集的 14K 合成样本
- **图提取**：使用 Claude-3.5-Sonnet 提取三元组
- **训练方式**：仅训练 GNN 和投影层，LLM 参数冻结
- **关键发现**：虽然在通用领域数据上训练，但图增强推理能力可泛化到医学领域

## 实验

### 评估基准

覆盖 7 个基准，跨通用和医学两个领域：

| 数据集 | 领域 | 大小 | 文档平均长度 | 声明平均长度 |
|--------|------|------|-------------|-------------|
| AggreFact-Xsum | 通用 | 558 | 324 | 23 |
| AggreFact-CNN | 通用 | 558 | 500 | 55 |
| SummEval | 通用 | 1600 | 359 | 63 |
| ExpertQA | 通用 | 3702 | 432 | 26 |
| COVID-Fact | 医学 | 4086 | 72 | 12 |
| SCIFact | 医学 | 809 | 249 | 12 |
| PubHealth | 医学 | 1231 | 77 | 14 |

### 主实验结果（Balanced Accuracy %）

| 方法 | Overall Avg |
|------|-------------|
| GPT-4 | 70.8 |
| GPT-4o | 70.1 |
| OpenAI o1 | 72.9 |
| Claude 3.5-Sonnet | 73.6 |
| DeepSeek-V3 671B | 71.7 |
| MiniCheck | 68.1 |
| GraphEval | 65.1 |
| **GraphCheck-Llama3.3 70B** | **71.1** |
| **GraphCheck-Qwen 72B** | **70.7** |

关键观察：
- GraphCheck 超越 GPT-4 和 GPT-4o，接近最强大模型（o1、Claude-3.5）
- 显著优于所有专门事实核查方法（比 MiniCheck +3%，比 ACUEval +10.5%）
- 在单次调用中完成，效率远高于需要多次调用的方法

### 领域分析

- **通用领域**：与 MiniCheck、GraphEval 持平
- **医学领域**：比 MiniCheck 高 **8.1%**，展现出强跨域泛化能力

### 消融实验

1. **图信息的增量贡献**：在轻量模型（Llama3 8B、Qwen2.5 7B）和大模型（70B+）上均有显著提升
2. **KG 作为文本 vs 图嵌入**：直接将 KG 文本加入 prompt 仅有微弱提升（66.4% vs 65.3%），而用 GNN 编码后效果显著（71.1%），说明 LLM 无法从纯文本 KG 中有效提取结构信息
3. **训练数据量**：性能随训练数据增加呈上升趋势，AggreFact-Xsum 从 60.1% 提升到 72.9%
4. **KG 质量影响**：短文本中不同模型提取的 KG 差异小；长文本中 KG 完整性至关重要，低质量 KG（如 Llama 8B 提取）会引入噪声

### 可解释性案例

通过可视化 GNN 的边注意力权重，可以观察模型在事实核查中重点关注的三元组关系。例如在医学案例中，模型聚焦于 "(Dr. Erica Pan, is, California state epidemiologist)" 和 "(Dr. Erica Pan, recommended pause of, Moderna COVID-19 vaccine)" 等关键三元组，与核查需求一致。

## 亮点与洞察

1. **单次推理实现细粒度核查**：相比 FactScore（多次原子核查）、MiniCheck（逐对比较），GraphCheck 在单次调用中完成，效率大幅提升
2. **KG 增强的跨域泛化**：在通用数据上训练的 GNN 能泛化到医学领域，说明图结构推理是领域无关的能力
3. **GNN 作为 soft prompt**：不修改 LLM 参数，仅通过投影模块将图嵌入注入，方法灵活且可复用
4. **可解释性强**：通过边注意力权重可追溯模型的推理过程，在医学领域尤其重要
5. **KG 质量实验**：系统比较了不同模型提取 KG 的质量影响，为实际部署提供了指导

## 局限性

1. **KG 质量依赖**：提取 KG 的质量直接影响核查性能，但目前缺乏可靠的 KG 质量自动评估方法
2. **长声明（claim）处理不佳**：在 AggreFact-CNN 和 SummEval（平均声明长度 > 50）上表现较弱，因为长声明使三元组提取更困难
3. **训练数据限制**：仅使用 14K 合成样本，数据规模有限
4. **KG 提取的计算成本**：需要额外的 LLM 调用来提取三元组，增加了预处理开销

## 相关工作

- **幻觉检测方法**：RAG 方法（外部知识库验证）、LLM+grounding document 方法（直接核查）
- **长文本事实核查**：FactScore、MiniCheck、ACUEval 将文本分解为原子单元逐一核查
- **图方法**：GraphEval（逐三元组 NLI 评估，不利用全局图结构）、FactGraph（pre-LLM 方法）、AMRFact（AMR 图引导摘要生成）
- **G-Retriever**：图增强的检索和问答，本文发现直接将 KG 文本作为 prompt 效果有限

## 评分

⭐⭐⭐⭐（4/5）

方法思路清晰——用 KG 捕获长文本中的多跳关系，用 GNN 编码后作为 soft prompt 注入 LLM。单次推理核查是实际部署的重要优势。跨域泛化能力（通用→医学）令人印象深刻。不足在于对 KG 提取质量的强依赖和长声明场景的局限性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](../../ICML2025/graph_learning/on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[ACL 2025\] Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors](fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)
- [\[ACL 2025\] Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](can_graph_neural_networks_learn_language.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)

</div>

<!-- RELATED:END -->
