---
title: >-
  [论文解读] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models
description: >-
  [ICML2025][图学习][知识图谱问答] 提出 Graph-constrained Reasoning (GCR)，通过将知识图谱编码为 KG-Trie 并嵌入 LLM 解码过程，实现零幻觉的忠实推理，在 KGQA 基准上达到 SOTA 且具备零样本跨图谱迁移能力。 LLM 在推理任务上表现优异…
tags:
  - "ICML2025"
  - "图学习"
  - "知识图谱问答"
  - "LLM推理"
  - "受限解码"
  - "Trie索引"
  - "幻觉消除"
---

# Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models

**会议**: ICML2025  
**arXiv**: [2410.13080](https://arxiv.org/abs/2410.13080)  
**代码**: [RManLuo/graph-constrained-reasoning](https://github.com/RManLuo/graph-constrained-reasoning)  
**领域**: 图学习  
**关键词**: 知识图谱问答, LLM推理, 受限解码, Trie索引, 幻觉消除

## 一句话总结

提出 Graph-constrained Reasoning (GCR)，通过将知识图谱编码为 KG-Trie 并嵌入 LLM 解码过程，实现零幻觉的忠实推理，在 KGQA 基准上达到 SOTA 且具备零样本跨图谱迁移能力。

## 研究背景与动机

LLM 在推理任务上表现优异，但在需要事实知识的场景中仍面临两大核心问题：**知识缺失**和**幻觉**。现有利用知识图谱 (KG) 增强 LLM 推理的方法分为两类：

- **检索式 (Retrieval-based)**：如 RoG、GNN-RAG，通过外部检索器从 KG 抽取相关事实输入 LLM。问题在于检索器精度有限，难以建模图结构，且对未见问题泛化不足。
- **代理式 (Agent-based)**：如 ToG、EffiQA，将 LLM 作为智能体与 KG 多轮交互。问题在于计算开销大、延迟高（ToG 平均需 11.6 次 LLM 调用）。

关键发现：即使是领先的 KG 增强方法 RoG，在图谱推理中仍有 **33% 的幻觉错误**（Sui et al., 2024）。这意味着 LLM 生成的推理路径并非真正锚定在 KG 上，可信度存疑。

**核心动机**：能否将 KG 的结构化约束直接融入 LLM 的解码过程，从根本上消除推理幻觉？

## 方法详解

GCR 框架包含三个核心组件：KG-Trie 构建、图约束解码、图归纳推理。

### 1. KG-Trie 构建

**目标**：将 KG 转化为可约束 LLM 解码的结构化索引。

给定问题 $q$ 和 KG $\mathcal{G}$，从问题实体 $\mathcal{E}_q$ 出发，用 BFS 检索 $L$ 跳内的所有推理路径，格式化为文本后用 LLM tokenizer 分词，构建前缀树：

$$\mathcal{W}_z = \text{BFS}(\mathcal{G}, \mathcal{E}_q, L)$$

$$\mathcal{T}_z = \text{Tokenizer}(\mathcal{W}_z)$$

$$\mathcal{C}_\mathcal{G} = \text{Trie}(\mathcal{T}_z)$$

**优势**：KG-Trie 支持恒定时间 $O(|\mathcal{W}_z|)$ 的路径遍历；可离线预构建也可按需实时构建；2-hop 平均构建时间仅 0.28 秒，空间 0.5 MB。

### 2. 图约束解码 (Graph-constrained Decoding)

核心思想：在 LLM 解码时用 KG-Trie 约束 token 生成，确保输出的推理路径必定存在于 KG 中。

生成概率建模为：

$$P_\phi(a, \boldsymbol{w}_z | q) = \underbrace{P_\phi(a|q, \boldsymbol{w}_z)}_{\text{常规解码}} \cdot \underbrace{\prod_{i=1}^{|\boldsymbol{w}_z|} P_\phi(w_{z_i}|q, w_{z_{1:i-1}}) \cdot \mathcal{C}_\mathcal{G}(w_{z_i}|w_{z_{1:i-1}})}_{\text{图约束解码}}$$

约束函数为硬约束（0/1 掩码）：

$$\mathcal{C}_\mathcal{G}(w_{z_i}|w_{z_{1:i-1}}) = \begin{cases} 1, & \exists \text{prefix}(w_{z_{1:i}}, \boldsymbol{w}_z), \boldsymbol{w}_z \in \mathcal{W}_z \\ 0, & \text{otherwise} \end{cases}$$

在生成推理路径（`<PATH>...</PATH>` 标记之间）时施加 Trie 约束，路径生成完毕后切回常规解码生成假设答案。

**训练**：微调一个轻量 KG 专用 LLM（如 Qwen2-0.5B 至 Llama-3.1-8B），损失函数为标准自回归：

$$\mathcal{L} = \mathbb{E}_{(q, \boldsymbol{w}_z, a) \sim \mathcal{D}_\mathcal{G}} \left[ \log \prod_{i} P_\phi(a_i|q, \boldsymbol{w}_z, a_{1:i-1}) \prod_{j} P_\phi(w_{z_j}|q, w_{z_{1:j-1}}) \right]$$

训练数据：用问题实体和答案实体之间的最短路径作为推理路径。WebQSP 训练集 28,307 条，CWQ 训练集 181,602 条。

### 3. 图归纳推理 (Graph Inductive Reasoning)

利用 beam search 同时生成 $K$ 条推理路径及假设答案，然后输入一个强力通用 LLM（如 ChatGPT、GPT-4o-mini）进行归纳推理，产出最终答案：

$$\mathcal{Z}_K = \{a_k, \boldsymbol{w}_{z_k}\}_{k=1}^K = \arg\text{top-}K \, P_\phi(a, \boldsymbol{w}_z | q)$$

$$P_\theta(\mathcal{A}|q, \mathcal{Z}_K) \simeq \prod_{k=1}^K P_\theta(\mathcal{A}|q, a_k, \boldsymbol{w}_{z_k})$$

采用 FiD 架构，在一次 LLM 调用中处理所有路径，无需额外微调通用 LLM。

## 实验关键数据

### 主实验：KGQA 推理性能 (Table 1)

| 方法 | WebQSP Hit | WebQSP F1 | CWQ Hit | CWQ F1 |
|------|-----------|-----------|---------|--------|
| ChatGPT+Self-Consistency | 83.5 | 63.4 | 56.0 | 48.1 |
| RoG (Llama-2-7B) | 85.7 | 70.8 | 62.6 | 56.2 |
| GNN-RAG+RA | 90.7 | 73.5 | 68.7 | 60.4 |
| **GCR (Llama-3.1-8B + ChatGPT)** | **92.6** | 73.2 | 72.7 | 60.9 |
| **GCR (Llama-3.1-8B + GPT-4o-mini)** | 92.2 | **74.1** | **75.8** | **61.7** |

GCR 在两个数据集上均 SOTA，WebQSP Hit 超第二名 2.1%，CWQ Hit 超第二名 9.1%。

### 效率对比 (Table 2, WebQSP)

| 方法 | Hit | 平均推理时间 (s) | 平均 LLM 调用次数 | 平均输入 Token 数 |
|------|-----|----------------|-----------------|-----------------|
| RoG | 85.7 | 2.60 | 2 | 521 |
| ToG (Agent) | 75.1 | 16.14 | 11.6 | 7,069 |
| **GCR** | **92.6** | 3.60 | 2 | **231** |

GCR 仅需 2 次 LLM 调用、231 个输入 token，效率远超代理式方法。

### 幻觉消除

- GCR **在 WebQSP 和 CWQ 上均实现 100% 忠实推理率**（推理路径完全可在 KG 中找到）。
- 移除 KG 约束后，CWQ 忠实推理率降至 48.1%，验证了 Trie 约束的必要性。

### 零样本跨图谱迁移 (Table 6)

| 模型 | FreebaseQA | CSQA | MedQA |
|------|-----------|------|-------|
| GPT-4o-mini | 89 | 91 | 75 |
| **GCR (GPT-4o-mini)** | **94** | **94** | **79** |

无需额外训练，直接接入新 KG（ConceptNet、医学 KG）的 Trie，FreebaseQA 提升 +5%，CSQA 提升 +3%。

### KG 专用 LLM 消融 (Table 4, WebQSP)

微调后的 Qwen2-0.5B 已达 Hit 87.5，超越零样本 Llama-3.1-70B 的 38.5，证明微调对图推理能力的关键作用。最佳 beam size $K=10$，最佳跳数 $L=2$。

## 亮点与洞察

1. **范式创新**：区别于检索式和代理式，GCR 开辟了"约束解码"的第三条路线——将图结构直接嵌入 LLM 解码过程，这是一个优雅且高效的设计。
2. **双 LLM 协作架构**：轻量 KG 专用 LLM（图推理） + 强力通用 LLM（归纳推理），各取所长，且通用 LLM 无需微调。
3. **零幻觉保证**：通过硬约束掩码实现理论层面的零幻觉，而非软约束或后验过滤。
4. **即插即用的跨图谱迁移**：更换 KG-Trie 即可迁移到新图谱，体现了方法的模块化优势。

## 局限与展望

1. **零幻觉的定义局限**：仅保证推理路径在 KG 中存在，但 KG 本身可能不完整或包含错误事实，无法检测 KG 固有的假阳性。
2. **复杂多跳问题的扩展性**：当 $L$ 增大时 KG-Trie 构建开销呈指数增长（$O(E^L)$），3 跳以上效率显著下降。论文建议与规划分解方法结合，但未实际验证。
3. **无关推理路径问题**：LLM 有时选择图中存在但与问题无关的路径（如 Case 1 中询问选区却走向政治职位路径），导致错误答案。
4. **对 KG 覆盖度的依赖**：KG 不完整时（如角色-演员关系缺失），方法完全失效。
5. **通用 LLM 成本**：归纳推理依赖 ChatGPT/GPT-4o-mini 等闭源模型，部署成本和隐私问题值得关注。

## 相关工作与启发

- **RoG** (Luo et al., ICLR 2024)：GCR 的前作，规划-检索-推理框架，但仍有 33% 幻觉。
- **GNN-RAG** (Mavromatis & Karypis, 2024)：用 GNN 做图检索，GCR 与之互补——GNN-RAG 输出可作为 KG-Trie 的子图输入。
- **Constrained Decoding** (De Cao et al., ICLR 2022)：GCR 的 Trie 约束解码直接继承自 GENRE 的自回归实体检索思路。
- **启发**：约束解码思路可推广到其他结构化数据（如数据库 schema、代码 AST），实现结构感知的忠实生成。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 将 Trie 约束解码引入 KG 推理是一个巧妙的跨领域迁移
- 实验充分度: ⭐⭐⭐⭐⭐ — 5 个数据集、22 个基线、消融/效率/幻觉/迁移全面覆盖
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，动机阐述充分，附录详尽
- 价值: ⭐⭐⭐⭐ — 零幻觉+高效率+零样本迁移，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](../../ACL2025/graph_learning/fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[ACL 2025\] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](../../ACL2025/graph_learning/ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)

</div>

<!-- RELATED:END -->
