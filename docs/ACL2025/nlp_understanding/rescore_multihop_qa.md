# ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2505.21250](https://arxiv.org/abs/2505.21250)  
**代码**: [项目页](https://leeds1219.github.io/ReSCORE)  
**领域**: 信息检索 / 多跳问答  
**关键词**: multi-hop QA, dense retriever training, label-free supervision, iterative RAG, LLM distillation  

## 一句话总结

提出 ReSCORE，利用 LLM 生成的文档-问题相关性（relevance）和文档-答案一致性（consistency）的联合概率作为伪标签，在迭代 RAG 框架中无监督训练 dense retriever，在三个多跳 QA 数据集上达到 SOTA。

## 背景与动机

多跳问答（MHQA）需要跨多个文档进行推理，当前 SOTA 系统采用迭代检索增强生成（Iterative RAG）的范式。然而存在两个关键痛点：

1. **Dense retriever 需要标注数据**：虽然 dense retriever（如 Contriever）在语义匹配上优于 BM25，但它们需要标注的 query-document 对来微调。在 MHQA 场景下，每一步迭代的 query（重写后的问题）会因 LLM 不同而不同，标注成本极高。
2. **现有迭代 RAG 方法不训练 retriever**：IRCoT、Adaptive-RAG、Adaptive-Note 等方法在迭代推理方面做得很好，但都依赖预训练的稀疏检索器（BM25）或非领域适配的 dense retriever，没有在目标域上微调检索器。

## 核心问题

如何在没有标注文档相关性标签的情况下，为多跳问答场景训练一个有效的 dense retriever？

## 方法详解

### 整体框架

ReSCORE 在一个迭代 RAG 框架内工作：给定问题 $q$，系统迭代地检索文档、生成中间"thought"、重写 query，直到 LLM 给出最终答案（而非 "unknown"）。训练时，用 LLM 生成的概率分布作为伪标签监督 retriever，通过 KL 散度损失更新 query encoder。完整系统称为 IQATR（Iterative Question Answerer with Trained Retriever）。

### 关键设计

1. **Relevance-Consistency 联合伪标签生成**：核心公式为 $Q_{\text{LM}}^{(i)}(d_j^{(i)} | q) \propto P_{\text{LM}}(a, q | d_j^{(i)}) = P_{\text{LM}}(q | d_j^{(i)}) \cdot P_{\text{LM}}(a | q, d_j^{(i)})$。其中第一项 $P_{\text{LM}}(q|d)$ 衡量文档与问题的**相关性**（relevance），第二项 $P_{\text{LM}}(a|q,d)$ 衡量文档在回答问题时的**一致性**（consistency）。单独使用 consistency 会产生大量假阳性（表面词汇匹配但语义不相关的文档得高分），而 relevance 项可以有效过滤这些无关文档。

2. **KL 散度训练损失**：将 LLM 概率分布 $Q_{\text{LM}}$ 作为软标签，通过最小化 $D_{\text{KL}}(Q_{\text{LM}}^{(i)} \| P_R^{(i)})$ 来训练 retriever。$P_R$ 是基于 query-document 向量点积的 softmax 分布。仅训练 query encoder，冻结 document encoder。计算伪标签时只取 top-$M$=32 个文档以控制计算开销。

3. **迭代式 query 重构**：每次迭代生成一个 "thought"（对已检索文档的关键信息压缩），将其拼接到原始 query 上构成新 query（Thought-concat 策略）。这种方式在复杂问题上优于 LLM 直接重写 query，因为保留了原始问题不易丢失焦点。

4. **迭代训练机制**：训练不是单步完成的，而是在整个迭代 RAG 过程中进行。每个 iteration 的 query 不同（因为重写了），对应不同的检索文档集，允许 retriever 学会在后续迭代中检索到与前几轮互补的文档。

## 实验关键数据

| Dataset | Metric | IQATR (ReSCORE) | IRCoT (Prev SOTA) | Adaptive-RAG | Improvement |
|---------|--------|------------------|--------------------|--------------|-------------|
| MuSiQue | EM / F1 | **23.4** / **32.7** | 22.0 / 31.8 | 23.6 / 31.8 | +1.4 / +0.9 |
| HotpotQA | EM / F1 | **47.2** / **59.3** | 44.4 / 56.2 | 42.0 / 53.8 | +2.8 / +3.1 |
| 2WikiMHQA | EM / F1 | **50.0** / **59.7** | 49.7 / 54.9 | 40.6 / 49.8 | +0.3 / +4.8 |

注：上表 Prev SOTA 使用 Flan-T5-XL + BM25，IQATR 使用 Llama-3.1-8B + Contriever (ReSCORE)。

**Contriever 微调前后对比（同框架内）：**

| Dataset | Baseline (Contriever) | + ReSCORE | Δ EM / Δ F1 |
|---------|----------------------|-----------|-------------|
| MuSiQue | 15.2 / 23.8 | 23.4 / 32.7 | +8.2 / +8.9 |
| HotpotQA | 39.4 / 52.3 | 47.2 / 59.3 | +7.8 / +7.0 |
| 2WikiMHQA | 32.8 / 41.6 | 50.0 / 59.7 | +17.2 / +18.1 |

### 消融实验要点

- **Pseudo-GT label 类型对比**（Table 3，单步 reranking）：仅用 $P(q|d)$（relevance）平均提升 recall 5.37%；仅用 $P(a|q,d)$（consistency）反而**下降 23.8%**（假阳性严重）；两者联合 $P(q,a|d)$ 提升 14.4%。
- **Pseudo-GT vs GT 标签**（Table 4）：令人惊讶的是，ReSCORE 的伪标签**优于人工 GT 标签**。原因是 GT 标签在单步训练中要求 query 同时对齐多个距离很远的文档（如 "Billie Eilish"、"Avocado"、"Mexico Presidents"），query encoder 被拉向这些文档的质心，不利于检索任何单个文档。而 ReSCORE 通过迭代过程逐步检索互补文档。
- **Query 重构策略**（Table 5）：Thought-concat 在复杂问题（MuSiQue、HotpotQA，平均 17+ tokens）上更优；LLM-rewrite 在简单问题（2WikiMHQA，11.7 tokens）上略优。原因是 LLM 重写复杂 query 时容易丢失焦点。

## 亮点

- **无需文档标注**的 dense retriever 训练方法，巧妙利用 LLM 概率信号作为伪标签
- Relevance + Consistency 联合建模的思路，解决了单独用 consistency 导致的假阳性问题
- 伪标签竟然优于人工 GT 标签，揭示了 GT 标签在多跳场景下作为训练信号的局限性
- ReSCORE 可以作为插件提升多种现有迭代 RAG 框架（Self-RAG、FLARE、Adaptive-Note）的性能
- 统计显著性测试充分（10 seeds, t-test, p < 0.05）

## 局限性 / 可改进方向

- **泛化性不足**：模型在特定数据集上微调，跨数据集（不同推理模式、不同 hop 数）的 OOD 泛化能力有限
- **计算开销**：迭代检索过程增加了延迟和计算成本，每次迭代都需要 LLM 推理
- 只训练 query encoder、冻结 document encoder，这限制了 retriever 的适配能力上限
- 依赖答案作为伪标签信号，对于无法轻易获得参考答案的场景适用性降低
- 最大迭代次数固定为 6，对于需要更多跳数的极复杂问题可能不足

## 与相关工作的对比

| 方法 | Retriever | 训练方式 | 迭代 | MHQA 适配 |
|------|-----------|---------|------|----------|
| ATLAS | Dense | LLM consistency 蒸馏 | ✗ | 单跳 |
| REPLUG | Dense | LLM consistency | ✗ | 单跳 |
| IRCoT | BM25 | 无训练 | ✓ | ✓ 但不训练 retriever |
| Adaptive-RAG | BM25 | 无训练（训分类器） | ✓ | ✓ 但不训练 retriever |
| **ReSCORE** | Dense | LLM relevance + consistency | ✓ | ✓ 迭代训练 |

与 ATLAS/REPLUG 的关键区别：ReSCORE 同时建模 relevance 和 consistency（而不仅是 consistency），且在迭代框架中训练（而非单步）。与 IRCoT/Adaptive-RAG 的区别：ReSCORE 实际训练了 retriever 而不只是使用预训练版本。

## 启发与关联

1. "LLM 概率作为软监督信号"的思路很有价值，可以拓展到其他需要无标注训练的检索场景（如对话检索、多轮搜索）
2. Pseudo-GT 优于 GT 的发现暗示：在多步推理任务中，硬标签可能不如迭代式软标签有效，值得在其他领域验证
3. Relevance + Consistency 的分解框架提供了一种可解释的文档评估方式，可用于 RAG 系统的文档质量评估

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心 idea（relevance+consistency 联合伪标签 + 迭代训练 retriever）有明确创新点，但各组件（LLM 蒸馏、迭代 RAG）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多个消融维度、与多种方法交叉对比、统计显著性测试，非常扎实
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、公式推导流畅、例子生动（FIFA World Cup 的假阳性例子），minor issue 是 table 较多
- 对我的价值: ⭐⭐⭐⭐ 对 RAG 系统中 retriever 训练方法有参考价值，尤其是 "伪标签优于 GT" 的发现和 relevance-consistency 解耦思路
