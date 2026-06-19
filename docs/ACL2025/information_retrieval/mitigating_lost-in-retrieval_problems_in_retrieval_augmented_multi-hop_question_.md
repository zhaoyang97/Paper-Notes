---
title: >-
  [论文解读] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA
description: >-
  [ACL 2025][信息检索/RAG][多跳问答] 本文识别 RAG 多跳问答中的"检索丢失"（lost-in-retrieval）问题——子问题分解后后续子问题因缺少关键实体导致检索性能骤降，提出 ChainRAG 框架通过构建句子图 + 渐进式检索 + 子问题重写（补全缺失实体）形成完整推理链，在 MuSiQue、2Wiki、HotpotQA 三个数据集上一致超越基线。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "多跳问答"
  - "RAG"
  - "子问题重写"
  - "句子图"
  - "实体补全"
---

# Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA

**会议**: ACL 2025  
**arXiv**: [2502.14245](https://arxiv.org/abs/2502.14245)  
**代码**: [GitHub](https://github.com/nju-websoft/ChainRAG)  
**领域**: 信息检索  
**关键词**: 多跳问答, RAG, 子问题重写, 句子图, 实体补全

## 一句话总结

本文识别 RAG 多跳问答中的"检索丢失"（lost-in-retrieval）问题——子问题分解后后续子问题因缺少关键实体导致检索性能骤降，提出 ChainRAG 框架通过构建句子图 + 渐进式检索 + 子问题重写（补全缺失实体）形成完整推理链，在 MuSiQue、2Wiki、HotpotQA 三个数据集上一致超越基线。

## 研究背景与动机

**关键发现："Lost-in-Retrieval"**：RAG 多跳 QA 通常将复杂问题分解为子问题依次检索，但**第二个子问题往往缺少明确实体**（用代词如"this author"替代），导致检索性能急剧下降。实证分析表明，在三个数据集上第二子问题的 Recall@2 平均比第一子问题低 18.29%。

**具体案例**：问题"《城市的隐忧》的作者的故乡是哪里？" → 子问题1"谁是《城市的隐忧》的作者？" → 子问题2"这位作者的故乡是哪里？"。子问题2 中"这位作者"缺乏具体实体，检索到无关文本，最终答案错误。

**现有方法局限**：迭代检索方法（Iter-RetGen、IRCoT）使用之前检索到的上下文构建后续查询，但未显式解决实体缺失问题；GraphRAG 等需要 LLM 抽取实体和关系构建知识图谱，成本高昂。

## 方法详解

### 整体框架

ChainRAG 包含四个阶段：

1. **句子图构建**：从文本中提取命名实体，构建以句子为节点、实体共现为边标签的句子图
2. **问题分解**：用 LLM 将多跳问题分解为子问题
3. **迭代处理**：对每个子问题执行 检索 → 回答 → 重写下一子问题 的循环
4. **答案整合**：汇总所有检索句子和子答案生成最终回答

### 关键设计

**1. 句子图构建（Sentence Graph with Entity Indexing）**

句子图是 ChainRAG 的核心数据结构，用于组织分散在不同文本中的知识并支持实体补全。

- **节点**：每个句子 $s_i$ 为一个节点
- **边**类型共三种：
    - **实体共现（EC）**：两个句子共享关键实体则连接。为减少冗余，只保留 BM25 重要性 top-$\alpha\%$（$\alpha=60$）的实体作为关键实体 $\mathcal{K}_i \subseteq \mathcal{E}_i$
    - **语义相似（SS）**：用 OpenAI text-embedding-3-small 计算句子嵌入，每个句子维护 top-$m$（$m=10$）最相似句子集合 $\mathcal{R}_i$，互相在对方 top-$m$ 中则连接
    - **结构邻接（SA）**：原文中距离 ≤ 3 个句子的添加边，帮助重建文本整体结构实现更宽泛检索

- **实体索引**：存储每个实体到其所有包含句子的映射，方便后续实体补全和检索扩展

**2. 种子句子检索 + 图扩展**

对每个子问题的检索分两步：

**Step A: 种子句子检索**
- 计算子问题嵌入与所有句子的相似度（矩阵乘法快速完成）
- 过滤低相似度句子缩小候选集
- 用 cross-encoder（BGE-Reranker）评估候选句子与子问题的相关性
- 选 top-$k$（$k=3$）句子作为种子

**Step B: 句子图上的检索扩展**
- 从种子句子出发，沿图边迭代扩展到邻居节点
- 每次扩展后用 LLM 判断是否已获取足够信息回答子问题，若是则停止
- 优化机制：初始扩展一次性获取所有一跳邻居（减少 LLM 调用）；设置总长度上限（防止上下文过长）

**3. 子问题重写（Sub-question Rewriting）**

这是解决"lost-in-retrieval"的核心步骤：

- **触发条件**：检查子问题是否包含代词（"this"、"it"、"they" 等）
- **重写过程**：将当前子问题 + 之前所有子问题及其答案输入 LLM，生成含具体实体的新子问题
- **例子**：子问题2 "In which region of S-Fone is this place located?" → 利用子问题1 答案 "Da Nang, Vietnam" → 重写为 "In which region of S-Fone is Da Nang, Vietnam located?"
- **特殊情况**：若之前子问题未被回答，则总结其对应上下文并整合到当前子问题的上下文中

**双重作用**：(1) 恢复检索性能；(2) 保留前序子问题的关键信息支持当前推理

**4. 答案整合的两种策略**

- **子答案整合（AnsInt）**：仅用各子问题及其答案推导最终答案。优势：每次只处理一个子问题的文本，不需要强长上下文能力。劣势：若某子问题答错则影响全局
- **子上下文整合（CxtInt）**：合并所有检索句子，去重 + 重排序后生成答案。优势：可缓解子问题答案错误的影响。劣势：需要 LLM 处理更长上下文，且噪声更多

### 损失函数 / 训练策略

ChainRAG 为**无训练方法（train-free）**，所有组件使用现成模型：
- 嵌入模型：OpenAI text-embedding-3-small
- Cross-encoder：BGE-Reranker
- 实体识别：spaCy
- LLM：GPT4o-mini / Qwen2.5-72B / GLM-4-Plus

## 实验关键数据

### 主实验：三个多跳 QA 数据集

| 方法 | MuSiQue F1 | MuSiQue EM | 2Wiki F1 | 2Wiki EM | HotpotQA F1 | HotpotQA EM |
|------|-----------|-----------|---------|---------|------------|------------|
| NaiveRAG | 29.82 | 19.00 | 50.61 | 42.50 | 56.92 | 42.00 |
| NaiveRAG w/ QD | 37.49 | 26.00 | 56.88 | 38.50 | 60.00 | 43.50 |
| Iter-RetGen | 38.41 | 33.00 | 58.43 | 50.50 | 57.77 | 42.00 |
| LongRAG | 44.88 | 32.00 | 62.39 | 49.00 | 64.74 | 51.00 |
| HippoRAG w/ IRCoT | 46.50 | 28.50 | 62.38 | 48.00 | 56.12 | 40.00 |
| **ChainRAG (AnsInt)** | **50.54** | 37.00 | 62.55 | 52.00 | 60.73 | 46.00 |
| **ChainRAG (CxtInt)** | 47.87 | **38.50** | 56.54 | 50.50 | **64.59** | **50.00** |

（以上为 GPT4o-mini 结果；Qwen2.5-72B 上 CxtInt 平均 F1 达 59.92，超越次优方法 HippoRAG 的 54.68 达 9.6%）

### 消融实验

| 消融项 | MuSiQue F1 | 2Wiki F1 | HotpotQA F1 |
|--------|-----------|---------|------------|
| ChainRAG (完整) | 50.54 | 62.55 | 60.73 |
| w/o 子问题重写 | ~35 (-30%) | ~55 | ~55 |
| w/o EC 边 | 略降 | 显著降 | 略降 |
| w/o SS 边 | 略降 | 略降 | 略降 |
| w/o SA 边 | 略降 | 略降 | 显著降 |
| w/o 句子图（chunk 模式） | 降低 | 降低 | EM 显著降 |

子问题重写移除后 MuSiQue 上 F1 和 EM 均下降约 30%，证实"lost-in-retrieval"问题的严重性。

### 检索性能验证

| 子问题 | MuSiQue Recall@2 | 2Wiki Recall@2 | HotpotQA Recall@2 |
|--------|-----------------|---------------|-------------------|
| 子问题 1 | 55.52 | 57.50 | 54.67 |
| 子问题 2 (原始) | 40.91 | 49.87 | 49.17 |
| 子问题 2 (重写后) | **58.81** | **54.32** | **61.83** |

重写后子问题 2 的 Recall 甚至超过子问题 1（利用了子问题 1 的信息辅助检索）。

### 关键发现

- **AnsInt vs CxtInt 取决于 LLM 能力**：GPT4o-mini（强推理）适合 AnsInt；Qwen2.5-72B、GLM-4-Plus（强长上下文）适合 CxtInt
- **问题分解有时帮助有限甚至有害**：NaiveRAG 加问题分解后 MuSiQue 上 GLM-4-Plus 性能反而下降，直接证实 lost-in-retrieval 问题
- **实体恢复错误比分解错误更致命**：错误分解的 F1 平均 55.43，错误实体恢复的 F1 平均 51.34，后者对性能伤害更大
- **spaCy 实体识别足够用**：换成 Qwen2.5-72B 做实体识别仅提升 ~1.5 F1，spaCy 在效率与效果间平衡最佳

## 亮点与洞察

1. **问题定义精准**："lost-in-retrieval"通过实证数据（18.29% Recall 下降）清晰量化，为多跳 QA 的检索失败提供了可操作的诊断框架
2. **句子图设计巧妙**：三种边类型（实体共现、语义相似、结构邻接）从不同维度组织信息，图上扩展实现了比传统 chunk 检索更细粒度且更全面的知识获取
3. **重写 > 构建 KG**：相比 GraphRAG/HippoRAG 需要 LLM 抽取实体关系构建知识图谱的高成本方案，子问题重写是一种更轻量的替代
4. **效率优势**：比 LongRAG 平均减少 17.3% LLM 调用次数，比 HippoRAG 少数倍

## 局限与展望

1. **迭代过程仍有额外开销**：虽然比 HippoRAG 高效得多，但相比 NaiveRAG 和 Iter-RetGen 仍需更多 LLM 调用
2. **领域适应性未验证**：三个数据集均为通用领域（维基百科），高度专业化场景（法律、医学）的效果未知
3. **实体识别错误传播**：错误的实体识别和恢复是最大性能瓶颈（79.3% 恢复准确率），且错误会在推理链中传播
4. **句子粒度可能过细**：在某些长文档场景下句子级别索引可能带来过多碎片化信息

## 相关工作与启发

- **RAG 迭代检索**：Iter-RetGen 用前次生成文本作下次查询；Self-RAG 包含自省机制；ChainRAG 独特之处在于显式补全缺失实体而非简单复用上下文
- **图结构 RAG**：RAPTOR（树结构）、GraphRAG/HippoRAG（LLM 构建 KG）成本较高；ChainRAG 的句子图仅需 spaCy + 嵌入模型，更轻量
- **启发**：(1) "lost-in-retrieval"是多跳 QA 的通用问题，任何使用问题分解的 RAG 系统都应考虑实体补全；(2) 句子图的三边设计可推广到其他细粒度检索任务

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 新颖性 | 8 | "lost-in-retrieval"问题定义有价值，句子图 + 重写的解决方案简洁有效 |
| 技术深度 | 7 | 句子图设计合理，两种整合策略覆盖不同 LLM 特性 |
| 实验充分性 | 8 | 3 数据集 × 3 LLM + 完整消融 + 检索分析 + 效率分析 + 错误分析 |
| 写作质量 | 8 | 问题动机通过实证分析建立，案例清晰有说服力 |
| 实用价值 | 8 | train-free，可直接集成，句子图 + 重写是低成本的通用改进 |
| 总分 | 7.8 | 问题定义精准，方案实用，实验全面 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](../../NeurIPS2025/information_retrieval/think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)
- [\[ACL 2025\] MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](main-rag_multi-agent_filtering_retrieval-augmented_generation.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](../../AAAI2026/information_retrieval/reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)

</div>

<!-- RELATED:END -->
