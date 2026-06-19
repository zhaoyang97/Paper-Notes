---
title: >-
  [论文解读] GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering
description: >-
  [ACL 2025][信息检索/RAG][法律问答] 提出GRAF算法，结合法律知识图谱（Law-RoG）和图注意力网络进行罗马尼亚语法律多选题问答，同时开源了首个罗马尼亚语法律MCQA数据集JuRO（10,836题）和法律语料库CROL。 领域现状： 法律问答(Legal QA)是NLP的新兴应用领域…
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "法律问答"
  - "知识图谱"
  - "图注意力网络"
  - "罗马尼亚语"
  - "低资源NLP"
---

# GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering

**会议**: ACL 2025  
**arXiv**: [2412.04119](https://arxiv.org/abs/2412.04119)  
**代码**: [github.com/craciuncg/GRAF](https://github.com/craciuncg/GRAF)  
**领域**: 信息检索  
**关键词**: 法律问答, 知识图谱, 图注意力网络, 罗马尼亚语, 低资源NLP  

## 一句话总结

提出GRAF算法，结合法律知识图谱（Law-RoG）和图注意力网络进行罗马尼亚语法律多选题问答，同时开源了首个罗马尼亚语法律MCQA数据集JuRO（10,836题）和法律语料库CROL。

## 研究背景与动机

**领域现状**: 法律问答(Legal QA)是NLP的新兴应用领域。英语已有PrivacyQA、JEC-QA等多个数据集，但罗马尼亚语尽管有jurBERT等预训练模型，却缺乏公开的法律问答数据集。多语言和低资源语言的法律NLP仍处于早期阶段。

**现有痛点**: 罗马尼亚语的NLP资源极度匮乏——没有公开的法律MCQA数据集、没有结构化的法律语料库、没有法律知识图谱。此前Masala等(2021, 2024)的工作虽然提出了jurBERT和司法预测系统，但数据集均不公开。

**核心矛盾**: 标准的检索增强生成(RAG)方法在法律领域表现不稳定——它在某些法律分支上表现良好但在其他分支上泛化差；微调LLM容易产生幻觉。传统的基于encoder的方法在处理需要跨文档法律推理的复杂问题时能力不足。

**本文目标**: 为罗马尼亚语法律问答提供完整的数据资源和高效的方法论。

**切入角度**: 构建法律知识图谱，将每个答案选项与知识图谱中的事实进行图级别的对齐匹配，而非传统的文本检索方法。

**核心 idea**: 通过LLM提取问题-选项对中的声明图(Claim Graph)，与法律知识子图用GAT编码后计算余弦对齐分数，实现基于事实验证的MCQA。

## 方法详解

### 整体框架

GRAF对每个问题的每个选项执行以下流程：
1. **Cross Claim Extraction**: 用LLM将(问题, 选项)对分解为声明图（实体-关系三元组）
2. **KG Sampling**: 从Law-RoG中通过BM25检索+BFS采样获取相关子图
3. **Graph Encoding**: 用改进的GAT编码声明图和知识子图
4. **Alignment & Scoring**: 通过余弦相似度计算声明与知识的对齐，经自注意力后输出选项得分

### 关键设计

**声明图构建**: 使用LLM（通过few-shot prompting）将问题+选项分解为(实体, 关系, 实体)三元组。例如，法律问题"自卫的情况下是否排除刑事责任?"可分解为实体"自卫"和"刑事责任排除"之间的关系。

**知识图谱采样**: 对Law-RoG（160K节点，320K边）使用分支过滤+BM25检索top-10实体+深度1的BFS扩展，最多选取50个不同实体。采用SpaCy进行罗马尼亚语分词和词形还原。

**改进的GAT编码**: 原始GAT仅处理无关系的图。本文扩展GAT以同时捕获节点间和边的拓扑信息：

节点注意力: $e_N^{ij} = \sigma_A(a_N^T [W_N h_N^i \| W_N h_N^j])$

边注意力: $e_E^{ij} = \sigma_A(a_E^T [W_N h_N^i \| W_E h_E^j])$

最终节点表示融合两者: $h' = h'_N + h'_E$

**对齐计算与最终评分**:

$$R^{ij} = \cos(h_c^i, h^j)$$

聚合知识表示: $\bar{H} = Rh$

通过自注意力机制融合选项编码 $\bar{c}$ 和知识表示 $\bar{H}$，经全连接层+sigmoid输出选项概率。

### 损失函数/训练策略

- 单选题使用交叉熵损失，多选题采用二元交叉熵
- 语言编码器使用jurBERT（罗马尼亚语法律预训练）
- Law-RoG知识图谱通过LLM的few-shot实体关系抽取构建，5位NLP专家验证质量

## 实验关键数据

### 主实验 - 晋升考试 (单选, Accuracy%)

| 方法 | 民法 | 刑法 | 民诉 | 刑诉 | 行政 | 劳动 | 家庭 | 国际 | 商法 | 平均 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| QBERT | 35.5 | 38.3 | 35.8 | 40.1 | 36.6 | 40.4 | 38.2 | 39.3 | 39.4 | 38.2 |
| ColBERT | 44.1 | 38.3 | 47.8 | 37.5 | 41.6 | 42.4 | 41.2 | 48.8 | 40.4 | 42.5 |
| LLM | 48.9 | 40.2 | 41.0 | 42.3 | 46.5 | 48.5 | 49.0 | 52.4 | 39.4 | 45.4 |
| LLM RAG | 53.2 | 43.8 | 38.8 | 42.8 | 57.4 | 56.6 | 63.7 | 52.4 | 57.6 | 51.8 |
| LLM LFT | 45.7 | 51.8 | 74.6 | 49.0 | 52.5 | 49.5 | 54.9 | 70.2 | 53.5 | 55.8 |
| **GRAF(本文)** | **49.5** | **52.7** | **78.5** | **51.1** | **59.2** | **57.3** | **68.7** | **67.1** | **56.8** | **60.1** |

### 入学考试 (多选, Accuracy%)

| 方法 | 民法 | 刑法 | 民诉 | 刑诉 | 平均 |
|------|:---:|:---:|:---:|:---:|:---:|
| ColBERT | 45.1 | 43.1 | 51.0 | 41.2 | 45.1 |
| LLM RAG | 43.1 | 27.5 | 21.6 | 27.5 | 29.9 |
| **GRAF(本文)** | **60.8** | **62.8** | **54.9** | **56.9** | **58.8** |

### 关键发现

1. **GRAF在9个法律分支中6个取得最佳**: 平均60.1%的准确率，比最强基线LLM LFT高4.3个百分点
2. **图方法 vs RAG**: GRAF比LLM RAG高8.3个百分点(60.1 vs 51.8)，说明结构化知识表示优于纯文本检索
3. **多选题优势更大**: 入学考试中GRAF(58.8%)远超ColBERT(45.1%)和LLM RAG(29.9%)
4. **法律专用模型的重要性**: 预训练的法律领域模型显著优于通用模型（如Figure中所示）
5. **LLM间的低一致性**: APPA在40-50%之间，Fleiss' κ略负，说明法律QA对LLM仍极具挑战性
6. **RAG在特定分支不稳定**: LLM RAG在民事诉讼法(38.8%)表现差，但在行政法(57.4%)表现好

## 亮点与洞察

- **四重资源贡献**: JuRO数据集(10,836题) + CROL法律语料(330K条文, 31.5M词) + Law-RoG知识图谱(160K节点, 320K边) + GRAF算法
- **符号+数值方法的协同**: 将知识图谱的符号推理与向量空间的数值匹配结合，两种范式互补
- **Cross Claim Extraction的巧思**: 将选项分解为可验证的声明，转化为事实核查问题
- **适应低资源场景**: 整个方法论专为资源匮乏的语言设计，可移植到其他低资源语言的法律NLP

## 局限与展望

- 最佳平均准确率仅60%，距离实际部署的可靠性要求还有较大差距
- Law-RoG通过LLM生成，虽经人工验证但可能存在遗漏
- 单一法律分支的训练数据可能不足
- 未探索数据增强（如用LLM扩展训练集）的可能性
- KG采样的BFS深度和top-k参数的敏感性分析不充分
- 对于跨法律分支的问题（涉及多个知识图谱），方法的处理方式不明确

## 相关工作与启发

- Edge等(2024)的GraphRAG思想是本文Claim Graph构建的直接灵感来源
- Zhong等(2020)的JEC-QA中文法律问答数据集(26,367题)提供了规模参考
- COLIEE共享任务（日本法律QA，始于2014）是法律NLP领域最早的系统化评测
- He等(2022)的医疗图谱+对话方法展示了KG-based QA在垂直领域的潜力
- 对低资源语言NLP的启示：先构建领域知识图谱，再用图方法弥补数据不足

## 评分

- **新颖性**: ⭐⭐⭐⭐ — GRAF算法将图对齐应用于MCQA是新颖的，同时贡献了多个原创数据资源
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖9个法律分支、3种考试类型、6个基线方法，分析详实
- **写作质量**: ⭐⭐⭐⭐ — 资源描述和方法阐述清晰，算法伪代码规范
- **价值**: ⭐⭐⭐⭐⭐ — 资源资源贡献(JuRO+CROL+Law-RoG)的长期价值极高，为罗马尼亚语及低资源语言法律NLP奠定基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry](comrag_retrieval-augmented_generation_with_dynamic_vector_stores_for_real-time_c.md)
- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](../../AAAI2026/information_retrieval/n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)

</div>

<!-- RELATED:END -->
