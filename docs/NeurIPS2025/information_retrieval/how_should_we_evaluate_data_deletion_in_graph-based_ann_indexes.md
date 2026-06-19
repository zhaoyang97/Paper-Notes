---
title: >-
  [论文解读] How Should We Evaluate Data Deletion in Graph-Based ANN Indexes?
description: >-
  [NeurIPS 2025 (Workshop on ML for Systems)][信息检索/RAG][ANNS] 针对图基ANN索引缺乏统一数据删除评估方法的问题，形式化定义了逻辑删除、物理删除和重建三种基准方法，提出面向实际部署的评估框架和指标体系，并基于实验分析提出Deletion Control算法在精度约束下动态切换删除策略。
tags:
  - "NeurIPS 2025 (Workshop on ML for Systems)"
  - "信息检索/RAG"
  - "ANNS"
  - "HNSW"
  - "数据删除"
  - "向量数据库"
  - "评估框架"
---

# How Should We Evaluate Data Deletion in Graph-Based ANN Indexes?

**会议**: NeurIPS 2025 (Workshop on ML for Systems)  
**arXiv**: [2512.06200](https://arxiv.org/abs/2512.06200)  
**代码**: 无  
**领域**: 信息检索 / 近似最近邻搜索  
**关键词**: ANNS, HNSW, 数据删除, 向量数据库, 评估框架

## 一句话总结

针对图基ANN索引缺乏统一数据删除评估方法的问题，形式化定义了逻辑删除、物理删除和重建三种基准方法，提出面向实际部署的评估框架和指标体系，并基于实验分析提出Deletion Control算法在精度约束下动态切换删除策略。

## 研究背景与动机

**领域现状**：近似最近邻搜索（ANNS）是RAG、推荐系统等应用的核心构建块。在这些应用中，数据频繁更新——新品上架、过期商品下架、文档更新等，因此需要ANNS算法支持高效的数据删除操作。图基方法（如HNSW）是当前最主流的ANNS算法之一。

**现有痛点**：现有对ANNS数据删除的研究存在三个关键问题。首先，实验设置不切实际——如FreshDiskANN和MN-RU的实验中会重新插入已删除的数据，这在实际应用中几乎不会发生。其次，评估指标不够全面——大多数工作只关注删除后的搜索精度，但实际部署还需考虑删除延迟、插入吞吐量、内存占用等。第三，缺乏对不同删除方法的统一和形式化的比较框架。

**核心矛盾**：删除操作的速度、搜索精度维持和内存效率三者之间存在根本性的trade-off——逻辑删除最快但精度退化，物理删除居中但破坏图结构，重建能恢复精度但代价极高。如何在实际部署中选择合适的策略缺乏理论指导。

**本文目标** (1) 建立图基ANNS数据删除的统一形式化框架；(2) 设计面向实际用例的评估协议和指标体系；(3) 提出根据精度要求自适应选择删除策略的算法。

**切入角度**：将三种删除方法用统一的数学语言形式化，在相同的实验条件下公平比较，基于实验观察到的规律（物理删除精度收敛到稳定值、逻辑删除精度线性退化）设计动态切换算法。

**核心 idea**：通过形式化定义和标准化评估揭示三种删除方法的特性，并利用这些特性设计精度感知的自适应删除控制策略。

## 方法详解

### 整体框架

工作分为三部分：(1) 形式化三种删除方法的伪代码和数学定义；(2) 设计包含多维度指标的评估框架；(3) 基于实验发现提出Deletion Control算法。图基ANN索引用节点集 $\mathcal{P}$ 和邻居集 $\mathcal{N}$ 表示，删除操作定义为 $DELETE: (\mathcal{D}, \mathcal{P}, \mathcal{N}) \mapsto (\mathcal{P}', \mathcal{N}')$。

### 关键设计

1. **三种删除方法的形式化**:

    - 功能：为图基ANNS的数据删除建立统一的形式框架
    - 核心思路：(a) 逻辑删除——不修改图结构，仅维护删除标记集 $\mathcal{F}$，搜索时从结果中排除被标记节点 $\mathcal{R} \leftarrow SEARCH(\mathbf{q}, \mathcal{P}, \mathcal{N}) \setminus \mathcal{F}$；(b) 物理删除——真正从图中移除节点，删除所有连接边，更新邻居节点的邻居列表 $\mathcal{N}_i \leftarrow \mathcal{N}_i \setminus \mathcal{D}$，释放内存；(c) 重建——移除待删数据后用剩余数据重新构建整个图 $\mathcal{N} \leftarrow CONSTRUCT(\mathcal{P})$
    - 设计动机：先前工作各自使用不同的实现和定义，缺乏可比性。形式化使得方法间的差异一目了然——逻辑删除保留图结构但浪费内存、物理删除清理内存但破坏连通性、重建恢复最优图但代价昂贵

2. **面向部署的评估指标体系**:

    - 功能：全面评估删除操作对ANNS系统的影响
    - 核心思路：使用四维度指标——(a) 搜索精度：1-Recall@10；(b) 搜索速度：QPS-search；(c) 删除速度：QPS-delete；(d) 插入速度：QPS-add。用QPS-Recall曲线同时考量速度和精度的trade-off。实验设置为反复执行相同batch size的插入和删除，每轮重新计算ground truth
    - 设计动机：现有评估只看删除后搜索精度，忽略了删除本身的效率和对后续操作的影响

3. **Deletion Control算法**:

    - 功能：在精度约束下自动选择最优删除策略
    - 核心思路：引入两个关键参数——$\theta$（多次物理删除后精度收敛的下限值）和 $\pi$（逻辑删除在精度降至目标 $\alpha$ 以下前的最大步数）。估计方式：$\theta$ 取物理删除实验中精度的最低值；$\pi$ 利用逻辑删除精度的线性退化特性估算 $\pi \approx (\alpha - R_0) / \Delta$。根据 $\alpha$ 与 $\theta$ 的关系选择策略：若 $\alpha < \theta$，只用物理删除（精度不会降到要求以下）；若 $\alpha \geq \theta$，交替执行 $\pi$ 步逻辑删除和一次重建（周期性恢复精度）
    - 设计动机：纯物理删除精度有下限、纯逻辑删除精度持续退化、纯重建太慢。混合策略利用逻辑删除的速度优势和重建的精度恢复能力

## 实验关键数据

### 主实验

SIFT1M数据集上三种删除方法的搜索性能比较（step 5后）：

| 删除方法 | 搜索精度(1-Recall@10) | QPS-delete (b=10⁵) | 内存效率 |
|---------|---------------------|---------------------|---------|
| 重建 | 最高 | 最低 | 中 |
| 物理删除 | 中 | 中 | 最优 |
| 逻辑删除 | 最低 | **最高** | 最差 |

SIFT1B数据集上不同batch size的删除速度：

| batch size | 逻辑删除QPS | 物理删除QPS | 重建QPS |
|-----------|------------|------------|---------|
| b=10⁵ | 最快 | 中 | 最慢 |
| b=10³ | 最快 | 中 | 比b=10⁵时更慢（相对于物理删除） |

### 消融实验

Deletion Control验证（SIFT1B, b=10⁵, α=0.84）：

| 策略 | 精度维持 | 总删除时间 |
|------|---------|-----------|
| 仅物理删除 | θ=0.816 < α=0.84, 不满足 | - |
| 仅逻辑删除 | 快速退化，不满足 | - |
| Deletion Control (逻辑+重建交替) | ≈满足 α=0.84 要求 | 所有满足精度要求的策略中**最小** |

### 关键发现

- 物理删除的精度退化不是无限的——经过多轮删除+插入后收敛到一个稳定值 $\theta$，这是一个重要的实验发现
- 逻辑删除的精度退化近似线性，可用 $\Delta = (R_S - R_0)/S$ 精确预测
- 高频小batch删除（b=10³）时重建的相对开销更大，此时物理删除更有优势
- Deletion Control仅需用10%的查询作为训练集即可准确估计 $\theta$ 和 $\pi$

## 亮点与洞察

- **评估框架的实用价值**：首次为图基ANNS删除建立了统一、现实的评估方法论。先前工作的实验设置（重新插入已删数据）掩盖了真实场景中的问题，本文的迭代插入-删除协议更贴近实际
- **物理删除精度收敛的发现很有趣**：直觉上物理删除会持续破坏图结构导致精度不断下降，但实验发现精度会收敛到稳定值。这意味着在精度要求不太严格的场景下，物理删除是一个又快又省内存的好选择
- **Deletion Control的设计简洁有效**：仅需两个可估计参数（$\theta$, $\pi$）就能自适应选择策略，容易在实际系统中部署

## 局限与展望

- 作为Workshop论文，实验规模相对有限——主要在HNSW上验证，未测试其他图基索引（如DiskANN、NSG等）
- Deletion Control假设逻辑删除精度线性退化，这在极端数据分布下可能不成立
- 未考虑更复杂的删除方法（如FreshDiskANN的边重连、IPGM的邻居重计算），只对比了三种基础方法
- 内存占用的定量分析不够深入
- 未探索与硬件特性（如SSD vs RAM）相关的性能差异
- 删除与查询并发执行的性能影响未讨论

## 相关工作与启发

- **vs FreshDiskANN**: FreshDiskANN在物理删除时重连邻居边以维持搜索质量，但其评估中重新插入已删数据的设置不现实。本文的评估框架更适合衡量此类高级方法的真实效果
- **vs MN-RU（HNSW-update）**: MN-RU通过维护备份图解决不可达节点问题，代价是额外内存。本文发现基础物理删除的精度收敛到稳定值，这提出了一个问题：在什么精度要求下值得付出MN-RU的额外开销？
- **vs Ada-IVF/SPFresh**: 这些基于IVF结构的方法通过重新分配向量到聚类来处理更新，与图基方法的删除机制完全不同，但评估框架的思路可以借鉴

## 评分

- 新颖性: ⭐⭐⭐ 形式化和评估框架有价值但技术创新有限，Deletion Control算法较为简单
- 实验充分度: ⭐⭐⭐ 多数据集验证但仅在HNSW上测试，Workshop论文篇幅限制
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，形式化严谨简洁
- 价值: ⭐⭐⭐⭐ 填补了ANNS删除评估的空白，对向量数据库工程实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](../../ICML2026/information_retrieval/seeing_to_generalize_how_visual_data_corrects_binding_shortcuts.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[ACL 2025\] When Should Dense Retrievers Be Updated in Evolving Corpora? Detecting Out-of-Distribution Corpora Using GradNormIR](../../ACL2025/information_retrieval/when_should_dense_retrievers_be_updated_in_evolving_corpora_detecting_out-of-dis.md)
- [\[ACL 2025\] On Synthetic Data Strategies for Domain-Specific Generative Retrieval](../../ACL2025/information_retrieval/on_synthetic_data_strategies_for_domain-specific_generative_retrieval.md)
- [\[ACL 2025\] KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](../../ACL2025/information_retrieval/knowshiftqa_rag_knowledge_shifts.md)

</div>

<!-- RELATED:END -->
