---
title: >-
  [论文解读] Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching
description: >-
  [ACL 2025][LLM 其他][成对排序] 本文重新审视了LLM成对排序提示（PRP）中排序算法的选择问题，提出以LLM推理调用次数（而非比较次数）为核心代价模型，发现经典最优算法Heapsort在引入批处理（batching）和缓存（caching）优化后不再最优，Quicksort在batch size≥2时推理次数减少44%，为PRP排序提供了新的最优选择。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "成对排序"
  - "排序算法"
  - "LLM推理成本"
  - "批处理"
  - "缓存优化"
---

# Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching

**会议**: ACL 2025  
**arXiv**: [2505.24643](https://arxiv.org/abs/2505.24643)  
**领域**: LLM / 信息检索  
**关键词**: 成对排序, 排序算法, LLM推理成本, 批处理, 缓存优化

## 一句话总结
本文重新审视了LLM成对排序提示（PRP）中排序算法的选择问题，提出以LLM推理调用次数（而非比较次数）为核心代价模型，发现经典最优算法Heapsort在引入批处理（batching）和缓存（caching）优化后不再最优，Quicksort在batch size≥2时推理次数减少44%，为PRP排序提供了新的最优选择。

## 研究背景与动机

**领域现状**：LLM成对排序提示（Pairwise Ranking Prompting, PRP）是一种流行的零样本重排方法——通过让LLM比较两个文档的相关性来排序检索结果。由于每次比较需要一次LLM推理，朴素的全对比较开销极大，因此研究者引入排序算法来减少比较次数。

**现有痛点**：先前工作（Qin et al., 2024）基于经典排序理论选择了Heapsort（$O(n\log n)$比较次数）作为PRP的标准排序算法。但这种分析将每次比较视为等代价的原子操作，忽略了LLM推理的实际特性——LLM推理支持批处理（一次推理处理多个比较），且重复比较可以被缓存避免。

**核心矛盾**：经典排序理论的代价模型是比较次数，而LLM场景的实际代价是推理调用次数。这两个代价在引入批处理和缓存后可能完全不同——一个算法可能比较次数多但推理次数少（因为比较可以批处理），反之亦然。

**本文目标**：建立以LLM推理调用次数为核心的新代价模型，重新评估不同排序算法在PRP场景中的效率。

**切入角度**：Heapsort的树结构使得每次比较都是顺序依赖的，无法批处理；而Quicksort的分区操作中多个元素与pivot的比较是独立的，天然支持批处理。这一算法结构差异在传统理论中无关紧要，但在LLM场景中至关重要。

**核心 idea**：以LLM推理次数替代比较次数作为代价函数，证明Quicksort在batch size≥2时显著优于Heapsort，并首次将Quicksort引入PRP框架。

## 方法详解

### 整体框架
以LLM推理调用次数为目标函数，系统分析三种排序算法（Heapsort、Bubblesort、Quicksort）在三种优化（批处理、缓存、Top-k提取）下的实际推理次数，通过理论分析和实验验证给出最优算法选择建议。

### 关键设计

1. **Quicksort + 批处理（Batching）**:

    - 功能：利用Quicksort分区阶段的并行性减少推理次数
    - 核心思路：Quicksort在分区阶段需要将多个元素与pivot比较，这些比较之间相互独立，可以在一次LLM推理中同时处理。例如，将20个文档与pivot的比较打包成一次推理调用（batch size=20），而非20次独立调用。使用median-of-three策略选择pivot以保证分区平衡
    - 设计动机：Quicksort的分区操作天然可并行，且Partial Quicksort变体支持高效的Top-k提取

2. **Bubblesort + 缓存（Caching）**:

    - 功能：通过缓存重复的相邻比较减少推理次数
    - 核心思路：Bubblesort在多轮遍历中重复比较相邻元素对。如果第i轮已经比较过(A, B)且两者位置未变，第i+1轮可以直接从缓存中读取结果而非重新调用LLM。用一个字典存储已完成的比较结果，内存开销忽略不计
    - 设计动机：Bubblesort虽然比较次数为 $O(n^2)$，但大量比较是重复的。缓存后实际推理次数可减少约46%

3. **以推理次数为核心的代价框架**:

    - 功能：为LLM排序场景提供正确的算法分析基础
    - 核心思路：将代价函数从比较次数 $C(n)$ 替换为推理次数 $I(n, b)$（b为batch size）。对于Quicksort，$I_{QS}(n, b) \approx \frac{2n\ln n}{b}$；对于Heapsort，$I_{HS}(n) \approx 2n\log_2 n$（无法批处理）。当 $b \geq 2$ 时，Quicksort的推理次数已经少于Heapsort
    - 设计动机：正确的代价模型才能给出正确的算法选择。batch size=2这样的最低并行度就足以逆转Heapsort的优势

### 损失函数 / 训练策略
不涉及模型训练。使用多个预训练LLM（Flan-T5-L/XL/XXL、Mistral-7B、Llama-3-8B）进行排序实验，排序质量用nDCG@10衡量。

## 实验关键数据

### 主实验

| 算法 | Batch Size=1 推理次数 | Batch Size=2 推理次数 | Batch Size=8 推理次数 | nDCG@10 |
|------|--------------------|--------------------|--------------------|---------| 
| Heapsort | 526±42 | 526±42 (无法批处理) | 526±42 | 0.698 |
| Bubblesort (无Cache) | 4950±0 | 4950±0 | 4950±0 | 0.702 |
| Bubblesort (有Cache) | 2673±312 | 2673±312 | 2673±312 | 0.702 |
| Quicksort (无Batch) | 631±95 | 631±95 | 631±95 | 0.695 |
| **Quicksort (有Batch)** | 631±95 | **354±58** | **127±22** | **0.695** |

### 消融实验

| 配置 | 推理次数 | 相比Heapsort增益 |
|------|---------|----------------|
| Heapsort (baseline) | 526 | — |
| Quicksort, batch=2 | 354 | **-33%** |
| Quicksort, batch=4 | 207 | **-61%** |
| Quicksort, batch=8 | 127 | **-76%** |
| Quicksort, batch=128 (A100) | ~50 | **-90%**, 5.52×实际加速 |
| Bubblesort + Cache | 2673 | +408% (仍远高于Heap) |

### 关键发现
- 仅需batch size=2，Quicksort的推理次数就比Heapsort少33%，颠覆了"Heapsort最优"的结论
- A100 GPU上batch size=128时，Quicksort的端到端排序速度是Heapsort的5.52倍
- 三种算法的排序质量（nDCG@10）几乎相同，选择哪种算法不影响排序效果，只影响速度
- Bubblesort虽然通过缓存减少了46%的推理次数，但绝对数量仍远高于其他两种算法
- GPU架构影响批处理收益：A100在batch=128前接近理想线性扩展，3090在batch=32后饱和

## 亮点与洞察
- 本文的核心洞察虽然简单但影响深远：**在LLM时代，经典算法理论的代价模型需要重新审视**。push和pop一个堆元素的成本不再等于一次LLM调用的成本，因为批处理改变了游戏规则。
- 这个发现立即可以应用于所有使用PRP的商业排序系统中——只需把Heapsort换成Quicksort并启用批处理，就能获得显著加速。
- 更广泛地看，任何涉及LLM在循环中被重复调用的场景（如Tree-of-Thought、MCTS搜索）都应该重新考虑调用模式的并行化。

## 局限与展望
- LLM的成对比较可能违反传递性（A>B, B>C不意味着A>C），排序算法假设传递性成立
- 实验仅使用中等规模模型（最大11B），超大规模API模型的批处理特性可能不同
- 未考虑混合策略——如先用高效算法粗排，再用精确算法精排
- 未来可探索不假设传递性的排序/排名算法（如noisy sorting）与LLM成本模型的结合

## 相关工作与启发
- **vs Qin et al. (2024)**: 对PRP的开创性工作推荐Heapsort，本文证明该推荐在批处理设置下不再成立
- **vs PRP-Graph (Luo et al., 2024)**: PRP-Graph通过图聚合减少比较次数，与本文的优化视角互补
- **vs Listwise ranking**: Listwise方法一次排多个文档，与PRP+Quicksort的思路殊途同归

## 评分
- 新颖性: ⭐⭐⭐⭐ 代价模型的重新定义虽然概念简单但影响重大，首次引入Quicksort到PRP
- 实验充分度: ⭐⭐⭐⭐ 5种LLM、8个数据集、3种算法、多种batch size、延迟分析
- 写作质量: ⭐⭐⭐⭐⭐ 论述清晰简洁，理论分析与实验验证完美配合
- 价值: ⭐⭐⭐⭐⭐ 立即可用于改进所有PRP排序系统，实际工程价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ACL 2025\] Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat](ranking_unraveled_recipes_for_llm_rankings_in_head-to-head_ai_combat.md)
- [\[ACL 2025\] SR-LLM: Rethinking the Structured Representation in Large Language Model](sr-llm_rethinking_the_structured_representation_in_large_language_model.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](../../NeurIPS2025/llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)

</div>

<!-- RELATED:END -->
