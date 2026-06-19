---
title: >-
  [论文解读] Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards
description: >-
  [NeurIPS 2025][信息检索/RAG][RAG一致性] 提出 Con-RAG 框架，通过 Paraphrased Set GRPO (PS-GRPO) 在语义等价查询的多次生成之间计算组相似度奖励，训练 RAG 系统的生成器在释义输入下产生信息一致的输出，无需显式真实标签监督即可同时提升一致性和准确性。
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "RAG一致性"
  - "GRPO"
  - "信息一致性"
  - "语义等价查询"
  - "强化学习对齐"
---

# Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards

**会议**: NeurIPS 2025  
**arXiv**: [2510.04392](https://arxiv.org/abs/2510.04392)  
**代码**: 待确认  
**领域**: 信息检索  
**关键词**: RAG一致性, GRPO, 信息一致性, 语义等价查询, 强化学习对齐

## 一句话总结

提出 Con-RAG 框架，通过 Paraphrased Set GRPO (PS-GRPO) 在语义等价查询的多次生成之间计算组相似度奖励，训练 RAG 系统的生成器在释义输入下产生信息一致的输出，无需显式真实标签监督即可同时提升一致性和准确性。

## 研究背景与动机

RAG 系统在高风险场景（医疗、金融、法律）中被广泛部署，但面临严重的输出不一致问题：

**检索端不一致**：语义等价的查询（如"如何关闭储蓄账户"与"关闭储蓄账户的步骤是什么"）可能检索到不同的文档集，导致后续生成不同

**生成端不一致**：即使检索到相同文档，LLM 对措辞变化也敏感，可能产生不同答案

**信任危机**：在金融等领域，同一问题的不同回答会严重损害用户信任和合规性

现有工作主要关注 RAG 的准确性和忠实度，而**信息一致性**——即输出在语义等价输入下传达相同核心内容——被严重忽视。不同于词汇一致性（可能惩罚合法的同义替换），信息一致性更关注事实层面的一致。

关键观察：在短问答中，一致性与准确性正相关；在长问答中，两者正交——模型可能准确但不一致。

## 方法详解

### 整体框架

Con-RAG 包含两部分：(1) 分层一致性评估框架，诊断不一致来源；(2) PS-GRPO 训练方法，通过组相似度奖励优化生成器。

### 一致性度量框架

将 RAG 一致性分解为三个层次：

1. **检索器一致性**：释义查询检索到的文档集之间的 Jaccard 相似度
$$\mathcal{C}_{\text{ret}}(q_0) = \frac{2}{n(n-1)} \sum_{i,j} \frac{|R(p_i) \cap R(p_j)|}{|R(p_i) \cup R(p_j)|}$$

2. **端到端一致性**：各释义输入通过完整 RAG 管道产出的输出之间的相似度
$$\mathcal{C}_{\text{e2e}}(q_0) = \frac{1}{n(n-1)} \sum_{i \neq j} \text{sim}(y_i, y_j)$$

3. **生成器一致性**：固定检索结果，仅变化输入措辞，衡量 LLM 自身的稳定性

### 关键设计：Paraphrased Set GRPO (PS-GRPO)

基于 GRPO（Group Relative Policy Optimization）框架扩展而来。核心思想是：对于一个规范查询 $q_0$ 的 $n$ 个释义 $\{p_1,...,p_n\}$，每个释义生成 $g$ 个 rollout，形成 $n \times g$ 的输出矩阵。

**组相似度奖励**：每个 rollout $o_{ij}$ 的奖励通过与其他释义的所有 rollout 计算平均相似度得到：

$$r_{ij} = \frac{1}{(n-1)g} \sum_{\substack{u=1 \\ u \neq i}}^{n} \sum_{m=1}^{g} \text{sim}(o_{ij}, o_{um})$$

实践中用 BLEU 作为相似度函数（实验验证优于 ROUGE-L、Exact Match 等）。

**带准确性的联合奖励**（有真实标签时）：

$$r_{ij}^{\text{final}} = \alpha \cdot r_{ij}^{\text{cons}} + \gamma \cdot \text{Acc}(o_{ij}, y^\star)$$

其中 Acc 用 token F1 度量。对于开放式长回答任务，不使用准确性奖励，仅优化一致性。

**优势归一化**：在每个释义内部做归一化 $\hat{A}_{ij} = (r_{ij} - \mu_i) / \sigma_i$，再用标准 GRPO 裁剪目标更新策略：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{g} \sum_{i=1}^{g} \sum_{t=1}^{|o_i|} \min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

### 高效近似

朴素计算需要 $n(n-1)g^2$ 次相似度比较（$n=5, g=6$ 时为 720 次/查询）。论文引入**松弛近似**：对每个 rollout 仅随机采样 $\kappa$ 个释义和每个释义 $s$ 个 rollout：

$$\tilde{r}_{ij} = \frac{1}{\kappa s} \sum_{u \in K} \sum_{m \in S_k} \text{sim}(o_{ij}, o_{um})$$

复杂度从 $O(n(n-1)g^2)$ 降至 $O(ng\kappa s)$，其中 $\kappa \ll n-1, s \ll g$。

### 实验配置

- 生成器：LLaMA-3.1-8B、Qwen-2.5-3B
- 检索器：e5-base-v2，KILT Wikipedia 语料，top-k=5
- 释义生成：LLaMA-3.1-70B 为每个查询生成 $n=6$ 个释义
- 训练：$g=4$ rollouts/释义，$\kappa=3, s=1$，AdamW lr=1e-6

## 实验关键数据

### 主实验：短问答一致性与准确性（LLaMA-3.1-8B）

| 数据集 | 方法 | EM↑ | F1↑ | RM↑ | 端到端一致(Lexical)↑ | 端到端一致(LLM-Judge)↑ | 生成器一致(Lexical)↑ |
|--------|------|-----|-----|-----|---------------------|----------------------|---------------------|
| TriviaQA | RAG | 56.0 | 66.1 | 74.0 | 53.0 | 77.8 | 67.3 |
| TriviaQA | DRAG | 54.0 | 63.7 | 72.0 | 56.8 | 78.7 | 68.2 |
| TriviaQA | SFT | 24.0 | 27.5 | 29.0 | 51.3 | 58.2 | 77.8 |
| TriviaQA | **Con-RAG** | **77.0** | **81.0** | **83.0** | **87.3** | **91.3** | **91.2** |
| HotpotQA | RAG | 37.0 | 44.1 | 42.0 | 42.5 | 62.5 | 53.7 |
| HotpotQA | **Con-RAG** | **45.0** | **51.9** | **48.0** | **63.9** | **73.6** | **80.9** |
| MuSiQue | RAG | 8.0 | 15.3 | 12.0 | 27.9 | 48.2 | 44.4 |
| MuSiQue | **Con-RAG** | **23.0** | **30.8** | **25.0** | **72.5** | **72.3** | **91.4** |

TriviaQA 上 Con-RAG 端到端一致性从 53.0→87.3（词汇），77.8→91.3（LLM-Judge），同时准确性从 56.0→77.0 EM。

### 长回答实验（ELI5，无真实标签监督）

| 方法 | ROUGE↑ | LLM-Acc↑ | 端到端一致(Lexical)↑ | 端到端一致(Info)↑ | 生成器一致(Lexical)↑ | 生成器一致(Info)↑ |
|------|--------|----------|---------------------|------------------|---------------------|------------------|
| RAG | 21.9 | 74.0 | 8.6 | 62.8 | 15.1 | 74.2 |
| DRAG | 22.0 | 76.0 | 8.0 | 62.2 | 15.0 | 72.5 |
| SFT | 23.5 | 51.0 | 15.3 | 40.8 | 16.6 | 41.7 |
| **Con-RAG** | **24.2** | **78.0** | **14.6** | **72.7** | **21.7** | **80.8** |

在无真实标签情况下，Con-RAG 仅靠一致性奖励就同时提升了准确性（74→78 LLM-Acc）和一致性（62.8→72.7 信息一致性）。

### 消融实验：奖励相似度指标对比（ELI5，Qwen-2.5-3B）

| 指标 | LLM-Acc↑ | 端到端一致(LLM-Judge)↑ | 生成器一致(LLM-Judge)↑ |
|------|----------|----------------------|----------------------|
| BLEU-1 | 54.0 | 38.2 | 69.8 |
| **BLEU-2** | **58.0** | **42.0** | **67.5** |
| BLEU-3 | 49.0 | 36.3 | 66.0 |
| ROUGE-L | 46.0 | 35.2 | 65.2 |
| Exact Match | 49.0 | 37.7 | 66.2 |

低阶 BLEU（1-2）效果最佳，强调词汇选择和局部流畅度，更契合信息一致性目标。

### 关键发现

1. **检索器是主要不一致来源**：Jaccard 一致性仅 27-52%，说明释义查询经常检索到不重叠的文档
2. **一致性训练隐式增强了准确性**：在所有短问答数据集上，Con-RAG 同时提升了 EM、F1 和 RM，可能得益于释义训练的数据增强效果
3. **SFT 在开放式任务上表现差**：在 ELI5 上 SFT 的 LLM-Judge 准确性仅 51.0，远低于 RAG 的 74.0，说明刚性监督不适合开放式问答
4. **联合一致性+F1奖励最优**：单独一致性或准确性奖励都不如联合训练（TriviaQA EM: 51.5 vs 54.0 vs 60.0）

## 亮点与洞察

1. **问题定义精准**：清晰区分信息一致性与词汇一致性，前者容许合法的措辞变化，后者过于严格
2. **分层诊断框架**：将 RAG 不一致分解为检索器/生成器/端到端三层，可精确定位瓶颈
3. **GRPO 的创新应用**：巧妙利用 GRPO 多 rollout 的特性，将跨释义的 rollout 组成更大的比较组计算相似度奖励
4. **无监督也能提升准确性**：ELI5 实验证明仅靠一致性奖励（无真实标签）就能提升准确性，说明一致性是准确性的代理信号
5. **计算高效**：松弛近似将奖励计算从二次降到线性，使大规模训练可行

## 局限性

1. **释义生成依赖强 LLM**：需要 LLaMA-70B 生成高质量释义，本身有成本
2. **检索器未优化**：框架仅优化生成器，检索端不一致仍然存在（Jaccard 仅 27-52%）
3. **BLEU 奖励可能导致泛化问题**：过度优化表面相似度可能牺牲回答多样性
4. **评估局限**：一致性与准确性的最优平衡点可能因任务而异，缺乏统一标准
5. **仅测试两个模型规模**：3B 和 8B，更大模型上的表现未验证

## 相关工作与启发

- GRPO（Shao et al., 2024）为本文方法的基础，PS-GRPO 是其在一致性维度的自然扩展
- 与 Self-Consistency（Wang et al., 2023）不同，Con-RAG 通过训练而非仅推理时采样来提升一致性
- 启发：类似的组奖励思路可用于多语言一致性（同一问题的多语言回答应一致）

## 评分

- **创新性**: ⭐⭐⭐⭐ — 将 GRPO 扩展到跨释义一致性优化是巧妙设计
- **实验充分性**: ⭐⭐⭐⭐ — 覆盖短/多跳/长问答，两个模型，多基线对比
- **实用性**: ⭐⭐⭐⭐ — 对高风险 RAG 部署有直接价值，松弛近似保证可扩展性
- **写作质量**: ⭐⭐⭐⭐ — 框架图清晰，理论和实验对应良好
- **总体评价**: ⭐⭐⭐⭐ — 填补了 RAG 一致性优化的空白，方法设计和实验都很扎实

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](../../ICML2025/information_retrieval/fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](../../ACL2026/information_retrieval/end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2025\] Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries](../../ACL2025/information_retrieval/logical_consistency_is_vital_neural-symbolic_information_retrieval_for_negative-.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](../../ACL2026/information_retrieval/quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
