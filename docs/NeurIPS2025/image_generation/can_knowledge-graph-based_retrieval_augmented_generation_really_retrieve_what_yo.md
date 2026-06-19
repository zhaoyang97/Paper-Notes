---
title: >-
  [论文解读] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?
description: >-
  [NeurIPS 2025 Spotlight][图像生成][知识图谱] 提出 GraphFlow 框架，将知识图谱上的检索建模为 GFlowNet 的流匹配问题，通过详细平衡目标和局部探索策略联合训练检索策略与流估计器，在 STaRK 基准上检索准确率和多样性均超越 GPT-4o 约 10%。 基于知识图谱（KG）的检索增…
tags:
  - "NeurIPS 2025 Spotlight"
  - "图像生成"
  - "知识图谱"
  - "检索增强生成"
  - "GFlowNet"
  - "过程奖励模型"
  - "最优传输"
---

# Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.16582](https://arxiv.org/abs/2510.16582)  
**代码**: [GitHub](https://github.com/Samyu0304/GraphFlow)  
**领域**: 图像生成  
**关键词**: 知识图谱, 检索增强生成, GFlowNet, 过程奖励模型, 最优传输

## 一句话总结

提出 GraphFlow 框架，将知识图谱上的检索建模为 GFlowNet 的流匹配问题，通过详细平衡目标和局部探索策略联合训练检索策略与流估计器，在 STaRK 基准上检索准确率和多样性均超越 GPT-4o 约 10%。

## 研究背景与动机

基于知识图谱（KG）的检索增强生成（RAG）是增强 LLM 准确性的重要途径，因为 KG 提供了结构化的关系信息和可解释性。然而，现有 KG-RAG 方法在处理**复杂查询**时面临两大挑战：

**结构与文本信息的融合**：简单关系查询（"谁是 Alice 的父亲？"）只需检索关系三元组即可回答，但复杂查询（如"列出 A 大学发表的与 B 主题相关的论文"）需要同时理解作者-机构关系和论文内容文本

**检索多样性**：复杂查询对应**多个**检索目标，现有方法倾向于最大化似然而忽略多样性

过程奖励模型（PRM）可以为多步检索提供逐步引导，但训练 PRM 需要昂贵的**过程级监督信号**。在 KG 检索中，只有**结果级奖励**（检索轨迹是否支持查询）容易获得，中间步骤的逐步标注成本极高。

## 方法详解

### 整体框架

GraphFlow 将 KG 检索建模为**基于能量的轨迹分布**：

$$P(\tau) = \prod_{t=0}^{T} P(s_{t+1} \mid s_t) \propto R(\tau)$$

即高奖励的检索轨迹应以更高概率被采样。这不同于传统方法的似然最大化目标——GraphFlow 天然地促进了检索结果的**多样性与准确性**平衡。

核心组件：
- **检索策略** $P(s_{t+1} \mid s_t)$：前向策略，决定在 KG 上从当前节点移动到哪个邻居
- **流估计器** $F(s): \mathcal{S} \to \mathbb{R}_{\geq 0}$：为每个中间状态分配非负流值，将终端奖励分解到中间步骤
- **LLM 编码器**：共享的 LLM backbone（+ LoRA）编码状态和状态-动作对

### 关键设计

**详细平衡条件（Detailed Balance）**：

$$F(s_t) \cdot P(s_{t+1} \mid s_t) = F(s_{t+1}) \cdot P_B(s_t \mid s_{t+1})$$

当此条件在所有转移上成立时，策略 $P(s_{t+1} \mid s_t)$ 所诱导的轨迹满足能量分布目标。由于 KG 检索不可回溯，后向策略设为 $P_B(s_t \mid s_{t+1}) = 1$。

**局部探索策略**：在全局 KG 上强制详细平衡计算量巨大。GraphFlow 对每个非终端状态 $s_t$ 执行 $k$ 次探索，生成 $k$ 个替代动作 $\{a'_{t,1}, \ldots, a'_{t,k}\}$，形成 $k+1$ 个候选转移。优化目标：

$$\mathcal{L}_{\text{DBLE}}(s_t) = \sum_{i=0}^{k} \left[\log F(s_t) - \log F(s'_{t+1,i}) + r_\theta(s_t, a'_{t,i}) - \log \sum_{i=0}^{k} e^{r_\theta(s_t, a'_{t,i})}\right]^2$$

**自循环终止机制**：引入特殊的自循环动作，让策略自适应地决定何时停止检索，而非依赖固定步数。

**与 SFT/PRM 的关系**：当设 $\log F(s_t) = 1$、$\log F(s'_{t+1,i}) = 0$ 时，GraphFlow 退化为行为克隆。完整的流估计赋予了模型更强的泛化与多样性能力。

### 损失函数

详细平衡损失 $\mathcal{L}_{\text{DBLE}}$ 在每条采样轨迹的每个非终端状态上独立计算，边界条件 $\log F(s_0) = \log F(s_T) = 1$ 确保流值在轨迹首尾一致。策略头和流头共享 LLM 编码器，通过 LoRA 高效微调。

## 实验关键数据

### 主实验

在 STaRK 基准（STaRK-AMAZON、STaRK-MAG、STaRK-PRIME）上的检索准确率：

| 方法 | AMAZON Hit@1 | AMAZON Hit@5 | MAG Hit@1 | MAG Hit@5 | PRIME Hit@1 | PRIME Hit@5 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| DenseRetriever | 6.10 | 15.85 | 24.44 | 40.23 | 5.43 | 13.07 |
| ToG+GPT4o | 20.67 | 41.38 | 23.33 | 56.67 | 16.67 | 39.77 |
| SFT | 8.16 | 15.30 | 26.53 | 28.57 | 27.50 | 40.07 |
| PRM | 20.09 | 26.25 | 26.05 | 28.00 | 21.01 | 46.72 |
| **GraphFlow** | **19.63** | **44.17** | **29.32** | **58.64** | **39.84** | **71.71** |
| **GraphFlow (Rerank)** | **47.85** | **65.03** | **39.09** | **57.51** | **51.39** | **72.11** |

### 消融实验

检索多样性指标（Recall@20 与 Diverse-Recall@20）：

| 方法 | AMAZON R@20 | AMAZON D-R@20 | MAG R@20 | MAG D-R@20 | PRIME R@20 | PRIME D-R@20 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| ToG+GPT4o | 25.81 | 23.70 | 48.03 | 47.71 | 54.35 | 54.35 |
| PRM | 35.72 | 18.94 | 36.73 | 36.73 | 45.97 | 45.97 |
| **GraphFlow** | **36.15** | **36.15** | **57.18** | **57.18** | **79.71** | **79.59** |

GraphFlow 的 R@20 与 D-R@20 几乎相等，说明检索结果不仅多而且多样。相比之下 PRM 在 AMAZON 上 R@20=35.72 但 D-R@20 仅 18.94，存在严重的重复检索问题。

### 关键发现

1. GraphFlow 在 Rerank 设置下平均 Hit@1 提升约 10%，超越 GPT-4o 基线
2. 在 PRIME 数据集上，GraphFlow 的 R@20 达到 79.71%，远超第二名 ToG+GPT4o 的 54.35%
3. 流估计器有效地将结果级奖励分解为过程级信号，无需显式的过程标注
4. 局部探索策略避免了低奖励区域的无效探索，显著提升训练效率

## 亮点与洞察

- ⭐ 巧妙地将 GFlowNet 引入 KG-RAG 领域，通过"采样概率正比于奖励"天然解决准确性-多样性权衡
- ⭐ 流估计器作为"免费"的过程奖励模型，避免了昂贵的过程级标注
- 自循环终止机制让检索策略自适应决策深度，比固定步数更灵活
- 与 SFT/PRM 的统一视角（GraphFlow 是它们的泛化形式）提供了清晰的理论理解

## 局限性

- 实验仅在 STaRK 基准的三个数据集上验证，未在其他主流 KG-QA 基准（如 WebQSP、CWQ）上测试
- LLM 编码器的计算开销较大，每步检索都需要编码长文档，扩展性值得关注
- 局部探索的超参数 $k$ 对性能影响的分析不够充分
- 训练时仅采集到达目标文档的正例轨迹，可能导致对负例的覆盖不足

## 相关工作与启发

GraphFlow 连接了 GFlowNet（生成式流网络）和 KG-RAG 两个研究方向。GFlowNet 最初用于分子生成中的多样性采样，本文将其推广到图结构上的多步检索决策。未来可以考虑将此思路扩展到更大规模的 KG（如 Wikidata），或结合 GraphRAG 的社区检测方法。流估计器作为过程奖励的替代品，对 LLM reasoning 中的 credit assignment 也有参考价值。

## 评分

⭐⭐⭐⭐ (4/5)

| 维度 | 评分 |
|:--|:--:|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

将 GFlowNet 引入 KG-RAG 的创新性很强，理论框架清晰，实验提升显著。主要不足是实验数据集覆盖面有限且计算效率分析欠缺。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](../../AAAI2026/image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)
- [\[NeurIPS 2025\] Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning](graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval](highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)
- [\[CVPR 2026\] ImageRAGTurbo: Towards One-step Text-to-Image Generation with Retrieval-Augmented Diffusion Models](../../CVPR2026/image_generation/imageragturbo_towards_one-step_text-to-image_generation_with_retrieval-augmented.md)
- [\[NeurIPS 2025\] Scaling Can Lead to Compositional Generalization](scaling_can_lead_to_compositional_generalization.md)

</div>

<!-- RELATED:END -->
