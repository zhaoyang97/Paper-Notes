---
title: >-
  [论文解读] Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation
description: >-
  [NeurIPS 2025 (NORA Workshop)][图学习][持续学习] 本文在 FB15k-237 上系统评估了弹性权重固化（EWC）对 TransE 知识图谱嵌入持续学习的效果，发现 EWC 将灾难性遗忘从 12.62% 降至 6.85%（减少 45.7%），并揭示了任务划分策略（基于关系 vs 随机）对遗忘度量的显著影响（9.8 个百分点差异）。
tags:
  - "NeurIPS 2025 (NORA Workshop)"
  - "图学习"
  - "持续学习"
  - "知识图谱"
  - "弹性权重固化"
  - "灾难性遗忘"
  - "链接预测"
---

# Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation

**会议**: NeurIPS 2025 (NORA Workshop)  
**arXiv**: [2512.01890](https://arxiv.org/abs/2512.01890)  
**代码**: 无  
**领域**: 图学习  
**关键词**: 持续学习, 知识图谱, 弹性权重固化, 灾难性遗忘, 链接预测

## 一句话总结
本文在 FB15k-237 上系统评估了弹性权重固化（EWC）对 TransE 知识图谱嵌入持续学习的效果，发现 EWC 将灾难性遗忘从 12.62% 降至 6.85%（减少 45.7%），并揭示了任务划分策略（基于关系 vs 随机）对遗忘度量的显著影响（9.8 个百分点差异）。

## 研究背景与动机

**领域现状**：知识图谱不断演化，需要持续更新。神经嵌入模型（如 TransE）通过学习向量表示进行链接预测，但顺序学习新任务时会遭受灾难性遗忘——旧任务性能急剧下降。

**现有痛点**：持续学习方法已在图像分类和 NLP 中广泛研究，但在 KG 链接预测上的实证评估仍然不足。特别是经典的正则化方法（EWC）在 KG 嵌入上的有效性缺乏系统验证。

**核心矛盾**：KG 嵌入具有结构化参数空间（特定维度编码语义属性），简单的参数保护是否足以保持语义结构？另外，任务划分方式对灾难性遗忘的测量影响不明确。

**本文目标**：(a) EWC 能否有效缓解 KG 嵌入的灾难性遗忘？(b) 任务构建策略如何影响遗忘度量？

**切入角度**：以 TransE+FB15k-237 为基础，设计严格的多种子实验，对比 EWC vs 朴素顺序训练 vs 经验回放，并对比关系划分 vs 随机划分两种任务构建方式。

**核心 idea**：通过系统实证证明 EWC 的 Fisher 信息矩阵正则化可有效保护 KG 嵌入参数，同时揭示任务划分策略是持续学习评估中一个被忽视但影响巨大的因素。

## 方法详解

### 整体框架
将 KG $\mathcal{G}=(\mathcal{E},\mathcal{R},\mathcal{T})$ 划分为 $T$ 个顺序任务 $\mathcal{G}_1,\dots,\mathcal{G}_T$，逐任务训练 TransE 嵌入，评估各任务遗忘程度。框架评估三类方法：朴素顺序训练、EWC 正则化、经验回放。

### 关键设计

1. **TransE 嵌入训练**:

    - 功能：学习实体/关系向量使得 $\mathbf{h}+\mathbf{r}\approx\mathbf{t}$。
    - 核心思路：最小化 margin-based 损失 $\mathcal{L}=\sum_{(h,r,t)\in\mathcal{T}}\sum_{(h',r,t')\in\mathcal{T}'}\max(0, \gamma+d(\mathbf{h}+\mathbf{r},\mathbf{t})-d(\mathbf{h'}+\mathbf{r},\mathbf{t'}))$，其中 $\gamma=1.0$，使用 L2 距离，50 维嵌入。

2. **弹性权重固化（EWC）**:

    - 功能：在学习新任务时保护对旧任务重要的参数。
    - 核心思路：在任务 $i$ 的损失中加入 Fisher 信息矩阵正则项 $\mathcal{L}^i_{\text{EWC}}=\mathcal{L}^i+\frac{\lambda}{2}\sum_k F_k(\theta_k-\theta^*_{k,i-1})^2$，其中 $F_k=\mathbb{E}_{(h,r,t)\sim\mathcal{G}_{i-1}}[(\frac{\partial\log p(y|x;\theta)}{\partial\theta_k})^2]$ 用前一任务全部数据 mini-batch（256）近似计算。
    - 设计动机：Fisher 信息矩阵识别对前序任务编码至关重要的参数，通过二次惩罚阻止这些参数大幅偏移，保持嵌入空间的语义结构。

3. **任务划分策略**:

    - **关系划分**：按关系频率排序，round-robin 分配到 4 个任务，每任务约 59 种关系。同一关系的所有三元组在同一任务中，任务间关系完全不重叠，造成较大分布偏移。
    - **随机划分**：将 272,115 个训练三元组随机均分为 4 块（每块约 68,000），关系在各任务间重叠。分布偏移最小。

### 遗忘度量
- 任务 $j$ 在学习任务 $i$ 后的遗忘：$F^j_i = M^j_j - M^j_i$（$i>j$）
- 平均遗忘：$\bar{F}=\frac{1}{T-1}\sum_{j=1}^{T-1}F^j_T$
- 评估指标：MRR（filtered ranking）

## 实验关键数据

### 主实验（关系划分）

| 方法 | 遗忘率(%) | Final MRR |
|------|-----------|-----------|
| Naive 顺序训练 | 12.62 ± 0.35 | 0.206 ± 0.006 |
| EWC (λ=0.1) | 10.44 ± 0.26 | 0.229 ± 0.005 |
| EWC (λ=1.0) | 7.51 ± 0.44 | 0.250 ± 0.006 |
| **EWC (λ=10)** | **6.85 ± 0.33** | **0.242 ± 0.004** |
| EWC + Wave Replay | 9.91 ± 0.20 | 0.234 ± 0.005 |
| Random Replay | 13.78 ± 0.44 | 0.196 ± 0.006 |
| Wave Replay | 12.54 ± 0.14 | 0.216 ± 0.007 |

### 任务划分影响

| 划分策略 | Naive 遗忘(%) | EWC 遗忘(%) | 差异 |
|----------|---------------|-------------|------|
| 关系划分 | 12.62 ± 0.35 | 6.85 ± 0.33 | — |
| 随机划分 | 2.81 ± 0.34 | 5.08 ± 0.22 | — |
| 差异 | **9.81 pp** | 1.77 pp | 关系划分难度远高于随机 |

### EWC 超参数敏感性

| λ | 关系划分遗忘(%) | 随机划分遗忘(%) | 关系划分 MRR |
|---|-----------------|-----------------|-------------|
| 0.1 | 10.44 | 2.88 | 0.229 |
| 1.0 | 7.51 | 3.88 | 0.250 |
| 10.0 | **6.85** | 5.08 | 0.242 |

### 关键发现
- **EWC 显著有效**：在关系划分上遗忘从 12.62% 降至 6.85%，降幅 45.7%。MRR 也从 0.206 提升到 0.242。
- **回放方法反而更差**：Random Replay 遗忘率 13.78% 高于朴素训练，说明简单回放不如参数保护。
- **任务划分策略影响巨大**：关系划分 vs 随机划分在朴素训练上差 9.8 个百分点，因为关系划分造成任务间更大的分布偏移。
- **最优 λ 取决于任务构建**：关系划分需要强正则化（λ=10），随机划分反而弱正则化（λ=0.1）更好。过强正则化在随机划分上会限制必要的参数适应。
- EWC 能大幅缩小两种划分策略的遗忘差距（从 9.81pp 降至 1.77pp），说明有效的持续学习方法可跨不同任务构建方式泛化。

## 亮点与洞察
- **任务划分效应的揭示**是本文最有价值的发现：同一方法在不同划分方式下遗忘率差异可达 9.8 个百分点，这提醒持续学习评估需要明确报告任务构建方式，否则结果不可比较。这个发现可迁移到所有持续学习实验设计中。
- **EWC 优于经验回放**的结论与神经科学的突触巩固理论一致：对结构化知识表示（KG 嵌入），保护参数比回放样本更有效，因为 KG 嵌入中特定维度编码特定语义。
- 实验设计严谨，5个随机种子+均值/标准差，消费级 GPU 可复现。

## 局限与展望
- **单模型单数据集**：仅评估 TransE + FB15k-237，不确定能否泛化到 RotatE、ComplEx 等更复杂嵌入和 WN18RR、YAGO 等数据集。
- **仅 4 个任务**：长序列任务（10+）的动态行为未探索，遗忘可能随任务数非线性增长。
- **Workshop 论文**深度有限：没有与 PS-CKGE 等专门的 KG 持续学习 benchmark 对比。
- 关系划分的 round-robin 策略是特定选择，实体划分或领域划分可能展现不同模式。
- 可考虑将 EWC 与更高级的回放策略（如 GEM、A-GEM）或架构方法（Progressive Networks）组合。

## 相关工作与启发
- **vs PS-CKGE (Zhao2025)**: PS-CKGE 关注模式偏移（pattern shifts）对遗忘的影响，本文侧重经典 EWC 的实证评估。两者互补：PS-CKGE 提供更全面的 benchmark，本文提供更深入的正则化分析。
- **vs Daruna2021**: 在机器人操控任务上评估 KG 持续学习，涉及 TransE/DistMult/ComplEx 多种架构。本文聚焦更窄但更深入（任务划分分析）。
- **vs Online EWC / SI**: 本文使用标准 EWC，未对比在线 EWC 或 Synaptic Intelligence 等变体，是可扩展的方向。

## 评分
- 新颖性: ⭐⭐⭐⭐ 方法层面无新贡献（直接应用 EWC），但任务划分效应的发现有独立价值
- 实验充分度: ⭐⭐⭐⭐ 多种子实验设计严谨，但局限于单模型单数据集
- 写作质量: ⭐⭐⭐⭐⭐ 实证论文写作规范，结论表述审慎
- 价值: ⭐⭐⭐⭐ Workshop 论文定位合理，任务划分效应的洞察对该领域有参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](../../ACL2025/graph_learning/a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra.md)
- [\[ICML 2025\] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](../../ICML2025/graph_learning/from_rag_to_memory_non-parametric_continual_learning_for_large_language_models.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](../../ACL2025/graph_learning/kg_llm_trustworthy_qa.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)

</div>

<!-- RELATED:END -->
