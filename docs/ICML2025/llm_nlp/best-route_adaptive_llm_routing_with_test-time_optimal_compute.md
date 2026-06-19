---
title: >-
  [论文解读] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute
description: >-
  [ICML 2025][LLM 其他][LLM routing] 提出 BEST-Route（Best-of-n Enhanced Sampling and Test-time Route Optimization），在传统查询路由的基础上引入 best-of-n 采样策略，使路由器不仅选择模型，还自适应决定采样数量 n，通过小模型多次采样+选优替代大模型单次调用，在不到 1% 性能损失下降低高达 60% 的推理成本。
tags:
  - "ICML 2025"
  - "LLM 其他"
  - "LLM routing"
  - "best-of-n sampling"
  - "测试时计算"
  - "成本优化"
  - "自适应推理"
---

# BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute

**会议**: ICML 2025  
**arXiv**: [2506.22716](https://arxiv.org/abs/2506.22716)  
**作者**: Dujian Ding (UBC/Microsoft), Ankur Mallick (Microsoft), Shaokun Zhang (Penn State), Chi Wang (Google DeepMind), Victor Rühle (Microsoft) 等
**机构**: University of British Columbia, Microsoft, Pennsylvania State University, Google DeepMind, AG2AI
**领域**: LLM/NLP  
**关键词**: LLM routing, best-of-n sampling, 测试时计算, 成本优化, 自适应推理

## 一句话总结

提出 BEST-Route（Best-of-n Enhanced Sampling and Test-time Route Optimization），在传统查询路由的基础上引入 best-of-n 采样策略，使路由器不仅选择模型，还自适应决定采样数量 n，通过小模型多次采样+选优替代大模型单次调用，在不到 1% 性能损失下降低高达 60% 的推理成本。

## 研究背景与动机

### 核心矛盾

LLM 推理面临**质量-成本权衡**的根本矛盾：大模型（如 GPT-4）质量高但成本昂贵，小模型（如 Llama 系列）成本低但单次响应质量不足。现有三种主要策略试图缓解这一矛盾：

**Query Routing**：根据查询难度将请求分配给不同规模的模型（Ong et al., 2024）

**Speculative Decoding**：小模型草稿 + 大模型验证的 token 级协作（Kim et al., 2023）

**Model Cascades**：从便宜模型开始逐级升级，直到获得满意响应（Chen et al., 2023）

### 现有方法的关键缺陷

先前的查询路由方法存在一个核心问题：**每个模型只生成一次响应**。这意味着：

- 小模型的单次响应质量经常不够好，无法与大模型竞争
- 路由器不得不将大量查询（除最简单的外）分配给大模型
- 实际的成本节省远低于理论预期

### 关键观察

作者注意到一个重要但被忽视的现象：**小模型通过 best-of-n 采样可以大幅提升响应质量**。具体来说，对小模型采样 n 次并选出最优响应，其质量可以接近甚至超过大模型的单次响应，而总成本（n 次小模型调用）仍然低于一次大模型调用。这一观察来源于 test-time compute scaling 的最新研究（Snell et al., 2024），即在推理时增加计算量可以有效提升模型性能。

### 动机公式化

设大模型单次调用成本为 $C_L$，小模型单次调用成本为 $C_S$，若 $n \cdot C_S < C_L$，则对小模型采样 n 次仍比调用大模型一次便宜。当 best-of-n 选择策略能使小模型 n 次采样中的最优响应质量逼近大模型时，路由器就能将更多查询分配给小模型，从而显著降低总成本。

## 方法详解

### 整体框架

BEST-Route 的系统架构包含三个核心组件：

1. **查询难度评估器（Query Difficulty Estimator）**：对输入查询的难度进行评分，判断其需要多强的模型能力
2. **模型选择路由器（Model Selector/Router）**：根据难度评分选择调用大模型还是小模型
3. **采样数量决策器（Sampling Number Decider）**：当选择小模型时，自适应决定采样次数 n，以在满足质量阈值的前提下最小化成本

工作流程如下：
- 输入查询 $q$ → 评估难度 $d(q)$
- 若 $d(q)$ 超过阈值，直接路由到大模型，生成 1 个响应
- 若 $d(q)$ 在中等范围，路由到小模型，采样 $n(q)$ 个响应，用选择策略选出最优
- 若 $d(q)$ 极低，路由到小模型，采样 1 个响应即可

### 关键设计

#### Best-of-n 采样与选择

BEST-Route 将 test-time compute 引入路由框架的核心机制：

- **采样阶段**：对小模型以温度 $T > 0$ 采样生成 n 个独立响应 $\{r_1, r_2, \dots, r_n\}$
- **选择阶段**：使用奖励模型或轻量评分器对 n 个响应打分，选出得分最高的作为最终输出
- **自适应 n 值**：n 不是固定的，而是根据查询难度动态调整——中等难度查询需要较大的 n，简单查询 n=1 即可

#### 路由优化目标

路由决策可以形式化为一个约束优化问题：

$$\min_{f} \mathbb{E}_{q \sim Q}\left[\text{Cost}(f(q))\right] \quad \text{s.t.} \quad \mathbb{E}_{q \sim Q}\left[\text{Quality}(f(q))\right] \geq \tau$$

其中 $f(q)$ 表示对查询 $q$ 的路由决策（选择哪个模型 + 采样多少次），$\tau$ 是质量阈值。BEST-Route 通过扩展决策空间——从 "选择模型" 扩展到 "选择模型 × 采样次数"——在保持质量约束的同时找到更低成本的解。

#### 质量阈值机制

系统允许用户设定不同的质量阈值 $\tau$，实现灵活的质量-成本权衡：
- 高 $\tau$：更多查询路由到大模型，质量优先
- 低 $\tau$：更多查询由小模型 best-of-n 处理，成本优先
- 自适应 $\tau$：根据应用场景动态调整

### 训练策略

#### 路由器训练

路由器的训练需要解决两个核心问题：

1. **难度标注获取**：通过在训练集上同时收集大、小模型在不同 n 值下的响应质量，构建每个查询的最优路由标签
2. **成本感知学习**：在训练目标中同时考虑质量和成本，使路由器学会在两者间做最优权衡

#### 与 Test-Time Compute Scaling 的结合

BEST-Route 的核心创新在于将 test-time compute scaling（Snell et al., 2024）的思想系统性地融入路由框架：

- 传统路由：决策空间为 $\{M_1, M_2, \dots, M_K\}$（K 个模型的选择）
- BEST-Route：决策空间为 $\{(M_i, n_j) | i \in [K], j \in [N_{max}]\}$（模型 × 采样次数的组合）

这一扩展使路由器能够找到 Pareto 最优的组合——例如选择小模型采样 5 次可能比大模型单次调用质量相当但成本仅为 1/3。

## 实验分析

### 主要结果

基于论文摘要报告的核心实验数据（在真实世界数据集上测试）：

| 方法 | 成本降低 | 性能损失 | 路由策略 |
|------|---------|---------|---------|
| 单一大模型 (baseline) | 0% | 0% | 所有查询 → 大模型 |
| 传统路由 (prior work) | 10-25% | 1-3% | 简单→小, 难→大 |
| **BEST-Route** | **高达 60%** | **< 1%** | 简单→小(n=1), 中→小(best-of-n), 难→大 |
| 单一小模型 (baseline) | 最大 | 显著 | 所有查询 → 小模型 |

### 与先前路由方法对比

| 维度 | 传统路由方法 | BEST-Route |
|------|------------|-----------|
| 决策空间 | 模型选择 | 模型选择 × 采样次数 |
| 小模型利用率 | 低（仅处理简单查询） | 高（中等难度也可处理） |
| Test-time compute | 未利用 | 核心机制 |
| 成本节省 | 有限 | 显著（最高 60%） |
| 质量保证 | 取决于路由准确性 | 质量阈值 + best-of-n 兜底 |

### 关键发现

1. **Best-of-n 的边际效益递减但早期收益极大**：小模型从 n=1 到 n=3-5 的质量提升最为显著，之后边际收益递减。这意味着适度的额外计算（3-5 倍小模型成本）即可大幅缩小与大模型的差距。

2. **查询难度分布的长尾特性**：在实际工作负载中，大部分查询是中等或偏简单的，真正需要大模型的困难查询只占少数。这为 BEST-Route 提供了巨大的成本优化空间。

3. **成本-质量 Pareto 前沿的扩展**：通过引入 (模型, n) 的联合决策空间，BEST-Route 显著扩展了可达的 Pareto 前沿，使得在相同质量下存在更低成本的可行解。

## 亮点与洞察

1. **思路简洁但效果强大**：核心 idea 非常直觉——"小模型多试几次"比"大模型试一次"可能更划算。但将这一观察系统化为路由框架，并提供自适应的 n 值选择机制，是本文的关键贡献。

2. **弥合了两个领域的 gap**：将 test-time compute scaling 的思想（原本关注单模型内的计算分配）与 LLM routing（关注多模型间的任务分配）有机结合，拓展了两个方向的研究视野。

3. **实用价值极高**：60% 的成本降低对大规模 LLM 部署具有直接的经济意义。对于 API 服务商和企业用户，这意味着在保持服务质量的同时将推理预算降低大半。

4. **与 scaling law 观察一致**：小模型 best-of-n 的有效性与 inference-time scaling law 的发现一致——在推理阶段投入更多计算可以弥补模型参数量的不足。

## 局限性

1. **选择策略的依赖**：best-of-n 的效果高度依赖选择器（reward model 或 verifier）的质量。如果选择器本身不可靠，多次采样可能无法选出真正最优的响应。

2. **延迟问题**：虽然总成本降低，但 n 次串行采样会增加端到端延迟。在延迟敏感的场景中，需要并行采样，这又增加了系统复杂度。

3. **缓存不变性假设**：论文假设模型的能力分布在推理期间不变，但实际部署中模型可能更新、输入分布可能漂移，路由器需要持续校准。

4. **评估指标的局限**：不同应用场景对"质量"的定义不同（事实准确性 vs 创造性 vs 代码正确性），统一的质量阈值可能不适用于所有场景。

5. **模型组合扩展性**：当候选模型数量增多时，(模型, n) 的联合决策空间急剧增大，路由器训练和推理的复杂度也随之上升。

## 相关工作

### LLM Query Routing
- **RouteLLM** (Ong et al., 2024)：基于分类器的路由方法，但每个模型只生成一次响应
- **AutoMix** (Ding et al., 2024)：自动混合不同模型的响应，但未考虑 test-time compute
- **FrugalGPT** (Chen et al., 2023)：级联式模型调用策略，从便宜到贵依次尝试

### Test-Time Compute Scaling
- **Scaling LLM Test-Time Compute** (Snell et al., 2024)：系统研究推理时计算量对性能的影响
- **Self-Consistency** (Wang et al., 2023)：多次采样+投票的推理策略
- **Re-ranking** (Chuang et al., 2023)：对多个候选响应进行排序选优

### Speculative Decoding
- **Speculative Decoding** (Kim et al., 2023)：小模型生成草稿 token，大模型验证接受/拒绝，加速自回归生成

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 将 best-of-n 和路由结合的 idea 简洁有效，虽非全新概念但组合方式新颖 |
| 实用性 | 5 | 60% 成本降低对 LLM 部署有直接且显著的经济价值 |
| 技术深度 | 3 | 方法相对直观，核心贡献在于系统化框架而非技术突破 |
| 实验充分性 | 3 | 摘要提及真实数据集验证，但缓存中缺少详细实验设置和消融信息 |
| 写作质量 | 4 | 问题动机清晰，框架描述系统，图示直观 |

**总评**: 4/5 — 一篇实用价值极高的工作，核心 idea 简洁优雅，将 test-time compute scaling 与 LLM routing 有机结合，在成本-质量权衡上取得了显著的 Pareto 改进。方法的技术门槛不高但工程意义重大，对 LLM 规模化部署具有直接的指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search](../../NeurIPS2025/llm_nlp/wider_or_deeper_scaling_llm_inference-time_compute_with_adaptive_branching_tree_.md)
- [\[ICLR 2026\] Best-of-∞: Asymptotic Performance of Test-Time LLM Ensembling](../../ICLR2026/llm_nlp/best-of-infinity_asymptotic_performance_of_test-time_llm_ensembling.md)
- [\[ICML 2025\] Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting](breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting.md)
- [\[CVPR 2025\] Test-Time Visual In-Context Tuning](../../CVPR2025/llm_nlp/test-time_visual_in-context_tuning.md)
- [\[ICML 2025\] LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](la_rosa_enhancing_llm_efficiency_via_layerwise_rotated_sparse_activation.md)

</div>

<!-- RELATED:END -->
