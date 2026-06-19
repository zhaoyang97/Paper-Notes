---
title: >-
  [论文解读] On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting
description: >-
  [AAAI 2026][核稳定性] 针对基于认可的多赢者投票中核稳定性（core stability）是否总存在这一重大开放问题，提出基于混合整数线性规划（MILP）的自动推理框架，证明了新的存在性结果，发现了核稳定性与其他公理（如 Lindahl 可定价性）之间此前未知的关系，并推翻了一个已有猜想。
tags:
  - "AAAI 2026"
  - "核稳定性"
  - "多赢者投票"
  - "混合整数线性规划"
  - "自动推理"
  - "比例代表"
---

# On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting

**会议**: AAAI 2026  
**arXiv**: [2512.16895](https://arxiv.org/abs/2512.16895)  
**代码**: [GitHub](https://github.com/emanueltewolde/Core-MILP)  
**领域**: 计算社会选择 / 自动推理  
**关键词**: 核稳定性, 多赢者投票, 混合整数线性规划, 自动推理, 比例代表

## 一句话总结

针对基于认可的多赢者投票中核稳定性（core stability）是否总存在这一重大开放问题，提出基于混合整数线性规划（MILP）的自动推理框架，证明了新的存在性结果，发现了核稳定性与其他公理（如 Lindahl 可定价性）之间此前未知的关系，并推翻了一个已有猜想。

## 研究背景与动机

在多赢者投票（multi-winner voting）中，任务是从候选人集合中选出一个委员会。核稳定性（core stability）是群体公平性的一个自然且重要的概念：一个委员会是核稳定的，当且仅当不存在任何"联盟偏离"——即没有一组选民能够通过其按比例分配的席位，联合支持一个让他们全部更满意的替代委员会。

核稳定性位于 Aziz 等人 (2017) 提出的代表性公理层级的顶端，蕴含了许多后来引入的公理。它的应用范围超越政治选举：联邦学习中的公平性（Chaudhury et al.）、AI 对齐中的多系统决策（Conitzer et al.）等都能用核稳定性刻画。

然而，核稳定性的一个**重大开放问题**是：在基于认可的选举中，核稳定委员会是否**总是**存在？此前已知的结论仅限于委员会大小 $k \leq 3$（Cheng et al. 2019），或 $k \leq 8$（Peters 2025，但仅针对 PAV 投票规则）。本文采用**不假设特定投票规则**的通用方法，从所有可能的选民偏好中搜索核稳定性最接近被破坏的情况。

本文的方法论创新在于：与计算社会选择中流行的 SAT 方法相比，MILP 方法可以**独立于选民数量**证明特定候选人数量的结果，从而获得显著的计算增益。

## 方法详解

### 整体框架

本文的技术路线如下：

1. 利用投票分布（vote distribution）将核稳定性问题线性化，消除对选民数量 $n$ 的依赖
2. 将"核是否为空"的判定问题转化为嵌套优化问题（max-min-max），再转化为单层 MILP
3. 通过求解器（Gurobi）计算不同参数 $(m,k)$ 下 MILP 的最优值
4. 从实验结果中发现规律，利用对偶理论证明匹配的上界
5. 修改 MILP 来探索核稳定性与其他公理（如可定价性）的关系

### 关键设计

1. **投票分布空间的线性化（Xia 2024）**：核稳定性可以用投票分布 $\mathbf{x} \in \Delta(2^C)$ 表达，而非依赖具体的选民集合。关键引理：委员会 $W$ 是核稳定的当且仅当对所有偏离 $W' \in \mathcal{M}_{\leq k}$：

$$\boldsymbol{\delta}_{W,W'}^T \mathbf{x} - \frac{|W'|}{k} < 0$$

其中 $\boldsymbol{\delta}_{W,W'} \in \{0,1\}^{2^C}$ 表示哪些投票 $A$ 严格偏好 $W'$ 而非 $W$。这一线性化消除了对 $n$ 的依赖。

2. **MILP 构建**：判定对给定 $m,k$ 核是否可能为空，等价于求解：

$$\max_{\mathbf{x} \in \Delta(2^C)} \min_{W \in \mathcal{M}_k} \max_{W' \in \mathcal{M}_{\leq k}} \boldsymbol{\delta}_{W,W'}^T \mathbf{x} - \frac{|W'|}{k}$$

该嵌套优化被转化为带有二元变量 $\mathbf{y}[W,W']$ 的单层 MILP：当 $\mathbf{y}[W,W']=1$ 时对应偏离 $W'$ 最大化内层 max 问题。MILP 的最优值 $\mu^*$ 若为负，则核对所有投票分布非空；若非负，则存在使核为空的投票分布。

3. **对偶分析（DLP）**：固定二元变量 $\mathbf{y}$ 后得到的线性规划取对偶，得到更紧凑的对偶线性规划 DLP。对偶变量 $\mathbf{q}$ 定义了委员会上的概率分布，可以给出 MILP 的上界。核心定理（Theorem 3）：MILP $\leq v$ 当且仅当对所有偏离函数 $D$，DLP $\leq v$。

   这导出了核非空问题的概率化重构（Corollary 7）：核对所有分布非空当且仅当对每个偏离函数 $D$，存在委员会上的分布 $\mathbf{q}$ 使得对所有投票 $A$，偏好偏离的概率严格小于偏离的平均大小除以 $k$。

4. **Droop 核与 Hare 配额的关系**：将核稳定性定义从 Hare 配额（$\frac{1}{k}$）替换为更强的 Droop 配额（$\frac{1}{k+1}$），得到 Droop 核。定理 2 证明 MILP 的下界为 $\frac{-1}{k(k+1)}$，恰好对应 Droop 边界——这意味着 Droop 配额是**保证核非空所能使用的最小配额**。

### 损失函数 / 训练策略

本文不涉及机器学习训练，而是数学优化问题。核心优化目标是 MILP：

$$\max_{\mathbf{x}, \mu, \mathbf{y}} \mu \quad \text{s.t.} \quad \forall W, W': \mu \leq \boldsymbol{\delta}_{W,W'}^T \mathbf{x} - \frac{|W'|}{k} + 3(1 - \mathbf{y}[W,W'])$$

使用 Gurobi 求解器保证全局最优性。

## 实验关键数据

### 主实验

MILP 在不同 $(m,k)$ 参数下的最优值：

| $k \backslash m$ | 4 | 5 | 6 | 7 | 公式 |
|---|---|---|---|---|---|
| 1 | -0.5000 | -0.5000 | -0.5000 | -0.5000 | $-\frac{1}{2}$ |
| 2 | -0.1667 | -0.1667 | -0.1667 | -0.1667 | $-\frac{1}{6}$ |
| 3 | -0.0833 | -0.0833 | -0.0833 | -0.0833 | $-\frac{1}{12}$ |
| 4 | — | -0.0500 | -0.0500 | -0.0500 | $-\frac{1}{20}$ |
| 5 | — | — | -0.0333 | -0.0333 | $-\frac{1}{30}$ |
| 6 | — | — | — | -0.0238 | $-\frac{1}{42}$ |

**关键发现**：所有最优值均为负数，且遵循公式 $\frac{-1}{k(k+1)}$，与候选人数 $m$ 无关。

### 消融实验（不同偏离函数类的上界）

| 偏离函数类型 | DLP 上界 | DrDLP 上界 | 说明 |
|------------|---------|-----------|------|
| 全部为单元素偏离 | $-\frac{1}{k(k+1)}$ | $\leq 0$ | Theorem 4: 下界严格匹配 |
| 至多一个非单元素偏离 $|D(W^*)| = t$ | $-\frac{1}{k(k+2-t)}$ | $\leq 0$ | 非单元素偏离只会让值更远离0 |
| 大委员会 $m = k+1$ | $-\frac{1}{k(k+1)}$ | $\leq 0$ | Theorem 5: 对任意偏离函数成立 |

### 关键发现

1. **$m \leq 7$ 时核总是非空的**：改进了此前 $m+n \leq 14$ 的实验结果
2. **Droop 配额是"最佳配额"**：不存在比 Droop 更小的配额能保证核非空（Corollary 6）
3. **大委员会核非空**：对任意 $m$ 和 $k = m-1$，核（甚至 Droop 核）总是非空的（Corollary 9）
4. **推翻 Lindahl 可定价性猜想**：Munagala et al. 猜测核稳定性等价于 Lindahl 可定价性，本文找到最小反例（$m=4, k=2$）证明 Droop 核不蕴含 Lindahl 可定价性（Theorem 6）
5. **最小反例的确认**：框架不仅能找到反例，还能确认其在参数空间中的最小性

## 亮点与洞察

1. **方法论创新——MILP 替代 SAT**：计算社会选择领域长期依赖 SAT 求解器，但 SAT 难以处理选民数量增大的情况。本文通过投票分布的线性化，将问题转化为 MILP，彻底消除了对选民数量 $n$ 的依赖。同一实例 $(m=7, k=3)$ 在 MILP 下 2.5 小时内解决，而 Peters (2025) 的方法 37 小时未收敛
2. **对偶理论的巧妙应用**：通过 LP 对偶，将计算实验中发现的模式（$\frac{-1}{k(k+1)}$）提升为严格的数学证明，体现了"计算启发 → 理论证明"的优雅范式
3. **概率化核非空条件（Corollary 7）**：将确定性的核非空问题重述为概率语言——对每个偏离函数，存在委员会上的分布使得偏好偏离的概率足够小——为未来的概率性证明方法铺平道路
4. **猜想推翻与最小反例**：不仅推翻了已有猜想，还精确定位了反例存在的最小参数值，并提供了人可读的反例证明
5. **博弈论视角**：将核非空问题解读为对抗性团队博弈（adversarial team game），与合作博弈理论中的 least core 概念建立联系

## 局限与展望

1. **计算可扩展性有限**：MILP 包含 $2^m + 1$ 个连续变量和 $\binom{m}{k} \cdot \sum_{l=1}^{k} \binom{m}{l}$ 个二元变量，当 $k \approx m/2$ 时超多项式增长。$m=8, k=4$ 时 72 小时未收敛
2. **未证明一般情况**：虽然实验和部分理论均指向 MILP 最优值为 $\frac{-1}{k(k+1)}$（暗示核总是非空），但一般情况的证明仍然开放
3. **Droop 核的非空性未完全解决**：虽然所有实验的 DrMILP 值都收敛到 0，但一般性证明缺失
4. **核与其他公理的相容性未知**：核稳定性是否与 EJR+、委员会单调性等公理相容仍是开放问题
5. **仅限认可投票**：未扩展到"点赞/点踩"投票等相关设置

## 相关工作与启发

- **Cheng et al. (2019)**：证明 $k \leq 3$ 核非空，并提出稳定抽签（stable lottery）的存在性。本文的 Corollary 7 与其结果有深刻联系
- **Peters (2025)**：证明 PAV 在 $k \leq 8$ 时给出核稳定委员会，但给出 $k=9$ 的 PAV 反例。本文方法不依赖特定投票规则
- **Xia (2024)**：核的"线性"性质——核只依赖投票分布而非具体选民——是本文 MILP 的理论基础
- **Munagala et al. (2022, 2024)**：提出 Lindahl 可定价性并猜测其等价于核稳定性，被本文推翻
- **启发**：MILP 方法可推广到其他社会选择问题（如公平分配、匹配市场），尤其是那些具有线性结构但目前仍依赖 SAT 的问题

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 方法论突破性创新，MILP 代替 SAT 的范式在计算社会选择中具有广泛影响
- **技术深度**: ⭐⭐⭐⭐⭐ — 混合整数规划、LP 对偶、博弈论等多个数学工具的精妙结合
- **实验充分性**: ⭐⭐⭐⭐ — 受限于计算资源但在可及范围内充分，定理证明弥补了实验的局限
- **实用价值**: ⭐⭐⭐ — 主要是理论贡献，但对民主选举制度设计和 AI 公平性有潜在指导意义
- **写作质量**: ⭐⭐⭐⭐⭐ — 极其清晰的数学写作，运行示例贯穿全文，理论-实验-理论的递进结构优雅

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Automated Reproducibility Has a Problem Statement Problem](automated_reproducibility_has_a_problem_statement_problem.md)
- [\[CVPR 2026\] Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach](../../CVPR2026/others/revisiting_f-measure_optimization_in_multi-label_classification_a_sampling-based.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[AAAI 2026\] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)

</div>

<!-- RELATED:END -->
