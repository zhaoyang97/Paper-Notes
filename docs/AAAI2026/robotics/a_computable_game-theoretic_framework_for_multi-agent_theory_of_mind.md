# A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind

**会议**: AAAI 2026  
**arXiv**: [2511.22536](https://arxiv.org/abs/2511.22536)  
**作者**: Fengming Zhu, Yuxin Pan, Xiaomeng Zhu, Fangzhen Lin
**代码**: 无  
**领域**: 多智能体 / 博弈论  
**关键词**: Theory of Mind, 认知层次, 博弈论, 贝叶斯推理, Gamma-Poisson 共轭, 随机博弈

## 一句话总结

提出基于 Poisson 认知层次（cognitive hierarchy）的博弈论框架，通过 Gamma-Poisson 共轭贝叶斯更新实现可计算的多智能体 Theory of Mind，在避免 POMDP 不可判定性的同时支持递归式有限理性决策与在线信念修正。

## 研究背景与动机

1. **Theory of Mind 的跨学科需求**：ToM 起源于心理学（Premack & Woodruff 1978），研究智能体推断他者目标、意图与信念的能力，在逻辑学、经济学和机器人学中均有广泛需求，但心理学研究缺乏可自动化的形式化计算框架。
2. **逻辑形式化的局限**：传统自动规划中目标用 fluent 集合表示、信念用可修正的心智状态建模，已扩展至多智能体（如 ConGolog、高阶信念变更），但这些逻辑范式难以刻画功利性（utilitarian）行为——智能体应最大化累积奖励而非仅满足逻辑约束。
3. **决策论/博弈论视角的优势**：博弈论模型天然支持(a)功利性最优响应——智能体同时应对环境和其他智能体做最优决策；(b)可无缝集成统计学习与现代 ML 技术以扩展到大规模场景。
4. **I-POMDP 的可计算性问题**：Interactive POMDP（Gmytrasiewicz & Doshi 2005）是多智能体 ToM 的经典框架，但其递归信念空间导致求解面临不可判定性（undecidability），实际场景中无法使用。
5. **现有认知层次方法的不足**：GR2（Wen et al. 2021）将 Poisson 层次嵌入 RL 训练，但(a)在执行阶段不更新信念——无法在线适应对手变化；(b)仅考虑 action-wise 最优响应而非 strategy-wise。
6. **缺乏统一的可计算 ToM 框架**：现有基于深度学习的 ToM 工作（如 Machine ToM、MMToM-QA）集中在评估 LLM 的 ToM 能力或单智能体域，尚未提出多智能体域中概念清晰、可计算的形式化框架。

## 方法详解

### 整体框架

系统建模为**随机博弈**（Stochastic Game）$\langle \mathcal{N}, \mathcal{S}, \mathcal{A}, T, R \rangle$：

- $\mathcal{N}$：$n$ 个智能体的有限集合
- $\mathcal{S}$：有限环境状态集
- $\mathcal{A} = \mathcal{A}_1 \times \cdots \times \mathcal{A}_n$：联合动作空间
- $T: \mathcal{S} \times \mathcal{A} \mapsto \Delta(\mathcal{S})$：随机状态转移
- $R_i: \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$：智能体 $i$ 的即时奖励

每个智能体最大化累积折扣奖励 $\mathbb{E}[\sum_t \gamma^t R_{i,t}]$，其策略 $\pi_i: \mathcal{S} \mapsto \Delta(\mathcal{A}_i)$ 即为形式化的"意图"。目标概念被奖励结构所包含（subsume），信念则通过认知层次结构建模。

### 关键设计

**1. Poisson 认知层次构建**

信念结构采用 Poisson($\lambda$) 认知层次，自底向上逐层构建策略：

- **Level-0**：随机策略或简单规则，$\pi_j|_0(a_j|S) \sim \text{Unif}(\mathcal{A}_j)$
- **Level-$(k+1)$**：对低于 level-$k$ 的策略做最优响应（best response）

框架提供两种实现方式：

| | 实现 1：Singleton BR | 实现 2：Mixed BR |
|---|---|---|
| **假设** | 对手全在 level-$k$ | 对手按截断 Poisson 分布在 level-$0 \sim k$ |
| **公式** | $\pi_j\|_{k+1} \in BR(\pi_{-j}\|_k)$ | $\pi_j\|_{k+1} \in BR(\pi_{-j}^{mixed}\|_k)$ |
| **求解** | 解诱导 MDP $\mathcal{M}(\pi_{-j}\|_k)$ | 需 QMDP 近似避免 POMDP 不可判定 |
| **初始计算** | $\Theta(K_j)$ 个 MDP | $\Theta(K_j)$ 个 MDP |
| **更新开销** | 仅更新信念分布，支撑策略不变 | 每次需重新求解（$g_\iota$ 变化导致 BR 变化）|
| **性质** | 策略平稳性封闭 | 策略平稳性封闭 |

**2. Gamma-Poisson 共轭贝叶斯更新**

平均推理水平 $\Lambda$ 服从先验 $\text{Gamma}(a, b)$，观测到 $m$ 轮交互中对手的推理等级 $(k_1, \ldots, k_m)$ 后，后验更新为：

$$\Lambda | (k_1, \ldots, k_m) \sim \text{Gamma}\left(a + \sum_r k_r, \ b + m\right)$$

最优估计：$\lambda' = \frac{a + \sum_r k_r}{b + m}$

这一共轭结构使得信念更新仅需维护两个标量参数 $(a, b)$，计算代价极低。

**3. QMDP 近似保证可计算性**

实现 2 中 $BR(\pi_{-j}^{mixed}|_k)$ 原则上需求解 POMDP（不可判定），论文采用 QMDP 近似：

$$\pi_j|_{k+1}(S) \in \arg\max_{a_j} \sum_{\iota=0}^{k} g_\iota \cdot Q^*_{\mathcal{M}(\pi_{-j}|\iota)}(S, a_j)$$

其中 $g_\iota$ 为截断 Poisson 的条件概率权重，$Q^*$ 为各诱导 MDP 的最优 Q 函数。该近似将 POMDP 降为若干 MDP 的加权 Q 值组合。

**4. 完整算法流程**

1. 初始化先验 $\text{Gamma}(a, b)$
2. 估计 $\lambda$，构建各层策略
3. 计算自身最优响应策略
4. 观测对手行为，更新 $(a, b)$ 参数
5. 回到步骤 2 循环迭代

### 与 I-POMDP 的关系

本框架可视为 I-POMDP 的一个可计算实例化（instantiation），通过 Poisson 层次约束递归信念空间、QMDP 近似保证可解性，同时保留了多智能体递归推理的核心能力。

## 实验

本文为**纯理论工作**，未包含实验评估，作者在论文中明确说明"due to the page limit, we will focus on elaborating our proposed theoretic framework, leaving experiments to future work"，并提到会在人机共居系统上进行实验验证。

| 方面 | 说明 |
|---|---|
| 实验数据 | 无 |
| 基线对比 | 无 |
| 评估指标 | 未定义 |
| 实验领域 | 计划用于人机共居系统（human-robot cohabiting） |

| 理论比较 | 本框架 | GR2 (Wen et al. 2021) | I-POMDP |
|---|---|---|---|
| 在线信念更新 | ✓ Gamma-Poisson | ✗ 训练后固定 | ✓ 但不可计算 |
| 最优响应类型 | Strategy-wise | Action-wise | Strategy-wise |
| 可计算性 | ✓ QMDP 近似 | ✓ | ✗ 一般不可判定 |
| 递归推理 | ✓ 层次式 | ✓ 层次式 | ✓ 无限递归 |

## 关键发现

- Poisson 认知层次 + Gamma 先验构成共轭对，使信念更新从复杂的后验推断简化为两个标量的增量更新
- 两种 BR 实现（singleton vs. mixed）体现了精度-效率权衡：实现 1 更新开销低（支撑策略不变），实现 2 建模更真实但每轮需重新计算
- 信念更新具有**非单调、非收敛**特性——当信念偏离真实分布时可被快速纠正，符合真实交互场景
- 框架在平稳策略空间上具有封闭性：若 level-0 策略平稳，则所有层级策略均为平稳策略

## 亮点

- **优雅的数学架构**：将 ToM 的核心概念（目标→奖励、意图→策略、信念→认知层次）系统映射到博弈论框架中，概念清晰、形式严谨
- **Gamma-Poisson 共轭的巧妙利用**：避免了通用贝叶斯推断的高计算成本，信念更新代价为 O(1)
- **可计算性保证**：通过 QMDP 近似巧妙绕过 POMDP 不可判定性，将递归推理限制在可解范围内
- **两种实现的泛化性**：singleton BR 和 mixed BR 分别适用于不同场景需求，框架具有灵活性
- **承上启下的理论定位**：统一了逻辑形式化（automated planning）和决策论（MDP/game theory）两条研究路线

## 局限性

1. **完全缺乏实验验证**：作为 AAAI 论文，纯理论无实验是显著短板，框架的实际效果未知
2. **QMDP 近似质量无保证**：QMDP 是一种乐观近似，在信息获取价值高的场景中可能严重偏离最优
3. **对手推理等级的观测假设过强**：框架假设可以观测到对手"played a level-$k_r$ strategy"，但实际中如何从行为准确推断推理等级未被讨论
4. **层次截断的影响未分析**：实际实现中必须截断层级数，截断带来的近似误差缺乏理论界定
5. **仅考虑完全可观测环境状态**：环境状态 $\mathcal{S}$ 假设完全可观测，限制了在部分可观测场景中的应用
6. **可扩展性存疑**：虽然声称可集成 ML 技术扩展到大规模域，但未给出任何具体说明或路径

## 相关工作

- **逻辑形式化 ToM**：ConGolog（De Giacomo et al. 2000）、高阶信念变更（Wan et al. 2021）、通用博弈（Genesereth & Thielscher 2014）
- **认知层次模型**：经典 Poisson cognitive hierarchy（Camerer et al. 2004）；GR2 将 level-k 嵌入 RL 训练（Wen et al. 2021）
- **I-POMDP**：多智能体递归信念框架（Gmytrasiewicz & Doshi 2005），可计算性问题广泛存在
- **对手建模**：Dirichlet-Categorical 共轭用于矩阵博弈（Boutilier 1996）；Type-based planners（Albrecht & Ramamoorthy 2015; Zhu & Lin 2025b）
- **机器 ToM**：Machine Theory of Mind（Rabinowitz et al. 2018）；MMToM-QA（Jin et al. 2024）；MuMA-ToM（Shi et al. 2025）；AutoToM（Zhang et al. 2025）
- **随机博弈理论**：Shapley 1953；Markov games（Littman 1994）；常数记忆策略（Zhu & Lin 2025a）

## 评分

- 新颖性: ⭐⭐⭐⭐ 博弈论视角重新形式化 ToM 核心概念，Gamma-Poisson 共轭信念更新有新意
- 实验充分度: ⭐⭐ 纯理论论文，完全无实验，框架有效性缺乏验证
- 写作质量: ⭐⭐⭐⭐ 数学形式化严谨清晰，符号系统统一，附录推导完整
- 价值: ⭐⭐⭐⭐ 对多智能体 ToM 计算框架有重要理论贡献，但距实际应用尚需实验验证
