---
title: >-
  [论文解读] Differentiable Structure Learning and Causal Discovery for General Binary Data
description: >-
  [NEURIPS2025][因果推理][因果发现] 提出基于多元伯努利分布（MVB）的通用可微结构学习框架，不假设特定数据生成过程，能捕获二值离散变量间的任意高阶依赖关系，并证明在一般设定下DAG不可识别但可恢复最小等价类（Markov等价类）。 领域背景：因果发现旨在从观测数据中学习有向无环图（DAG），是NP完全问题…
tags:
  - "NEURIPS2025"
  - "因果推理"
  - "因果发现"
  - "结构学习"
  - "离散数据"
  - "DAG学习"
  - "多元伯努利分布"
---

# Differentiable Structure Learning and Causal Discovery for General Binary Data

**会议**: NEURIPS2025  
**arXiv**: [2509.21658](https://arxiv.org/abs/2509.21658)  
**代码**: 待确认  
**领域**: 因果推理  
**关键词**: 因果发现, 结构学习, 离散数据, DAG学习, 多元伯努利分布  

## 一句话总结

提出基于多元伯努利分布（MVB）的通用可微结构学习框架，不假设特定数据生成过程，能捕获二值离散变量间的任意高阶依赖关系，并证明在一般设定下DAG不可识别但可恢复最小等价类（Markov等价类）。

## 研究背景与动机

**领域背景**：因果发现旨在从观测数据中学习有向无环图（DAG），是NP完全问题。近年来NOTEARS等方法将其转化为可微约束优化，取得重大突破。

**已有方法局限**：现有可微结构学习方法主要针对连续数据（高斯假设），少数扩展到离散数据的方法（如Zeng et al. 2022的广义线性SEM）假设特定参数形式，仅考虑线性/加性效应。

**离散数据的挑战**：许多真实数据涉及二值或离散变量（疾病有无、基因标记、问卷回答），其复杂的高阶依赖结构无法被线性模型捕获。

**理论空缺**：Bello et al. (2022) 将连续优化方法应用于logistic回归模型的离散数据，但缺乏可识别性的理论保证。

**传统方法不足**：约束方法（PC）和分数方法（GES）要么缺乏统计鲁棒性，要么依赖强参数假设（加性噪声、隐表示、线性效应、潜在高斯变量）。

**核心动机**：需要一个通用的、理论健全的、可微的离散数据结构学习框架，能建模任意依赖关系，不受限于特定数据生成假设。

## 方法详解

### 整体框架

本文基于**多元伯努利分布（Multivariate Bernoulli Distribution, MVB）**构建通用离散数据的可微结构学习框架。核心思路分三步：(1) 用MVB作为任意二值数据的通用表示；(2) 证明不可识别性并刻画完整的等价结构-参数对集合；(3) 将最稀疏DAG的搜索问题转化为单一可微优化问题。

### 关键设计

#### 模块一：多元伯努利参数化与高阶交互

将联合分布写成指数族形式：$P(X=x) = \exp(\sum_{S \subseteq [p]} f^S B^S(x))$，其中 $B^S(x) = \prod_{j \in S} x_j$ 为交互特征。条件分布自然表示为：

$$\Pr(X_p = 1 | X_{-p}) = \sigma\left(\sum_{S \subseteq [p-1]} f^{S,p} B^S(x)\right)$$

这里 σ 为sigmoid函数。**关键洞察**是：对于一般二值数据，**所有高阶交互项都存在**，忽略它们（如仅保留一阶项）会导致错误的因果结论。通过扩展特征映射 $\Phi(X) = [B^S(X)]_{S \in 2^{[p]}}$，将所有 $2^p$ 个交互项统一编码。

#### 模块二：不可识别性刻画与最小等价类

**定理1（不可识别性）**：对任意拓扑排序 π，都存在唯一的结构方程模型 $(f_\pi, G_\pi)$ 精确再现观测分布。因此共有至多 p! 个等价的DAG-参数对。

**定义最小等价类** $\mathcal{E}_{\min}(p)$：在所有等价对中选择边数最少的DAG。

**定理2**：在SMR（最稀疏Markov表示）或忠实性假设下，最小等价类中的所有DAG属于同一个Markov等价类。

#### 模块三：可微优化求解（BiNOTEARS）

定义参数矩阵 $\mathbf{H} \in \mathbb{R}^{2^p \times p}$，通过加权邻接矩阵 $[W(\mathbf{H})]_{ij} = \sum_{S \subseteq [p], i \in S} (h^{S,j})^2$ 判定边的存在。约束 $h^{S,j} = 0$ 当 $j \in S$（禁止自环）。

**两阶段策略（BiNOTEARS）**用于扩展到大规模图：
- **第一阶段**：用 NOTEARS-MLP 适配离散数据，学习粗结构并提取拓扑排序 π
- **第二阶段**：沿拓扑排序构建一阶+二阶交互特征，拟合logistic回归确定最终边结构

### 损失函数

正则化得分函数为：

$$s(\mathbf{H}; \lambda, \delta, \mathbf{X}) = \ell(\mathbf{H}; \mathbf{X}) + p_{\lambda,\delta}(W(\mathbf{H}))$$

其中 $\ell$ 为交叉熵（负对数似然），$p_{\lambda,\delta}$ 为**准MCP（quasi Minimax-Concave Penalty）**正则项：

$$p_{\lambda,\delta}(t) = \lambda \left[ (|t| - \frac{t^2}{2\delta}) \mathbb{1}(|t|<\delta) + \frac{\delta}{2} \mathbb{1}(|t|>\delta) \right]$$

该正则在 $[0,\delta]$ 区间为二次函数、超过 $\delta$ 后变平，光滑近似 $\ell_0$ 惩罚，鼓励稀疏邻接矩阵。最终优化问题加上可微无环约束 $h(W(\mathbf{H})) = 0$。

**定理3**：存在足够小的 $\lambda, \delta > 0$，使全局最优解集恰好等于最小等价类。

## 实验关键数据

### 主实验：合成数据（小规模图, p ∈ {5,...,9}）

| 设定 | BiNOTEARS(Ours) | DAGMA | PC | FGES |
|------|----------------|-------|-----|------|
| ER-1 (一阶+二阶) | **最优** | 较差（仅一阶） | 中等 | 中等 |
| SF-1 (一阶+二阶) | **最优** | 较差 | 中等 | 中等 |
| ER-2 (一阶+二阶) | **最优** | 显著劣于其他 | 中等 | 中等 |
| SF-2 (一阶+二阶) | **最优** | 显著劣于其他 | 中等 | 中等 |
| 仅最高阶交互 | **近最优恢复** | 完全失败 | 部分恢复 | 部分恢复 |

> DAGMA仅建模一阶效应，在存在高阶交互时SHD显著恶化；BiNOTEARS在所有设定下竞争力最强。

### 真实数据：Sachs蛋白质信号网络（d=11, n=7466）

| 方法 | SHD ↓ | 边数 |
|------|-------|------|
| NOTEARS (linear) | 22 | 18 |
| NOTEARS-MLP | 16 | 13 |
| **BiNOTEARS (Ours)** | **13** | 15 |

### 消融与关键发现

1. **高阶交互至关重要**：当数据包含二阶交互时，仅使用一阶模型（DAGMA）性能急剧下降，验证了捕获高阶依赖的必要性。
2. **两阶段策略有效扩展**：BiNOTEARS在大规模图（p=10,20,30,40）上保持竞争力，尽管所有方法在图密度增加时都有退化。
3. **真实数据验证**：在Sachs数据集上SHD大幅领先NOTEARS(22→13)，减少41%的结构误差。

## 亮点与洞察

1. **理论贡献突出**：完整刻画了一般二值数据中DAG的不可识别性（定理1），并证明最小等价类可通过单一可微程序恢复（定理3），为先前工作提供了理论支撑。
2. **一般性框架**：不假设特定SEM形式，通过MVB分布表示任意二值数据的联合分布，是首个真正通用的离散可微结构学习方法。
3. **实用两阶段方法**：BiNOTEARS通过NOTEARS-MLP获取拓扑排序+logistic回归精细化，巧妙平衡了表达力与计算效率。
4. **关键洞察**：离散数据中高阶交互项不可忽略，线性假设会导致本质性错误。

## 局限性

1. **指数级参数空间**：一般MVB参数量为 $2^p$，限制了全阶交互方法只能处理小规模图（p < 10）。
2. **两阶段方法缺乏理论保证**：BiNOTEARS的第二阶段（截断到二阶交互）虽然实用，但理论性质未被分析。
3. **仅限二值数据**：虽然论文指出多类别可用二值指示器编码，但会进一步加剧维度爆炸。
4. **缺少与更多非参数方法的比较**：如基于核方法或条件独立测试的最新方法。

## 相关工作与启发

- **vs. DAGMA/NOTEARS**：本文直接扩展了DAGMA框架，但增加了高阶项以处理通用离散分布，并提供了理论保证
- **vs. PC/GES等传统方法**：传统方法依赖离散条件独立测试或BIC分数，缺乏灵活性
- **vs. Zeng et al. 2022**：该方法假设广义线性SEM，是本文的特例（仅一阶交互）
- **启发**：二值变量的高阶交互观点可迁移到其他离散结构学习问题，如基因调控网络发现

## 评分
- 新颖性: ⭐⭐⭐⭐ (通用MVB参数化 + 完整不可识别性刻画是新颖贡献)
- 实验充分度: ⭐⭐⭐ (合成实验充分但大规模和真实数据实验偏少)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，动机明确)
- 价值: ⭐⭐⭐⭐ (填补了离散数据可微因果发现的理论空白)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)
- [\[ACL 2025\] Leveraging Variation Theory in Counterfactual Data Augmentation for Optimized Active Learning](../../ACL2025/causal_inference/leveraging_variation_theory_in_counterfactual_data_augmentation_for_optimized_ac.md)

</div>

<!-- RELATED:END -->
