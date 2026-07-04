---
title: >-
  [论文解读] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning
description: >-
  [ICML2025][预训练][Bayesian Optimization] 提出 DRE-BO-SSL，将半监督学习（标签传播/标签扩散）引入密度比估计型贝叶斯优化，通过无标签数据点缓解监督分类器的过度利用(over-exploitation)问题，在探索与利用之间取得更好平衡。 贝叶斯优化范式回顾 贝叶斯优化 (BO)…
tags:
  - "ICML2025"
  - "预训练"
  - "Bayesian Optimization"
  - "Density Ratio Estimation"
  - "半监督学习"
  - "Label Propagation"
  - "Label Spreading"
---

# Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning

**会议**: ICML2025  
**arXiv**: [2305.15612](https://arxiv.org/abs/2305.15612)  
**代码**: [https://github.com/jungtaekkim/bayeso](https://github.com/jungtaekkim/bayeso)  
**领域**: LLM预训练  
**关键词**: Bayesian Optimization, Density Ratio Estimation, Semi-Supervised Learning, Label Propagation, Label Spreading

## 一句话总结

提出 DRE-BO-SSL，将半监督学习（标签传播/标签扩散）引入密度比估计型贝叶斯优化，通过无标签数据点缓解监督分类器的过度利用(over-exploitation)问题，在探索与利用之间取得更好平衡。

## 研究背景与动机

### 贝叶斯优化范式回顾

贝叶斯优化 (BO) 用于高效求解评估代价高昂的黑箱函数的全局最优。传统方法使用概率回归模型（如高斯过程）作为代理函数，显式建模 $p(y|\mathbf{x}, \mathcal{D})$，再结合采集函数（EI、PI 等）选择下一个查询点。

### 密度比估计型 BO 的兴起

区别于回归型 BO，密度比估计 (DRE) 型 BO 将搜索空间中的观测点按阈值 $y^\dagger$ 分为两组——接近全局最优的"好组" ($y \leq y^\dagger$) 和"差组" ($y > y^\dagger$)，然后估计两组的密度比作为采集函数。代表方法包括：

- **TPE** (Bergstra et al., 2011)：用两个树结构 Parzen 估计器分别估计两组密度
- **BORE** (Tiao et al., 2021)：将密度比估计转化为二分类的类概率估计
- **LFBO** (Song et al., 2022)：提出通用的 likelihood-free 框架，证明 BORE 等价于 Probability of Improvement

### 核心问题：过度利用

本文识别出 DRE 型 BO 中监督分类器的关键缺陷——**过度利用问题 (over-exploitation)**。与一般分类中的过度自信不同，此处指分类器对已知的全局候选区域过于自信，导致搜索集中在极小范围内，丧失探索能力。具体表现为：

1. BO 早期迭代中，训练数据 $\mathcal{D}_t$ 规模很小，监督分类器容易过拟合
2. 分类器给已知好点所在区域分配极高概率，其余区域概率趋零
3. 阈值 $y^\dagger$ 无法剧烈变化，算法容易陷入局部最优

传统回归型 BO（如 GP-EI）天然通过不确定性估计实现探索，而 DRE 型分类器缺乏这一机制。

## 方法详解

### 整体框架：DRE-BO-SSL

核心想法：引入**无标签数据点**，使用半监督分类器替代监督分类器来估计类概率，从而降低对已知区域的过度自信。

#### 采集函数推导

DRE 型 BO 的采集函数基于 $\zeta$-相对密度比：

$$A(\mathbf{x}|\zeta, \mathcal{D}_t) = \frac{p(\mathbf{x}|y \leq y^\dagger, \mathcal{D}_t)}{\zeta \cdot p(\mathbf{x}|y \leq y^\dagger, \mathcal{D}_t) + (1-\zeta) \cdot p(\mathbf{x}|y > y^\dagger, \mathcal{D}_t)}$$

其中 $\zeta = p(y \leq y^\dagger) \in [0,1)$ 为阈值比例。通过贝叶斯定理可将其转化为类概率估计：

$$A(\mathbf{x}|\zeta, \mathcal{D}_t) = \zeta^{-1} \pi(\mathbf{x})$$

其中 $\pi(\mathbf{x})$ 是输入 $\mathbf{x}$ 属于 Class 1（好组）的概率。DRE-BO-SSL 将 $\pi$ 替换为半监督分类器的输出：

$$\mathbf{x}_{t+1} = \arg\max_{\mathbf{x} \in \mathcal{X}} \pi_{\hat{\mathbf{C}}_t}(\mathbf{x}; \zeta, \mathcal{D}_t, \mathbf{X}_u)$$

### 半监督学习组件

采用两种经典的基于图的半监督方法：

#### 1. Label Propagation (Zhu & Ghahramani, 2002)

在由 $n_l$ 个有标签点和 $n_u$ 个无标签点构成的相似度图上，通过迭代传播标签。基于 RBF 核构建相似度矩阵 $\mathbf{W}$，然后迭代更新无标签点的伪标签直到收敛，每次迭代后重置有标签点的标签（硬约束）。

#### 2. Label Spreading (Zhou et al., 2003)

与 Label Propagation 类似，但允许有标签点的标签在迭代中也被修改（软约束），通过参数 $\alpha \in [0,1)$ 控制标签传播与初始标签之间的权衡。

### Transductive → Inductive 扩展

半监督方法天然是 transductive 的（只对已知无标签点预测），但 BO 需要对搜索空间中的任意点 $\mathbf{x}$ 给出预测。本文通过以下方式实现 inductive 预测：用得到伪标签的"有标签 + 无标签"全部数据训练一个标准分类器（如 1-NN），再对任意新查询点进行预测。

### 无标签点的采样策略

当无预定义池（fixed-size pool）时，需要主动采样无标签点。本文从截断多元正态分布中采样，使用 minimax tilting 方法 (Botev, 2017)，使无标签点覆盖搜索空间 $\mathcal{X}$ 且符合 cluster assumption。

### 算法流程 (Algorithm 1)

1. 随机初始化并评估 $\mathcal{D}_0$
2. 在每轮 $t$：计算阈值 $y_t^\dagger$ → 分配类标签 $\mathbf{C}_t$ → 采样/获取无标签点 $\mathbf{X}_u$ → 通过半监督学习估计伪标签 $\hat{\mathbf{C}}_t$ → 最大化 $\pi_{\hat{\mathbf{C}}_t}$ 选择下一查询点 → 评估并更新数据集

优化采用 multi-started L-BFGS-B；对于平坦的概率景观，随机选择最高概率点之一作为查询点。

## 实验设置与主要结果

### 实验设置

- **合成基准**：Branin (2D)、Hartmann (6D) 等标准测试函数
- **Tabular Benchmarks** (Klein & Hutter, 2019)：超参数优化的表格化基准（固定池场景）
- **NATS-Bench** (Dong et al., 2021)：神经架构搜索基准（固定池场景）
- **64D minimum multi-digit MNIST search**：高维搜索问题（固定池场景）
- **基线方法**：GP-EI, GP-PI, TPE, BORE (MLP/RF/XGBoost), LFBO (MLP/RF/XGBoost)

### 两种实验场景

| 场景 | 无标签点来源 | 代表任务 |
|------|-------------|---------|
| 随机采样 | 每轮从截断正态分布采样 $n_u$ 个无标签点 | 合成函数 |
| 固定池 | 预定义的有限候选集合 | Tabular Benchmarks, NATS-Bench, MNIST search |

### 关键实验结果

1. **合成函数上的可视化**（Figure 1, Branin）：BORE/LFBO 的 MLP 分类器在迭代前5步中，决策边界集中在极小区域（过度利用）；DRE-BO-SSL (Label Propagation/Spreading) 的决策边界更平滑、覆盖更广（更好的探索-利用平衡）

2. **合成基准定量结果**：DRE-BO-SSL 在多个合成函数上的收敛速度和最终解质量优于或持平 BORE/LFBO；尤其在早期迭代中优势明显

3. **Tabular Benchmarks**：在固定池场景下，DRE-BO-SSL 在超参数优化任务中表现出竞争力，某些配置下优于所有基线

4. **NATS-Bench 神经架构搜索**：DRE-BO-SSL 成功找到高质量架构，性能优于或媲美 GP 和 BORE/LFBO 变体

5. **64D MNIST search**：在高维固定池搜索中展示了 DRE-BO-SSL 的可扩展性

### 消融分析（Section 6）

- **阈值比例 $\zeta$ 的影响**：分析不同 $\zeta$ 值对性能的影响，验证过度利用问题随 $\zeta$ 变化的行为
- **无标签点数量 $n_u$ 的影响**：更多无标签点通常带来更平滑的概率景观，但存在计算开销
- **Label Propagation vs. Label Spreading**：两者表现相近，Label Spreading 因软约束在某些场景更稳定

## 亮点与洞察

1. **问题识别精准**：清晰区分了 DRE-BO 中的"over-exploitation"与一般分类下的"overconfidence"——前者是区域级的过度集中，后者是单样本级的过度自信
2. **方法轻量优雅**：不需要修改 BO 框架本身，只需将监督分类器替换为半监督分类器，改动最小化
3. **理论兼容性好**：与 BORE/LFBO 的理论框架完全兼容，半监督分类器仍然输出类概率，可无缝接入 $\zeta$-相对密度比的计算
4. **两种场景统一处理**：通过截断正态采样和固定池两种方式获取无标签点，覆盖了连续和离散搜索空间

## 局限与展望

1. **可扩展性存疑**：Label Propagation/Spreading 需要计算 $(n_l + n_u) \times (n_l + n_u)$ 的相似度矩阵，当 $n_u$ 较大或维度较高时计算开销显著
2. **半监督方法选择有限**：仅使用了 Label Propagation 和 Label Spreading 两种最经典的方法，未探索更现代的半监督技术（如 FixMatch、MixMatch 等）
3. **RBF 核参数敏感性**：相似度矩阵依赖 RBF 核的带宽参数，论文对该超参数的选择和鲁棒性讨论不足
4. **无标签点采样策略简单**：截断正态分布采样假设较强，未探索自适应采样策略（如基于当前搜索状态调整分布）
5. **缺乏理论收敛保证**：相比 GP-based BO 有明确的 regret bound，DRE-BO-SSL 缺乏类似的理论分析
6. **高维连续空间上的实验不够充分**：合成函数最高 6D，固定池场景虽有 64D 但本质上是离散搜索

## 相关工作与启发

- **BORE** (Tiao et al., 2021)：首次将密度比估计转化为二分类，是本文的直接竞争者
- **LFBO** (Song et al., 2022)：统一了 DRE-BO 的理论框架，证明其与 PI/EI 的等价性
- **TPE** (Bergstra et al., 2011)：DRE-BO 的开创性工作，用 Parzen 估计器建模两个密度
- **Label Propagation** (Zhu & Ghahramani, 2002) / **Label Spreading** (Zhou et al., 2003)：本文采用的两种半监督学习核心算法
- 启发：半监督学习在其他优化/决策框架中也可能缓解类似的过度利用问题，如进化算法中的代理模型

## 评分
- 新颖性: ⭐⭐⭐⭐ — 问题识别有洞察力，但方法本身是现有技术的组合（DRE-BO + 经典半监督学习），创新增幅有限
- 实验充分度: ⭐⭐⭐⭐⭐ — 涵盖合成、超参优化、NAS、高维搜索等多个场景，消融实验较充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰但篇幅较长，公式推导详尽但可读性一般
- 价值: ⭐⭐⭐⭐ — 对 DRE-BO 社区有实用价值，但该方向整体上仍是 BO 的小众分支

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] A Unified Framework for Heterogeneous Semi-supervised Learning](../../CVPR2025/llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)
- [\[ICML 2025\] A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning](a_square_peg_in_a_square_hole_meta-expert_for_long-tailed_semi-supervised_learni.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)
- [\[ICML 2025\] Position: The Future of Bayesian Prediction Is Prior-Fitted](position_the_future_of_bayesian_prediction_is_prior-fitted.md)

</div>

<!-- RELATED:END -->
