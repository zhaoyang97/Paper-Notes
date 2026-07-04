---
title: >-
  [论文解读] Transferring Causal Effects using Proxies
description: >-
  [NEURIPS2025][因果推理][proximal causal inference] 提出基于代理变量（proxy）的多域因果效应迁移方法，在目标域仅观测到代理变量 W 的条件下，利用多源域数据识别并估计目标域中含未观测混淆因子的干预分布，给出两种一致性估计器及渐近置信区间。 核心问题：估计处理变量 X 对结果 Y…
tags:
  - "NEURIPS2025"
  - "因果推理"
  - "proximal causal inference"
  - "域适应"
  - "unobserved confounders"
  - "proxy variables"
  - "interventional distribution"
---

# Transferring Causal Effects using Proxies

**会议**: NEURIPS2025  
**arXiv**: [2510.25924](https://arxiv.org/abs/2510.25924)  
**代码**: [manueligal/proxy-intervention](https://github.com/manueligal/proxy-intervention)  
**领域**: 因果推理  
**关键词**: proximal causal inference, domain adaptation, unobserved confounders, proxy variables, interventional distribution

## 一句话总结

提出基于代理变量（proxy）的多域因果效应迁移方法，在目标域仅观测到代理变量 W 的条件下，利用多源域数据识别并估计目标域中含未观测混淆因子的干预分布，给出两种一致性估计器及渐近置信区间。

## 研究背景与动机

**核心问题**：估计处理变量 X 对结果 Y 的因果效应是科学研究的核心目标，但未观测混淆因子 U 的存在使得从观测数据中进行因果推断极其困难

**现有方案局限**：随机对照试验（RCT）是因果推断的金标准，但常因伦理或实际约束不可行；现有代理变量方法（proximal causal inference）通常假设因果效应在不同域间不变，无法处理域间分布偏移

**多域设置的独特挑战**：当隐藏混淆因子 U 的分布随域改变时（latent shift），X 对 Y 的因果效应在不同域中是不同的，传统的单域代理方法无法直接应用

**目标域数据稀缺**：在目标域中可能只能观测到代理变量 W，无法直接观测 X 和 Y，需要从源域迁移因果信息

**与已有工作的关键区别**：Tsai et al. [2024] 的工作需要在目标域观测 W 和 X，且目标是条件均值预测而非干预分布；本文仅需在目标域观测 W，目标是估计干预分布 Q(Y|do(x))

**实际应用场景**：例如研究网站排名对消费者选择的因果效应——酒店特征（U）影响排名（X）和点击（Y），价格（W）是 U 的代理，不同酒店构成不同域

## 方法详解

### 整体框架

本文考虑的数据生成过程由结构因果模型（SCM）描述。源域中观测 (E, W, X, Y)，目标域中仅观测 W。核心思路是利用多域数据中代理变量 W 的条件分布矩阵 P(W|E,x) 的可逆性，建立干预分布的可识别性公式 q(y|do(x)) = P(y|E,x) · P(W|E,x)† · Q(W)，其中 † 表示右伪逆。

### 模块一：可识别性理论（Identifiability）

- **功能**：证明在什么条件下，目标域的干预分布 q(y|do(x)) 能从可观测数据中唯一确定
- **核心思路**：从协变量调整公式 q(y|do(x)) = Σ_u q(y|u,x)·q(u) 出发，通过矩阵分解将不可观测的 U 替换为可观测量。关键在于利用 P(W|U) 在域间的不变性（模块性假设），将 Q(y|U,x) 重写为仅依赖于 P(W|U) 的形式，边际化后 U 消失
- **设计动机**：Assumption 1 要求 rank(P(W|E,x)) ≥ k_U，即代理变量 W 在不同域间的条件分布足够"多样"，使得域偏移在 U 中产生的变化能通过 W 充分反映。这是矩阵伪逆存在的前提。该假设不仅充分且（在本设置下）必要——违反时可构造反例使识别性丧失
- **推广**：Theorem 1 可推广至 X 和 Y 连续的情形、包含额外观测协变量 Z 的情形（可识别 CATE），以及允许 E→X 直接边的更一般因果图

### 模块二：因果参数化估计器（Causal Parametrisation Estimator）

- **功能**：显式参数化 SCM 中所有因果机制的条件概率矩阵，通过最大似然估计获取参数后计算干预分布
- **核心思路**：参数 θ 包含 P(U|E), Q(U), P(W|U), P(X|U), P(y|U,W,x) 的所有条目。用 softmax 变换将无约束的 logit 参数转化为合法概率。通过 Proposition 3 给出的公式 q(y|do(x)) = diag(P(y|U,W,x)·P(W|U))·Q(U)，将 MLE 得到的 θ̂ 代入得到估计值 q̂_{C,n}
- **设计动机**：虽然 θ 中部分参数不可识别（过参数化），但干预分布作为 θ 的函数是可识别的。Proposition 4 证明了一致性——即使 θ̂ 本身不收敛到真值，其诱导的观测变量分布收敛，足以保证因果效应估计的一致性
- **计算特点**：需要非凸优化求解 MLE，计算开销相对较大

### 模块三：简约参数化估计器（Reduced Parametrisation Estimator）

- **功能**：仅估计计算 Eq.4 所需的最少参数，避免过参数化带来的计算负担
- **核心思路**：直接基于可识别性公式 q(y|do(x)) = P(y|E,x)·P(W|E,x)†·Q(W)，参数向量 η 仅包含可从数据中直接用经验频率估计的联合/条件概率（如 P(W,x,E), P(y,x,E), P(x,E), Q(W,e_T)）。估计器为 q̂_{R,n}(y|do(x)) = h(η̂)，其中 h 是将经验概率转换为干预分布的映射
- **设计动机**：避免了因果参数化中的非凸优化问题，直接用经验频率估计。更重要的是，Proposition 5 证明了渐近正态性，可以通过 delta 方法构造置信区间：σ̂² = ∇h(η̂)ᵀ · Σ̂ · ∇h(η̂)，得到 (1-α) 水平的渐近置信区间
- **实用处理**：估计值和置信区间可能超出 [0,1]，需要裁剪（clipping）

### 损失函数与优化

- **因果参数化**：最大化条件似然 L(θ) = Π p_θ(y,x,w|e)^{n(y,x,w,e)} · Π q_θ(w)^{n(w)}，使用 softmax 约束参数空间，通过梯度迭代优化（非凸问题）
- **简约参数化**：无需优化，直接用经验频率估计 η̂ = (1/n)Σ η^i，计算复杂度低
- **条件数监控**：κ(P(W|E,x)) 可从数据估计（κ̂ ≈ κ），当 κ̂ 较大时标记估计为不可靠

## 实验关键数据

### 表1：模拟实验——点估计绝对误差对比（n=20000, k_E=3, M=10, N=5）

| 方法 | 平均绝对误差 | 说明 |
|------|-------------|------|
| Oracle（干预数据） | 最低 | 使用不可观测的干预分布数据 |
| Causal Estimator | 0.040 | 本文因果参数化估计器 |
| Reduced Estimator | 0.058 | 本文简约参数化估计器，计算更快 |
| NoAdj（无调整） | 较大偏移 | 直接用观测分布估计 |
| NoAdj*（目标域无调整） | 较大偏移 | 使用目标域观测数据（不可用） |
| WAdj（W 调整） | 较大偏移 | 用 W 替代 U 做调整（不合法） |
| WAdj*（目标域 W 调整） | 较大偏移 | 使用目标域数据（不可用） |

### 表2：酒店排名真实数据——q(Y=1|do(X=1)) 估计（25 源域，18 目标域）

| 方法 | 平均绝对误差 | 置信区间中位长度 |
|------|-------------|-----------------|
| Reduced Estimator | **0.044** | **0.14** |
| NoAdj | 0.051 | 0.17 |
| NoAdj* | 0.080 | — |
| WAdj | 0.053 | — |
| WAdj* | 0.075 | — |

### 关键发现

1. 两种估计器的绝对误差均显著低于所有非Oracle基线方法，验证了可识别性理论的实际有效性
2. 估计误差随条件数 κ(P(W|E,x)) 增大而增长，且 κ 可从数据中准确估计（κ̂ ≈ κ），为实际应用提供了可靠性诊断工具
3. 渐近置信区间覆盖率接近名义水平（95%），且区间长度随样本量增大而缩短，验证了一致性
4. 在 Expedia 酒店搜索真实数据上，Reduced Estimator 的置信区间在所有18个目标域上均与 Oracle 区间重叠

## 亮点与洞察

1. **理论贡献扎实**：在目标域仅观测代理变量 W 的极端设置下证明了干预分布的可识别性，且条件（Assumption 1）被证明是必要的，不存在更弱的替代条件
2. **两种互补估计器**：Causal Estimator 误差略低但需非凸优化，Reduced Estimator 提供解析的渐近置信区间且计算高效，实际中可根据需求选择
3. **可诊断性**：条件数 κ 可直接从数据估计，用户可在应用前评估估计的可靠性
4. **连续变量兼容**：虽然核心推导在离散设置下展开，但 X、Y 连续时仍成立；W 连续时可通过适当离散化处理

## 局限性

1. 要求 U 离散且其支撑集大小 k_U 需事先指定或估计——在实际中往往较难确定
2. Assumption 1 要求足够多的"多样"源域（k_E ≥ k_U），在源域数量有限时可能不满足
3. Causal Estimator 需要求解非凸的 MLE，存在局部最优问题；Reduced Estimator 在小样本下可能产生超出 [0,1] 的估计值
4. 目前限于离散 W（连续 W 仅在附录中讨论离散化方案），缺少对高维代理变量的系统处理
5. 实际数据实验仅在酒店排名场景上验证，缺乏更广泛的应用领域测试

## 相关工作与启发

- **Proximal Causal Inference [Miao et al., 2018; Tchetgen et al., 2024]**：需要两个代理变量，且假设因果效应跨域不变。本文只需一个代理+多域数据
- **Domain Adaptation with Proxies [Tsai et al., 2024]**：需在目标域观测 W 和 X，目标是预测性能。本文仅需观测 W，目标是干预分布
- **Latent Variable Models [Louizos et al., 2017; Wang & Blei, 2019]**：显式估计 U 再做调整。本文绕过 U 的估计
- **Transportability [Bareinboim & Pearl, 2013]**：基于纯图形准则判断可识别性。本文额外利用代理的信息量假设

## 评分
- 新颖性: ⭐⭐⭐⭐ 在多域代理因果推断中提出目标域仅需观测W的新设置，理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 模拟实验覆盖一致性/覆盖率/条件数敏感性，真实数据有Oracle对照
- 写作质量: ⭐⭐⭐⭐⭐ 符号体系清晰，定理-证明结构严谨，图表信息量充足
- 价值: ⭐⭐⭐⭐ 建立了新的可识别性结果和实用估计框架，对因果推断和域适应领域有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Isolated Causal Effects of Natural Language](../../ICML2025/causal_inference/isolated_causal_effects_of_natural_language.md)
- [\[ICML 2025\] Estimating Causal Effects in Gaussian Linear SCMs with Finite Data](../../ICML2025/causal_inference/estimating_causal_effects_in_gaussian_linear_scms_with_finite_data.md)
- [\[CVPR 2025\] Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference](../../CVPR2025/causal_inference/image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](../../AAAI2026/causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[ICML 2026\] Rank-Learner: Orthogonal Ranking of Treatment Effects](../../ICML2026/causal_inference/rank-learner_orthogonal_ranking_of_treatment_effects.md)

</div>

<!-- RELATED:END -->
