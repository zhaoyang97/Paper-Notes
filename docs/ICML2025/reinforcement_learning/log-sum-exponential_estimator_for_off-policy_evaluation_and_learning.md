---
title: >-
  [论文解读] Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning
description: >-
  [ICML2025 Spotlight][强化学习][off-policy learning] 提出基于 log-sum-exponential (LSE) 算子的新型非线性估计器，用于离线策略评估与学习，在重尾奖励和噪声倾向分数场景下显著降低方差并提供理论保证。 现有痛点 现有痛点：领域现状：现状：离线策略学习与评估 (O…
tags:
  - "ICML2025 Spotlight"
  - "强化学习"
  - "off-policy learning"
  - "off-policy evaluation"
  - "inverse propensity score"
  - "heavy-tailed reward"
  - "variance reduction"
  - "log-sum-exponential"
---

# Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning

**会议**: ICML2025 Spotlight  
**arXiv**: [2506.06873](https://arxiv.org/abs/2506.06873)  
**代码**: [GitHub](https://github.com/armin-behnamnia/lse-offpolicy-learning)  
**领域**: 强化学习  
**关键词**: off-policy learning, off-policy evaluation, inverse propensity score, heavy-tailed reward, variance reduction, log-sum-exponential

## 一句话总结
提出基于 log-sum-exponential (LSE) 算子的新型非线性估计器，用于离线策略评估与学习，在重尾奖励和噪声倾向分数场景下显著降低方差并提供理论保证。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：**现状**：离线策略学习与评估 (OPL/OPE) 利用已有的日志化 bandit 反馈数据集进行策略评估和学习，被广泛应用于推荐系统、个性化医疗和广告投放等领域。标准方法是基于逆倾向分数 (IPS) 的估计器。

**痛点**：

**高方差问题**：IPS 估计器在策略分布差异大时方差极高，导致评估不稳定

**重尾奖励**：金融市场、网络广告等场景中奖励分布呈重尾分布，方差甚至未定义，现有估计器（PM、ES、IX 等）均假设有界奖励，无法处理

**噪声倾向分数**：实际场景中常需估计倾向分数而非使用真值，估计误差进一步恶化性能

**Idea**：利用 log-sum-exponential 算子的天然鲁棒性——当 $\lambda < 0$ 时，异常大值 $z_i$ 被 $e^{\lambda z_i} \to 0$ 自动抑制——构建新型非线性估计器，同时在重尾和噪声场景下保持低方差。

## 方法详解

### 核心框架：LSE 估计器

给定日志化 bandit 反馈数据集 $S = (x_i, a_i, p_i, r_i)_{i=1}^n$，LSE 估计器定义为：

$$\hat{V}_{\text{LSE}}^{\lambda}(S, \pi_\theta) = \frac{1}{\lambda} \log\left(\frac{1}{n} \sum_{i=1}^{n} e^{\lambda r_i w_\theta(a_i, x_i)}\right)$$

其中 $\lambda < 0$ 是可调参数，$w_\theta(a,x) = \pi_\theta(a|x) / \pi_0(a|x)$ 是重要性权重。

**关键性质**：
- 当 $\lambda \to 0$ 时退化为标准 IPS 估计器
- 对于 $\lambda < 0$，异常大的加权奖励 $r_i w_\theta$ 通过指数运算被自然压制
- LSE 关于 $\lambda$ 是单调递增函数
- 区别于所有已有线性估计器，LSE 是关于整体加权奖励样本的非线性函数

### 重尾假设

假设加权奖励的 $(1+\epsilon)$ 阶矩有界（$\epsilon \in [0,1]$）：

$$\mathbb{E}\left[(w_\theta(A,X) R)^{1+\epsilon}\right] \leq \nu$$

该假设比传统有界奖励假设弱得多，允许奖励无界。

### 理论结果

**Regret 上界（Theorem 5.3）**：有限策略集 $|\Pi_\theta| < \infty$ 下，以概率 $\geq 1-\delta$ 成立，regret 受控于 $\lambda$、$\nu$、$n$ 的函数。

**收敛速率（Proposition 5.4）**：选择 $\lambda = -n^{-1/(1+\epsilon)}$ 时，regret 上界收敛速率为 $O(n^{-\epsilon/(1+\epsilon)})$。当 $\epsilon=1$（二阶矩有界）时达到 $O(n^{-1/2})$。

**偏差界（Proposition 5.5）**：偏差上界为 $\frac{|\lambda|^\epsilon}{1+\epsilon}\nu + O(1/(n\lambda))$，选择 $\lambda(n) = -n^{-\varsigma}$ 可使偏差渐近为零（渐近无偏）。

**方差界（Proposition 5.7）**：$\mathbb{V}(\hat{V}_{\text{LSE}}^{\lambda}) \leq \frac{1}{n}\mathbb{V}(w_\theta R) \leq \frac{\nu_2}{n}$，对所有 $\lambda < 0$ 均不超过 IPS 方差。

**噪声鲁棒性（Theorem 5.9）**：在奖励分布偏移 $\tilde{P}_{R|X,A}$ 下，regret 额外项正比于 $\text{TV}(P_{R|X,A}, \tilde{P}_{R|X,A}) / \lambda^2$，增大 $|\lambda|$ 可降低噪声代价但增加偏差。

## 实验关键数据

### 主实验：EMNIST 分类

| 场景 | LSE | PM | ES | IX | BanditNet | LS-LIN | OS |
|------|-----|----|----|----|-----------|---------|----|
| τ=1, 真实 PS | **89.29** | 89.08 | 88.45 | 88.14 | 59.90 | 88.30 | 88.74 |
| τ=1, 噪声 PS (b=0.01) | **86.07** | 85.62 | 85.71 | 81.39 | 66.55 | 84.64 | 84.59 |
| τ=10, 噪声 PS (b=0.01) | **82.15** | 80.85 | 81.07 | 77.49 | 27.02 | 78.43 | 21.70 |
| τ=10, 噪声奖励 (Pf=0.1) | **88.29** | 88.22 | 88.19 | 87.93 | 84.89 | 87.50 | 87.68 |

### 关键发现

- **噪声倾向分数**：LSE 在低质量倾向分数（b=0.01）下优势显著，τ=10 时领先 PM 约 1.3%，领先 OS 超过 60%
- **噪声奖励**：Pf=0.1 时 LSE 稳定领先；Pf=0.5 高噪声下 LSE 仍保持竞争力
- **方差稳定性**：LSE 的标准差普遍最低，体现了理论预测的方差缩减效果
- **Pareto 分布实验**（Table 1）：LSE 在 n=10 时 MSE 仅 0.13（Monte-Carlo 为 1.54），方差降低 15 倍

## 亮点与洞察

1. **优雅的设计直觉**：利用 $e^{\lambda z} \to 0$（$\lambda<0, z\to\infty$）自然压制异常值，无需显式截断
2. **理论全面性**：同时覆盖 OPE 和 OPL 场景，包含 regret、偏差、方差、鲁棒性的完整理论体系
3. **更弱的假设**：仅要求加权奖励 $(1+\epsilon)$ 阶矩有界，允许无界奖励和重尾分布
4. **可微性**：LSE 关于策略参数可微，避免了截断 IPS 的优化困难
5. **亚高斯尾部**：LSE 的集中度具有亚高斯型尾部行为

## 局限与展望

1. **有限策略集假设**：主要定理假设 $|\Pi_\theta| < \infty$，需要 VC 维或 PAC-Bayes 扩展到连续策略空间
2. **参数 $\lambda$ 选择**：理论最优 $\lambda = -n^{-1/(1+\epsilon)}$ 需要知道 $\epsilon$，实际中需要交叉验证
3. **偏差代价**：LSE 引入的偏差在小样本时不可忽略，Bias-Variance 权衡需要仔细调节
4. **Model-free 限制**：仅考虑 model-free 设置，与 doubly-robust 等 model-based 方法的结合值得探索（附录 G.3 初步讨论）
5. **实验规模**：主要在 EMNIST 等中等规模数据集上验证，大规模推荐系统等场景待验证

## 相关工作与启发

- **IPS 系列**：IPS → SN-IPS → Truncated IPS → PM → ES → IX → LS，LSE 在此谱系中开辟了非线性方向
- **Tilted ERM（Li et al., 2023）**：LSE 与 tilted empirical risk 的联系，从监督学习借鉴鲁棒性分析
- **Heavy-tailed Bandits（Bubeck et al., 2013）**：有限矩假设的灵感来源
- **PAC-Bayes 扩展**（附录 D.6）：连接到 PAC-Bayesian 分析框架

## 评分
- 新颖性: ⭐⭐⭐⭐ (LSE 算子在 off-policy 中的应用新颖，非线性估计器方向独特)
- 实验充分度: ⭐⭐⭐⭐ (多场景覆盖，消融充分，但规模偏小)
- 写作质量: ⭐⭐⭐⭐ (理论与实验结合紧密，符号体系清晰)
- 价值: ⭐⭐⭐⭐ (为离线强化学习提供了实用且有理论保障的新工具)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](../../NeurIPS2025/reinforcement_learning/bootstrap_off-policy_with_world_model.md)
- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](../../AAAI2026/reinforcement_learning/behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](solving_zero-sum_convex_markov_games.md)

</div>

<!-- RELATED:END -->
