---
title: >-
  [论文解读] QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions
description: >-
  [ICML 2025][LLM 其他][分位数估计] 提出 QuEst 框架，将少量高质量观测数据与大量模型预测（imputed）数据相结合，对分位数相关的分布度量（QBDM）给出更精确的点估计和严格的置信区间，覆盖 CVaR、Interval-VaR 等经典指标。 现有的 Prediction-Powered Infere…
tags:
  - "ICML 2025"
  - "LLM 其他"
  - "分位数估计"
  - "预测驱动推断"
  - "置信区间"
  - "分布度量"
  - "LLM自动评估"
---

# QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions

**会议**: ICML 2025  
**arXiv**: [2507.05220](https://arxiv.org/abs/2507.05220)  
**代码**: [btleyre/quest](https://github.com/btleyre/quest)  
**领域**: LLM/NLP  
**关键词**: 分位数估计, 预测驱动推断, 置信区间, 分布度量, LLM自动评估

## 一句话总结

提出 QuEst 框架，将少量高质量观测数据与大量模型预测（imputed）数据相结合，对分位数相关的分布度量（QBDM）给出更精确的点估计和严格的置信区间，覆盖 CVaR、Interval-VaR 等经典指标。

## 研究背景与动机

现有的 Prediction-Powered Inference（PPI）框架可以将少量金标准标签与大量模型预测结合，对均值、回归系数等 M-estimator 进行混合推断。然而，PPI 的适用范围局限于 M-estimator，无法处理许多关键的**分位数相关分布度量**（Quantile-Based Distributional Measures, QBDM），例如：

- **CVaR**（条件风险值）：刻画尾部极端表现，在金融风控和 LLM 安全评估中至关重要
- **Interval-VaR**：关注某段人群/数据的平均值（如收入最低 20%）
- **VaR**（单分位数）：虽可被 PPI 处理，但 PPI 中缺少最优 λ 选择

在经济学、社会学、基因组学和 LLM 安全评估等领域，决策者往往需要理解分布的尾部行为或特定人群分段，而非仅关注均值。在标注数据昂贵的场景下，仅用少量观测数据估计这些指标会产生很大方差，而纯模型预测又存在系统偏差。QuEst 正是为填补这一空白而提出的。

## 方法详解

### 整体框架

QuEst 的核心思想是对 QBDM 定义一个混合估计量，利用大量预测数据降低方差，同时用少量观测数据进行偏差校正。

**QBDM 的统一定义**：给定 CDF $F$，QBDM 定义为

$$Q_\psi(F) = \int_0^1 \psi(p) F^{-1}(p) dp$$

其中 $\psi \geq 0$ 且 $\int \psi(p) dp = 1$ 是权重函数。通过选取不同 $\psi$ 可以恢复多种经典指标：

| 度量 | 权重函数 $\psi(p)$ | 应用场景 |
|------|-------------------|---------|
| 期望均值 | $\psi(p) = 1$ | 通用 |
| β-VaR | $\psi(p) = \delta_\beta(p)$ | 金融风险、分位数 |
| β-CVaR | $\psi(p) = \frac{1}{1-\beta}$ 当 $p \geq \beta$ | 尾部风险评估 |
| Interval-VaR | $\psi(p) = \frac{1}{\beta_2 - \beta_1}$ 当 $p \in [\beta_1, \beta_2]$ | 人群分段分析 |

**数据设置**：

- 少量标注数据集（观测数据）：$\{(X_i, Y_i, \tilde{Y}_i)\}_{i=1}^n$，$n$ 较小
- 大量未标注数据集（预测数据）：$\{(X_j^u, \tilde{Y}_j^u)\}_{j=1}^N$，$N \gg n$
- $\tilde{Y}$ 由某预测模型 $g$ 产生，$M(X,Y)$ 为用户定义的度量函数

### 关键设计

**1. QuEst 混合估计量**

$$\hat{Q}_\psi(\lambda) = \lambda \cdot Q_\psi(\tilde{F}_N^u) + \left( Q_\psi(F_n) - \lambda \cdot Q_\psi(\tilde{F}_n) \right)$$

- 第一项 $\lambda Q_\psi(\tilde{F}_N^u)$：利用大量预测数据的经验分位数函数，提供低方差估计
- 第二项 $Q_\psi(F_n) - \lambda Q_\psi(\tilde{F}_n)$：用观测数据估计并校正预测偏差
- $\lambda = 0$ 退化为仅用观测数据的经典估计量
- $\lambda = 1$ 完全依赖预测数据并用观测数据校正偏差

**2. 渐近正态性（核心定理）**

论文利用 L-统计量的经典理论，将 $Q_\psi(F_n)$ 展开为有序统计量的加权和：

$$Q_\psi(F_n) = \sum_{i=1}^n \left[ \int_{(i-1)/n}^{i/n} \psi(p)dp \right] M_{(i)}$$

基于此推导出 Theorem 3.1：对任意固定 $\lambda$，在正则条件下

$$\sqrt{n} \left( \hat{Q}_\psi(\lambda) - Q_\psi(F) \right) \xrightarrow{D} \mathcal{N}(0, \rho_\psi^2(\lambda, F, \tilde{F}))$$

其中渐近方差为

$$\rho_\psi^2(\lambda, F, \tilde{F}) = \lambda^2(1+r)\sigma_\psi^2(\tilde{F}) + \sigma_\psi^2(F) - 2\lambda \eta_\psi(F, \tilde{F})$$

这里 $r = n/N$，$\eta_\psi$ 为观测和预测分布间 QBDM 的协方差。

**3. 最优 λ 的闭式解**

最小化 $\rho_\psi^2$ 关于 $\lambda$ 得到：

$$\hat{\lambda} = \frac{\eta_\psi(F_n, \tilde{F}_n)}{(1 + n/N) \sigma_\psi^2(\tilde{F}_N^u)}$$

关键性质：最终渐近方差保证不超过经典估计量的方差，即

$$\rho_\psi^2(\hat{\lambda}) = \sigma_\psi^2(F_n) - \frac{(\eta_\psi(F_n, \tilde{F}_n))^2}{(1+n/N)\sigma_\psi^2(\tilde{F}_N^u)} \leq \sigma_\psi^2(F_n)$$

**4. 多维扩展（Multidimensional QuEst）**

同时估计 $k$ 个 QBDM，通过多维 CLT 给出联合置信区域：

$$\hat{V}^{-1/2}\sqrt{n}\left(\hat{\mathbf{Q}}(\psi_{1:k}, \hat{\lambda}_{1:k}) - \mathbf{Q}(\psi_{1:k}, F)\right) \xrightarrow{D} \mathcal{N}(\mathbf{0}, I)$$

相比逐个估计后 Bonferroni 校正，避免了过于保守的置信区域。

**5. QuEst-Opt：广义权重函数优化**

进一步允许对预测数据施加不同于目标 QBDM 的自适应权重函数 $\tilde{\psi}$：

$$\hat{Q}(\psi, \tilde{\psi}) = \underbrace{Q_{\tilde{\psi}}(\tilde{F}_N^u) - Q_{\tilde{\psi}}(\tilde{F}_n)}_{\text{自适应项}} + \underbrace{Q_\psi(F_n)}_{\text{目标 QBDM}}$$

将 $\tilde{\psi}(\cdot) = \xi^T \phi(\cdot)$ 参数化后，方差关于 $\xi$ 是凸函数，通过带正则化的凸优化求解最优 $\xi$，仍满足 CLT。在数据存在异方差性时效果显著。

### 损失函数 / 训练策略

QuEst 是纯推断框架，不涉及模型训练，其核心"优化"在于：

1. **λ 选择**：最小化渐近方差的闭式解，无需超参数
2. **ξ 优化**（QuEst-Opt）：带 L2 正则化的凸优化问题 $\min_\xi \rho_\psi^2(\xi) + \frac{\alpha}{2}\|\xi\|^2$
3. **置信区间构造**：$\hat{Q}_\psi(\hat{\lambda}) \pm z_{1-\alpha/2} \cdot \hat{SE}$，其中 $\hat{SE} = \rho_\psi(\hat{\lambda}) / \sqrt{n}$

## 实验关键数据

### 主实验

实验跨越 3 类场景、6+ 个数据集，涵盖经济学、基因组学、公众舆论和 LLM 评估。

| 数据集 | 指标 | QuEst 效果（n=100） | 仅观测 | 提升 |
|--------|------|---------------------|--------|------|
| PovertyMap | Interval-VaR | 显著更低误差 | 基线 | ~30-40% 误差降低 |
| GeneExpression | CVaR (top 20%) | ~50% 降低 | 基线 | ~50% 误差与区间宽度降低 |
| OpinionQA | Interval-VaR (中间50%) | 明显更优 | 基线 | 显著优于纯预测和纯观测 |
| 红队毒性评估 | CVaR (最差25%) | 持续更低误差 | 基线 | 模型排名相关性更高 |
| 新闻摘要 (XSum) | 多维 Interval-VaR | MSE 降低 7% | 基线 | 置信域体积降低 34% (单变量) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| QuEst vs PPI (VaR) | 估计误差更低 | 自适应 λ 选择优于 PPI 固定 λ |
| QuEst-Opt vs QuEst | 异方差数据上进一步降低误差 | 广义权重带来额外方差缩减 |
| 单变量 vs 多变量置信域 | 体积：1.04e-2 vs 5.17e-3 | 多维 QuEst 置信域最小 |
| 标注者相关性影响 | 相关性↑ → QuEst 收益↑ | 与直觉一致 |
| λ clipping [0,1] | 小样本更稳定 | 不影响渐近性质 |

### 关键发现

1. **低数据量场景收益最大**：仅 100 个金标准样本时，QuEst 在 GeneExpression 上实现约 50% 的误差和区间宽度降低
2. **保证不差于经典方法**：最优 λ 的渐近方差严格不超过仅用观测数据的方差
3. **自适应程度**：QuEst 自动根据预测质量调节 λ，烂模型时 λ→0 退化为经典估计
4. **多维推断避免 Bonferroni 损失**：联合置信域比逐个估计后 Bonferroni 校正更紧
5. **红队评估中模型排名**：QuEst 产生的 8 个候选模型毒性排名与真实排名的相关性最高

## 亮点与洞察

1. **统一且优雅的理论框架**：通过 L-统计量理论将 QBDM 的混合推断问题纳入统一框架，λ 有闭式解且保证不劣于经典方法
2. **零超参数**：λ 自动求解，用户只需选择目标 QBDM 的权重函数 ψ
3. **广泛适用性**：从经济学财富分析到 LLM 安全红队评估，应用场景丰富
4. **QuEst-Opt 的可扩展性**：通过参数化自适应权重函数实现进一步方差缩减，方差目标保持凸性
5. **对 LLM Eval 的实际价值**：用昂贵大模型标注少量样本 + 便宜小模型大量标注，同时获得可靠的尾部风险估计，在 LLM 安全评估中尤为重要

## 局限与展望

1. **需要预测数据与观测数据来自同一输入分布**：协变量漂移场景未被覆盖
2. **异方差严重时仍依赖 QuEst-Opt**：基础 QuEst 仅用标量 λ，自适应能力有限
3. **缺少对"需要多大 N"的理论刻画**：未明确给出预测数据量下界
4. **置信区间依赖渐近理论**：小样本（n < 50）时有限样本覆盖率可能不理想
5. **正则化参数 α 的选择**：QuEst-Opt 中 α 理论上可任意小，但实际影响未深入探讨

## 相关工作与启发

- **PPI / PPI++**（Angelopoulos et al., 2023, 2024）：QuEst 直接扩展 PPI 到 QBDM 家族
- **Cross-PPI**（Fisch et al., 2024）：另一种混合推断方法
- **LLM Auto-Evaluation**（Boyeau et al., 2024; Eyre & Madras, 2024）：QuEst 在此方向的应用具有实际意义
- **启发**：该框架的思想可以推广到其他非 M-estimator 的统计量，如 Gini 系数、Lorenz 曲线等

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 首次将 PPI 推广到完整 QBDM 家族，理论扎实 |
| 实用性 | 4.5 | 开源代码，零超参数，LLM eval 场景直接可用 |
| 理论严谨性 | 5 | CLT 完整推导，最优 λ 闭式解，凸优化保证 |
| 实验充分性 | 4 | 多场景覆盖全面，但缺少极端小样本分析 |
| 写作质量 | 4 | 结构清晰，符号一致，但公式密集 |
| **总分** | **4.3** | 扎实的理论工作，弥补 PPI 的重要空白 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](la_rosa_enhancing_llm_efficiency_via_layerwise_rotated_sparse_activation.md)
- [\[ACL 2025\] Not Quite Sherlock Holmes: Language Model Predictions Do Not Reliably Differentiate Impossible from Improbable Events](../../ACL2025/llm_nlp/not_quite_sherlock_holmes_language_model_predictions_do_not_reliably_differentia.md)
- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](../../ACL2026/llm_nlp/text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[ACL 2025\] LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](../../ACL2025/llm_nlp/llm_braces_straightening.md)

</div>

<!-- RELATED:END -->
