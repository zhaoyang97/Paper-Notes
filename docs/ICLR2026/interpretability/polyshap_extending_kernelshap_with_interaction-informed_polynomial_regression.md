---
title: >-
  [论文解读] PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression
description: >-
  [ICLR 2026][可解释性][Shapley值] 本文提出 PolySHAP，通过将 KernelSHAP 的线性近似扩展为高阶多项式回归来捕获特征间的非线性交互，从而提升 Shapley 值的估计精度；并从理论上证明了配对采样（paired sampling）等价于二阶 PolySHAP，首次解释了配对采样启发式方法优越性能的根本原因。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "Shapley值"
  - "可解释AI"
  - "多项式回归"
  - "特征交互"
  - "KernelSHAP"
---

# PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression

**会议**: ICLR 2026  
**arXiv**: [2601.18608](https://arxiv.org/abs/2601.18608)  
**代码**: [GitHub](https://github.com/FFmgll/PolySHAP)  
**领域**: 可解释性  
**关键词**: Shapley值, 可解释AI, 多项式回归, 特征交互, KernelSHAP

## 一句话总结

本文提出 PolySHAP，通过将 KernelSHAP 的线性近似扩展为高阶多项式回归来捕获特征间的非线性交互，从而提升 Shapley 值的估计精度；并从理论上证明了配对采样（paired sampling）等价于二阶 PolySHAP，首次解释了配对采样启发式方法优越性能的根本原因。

## 研究背景与动机

Shapley 值是可解释 AI 中最核心的博弈论工具之一，用于量化各特征对模型预测的贡献。然而，对于 $d$ 个特征的模型，精确计算 Shapley 值需要 $2^d$ 次博弈评估，计算代价极高。KernelSHAP 通过将博弈函数 $\nu$ 近似为线性函数来规避指数级开销，但线性近似本质上无法捕获特征间的非线性交互效应，限制了估计精度。

此外，配对采样（paired sampling）作为一种广泛使用的启发式策略，能显著提升 KernelSHAP 的估计质量，但其优越性能背后的理论机制一直未被充分理解。本文从多项式回归的角度，为以上两个问题提供了统一的理论框架和实践方案。

## 方法详解

### 整体框架

PolySHAP 把 KernelSHAP 的线性博弈近似换成带交互项的高阶多项式：先指定一个交互前沿 $\mathcal{I}$（即要显式建模哪些特征组合），用带权最小二乘拟合出含交互项的多项式系数，再用闭式公式把这些系数折算回满足 efficiency 的 Shapley 值。整套方法不改变 KernelSHAP 的采样—回归骨架，只是把回归的特征空间从单特征扩展到了交互项。

### 关键设计

**1. PolySHAP 交互表示：用多项式回归代替线性回归**

KernelSHAP 的瓶颈在于它用线性函数拟合博弈 $\nu$，天然丢掉了所有特征间的交互效应，于是只能捕捉每个特征的"独立"贡献。PolySHAP 直接把回归目标从单特征扩展成带交互项的多项式 $\phi^{\mathcal{I}} \in \mathbb{R}^{d'}$（维度 $d' = d + |\mathcal{I}|$）：既保留 $d$ 个单特征项，又显式纳入交互前沿 $\mathcal{I}$ 里指定的那些特征组合项，再做一次带约束的加权最小二乘把它们一起拟合出来：

$$\phi^{\mathcal{I}}[\nu] := \arg\min_{\phi \in \mathbb{R}^{d'}: \langle\phi,\mathbf{1}\rangle = \nu(D)} \sum_{S \subseteq D} \mu(S)\left(\nu(S) - \sum_{T \in D \cup \mathcal{I}} \phi_T \prod_{j \in T} \mathbb{1}[j \in S]\right)^2$$

拟合出来的多项式系数并不直接就是 Shapley 值——交互项 $\phi_S^{\mathcal{I}}$ 的贡献还混在组合里，需要按定理 4.3 把它沿阶数均摊回参与该交互的每个特征：$\phi_i^{SV}[\nu] = \phi_i^{\mathcal{I}} + \sum_{S \in \mathcal{I}: i \in S} \frac{\phi_S^{\mathcal{I}}}{|S|}$。多项式比线性表达力强、对 $\nu$ 拟合得更准，折算回去的 Shapley 估计自然误差更小，这是整个方法精度提升的根源。

**2. 配对采样等价性定理：解释一个长期没被理解的启发式**

配对采样（每次同时采子集 $S$ 和它的补集 $D \setminus S$）一直被当成能显著提升 KernelSHAP 的经验技巧，但为什么有效从没被讲清。本文用定理 5.1 给出答案：在配对采样下，KernelSHAP 的输出**精确等于** 2-PolySHAP——也就是说，"配对"这一步本身就隐式地把所有二阶交互都吃进了估计里，等价于免费做了一次二阶多项式回归。这第一次从理论上点明了配对采样增益的来源，同时也带来一个实践推论：若已经在用配对采样，PolySHAP 真正额外的收益要等到加入三阶交互才开始显现。

**3. $k$-可加交互前沿：按预算逐阶加交互项**

交互项不能无脑全加——$k$ 阶交互的数量按 $\binom{d}{k}$ 爆炸，高维下根本拟合不起。PolySHAP 用交互前沿 $\mathcal{I}$ 来界定"这一次到底显式建模哪些组合"，并定义 $k$-可加前沿 $\mathcal{I}_{\leq k} = \{S \subseteq D : 2 \leq |S| \leq k\}$，对应 $k$-PolySHAP：$k=1$ 时没有任何交互项、退化回 KernelSHAP，$k=2$ 含全部二阶交互，逐阶往上加表达力越强。为了在高维下仍能平滑降级，论文进一步引入**部分交互前沿** $\mathcal{I}_\ell$——当采样预算撑不起完整 $k$ 阶时，只选择性地纳入一部分高阶项，让方法在算力受限时不至于直接失效。

**4. 杠杆分数采样：在扩大的特征空间里保住近似保证**

特征空间从 $d$ 涨到 $d'$ 之后，如果还沿用 KernelSHAP 按 Shapley 权重采样子集，回归矩阵的条件数会变差、解的方差不可控。PolySHAP 改用杠杆分数采样（leverage score sampling）：按每个子集对回归矩阵的杠杆分数（leverage score，即它对最小二乘解的影响力）来决定采样概率，把采样预算优先花在对解最关键的子集上。由此可证，在 $m = O(d' \log(d'/\delta) + d'/({\epsilon\delta}))$ 的子集预算下，就能以 $1-\delta$ 的概率保证 $\epsilon$ 级的近似质量，使扩大后的多项式回归仍有理论保证。

### 损失函数 / 训练策略

求解的是带约束加权最小二乘，约束 $\langle\phi,\mathbf{1}\rangle = \nu(D)$ 对应 Shapley 值的 efficiency 性质（各特征贡献加和等于全特征博弈值）。实现上用投影矩阵 $\mathbf{P}_{d'}$ 把约束问题转成无约束问题闭式求解，并用 border trick 对极小尺寸的子集直接穷举而非采样，进一步降低小子集上的估计方差。

## 实验关键数据

### 主实验

在 15 个不同的解释博弈上（涵盖表格、图像、语言等领域，$d$ 从 8 到 101），对比 PolySHAP 与多种基线方法。

| 数据集/博弈 | 指标 | PolySHAP (3阶) | KernelSHAP | 改进 |
|-------------|------|---------------|------------|------|
| Housing ($d=8$) | MSE | 最优 | 基线 | 显著降低 MSE |
| Adult ($d=14$) | MSE | 最优 | 基线 | 显著降低 MSE |
| Estate ($d=15$) | MSE | 最优 | 基线 | 显著降低 MSE |
| Cancer ($d=30$) | MSE | 最优 | 基线 | 显著降低 MSE |
| CG60 ($d=60$) | MSE | 小幅改进 | 基线 | 仅小幅提升（高维限制） |

### 消融实验

| 配置 | 关键指标 (MSE) | 说明 |
|------|---------------|------|
| 1-PolySHAP (= KernelSHAP) | 基线 | 无交互项 |
| 2-PolySHAP | 显著改进 | 加入所有二阶交互 |
| 2-PolySHAP (50%) | 中等改进 | 仅加入 50% 二阶交互 |
| 3-PolySHAP | 最优 | 加入三阶交互，低维场景增益最大 |
| 配对 KernelSHAP vs 配对 2-PolySHAP | 完全一致 | 实验验证定理 5.1 |
| 配对 3-PolySHAP vs 配对 4-PolySHAP | 几乎一致 | 暗示更高阶存在类似等价关系 |

### 关键发现

- 加入任意数量的交互项都能改善 Shapley 值近似质量
- 在配对采样下，KernelSHAP 自动获得了 2-PolySHAP 的性能，因此实践中 PolySHAP 的增益从三阶交互开始体现
- 高维场景（$d \geq 60$）中，可加入的三阶交互数量有限，增益较小
- RegressionMSR 是唯一能与 PolySHAP 媲美的基线，但它依赖 XGBoost 树模型，在某些博弈上表现不稳定

## 亮点与洞察

- **理论贡献重大**: 配对采样等价于 2-PolySHAP 的发现是一个优美的理论结果，解答了长期以来的实践困惑
- **方法自然优雅**: 从线性到多项式的扩展思路简洁，且保持了一致性保证
- **统一视角**: 将 KernelSHAP、Faith-SHAP、$k_{ADD}$-SHAP 等方法纳入统一框架
- **投影引理**: 提出的技术性投影引理（Lemma A.1）在证明多个定理中发挥关键作用

## 局限与展望

- 高维场景中三阶交互项组合数爆炸（$\binom{d}{3}$），实际可加入的交互项有限
- 猜测配对 $k$-PolySHAP 等价于 $(k+1)$-PolySHAP（$k$ 为奇数）但未能证明
- 交互前沿的选择较为通用（按阶全加），未利用特定问题的交互结构信息
- 运行时间分析较为理论化，大规模实际应用中的效率有待验证

## 相关工作与启发

- 与 RegressionMSR（Witter et al., 2025）对比：PolySHAP 无需额外回归调整步骤即可保持一致性
- 与 $k_{ADD}$-SHAP（Pelegrina et al., 2023）关系：PolySHAP 简化并推广了其收敛性证明
- 启发：未来可结合交互检测方法（如 Tsang et al., 2020）或图结构信息来构建更智能的交互前沿

## 评分

- 新颖性: ⭐⭐⭐⭐ 多项式扩展思路自然但不算突破性，配对采样等价性定理是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个博弈覆盖表格/图像/语言，多种基线对比全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，图表直观，叙述流畅
- 价值: ⭐⭐⭐⭐ 对 XAI 领域的 Shapley 值估计方法有实质推进，配对采样理论解释意义深远

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](../../ICML2026/interpretability/polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](../../ICML2026/interpretability/a_deep_learning_model_of_mental_rotation_informed_by_interactive_vr_experiments.md)
- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](../../ICML2026/interpretability/breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)
- [\[ACL 2025\] Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages](../../ACL2025/interpretability/bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)

</div>

<!-- RELATED:END -->
