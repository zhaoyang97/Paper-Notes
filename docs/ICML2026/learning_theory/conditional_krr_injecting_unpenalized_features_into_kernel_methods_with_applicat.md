---
title: >-
  [论文解读] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding
description: >-
  [ICML2026][机器学习理论][条件核岭回归] 本文提出条件核岭回归（Conditional KRR）框架，将一组非惩罚特征注入核方法中，通过残差核将其归约为标准 KRR，证明了归约代价为 $\mathcal{O}(1/\sqrt{N})$，并在硬阈值（top-k 本征函数）和软阈值（随机高斯特征）两种设定下验证了条件 KRR 优于标准 KRR 的充分条件。
tags:
  - "ICML2026"
  - "机器学习理论"
  - "条件核岭回归"
  - "条件正定核"
  - "残差核"
  - "核阈值化"
  - "非惩罚特征"
---

# Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding

**会议**: ICML2026  
**arXiv**: [2605.26067](https://arxiv.org/abs/2605.26067)  
**代码**: 无  
**领域**: 机器学习理论  
**关键词**: 条件核岭回归, 条件正定核, 残差核, 核阈值化, 非惩罚特征  

## 一句话总结

本文提出条件核岭回归（Conditional KRR）框架，将一组非惩罚特征注入核方法中，通过残差核将其归约为标准 KRR，证明了归约代价为 $\mathcal{O}(1/\sqrt{N})$，并在硬阈值（top-k 本征函数）和软阈值（随机高斯特征）两种设定下验证了条件 KRR 优于标准 KRR 的充分条件。

## 研究背景与动机

**领域现状**：核岭回归（KRR）是监督学习中的经典方法，近年来在神经切线核、算子逼近、强化学习等领域得到广泛应用。使用 KRR 时需要指定一个正定（PD）核函数 $K(x,y)$，但正定性条件可以放宽为条件正定（CPD），即二次型 $\sum_{ij} K(x_i, x_j) \zeta_i \zeta_j$ 仅需在 $\zeta$ 正交于某个函数类 $\mathcal{F}$ 时非负。

**现有痛点**：已有的 CPD 核研究几乎全部聚焦于 $\mathcal{F}$ 为多项式子空间的情形，对一般函数类 $\mathcal{F}$ 的统计理论几乎空白。更重要的是，即使原始核 $K$ 本身是正定的，将其视为关于一般 $\mathcal{F}$ 的 CPD 核可以自然地引出一种"先线性回归、再核回归残差"的两阶段学习框架，但这种框架缺乏严格的统计保证。

**核心矛盾**：条件 KRR 的优化问题可以分解为线性回归 + 残差核 KRR，但这种分解依赖于对 $\mathcal{F}$-分量的精确知识。在实际中学习器并不知道真实回归函数在 $\mathcal{F}$ 中的投影，因此需要量化"不知道 $f_\parallel$ 所付出的代价"——即 conditioning cost。

**本文目标**：(1) 建立条件 KRR 到标准 KRR 的严格归约理论；(2) 给出 conditioning cost 的高概率上界；(3) 将理论应用于硬阈值和软阈值两种实用设定，给出条件 KRR 优于标准 KRR 的充分条件。

**切入角度**：作者引入"残差核" $K_P = ((I - \Pi_P) \otimes (I - \Pi_P))[K]$，证明其正定性，并将条件 KRR 的原生空间等价于残差核的 RKHS 加上 $\mathcal{F}$。这使得条件 KRR 可以被理解为对目标变量做线性回归后，对残差再跑标准 KRR。

**核心 idea**：将条件 KRR 严格归约为残差核上的标准 KRR，代价仅为 $\mathcal{O}(\log k / \sqrt{N})$，从而可以直接复用已有 KRR 统计理论来分析条件 KRR 的收敛性和泛化性。

## 方法详解

### 整体框架

给定训练集 $\{(x_i, y_i)\}_{i=1}^N$ 和一组非惩罚特征 $\mathcal{F} = \text{span}\{f_1, \ldots, f_k\}$，条件 KRR 求解优化问题 $\min_{f \in \mathcal{H}_K^{\mathcal{F}}} \frac{1}{N}\sum_i (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2$，其中半范数 $\|f\|_{\mathcal{H}_K^{\mathcal{F}}}$ 仅惩罚 $f$ 在 $\mathcal{F}$ 正交补上的分量。等价地，该过程可分解为：(1) 先用 $\mathcal{F}$ 中的特征对 $y$ 做线性回归得到残差 $r$；(2) 再用残差核 $K_P$ 对残差 $r$ 做标准 KRR。

### 关键设计

**1. 残差核构造与原生空间等价：把条件正定核的学习问题翻译成标准正定核问题**

条件 KRR 的麻烦在于核 $K$ 只是关于函数类 $\mathcal{F}$ 条件正定的，半范数只惩罚 $\mathcal{F}$ 正交补上的分量，已有 KRR 理论用不上。作者的破局点是构造残差核

$$K_P(x,y) = \big((I-\Pi_P)\otimes(I-\Pi_P)\big)[K],$$

其中 $\Pi_P$ 是 $L_2(\mathcal{X},P)$ 中到 $\mathcal{F}$ 的投影算子。Theorem 2.1 证明 $K_P$ 是货真价实的正定核，Theorem 3.1 证明条件 KRR 的原生空间 $\mathcal{H}_K^{\mathcal{F}}$ 与 $\mathcal{H}_{K_P}\oplus\mathcal{F}$ 同构，Theorem 3.2 进一步表明条件 KRR 等价于"先用 $\mathcal{F}$ 中特征对 $y$ 做线性回归、再用 $K_P$ 对残差跑标准 KRR"。这条等价关系是全文理论的基石——一旦把问题搬回标准 PD 核，所有已有的 KRR 统计结论（收敛率、泛化界）就能原封不动迁移过来。

**2. Conditioning Cost 的高概率上界：量化"不知道真实 $\mathcal{F}$-投影"要付多少代价**

实际学习器并不知道真回归函数在 $\mathcal{F}$ 上的投影 $f_\parallel$，所以必须刻画与"已知 $f_\parallel$ 的理想学习器"之间的差距。作者定义 conditioning cost $c_{\text{con}}=\mathbb{E}[(\hat f(X)-f_\parallel(X)-h(X))^2]$，Theorem 4.2 证明在概率 $\ge 1-\delta$ 下

$$\mathbb{E}_\varepsilon[c_{\text{con}}] \le C\cdot\|f\|_{\mathcal{H}_K^{\mathcal{F}}}^2\cdot\frac{k\log^{1/2}(2k/\delta)}{\sqrt N}+\frac{c_2\sigma^2}{N}.$$

这条界的关键在于把信号分量和噪声的贡献解耦：当信号完全落在 $\mathcal{F}$ 中（$f_\perp=0$）时第一项消失，代价缩到 $\mathcal{O}(\sigma^2 k/N)$。正是这种解耦，让人能精确判断什么时候注入非惩罚特征划算、什么时候反受其害。

**3. 硬/软阈值的统一分析：揭示谱截断与随机特征在残差核层面的等价**

最后把理论落到两种实用设定上，得出测试误差关于 $k$ 的 U 形条件。硬阈值取 $\mathcal{F}$ 为核 $K$ 的前 $k$ 个 Mercer 本征函数，此时残差核恰是核谱的尾部 $K_P(x,y)=\sum_{i>k}\lambda_i\phi_i(x)\phi_i(y)$；软阈值取 $\mathcal{F}$ 为 $k$ 个高斯随机特征，Theorem 5.2 证明残差核的期望本征值满足 $\mu_i/\lambda_i\approx c\varkappa^2/(\lambda_i+\varkappa)^2$，即大本征值被"软压制"。两种设定都呈现同一条 U 形曲线——$k$ 太小没充分利用非惩罚特征、$k$ 太大 conditioning cost 又涨上来。这个统一视角的价值在于点破：传统谱截断和随机特征方法看似两套，在残差核层面其实定性等价。

## 实验关键数据

### 主实验：Conditioning Cost 验证

| 参数变化 | 实验观测 | 理论预测 |
|----------|----------|----------|
| $N$ 增大 | $\hat{c}_{\text{con}} \sim 1/N$ | $\mathcal{O}(1/\sqrt{N})$（更松） |
| $k$ 增大 | 线性增长 | $\mathcal{O}(k)$ |
| $\sigma^2$ 增大 | 线性增长 | $\mathcal{O}(\sigma^2)$ |
| $f \in \mathcal{F}$ 时 | 仅剩 $\mathcal{O}(\sigma^2 k / N)$ | 与理论一致 |

### U 形测试误差（硬阈值）

| 数据集 / 核 | 标准 KRR (k=0) | 最优条件 KRR | 最优 k | U 形出现 |
|-------------|----------------|-------------|--------|----------|
| 合成数据 ($s=2$) | 基线 | 显著更低 | $k=5$ | 是 |
| MNIST 7-vs-9 / Gaussian 核 | 基线 | 更低 | 中等 $k$ | 是 |
| MNIST 7-vs-9 / NNGP-erf 核 | 基线 | 微弱改善 | 接近 $N$ 时过拟合 | 是（温和） |
| MNIST 7-vs-9 / Laplace 核 | 基线 | 无改善 | — | 否 |

### 关键发现

- **Conditioning cost 实际衰减速率** $\sim 1/N$ 比理论上界 $1/\sqrt{N}$ 更快，提示理论存在收紧空间
- **U 形行为的充分条件**（公式 3）：当信号在前 $k$ 个本征函数上的投影足够强时，条件 KRR 优于标准 KRR；纯噪声数据（$f=0$）下测试 MSE 随 $k$ 单调递增，无 U 形
- **软阈值与硬阈值的等价性**：随机高斯特征产生的残差核在期望意义下等价于对核谱做"软截断"，大本征值被压制、小本征值被放大，与硬截断定性一致
- **Laplace 核不出现过拟合**：因为前 11000 个经验本征函数均对预测有贡献，需要更大样本量才能观察到 U 形

## 亮点与洞察

- **两阶段分解的严格理论基础**：虽然"先回归主信号、再核方法学残差"的策略在 boosting 和残差网络中广泛使用，本文首次为核方法框架下的这种策略提供了完整的统计理论，conditioning cost 上界的推导技巧（利用矩阵扰动和随机矩阵集中不等式）具有普适价值
- **硬/软阈值的统一视角**：将 top-k 本征函数截断和随机特征注入统一为条件 KRR 的两种实例化，通过 Theorem 5.2 揭示了两者在残差核层面的渐近等价性，这个视角可以迁移到神经网络的随机特征分析中
- **弱良性过拟合的刻画**：当信号完全在 $\mathcal{F}$ 中时，噪声只能在正交补方向上被记忆，不影响信号恢复精度——这提供了一种介于良性过拟合与灾难性过拟合之间的新的过拟合模式

## 局限与展望

- **理论仅覆盖 $\lambda > 0$**：当正则化参数 $\lambda = 0$ 时（完美记忆训练集），现有理论失效，而实践中无正则化的过参数化场景正是关注重点
- **Conditioning cost 上界偏松**：实验中观测到 $\sim 1/N$ 的衰减，但理论只给出 $\sim 1/\sqrt{N}$，收紧该上界是开放问题
- **Mercer 本征函数的估计**：硬阈值方法依赖本征函数的精确估计，实际中需要用经验本征函数 $\hat{\phi}_k$ 替代，引入额外误差但文中未详细分析
- **未与深度学习方法对比**：作为纯核方法的理论工作，缺乏与现代神经网络两阶段方法（如残差学习）的实证对比

## 相关工作与启发

- KRR 统计理论：Caponnetto & De Vito 2007（收敛率）、Simon et al. 2023（本征学习框架）
- CPD 核理论：Duchon 1977（多项式 CPD 核）、Meinguet 1979（原生空间构造）
- 两阶段学习：Yang et al. 2023（基网络 + 残差学习）、Freund & Schapire 1997（Boosting）
- 启发：条件 KRR 的思路可推广到深度核学习中——将神经网络特征作为非惩罚分量注入核方法，可能在理论上解释"预训练 + 核微调"范式的优势

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Sequential Kernel-based Conditional Independence Testing via Adaptive Betting](sequential_kernel-based_conditional_independence_testing_via_adaptive_betting.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/learning_theory/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](../../ICLR2026/learning_theory/scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)

</div>

<!-- RELATED:END -->
