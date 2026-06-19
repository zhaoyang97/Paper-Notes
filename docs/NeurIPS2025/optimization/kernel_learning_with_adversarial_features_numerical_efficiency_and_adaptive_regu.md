---
title: >-
  [论文解读] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization
description: >-
  [NeurIPS 2025][优化/理论][adversarial training] 提出在再生核希尔伯特空间（RKHS）中将对抗扰动从输入空间转移到特征空间的新范式，使内层最大化可精确求解，并通过迭代加权核岭回归高效优化，同时自适应正则化无需调参即可匹配交叉验证性能。 领域现状：对抗训练是提升模型鲁棒性的核心技术…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "adversarial training"
  - "kernel methods"
  - "RKHS"
  - "adaptive regularization"
  - "multiple kernel learning"
---

# Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization

**会议**: NeurIPS 2025  
**arXiv**: [2510.20883](https://arxiv.org/abs/2510.20883)  
**代码**: [antonior92/adversarial_training_kernel](https://github.com/antonior92/adversarial_training_kernel)  
**领域**: 优化  
**关键词**: adversarial training, kernel methods, RKHS, adaptive regularization, multiple kernel learning

## 一句话总结
提出在再生核希尔伯特空间（RKHS）中将对抗扰动从输入空间转移到特征空间的新范式，使内层最大化可精确求解，并通过迭代加权核岭回归高效优化，同时自适应正则化无需调参即可匹配交叉验证性能。

## 研究背景与动机
**领域现状**：对抗训练是提升模型鲁棒性的核心技术，但经典输入空间对抗训练（如 PGD）需求解难解的 min-max 问题，计算开销大。

**现有痛点**：(a) 输入空间扰动下内层最大化非凹，通常需多步梯度近似；(b) 线性模型中对抗训练与参数收缩（ridge/lasso）的等价关系已被广泛研究，但无穷维核空间中的类似理论尚未建立。

**核心矛盾**：如何在保持对抗鲁棒性保证的前提下，让核方法中的对抗训练既高效可解又具有自适应正则化性质？

**切入角度**：将对抗扰动施加在 RKHS 特征空间而非输入空间，利用特征空间的线性结构获得闭式解。

## 方法详解

### 核心思想：特征空间扰动

经典输入空间对抗训练求解：

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \max_{\|\Delta x_i\| \le \delta} \ell(y_i, f(x_i + \Delta x_i))$$

本文将扰动转移到特征空间 $\mathcal{H}$：

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \max_{\|d\|_\mathcal{H} \le \delta} (y_i - \langle f, \phi(x_i) + d \rangle)^2$$

**关键定理（Proposition 2，内层闭式解）**：当 $\Omega_\mathcal{H} = \{d: \|d\|_\mathcal{H} \le \delta\}$ 时，

$$\max_{d \in \Omega_\mathcal{H}} (y - \langle f, \phi(x) + d \rangle)^2 = (|y - f(x)| + \delta\|f\|_\mathcal{H})^2$$

因此原问题等价于：

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n (|y_i - f(x_i)| + \delta \|f\|_\mathcal{H})^2$$

这与核岭回归形式极为相似，但正则项在括号内而非括号外。

### 松弛性保证（Proposition 1）

特征空间扰动是输入空间扰动的上界松弛：对 $\Omega_\mathcal{X} = \{\Delta x: D_\mathcal{H}(x, x+\Delta x) \le \delta\}$，

$$\max_{d \in \Omega_\mathcal{H}} (y - \langle f, \phi(x) + d \rangle)^2 \ge \max_{\Delta x \in \Omega_\mathcal{X}} (y - f(x + \Delta x))^2$$

对 Gaussian、Matérn、多项式等核均可建立输入空间扰动半径与特征空间扰动半径的对应关系。

### 优化算法：迭代加权核岭回归（Algorithm 1）

利用 $\eta$-trick 变分重写将损失分解为加权最小二乘 + 正则项之和，交替优化：

1. **求解加权核岭回归**：$\hat{f} = \arg\min_f \frac{1}{n}\sum_i w_i (y_i - f(x_i))^2 + \lambda \|f\|_\mathcal{H}^2$
2. **更新权重**：$w_i = 1/\eta_i^0$，$\lambda = \frac{1}{n}\sum_i \delta^2/\eta_i^1$
3. 其中 $\eta_i^0, \eta_i^1$ 由当前残差和范数之比决定
4. 重复直到收敛（通常仅需几步迭代）

计算复杂度与核岭回归相同，$O(n^3)$ 由核矩阵分解主导，兼容 Nyström 近似。

### 泛化界

**Theorem 1（对抗核训练超额风险）**：设 $R = \|f^*\|_\mathcal{H}$，$\sigma$ 为噪声量级，$\delta$ 为对抗半径，则超额风险满足 $\mathcal{E}(\hat{f} - f^*) \le \min(\mathcal{B}_\gamma^{\rm adv}, \mathcal{B}_\beta^{\rm adv})$，其中：

- $\mathcal{B}_\gamma^{\rm adv} = O(\sigma R \gamma)$（取 $\delta \propto \gamma$），$\gamma$ 为 Gaussian 复杂度
- $\mathcal{B}_\beta^{\rm adv} = O(\sigma^2 \beta^2)$（取 $\delta$ 充分小），$\beta$ 为局部 Gaussian 复杂度

**核心优势**：对抗训练以默认 $\delta$ 无需知道噪声水平 $\sigma$ 即可自适应达到近最优率，而核岭回归需要 $\lambda \propto \sigma/R$ 才行。

高斯噪声下对平移不变核，$\gamma = O(1/\sqrt{n})$，得 $\mathcal{B}_\gamma^{\rm adv} = O(\sigma R / \sqrt{n})$（无维度依赖）。

### 多核学习扩展

将框架推广到多核空间 $\bar{\mathcal{H}} = \bigoplus_{j=1}^D \mathcal{H}_j$，扰动集取各核空间球的交集，等价优化：

$$\min \frac{1}{n}\sum_{i=1}^n \left(|y - \sum_j f_j(x)| + \delta \sum_j \|f_j\|_{\mathcal{H}_j}\right)^2$$

形式与经典多核学习高度类似，同样可用迭代加权算法求解。

## 实验关键数据

### Clean 数据性能

| 数据集 | 对抗核训练 $R^2$ | 核岭回归（CV）$R^2$ |
|--------|:---:|:---:|
| Abalone | **0.57** | 0.57 |
| 多数据集平均 | ≈持平或更优 | 需 CV 调参 |

对抗核训练的超参空间更小（仅需调 $\gamma$，$\delta$ 用默认值），但性能与 CV 调参后的核岭回归持平或更好。

### 对抗鲁棒性（Abalone 数据集）

| 训练方法 | 无攻击 | $\ell_2 \le 0.01$ | $\ell_2 \le 0.1$ | $\ell_\infty \le 0.01$ | $\ell_\infty \le 0.1$ |
|----------|:---:|:---:|:---:|:---:|:---:|
| Adv Kern ($\delta \propto n^{-1/2}$) | 0.57 | 0.55 | **0.39** | 0.54 | **0.13** |
| Ridge Kernel (CV) | 0.57 | 0.55 | 0.26 | 0.52 | -0.16 |
| Adv Kern ($\|d\| \le 0.01$) | 0.56 | 0.55 | **0.40** | 0.53 | **0.18** |
| Adv Input ($\ell_2 \le 0.1$) | 0.57 | 0.55 | 0.27 | 0.52 | -0.14 |

**关键发现**：特征空间对抗训练在抵御输入空间攻击时优于直接输入空间对抗训练，尤其在大攻击预算（$\ell_\infty \le 0.1$）下优势显著。

### 自适应正则化

在不同平滑度目标函数和不同核（Matérn-1/2, Matérn-3/2, Gaussian）上，对抗核训练以默认 $\delta \propto 1/\sqrt{n}$ 的测试 MSE 与交叉验证核岭回归的衰减率高度一致，展示出对目标函数正则性的自动适配能力。

## 亮点
1. **闭式内层解**使对抗训练计算量与标准核岭回归相当，避免了多步 PGD
2. **自适应正则化**性质无需调超参即具备——$\delta$ 不依赖噪声水平
3. 泛化界严格建立了特征扰动与输入扰动的松弛关系，给出了理论保证
4. 多核学习扩展使框架可应对多模态场景

## 局限与展望
1. 计算复杂度仍为 $O(n^3)$，大规模数据需 Nyström 近似（文中未展开）
2. 理论分析聚焦 fixed-design 回归，random-design 和分类场景需进一步研究
3. 仅测试了中等规模 UCI 数据集，缺少大规模或高维数据集验证
4. 未与神经网络对抗训练做直接比较

## 与相关工作的对比
- **vs. 线性模型对抗训练**（Xing et al. 2021）：本文将线性结论推广到无穷维 RKHS
- **vs. PGD 对抗训练**（Madry et al. 2018）：闭式内层解避免多步近似，优化更高效
- **vs. TRADES**（Zhang et al. 2019）：TRADES 是近似折中方案，本文在核框架下有精确解
- **vs. 鲁棒核回归（M-estimator）**：目标不同——本文关注对抗扰动鲁棒性而非离群点鲁棒性

## 启发与关联
- 自适应正则化性质可启发设计无需超参搜索的正则化策略
- 多核对抗学习框架可直接应用于多模态学习场景
- 理论可延伸到神经切线核（NTK），为深度网络对抗训练提供理论指导

## 评分
- ⭐ 新颖性: 4/5 — 特征空间扰动视角新颖，闭式解带来实质性计算优势
- ⭐ 实验充分度: 3/5 — 合成+UCI 数据覆盖够用，但缺少高维/大规模验证
- ⭐ 写作质量: 4/5 — 理论推导清晰，结构完整
- ⭐ 综合价值: 4/5 — 理论贡献扎实，方法实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models](fantastic_features_and_where_to_find_them_a_probing_method_to_combine_features_f.md)
- [\[ICML 2025\] AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs](../../ICML2025/optimization/advprompter_fast_adaptive_adversarial_prompting_for_llms.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)

</div>

<!-- RELATED:END -->
