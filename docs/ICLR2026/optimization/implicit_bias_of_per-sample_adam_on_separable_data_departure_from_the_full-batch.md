---
title: >-
  [论文解读] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime
description: >-
  [ICLR 2026][优化/理论][Adam] 首次证明mini-batch Adam的隐式偏差与full-batch不同：构造数据集使单样本Adam收敛到 $\ell_2$ 最大间隔分类器（而full-batch Adam收敛到 $\ell_\infty$），并通过AdamProxy刻画一般数据集上的数据自适应Mahalanobis范数间隔最大化行为。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "Adam"
  - "隐式偏差"
  - "最大间隔"
  - "Mini-batch"
  - "Mahalanobis范数"
---

# Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime

**会议**: ICLR 2026  
**arXiv**: [2510.26303](https://arxiv.org/abs/2510.26303)  
**代码**: 无  
**领域**: 其他  
**关键词**: Adam, 隐式偏差, 最大间隔, Mini-batch, Mahalanobis范数

## 一句话总结
首次证明mini-batch Adam的隐式偏差与full-batch不同：构造数据集使单样本Adam收敛到 $\ell_2$ 最大间隔分类器（而full-batch Adam收敛到 $\ell_\infty$），并通过AdamProxy刻画一般数据集上的数据自适应Mahalanobis范数间隔最大化行为。

## 研究背景与动机

**领域现状**：优化算法的隐式偏差决定了过参数化模型中哪个全局最优被选择。GD收敛到 $\ell_2$ 最大间隔解，full-batch Adam收敛到 $\ell_\infty$ 最大间隔解。SGD不改变GD的偏差（任何batch size都收敛到 $\ell_2$）。

**现有痛点**：现有Adam隐式偏差分析局限于full-batch设置。实际训练使用mini-batch，但不清楚mini-batch是否改变Adam的 $\ell_\infty$ 偏差。直觉上SGD不变→Adam也不变？

**核心矛盾**：实验发现mini-batch Adam（batch=1）在高斯数据上收敛到的方向与full-batch不同，更接近 $\ell_2$ 最大间隔！这与SGD的行为形成鲜明对比。

**切入角度**：通过分析单样本Adam（Inc-Adam）的epoch-wise更新的渐近形式，揭示预条件器跟踪的是单样本梯度平方的加权和（而非full-batch梯度的平方），导致自适应性质改变。

## 方法详解

### 整体框架

整篇分析围绕单样本 Adam（Inc-Adam，batch=1 且逐样本顺序更新）展开，分两步推进：先在一类高度结构化的 Scaled Rademacher 数据上给出精确收敛结论，证明 Inc-Adam 收敛到 $\ell_2$ 最大间隔，从而与 full-batch Adam 的 $\ell_\infty$ 偏差正面对撞；再对任意可分数据集引入 AdamProxy 代理算法，把一般情形的隐式偏差刻画为一个数据自适应的 Mahalanobis 范数间隔最大化问题。

### 关键设计

**1. Epoch-wise 近似：把逐样本更新折叠成一步等效梯度**

直接分析 batch=1 的逐样本轨迹很难，因为一个 epoch 内权重被更新了 $n$ 次、预条件器也在不断变化。本文的做法是把整个 epoch 的净位移近似成单步更新（Proposition 2.5）：

$$w_{r+1}^0 - w_r^0 \approx -\eta \sum_i \frac{\sum_j \beta_1^{(i,j)} \nabla \mathcal{L}_j(w)}{\sqrt{\sum_j \beta_2^{(i,j)} \nabla \mathcal{L}_j(w)^2}}$$

把它和 full-batch Adam 的渐近形式（退化为 SignGD：$w_{t+1} - w_t \approx -\eta \cdot \text{sign}(\nabla \mathcal{L}(w))$）放在一起，关键差异立刻显现：Inc-Adam 的预条件器分母里是**单样本梯度平方的加权和** $\sum_i (\nabla \mathcal{L}_i)^2$，而 full-batch 用的是全批梯度的平方 $(\sum_i \nabla \mathcal{L}_i)^2$，两者并不相等。正是这一项差异在后续传导成完全不同的隐式偏差，也解释了为什么单纯类比 SGD「batch size 无关」会失效。这一步是整篇分析的支点：它把不可解的完整 Adam 动力学换成只依赖当前权重的可处理表达式。

**2. Scaled Rademacher 数据上的精确结果：构造一个让自适应性消失的极端场景**

为了把上面的直觉坐实成定理，作者构造了 Scaled Rademacher（SR）数据——每个样本各坐标绝对值相等（形如 $x_i = (a_i, \pm a_i, \pm a_i, \pm a_i)$）。在这种对称结构下，Inc-Adam 预条件器的逐坐标自适应被完全抹平，第一步得到的等效更新退化成一个加权归一化 GD（其中权重 $a_i(r)$ 随 epoch 数 $r$ 变化，需借助 Corollary 3.2 把它夹在两个正常数之间，保证没有样本贡献消失）。于是 Theorem 3.3 给出它收敛到 $\ell_2$ 最大间隔分类器。与此同时 full-batch Adam 在同一数据上仍走向 $\ell_\infty$ 最大间隔，两者构成最干净的极端对比，证明「mini-batch 改变 Adam 偏差」不是数值假象，而是可证的结构性事实。

**3. AdamProxy：把一般数据集的偏差归约成 Mahalanobis 间隔最大化**

SR 数据太特殊，要覆盖任意可分数据集，作者在 $\beta_2 \to 1$ 的极限下取简化的代理更新（Proposition 4.1），称为 AdamProxy：

$$\delta_t = \frac{\nabla \mathcal{L}(w)}{\sqrt{\sum_i \nabla \mathcal{L}_i(w)^2}}$$

这个极限有实际意义（实践中 $\beta_2$ 本就取到接近 1），它保留了「单样本梯度平方求和」这一关键项，却剥掉了难以处理的瞬态动量耦合。在此基础上证明其收敛方向求解的是一个 Mahalanobis 范数下的间隔最大化 $\max \min_i \frac{x_i^\top w}{\|w\|_M}$，其中度量矩阵 $M$ 由数据通过一个对偶不动点方程 $T(\mathbf{c})=\mathbf{c}$ 自适应确定（Theorem 4.8，可用 Algorithm 3 的不动点迭代数值求解）。这把「Adam 收敛到什么方向」从逐例分析升格成一个统一的几何问题：$\ell_2$ 与 $\ell_\infty$ 都只是 $M$ 取特殊形式时的特例，而一般数据上的收敛方向可以两者都不是。

**4. Signum 的不变性：用对照说明偏差变化源自预条件器而非动量**

为排除「是动量或 sign 操作导致差异」的解释，作者把带动量的 SignSGD（Signum）作为对照：当动量参数足够接近 1 时，它在任意 batch size 下都仍收敛到 $\ell_\infty$ 最大间隔（Theorem 5.1）。原因是 sign 操作只看梯度符号、把幅值信息整体丢弃，因而对「单样本平方和 vs 全批平方」这一差异天然免疫。这条对照反过来坐实了 Adam 偏差对采样方式敏感的根源正是其平方根预条件器，而非动量本身。

## 实验关键数据

### SR数据验证

| 方法 | batch=full | batch=1 |
|------|-----------|---------|
| Adam | $\ell_\infty$ 间隔 | **$\ell_2$ 间隔** |
| SGD | $\ell_2$ 间隔 | $\ell_2$ 间隔 |
| Signum | $\ell_\infty$ 间隔 | $\ell_\infty$ 间隔 |

### 高斯数据验证

| 方法 | 与 $\ell_2$ 余弦 | 与 $\ell_\infty$ 余弦 |
|------|----------------|-------------------|
| Full-batch Adam | 低 | **1.0** |
| Inc-Adam | 高 | 低 |
| Adam (batch=1, replacement) | 高 | 低 |
| Adam (batch=1, reshuffling) | 高 | 低 |

### 关键发现
- Inc-Adam与有替换/reshuffling的batch=1 Adam行为一致→Inc-Adam是好的理论代理
- 对任何 $\beta_1 \leq \beta_2$，SR数据上结论成立→偏差变化不是特定超参的问题
- AdamProxy的Mahalanobis范数在某些数据上退化为 $\ell_2$、另一些退化为 $\ell_\infty$

## 亮点与洞察
- **反常识发现**：SGD的隐式偏差不依赖batch size→自然推测Adam也不依赖→但事实相反。这揭示了自适应方法的预条件器对采样方式敏感的本质。
- **预条件器的核心差异**：$\sum_i (\nabla \mathcal{L}_i)^2 \neq (\sum_i \nabla \mathcal{L}_i)^2$——单样本梯度的平方之和不等于全批梯度的平方。这个简单的数学事实导致了完全不同的隐式偏差。
- **Signum的鲁棒性**：sign操作使Signum对采样方式免疫——这可能部分解释了Signum/SignSGD在某些场景下的稳定性。

## 局限与展望
- AdamProxy分析需要假设方向收敛存在（Assumption 4.4）
- 仅分析了batch=1的极端情况，中间batch size的行为开放
- 仅限于线性分类+可分数据，深层网络的情况更复杂
- $\beta_2 \to 1$的极限可能与实践中 $\beta_2=0.999$ 有偏差

## 相关工作与启发
- **vs Zhang等人(2024)**: 他们证明full-batch Adam → $\ell_\infty$，本文证明mini-batch可以不同
- **vs Soudry等人(2018)**: GD → $\ell_2$ 不受batch size影响，但Adam受影响——自适应方法本质不同
- **启示**: 实践中Adam的行为可能同时受数据结构和batch size影响——不能简单从full-batch理论推断

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次发现并理论化Adam的batch-dependent隐式偏差
- 实验充分度: ⭐⭐⭐⭐ 结构化+随机数据验证，多种采样方案对比
- 写作质量: ⭐⭐⭐⭐ 数学严谨，直觉解释到位
- 价值: ⭐⭐⭐⭐⭐ 对理解Adam在实际训练中的行为有根本意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](../../ICML2026/optimization/the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[ICLR 2026\] Hyperbolic Aware Minimization: Implicit Bias for Sparsity](hyperbolic_aware_minimization_implicit_bias_for_sparsity.md)

</div>

<!-- RELATED:END -->
