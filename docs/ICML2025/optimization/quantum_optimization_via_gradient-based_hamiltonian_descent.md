---
title: >-
  [论文解读] Quantum Optimization via Gradient-Based Hamiltonian Descent
description: >-
  [ICML2025][优化/理论][量子哈密顿下降] 将梯度信息融入量子哈密顿下降 (QHD) 框架，提出 gradient-based QHD，在凸和非凸优化中均实现了比原始 QHD 及经典方法（NAG、SGDM）快至少一个数量级的收敛速度和更高的全局最优命中概率。 - 经典优化瓶颈：一阶梯度方法（GD、NAG）在高度非凸…
tags:
  - "ICML2025"
  - "优化/理论"
  - "量子哈密顿下降"
  - "梯度信息"
  - "Lyapunov 收敛分析"
  - "Schrödinger 方程"
  - "非凸优化"
  - "量子算法"
---

# Quantum Optimization via Gradient-Based Hamiltonian Descent

**会议**: ICML2025  
**arXiv**: [2505.14670](https://arxiv.org/abs/2505.14670)  
**代码**: [jiaqileng/Gradient-Based-QHD](https://github.com/jiaqileng/Gradient-Based-QHD)  
**领域**: 量子优化 / optimization  
**关键词**: 量子哈密顿下降, 梯度信息, Lyapunov 收敛分析, Schrödinger 方程, 非凸优化, 量子算法

## 一句话总结

将梯度信息融入量子哈密顿下降 (QHD) 框架，提出 gradient-based QHD，在凸和非凸优化中均实现了比原始 QHD 及经典方法（NAG、SGDM）快至少一个数量级的收敛速度和更高的全局最优命中概率。

## 研究背景与动机

- **经典优化瓶颈**：一阶梯度方法（GD、NAG）在高度非凸景观中容易陷入局部极值或鞍点，无法保证全局收敛。
- **量子哈密顿下降 (QHD)**：Leng et al. 提出将 Nesterov 加速梯度法对应的 Bregman Lagrangian 经正则量子化后，利用 Schrödinger 方程演化量子波函数，借助量子隧穿效应逃逸局部极值。但原始 QHD 仅使用函数值（零阶信息），收敛速率偏慢。
- **高分辨率 ODE 的启示**：Shi et al. 的高分辨率 ODE 框架揭示了 NAG 加速机制来源于 Lyapunov 函数中梯度项的耦合。本文受此启发，将梯度信息显式嵌入量子哈密顿量，构建更强的量子优化动力学。

## 方法详解

### 1. 含梯度的经典 Lagrangian

在标准 Bregman Lagrangian 基础上增加梯度耦合项：

$$
\mathcal{L}(t,X,\dot{X}) = \frac{t^3}{2}|\dot{X}|^2 - \alpha t^3 \dot{X}^\top \nabla f(X) - \frac{\beta t^3}{2}|\nabla f(X)|^2 - (t^3 + \gamma t^2)f(X)
$$

其中 $\alpha, \beta, \gamma$ 为可调参数。当 $\alpha = \beta = \gamma = 0$ 时退化为原始 QHD。

### 2. Legendre 变换得 Hamiltonian

$$
H(t,X,P) = \frac{1}{2}\|t^{-3/2}P + \alpha t^{3/2}\nabla f\|^2 + \frac{\beta t^3}{2}\|\nabla f\|^2 + (t^3 + \gamma t^2)f(X)
$$

### 3. 正则量子化 → Gradient-Based QHD

将经典动量 $P \mapsto \hat{p} = -i\nabla$ 代入，得到含梯度的量子哈密顿算符：

$$
\hat{H}(t) = \frac{1}{2}\sum_{j=1}^d A_j^2 + \frac{\beta}{2}t^3\|\nabla f\|^2 + (t^3 + \gamma t^2)f
$$

其中 $A_j = t^{-3/2}\hat{p}_j + \alpha t^{3/2}\hat{v}_j$，$\hat{v}_j$ 是梯度分量的乘法算符。演化由含时 Schrödinger 方程驱动：

$$
i\partial_t \Psi(t,x) = \hat{H}(t)\Psi(t,x)
$$

### 4. 离散化量子算法 (Algorithm 1)

将 $\hat{H}(t_k)$ 分解为三项：动能 $H_{k,1} = -\frac{1}{2t_k^3}\Delta$、梯度耦合 $H_{k,2} = \frac{\alpha}{2}\{-i\nabla, \nabla f\}$、势能 $H_{k,3}$，然后用乘积公式（算符分裂）逐步演化：

$$
e^{-ih\hat{H}(t_k)} \approx e^{-ihH_{k,1}} e^{-ihH_{k,2}} e^{-ihH_{k,3}}
$$

最终测量量子态的位置观测量，输出候选解 $\xi \in \mathbb{R}^d$。

### 5. 收敛性理论

**定理 1（函数值收敛，凸情形）**：当 $\beta=0$，$\gamma \geq \max(3\alpha, 0)$ 时，

$$
\mathbb{E}[f(X_t)] \leq \frac{\mathscr{K}_0 + \mathscr{D}_0}{t^2 + \omega t} = O(t^{-2})
$$

其中 $\omega = \gamma - 3\alpha$，证明依赖于构造量子算符 Lyapunov 函数 $\mathcal{E}(t) = \langle \hat{O}(t)\rangle_t$ 并证其单调递减。

**定理 4（梯度范数收敛）**：当 $\beta > 0$ 且梯度范数函数 $G(x) = \|\nabla f\|^2$ 满足凸性条件时，

$$
\mathbb{E}[\|\nabla f(X_t)\|^2] \leq \frac{2(\mathscr{K}_0 + \mathscr{D}_0')}{\beta t^2} = O(t^{-2})
$$

### 6. 门复杂度

**定理 6**：$K$ 步迭代需 $O(K)$ 次零阶量子 oracle 查询，$\widetilde{O}(\alpha d h K L)$ 次一阶量子 oracle 查询，每步复杂度关于问题维度 $d$ 线性增长，算法可扩展至大规模问题。

## 实验关键数据

| 测试函数 | 类型 | 景观特点 | Grad-QHD vs QHD | Grad-QHD vs NAG/SGDM |
|---|---|---|---|---|
| 凸函数 $(x+y)^4/256+(x-y)^4/128$ | 凸，非强凸 | 全局极值处 Hessian 奇异 | 函数值/梯度范数收敛显著更快 | 大幅领先 |
| Styblinski-Tang | 非凸 | 3个局部极值+1个全局极值 | 成功概率显著更高，解分布更集中 | 高出一个数量级 |
| Michalewicz | 非凸 | 平坦台地+狭窄峡谷中隐藏全局极值 | 函数值和成功概率均优 | 明显优势 |
| Cube-Wave | 非凸 | 10+个局部极值，4个全局极值 | 终端函数值低约1个数量级 | 终端函数值低约2个数量级 |
| Rastrigin | 非凸 | 强振荡景观 | 收敛率和成功概率均优 | 大幅领先 |

**实验设置**：步长 $h=0.2$，经典方法用 1000 次独立运行估计期望，量子方法通过波函数数值积分直接计算。参数 $\alpha=-0.1$（凸）/ $\alpha=-0.05$（非凸），$\beta=0$，$\gamma=5$。实验在 MacBook M4 上完成。

## 亮点与洞察

1. **物理直觉+优化的优美结合**：将高分辨率 ODE 框架中梯度耦合加速机制移植到量子力学，从 Lagrangian → Hamiltonian → 正则量子化形成完整理论管线。
2. **非平凡的 Lyapunov 分析**：在量子力学（非交换算符）框架下构造 Lyapunov 函数并证明单调性，需要处理复杂的对易关系（Lemma 3），技术难度远超经典情形。
3. **参数灵活性**：$\alpha, \beta, \gamma$ 不限于高分辨率 ODE 对应值，提供更广泛的动力学探索空间。
4. **渐近可比性**：每步门复杂度 $\widetilde{O}(d)$，与经典 NAG 的 $O(d)$ 渐近可比，不引入指数级量子开销。
5. **全局优化能力**：利用量子隧穿 + 梯度引导，在多个非凸 benchmark 上成功概率远超经典方法。

## 局限与展望

1. **数值实验仅限 2D**：由于量子态模拟的指数开销，所有实验均在二维进行，高维优势尚未验证。
2. **离散化收敛理论缺失**：连续时间收敛已证，但离散时间算法（Algorithm 1）的严格收敛分析留待未来。
3. **步长选择缺乏理论指导**：理论分析建议 $h \sim t_k^{-3/2}$，但实验中固定步长已经 work，gap 未弥合。
4. **需要一阶量子 oracle**：与原始 QHD（零阶）相比，需要额外的梯度 oracle，构建成本取决于问题。
5. **实际量子硬件未测试**：算法假设容错量子计算机，当前 NISQ 设备上的可行性未探讨。
6. **凸性假设**：理论收敛保证需要 $f$ 凸或 $G(x) = \|\nabla f\|^2$ 凸，非凸情形的理论结果仍为开放问题。

## 相关工作与启发

- **QHD 原始工作** [Leng et al.]：本文直接扩展的基础。
- **高分辨率 ODE** [Shi et al.]：NAG 加速机制的连续化理解，Lyapunov 函数设计的灵感来源。
- **Bregman Lagrangian** [Wibisono et al.]：统一加速方法的变分视角。
- **量子模拟** [Childs et al., Berry et al.]：含时哈密顿量模拟的算法基础。
- **QSVT** [Gilyén et al.]：实现梯度耦合项 $H_{k,2}$ 的关键量子子程序。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将梯度信息融入量子哈密顿框架，理论构造新颖优美
- 实验充分度: ⭐⭐⭐ — benchmark 多样但仅限 2D，缺乏高维和实际问题验证
- 写作质量: ⭐⭐⭐⭐⭐ — 从物理动机到数学推导再到算法实现，逻辑清晰完整
- 价值: ⭐⭐⭐⭐ — 为量子优化算法设计提供了有力的新方向，但实用性受限于量子硬件

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICML 2025\] Incremental Gradient Descent with Small Epoch Counts is Surprisingly Slow on Ill-Conditioned Problems](incremental_gradient_descent_with_small_epoch_counts_is_surprisingly_slow_on_ill.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](../../NeurIPS2025/optimization/optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)

</div>

<!-- RELATED:END -->
