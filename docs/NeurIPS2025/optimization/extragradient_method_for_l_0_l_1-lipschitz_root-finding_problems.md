---
title: >-
  [论文解读] Extragradient Method for $(L_0, L_1)$-Lipschitz Root-finding Problems
description: >-
  [NeurIPS 2025][优化/理论][Extragradient方法] 本文在 $\alpha$-对称 $(L_0,L_1)$-Lipschitz 条件下（放松经典 $L$-Lipschitz 假设）为 extragradient (EG) 方法提出自适应步长策略 $\gamma_k = 1/(c_0 + c_1\|F(x_k)\|^\alpha)$，建立了强单调（线性收敛）、单调（次线性收敛）和 weak Minty（局部收敛）三类根问题的首个完整收敛保证。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "Extragradient方法"
  - "$(L_0"
  - "L_1)$-Lipschitz"
  - "根问题"
  - "自适应步长"
  - "min-max优化"
---

# Extragradient Method for $(L_0, L_1)$-Lipschitz Root-finding Problems

**会议**: NeurIPS 2025  
**arXiv**: [2510.22421](https://arxiv.org/abs/2510.22421)  
**代码**: [github.com/isayantan/L0L1extragradient](https://github.com/isayantan/L0L1extragradient)  
**领域**: 优化理论 / 变分不等式  
**关键词**: Extragradient方法, $(L_0,L_1)$-Lipschitz, 根问题, 自适应步长, min-max优化

## 一句话总结
本文在 $\alpha$-对称 $(L_0,L_1)$-Lipschitz 条件下（放松经典 $L$-Lipschitz 假设）为 extragradient (EG) 方法提出自适应步长策略 $\gamma_k = 1/(c_0 + c_1\|F(x_k)\|^\alpha)$，建立了强单调（线性收敛）、单调（次线性收敛）和 weak Minty（局部收敛）三类根问题的首个完整收敛保证。

## 研究背景与动机

**领域现状**：Extragradient (EG) 方法是求解变分不等式和根问题 $F(x_*) = 0$ 的经典算法，广泛用于 min-max 优化（GAN 训练、分布式鲁棒优化、多智能体博弈）。现有收敛理论几乎全部建立在算子 $F$ 的 $L$-Lipschitz 假设 $\|F(x) - F(y)\| \leq L\|x-y\|$ 上。

**现有痛点**：(a) $L$-Lipschitz 假设对很多问题过于严格——例如 $F(x) = x^2$ 不满足任何有限 $L$ 的 Lipschitz 条件；(b) 张等人 (2020) 为最小化问题提出的 $(L_0, L_1)$-光滑假设 $\|\nabla^2 f(x)\| \leq L_0 + L_1\|\nabla f(x)\|$ 在 LSTM/Transformer 上得到实验验证，但尚未推广到 EG 方法和变分不等式设定。

**核心矛盾**：对于非 Lipschitz 算子（如立方 min-max 问题），Jacobian 范数 $\|\mathbf{J}(x)\|$ 随 $\|F(x)\|$ 增长，常数步长 EG 可能发散——需要自适应步长，但现有理论空白。

**本文目标** 在 $\alpha$-对称 $(L_0,L_1)$-Lipschitz 条件下为 EG 建立完整收敛理论（覆盖强单调、单调、weak Minty 三类问题），并设计配套的自适应步长。

**切入角度**：利用等价刻画 $\|\mathbf{J}(x)\| \leq L_0 + L_1\|F(x)\|^\alpha$（Theorem 2.1），设计与 $\|F(x_k)\|^\alpha$ 成反比的自适应步长。

**核心 idea**：步长与当前算子范数 $\|F(x_k)\|^\alpha$ 成反比，在远离解时自动缩小（保证安全），接近解时自动增大（加速收敛）。

## 方法详解

### 整体框架
EG 方法的两步迭代：$\hat{x}_k = x_k - \gamma_k F(x_k)$（extrapolation），$x_{k+1} = x_k - \omega_k F(\hat{x}_k)$（update）。本文核心贡献是为不同问题类设计 $(\gamma_k, \omega_k)$ 并证明收敛。

### 关键设计

1. **$\alpha$-对称 $(L_0,L_1)$-Lipschitz 条件 (Assumption 1.1)**:

    - 功能：放松经典 Lipschitz 假设
    - 核心定义：$\|F(x) - F(y)\| \leq (L_0 + L_1 \max_{\theta \in [0,1]} \|F(\theta x + (1-\theta)y)\|^\alpha)\|x-y\|$
    - 等价 Jacobian 刻画 (Theorem 2.1)：对 min-max 问题 $\|\mathbf{J}(x)\| \leq L_0 + L_1\|F(x)\|^\alpha$
    - 当 $L_1 = 0$ 退化为标准 $L$-Lipschitz；$\alpha = 1$ 对应张等人的 $(L_0, L_1)$-光滑条件
    - 意义：允许 Lipschitz"常数"随算子范数增长，更准确刻画实际问题结构

2. **消除路径最大值的关键引理 (Proposition 3.1)**:

    - 功能：将假设中的 $\max_\theta$ 消除，得到可操作的上界
    - 对 $\alpha = 1$：$\|F(x) - F(y)\| \leq (L_0 + L_1\|F(x)\|)\exp(L_1\|x-y\|) \cdot \|x-y\|$
    - 对 $\alpha \in (0,1)$：$\|F(x) - F(y)\| \leq (K_0 + K_1\|F(x)\|^\alpha + K_2\|x-y\|^{\alpha/(1-\alpha)}) \cdot \|x-y\|$
    - 其中 $K_0 = L_0(2^{\alpha^2/(1-\alpha)} + 1)$, $K_1 = L_1 \cdot 2^{\alpha^2/(1-\alpha)}$

3. **自适应步长策略**:

    - **统一形式**：$\gamma_k = \frac{1}{c_0 + c_1\|F(x_k)\|^\alpha}$
    - **强单调 $\alpha = 1$**：$\gamma_k = \omega_k = \frac{0.21}{L_0 + L_1\|F(x_k)\|}$（Theorem 3.2, 线性收敛）
    - **单调 $\alpha = 1$**：$\gamma_k = \omega_k = \frac{0.45}{L_0 + L_1\|F(x_k)\|}$（Theorem 3.5, 次线性）
    - **Weak Minty $\alpha = 1$**：$\gamma_k = \frac{0.56}{L_0 + L_1\|F(x_k)\|}$, $\omega_k = \gamma_k/2$（Theorem 3.8, 局部次线性）
    - 对 $\alpha \in (0,1)$，步长中 $c_0, c_1$ 依赖 $K_0, K_1, K_2$

4. **消除指数依赖的精细化分析**:

    - 问题：Theorem 3.2 的收敛率中出现 $\exp(L_1\|x_0 - x_*\|)$ 项
    - **Corollary 3.3**：证明经过 $K'$ 步后 $\|F(x_k)\| \leq L_0/L_1$，此后步长 $\geq \nu/2L_0$。总迭代量分为两项：
        - Term I：$\frac{2L_0}{\nu\mu}\log(1/\varepsilon)$（标准复杂度，无指数依赖）
        - Term II：与 $\varepsilon$ 无关的"warmup"阶段
    - 这是对 Vankov et al. (2024) 结果的本质改进——后者的收敛率指数依赖 $\|x_0 - x_*\|$

### 问题实例验证
- **Example 1**（逻辑回归）：$f(x) = \log(1 + \exp(-a^\top x))$，$L = \|a\|^2$ 但 $L_0 = 0, L_1 = \|a\|$ → 当 $\|a\| \gg 1$ 时 $(L_0, L_1)$ 界远紧于 $L$ 界
- **Example 2**：$F(x) = (u_1^2, u_2^2)$，不满足任何有限 $L$ 的 Lipschitz，但满足 $1/2$-对称 $(0,2)$-Lipschitz
- **Example 3**（双线性耦合）：$\frac{1}{p+1}\|w_1\|^{p+1} + w_1^\top \mathbf{B} w_2 - \frac{1}{p+1}\|w_2\|^{p+1}$，对任意 $p > 1$ 满足 1-对称 $(L_0, L_1)$-Lipschitz

## 实验关键数据

### 强单调问题：自适应步长 vs Vankov et al.

| 方法 | 初始步长 | 最终步长 | 收敛速度 |
|------|---------|---------|---------|
| Vankov et al. (2024) | ~0.02 | ~0.02（恒定） | 较慢 |
| **本文** | ~0.02 | >0.032（自增）| **更快** |

关键观察：本文步长随 $\|F(x_k)\|$ 减小而**自动增大**，而 Vankov 的步长保持恒定。

### 单调问题 (式 (20)，凸凹 min-max)

| 步长策略 | $c$ 或 $(c_0, c_1)$ | 收敛性 | 备注 |
|---------|-------------------|--------|------|
| 常数 $\gamma = 1/c$ | $c = 10^5$（最优网格搜索）| 较慢 | $c < 10^5$ 发散 |
| 自适应 (本文) | $(c_0, c_1) = (10, 10)$（最优）| **最快** | 9/9 组合均优于常数 |

### Weak Minty (GlobalForsaken 问题)

| 算法 | 步长 | 收敛到 $(0,0)$ 速度 |
|------|------|-------------------|
| AdaptiveEG+ | $\gamma = 0.1$（固定） | 中等 |
| EG+ | $\gamma = 0.1$（固定） | 中等 |
| **本文 EG** | $(c_0, c_1) = (1, 1)$ | **显著最快** |

### 关键发现
- 自适应步长的核心优势：**自动从小到大**——初始远离解时步长小（安全），接近解时步长大（加速），无需手动调整
- 在单调问题上，9 组自适应参数中的**绝大多数**均优于最优常数步长——对超参数不敏感
- $\alpha < 1$ 的情况（如 $\alpha = 1/2$）允许覆盖 $L$-Lipschitz 完全失效的算子类

## 亮点与洞察
- **统一的步长范式**：$\gamma_k = 1/(c_0 + c_1\|F(x_k)\|^\alpha)$ 一个公式覆盖三类问题（强单调/单调/weak Minty），仅系数不同——简洁优雅
- **消除指数依赖的技巧**：Corollary 3.3 的两阶段分析思路（先证步长增长到安全水平，再用固定步长分析）可推广到其他自适应方法
- **Theorem 2.1 的实用性**：将抽象的路径最大值条件等价为 Jacobian 范数条件——后者在实际中可直接验证（如图散点图），极大提升了假设的可操作性

## 局限与展望
- **仅确定性分析**：未涉及随机 EG（SVRE-EG 等），实际 min-max 训练通常使用随机梯度
- **Weak Minty 局部收敛**：需要初始点足够接近解（条件 (17)：$\Delta_1 > 0$），无全局保证
- **单调问题的指数依赖**：Theorem 3.5 中 $\exp(L_1\|x_0 - x_*\|)$ 项虽在 Theorem 3.6 中消除，但后者需 $K+1 \geq 2L_1^2\|x_0 - x_*\|^2/\nu^2$ 的 warmup
- **$\alpha$ 的选择**：理论覆盖 $\alpha \in (0,1]$，但实际问题的最优 $\alpha$ 如何确定未讨论
- **实验规模小**：仅低维合成问题（$d \leq 2$），缺乏大规模 GAN/DRO 训练的验证

## 相关工作与启发
- **vs Vankov et al. (2024)**: 仅对强单调 $\alpha = 1$ 分析 EG，步长方案用 $\min$ 取多个条件（更保守）。本文步长更简洁、理论更紧、覆盖面更广
- **vs Gorbunov et al. (2025)**: 在 $(L_0, L_1)$-光滑下分析自适应梯度下降（最小化问题），本文是首次将此框架推广到 EG 和根问题
- **vs Zhang et al. (2020)**: 原始 $(L_0, L_1)$-光滑定义者，关注最小化设定中裁剪 GD 的收敛。本文推广到变分不等式
- **vs Diakonikolas et al. (2021)**: weak Minty 算子下 EG 的收敛分析，但限于 $L$-Lipschitz。本文放松为 $(L_0, L_1)$-Lipschitz

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在 $(L_0,L_1)$-Lipschitz 下建立 EG 的完整收敛理论，覆盖三类问题
- 实验充分度: ⭐⭐⭐ 合成实验清晰验证理论，但缺乏大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ 表格汇总（Table 1）极其清晰，理论展开层次分明
- 价值: ⭐⭐⭐⭐ 填补了 EG 在广义 Lipschitz 条件下的理论空白，步长策略有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning](escaping_saddle_points_without_lipschitz_smoothness_the_power_of_nonlinear_preco.md)
- [\[NeurIPS 2025\] Revisiting Orbital Minimization Method for Neural Operator Decomposition](revisiting_orbital_minimization_method_for_neural_operator_decomposition.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[NeurIPS 2025\] Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models](fantastic_features_and_where_to_find_them_a_probing_method_to_combine_features_f.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)

</div>

<!-- RELATED:END -->
