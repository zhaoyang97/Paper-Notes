---
title: >-
  [论文解读] Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations
description: >-
  [ICML 2025][物理/科学计算][符号回归] 本文提出 SymPDE 框架，利用深度强化学习直接搜索 PDE 的闭式符号解，绕过了 PINNs 数值解精度不足和可解释性差的问题，在 Poisson 方程和热方程上达到 90% 的恢复率。 领域现状： 偏微分方程 (PDE) 广泛存在于物理、数学等领域，其求解是核心问题…
tags:
  - "ICML 2025"
  - "物理/科学计算"
  - "符号回归"
  - "偏微分方程"
  - "深度强化学习"
  - "闭式解"
  - "PINNs"
---

# Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations

**会议**: ICML 2025  
**arXiv**: [2405.14620](https://arxiv.org/abs/2405.14620)  
**代码**: 无  
**领域**: Reinforcement Learning / Scientific Computing  
**关键词**: 符号回归, 偏微分方程, 深度强化学习, 闭式解, PINNs

## 一句话总结
本文提出 SymPDE 框架，利用深度强化学习直接搜索 PDE 的闭式符号解，绕过了 PINNs 数值解精度不足和可解释性差的问题，在 Poisson 方程和热方程上达到 90% 的恢复率。

## 研究背景与动机
**领域现状**: 偏微分方程 (PDE) 广泛存在于物理、数学等领域，其求解是核心问题。传统解析方法 (如 Green 函数) 对非线性 PDE 几乎不可行；PINNs 虽然利用深度学习给出数值解，但本质上是连续函数空间中的近似。

**现有痛点**: (a) PINNs 拟合高频和陡变函数时精度不够，数值解在振荡或畸变区域偏差大；(b) 神经算子方法 (DeepONet, FNO) 需要大量标注数据；(c) 所有神经网络方案得到的都是数值解，缺乏可解释性，外推能力差。

**核心矛盾**: 数值解 vs 符号解——符号解天然具备精确性、可解释性和外推能力，但搜索空间巨大、优化困难。

**已有尝试的不足**: 先用 PINN 求数值解、再用符号回归拟合符号表达式 (DSR* 范式) 的两步法，因数值解本身的近似误差会误导符号回归。

**本文切入**: 跳过数值解中间步骤，用强化学习直接在符号空间搜索满足 PDE 定义的闭式解。

**核心 idea**: 用 RNN 生成表达式骨架 (skeleton)，通过 BFGS 优化常数使其满足 PDE 约束，以满足程度为奖励训练 RNN (risk-seeking policy gradient)。

## 方法详解

### 整体框架
输入：PDE 定义 (方程形式 + 边界/初始条件 + 计算域) → RNN 自回归生成符号表达式树 → 提取骨架中的常数用 BFGS 优化 → 计算 MSE 作为奖励 → 更新 RNN 策略 → 直到奖励 > 0.9999 → 输出闭式解。

### 关键设计

1. **多系统 PDE 建模**:

    - 时间无关系统：$\mathcal{F}[\mathbf{x}, u(\mathbf{x}), \nabla u, ..., \nabla^k u] = 0$，损失为 $\mathcal{L}_s = \text{MSE}_\mathcal{F} + \text{MSE}_\mathcal{B}$
    - 时空连续模型：$u_t = \mathcal{N}(\mathbf{x},t,\nabla u,...,\nabla^k u)$，增加初始条件损失 $\text{MSE}_\mathcal{I}$
    - 时空离散模型：将时间参数化，每个时间步的解共享同一骨架但参数不同 $\hat{u}(\mathbf{x}; \vec{\alpha}_t)$，用参数神经网络 (PNN) 学习 $t \to \vec{\alpha}_t$ 的映射
    - 设计动机：离散时间模型降低了时空耦合表达式的搜索空间复杂度

2. **基于 RL 的表达式生成与优化**:

    - 用 RNN 自回归生成表达式树的前序遍历序列
    - 每个 token 的选择概率由 softmax 输出，同时提供父节点和兄弟节点信息以强化结构理解
    - 每生成一个完整表达式即为一个 episode；对骨架中的常数用 BFGS/Adam 优化 → 计算奖励
    - 设计动机：将 PDE 求解转化为 MDP，利用 RL 自动化搜索

3. **Risk-Seeking Policy Gradient**:

    - 奖励函数：$\mathcal{R}_s(\tau) = \frac{1}{1 + \sqrt{\text{MSE}_\mathcal{F} + \text{MSE}_\mathcal{B}}}$
    - 不优化平均表现，而是最大化最佳情况表现：$J_{\text{risk}}(\theta; \epsilon) \approx \mathbb{E}[\mathcal{R}(\tau) | \mathcal{R}(\tau) \geq \mathcal{R}_\epsilon]$
    - 仅用奖励超过 $(1-\epsilon)$-分位数的样本更新策略
    - 加入熵正则化促进探索
    - 设计动机：找到 PDE 的精确解而非平均好的近似解

### 损失函数 / 训练策略
- 常数优化：BFGS 最小化 $\mathcal{L}_s$ 或 $\mathcal{L}_{s\text{-}t}$
- 策略优化：risk-seeking policy gradient + 熵正则
- 终止条件：$\mathcal{R} > 0.9999$ 或达最大 episode 数

## 实验关键数据

### 主实验

| 数据集/基准 | 指标 | SymPDE | DSR* | 提升 |
|-------------|------|--------|------|------|
| Nguyen 12 基准平均 | $\bar{\mathcal{R}}_s$ | 0.9746 | 0.9144 | +6.6% |
| Nguyen 12 基准平均 | 恢复率 $P_{\text{Re}}$ | 90.0% | 33.3% | +56.7% |
| 周期势场 (Eq.10) | $R^2$ | 1.00 | 0.00 | — |
| 点电荷 (Eq.11) | $R^2$ | 1.00 | 0.00 | — |
| 连续时间热方程 (Eq.12) | 表达式正确性 | ✓ | ✗ | — |
| 离散时间热方程 (Eq.13) | $\mathcal{L}_2$ 相对误差 | 9.84×10⁻⁴ | — | — |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SymPDE (连续时间) | 正确骨架 $x^2 e^{-t}$ | 端到端发现正确的时空耦合表达式 |
| SymPDE (离散时间) | 骨架 $c_0 e^{c_1 x^2}$ | 成功解耦时间与空间，PNN 拟合参数 |
| DSR* (PINN+DSR) | 错误表达式 | PINN 微小偏差被符号回归放大 |
| 纯 PINN | $\mathcal{L}_2=0.0283$ | 数值精度尚可但不可解释 |

### 关键发现
- SymPDE 在 12 个 Nguyen 基准中 10 个实现 100% 恢复率，DSR* 仅 4 个
- 对高频振荡 (sin(5x)) 和陡变函数 (1/r)，SymPDE 完美恢复而 DSR* 完全失败
- PINN 的微小数值偏差通过符号回归会被显著放大，验证了端到端方案的必要性

## 亮点与洞察
- **范式创新**：首次提出直接用 RL 搜索 PDE 闭式解的范式，绕开了数值解中间步骤
- **离散时间模型**：受 FDTD 启发，用参数化表达式骨架处理时空 PDE，巧妙降维
- **实际意义**：闭式解可精确外推到训练域之外，这是所有数值方法做不到的

## 局限与展望
- 仅在 Poisson 方程和热方程上验证，未涉及更复杂的非线性 PDE (如 Navier-Stokes)
- 搜索空间随变量数和算子种类指数增长，可扩展性有待验证
- 需要预先指定允许的运算符集合，这需要领域先验知识

## 相关工作与启发
- 与 Petersen et al. 2020 的 DSR 算法共享 RL 框架，本文将其从数据拟合扩展到 PDE 求解
- PINNs 系列 (Raissi 2019, DeepONet, FNO) 提供数值解基线
- X-Net (Li 2024) 等端到端符号回归方法可作为骨架生成器的替代方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 RL 直接用于 PDE 闭式符号解搜索，范式新颖
- 实验充分度: ⭐⭐⭐ 基准较为基础，仅涉及线性 PDE，缺乏挑战性更高的测试
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法阐述系统
- 价值: ⭐⭐⭐⭐ 为 AI for Science 中的符号计算开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Towards a Foundation Model for Partial Differential Equations Across Physics Domains](../../AAAI2026/physics/towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)
- [\[ICML 2025\] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[ICML 2025\] OmniArch: Building Foundation Model For Scientific Computing](omniarch_building_foundation_model_for_scientific_computing.md)
- [\[ICML 2026\] Foundation Inference Models for Ordinary Differential Equations](../../ICML2026/physics/foundation_inference_models_for_ordinary_differential_equations.md)

</div>

<!-- RELATED:END -->
