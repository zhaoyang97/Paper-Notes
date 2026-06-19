---
title: >-
  [论文解读] Incremental Gradient Descent with Small Epoch Counts is Surprisingly Slow on Ill-Conditioned Problems
description: >-
  [ICML 2025][优化/理论][incremental gradient descent] 系统研究了增量梯度下降（IGD）在小 epoch 体制（$K \lesssim \kappa$）下的收敛行为，证明 IGD 在此体制下比有放回 SGD 慢至少 $n$ 倍，且当分量函数非凸时收敛速度急剧恶化至指数级慢。
tags:
  - "ICML 2025"
  - "优化/理论"
  - "incremental gradient descent"
  - "permutation-based SGD"
  - "convergence lower bound"
  - "small epoch regime"
  - "condition number"
---

# Incremental Gradient Descent with Small Epoch Counts is Surprisingly Slow on Ill-Conditioned Problems

---

**会议**: ICML 2025  
**arXiv**: [2506.04126](https://arxiv.org/abs/2506.04126)  
**代码**: 无  
**领域**: 优化  
**关键词**: incremental gradient descent, permutation-based SGD, convergence lower bound, small epoch regime, condition number

## 一句话总结

系统研究了增量梯度下降（IGD）在小 epoch 体制（$K \lesssim \kappa$）下的收敛行为，证明 IGD 在此体制下比有放回 SGD 慢至少 $n$ 倍，且当分量函数非凸时收敛速度急剧恶化至指数级慢。

---

## 研究背景与动机

**领域现状**：有限和优化问题 $\min_x F(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$ 是机器学习的核心。基于排列的 SGD（如 Random Reshuffling）因实际表现优于有放回 SGD 而被广泛使用。近年来理论研究已证明 RR 在光滑强凸目标下达到 $\mathcal{O}(1/nK^2)$ 的收敛速率，优于有放回 SGD 的 $\mathcal{O}(1/nK)$。

**现有痛点**：上述收敛分析几乎全部限定在"大 epoch 体制"——即 epoch 数 $K$ 远大于条件数 $\kappa = L/\mu$。但在实际深度学习训练中，神经网络的优化景观条件数极大，而由于计算资源限制，epoch 数 $K$ 通常远小于 $\kappa$。Safran & Shamir (2021) 的下界表明，在小 epoch 体制下 RR 甚至无法超越有放回 SGD，但 IGD 在此体制的精确行为仍是开放问题。

**核心矛盾**：现有分析需要步长 $\eta \leq \mathcal{O}(1/nL)$ 来控制单 epoch 内的累积误差，但当 $K$ 小时步长需要设为至少 $\Omega(1/K)$ 才能让迭代点接近最优解——这两个要求在 $K \ll \kappa$ 时互相矛盾，导致现有上界不紧。

**本文目标** 作为理解小 epoch 体制下排列式 SGD 收敛行为的第一步，全面刻画 IGD（最简单的确定性排列 SGD）在此体制下的收敛速率下界和上界。

**切入角度**：IGD 是确定性的（固定排列 $\sigma_k =$ identity），技术上比分析随机排列更可控，可以通过精心构造目标函数来获得紧的下界。

**核心 idea**：通过三类精巧的函数构造（相同 Hessian、强凸分量、非凸分量），揭示 IGD 在小 epoch 体制下呈现出从 $\Omega(G^2/\mu K)$ 到指数级 $\Omega((1+\kappa/2nK)^n)$ 的逐级恶化收敛行为。

## 方法详解

### 整体框架

论文围绕有限和问题 $F(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$，其中 $F$ 是 $\mu$-强凸的，各 $f_i$ 是 $L$-光滑的。考虑排列式 SGD 的最简形式 IGD（每个 epoch 按固定顺序 1 到 $n$ 依次更新），研究其在 $K \leq \kappa$ 时的收敛速率。分析分为三个层次：（1）分量共享 Hessian；（2）分量各自强凸；（3）允许非凸分量。同时给出大 epoch 体制的紧界作为参照。

### 关键设计

1. **相同 Hessian 下界 (Theorem 3.1)**:

    - 功能：建立最友好情况下的基本下界
    - 核心思路：构造 3 维目标函数，其中所有分量共享相同的 Hessian（$\nabla^2 f_i = \nabla^2 F$），证明对任意常数步长，IGD 的最终迭代满足 $F(x_n^K) - F(x^*) \gtrsim G^2/(\mu K)$。这比有放回 SGD 的上界 $G^2/(\mu nK)$ 慢了 $n$ 倍
    - 设计动机：即使在分量函数"最相似"的情况下（相同曲率），IGD 仍然慢于有放回 SGD，说明确定性排列本身在小 epoch 体制下就是不利的。对应的上界 (Theorem 3.2) 在 1 维情况下匹配，证明此下界本质上是紧的

2. **强凸分量下界 (Theorem 3.3)**:

    - 功能：揭示去掉相同 Hessian 假设后收敛速率的急剧恶化
    - 核心思路：构造 4 维目标函数，通过旋转变换使各分量的极小值点构成正 $n$ 角形。精心选择初始化点使迭代序列保持旋转对称性，同样形成正 $n$ 角形并与全局最优点保持恒定距离。证明下界为 $\Omega(LG^2/\mu^2 \cdot \min\{1, \kappa^2/K^4\})$
    - 设计动机：相比 Theorem 3.1，此下界在整个小 epoch 体制内始终更大（大 $\kappa K$ 或 $\kappa^3/K^3$ 倍），且与 Theorem 3.2 的上界在 $K = \Theta(\sqrt{\kappa})$ 时匹配。这揭示了分量 Hessian 异质性对 IGD 收敛的严重影响

3. **非凸分量下界 (Theorem 3.5)**:

    - 功能：展示非凸分量导致的灾难性减速
    - 核心思路：构造 2 维函数，允许部分分量在某些方向上凹。证明下界包含 $(1 + L/(2\mu nK))^n$ 项，近似为 $\exp(\kappa/(2K))$——当 $K \ll \kappa$ 时这是指数级大的
    - 设计动机：这与强凸分量情况形成鲜明对比。有趣的是实验表明 RR 在相同构造上表现鲁棒，暗示随机排列对非凸分量有天然的缓解作用。这是文献中首次展示排列式 SGD 在小 epoch 体制下可能遭遇指数级慢收敛

### 附加结果

**最优排列的存在性 (Theorem 3.7)**：利用 Herding 算法，证明存在某种排列使得排列式 SGD 在小 epoch 体制下也能超越有放回 SGD，达到 $\mathcal{O}(H^2 L^2 G_*^2 / (\mu^3 n^2 K^2))$ 的收敛速率。但此排列依赖最优点处的梯度信息，不具实用性。

**大 epoch 体制紧界 (Section 4)**：建立了 $K \gtrsim \kappa$ 时凸和非凸分量情况下的匹配上下界。非凸分量情况下大 epoch 体制的速率差距仅为 $\kappa$ 倍，与小 epoch 体制的指数级差距形成鲜明对比。

## 实验关键数据

### 主要收敛速率对比

| 体制 | 分量假设 | 下界 (IGD) | 上界 (任意排列 SGD) | 有放回 SGD |
|------|---------|-----------|-------------------|-----------|
| 小 epoch, 相同 Hessian | 强凸 | $\Omega(G^2/\mu K)$ | $\mathcal{O}(G_*^2/\mu K)$ | $\mathcal{O}(G^2/\mu nK)$ |
| 小 epoch | 强凸 | $\Omega(LG^2/\mu^2 \cdot \min\{1,\kappa^2/K^4\})$ | $\mathcal{O}(L^2G_*^2/\mu^3 K^2)$ | $\mathcal{O}(G^2/\mu nK)$ |
| 小 epoch | 含非凸 | $\Omega(G^2/L \cdot (1+L/2\mu nK)^n)$ | — | $\mathcal{O}(G^2/\mu nK)$ |
| 大 epoch | 凸 | $\Omega(LG^2/\mu^2 K^2)$ | $\mathcal{O}(LG_*^2/\mu^2 K^2)$ | — |
| 大 epoch | 含非凸 | $\Omega(L^2G^2/\mu^3 K^2)$ | $\mathcal{O}(L^2G^2/\mu^3 K^2)$ | — |

### 关键速率比较

| 比较 | 差距因子 |
|------|---------|
| IGD vs 有放回 SGD（小 epoch, 相同 Hessian） | $n$ 倍 |
| IGD（强凸分量 vs 相同 Hessian） | $\kappa K$ 或 $\kappa^3/K^3$ 倍 |
| IGD（非凸 vs 强凸分量，小 epoch） | 指数级 vs 多项式级 |
| IGD（非凸 vs 凸分量，大 epoch） | 仅 $\kappa$ 倍 |

### 关键发现

- IGD 在小 epoch 体制下比有放回 SGD 至少慢 $n$ 倍，即使在最友好的假设下
- 去掉分量 Hessian 相同假设后，IGD 的收敛进一步恶化，下界中出现 $\kappa$ 的高次幂
- 允许非凸分量时，小 epoch 体制的恶化是指数级的，而大 epoch 体制仅是多项式级的——小 epoch 和大 epoch 体制的行为存在质的差异
- 存在（不实用的）最优排列能在小 epoch 体制超越有放回 SGD，说明排列选择至关重要

## 亮点与洞察

- 论文的核心贡献在于首次系统刻画了小 epoch 体制的收敛景观，揭示了其与大 epoch 体制的本质不同——这对理解实际深度学习训练有重要指导意义
- Theorem 3.3 的旋转多边形构造非常精巧，将几何直觉与收敛分析完美结合——迭代序列保持正 $n$ 角形对称性的想法令人印象深刻
- Theorem 3.5 的指数级下界出人意料，表明非凸分量在小 epoch 下可能导致灾难性的慢收敛
- 同时在大 epoch 体制给出紧界，与小 epoch 结果对照形成了完整画面——非凸分量在大 epoch 体制仅导致 $\kappa$ 倍减速而在小 epoch 体制导致指数级减速
- 实验验证（附录 G）确认了理论构造的迭代行为，增强了结果的可信度
- 论文写作结构清晰，Table 1 和 Figure 1 的概览极具参考价值，使复杂的理论结果一目了然

## 局限与展望

- 下界仅针对 IGD（固定排列），尚未推广到 RR 或其它随机排列策略——这是最重要的开放问题
- Theorem 3.2 的上界限于 1 维，高维推广是技术挑战（虽然可直接应用于对角 Hessian 目标）
- 所有结果假设常数步长，变步长设置（如 cosine schedule）下的行为未知——实际训练中步长几乎总是变化的
- 未涉及非凸目标函数（即 $F$ 本身非凸的情况），只考虑了强凸 $F$ 但某些 $f_i$ 非凸
- 离实际深度学习场景（随机化排列、自适应步长、大规模非凸问题）仍有距离
- Theorem 3.7 证明的最优排列需要知道最优解处的梯度信息，不具实用性——能否设计实用的接近最优排列策略是重要方向
- 分量函数之间的异质性度量（如 Hessian 差异程度）与收敛速率的定量关系尚未完全揭示

## 相关工作与启发

- Safran & Shamir (2021) 的 RR 小 epoch 下界 $\Omega(G^2/\mu nK)$ 是本文的重要基准，本文将 IGD 的下界从 $G^2/\mu K^2$ 改进到 $G^2/\mu K$
- GraB (Lu et al., 2022) 的梯度平衡排列策略在大 epoch 体制最优，但其小 epoch 行为未知——这是有趣的开放问题
- Mishchenko et al. (2020) 的上界是直接对照，本文在多个设置下揭示了其不紧性
- 本文结果暗示：在条件数大、训练 epoch 少的实际场景中（如大模型训练初期），排列策略的选择可能比之前认为的更加重要
- Herding 算法从组合优化借用到优化理论，展示了跨领域工具在分析排列策略中的价值
- 对实践的启示：在高条件数问题上，可能需要针对小 epoch 体制设计专门的排列策略或学习率调度

## 评分

⭐⭐⭐⭐ 理论贡献扎实，三层递进的下界构造（相同 Hessian → 强凸 → 非凸分量）层次分明，每一层都揭示了新的数学现象。特别是非凸分量下从多项式级到指数级的跳变非常出人意料。论文结构清晰，总结表格一目了然。但作为纯理论工作，对实际算法设计的直接指导有限，且主要结果限于 IGD 而非更广泛使用的 RR，留下了大量开放问题。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](quantum_optimization_via_gradient-based_hamiltonian_descent.md)
- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[ICLR 2026\] Conditioned Initialization for Attention](../../ICLR2026/optimization/conditioned_initialization_for_attention.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
