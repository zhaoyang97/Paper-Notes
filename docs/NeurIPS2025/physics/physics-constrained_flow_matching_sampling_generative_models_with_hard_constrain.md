---
title: >-
  [论文解读] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints
description: >-
  [NeurIPS 2025][物理/科学计算][流匹配] 提出 Physics-Constrained Flow Matching (PCFM)，一种零样本推理框架，通过在预训练流匹配模型的采样过程中交替执行前向投射、OT 插值反向更新和松弛惩罚校正，实现任意非线性等式约束的精确满足（达到机器精度），在含激波和间断的 PDE 问题上相比基线方法提升高达 99.5%。
tags:
  - "NeurIPS 2025"
  - "物理/科学计算"
  - "流匹配"
  - "物理约束"
  - "硬约束满足"
  - "偏微分方程"
  - "零样本推理"
---

# Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints

**会议**: NeurIPS 2025  
**arXiv**: [2506.04171](https://arxiv.org/abs/2506.04171)  
**代码**: [Python](https://github.com/cpfpengfei/PCFM) / [Julia](https://github.com/utkarsh530/PCFM.jl)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 流匹配, 物理约束, 硬约束满足, PDE求解, 零样本推理

## 一句话总结
提出 Physics-Constrained Flow Matching (PCFM)，一种零样本推理框架，通过在预训练流匹配模型的采样过程中交替执行前向投射、OT 插值反向更新和松弛惩罚校正，实现任意非线性等式约束的精确满足（达到机器精度），在含激波和间断的 PDE 问题上相比基线方法提升高达 99.5%。

## 研究背景与动机

深度生成模型已被应用于 PDE 物理系统的仿真和不确定性推断，但如何确保生成样本严格满足守恒律、边界条件等物理约束仍是核心挑战。

现有方法的局限：

**软约束方法**（如 PINN 式罚项、DiffusionPDE）只能近似满足约束，在精确约束至关重要的场景下可能导致关键失效——尤其是在含激波的双曲方程中

**ECI 框架**仅验证了简单的线性、非重叠约束（如逐点 Dirichlet 条件），依赖解析投影的闭式解，无法处理非线性或耦合约束

**D-Flow** 需要通过 ODE 求解器反向传播梯度，计算成本极高（4-6x 慢于 PCFM）

**PDM** 在每一步都投影到约束流形上，对非线性或全局约束过度约束采样轨迹

核心矛盾：生成模型的采样过程内在地具有随机性和渐进性，约束只需在最终去噪解上精确满足。如何在保持生成一致性的同时插入物理约束校正？PCFM 的洞察是：可以在每一步"前瞻"到终态、在终态投影到约束流形、再"回溯"到当前步，从而实现约束满足与生成流对齐的兼顾。

## 方法详解

### 整体框架
PCFM 对预训练的 Functional Flow Matching (FFM) 模型进行零样本推理增强。在采样的每个子步 $\tau \to \tau + \delta\tau$ 中执行三个操作：前向射击+投影、OT 反向更新、松弛约束校正。

### 关键设计

1. **前向射击与 Gauss-Newton 投影**:

    - 在每个子步，先前向积分到终态 $\tau=1$：$u_1 = \text{ODESolve}(u(\tau), v_\theta, \tau, 1)$
    - 对终态做一次 Gauss-Newton 投影到约束流形的切空间：$u_{\text{proj}} = u_1 - J^\top(JJ^\top)^{-1}h(u_1)$
    - 当约束 $h$ 是仿射时，该投影恰好是精确解（Proposition E.1）
    - 设计动机：前向射击只需粗精度积分器（如大步长 Euler），成本低

2. **OT 位移插值反向更新**:

    - 将投影后的终态 $u_{\text{proj}}$ 沿 OT 直线路径回溯到 $\tau' = \tau + \delta\tau$：$\hat{u}_{\tau'} = \text{ODESolve}(u_{\text{proj}}, -(u_{\text{proj}} - u_0), 1, \tau')$
    - 核心理论 (Proposition 3.1)：在步长 $\delta\tau$ 趋于 0 时，OT 位移插值 $\bar{v}(u) = u_1 - u_0$ 以 $O(\delta\tau^p)$ 量级逼近真实向量场 $v_\theta$，且反向更新无条件稳定
    - 设计动机：直接反向积分 ODE 因 Jacobian 特征值符号翻转而数值不稳定，OT 插值完全回避此问题

3. **松弛约束校正**:

    - 对回溯得到的 $\hat{u}_{\tau'}$ 做带惩罚的校正：$u_{\tau'} = \arg\min_u \|u - \hat{u}_{\tau'}\|^2 + \lambda\|h(u + \gamma v_\theta(u, \tau'))\|^2$，$\gamma = 1-\tau'$
    - 在外推点 $u + \gamma v_\theta$ 上评估约束，而非当前点，鼓励终态约束满足
    - 设计动机：离散化和非线性会引入残差误差，惩罚项在粗网格下提供鲁棒性

4. **最终投影保证**:

    - 若采样结束后约束残差仍超过阈值 $\epsilon$，执行完整的约束投影：$\min_{u'} \|u' - u_1\|^2 \text{ s.t. } h(u') = 0$
    - 使用自定义批量可微求解器，仅求解 $m \times m$ Schur 补系统（$m = \dim h \ll n$），成本仅占总采样时间的 1-3%

### 约束类型覆盖
- 线性：Dirichlet 初始/边界条件、周期边界下的全局质量守恒
- 非线性：非线性守恒律（如反应扩散方程的非线性质量）、Godunov 通量的局部守恒
- 本框架下 ECI 是特例：$\lambda = 0$ + 线性非重叠约束时退化为 ECI

## 实验关键数据

### 主实验：多PDE约束生成性能

| PDE | 方法 | MMSE ($\times 10^{-2}$) | CE(IC) ($\times 10^{-2}$) | CE(CL) ($\times 10^{-2}$) |
|-----|------|------------------------|--------------------------|--------------------------|
| Heat | PCFM | **0.241** | **0** | **0** |
| Heat | ECI | 0.697 | 0 | 0 |
| Heat | FFM | 4.56 | 579 | 2.11 |
| N-S | PCFM | **4.59** | **0** | **0** |
| N-S | ECI | 5.23 | 0 | 0 |
| Burgers IC | PCFM | **0.052** | **0** | **0** |
| Burgers IC | ECI | 10.0 | 0 | 205 |
| R-D | PCFM | **0.026** | **0** | **0** |
| R-D | ECI | 0.324 | 0 | 6.00 |

### 消融实验：约束配点数量的影响（Burgers IC）

| 配点数 k | MMSE | CE(CL) | 说明 |
|----------|------|--------|------|
| 0 | ~0.3 | ~0 | 仅 IC + 质量守恒 |
| 1 | ~0.15 | ~0 | 添加局部通量约束开始改善 |
| 3 | ~0.08 | ~0 | 持续改善 |
| 5 | **~0.05** | **~0** | 最佳，且 IC/质量约束不受影响 |

### 运行时间对比

| 方法 | Heat (ms/sample) | Burgers (ms/sample) |
|------|-----------------|-------------------|
| PCFM | 291.8 | 1371.0 |
| ECI | 65.6 | 780.6 |
| D-Flow | 2770.5 | 8048.9 |
| DiffusionPDE | 128.6 | 211.2 |

### 关键发现
- PCFM 是唯一能同时精确满足非线性约束（如 Burgers 方程的非线性质量守恒）并捕获激波动力学的方法
- 与 PINN 式软约束方法不同，PCFM 中添加更多互补约束反而改善生成质量——因为硬约束之间不存在梯度竞争
- 松弛约束校正 ($\lambda > 0$) 在步数较少时提供显著的鲁棒性增益

## 亮点与洞察
- 将数值 PDE 求解器的经典思想（投影法、约束满足）引入生成模型推理，是数值方法与深度学习的优雅桥接
- Proposition 3.1 提供了 OT 反向更新的理论保障，解决了神经 ODE 反向积分的数值不稳定性这一长期痛点

## 局限与展望
- 目前仅支持等式约束，不等式约束（如 TVD 条件）的严格执行需要进一步研究
- Gauss-Newton 投影假设约束 Jacobian 满秩，退化情况下需要正则化处理
- 运行时间介于快速但不精确的方法（ECI、DiffusionPDE）和慢但精确的方法（D-Flow）之间

## 相关工作与启发
- **vs ECI**：ECI 是 PCFM 的特例（$\lambda=0$，线性约束），PCFM 支持任意非线性耦合约束
- **vs DiffusionPDE**：DiffusionPDE 使用 PINN 式软罚项，无法精确满足约束且需要反向传播
- **vs PDM**：PDM 在每一步都投影，对全局约束过度约束轨迹；PCFM 通过前瞻-投影-回溯避免此问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 数值方法与生成模型的创新结合，零样本非线性硬约束前所未有
- 实验充分度: ⭐⭐⭐⭐ 覆盖四种 PDE、线性和非线性约束、多个基线对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐⭐ 对科学计算中的约束生成有直接实用价值，框架可推广到分子设计等领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)
- [\[NeurIPS 2025\] Guided Diffusion Sampling on Function Spaces with Applications to PDEs](guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/physics/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[NeurIPS 2025\] Balanced Conic Rectified Flow](balanced_conic_rectified_flow.md)

</div>

<!-- RELATED:END -->
