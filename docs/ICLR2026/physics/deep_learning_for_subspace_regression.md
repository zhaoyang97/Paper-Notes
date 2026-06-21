---
title: >-
  [论文解读] Deep Learning for Subspace Regression
description: >-
  [ICLR2026][物理/科学计算][subspace regression] 将缩减阶建模（ROM）中的子空间预测问题形式化为 Grassmann 流形上的回归任务，提出专用损失函数与子空间嵌入（subspace embedding）技术——通过预测比目标更大维度的子空间来降低映射复杂度——在特征值问题、参数化 PDE 和迭代法加速等场景中均取得显著效果。
tags:
  - "ICLR2026"
  - "物理/科学计算"
  - "subspace regression"
  - "Grassmann manifold"
  - "reduced order modeling"
  - "神经算子"
  - "eigenspace"
---

# Deep Learning for Subspace Regression

**会议**: ICLR2026  
**arXiv**: [2509.23249](https://arxiv.org/abs/2509.23249)  
**代码**: [GitHub](https://github.com/VLSF/subreg)  
**领域**: 科学计算  
**关键词**: subspace regression, Grassmann manifold, reduced order modeling, neural operator, eigenspace

## 一句话总结

将缩减阶建模（ROM）中的子空间预测问题形式化为 Grassmann 流形上的回归任务，提出专用损失函数与子空间嵌入（subspace embedding）技术——通过预测比目标更大维度的子空间来降低映射复杂度——在特征值问题、参数化 PDE 和迭代法加速等场景中均取得显著效果。

## 研究背景与动机

**领域现状**：缩减阶建模（Reduced Order Modelling, ROM）通过识别线性子空间来丢弃无信息的自由度，从而简化系统的分析和模拟。这一思路在参数化偏微分方程（PDE）、最优控制、不确定性量化等反复求解相关问题的场景中极具价值。典型方法如 Proper Orthogonal Decomposition（POD）通过对代表性参数的解做最佳低秩逼近来构建缩减基。

**现有痛点**：当子空间显式依赖参数时（即 local POD），需要从已知参数对应的子空间"插值"到未知参数对应的子空间。然而，Grassmann 流形上的经典插值方法（如 Riemannian 正规坐标插值）在高维参数空间中极不可靠——观测稀疏导致切空间近似质量低下，且计算代价随维度快速增长。

**核心矛盾**：子空间数据具有特殊的代数结构——右 $\text{GL}(k)$ 不变性，即矩阵 $W_1$ 和 $W_2$ 若列空间相同则等价。这使得标准的 $\ell_2$ 回归损失完全不适用。同时，参数化特征问题的复杂度（不同子空间配置数）随参数维度和特征值序号急剧增长，直接学习单个特征向量面临组合爆炸。

**本文方案**：将插值问题放松为回归问题，设计满足 $\text{GL}(k)$ 不变性的损失函数，用神经网络参数化高维参数到子空间的映射。核心创新是引入**子空间嵌入**——预测包含目标子空间的更大维度子空间，理论证明可降低映射的导数范数（更平滑），从而契合神经网络的 F-principle（频谱偏差），大幅提升精度。

## 方法详解

### 整体框架

本文把"参数到子空间"的插值难题改写成一个 Grassmann 流形上的回归问题：给定数据集 $\mathcal{D} = \{(r_i, V_i)\}_{i=1}^m$，其中 $r \in \mathbb{R}^p$ 是参数、$V(r) \in \text{Gr}(k, n)$ 是对应的 $k$ 维子空间，用一个 FFNO（Factorized Fourier Neural Operator）参数化的网络 $Y_\theta: \mathbb{R}^p \to \text{Gr}(r, n)$（允许预测维度 $r \geq k$）去逼近这个映射，目标是最小化一个能容忍子空间冗余表示的损失 $\theta^\star = \arg\min_\theta \frac{1}{m} \sum_i L(Y_\theta(r_i), V_i)$。整套方法的三块拼图分别是：一个对子空间等价类不变的损失函数、一个用"预测更大子空间"换取平滑性的嵌入技巧，以及把特征问题、local POD、迭代法加速等场景统一到同一框架下的建模视角。

### 关键设计

**1. 子空间损失函数：让 $\ell_2$ 回归失效的 $\text{GL}(k)$ 不变性变成可优化目标**

子空间数据有个麻烦的代数结构——两个矩阵只要列空间相同就等价（右 $\text{GL}(k)$ 不变），所以逐元素的 $\ell_2$ 损失会把同一个子空间的不同基当成不同目标。本文要求损失 $L$ 同时满足两条：对等价类不变 $L(A,B) = L(\tilde{A}, \tilde{B})$，以及 $L(A,B)=0$ 当且仅当 $\mathcal{S}(B) \subset \mathcal{S}(A)$（注意是包含，不是相等，这为后面的嵌入留了口子）。Theorem 1 给出两种构造。确定性损失 $L_1(A, B) = p - \|Q_B^\top Q_A\|_F^2$ 通过对 $A,B$ 各做一次 QR 分解（$A = Q_A R_A$、$B = Q_B R_B$）来度量正交投影子之差，干净但要算两次 QR；随机化损失 $L_2(A, B; z) = \min_u \|Au - Q_B z\|_2^2$（$z \sim \mathcal{N}(0, I_k)$）则用一个最小二乘问题替掉投影子，再借 Hutchinson 迹估计省掉第二次 QR，并且在期望意义上严格等于前者 $\mathbb{E}_z[L_2] = L_1$。$L_2$ 可以直接用正规方程求解，子空间维度一大训练效率就明显碾压 $L_1$；代价是正规方程在大维度下条件数会恶化，所以再叠一层 Cholesky-QR2 稳定化得到 $L_2^{\text{stab}}$，把数值精度找回来。

**2. 子空间嵌入：用"多预测几维冗余子空间"换映射平滑性**

这是全文最核心的招数，思路出奇地简单——既然损失只要求 $\mathcal{S}(V) \subset \mathcal{S}(Y_\theta)$，那就索性让网络预测一个 $r > k$ 维的更大子空间，只要它包住目标即可。冗余看似浪费，却同时改善了优化难度和逼近难度两件事。一方面，Theorem 2 证明对任意连续可微的 $V: \mathbb{R} \to \text{Gr}(k, n)$，总能构造一个 $W: \mathbb{R} \to \text{Gr}(r, n)$ 使 $\|\dot{W}(t)\|_F^2 \leq \|\dot{V}(t)\|_F^2$ 且在导数非零处严格更小，也就是嵌进更大子空间后映射变得更平滑，这恰好踩中神经网络的 F-principle（频谱偏差，网络天生偏爱低频平滑函数）。另一方面，Theorem 3 用组合数学刻画了为什么直接学不动：对常系数椭圆特征问题 $-\sum a_i \partial^2 \phi / \partial x_i^2 = \lambda \phi$，参数到第 $k$ 个特征向量的映射是分段常数，常数区域数随维度 $D$ 增长为 $\#_{F_k}(k, D) \sim \frac{1}{(D-1)!} k(\log k)^{D-1}$，而朴素的子空间映射区域更多（$\#_{G_k} \geq \frac{1}{(D-1)!} k^{D-1}$）；嵌入则反过来利用"有限几种特征向量的不同组合"来合并区域，极端情况下当预测维度涨到所有可能特征向量数 $\#_{F_k}$ 时，映射直接退化成常数函数，最易学。这也解释了实验里把预测维度从 10 加到 40、误差从 30% 掉到 2% 的现象。

**3. 多场景统一覆盖：同一套回归框架接管科学计算里反复出现的"找子空间"任务**

一旦把问题抽象成"参数 → 子空间"的回归，许多看似无关的缩减阶建模任务就落进了同一个框架，只是子空间的物理含义不同。参数化特征问题里，子空间是前 $K$ 个特征函数张成的 $\text{span}\{\phi_1, \dots, \phi_K\}$；Burgers 方程的 local POD 里是随 PDE 参数变化的缩减基 $\{\psi_1, \dots, \psi_k\}$；共轭梯度 deflation 里是待消除的小特征值特征空间 $\mathcal{S}(V)$；Jacobi 粗网格修正里是误差传播矩阵的主特征空间；最优控制的平衡截断里则是可控可观性约简基 $\mathcal{S}(\bar{\mathcal{T}})$。下表汇总了这些映射：

| 应用场景 | 映射关系 | 子空间含义 |
|---------|---------|-----------|
| 参数化特征问题 | $U(x) \to \text{span}\{\phi_1, \dots, \phi_K\}$ | 前 $K$ 个特征函数 |
| 本地 POD（Burgers 方程） | PDE 参数 $\to \{\psi_1, \dots, \psi_k\}$ | POD 缩减基 |
| 共轭梯度 Deflation | $k(x) \to \mathcal{S}(V)$ | 小特征值特征空间 |
| 粗网格修正（Jacobi） | $k(x) \to \mathcal{S}(V)$ | 误差传播矩阵主特征空间 |
| 平衡截断（最优控制） | $A, B, C \to \mathcal{S}(\bar{\mathcal{T}})$ | 可控可观性约简基 |

## 实验关键数据

### 量子力学特征空间预测

| 数据集 | Riemannian 插值 | $\mathbb{Z}_2$-adjusted $\ell_2$ | 子空间回归 $L_1$ |
|--------|:--------:|:--------:|:--------:|
| $D=1$ Schrödinger | 4.69% | 2.33% | **0.09%** |
| $D=2$, 数据集 a | 31.9% | 19.52% | **0.65%** |
| $D=2$, 数据集 b | 92.64% | 48.56% | **15.58%** |

子空间回归在所有数据集上均大幅优于经典插值和直接特征向量预测。

### 子空间嵌入效果（$D=2$ 椭圆问题，前 10 特征向量）

| 预测子空间维度 $N_{\text{sub}}$ | 测试误差 | 说明 |
|:---:|:---:|:---|
| 10（无嵌入） | ~30% | 与目标维度相同 |
| 20 | ~10% | 嵌入开始生效 |
| 30 | ~5% | 显著改善 |
| 40 | **~2%** | 仅占总自由度的 0.4% |

嵌入技术将测试误差从 30% 降至 2%，同时训练-测试泛化差距也系统性缩小。

### $D=3$ 椭圆问题损失函数对比

| $N_{\text{sub}}$ | $L_1(A,B)$ | $L_2(A,B;z)$ | $L_2^{\text{stab}}(A,B;z)$ |
|:---:|:---:|:---:|:---:|
| 6 | 24.77% | 31.46% | 28.28% |
| 12 | 13.69% | 17.12% | 15.88% |
| 24 | 9.71% | 失败 | 9.49% |
| 48 | 7.54% | 16.3% | **7.4%** |

$L_2$ 在大维度下因正规方程病态条件数而不稳定，Cholesky-QR2 稳定化后恢复甚至略优于 $L_1$。

## 亮点与洞察

- **子空间嵌入思想**极为精妙：利用子空间包含关系的单调性，用冗余换平滑性，完美契合神经网络的频谱偏差（F-principle）。代价是学到的子空间维度远大于最优，但缩减模型的额外计算量可忽略
- **Theorem 3 的复杂度刻画**将组合数学与逼近论联系，为子空间学习提供了精确的困难度量
- **随机化损失 $L_2$** 将 Hutchinson 迹估计引入子空间学习，大维度下训练时间显著优于 QR 方案
- **迭代法实验的意外发现**：标准 Jacobi 方法的主特征空间包含高低频混合函数导致学习完全失败，改用阻尼 Jacobi ($\omega=0.9$) 后变为纯低频函数，学习难度骤降——问题公式化对学习难度有决定性影响

## 局限与展望

- 子空间嵌入的最优冗余维度 $r - k$ 缺乏自动选择机制，目前需手动网格搜索
- 仅在线性子空间上验证，向非线性流形（如 Stiefel 流形上的约束逼近）的扩展尚未涉及
- 结果为单次运行无误差条（虽然附录有方差分析表明影响小）
- 所有神经网络学到的表示效率远低于最优子空间，这一根本性效率差距是否可弥补仍是开放问题

## 相关工作与启发

- **vs Grassmann 流形插值**：经典方法在高维参数空间中因稀疏观测而失效，本文用回归+神经网络克服维度灾难
- **vs 直接特征向量预测**：$\mathbb{Z}_2$-adjusted $\ell_2$ 虽处理符号不定性，但无法利用子空间结构避免 Theorem 3 的组合爆炸
- **vs 神经算子（FNO/DeepONet）**：互补关系——神经算子直接预测 PDE 解，本文预测缩减基然后求解缩减模型
- **vs DeepPOD**：类似思路但用投影损失直接从快照矩阵提取基，子空间回归在精度上匹配或略优

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 子空间回归问题形式化 + 嵌入技术 + Theorem 3 复杂度分析的理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 多种应用场景覆盖广，与多种 baseline 比较全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，数学符号规范，结构清晰
- 价值: ⭐⭐⭐⭐⭐ 为缩减阶建模提供了强大的新工具，子空间嵌入思想具有广泛的方法论启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials](advancing_universal_deep_learning_for_electronic-structure_hamiltonian_predictio.md)
- [\[ICLR 2026\] Locally Subspace-Informed Neural Operators for Efficient Multiscale PDE Solving](locally_subspace-informed_neural_operators_for_efficient_multiscale_pde_solving.md)
- [\[ICLR 2026\] Accelerating Eigenvalue Dataset Generation via Chebyshev Subspace Filter](accelerating_eigenvalue_dataset_generation_via_chebyshev_subspace_filter.md)
- [\[ICLR 2026\] GenSR: Symbolic Regression based on Equation Generative Space](gensr_symbolic_regression_based_on_equation_generative_space.md)
- [\[NeurIPS 2025\] Toward Complete Merger Identification at Cosmic Noon with Deep Learning](../../NeurIPS2025/physics/toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)

</div>

<!-- RELATED:END -->
