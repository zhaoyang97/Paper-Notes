---
title: >-
  [论文解读] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers
description: >-
  [ICLR 2026][物理/科学计算][神经算子] 论证神经 PDE 求解器在边界条件变化时学到的不是单一的解算子，而是由边界条件索引的算子族，并从学习理论角度形式化了 ERM 下边界分布偏移导致的不可辨识性问题。 神经 PDE 求解器（如 FNO）通常被描述为"学习解算子"——从问题输入映射到 PDE 解。然而…
tags:
  - "ICLR 2026"
  - "物理/科学计算"
  - "神经算子"
  - "边界条件"
  - "分布偏移"
  - "不可辨识性"
---

# One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers

**会议**: ICLR 2026  
**arXiv**: [2603.01406](https://arxiv.org/abs/2603.01406)  
**代码**: [有](https://github.com/lennonshikhman/boundary-indexed-neural-pde)  
**领域**: Scientific Computing / Neural PDE Solvers  
**关键词**: Neural Operator, 边界条件, 分布偏移, 不可辨识性, Fourier Neural Operator

## 一句话总结

论证神经 PDE 求解器在边界条件变化时学到的不是单一的解算子，而是由边界条件索引的算子族，并从学习理论角度形式化了 ERM 下边界分布偏移导致的不可辨识性问题。

## 研究背景与动机

神经 PDE 求解器（如 FNO）通常被描述为"学习解算子"——从问题输入映射到 PDE 解。然而，从经典 PDE 理论看，解算子不仅由微分方程本身定义，**边界条件**是良定性和唯一性的核心要素。

现有方法通常将边界条件隐式编码（如边界填充、辅助通道），这引出一个根本性问题：当边界条件不固定时，神经求解器到底在近似什么？作者指出，它们学到的映射内在地绑定于训练时见过的边界条件分布，这导致在边界分布偏移时会出现不可预期的失败——而这种失败不是架构不足或优化问题。

## 方法详解

### 整体框架

本文不提出新架构，而是把神经 PDE 求解器的训练放回学习理论的显微镜下：将算子学习写成在边界条件分布上的条件风险最小化，再追问当边界条件本身可变时，经验风险最小化（empirical risk minimization, ERM）究竟把模型推向了什么解。整篇论证由两条相互衔接的分析串起来：先在 ERM 框架里证明模型学到的并非边界无关的单一解算子，而是被训练时见过的那簇边界条件索引出来的**算子族**，且在训练边界分布支撑集之外**不可辨识**；再进一步说明当边界信息从输入里被抹掉时，模型的最优预测会**退化成对所有边界取平均的条件期望**，因而在任何一个具体边界条件上都失败。两条分析分别由下面两个关键设计承接，并用同一套受控实验（同方程、同 forcing、同分辨率）逐一坐实。

### 关键设计

**1. 边界索引算子族的形式化：把"学解算子"重述为"学一族被边界索引的算子"**

经典 PDE 理论里，解的唯一性同时依赖微分方程和边界条件 $\mathcal{B}$，但神经求解器常把 $\mathcal{B}$ 隐式塞进边界填充或辅助通道里，于是"学解算子"这句话掩盖了它对训练边界分布的依赖。作者把训练目标写成条件风险最小化 $\min_\theta \mathbb{E}_{(f,\mathcal{B})\sim\mu}\big[\ell(\hat{\mathcal{S}}_\theta(f,\mathcal{B}),\,\mathcal{S}(f,\mathcal{B}))\big]$：当 $\mathcal{B}$ 固定时学到的是单一算子 $\mathcal{S}_\mathcal{B}:f\mapsto u$，一旦 $\mathcal{B}$ 随分布 $\mu_\mathcal{B}$ 变化，学到的就是联合映射 $\mathcal{S}:(f,\mathcal{B})\mapsto u$。关键在于 ERM 只在 $\mu_\mathcal{B}$ 的支撑集上对模型施加约束，支撑集之外多个不同映射可以达到同样低的训练损失，因此模型在分布外的行为不可辨识。这正解释了为何对外推力 $f$ 的鲁棒性并不等价于对边界条件的鲁棒性——$f$ 在输入中被密集采样，而边界条件往往只占据一个低维、稀疏的子空间，留下大片未被约束的方向。

**2. 条件期望退化行为：解释边界信息被抹掉时模型为何退化成"平均解"**

当边界信息被删除或表示得太弱（比如去掉边界通道）时，模型拿不到区分不同边界条件所需的输入，其在 MSE 下的最优预测被迫塌缩为对未观测边界变量取平均的条件期望 $\hat{u}(f)\approx\mathbb{E}_{\mathcal{B}\sim\mu_\mathcal{B}}[\mathcal{S}(f,\mathcal{B})\mid f]$。这个平均量不对应任何一个固定边界条件下的合法解，于是模型在每一个具体分布上都失败——它学到的是横跨所有边界条件的"平均解"而非某个特定解。实验中固定一个 forcing function $f^*$、把无边界版模型的输出与 $\mathbb{E}[u\mid f^*]$ 的蒙特卡洛估计逐点对比，二者几乎完全吻合，从经验上坐实了这条退化机制。

### 损失函数 / 训练策略

实验统一采用 FNO 架构、Adam 优化器（学习率 $8\times10^{-4}$）、MSE 损失，在 $64\times64$ 网格上做 2500 步梯度更新、batch size 为 12；边界函数用截断 Fourier 展开参数化，带宽 $K=6$。这套配置刻意保持简单且固定，是为了把跨边界分布的失败干净地归因到 ERM 目标本身，而非架构容量或优化超参。

## 实验关键数据

### 主实验（表格）

**跨分布泛化（Poisson 方程）**：

| 模型 | 测试 $\mu_{B_0}$ | 测试 $\mu_{B_1}$ |
|------|-------------------|-------------------|
| FNO (训练 $\mu_{B_0}$) | 0.078 ± 0.005 | **0.489 ± 0.022** |
| FNO (训练 $\mu_{B_1}$) | **0.601 ± 0.036** | 0.102 ± 0.003 |
| FNO (无边界通道) | 0.999 ± 0.001 | 1.001 ± 0.001 |

每个模型仅在训练边界分布上表现良好，在另一分布上误差飙升 5-6 倍。无边界通道的模型在所有分布上完全失败（误差 ≈ 1.0）。

### 消融实验（表格）

**边界外推（Dirichlet 均值偏移）**：

| 偏移量 $\delta$ | -1.0 | -0.5 | 0 | 0.5 | 1.0 |
|----------------|------|------|---|-----|-----|
| 误差趋势 | 高 | 中 | 低（域内） | 中 | 高 |

误差随偏移量对称增长，连续退化而非突变。频率外推（将 Dirichlet 带宽从 $K=6$ 增至 12）同样导致单调性能下降。

### 关键发现

- **跨分布失败是结构性的**：不是模型容量不足或优化不稳定，而是 ERM 目标的内在限制——在相同 FNO 架构下，域内性能良好证明了这一点
- **条件期望行为验证**：固定一个 forcing function $f^*$，无边界版模型的输出几乎完全匹配 $\mathbb{E}[u \mid f^*]$ 的蒙特卡洛估计
- **分辨率鲁棒性 ≠ 边界鲁棒性**：能跨网格分辨率泛化的模型在边界偏移下仍会严重失败

## 亮点与洞察

- 不是提出新方法、而是提供新理解的典范性工作——用条件风险最小化和不可辨识性精确刻画了一个被忽视的基本问题
- 对"基础模型 for PDE"的热潮提出了冷静的结构性警示：仅靠扩大数据规模和模型容量无法解决边界泛化问题
- 实验设计简洁精巧：通过控制变量（同一方程、同一 forcing、同一分辨率）精确隔离边界条件的影响

## 局限与展望

- 仅在二维 Poisson 方程上实验，未扩展到抛物线、双曲或非线性 PDE
- 未考虑时间依赖系统中边界条件与初始条件的交互
- 提供了诊断框架但未提出解决方案（如边界感知架构或不变表示）
- 未探索跨随机初始化种子的变异性

## 相关工作与启发

- 与之前观察到边界敏感性并提出边界感知架构的工作互补（本文提供理论解释）
- 启发方向：将边界条件作为"一等公民"纳入算子学习的设计——如条件算子分解、结构化边界编码、不变表示学习
- 对 PDE 基础模型（如 Subramanian et al., 2023）的评估协议提出重要启示：需要显式测试边界分布偏移

## 评分

⭐⭐⭐⭐ 理论洞察深刻，用简洁实验精确验证了核心论点，对神经 PDE 求解器社区具有重要的警示意义，但缺少建设性解决方案。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](../../ICML2026/physics/topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[CVPR 2026\] Spatial-Spectral Residuals Informed Diffusion Neural Operator for Pan-sharpening](../../CVPR2026/physics/spatial-spectral_residuals_informed_diffusion_neural_operator_for_pan-sharpening.md)
- [\[ICLR 2026\] ATOM: A Pretrained Neural Operator for Multitask Molecular Dynamics](atom_a_pretrained_neural_operator_for_multitask_molecular_dynamics.md)
- [\[ICCV 2025\] JPEG Processing Neural Operator for Backward-Compatible Coding](../../ICCV2025/physics/jpeg_processing_neural_operator_for_backward-compatible_coding.md)

</div>

<!-- RELATED:END -->
