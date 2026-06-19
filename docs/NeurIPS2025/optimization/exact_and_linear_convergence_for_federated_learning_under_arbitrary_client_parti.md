---
title: >-
  [论文解读] Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] 本文引入随机矩阵和时变图作为建模工具，将联邦学习的客户端参与和本地更新过程统一为矩阵乘法形式，并提出 FOCUS 算法（基于 push-pull 策略），在任意客户端参与：和数据异构下首次实现精确收敛与线性收敛速率。 领域现状：联邦学习（FL）允许多客户端协作训练而不共…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "任意客户端参与"
  - "精确收敛"
  - "push-pull策略"
  - "随机矩阵"
---

# Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable

**会议**: NeurIPS 2025  
**arXiv**: [2503.20117](https://arxiv.org/abs/2503.20117)  
**代码**: [github.com/BichengYing/FedASL](https://github.com/BichengYing/FedASL)  
**领域**: 优化 / 联邦学习  
**关键词**: 联邦学习, 任意客户端参与, 精确收敛, push-pull策略, 随机矩阵

## 一句话总结
本文引入随机矩阵和时变图作为建模工具，将联邦学习的客户端参与和本地更新过程统一为矩阵乘法形式，并提出 FOCUS 算法（基于 push-pull 策略），在**任意客户端参与**和数据异构下首次实现精确收敛与线性收敛速率。

## 研究背景与动机

**领域现状**：联邦学习（FL）允许多客户端协作训练而不共享原始数据。FedAvg 是最流行的算法，但存在根本性的收敛偏差问题。

**现有痛点**：(a) 多步本地更新 + 非 i.i.d. 数据导致**客户端漂移**——FedAvg 的不动点偏离真实全局最优解，偏差为 $\|x^o - x^\star\|^2 = \Omega((\tau-1)\eta)\|x^\star\|^2$；(b) 任意客户端参与引入额外目标偏差——全局模型收敛到参与概率加权的扭曲目标的稳定点，而非问题 (1) 的真实最优解；(c) 现有方法需**衰减学习率**来缓解偏差，导致收敛速度变慢。

**核心矛盾**：在任意（未知）客户端参与概率下，能否用**常数学习率**实现精确收敛到全局最优解？

**本文目标** 设计一个在任意客户端参与下仍能精确收敛（线性速率）的 FL 算法，且不需要有界异质性假设或学习参与概率。

**切入角度**：将 FL 的三个核心操作（客户端参与、本地更新、模型聚合）统一表示为随机矩阵乘法，从去中心化优化视角重新设计 FL 算法。

**核心 idea**：用行随机矩阵建模 pull 操作、列随机矩阵建模 push 操作，通过 push-pull 对偶策略同时消除客户端漂移和参与偏差。

## 方法详解

### 整体框架
FOCUS 将 FL 问题重新形式化为带约束的去中心化优化：$\min_{\{x_0,...,x_N\}} \frac{1}{N}\sum_{i=0}^N f_i(x_i)$，约束 $R(S_r)\boldsymbol{x} = \boldsymbol{x}$（一致性约束）。通过 push-pull 原始-对偶方法求解，核心迭代：

$$\boldsymbol{x}_{k+1} = R_k(\boldsymbol{x}_k - \eta D_k \boldsymbol{y}_k), \quad \boldsymbol{y}_{k+1} = C_k(\boldsymbol{y}_k + \nabla\boldsymbol{f}(\boldsymbol{x}_{k+1}) - \nabla\boldsymbol{f}(\boldsymbol{x}_k))$$

### 关键设计

1. **随机矩阵建模框架**:

    - 功能：将 FL 的所有操作统一为矩阵-向量乘法
    - 核心思路：定义四种矩阵——
        - **模型分配矩阵** $R(S_r)$：行随机矩阵，对参与客户端 $i \in S_r$ 置 $R[i,0]=1$（从服务器拉取模型），非参与者 $R[i,i]=1$（保持不变）
        - **模型平均矩阵** $A(S_r)$：行随机矩阵，服务器行取参与者平均 $A[0,j] = 1/|S_r|$
        - **列随机矩阵** $C(S_r) = R(S_r)^\top$：用于 push 操作
        - **双随机矩阵** $W(S_r)$：用于去中心化场景（本文不直接使用）
    - 设计动机：经典 FL 用 per-client 符号描述，无法利用矩阵分析的强大工具。堆叠向量-矩阵表示 $\boldsymbol{x}_k = \text{vstack}[x_{k,0}; x_{k,1}; ...; x_{k,N}]$ 使问题自然映射到图和矩阵理论

2. **任意客户端参与建模 (Assumption 1)**:

    - 功能：用统一框架覆盖全参与、主动参与、被动参与三种模式
    - 核心思路：每个客户端 $i$ 以未知概率 $p_i \in (0,1]$ 参与，平均权重 $q_i = \mathbb{E}[\mathbb{I}_i / \sum_j \mathbb{I}_j]$。期望矩阵 $\bar{R}$ 和 $\bar{A}$ 的结构被显式推导
    - 关键特例：均匀采样（$p_i \equiv m/N, q_i \equiv 1/m$）；Active 参与（独立 Bernoulli）；Passive 参与（无放回类别分布）

3. **FOCUS 算法的 Push-Pull 机制**:

    - 功能：通过对偶变量跟踪全局梯度，消除本地更新和非均匀参与引入的偏差
    - 核心思路：
        - **Pull (行随机矩阵 $R$)**：参与客户端从服务器拉取模型 $x_{0,i}^{(r)} \Leftarrow x_r$
        - **本地更新**：客户端执行 $\tau$ 步梯度跟踪更新 $y_{t+1,i}^{(r)} = y_{t,i}^{(r)} + \nabla f_i(x_{t,i}^{(r)}) - \nabla f_i(x_{t-1,i}^{(r)})$，$x_{t+1,i}^{(r)} = x_{t,i}^{(r)} - \eta y_{t+1,i}^{(r)}$
        - **Push (列随机矩阵 $C$)**：客户端推送 $y_{\tau,i}^{(r)}$ 到服务器（**求和而非平均**），$y_{r+1} \Leftarrow y_r + \sum_{i \in S_r} y_{\tau,i}^{(r)}$
    - **两个关键性质**：
        - **一致性 (Consensus)**：$R(S_r)\boldsymbol{x}^\star = \boldsymbol{x}^\star$ → 所有模型最终收敛到同一值
        - **跟踪性 (Tracking)**：$\mathbf{1}^\top \boldsymbol{y}_k = \mathbf{1}^\top \nabla\boldsymbol{f}(\boldsymbol{x}_k)$ → 对偶变量之和始终等于全局梯度和
    - 设计动机：FedAvg 的混合矩阵 $W_k$ 在非全参与时**不是双随机矩阵**，导致无法精确收敛。Push-pull 用两种不同随机矩阵分别处理模型一致性（行随机）和梯度跟踪（列随机），绕开此限制

4. **与 FedAvg 的关键区别**:

    - FedAvg 拉取模型 $x$ 并平均模型 → FOCUS 拉取模型 $x$ 但推送梯度 $y$
    - FedAvg 用行随机矩阵 $A$ 聚合 → FOCUS 用列随机矩阵 $C$ 推送
    - FedAvg 不需要维护对偶变量 → FOCUS 服务器维护全局 $y_r$

### 随机梯度变体 SG-FOCUS
将确定性梯度替换为随机梯度 $\hat{\nabla}f_i$，理论和实验均展示更快收敛和更高精度。

## 实验关键数据

### 收敛性能对比

| 算法 | 精确收敛 | 强凸复杂度 | 非凸复杂度 | 参与假设 | 异质梯度假设 |
|------|:---:|---------|---------|---------|-----------|
| FedAvg | ✗ | $O(1/\epsilon)$ | $O(1/\epsilon^2)$ | 均匀 | 有界 |
| SCAFFOLD | ✓* | $O(\log(1/\epsilon))$ | $O(1/\epsilon)$ | 均匀 | 无 |
| FedAU | ✗ | — | $O(1/\epsilon^2)$ | 任意 | 有界 |
| FedAWE | ✗ | — | $O(1/\epsilon^2)$ | 任意 | 有界 |
| MIFA | ✗ | — | $O(1/\epsilon^2)$ | 任意 | 有界 |
| **FOCUS** | **✓** | $O(\log(1/\epsilon))$ | $O(\log(1/\epsilon))$† | **任意** | **无** |

*SCAFFOLD 在任意参与下无收敛证明；†需 PL 条件

### FedAvg 不动点偏差分析

| 场景 | FedAvg 不动点偏差 | FOCUS |
|------|----------------|-------|
| 均匀参与 + i.i.d. | $\Omega((\tau-1)\eta)$ | 0（精确） |
| 非均匀参与 | 额外偏差 $\delta_q^2$ | 0（精确） |
| 数据异质 | $\sigma_g^2$ 相关 | 0（精确） |

### 关键发现
- FOCUS 是**首个**在任意客户端参与下实现精确收敛 + 线性速率的 FL 算法
- 不需要学习参与概率（FedAU、FedAWE 需要），不需要有界异质梯度假设
- 在 PL 条件下非凸问题也实现对数复杂度（vs 所有现有方法的多项式复杂度）
- SG-FOCUS 随机变体在理论和实验中均优于确定性版本
- 通信开销与 SCAFFOLD 类似（每轮 $2d$ 向量），但支持任意参与

## 亮点与洞察
- **视角转换的力量**：将 FL 从 per-client 描述转化为矩阵乘法形式，这一视角转换直接启用了去中心化优化的成熟工具（push-pull、时变图）。简单的记号变换带来了全新的算法设计空间
- **Push-Pull 的优雅性**：行随机矩阵保证一致性、列随机矩阵保证跟踪性，两种矩阵的对偶配合恰好消除了 FedAvg 的两大偏差源。且只需服务器维护一个额外变量 $y_r$
- **Tracking Property 的关键作用**：$\mathbf{1}^\top \boldsymbol{y}_k = \mathbf{1}^\top \nabla\boldsymbol{f}(\boldsymbol{x}_k)$ 这一不变量确保了即使参与客户端集合在变化，全局梯度信息始终被精确追踪

## 局限与展望
- 通信开销每轮为 $2d$（模型 $x$ + 对偶 $y$），比 vanilla FedAvg 的 $d$ 多一倍
- 理论分析假设确定性梯度（SG-FOCUS 有实验但理论尚不完整）
- 非凸的线性收敛速率需要 PL 条件——一般非凸问题下 FOCUS 的具体复杂度未给出
- 实际大规模实验（数千客户端、真实异构数据）的验证未在主文中展示
- 服务器需要记住每个参与客户端的贡献 $y_{\tau,i}^{(r)}$ 进行求和——对极大规模参与有存储压力

## 相关工作与启发
- **vs FedAvg [McMahan et al., 2017]**: 简单直观但在非均匀参与下有不可消除的偏差。FOCUS 从数学原理出发彻底消除偏差
- **vs SCAFFOLD [Karimireddy et al., 2020]**: 通过控制变量消除客户端漂移，但要求均匀采样。FOCUS 不假设采样分布
- **vs FedAU/FedAWE [Wang & Ji, 2024; Xiang et al., 2024]**: 通过学习参与概率来校正偏差，但学习过程本身减慢收敛且无法实现精确收敛
- **vs MIFA [Gu et al., 2021]**: 用方差减少处理参与偏差，但需服务器存储每个客户端模型且有有界延迟假设

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 视角转换（FL → 矩阵乘法 → push-pull）非常巧妙，首次实现任意参与下精确线性收敛
- 实验充分度: ⭐⭐⭐ 理论贡献为主，实验验证在附录中，规模有限
- 写作质量: ⭐⭐⭐⭐ 矩阵建模框架的逐步引入非常清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了 FL 领域的根本性理论问题，具有很强的实践指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)

</div>

<!-- RELATED:END -->
