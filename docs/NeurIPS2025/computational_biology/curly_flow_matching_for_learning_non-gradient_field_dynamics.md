---
title: >-
  [论文解读] Curly Flow Matching for Learning Non-gradient Field Dynamics
description: >-
  [NeurIPS 2025][计算生物][Flow Matching] 提出 Curly Flow Matching (Curly-FM)，通过设计带有非零参考漂移的 Schrödinger Bridge 问题，使 flow matching 能够学习周期性、旋转性等非梯度场动力学，突破了传统方法只能建模梯度场的限制。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "Flow Matching"
  - "Schrödinger Bridge"
  - "非梯度场动力学"
  - "轨迹推断"
  - "最优传输"
---

# Curly Flow Matching for Learning Non-gradient Field Dynamics

**会议**: NeurIPS 2025  
**arXiv**: [2510.26645](https://arxiv.org/abs/2510.26645)  
**代码**: [GitHub](https://github.com/kpetrovicc/curly-flow-matching)  
**领域**: 计算生物  
**关键词**: Flow Matching, Schrödinger Bridge, 非梯度场动力学, 轨迹推断, 最优传输

## 一句话总结

提出 Curly Flow Matching (Curly-FM)，通过设计带有非零参考漂移的 Schrödinger Bridge 问题，使 flow matching 能够学习周期性、旋转性等非梯度场动力学，突破了传统方法只能建模梯度场的限制。

## 研究背景与动机

从群体级别观测中建模自然过程的传输动力学是自然科学中的普遍问题。当前主流方法（如 Conditional Flow Matching, OT-CFM）依赖最小作用量原理，假设系统遵循梯度场动力学，即轨迹最小化两个概率测度之间的能量泛函。

然而，许多真实系统天然表现出**非梯度场**的周期性行为：
- **单细胞 RNA 细胞周期**：细胞在基因表达空间中沿周期性轨迹运动
- **计算流体动力学**：涡旋结构产生旋转性流场
- **海洋洋流**：具有复杂的旋转和周期性模式

这些系统的非梯度（curl）分量从根本上无法被现有的 flow matching 和 bridge matching 方法捕获。传统方法生成的轨迹是直线插值，完全忽略了底层系统的真实动态。

## 方法详解

### 整体框架

Curly-FM 是一个两阶段的 simulation-free 框架：

**第一阶段 — 学习参考漂移（Algorithm 1）**：通过神经路径插值器（neural path interpolant）学习参考过程的漂移，使生成的路径能够弯曲以匹配已知的速度场信息。

**第二阶段 — 求解传输计划（Algorithm 2）**：基于学到的参考漂移，通过边际 flow matching 目标计算最优耦合（coupling），训练最终的漂移模型。

### 关键设计

**1. 非零漂移参考过程**

与传统 Schrödinger Bridge 方法使用零漂移（布朗运动）参考过程不同，Curly-FM 构建一个具有非零漂移的参考过程 $Q$，该漂移从数据中推断的速度信息构建。具体地，参考过程的漂移 $f_t$ 是利用核函数 $\kappa_t$ 对离散观测点的速度进行平滑得到的连续场。

**2. 神经路径插值器**

使用神经网络 $\phi_{t,\eta}(x_0, x_1)$ 参数化条件桥的均值，使其不再是 $x_0$ 和 $x_1$ 之间的直线，而是弯曲的路径。条件分布形式为：

$$P_{t|0,1}(x_t) = \mathcal{N}(x_t; t x_1 + (1-t)x_0 + t(1-t)\phi_{t,\eta}(x_0, x_1), \sigma_t^2 t(1-t))$$

插值器通过最小化 $\mathbb{E}[\|\partial_t X_{t,\eta} - f_t(X_{t,\eta})\|^2]$ 来匹配参考漂移。

**3. 最优耦合计算**

利用已学习的神经路径插值器估计 OT 代价 $c(x_0, x_1)$，通过 mini-batch OT 近似求解熵正则化最优传输来获得耦合 $\pi^*$。代价通过 Monte Carlo 估计路径长度的积分来近似。

### 损失函数 / 训练策略

**阶段一损失**（学习参考漂移匹配）：
$$\mathcal{L}(\eta) = \mathbb{E}_{(X_0, X_1, t)}\left[\|\partial_t X_{t,\eta} - f_t(X_{t,\eta})\|^2\right]$$

**阶段二损失**（flow matching + score matching）：
```python
flow_loss = E[||v_t(x_t, t) - x_dot_t||^2]
score_loss = E[||(lambda_t * s_t(x_t, t) + eps)||^2]
loss = flow_loss + score_loss
```

其中 score model 用于支持随机性（$\sigma > 0$）的情况。

**推理**：使用学到的漂移模型通过 SDE/ODE 积分前向演化。

## 实验关键数据

### 主实验

**Table 1：单细胞 RNA 轨迹推断（细胞周期数据）**

| 方法 | W1 ↓ | W2 ↓ | Cosine Dist ↓ |
|------|------|------|---------------|
| CFM | 高 | 高 | ~1.0（直线轨迹） |
| OT-CFM | 高 | 高 | ~1.0 |
| CurlyFM | **最低** | **最低** | **~0.03** |

**Table 2：海洋洋流轨迹推断**

| 方法 | W1 ↓ | Cosine Dist ↓ | 计算时间 (h) |
|------|------|---------------|-------------|
| DM-SB | - | - | 15.44 |
| TrajectoryNet | - | - | 7.44 |
| SBIRR | - | - | 4.67 |
| Vanilla-SB | - | - | 0.43 |
| CurlyFM | **0.062±0.003** | **0.034±0.006** | **0.06** |

CurlyFM 在海洋洋流任务上，训练时间仅需约 4 分钟，比 SBIRR 快 ~78 倍，比 TrajectoryNet 快 ~124 倍。

### 消融实验

**噪声参考漂移鲁棒性（Table R2）**

| 噪声比 β | W1 ↓ | W2 ↓ | Cosine Dist ↓ |
|-----------|------|------|---------------|
| 0.00 | 0.062±0.003 | 0.143±0.010 | 0.034±0.006 |
| 0.25 | 0.057±0.033 | 0.021±0.036 | 0.051±0.030 |
| 0.50 | 0.087±0.047 | 0.301±0.085 | 0.091±0.046 |
| 0.75 | 0.261±0.123 | 0.381±0.120 | 0.145±0.062 |
| 1.00 | 0.428±0.157 | 0.445±0.121 | 0.237±0.079 |

CurlyFM 在 25% 噪声水平下仍保持良好性能，表现出对参考速度场近似误差的鲁棒性。

**随机性消融（σ 值影响）**

| σ | W1 ↓ | W2 ↓ | Cosine Dist ↓ |
|---|------|------|---------------|
| 0.01 | 0.061±0.003 | 0.141±0.009 | 0.028±0.066 |
| 0.10 | 0.062±0.002 | 0.145±0.011 | 0.066±0.008 |
| 1.00 | 0.145±0.009 | 0.474±0.058 | 0.871±0.048 |

低随机性设置性能最佳，与应用场景中噪声水平低的特性一致。

### 关键发现

1. **传统 flow matching 的根本局限**：OT-CFM 的 cosine distance 接近 1.0，说明其轨迹几乎是纯直线，完全无法恢复旋转动态
2. **效率优势显著**：CurlyFM 是 simulation-free 方法，比需要模拟的 TrajectoryNet 和 SBIRR 快 1-2 个数量级
3. **多边际扩展**：CurlyFM 可自然扩展到多时间点设置，通过交替相邻边际对进行训练
4. **与 GSBM 对比**：CurlyFM 在 W1 和 cosine distance 上优于 GSBM（15 控制点），虽然计算代价略高

## 亮点与洞察

- **核心洞察**：通过在 Schrödinger Bridge 框架中引入非零漂移参考过程，打破了 flow matching 只能建模梯度场的限制
- **两阶段设计的优雅性**：先拟合参考漂移再求解传输问题，避免了迭代式算法的收敛困难
- **实用性强**：只需要在数据采样点处有速度信息（如 RNA velocity），通过核平滑即可构建连续参考场
- **Simulation-free**：相比需要反向传播或模拟的方法，训练效率提升数十倍

## 局限与展望

1. **依赖速度场质量**：性能取决于参考速度场的近似精度，在速度信息不可用的场景下无法直接应用
2. **确定性设为主**：主要实验在 σ=0 的确定性极限下进行，高随机性场景性能下降明显
3. **理论保证有限**：方法基于一系列实用近似（桥的布朗桥建模、mini-batch OT），缺乏严格的收敛性保证
4. **核选择**：核函数 κ 的选取对参考漂移质量有影响，但paper中缺少系统的核选择指导

## 相关工作与启发

- **Flow Matching (Lipman et al. 2023)**：Curly-FM 的基础框架
- **OT-CFM (Tong et al. 2024)**：mini-batch OT + CFM，但仅限梯度场
- **DSBM (Shi et al. 2024)**：Diffusion Schrödinger Bridge Matching，处理零漂移参考
- **GSBM (Liu et al. 2024)**：Generalized SBM，使用样条插值，但 Curly-FM 用神经网络更灵活
- **TrajectoryNet (Tong et al. 2020)**：基于 ODE 的轨迹推断，需要模拟，计算昂贵

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 — 非零漂移参考过程是关键创新 |
| 技术质量 | 4 — 方法合理，但部分理论细节需改进 |
| 实验充分性 | 4 — 多场景验证 + 充分消融 |
| 写作质量 | 3 — 方法描述（Section 3.1）较难理解 |
| 影响力 | 4 — 对单细胞生物学等领域有直接应用价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](../../ICML2025/computational_biology/improving_flow_matching_by_aligning_flow_divergence.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](../../ICML2025/computational_biology/flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)

</div>

<!-- RELATED:END -->
