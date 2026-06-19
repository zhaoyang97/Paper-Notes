---
title: >-
  [论文解读] Inferring Stochastic Dynamics with Growth from Cross-Sectional Data
description: >-
  [NeurIPS 2025][计算生物][概率流推断] 提出非平衡概率流推断（UPFI），通过Fokker-Planck方程的Lagrangian形式化，从横截面数据中联合推断随机动力学系统的漂移项、扩散项和增长率，首次准确处理含细胞增殖/死亡的场景。 单细胞测序（scRNA-seq）技术是破坏性的——每个细胞只能在一个时间…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "概率流推断"
  - "Fokker-Planck方程"
  - "细胞动力学"
  - "最优传输"
  - "分支扩散过程"
---

# Inferring Stochastic Dynamics with Growth from Cross-Sectional Data

**会议**: NeurIPS 2025  
**arXiv**: [2505.13197](https://arxiv.org/abs/2505.13197)  
**代码**: 无  
**领域**: 计算生物学 / 随机动力学推断  
**关键词**: 概率流推断, Fokker-Planck方程, 细胞动力学, 最优传输, 分支扩散过程

## 一句话总结

提出非平衡概率流推断（UPFI），通过Fokker-Planck方程的Lagrangian形式化，从横截面数据中联合推断随机动力学系统的漂移项、扩散项和增长率，首次准确处理含细胞增殖/死亡的场景。

## 研究背景与动机

单细胞测序（scRNA-seq）技术是破坏性的——每个细胞只能在一个时间点测量，因此只能获取不同时间点的种群横截面快照。从这些快照重建底层动力学系统是一个核心逆问题。

现有方法的局限：

- 大多数方法假设无噪声或各向同性常数扩散率，且不处理细胞增殖/死亡
- 忽略增殖/死亡会导致推断出错误的状态转移（例如将凋亡细胞错误连接到多能性细胞）
- DeepRUOT 等方法虽考虑了增长，但训练过程多阶段且不稳定
- 原始PFI方法不处理细胞质量变化（不平衡传输）

关键挑战：漂移（drift）和增长率（fitness）的可辨识性问题——同一组分布快照可能由不同的漂移+增长组合产生。

## 方法详解

### 整体框架

UPFI采用两步训练方案：

1. **离线得分匹配**：用去噪得分匹配从快照数据估计时间依赖的得分函数 $\mathbf{s}_t(\mathbf{x}) \approx \nabla \log \rho_t(\mathbf{x})$
2. **在线ODE拟合**：在Lagrangian参考系下，学习漂移 $\mathbf{v}_t$ 和增长率 $g_t$，使得推送后的分布与观测匹配

核心洞察：含增长的Fokker-Planck方程可改写为 $d+1$ 维ODE系统（位置 + 质量），避免了高维PDE求解。

### 关键设计

1. **Lagrangian形式化与质量方程**:
    - 功能：将含源项的FPE转化为特征线ODE系统，其中位置演化包含概率流速度（漂移 - 散度修正 - 得分项），质量沿特征线以增长率指数增长
    - 核心思路：$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_t - \nabla \cdot \mathbf{D}_t - \mathbf{D}_t \nabla \log \rho_t$，$\frac{dm_t}{dt} = g_t(\mathbf{x}_t) m_t$
    - 设计动机：Lagrangian视角将PDE降为ODE，得分函数独立于动力学参数，可离线预计算

2. **非平衡Sinkhorn散度作为损失**:
    - 功能：选择非平衡Sinkhorn散度 $S_{\varepsilon,\gamma}$ 度量推送分布和观测分布的差异
    - 核心思路：Sinkhorn散度直接作用于离散测度，允许质量不守恒，无需计算密度
    - 设计动机：传统Wasserstein距离要求质量守恒，不适用于含增长系统；Sinkhorn散度有好的几何和计算性质

3. **Wasserstein-Fisher-Rao正则化解决可辨识性**:
    - 功能：添加 $\lambda \int (\|\mathbf{v}_t\|^2 + \alpha |g_t|^2) d\rho_t dt$ 正则项
    - 核心思路：对Ornstein-Uhlenbeck过程分析表明，漂移和增长不可唯一辨识（Corollary 2.2），正则化保证唯一解
    - 设计动机：Proposition 2.1证明即使限制为自治漂移，漂移矩阵的对称和反对称部分都可与增长混淆

### 损失函数 / 训练策略

总损失：$L = \sum_{i=1}^K S_{\varepsilon,\gamma}(\hat{\rho}_{t_i}, \rho_{t_i}) + \lambda(t_i - t_{i-1}) \int_{t_{i-1}}^{t_i} \int (\|\mathbf{v}_t\|^2 + \alpha |g_t|^2) d\rho_t dt$

Theorem 2.3 证明在连续时间极限下，此损失对OU过程有唯一最小值。实际训练中ODE用2-3个Euler步即可。

## 实验关键数据

### 主实验（表格）

双稳态系统上的Path Energy Distance（越低越好）：

| 维度 $d$ | UPFI | PFI | fitness-ODE | TIGON++ | DeepRUOT | OTFM | UOTFM |
|-----------|------|-----|-------------|---------|----------|------|-------|
| 2 | **0.14±0.09** | 1.41±0.16 | 0.30±0.18 | 0.46±0.12 | 2.15±0.01 | 1.16±0.13 | 0.42±0.13 |
| 5 | **0.04±0.03** | 1.34±0.06 | 0.30±0.14 | 0.63±0.16 | 0.47±0.04 | 1.07±0.11 | 0.36±0.10 |
| 10 | **0.05±0.04** | 1.03±0.18 | 0.29±0.15 | 0.61±0.06 | 1.32±0.05 | 1.09±0.19 | 0.38±0.08 |
| 50 | **0.15±0.02** | — | — | — | — | — | — |

### 消融实验

- 不处理增长的PFI在双稳态系统上推断出错误的跨分支流线（误差高10倍以上）
- OU过程验证：在已知解析解的线性-二次场景下验证了UPFI的正确性
- 正则化强度 $\lambda$ 对漂移-增长分离的影响：过小则不可辨识，过大则欠拟合

### 关键发现

- UPFI在所有维度和所有对比方法中一致最优或接近最优
- 不考虑增长的PFI会系统性产生错误的粒子流向（将两个分支错误连接）
- 计算复杂度：Sinkhorn散度 $O(B^2)$，得分匹配 $O(Bd)$，可处理中等高维
- 在真实scRNA-seq数据（造血干细胞分化）上也展现了良好效果

## 亮点与洞察

- **理论贡献**：Proposition 2.1和Corollary 2.2首次形式化了漂移-增长不可辨识性问题
- **简洁架构**：两步训练（得分匹配 + ODE拟合），相比DeepRUOT等多阶段方法更稳定
- **物理正当性**：Lagrangian形式化保留了物理可解释性，推断出的漂移场和增长率有生物学意义
- Theorem 2.3证明了正则化训练在OU情形下有唯一解

## 局限与展望

- 漂移和增长的可辨识性在非线性情形下未完全解决，正则化引入了归纳偏置
- 得分匹配在高维稀疏数据上可能不准确
- 假设扩散系数 $\mathbf{D}_t$ 已知，实际中通常未知
- 计算扩展性受限于ODE积分和Sinkhorn散度的 $O(B^2)$ 复杂度
- 未处理批次效应等实际数据挑战

## 相关工作与启发

- **PFI (Zhang & Chardès 2023)**：本文的直接前驱，但不处理增长
- **Waddington-OT (Schiebinger et al. 2019)**：用非平衡最优传输处理增长，但不推断漂移
- **TIGON++ / fitness-ODE**：处理增长但假设确定性动力或全局fitness
- 连接了随机分析、最优传输和生物动力学推断三个领域

## 评分

⭐⭐⭐⭐ — 理论扎实，方法简洁有效，首次系统解决了含增长的随机动力学推断问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)
- [\[NeurIPS 2025\] Evaluating Multiple Models Using Labeled and Unlabeled Data](evaluating_multiple_models_using_labeled_and_unlabeled_data.md)
- [\[NeurIPS 2025\] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)

</div>

<!-- RELATED:END -->
