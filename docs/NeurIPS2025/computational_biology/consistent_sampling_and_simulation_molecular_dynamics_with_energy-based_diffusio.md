---
title: >-
  [论文解读] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models
description: >-
  [NeurIPS 2025][计算生物][扩散模型] 本文发现扩散模型在采样和模拟之间存在不一致性问题（尤其在小扩散时间步），提出基于 Fokker-Planck 方程的正则化项来强制一致性，并结合时间分段的混合专家（MoE）策略，实现了在多个生物分子系统上一致且高效的采样与分子动力学模拟。 领域现状：分子动力学（MD）模拟…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "扩散模型"
  - "分子动力学模拟"
  - "Fokker-Planck方程"
  - "能量模型"
  - "粗粒化"
---

# Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.17139](https://arxiv.org/abs/2506.17139)  
**代码**: [https://github.com/noegroup/ScoreMD](https://github.com/noegroup/ScoreMD)  
**领域**: 分子动力学 / 扩散模型  
**关键词**: 扩散模型, 分子动力学模拟, Fokker-Planck方程, 能量模型, 粗粒化

## 一句话总结

本文发现扩散模型在采样和模拟之间存在不一致性问题（尤其在小扩散时间步），提出基于 Fokker-Planck 方程的正则化项来强制一致性，并结合时间分段的混合专家（MoE）策略，实现了在多个生物分子系统上一致且高效的采样与分子动力学模拟。

## 研究背景与动机

**领域现状**：分子动力学（MD）模拟是研究生物分子行为的基本工具，但达到生物学相关的时间尺度（μs-ms）计算代价极高。粗粒化（CG）方法通过降低系统维度来加速模拟，但需要学习粗粒化力场。扩散模型近年在分子生成和采样中表现出色，可以从平衡态分子分布中训练并生成新构型。

**现有痛点**：
   - 扩散模型学到的 score $\nabla_x \log p_t(x)$ 在 $t=0$ 时等价于力（$-\nabla U / k_BT$），理论上可以直接用于 MD 模拟
   - 但实践中，虽然经典扩散采样（iid）能正确恢复训练分布，但当使用同一模型的 score 进行 Langevin 模拟时，产生的分布与 iid 采样不一致
   - 即使在简单的二维玩具系统（如双高斯混合物）上也能观察到这种不一致性：iid 采样恢复两个模式，但模型能量在 $t=0$ 却学出了第三个错误模式
   - 先前工作 Two For One 通过在 $t>0$ 评估模型来缓解不一致性，但引入了额外噪声，降低了结构精度

**核心矛盾**：扩散模型的 score 在接近数据分布（$t \to 0$）时存在数值不稳定性，导致 Fokker-Planck 方程不被满足，从而使基于 score 的能量估计和力不准确

**本文解决什么**：
   - 如何使扩散模型的 iid 采样和 Langevin 模拟产生一致的分布？
   - 如何在保持采样质量的同时改善模拟准确性？
   - 如何高效训练能量一致的扩散模型？

**切入角度**：从 Fokker-Planck 方程出发，该方程描述了 score 应如何随扩散时间演化。如果模型不满足此方程，则其能量估计必然不一致。通过最小化 Fokker-Planck 残差来正则化训练。

**核心idea**：用 Fokker-Planck 方程推导的正则化项约束能量参数化的扩散模型，使其 score 在小 $t$ 处更加自洽，从而统一采样和模拟能力。

## 方法详解

### 整体框架

方法包含三个核心组件：

1. **保守能量参数化**：将 score 参数化为能量函数的梯度 $\nabla_x \log p_t^\theta = \nabla_x \text{NNET}(x, t)$
2. **Fokker-Planck 正则化**：添加残差损失确保模型满足 Fokker-Planck 方程
3. **混合专家（MoE）**：将扩散时间轴分段，用不同模型处理，仅对小 $t$ 区域施加正则化

输入：Boltzmann 分布的平衡态分子构型
输出：能同时用于 iid 采样和 Langevin 模拟的统一模型

### 关键设计

**保守 Score 参数化**：通常扩散模型直接参数化 score $s_\theta = \text{NN}(x, t)$，但这不保证场是保守的（无旋的）。本文使用能量参数化 $\nabla_x \log p_t^\theta = \nabla_x \text{NN}(x, t)$，确保 score 是某个能量函数的梯度。这不仅物理合理（分子力应为保守力），还使得 Fokker-Planck 正则化成为可能（需要评估 $\partial_t \log p_t^\theta$）。实验证明，保守参数化对模拟稳定性至关重要——非保守模型的模拟在数千步后发散。

**Fokker-Planck 正则化**：对于 VP-SDE 定义的扩散过程，Fokker-Planck 方程的对数形式为：

$$\partial_t \log p_t(x) = \frac{1}{2}g^2(t)[\text{div}_x(\nabla_x \log p_t) + \|\nabla_x \log p_t\|^2] - \langle f, \nabla_x \log p_t \rangle - \text{div}_x(f)$$

定义 Fokker-Planck 残差 $R(x,t) = \mathcal{F}_{p^\theta}(x,t) - \partial_t \log p_t^\theta(x)$，正则化损失为：

$$\mathcal{L}_{FP}[\log p^\theta](x,t) = \lambda_{FP}(t) D^{-2} \|R(x,t)\|^2$$

总训练目标结合标准 denoising score matching 和 FP 正则化：

$$\min_\theta \mathbb{E}_{t,x(0),x(t)}[\mathcal{L}_{DSM}[\nabla_x \log p^\theta](x(t), t) + \alpha \cdot \mathcal{L}_{FP}[\log p^\theta](x(t), t)]$$

**弱残差公式（Theorem 1）**：直接计算 FP 残差需要代价高昂的高阶导数（div 项需要 Hessian trace）。本文推导出一种弱残差估计器，仅需一阶导数：

$$\tilde{R}(x,t;v) = \frac{1}{2}g^2(t)\left[\left(\frac{v}{\sigma}\right)^\top \frac{s_\theta(x+v,t) - s_\theta(x-v,t)}{2\sigma} + \|s_\theta(x+v,t)\|^2\right] - \langle f(x+v,t), s_\theta(x+v,t)\rangle - \text{div}_x(f(x+v,t))$$

其中 $v \sim \mathcal{N}(0, \sigma^2 I)$，$\sigma = 0.0001$。时间导数 $\partial_t \log p_t^\theta$ 通过有限差分近似。

**混合专家（MoE）策略**：将扩散时间区间 $(0,1)$ 分为不相交子区间 $\mathcal{I}_0, \mathcal{I}_1, ...$，每个区间由独立专家处理：

$$s_\theta(x,t) = \sum_i w_i(t) s_i^\theta(x,t), \quad \sum_i w_i(t) = 1$$

典型配置：$(0, 0.1)$, $[0.1, 0.6)$, $[0.6, 1.0)$。只有小 $t$ 区间的模型使用保守参数化和 FP 正则化，大 $t$ 区间使用更简单的模型。这有两个优势：
1. 避免过度正则化——大 $t$ 模型不需要精确的力
2. 降低计算成本——仅对关键区域使用昂贵的保守模型

**物理约束的架构**：使用图Transformer 架构，输入节点特征为原子类型和扩散时间，边特征为原子间距离向量。通过使用成对距离（而非绝对坐标）实现平移不变性，通过训练时随机旋转实现学习的旋转等变性。最后通过标量能量映射 $\psi: \mathbb{R}^K \to \mathbb{R}$ 确保守恒性。

### 损失函数 / 训练策略

完整训练目标（以 Both 模型为例）：
- 小 $t$ 专家 $[0, 0.1)$：$\mathcal{L}_{DSM} + \alpha \cdot \mathcal{L}_{FP}$（保守参数化，$\alpha = 0.0001$）
- 中 $t$ 专家 $[0.1, 0.6)$：$\mathcal{L}_{DSM}$（非保守，更简单架构）
- 大 $t$ 专家 $[0.6, 1.0)$：$\mathcal{L}_{DSM}$（非保守，最简架构）

VP-SDE 参数：$\beta_{min} = 0.1$, $\beta_{max} = 20$。优化器：AdamW。

## 实验关键数据

### 主实验

**丙氨酸二肽（Alanine Dipeptide）— 5 原子粗粒化**：

| 方法 | iid JS ↓ | sim JS ↓ | iid PMF ↓ | sim PMF ↓ |
|------|---------|---------|----------|----------|
| Diffusion | 0.0081±0.0003 | 0.0695±0.0517 | 0.095±0.003 | 1.047±0.924 |
| Two For One | 0.0081±0.0003 | 0.0158±0.0002 | 0.098±0.006 | 0.206±0.004 |
| Mixture | 0.0080±0.0004 | 0.0353±0.0117 | 0.092±0.007 | 0.388±0.109 |
| Fokker-Planck | 0.0084±0.0002 | 0.0088±0.0006 | 0.098±0.006 | 0.105±0.011 |
| **Both** | **0.0079±0.0002** | **0.0086±0.0004** | **0.089±0.005** | **0.099±0.003** |

关键观察：标准 Diffusion 的 sim JS（0.0695）比 iid JS（0.0081）高一个数量级，证实不一致性。Both 将 sim JS 降至 0.0086，接近 iid 水平。

**跨二肽可迁移模型（400 种二肽，10 原子粗粒化）**：

| 方法 | iid JS ↓ | sim JS ↓ | iid PMF ↓ | sim PMF ↓ |
|------|---------|---------|----------|----------|
| Transferable BG | 0.0183±0.0070 | - | 0.230±0.119 | - |
| Diffusion | 0.0155±0.0083 | 0.2256±0.1304 | 0.206±0.159 | 6.515±3.175 |
| Two For One | 0.0153±0.0080 | 0.0466±0.0114 | 0.203±0.149 | 0.741±0.319 |
| Mixture | 0.0155±0.0078 | 0.0444±0.0237 | 0.200±0.127 | 0.658±0.407 |
| Fokker-Planck | 0.0154±0.0060 | 0.0200±0.0106 | 0.192±0.118 | 0.290±0.222 |
| **Both** | **0.0158±0.0077** | **0.0158±0.0052** | **0.197±0.124** | **0.183±0.070** |

Both 模型实现了几乎完美的 iid-sim 一致性（二者的 JS 散度相同）。

### 消融实验

**Müller-Brown 势（2D 玩具系统）**：

| 方法 | iid JS ↓ | sim JS ↓ | iid PMF ↓ | sim PMF ↓ |
|------|---------|---------|----------|----------|
| Reference | - | - | 0.0119±0.0004 | 0.087±0.002 |
| Diffusion | 0.0122±0.0013 | 0.0448±0.0125 | 0.111±0.006 | 0.504±0.150 |
| Mixture | 0.0109±0.0007 | 0.0254±0.0109 | 0.097±0.004 | 0.247±0.113 |
| Fokker-Planck | 0.0130±0.0010 | 0.0166±0.0009 | 0.122±0.006 | 0.163±0.008 |
| **Both** | **0.0110±0.0007** | **0.0108±0.0008** | **0.098±0.003** | **0.099±0.004** |

**保守 vs 非保守参数化**（丙氨酸二肽）：
- 保守模型：iid 采样略优，模拟稳定
- 非保守模型：iid 采样相近，但模拟在数千步后发散——力场不保守导致系统能量不守恒

**C-N 键长的 Wasserstein 距离**：

| 方法 | iid 相对 W1 ↓ | sim 相对 W1 ↓ |
|------|-------------|-------------|
| Diffusion | 1.51±1.28 | 1.70±0.38 |
| Two For One | 0.96±0.34 | **48.92±11.25** |
| Mixture | 1.36±0.21 | 0.94±0.21 |
| Fokker-Planck | 2.05±0.62 | 2.51±0.59 |
| **Both** | **1.00±0.00** | **1.00±0.00** |

Two For One 的 sim W1 高达 48.92——因为在 $t>0$ 评估引入了过多噪声，严重破坏键长结构。

**运行时间对比**：

| 系统 | 阶段 | Diffusion | Mixture | Fokker-Planck | Both |
|------|------|-----------|---------|---------------|------|
| 丙氨酸二肽 | 训练 | 49min | 50min | 4h 39min | 3h 59min |
| 丙氨酸二肽 | 推理 | 3min | 4min | 3min | 4min |
| 二肽 | 训练 | 4h 5min | 3h 50min | 28h 39min | 27h 5min |
| 二肽 | 推理 | 8min | **4min** | 8min | **4min** |

MoE 将推理时间减半（仅需运行小模型进行模拟），但 FP 正则化显著增加训练时间。

### 关键发现

1. **不一致性的根源确认**：Fokker-Planck 残差在 $t \to 0$ 时最大，与理论预测一致
2. **FP 正则化和 MoE 通过不同机制改善一致性**：FP 直接减小方程残差，MoE 让模型专注于关键区间——两者组合效果最佳
3. **Two For One 的结构精度代价**：通过在 $t>0$ 评估确实改善了模式覆盖，但引入的噪声严重破坏了结构特征（键长 W1 高 48 倍）
4. **可迁移性验证**：在 400 种二肽上训练的单一模型实现了跨二肽的一致采样和模拟

## 亮点与洞察

- **从理论到实践的完整闭环**：从 Fokker-Planck 方程出发识别问题，推导弱残差公式降低计算成本，通过 MoE 选择性应用正则化，最终在多个分子系统上验证
- **弱残差公式的工程意义**：将高阶导数（Hessian trace）降为一阶导数，使正则化在高维分子系统上可行
- **MoE 的双重作用**：不仅提高效率，还防止大 $t$ 区域过度正则化导致 iid 采样退化
- **采样与模拟的统一**：同一模型可以同时提供热力学（iid 采样）和动力学（Langevin 模拟）信息

## 局限与展望

- 目前仅在粗粒化小分子（≤10 原子）上验证，大蛋白质系统的可扩展性待验证
- FP 正则化增加了约 6 倍的训练时间（弱残差公式仍需多次前向传播）
- 不能保证完美一致性——由于扩散采样和 Langevin 模拟的根本差异，完美对齐可能需要限制模型表达能力
- MoE 模型在区间边界处引入不连续性，虽然实验中影响不大
- 需要已知的 Boltzmann 分布样本进行训练，不能直接用于未知势函数

## 相关工作与启发

- **Two For One (Arts et al., 2023)**：最直接的前身，将扩散模型用于粗粒化 MD，但通过在 $t>0$ 评估来缓解不一致性
- **力匹配方法（Husic et al., Charron et al.）**：直接学习力场而非分布，但需要力标签
- **Transferable Boltzmann Generator (Klein & Noé, 2024)**：可迁移的 Boltzmann 生成器，支持独立采样但不支持模拟
- **FP-Diffusion (Lai et al., 2023)**：同样使用 FP 正则化，但目标是改善 iid 采样质量，而非模拟一致性
- **AlphaFold 3 / RFDiffusion**：大规模蛋白质结构预测模型，证明不一定需要等变架构

## 评分

- **新颖性**: ⭐⭐⭐⭐ — FP 正则化 + MoE 的组合有理论深度和实践价值
- **技术深度**: ⭐⭐⭐⭐⭐ — 从 SDE 理论到弱形式推导到实验验证，技术链条完整
- **实验充分度**: ⭐⭐⭐⭐ — 玩具系统+丙氨酸二肽+跨二肽迁移，层层递进
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，实验结果呈现系统
- **实用性**: ⭐⭐⭐⭐ — 代码开源（JAX/PyTorch），对计算化学社区有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](energy_loss_functions_for_physical_systems.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)

</div>

<!-- RELATED:END -->
