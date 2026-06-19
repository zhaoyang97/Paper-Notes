---
title: >-
  [论文解读] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling
description: >-
  [NeurIPS 2025][计算生物][能量模型] 提出 Energy Matching，通过学习一个时间无关的标量势能场统一流匹配与能量模型：远离数据流形时沿最优传输路径高效传输，靠近流形时过渡为 Boltzmann 平衡分布以建模似然，在 CIFAR-10 上 FID 3.34 大幅超越现有 EBM（>50%提升）。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "能量模型"
  - "流匹配"
  - "最优传输"
  - "Boltzmann分布"
  - "逆问题"
  - "局部内在维度"
---

# Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2504.10612](https://arxiv.org/abs/2504.10612)  
**代码**: [GitHub](https://github.com/m1balcerak/EnergyMatching)  
**领域**: 图像生成  
**关键词**: 能量模型, 流匹配, 最优传输, Boltzmann分布, 逆问题, 局部内在维度

## 一句话总结

提出 Energy Matching，通过学习一个时间无关的标量势能场统一流匹配与能量模型：远离数据流形时沿最优传输路径高效传输，靠近流形时过渡为 Boltzmann 平衡分布以建模似然，在 CIFAR-10 上 FID 3.34 大幅超越现有 EBM（>50%提升）。

## 研究背景与动机

**流匹配/扩散模型的局限**：当前 SOTA 生成模型（流匹配、扩散）将噪声映射到数据分布，但不直接捕获数据似然——无法在数据流形上自由导航，难以自然融合额外观测和先验（如逆问题中的测量似然）

**传统 EBM 的困境**：能量模型通过标量函数 $E(x)$ 定义未归一化密度 $p(x) \propto \exp(-E(x))$，理论优美但实际生成质量差——MCMC 训练在高维空间难以充分探索能量景观，导致模式崩塌和不稳定

**现有改进方案复杂**：为提升 EBM 性能，现有方法依赖时间条件集成、分层潜在集成、或配合独立生成器协作训练——参数量大、训练复杂

**核心 idea**：用一个时间无关标量场同时实现两个目标：(a) 远处做最优传输高效采样；(b) 近处做 Boltzmann 密度建模保留似然信息

## 方法详解

### 理论基础：JKO 方案

方法建立在 Jordan-Kinderlehrer-Otto (JKO) 方案上，描述概率分布在 Wasserstein 空间中沿能量最小化轨迹的离散时间演化：

$$\rho_{t+\Delta t} = \arg\min_{\rho} \underbrace{\frac{W_2^2(\rho, \rho_t)}{2\Delta t}}_{\text{传输代价}} + \underbrace{\int V_\theta(x) \mathrm{d}\rho(x)}_{\text{势能}} + \underbrace{\varepsilon(t) \int \rho(x) \log \rho(x) \mathrm{d}x}_{\text{内能（-熵）}}$$

其中 $V_\theta(x)$ 是可学习标量势能，$\varepsilon(t)$ 是时间相关的温度参数。

### 两阶段机制

通过分析一阶最优性条件，揭示两种行为模式：

**regime 1**（远离数据流形，$t < \tau^*$）：$\varepsilon(t) = 0$，退化为最优传输：
$$\frac{1}{\Delta t}(x - y) + \nabla_x V_\theta(x) = 0$$
系统沿确定性 OT 路径高效传输。

**regime 2**（靠近数据流形，$t \geq 1$）：$\varepsilon(t) = \varepsilon_{\max}$，平衡分布服从 Boltzmann：
$$\rho_{\text{eq}}(x) \propto \exp\left(-\frac{V_\theta(x)}{\varepsilon_{\max}}\right)$$
系统进入 EBM 模式，精确建模数据密度。

温度调度采用线性方案：
$$\varepsilon(t) = \begin{cases} 0, & 0 \leq t < \tau^* \\ \varepsilon_{\max} \frac{t - \tau^*}{1 - \tau^*}, & \tau^* \leq t < 1 \\ \varepsilon_{\max}, & t \geq 1 \end{cases}$$

### 训练目标

**Phase 1 (warm-up): 流式目标 $\mathcal{L}_{\text{OT}}$**

计算 mini-batch 间的 OT 耦合 $\gamma^*$，沿测地线插值 $x_t = (1-t)T(x_{\text{data}}) + t \cdot x_{\text{data}}$，要求梯度场近似速度场：

$$\mathcal{L}_{\text{OT}} = \mathbb{E}_{x_{\text{data}} \in \mathcal{D}, t \sim U(0, \tau^*)} \left[\|\nabla_x V_\theta(x_t) + x_{\text{data}} - T(x_{\text{data}})\|^2\right]$$

这等价于施加无旋条件（速度场为标量势能梯度）的流匹配，与 OT 天然对齐。

**Phase 2 (main): 对比目标 $\mathcal{L}_{\text{CD}}$**

在数据流形附近精调 $V_\theta$ 使 Boltzmann 分布匹配数据分布：

$$\mathcal{L}_{\text{CD}} = \mathbb{E}_{x \sim p_{\text{data}}}\left[\frac{V_\theta(x)}{\varepsilon_{\max}}\right] - \mathbb{E}_{\tilde{x} \sim \text{sg}(p_{\text{eq}})}\left[\frac{V_\theta(\tilde{x})}{\varepsilon_{\max}}\right]$$

负样本通过 Langevin 链近似生成（一半从真实数据初始化，一半从噪声初始化）。联合优化 $\mathcal{L} = \mathcal{L}_{\text{OT}} + \lambda_{\text{CD}} \mathcal{L}_{\text{CD}}$。

### 逆问题求解

给定测量 $y = A(x) + w$，后验分布分解为：
$$p(x|y) \propto \underbrace{\exp\left(-\frac{\|y - A(x)\|^2}{\zeta^2}\right)}_{p(y|x)} \underbrace{\exp\left(-E_\theta(x)\right)}_{p(x)}$$

直接用学到的 $E_\theta(x) = V_\theta(x)/\varepsilon_{\max}$ 作为先验，加上测量保真项即可 Langevin 采样——无需在噪声/数据分布间来回穿梭。可进一步引入交互能 $W(x_1, x_2)$ 鼓励多样重建。

## 实验关键数据

### 无条件生成（FID↓）

**CIFAR-10**：

| 方法 | 类别 | 参数量 | FID↓ |
|------|------|--------|------|
| ImprovedCD | EBM | - | 25.1 |
| CLEL-large | EBM | 32M | 8.61 |
| Cooperative DRL-large | 集成 | 145M | 3.68 |
| DDPM | 扩散 | - | 6.45 |
| NCSN++ | 扩散 | 107M | 2.45 |
| OT-CFM | 流 | 37M | 4.04 |
| **Energy Matching** | **EBM** | **50M** | **3.34** |

**ImageNet 32×32**：

| 方法 | 类别 | FID↓ |
|------|------|------|
| ImprovedCD | EBM | 32.48 |
| CLEL-large | EBM | 15.47 |
| Cooperative DRL | 集成 | 9.35 |
| DDPM++ | 扩散 | 8.42 |
| Flow-matching | 流 | 5.02 |
| **Energy Matching** | **EBM** | **6.64** |

- CIFAR-10 上 FID 3.34，**超越所有现有 EBM 50%以上**
- 50M 参数即超过 145M 参数的 Cooperative DRL-large（3.34 vs 3.68）
- ImageNet 上 FID 6.64，首次使 EBM 接近流模型水平

### 局部内在维度（LID）估计

Spearman 相关系数 vs PNG 压缩率（4096 张图）：

| 方法 | MNIST | CIFAR-10 |
|------|-------|----------|
| ESS | 0.444 | 0.326 |
| FLIPD | 0.837 | 0.819 |
| NB (扩散) | 0.864 | 0.894 |
| **Energy Matching** | **0.877** | **0.901** |

直接在数据流形上计算 $\nabla_x^2 V(x_{\text{data}})$ 的 Hessian 谱，近零特征值数量反映局部维度——比扩散模型更准确（无需近似）。

### 蛋白质逆向设计

在 AAV 壳蛋白分段的适应性-多样性权衡中：通过可调斥力项 $\propto 1/\sigma^2$ 显式控制生成多样性，在 Medium 和 Hard 基准上实现比流模型和分数模型更好的 Pareto 前沿。

## 亮点

- ⭐⭐⭐⭐⭐ **理论统一**：首次严格统一 OT 流匹配和 EBM，从 JKO 方案自然推导出两阶段机制
- ⭐⭐⭐⭐⭐ **极简设计**：单个时间无关标量场，无辅助生成器、无时间条件、无额外网络
- ⭐⭐⭐⭐ **性能突破**：EBM 类别 CIFAR-10 FID 3.34，首次接近扩散/流模型水平
- ⭐⭐⭐⭐ **多功能性**：同一模型可做生成、逆问题求解、LID 估计——都源于显式似然建模
- ⭐⭐⭐ **蛋白质设计**：交互能项轻松扩展到受控蛋白质生成，展示方法的灵活性

## 局限与展望

1. **分辨率有限**：实验仅在 CIFAR-10 (32×32) 和 ImageNet 32×32 上验证，未扩展到高分辨率
2. **采样步数多**：采样需要 325 步 Euler-Heun，比一些加速流匹配方法慢
3. **MCMC 开销**：Phase 2 的 Langevin 链负样本生成增加训练成本
4. **绝对 FID 仍有差距**：虽然大幅超越 EBM，但绝对 FID（3.34）仍不及最优扩散模型（NCSN++ 2.45）
5. **超参数敏感**：$\tau^*$、$\varepsilon_{\max}$、$\lambda_{\text{CD}}$ 等超参数需要针对每个数据集调节

## 总体评价 ⭐⭐⭐⭐⭐

非常优雅的理论工作，从 JKO 方案出发自然统一了两大生成模型范式。用一个标量场同时实现高效传输和密度建模的 idea 简洁而深刻。虽然目前在绝对生成质量上仍有提升空间，但为 EBM 开辟了全新方向——显式似然+逆问题灵活性+LID 估计一举三得。对推动 EBM 在更多领域的应用具有重要意义。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](energy_loss_functions_for_physical_systems.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](../../ICML2025/computational_biology/improving_flow_matching_by_aligning_flow_divergence.md)

</div>

<!-- RELATED:END -->
