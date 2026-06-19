---
title: >-
  [论文解读] UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation
description: >-
  [CVPR 2025][逆物理仿真] 提出 UniPhy，首个统一的潜变量条件本构模型，在共享潜空间中编码弹性体/沙子/塑料/牛顿/非牛顿流体等多种材料属性，推理时通过可微 MPM 仿真器优化潜变量以匹配观测粒子轨迹，重建误差比 NCLaw 低 1-2 个数量级。 领域现状：逆物理仿真从观测数据推断材料属性（如弹性模量、粘度…
tags:
  - "CVPR 2025"
  - "逆物理仿真"
  - "本构模型"
  - "潜变量优化"
  - "MPM仿真"
  - "材料参数估计"
---

# UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation

**会议**: CVPR 2025  
**arXiv**: [2505.16971](https://arxiv.org/abs/2505.16971)  
**代码**: [https://himangim.github.io/UniPhy](https://himangim.github.io/UniPhy)  
**领域**: 其他 / 物理仿真  
**关键词**: 逆物理仿真, 本构模型, 潜变量优化, MPM仿真, 材料参数估计

## 一句话总结

提出 UniPhy，首个统一的潜变量条件本构模型，在共享潜空间中编码弹性体/沙子/塑料/牛顿/非牛顿流体等多种材料属性，推理时通过可微 MPM 仿真器优化潜变量以匹配观测粒子轨迹，重建误差比 NCLaw 低 1-2 个数量级。

## 研究背景与动机

**领域现状**：逆物理仿真从观测数据推断材料属性（如弹性模量、粘度）。现有方法要么为每种材料设计专用模型（如弹性用 Neo-Hookean、流体用 Navier-Stokes），要么用 NCLaw 等学习方法但需要预先指定材料类型。

**现有痛点**：真实场景中材料类型往往未知——一团看起来像软体的东西可能是弹性体也可能是塑料。现有方法需要先分类再推断属性，两步都可能出错。

**核心矛盾**：不同材料的物理行为差异巨大（弹性体回弹 vs 沙子崩塌 vs 流体流动），用一个模型统一表示似乎不切实际。

**切入角度**：将材料属性编码为连续潜向量——不同材料对应潜空间中的不同区域，但共享相同的神经网络架构。推理时只需优化潜向量就能自动"发现"材料类型。

**核心 idea**：潜变量条件的变形投影 $g_\phi(F,z)$ + 本构律 $f_\theta(F_{proj},z)$ = 统一的多材料物理模型。

## 方法详解

### 关键设计

1. **潜变量条件的双网络架构**:

    - 功能：统一编码不同材料的物理行为
    - 核心思路：两个神经网络：变形梯度投影函数 $g_\phi(F,z)$ 将变形梯度投影到材料相关的子空间（如弹性体投影到体积保持流形），本构律函数 $f_\theta(F_{proj},z)$ 从投影后的变形计算应力。潜向量 $z$ 编码材料属性
    - 设计动机：分离"变形如何分解"和"应力如何计算"——不同材料的变形投影方式不同（弹性保体积、沙子有屈服面），但应力计算的网络结构可以共享

2. **通过可微 MPM 优化潜变量**:

    - 功能：从观测粒子轨迹推断材料属性
    - 核心思路：将预训练的 $g_\phi, f_\theta$ 嵌入可微物质点法（MPM）仿真器，固定网络参数，只优化潜向量 $z$ 使仿真轨迹匹配观测。L2 距离作为损失
    - 设计动机：通过仿真器的梯度流回到潜空间，实现"观测→材料"的端到端推断

### 损失函数 / 训练策略

训练：$L = \sum_n \sum_t \sum_p (L(F_{proj}, \hat{F}_{proj}) + L(S, \hat{S}) + \frac{1}{\sigma^2}\|z_n\|^2)$，含变形投影误差 + 应力误差 + 潜变量正则化。推理：L2 粒子位置误差。32 维潜空间最优。

## 实验关键数据

### 主实验

重建误差（×10⁻⁵）：

| 材料 | UniPhy | NCLaw |
|------|--------|-------|
| 弹性体 | **0.052** | 2.40 |
| 沙子 | **1.50** | 2.60 |
| 塑料 | **3.90** | 6.50 |
| 牛顿流体 | **0.011** | 2.00 |

### 消融实验

| 配置 | 弹性体误差 |
|------|-----------|
| UniPhy (w/ teacher forcing) | 0.052 |
| UniPhy (w/o TF) | 1.10 |
| NCLaw (w/o TF) | 36.0 |
| 潜空间 4 维 | 7.80 |
| 潜空间 32 维 | **1.10** |
| 潜空间 256 维 | 1.50 |

### 关键发现
- UniPhy 比 NCLaw 低 1-2 个数量级——统一模型不仅可行，还比专用模型更好
- 32 维潜空间是最佳平衡——太小无法区分材料，太大过拟合
- 优化后的潜变量确实反映了材料类型——不同材料聚类到不同区域

## 亮点与洞察
- **统一 > 专用的反直觉结果**——一个模型处理所有材料比为每种设计专用模型更好，可能因为跨材料的共享知识（如连续体力学的通用结构）
- **潜向量 = 材料指纹**——不需要预先分类材料类型，优化后的潜向量自动编码了材料属性

## 局限与展望
- 仅支持均匀材料（无多材料混合）
- 需要已知初始几何和 3D 运动观测
- 仅在仿真数据上训练，真实世界泛化未验证

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个统一多材料本构模型
- 实验充分度: ⭐⭐⭐⭐ 多材料类型，泛化测试充分
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 为逆物理仿真提供了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](../../CVPR2026/others/physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [\[CVPR 2025\] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)
- [\[CVPR 2025\] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)
- [\[CVPR 2025\] Uncertainty Weighted Gradients for Model Calibration](uncertainty_weighted_gradients_for_model_calibration.md)
- [\[NeurIPS 2025\] Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free](../../NeurIPS2025/others/learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)

</div>

<!-- RELATED:END -->
