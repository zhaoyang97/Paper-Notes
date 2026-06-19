---
title: >-
  [论文解读] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion
description: >-
  [NeurIPS 2025][人体理解][人体-物体交互] 将人体-物体交互（HOI）生成建模为 Driver-Responder 系统，通过轻量级 Transformer 交互动力学模型显式预测物体对人体动作的响应，利用残差动力学损失在训练时增强因果一致性，同时保持推理效率。 生成逼真的 3D 人体-物体交互（HOI）是…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "人体-物体交互"
  - "运动扩散"
  - "交互动力学"
  - "Driver-Responder"
  - "Transformer"
---

# HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion

**会议**: NeurIPS 2025  
**arXiv**: [2507.01737](https://arxiv.org/abs/2507.01737)  
**代码**: [有](https://wulin97.github.io/hoi-dyn)  
**领域**: 人体理解  
**关键词**: 人体-物体交互, 运动扩散, 交互动力学, Driver-Responder, Transformer

## 一句话总结

将人体-物体交互（HOI）生成建模为 Driver-Responder 系统，通过轻量级 Transformer 交互动力学模型显式预测物体对人体动作的响应，利用残差动力学损失在训练时增强因果一致性，同时保持推理效率。

## 研究背景与动机

生成逼真的 3D 人体-物体交互（HOI）是 VR/AR、计算机动画和机器人领域的重要问题。现有方法主要存在以下局限：

**独立建模问题**：大多数方法将人体运动和物体运动独立处理，导致物理不合理和因果不一致的行为

**接触建模困难**：一些方法聚焦于物体 affordance 或接触点预测，但精确建模接触区域本质上非常困难

**缺乏因果关系**：现有扩散模型虽然能生成全局合理的序列，但未捕捉物体如何响应人体动作的细粒度动力学

作者提出一个关键观察：HOI 本质上是一个**非对称系统**——人体运动遵循内部引导的动力学（自主运动），而物体运动是外部驱动的（不能自主发生）。这种不对称性自然引出了 Driver-Responder 的建模范式。

## 方法详解

### 整体框架

HOI-Dyn 由两个核心组件构成：

1. **Conditional Motion Diffusion**：基于 Transformer 的条件扩散模型，联合编码人体、物体和交互上下文
2. **Interaction Dynamics**：辅助监督模块，在训练时强化细粒度因果一致性

关键设计思想是 **Driver-Responder 形式化**：

$$\text{Driver (Human)}: h^{(t+1)} = h^{(t)} + \Delta t \cdot F_h(h^{(t)})$$
$$\text{Responder (Object)}: o^{(t+1)} = o^{(t)} + \Delta t \cdot F_o(o^{(t)}, s^{(t)}, u^{(t)})$$

其中 $u^{(t)}$ 是控制信号（基于人体意图与物体行为的误差反馈），$s^{(t)}$ 是交互上下文（接触状态、物体几何等）。

### 关键设计

#### 1. 交互动力学模型

核心思想：物体的相对运动可以用人体的相对运动来预测：

$$\Delta o^{(t)} \approx \mathcal{D}(s^{(t)}, o^{(t)}, \Delta h^{(t)}; \theta_\mathcal{D})$$

为增强对不同交互幅度的敏感性，将预测范围从 1 扩展到 $k$，其中 $k$ 从 $[1, K]$ 随机采样：

$$\Delta o_{t \to t+k}^* \approx \mathcal{D}(s^{(t)}, o^{(t)}, \Delta h_{t \to t+k}; \theta_\mathcal{D})$$

预测结果表示为刚体变换（旋转 $\hat{\mathcal{R}} \in SO(3)$ + 平移 $\hat{\mathcal{T}} \in \mathbb{R}^3$），通过 SVD 投影确保旋转矩阵有效性。

#### 2. 物体动力学代价函数

基于关键点变换误差：

$$\Phi(\Delta o, \Delta o^*) = \|\mathcal{P}^{(t+k)} - \hat{\mathcal{P}}^{(t+k)}\|_1$$

#### 3. 隐式接触处理

一个优雅的设计：**无需显式建模接触**。如果没有接触就没有响应；如果有接触，物体响应由交互动力学自然决定。

#### 4. 网络架构

交互动力学模型采用轻量级 Transformer（仅 0.5M 参数），输入为当前物体状态、交互上下文和累积人体运动，输出物体运动的刚体变换。采用耦合设计（rotation + translation 联合预测）优于解耦设计。

### 损失函数 / 训练策略

**两阶段训练**：

1. **阶段一**：预训练交互动力学模型 $\mathcal{D}$，使用动力学损失：
$$\mathcal{L} = \mathbb{E}_{t, k \sim \mathcal{U}(1,K)} \left[\frac{1}{k} \cdot \Phi(\Delta o_{t \to t+k}, \Delta o^*_{t \to t+k})\right]$$

2. **阶段二**：训练扩散模型，总损失 = $\mathcal{L}_{\text{hoi}} + \mathcal{L}_{\text{dyn}} + \mathcal{L}_{\text{obj}}$

**残差动力学损失**（核心创新）：

$$\mathcal{L}_{\text{dyn}} = \mathbb{E}_t \left[\|\Phi(\Delta\hat{o}_t^*, \Delta\hat{o}_t) - \Phi(\Delta o_t^*, \Delta o_t)\|_1\right]$$

这个设计巧妙之处在于：即使动力学模型 $\mathcal{D}$ 不完美，通过对生成序列和 GT 序列残差做差，系统性偏差被抵消，使学习聚焦于真正的生成不一致性。关键假设是 $\mathcal{D}$ 局部平滑且时间齐次。

**推理时动力学模型不使用**，保持推理效率。

## 实验关键数据

### 主实验

| 方法 | FID ↓ | $C_{F1}$ ↑ | C% ↑ | MPJPE ↓ | $T_{\text{obj}}$ ↓ | $R_{\text{obj}}$ ↓ |
|------|-------|---------|------|---------|---------|---------|
| InterDiff | 208.0 | 0.33 | 0.27 | 25.91 | 88.35 | 1.65 |
| MDM | 6.16 | 0.53 | 0.43 | 17.86 | 24.46 | 1.85 |
| CHOIS | 0.87 | 0.66 | 0.54 | 16.01 | 14.29 | 0.99 |
| **HOI-Dyn** | **0.48** | **0.71** | **0.60** | **15.60** | **12.47** | **0.90** |

| 方法 (3D-FUTURE) | FID ↓ | C% ↑ | FS ↓ |
|------|-------|------|------|
| CHOIS | 1.67 | 0.47 | 0.42 |
| **HOI-Dyn** | **1.62** | **0.54** | **0.37** |

### 消融实验

| 架构 | K | 参数量 | K=2 Loss |
|------|---|--------|----------|
| Coupled (K=2) D4-F64-H8 | 2 | 0.483M | 0.462 |
| Coupled (K=1) D4-F64-H8 | 1 | 0.483M | 0.514 |
| Decoupled (K=2) (D1-F64-H8)×2 | 2 | 0.463M | 0.532 |
| Coupled (K=2) D8-F128-H8 | 2 | 0.994M | 0.845 |

### 关键发现

1. **预测范围 K**：K=2 或 K=3 最优，过小（K=1）无法捕捉大幅度运动，过大（K=10）削弱细微交互建模
2. **耦合 vs 解耦**：联合预测旋转和平移的耦合设计在相似参数量下显著优于分别预测的解耦设计，验证了 HOI 中旋转和平移的内在耦合性
3. **模型规模**：0.5M 参数的轻量模型已足够，增大到 1M 参数反而性能下降（过拟合）
4. **定性对比**：CHOIS 产生提前运动（物体在人体动作前自发移向接触点），HOI-Dyn 消除了这类伪影

## 亮点与洞察

1. **Driver-Responder 视角新颖**：从控制论角度建模 HOI，自然解决了接触建模难题
2. **残差损失设计精巧**：通过差分消除动力学模型的系统偏差，使不完美的辅助模型仍能提供有效监督
3. **训练-推理解耦**：动力学模型仅用于训练，不增加推理开销
4. **轻量高效**：0.5M 参数的动力学模型 + 单卡 A4500 约 10 小时训练

## 局限与展望

1. 仅在 FullBodyManipulation 数据集上训练，场景多样性有限
2. K 需要手动选择，可以探索自适应预测范围
3. 动力学模型的局部平滑和时间齐次假设在快速、剧烈交互中可能不成立
4. 刚体变换假设限制了对柔性物体交互的建模能力

## 相关工作与启发

- **CHOIS**（SOTA baseline）：基于稀疏路径点引导扩散，但未建模交互因果关系
- **OMOMO**：以完整物体轨迹为输入生成人体姿态，但缺乏双向因果建模
- **CG-HOI**：使用人体网格上的接触场作为先验，但接触场预测本身就很困难
- **启发**：残差损失的思想可推广到其他使用不完美辅助模型的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — Driver-Responder 形式化和残差动力学损失概念新颖
- 实验充分度: ⭐⭐⭐⭐ — 定量定性全面，消融充分，但数据集有限
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，动机阐述有说服力
- 价值: ⭐⭐⭐⭐ — 对 HOI 生成领域有重要贡献，方法可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] IMU-HOI: A Symbiotic Framework for Coherent Human-Object Interaction and Motion Capture via Contact-Conscious Inertial Fusion](../../CVPR2026/human_understanding/imu-hoi_a_symbiotic_framework_for_coherent_human-object_interaction_and_motion_c.md)
- [\[CVPR 2026\] Decoupled Generative Modeling for Human-Object Interaction Synthesis](../../CVPR2026/human_understanding/decoupled_generative_modeling_for_human-object_interaction_synthesis.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[CVPR 2025\] Homogeneous Dynamics Space for Heterogeneous Humans](../../CVPR2025/human_understanding/homogeneous_dynamics_space_for_heterogeneous_humans.md)
- [\[CVPR 2026\] Interact2Ar: Full-Body Human-Human Interaction Generation via Autoregressive Diffusion Models](../../CVPR2026/human_understanding/interact2ar_full-body_human-human_interaction_generation_via_autoregressive_diff.md)

</div>

<!-- RELATED:END -->
