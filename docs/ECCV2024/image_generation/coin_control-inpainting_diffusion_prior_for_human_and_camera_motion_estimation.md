---
title: >-
  [论文解读] COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation
description: >-
  [ECCV 2024][图像生成][全局人体运动估计] 提出COIN方法，通过控制-补绘（Control-Inpainting）的改进版Score Distillation Sampling，结合人-场景关系损失，从单目动态相机视频中同时估计高质量的全局人体运动和相机运动。 从单目移动相机视频中恢复全局人体运动和相机运动是一…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "全局人体运动估计"
  - "相机运动估计"
  - "Score Distillation Sampling"
  - "运动扩散先验"
  - "控制补绘"
---

# COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation

**会议**: ECCV 2024  
**arXiv**: [2408.16426](https://arxiv.org/abs/2408.16426)  
**代码**: [有](https://nvlabs.github.io/COIN/)  
**领域**: 人体运动估计 / 扩散模型  
**关键词**: 全局人体运动估计, 相机运动估计, Score Distillation Sampling, 运动扩散先验, 控制补绘

## 一句话总结

提出COIN方法，通过控制-补绘（Control-Inpainting）的改进版Score Distillation Sampling，结合人-场景关系损失，从单目动态相机视频中同时估计高质量的全局人体运动和相机运动。

## 研究背景与动机

从单目移动相机视频中恢复全局人体运动和相机运动是一个重要但极具挑战性的问题，核心难点在于人体运动和相机运动的**纠缠**：在相机坐标系下观察到的运动是二者的叠加。

现有方法的主要问题：

**回归方法**（如GLAMR）：忽略相机运动，或假设简单场景，泛化能力差

**运动先验+SLAM方法**（如SLAHMR、PACE）：使用VAE运动先验将人体运动约束在低维空间，导致过度平滑；相机优化完全依赖全局人体运动，初始人体运动不准时会灾难性失败

**扩散模型作为先验的天然候选**：虽然理论上编码了丰富的运动先验，但直接应用SDS会产生不一致的、过度平滑的运动

作者在Fig.1中展示了一个滑板场景：PACE和WHAM完全失败，而COIN能正确恢复运动——这得益于其改进的扩散先验和人-场景关系约束。

## 方法详解

### 整体框架

COIN是一个**迭代优化框架**，联合优化全局人体运动 $\mathbf{H}$、相机运动 $\mathcal{C}$、相机尺度 $s$、初始帧相机姿态（高度 $h_0$ 和朝向 $R_0$）以及人体形状 $\beta$。

整体优化目标：$\min_{\mathbf{H}, \mathcal{C}, s, h_0, R_0, \beta} \mathcal{L}_{\text{body}} + \mathcal{L}_{\text{COIN-SDS}} + \mathcal{L}_{\text{HSR}}$

其中：
- $\mathcal{L}_{\text{body}}$：标准人体重建损失（2D重投影 + 3D关节 + 形状正则 + 时序平滑 + 脚部接触）
- $\mathcal{L}_{\text{COIN-SDS}}$：改进的Score Distillation Sampling损失（核心贡献）
- $\mathcal{L}_{\text{HSR}}$：人-场景关系损失（解决尺度模糊）

初始化流程：HybrIK获取逐帧局部SMPL参数 → DROID-SLAM获取相机到世界变换 → 将局部运动转换到世界坐标（此时因SLAM尺度未知，存在漂移）。

### 关键设计

**1. 多步去噪替代单步SDS**

标准SDS使用扩散模型单步去噪得到伪真值 $\hat{\mathbf{H}}_0^t$，但输入的微小扰动会导致输出剧烈变化（不一致性）。COIN改用10步DDIM去噪：

$$\tilde{\mathbf{H}}_{t-\Delta t} = \sqrt{\bar{\alpha}_{t-\Delta t}} \cdot \hat{\mathbf{H}}_0^t + \sqrt{1-\bar{\alpha}_{t-\Delta t}} \cdot \boldsymbol{\epsilon}_\phi^t$$

多步去噪产生更高质量、更一致的伪真值运动。

**2. 动态控制采样（Dynamic Controlled Sampling）**

在预训练扩散模型上附加ControlNet风格的控制分支 $\phi_c$，以部分观测的人体运动作为控制信号引导生成：

$$\tilde{\mathbf{H}}_0^t = \mathcal{D}_{\phi, \phi_c}(\tilde{\mathbf{H}}_t, t, \mathbf{c} \odot \mathbf{M})$$

关键创新在于**动态控制**：不使用固定的初始估计作为控制信号，而是每次迭代用上一步优化结果 $\mathbf{c} = \mathbf{H}$ 更新控制信号，形成自演化的控制。这避免了初始估计不准导致的性能退化。

控制分支采用ControlNet架构：复制预训练模型的4个编码块，接zero convolution，预训练模型冻结仅训练控制分支。

**3. 软补绘（Soft Inpainting）**

在去噪过程中区分"已知区域"和"未知区域"，对已知区域保留观测值，未知区域由扩散模型采样。关键是使用**连续权重掩码**而非二值掩码：

$$\tilde{\mathbf{M}} = w(t) \cdot \mathbf{S} \odot \mathbf{M}$$

其中 $w(t) = \max(0, \frac{t-0.5}{0.5})$ 随去噪时步递减，$\mathbf{S}$ 是观测置信度分数。这允许扩散模型对高置信观测做轻微修正，而对低置信区域大幅重建。

**4. 人-场景关系损失（HSR Loss）**

利用SLAM恢复的场景点云约束相机尺度。核心想法：投影到人体mesh可见顶点上的场景点应被人遮挡（深度更大）。

$$\mathcal{L}_{\text{HSR}} = -\frac{1}{|\mathcal{P}|} \sum_{i=1}^T \sum_{p \in \mathcal{P}^*} \min(0, \mathcal{T}^{(i)}(p)_z - j^{(i)}(p)_z) \cdot \mathbb{1}(\text{invisible})$$

这利用人-场景的深度关系提供了与运动先验互补的约束信息，解耦了相机尺度对人体运动的依赖。

### 损失函数 / 训练策略

- COIN-SDS损失：$\mathcal{L}_{\text{COIN-SDS}} = \frac{\omega(t)\sqrt{\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}} \|\mathbf{H} - \tilde{\mathbf{H}}_0\|_2^2$
- 人体损失：$\mathcal{L}_{\text{body}} = \mathcal{L}_{\text{2D}} + \mathcal{L}_{\text{3D}} + \mathcal{L}_\beta + \mathcal{L}_{\text{smooth}} + \mathcal{L}_{\text{contact}}$
- 运动扩散模型在AMASS数据集上训练，控制分支基于冻结的预训练模型高效微调

## 实验关键数据

### 主实验

**表1：RICH数据集全局人体运动估计**

| 方法 | PA-MPJPE↓ | W-MPJPE↓ | WA-MPJPE↓ | ACCEL↓ |
|------|-----------|----------|-----------|--------|
| GLAMR | 79.9 | 653.7 | 365.1 | 107.7 |
| SLAHMR | 52.5 | 571.6 | 323.7 | 9.4 |
| WHAM | 46.2 | 497.6 | 272.7 | 6.7 |
| PACE | 49.3 | 380.0 | 197.2 | 8.8 |
| **COIN** | **42.9** | **254.5** | **169.5** | **7.5** |

COIN在W-MPJPE上比PACE提升33%（380.0→254.5），比WHAM提升49%。

**表2：EMDB数据集全局人体运动估计**

| 方法 | PA-MPJPE↓ | W-MPJPE100↓ | RTE↓ | ROE↓ |
|------|-----------|-------------|------|------|
| WHAM | 41.9 | 439.2 | 8.4 | 36.3 |
| **COIN** | **32.7** | **407.3** | **3.5** | **34.1** |

### 消融实验

**RICH数据集消融（W-MPJPE）**

| 变体 | W-MPJPE↓ |
|------|----------|
| Vanilla SDS | 1453.5 |
| COIN w/o Controlled Sampling | 825.0 |
| COIN w/o Dynamic Control | 293.8 |
| COIN w/o Soft Inpainting | 325.8 |
| COIN w/o $\mathcal{L}_{\text{HSR}}$ | 273.0 |
| **COIN (完整)** | **254.5** |

每个组件都有显著贡献，其中Controlled Sampling的影响最大（移除后W-MPJPE从254.5退化到825.0）。

### 关键发现

1. Vanilla SDS在运动估计中完全失败（W-MPJPE=1453.5），证实了直接应用SDS的不一致性问题
2. 动态控制是最关键的组件，静态控制（w/o Dynamic Control）仍有不错表现但差距明显
3. HSR损失主要帮助解决尺度问题，对局部运动（PA-MPJPE）影响较小
4. COIN不仅提升全局运动，也改善局部运动质量（PA-MPJPE在三个数据集上均最优）

## 亮点与洞察

- **SDS的改良方案具有通用性**：控制-补绘的思想可迁移到其他需要从扩散模型蒸馏先验的任务（3D生成、运动合成等）
- **自演化控制信号**：动态更新控制条件形成正反馈循环，是解决初始估计不准问题的优雅方案
- **人-场景深度一致性**：利用场景点云的遮挡关系来标定相机尺度，是SLAM与人体重建结合的巧妙桥梁

## 局限与展望

1. **计算成本**：10步DDIM去噪在每次优化迭代中都要执行，整体优化较慢
2. **依赖SLAM质量**：如果DROID-SLAM在复杂场景中失败，整个管线会受影响
3. **单人假设**：当前框架针对单人场景，多人交互场景需要扩展
4. **运动扩散模型的训练数据限制**：AMASS数据集覆盖的运动类型有限，极端运动（如体操）可能失败

## 相关工作与启发

- COIN的控制-补绘SDS为其他扩散先验应用（text-to-3D、motion generation）提供了改良SDS的新模板
- 人-场景关系损失的理念可扩展到多人场景中的人-人关系约束
- 与WHAM等回归方法互补，可将COIN的优化结果作为回归方法的训练数据

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

## 与相关工作的对比

| 方法 | 核心思路 | 运动先验 | 相机估计 | 主要不足 |
|------|---------|---------|---------|---------|
| GLAMR | 回归全局运动 | 无 | 不估计 | 忽略相机运动，泛化差 |
| SLAHMR | SLAM + VAE先验优化 | HuMoR (VAE) | SLAM初始化+优化 | 过度平滑，2D对齐差 |
| PACE | SLAM + 条件VAE先验 | LEMO (cVAE) | 依赖人体运动优化 | 人体运动初始化差时相机灾难性失败 |
| WHAM | 回归+2D关键点提升 | 隐式学习 | 角速度输入 | 不显式恢复相机运动 |
| RoHM | 运动扩散模型恢复 | MDM (扩散) | 不估计 | 仅恢复局部运动，不处理相机 |
| **COIN** | **控制-补绘SDS + HSR** | **改良的运动扩散** | **联合优化+HSR约束** | **计算成本高，依赖SLAM** |

COIN与SLAHMR/PACE的核心区别：用扩散模型替代VAE先验，解决了VAE将运动压缩到低维空间导致的过度平滑；用COIN-SDS替代Vanilla SDS，解决了去噪不一致性；用HSR损失补充运动先验，避免人体运动初始化差时的级联失败。

## 启发与关联

- **改良SDS的通用范式**：COIN提出的控制-补绘策略（动态控制信号 + 软补绘 + 多步去噪）是对标准SDS的系统性改进，可推广到text-to-3D、视频生成等任何需要从扩散模型蒸馏先验的下游任务
- **自演化优化策略**：将上一轮优化结果作为下一轮控制信号的设计思想，类似于EM算法或自举（bootstrapping），适用于初始估计不可靠的迭代优化场景
- **跨模态一致性约束**：HSR损失利用场景几何与人体的深度遮挡关系来标定相机尺度，这种跨模态（运动 vs 场景）的一致性约束思想可以扩展到人-物交互、多人场景等更复杂的设定

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->
