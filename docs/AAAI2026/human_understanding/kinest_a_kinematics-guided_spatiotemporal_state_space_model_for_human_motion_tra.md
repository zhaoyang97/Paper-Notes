---
title: >-
  [论文解读] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals
description: >-
  [AAAI 2026][人体理解][全身运动追踪] 提出 KineST，一种运动学引导的状态空间模型，通过运动学树双向扫描策略和混合时空表征学习，从头显稀疏信号高效重建全身运动，在精度和时序一致性上均超越 SOTA。 领域现状 全身运动追踪在 AR/VR 中扮演关键角色，但头显设备（HMD）仅提供头部和双手的 3 个稀疏跟踪…
tags:
  - "AAAI 2026"
  - "人体理解"
  - "全身运动追踪"
  - "状态空间模型"
  - "运动学先验"
  - "AR/VR"
  - "稀疏信号"
---

# KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals

**会议**: AAAI 2026  
**arXiv**: [2512.16791](https://arxiv.org/abs/2512.16791)  
**代码**: [项目页](https://kaka-1314.github.io/KineST/)  
**领域**: 视频理解  
**关键词**: 全身运动追踪, 状态空间模型, 运动学先验, AR/VR, 稀疏信号

## 一句话总结

提出 KineST，一种运动学引导的状态空间模型，通过运动学树双向扫描策略和混合时空表征学习，从头显稀疏信号高效重建全身运动，在精度和时序一致性上均超越 SOTA。

## 研究背景与动机

### 领域现状
全身运动追踪在 AR/VR 中扮演关键角色，但头显设备（HMD）仅提供头部和双手的 3 个稀疏跟踪信号，从中推断 22 个关节的全身运动是一项极具挑战性的问题。

### 核心痛点
现有方法面临**精度与流畅性难以兼顾**的困境：
- **高计算成本方案**（AvatarJLM 63M 参数, SAGE 137M）：通过堆叠 Transformer 块或大型生成模型取得好效果，但部署成本高，不适合 AR/VR 实时应用
- **轻量化方案**存在取舍：
    - RPM：引入预测一致性锚改善流畅性，但**牺牲了姿态精度**
    - 分离的时空模块（HMD-Poser, MMD）：提升关节交互建模，但将部分建模能力转移到单帧空间特征上，**损害运动流畅性**
- **State Space Duality (SSD)** 框架虽然在时序建模上有效率优势，但直接应用到运动追踪效果不佳——**单向扫描和缺乏姿态先验**

### 核心 Idea

**将运动学先验嵌入 SSD 的扫描策略**：(1) 重新定义扫描顺序为沿人体运动学树的双向扫描；(2) 将时空上下文紧密耦合而非分离建模；(3) 在 Lie 群 SO(3) 上定义几何角速度损失提升运动连续性。

## 方法详解

### 整体框架

KineST 由两类核心模块交替堆叠组成：

1. **Temporal Flow Module (TFM)**：N 个，学习帧间动态
2. **Spatiotemporal Kinematic Flow Module (SKFM)**：M 个，运动学引导的时空建模

输入：稀疏 IMU 信号 $X \in \mathbb{R}^{L \times C}$（$C = 3 \times (3+6+3+6) = 54$，每个跟踪部件含 3D 位置 + 6D 旋转 + 线速度 + 角速度）

输出：全身姿态 $Y \in \mathbb{R}^{L \times V}$（$V = 22 \times 6$，22 个 SMPL 关节的 6D 旋转）

处理流程：线性嵌入 → N 个 TFM → M 个 SKFM → 线性回归头

### 关键设计

#### 1. **Temporal Flow Module (TFM)**

- **Bi-SSD 块**：并行前向和后向分支进行双向时序建模
- 前向分支：
    - 通过 LN + Linear + Conv + SiLU 生成状态向量 X, B, C
    - 通过 Linear + LN 生成状态转移矩阵 A
    - 自适应门控向量 $f_1$ 与 SSM 输出逐元素相乘
- 后向分支：对输入时间翻转，相同操作后再翻转回来
- **Local Motion Aggregator (LMA)**：基于卷积的局部依赖建模
- **Global Motion Aggregator (GMA)**：基于轻量 Transformer 的全局运动周期性建模
- 输出：$T_1 = \text{GMA}(\text{LMA}(F_f^t + F_b^t))$

#### 2. **运动学树扫描策略（Kinematic Tree Scanning Strategy, KTSS）**

- **核心创新**：将 SSD 的扫描顺序从标准的索引顺序重新定义为沿人体运动学树的双向扫描
- **两种变体**：
    - **Five-branch Kinematic Scan (FKS)**：严格沿五条运动学分支扫描（头→左臂→右臂→左腿→右腿），更好感知局部运动学依赖，但分支式设计损害全身完整性
    - **Unified Kinematic Scan (UKS)**：将根关节置于中心，有效耦合上下体运动：`[21,19,17,14,15,12,20,18,16,13,9,6,3,0,1,4,7,10,2,5,8,11]`
- **最终采用 UKS**：兼顾局部运动学依赖和全局运动连贯性
- **设计动机**：SSD 的序列特性使每个关节特征可从前一个关节状态推断，运动学树定义的扫描顺序让特征沿亲子关节层级流动

#### 3. **时空混合机制（Spatiotemporal Mixing Mechanism, STMM）**

- **核心思路**：将空间关节维度和时间帧维度合并为统一轴，实现时空紧密耦合
- **具体流程**（Algorithm 1）：
  1. 时序特征 $T_N$ 投影到关节空间 $S_l \in \mathbb{R}^{L \times H}$，reshape 为 $S_l' \in \mathbb{R}^{L \times J \times D}$
  2. 按 KTSS 重排为前向和后向关节序列 $S_f, S_b$
  3. **关键步骤**：将序列和关节维度合并：$S_f' \in \mathbb{R}^{(LJ_f) \times D}$
  4. 用 Bi-SSD 处理混合张量
  5. 双向特征求和 + 线性投影 + LMA + GMA
- **设计动机**：独立建模空间或时间都会导致另一方面的退化，紧密耦合确保精度和流畅性的平衡

### 损失函数 / 训练策略

#### 几何角速度损失

在 Lie 群 SO(3) 的切空间（Lie 代数 $\mathfrak{so}(3)$）中计算角速度：

$$V_t = R_{t-1}^{-1} R_t$$

$$\theta_V = \arccos\left(\frac{\text{Tr}(V)-1}{2}\right)$$

$$\log V = \theta_V \cdot \frac{1}{2\sin\theta_V}\begin{bmatrix}V_{32}-V_{23}\\ V_{13}-V_{31}\\ V_{21}-V_{12}\end{bmatrix}$$

$$\mathcal{L}_{\text{angvel}}^{\text{geo}} = \sum_{t=1}^{T-1}\|\log(V_t) - \log(\hat{V}_t)\|_1$$

- **与现有方法区别**：先前方法在欧氏空间用一阶有限差分近似角速度，忽略了旋转的非线性流形结构
- **设计动机**：旋转位于 SO(3) 李群上，角速度必须在切空间中计算才有物理意义

总损失：$\mathcal{L} = \alpha \mathcal{L}_{\text{rot}} + \beta \mathcal{L}_{\text{ori}} + \delta \mathcal{L}_{\text{angvel}}^{\text{geo}}$（$\alpha=1, \beta=0.02, \delta=1$）

训练配置：NVIDIA 4090, batch_size=256, Adam, lr=3e-4 (衰减至 3e-5), 序列长度 L=96

## 实验关键数据

### 主实验（Protocol 1, AMASS 数据集）

| 方法 | MPJRE↓ | MPJPE↓ | MPJVE↓ | Hand PE↓ | Upper PE↓ | Lower PE↓ | Jitter↓ | 参数量 |
|------|--------|--------|--------|----------|----------|----------|---------|--------|
| AvatarPoser | 3.08 | 4.18 | 27.70 | 2.12 | 1.81 | 7.59 | 14.49 | 4M |
| AvatarJLM | 2.90 | 3.35 | 20.79 | 1.24 | 1.72 | 6.20 | 8.39 | 63M |
| SAGE | 2.53 | 3.28 | 20.62 | 1.18 | 1.39 | 6.01 | 6.55 | 137M |
| HMD-Poser | 2.32 | 3.15 | 18.15 | 1.35 | 1.34 | 5.76 | 6.21 | 17M |
| MMD | 2.31 | 3.22 | 17.88 | 0.94 | 1.29 | 6.01 | 7.39 | 14M |
| **KineST** | **2.25** | **2.86** | **15.26** | **1.04** | **1.24** | **5.20** | **5.97** | **11M** |

KineST 以 **11M 参数**（SAGE 的 1/12）实现全面最优：MPJRE -2.6%, MPJPE -11.2%, MPJVE -14.7%。

### Protocol 2 & 3

| 方法 | Protocol 2 MPJRE↓ | MPJPE↓ | MPJVE↓ | Protocol 3 MPJRE↓ | MPJPE↓ | MPJVE↓ |
|------|-------------------|--------|--------|-------------------|--------|--------|
| AvatarJLM | 4.30 | 4.93 | 26.17 | 7.01 | 9.72 | 27.59 |
| AGRoL | 4.30 | 6.17 | 24.40 | - | - | - |
| **KineST** | **4.28** | **5.17** | **24.08** | **6.91** | **9.68** | **25.16** |

在真实设备采集数据（Protocol 3）上也取得全面最优，验证实用性。

### 消融实验

| 扫描策略 | MPJRE↓ | MPJPE↓ | MPJVE↓ | Jitter↓ |
|---------|--------|--------|--------|---------|
| Index-order (SMPL) | 2.32 | 3.11 | 17.81 | 8.27 |
| FKS (五分支) | 2.28 | 3.00 | 16.25 | 7.01 |
| **UKS (统一)** | **2.25** | **2.86** | **15.26** | **5.97** |

| SKFM 建模机制 | MPJRE↓ | MPJPE↓ | MPJVE↓ | Jitter↓ |
|--------------|--------|--------|--------|---------|
| 纯时序 | 2.27 | 2.97 | 16.84 | 7.83 |
| 纯空间(holistic) | 2.41 | 3.10 | 16.77 | 7.72 |
| 纯空间(token-wise) | 2.23 | 2.93 | 17.85 | 9.31 |
| **STMM (混合)** | **2.25** | **2.86** | **15.26** | **5.97** |

| 损失函数 | MPJRE↓ | MPJPE↓ | MPJVE↓ | Jitter↓ |
|---------|--------|--------|--------|---------|
| Baseline | 2.25 | 2.87 | 16.10 | 6.75 |
| + $L_{angvel}^{diff}$ (有限差分) | 2.29 | 3.03 | 15.91 | 6.44 |
| + **$L_{angvel}^{geo}$ (几何)** | **2.25** | **2.86** | **15.26** | **5.97** |

### 关键发现

1. **UKS 全面优于 FKS 和索引扫描**：Jitter 从 8.27 降至 5.97，证明全身统一扫描比分支式更好
2. **STMM 是精度+流畅性的关键**：token-wise 空间建模虽然 MPJRE 最低，但 Jitter 最高（9.31）；STMM 在所有指标上实现平衡
3. **几何角速度损失优于有限差分版本**：后者虽降低 MPJVE 但显著增加旋转和位置误差，前者则在不损害精度的前提下改善流畅性
4. **每个 Flow Module 组件都必不可少**：去除 Bi-SSD 导致 Jitter 从 5.97 飙升至 13.57

## 亮点与洞察

1. **运动学先验的优雅嵌入**：将 SSD 扫描顺序重定义为运动学树遍历，概念简洁但效果显著（Jitter 降低 28%）
2. **时空混合而非分离**：通过维度合并实现紧密的时空耦合，解决了分离建模导致的精度/流畅性权衡
3. **几何一致性**：在 SO(3) 李群上定义角速度损失，数学上严谨，物理上有意义
4. **极致轻量化**：11M 参数超越 137M 的 SAGE，推理 96 帧仅需 12.9ms，适合 AR/VR 部署

## 局限与展望

1. 在复杂运动（体操、杂技等）上性能受限，需引入更丰富的运动先验
2. 当前仅从 3 个 HMD 跟踪点推断，未利用可能的额外传感器（如腰部 IMU）
3. 序列长度固定为 96 帧，更长或更短的序列需要重新调优
4. SMPL 模型仅前 22 个关节，手指等精细关节未覆盖

## 相关工作与启发

- 与 MMD 同属 SSM 方法，但 MMD 分离时空建模，KineST 通过 STMM 紧密耦合
- 运动学树扫描策略可推广到其他骨架相关任务（动作识别、运动生成）
- 几何角速度损失的 Lie 群思想可应用于任何涉及旋转预测的问题
- LMA + GMA 的局部/全局聚合器设计是一个通用的序列建模工具

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — KTSS 和 STMM 都是创新性设计，几何角速度损失有深厚数学基础
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三种 protocol、真实设备数据、详尽消融（扫描策略、建模机制、损失函数、组件、序列长度）
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，算法伪代码清楚，可视化丰富
- **价值**: ⭐⭐⭐⭐⭐ — 直接面向 AR/VR 实时部署需求，轻量高效的方案有很高实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation](../../ICCV2025/human_understanding/high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)
- [\[ICLR 2026\] QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture](../../ICLR2026/human_understanding/quamo_quaternion_motion_kinematics.md)
- [\[AAAI 2026\] Improving Sparse IMU-based Motion Capture with Motion Label Smoothing](improving_sparse_imu-based_motion_capture_with_motion_label_smoothing.md)

</div>

<!-- RELATED:END -->
