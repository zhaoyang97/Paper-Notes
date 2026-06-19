---
title: >-
  [论文解读] Improving Sparse IMU-based Motion Capture with Motion Label Smoothing
description: >-
  [AAAI2026][人体理解][稀疏IMU] 提出 Motion Label Smoothing，将经典 label smoothing 从分类任务适配到稀疏IMU运动捕捉中，通过融合骨骼结构感知的Perlin噪声作为平滑标签，在不修改模型架构的前提下以即插即用方式提升三种SOTA方法在四个数据集上的精度，GlobalPose在TotalCapture上SIP误差降低20.41%。
tags:
  - "AAAI2026"
  - "人体理解"
  - "稀疏IMU"
  - "人体运动捕捉"
  - "Label Smoothing"
  - "Perlin噪声"
  - "正则化"
---

# Improving Sparse IMU-based Motion Capture with Motion Label Smoothing

**会议**: AAAI2026  
**arXiv**: [2511.22288](https://arxiv.org/abs/2511.22288)  
**作者**: Zhaorui Meng, Lu Yin, Yangqing Hou, Anjun Chen, Shihui Guo, Yipeng Qin (厦门大学, 卡迪夫大学)  
**代码**: 未公开  
**领域**: 人体理解  
**关键词**: 稀疏IMU, 人体运动捕捉, Label Smoothing, Perlin噪声, 正则化  

## 一句话总结

提出 Motion Label Smoothing，将经典 label smoothing 从分类任务适配到稀疏IMU运动捕捉中，通过融合骨骼结构感知的Perlin噪声作为平滑标签，在不修改模型架构的前提下以即插即用方式提升三种SOTA方法在四个数据集上的精度，GlobalPose在TotalCapture上SIP误差降低20.41%。

## 背景与动机

稀疏IMU运动捕捉系统仅使用6个IMU传感器（分别放置在双手腕、双脚踝、头部和臀部），实现实时人体运动重建。相比光学动捕系统，该方案具备便携、低成本、抗遮挡等优势，在电影制作、游戏和医疗康复中有广泛应用前景。

近年来，该领域的研究主要集中在**模型架构设计**上：TransPose引入Transformer架构提升精度，PIP结合物理优化增强运动合理性，PNP通过自回归MLP校准加速度信号，GlobalPose实现全3D空间的平移估计。然而，这些工作几乎都忽视了**正则化方法**——深度学习中同样关键的组件，在稀疏IMU运动捕捉的AI工具箱中留下了明显空白。

Label smoothing是分类任务中广泛使用的正则化技术，通过将one-hot标签与均匀分布混合来防止模型过度自信。但将其从分类领域直接迁移到运动捕捉并非显然——分类中的"均匀向量"对应到运动空间会退化为静态姿态（如T-pose），反而降低标签熵而非增加熵，与label smoothing的本质目的相矛盾。

## 核心问题

如何在不修改模型架构的前提下，设计一种适用于连续运动表示的label smoothing正则化方法，使其在提高标签熵的同时保持人体运动数据的三个内在属性：(1) 时间平滑性、(2) 关节相关性、(3) 低频主导性？

## 方法详解

### 整体框架

Motion Label Smoothing将经典label smoothing公式 $y' = (1-\epsilon)y + \epsilon u$ 中的 $y$ 替换为运动标签 $R$（24个SMPL关节旋转），核心在于设计满足运动属性的噪声 $u$：

$$R' = (1-\epsilon)R + \epsilon u$$

其中 $u$ 使用本文提出的 skeleton-based Perlin noise 实现。

### 关键设计1：运动标签三属性分析

**属性1 — 时间平滑性**：人体运动受肌肉力量和关节活动范围约束，关节角速度有上界：

$$\omega(t) \approx \frac{R(t+\Delta t) - R(t)}{\Delta t}, \quad \|\omega(t)\| \leq M$$

**属性2 — 关节相关性**：人体为刚性骨骼链耦合系统，父子关节旋转范围相互约束：

$$R_{\text{child}}(t) \in \mathcal{A}(R_{\text{parent}}(t))$$

**属性3 — 低频主导性**：运动信号以低频成分为主（实验中 $f_c=5\text{Hz}$ 时 $\alpha=0.7$），功率谱密度为：

$$P_c(f) = \left|\int_0^T R_c(t) e^{-i2\pi ft} dt\right|^2$$

### 关键设计2：Skeleton-based Perlin Noise

普通i.i.d.噪声（Gaussian/Uniform）的平坦功率谱和独立采样特性会破坏上述三个属性。本文提出基于骨骼结构的Perlin噪声：

$$u = \texttt{sk-Perlin}(JC, \mathcal{H}, size)$$

其中 $JC$ 为SMPL骨骼的6条关节链，$\mathcal{H} = \{S_b, S_t, S_s, p, oct, l\}$ 为Perlin噪声参数。

关键特性：
- **幅度解耦**：Perlin噪声的幅度由 $S_b$ 控制，平滑度由插值函数控制，两者独立——不像i.i.d.噪声存在幅度与平滑性的tradeoff
- **满足属性1&2**：沿时间轴插值保证时间连续性；先为每条关节链生成基础噪声，再为链内各关节叠加单层octave偏移，确保链内关节相关且可区分
- **满足属性3**：octave叠加中振幅按 $1/2^i$ 指数衰减，persistence $p=0.5$，octave数 $oct=5$，保证低频主导

## 实验关键数据

### 主实验：三种方法 × 四个数据集的误差降低比例

| 方法 | 数据集 | SIP误差↓ | Joint误差↓ | Mesh误差↓ |
|------|--------|---------|-----------|----------|
| TransPose+Ours | TotalCapture | 12.49 (↓12.54%) | 5.00 (↓5.84%) | 5.55 (↓5.77%) |
| PIP+Ours | TotalCapture | 10.54 (↓5.56%) | 4.38 (↓3.74%) | 5.07 (↓3.61%) |
| **GlobalPose+Ours** | **TotalCapture** | **7.84 (↓20.41%)** | **3.26 (↓17.68%)** | **3.75 (↓13.79%)** |
| TransPose+Ours | DIP-IMU | 13.57 (↓3.35%) | 4.64 (↓4.53%) | 5.50 (↓5.17%) |
| PIP+Ours | DIP-IMU | 11.62 (↓3.81%) | 4.18 (↓3.46%) | 4.88 (↓3.56%) |
| GlobalPose+Ours | DIP-IMU | 13.50 (↓1.96%) | 4.27 (↓2.06%) | 4.98 (↓1.97%) |

### 消融实验：逐步添加运动属性

| 配置 | SIP(°)↓ | Ang(°)↓ | Joint(cm)↓ | Mesh(cm)↓ |
|------|---------|---------|-----------|----------|
| Baseline (GlobalPose) | 9.85 | 9.55 | 3.96 | 4.35 |
| +Label Smoothing (Gaussian) | 8.82 | 8.65 | 3.82 | 4.43 |
| +Temporal Smoothness (T) | 8.59 | 8.30 | 3.77 | 4.37 |
| +Joint Correlation (T+J) | 8.22 | 8.02 | 3.52 | 4.12 |
| +Low-Freq Dominance (T+J+L, Ours) | **7.84** | **7.87** | **3.26** | **3.75** |

### 替代策略对比

| 策略 | SIP↓ | Joint↓ | Mesh↓ |
|------|------|--------|-------|
| T-pose Vector | 8.97 | 3.96 | 4.73 |
| AMASS Mean Vector | 8.75 | 3.87 | 4.60 |
| Uniform Noise | 8.72 | 3.82 | 4.44 |
| Gaussian Noise | 8.82 | 3.82 | 4.43 |
| Temporal Smoothing | 8.23 | 3.57 | 4.15 |
| Knowledge Distillation | 8.46 | 3.59 | 4.17 |
| **Ours** | **7.84** | **3.26** | **3.75** |

## 亮点

- **首个面向稀疏IMU运动捕捉的正则化方法**：填补了该领域AI工具箱中正则化模块的空白，且为即插即用设计，无需修改任何模型架构
- **对label smoothing的深刻洞察**：揭示了label smoothing的核心机制是增加标签熵而非简单的均匀化，纠正了长期存在的误解——这一洞察对其他连续回归任务同样有启发
- **系统化的运动属性分析**：首次严格定义并验证了运动标签的三个关键属性（时间平滑性、关节相关性、低频主导性），为Perlin噪声的设计提供了理论依据
- **跨方法和数据集的强泛化性**：在3种SOTA方法 × 4个数据集上均一致提升，其中GlobalPose在TotalCapture上SIP误差降低20.41%

## 局限与展望

- **继承底层方法的固有限制**：方法依赖SMPL模板人体模型，忽略了体型差异对IMU数据的影响，对儿童或极端体型用户的泛化能力有限
- **数据集运动类型有限**：训练数据（AMASS）虽然规模大但运动类型受限，对滑倒、街舞等复杂运动的重建仍有困难
- **缺乏理论收敛分析**：虽然从熵增角度给出了直觉解释，但未提供motion label smoothing对收敛速度或泛化界的严格理论分析

## 与相关工作的对比

- **TransPose/PIP/GlobalPose等**：专注于模型架构创新（RNN→Transformer→物理优化），本文从正则化角度提供互补提升
- **经典Label Smoothing (Szegedy et al. 2016)**：为分类任务设计，使用均匀向量；本文揭示直接迁移到回归任务并不有效
- **Knowledge Distillation (Yuan et al. 2020)**：知识蒸馏可视为隐式label smoothing，但在运动捕捉上效果不如本方法（SIP 8.46 vs 7.84）
- **Label Relaxation (Lienen & Hüllermeier 2021)**：将标签表示为概率集合，但未考虑运动数据的结构化属性

## 启发与关联

- **Perlin噪声的新应用**：原本用于计算机图形学中的纹理生成，本文巧妙地将其应用于运动标签的正则化，展示了跨领域技术迁移的价值
- **从分类到回归的正则化迁移**：label smoothing的核心是增加熵，这一洞察提示在其他连续信号回归任务（如语音合成、轨迹预测）中同样可以设计类似的结构化噪声正则化
- **属性感知的数据增强**：三个运动属性的识别为设计其他领域的任务特定数据增强策略提供了方法论参考

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将label smoothing适配到运动捕捉任务，Perlin噪声的应用新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ — 3种方法×4个数据集的全面评估+丰富的消融和替代策略对比
- 写作质量: ⭐⭐⭐⭐ — 逻辑链清晰：问题定义→属性分析→方法设计→实验验证，层层递进
- 价值: ⭐⭐⭐⭐ — 即插即用的正则化工具，对稀疏IMU领域有实际价值，但理论深度可进一步加强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] IMU-HOI: A Symbiotic Framework for Coherent Human-Object Interaction and Motion Capture via Contact-Conscious Inertial Fusion](../../CVPR2026/human_understanding/imu-hoi_a_symbiotic_framework_for_coherent_human-object_interaction_and_motion_c.md)
- [\[CVPR 2026\] Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors](../../CVPR2026/human_understanding/ground_reaction_inertial_poser_physics-based_human_motion_capture_from_sparse_im.md)
- [\[ICCV 2025\] MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances](../../ICCV2025/human_understanding/magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)
- [\[CVPR 2026\] MoBind: Motion Binding for Fine-Grained IMU-Video Pose Alignment](../../CVPR2026/human_understanding/mobind_motion_binding_for_fine-grained_imu-video_pose_alignment.md)
- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)

</div>

<!-- RELATED:END -->
