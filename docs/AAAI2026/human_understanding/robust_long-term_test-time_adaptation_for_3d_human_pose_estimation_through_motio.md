---
title: >-
  [论文解读] Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization
description: >-
  [AAAI 2026][人体理解][3D 人体姿态估计] 针对 3D 人体姿态估计在线测试时自适应中的误差累积问题，提出基于运动离散化（无监督聚类获得锚运动集）+ 自回放机制 + 软重置策略的解决方案，使模型能在长时间持续适应中稳健利用个人形态和习惯性运动特征，在 Ego-Exo4D 和 3DPW 上超越所有现有在线 TTA 方法。
tags:
  - "AAAI 2026"
  - "人体理解"
  - "3D 人体姿态估计"
  - "测试时自适应"
  - "运动离散化"
  - "误差累积"
  - "个性化适应"
  - "软重置"
  - "自回放"
---

# Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization

**会议**: AAAI 2026  
**arXiv**: [2511.18851](https://arxiv.org/abs/2511.18851)  
**代码**: 待确认  
**领域**: 人体理解  
**关键词**: 3D 人体姿态估计, 测试时自适应, 运动离散化, 误差累积, 个性化适应, 软重置, 自回放  

## 一句话总结

针对 3D 人体姿态估计在线测试时自适应中的误差累积问题，提出基于运动离散化（无监督聚类获得锚运动集）+ 自回放机制 + 软重置策略的解决方案，使模型能在长时间持续适应中稳健利用个人形态和习惯性运动特征，在 Ego-Exo4D 和 3DPW 上超越所有现有在线 TTA 方法。

## 研究背景与动机

预训练的 3D 人体姿态估计器在部署到训练域外的真实场景时，性能显著退化。在线测试时自适应（Online TTA）通过在未标注的流式视频上用自监督方式持续更新模型来缓解这一问题。但现有方法存在两大核心缺陷：

**误差累积**：自监督信号来自不完美的 2D 检测和 3D 估计，预测错误随时间复合叠加，长时间适应后性能反而下降

**个人特征未利用**：同一个人具有一致的身体形状和习惯性运动模式，持续观察应该能捕获这些个性化特征来增强估计精度，但误差累积使得持续适应反而成为风险

具体来说：BOA/DynaBOA 缺乏有效 3D 引导导致深度估计不准；CycleAdapt 过度依赖不完美估计作为伪标签，长期自监督后陷入恶性循环。

## 方法详解

### 整体框架

框架包含两个交替适应的组件：
- **姿态估计器 F**：ResNet-50 骨干，从图像回归 SMPL 参数 (pose θ, shape β, translation ψ)
- **运动去噪网络 M**：自编码器架构 (Encoder E + Decoder D + Codebook C)，处理 16 帧@15fps 的连续姿态序列

两者在每个 batch（160 帧@30fps）中交替更新 12 个 cycle：F 产生姿态估计，M 对其去噪和离散化产生锚运动和去噪运动，反过来指导 F 的适应。

### 关键设计 1：运动离散化（Motion Discretization）

在 M 的预训练阶段，对其潜在空间进行**无监督聚类**，构建残差码本 $\mathcal{C} = \{C^i \in \mathbb{R}^{N_c \times d} | i=1,...,k\}$（k=3 层 × 512 码字 × 512 维），每层递归检索最近码字并减去残差。

测试时用途：
- 将 F 的输出 $\theta_{1:t}$ 编码为潜在向量 $z = E(\theta_{1:t})$，量化得到 $c = \sum_{i=1}^k c_i$
- 解码锚运动 $\theta^* = D(c)$：经过离散化的运动**过滤了高频噪声**，保留核心运动模式
- 用锚损失 $L_{ach} = ||\theta - \text{sg}(\theta^*)||$ 正则化 F 的更新

**关键洞察**：离散化充当了"信息瓶颈"——去噪运动 $\theta'$ 可能被不完美自监督污染，但锚运动 $\theta^*$ 通过码本量化有效滤除了误差，提供可靠的正则信号。

### 关键设计 2：自回放（Self-Replay）

M 的在线适应会导致**表征漂移**——潜在码字逐渐丧失解码出一致/规则锚运动的能力。为此设计自回放机制：

1. 从预训练码本 $\bar{\mathcal{C}}$ 随机采样码字 $\bar{c}$
2. 用预训练的解码器 $\bar{D}$ 解码得到回放运动 $\bar{\theta}_{1:t} = \bar{D}(\bar{c})$
3. 将回放运动和测试时估计同时输入 M，用重建损失联合更新

$$L_M = ||\bar{\theta}'_{1:t} - \text{sg}(\bar{\theta}_{1:t})|| + ||\theta'_{1:t} - \text{sg}(\theta_{1:t})||$$

同时用回放潜在向量的 EMA（衰减 0.999）更新码本，保持码本与演化的潜在空间同步。**无需访问原始预训练数据**，解决隐私问题。

### 关键设计 3：软重置（Soft Reset）

适应每个 batch 后，对 F 执行 EMA 重置：

$$F \leftarrow \mu_F \cdot F_{pre} + (1 - \mu_F) \cdot F$$

其中 $\mu_F = 0.95$。作用：减少单个 batch 上噪声更新的影响，同时保留历史适应中学到的关键个人特征。相比完全重置（$\mu_F=1$，丢弃所有适应信息）和不重置（$\mu_F=0$，噪声累积），取得最佳平衡。

### 损失函数

姿态估计器 F 的总损失：

$$L_F = L_p + \lambda_1 L_s + \lambda_2 L_{2D} + \lambda_3 L_{ach}$$

其中 $L_p$ 为去噪运动伪标签损失，$L_s$ 为形状一致性损失，$L_{2D}$ 为 2D 重投影误差，$L_{ach}$ 为锚运动损失（$\lambda_1=0.001, \lambda_2=0.1, \lambda_3=0.3$）。

## 实验

### 主实验表：与在线 TTA 方法对比

| 方法 | Ego-Exo4D MPJPE↓ | Ego-Exo4D PA↓ | 3DPW MPJPE↓ | 3DPW PA↓ | 3DPW MPVPE↓ |
|------|:-:|:-:|:-:|:-:|:-:|
| Pre-trained F | 205.8 | 116.5 | 230.3 | 123.4 | 253.4 |
| BOA† | 135.1 | 70.0 | 98.2 | 55.8 | 114.2 |
| DynaBOA† | 153.3 | 71.6 | 139.7 | 63.8 | 155.1 |
| CycleAdapt | 145.0 | 80.5 | 141.0 | 79.6 | 155.6 |
| **Ours (OpenPose)** | **121.5** | **68.1** | **83.9** | **51.6** | **100.3** |
| **Ours (ViTPose)** | **116.4** | **60.8** | **85.0** | **53.3** | **100.4** |

†使用原始预训练数据。MPJPE 单位为 mm，越低越好。

### 消融实验表：Ego-Exo4D 全场景

| 软重置 | 锚损失 | 自回放 | MPJPE↓ | PA↓ |
|:------:|:------:|:------:|:------:|:---:|
| ✗ | ✗ | ✗ | 144.3 | 80.6 |
| ✓ | ✗ | ✗ | 122.9 | 69.2 |
| ✗ | ✓ | ✓ | 138.2 | 74.8 |
| ✓ | ✓ | ✓ | **121.5** | **68.1** |

| 软重置衰减 $\mu_F$ | MPJPE↓ |
|:-:|:-:|
| 0 (无重置) | 138.2 |
| 0.9 | 122.7 |
| **0.95** | **121.5** |
| 1.0 (完全重置) | 125.3 |

### 关键发现

1. **运动离散化 + 自回放 + 软重置三者协同**：仅有软重置已将 MPJPE 从 144.3 降到 122.9，加入运动离散化（含自回放）进一步降到 121.5；单独使用锚损失（无自回放）因表征漂移反而增加误差
2. **持续适应有效**：相比每次重置到预训练权重（125.3），持续适应达到 121.5，说明成功捕获了个人特征
3. **与领域泛化方法可比**：使用 ResNet-50 骨干的本方法在 Ego-Exo4D 上超过了使用 ViT-H/16 + 更多训练数据的 HMR-2.0b（125.2 MPJPE）
4. **篮球场景的挑战**：鱼眼相机导致的畸变使部分参与者持续适应效果有限，揭示了方法的边界条件
5. **运行效率**：处理 160 帧（5.3 秒视频）仅需 6.8 秒，接近实时

## 亮点

- **运动离散化作为信息瓶颈**的设计理念优雅——用码本量化自动过滤自监督噪声，无需手动设计阈值或质量评估
- **自回放机制**的巧妙之处在于完全不需要原始训练数据，仅用预训练码本和解码器即可"回忆"正则运动
- **个性化 TTA 范式**的提出有启发意义——不再把每个视频片段独立对待，而是利用同一人的长期观察
- 理论分析与实践兼顾：统计显著性检验（Wilcoxon 检验）验证了方法的一致性优势

## 局限性

1. 使用 ResNet-50 骨干以对齐先前工作，未验证 ViT 等现代骨干的效果
2. 对严重畸变（鱼眼相机矫正后的边缘区域）适应能力有限
3. 回放运动质量受限于预训练数据的多样性和码本容量
4. 仅在第三人称/外中心视角验证，自中心/第一人称视角的严重自遮挡场景未探索
5. 损失权重和 EMA 衰减系数作为固定超参数，未做自适应调整

## 相关工作

- **BOA / DynaBOA** (Guan et al., 2021/2022): 在线双层适应 + GT 2D + 源域 exemplar，缺乏 3D 引导
- **CycleAdapt** (Nam et al., 2023): 循环适应 F 和 M，但未解决误差累积
- **TokenHMR** (Dwivedi et al., 2024): ViT-H 骨干 + 量化姿态表示用于领域泛化
- **VQ-VAE 运动生成** (Zhang et al., 2023; Guo et al., 2024): 离散运动token用于GPT式生成

## 评分

- 新颖性: ⭐⭐⭐⭐ （运动离散化抑制误差累积的思路新颖，个性化 TTA 范式有前瞻性）
- 实验充分度: ⭐⭐⭐⭐⭐ （两大数据集 + 30 参与者 + 分场景消融 + 统计检验 + 运行时分析）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，问题动机论述到位）
- 价值: ⭐⭐⭐⭐ （长期在线适应是实际部署的关键问题，方案实用且高效）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](../../CVPR2025/human_understanding/crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[ECCV 2024\] Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework](../../ECCV2024/human_understanding/human_motion_forecasting_in_dynamic_domain_shifts_a_homeostatic_continual_test-t.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [\[CVPR 2026\] SAM 3D Body: Robust Full-Body Human Mesh Recovery](../../CVPR2026/human_understanding/sam_3d_body_robust_full-body_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
