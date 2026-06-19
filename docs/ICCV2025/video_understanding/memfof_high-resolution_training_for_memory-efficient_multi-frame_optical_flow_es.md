---
title: >-
  [论文解读] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation
description: >-
  [ICCV 2025][视频理解][光流估计] MEMFOF 是首个面向显存效率的多帧光流方法，通过降低相关体积分辨率并引入高分辨率训练策略，在 1080p 推理仅需 2.09GB 显存的同时在 Spring、Sintel、KITTI 等基准上达到 SOTA 精度。 光流估计是低级视觉中的基础任务，广泛应用于视频动作识别、目…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "光流估计"
  - "显存优化"
  - "多帧估计"
  - "高分辨率训练"
  - "RAFT"
---

# MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation

**会议**: ICCV 2025  
**arXiv**: [2506.23151](https://arxiv.org/abs/2506.23151)  
**代码**: [https://github.com/msu-video-group/memfof](https://github.com/msu-video-group/memfof)  
**领域**: 视频理解  
**关键词**: 光流估计, 显存优化, 多帧估计, 高分辨率训练, RAFT

## 一句话总结

MEMFOF 是首个面向显存效率的多帧光流方法，通过降低相关体积分辨率并引入高分辨率训练策略，在 1080p 推理仅需 2.09GB 显存的同时在 Spring、Sintel、KITTI 等基准上达到 SOTA 精度。

## 研究背景与动机

光流估计是低级视觉中的基础任务，广泛应用于视频动作识别、目标检测、视频修复与合成。自 RAFT 提出以来，基于全对相关体积（all-pairs correlation volume）的迭代 GRU 细化范式成为主流，但其显存消耗随图像分辨率呈二次增长：FullHD（1920×1080）下 RAFT 需要约 8GB 显存，WQHD 更超过 25GB，严重限制了在消费级 GPU 上的部署。

现有工作沿两条路线改进：(1) **显存高效方法**——Flow1D 将运动分解为 1D、SCV 稀疏候选匹配、HCV 混合体积等，但往往牺牲精度；(2) **多帧方法**——VideoFlow、MemFlow、StreamFlow 利用时序一致性解决遮挡，但它们同样没有解决高分辨率下的显存瓶颈问题。例如 StreamFlow 在 1080p 下需要 18.97GB、VideoFlow-MOF 直接 OOM。

核心矛盾在于：**多帧利用时序信息能提升精度，但多帧意味着更多的相关体积和更大的显存开销**，二者在高分辨率场景下难以兼顾。

本文的切入角度：与其在推理时做降采样或拼贴（tile）来妥协，不如从架构层面压缩相关体积分辨率（从 1/8 到 1/16），使三帧方法的两个相关体积总显存从 10.4GB 降至 0.65GB，同时通过高分辨率训练策略弥补精度损失，实现"设计上的"显存-精度-速度三者兼顾。

## 方法详解

### 整体框架

MEMFOF 基于 SEA-RAFT 架构扩展为三帧输入。给定连续三帧 $I_{t-1}, I_t, I_{t+1}$，方法同时估计前向和后向双向光流 $f_{t \to t-1}$ 和 $f_{t \to t+1}$。核心流程为：

1. **特征提取**：共享 ResNet34 backbone 提取三帧特征 $F_t, F_{t-1}, F_{t+1}$
2. **上下文网络**：将三帧输入 ContextNetwork 得到初始流 $f^0$、隐状态 $h^0$ 和上下文特征 $g$
3. **双相关体积**：计算 $C_{t,t-1}$ 和 $C_{t,t+1}$
4. **迭代细化**：通过 N 次 GRU 更新，逐步细化双向流预测
5. **凸上采样**：最终流预测上采样到输入分辨率

在视频序列处理时，特征图和相关体积可跨帧复用，进一步节省计算。

### 关键设计

1. **相关体积分辨率降低（1/8 → 1/16）**：

    - 功能：将工作分辨率从标准的 1/8 降至 1/16，使相关体积大小缩减为原来的 1/16
    - 核心思路：在 SEA-RAFT 的 ResNet34 backbone 上增加一层步长卷积，将 1/8 特征图降至 1/16。同时将特征维度 $D_f$ 从 256 增大到 1024、更新模块维度 $D_c$ 从 128 增大到 512，以补偿降分辨率导致的信息损失
    - 关键影响：三帧方案的两个相关体积显存从 10.4GB 降至 0.65GB；总显存从 SEA-RAFT 的 8.19GB 降至 2.09GB（1080p 推理）
    - 设计动机：相关体积是 RAFT 类方法中显存消耗的核心瓶颈，其复杂度为 $\mathcal{O}((HW)^2)$。降低分辨率是最直接的解决方案，但需要通过增加通道数来弥补空间信息的损失

2. **高分辨率训练策略（FullHD-centric）**：

    - 功能：在训练时将标准数据集（如 Things、Sintel、KITTI）2x 上采样，使用接近 FullHD 的 crop size（如 864×1920）训练
    - 核心思路：标准光流数据集分辨率较低（如 Sintel 约 436×1024），直接训练后在 FullHD 上推理会出现欠拟合，特别是大运动区域精度不足。通过上采样数据集并使用大 crop 训练，模型能更好地学习高分辨率下的大运动模式
    - 训练流程（多阶段）：TartanAir（2x, crop 480×960）→ Things（2x, crop 864×1920）→ TSKH 混合（2x, crop 864×1920）→ 特定基准微调（Spring 用原始 1080p）
    - 设计动机：消融实验（Table 4）显示，2x 上采样全图训练比原始尺度训练在大运动区域（s40+）的 1px 误差从 33.9% 降至 28.5%，EPE 从 0.430 降至 0.341

3. **GMA 全局运动注意力与自适应缩放**：

    - 功能：重新引入 GMA 模块以增强运动一致性，并修改注意力缩放因子以适应不同分辨率
    - 核心思路：将 attention 中的缩放因子从 $1/\sqrt{D_c}$ 修改为 $\log_3(HW)/\sqrt{D_c}$，使模型在不同分辨率下表现更稳定
    - 设计动机：借鉴 MemFlow 的做法，使注意力机制能够随分辨率自适应调整

### 损失函数 / 训练策略

- **Mixture-of-Laplace (MoL) Loss**：沿用 SEA-RAFT，替代 L1 loss，对 T 个光流帧预测和 N 次迭代细化加权求和：$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{k=0}^{N} \gamma^{N-k} \mathcal{L}_{MoL}^{t,k}$，其中 $\gamma=0.85$
- 跳过 FlyingChairs 数据集（只支持两帧），使用 TartanAir 作为预训练起点
- 训练使用 32 块 A100 GPU，混合精度训练，总耗时 3-4 天
- Spring 微调阶段使用原始 1x 分辨率（1080×1920），训练时显存 28.5GB/GPU

## 实验关键数据

### 主实验

**Spring 基准（1080p 高分辨率）**：

| 方法 | #Frames | 显存(GB) | 1px↓ | EPE↓ | WAUC↑ |
|------|---------|---------|------|------|-------|
| RAFT | 2 | 7.97 | 6.790 | 1.476 | 90.92 |
| SEA-RAFT(M) ft | 2 | 8.19 | 3.686 | 0.363 | 94.53 |
| StreamFlow ft | 4 | 18.97 | 4.152 | 0.467 | 94.40 |
| MemFlow ft | 3 | 8.08 | 4.482 | 0.471 | 93.86 |
| **MEMFOF ft** | **3** | **2.09** | **3.289** | **0.355** | **95.19** |
| MEMFOF (zero-shot) | 3 | 2.09 | 3.600 | 0.432 | 94.48 |

**Sintel & KITTI**：

| 方法 | Sintel Clean EPE↓ | Sintel Final EPE↓ | KITTI Fl-all↓ |
|------|-------------------|-------------------|---------------|
| VideoFlow-MOF | 0.991 | 1.649 | 3.65 |
| StreamFlow | 1.041 | 1.874 | 4.24 |
| **MEMFOF** | **0.963** | 1.907 | **2.94** |

### 消融实验

| 配置 | EPE↓ | 1px(s40+)↓ | WAUC↑ | 说明 |
|------|------|-----------|-------|------|
| Bi, 1x, half推理 | 0.402 | 35.4% | 93.84 | Baseline: 原始尺度训练+半分辨率推理 |
| Bi, 1x, full推理 | 0.430 | 33.9% | 94.23 | 原始尺度训练+全分辨率推理 |
| Bi, 2x, crop, full推理 | 0.378 | 31.9% | 94.19 | 2x上采样+crop训练 |
| **Bi, 2x, full, full推理** | **0.341** | **28.5%** | **94.52** | 2x上采样全图训练（最优） |
| Uni, 2x, full, full推理 | 0.423 | 34.8% | 93.62 | 单向流，明显劣于双向 |

### 关键发现

- MEMFOF 的 zero-shot 结果（3.600 1px）甚至优于 SEA-RAFT(M) 的微调结果（3.686 1px），证明高分辨率训练的泛化能力
- 显存仅 2.09GB，是 StreamFlow (18.97GB) 的 1/9，使 FullHD 光流估计在消费级 GPU 上成为可能
- 双向流比单向流在 EPE 上提升约 20%，验证了多帧时序信息的价值
- 高分辨率训练对大运动区域（s40+）提升最显著：1px 从 33.9% 降至 28.5%

## 亮点与洞察

1. **"用更粗的空间分辨率 + 更宽的通道"替代"高空间分辨率 + 窄通道"**，在显存约束下是一个极具实用价值的 trade-off。1/16 分辨率看似激进，但通过 4x 通道补偿和高分辨率训练，精度不降反升
2. 高分辨率训练策略解决了一个长期被忽视的问题：低分辨率数据集训练 → 高分辨率推理的 domain gap，特别是大运动区域的欠拟合
3. 视频序列处理时的特征图/相关体积复用策略，使得实际批处理场景下效率进一步提升

## 局限与展望

- 三帧设计限制了时序信息的利用深度，未来可探索更长时序记忆机制
- 1/16 分辨率对小物体或细微运动的分辨能力可能有损失，论文未充分讨论这一 failure case
- 显存优化主要针对相关体积，backbone 和 context network 的显存占比随分辨率降低而变得更显著，需要进一步优化

## 相关工作与启发

- SEA-RAFT 的三个技巧（MoL loss、直接回归初始流、rigid-flow 预训练）被本文继承并扩展到三帧
- Flow1D/MeFlow 的分辨率降低思路被本文系统化应用
- 对于其他需要高分辨率推理的视觉任务（如深度估计、视频超分），本文的"上采样训练 + 降分辨率中间表示"策略有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[CVPR 2026\] Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation](../../CVPR2026/video_understanding/efficient_all-pairs_correlation_volume_sampling_for_optical_flow_estimation.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[CVPR 2025\] DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework](../../CVPR2025/video_understanding/dpflow_adaptive_optical_flow_estimation_with_a_dual-pyramid_framework.md)

</div>

<!-- RELATED:END -->
