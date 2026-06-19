---
title: >-
  [论文解读] SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens
description: >-
  [CVPR 2025][3D视觉][多人3D网格估计] 提出 SAT-HMR，一种基于 DETR 的实时多人 3D 人体网格估计框架，通过引入尺度自适应 token——对小尺度人物使用高分辨率 token、大尺度人物使用低分辨率 token、背景 token 进行池化压缩——在保持高分辨率输入精度的同时将推理速度提升至 24 FPS，实现了精度和速度的最佳平衡。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "多人3D网格估计"
  - "尺度自适应"
  - "高效ViT"
  - "DETR"
  - "实时推理"
---

# SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens

**会议**: CVPR 2025  
**arXiv**: [2411.19824](https://arxiv.org/abs/2411.19824)  
**代码**: [项目主页](https://ChiSu001.github.io/SAT-HMR/)  
**领域**: 3D视觉/人体理解  
**关键词**: 多人3D网格估计, 尺度自适应, 高效ViT, DETR, 实时推理

## 一句话总结

提出 SAT-HMR，一种基于 DETR 的实时多人 3D 人体网格估计框架，通过引入尺度自适应 token——对小尺度人物使用高分辨率 token、大尺度人物使用低分辨率 token、背景 token 进行池化压缩——在保持高分辨率输入精度的同时将推理速度提升至 24 FPS，实现了精度和速度的最佳平衡。

## 研究背景与动机

- **多人 3D 网格估计的挑战**：从单张 RGB 图像估计所有人的 SMPL 参数，既需要局部细节（关节姿态）又需要全局上下文（相对位置、遮挡关系）。
- **多阶段 vs 一阶段**：多阶段方法先检测再逐人裁剪估计，精度好但丢失全局上下文且难以处理遮挡。一阶段方法（如 ROMP、BEV）基于 CNN 处理整图，但低分辨率输入限制了表达能力。
- **高分辨率的代价**：新兴的 DETR 式方法（AiOS、Multi-HMR）通过使用高分辨率输入（1288 分辨率）达到 SOTA，但推理速度仅约 5 FPS。关键观察：**高分辨率主要有利于小尺度人物（远距离/儿童/蜷缩姿势）**——0-10% 尺度范围内误差下降 35mm，而 30%+ 尺度范围改善极小。
- **核心洞察**：对大尺度人物（靠近相机、占图像大面积）使用高分辨率 token 是计算浪费——他们已经由足够多的 token 表示。应将高分辨率计算资源集中在真正需要的小尺度人物上。
- **背景区域可压缩**：背景 token 提供有用的上下文信息（不应完全丢弃），但可以通过空间池化进一步压缩。

## 方法详解

### 整体框架

SAT-HMR 采用 DETR 式管线：(1) 从低分辨率图像提取 token 并用浅层 Transformer 编码 → (2) 用尺度头预测 patch 级尺度图 $\mathbf{S}$ → (3) 根据尺度图将 token 分为背景/小尺度/大尺度三类 → (4) 小尺度 token 替换为高分辨率对应 token，背景 token 池化压缩，大尺度 token 保持不变 → (5) 拼接得到尺度自适应 token $\mathcal{T}_{\text{SA}}$ → (6) 后续 Transformer 编码器 + 解码器 + 预测头回归 SMPL 参数。

### 关键设计

**设计一：Patch 级尺度图预测**
- **功能**：判断每个 patch 是否覆盖人物以及该人物的相对尺度
- **核心思路**：尺度图 $\mathbf{S}(i,j) = (c, s)$ 包含两个值：$c$ 为人物置信度（0 表示背景），$s = \min(d_{\text{bb}} / S_{\text{hr}}, 1)$ 为人物尺度（bounding box 对角线与图像最大边的比值）。当 patch 覆盖多人时取最近人物的尺度。从低分辨率 token 经 $N_{\text{lr}}$ 层 Transformer + MLP 尺度头预测。
- **设计动机**：尺度定义直接反映了人物在图像中的占比，与该区域是否需要高分辨率token强相关；使用轻量头网络预测，引入的计算开销极小。

**设计二：尺度自适应 Token 选择与替换**
- **功能**：动态调整各区域的 token 分辨率
- **核心思路**：根据阈值 $\alpha_c$（置信度）和 $\alpha_s$（尺度）将 token 分为三组：(1) 背景 $\mathcal{T}_B$：每 4 个相邻 token 池化为 1 个得到 $\mathcal{T}_B'$；(2) 小尺度 $\mathcal{T}_{\text{SMALL}}$：剪除后用高分辨率图像对应位置的 token 替换，$k_{\text{hr}} = 4 k_{\text{small}}$；(3) 大尺度 $\mathcal{T}_{\text{LARGE}}$：保持低分辨率不变。最终拼接 $\mathcal{T}_{\text{SA}} = \{\mathcal{T}_B', \mathcal{T}_{\text{LARGE}}, \mathcal{T}_{\text{HR}}\}$。
- **设计动机**：小尺度人物在低分辨率下特征不足（主导了高分辨率带来的增益），用 4 倍分辨率的 token 替换正好弥补；大尺度人物已有足够 token 覆盖；背景不丢弃而是池化保留上下文。

**设计三：双分辨率编码器对齐**
- **功能**：确保低分辨率和高分辨率 token 在同一特征空间中可拼接
- **核心思路**：低分辨率和高分辨率分支各自使用浅层 Transformer 编码器处理（$N_{\text{lr}} = N_{\text{hr}} = 3$ 层），共享相同的 DINOv2 预训练权重。两分支独立编码后，在拼接点进行特征空间对齐，后续由 $N_{\text{sa}} = 9$ 层统一 Transformer 编码器处理。
- **设计动机**：层数相同保证了特征抽象层级一致，便于后续统一编码器无缝处理混合分辨率 token。

### 损失函数

总损失为多项损失的加权和：$\mathcal{L} = \lambda_{\text{map}} \mathcal{L}_{\text{map}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{pose}} \mathcal{L}_{\text{pose}} + \lambda_{\text{shape}} \mathcal{L}_{\text{shape}} + \lambda_{\text{j3d}} \mathcal{L}_{\text{j3d}} + \lambda_{\text{j2d}} \mathcal{L}_{\text{j2d}} + \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{det}} \mathcal{L}_{\text{det}}$。其中 $\mathcal{L}_{\text{map}}$ 包含尺度图的 focal loss + L1 loss，$\mathcal{L}_{\text{depth}}$ 为归一化深度 L1 loss，$\mathcal{L}_{\text{det}}$ 为检测 focal loss，其余采用 L1 距离。

## 实验关键数据

### 主实验：AGORA 测试集

| 方法 | 分辨率 | 时间 (ms) | MACs (G) | F1 ↑ | MPJPE ↓ | MVE ↓ |
|------|-------|----------|---------|------|--------|------|
| ROMP | 512 | 38.7 | 43.6 | 0.91 | 108.1 | 103.4 |
| BEV | 512 | 50.6 | 48.9 | 0.93 | 105.3 | 100.7 |
| AiOS | 1333 | 405.2 | 314.5 | 0.94 | 63.9 | 57.5 |
| Multi-HMR | 1288 | 231.7 | 6104.6 | 0.95 | 65.3 | 61.1 |
| **SAT-HMR** | **644*** | **42.0** | **133.1** | **0.95** | **67.9** | **63.3** |

### 其他数据集泛化性

| 方法 | 3DPW PA-MPJPE ↓ | MuPoTS PCK All ↑ | CMU Panoptic Avg ↓ |
|------|----------------|------------------|-------------------|
| Multi-HMR | 41.7 | 85.0 | - |
| BEV | 46.9 | 70.2 | 109.5 |
| **SAT-HMR** | **41.6** | **89.0** | **84.2** |

### 消融实验：背景 token 策略

| 策略 | 0-20% MVE | 80%+ MVE | Avg MVE |
|------|----------|---------|---------|
| 全部丢弃 | 59.9 | 70.8 | 57.2 |
| 不池化 | 60.3 | 64.1 | 56.1 |
| 池化×2 | 60.7 | 66.5 | 56.3 |
| **池化×1 (Ours)** | **60.0** | **62.7** | **56.0** |

### 关键发现

1. SAT-HMR 以 24 FPS (42ms) 实现了与 Multi-HMR (4 FPS) 可比的精度，速度提升约 5.5 倍
2. MACs 从 Multi-HMR 的 6104.6G 降至 133.1G，降低 97.8%
3. 完全丢弃背景 token 会导致大尺度人物误差从 62.7 升至 70.8，证明背景上下文的重要性
4. CMU Panoptic 上从 BEV 的 109.5 降至 84.2（提升 23.1%），展现超强泛化能力

## 亮点与洞察

- **精准的问题洞察**："高分辨率主要帮助小尺度人物"这一实证观察直接指导了方法设计
- **计算资源的智能分配**：将宝贵的高分辨率 token 集中在最需要的区域，背景压缩但不丢弃
- **首个实时 SOTA**：在 AGORA 排行榜上实现了实时级别（24 FPS）的 SOTA 性能，实用价值极高
- 方法设计简洁优雅，没有引入复杂的新模块，而是通过 token 层面的重新分配实现效果

## 局限与展望

- 尺度定义未考虑人物高度信息（如蹲下的成年人与站立的儿童 bounding box 相似），可能导致深度估计偏差
- 当前仅估计 SMPL body mesh，未扩展到 SMPL-X 全身 mesh（包括手、面部）
- 尺度阈值 $\alpha_s$ 为固定超参数，场景变化大时可能需要自适应调整
- 需要同时处理高低分辨率两张图像，内存占用仍有优化空间

## 相关工作与启发

- 与 FlexiViT/TORE 等高效 ViT 方法相比，SAT-HMR 的 token 重分配策略更具任务适应性
- 尺度自适应的思路可迁移到其他密集预测任务（如全景分割、深度估计）
- 背景池化而非丢弃的设计对所有需要全局上下文的检测/估计任务都有参考价值

## 评分

⭐⭐⭐⭐ — 问题洞察精准、方法设计简洁、实验充分，实现了实时 SOTA 的突破性成果。对高效 Vision Transformer 在人体理解任务中的应用有很好的参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](../../ECCV2024/3d_vision/multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[CVPR 2025\] MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction](mac-ego3d_multi-agent_gaussian_consensus_for_real-time_collaborative_ego-motion_.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)

</div>

<!-- RELATED:END -->
