---
title: >-
  [论文解读] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion
description: >-
  [CVPR 2025][视频生成][轨迹引导视频生成] 提出 PoseTraj，一个姿态感知的轨迹引导视频生成模型，通过两阶段姿态感知预训练（利用合成数据集 PoseTraj-10K 和3D包围盒中间监督）和相机运动解耦微调，实现从2D轨迹生成3D对齐的旋转运动视频。 - 轨迹引导视频生成因其交互友好性受到广泛关注…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "轨迹引导视频生成"
  - "6D姿态感知"
  - "合成数据预训练"
  - "相机运动解耦"
  - "3D包围盒监督"
---

# PoseTraj: Pose-Aware Trajectory Control in Video Diffusion

**会议**: CVPR 2025  
**arXiv**: [2503.16068](https://arxiv.org/abs/2503.16068)  
**代码**: [项目主页](https://robingg1.github.io/Pose-Traj/)  
**领域**: 目标检测/视频生成  
**关键词**: 轨迹引导视频生成, 6D姿态感知, 合成数据预训练, 相机运动解耦, 3D包围盒监督

## 一句话总结

提出 PoseTraj，一个姿态感知的轨迹引导视频生成模型，通过两阶段姿态感知预训练（利用合成数据集 PoseTraj-10K 和3D包围盒中间监督）和相机运动解耦微调，实现从2D轨迹生成3D对齐的旋转运动视频。

## 研究背景与动机

- 轨迹引导视频生成因其交互友好性受到广泛关注，但现有模型难以处理涉及6D姿态变化（特别是大角度旋转）的物体运动
- DragNUWA 和 DragAnything 等方法仅在2D图像空间约束物体跟随轨迹，未考虑物体姿态的变化
- 旋转轨迹在真实视频数据中稀少且难以自动标注，模型缺乏3D理解能力
- 从2D轨迹推断潜在旋转本质上是一个不适定问题
- 真实视频中相机运动和物体运动耦合，难以准确分离
- 现有方法在大角度旋转场景下容易出现物体实体崩塌（entity collapse）
- 现有评估数据集缺乏包含旋转轨迹的精确3D标注
- SVD等预训练模型对旋转运动的理解不足

## 方法详解

### 整体框架

PoseTraj 基于 Stable Video Diffusion（SVD）构建，包含三个训练阶段：**第一阶段**在合成数据集 PoseTraj-10K 上进行3D包围盒引导预训练（同时生成物体和3D bbox）；**第二阶段**去除bbox监督，专注于物体外观细节；**第三阶段**在真实视频（VIPSeg）上进行相机运动解耦微调。推理时用户自由绘制轨迹，模型生成姿态感知的视频。核心模块 Traj-ControlNet 是 SVD 编码器块的可训练副本，接收轨迹特征并预测残差特征。

### 关键设计

**设计一：两阶段姿态感知预训练**
- **功能**：让模型感知2D轨迹中潜在的3D旋转变化
- **核心思路**：第一阶段将3D包围盒渲染到像素空间与物体一起生成，bbox提供位置和姿态信息作为中间监督；第二阶段去除bbox，微调模型专注外观细节。采用"注入式重建"（injection-by-reconstruction），即直接将bbox作为重建目标而非输入条件
- **设计动机**：直接回归3D参数困难，而将bbox生成到图像空间可以利用扩散模型的像素级重建能力，增强连续3D感知；且第二阶段可简单更换重建目标来移除bbox，避免推理时的信号不匹配

**设计二：PoseTraj-10K 合成数据集**
- **功能**：提供包含旋转轨迹和精确3D包围盒标注的大规模训练数据
- **核心思路**：从Objaverse采样2000个高质量3D模型（GPT-4v筛选+人工选择），在Blender中为每个模型生成5条随机旋转轨迹，渲染10000个视频（14帧，5fps，$320\times576$）
- **设计动机**：真实视频中旋转运动稀少且难以标注6D姿态，合成数据提供精确轨迹和3D bbox标注，避免相机运动干扰

**设计三：相机运动解耦微调 + 空间增强损失**
- **功能**：增强从合成到真实的泛化能力，分离物体运动和相机运动
- **核心思路**：在VIPSeg上微调时引入相机外参作为额外输入（通过MLP编码后与轨迹特征拼接），训练时50%概率随机丢弃相机信息。空间增强损失 $\mathcal{L}_{\text{SPA}}$ 随机采样单帧轨迹进行图像重建，仅更新空间层
- **设计动机**：合成数据中相机静止，真实视频中相机运动不可预测；空间增强损失解决大角度旋转下物体实体崩塌的问题

### 损失函数

总损失 $\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{SPA}}\mathcal{L}_{\text{SPA}}$。其中 $\mathcal{L}_{\text{MSE}}$ 为标准视频扩散去噪损失（不同阶段条件不同），$\mathcal{L}_{\text{SPA}}$ 为单帧空间重建损失，反向传播只更新空间层。

## 实验关键数据

### 主实验：轨迹引导视频生成比较

| 方法 | VIPSeg 320×576 ObjMC↓ | FID↓ | FVD↓ | DAVIS ObjMC↓ | FVD↓ |
|------|----------------------|------|------|-------------|------|
| DragNUWA 1.5 | 133.05 | 41.88 | 289.15 | 74.07 | 952.87 |
| DragAnything | 91.12 | 39.29 | 275.93 | 47.01 | 771.78 |
| **PoseTraj** | **77.48** | **38.41** | **267.33** | **29.92** | **729.16** |

### 消融实验：预训练设计影响

| 变体 | ObjMC↓ | FID↓ | FVD↓ |
|------|--------|------|------|
| Full method | 77.48 | 38.41 | 267.33 |
| No bbox stage | 81.36 | 41.90 | 275.40 |
| No pretrain | 145.72 | 42.62 | 486.84 |
| No Cam-disen | 83.22 | 39.71 | 279.15 |
| No SPA-loss | 137.26 | 39.79 | 436.56 |

### 关键发现
- PoseTraj在VIPSeg上轨迹精度（ObjMC）较DragAnything提升15%，DAVIS上提升36%
- 去除两阶段预训练后ObjMC退化至145.72（+88%），空间增强损失同样关键（+77%退化）
- 3D bbox预训练主要影响物体姿态定位的视觉准确性，定量指标上影响相对较小
- 用户评估中PoseTraj在轨迹跟随准确度和视觉质量上分别获43%和39%更多投票

## 亮点与洞察

1. **合成数据+3D bbox中间监督的预训练策略**：巧妙利用合成数据解决真实视频中旋转标注稀缺的问题
2. **注入式重建范式**：将3D信息作为重建目标而非输入条件，训练后可无缝切换
3. **空间增强损失**：通过单帧子任务显著改善大角度旋转下的物体保持能力
4. **OOD泛化能力强**：未在DAVIS上训练但用于评估，表现优异

## 局限与展望

- 合成数据中的物体种类（2000个）和渲染质量仍有局限
- 对于非刚体变形（如人体运动）的旋转控制尚未验证
- 相机运动解耦模块在推理时不使用相机位姿，可能限制其效果
- 未来可探索基于DiT架构（如Tora）的姿态感知扩展

## 相关工作与启发

- 与DragAnything使用分割掩码提取实体特征不同，PoseTraj通过3D预训练内化3D理解
- PuppetMaster在部件级动画上类似使用合成数据，PoseTraj聚焦于旋转轨迹
- 空间增强损失的思路可推广到其他需要空间一致性保证的视频生成任务

## 评分

⭐⭐⭐⭐ — 预训练策略设计合理有效，合成数据+3D bbox中间监督是亮点；但旋转场景的实际应用需求相对小众。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[ICLR 2026\] Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control](../../ICLR2026/video_generation/learning_video_generation_for_robotic_manipulation_with_collaborative_trajectory.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)

</div>

<!-- RELATED:END -->
