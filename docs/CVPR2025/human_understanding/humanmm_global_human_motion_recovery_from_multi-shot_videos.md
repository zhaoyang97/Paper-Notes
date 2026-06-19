---
title: >-
  [论文解读] HumanMM: Global Human Motion Recovery from Multi-shot Videos
description: >-
  [CVPR 2025][人体理解][人体运动恢复] HumanMM首次提出从多镜头视频中恢复世界坐标系下3D人体运动的框架，通过镜头转换检测器、增强SLAM、基于立体标定的朝向对齐和运动积分器，实现了跨镜头的连续运动重建。 领域现状：3D人体运动恢复（HMR）已取得显著进展，HMR2.0等方法在相机坐标系下表现出色…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "人体运动恢复"
  - "多镜头视频"
  - "世界坐标系"
  - "镜头转换对齐"
  - "SLAM"
---

# HumanMM: Global Human Motion Recovery from Multi-shot Videos

**会议**: CVPR 2025  
**arXiv**: [2503.07597](https://arxiv.org/abs/2503.07597)  
**代码**: 有（项目页面提供）  
**领域**: 人体理解  
**关键词**: 人体运动恢复, 多镜头视频, 世界坐标系, 镜头转换对齐, SLAM

## 一句话总结
HumanMM首次提出从多镜头视频中恢复世界坐标系下3D人体运动的框架，通过镜头转换检测器、增强SLAM、基于立体标定的朝向对齐和运动积分器，实现了跨镜头的连续运动重建。

## 研究背景与动机

**领域现状**：3D人体运动恢复（HMR）已取得显著进展，HMR2.0等方法在相机坐标系下表现出色。WHAM、GVHMR等方法通过集成SLAM估计相机参数，进一步实现了世界坐标系下的运动恢复。但这些方法都聚焦于单镜头视频。

**现有痛点**：大量在线视频（体育转播、谈话节目、音乐会等）都是多镜头拍摄的，包含镜头转换。将多镜头视频按镜头切割会大幅缩短序列长度（现有数据集最长片段不到20秒），这对需要长序列的任务（如长运动生成）极为不利。直接处理多镜头视频面临两个基本挑战。

**核心矛盾**：多镜头视频中的人体运动在物理上是连续的，但镜头转换导致的视角突变使得相机坐标系下的运动估计在转换点处不连续——需要解决"运动连续性"与"视角不连续性"之间的矛盾。

**本文目标** (1) 如何在镜头转换时对齐世界坐标系中的人体朝向和运动？(2) 如何在世界坐标系中重建准确的人体运动（解决脚部滑动、时序一致性等问题）？

**切入角度**：作者基于一个关键观察——多镜头视频中人体运动在镜头转换期间通常是连续的，只是相机视角发生了变化。因此，通过估计转换帧间的相对相机旋转，可以将不同镜头的运动对齐到统一的世界坐标系。

**核心 idea**：通过镜头检测+增强SLAM+基于2D关键点的相机标定对齐朝向+跨镜头Transformer平滑姿态，实现多镜头视频的世界坐标系运动恢复。

## 方法详解

### 整体框架
输入为包含多次镜头转换的长视频$\mathbf{V}=\{I_t\}_{t=1}^T$，输出为世界坐标系下的SMPL运动参数。流程分五步：(1) 提取运动特征并检测镜头转换帧；(2) 对每个单镜头片段用Masked LEAP-VO估计相机位姿，用GVHMR恢复初始运动；(3) 通过相机标定对齐跨镜头的人体朝向；(4) 通过ms-HMR Transformer平滑跨镜头的人体姿态；(5) 通过运动积分器（BiLSTM + 轨迹优化器）恢复轨迹并消除脚部滑动。

### 关键设计

1. **镜头转换检测器（Shot Transition Detector）**:

    - 功能：准确识别视频中的镜头转换帧
    - 核心思路：串联使用三个互补模块——(1) SceneDetect检测明显的场景背景变化；(2) 边界框追踪检测人物尺寸突变（计算相邻帧bbox的IoU，低于阈值即判定转换）；(3) 人体关键点追踪检测细粒度的姿态/朝向转换（计算相邻帧对应关键点的IoU）。三个模块串行协作，覆盖从粗到细的各类镜头转换情况。
    - 设计动机：单一检测器无法覆盖所有类型的转换——SceneDetect无法处理背景相似的视角切换，bbox无法处理尺寸不变但姿态变化的情况，必须多模块互补。

2. **增强相机位姿估计 + 朝向对齐（Masked LEAP-VO + Orientation Alignment）**:

    - 功能：精确估计每个镜头的相机轨迹，并跨镜头对齐人体朝向
    - 核心思路：基于LEAP-VO改进——先用SAM生成人体mask，将mask内的特征点设为不可见，排除动态人体对bundle adjustment的干扰（"Masked LEAP-VO"）。对于跨镜头朝向对齐，提出朝向对齐模块（OAM）：基于**假设1**（镜头转换时人体朝向和位移在世界坐标系中连续），将朝向对齐问题转化为估计相对相机旋转$\mathbf{R}_{\delta_{cam}}$。具体做法是提取转换帧两侧的2D关键点，用RANSAC筛选匹配点，通过本质矩阵$\mathbf{E}=[\mathbf{T}]_\times \mathbf{R}$的SVD分解求解相对旋转。
    - 设计动机：DROID-SLAM等方法在人体遮挡大量画面时mask后特征点过少导致不准确。LEAP-VO利用CoTracker进行长程特征追踪，mask后仍保留足够信息。朝向对齐不直接mask人体而是利用人体关键点作为显式特征匹配，因为此时人体是唯一跨镜头的可靠对应。

3. **多镜头HMR编码器（ms-HMR）+ 运动积分器**:

    - 功能：跨镜头平滑人体姿态，恢复轨迹并消除脚部滑动
    - 核心思路：ms-HMR是一个Transformer编码器，输入为跨所有镜头的初始运动参数$\{\theta_t\}_{t=1}^T$（包含镜头索引位置编码），输出为refined运动参数$\{\phi_t\}_{t=1}^T$。训练时在根姿态上添加随机旋转噪声（0-1 radian）模拟镜头转换的不准确性。运动积分器使用双向LSTM预测脚-地面接触概率和根速度，再用轨迹优化器（扩展自WHAM）消除脚部滑动。
    - 设计动机：镜头转换导致部分遮挡，不同镜头中可见的身体部位互补，Transformer的全局注意力机制可以跨镜头利用这种互补信息。加入噪声的训练策略使模型对镜头转换引入的不准确性具有鲁棒性。

### 损失函数 / 训练策略
ms-HMR、轨迹预测器和脚部滑动优化器在AMASS、3DPW、Human3.6M和BEDLAM数据集上训练80个epoch。接触概率和速度用MSE loss监督。训练时添加随机旋转噪声和身体姿态噪声模拟镜头转换误差。

## 实验关键数据

### 主实验（ms-Motion数据集，2-shot设置）

| 数据集 | 方法 | PA-MPJPE↓ | WA-MPJPE↓ | RTE↓ | ROE↓ | Foot Sliding↓ |
|--------|------|-----------|-----------|------|------|--------------|
| ms-AIST | GVHMR | 60.72 | 231.36 | 6.20 | 96.58 | 7.65 |
| ms-AIST | WHAM | 65.34 | 336.82 | 4.39 | 84.48 | 2.75 |
| ms-AIST | **Ours** | **36.82** | **121.35** | **2.56** | **69.23** | **2.66** |
| ms-H3.6M | GVHMR | 64.63 | 254.30 | 6.94 | 81.93 | 8.80 |
| ms-H3.6M | **Ours** | **40.52** | **132.13** | **3.65** | **53.39** | **4.17** |

### 不同镜头数量的性能变化

| 设置 | PA-MPJPE↓ (ms-AIST) | WA-MPJPE↓ | ROE↓ |
|------|---------------------|-----------|------|
| 2-shot | 36.82 | 121.35 | 69.23 |
| 3-shot | 38.52 | 141.38 | 67.71 |
| 4-shot | 39.63 | 161.52 | 70.31 |

### 关键发现
- HumanMM在PA-MPJPE上比GVHMR降低了39%（ms-AIST），在ROE上降低了28%，说明跨镜头对齐和姿态平滑效果显著
- 随着镜头数量增加（2→3→4），PA-MPJPE仅增加7.6%，说明方法对多镜头数量具有较好的可扩展性
- RTE（轨迹误差）大幅优于所有baseline，验证了Masked LEAP-VO和轨迹优化器的有效性
- SLAHMR在所有指标上最差，说明现有单镜头方法直接应用于多镜头场景完全不可行
- 脚部滑动指标与WHAM接近，说明运动积分器在消除滑动方面效果好

## 亮点与洞察
- **问题定义的开创性**：首次正式定义并解决多镜头视频的世界坐标系运动恢复问题，填补了重要空白。在线视频的很大比例是多镜头的，这个问题的解决能极大扩展运动数据的可用规模。
- **朝向对齐的巧妙设计**：将跨镜头朝向对齐转化为相对相机旋转估计，利用人体2D关键点作为跨镜头对应点进行相机标定——在正常SLAM中人体是需要排除的动态物体，但在跨镜头对齐中恰恰是唯一可靠的对应。
- **ms-Motion数据集构建**：利用AIST和H3.6M的多机位数据构造多镜头评测集，方法巧妙且为后续研究提供了标准benchmark。

## 局限与展望
- 假设镜头转换时人体朝向连续，在某些快速动作场景下可能不成立
- ms-Motion数据集是合成的多镜头视频（从多机位数据拼接），与真实在线多镜头视频仍有分布差异
- 镜头转换检测依赖手动调节的IoU阈值，鲁棒性有限
- 改进方向：引入预训练视觉模型增强镜头转换检测、支持多人多镜头场景、直接在真实在线视频上训练和评测

## 相关工作与启发
- **vs GVHMR**: GVHMR是当前最好的单镜头世界坐标HMR方法，但直接应用于多镜头时因不处理镜头转换导致朝向跳变和轨迹断裂；HumanMM在其基础上增加了对齐和平滑模块
- **vs WHAM**: WHAM用DROID-SLAM估计相机，在人体遮挡大时特征不足；HumanMM用Masked LEAP-VO利用长程追踪保留更多特征
- **vs Pavlakos et al.**: 之前唯一的多镜头工作，但仅处理相机坐标系下的远景-近景转换，无法处理世界坐标系和任意镜头转换
- 可作为运动生成任务的数据获取工具，从海量在线视频中自动提取长序列运动数据

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义并解决多镜头视频世界坐标HMR问题，意义重大
- 实验充分度: ⭐⭐⭐⭐ 自建benchmark评测全面，但真实在线视频的定性结果有限
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法流程完整，但pipeline较复杂
- 价值: ⭐⭐⭐⭐⭐ 填补了重要空白，能直接扩展运动数据规模，对运动生成等下游任务有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance](reperformer_immersive_human-centric_volumetric_videos_from_playback_to_photoreal.md)
- [\[CVPR 2025\] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)
- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[CVPR 2026\] Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining](../../CVPR2026/human_understanding/mocap-2-to-3_multi-view_lifting_for_monocular_motion_recovery_with_2d_pretrainin.md)
- [\[CVPR 2025\] Human Motion Instruction Tuning](human_motion_instruction_tuning.md)

</div>

<!-- RELATED:END -->
