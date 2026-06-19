---
title: >-
  [论文解读] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions
description: >-
  [ICCV 2025][目标检测][人体姿态估计] 提出 VOccl3D，一个基于3DGS渲染的大规模合成视频数据集（25万帧，400视频序列），专注于真实遮挡场景的3D人体姿态与形状估计，在该数据集上微调的模型显著提升了遮挡场景下的HPS性能。 3D人体姿态与形状估计（HPS）方法在标准场景下已经表现优异…
tags:
  - "ICCV 2025"
  - "目标检测"
  - "人体姿态估计"
  - "遮挡"
  - "合成数据集"
  - "3D Gaussian Splatting"
  - "SMPL-X"
---

# VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions

**会议**: ICCV 2025  
**arXiv**: [2508.06757](https://arxiv.org/abs/2508.06757)  
**代码**: [https://yashgarg98.github.io/VOccl3D-dataset/](https://yashgarg98.github.io/VOccl3D-dataset/)  
**领域**: 目标检测  
**关键词**: 人体姿态估计, 遮挡, 合成数据集, 3D Gaussian Splatting, SMPL-X

## 一句话总结

提出 VOccl3D，一个基于3DGS渲染的大规模合成视频数据集（25万帧，400视频序列），专注于真实遮挡场景的3D人体姿态与形状估计，在该数据集上微调的模型显著提升了遮挡场景下的HPS性能。

## 研究背景与动机

3D人体姿态与形状估计（HPS）方法在标准场景下已经表现优异，但在**严重遮挡**场景中仍然困难重重。核心问题在于：

**现有遮挡数据集不够真实**: 多数数据集使用随机色块/剪贴画遮挡（如 3DPW-AdvOcc），远非真实世界的遮挡形态

**自然遮挡数据集遮挡程度不足**: 如3DPW和OCMotion中遮挡稀疏且不频繁

**缺乏大规模、多样化、高遮挡程度的训练数据**: 导致模型无法学习遮挡场景下的鲁棒估计

VOccl3D 的目标是填补这一空白：提供具有真实遮挡的大规模合成视频数据集。

## 方法详解

### 整体框架

数据集构建流程：
1. 使用 **3D Gaussian Splatting** 从 DL3DV 数据集的真实视频中学习40个背景场景的3D表示
2. 从 **AMASS** 动捕数据集采样约400个 SMPL-X 人体运动序列
3. 从 **SMPLitex** 数据集选取约200种人体纹理
4. 在 **Unity** 引擎中整合渲染：3DGS背景 + 人体动画 + 纹理 → 合成视频

### 关键设计

1. **3DGS 背景场景**: 不同于 BEDLAM 等使用图形资产的方法，VOccl3D 利用3D高斯飞溅从真实环境视频中学习3D表示。这带来两个优势：

    - 渲染场景更接近真实数据分布
    - 可灵活创建特定领域数据集（农业、街道、室内等），仅需原始RGB视频
    - 从 DL3DV 数据集挑选了包含自然遮挡物（花环、椅子、长凳、雕塑、汽车、垃圾桶等）的场景

2. **运动序列处理**: 

    - 使用 VPoser 评估姿态难度，筛选具有挑战性的复杂姿态（$\|\epsilon_\theta\|_2 > 40$）
    - 确保每个序列至少180帧（6秒 @30fps）
    - 实施边界约束防止人体移出遮挡区域
    - 应用随机旋转和平移增加多样性

3. **数据集属性与标注**:

    - 250,000+ 帧，400 视频序列，总时长 > 2.5小时
    - 40个背景场景，每个场景10个视频
    - 提供完整标注：相机内外参、SMPL-X姿态/形状参数、全局朝向、平移、性别、2D关键点
    - **关键点级别遮挡标注**: 每个关键点的二值遮挡标签
    - 三级遮挡分类：硬遮挡（4-9可见关键点）、中等遮挡（10-15）、低遮挡（16-20）

4. **多模态标注**: 除3D姿态外还支持人体轮廓、身体部位分割、2D关键点、人体检测框

### 损失函数 / 训练策略

- 基于 VOccl3D 微调 CLIFF 和 BEDLAM-CLIFF → VOccl3D-CLIFF 和 VOccl3D-B-CLIFF
- 约200k训练图像 + 50k测试图像
- 使用五折交叉验证报告平均性能
- 同时微调 YOLO11 人体检测器 → VOccl3D-YOLO11

## 实验关键数据

### 主实验 (表格)

**VOccl3D 测试集 (Ground-truth bounding boxes)：**

| Method | Hard-Occ MPJPE | Hard-Occ PA-MPJPE | Med-Occ MPJPE | Med-Occ PA-MPJPE | Low-Occ MPJPE | Low-Occ PA-MPJPE |
|--------|---------------|-------------------|---------------|-------------------|---------------|-------------------|
| CLIFF | 192.22 | 114.35 | 121.70 | 78.56 | 98.82 | 67.64 |
| BEDLAM-CLIFF | 154.86 | 99.53 | 90.97 | 65.03 | 74.95 | 52.65 |
| HMR2.0 | 169.71 | 100.49 | 113.88 | 71.78 | 88.53 | 59.08 |
| WHAM | 152.15 | 102.14 | 110.97 | 76.81 | 93.90 | 66.68 |
| **VOccl3D-B-CLIFF** | **136.34** | **89.94** | **82.48** | **58.78** | **69.46** | **46.32** |

VOccl3D-B-CLIFF 在 Hard-Occlusion MPJPE 上比 BEDLAM-CLIFF 降低约 **18.5mm**。

### 消融实验 (表格)

**真实数据集评估（检测器影响）：**

| Method | 3DPW MPJPE | 3DPW PA-MPJPE | OCMotion MPJPE | OCMotion PA-MPJPE |
|--------|-----------|---------------|---------------|-------------------|
| VOccl3D-CLIFF w/ GT Box | 71.10 | 45.98 | 64.29 | 39.64 |
| VOccl3D-CLIFF w/ YOLO11 | 116.52 | 63.35 | 67.16 | 41.30 |
| VOccl3D-CLIFF w/ VOccl3D-YOLO11 | **114.85** | **62.66** | **66.65** | **41.15** |

**遮挡增强的3DPW变体：**

| Method | 3DPW MPJPE | OcclType1-3DPW MPJPE | OcclType2-3DPW MPJPE |
|--------|-----------|---------------------|---------------------|
| CLIFF | 73.9 | 98.15 | 99.49 |
| BEDLAM-CLIFF | 72.0 | 98.71 | 96.80 |
| **VOccl3D-B-CLIFF** | **72.0** | **95.89** | **93.74** |

### 关键发现

- **遮挡程度越高差距越大**: 所有方法从低遮挡到高遮挡性能均大幅下降，但 VOccl3D 微调模型在高遮挡下优势最显著
- **合成数据迁移到真实场景有效**: 在3DPW和OCMotion（真实数据集）上也能取得优于或持平的性能
- **检测器是HPS瓶颈**: GT box 与 YOLO11 检测框导致 3DPW 上 MPJPE 差距约45mm，凸显检测器在遮挡下的重要性
- **STRIDE + VOccl3D 伪标签协同**: 使用 VOccl3D-B-CLIFF 的伪标签比原始 BEDLAM-CLIFF 伪标签效果更好

## 亮点与洞察

- **3DGS 替代传统图形资产**: 用3D高斯飞溅学习真实场景表示，使背景更逼真且制作成本更低
- **关键点级遮挡标注**: 比图像级遮挡标签更精细，支持更深入的遮挡分析
- **多任务基准**: 同一数据集支持HPS估计、检测、分割多任务评估
- **端到端分析**: 从检测器到姿态估计器的完整链路分析，揭示了检测器对HPS的关键影响

## 局限与展望

- 合成数据与真实数据仍存在域差距（lighting, texture, physics）
- 仅使用 SMPL-X 模型，缺少手部/面部精细估计
- 背景场景数量（40个）相对有限，可以扩展更多多样化场景
- 未考虑动态遮挡物（如移动的车辆或其他行人）
- 人体-遮挡物的物理交互（如碰撞）未建模

## 相关工作与启发

- 与 BEDLAM 一脉相承但聚焦遮挡，填补了遮挡数据集的空白
- 3DGS 背景生成方法可推广到其他合成数据需求（如自动驾驶、机器人）
- 检测器微调带来的级联提升提示：在下游任务中需要全链路优化
- 可与扩散模型结合生成更多样的遮挡场景

## 评分

- **新颖性**: ⭐⭐⭐ 方法本身创新有限，主要贡献在数据集构建
- **实验充分度**: ⭐⭐⭐⭐ 多模型、多数据集、多遮挡级别、检测器影响分析全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据集构建细节充分
- **价值**: ⭐⭐⭐⭐ 填补了HPS领域遮挡数据集的重要空白，可作为标准基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ProbPose: A Probabilistic Approach to 2D Human Pose Estimation](../../CVPR2025/object_detection/probpose_a_probabilistic_approach_to_2d_human_pose_estimation.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](yoloe_realtime_seeing_anything.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](kaputt_a_large-scale_dataset_for_visual_defect_detection.md)

</div>

<!-- RELATED:END -->
