---
title: >-
  [论文解读] egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks
description: >-
  [ICCV 2025][视频理解][egocentric vision] 提出egoPPG这一新的自中心视觉任务，通过PulseFormer方法从未修改的自中心头戴设备的眼部追踪摄像头估计心率（MAE=7.67 bpm），并证明心率估计在EgoExo4D的技能水平评估下游任务中可提升14.1%的准确率。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "egocentric vision"
  - "heart rate estimation"
  - "rPPG"
  - "eye tracking"
  - "physiological sensing"
---

# egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks

**会议**: ICCV 2025  
**arXiv**: [2502.20879](https://arxiv.org/abs/2502.20879)  
**代码**: [https://siplab.org/projects/egoPPG](https://siplab.org/projects/egoPPG)  
**领域**: 视频理解  
**关键词**: egocentric vision, heart rate estimation, rPPG, eye tracking, physiological sensing

## 一句话总结
提出egoPPG这一新的自中心视觉任务，通过PulseFormer方法从未修改的自中心头戴设备的眼部追踪摄像头估计心率（MAE=7.67 bpm），并证明心率估计在EgoExo4D的技能水平评估下游任务中可提升14.1%的准确率。

## 研究背景与动机
自中心视觉系统（如MR/AR眼镜）旨在理解用户的空间环境和行为，包括运动、活动和交互。→ 但现有系统忽视了一个关键维度：生理状态的检测，如心率（HR）反映注意力、认知表现、情绪、压力和疲劳。→ 虽然Meta的Project Aria 2引入了接触式HR传感器，但大量已有设备和已录制数据集（如EgoExo4D）并不具备这一功能。→ 核心idea：利用自中心头戴设备上已有的眼部追踪摄像头，从眼周区域皮肤的细微光强变化提取光电容积脉搏（PPG）信号，结合IMU运动信息，实现非接触式HR估计。

## 方法详解

### 整体框架
PulseFormer以眼部追踪视频（连续帧差分标准化）和IMU数据为输入，通过3D CNN主干网络（PhysNet）处理，配合运动感知时间注意力（MITA）和空间注意力（SA）模块，输出BVP（血容量脉搏）信号，再通过Butterworth滤波和峰值检测计算HR。输入为T=128帧（4.3秒），分辨率下采样至48×128。

### 关键设计
1. **运动感知时间注意力（MITA）模块**:

    - 功能：利用IMU数据为每帧视频分配基于运动强度的时间权重
    - 核心思路：使用ResNet18+线性层从视频帧提取图像嵌入$\mathbf{F_e} \in \mathbb{R}^{T \times D}$，使用两层1D卷积从IMU提取嵌入$\mathbf{I_e} \in \mathbb{R}^{T \times D}$，通过交叉注意力计算：$\mathbf{A} = \text{softmax}(\frac{\mathbf{QK}^\top}{\sqrt{D}})\mathbf{V}$，其中IMU嵌入作为Q，图像嵌入作为K和V。输出时间注意力$\mathbf{T} \in \mathbb{R}^{T \times 1 \times 1 \times 1}$与输入逐帧相乘
    - 设计动机：自中心设备在用户运动时会产生严重运动伪影，MITA让模型能降低运动剧烈帧的权重（如跳舞时MAE从10.54降至7.85 bpm）

2. **空间注意力（SA）模块**:

    - 功能：让网络自动聚焦于高信噪比的眼周皮肤区域，抑制眼球区域的低SNR影响
    - 核心思路：在每个pooling前插入空间注意力模块，使用平均池化和最大池化特征图经7×7卷积生成空间注意力图：$\mathbf{M_s}(\mathbf{F}) = \sigma(f^{7 \times 7}([\mathbf{F_{avg}}; \mathbf{F_{max}}]))$
    - 设计动机：虽然球结膜（白眼部分）含丰富血管，但眼球运动和眨眼导致极大噪声；眼周皮肤运动少、血管分布适合PPG信号提取。学习到的空间注意力图也验证了这一发现——模型自动排除了眼球区域

3. **数据增强策略**:

    - 功能：应对不同用户佩戴眼镜时眼部区域可见性的差异
    - 核心思路：随机旋转（±20度）、随机水平裁剪、水平和垂直翻转
    - 设计动机：眼镜佩戴位置因人而异，摄像头可能只捕捉到上方、下方或倾斜的眼部区域

### 损失函数 / 训练策略
使用MSE损失，标签为PPG信号的标准化连续差分。5折交叉验证按参与者划分，保证训练/验证/测试集严格分离。Batch size=4，100 epochs，学习率0.0009。在RTX 4090上约20小时完成所有折叠的训练。模型仅~12M参数，推理速度2.9k fps（RTX 4090）或180 fps（CPU）。

## 实验关键数据

### 主实验

| 方法 | MAE↓ | RMSE↓ | MAPE↓ | r↑ |
|------|------|-------|-------|-----|
| DeepPhys | 28.26 | 31.97 | 36.68 | 0.08 |
| PhysNet | 12.09 | 15.43 | 15.14 | 0.66 |
| PhysFormer | 10.71 | 13.97 | 12.69 | 0.72 |
| FactorizePhys (SOTA) | 10.07 | 13.43 | 12.36 | 0.67 |
| PulseFormer w/o SA | 10.49 | 13.62 | 12.83 | 0.73 |
| PulseFormer w/o MITA | 8.82 | 12.03 | 10.82 | 0.81 |
| **PulseFormer (ours)** | **7.67** | **10.69** | **9.45** | **0.85** |
| 相比次优改进 | -2.40 | -2.74 | -2.91 | +0.13 |

下游任务 - EgoExo4D技能评估（Top-1准确率）：

| 配置 | Basketball | Cooking | Dancing | Overall |
|------|-----------|---------|---------|---------|
| Ego only | 45.45 | 20.00 | 43.44 | 39.69 |
| **Ego + HR** | 47.47 | **40.00** | **53.27** | **45.29** |
| Ego + Exo | 49.49 | 25.00 | 50.82 | 39.00 |
| **Ego + Exo + HR** | **50.50** | **40.00** | **59.84** | **43.94** |

### 消融实验

| 配置 | MAE↓ | 说明 |
|------|------|------|
| Baseline 信号处理(皮肤) | 12.40 | 手动定义皮肤区域+信号处理 |
| Baseline 信号处理(眼部) | 14.60 | 手动定义眼部区域+信号处理 |
| PulseFormer w/o SA | 10.49 | 去除空间注意力，MAE增加2.82 |
| PulseFormer w/o MITA | 8.82 | 去除运动注意力，MAE增加1.15 |
| **PulseFormer** | **7.67** | 完整模型 |

按活动分析（MAE）：

| 活动 | 平均HR | 运动强度 | PulseFormer | w/o MITA |
|------|--------|---------|-------------|----------|
| 看视频 | 71.5 | 0 | 5.52 | 5.97 |
| 办公 | 75.7 | 0.45 | 7.50 | 8.22 |
| 厨房 | 85.3 | 0.54 | 7.22 | 8.89 |
| 跳舞 | 89.1 | 1.00 | 7.85 | 10.54 |
| 骑车 | 113.1 | 0.77 | 12.91 | 14.62 |

帧率影响：

| 帧率 | MAE↓ | r↑ |
|------|------|-----|
| 30 fps（原始） | 7.67 | 0.85 |
| 10 fps（下采样） | 11.13 | 0.70 |
| 10→30 fps（插值上采样） | 10.18 | 0.77 |

### 关键发现
- 眼周皮肤区域比眼球区域的PPG信号质量显著更好（信号处理baseline: 12.40 vs 14.60 MAE）
- MITA在跳舞活动中效果最显著（MAE从10.54降至7.85），说明IMU运动信息对抑制运动伪影至关重要
- HR信息对烹饪（20%→40%）和跳舞（43%→53%）的技能评估提升最大
- 10 fps帧率严重降低性能，但线性插值上采样可以部分恢复

## 亮点与洞察
- 开创性地提出egoPPG任务，将生理感知引入自中心视觉系统，扩展了该领域的研究范围
- 实用性强：无需修改硬件，利用现有眼部追踪摄像头即可工作
- 数据集egoPPG-DB（13小时，25参与者）包含丰富的日常活动和HR变化，贡献了ground-truth PPG和ECG双重验证
- 下游任务验证有说服力：HR信息确实能提升视觉任务性能

## 局限与展望
- 数据集规模相对有限（25参与者），人口统计学多样性有待扩展
- 高心率（如骑车113 bpm）下MAE仍较高（12.91 bpm），可能需要更好的高HR建模
- 当前仅在EgoExo4D的技能评估上验证了下游效果，可探索更多下游任务（如情绪识别、注意力检测）
- 10 fps的性能下降问题影响对已有低帧率数据集的适用性

## 相关工作与启发
- 将rPPG技术从面部视频扩展到眼部追踪视频是一个创新性的应用转移
- 利用IMU融合视频信息的多模态方法可推广到其他体征估计任务
- egoPPG为已有大规模数据集（EgoExo4D、Nymeria）提供生理状态增强，是一种有价值的数据增强范式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全新的egoPPG任务定义，从眼部追踪器估计心率极具创意
- 实验充分度: ⭐⭐⭐⭐ 数据集+方法+下游任务全链条验证，消融分析详细
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、贡献明确、实验设计严谨
- 价值: ⭐⭐⭐⭐ 开创新任务+开源数据集+下游任务验证，对自中心视觉社区有重要影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)

</div>

<!-- RELATED:END -->
