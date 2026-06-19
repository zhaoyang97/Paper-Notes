---
title: >-
  [论文解读] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation
description: >-
  [ECCV 2024][语义分割][Event Camera] 首个无需标注的事件相机独立运动物体(IMO)分割框架，利用光流与几何约束生成伪标签训练分割网络，在 EVIMO 数据集上取得与有监督方法可比的性能。 事件相机具有高时间分辨率、高动态范围和低功耗等特性，非常适合处理需要快速响应的运动分割任务。然而…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "Event Camera"
  - "图像分割"
  - "Unsupervised"
  - "伪标签"
  - "光流"
---

# Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation

**会议**: ECCV 2024  
**arXiv**: [2312.00114](https://arxiv.org/abs/2312.00114)  
**代码**: [https://www.cis.upenn.edu/~ziyunw/un_evmoseg/](https://www.cis.upenn.edu/~ziyunw/un_evmoseg/)  
**领域**: 运动分割 / 事件相机  
**关键词**: Event Camera, Independent Motion Segmentation, Unsupervised, Pseudo-label, optical flow

## 一句话总结

首个无需标注的事件相机独立运动物体(IMO)分割框架，利用光流与几何约束生成伪标签训练分割网络，在 EVIMO 数据集上取得与有监督方法可比的性能。

## 研究背景与动机

事件相机具有高时间分辨率、高动态范围和低功耗等特性，非常适合处理需要快速响应的运动分割任务。然而，现有的事件相机运动分割方法严重依赖标注数据，标注成本极高（如 EVIMO 数据集需要 Vicon 多相机系统追踪物体并投影生成 mask）。

生物视觉系统（如昆虫捕食、人类驾驶避障）在不使用显式标签的情况下就能完成独立运动物体的检测。受此启发，作者提出了一个核心问题：**能否仅通过观察运动模式，利用事件相机在无标注情况下学习运动分割？**

现有方法的主要局限：

**混合模型方法**（如 EMSGC、EVIMO）需要预设固定的运动模型数量和参数化形式，泛化能力差

**有监督方法**（如 SpikeMS、EVDodgeNet）需要大量标注的 IMO mask

**优化方法**（如 EMSGC）需要逐场景调参，推理效率极低

## 方法详解

### 整体框架

Un-EVIMO 由两个核心模块组成：
1. **几何自标注模块**（训练阶段）：利用光流和深度信息，通过 RANSAC 估计相机运动，计算残差光流场，再通过自适应阈值生成 IMO 伪标签
2. **事件运动分割网络**（推理阶段）：仅以事件流作为输入，通过前馈 UNet 网络直接预测 IMO 二值分割 mask

关键优势：训练时需要深度和光流，但推理时仅需事件流，无需任何额外传感器输入。

### 关键设计

1. **带独立运动的光流估计**：在 DSEC 上预训练的 E-RAFT 由于训练数据缺少独立运动物体，对 IMO 区域的光流估计效果很差。作者用 RAFT 从灰度图预测的光流作为监督信号对 E-RAFT 进行微调，使其能正确估计 IMO 区域的光流。EPE 从 E-RAFT 的 11.15 降到 1.55（Table 场景）。

2. **鲁棒相机运动估计（RANSAC）**：IMO 运动与相机运动不一致，直接优化会被近距离快速运动物体偏置。作者利用完整的刚体运动场模型（6-DOF），通过 RANSAC 采样 3 个点求解线性方程 $A\theta = b$（公式 7），用 SVD 对所有 inlier 像素求解超定最小二乘问题。最大迭代 300 次或停止概率达 0.999。相机位姿估计的平移误差在 Table 和 Floor 场景下达到亚厘米级（0.0082m, 0.0075m）。

3. **自适应几何阈值化（Otsu 方法）**：残差光流 $r(q_i) = \|\Psi(q_i) - \Psi_{cam}(q_i)\|_2$ 通常呈双峰分布——一个峰对应刚体背景（低残差），另一个对应 IMO（高残差）。采用 Otsu 方法最大化两类间方差来自动选择阈值，避免不同场景噪声和深度变化导致的固定阈值失效。此外引入两级置信度过滤：若总方差过大（光流预测边界不清晰）或类间方差过小，则丢弃该训练样本。

4. **可选深度输入与参数化光流**：深度仅在训练阶段用于伪标签生成。作者还提供了无需深度的替代方案——使用 6-DOF 或 12-DOF 二次参数化光流模型（公式 10-12），虽然性能稍有下降，但仍优于 EMSGC。

### 损失函数 / 训练策略

- **Focal Loss**：由于 IMO 通常只占画面很小区域，存在严重的类别不平衡，使用 Focal Loss 替代标准交叉熵
- **事件体积表示**：使用 15 通道的事件体积，通过双线性插值核将事件分配到离散时空 bin 中，保留丰富的时间信息
- **网络架构**：UNet 结构，ResNet34 编码器（ImageNet 预训练），瓶颈层聚合全局特征用于区分全局相机运动和局部 IMO 运动
- **优化器**：Adam，学习率 $2 \times 10^{-4}$

## 实验关键数据

### 主实验

在 EVIMO 数据集上的 Event-masked IoU 评估（公式 13，40Hz 评估频率）：

| 场景 | Baseline CNN(监督) | EVIMO(监督) | SpikeMS(监督) | EMSGC Top30%(无监督) | **Un-EVIMO(无监督)** |
|------|:---:|:---:|:---:|:---:|:---:|
| Table | 66±23 | 79±6 | 50±8 | 55±17 | **50±21** |
| Box | 50±23 | 70±5 | 65±8 | 24±28 | **45±24** |
| Floor | 74±13 | 59±9 | 53±16 | 18±29 | **56±15** |
| Wall | 60±20 | 78±5 | 63±6 | 24±33 | **53±19** |
| Fast | 52±24 | 67±3 | 38±10 | 43±27 | **44±21** |

### 消融实验

| 配置 | Table | Box | Floor | Wall | Fast | 说明 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| (a) E-RAFT 未微调 | 32±23 | 28±21 | 35±19 | 42±22 | 27±23 | 预训练光流缺少 IMO |
| (b) 6-DOF 参数化 | 43±26 | 42±25 | 51±21 | 47±23 | 37±24 | 无需深度，简化模型 |
| (c) 12-DOF 参数化 | 47±24 | 40±25 | 56±18 | 49±22 | 37±25 | 更灵活的参数化 |
| **完整模型** | **50±21** | **45±24** | **56±15** | **53±19** | **44±21** | 深度+全运动场 |

### 关键发现

1. **光流质量是关键瓶颈**：使用未微调 E-RAFT 的消融实验性能大幅下降（Table 从 50 降到 32），说明对 IMO 区域的光流准确性至关重要
2. **实时推理**：Un-EVIMO 总推理时间仅 6.57ms（3.35ms 预处理 + 3.22ms 推理），远快于 EMSGC 的 9529ms 和 SpikeMS 的 120ms
3. **相机姿态估计精度高**：平移误差亚厘米级，旋转误差约 0.03 rad，证明了 flow+RANSAC 的鲁棒性
4. **合成运动模糊视频**上，有监督 RGB 方法性能严重退化（Table IoU 仅 24），凸显事件相机在高速场景的优势

## 亮点与洞察

1. **几何自标注的可扩展性**：不依赖语义信息，纯几何方法可迁移到任意场景，无需物体扫描或 Vicon 系统
2. **训练-推理解耦**：训练时需要深度和光流生成伪标签，推理时仅需事件流——一种优雅的知识蒸馏思路
3. **不假定固定物体数目**：与混合模型方法不同，本文的逐像素分类方式可自然处理任意数量的 IMO
4. **完整运动场模型**：不简化几何，使用完整的 6-DOF 刚体运动场方程，保证理论正确性

## 局限与展望

1. **缺乏时序一致性**：当前方法在单个事件切片上独立预测，连续帧的预测可能出现不一致（图 5b），可引入时序约束或 CRF
2. **静止/低速 IMO 漏检**：残差光流很小的物体难以被检测，需要结合历史运动信息
3. **边界模糊**：伪标签天然带噪，导致网络预测的 mask 边界不如有监督方法锐利
4. **深度依赖**：虽然推理不需要深度，但训练阶段最优性能仍依赖深度输入，参数化光流模型是潜在的无深度替代方案

## 相关工作与启发

- 与 EMSGC 的对比凸显了端到端学习相比逐切片优化的优势（速度提升 1400x+，且无需调参）
- 自标注思路与 Yang & Ramanan (CVPR) 基于场景流误差的自监督分割一脉相承，区别在于事件相机的高时间分辨率使光流估计更精确
- 启发：该框架可推广到其他自标注场景——任何可通过几何约束分离的信号都可用作伪标签

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将几何自标注应用于事件相机 IMO 分割，解耦训练和推理的设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ 涵盖多场景定量评估、消融实验、速度对比、失败案例分析，但仅在 EVIMO 一个数据集评估
- **写作质量**: ⭐⭐⭐⭐ 数学推导完整清晰，从运动场方程到 RANSAC 到 Otsu 阈值化的叙述逻辑通顺
- **实用价值**: ⭐⭐⭐⭐ 实时推理 + 无需标注训练，对事件相机实际应用（自动驾驶、无人机避障）有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[ECCV 2024\] ColorMAE: Exploring Data-Independent Masking Strategies in Masked AutoEncoders](colormae_exploring_data-independent_masking_strategies_in_masked_autoencoders.md)
- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](../../ICCV2025/segmentation/skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[ECCV 2024\] ActionVOS: Actions as Prompts for Video Object Segmentation](actionvos_actions_as_prompts_for_video_object_segmentation.md)

</div>

<!-- RELATED:END -->
