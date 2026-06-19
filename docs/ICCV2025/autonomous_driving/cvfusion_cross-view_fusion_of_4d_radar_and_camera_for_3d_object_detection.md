---
title: >-
  [论文解读] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection
description: >-
  [ICCV 2025][自动驾驶][4D毫米波雷达] 提出CVFusion——首个4D雷达-相机两阶段融合网络，第一阶段通过雷达引导迭代（RGIter）BEV融合生成高召回率提案框，第二阶段利用点引导融合（PGF）和网格引导融合（GGF）聚合多视角异构特征进行提案精化，在VoD和TJ4DRadSet上分别取得9.10%和3.68%的mAP提升。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "4D毫米波雷达"
  - "相机融合"
  - "3D目标检测"
  - "两阶段检测"
  - "BEV融合"
---

# CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection

**会议**: ICCV 2025  
**arXiv**: [2507.04587](https://arxiv.org/abs/2507.04587)  
**代码**: [https://github.com/zhzhzhzhzhz/CVFusion](https://github.com/zhzhzhzhzhz/CVFusion)  
**领域**: 自动驾驶  
**关键词**: 4D毫米波雷达, 相机融合, 3D目标检测, 两阶段检测, BEV融合

## 一句话总结

提出CVFusion——首个4D雷达-相机两阶段融合网络，第一阶段通过雷达引导迭代（RGIter）BEV融合生成高召回率提案框，第二阶段利用点引导融合（PGF）和网格引导融合（GGF）聚合多视角异构特征进行提案精化，在VoD和TJ4DRadSet上分别取得9.10%和3.68%的mAP提升。

## 研究背景与动机

4D毫米波雷达因其在恶劣天气下的鲁棒性（雨、雾、雪）受到自动驾驶领域的广泛关注。与传统3D雷达相比，4D雷达额外提供了高程测量，可产生类似LiDAR的3D点云。但4D雷达存在两大固有问题：(1) **极度稀疏**——每帧仅0.1k-2k个点（LiDAR为10k-100k）；(2) **噪声多**——毫米波的多径效应导致测量不精确。因此仅依赖4D雷达的检测效果不佳，需与相机融合。

现有4D雷达-相机融合方法（RCFusion、LXL等）均采用**单阶段流水线**，将两种模态分别转换到BEV空间后直接融合。但这存在**特征错位**问题：单目图像的深度模糊导致前视图到BEV的投影偏移，而稀疏且有噪声的雷达点不足以提供准确的3D位置信息弥补图像深度不确定性。直接拼接两种BEV特征会导致错位，特别是对小目标影响更大，容易产生位置偏移和假阳性。

作者指出：任务不应一步到位完成，而应分阶段逐步精化。两阶段范式在LiDAR检测中已被证明有效（PV-RCNN等），但从未被引入4D雷达-相机融合——主要困难在于雷达太稀疏，如何设计提案精化中的有效特征聚合非常关键。

## 方法详解

### 整体框架

CVFusion分为两个阶段：
- **Stage 1**: 4D雷达和图像分支分别提取特征并转换到BEV平面，通过RGIter-BEV融合模块融合后送入RPN生成提案框。
- **Stage 2**: 对每个提案框，通过PGF（点引导融合）和GGF（网格引导融合）两个分支聚合雷达点、图像前视图和BEV三种视角的特征，进行精化预测。

### 关键设计

1. **雷达引导迭代BEV融合（RGIter-BEV）**: 针对图像BEV特征的深度不确定性，利用多尺度雷达BEV特征 $F_{B,k}^{Rad}$（$k=0,1,2$）生成占据概率权重图 $W_k = \text{Sigmoid}(\text{Conv2d}(F_{B,k}^{Rad}))$，对相机BEV特征进行加权：$F_{B,k}^{Cam'} = F_{B,k}^{Cam} \odot W_k$。这一加权强化了图像特征中有雷达支持的位置，抑制了深度不确定区域。关键创新在于**迭代式多尺度处理**：加权后的图像特征通过卷积+下采样生成下一尺度的输入：$F_{B,k+1}^{Cam} = \text{Conv2d}(F_{B,k}^{Cam'}, \text{stride}=2)$，确保雷达引导的位置信息在不同尺度间传播。最终三个尺度的融合特征通过上/下采样统一后拼接。

2. **点引导融合（PGF）**: 处理提案框内有雷达点的情况。对每个雷达点 $p$，将其体素特征 $f_p$ 投影到2D图像平面获取位置 $r_p$，然后通过**跨模态可变形注意力（CMDA）**查询附近图像特征：$f_I^* = \text{CMDA}(f_p, r_p, F_I)$。CMDA使用点特征作为Query，图像特征作为Key/Value，通过可学习的采样偏移和注意力权重聚合局部图像信息。查询结果与点特征拼接后通过MLP融合：$f_p^* = \text{MLP}([f_p, f_I^*])$。额外引入**核密度估计（KDE）**特征区分孤立噪声点，最后通过RoI-Pooling（$U^3$网格）聚合为提案级特征 $f_b^{pt}$。

3. **网格引导融合（GGF）**: 解决大量提案框内无雷达点的问题。将提案框划分为 $U^3$ 个均匀网格（与雷达点无关），每个网格通过**网格位置编码器（GPE）**生成初始特征：$f_{g_j}^{pos} = \text{MLP}(\delta_j, c_b, \log(|N_{g_j}| + \epsilon))$，编码了网格到框中心的相对位置和网格内点数。然后通过**两级CMDA串联**聚合特征：第一个CMDA将网格投影到图像平面查询前视图特征 $f_{g_j}^{fv}$；第二个CMDA将 $f_{g_j}^{fv}$ 投影到BEV平面查询融合BEV特征。FV和BEV的**空间正交性**使得边界框的精化在两个正交方向上都有约束。

4. **PGF+GGF融合**: 两个分支的输出 $f_b^{pt}$ 和 $f_b^{gd}$ 通过Transformer自注意力深度融合：$f_b = \text{SelfAttn}(f_b^{pt} + f_b^{gd})$，最终送入Refine Head预测置信度和框偏移。

### 损失函数 / 训练策略

总损失 $L = L_{RPN} + L_{refine}$，其中 $L_{refine}$ 包含置信度损失和回归损失。使用OpenPCDet框架，batch size=2，学习率0.01，80 epochs，4张GTX 3090训练。2D骨干使用ImageNet预训练的Swin-Tiny并冻结，其余模块从头训练。后处理使用置信度阈值0.1和NMS阈值0.01。

## 实验关键数据

### 主实验

| 方法 | 模态 | VoD Entire mAP↑ | VoD Corridor mAP↑ | TJ4DRad 3D mAP↑ |
|------|------|-----------------|-------------------|-----------------|
| PointPillars | LiDAR | 61.16 | 79.93 | 43.76(BEV) |
| LXL | R+C | 56.31 | 72.93 | 41.20(BEV) |
| RCFusion | R+C | 49.65 | 69.23 | - |
| RCBEVDet | R+C | 49.99 | 69.80 | - |
| CVFusion(Stage1) | R+C | 59.70 | 76.66 | - |
| **CVFusion** | **R+C** | **65.41** | **82.42** | **44.07(BEV)** |

CVFusion在VoD上将SOTA从56.31提升至65.41（+9.10%），在Corridor区域甚至超越了LiDAR方法PointPillars（82.42 vs 79.93）。

### 消融实验

| 配置 | VoD Entire mAP↑ | VoD Corridor mAP↑ | 说明 |
|------|-----------------|-------------------|------|
| 仅Camera | 18.8 | 35.3 | 单模态基线 |
| 仅Radar | 46.7 | 67.8 | 雷达独立检测 |
| Radar+Cam(无RGIter) | 57.1 | 76.3 | 简单BEV拼接 |
| +RGIter | 59.7 | 76.7 | 迭代融合+2.6% |
| +RGIter+PGF | 61.7 | 81.5 | 点级精化+2.0% |
| +RGIter+GGF | 63.0 | 81.9 | 网格级精化+3.3% |
| **+RGIter+PGF+GGF** | **65.4** | **82.4** | PGF+GGF互补 |

### 关键发现

- **两阶段带来巨大收益**：仅Stage1已显著优于现有单阶段方法（59.70 vs LXL的56.31），加上Stage2进一步提升至65.41。
- **GGF比PGF贡献更大**（+3.3% vs +2.0%）：因为大量提案框内无雷达点，GGF的网格策略保证了所有提案框都有特征输入。
- **PGF和GGF互补**：两者同时使用比单独使用任一分支提升更多（+5.7% vs +2.0%/+3.3%），说明点级精确信息和网格级覆盖信息各有所长。
- **VoD Corridor区域超越LiDAR**：82.42 vs 79.93，证明4D雷达+相机融合在近距离检测区域的巨大潜力。
- Stage1-only的FPS为6.9，完整模型为5.4，速度损失可接受。

## 亮点与洞察

- 首次将两阶段范式引入4D雷达-相机融合，思路清晰且效果显著。
- RGIter的"雷达占据概率加权+迭代下采样"设计简洁有效，利用雷达的空间先验来纠正图像BEV的深度模糊。
- PGF+GGF的双分支设计巧妙应对了4D雷达稀疏性问题：有点用点，无点用网格，互不影响又互相补充。
- FV-BEV空间正交性的利用为提案精化提供了两个独立方向的约束。

## 局限与展望

- 两阶段带来额外计算开销（FPS从6.9降至5.4），对实时性要求高的自驾场景是挑战。
- 仅在VoD和TJ4DRadSet两个中等规模数据集上验证，缺少更大规模数据集（如nuScenes 4D radar subset）的评估。
- 未讨论恶劣天气场景下的具体性能提升，而这正是4D雷达的核心优势场景。
- KDE去噪是事后处理手段，更好的雷达点云预处理可能带来更大收益。

## 相关工作与启发

- 两阶段设计借鉴了LiDAR领域的PV-RCNN/LoGoNet，但针对4D雷达稀疏性做了关键设计调整。
- CMDA（跨模态可变形注意力）衍生自Deformable DETR，用于点-图像和网格-图像的跨模态查询。
- 与同时期的RCBEVDet（CAMF对齐）相比，CVFusion从"多视角多粒度"角度解决对齐问题，思路更系统。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个4D雷达-相机两阶段融合，PGF+GGF双分支设计有创意
- 实验充分度: ⭐⭐⭐⭐ 消融完整，与多种模态配置对比充分
- 写作质量: ⭐⭐⭐⭐ 架构图清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐⭐ 4D雷达融合超越LiDAR基线，对自动驾驶感知有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](../../CVPR2025/autonomous_driving/racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](../../CVPR2025/autonomous_driving/v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [\[CVPR 2026\] RPGFusion: 4D Radar Prior-Guided Multi-Modal Fusion for 3D Detection](../../CVPR2026/autonomous_driving/rpgfusion_4d_radar_prior-guided_multi-modal_fusion_for_3d_detection.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)

</div>

<!-- RELATED:END -->
