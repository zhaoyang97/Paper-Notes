---
title: "BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions"
authors: "Wonyong Seo, Jihyong Oh, Munchurl Kim"
affiliations: "KAIST"
venue: "CVPR 2025"
date: 2024-12-16
tags: ["video frame interpolation", "optical flow", "motion estimation", "knowledge distillation", "lightweight"]
arxiv: "2412.11365"
code: "https://kaist-viclab.github.io/BiM-VFI_site/"
---

# BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions

## 研究背景与动机

视频帧插值（Video Frame Interpolation, VFI）旨在合成两个给定帧之间的中间帧，广泛应用于慢动作生成、视频编解码和帧率提升。现有方法主要基于**光流估计**：先估计前向和后向光流，然后通过 warping 合成中间帧。

然而，现实视频中的运动场往往是**非均匀的**（non-uniform），即同一帧内不同区域的运动速度和方向差异巨大。典型场景包括：

**前景高速运动 + 静止背景**：如体育赛事中的运动员

**多物体异速运动**：如交通场景中的多辆车

**旋转与平移混合**：如手持拍摄中的复杂相机运动

传统方法通常假设双向运动是独立的，分别估计前向和后向光流。这忽略了一个关键信息：**双向光流之间存在内在的几何约束关系**。对于同一个3D场景点，其在两帧中的投影位移之间满足特定的数学关系。

本文提出了 **BiM（Bidirectional Motion）描述子**，一种同时编码双向运动关系的紧凑表示，以及基于BiM的轻量级帧插值框架。

## 方法详解

### BiM 描述子

BiM 描述子 $[R, \Phi]$ 由两个分量组成：

#### 幅度比 $R$ (Magnitude Ratio)

$$R = rac{\|\mathbf{f}_{0 \to 1}\|}{\|\mathbf{f}_{1 \to 0}\|}$$

其中 $\mathbf{f}_{0 \to 1}$ 和 $\mathbf{f}_{1 \to 0}$ 分别是前向和后向光流。$R$ 捕获了运动的相对速度信息。

#### 角度差 $\Phi$ (Angle Difference)

$$\Phi = \angle(\mathbf{f}_{0 \to 1}) - \angle(\mathbf{f}_{1 \to 0}) - \pi$$

$\Phi$ 衡量前向和后向光流方向的偏差。对于严格的线性运动，$\Phi = 0$；对于非线性运动（如旋转、加速），$\Phi \neq 0$。

| 运动类型 | $R$ | $\Phi$ | 描述 |
|---------|-----|--------|------|
| 均匀平移 | 1.0 | 0 | 前后帧等速反向运动 |
| 加速运动 | >1.0 | 0 | 后半段更快 |
| 减速运动 | <1.0 | 0 | 前半段更快 |
| 弧线运动 | ≈1.0 | ≠0 | 存在方向偏差 |
| 复杂非线性 | ≠1.0 | ≠0 | 速度和方向均变化 |

### BiM-guided FlowNet

BiM 描述子作为额外输入通道注入光流估计网络：

$$\mathbf{f}_{t} = 	ext{FlowNet}(I_0, I_1, t, R, \Phi)$$

与传统方法直接估计中间帧光流不同，BiM-guided FlowNet 利用双向运动的全局约束信息，显著提高了运动不均匀区域的光流精度。

### Content-Aware Upsampling Network (CAUN)

传统帧插值使用双线性插值或可分离卷积对 warped 特征进行上采样。本文提出 CAUN，一种内容感知的上采样模块：

- 输入：低分辨率 warped 特征、高分辨率原始帧
- 核心：基于局部内容的自适应采样核生成
- 输出：高分辨率合成帧

CAUN 在边缘和纹理区域使用更精细的采样策略，在平坦区域使用更大感受野，实现质量和效率的平衡。

### 知识蒸馏 (KDVCF)

为进一步压缩模型，本文设计了 KDVCF（Knowledge Distillation for Video Content-aware Frame interpolation）策略：

| 组件 | 教师模型 | 学生模型 |
|------|---------|---------|
| 骨干网络 | ResNet-50 | MobileNetV3 |
| 参数量 | 28.3M | **6.88M** |
| 蒸馏损失 | - | 特征对齐 + 输出匹配 |
| 推理速度 | 1× | **3.2×** |

蒸馏策略包括：
1. **特征对齐蒸馏**：中间层特征的 L2 距离最小化
2. **输出匹配蒸馏**：最终合成帧的感知损失匹配

## 实验结果

### 标准数据集对比

| 方法 | 参数量 | Vimeo90K PSNR↑ | SSIM↑ | UCF101 PSNR↑ | SNU-FILM Hard↑ |
|------|--------|---------------|-------|-------------|---------------|
| RIFE | 9.8M | 35.61 | 0.978 | 35.28 | 29.27 |
| IFRNet | 19.7M | 35.80 | 0.979 | 35.36 | 29.51 |
| AMT-S | 12.3M | 35.72 | 0.978 | 35.31 | 29.39 |
| EMA-VFI | 21.5M | 35.86 | 0.979 | 35.40 | 29.56 |
| **BiM-VFI** | **6.88M** | **36.01** | **0.980** | **35.52** | **29.72** |

BiM-VFI 以最少的参数量（6.88M）在所有数据集上取得最佳结果。

### 非均匀运动场景

在包含大量非线性运动的 X-TEST 和 Xiph-4K 数据集上，BiM-VFI 的优势更加明显：

| 方法 | X-TEST PSNR↑ | Xiph-4K PSNR↑ |
|------|-------------|--------------|
| RIFE | 28.93 | 31.42 |
| EMA-VFI | 29.34 | 31.89 |
| **BiM-VFI** | **30.12** | **32.47** |

### 消融实验

| 配置 | Vimeo90K PSNR↑ | 参数量 |
|------|---------------|--------|
| Full BiM-VFI | 36.01 | 6.88M |
| w/o BiM描述子 | 35.42 | 6.85M |
| w/o CAUN (双线性上采样) | 35.67 | 5.91M |
| w/o KDVCF (教师模型) | 36.23 | 28.3M |
| 仅用$R$ | 35.78 | 6.86M |
| 仅用$\Phi$ | 35.71 | 6.86M |

BiM 描述子的两个分量均对性能有贡献，完整 BiM 描述子带来 +0.59dB 提升。

## 总结与展望

BiM-VFI 通过引入 BiM 描述子 $[R, \Phi]$ 显式建模双向运动场的内在关系，结合内容感知上采样和知识蒸馏，在仅6.88M参数下实现了 SOTA 的帧插值质量。该方法特别适合处理非均匀运动场景，其设计理念——利用双向光流的约束关系——可以推广到其他需要运动估计的视频理解任务中。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map](../../ECCV2024/video_understanding/iam-vfi_interpolate_any_motion_for_video_frame_interpolation_with_motion_complex.md)
- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/video_understanding/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)
- [\[CVPR 2025\] FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video](frame_floor-aligned_representation_for_avatar_motion_from_egocentric_video.md)
- [\[AAAI 2026\] VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation](../../AAAI2026/video_understanding/vtinker_guided_flow_upsampling_and_texture_mapping_for_high-resolution_video_fra.md)
- [\[CVPR 2025\] Progress-Aware Video Frame Captioning](progress-aware_video_frame_captioning.md)

</div>

<!-- RELATED:END -->
