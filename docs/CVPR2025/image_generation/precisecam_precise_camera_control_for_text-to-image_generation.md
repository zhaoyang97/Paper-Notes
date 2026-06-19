---
title: >-
  [论文解读] PreciseCam: Precise Camera Control for Text-to-Image Generation
description: >-
  [CVPR 2025][图像生成][相机控制] PreciseCam 通过 4 个相机参数（roll、pitch、vFoV、畸变 ξ）和 Perspective Field-Unified Spherical 表示，实现文本到图像生成中的精确相机视角控制，无需 3D 几何或多视图数据。 领域现状 领域现状：图像的"镜头语言"…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "相机控制"
  - "文本生成图像"
  - "Perspective Field"
  - "ControlNet"
  - "镜头语言"
---

# PreciseCam: Precise Camera Control for Text-to-Image Generation

**会议**: CVPR 2025  
**arXiv**: [2501.12910](https://arxiv.org/abs/2501.12910)  
**代码**: [项目页面](https://graphics.unizar.es/projects/PreciseCam2024)  
**领域**: 3D Vision / Camera Control  
**关键词**: 相机控制, 文本生成图像, Perspective Field, ControlNet, 镜头语言

## 一句话总结

PreciseCam 通过 4 个相机参数（roll、pitch、vFoV、畸变 ξ）和 Perspective Field-Unified Spherical 表示，实现文本到图像生成中的精确相机视角控制，无需 3D 几何或多视图数据。

## 研究背景与动机

### 领域现状

**领域现状**：图像的"镜头语言"在传递情感中至关重要（低角度=威严，Dutch angle=不安），但现有 T2I 模型严重缺乏相机控制能力：

### 现有痛点

**现有痛点**：模型生成的图像通常呈现**平视角**——相机平行于地平面、地平线居中

### 核心矛盾

**核心矛盾**：提示工程（prompt engineering）是唯一控制手段，但粗糙且需要试错

### 解决思路

**解决思路**：Firefly 等模型提供"广角""俯拍"等标签控制，但不精确

### 补充说明

**补充说明**：基于 3D 表示的方法（NeRF）需要多视图图像，且无法处理复杂场景和多对象

### 补充说明

**补充说明**：ControlNet 用深度/边缘图控制会施加超出相机控制的严格约束

### 补充说明

**补充说明**：相对相机运动控制（视频生成）已有探索，但**绝对相机位置控制**被忽视

### 补充说明

**补充说明**：需要一个通用方案，仅通过简单直觉的相机参数即可精确控制

## 方法详解

### 整体框架

用户提供文本提示 $p$ 和 4 个相机参数 $\Omega = (\text{roll}, \text{pitch}, \text{vFoV}, \xi)$。将参数转换为 PF-US（Perspective Field - Unified Spherical）逐像素表示图。通过 ControlNet 模块将 PF-US 图注入 SDXL 的生成过程。仅需训练 ControlNet 模块，SDXL 权重冻结。

### 关键设计1：PF-US 相机视角表示

**功能**：将 4 个相机参数转化为逐像素的几何信息图，编码相机参数对每个像素外观的影响。

**核心思路**：基于 Perspective Field 表示，为每个像素分配上方向向量 $\mathbf{u}_x$（重力方向的投影）和纬度角 $\varphi_x$（光线与水平面的夹角）。采用 Unified Spherical 相机模型，投影函数 $u = \frac{xf}{\xi\sqrt{x^2+y^2+z^2}+z} + u_0$，参数 $\xi \in (0,1)$ 控制畸变程度（$\xi=0$ 为针孔相机）。PF-US 图仅从相机参数计算，不需要任何 3D 场景几何。

**设计动机**：PF-US 提供的是**局部像素级**信息而非全局 3D 表示，使模型可以学习相机参数与像素外观的关系，无需沉重的 3D 表示。排除 yaw（偏航）参数——2D 图像中无左右参考方向。

### 关键设计2：ControlNet 中间块注入策略

**功能**：在最小化对生成质量的干扰前提下实现精确的相机条件控制。

**核心思路**：训练时，ControlNet 的编码器和中间块输出通过零卷积注入 SDXL 的 U-Net 瓶颈和解码器跳连。推理时发现**仅注入中间块**（bottleneck）输出即可在不损害条件遵循度的前提下提高生成一致性。这是因为相机参数是全局属性（roll、pitch 影响整图），中间块的全局特征足以编码。

**设计动机**：全部注入会过度约束生成过程导致内容质量下降；仅中间块注入实现了控制精度和生成质量的最佳平衡。

### 关键设计3：360° 图像数据集构建

**功能**：提供 57,380 张带 GT 相机参数的训练图像，覆盖宽广的相机参数范围。

**核心思路**：从 6 个 360° 图像数据集中采样相机参数：roll ∈ (-90°, 90°)、pitch ∈ (-90°, 90°)、vFoV ∈ [15°, 140°]、$\xi \in (0,1)$。根据每组参数从 360° 图中裁剪对应区域并计算 PF-US 图。使用 BLIP-2 为每张图生成文本描述。BLIP-2 的不准确描述不影响训练——ControlNet 需学习的是独立于提示的相机视角。

**设计动机**：现有 PF 数据集主要是城市户外且参数范围窄（缺少大畸变和大 FoV）；360° 图像天然包含所有可能的视角方向。

### 损失函数

标准 ControlNet 训练损失（扩散模型去噪损失），SDXL 冻结，仅训练 ControlNet 模块。

## 实验关键数据

### 主实验：相机参数遵循度评估

| 方法 | Roll 误差↓ | Pitch 误差↓ | FoV 误差↓ |
|------|-----------|------------|----------|
| **PreciseCam** | **最优** | **最优** | **最优** |
| Prompt Engineering | 较差 | 较差 | 较差 |
| Firefly Tags | 中等 | 中等 | 中等 |
| ControlNet (Depth) | N/A | N/A | N/A |

### 消融实验：ControlNet 注入策略

| 注入层 | 相机遵循度 | 图像质量 |
|--------|----------|---------|
| 仅中间块 | 高 | **最优** |
| 编码器+中间块 | 更高 | 较差 |
| 全部注入 | 最高 | 最差 |

### 关键发现

- PreciseCam 可以精确控制 roll、pitch、vFoV 和畸变，远超提示工程方法
- 仅中间块注入是最优策略——全局相机属性不需要解码器级别的细粒度控制
- 方法同时支持摄影风格和艺术风格的生成
- 可扩展到视频生成——提供精确的初始相机位置

## 亮点与洞察

- **以像素级表示替代 3D 场景**是解决相机控制问题的优雅方案
- 排除 yaw 参数的决策体现了对 2D 图像本质的深刻理解
- 数据集构建策略（从 360° 裁剪）简单但覆盖全面

## 局限与展望

- 不控制 yaw（偏航），限制了左右视角的精确控制
- 基于 ControlNet 的方法可能在某些复杂场景中影响内容多样性
- 训练数据来自 360° 图像裁剪，可能存在分辨率和风格偏差
- 未来可扩展到视频生成中的绝对相机轨迹控制

## 相关工作与启发

- PF-US 表示可推广到其他需要几何条件控制的生成任务
- ControlNet 中间块注入的发现对其他全局属性控制（如光照）也有参考价值
- 与 CameraCtrl 等视频相对相机控制方法互补——提供绝对起始位置

## 评分

⭐⭐⭐ — 解决了一个实际且重要的创意工具问题。PF-US 表示设计精巧，但方法本身主要是 ControlNet 的应用。数据集构建和推理策略的贡献值得肯定。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] CamFreeDiff: Camera-free Image to Panorama Generation with Diffusion Model](camfreediff_camera-free_image_to_panorama_generation_with_diffusion_model.md)
- [\[CVPR 2026\] Camera Control for Text-to-Image Generation via Learning Viewpoint Tokens](../../CVPR2026/image_generation/camera_control_for_text-to-image_generation_via_learning_viewpoint_tokens.md)
- [\[CVPR 2025\] GPS as a Control Signal for Image Generation](gps_as_a_control_signal_for_image_generation.md)
- [\[CVPR 2025\] Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)

</div>

<!-- RELATED:END -->
