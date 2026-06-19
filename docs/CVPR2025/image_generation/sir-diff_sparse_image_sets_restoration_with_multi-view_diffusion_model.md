---
title: >-
  [论文解读] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model
description: >-
  [CVPR 2025][图像生成][多视角图像修复] 提出 SIR-Diff，一种多视角扩散模型，通过联合去噪多张同场景退化图像来实现跨视角一致的图像修复，利用 Spatial-3D ResNet 和 3D 自注意力 Transformer 融合多视角互补信息，在去模糊和超分辨率任务上超越单视角和视频修复方法。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "多视角图像修复"
  - "扩散模型"
  - "3D一致性"
  - "去模糊"
  - "超分辨率"
---

# SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model

**会议**: CVPR 2025  
**arXiv**: [2503.14463](https://arxiv.org/abs/2503.14463)  
**代码**: [项目页面](https://myc634.github.io/sirdiff/)  
**领域**: 3D视觉  
**关键词**: 多视角图像修复, 扩散模型, 3D一致性, 去模糊, 超分辨率

## 一句话总结

提出 SIR-Diff，一种多视角扩散模型，通过联合去噪多张同场景退化图像来实现跨视角一致的图像修复，利用 Spatial-3D ResNet 和 3D 自注意力 Transformer 融合多视角互补信息，在去模糊和超分辨率任务上超越单视角和视频修复方法。

## 研究背景与动机

传统图像修复方法独立处理每张退化图像，但实际中我们经常对同一场景拍摄多张照片。核心假设是：同一场景的多张退化照片包含互补信息，联合处理可以更好地约束修复问题。例如，一张照片中模糊的区域可能在另一视角中清晰。

这种多视角修复对 3D 计算机视觉尤为重要——SLAM、新视角合成等任务依赖多视角图像的几何一致性。单视角修复方法不可避免地引入跨视角不一致性，破坏底层 3D 场景假设。而传统多图像融合（如经典超分辨率中的亚像素位移融合）局限于简单场景。

SIR-Diff 将图像修复重新定义为多视角协同修复任务，通过神经注意力机制隐式融合一般多视角图像中的重叠信息。

## 方法详解

### 整体框架

SIR-Diff 基于 SD 2.1 的潜在扩散模型架构。输入为一组退化图像 $\{I^c\}_{i=1}^N$（模糊/低分辨率），输出为修复后的 3D 一致图像集。核心修改是将 SD 的 UNet 扩展为多视角联合去噪模型，包含 Spatial-3D ResNet 和 3D 自注意力 Transformer 两个关键组件。退化图像经 VAE 编码后与噪声潜在表示通道拼接作为条件输入。

### 关键设计1：Spatial-3D ResNet

**功能**：在卷积层同时捕获 2D 空间和跨视角 3D 关系。

**核心思路**：在 SD 的标准 2D 卷积基础上并行添加 3D 卷积层。2D 卷积用 SD 2.1 权重初始化处理空间信息，3D 卷积用 Stable Video Diffusion (SVD) 权重初始化处理跨视角信息。两者输出通过可学习权重混合：$O_{\text{ResNet}} = \sigma(\alpha) \times O_{2D} + \sigma(1-\alpha) \times O_{3D}$。

**设计动机**：纯 2D 卷积无法捕获跨视角几何关系。从零训练 3D 卷积不稳定且耗时。时间域的相似性（视频中相邻帧）和空间域的相似性（多视角中相邻视角）有共通性，因此 SVD 的时序卷积权重可以初始化空间 3D 卷积。

### 关键设计2：3D 自注意力 Transformer

**功能**：使每个 token 能关注所有视角的所有空间位置，实现全局跨视角信息融合。

**核心思路**：将 $N$ 个视角的潜在特征 patch 化后得到 $N \times p$ 个 token，每个 token 对所有 $N \times p$ 个 token 计算注意力分数。受 CAT3D 启发，仅在低分辨率层注入此模块以控制计算开销，并移除 cross-attention 模块加速收敛。

**设计动机**：自注意力使每个视角的每个像素都能访问其他所有视角的信息，实现最大程度的信息互补。推理时可适配不同数量的输入帧，提供灵活性。

### 关键设计3：统一退化输入编码器

**功能**：处理不同类型的退化输入（模糊、低分辨率、正常 RGB）。

**核心思路**：使用 SD 2.1 的 VAE 编码器对退化图像编码，通过零初始化卷积层适配通道数变化（退化条件与噪声潜在表示的通道拼接）。

**设计动机**：统一的编码方式使模型能同时处理多种退化类型，增强通用性。

### 损失函数

标准扩散训练损失：$\mathcal{L} = \mathbb{E}\|\epsilon - \hat{\epsilon}\|_2^2$，监督 UNet 预测的噪声与前向扩散过程中添加的噪声一致。在合成数据集（Hypersim + TartanAir）上训练。

## 实验关键数据

### 多视角去模糊（零样本评估）

| 方法 | Scannet++ FID ↓ | Scannet++ LPIPS ↓ | Scannet++ VConsis ↓ |
|------|--------|---------|----------|
| PromptIR | 81.28 | 0.248 | 7.81 |
| Restormer | 49.72 | 0.232 | 6.52 |
| VRT (视频) | 134.5 | 0.371 | 7.67 |
| SIR-Diff (单帧) | 81.58 | 0.247 | 6.45 |
| **SIR-Diff** | **40.09** | **0.160** | **5.75** |

### 关键发现

- SIR-Diff 在所有数据集上的 FID、LPIPS 和视觉一致性指标均超越单视角和视频修复方法。
- 多视角版本远优于单帧版本（FID 40.09 vs 81.58），验证了多视角互补信息的价值。
- 视频修复方法（VRT）在稀疏多视角场景下表现最差，因为其假设帧间连续性而非稀疏采样的不同视角。
- 在合成数据上训练，零样本迁移到真实场景（Scannet++、ETH3D、CO3D）效果良好。
- 修复后的图像可直接用于 3DGS 重建，显著提升新视角合成质量和特征匹配成功率。
- 推理时支持任意数量输入帧，训练时仅用少量图像。

## 亮点与洞察

1. **范式转换**：将图像修复从"单视角独立处理"转向"多视角协同修复"，充分利用多捕获场景的互补信息。
2. **SVD 初始化技巧**：时间域→空间域的权重迁移巧妙利用了已有视频扩散模型的几何理解能力。
3. **VConsis 指标**：提出评估生成视角集内部一致性的新指标，对多视角生成领域有参考价值。

## 局限与展望

- 需要已知相机位姿来计算视角间重叠区域（训练时），限制了某些应用场景。
- 3D 自注意力的计算开销随视角数 $N$ 二次增长，大量视角时可能成为瓶颈。
- 目前仅处理去模糊和超分辨率两种退化，其他类型（去雾、去雨等）未验证。
- 在合成数据上训练可能存在域差异，尤其对复杂真实退化。

## 相关工作与启发

- **MVDream / CAT3D**：多视角生成扩散模型，提供了 3D 自注意力层的设计参考。
- **Stable Video Diffusion**：其 3D 卷积权重被复用于初始化多视角理解的 3D 卷积。
- **Restormer / PromptIR**：强单视角修复基线，本文证明多视角方法可超越它们。

## 评分

⭐⭐⭐⭐ — 问题定义清晰实用（多视角修复），SVD 权重迁移技巧聪明，零样本泛化效果好。对 3D 重建管线（3DGS 等）的下游价值突出。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Pippo: High-Resolution Multi-View Humans from a Single Image](pippo_high-resolution_multi-view_humans_from_a_single_image.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [\[CVPR 2026\] InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization](../../CVPR2026/image_generation/instructmix2mix_consistent_sparse-view_editing_through_multi-view_model_personal.md)
- [\[CVPR 2025\] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)
- [\[CVPR 2025\] RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing](roompainter_view-integrated_diffusion_for_consistent_indoor_scene_texturing.md)

</div>

<!-- RELATED:END -->
