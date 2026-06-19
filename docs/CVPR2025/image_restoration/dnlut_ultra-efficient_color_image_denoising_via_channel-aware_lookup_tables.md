---
title: >-
  [论文解读] DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables
description: >-
  [CVPR 2025][图像恢复][查找表] 提出基于查找表(LUT)的超高效彩色图像去噪框架 DnLUT，通过 Pairwise Channel Mixer (PCM) 捕获通道间相关性和 L 形卷积核扩展感受野，仅需 500KB 存储和 DnCNN 0.1% 的能耗即可实现 SOTA 的 LUT 去噪性能。
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "查找表"
  - "彩色图像去噪"
  - "边缘设备部署"
  - "通道相关性"
  - "轻量化"
---

# DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables

**会议**: CVPR 2025  
**arXiv**: [2503.15931](https://arxiv.org/abs/2503.15931)  
**代码**: [GitHub](https://github.com/Stephen0808/DnLUT)  
**领域**: 图像复原  
**关键词**: 查找表, 彩色图像去噪, 边缘设备部署, 通道相关性, 轻量化

## 一句话总结

提出基于查找表(LUT)的超高效彩色图像去噪框架 DnLUT，通过 Pairwise Channel Mixer (PCM) 捕获通道间相关性和 L 形卷积核扩展感受野，仅需 500KB 存储和 DnCNN 0.1% 的能耗即可实现 SOTA 的 LUT 去噪性能。

## 研究背景与动机

彩色图像去噪是一个重要且具有挑战性的任务。深度神经网络虽然大幅提升了去噪质量，但其计算复杂度和内存需求使其难以在边缘设备上部署。查找表(LUT)方法通过将运行时计算替换为简单的数组索引操作提供了一种解决方案，但面临以下核心瓶颈：

- **存储呈指数增长**：LUT 存储需求随输入维度指数增长，例如 $2 \times 2$ 深度为 3 的核需要约 582TB 存储
- **通道信息缺失**：现有 LUT 方法通常对每个通道独立处理（空间 LUT），忽略了彩色图像中 RGB 通道间的强相关性
- **通道-空间二选一**：通道 LUT ($1 \times 1$ 深度 3) 忽略空间关系，空间 LUT ($2 \times 2$ 深度 1) 忽略通道关系
- **旋转冗余**：传统旋转集成策略中约一半像素被重复访问，存在设计冗余

## 方法详解

### 整体框架

DnLUT 由训练阶段的 DnNet 和推理阶段的 DnLUT 两部分组成。DnNet 由两个阶段构成：第一阶段使用多组 LUT 生成多通道特征并在融合模块中拼接；第二阶段整合 PCM 和 L 形卷积两个核心模块。训练完成后，所有组件被转换为 3D 或 4D LUT 进行高效推理。

### 关键设计1: Pairwise Channel Mixer (PCM) — 同时捕获空间和通道信息

**功能**: 在 LUT 可承受的索引维度内同时处理空间和通道信息，解决通道-空间不可得兼的问题。

**核心思路**: 将 RGB 三通道重组为三个成对组合 (RG, GB, BR)，每对通过 $1 \times 2$ 空间维度、深度为 2 的卷积核并行处理。每次卷积处理 4 个像素值产生 1 个通道输出，可高效转换为 4D LUT。输出公式为：

$$
(V_R, V_G, V_B) = \text{Cat}(LUT_{RG}[I_{R}][I_{R'}][I_{G}][I_{G'}], LUT_{GB}[\cdot], LUT_{BR}[\cdot])
$$

**设计动机**: 直接使用深度为 3 的 $2 \times 2$ 卷积核需要 12 维 LUT（约 582TB），而成对组合将维度控制在 4D（约 83.5KB），在存储可行范围内实现通道-空间联合建模。PCM 还可作为即插即用模块，仅增加 12% 运行时间和 8% 存储即可为现有 LUT 方法带来超过 1dB 的提升。

### 关键设计2: L 形旋转无重叠卷积核 — 最大化像素利用率

**功能**: 扩展感受野的同时避免像素重复访问，并将 4D LUT 降为更高效的 3D LUT。

**核心思路**: 设计 L 形卷积核，每次旋转只处理中心像素之外的 2 个额外像素且互不重叠。4 次旋转后每个周围像素恰好贡献一次，实现 $3 \times 3$ 等效感受野覆盖。

**设计动机**: 传统 SR-LUT 使用 $2 \times 2$ 核 + 4 次旋转扩展到 $3 \times 3$ 感受野，但约一半像素被重复查找。L 形核消除了这种冗余，且由于每次旋转只需 3 个像素索引，可转换为 3D LUT 而非 4D LUT，存储减少 17 倍。

### 关键设计3: 多尺度融合架构 — 逐级扩展感受野

**功能**: 通过多级 LUT 组合实现大范围特征聚合。

**核心思路**: 第一阶段使用多组不同模式的 LUT 提取多通道特征并拼接融合；第二阶段在融合特征基础上应用 PCM 和 L 形卷积进一步精炼。

**设计动机**: 单一 LUT 的感受野有限，多级级联允许信息在更大范围内传播，同时保持每个 LUT 的低维度索引。

### 损失函数

训练使用标准 $L_1$ 损失函数，直接优化去噪后图像与干净图像之间的像素差异。

## 实验关键数据

### 主实验：高斯彩色图像去噪 (CPSNR/dB)

| 方法 | CBSD68 σ=15 | CBSD68 σ=25 | CBSD68 σ=50 | Urban100 σ=25 |
|------|------------|------------|------------|--------------|
| SR-LUT | 29.76 | 26.71 | 22.41 | 26.04 |
| MuLUT | 30.52 | 28.11 | 24.85 | 27.67 |
| SPF-LUT | 30.97 | 28.56 | 25.33 | 28.26 |
| **DnLUT** | **32.41** | **29.88** | **26.03** | **28.87** |
| DnCNN (DNN) | 33.90 | 31.24 | 27.95 | 30.81 |

### 真实世界去噪

| 数据集 | SPF-LUT | DnLUT | DnCNN |
|--------|---------|-------|-------|
| SIDD (CPSNR) | 34.91 | **35.44** | 36.45 |
| DnD (PSNR) | 36.22 | **36.67** | 37.11 |

### 效率对比

| 指标 | DnLUT | DnCNN |
|------|-------|-------|
| 存储 | **500KB** | ~500× 更大 |
| 能耗 | **0.1%** | 100% |
| 推理速度 | **20× 快** | 基准 |

### 关键发现

- DnLUT 在所有 LUT 方法中取得最佳性能，超过 SPF-LUT 最多 **1.3dB**
- PCM 作为插件可为现有 LUT 方法带来 **>1dB** 的一致提升
- L 形卷积核在保持感受野的同时将存储减少 **17×**
- 在真实世界去噪任务中，DnLUT 超过经典方法 CBM3D 近 **5dB**

## 亮点与洞察

1. **通道成对分组是关键洞察**：将 RGB 三通道问题分解为三个两通道子问题，将不可行的 12 维 LUT 转化为可行的 4D LUT，是这项工作最精妙之处
2. **旋转无重叠设计具有数学优雅性**：L 形核利用几何对称性确保每个像素恰好被访问一次，是对 LUT 方法存储-性能权衡的理论最优解
3. **PCM 插件化设计展示了通用性**：不需要修改现有架构即可带来显著提升，降低了采用门槛

## 局限与展望

- 与 DNN 方法（如 SwinIR 34.42dB vs DnLUT 32.41dB）仍有显著差距，LUT 方法的表达能力本质上受限于索引维度
- 目前主要针对高斯噪声和真实噪声场景，对其他退化类型（如模糊、JPEG 压缩）的适用性有待验证
- 级联 LUT 的深度和组合策略值得进一步探索
- 未来可探索与神经网络 + LUT 混合架构，在不同计算预算下灵活切换

## 相关工作与启发

- **SR-LUT / MuLUT / RC-LUT**: 基于 LUT 的超分方法，仅关注空间信息
- **SPF-LUT**: 引入 shift 聚合和多 LUT 级联，但仍缺少通道建模
- **DnCNN**: 经典 CNN 去噪方法，性能强但计算量大
- **CBM3D**: 经典的彩色图像去噪方法，在亮度-色度空间操作以利用通道相关性

## 评分

⭐⭐⭐⭐ — 在 LUT 去噪领域做出了很有意义的贡献，PCM 和 L 形核的设计都颇具巧思。虽然与 DNN 方法差距仍大，但在边缘设备部署场景下具有极强的实用价值。方法清晰、实验充分、即插即用性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](../../ICCV2025/image_restoration/lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)
- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](hvi_a_new_color_space_for_low-light_image_enhancement.md)
- [\[CVPR 2026\] ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](../../CVPR2026/image_restoration/shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)

</div>

<!-- RELATED:END -->
