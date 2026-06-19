---
title: >-
  [论文解读] VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation
description: >-
  [AAAI 2026][视频理解][视频帧插值] 提出 VTinker 流水线，通过引导式光流上采样（GFU）解决光流边界模糊问题，并采用纹理映射替代传统逐像素融合策略来消除鬼影和不连续，在高分辨率视频帧插值上取得 SOTA。 领域现状：基于光流的视频帧插值（VFI）是主流方法，通常包含三个阶段：运动估计（在低分辨率下进行以…
tags:
  - "AAAI 2026"
  - "视频理解"
  - "视频帧插值"
  - "光流上采样"
  - "纹理映射"
  - "高分辨率视频"
  - "运动估计"
---

# VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation

**会议**: AAAI 2026  
**arXiv**: [2511.16124](https://arxiv.org/abs/2511.16124)  
**代码**: [https://github.com/Wucy0519/VTinker](https://github.com/Wucy0519/VTinker)  
**领域**: 视频理解 / 视频帧插值  
**关键词**: 视频帧插值, 光流上采样, 纹理映射, 高分辨率视频, 运动估计

## 一句话总结

提出 VTinker 流水线，通过引导式光流上采样（GFU）解决光流边界模糊问题，并采用纹理映射替代传统逐像素融合策略来消除鬼影和不连续，在高分辨率视频帧插值上取得 SOTA。

## 研究背景与动机

**领域现状**：基于光流的视频帧插值（VFI）是主流方法，通常包含三个阶段：运动估计（在低分辨率下进行以降低计算量）→ 光流上采样（从低分辨率到高分辨率）→ 帧合成（基于上采样光流 warp 两帧并融合）。

**现有痛点**：现有方法存在三个层面的缺陷。(1) **光流上采样**：双线性上采样会导致光流边界模糊，自适应核上采样（AFU）在端到端训练的任务导向光流中容易产生块状伪影；(2) **运动精度**：低分辨率运动估计无法捕捉高分辨率细粒度像素运动，上采样后的光流存在错位；(3) **帧合成**：传统 Mask&Res 机制逐像素地融合两个 warp 后的帧，当光流不准确时产生鬼影、模糊和不连续。

**核心矛盾**：在高分辨率视频中，运动跨度可能超过 100 像素，必须在低分辨率做运动估计以保证效率，但低分辨率到高分辨率的光流映射和逐像素双源纹理融合都会放大误差。

**本文目标** (1) 如何让上采样光流的边界与图像边界对齐；(2) 如何避免逐像素双源融合带来的鬼影和不连续。

**切入角度**：受 UPFlow 启发，利用高分辨率输入帧作为引导信息来细化光流上采样；用纹理映射替代逐像素融合，从单一源帧中选取整块纹理来保证连续性。

**核心 idea**：用输入帧引导光流上采样得到清晰边界，用块级纹理映射替代像素级双源融合来消除鬼影。

## 方法详解

### 整体框架

给定两帧 $I_0, I_1 \in \mathbb{R}^{H \times W \times 3}$，VTinker 首先在低分辨率估计双向光流 $F_{0\to1}, F_{1\to0}$，通过 GFU 上采样到高分辨率，用上采样光流 warp 两帧生成中间代理（proxy），然后从输入帧中提取纹理块，通过光流引导搜索和局部匹配选择最佳纹理块映射到 proxy 上，最后通过重建模块生成插值帧。

### 关键设计

1. **引导式光流上采样（Guided Flow Upsampling, GFU）**:

    - 功能：将低分辨率光流上采样到高分辨率，同时保持清晰的运动边界
    - 核心思路：先用双线性插值对低分辨率光流进行初始上采样，然后用卷积层从输入帧中提取引导信息，利用引导信息修正双线性上采样中的模糊边界。由于光流边界应与对应帧的图像边界对齐（如 $F_{0\to1}^{up}$ 与 $I_0$ 的边界对齐），引导信息自然来自对应输入帧。
    - 设计动机：双线性上采样在边界处平滑了不同运动区域的光流值，导致边界模糊；AFU 在任务导向光流（无 GT 监督）下产生不连续边界。GFU 通过引入高分辨率图像结构信息来引导边界对齐。

2. **纹理映射（Texture Mapping）**:

    - 功能：用整块纹理替代逐像素融合，消除因 warp 错位导致的鬼影和不连续
    - 核心思路：分为三步。(a) **Proxy 生成**：将两个 warp 后的帧 $I_t^0, I_t^1$ 通过多层卷积生成中间 proxy $\mathcal{Q} \in \mathbb{R}^{H/2 \times W/2 \times C}$。(b) **纹理提取与块划分**：从 $I_0, I_1$ 提取纹理特征 $\mathcal{T}_0, \mathcal{T}_1$，按块大小 $s$ 划分为重叠的纹理块 $\mathcal{B}_0^{x,y}, \mathcal{B}_1^{x,y}$。(c) **纹理块选择**：先通过光流引导进行粗搜索（将光流降采样到块级别，用 nearest mode 的 GridSample 索引纹理块），再通过局部匹配精细对齐——将 proxy 块和纹理块通过卷积压缩为索引向量，在 $N \times N$ 邻域内选择相关性最高的纹理块。
    - 设计动机：传统 Mask&Res 的输出纹理是像素级双源的（来自 $I_0$ 和 $I_1$ 的混合），当 warp 不准确时混合两个错位源产生鬼影。纹理映射确保每个位置的纹理块来自单一源（仅 $I_0$ 或仅 $I_1$），保证块内纹理的连续性。

3. **重建模块与纹理质量保证**:

    - 功能：将映射了纹理块的 proxy 重建为最终插值帧，并确保纹理质量
    - 核心思路：使用 UNet-like 网络将潜空间特征变换到图像空间。关键设计是用同一个权重共享的重建模块分别重建 $\mathcal{T}_0 \to \hat{I}_0$ 和 $\mathcal{T}_1 \to \hat{I}_1$，并用输入帧 $I_0, I_1$ 作为监督，确保提取的纹理是高质量的。推理时只需要重建插值帧，不需要重建 $\hat{I}_0, \hat{I}_1$。
    - 设计动机：如果纹理块本身质量差，即使映射策略正确也无法生成好的结果。通过额外的纹理重建损失约束纹理提取器的质量。

### 损失函数 / 训练策略

使用 FILM 提出的 Style loss：$\mathcal{L}_S = w_l \mathcal{L}_1 + w_{VGG} \mathcal{L}_{VGG} + w_{Gram} \mathcal{L}_{Gram}$。总损失为三个 Style loss 的加权和：$\mathcal{L}_S^{all} = w_t \times \mathcal{L}_S^t + w_0 \times \mathcal{L}_S^0 + w_1 \times \mathcal{L}_S^1$，分别对应插值帧和两个输入帧的重建。运动估计器基于 UPR-Net 重新设计，参考 PWC-Net 结构做特征对齐。

## 实验关键数据

### 主实验

| 数据集 | 指标 | VTinker | 之前最佳 | 提升 |
|--------|------|---------|---------|------|
| DAVIS(1080p) | PSNR↑ | 26.778 | 26.927 (SGM-1/2) | -0.15 |
| DAVIS(1080p) | LPIPS↓ | 0.108 | 0.114 (PerVFI) | +0.006 |
| DAVIS(1080p) | DISTS↓ | 0.039 | 0.042 (PerVFI) | +0.003 |
| DAVIS(4K) | PSNR↑ | 26.610 | 26.798 (SGM-1/2) | -0.19 |
| DAVIS(4K) | LPIPS↓ | 0.115 | - | 最佳 |
| Xiph-4K | LPIPS↓ | 0.066 | 0.084 (RIFE) | +0.018 |
| Xiph-4K | DISTS↓ | 0.025 | 0.035 (RIFE) | +0.010 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 双线性上采样 vs GFU | GFU 优于双线性和 AFU | 光流边界更清晰 |
| Mask&Res vs 纹理映射 | 纹理映射减少鬼影和不连续 | 视觉质量提升明显 |
| 无纹理监督 vs 有纹理监督 | 有监督提升纹理质量 | 权重共享重建模块有效 |

### 关键发现

- 在感知指标（LPIPS, DISTS）上 VTinker 全面领先，尤其在 4K 超高分辨率下优势更大
- PSNR 上略逊于 SGM-1/2Point，但感知质量（人眼感受）更好
- GFU 的提升主要体现在运动边界区域，纹理映射的提升主要在遮挡和大运动区域
- 许多方法在 4K 分辨率下 OOM，VTinker 保持可运行

## 亮点与洞察

- 将 VFI 问题分解为"光流质量"和"合成策略"两个层面，分别用 GFU 和纹理映射来解决，思路清晰
- 纹理映射的"单源块选择"思想简单但非常有效——保证每个位置的纹理来自单一帧避免了混合双源的问题
- 权重共享重建模块同时服务于纹理质量保证和最终帧重建，设计经济
- 在 4K 分辨率上相比其他方法有显著优势，real-world 实用性强

## 局限与展望

- 计算量较大（121ms@1080p, 765G FLOPs），不适合实时应用
- 块大小 $s$ 是固定的，自适应块大小可能在不同运动区域取得更好的效果
- 纹理块匹配依赖光流引导的粗搜索，光流严重错误时匹配也会失败
- PSNR 指标未达最佳，说明在像素精确度上仍有提升空间

## 相关工作与启发

- **vs SGM**: SGM 使用 AFU 进行光流上采样，在端到端训练中产生块状伪影；VTinker 的 GFU 通过图像引导避免此问题
- **vs UPR-Net**: VTinker 重新设计了 UPR-Net 的运动估计器，在原始时间步做对齐而非插值时间步，提高光流估计效率
- **vs FILM / PerVFI**: 感知质量导向的方法，VTinker 在感知指标上超越它们，且在 4K 分辨率上不会 OOM

## 评分

- 新颖性: ⭐⭐⭐⭐ 纹理映射的块级单源选择思想新颖，GFU 虽受 UPFlow 启发但适配 VFI 有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多分辨率（720p/1080p/2K/4K）多数据集全面评测，消融充分，视觉对比清晰
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，方法流程图清晰，消融实验设计合理
- 价值: ⭐⭐⭐⭐ 对高分辨率 VFI 的实际应用有显著意义，代码开源增加了可复现性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/video_understanding/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)
- [\[CVPR 2025\] VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video SpatioTemporal Augmentation](../../CVPR2025/video_understanding/vista_enhancing_long-duration_and_high-resolution_video_understanding_by_video_s.md)
- [\[CVPR 2026\] FPS-Bench: A Benchmark for High Frame-Rate Video Understanding](../../CVPR2026/video_understanding/fps-bench_a_benchmark_for_high_frame-rate_video_understanding.md)
- [\[ECCV 2024\] UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation](../../ECCV2024/video_understanding/uniinr_event-guided_unified_rolling_shutter_correction_deblurring_and_interpolat.md)

</div>

<!-- RELATED:END -->
