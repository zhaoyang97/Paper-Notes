---
title: >-
  [论文解读] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution
description: >-
  [ICCV 2025][视频生成][视频超分辨率] 首次将 Mamba 引入视频超分辨率任务，提出双聚合 Mamba 模块（DAMB）捕获时空长程依赖、可变形交叉 Mamba 对齐模块（DCA）增强帧间对齐灵活性，以及频域 Charbonnier 损失（FCL）改善高频细节恢复，在 REDS4/Vid4/Vimeo-90K 上取得 SOTA。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "视频超分辨率"
  - "Mamba"
  - "状态空间模型"
  - "可变形对齐"
  - "频域损失"
---

# VSRM: A Robust Mamba-Based Framework for Video Super-Resolution

**会议**: ICCV 2025  
**arXiv**: [2506.22762](https://arxiv.org/abs/2506.22762)  
**代码**: 无  
**领域**: Image Restoration / Video Super-Resolution  
**关键词**: 视频超分辨率, Mamba, 状态空间模型, 可变形对齐, 频域损失

## 一句话总结

首次将 Mamba 引入视频超分辨率任务，提出双聚合 Mamba 模块（DAMB）捕获时空长程依赖、可变形交叉 Mamba 对齐模块（DCA）增强帧间对齐灵活性，以及频域 Charbonnier 损失（FCL）改善高频细节恢复，在 REDS4/Vid4/Vimeo-90K 上取得 SOTA。

## 研究背景与动机

视频超分辨率（VSR）旨在从低分辨率视频生成高分辨率帧，需要利用多帧互补信息。当前方法主要基于 CNN 或 Transformer：

- **CNN** 方法（如 BasicVSR）受限于局部感受野，无法有效捕获远距离帧间信息
- **Transformer** 方法（如 PSRT、IART）提供了强大的注意力机制，但全注意力的二次复杂度使其在长序列处理中不实际；窗口注意力虽降低复杂度，但感受野仍受限
- **对齐模块**：现有方法多使用双线性/最近邻插值进行空间对齐，固定权重导致特征畸变；IART 虽提出基于注意力的隐式插值，但在固定参考窗口内计算，灵活性不足
- **损失函数**：像素级损失过于平滑，感知损失引入更大失真；VSR 是病态问题，重建帧与 GT 之间的差异在频域中尤为明显

Mamba 以线性复杂度实现长序列建模和数据依赖特性，非常适合 VSR 场景。本文首次探索 Mamba 在 VSR 中的应用。

## 方法详解

### 整体框架

VSRM 由两部分组成：特征提取（Conv2d + 特征传播模块 FPB）和上采样器（重建模块）。FPB 包含可变形交叉 Mamba 对齐（DCA）和双聚合 Mamba 模块（DAMB），先对齐相邻帧特征，再提取深层时空特征，最后通过上采样器生成高分辨率输出。

### 关键设计

1. **双聚合 Mamba 模块 (DAMB)**：由 N 个 S2TMB 和 1 个 T2SMB 组成，充分建模空间和时间两个维度的长程依赖。

    - **S2T-Mamba（空间到时间）**：将 3D 视频序列展平为 1D 序列，采用空间优先→时间的扫描顺序，使用前向和反向双向 SSM 处理，公式：$S2T\text{-}Mamba(x,z) = Linear(x_1 \odot z + x_2 \odot z)$
    - **T2S-Mamba（时间到空间）**：采用时间优先→空间的扫描顺序，仅使用单向前向扫描。实验表明 S2TMB 偏重空间信息提取，T2SMB 显式优先提取时间信息，二者互补
    - **TGFN（时间门控前馈网络）**：包含 3D 深度可分离卷积和门控机制，替代标准 FFN，更好地建模时空相邻像素关系并优化信息流

2. **可变形交叉 Mamba 对齐 (DCA)**：使用 SpyNet 估计光流，在补偿阶段引入可变形窗口方案。核心思路：
    - 对每个目标像素，根据光流在参考帧中定位采样点
    - 在采样位置周围构建窗口 $w$，初始化参考区域 $r$
    - 通过可学习偏移网络 $\mathcal{S}(w)$ 学习偏移 $\epsilon_r$，获得动态参考区域 $\bar{r} = \phi(w; r + \epsilon_r)$
    - 通过 cross-mamba 模块融合参考点与目标点信息完成对齐：$\bar{X}(x,y) = cross\text{-}mamba(R, Q)$
    - cross-mamba 基于 SSM 递推：$H_t = \bar{A}_R H_{t-1} + \bar{B}_R \bar{R}_t, \bar{X}_t = C_Q H_t$

3. **频域 Charbonnier 损失 (FCL)**：在频域中分别计算实部和虚部的 Charbonnier 损失，而非使用幅度/相位（避免平方根和反正切函数带来的不连续性）。公式：

   $$\mathcal{L}_{FCL} = \sum_{i \in \{Re, Im\}} \lambda_i \sqrt{\|i\mathcal{F}(\mathbf{I}_{SR}) - i\mathcal{F}(\mathbf{I}_{HR})\|^2 + \epsilon^2}$$

### 损失函数 / 训练策略

总损失为空间域 Charbonnier 损失与频域 FCL 的加权组合：

$$\mathcal{L}_{total} = \lambda \mathcal{L}_{CL} + \mathcal{L}_{FCL}$$

其中 $\lambda = 1.0$，$\lambda_{Re} = \lambda_{Im} = 0.02$，$\epsilon = 10^{-3}$。训练集使用 REDS 和 Vimeo-90K。

## 实验关键数据

### 主实验

| 方法 | 输入帧数 | 参数量(M) | REDS4 PSNR | REDS4 SSIM | Vid4 PSNR | Vid4 SSIM |
|------|---------|----------|------------|------------|-----------|-----------|
| BasicVSR++ | 30/14 | 7.3 | 32.39 | 0.9069 | 27.79 | 0.8400 |
| VRT | 16/7 | 35.6 | 32.19 | 0.9006 | 27.93 | 0.8425 |
| RVRT | 30/14 | 10.8 | 32.75 | 0.9113 | 27.99 | 0.8462 |
| PSRT-rec | 16/14 | 13.4 | 32.72 | 0.9106 | 28.07 | 0.8485 |
| IART | 16/7 | 13.4 | 32.90 | 0.9138 | 28.26 | 0.8517 |
| **VSRM** | **16/7** | **17.1** | **33.11** | **0.9162** | **28.44** | **0.8552** |

VSRM 在 REDS4 上比 IART 提升 0.21dB（16帧设置），在 Vid4 上提升 0.18dB，同时也在 Vimeo-90K-T 上取得最优 38.33dB。

### 消融实验

| 消融项 | PSNR (dB) | 参数量(M) | FLOPs(G) |
|--------|-----------|----------|----------|
| 3D DW-Conv (替换 Mamba) | 30.84 | 19.49 | 149.8 |
| 窗口注意力 (替换 Mamba) | 30.97 | 7.68 | 152.4 |
| 全注意力 (替换 Mamba) | 31.06 | 7.68 | 1018.1 |
| **Mamba (ours)** | **31.09** | **8.61** | **159.2** |
| 无 DCA 对齐 | 30.87 | 8.53 | 120.4 |
| FGDA 对齐 | 30.92 | 8.70 | 154.3 |
| IA 对齐 | 31.00 | 8.57 | 148.7 |
| **DCA 对齐 (ours)** | **31.09** | **8.61** | **159.2** |
| 无 T2SMB | 30.95 | 7.87 | 155.6 |
| T2SMB (双向) | 31.02 | 8.65 | 162.2 |
| **T2SMB (单向, ours)** | **31.09** | **8.61** | **159.2** |
| FFN | 30.90 | 8.68 | 136.2 |
| **TGFN (ours)** | **31.09** | **8.61** | **159.2** |
| 无 FCL (λ=0) | 30.97 | - | - |
| **FCL (λ=0.02)** | **31.09** | - | - |

### 关键发现

- Mamba 在达到与全注意力相似 PSNR 的同时，FLOPs 仅为其 1/6 (159G vs 1018G)
- DCA 比 FGDA 和 IA 对齐分别提升 0.17dB 和 0.09dB，验证了可变形窗口机制的优势
- T2SMB 补充了 S2TMB 不足的时间信息提取（+0.14dB），且单向扫描优于双向
- 移除 FCL 损失导致 0.12dB 下降，证明频域约束对高频细节恢复的重要性
- VSRM 的有效感受野（ERF）远大于 CNN 和 Transformer 方法

## 亮点与洞察

- **首次 Mamba + VSR**：成功验证 Mamba 在视频超分辨率中的可行性，兼具线性复杂度和全局感受野
- **S2T 和 T2S 互补扫描**：将空间优先和时间优先两种扫描策略组合，完整提取时空特征，是 VSR 特有的 Mamba 适配方案
- **DCA 的可变形参考区域**：不同于固定窗口的隐式对齐，通过学习偏移动态调整参考区域，更好地处理运动幅度差异
- **FCL 损失设计简洁有效**：直接在实/虚部上计算 Charbonnier 损失，避免了幅度/相位计算的数值不稳定问题

## 局限与展望

- 参数量（17.1M）和运行时间（223ms）略高于 PSRT/IART（13.4M, 173-180ms），Mamba 的加速优化仍在探索中
- 仅探索了 4× 超分任务，未验证其他缩放因子和降质模型
- Mamba 在视觉领域的加速库和硬件支持不如 Transformer 成熟
- 可进一步扩展到其他低级视觉任务（去模糊、去噪、着色等）

## 相关工作与启发

- BasicVSR 系列展示了通用组件（残差块+光流）的有效性，VSRM 在此基础上引入 Mamba 提升时空建模
- MambaIR 验证了 Mamba 在图像恢复中的有效性，本文将其拓展到视频领域
- 频域损失（FFL、WHFL）的比较证明了 FCL 在平衡低高频方面的优势

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将 Mamba 引入 VSR，双向扫描和 DCA 设计有创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融覆盖每个模块，多指标多数据集对比
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富
- **价值**: ⭐⭐⭐⭐ 为 Mamba 在低级视觉的应用提供了坚实 baseline

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](../../CVPR2025/video_generation/videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](../../CVPR2025/video_generation/patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2026\] Compressed-Domain-Aware Online Video Super-Resolution](../../CVPR2026/video_generation/compressed-domain-aware_online_video_super-resolution.md)
- [\[CVPR 2025\] BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](../../CVPR2025/video_generation/bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatia.md)
- [\[CVPR 2026\] VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment](../../CVPR2026/video_generation/vsrell_a_simple_baseline_for_video_super-resolution_and_enhancement_in_low-light.md)

</div>

<!-- RELATED:END -->
