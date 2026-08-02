---
title: >-
  [论文解读] BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution
description: >-
  [CVPR 2025][视频生成][视频超分辨率] 提出 BF-STVSR 框架，用 B-spline Mapper 建模时间运动插值、Fourier Mapper 捕获空间高频细节，无需外部光流网络即可实现连续时空视频超分辨率的 SOTA 性能。 1. 领域现状：连续时空视频超分辨率（C-STVSR）旨在同时以任意倍率提升…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频超分辨率"
  - "B样条"
  - "傅里叶"
  - "连续时空超分"
  - "运动插值"
---

# BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2501.11043](https://arxiv.org/abs/2501.11043)  
**代码**: 有（论文中提及）  
**领域**: 视频生成  
**关键词**: 视频超分辨率, B样条, 傅里叶, 连续时空超分, 运动插值

## 一句话总结

提出 BF-STVSR 框架，用 B-spline Mapper 建模时间运动插值、Fourier Mapper 捕获空间高频细节，无需外部光流网络即可实现连续时空视频超分辨率的 SOTA 性能。

## 研究背景与动机

1. **领域现状**：连续时空视频超分辨率（C-STVSR）旨在同时以任意倍率提升视频的时间和空间分辨率。现有方法如 VideoINR 和 MoTIF 采用隐式神经表示（INR）将时空坐标映射为像素值，但对视频数据的复杂性建模能力不足。

2. **现有痛点**：VideoINR 和 MoTIF 的 INR 部分仅使用简单的坐标拼接，未引入有效的位置编码，导致**谱偏置**问题——难以捕获高频空间细节。MoTIF 还依赖预训练光流网络（RAFT）提供运动指导，增加了计算开销且限制了模型灵活性。

3. **核心矛盾**：作者发现一个反直觉的现象——直接添加位置编码（如傅里叶编码）在 C-STVSR 中不仅没有改善效果，反而**降低**了性能。这与位置编码在图像超分中的广泛成功形成鲜明对比。问题在预训练光流网络存在时尤为严重，光流网络可能限制了模型利用多样视频信息的灵活性。

4. **本文要解决什么？** (1) 如何在不依赖外部光流的情况下有效建模视频运动；(2) 如何克服谱偏置，捕获空间高频细节；(3) 如何设计适合视频时空特性的连续表示。

5. **切入角度**：视频的时间轴（运动）和空间轴（细节）具有截然不同的特性——运动是平滑连续的，空间细节由频率信息主导。因此应针对两个轴分别设计专用模块，而非用统一的 MLP 处理。

6. **核心 idea 一句话**：用 B 样条基函数建模平滑的时间运动轨迹，用傅里叶基函数建模空间频率细节，取代统一 MLP + 外部光流的方案。

## 方法详解

### 整体框架

BF-STVSR 的整体流程：给定两帧低分辨率输入 ^L, I_1^L \in \mathbb{R}^{3 \times H \times W}$，目标是生成任意时间  \in [0,1]$、任意空间放大倍率 $ 下的高分辨率中间帧 ^H \in \mathbb{R}^{3 \times sH \times sW}$。编码器提取三个特征图 ^L, F_{(0,1)}^L, F_1^L$，其中 {(0,1)}^L$ 是融合两帧信息的模板特征。B-spline Mapper 预测到目标时刻的运动向量，Fourier Mapper 预测高分辨率空间特征。最后通过 softmax splatting 前向扭曲生成中间帧。

### 关键设计

1. **B-spline Mapper（时间运动建模）**:
    - 做什么：预测高分辨率运动向量 {0 \to t}^H, M_{1 \to t}^H$ 和可靠性图
    - 核心思路：不直接预测到目标时刻 $ 的运动，而是预测 B 样条系数 $ 和节点 $，通过 B 样条基函数 $\beta^n$ 在时间轴上平滑插值：\psi(z_r, \delta_r, \hat{t}) = c_r \odot \beta^n\left(\frac{\hat{t} - k_r}{d}
\right)$。系数/节点由三层 SIREN 网络从编码特征估计，膨胀因子 $ 由帧间隔 $ 预测
    - 设计动机：B 样条天然适合建模连续平滑信号，视频中物体运动本身就是平滑连续的，比 MLP 直接预测运动向量更优雅。同时直接从编码特征而非外部光流网络估计运动，消除了对 RAFT 的依赖

2. **Fourier Mapper（空间频率建模）**:
    - 做什么：从低分辨率特征预测高分辨率空间特征 ^H, F_1^H$
    - 核心思路：估计每个查询坐标的主频率 $ 和振幅 $，构建傅里叶基：\phi(z_r, \delta_r) = A_r \odot [\cos(\pi F_r \delta_r); \sin(\pi F_r \delta_r)]$。频率和振幅分别由 SIREN 网络估计，后接线性投影得到最终特征
    - 设计动机：INR 的谱偏置问题导致高频细节丢失，通过显式预测主频率信息可以有效捕获空间细节，灵感来自图像超分中的 LTE 方法。与 LTE 不同的是不包含相位估计器

3. **前向扭曲与解码**:
    - 做什么：将空间特征传播到目标时刻并生成最终高分辨率帧
    - 核心思路：使用 softmax splatting 将 ^H, F_1^H$ 按运动向量 {0 \to t}^H, M_{1 \to t}^H$ 前向扭曲并融合，得到中间特征 ^H$。扭曲后的特征与时间 $ 和模板特征 {(0,1)}^H$ 拼接后解码
    - 设计动机：前向扭曲相比后向扭曲可以更自然地处理遮挡和多对一映射

### 损失函数 / 训练策略

- **简化损失函数**：仅使用 Charbonnier 损失 $\mathcal{L} = \mathcal{L}_{char}(\hat{I}_t^H, I_t^H)$，去除了 MoTIF 中的光流监督 $\mathcal{L}_{RAFT}$
- **两阶段训练**：前 450K 迭代固定空间 4× 放大，后 150K 迭代从 [2,4] 均匀采样放大倍率
- Adam 优化器，余弦退火学习率（^{-4} \to 10^{-7}$），批量大小 32
- 训练稳定性：用一定概率（从 1.0 逐渐降到 0）替换预测光流为真实光流

## 实验关键数据

### 主实验（固定倍率 STVSR，4×空间 8×时间）

| 方法 | Vid4 PSNR/SSIM | GoPro-Center | GoPro-Avg | Adobe-Center | Adobe-Avg | 参数量 |
|------|---------------|-------------|-----------|-------------|-----------|--------|
| VideoINR | 25.61/0.7709 | 30.26/0.8792 | 29.41/0.8669 | 29.92/0.8746 | 29.27/0.8651 | 11.31M |
| MoTIF | 25.79/0.7745 | 31.04/0.8877 | 30.04/0.8773 | 30.63/0.8839 | 29.82/0.8750 | 12.55M |
| BF-STVSR+$\mathcal{L}_{RAFT}$ | 25.80/0.7754 | 31.14/0.8893 | 30.20/0.8799 | **30.84/0.8877** | **30.14/0.8808** | 13.47M |
| **BF-STVSR** | **25.85/0.7772** | **31.17/0.8898** | **30.22/0.8802** | 30.83/0.8880 | 30.12/0.8808 | 13.47M |

在 GoPro-Avg 上比 MoTIF 提升 +0.18 dB PSNR，Adobe-Avg 提升 +0.30 dB。不使用光流监督的版本反而略优。

### 消融实验（光流与位置编码的影响）

| 配置 | 光流网络 | B-spline | Fourier | $\mathcal{L}_{RAFT}$ | GoPro-Avg | Adobe-Avg |
|------|---------|----------|---------|--------|-----------|-----------|
| MoTIF 基线 | ✓ | ✗ | ✗ | ✓ | 30.04/0.8773 | 29.82/0.8750 |
| +光流+Fourier | ✓ | ✗ | ✓ | ✓ | 29.94/0.8764 | 29.73/0.8741 |
| +光流+B-spline | ✓ | ✓ | ✗ | ✓ | 30.03/0.8774 | 29.81/0.8756 |
| 仅 B-spline | ✗ | ✓ | ✗ | ✗ | 30.12/0.8783 | 30.02/0.8784 |
| 仅 Fourier | ✗ | ✗ | ✓ | ✓ | 30.16/0.8792 | 30.11/0.8801 |
| B+F+$\mathcal{L}_{RAFT}$ | ✗ | ✓ | ✓ | ✓ | 30.20/0.8799 | 30.14/0.8808 |
| **B+F（完整版）** | ✗ | ✓ | ✓ | ✗ | **30.22/0.8802** | 30.12/0.8808 |

### 关键发现

- **光流网络是累赘**：将 B-spline/Fourier 与预训练光流网络结合反而降低性能（第2、3行），去掉光流网络后性能全面提升
- **两个模块缺一不可**：单独使用 B-spline 或 Fourier 均不如两者结合，B-spline 主要贡献时间一致性，Fourier 主要贡献空间细节
- **无需光流监督**：去掉 $\mathcal{L}_{RAFT}$ 后性能不降反升，说明模型已能从编码特征中自主学习运动
- **计算效率更高**：去掉光流网络后 FLOPs 和推理时间均为最低，且实现了自定义 CUDA 核加速 B 样条计算

## 亮点与洞察

- **反直觉发现驱动创新**：位置编码在 C-STVSR 中失效的发现非常有价值，揭示了预训练光流网络与位置编码之间的冲突——光流提供的硬约束限制了位置编码发挥作用的空间
- **轴分离设计范式**：时间和空间特性不同就用不同数学工具处理，B 样条的平滑性契合运动连续性，傅里叶基的频率表示契合空间细节——这种用对的数学工具做对的事的思路可迁移到其他多轴信号处理任务
- **自定义 CUDA 核**：为 B 样条基函数实现了专用 CUDA 核，这种工程优化使方法具备实际部署价值

## 局限性 / 可改进方向

- **大运动场景仍有困难**：当帧间物体运动幅度极大时，所有 C-STVSR 方法（包括本文）仍产生模糊和伪影
- **训练代价高**：450K+150K 迭代的两阶段训练耗时较长
- 仅在 Adobe240 上训练，泛化到其他域（如手术视频、监控等）的效果未验证
- Vid4 上被 TMNet 超越，原因是 TMNet 使用了与 Vid4 特性相似的 Vimeo90K 训练——训练数据的域匹配仍然重要

## 相关工作与启发

- **vs MoTIF**: MoTIF 用外部 RAFT 光流网络指导运动建模，本文证明直接从编码特征学习运动更高效、效果更好
- **vs VideoINR**: VideoINR 用简单 MLP + 坐标拼接做连续映射，本文用 B 样条和傅里叶基函数替代，针对性更强
- **vs LTE（图像超分）**: 本文的 Fourier Mapper 借鉴了 LTE 估计主频率的思路，但适配到视频场景，省去了相位估计

## 评分

- 新颖性: ⭐⭐⭐⭐ 反直觉发现有启发性，轴分离设计思路清晰但各模块并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集多指标全面对比，消融实验详尽，含计算效率分析
- 写作质量: ⭐⭐⭐⭐ 动机推导逻辑清晰，图表设计好
- 价值: ⭐⭐⭐⭐ 对 C-STVSR 领域有实质推进，效率和效果双重改善

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2026\] Thermal Diffusion Matters: Infrared Spatial-Temporal Video Super-Resolution through Heat Conduction Priors](../../CVPR2026/video_generation/thermal_diffusion_matters_infrared_spatial-temporal_video_super-resolution_throu.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[CVPR 2025\] MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)

</div>

<!-- RELATED:END -->
