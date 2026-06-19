---
title: >-
  [论文解读] TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation
description: >-
  [ICCV 2025][图像生成][视频帧插值] 提出 TLB-VFI，一种高效的视频扩散模型用于帧插值：通过时域感知自编码器（隐空间时域块+像素空间3D小波门控）提取丰富的时间信息，结合重新设计的布朗桥扩散过程，在参数量仅 46.7M（比图像扩散方法少 3×、比视频扩散方法少 20×）的情况下，在 SNU-FILM extreme 和 Xiph-4K 上 FID 提升约 20%。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "视频帧插值"
  - "布朗桥扩散"
  - "时域感知自编码器"
  - "3D小波"
  - "光流引导"
---

# TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation

**会议**: ICCV 2025  
**arXiv**: [2507.04984](https://arxiv.org/abs/2507.04984)  
**代码**: [项目页面](https://github.com/)  
**领域**: 视频帧插值/扩散模型  
**关键词**: 视频帧插值, 布朗桥扩散, 时域感知自编码器, 3D小波, 光流引导

## 一句话总结

提出 TLB-VFI，一种高效的视频扩散模型用于帧插值：通过时域感知自编码器（隐空间时域块+像素空间3D小波门控）提取丰富的时间信息，结合重新设计的布朗桥扩散过程，在参数量仅 46.7M（比图像扩散方法少 3×、比视频扩散方法少 20×）的情况下，在 SNU-FILM extreme 和 Xiph-4K 上 FID 提升约 20%。

## 研究背景与动机

### 问题定义

视频帧插值（VFI）旨在给定两帧相邻视频帧 $I_0$ 和 $I_1$，预测中间帧 $I_n$。核心挑战在于：（1）处理大幅运动变化；（2）保持时间一致性；（3）生成高质量且感知自然的中间帧。

### 已有方法的不足

**传统方法和图像扩散方法（LDMVFI, Consec.BB）**：仅提取空间信息，缺乏显式的时间信息提取。图像扩散方法比传统方法效率更低，且无法构建帧间时序关系

**视频扩散方法（VIDIM, Dreammover, ViBiDSampler）**：虽然能提取时间信息，但：
   - 需要超过 1000 万视频训练（vs VFI 标准训练集仅 51K 三元组）
   - 模型参数量巨大（>943M）
   - 推理极慢（8.48-52.55 秒/帧）
   - 缺乏光流等像素级引导

**Consecutive Brownian Bridge**：将布朗桥应用于相邻帧之间，但相邻帧特征高度相似导致布朗桥退化为恒等映射，失去功能

### 核心动机

需要一个**视频扩散模型**来提取时间信息（3D UNet 在采样过程中逐步构建时序关系），但必须保持训练规模、模型大小和推理时间在合理范围内。关键在于：（1）重新设计自编码器使其能在隐空间和像素空间双重提取时间信息；（2）利用光流引导减少训练数据需求。

## 方法详解

### 整体框架

方法分为两个组件：（1）时域感知自编码器——包含图像编码器+时域块+图像解码器+3D小波门控，预测 mask $M$ 和残差 $\Delta$；（2）布朗桥扩散模型——在隐空间对齐 $\mathcal{E}(V)$ 和 $\mathcal{E}(\tilde{V})$ 的分布差异。最终输出：
$$\hat{I}_n = M \odot \text{warp}(I_0) + (1-M) \odot \text{warp}(I_1) + \Delta$$

### 关键设计

#### 1. **隐空间时域特征提取**

- **功能**：在自编码器的编码器和解码器之间插入时域块，在隐空间提取帧间时序信息
- **核心思路**：使用共享的图像编码器分别编码每帧得到空间特征，然后通过 3D 卷积 + 时空注意力在隐空间提取时间关系。解码器端使用时空交叉注意力聚合将视频特征（$F \in \mathbb{R}^{C \times T \times H \times W}$）转换为单帧特征：
  $$V_{out} = \text{softmax}\left(\frac{QK^T}{\sqrt{C}}\right)V$$
  其中 $F_Q = F[t].\text{flatten}(1)$（中间帧特征），$F_{KV} = F.\text{flatten}(1)$（所有帧特征）。同时利用 $I_0, I_1$ 编码器的多层级特征通过 warp + 交叉注意力引导解码器。
- **设计动机**：将时域提取与空间编码分离的关键原因是：推理时 $I_n$ 被替换为零矩阵（$\tilde{V}=[I_0,0,I_1]$），如果编码器本身是时域感知的，多层级特征会因零替换而包含不完整信息，损害解码性能。共享图像编码器确保 $I_0, I_1$ 的多层级特征不受影响。

#### 2. **3D 小波特征门控**

- **功能**：在像素空间利用 3D 小波变换提取时域高频信息（运动变化区域），作为隐空间时域信息的补充
- **核心思路**：对输入视频 $V=[I_0,I_n,I_1]$ 应用 3D 小波变换，使用低通滤波器 $[\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}]$ 和高通滤波器 $[-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}]$ 在高度、宽度、时间三个维度的组合上提取 8 种频率图。两次小波变换分别捕捉：（1）$I_0$ 与 $I_n$ 之间、$I_n$ 与 $I_1$ 之间的时间信息；（2）全帧的时间信息。将编码后的频率特征 $f_w$ 通过门控机制融合：
  $$f = \sigma(f_w) \odot f_i + f_i$$
  其中 $\sigma$ 是 sigmoid，$f_i$ 是图像编码器的隐特征。
- **设计动机**：隐空间被高度压缩，仅在隐空间提取时间信息不够——像素级的时间信息能告诉模型哪些区域运动变化更剧烈，通过门控机制引导模型聚焦这些区域。

#### 3. **重新设计的布朗桥扩散过程**

- **功能**：将布朗桥应用于 $\mathcal{E}(V)$（含 $I_n$ 的完整视频编码）和 $\mathcal{E}(\tilde{V})$（$I_n$ 替换为零的视频编码）之间，避免恒等映射问题
- **核心思路**：BBDM 的扩散过程：
  $$q(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_T) = \mathcal{N}\left(\frac{t}{T}\mathbf{x}_0 + (1-\frac{t}{T})\mathbf{x}_T, \frac{t(T-t)}{T}\mathbf{I}\right)$$
  其中 $\mathbf{x}_0 = \mathcal{E}(V)$，$\mathbf{x}_T = \mathcal{E}(\tilde{V})$。采样过程中去噪网络预测 $\mathbf{x}_t - \mathbf{x}_0$。
  
  **Proposition 1**：布朗桥扩散有效的充分条件是 $\mathbb{E}(\mathbf{x}_0)$ 和 $\mathbb{E}(\mathbf{x}_T)$ 之间存在显著分布偏移（可通过 t-test 验证 $H_0: \mathbb{E}(\mathbf{x}_0 - \mathbf{x}_T)=0$ 被拒绝）。
  
  本文设计中 MAPE 达到 40-50%（vs Consec.BB 的 <1%），t-statistic>21（远超 0.001 显著性水平阈值 3.291）。
- **设计动机**：Consec.BB 将布朗桥应用在相邻帧特征（几乎相同）之间导致恒等映射。本文通过零替换 $I_n$ 制造显著分布差异，使布朗桥真正发挥信息恢复的功能。

### 损失函数 / 训练策略

- 自编码器：L1 重建损失 + 感知损失
- 扩散模型：预测 $\mathbf{x}_t - \mathbf{x}_0$ 的 MSE 损失
- 训练数据：Vimeo 90K 三元组（51K）
- 数据增强：随机翻转、裁剪、旋转、时间顺序反转
- 推理：10 步扩散采样

## 实验关键数据

### 主实验

**LPIPS↓/FloLPIPS↓/FID↓ 主要结果**：

| 方法 | 参数量 | Xiph-4K | SNU-FILM extreme | 运行时间(s/帧) |
|------|--------|---------|-------------------|---------------|
| LDMVFI | 439.0M | OOM | 0.123/0.204/47.04 | 2.48 |
| Consec.BB | 146.4M | 0.097/0.135/24.42 | 0.104/0.184/36.63 | 1.62 |
| PerVFI* | — | 0.086/0.128/18.85 | 0.090/0.151/32.37 | 1.52 |
| **Ours** | **46.7M** | **0.077/0.113/19.11** | **0.095/0.151/29.87** | **0.69** |

*PerVFI 训练规模约为其他方法 2 倍（含高分辨率数据），灰色标注不参与排名。

### 消融实验

**时域感知设计消融（FID↓）**：

| 配置 | Xiph-4K | Xiph-2K | SNU-FILM extreme |
|------|---------|---------|-------------------|
| Full model | **19.114** | **9.901** | **29.868** |
| - 3D Wavelet | 19.247 | 10.092 | 30.717 |
| - Cross-attn aggregation | 19.663 | 10.499 | 30.903 |
| - Temporal attention | 19.944 | 10.911 | 32.061 |
| - 3D Convolution (→2D) | 23.481 | 12.679 | 33.155 |
| Ours†(时域编码器) | 22.731 | 13.410 | 34.982 |

### 关键发现

1. **3D 卷积是最关键组件**：移除后 FID 退化最严重（19.11→23.48 on Xiph-4K），因为这是隐空间时域信息提取的基础
2. **共享图像编码器不可替代**：使用时域编码器（Ours†）导致性能大幅下降，验证了多层级特征不受零替换影响的设计必要性
3. **3D 小波门控提供互补的像素级信息**：虽然单独贡献较小（FID 仅恶化 0.1），但在难例上效果更明显
4. **布朗桥分布偏移的重要性**：t-statistic 从 Consec.BB 的 0.0001 提升到本文的 21+，验证了 Proposition 1
5. **PSNR/SSIM 与视觉质量不一致**：论文展示 EMAVFI 在视觉上明显失真但 PSNR/SSIM 反而更高的案例，强调应使用 LPIPS/FloLPIPS/FID 评估

## 亮点与洞察

1. **训练效率极高**：仅用 51K 三元组（vs 视频扩散方法的 >10M 视频），参数量 46.7M（vs VIDIM 的 >1B），训练成本降低 3-4 个数量级
2. **理论贡献**：Proposition 1 给出了布朗桥扩散有效性的充分条件，解释了为什么 Consec.BB 中布朗桥退化
3. **2.3× 推理加速**：相比同类扩散方法在相同采样步数下更快（0.69s vs 1.62s）
4. **困难样本上优势更大**：在 SNU-FILM 系列从 easy→extreme 的性能提升幅度递增，说明时域信息对大运动场景尤为关键

## 局限与展望

1. **推理仍需 10 步扩散采样**：虽然已比同类方法快，但仍慢于非扩散方法（IFRNet 0.10s）
2. **仅评估三帧插值**：对连续多帧插值（如 8× 插帧）的效果在补充材料中展示但未深入分析
3. **光流估计质量依赖**：当光流估计失败（如遮挡、透明区域），残差项 $\Delta$ 能否完全补偿未知
4. **未在非自然视频上评估**：动画、游戏画面等合成视频的效果未知
5. **3D 小波的计算开销**：虽然轻量但两次小波变换仍增加了像素空间处理成本

## 相关工作与启发

- 与 Consec.BB 的核心区别：Consec.BB 在相邻帧特征间做布朗桥（≈恒等映射），本文在完整/零替换视频编码间做布朗桥（大分布偏移）
- 与 LDMVFI 的区别：LDMVFI 使用核方法引导自编码器、基于图像的扩散；本文使用光流引导、基于视频的扩散
- 启发：分离空间编码和时域编码的设计思路可推广到视频修复、去模糊等任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 重新设计布朗桥端点和时域感知自编码器有创新性，Proposition 1 有理论贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多数据集、多指标（LPIPS/FloLPIPS/FID）、运行时间、训练成本对比、定性分析等非常全面
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，动机逻辑链完整，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ — 在保持 SOTA 质量的同时大幅降低训练和推理成本，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](../../CVPR2025/image_generation/eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[CVPR 2025\] Hierarchical Flow Diffusion for Efficient Frame Interpolation](../../CVPR2025/image_generation/hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ICML 2025\] Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion](../../ICML2025/image_generation/synthetic_face_datasets_generation_via_latent_space_exploration_from_brownian_id.md)

</div>

<!-- RELATED:END -->
