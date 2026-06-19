---
title: >-
  [论文解读] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs
description: >-
  [ICCV 2025][3D视觉][3D高斯喷溅] 本文提出 CodecGS，通过将 3DGS 的所有高斯属性用紧凑的 Tri-plane 特征平面表示，并结合频率域 DCT 熵建模和通道级比特分配策略，使特征平面能高效利用标准视频编解码器（HEVC）压缩，实现在保持高渲染质量的同时将存储大小减少至约10MB以内（相比原始3DGS压缩比高达146×）。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D高斯喷溅"
  - "压缩"
  - "特征平面"
  - "视频编解码器"
  - "熵建模"
---

# Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs

**会议**: ICCV 2025  
**arXiv**: [2501.03399](https://arxiv.org/abs/2501.03399)  
**代码**: [https://fraunhoferhhi.github.io/CodecGS](https://fraunhoferhhi.github.io/CodecGS)  
**领域**: 3D Vision / 3DGS Compression  
**关键词**: 3D高斯喷溅, 压缩, 特征平面, 视频编解码器, 熵建模

## 一句话总结

本文提出 CodecGS，通过将 3DGS 的所有高斯属性用紧凑的 Tri-plane 特征平面表示，并结合频率域 DCT 熵建模和通道级比特分配策略，使特征平面能高效利用标准视频编解码器（HEVC）压缩，实现在保持高渲染质量的同时将存储大小减少至约10MB以内（相比原始3DGS压缩比高达146×）。

## 研究背景与动机

- **3DGS** 以高渲染质量和速度著称，但表示一个3D场景通常需要数百万个高斯体，占用数百MB甚至数GB存储
- 这使得在移动设备、头盔显示器等资源受限设备上的部署面临巨大挑战
- **现有压缩方法**主要通过点剪枝（pruning）+ 向量量化（VQ）减少高斯数量和属性大小，但往往：
    - 专注于减少渲染失真而忽略高斯属性间的冗余
    - 未能充分利用高斯属性的空间相关性
    - 无法与成熟的视频编解码器技术衔接
- **视频编解码器**（HEVC/VVC）在率失真优化方面非常成熟，已有工作将其用于 NeRF 压缩，但3DGS的非结构化特性使其难以直接适配
- **核心动机**：能否将3DGS属性组织为结构化的2D特征平面，从而利用标准视频编解码器的高效压缩能力？

## 方法详解

### 整体框架

1. 使用原始3DGS训练15k迭代完成点密化
2. 用 Tri-plane 特征平面 + MLP 解码器预测所有高斯属性（颜色、缩放、旋转、不透明度）
3. 通过 DCT 域熵建模优化特征平面使其更适合视频编解码
4. 利用通道重要性评分进行比特分配
5. 用标准 HEVC 编码器压缩特征平面

### 关键设计

1. **Tri-plane 特征平面表示**：

    - 采用 k-planes 模型的静态版本（Tri-plane），对3D位置通过三个平面（XY、XZ、YZ）的 Hadamard 积分解得到紧凑特征
    - 每个属性（颜色、缩放、旋转、不透明度）由独立的小型 MLP $g$ 解码
    - 每个平面 $512 \times 512$ 分辨率，8通道，共32通道预测所有属性
    - **两阶段训练**：先用原始3DGS训练15k迭代完成密化，再切换到特征平面训练，避免平面优化与点密化的冲突

2. **渐进训练（Progressive Training）**：

    - 采用通道级渐进掩码策略：在迭代阶段 $T_i$，仅更新 $[0, L_i]$ 通道
    - $T_i = \{0, 5000, 10000, 15000\}$，$L_i = \{2, 4, 6, 8\}$
    - 低层通道捕获全局/低频信息，高层通道捕获高频细节
    - 形成多级表示，为后续的通道级比特分配提供天然基础

3. **频率域 DCT 熵建模**：

    - 核心观察：标准视频编解码器内部使用 DCT 进行频率域压缩
    - 不直接最小化空间域参数的熵，而是对特征平面施加 $N \times M$ 块级 DCT 变换 $\mathcal{F}$，然后最小化变换系数的熵：$I(\mathcal{F}(\mathcal{P}))$
    - 使用 $4 \times 4$ 块大小（与标准编解码器最小 TU 单元一致）
    - 量化步长 $Q_{\text{step}} = 2^8$（配合16位标量量化）
    - 相比 $\mathcal{L}_1$ 稀疏化，DCT 熵建模以块级近似方式保留信号，更有效地维持原始信息

4. **通道重要性比特分配（Channel Importance-based Bit Allocation）**：

    - 利用通道重要性评分 $CI_c(\mathcal{P}) = \frac{1}{\sum P_i} \left| \frac{\partial E_i}{\partial \mathcal{P}_c} \right|$ 衡量每个通道对视觉质量的敏感度
    - 定义权重 $w_c = CI_1 / CI_c$，高层通道（低重要性）获得更高权重，从而压缩更多
    - 加权熵损失：$\mathcal{L}_{\text{ent}} = \sum_c w_c I(\mathcal{F}(\mathcal{P}_c))$
    - 自动确定每个通道的比特分配，避免穷举超参搜索

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{render}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}} + \lambda_1 \mathcal{L}_1$$

- $\mathcal{L}_{\text{render}}$：标准3DGS渲染损失
- $\mathcal{L}_{\text{ent}}$：DCT域加权熵损失，在30k迭代后启用
- $\mathcal{L}_1$：特征平面的 $\ell_1$ 正则化，减少未占用区域的噪声
- 通道重要性在30k迭代时计算并固定
- 训练后特征平面归一化到 $[0,1]$ 并缩放为16位整数，以 YUV400 格式送入 HEVC

## 实验关键数据

### 主实验 (表格)

**Mip-NeRF360 / DeepBlending / Tank&Temples**：

| 方法 | PSNR↑ (360) | Size↓ (360) | PSNR↑ (DB) | Size↓ (DB) | PSNR↑ (T&T) | Size↓ (T&T) |
|------|------------|------------|------------|------------|-------------|-------------|
| 3DGS | 27.49 | 745 MB | 29.42 | 664 MB | 23.69 | 431 MB |
| LightGaussian | 27.00 | 44.5 | 27.01 | 33.9 | 22.83 | 22.4 |
| C3DGS | 26.98 | 28.8 | 29.38 | 25.3 | 23.32 | 17.3 |
| CompGS | 27.26 | 16.5 | 29.69 | 8.77 | 23.71 | 9.61 |
| HAC | 27.53 | 15.3 | **30.19** | 7.46 | 23.70 | 8.44 |
| **Ours** | 27.30 | **9.78** | 29.82 | 8.62 | 23.63 | **7.46** |

> 在 Mip-NeRF360 上实现 **76× 压缩**，仅损失 0.19dB PSNR。三个数据集均在约10MB以内。

### 消融实验 (表格)

| PC | $w_c$ | $\mathcal{L}_{\text{ent}}$ | PR | $\mathcal{L}_1$ | PSNR(dB) | Size(MB) |
|----|-------|----------|----|----|----------|----------|
| ✓ | | | | | 27.27 | 23.45 |
| ✓ | | | ✓ | | 27.40 | 22.81 |
| ✓ | | ✓ | ✓ | | 27.31 | **10.68** |
| ✓ | ✓ | ✓ | ✓ | | 27.29 | 9.96 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 27.30 | 9.78 |

> - $\mathcal{L}_{\text{ent}}$ 是最关键组件：引入后大小从 22.81MB 骤降至 10.68MB，PSNR 仅下降 0.09dB
> - 通道比特分配 $w_c$ 进一步减少到 9.96MB
> - 每个组件都贡献了压缩性能

### 关键发现

- **DCT 熵建模**远优于简单的 $\mathcal{L}_1$ 稀疏化：后者导致显著的质量下降，而 DCT 以块级近似保留信号
- **渐进训练**自然形成多级表示：低层通道能量集中，高层稀疏，为差异化比特分配提供基础
- **与 HAC/CompGS 对比**：这些方法基于 Scaffold-GS 利用锚点关系，而本方法从原始3DGS出发，互补性好
- **视频编解码器的通用性**：方法同时适用于 HM（参考实现）和 FFmpeg libx265（硬件加速），后者编码仅需 25s
- **Piecewise-projective contraction** 将无界 360° 场景有效映射到有界平面，增强空间相关性

## 亮点与洞察

- **标准视频编解码器集成**是最大亮点：利用了数十年视频编码技术的积累，无需自定义解码器，硬件解码广泛可用
- **DCT 域熵建模**的设计洞察优雅：既然编解码器内部用DCT，就在训练时直接优化DCT系数的分布
- **通道重要性自动比特分配**避免了穷举调参，实用性强
- 渐进训练 + 通道重要性的组合形成了完整的"粗到细 + 差异化压缩"策略
- 方法独立于密化过程，与原始3DGS渲染管线完全兼容，渲染速度无额外开销

## 局限与展望

- 训练时间较长（~90分钟/场景），主要受限于网格方法的收敛速度
- 未引入点剪枝，点位置的存储占比随压缩率提高而增大，有进一步优化空间
- 在 Tank&Temples 上 SSIM/LPIPS 略低于原始3DGS，说明高频细节的保真度仍有提升空间
- 当前使用固定 QP=1 编码，更灵活的率控策略可能进一步优化率失真性能
- 可探索 VVC（下一代编解码器）替代 HEVC 以获得更好压缩效率

## 相关工作与启发

- **VideoRF / TeTriRF**：最先将标准视频编解码器用于 NeRF 张量分解压缩，但不适用于非结构化的3DGS
- **HAC / CompGS**：基于 Scaffold-GS 的3DGS压缩方法，利用锚点关系，与本文方法互补
- **Self-Organizing Gaussian**：利用排序算法+图像编解码器压缩高斯属性
- **k-planes**：Tri-plane 特征分解的理论基础
- 启发：将非结构化3D表示转化为结构化2D表示，就能桥接成熟的2D压缩技术

## 评分

- **新颖性**: ⭐⭐⭐⭐ — DCT域熵建模+通道重要性比特分配的组合设计有创新性
- **实验充分度**: ⭐⭐⭐⭐ — 三个标准数据集、RD曲线、详细消融、编解码器对比
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图表丰富，消融层次分明
- **价值**: ⭐⭐⭐⭐ — 实现了3DGS与标准视频编解码器的无缝集成，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Neural Compression for 3D Geometry Sets](neural_compression_for_3d_geometry_sets.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2025\] GIFStream: 4D Gaussian-based Immersive Video with Feature Stream](../../CVPR2025/3d_vision/gifstream_4d_gaussian-based_immersive_video_with_feature_stream.md)
- [\[CVPR 2025\] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video](../../CVPR2025/3d_vision/4dgc_rate-aware_4d_gaussian_compression_for_efficient_streamable_free-viewpoint_.md)
- [\[ICCV 2025\] CF³: Compact and Fast 3D Feature Fields](cf3_compact_and_fast_3d_feature_fields.md)

</div>

<!-- RELATED:END -->
