---
title: >-
  [论文解读] VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction
description: >-
  [CVPR 2025][人体理解][3D虚拟试穿] 提出 VTON 360，通过将 3D 虚拟试穿重新建模为多视角一致的 2D 虚拟试穿扩展问题，结合伪 3D 姿态表示、多视角空间注意力和多视角 CLIP 嵌入三项技术，实现从任意视角的高保真虚拟试穿。 虚拟试穿（VTON）在电商和时尚设计中需求巨大。2D VTON 方法虽已…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "3D虚拟试穿"
  - "多视角一致性"
  - "高斯溅射"
  - "扩散模型"
  - "服装纹理保持"
---

# VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction

**会议**: CVPR 2025  
**arXiv**: [2503.12165](https://arxiv.org/abs/2503.12165)  
**代码**: [项目页面](https://scnuhealthy.github.io/VTON360)  
**领域**: 人体理解  
**关键词**: 3D虚拟试穿, 多视角一致性, 高斯溅射, 扩散模型, 服装纹理保持

## 一句话总结

提出 VTON 360，通过将 3D 虚拟试穿重新建模为多视角一致的 2D 虚拟试穿扩展问题，结合伪 3D 姿态表示、多视角空间注意力和多视角 CLIP 嵌入三项技术，实现从任意视角的高保真虚拟试穿。

## 研究背景与动机

虚拟试穿（VTON）在电商和时尚设计中需求巨大。2D VTON 方法虽已取得显著进展，但无法支持多视角渲染。传统 3D VTON 方法要么依赖昂贵的 3D 扫描设备，要么从 2D 图像重建 3D 服装模型但缺乏多视角信息导致保真度不足。

DreamVTON 等基于 SDS 的方法虽可任意视角渲染，但 T2I 模型学习的是语义级"概念"而非像素级精确，保真度有限。GaussianVTON 将 3D VTON 建模为场景编辑任务，但由于没有 2D VTON 方法能生成多视角 3D 一致的图像，服装的保真度和一致性仍存在问题。

本文的核心洞察是：3D 模型与其多视角 2D 渲染图像之间存在等价关系，因此可将 3D VTON 转化为在多视角 2D 图像上的一致编辑问题，并通过 3D 重建恢复编辑后的 3D 模型。

## 方法详解

### 整体框架

给定输入 3D 人体模型 $\mathbf{G}_{\text{src}}$ 和一对前后视角服装图像 $(g_f, g_b)$，方法分三步：(1) 将 3D 模型渲染为多视角 2D 图像；(2) 使用增强的 2D VTON 网络对多视角图像进行 3D 一致编辑；(3) 用 Gaussian Splatting 重建编辑后的 3D 模型 $\mathbf{G}_{\text{VTON}}$。核心创新在第二步，扩展典型 2D VTON 框架（Main UNet + GarmentNet）以支持多视角一致生成。

### 关键设计1：伪 3D 姿态表示

**功能**：替代传统 DensePose 提供跨视角几何一致的人体姿态表示。

**核心思路**：使用 SMPL-X 3D 人体模型导出的法线图 $\mathbf{N}$ 作为姿态条件，通过轻量级 PoseEncoder $\mathcal{E}'$ 编码后送入 Main UNet。法线图捕获细粒度的表面朝向信息，在不同视角间保持几何结构一致性。

**设计动机**：DensePose 对每个身体部位分配统一语义标签，缺乏 3D 几何一致性，在多视角下会产生伪影和时序不一致（如肢体边界处理不当）。法线图提供更平滑、几何一致的过渡，支持真实感着色效果。

### 关键设计2：多视角空间注意力（MVAttention）

**功能**：建模不同视角特征间的相关性，确保多视角生成的 3D 一致性。

**核心思路**：受视频生成中时序注意力启发，设计空间注意力层。Query 来自多视角特征 $\mathbf{F}^l$，Key/Value 为多视角特征与前后服装特征的拼接 $[\mathbf{F}^l \oplus F_f^l \oplus F_b^l]$。关键创新是引入基于相机旋转矩阵角度差异的"相关性"矩阵 $C$：

$$C_{ij} = ((\text{trace}(R_i^T R_j) - 1) / 2 + 1) / 2$$

相似视角获得更高相关权重，远离视角权重较低，注意力得分被 $C_{ij}$ 调制。

**设计动机**：多视角输入来自非均匀空间间隔的随机方位角，相似视角的特征相关性高而不同视角相对独立。直接使用标准注意力无法建模这种空间关系。

### 关键设计3：多视角 CLIP 嵌入

**功能**：在 CLIP 特征中注入相机视角信息，使网络学习与特定视角相关的服装特征。

**核心思路**：将相机旋转矩阵提取为 9 维张量 $\mathbf{r}_i$，经位置编码后通过 MLP 投影到与 CLIP 嵌入相同维度，与服装 CLIP 特征 $F^g$ 沿 token 轴拼接形成 $Y_i = F^g \oplus \text{MLP}(F_i^c)$，用于 Main UNet 的 cross-attention 层。

**设计动机**：标准 2D VTON 的 CLIP 嵌入不含视角信息，无法区分不同视角下应呈现的服装细节（如前面的 logo vs 后面的标签），注入相机条件可提升视角感知能力。

### 损失函数

使用标准 LDM 去噪损失 $\mathcal{L}_{\text{ldm}} = \mathbb{E}[\|\epsilon - \hat{\epsilon}_\theta(z_t, t, \eta, \psi, \zeta)\|_2^2]$，其中 $\eta$ 为服装和法线图的潜在表示，$\psi$ 为多视角 CLIP 嵌入，$\zeta$ 为去衣人体图像。训练分两阶段：先单视角训练基础能力，后多视角训练 MVAttention 模块。

## 实验关键数据

### 主实验：与 SOTA 方法对比

| 方法 | CLIP_cons ↑ | DINO_sim ↑ | Vote_quality | Vote_align |
|------|------------|-----------|-------------|-----------|
| DreamWaltz | 0.887 | 0.556 | 0.46% | 1.54% |
| TIP-Editor | 0.939 | 0.569 | 0.92% | 0.62% |
| GaussCtrl | 0.931 | 0.577 | 1.08% | 1.38% |
| **VTON 360** | 0.923 | **0.633** | **97.54%** | **96.46%** |

（Thuman2.0 数据集；MVHumanNet 上趋势一致，DINO_sim 0.623 vs 次优 0.521）

### 消融实验：三项技术的逐步贡献（Thuman2.0）

| 配置 | CLIP_cons ↑ | DINO_sim ↑ |
|------|-----------|-----------|
| 2D-VTON baseline | 0.892 | 0.609 |
| + 伪3D姿态 | 0.910 | 0.626 |
| + 多视角CLIP嵌入 | 0.913 | 0.631 |
| + MVAttention | **0.923** | **0.633** |

### 关键发现

- DINO 相似度（服装纹理保持能力）显著超越所有基线方法，主要得益于像素级的细节传递而非语义级概念。
- 用户投票中获得 97.54% 的质量偏好和 96.46% 的对齐偏好，说明在主观感知上优势压倒性。
- 在电商平台服装（YOOX、淘宝、TikTok）上展现良好泛化能力，可准确保持条纹、logo、纽扣等细节。
- 伪 3D 姿态对肢体生成质量改善最大，MVAttention 进一步增强跨视角一致性。

## 亮点与洞察

1. **问题建模转换巧妙**：将 3D VTON 转化为"多视角一致的 2D VTON 扩展"，利用成熟的 2D VTON 技术栈，避免了从零搭建 3D 流程。
2. **相关性矩阵设计精妙**：利用相机旋转矩阵的 trace 关系构建视角间相关性权重，物理含义直观且计算简单。
3. **训练策略分阶段**：先单视角学基础 VTON 能力，后多视角学一致性，降低了训练难度。

## 局限与展望

- 需要前后两张服装图像作为输入，限制了单图服装输入的应用场景。
- 训练受限于 GPU 内存，训练时仅用 8 个视角（测试 16 个），更多视角可能进一步提升效果。
- 依赖 SMPL-X 拟合质量，对非标准体型或复杂姿态的鲁棒性有待验证。
- 输入分辨率固定为 $768 \times 576$，对高分辨率细节保持有影响。
- 未处理复杂配饰（如围巾、帽子）和下装的试穿场景。

## 相关工作与启发

- **CatVTON / OOTDiffusion**：代表性 2D VTON 方法，本文以其 GarmentNet+Main UNet 框架为基础进行 3D 扩展。
- **GaussianVTON**：同期工作，也使用 2D VTON + 3DGS 路线，但缺少多视角一致性设计导致效果不佳。
- **GaussCtrl**：利用深度条件和注意力对齐实现 3D 感知编辑，启发了多视角一致编辑思路。

## 评分

⭐⭐⭐⭐ — 问题建模清晰，三项技术设计动机明确且效果显著，用户研究以压倒性优势胜出。限制在于需要双视角服装输入和对 SMPL-X 的依赖。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](../../CVPR2026/human_understanding/mobile_vton_ondevice_virtual_tryon.md)
- [\[ECCV 2024\] Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment](../../ECCV2024/human_understanding/wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)
- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[CVPR 2025\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)

</div>

<!-- RELATED:END -->
