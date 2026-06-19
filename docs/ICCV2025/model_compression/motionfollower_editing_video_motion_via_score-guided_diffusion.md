---
title: >-
  [论文解读] MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion
description: >-
  [ICCV2025][模型压缩][视频运动编辑] 提出 MotionFollower，通过两个轻量卷积控制器（姿态+外观）和基于分数函数正则化的一致性引导机制，实现视频运动编辑，在 GPU 显存消耗减少约 80% 的同时超越 MotionEditor 等强基线。 - 现有视频编辑局限：当前扩散模型驱动的视频编辑主要集中在属性…
tags:
  - "ICCV2025"
  - "模型压缩"
  - "视频运动编辑"
  - "扩散模型"
  - "分数引导"
  - "轻量化控制器"
  - "姿态迁移"
---

# MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion

**会议**: ICCV2025  
**arXiv**: [2405.20325](https://arxiv.org/abs/2405.20325)  
**代码**: [francis-rings/MotionFollower](https://francis-rings.github.io/MotionFollower/)  
**领域**: 模型压缩  
**关键词**: 视频运动编辑, 扩散模型, 分数引导, 轻量化控制器, 姿态迁移

## 一句话总结

提出 MotionFollower，通过两个轻量卷积控制器（姿态+外观）和基于分数函数正则化的一致性引导机制，实现视频运动编辑，在 GPU 显存消耗减少约 80% 的同时超越 MotionEditor 等强基线。

## 研究背景与动机

- **现有视频编辑局限**：当前扩散模型驱动的视频编辑主要集中在属性级编辑（风格迁移、外观/背景修改），对视频中最独特且复杂的特征——运动信息——关注极少
- **MotionEditor 的不足**：作为唯一可编辑运动的先驱工作，MotionEditor 依赖 ControlNet + 注意力注入机制，存在三大问题：
    1. 时序一致性较差
    2. 计算开销巨大（ControlNet 含大量注意力操作）
    3. 面对大幅相机运动和复杂背景时性能退化
- **人体运动迁移 vs 运动编辑**：现有人体运动迁移方法（MagicAnimate、AnimateAnyone、Champ）旨在从单张图像生成动画，与视频运动编辑有本质差异——后者需同时保持相机运动、逐帧背景变化和主体外观
- **核心动机**：设计一个轻量、高效且能在复杂场景（大相机运动、复杂背景）下保持源视频细节的运动编辑框架

## 方法详解

### 整体架构

MotionFollower 基于 T2I 扩散模型（LDM/SD1.5）构建，将 2D U-Net 通过插入时序层（temporal layers）膨胀为 3D U-Net。整体包含两个核心设计：

1. **两个轻量条件控制器**：Pose Controller (PoCtr) + Reference Controller (ReCtr)
2. **推理阶段的分数引导一致性正则化**：双分支架构（重建分支 + 编辑分支）+ 多损失函数

### 轻量条件控制器

#### Pose Controller (PoCtr)
- **作用**：编码目标姿态信号以控制运动修改
- **结构**：仅由 4 个卷积块组成，每块包含 2 层卷积，**完全不含注意力计算**
- **工作方式**：将姿态信号编码为与初始潜变量同维度的表示，在去噪前直接加到潜变量上
- **初始化策略**：最终投影层权重初始化为零，避免训练初期引入过大干扰
- 对比 ControlNet：参数量和计算量大幅减少，且避免了从随机噪声直接生成修改运动时鲁棒性下降的问题

#### Reference Controller (ReCtr)
- **作用**：保留源视频的外观信息，替代耗时的 DDIM 反演
- **结构**：同样 4 个卷积块（每块 2 层卷积），将源帧特征下采样到与潜变量同维度
- **多尺度注入**：将多尺度源特征直接加到 U-Net 编码器各块的特征上
- **CLIP 嵌入**：源帧额外通过 CLIPProjector 转为 CLIP 嵌入，经交叉注意力与潜变量交互，保留精确的低级细节
- 对比 MagicAnimate/AnimateAnyone：后者使用完整 U-Net 作为参考网络，含大量注意力操作且需拼接操作增加通道维度，显存开销约为 ReCtr 的 2 倍

### 训练策略（两阶段）

| 阶段 | 数据 | 可训练模块 | 目标 |
|------|------|------------|------|
| 第一阶段（图像） | 单帧图像，源/目标随机从同一视频提取 | PoCtr + ReCtr + U-Net（移除时序层） | 学习给定源图像和目标姿态的图像编辑能力 |
| 第二阶段（视频） | 视频片段 + 40% 概率退化为图像训练 | 仅时序层（其余冻结） | 增强时序一致性，时序层由 AnimateDiff 预训练权重初始化 |

- 训练数据：3K 条互联网视频（60-90 秒），裁剪至 512×512
- 两阶段各训练 100K 步，学习率 1e-5
- 使用 DWPose 提取姿态，自研轻量分割模型提取 mask

### 分数引导一致性正则化（推理阶段核心）

**核心思想**：扩散模型的去噪过程可视为一个连续过程，由分数函数（log 概率密度对数据的梯度）引导去噪方向。通过在分数函数中加入正则化项，可强制模型在不更新权重的情况下保持源视频的语义细节。

**双分支架构**：
- **重建分支**：输入随机噪声 + 源姿态 + 源帧 → 重建源视频
- **编辑分支**：输入随机噪声 + 目标姿态 + 源帧 → 生成编辑视频

**分数函数分解**：

$$\nabla_{z_t^e} \log q(z_t^e, F_t^e, F_t^r) = \underbrace{\nabla_{z_t^e} \log q(z_t^e)}_{\text{原始去噪分数}} + \underbrace{\nabla_{z_t^e} \log q(F_t^e, F_t^r | z_t^e)}_{\text{一致性正则化}}$$

**区域损失设计**：利用分割 mask 将损失分为前景和背景两部分

#### 前景损失 $\mathcal{L}_{fg}$
- 最大化编辑分支和重建分支前景特征的相似度
- 使用 mask pooling + 空间平均余弦相似度
- 确保主体外观在运动修改后保持一致

#### 背景损失 $\mathcal{L}_{bg}$ （三个子损失）

| 损失 | Mask 定义 | 作用 |
|------|-----------|------|
| $\mathcal{L}_{over}$（重叠背景） | $(1-M^e) \odot (1-M^r)$ | 强制编辑/重建的重叠背景区域一致 |
| $\mathcal{L}_{body}$（非重叠人体） | $M^r \odot (1-M^e)$ | 减轻源姿态偏置导致的鬼影/模糊 |
| $\mathcal{L}_{com}$（补充） | 非重叠人体 vs 源背景 | 引导模型用背景信息修补非重叠人体区域 |

**最终引导**：通过 mask 组合前景和背景梯度，按区域更新潜变量：

$$\nabla_{z_t^e} \log q(F_t^e, F_t^r | z_t^e) = \frac{d\mathcal{L}_{fg}}{dz_t^e} \odot M^e + \frac{d\mathcal{L}_{bg}}{dz_t^e} \odot (1-M^e)$$

关键：**仅优化潜变量，不更新模型权重**。

## 实验关键数据

### 主实验（100 个野外视频）

| 模型 | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FID-VID↓ | FVD↓ | 显存↓ |
|------|-----|-------|-------|--------|------|----------|------|-------|
| MotionEditor | 9.13E-5 | 17.34 | 0.68 | 0.34 | 31.98 | 20.57 | 395.43 | 42.6G |
| MagicAnimate | 1.09E-4 | 16.22 | 0.62 | 0.35 | 33.04 | 26.59 | 477.65 | 21.8G |
| AnimateAnyone | 9.45E-5 | 16.18 | 0.65 | 0.32 | 35.81 | 28.31 | 515.57 | 16.1G |
| Champ | 9.94E-5 | 16.12 | 0.58 | 0.36 | 36.55 | 25.89 | 452.65 | 17.4G |
| **MotionFollower** | **6.31E-5** | **20.85** | **0.75** | **0.22** | **26.30** | **12.42** | **276.04** | **9.8G** |

- 所有指标全面领先，FVD 从 395→276（↓30%），显存 42.6G→9.8G（↓77%）
- 单张 A100 上 MotionEditor 只能编辑 16 帧，MotionFollower 可编辑 56 帧

### 人体运动迁移（TikTok 基准）

| 模型 | SSIM↑ | LPIPS↓ | FVD↓ |
|------|-------|--------|------|
| Champ | 0.773 | 0.235 | 170.20 |
| **MotionFollower** | **0.793** | **0.230** | **159.88** |

在更简单的运动迁移任务上同样取得最优结果。

### 消融实验

| 配置 | FID↓ | FID-VID↓ | FVD↓ | 显存 |
|------|------|----------|------|------|
| w/o ReCtr | 64.82 | 47.28 | 545.39 | 8.3G |
| 替换 ReCtr→RNet | 26.35 | 13.12 | 288.57 | **17.8G** |
| 替换 PoCtr→CNet | 30.57 | 23.75 | 381.52 | **15.2G** |
| w/o 分数引导 | 35.91 | 28.10 | 437.69 | 7.1G |
| **完整模型** | **26.30** | **12.42** | **276.04** | **9.8G** |

- 去掉 ReCtr 性能断崖式下降，说明外观控制至关重要
- 用 RNet 替换 ReCtr 显存翻倍但性能略差，证明纯卷积方案更优
- 分数引导对背景一致性贡献显著

## 亮点与洞察

1. **极致轻量化设计**：用纯卷积控制器替代 ControlNet 和 Reference Net，GPU 显存减少 80%，这是一个"少即是多"的优秀示范——复杂的注意力机制反而会破坏参考网络与 U-Net 之间的分布关系
2. **分数引导的优雅性**：不修改模型权重，仅通过损失梯度在推理时优化潜变量，这种"测试时优化"可无缝迁移到其他扩散模型
3. **精细的区域损失设计**：通过分割 mask 将前景/背景/重叠区域/非重叠区域分别施加不同约束，解决了运动编辑中的鬼影、模糊和背景不一致问题
4. **长视频与大相机运动**：MotionEditor 在 600 帧长视频和大幅相机运动场景中出现严重退化，MotionFollower 凭借一致性引导有效应对
5. **两阶段训练 + 混合训练策略**：第二阶段 40% 概率退化为图像训练，有效防止时序层训练破坏单帧编辑能力

## 局限与展望

1. **依赖姿态检测质量**：PoCtr 依赖 DWPose 提取姿态，当姿态检测失败（严重遮挡、非常规姿态）时编辑质量可能下降
2. **分辨率限制**：训练和测试均在 512×512，未探索高分辨率（1024+）场景的可扩展性
3. **基于 SD1.5 架构**：未利用更新的 SDXL 或 DiT 架构，可能限制生成质量上限
4. **推理速度**：24 帧需 50 秒（A100），实时应用仍有差距；分数引导需双分支前向传播，增加了推理开销
5. **仅处理人体运动**：当前仅针对人体姿态驱动的运动编辑，未涉及物体运动或场景级运动编辑
6. **分割模型依赖**：分数引导需要准确的前景/背景分割 mask，分割错误会传播到编辑结果

## 相关工作与启发

- **MotionEditor**（ICLR 2024）：运动编辑先驱，ControlNet+注意力注入，本文的主要对标对象
- **AnimateDiff**（ICLR 2024）：提供时序层预训练权重，MotionFollower 第二阶段训练的初始化来源
- **MagicAnimate / AnimateAnyone / Champ**：人体运动迁移方法，使用完整 U-Net 作为参考网络，本文证明纯卷积替代方案更优
- **分数引导思想**：源自 SDE 和 classifier guidance 的思路，但本文创新地用于区域级一致性约束而非全局引导
- **启发**：轻量化控制器 + 测试时潜变量优化的范式可推广到其他条件视频生成任务（如视频修复、视频超分）

## 评分
- 新颖性: ⭐⭐⭐⭐ — 分数引导一致性正则化和纯卷积控制器设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 定量/定性/消融/人类评估/长视频/相机运动全覆盖
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ — 80%显存节省有重要实用价值，轻量化设计思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[ICLR 2026\] DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing](../../ICLR2026/model_compression/diffvax_optimization-free_image_immunization_against_diffusion-based_editing.md)
- [\[CVPR 2026\] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](../../CVPR2026/model_compression/prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[CVPR 2026\] F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling](../../CVPR2026/model_compression/f2hdr_two-stage_hdr_video_reconstruction_via_flow_adapter_and_physical_motion_mo.md)

</div>

<!-- RELATED:END -->
