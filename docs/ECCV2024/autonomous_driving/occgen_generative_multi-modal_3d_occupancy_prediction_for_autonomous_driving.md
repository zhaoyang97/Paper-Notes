---
title: >-
  [论文解读] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving
description: >-
  [ECCV 2024][自动驾驶][3D Occupancy Prediction] OccGen 将 3D 语义占用预测重新定义为"noise-to-occupancy"的生成式范式，通过条件编码器提取多模态特征、渐进式精炼解码器执行扩散去噪，以由粗到精的方式逐步生成占用图，在 nuScenes-Occupancy 上多模态/纯LiDAR/纯相机设置下分别相对提升 9.5%/6.3%/13.3% 的 mIoU。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "3D Occupancy Prediction"
  - "扩散模型"
  - "多模态"
  - "Generative Perception"
  - "Coarse-to-Fine Refinement"
---

# OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving

**会议**: ECCV 2024  
**arXiv**: [2404.15014](https://arxiv.org/abs/2404.15014)  
**代码**: [https://occgen-ad.github.io/](https://occgen-ad.github.io/)  
**领域**: 自动驾驶  
**关键词**: 3D Occupancy Prediction, diffusion model, Multi-modal Fusion, Generative Perception, Coarse-to-Fine Refinement

## 一句话总结

OccGen 将 3D 语义占用预测重新定义为"noise-to-occupancy"的生成式范式，通过条件编码器提取多模态特征、渐进式精炼解码器执行扩散去噪，以由粗到精的方式逐步生成占用图，在 nuScenes-Occupancy 上多模态/纯LiDAR/纯相机设置下分别相对提升 9.5%/6.3%/13.3% 的 mIoU。

## 研究背景与动机

**领域现状**：3D 语义占用预测是自动驾驶的核心感知任务，旨在为感知范围内每个体素分配语义标签，比 BEV 表示更能保留垂直维度信息。

**现有痛点**：现有方法（LiDAR-based、vision-based、multi-modal）将占用预测视为一次性 (one-shot) 的体素分割问题，用单次前向推理完成预测，但这种判别式方法存在两个问题：(1) 只学习输入到输出的映射而忽略了占用图的分布建模；(2) 单次推理不足以完成精细场景的补全。

**核心矛盾**：判别式方法缺乏逐步精炼和场景想象力，无法像人类一样通过持续观察来完善对整个场景的感知。

**本文目标**：如何在占用预测中引入由粗到精的逐步精炼范式，同时支持不确定性估计。

**切入角度**：扩散模型的去噪过程天然能建模占用图从粗到精的精炼过程——从随机高斯噪声逐步生成精确的占用预测。

**核心 idea**：提出 OccGen，采用"noise-to-occupancy"的生成式范式，条件编码器只运行一次，解码器多步迭代精炼，实现与单次前向方法相当的推理延迟。

## 方法详解

### 整体框架

OccGen 是一个编码器-解码器结构的生成式感知模型。条件编码器 (conditional encoder) 处理多模态输入提取条件特征（仅运行一次），渐进式精炼解码器 (progressive refinement decoder) 使用多模态特征作为条件，通过扩散去噪从 3D 高斯噪声图逐步精炼生成占用预测。训练阶段将高斯噪声添加到 GT 占用图上，学习去噪；推理阶段从纯高斯噪声出发逐步去噪生成占用。

### 关键设计

1. **Noise-to-Occupancy 生成范式**:

    - 功能：将占用预测建模为从噪声到占用的逐步生成过程
    - 核心思路：通过 $T$ 步扩散过程 $Y_T \xrightarrow{f_\theta} Y_{T-1} \xrightarrow{f_\theta} \ldots \rightarrow Y_0$ 逐步精炼，每步通过模型 $f_\theta$ 预测并消除噪声。前向扩散过程为 $z_t = \sqrt{\alpha_t} z_0 + \sqrt{1-\alpha_t} Z$，其中 $Z \sim \mathcal{N}(0, I)$。每步精炼公式为 $\Delta Y_t = f_\theta(x, t, Y_{t+1})$，$Y_t = Y_{t+1} \oplus \Delta Y_t$
    - 设计动机：扩散去噪过程天然建模了 3D 占用图的由粗到精精炼，而非一次性生成。同时支持计算换精度的灵活权衡和不确定性估计

2. **条件编码器 (Conditional Encoder)**:

    - 功能：融合 LiDAR 和相机输入的多模态特征，生成条件信息供解码器使用
    - 核心思路：采用双流结构——LiDAR 流用 VoxelNet + 3D 稀疏卷积提取体素特征 $F_p$，相机流用 2D backbone + FPN 提取多视角图像特征后通过 Hard 2D-to-3D 视图变换得到相机体素特征 $F_c$。融合模块采用自适应加权：$W = \mathcal{G}_C([\mathcal{G}_C(F_p), \mathcal{G}_C(F_c)])$，$F_m = \sigma(W) \odot F_p + (1-\sigma(W)) \odot F_c$
    - 设计动机：(1) Hard 2D-to-3D 视图变换用 Gumbel-Softmax 替代传统 softmax 深度估计，保证更精确的深度（可微one-hot编码）；(2) 几何掩码 (Geometry Mask) 利用 LiDAR 体素特征生成掩码约束相机特征的空间分布，弥补相机深度模糊性

3. **渐进式精炼解码器 (Progressive Refinement Decoder)**:

    - 功能：以条件特征为引导，对噪声占用图进行多轮迭代去噪精炼
    - 核心思路：解码器由多层 Refinement Layer 和占用头组成。每层包含三个核心操作：(1) 3D 可变形交叉注意力 (DCA)：从条件输入中学习知识，$\text{DCA}_{3D}(Y_t^i, F_m) = \sum_{n \in F_m} \text{DA}_{3D}(q, \text{proj}(q,n), F_m)$；(2) 3D 可变形自注意力 (DSA)：增强自补全能力 $\text{DSA}_{3D}(Y_t^i, Y_t^i)$；(3) 时间扩散模块：利用步索引 $t$ 的嵌入对噪声图执行 scale-and-shift 操作 $Y_t^i := \text{Diff}(Y_t^i, \text{ToEmbed}(t))$。为效率考虑，先将噪声图下采样为多尺度 $Y_t^i \in \mathbb{R}^{D/2^i \times H/2^i \times W/2^i \times C_i}$
    - 设计动机：编码器只运行一次而解码器多步迭代，不引入显著计算开销，使得推理延迟与单次前向方法相当

### 损失函数 / 训练策略

训练时使用 cosine noise schedule 将高斯噪声添加到 GT 占用图（经验发现 cosine schedule 优于 linear schedule）。总损失函数为：

$$\mathcal{L}_{total} = \mathcal{L}_{ce} + \mathcal{L}_{ls} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem} + \mathcal{L}_d$$

其中 $\mathcal{L}_{ce}$ 为交叉熵损失，$\mathcal{L}_{ls}$ 为 lovász-softmax 损失，$\mathcal{L}_{scal}^{geo}$ 和 $\mathcal{L}_{scal}^{sem}$ 为 scene-wise 和 class-wise 的 affinity loss，$\mathcal{L}_d$ 为深度损失。推理时采用 DDIM 策略和非对称时间间隔（$td=1$）。

## 实验关键数据

### 主实验

nuScenes-Occupancy 验证集结果：

| 设置 | 方法 | IoU | mIoU | 相对提升 |
|------|------|-----|------|---------|
| 多模态 | CONet | 29.5 | 20.1 | - |
| 多模态 | **OccGen** | **30.3** | **22.0** | +9.5% |
| 纯LiDAR | L-CONet | 30.9 | 15.8 | - |
| 纯LiDAR | **L-OccGen** | **31.6** | **16.8** | +6.3% |
| 纯相机 | C-CONet | 20.1 | 12.8 | - |
| 纯相机 | **C-OccGen** | **23.4** | **14.5** | +13.3% |

SemanticKITTI 验证集结果：

| 方法 | IoU | mIoU |
|------|-----|------|
| OccFormer | 36.50 | 13.46 |
| Symphonize | 41.44 | 13.44 |
| **OccGen** | **36.87** | **13.74** |

### 消融实验

| 配置 | IoU | mIoU | 说明 |
|------|-----|------|------|
| Baseline | 28.1 | 20.4 | 无编码器改进+无解码器改进 |
| +提出的编码器 | 28.6 | 20.7 | Hard LSS + Geo Mask |
| +提出的解码器 | 30.1 | 21.6 | 渐进式精炼解码器 |
| 完整 OccGen | 30.3 | 22.0 | 编码器+解码器 |
| w/o DSA | 30.1 | 21.4 | 去除自注意力 |
| w/o Diffusion | 29.3 | 21.7 | 去除扩散去噪 |

### 关键发现

- 解码器（渐进式精炼）贡献大于编码器改进，说明生成式范式本身是提升的关键
- 交叉注意力影响 > 自注意力，因为从条件输入中学习知识更关键
- 扩散模块去除后 mIoU 从 22.0% 降至 21.7%，证明时间扩散过程的必要性
- 多步推理可持续提升性能：1步21.7% → 3步22.0%，且无需重新训练即可调整计算-精度权衡
- 多模态 OccGen 比纯相机高 7.5% mIoU，比纯 LiDAR 高 5.8% mIoU

## 亮点与洞察

- **首次将扩散模型引入 3D 占用预测**：将传统判别式任务重新定义为生成式问题，赋予模型逐步精炼和不确定性估计能力
- **高效推理设计**：编码器只需运行一次，解码器多步迭代，推理延迟与单步方法相当（约 294ms vs CONet 的 286ms）
- **不确定性估计**：随机采样过程可自然计算体素级预测不确定性，这是判别式方法无法实现的
- **Hard 2D-to-3D 视图变换**：用 Gumbel-Softmax 实现可微的 one-hot 深度编码，比传统 softmax 更精确

## 局限与展望

- 多步推理带来的精度提升幅度有限（3步仅提升 0.3% mIoU），更多步后提升趋于饱和
- 当前生成过程在体素空间直接操作，未探索潜在空间的扩散可能更高效
- 与最新的纯视觉方法（如 FlashOcc, SparseOcc 等）的比较不够充分
- 多模态融合策略相对简单（自适应加权），可考虑更复杂的交互机制

## 相关工作与启发

- **vs CONet**: OccGen 以生成式方法替代判别式方法，在三种模态设置下全面超越，展示扩散范式对密集预测的优势
- **vs DiffusionDet / DDP**: 沿用了扩散模型从"noise-to-X"的范式，但扩展到了 3D 占用的稀疏体素空间，解决了 3D 场景中的高分辨率计算问题
- **vs MonoScene / VoxFormer**: 与单次前向方法不同，OccGen 可灵活权衡计算与精度，且提供不确定性信息

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将扩散模型引入 3D 占用预测，生成式范式有新意但扩散应用于感知已非首创
- 实验充分度: ⭐⭐⭐⭐ 两个benchmark + 详细消融，但缺少与更多最新方法的对比
- 写作质量: ⭐⭐⭐⭐ 条理清晰，方法描述详尽，公式推导完整
- 价值: ⭐⭐⭐⭐ 为占用预测提供了生成式新思路，不确定性估计和灵活推理有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving](unim2ae_multi-modal_masked_autoencoders_with_unified_3d_representation_for_3d_pe.md)
- [\[CVPR 2026\] Gau-Occ: Geometry-Completed Gaussians for Multi-Modal 3D Occupancy Prediction](../../CVPR2026/autonomous_driving/gau-occ_geometry-completed_gaussians_for_multi-modal_3d_occupancy_prediction.md)
- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)

</div>

<!-- RELATED:END -->
