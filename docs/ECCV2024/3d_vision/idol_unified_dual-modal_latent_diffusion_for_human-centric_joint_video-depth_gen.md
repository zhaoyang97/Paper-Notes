---
title: >-
  [论文解读] IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation
description: >-
  [ECCV 2024][3D视觉][视频生成] 提出IDOL框架，通过统一双模态U-Net和运动一致性损失，实现以人为中心的视频与深度图联合生成，显著优于现有方法。 领域现状： 基于扩散模型的人体视频生成已取得显著进展，但生成的视频缺乏深度信息，限制了AR/VR等需要空间感知的下游应用。 现有痛点： 判别式单目深度估计方法（…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "视频生成"
  - "深度估计"
  - "双模态扩散"
  - "人体动画"
  - "视频-深度对齐"
---

# IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation

**会议**: ECCV 2024  
**arXiv**: [2407.10937](https://arxiv.org/abs/2407.10937)  
**代码**: [https://yhzhai.github.io/idol/](https://yhzhai.github.io/idol/) (有项目页)  
**领域**: 3D视觉  
**关键词**: 视频生成, 深度估计, 双模态扩散, 人体动画, 视频-深度对齐

## 一句话总结

提出IDOL框架，通过统一双模态U-Net和运动一致性损失，实现以人为中心的视频与深度图联合生成，显著优于现有方法。

## 研究背景与动机

**领域现状**: 基于扩散模型的人体视频生成已取得显著进展，但生成的视频缺乏深度信息，限制了AR/VR等需要空间感知的下游应用。

**现有痛点**: 判别式单目深度估计方法（如MiDaS、HDNet）在合成图像上泛化能力差，生成不完整或过度简化的深度图；多视角方法难以控制人体外观和动作。

**核心矛盾**: 视频（3通道RGB序列）和深度（标量深度图序列）是本质不同的两种模态，现有预训练扩散模型仅针对单模态图像生成设计；同时在latent空间中维持精确的视频-深度空间对齐是一个困难问题。

**本文目标**: 如何设计一个统一框架，同时生成高质量的人体视频及其对应的深度图，且保证两者的时空对齐。

**切入角度**: 将深度图渲染为RGB热力图，将深度生成重新定义为"风格化视频生成"问题，从而可以直接利用预训练图像生成模型；并通过参数共享和一致性损失实现跨模态信息交互。

**核心 idea**: 统一双模态U-Net共享参数进行联合视频-深度去噪，辅以运动一致性损失和交叉注意力一致性损失促进精确的视频-深度空间对齐。

## 方法详解

### 整体框架

IDOL基于3D U-Net构建，采用两阶段训练：(1) 人体属性外绘预训练（HAOP）学习人体外观，(2) 联合视频-深度去噪训练。输入为人体前景图像 $f$、背景图像 $b$ 和姿态序列 $p=\{p_1,...,p_L\}$，输出视频 $v$ 和对应深度图序列 $d$。前景外观通过CLIP编码后送入cross-attention，姿势通过ControlNet控制，背景潜变量直接加到输入噪声上。

### 关键设计

1. **统一双模态U-Net (Unified Dual-Modal U-Net)**: 视频和深度去噪共享同一U-Net的架构和参数。通过一个可学习的模态嵌入（one-hot模态标签 $y_v$ 或 $y_d$）加到时间步嵌入上来控制输出模态。联合去噪目标为：

$$\mathcal{L}_{\text{denoise}} = \mathbb{E}\left[\|\epsilon_v - \epsilon_\theta(z_{v,t}, t, f, b, p; y_v)\|_2^2 + \|\epsilon_d - \epsilon_\theta(z_{d,t}, t, f, b, p; y_d)\|_2^2\right]$$

设计动机：共享参数不仅节省参数量（仅1.39B vs 2×1.39B），还能隐式学习来自深度的结构信息以提升视频质量。在每个U-Net block末端添加跨模态注意力层，将视频和深度特征拼接后做空间self-attention，实现显式的信息交互。

2. **运动一致性损失 (Motion Consistency Loss)**: 发现虽然视频和深度的中间特征具有相似的空间布局，但它们的时序运动模式可能不同步，导致输出不对齐。通过计算相邻帧特征间的代价体并归一化为运动场：

$$u_{v,l,i,j,h,k} = \frac{\exp(c_{v,l,i,j,h,k}/\tau)}{\sum_{h'}\sum_{k'}\exp(c_{v,l,i,j,h',k'}/\tau)}$$

然后最小化视频与深度运动场的MSE：$\mathcal{L}_{\text{mo}} = \frac{1}{LHWHW}\sum\|u_v - u_d\|_2^2$。

设计动机：U-Net中间层的自注意力特征包含语义信息，强制两模态特征运动同步可促进视频-深度对齐。

3. **交叉注意力图一致性损失 (Cross-Attention Map Consistency Loss)**: 交叉注意力图影响生成图像的空间布局。通过MSE损失强制视频流和深度流的交叉注意力图一致：

$$\mathcal{L}_{\text{xattn}} = \|M_v - M_d\|_2^2$$

设计动机：借鉴已有发现——交叉注意力图对图像布局有决定性影响，扩展到跨模态一致性场景。

4. **人体属性外绘预训练 (HAOP)**: 改进DisCo的HAP预训练：(a) 通过膨胀操作扩展背景mask，迫使模型填充前景周围的背景；(b) 对前景图像随机裁剪和缩放，促使模型外推部分人体属性。解决了当目标姿势偏离原始位置时出现明显背景遮罩的问题。

### 损失函数 / 训练策略

总训练损失：

$$\mathcal{L} = \mathcal{L}_{\text{denoise}} + \sum_{n=1}^{N}(w_{\text{mo}}\mathcal{L}_{\text{mo},n} + w_{\text{xattn}}\mathcal{L}_{\text{xattn},n})$$

一致性损失仅施加在U-Net上采样块中，因为ControlNet的特征融合发生在这些位置。

## 实验关键数据

### 主实验

| 方法 | TikTok FID-FVD↓ | TikTok FVD↓ | TikTok Depth L2↓ | NTU120 FID-FVD↓ | NTU120 FVD↓ | NTU120 Depth L2↓ |
|------|---------|---------|---------|---------|---------|---------|
| FOMM | 38.36 | 404.31 | - | 40.34 | 1439.50 | - |
| DisCo | 20.75 | 257.90 | 0.0975† | 26.21 | 458.92 | 0.0371† |
| LDM3D | 45.30 | 553.03 | 0.0637 | 71.11 | 587.84 | 0.0650 |
| MM-Diff | 48.92 | 771.32 | 0.0367 | 58.44 | 504.05 | 0.0404 |
| **IDOL** | **17.86** | **223.69** | **0.0336** | **20.23** | **314.82** | **0.0317** |

†表示从合成图像上用HDNet估计的深度。

### 消融实验

| 设置 | 参数量 | FID-FVD↓ | FVD↓ | Depth L2↓ | FID↓ |
|------|--------|---------|------|-----------|------|
| 分离U-Net | 2×1.39B | 24.28 | 282.50 | 0.0822 | 41.72 |
| 共享U-Net | 1.39B | 22.10 | 272.37 | 0.0369 | 39.43 |
| +跨模态注意力 | 1.41B | 19.28 | 260.65 | 0.0360 | 39.01 |
| +$\mathcal{L}_{\text{xattn}}$ | - | 19.99 | 244.58 | 0.0351 | 37.89 |
| +$\mathcal{L}_{\text{mo}}$+$\mathcal{L}_{\text{xattn}}$ | - | **17.86** | **223.69** | **0.0336** | **36.04** |

### 关键发现

- 共享U-Net参数量减半但性能更优，说明跨模态隐式结构学习是有益的
- 运动一致性损失同时改善视频、深度和图像质量三个指标
- IDOL的FLOPs最低（39.35T），参数最少（1.41B），推理最快（12.23s）
- 即使使用缺少手部关键点的OpenPose，IDOL仍能生成合理的手部

## 亮点与洞察

- **深度作为RGB热力图**：巧妙地将深度生成重新定义为风格化视频生成，避免修改预训练模型输出层
- **运动场一致性**：从U-Net中间特征的语义属性出发，提出在运动层面对齐而非像素层面，思路新颖
- **参数效率**：共享U-Net设计既省参数又提性能，是win-win的设计
- HAOP预训练策略简单有效，解决了姿势偏移时的背景遮罩问题

## 局限与展望

- 双模态高分辨率处理计算开销大，不适合实时应用
- 依赖高质量深度图训练数据，限制了场景泛化能力
- 可探索无监督深度或数据增强策略减轻数据质量约束
- 存在deepfake伦理风险，需要考虑不可见水印等措施

## 相关工作与启发

- 与LDM3D（修改autoencoder输出RGB+深度）和MM-Diffusion（耦合U-Net同时去噪视频和音频latent）相比，IDOL的参数共享+模态标签方案更加优雅高效
- 可以推广到其他多模态联合生成任务（如视频+法线图、视频+分割图）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 统一双模态U-Net和运动一致性损失是新颖且有效的设计
- **实验充分度**: ⭐⭐⭐⭐⭐ — 双数据集、多种深度类型、丰富的消融实验、计算复杂度分析
- **写作质量**: ⭐⭐⭐⭐ — 论文结构清晰，图示直观
- **实用价值**: ⭐⭐⭐⭐ — 在VR/AR、视频游戏等领域有应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](../../ICCV2025/3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[ECCV 2024\] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image](compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)

</div>

<!-- RELATED:END -->
