---
title: >-
  [论文解读] Using Diffusion Priors for Video Amodal Segmentation
description: >-
  [CVPR 2025][语义分割][非模态分割] 本文将视频非模态分割（video amodal segmentation）重新建模为条件生成任务，利用预训练视频扩散模型（Stable Video Diffusion）的形状先验，以模态掩码和伪深度图为条件，在遮挡区域实现高达 13% mIoU 提升的补全效果，并首次实现视频级非模态内容补全。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "非模态分割"
  - "视频扩散模型"
  - "遮挡补全"
  - "时序一致性"
  - "深度条件"
---

# Using Diffusion Priors for Video Amodal Segmentation

**会议**: CVPR 2025  
**arXiv**: [2412.04623](https://arxiv.org/abs/2412.04623)  
**代码**: 无  
**领域**: 分割  
**关键词**: 非模态分割, 视频扩散模型, 遮挡补全, 时序一致性, 深度条件

## 一句话总结

本文将视频非模态分割（video amodal segmentation）重新建模为条件生成任务，利用预训练视频扩散模型（Stable Video Diffusion）的形状先验，以模态掩码和伪深度图为条件，在遮挡区域实现高达 13% mIoU 提升的补全效果，并首次实现视频级非模态内容补全。

## 研究背景与动机

**领域现状**：当前主流分割方法（如 SAM 系列）只处理物体的可见部分（模态分割），不考虑被遮挡的完整形状。非模态分割（amodal segmentation）旨在预测物体的完整轮廓，包括被遮挡部分，这在机器人操作、自动驾驶和视频编辑中至关重要。

**现有痛点**：(1) 单帧非模态方法无法处理严重遮挡或完全遮挡的情况——当物体被大面积遮挡时，单帧信息不足以推断其完整形状；(2) 现有视频非模态方法（如 SaVos、EoRaS）局限于刚性物体，且依赖额外输入（相机姿态、光流），泛化能力差；(3) 缺乏真实世界的非模态标注数据，限制了方法的训练和评估。

**核心矛盾**：非模态感知本质上是一个病态问题（ill-posed），因为遮挡区域有多种合理的补全方式。单帧方法缺乏足够信息，而多帧方法又受限于数据和表示能力。

**本文目标**：(1) 利用时序信息推断严重/完全遮挡下的物体完整形状；(2) 不依赖相机姿态或光流等额外输入；(3) 同时实现非模态掩码预测和 RGB 内容补全。

**切入角度**：大规模预训练的视频扩散模型内化了丰富的物体形状先验——这些模型在生成像素的同时也"学会了"物体边界应该如何延伸。利用此先验进行条件生成，可以自然地处理遮挡补全。

**核心 idea**：将 Stable Video Diffusion 改造为条件生成模型，以模态掩码序列和伪深度图为条件，生成非模态掩码序列（第一阶段），再以模态 RGB 内容和非模态掩码为条件补全遮挡区域的外观（第二阶段）。

## 方法详解

### 整体框架

方法分两阶段：第一阶段以模态掩码序列 + 伪深度图为输入，通过改造的 SVD 模型生成非模态掩码序列；第二阶段以第一阶段预测的非模态掩码和物体的模态 RGB 内容为条件，通过另一个 SVD 模型补全遮挡区域的 RGB 内容。两阶段共用相同的 3D UNet 架构但使用不同条件。

### 关键设计

1. **模态到非模态的条件扩散（Stage 1: Modal-to-Amodal）**:

    - 功能：从模态掩码序列预测完整的非模态掩码序列
    - 核心思路：将 SVD 的输入条件从 RGB 图像替换为二值模态掩码序列。由于 VAE 需要 3 通道输入，将单通道掩码复制3次后编码为 latent。编码后的 latent 与噪声 latent 拼接作为 3D UNet 的输入。同时使用模态掩码的 CLIP 嵌入通过交叉注意力注入，提供关于物体在各帧可见性的时序信息。与原始 SVD 复制单帧 T 次不同，这里输入 T 帧独立的模态掩码
    - 设计动机：利用 SVD 在 1.52 亿样本上预训练获得的强大形状先验，通过条件生成的方式将这些先验迁移到非模态分割任务

2. **伪深度条件（Pseudo-Depth Conditioning）**:

    - 功能：提供关于场景中遮挡关系的隐式线索
    - 核心思路：使用 Depth Anything V2 将 RGB 帧转换为伪深度图，作为额外通道拼接到 3D UNet 的输入中。这使输入 latent 形状变为 $\mathcal{R}^{T \times 3C_1 \times \frac{H}{F} \times \frac{W}{F}}$。训练采用两阶段微调策略：先只用掩码条件训练模型，再在此基础上初始化掩码+深度条件的模型。新增深度通道采用零卷积初始化（zero convolution），保留已学到的掩码条件能力
    - 设计动机：遮挡通常由距离相机更近的物体造成，深度图能直接揭示遮挡者-被遮挡者关系。实验证明深度比 RGB 帧更有效：RGB 帧中的纹理和外观依赖反而会损害泛化能力

3. **非模态内容补全（Stage 2: Amodal Content Completion）**:

    - 功能：补全遮挡区域中物体的 RGB 外观
    - 核心思路：使用相同 3D UNet 架构但不同条件的第二个 SVD 模型。条件包括：物体的模态 RGB content（可见区域的外观）和第一阶段预测的非模态掩码。由于合成数据集中也缺乏遮挡区域的真实 RGB 标注，采用自监督训练对构造：选择高可见度（>95%）的物体，随机叠加其他物体的非模态掩码来模拟遮挡
    - 设计动机：这是首个尝试视频级非模态内容补全的工作。通过自监督方式解决了训练数据缺失的问题

### 损失函数 / 训练策略

采用 EDM 框架训练，使用加权 L2 去噪目标。两阶段微调策略（先掩码后加深度）配合零卷积初始化，确保训练稳定。在 SAIL-VOS 合成数据上训练（128x256 分辨率，batch size 8），使用 8x RTX 3090 约 30 小时。推理时用更高分辨率（256x512）以获得更精确的像素级预测。

## 实验关键数据

### 主实验

| 方法 | SAIL-VOS mIoU | SAIL-VOS mIoU_occ | TAO-Amodal AP50 | 类型 |
|------|-------------|------------------|----------------|------|
| PCNet-M | 74.20 | 42.52 | 85.11 | 单帧 |
| AISFormer | 73.51 | 39.16 | 81.93 | 单帧 |
| pix2gestalt (Top-1) | 54.83 | 26.59 | 57.50 | 单帧扩散 |
| EoRaS | 81.76 (MOVi-B) | 49.39 | - | 多帧 |
| 3D-UNet baseline | 72.79 | 39.54 | 83.83 | 多帧 |
| **Ours (Top-1)** | **77.07** | **55.12** | **89.25** | 多帧扩散 |
| **Ours (Top-3)** | **79.23** | **59.69** | **92.46** | 多帧扩散 |

### 消融实验

| 条件配置 | SAIL-VOS mIoU | mIoU_occ | TAO AP50 | 说明 |
|---------|-------------|----------|---------|------|
| 仅掩码 | 75.17 | 51.28 | 85.03 | 基础条件 |
| 掩码 + RGB | 76.59 | 53.30 | 86.59 | RGB有帮助但有限 |
| 掩码 + 深度 | 77.07 | 55.12 | 89.25 | 深度效果更好 |
| 掩码 + RGB + 深度 | 77.19 | 54.59 | 87.16 | 加RGB反而有损泛化 |

### 关键发现

- 在遮挡区域指标 mIoU_occ 上提升最为显著（比第二名 PCNet-M 提升近 13%），说明方法核心优势在于处理严重遮挡
- 仅在合成数据 SAIL-VOS 上训练，在真实数据 TAO-Amodal 上零样本评估仍大幅领先，体现了 SVD 先验的强大泛化能力
- 伪深度条件比 RGB 条件更有效且泛化性更好——RGB 包含的纹理信息在跨数据集时反而成为噪声
- 用户研究中，85.6% 的用户偏好本方法的内容补全结果（相对 pix2gestalt）

## 亮点与洞察

- **将分割任务转化为条件视频生成**是一个非常巧妙的视角转换。这不仅自然地引入了时序一致性，还利用了大规模预训练模型中的丰富先验。这种"借生成模型做判别任务"的思路可以迁移到许多其他视频理解任务
- **伪深度比 RGB 更好地编码遮挡信息**这个发现很有洞察力。深度图天然表达了前后关系，而 RGB 中的纹理细节在跨域时反而有害，这为条件选择提供了指导原则
- **首个视频级非模态内容补全**是重要贡献。自监督训练对构造方法简单有效，可以扩展到更多缺乏标注的视频理解任务

## 局限与展望

- 主要在合成数据上训练和评估，真实世界的非模态标注仍然缺乏，限制了更全面的评估
- 推理效率未详细讨论——使用 25 步 EDM 去噪对于实时应用可能仍然较慢
- 对于画面外遮挡（out-of-frame occlusion）的处理能力有限
- 内容补全阶段的质量评估还停留在用户研究层面，缺乏定量指标

## 相关工作与启发

- **vs PCNet-M**: PCNet-M 是经典的模态到非模态的单帧预测方法，在简单遮挡场景下效果不错，但面对严重遮挡时信息不足，本文通过多帧+扩散先验大幅超越
- **vs pix2gestalt**: 同为扩散模型方法但仅处理单帧，无法保证时序一致性，且在高遮挡场景下表现很差
- **vs EoRaS**: EoRaS 依赖光流等额外输入且限于刚性物体，本文不需要额外输入且能处理非刚性物体

## 评分

- 新颖性: ⭐⭐⭐⭐ 将视频扩散模型创造性地用于非模态分割，视角转换巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、多种baseline、详细消融、用户研究
- 写作质量: ⭐⭐⭐⭐ 清晰系统，动机链条完整
- 价值: ⭐⭐⭐⭐ 在遮挡理解这个重要问题上取得显著进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning and Aligning Click-Aware Shape Prior for Interactive Amodal Instance Segmentation](../../CVPR2026/segmentation/learning_and_aligning_click-aware_shape_prior_for_interactive_amodal_instance_se.md)
- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](../../CVPR2026/segmentation/from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)
- [\[CVPR 2025\] LiVOS: Light Video Object Segmentation with Gated Linear Matching](livos_light_video_object_segmentation_with_gated_linear_matching.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
