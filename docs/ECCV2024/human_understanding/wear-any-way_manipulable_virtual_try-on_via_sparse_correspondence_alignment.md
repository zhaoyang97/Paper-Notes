---
title: >-
  [论文解读] Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment
description: >-
  [ECCV 2024][人体理解][虚拟试穿] 提出 Wear-Any-Way 框架，基于双 U-Net 扩散模型构建强基线实现高保真虚拟试穿，并通过稀疏对应对齐（Sparse Correspondence Alignment）引入点控制机制，支持用户通过点击和拖拽精确操控穿着方式（如卷袖子、开合外套、塞衣角等），在标准试穿和可操控试穿两个维度均达到 SOTA。
tags:
  - "ECCV 2024"
  - "人体理解"
  - "虚拟试穿"
  - "扩散模型"
  - "点控制"
  - "稀疏对应对齐"
  - "可定制生成"
---

# Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment

**会议**: ECCV 2024  
**arXiv**: [2403.12965](https://arxiv.org/abs/2403.12965)  
**代码**: 未公开（项目页面: mengtingchen.github.io/wear-any-way-page）  
**领域**: 人体理解  
**关键词**: 虚拟试穿, 扩散模型, 点控制, 稀疏对应对齐, 可定制生成

## 一句话总结

提出 Wear-Any-Way 框架，基于双 U-Net 扩散模型构建强基线实现高保真虚拟试穿，并通过稀疏对应对齐（Sparse Correspondence Alignment）引入点控制机制，支持用户通过点击和拖拽精确操控穿着方式（如卷袖子、开合外套、塞衣角等），在标准试穿和可操控试穿两个维度均达到 SOTA。

## 研究背景与动机

虚拟试穿（Virtual Try-on）旨在合成特定人物穿着指定服装的图像，是时尚电商领域的关键技术。现有方法存在两个主要缺陷：

**生成质量不足**：大部分方法只能处理简单场景（单件服装、简单纹理），对复杂纹理/图案的细节保持力差；不支持真实场景中的模特到模特试穿、多件服装试穿等需求

**穿着方式不可控**：现有方法无法控制穿着风格。但在时尚领域，袖子是否卷起、外套是否敞开、上衣是否塞进裤子等穿着方式的变化对展示效果至关重要

本文的核心创新：**在构建高质量虚拟试穿基线的基础上，首次引入基于控制点的穿着风格操控机制，使用户能通过简单的点击和拖拽交互定制穿着方式**。

## 方法详解

### 整体框架

Wear-Any-Way 采用双 U-Net 架构：主 U-Net（inpainting Stable Diffusion 初始化）负责生成试穿结果，参考 U-Net（标准 SD 初始化）提取服装的细粒度特征。整体流程分为两个阶段：(1) 构建强基线实现标准虚拟试穿；(2) 通过稀疏对应对齐模块添加点控制能力。

### 关键设计

1. **强基线：双 U-Net 虚拟试穿 Pipeline**：实现高保真标准试穿的基础架构

    - **主 U-Net**：以 Stable Diffusion inpainting 模型初始化，输入 9 通道张量（4 通道隐空间噪声 + 4 通道去衣服区域的人物图像 latent + 1 通道二值 mask）
    - **参考 U-Net**：标准 SD 模型，以服装图像为输入提取多层特征。通过在 self-attention 层中拼接参考特征的 Key/Value 实现特征注入：
    $\text{Attention} = \text{softmax}\left(\frac{Q_m \cdot \text{cat}(K_m, K_r)^T}{\sqrt{d_k}}\right) \cdot \text{cat}(V_m, V_r)$
    - **CLIP 图像编码器**：替换文本嵌入，提供服装的全局颜色和纹理引导
    - **姿态控制**：用 DW-Pose 提取人物姿态，通过小型 CNN 编码后直接加到主 U-Net 的隐空间噪声上
    - 设计动机：Reference U-Net 比 CLIP/DINOv2/ControlNet 等特征提取器能更好地保留服装细节（logo、图案、文字等），实验验证了这一点

2. **稀疏对应对齐（Sparse Correspondence Alignment）**：核心创新，实现点控制

    - **点嵌入（Point Embedding）**：用 disk map $D_{g/p}^{1 \times H \times W}$ 表示控制点，背景为 0，点位置填入 1~K 的随机数值（K=24 为最大控制点数）。服装图像和人物图像上对应点填入相同数值，通过随机分配解耦语义与点的关联，使点表示具有置换不变性。用卷积网络将 disk map 投射为高维嵌入 $E_{g/p}^{C \times H \times W}$
    - **嵌入注入**：将点嵌入添加到 attention 层的 Query 和 Key 上：
    $\text{Attention} = \text{softmax}\left(\frac{(Q_m + E_p) \cdot \text{cat}(K_m + E_p, K_r + E_g)^T}{\sqrt{d_k}}\right) \cdot \text{cat}(V_m, V_r)$
    - 设计动机：通过在 attention 的 Q/K 上添加相同数值编码的点嵌入，使得服装图像上标记位置的特征在生成时自然对齐到人物图像上目标位置，从而实现精确的空间控制

3. **训练点对收集**：解决缺少服装-人物密集对应标注的问题

    - 利用预训练的 Siamese Stable Diffusion 提取人物和服装图像的特征
    - 取最后一层特征图，在多个时间步上集成预测结果以获得鲁棒匹配
    - 对人物图像服装区域随机采样内部和边界点作为查询，通过最大余弦相似度在服装图像上找到对应点
    - 匹配方向从人物到服装（因为复杂姿态下服装上某些点在穿着后可能不可见）
    - 实验对比了 SuperGlue、CLIP、DINOv2、Reference U-Net 和 SD 特征，SD 特征匹配效果最佳

### 损失函数 / 训练策略

- **基础损失**：标准扩散模型 MSE 噪声预测损失
- **点加权损失（Point-weighted Loss）**：在采样点周围增加损失权重，强化点控制的监督信号
- **条件丢弃（Condition Dropping）**：增大丢弃姿态图的概率，并将 inpainting mask 退化为 bounding box mask，迫使模型学习从控制点获取穿着风格信息
- **零初始化（Zero-initialization）**：在点嵌入网络输出处添加零初始化卷积层，参考 ControlNet 的做法实现渐进式集成，提升训练稳定性

训练配置：
- 8×A100 GPU，batch size 64，学习率 5e-5
- 主 U-Net 训练 decoder 和 encoder 的 self-attention；参考 U-Net 全参数训练
- 训练分辨率 768×576（自有数据）/ 512×384（公开数据集对比）
- 0.3M 高质量试穿三元组数据（人物+上衣+下装）

## 实验关键数据

### 主实验

**VITON-HD 和 DressCode 定量对比**

| 方法 | VITON-HD FID↓ | VITON-HD KID↓ | D.C. Upper FID↓ | D.C. Upper KID↓ |
|------|--------------|--------------|-----------------|-----------------|
| VITON-HD | 12.117 | 3.23 | - | - |
| HR-VITON | 11.265 | 2.73 | 13.820 | 2.71 |
| DCI-VTON | 8.754 | 1.10 | 11.920 | 1.89 |
| StableVITON | 8.698 | 0.88 | 11.266 | 0.72 |
| **Wear-Any-Way** | **8.155** | **0.78** | 11.72 | **0.33** |

在 FID 和 KID（生成质量核心指标）上取得最佳或次佳结果。

### 消融实验

**点嵌入注入方式和增强策略消融（Landmark Distance↓）**

| 配置 | Dist_upper | Dist_down | Dist_coat |
|------|-----------|-----------|-----------|
| 无点控制 | 35.65 | 21.13 | 43.34 |
| Latent Noise 注入 | 27.32 | 16.34 | 30.38 |
| Attention Q,K 注入 | 24.35 | 15.79 | 27.27 |
| + Zero-init | 22.65 | 15.33 | 25.56 |
| + Condition-dropping | 18.39 | 12.04 | 20.44 |
| + Point-weighted loss | **17.65** | **10.32** | **20.32** |

**训练点对收集方法对比（Landmark Distance↓）**

| 匹配方法 | Dist_upper | Dist_down | Dist_coat |
|----------|-----------|-----------|-----------|
| SuperGlue | 134.04 | 128.34 | 187.30 |
| CLIP | 93.42 | 89.23 | 129.24 |
| DINOv2 | 83.24 | 70.08 | 103.54 |
| Reference U-Net | 59.34 | 35.23 | 79.98 |
| **Stable Diffusion** | **43.44** | **29.94** | **59.45** |

### 关键发现

- Reference U-Net 是保持服装细节的关键，CLIP/DINOv2/ControlNet 都无法保留 logo、文字等精细图案
- Attention Q/K 注入比 Latent Noise 注入效果更好，因为可以在特征聚合阶段直接建立对应关系
- 三个增强策略（zero-init、condition-dropping、point-weighted loss）逐步叠加各有贡献，condition-dropping 的提升最大
- 预训练 SD 特征在非刚体（服装可变形）匹配上远优于传统匹配方法（如 SuperGlue），验证了扩散模型特征的语义对应能力

## 亮点与洞察

1. **开创性的交互范式**：首次在虚拟试穿中引入点击/拖拽操控穿着风格的概念，从"被动生成"转向"主动定制"
2. **点嵌入的置换不变性设计**：通过随机分配数值实现控制点的 permutable 性质，使模型支持任意数量、任意位置的控制点
3. **统一框架**：单一模型一次推理即可完成标准试穿、可操控试穿、多件服装试穿、模特到模特试穿等多种任务
4. **巧妙利用扩散模型做非刚体匹配**：发现预训练 SD 天然具有语义对应能力，用于收集训练数据的点对

## 局限与展望

1. 作者指出在手部等小区域仍可能产生伪影，可通过更高分辨率或 SDXL 等大模型改善
2. 点对收集依赖预训练 SD 特征的匹配质量，对于极端形变可能失效
3. 当前仅支持静态图像，扩展到视频试穿是有价值的方向
4. K=24 的最大控制点数是否足够覆盖所有精细控制需求值得进一步探索
5. 定量评估可操控性的指标（landmark distance）依赖 FashionAI 检测器的准确性

## 相关工作与启发

- **DragGAN/DragDiffusion**：基于拖拽的图像编辑方法，但在服装场景下精度不足且易破坏人体结构
- **TryOnDiffusion**：也使用双 U-Net，但需要大量多姿态数据且不支持穿着风格控制
- **StableVITON**：使用 zero cross-attention 条件化空间编码器，但细节保持不如 Reference U-Net
- **启发**：扩散模型的语义对应能力可作为通用工具用于非刚体物体间的稠密匹配

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次提出可操控虚拟试穿的概念和完整解决方案，稀疏对应对齐设计优雅
- **实验充分度**: ⭐⭐⭐⭐ 标准和可控两个维度均有评估，消融完整，但缺少用户研究
- **写作质量**: ⭐⭐⭐⭐ 图示丰富直观，方法描述清晰，应用场景展示充分
- **价值**: ⭐⭐⭐⭐⭐ 对时尚行业有直接应用价值，开辟了虚拟试穿的新交互范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction](../../CVPR2025/human_understanding/vton_360_high-fidelity_virtual_try-on_from_any_viewing_direction.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](../../CVPR2026/human_understanding/mobile_vton_ondevice_virtual_tryon.md)
- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](../../CVPR2026/human_understanding/refton_reference_person_shot_assist_virtual_try-on.md)
- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [\[CVPR 2026\] MOFA-VTON: More Fashion Possibilities with Fine-Grained Adaptations in Virtual Try-On](../../CVPR2026/human_understanding/mofa-vton_more_fashion_possibilities_with_fine-grained_adaptations_in_virtual_tr.md)

</div>

<!-- RELATED:END -->
