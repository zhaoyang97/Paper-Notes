---
title: >-
  [论文解读] High-Fidelity 3D Textured Shapes Generation by Sparse Encoding and Adversarial Decoding
description: >-
  [ECCV 2024][3D生成] 本文提出了一种基于稀疏编码模块和对抗解码模块的 3D 纹理形状生成框架，通过对 StableDiffusion 的最小适配扩展到 3D 领域，在 ShapeNet 和 G-Objaverse（200K 样本）上实现了开放词汇的高保真 3D 生成，超越了现有 SOTA 方法。
tags:
  - "ECCV 2024"
  - "3D生成"
  - "纹理形状生成"
  - "稀疏编码"
  - "对抗解码"
  - "开放词汇"
---

# High-Fidelity 3D Textured Shapes Generation by Sparse Encoding and Adversarial Decoding

**会议**: ECCV 2024  
**代码**: 无（数据集发布于 [https://aigc3d.github.io/gobjaverse/](https://aigc3d.github.io/gobjaverse/)）  
**领域**: 其他  
**关键词**: 3D生成、纹理形状生成、稀疏编码、对抗解码、开放词汇

## 一句话总结

本文提出了一种基于稀疏编码模块和对抗解码模块的 3D 纹理形状生成框架，通过对 StableDiffusion 的最小适配扩展到 3D 领域，在 ShapeNet 和 G-Objaverse（200K 样本）上实现了开放词汇的高保真 3D 生成，超越了现有 SOTA 方法。

## 研究背景与动机

**领域现状**：3D 内容生成是计算机视觉和图形学的重要任务，近年来随着 2D 扩散模型（如 StableDiffusion）的成功，研究者开始探索将其扩展到 3D 领域。主流方法大致分为两类：基于 SDS（Score Distillation Sampling）的优化方法（如 DreamFusion）和直接在 3D 数据上训练的生成模型。

**现有痛点**：存在两个关键问题。（1）3D 数据天然具有稀疏的空间结构——与 2D 图像的密集像素网格不同，3D 物体仅在空间中的有限区域占据位置，现有方法直接将密集编码架构套用到 3D 上会造成大量计算浪费。（2）3D 训练数据量远少于 2D——现有最大的 3D 数据集（Objaverse）也只有百万级别，远低于 LAION-5B 等 2D 数据集，这严重影响了模型的泛化能力。如果从头训练 3D 生成模型，数据不足会导致生成质量和多样性均受限。

**核心矛盾**：一方面需要利用大规模预训练 2D 模型的强大生成能力，另一方面 2D 模型的架构设计（密集卷积、2D 注意力）并不适合 3D 稀疏数据。如何在保留 2D 预训练权重的同时高效适配 3D 稀疏结构，是核心挑战。

**本文目标** （a）设计适合 3D 稀疏结构的编码方案，避免在空白区域浪费计算；（b）提升 3D 解码的形状恢复质量；（c）构建大规模 3D 基准，推动开放词汇 3D 生成。

**切入角度**：作者注意到 StableDiffusion 的核心 U-Net 架构可以通过少量修改适配 3D 任务——关键是在编码端引入稀疏操作来处理 3D 的空间稀疏性，在解码端引入对抗训练来提升形状质量。这种"最小改动"策略可以最大程度继承 2D 预训练的语义知识。

**核心 idea**：通过稀疏编码保持 3D 空间效率 + 对抗解码提升形状质量，以最小代价将 StableDiffusion 扩展为高保真 3D 纹理生成器。

## 方法详解

### 整体框架

框架基于 StableDiffusion 架构，输入可以是文本提示或条件图像，输出为带纹理的 3D 形状。整体 pipeline 分为三个阶段：首先通过稀疏编码模块将 3D 体素化表示中的有效区域编码为紧凑的潜在特征，然后使用经过适配的 U-Net 进行去噪扩散过程，最后通过对抗解码模块将潜在特征解码为完整的 3D 纹理形状。3D 表示采用体素化的三平面特征（triplane），将 3D 信息投影到三个正交平面上进行处理。

### 关键设计

1. **稀疏编码模块（Sparse Encoding Module）**:

    - 功能：将 3D 物体的体素化表示编码为紧凑的潜在特征，同时保留细节信息
    - 核心思路：观察到 3D 物体在体素网格中仅占据一小部分（通常 <20%），稀疏编码模块在编码时只处理被 3D 物体占据的体素位置。具体实现基于稀疏卷积（Sparse Convolution），使用 MinkowskiEngine 框架。输入的三平面特征首先通过掩码标记有效和无效区域，然后稀疏卷积仅在有效位置进行计算，减少了约 80% 的冗余计算。编码后的稀疏特征随后被展平为序列，送入后续的 U-Net 处理。关键在于稀疏-密集转换层（Sparse-to-Dense）的设计，确保 U-Net 各层都能正确处理变长序列
    - 设计动机：密集编码不仅计算浪费，更会引入大量"空气区域"的特征噪声，干扰模型学习物体表面的精确几何。稀疏编码让模型聚焦于物体本身，提升了细节保真度

2. **对抗解码模块（Adversarial Decoding Module）**:

    - 功能：将潜在特征解码为高质量的 3D 纹理形状，弥补扩散模型在几何细节上的不足
    - 核心思路：标准的解码器通常使用 L1/L2 重建损失训练，倾向于生成过度平滑的形状。对抗解码模块引入了一个 PatchGAN 风格的判别器 $D$，在多视角渲染的 2D 图像上进行对抗训练。判别器同时评估几何（深度图、法线图）和纹理（RGB 渲染），总损失为 $\mathcal{L}_{dec} = \mathcal{L}_{recon} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{feat} \mathcal{L}_{feat}$，其中 $\mathcal{L}_{feat}$ 是判别器中间层特征的匹配损失，有助于稳定训练。解码器从稀疏的潜在空间特征恢复密集的三平面表示，再通过小型 MLP 网络查询任意 3D 点的 SDF 值和颜色
    - 设计动机：扩散模型擅长生成整体结构和语义，但弱于精细的几何边缘和纹理细节。对抗训练引入的高频监督信号可以迫使解码器恢复更锐利的边缘和更丰富的表面纹理

3. **开放词汇训练策略**:

    - 功能：使模型能够生成训练集中未见过的类别的 3D 物体
    - 核心思路：打破传统 3D 生成中"特定类别"的概念，将 3D 生成视为开放词汇问题。在训练时使用 CLIP 图像编码器提取条件图像的语义特征，并通过 cross-attention 注入 U-Net。由于使用了经过清洗的 G-Objaverse 数据集（200K 高质量样本），模型接触到了足够多样的物体类别。测试时可以使用任意图像作为条件生成对应的 3D 物体，不再受限于固定的类别标签
    - 设计动机：传统 3D 生成模型在特定类别（如椅子、汽车）上训练和评估，不具备实际应用所需的通用性。开放词汇能力是 3D 生成走向实用的关键一步

### 损失函数 / 训练策略

训练分为两阶段：第一阶段训练 VAE（稀疏编码器 + 对抗解码器）使用重建损失、对抗损失和 KL 正则化；第二阶段在冻结 VAE 的基础上训练扩散 U-Net，使用标准的去噪损失 $\mathcal{L}_{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2]$。在 ShapeNet 上先验证网络设计，再在 G-Objaverse 上进行大规模训练。

## 实验关键数据

### 主实验

**ShapeNet 无条件生成（单类别）**:

| 方法 | 类别 | FID↓ | COV↑ | 1-NNA↓ |
|------|------|------|------|--------|
| GET3D | Chair | 46.8 | 47.2 | 65.3 |
| 3DShape2VecSet | Chair | 38.2 | 52.1 | 58.7 |
| Ours | Chair | **28.5** | **58.9** | **51.2** |
| GET3D | Car | 51.3 | 42.8 | 68.1 |
| Ours | Car | **32.1** | **55.4** | **53.8** |

**G-Objaverse 图像条件生成（开放词汇）**:

| 方法 | FID↓ | CLIP-Score↑ | 几何精度 (CD)↓ |
|------|------|-------------|---------------|
| Shap-E | 89.3 | 0.72 | 0.082 |
| One-2-3-45 | 73.5 | 0.76 | 0.068 |
| Ours | **52.1** | **0.81** | **0.047** |

### 消融实验

| 配置 | FID↓ | CD↓ | 说明 |
|------|------|-----|------|
| Full model | 28.5 | 0.047 | 完整模型 |
| w/o Sparse Encoding | 35.2 | 0.059 | 密集编码，细节丢失 |
| w/o Adversarial Decoding | 33.8 | 0.063 | 形状过度平滑 |
| w/o Feature Matching | 31.2 | 0.052 | 对抗训练不稳定 |
| Dense Encoding + L2 Dec | 41.7 | 0.071 | 基线配置 |

### 关键发现

- 稀疏编码和对抗解码各自贡献显著，FID 分别从 35.2 和 33.8 降至 28.5
- 联合使用的效果远优于单独使用（FID 41.7 → 28.5），说明两个模块存在协同效应
- 在 G-Objaverse 的大规模实验中，FID 比 Shap-E 降低 41%，CLIP-Score 提升 12.5%，验证了开放词汇能力
- 数据量从 ShapeNet 55K 扩展到 G-Objaverse 200K 后，生成多样性和质量均有明显提升

## 亮点与洞察

- **最小适配策略**是最巧妙的设计决策——不从头设计 3D 架构，而是对成熟的 2D 架构做最小改动，继承预训练权重的同时适配 3D 特性。这个策略可以推广到视频生成等其他模态
- **稀疏编码的引入**不仅是效率优化，更是质量优化——减少空白区域的特征污染对精细几何的恢复至关重要
- **G-Objaverse 数据集的清洗和基准构建**可能是本文隐含的重大贡献，200K 高质量 3D 数据对后续研究有长期价值

## 局限与展望

- 当前方法基于三平面表示，对于拓扑复杂的物体（如嵌套结构、高 genus 表面）表达能力受限
- 开放词汇能力仍然受限于 Objaverse 的类别分布，对于训练集中极少出现的类别（如特殊工具、乐器），生成质量可能下降
- 对抗解码引入的判别器增加了训练不稳定性的风险，超参数调节需要仔细
- 未探索文本到 3D 的生成路径，当前仅支持图像条件生成

## 相关工作与启发

- **vs GET3D (NVIDIA)**: GET3D 也做纹理 3D 生成，但使用 StyleGAN-based 架构，在单类别上有不错效果但难以扩展到开放词汇场景。本文通过继承 StableDiffusion 的语义理解能力实现了开放词汇
- **vs Shap-E (OpenAI)**: Shap-E 直接在 3D tokens 上做扩散，但缺乏对稀疏结构的特殊处理，导致在大规模数据上训练时细节保真度不足
- **vs One-2-3-45**: 基于 SDS 的优化方法，生成质量好但速度极慢（每个物体数分钟），本文的前馈生成在速度上有数量级优势

## 评分

- 新颖性: ⭐⭐⭐⭐ 稀疏编码+对抗解码的组合对 3D 生成很有针对性，但各单独技术并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 ShapeNet 单类、多类和 G-Objaverse 开放词汇三个层级，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题motivation清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 数据集+方法的组合贡献对 3D 生成领域有实质推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Differentiable Stroke Planning with Dual Parameterization for Efficient and High-Fidelity Painting Creation](../../CVPR2026/others/differentiable_stroke_planning_with_dual_parameterization_for_efficient_and_high.md)
- [\[CVPR 2026\] InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space](../../CVPR2026/others/instantretouch_efficient_and_high-fidelity_instruction-guided_image_retouching_w.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](../../CVPR2026/others/order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](../../ACL2025/others/coachme_sport_instruction.md)

</div>

<!-- RELATED:END -->
