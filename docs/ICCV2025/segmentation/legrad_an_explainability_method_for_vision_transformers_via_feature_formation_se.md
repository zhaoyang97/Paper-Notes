---
title: >-
  [论文解读] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity
description: >-
  [ICCV 2025][语义分割][可解释性] 提出LeGrad——一种专为ViT设计的逐层可解释性方法，通过计算激活值对各层注意力图的梯度作为解释信号，并跨层聚合以生成高质量的空间显著性图，在分割、扰动和开放词汇场景均展现出优越的空间保真度。 Vision Transformer（ViT）已成为计算机视觉的标准架构…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "可解释性"
  - "Transformer"
  - "注意力梯度"
  - "开放词汇分割"
  - "CLIP"
---

# LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity

**会议**: ICCV 2025  
**arXiv**: [2404.03214](https://arxiv.org/abs/2404.03214)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: 可解释性, Vision Transformer, 注意力梯度, 开放词汇分割, CLIP

## 一句话总结

提出LeGrad——一种专为ViT设计的逐层可解释性方法，通过计算激活值对各层注意力图的梯度作为解释信号，并跨层聚合以生成高质量的空间显著性图，在分割、扰动和开放词汇场景均展现出优越的空间保真度。

## 研究背景与动机

Vision Transformer（ViT）已成为计算机视觉的标准架构，但由于自注意力机制的长距离依赖建模，其可解释性仍是挑战。现有可解释性方法存在以下问题：

**传统方法不适用**：GradCAM依赖卷积层、LRP需要特定层传播规则，无法直接用于ViT

**注意力方法的局限**：Raw Attention和Rollout忽略了非线性交互和正负贡献的区分，可能产生误导性解释

**CheferCAM的问题**：使用梯度加权注意力图，计算量大且对架构变化不灵活

**开放词汇场景不足**：现有方法在大规模开放词汇数据集（如OpenImagesV7，5827个类别）上表现严重下降

**大模型扩展性差**：多数方法在ViT-BigG等超大模型上难以有效计算

**核心洞察**：ViT中特征的形成是逐层迭代进行的，解释方法应该捕获每层对最终表征的贡献，而不仅仅使用最终输出。LeGrad将梯度本身（而非梯度加权的注意力）作为解释信号，实现层间可叠加。

## 方法详解

### 整体框架

LeGrad的核心思想非常简洁：对每层ViT的注意力图计算目标类别激活值的梯度，经ReLU裁剪后跨头和patch维度平均，最后跨层聚合。

### 关键设计

1. **单层解释图计算**：

    - 对于给定层 $l$，使用中间token表征 $Z^l$ 的均值 $\bar{z}^l$ 通过分类器/文本嵌入 $\mathcal{C}$ 计算激活值 $s^l = \bar{y}^l_{[\hat{c}]}$
    - 计算激活 $s^l$ 对该层注意力图 $\mathbf{A}^l \in \mathbb{R}^{h \times (n+1) \times (n+1)}$ 的梯度：$\nabla\mathbf{A}^l = \frac{\partial s^l}{\partial \mathbf{A}^l}$
    - 关键步骤：用ReLU裁剪负梯度 $(\nabla\mathbf{A}^l_{h,i,.})^+$，防止负梯度影响正激活
    - 跨头和patch维度取平均得到 $\hat{E}^l(s^l) = \frac{1}{h \cdot (n+1)}\sum_h\sum_i(\nabla\mathbf{A}^l_{h,i,.})^+$
    - 设计动机：梯度直接反映注意力图对预测的敏感度，比梯度加权注意力更直接；逐层梯度天然可加，无需额外归一化

2. **多层聚合**：

    - 逐层计算解释图后取平均：$\bar{\mathbf{E}} = \frac{1}{L}\sum_l \hat{E}^l(s^l)_{1:}$
    - 去掉CLS token列，reshape为2D，min-max归一化：$\mathbf{E} = \text{norm}(\text{reshape}(\bar{\mathbf{E}}))$
    - 设计动机：信息聚合在ViT中跨多层分布式进行，特别是大模型中更分散；仅用最终层会丢失中间层的丰富信息

3. **Attentional Pooler适配**（如SigLIP）：

    - 对使用注意力池化的ViT，在每层用Attentional Pooler处理中间表征 $Z^l$ 得到池化查询 $q^l$
    - 使用Pooler的注意力图 $\mathbf{A}_{pool} \in \mathbb{R}^{h \times 1 \times n}$ 替换自注意力图计算梯度
    - 设计动机：使LeGrad能够适配不同的特征聚合策略，不局限于CLS token架构

### 损失函数 / 训练策略

LeGrad是一个**无需训练**的后处理可解释性方法，不涉及损失函数或训练过程。仅需一次前向+反向传播即可生成解释图。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | LeGrad | CheferCAM | TextSpan | Rollout | 提升 |
|---|---|---|---|---|---|---|
| ImageNet-S 分割 (ViT-B/16) | mIoU↑ | **58.66** | 47.47 | 40.26 | 40.64 | +11.19 |
| ImageNet-S 分割 (ViT-B/16) | Pixel Acc↑ | **77.52** | 69.21 | 73.01 | 60.63 | +4.51 |
| OpenImagesV7 OV (ViT-B/16) | p-mIoU↑ | **48.38** | 5.87 | 9.44 | 8.75 | +38.94 |
| OpenImagesV7 OV (ViT-L/14) | p-mIoU↑ | **47.69** | 2.51 | 21.73 | 6.85 | +25.96 |
| OpenImagesV7 OV (ViT-H/14) | p-mIoU↑ | **46.51** | 9.49 | 23.74 | 5.82 | +22.77 |
| SigLIP-B/16 OV | p-mIoU↑ | **25.40** | 1.94 | - | 0.07 | +23.46 |
| ADE20K Sound Seg. | mIoU↑ | **38.9** | - | - | - | +14.7(vs DenseAV) |

### 消融实验

| 配置 | 说明 | 关键发现 |
|---|---|---|
| 层数实验 (ViT-B/16) | 使用不同数量的层 | 小模型用少数层即可；大模型需要更多层 |
| 层数实验 (ViT-L/14) | 使用更多层 | 信息聚合更分散，需更多层覆盖 |
| 层数实验 (ViT-H/14) | 层数增加 | 大模型layer贡献更均匀，凸显多层聚合的必要性 |
| 每层可视化分析 | 逐层热力图 | 定位信号分布在多层而非集中于单层 |

### 关键发现

- **开放词汇场景的碾压性优势**：在OpenImagesV7上LeGrad达到48.38 p-mIoU，第二好的TextSpan仅9.44，提升5倍以上
- **推理速度快**：96 FPS（ViT-B/16），接近GradCAM的108 FPS，远快于CheferCAM（21 FPS）和TextSpan（3.8 FPS）
- **大模型扩展好**：在ViT-BigG/14（25亿参数）上仍能有效工作
- **梯度分布分析**：不同预训练权重（Laion400M vs OpenAI vs MetaCLIP）的层重要性分布差异显著，可作为模型"指纹"

## 亮点与洞察

- **概念简洁**：核心思想极简——用梯度本身而非梯度加权作为解释信号，实现跨层可加
- **通用性极强**：适配CLS token聚合和Attentional Pooler两种架构，甚至可用于音频-视觉模型（ImageBind）
- **开放词汇爆发力**：在5827类的OpenImagesV7上5倍超越SOTA，说明LeGrad在细粒度识别中的优势
- **ReLU裁剪的重要性**：简单的负梯度裁剪有效去除了ViT中常见的均匀噪声激活
- **模型"指纹"发现**：层梯度分布可作为不同预训练策略的诊断工具

## 局限与展望

- 对于分辨率极高的图像，patch粒度限制了空间精度（受限于ViT的patch size）
- 未在分割任务中结合后处理（如CRF）进一步精化边界
- 多层聚合使用简单平均，未探索自适应加权策略
- 未在视频理解或3D场景中验证
- ReLU裁剪过于激进，可能丢失某些有意义的负梯度信息

## 相关工作与启发

- GradCAM是卷积网络时代的可解释性标志方法，LeGrad可视为其在ViT时代的对应
- CheferCAM用梯度加权注意力再矩阵乘传播，计算量大；LeGrad改为梯度直接求和，O(L) vs O(L)矩阵乘
- TextSpan在不使用梯度的情况下也能工作，但速度慢（3.8 FPS）且在大模型上不够稳定
- 可启发将LeGrad应用于CLIP的zero-shot分割任务中，直接生成像素级解释

## 评分

- **新颖性**: ⭐⭐⭐⭐ 思路简洁但有效，用梯度本身代替梯度加权注意力
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖分割/扰动/OV/音频/速度/大模型，非常全面
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，实验组织有条理
- **价值**: ⭐⭐⭐⭐ 在开放词汇可解释性上贡献突出，实际应用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[CVPR 2025\] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers](../../CVPR2025/segmentation/da-vpt_semantic-guided_visual_prompt_tuning_for_vision_transformers.md)
- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](../../ICLR2026/segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[ICLR 2026\] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](../../ICLR2026/segmentation/thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)

</div>

<!-- RELATED:END -->
