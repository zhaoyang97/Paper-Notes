---
title: >-
  [论文解读] Exploring Simple Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2025][语义分割][开放词汇分割] 本文提出 S-Seg，一个极简的开放词汇语义分割模型，不依赖 CLIP 预训练、不需要标注掩码、不使用定制分组编码器，仅用伪掩码（DINO K-Means 聚类）和图像-文本对比损失训练 MaskFormer，在 Pascal VOC、Pascal Context 和 COCO 上取得了与复杂方法相当的性能，自训练后平均 mIoU 提升 5.5%。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "开放词汇分割"
  - "伪掩码"
  - "MaskFormer"
  - "图像-文本对"
  - "自训练"
---

# Exploring Simple Open-Vocabulary Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2401.12217](https://arxiv.org/abs/2401.12217)  
**代码**: [https://github.com/zlai0/S-Seg](https://github.com/zlai0/S-Seg)  
**领域**: 分割  
**关键词**: 开放词汇分割, 伪掩码, MaskFormer, 图像-文本对, 自训练

## 一句话总结

本文提出 S-Seg，一个极简的开放词汇语义分割模型，不依赖 CLIP 预训练、不需要标注掩码、不使用定制分组编码器，仅用伪掩码（DINO K-Means 聚类）和图像-文本对比损失训练 MaskFormer，在 Pascal VOC、Pascal Context 和 COCO 上取得了与复杂方法相当的性能，自训练后平均 mIoU 提升 5.5%。

## 研究背景与动机

**领域现状**：开放词汇语义分割（OVS）目标是从任意文本类别中为每个像素分配语义标签。当前主流方法通常依赖三种策略的组合：(1) 基于 CLIP 等图像级视觉-语言模型进行适配；(2) 在标注掩码上训练以学习像素级特征；(3) 设计专门的分组编码器（如 GroupViT）将像素聚合为语义区域。

**现有痛点**：这三种策略各有缺陷——CLIP 适配方法继承了图像级模型在像素级任务上的局限性（MaskCLIP 分割噪声大）；依赖标注掩码限制了可扩展性；定制分组编码器增加了设计复杂度。更重要的是，没有人验证过是否可以完全不依赖这些策略也能做好 OVS。

**核心矛盾**：开放词汇分割的核心需求是像素级的视觉-语言对齐，但获取大规模像素级标注不现实。现有方法要么从图像级模型"下采样"对齐信号（信号损耗），要么在有限的标注类别上训练（泛化受限），要么设计复杂的分组机制（简洁性差）。

**本文目标**：证明一个极简方案——用伪掩码提供形状监督 + 用图像-文本对提供语义监督，直接训练标准 MaskFormer——就能在不使用 CLIP、标注掩码或定制编码器的情况下取得competitive的分割性能。

**切入角度**：作者观察到 DINO 的自监督 ViT 特征已经包含了很好的物体分割先验（通过 K-Means 聚类即可得到高质量伪掩码），而网络文本提供了覆盖长尾概念的语义监督。将这两种免费可获取的监督信号结合，就可以解耦掩码监督和语义监督来训练分割模型。

**核心 idea**：将开放词汇分割的监督解耦为掩码监督（来自 DINO 伪掩码）和语义监督（来自图像-文本对比学习），用这两种弱监督信号直接训练 MaskFormer 实现像素级特征与语言的对齐。

## 方法详解

### 整体框架

S-Seg 在训练时接受图像-文本对，图像通过 MaskFormer 预测 $N$ 个掩码和对应的掩码特征；同时伪掩码生成器从图像生成 $K$ 个伪掩码用于掩码预测的监督；文本通过语言模型编码为文本特征，与掩码特征的平均值进行对比学习。推理时，给定候选类别名列表，语言模型编码各类别特征，与掩码特征计算余弦相似度来分类每个掩码，最终组合产生语义分割图。

### 关键设计

1. **基于 DINO 的伪掩码生成器**:

    - 功能：提供高质量的类别无关掩码监督，替代人工标注
    - 核心思路：使用 DINO 预训练的 ViT-S/8 提取图像的 patch token 特征，然后对这些特征应用 K-Means 聚类（$K=8$），将每个 token 分配到一个聚类中，reshape 回图像尺寸得到伪掩码。预测的 $N$ 个掩码与 $K$ 个伪掩码通过二分匹配（bipartite matching）对齐，$N-K$ 个未匹配的掩码不做惩罚
    - 设计动机：DINO 的自监督学习产生的特征在物体边界处有自然的不连续性，K-Means 聚类能很好地捕获这一点。实验显示其 oracle 性能（78.8% VOC mIoU）远超 GroupViT（73.7%），且速度极快（128 样本仅需 0.002s）。不使用 ImageNet 监督的聚类（68.8%）或原始像素聚类（49.5%），确保了完全自监督

2. **图像-文本对比学习的语义监督**:

    - 功能：学习掩码特征与语言的对齐，赋予模型开放词汇分类能力
    - 核心思路：$N$ 个掩码特征取平均作为图像全局表示，通过 2 层 MLP 投影到共享嵌入空间；文本通过 12 层 Transformer（从头训练）编码后取 [EOS] token 的嵌入，也投影到共享空间。使用标准的 CLIP 式双向对比损失 $\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}$，带可学习温度参数。跨 GPU 汇聚负样本以提高对比效率
    - 设计动机：图像-文本对比学习是经过 CLIP 等大量工作验证的有效范式。S-Seg 直接在 MaskFormer 的掩码特征上做对比学习，使像素级特征天然与语言对齐，而非像 CLIP 那样只对齐图像级特征后再适配

3. **自训练增强（S-Seg+）**:

    - 功能：利用测试域的未标注图像和类别信息进一步提升性能
    - 核心思路：用训练好的 S-Seg 对目标数据集的训练集生成伪标签，然后用这些伪标签训练一个 UperNet（MAE 预训练 ViT 骨干）进行全监督语义分割。这一步利用了测试时可获得的两个信息：目标域的未标注图像和候选类别列表
    - 设计动机：自训练利用了目标域的分布信息和类别先验，可以纠正 S-Seg 的一些预测错误。实验显示平均 mIoU 提升 5.5%（37.1% → 42.6%），且随训练数据规模增大持续改善

### 损失函数 / 训练策略

总损失为掩码损失和对比损失的加权和：$L = \lambda_{mask}\mathcal{L}_{mask} + \lambda_{contrastive}\mathcal{L}_{contrastive}$，其中 $\mathcal{L}_{mask} = \lambda_{dice}\mathcal{L}_{dice} + \lambda_{focal}\mathcal{L}_{focal}$。超参数为 $\lambda_{mask} = 1.0, \lambda_{contrastive} = 1.0, \lambda_{dice} = 1.0, \lambda_{focal} = 20.0$。使用 Swin-S 骨干、6 层 Transformer 解码器、$N=64$ 查询；AdamW 优化器，基础学习率 $5 \times 10^{-4}$，batch size 4096，训练 30 个 epoch。训练数据使用 CC3M + CC12M + RedCaps（最多 26M 图像-文本对）。

## 实验关键数据

### 主实验（含背景类评估）

| 方法 | 监督类型 | P. VOC | P. Context | COCO | 3-Avg |
|------|---------|--------|-----------|------|-------|
| CLIP | text | 13.5 | 8.1 | 5.9 | 9.2 |
| MaskCLIP | text | 26.8 | 22.8 | 12.8 | 20.8 |
| GroupViT | text | 50.8 | 23.7 | 27.5 | 34.0 |
| SegCLIP | text | 52.6 | 24.7 | 26.5 | 34.6 |
| TCL | text | 55.0 | 30.4 | — | — |
| **S-Seg** | text | 53.2 | 27.9 | 30.3 | **37.1** |
| **S-Seg+** | text | 62.0 | 30.2 | 35.7 | **42.6** |

### 消融实验——数据规模与自训练

| 数据 | S-Seg VOC | S-Seg Ctx | S-Seg COCO | S-Seg+ VOC | S-Seg+ COCO |
|------|----------|----------|-----------|-----------|------------|
| 12M | 44.9 | 22.9 | 22.5 | 53.1 | 26.2 |
| 15M | 45.1 | 23.8 | 27.9 | 54.2 | 28.0 |
| 26M | 53.2 | 27.9 | 30.3 | 62.0 | 35.7 |

### 关键发现

- **不依赖 CLIP 也能做好 OVS**：S-Seg 从头训练文本编码器和 MaskFormer，不使用任何预训练 VL 模型，依然取得了competitive的性能
- **简单基线不可行**：伪掩码+CLIP 分类（6.6% avg）和伪掩码 ViT（14.9% avg）都远不如 S-Seg（30.1% avg），说明多任务联合学习至关重要
- **数据可扩展性好**：从 12M 到 26M，S-Seg 在 VOC 上提升 +8.3%，说明方法能充分利用更多数据
- **自训练稳定有效**：在所有数据量和数据集设定下，自训练都带来了一致的提升（平均 +5.5%）
- 在去除背景类的评估下，S-Seg (81.8% VOC) 已超过早期需要标注掩码的方法 ZegFormer (80.7%)

## 亮点与洞察

- **"极简主义"的研究哲学很有价值**：在领域越来越复杂的趋势下，S-Seg 通过去除所有"必需"组件（CLIP、GT 掩码、定制编码器），逆向验证了什么是真正必要的。结论是：伪掩码 + 语言监督 + 标准架构就够了
- **监督信号解耦的思想巧妙**：将掩码预测的监督（来自伪掩码）和语义分类的监督（来自文本）完全解耦，使得两种监督可以独立扩展。这与传统的"掩码+类别标签"耦合监督形成鲜明对比
- **伪掩码生成器的质量出乎意料地好**：DINO K-Means 聚类在 VOC 上达到 78.8% oracle mIoU，甚至超过了经过 VL 训练的 GroupViT (73.7%)，这个发现很有启发性

## 局限与展望

- 在 COCO（81 类）上性能明显低于 VOC（21 类），说明类别数量增多对方法挑战较大
- 伪掩码生成器对小物体和纹理丰富的场景可能不够准确
- 自训练虽然有效但引入了额外的训练流程和对目标域的依赖
- 语言模型从头训练的规模有限（12 层），更大的预训练语言模型可能带来更好的语义理解
- 可以考虑结合扩散模型生成更多样的伪掩码，或使用 SAM 替代 DINO 聚类作为伪掩码源

## 相关工作与启发

- **vs GroupViT**：GroupViT 设计了专门的分组 token 机制来从文本监督中涌现分割能力，S-Seg 使用标准 MaskFormer 加伪掩码监督实现类似目标。S-Seg 的方法更简洁且略优
- **vs ZegFormer**：ZegFormer 使用 CLIP + GT 掩码训练，S-Seg 不用两者。有趣的是，S-Seg 在未见类上泛化更好，说明 CLIP 和 GT 掩码可能带来过拟合
- **vs TCL**：TCL 使用 CLIP 进行区域级 grounding + 对比学习，在 VOC 上略优于 S-Seg，但 TCL 依赖 CLIP 预训练
- **vs OpenSeg/OVSeg/SAN**：这些方法使用标注掩码，性能上限更高但需要大量标注成本

## 评分

- 新颖性: ⭐⭐⭐⭐ 思路新颖但各组件（伪掩码、对比学习、MaskFormer）都是现有技术
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种评估协议、与 16+ 方法对比、数据规模消融、简单基线验证
- 写作质量: ⭐⭐⭐⭐ 论文组织清晰，简单基线分析很有说服力
- 价值: ⭐⭐⭐⭐ 为开放词汇分割确立了一个strong yet simple的基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
