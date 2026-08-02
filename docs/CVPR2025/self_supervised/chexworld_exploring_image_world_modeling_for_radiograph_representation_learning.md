---
title: "CheXWorld: Image World Modeling for Radiograph Representation Learning"
description: "提出基于JEPA的胸片世界模型，通过局部解剖结构、全局布局和域变化三重任务学习，VinDr AUROC达95.24%，1-shot学习达64.60%"
tags:
  - CVPR2025
  - 自监督
  - 自监督学习
  - JEPA
  - 胸片分析
---

# CheXWorld: Image World Modeling for Radiograph Representation Learning

**会议**: CVPR 2025  
**机构**: 清华大学 / 解放军总医院  
**关键词**: 胸片、JEPA、世界模型、自监督学习、表征学习  

## 研究背景与动机

胸部X光（Chest X-ray）是全球最常见的医学影像检查，每年产生数十亿张图像。自监督学习在自然图像上取得了巨大成功（MAE、DINO、JEPA等），但直接迁移到胸片领域面临独特挑战：

**解剖结构的固定性**：与自然图像不同，胸片中的器官位置相对固定（心脏在左下、肺野在两侧），模型需要理解这种固定的空间布局。

**病灶的局部性**：大多数病变（结节、浸润等）仅占图像的很小区域，全局特征学习可能忽略这些关键局部。

**域差异的多样性**：不同设备、不同拍摄参数下的同一患者胸片可能差异极大，模型需要对这些域变化保持不变性。

现有方法通常只关注上述一个方面。例如，MAE关注局部重建但忽略全局布局；对比学习关注增广不变性但可能丢失局部细节。CheXWorld的核心动机是：**构建一个理解胸片"世界"的模型，同时建模局部结构、全局布局和域变化。**

JEPA（Joint-Embedding Predictive Architecture）提供了天然的框架——它在表征空间而非像素空间做预测，避免了像素重建的冗余性，同时支持灵活的预测任务设计。

## 方法详解

### 整体架构

CheXWorld基于I-JEPA框架，使用ViT作为编码器，设计了三个互补的预测任务来建模胸片世界的不同方面。

### 任务1：局部解剖结构建模（Local Anatomical Structure）

目标：理解胸片中的局部解剖结构和病灶模式。

方法：类似I-JEPA的掩码-预测范式：
- 从输入胸片中随机掩码若干patch块
- 用上下文编码器编码可见patch
- 用预测器从可见patch的表征预测被掩码patch的表征
- 损失函数：被掩码patch的预测表征与目标编码器输出的L2距离

$$\mathcal{L}_{\text{local}} = \frac{1}{|M|} \sum_{i \in M} \|f_{\text{pred}}(z_{\text{ctx}}) - \text{sg}(f_{\text{target}}(x_i))\|_2^2$$

其中 $M$ 是掩码位置集合，$\text{sg}$ 表示停止梯度。

### 任务2：全局布局建模（Global Layout）

目标：理解器官的空间关系和整体胸腔结构。

方法：跨裁剪预测（Cross-crop Prediction）+ 相对位置编码：
- 从同一胸片生成两个不同的裁剪视图（crop A, crop B）
- 用crop A的特征预测crop B中对应位置的表征
- **关键创新**：将两个crop的相对空间位置作为条件输入预测器

$$\mathcal{L}_{\text{global}} = \|f_{\text{pred}}(z_A, \Delta_{\text{pos}}) - \text{sg}(f_{\text{target}}(x_B))\|_2^2$$

其中 $\Delta_{\text{pos}}$ 编码了crop A和crop B的相对位置。这迫使模型理解"如果心脏在左下裁剪中可见，那么右侧裁剪中应该出现什么？"

### 任务3：域变化建模（Domain Variation）

目标：学习对不同成像条件的不变性和敏感性。

方法：增广条件预测（Augmentation-conditioned Prediction）：
- 对同一图像施加不同的数据增广（亮度、对比度、噪声等，模拟不同设备参数）
- 将增广参数作为条件，预测增广后图像的表征

$$\mathcal{L}_{\text{domain}} = \|f_{\text{pred}}(z_{\text{orig}}, c_{\text{aug}}) - \text{sg}(f_{\text{target}}(T(x)))\|_2^2$$

其中 $c_{\text{aug}}$ 编码了增广类型和强度，$T(x)$ 是增广后的图像。

### 统一训练

四个损失函数联合优化：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{local}} + \lambda_2 \mathcal{L}_{\text{global}} + \lambda_3 \mathcal{L}_{\text{domain}} + \lambda_4 \mathcal{L}_{\text{reg}}$$

其中 $\mathcal{L}_{\text{reg}}$ 是方差-不变性-协方差（VICReg）正则化项，防止表征坍缩。

## 实验结果

### 胸片分类

| 方法 | VinDr AUROC | ShenZhen Acc | CheXpert AUROC |
|------|------------|-------------|----------------|
| ImageNet预训练 | 88.45% | 93.12% | 86.72% |
| MAE (胸片微调) | 91.86% | 96.44% | 89.13% |
| DINO v2 | 93.15% | 97.53% | 90.45% |
| I-JEPA | 93.89% | 97.81% | 91.02% |
| **CheXWorld** | **95.24%** | **98.88%** | **92.56%** |

### 分割任务

| 方法 | SIIM-ACR Dice |
|------|-------------|
| U-Net (ImageNet) | 78.32% |
| TransUNet | 81.45% |
| MAE + U-Net | 82.19% |
| **CheXWorld + U-Net** | **84.58%** |

### Few-shot学习

| 方法 | 1-shot AUROC | 5-shot AUROC | 10-shot AUROC |
|------|-------------|-------------|--------------|
| DINO v2 | 58.34% | 72.15% | 79.82% |
| I-JEPA | 60.12% | 74.53% | 81.34% |
| **CheXWorld** | **64.60%** | **78.21%** | **84.15%** |

### 消融实验

| 配置 | VinDr 1% AUROC | VinDr 100% AUROC |
|------|---------------|-----------------|
| 仅Local | 84.71% | 93.89% |
| Local + Global | 87.43% | 94.56% |
| Local + Domain | 86.92% | 94.34% |
| **Local + Global + Domain（完整）** | **90.53%** | **95.24%** |

从84.71%到90.53%的提升（+5.82%）验证了三个任务的互补性，特别是在数据稀缺（1%）时影响更大。

## 创新点

1. **世界模型视角**：首次将胸片表征学习形式化为"世界模型"问题，统一建模结构、布局和域变化
2. **JEPA在医学影像的创新应用**：设计了三个医学影像特有的预测任务
3. **跨裁剪位置预测**：利用胸片解剖结构的空间固定性，通过相对位置条件实现全局理解

## 局限性

- 仅在胸片上验证，其他医学影像模态（CT、MRI）的适用性未知
- 三个任务的权重 $\lambda$ 需要手动调优
- 预训练数据规模的影响未充分探讨

## 总结

CheXWorld将JEPA框架创造性地应用于胸片表征学习，通过三重任务建模了胸片世界的不同方面。在分类、分割和少样本学习上全面超越了现有方法，特别是在低数据场景下表现突出。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)
- [\[CVPR 2025\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)
- [\[ICML 2025\] AdaWorld: Learning Adaptable World Models with Latent Actions](../../ICML2025/self_supervised/adaworld_learning_adaptable_world_models_with_latent_actions.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](../../CVPR2026/self_supervised/d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)

</div>

<!-- RELATED:END -->
