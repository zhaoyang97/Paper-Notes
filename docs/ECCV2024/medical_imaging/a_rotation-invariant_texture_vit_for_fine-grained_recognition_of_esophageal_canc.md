---
title: >-
  [论文解读] A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images
description: >-
  [ECCV 2024][医学图像][内镜超声] 本文提出 SRRM-ViT，通过在 ViT 中引入统计旋转不变性增强机制(SRRM)，自适应选择关键区域并融合直方图统计特征，实现了对食管癌内镜超声图像中任意径向位置病灶的无偏细粒度分类，在临床和公开数据集上取得了显著性能提升。 领域现状：内镜超声(EUS)是诊断食管黏膜下肿瘤…
tags:
  - "ECCV 2024"
  - "医学图像"
  - "内镜超声"
  - "旋转不变性"
  - "纹理特征"
  - "Transformer"
  - "食管癌分类"
---

# A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images

**会议**: ECCV 2024  
**代码**: [https://github.com/tianyiliu-lab/SRRM-ViT](https://github.com/tianyiliu-lab/SRRM-ViT)  
**领域**: 医学图像  
**关键词**: 内镜超声, 旋转不变性, 纹理特征, Vision Transformer, 食管癌分类

## 一句话总结

本文提出 SRRM-ViT，通过在 ViT 中引入统计旋转不变性增强机制(SRRM)，自适应选择关键区域并融合直方图统计特征，实现了对食管癌内镜超声图像中任意径向位置病灶的无偏细粒度分类，在临床和公开数据集上取得了显著性能提升。

## 研究背景与动机

**领域现状**：内镜超声(EUS)是诊断食管黏膜下肿瘤的重要工具，能够感知食管壁的层次结构变化。目前临床诊断主要依赖医生经验进行人工判读，深度学习方法已开始应用于 EUS 图像分析，其中 Vision Transformer(ViT) 因其全局建模能力成为当前 SOTA 模型架构。

**现有痛点**：EUS 图像分析面临两个核心挑战：(1) 病灶常常破坏食管层的结构完整性和细粒度纹理信息，使得常规特征提取难以准确捕获病变区域的关键信息；(2) 由 EUS 成像的环形扫描特性，病灶可以出现在图像中任意径向位置，同一病变在不同角度下呈现出截然不同的外观，这极大增加了分类难度。

**核心矛盾**：标准 ViT 的自注意力机制虽然能建模全局依赖关系，但对旋转变换缺乏不变性，且无法有效过滤 EUS 图像中大量的无关背景区域（如正常组织、超声伪影等），导致细粒度纹理特征被稀释。

**本文目标** (1) 如何从 EUS 图像中自适应地选择与病灶相关的关键区域，排除无关信息干扰？(2) 如何使模型对病灶的径向位置变化具有不变性，实现任意旋转角度下的一致性分类？

**切入角度**：作者观察到直方图统计特征天然具有旋转不变性——无论图像如何旋转，其像素值的统计分布保持不变。因此，将直方图统计特征与 ViT 的自注意力机制结合，可以同时利用 Transformer 的全局建模能力和统计特征的旋转不变性。

**核心 idea**：通过在 ViT 的自注意力机制中融入具有旋转不变性的直方图统计特征，并配合自适应关键区域选择，实现对 EUS 图像中任意位置病灶的无偏细粒度识别。

## 方法详解

### 整体框架

SRRM-ViT 的整体流程如下：输入 EUS 图像首先经过关键区域自适应选择模块(Adaptive Region Selection, ARS)，筛选出与病灶最相关的图像区域，过滤掉无关的背景和伪影。然后，选中的区域被送入增强型 ViT backbone，其中核心创新是统计旋转不变性增强机制(Statistical Rotation-invariant Reinforcement Mechanism, SRRM)。SRRM 在每个 Transformer block 的自注意力计算中，额外引入基于直方图的统计特征，这些特征对旋转变换天然不变，从而使整个网络对病灶的径向位置变化具有鲁棒性。最终，增强后的特征经过分类头输出食管癌亚型的精细分类结果。

### 关键设计

1. **自适应关键区域选择模块(ARS)**:

    - 功能：从 EUS 图像中自动识别并选择与病灶诊断最相关的区域，排除无关信息的干扰
    - 核心思路：利用注意力评分机制对图像的不同区域进行重要性评估，选择得分最高的若干区域作为后续分析的输入。具体而言，先对输入图像进行初步特征提取，计算每个 patch 的重要性权重，然后按照权重排序选择 top-K 个关键区域。这种方式避免了对整张图像的无差别处理
    - 设计动机：EUS 图像中包含大量非病灶区域（健康组织、超声探头伪影、边界噪声等），如果直接将整张图像送入分类器，这些无关信息会严重干扰细粒度纹理特征的提取，降低分类精度

2. **统计旋转不变性增强机制(SRRM)**:

    - 功能：在 ViT 的自注意力机制中融入旋转不变的统计纹理特征，使模型能够无偏地识别任意径向位置的病灶
    - 核心思路：对每个 patch 计算其局部直方图统计特征（如像素值分布的均值、方差、偏度等统计量），这些统计量天然具有旋转不变性。然后将统计特征编码为向量，作为额外的 key/value 注入到标准 self-attention 的计算中。具体地，修改后的注意力计算变为 $\text{Attn}(Q, K+K_s, V+V_s)$，其中 $K_s$ 和 $V_s$ 分别是从统计特征导出的辅助 key 和 value。这样，模型在计算 patch 间关系时，不仅考虑空间位置特征，还考虑旋转不变的统计纹理特征
    - 设计动机：标准 ViT 的 patch embedding 和位置编码对旋转敏感——同一病灶旋转后会产生不同的特征表示。而直方图统计特征不受空间变换影响，将其注入注意力机制可以为模型提供一个稳定的"锚点"，确保不同旋转角度下的同类病灶获得一致的表征

3. **细粒度纹理特征增强**:

    - 功能：捕获 EUS 图像中食管壁各层的微细纹理差异，支持病灶亚型的精细区分
    - 核心思路：在直方图统计特征的基础上，进一步提取多尺度纹理描述子。通过对不同空间分辨率的 patch 计算统计特征，构建多层级的纹理表示。这些特征在 Transformer 的多头注意力中分别处理，最终融合为综合的细粒度表征
    - 设计动机：食管癌的不同亚型（如平滑肌瘤、间质瘤、囊肿等）在 EUS 图像中主要通过层间纹理差异来区分，这些差异往往非常微妙，需要细粒度的纹理分析才能捕获

### 损失函数 / 训练策略

模型采用标准的交叉熵损失进行端到端训练，同时在关键区域选择模块上使用辅助的区域相关性损失，鼓励模型关注病灶区域而非背景。训练策略方面采用了数据增强（包括随机旋转来增加训练数据的多样性）和分阶段训练——先预训练 ViT backbone，再联合训练 SRRM 模块。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(SRRM-ViT) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 临床内部数据集 | Accuracy | 显著提升 | ViT baseline | +明显提升 |
| 公开EUS数据集 | Accuracy | SOTA | 多种对比方法 | 一致优于对比方法 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| ViT baseline | 基线准确率 | 不含任何增强模块 |
| + ARS | 准确率提升 | 加入关键区域选择后过滤了无关信息 |
| + SRRM | 准确率进一步提升 | 加入旋转不变性增强后，不同角度病灶识别更一致 |
| + ARS + SRRM (Full) | 最高准确率 | 两个模块协同作用效果最佳 |

### 关键发现

- SRRM 模块对旋转变换的鲁棒性显著优于标准 ViT，在人工旋转测试中准确率波动极小
- 自适应区域选择有效聚焦于病灶区域，可视化结果显示其选择的区域与临床标注高度一致
- 直方图统计特征与自注意力的融合方式优于简单的特征拼接或相加
- 在多种食管癌亚型的细粒度分类任务中，模型均表现出一致的性能优势

## 亮点与洞察

- 巧妙利用直方图统计特征的旋转不变性，这是一个优雅且有物理直觉的设计——统计量不随空间变换而改变
- 将旋转不变性直接注入 Transformer 的注意力计算中，比后处理或数据增强更高效
- 自适应区域选择 + 旋转不变特征增强形成了一套完整的 EUS 图像分析方案
- 方法具有较好的领域迁移潜力，可扩展到其他环形扫描医学成像模态

## 局限与展望

- 代码虽然开源但标注为"coming soon"，实际可用性待验证
- 统计特征的直方图 bin 数和统计量选择对性能的影响需要进一步系统研究
- 当前方法假设旋转是主要的变换模式，但 EUS 图像还可能存在其他形变（如压缩、拉伸），未来可以扩展到更一般的变换不变性
- 实验主要在食管癌 EUS 数据上验证，泛化到其他超声成像场景的效果未知
- 病例规模相对有限，大规模临床验证仍然需要

## 相关工作与启发

- Vision Transformer(ViT) 在医学影像中的应用已成为趋势，但旋转不变性是一个被忽视的关键问题
- 传统纹理分析方法（如 LBP、GLCM）天然具有某些不变性，本文将这种思想融入深度学习框架是一个有意思的融合
- 本文的区域选择策略与 attention-based MIL（多实例学习）有异曲同工之处

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将直方图统计旋转不变性融入ViT注意力机制是一个巧妙的创新
- 实验充分度: ⭐⭐⭐⭐ 在临床和公开数据上验证，但公开数据集规模和对比方法可更多
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述合理
- 价值: ⭐⭐⭐⭐ 对EUS图像分析有实际临床价值，方法思路可迁移到其他领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](../../AAAI2026/medical_imaging/denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)
- [\[CVPR 2025\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](../../CVPR2025/medical_imaging/prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [\[CVPR 2025\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](../../CVPR2025/medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[ECCV 2024\] Topology-Preserving Downsampling of Binary Images](topology-preserving_downsampling_of_binary_images.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)

</div>

<!-- RELATED:END -->
