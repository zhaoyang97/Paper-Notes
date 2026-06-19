---
title: >-
  [论文解读] EAFormer: Scene Text Segmentation with Edge-Aware Transformers
description: >-
  [ECCV 2024][语义分割][场景文字分割] 提出边缘感知Transformer（EAFormer），通过文本边缘提取器过滤非文本区域边缘、对称交叉注意力在编码器中融合文本边缘信息，显著提升文字边缘区域的分割精度，并重标注COCO_TS和MLT_S数据集以实现更公平评估。 领域现状：场景文字分割旨在像素级区分前景文字与…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "场景文字分割"
  - "边缘感知"
  - "Transformer"
  - "对称交叉注意力"
  - "数据集重标注"
---

# EAFormer: Scene Text Segmentation with Edge-Aware Transformers

**会议**: ECCV 2024  
**arXiv**: [2407.17020](https://arxiv.org/abs/2407.17020)  
**代码**: [https://hyangyu.github.io/EAFormer/](https://hyangyu.github.io/EAFormer/)  
**领域**: 图像分割  
**关键词**: 场景文字分割, 边缘感知, Transformer, 对称交叉注意力, 数据集重标注

## 一句话总结
提出边缘感知Transformer（EAFormer），通过文本边缘提取器过滤非文本区域边缘、对称交叉注意力在编码器中融合文本边缘信息，显著提升文字边缘区域的分割精度，并重标注COCO_TS和MLT_S数据集以实现更公平评估。

## 研究背景与动机

**领域现状**：场景文字分割旨在像素级区分前景文字与背景，广泛应用于文本擦除、文档分析、场景理解等下游任务。近年来基于深度学习的方法不断涌现，如TexRNet引入字符级监督、TextFormer加入识别头，性能持续提升。

**现有痛点**：

**忽视文字边缘**：现有方法虽提升了整体分割精度，但忽略了文字边缘区域的准确性。边缘不精确的文字掩码在文本擦除等下游任务中会导致残留/误删像素

**边缘信息引入困难**：传统边缘检测（Canny）能精确捕捉边缘，但无法区分文字与非文字区域，直接使用全图边缘会引入大量干扰

**评估数据集标注粗糙**：COCO_TS和MLT_S等数据集的标注基于bounding box生成，标注质量差（缺失标注、边缘不精确），影响了方法评估的公平性

**核心矛盾**：如何有效利用传统边缘检测的精确边缘信息来增强文字分割，同时避免非文字区域边缘带来的负面干扰？

**切入角度**：设计一个"先过滤再融合"的两阶段策略——先用轻量文本检测器过滤非文字区域的边缘，再通过对称交叉注意力将过滤后的文字边缘融入编码器早期阶段。

**核心idea**：文字边缘的精准分割可以通过"文本边缘提取→边缘过滤→边缘引导编码"三步走实现，利用Canny的强边缘检测能力 + 轻量检测器的区域过滤 + 对称交叉注意力的特征融合。

## 方法详解

### 整体框架
EAFormer由三个模块组成：
1. **Text Edge Extractor**：提取并过滤文本边缘
2. **Edge-Guided Encoder**：基于SegFormer的4阶段编码器，在第一阶段融合边缘引导
3. **Text Segmentation Decoder**：基于MLP的解码器，融合多尺度特征预测文字掩码

输入场景图像 $\mathbf{X} \in \mathbb{R}^{3 \times H \times W}$，输出文字掩码 $\mathbf{M}_t$。

### 关键设计

1. **文本边缘提取器（Text Edge Extractor）**：

    - 用Canny算法提取全图边缘 $\mathbf{E}_w$（阈值100/200）
    - 用轻量ResNet backbone提取多尺度特征 $\{\mathbf{F}_1^d, ..., \mathbf{F}_4^d\}$，经1×1卷积预测文本区域掩码：
    $\mathbf{M}_a = \text{Conv}_{1\times 1}(\text{Concat}(\{\mathbf{F}_1^d, \mathbf{F}_2^d, \mathbf{F}_3^d, \mathbf{F}_4^d\}))$
    - 通过逐像素乘法过滤非文字边缘：$\mathbf{E}_t = \mathbf{M}_a \odot \text{SoftArgmax}(\mathbf{E}_w)$
    - 使用SoftArgmax使文本检测和分割分支可以联合端到端优化

2. **边缘引导编码器（Edge-Guided Encoder）**：

    - 基于SegFormer的4阶段层级Transformer编码器
    - 在**第一阶段**后引入**对称交叉注意力（Symmetric Cross-Attention）**：
        - 边缘作为Query、视觉特征作为Key/Value → 提取边缘感知视觉信息 $\mathbf{F}^{ev}$
        - 视觉特征作为Query、边缘作为Key/Value → 提取文字边缘信息 $\mathbf{F}^{te}$
    $\hat{\mathbf{F}}_1^s = \mathbf{F}^{ev} \oplus \mathbf{F}^{te} \oplus \mathbf{F}_1^s$
    - **仅在第一阶段融合边缘**的设计动机：K-Means聚类可视化显示只有第一阶段特征关注边缘信息，高层特征已不包含边缘细节。实验也验证在更高阶段引入边缘信息反而损害性能

3. **MLP文字分割解码器**：

    - 各阶段特征通过MLP统一通道维度并上采样到相同分辨率
    - 拼接后再经MLP融合，最终预测二分类文字掩码

### 损失函数 / 训练策略
仅使用两个交叉熵损失，避免复杂超参调节：
$$\mathcal{L} = \underbrace{\text{CE}(\mathbf{M}_t, \hat{\mathbf{M}}_t)}_{\mathcal{L}_{seg}} + \lambda \underbrace{\text{CE}(\mathbf{M}_a, \hat{\mathbf{M}}_a)}_{\mathcal{L}_{det}}$$
$\lambda = 1.0$；文本检测的bounding box级监督可从语义标注直接获取，不需要额外标注。AdamW优化器，lr=6×10⁻⁵，batch size 4，8卡RTX 4090。

## 实验关键数据

### 主实验

**英文文字分割基准**：

| 方法 | TextSeg fgIoU | TextSeg F-score | COCO_TS fgIoU | MLT_S fgIoU | BTS fgIoU |
|------|--------------|-----------------|---------------|-------------|-----------|
| SegFormer (Baseline) | 84.59 | 0.916 | 63.17 | 78.77 | 84.99 |
| TextFormer | 87.42 | 0.933 | 73.20 | 86.66 | 86.97 |
| TFT | 87.11 | 0.931 | 73.40 | 87.80 | 87.84 |
| **EAFormer (Ours)** | **88.06** | **0.939** | **81.03** | **89.02** | **88.08** |

在COCO_TS上相比TFT提升7.63% fgIoU；在TextSeg上超越TextFormer 0.64% fgIoU。

**重标注数据集结果**：

| 方法 | COCO_TS (重标注) fgIoU | COCO_TS F-score | MLT_S (重标注) fgIoU | MLT_S F-score |
|------|------------------------|-----------------|---------------------|---------------|
| TextFormer | 52.73 | 0.688 | 74.83 | 0.861 |
| **EAFormer** | **64.82** | **0.786** | **81.92** | **0.900** |

使用更精确标注训练和测试时，EAFormer优势更加明显（COCO_TS提升12.09%）。

### 消融实验

**边缘过滤与边缘引导**：

| 边缘过滤(EF) | 边缘引导(EG) | TextSeg fgIoU | BTS fgIoU |
|------|------|-------------|-----------|
| ✗ | ✗ | 84.59 | 84.99 |
| ✓ | ✗ | 86.85 | 87.35 |
| ✗ | ✓ | 81.03 | 80.35 |
| ✓ | ✓ | **88.06** | **88.08** |

**超参λ选择**：

| λ | TextSeg fgIoU | TextSeg F-score |
|---|--------------|-----------------|
| 0.1 | 84.03 | 0.910 |
| 0.5 | 87.33 | 0.926 |
| 1.0 | **88.06** | **0.939** |
| 5.0 | 87.67 | 0.934 |
| 10.0 | 87.94 | 0.937 |

### 关键发现
- **仅用边缘引导不过滤，反而损害性能**（TextSeg从84.59降到81.03），因为非文字区域边缘引入严重干扰
- 边缘应在**第一阶段**融合，在第三/四阶段融合时性能甚至低于baseline
- 使用预训练文本检测器（DBNet）替换轻量检测器时，TextSeg可达90.16%/95.2% fgIoU/F-score
- 参数量增加可控：从TextFormer的85M增至92M，推理时间从0.42s到0.47s/image

## 亮点与洞察
- **"先过滤再引导"策略**是文字分割场景下引入边缘信息的关键，直接用Canny边缘会适得其反
- **对称交叉注意力**双向信息交换比单向融合更有效，同时让视觉特征感知边缘、让边缘特征感知视觉上下文
- **重标注数据集**是重要贡献：COCO_TS和MLT_S的原始标注质量差，重标注后能更公平地评估方法
- 在低层特征中融合边缘的设计符合视觉感知层级：底层关注纹理/边缘，高层关注语义

## 局限与展望
- 引入轻量文本检测器增加了参数量（+7M）
- 只使用Canny边缘检测，使用深度学习边缘检测方法（如HED/BDCN）可能进一步提升
- 对模糊文字的边缘检测仍有困难，特别是低分辨率场景
- 可探索将边缘引导扩展到实例级文字分割

## 相关工作与启发
- SegFormer的层级Transformer设计 → 提供了引入边缘引导的自然接入点
- BCANet/BSNet等边缘引导分割方法需要边缘标注 → EAFormer用Canny免去标注需求
- DBNet的可微分二值化 → 启发了文本检测分支的设计
- 启发：对称交叉注意力的设计可推广到其他需要辅助信息引导的分割任务（如医学图像的轮廓引导）

## 评分
- 新颖性: ⭐⭐⭐⭐ 边缘过滤+对称交叉注意力的组合设计新颖，解决了实际存在的问题
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集全面测试，消融实验充分，重标注数据集增加了评估可信度
- 写作质量: ⭐⭐⭐⭐ 动机清晰，可视化分析（K-Means聚类、各阶段特征）有说服力
- 价值: ⭐⭐⭐⭐ 重标注数据集贡献有持久价值，方法在文字分割社区具有实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Beyond Detection: A Structure-Aware Framework for Scene Text Tracking](../../ICML2026/segmentation/beyond_detection_a_structure-aware_framework_for_scene_text_tracking.md)
- [\[ECCV 2024\] Occlusion-Aware Seamless Segmentation](occlusion-aware_seamless_segmentation.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)
- [\[ECCV 2024\] OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)
- [\[ICCV 2025\] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation](../../ICCV2025/segmentation/spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)

</div>

<!-- RELATED:END -->
