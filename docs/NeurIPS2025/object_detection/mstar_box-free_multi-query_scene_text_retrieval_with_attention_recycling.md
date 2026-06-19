---
title: >-
  [论文解读] MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling
description: >-
  [NeurIPS 2025][目标检测][场景文本检索] 提出 MSTAR，首个无需边界框标注的多查询场景文本检索方法，通过渐进式视觉嵌入（PVE）逐步将注意力从显著区域转移到不显著区域，结合风格感知指令和多实例匹配模块，实现了对单词、短语、组合和语义四种查询类型的统一检索，并构建了首个多查询文本检索基准 MQTR。
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "场景文本检索"
  - "无框注释"
  - "多查询检索"
  - "注意力回收"
  - "视觉语言模型"
---

# MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling

**会议**: NeurIPS 2025  
**arXiv**: [2506.10609](https://arxiv.org/abs/2506.10609)  
**代码**: [GitHub](https://github.com/yingift/MSTAR)  
**领域**: 目标检测 / 场景文本检索  
**关键词**: 场景文本检索, 无框注释, 多查询检索, 注意力回收, 视觉语言模型

## 一句话总结

提出 MSTAR，首个无需边界框标注的多查询场景文本检索方法，通过渐进式视觉嵌入（PVE）逐步将注意力从显著区域转移到不显著区域，结合风格感知指令和多实例匹配模块，实现了对单词、短语、组合和语义四种查询类型的统一检索，并构建了首个多查询文本检索基准 MQTR。

## 研究背景与动机

**领域现状**：场景文本检索（Scene Text Retrieval）旨在根据查询从图像集中搜索包含相关文本的图像，在签名检索、关键帧提取等领域有广泛应用。近年来在准确文本定位的辅助下取得了显著进展。

**现有痛点**：(1) 现有方法通常需要昂贵的边界框标注用于训练（word级、text-line级等多种标注），成本高昂；(2) 大多数方法采用定制化检索策略，难以统一处理多种查询类型（单词、短语、组合、语义查询）。

**核心矛盾**：视觉语言模型（VLM）在大规模无框预训练中表现强大，但倾向于关注显著视觉概念而忽略细粒度的场景文本实例（如图像中的小字）。

**本文目标**：实现无边界框监督的场景文本检索，同时统一多种查询类型。

**切入角度**：利用 VLM 的注意力机制，通过注意力回收（Attention Recycling）逐步从高注意力区域转移到被忽视的区域。

**核心 idea**：通过渐进式遮蔽高注意力区域迫使模型关注不显著文本，并通过风格指令统一多种查询类型。

## 方法详解

### 整体框架

MSTAR 基于 BLIP-2 构建，由四个核心组件组成：视觉编码器 $\phi$（SigLIP ViT-Base-512）、渐进式视觉嵌入（PVE）、多模态编码器 $\psi$（BLIP-2）和多实例匹配模块（MIM）。训练时使用对比学习和图文匹配损失联合优化；推理时先通过余弦相似度初步排序，再对 top K 图像进行重排序。

### 关键设计

1. **Progressive Vision Embedding (PVE)**:

    - **功能**：渐进式提取视觉嵌入，将注意力从显著区域转移到不显著的细节文本区域
    - **为什么**：VLM 倾向于关注显著视觉元素（如红色圆圈），容易忽略细小场景文本，导致小文本检索漏检率高
    - **怎么做**：
        - 视觉编码器提取初始图像特征 $f_0$，通过多模态编码器生成初始视觉嵌入 $E_V^0$
        - Salient Attention Shift（SAS）模块从交叉注意力权重中计算注意力图 $C_{t-1}$，二值化后取反得到掩码 $M_{t-1} = 1 - \sigma(C_{t-1})$
        - 掩码注意力层迫使自注意力降低已高关注区域的权重，聚焦被忽视区域
        - 迭代 T 步后拼接所有嵌入 $E_V \in \mathbb{R}^{(T+1)Q \times d}$
    - **区别**：不同于需要 GT 监督的掩码方法，SAS 的掩码完全从模型自身的交叉注意力中导出

2. **Style-Aware Instruction**:

    - **功能**：通过短文本指令引导多模态编码器区分不同查询风格
    - **为什么**：统一训练多种查询类型（单词/短语/组合/语义）时，格式和语义差异会造成表征不一致
    - **怎么做**：$E_T = \psi(\text{Concat}[T_i, T_Q])$，其中 $T_i$ 为风格指令，$T_Q$ 为文本查询。为加速训练，同一图像的所有查询一起编码
    - **区别**：无需为每种查询类型设计独立模型或分支

3. **Multi-Instance Matching (MIM)**:

    - **功能**：显式建立视觉嵌入和文本嵌入之间的一对一匹配关系
    - **为什么**：传统的嵌入聚合或晚期交互策略需要大量训练才能实现视觉-语言对齐
    - **怎么做**：两个并行分支：
        - 单词分支：使用匈牙利匹配算法在 $E_w$ 和 $E_V$ 之间建立一对一对应
        - 多词分支：通过轻量交叉注意力层在文本约束下聚合视觉特征 $E_{vt} = \mathcal{F}(\mathcal{C}(\mathcal{Q}=E_w, \mathcal{K,V}=E_V))$

### 损失函数 / 训练策略

- 对比损失 $\mathcal{L}_c = \alpha \mathcal{L}_{t2v} + \mathcal{L}_{v2t}$，$\alpha=1.5$
- 图文匹配损失 $\mathcal{L}_m$（交叉熵）
- 总损失 $\mathcal{L} = \mathcal{L}_c + \mathcal{L}_m$
- 多阶段训练：分辨率从 $512 \to 640 \to 800$ 渐进增大
- 重排序Top 2%图像

## 实验关键数据

### 主实验

**MQTR 多查询检索 MAP%：**

| 方法 | 类型 | AVG | Word | Phrase | Combined | Semantic |
|------|------|-----|------|--------|----------|----------|
| TDSL | Box-Based | 58.25 | 69.11 | 40.83 | 72.71 | 50.36 |
| TG-Bridge | Box-Based | 54.09 | 69.89 | 30.21 | 75.53 | 40.73 |
| BLIP-2 (FT) | Box-Free | 58.11 | 58.09 | 42.23 | 60.84 | 71.24 |
| **MSTAR** | **Box-Free** | **66.78** | **73.27** | **44.22** | **74.48** | **75.14** |

MSTAR 平均 MAP 超越先前最优方法 8.53%。

**6个公开数据集词级检索 MAP%：**

| 方法 | SVT | STR | CTR | Total-Text | CTW | IC15 | Avg |
|------|-----|-----|-----|------------|-----|------|-----|
| TDSL | 89.38 | 77.09 | 66.45 | 74.75 | 59.34 | 77.67 | 74.16 |
| FDP-RN50×16 | 89.63 | **89.46** | - | 79.18 | - | - | - |
| MSTAR | **91.31** | 86.25 | 60.13 | **85.55** | **90.87** | **81.21** | **82.56** |
| MSTAR (+re-rank) | 91.11 | 86.14 | **65.25** | **86.96** | **92.95** | **82.69** | **84.18** |

MSTAR 在不使用边界框标注的情况下，平均 MAP 超越 TDSL 8.40%，在 Total-Text 上超越 FDP 6.37%。

### 消融实验

| Ins | MIM | PVE | CTR | SVT | STR | Total-Text | CTW | IC15 | MQTR |
|-----|-----|-----|-----|-----|-----|------------|-----|------|------|
| ✗ | ✗ | ✗ | 52.87 | 90.07 | 81.57 | 82.32 | 87.28 | 76.71 | 65.79 |
| ✓ | ✗ | ✗ | 54.65 | 90.70 | 82.81 | 83.19 | 88.96 | 77.15 | 66.15 |
| ✓ | ✓ | ✗ | 55.77 | 91.02 | 85.00 | 84.01 | 90.31 | 79.23 | 65.69 |
| ✓ | ✓ | ✓ | **60.13** | **91.31** | **86.25** | **85.55** | **90.87** | **81.21** | **66.78** |

- PVE 对小文本场景效果最显著：CTR 提升 4.36%，IC15 提升 1.98%
- MIM 显著提升字级检索：STR +2.19%，CTW +1.35%

### 关键发现

- 无框方法 MSTAR 可以与全监督文本定位方法（TG-Bridge）达到竞争性性能，且推理速度快一倍（14.2 FPS vs 6.7 FPS）
- PVE 的注意力回收机制有效解决了 VLM 忽略小文本的问题
- 在短语检索数据集上达到 95.71% MAP

## 亮点与洞察

- **范式创新**：首个无框场景文本检索方法，证明了无需昂贵的边界框标注也能达到甚至超越有框方法
- **统一多查询**：通过风格指令实现四种查询类型的统一处理，避免为每种类型训练独立模型
- **注意力回收的优雅设计**：利用模型自身的注意力图来引导注意力转移，无需额外监督
- **MQTR 基准贡献**：首个多查询场景文本检索基准，填补了评测空白
- 实际推理速度具有优势：无需文本检测模块

## 局限与展望

- 在 CTR 数据集（极小文本）上仍不及基于框的方法 TDSL，这是无框方法的固有限制
- 匈牙利匹配在处理复杂组合查询时在 MQTR 上产生了轻微的性能下降
- 对中文场景文本的支持和评估不够充分
- PVE 的迭代步数 T 是超参数，不同场景可能需要不同设置

## 相关工作与启发

- **BLIP-2 (li2023blip)**：基础架构，提供预训练的多模态编码器
- **SigLIP (zhai2023sigmoid)**：视觉编码器初始化
- **FDP (zeng2024focus)**：先前 SOTA，利用 CLIP 加框监督进行文本区域定位
- **TDSL (wang2021scene)**：经典端到端场景文本检索方法
- 启发：VLM 的注意力偏向性是一个系统性问题，注意力回收是一种通用的解决思路，可推广到其他细粒度检索任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个无框多查询文本检索，PVE注意力回收设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 7个公开数据集+自建MQTR基准，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述有说服力
- 价值: ⭐⭐⭐⭐ 显著降低标注成本，MQTR基准对社区有贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](../../CVPR2025/object_detection/search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)
- [\[NeurIPS 2025\] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[AAAI 2026\] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](../../AAAI2026/object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)
- [\[CVPR 2026\] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding](../../CVPR2026/object_detection/mrd_multi-resolution_retrieval-detection_fusion_for_high-resolution_image_unders.md)

</div>

<!-- RELATED:END -->
