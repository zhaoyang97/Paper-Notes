---
title: >-
  [论文解读] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding
description: >-
  [CVPR 2025][多模态VLM][多模态] 提出ASAP框架，通过大模型辅助对齐(LMA)、篡改引导交叉注意力(MGCA)和补丁篡改建模(PMM)三个核心模块，系统性地推进图文语义对齐以提升多模态篡改检测与定位性能——在DGM4基准上AUC达94.38%，文本定位F1达76.52%，显著超越现有方法。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "多模态"
  - "Semantic Alignment"
  - "注意力机制"
  - "DGM4"
  - "Hard Negative Mining"
---

# ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding

**会议**: CVPR 2025  
**arXiv**: [2412.12718](https://arxiv.org/abs/2412.12718)  
**代码**: [https://github.com/CriliasMiller/ASAP](https://github.com/CriliasMiller/ASAP)  
**领域**: 机器人  
**关键词**: Multi-Modal Manipulation Detection, Semantic Alignment, Cross Attention, DGM4, Hard Negative Mining  
**作者**: Zhenxing Zhang, Yaxiong Wang 等 (合肥工业大学)

## 一句话总结
提出ASAP框架，通过大模型辅助对齐(LMA)、篡改引导交叉注意力(MGCA)和补丁篡改建模(PMM)三个核心模块，系统性地推进图文语义对齐以提升多模态篡改检测与定位性能——在DGM4基准上AUC达94.38%，文本定位F1达76.52%，显著超越现有方法。

## 背景与动机
随着AIGC技术（如扩散模型、LLM）的快速发展，高质量的图文篡改内容越来越容易生成，对社交媒体信息可信度构成严重威胁。多模态篡改检测任务要求同时检测图像和文本中的篡改区域，不仅需要判断整体是否被篡改（分类），还需精确定位篡改的图像区域和文本片段（定位）。

现有方法（如HAMMER等）虽然取得了一定进展，但仍存在两个核心瓶颈：
1. **图文语义对齐不足**：篡改检测的关键在于发现图文之间的语义不一致，但现有方法的对齐学习不够充分，难以捕捉细粒度的语义差异
2. **区域级定位精度有限**：图像篡改定位通常需要patch级别的精确判断，现有方法缺乏有效的区域级监督和难负例挖掘机制

## 核心问题
如何系统性地增强图文语义对齐，使模型能更准确地检测和定位多模态内容中的篡改区域？

## 方法详解

### 整体框架
ASAP构建在CLIP双编码器架构之上，包含三个核心模块，总损失函数为：

35714L = L_{DGM} + L_{LMA} + \alpha \cdot L_{MGCA} + \lambda \cdot L_{PMM}35714

其中 $\alpha=0.1$，$\lambda=0.01$。{DGM}$ 是基础的DGM4多任务损失。

### 模块一：大模型辅助对齐 (LMA)
LMA模块利用预训练大模型生成丰富的文本描述，增强视觉-语言对齐学习：

1. **图像描述生成**：使用多模态大语言模型（MLLM，如InstructBLIP）为每张图像生成详细的视觉描述（caption）
2. **篡改解释生成**：使用大语言模型（LLM，如ChatGLM）基于原始文本和篡改文本对生成篡改解释（explanation），描述文本在何处发生了何种篡改
3. **对比学习对齐**：构建视觉-描述对和文本-解释对进行对比学习，拉近匹配对、推远不匹配对，损失为：

35714L_{LMA} = L_{cap} + L_{exp}35714

| 文本类型 | 生成模型 | 用途 | 示例 |
|---------|---------|-----|------|
| Caption（图像描述） | InstructBLIP (MLLM) | 图像到文本的语义桥接 | 一位男子站在红色汽车旁 |
| Explanation（篡改解释） | ChatGLM (LLM) | 描述篡改具体变化 | 将红色汽车替换为蓝色卡车 |
| 原始文本 | 数据集提供 | 基准文本 | 一位男子站在红色汽车旁 |
| 篡改文本 | 数据集提供 | 检测目标 | 一位男子站在蓝色卡车旁 |

### 模块二：篡改引导交叉注意力 (MGCA)
MGCA模块通过显式的篡改区域引导，增强跨模态注意力对篡改区域的关注：

1. **引导掩码生成**：根据图像篡改标注生成二值引导掩码 $，标记哪些patch被篡改
2. **掩码增强注意力**：在标准交叉注意力基础上，通过引导掩码调制注意力权重，使模型更多关注篡改相关的区域：

35714Attn_{MGCA} = \text{softmax}(\frac{QK^T}{\sqrt{d}} + \beta \cdot M_g)35714

3. **辅助损失**：额外的交叉注意力对齐损失 {MGCA}$ 鼓励注意力权重集中在实际篡改区域

### 模块三：补丁篡改建模 (PMM)
PMM模块通过难负例补丁选择策略提升区域级篡改定位精度：

1. **Hard Negative Patch Selection (HNP)**：在每个batch中，选择与篡改patch视觉特征最相似但未被篡改的patch作为难负样例
2. **对比学习**：在patch级别构建对比学习目标，拉近同一篡改区域的patch表示、推远与难负例的表示
3. **区域定位增强**：通过HNP策略，模型学习区分视觉相似但语义不同的patch，从而提升定位精度

35714L_{PMM} = -\log \frac{\exp(sim(z_i^+, z_i) / \tau)}{\exp(sim(z_i^+, z_i) / \tau) + \sum_j \exp(sim(z_j^-, z_i) / \tau)}35714

## 实验结果

### 主实验
在DGM4基准数据集上与主要方法的全面对比：

| 方法 | AUC (%) | ACC (%) | mAP (%) | 文本F1 (%) | 图像IoU (%) |
|-----|---------|---------|---------|-----------|------------|
| HAMMER | 93.09 | 86.42 | 87.20 | 72.22 | 76.10 |
| DGM4-baseline | 91.56 | 84.90 | 85.03 | 70.15 | 74.82 |
| MFCLIP | 92.47 | 85.88 | 86.51 | 71.68 | 75.63 |
| **ASAP (本文)** | **94.38** | **87.71** | **88.53** | **76.52** | **77.35** |
| ASAP vs HAMMER | +1.29 | +1.29 | +1.33 | +4.30 | +1.25 |

核心发现：
- 在所有五个指标上统一超越现有最优方法
- 文本定位F1提升最显著(+4.30%)，说明语义对齐对文本篡改定位尤为关键
- AUC和ACC的同步提升证明整体检测能力的增强

### 不同篡改类型的性能

| 篡改类型 | AUC (%) | 文本F1 (%) | 图像IoU (%) |
|---------|---------|-----------|------------|
| 仅文本篡改 | 95.12 | 79.83 | - |
| 仅图像篡改 | 93.67 | - | 78.91 |
| 图文联合篡改 | 94.45 | 73.72 | 75.23 |

图文联合篡改最具挑战性，但ASAP仍保持较高性能。

## 消融实验

### 各模块贡献
逐步添加各模块的性能变化：

| 设置 | AUC (%) | ACC (%) | 文本F1 (%) | 图像IoU (%) |
|------|---------|---------|-----------|------------|
| Baseline (DGM4) | 93.16 | 86.01 | 72.05 | 75.88 |
| + LMA | 94.28 | 87.30 | 75.41 | 76.72 |
| + LMA + MGCA | 94.40 | 87.55 | 76.18 | 77.10 |
| + LMA + MGCA + PMM (完整ASAP) | 94.38 | 87.71 | 76.52 | 77.35 |

关键观察：
- LMA贡献最大（AUC +1.12%），证明大模型辅助的语义对齐学习是核心
- MGCA进一步提升定位精度（文本F1 +0.77%）
- PMM主要提升图像定位（IoU +0.25%）和文本定位（F1 +0.34%），AUC略有波动（-0.02%）表明PMM更侧重定位而非分类

### LMA中不同文本类型的影响

| 设置 | AUC (%) | 文本F1 (%) | 图像IoU (%) |
|------|---------|-----------|------------|
| 无LMA | 93.16 | 72.05 | 75.88 |
| 仅Caption | 93.85 | 74.12 | 76.30 |
| 仅Explanation | 93.72 | 73.88 | 76.15 |
| Caption + Explanation | 94.28 | 75.41 | 76.72 |

Caption和Explanation各自贡献互补信息，联合使用效果最佳。Caption提供视觉语义描述，Explanation提供篡改操作的显式描述。

### HNP策略的影响
- 有HNP：图像IoU 77.35%
- 无HNP：图像IoU 76.92%（-0.43%）
- HNP通过挖掘视觉相似但语义不同的难负例，有效提升了边界区域的定位精度

## 总结与评价

### 优点
1. **系统性设计**：三个模块从不同角度增强语义对齐，互为补充
2. **大模型赋能**：巧妙利用MLLM和LLM生成辅助文本，无需额外人工标注
3. **全面领先**：在检测和定位两个层面的所有指标上均取得最优

### 局限性
1. **推理效率**：LMA模块的大模型生成在训练阶段可离线完成，但增加了数据预处理成本
2. **引导掩码依赖**：MGCA需要训练时的篡改区域标注，限制了对无标注场景的适用性
3. **数据集单一**：实验仅在DGM4基准上验证，对其他多模态篡改检测数据集的泛化性有待验证

### 启发
- 利用大模型生成辅助信号进行对齐学习，是一种低成本但高效的增强策略
- 难负例挖掘在区域级任务中的重要性被再次验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ASAP: Advancing Semantic Alignment for Multi-Modal Manipulation Detection](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_detecting_an.md)
- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)
- [\[CVPR 2025\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)
- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[CVPR 2025\] GeoMM: On Geodesic Perspective for Multi-Modal Learning](geomm_on_geodesic_perspective_for_multi-modal_learning.md)

</div>

<!-- RELATED:END -->
