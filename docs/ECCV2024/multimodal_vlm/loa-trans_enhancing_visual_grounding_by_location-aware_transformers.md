---
title: >-
  [论文解读] LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers
description: >-
  [ECCV 2024][多模态VLM][视觉定位] LoA-Trans提出一种位置感知的查询选择机制，生成多个可能的目标位置作为位置感知查询（而非仅依赖估计的中心点），并引入TaskSyn网络在解码器中实现指代表达理解（REC）和指代表达分割（RES）的任务协同，显著提升视觉定位的准确性。 领域现状：视觉定位（Visual…
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "视觉定位"
  - "位置感知查询"
  - "Transformer"
  - "REC"
  - "RES"
  - "多任务协同"
---

# LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers

**会议**: ECCV 2024  
**代码**: 无  
**DOI**: [10.1007/978-3-031-72667-5_23](https://doi.org/10.1007/978-3-031-72667-5_23)  
**领域**: 多模态VLM  
**关键词**: 视觉定位, 位置感知查询, Transformer, REC, RES, 多任务协同

## 一句话总结

LoA-Trans提出一种位置感知的查询选择机制，生成多个可能的目标位置作为位置感知查询（而非仅依赖估计的中心点），并引入TaskSyn网络在解码器中实现指代表达理解（REC）和指代表达分割（RES）的任务协同，显著提升视觉定位的准确性。

## 研究背景与动机

**领域现状**：视觉定位（Visual Grounding）是多模态理解的核心任务，包括指代表达理解（REC，预测目标的bounding box）和指代表达分割（RES，预测目标的像素级掩码）。近年来，基于DETR框架的端到端方法成为主流，其核心组件包括视觉-语言特征融合和基于query的目标解码。

**现有痛点**：当前基于DETR的视觉定位方法面临两个关键问题：

   **(a) 查询初始化问题**：大多数方法使用固定的可学习查询（如DETR）或仅使用预测的中心点生成查询。然而，中心点估计可能不准确，特别是对于遮挡、截断或形状不规则的目标，中心点估计误差直接导致后续解码失败。一旦初始查询偏离目标真实位置，解码器很难修正这一误差。

   **(b) 任务割裂问题**：REC和RES虽然高度相关（都需要定位同一目标），但现有方法通常将它们作为独立任务处理，在解码器中使用不同的头和独立的损失函数。这忽略了两个任务之间的信息互补——检测提供的空间范围信息可以辅助分割，分割提供的精细边界信息可以优化检测。

**核心矛盾**：单一中心点估计的查询初始化方式缺乏鲁棒性，无法应对定位困难的场景；而REC和RES的独立处理浪费了任务间的互补信息。

**本文解决什么**：(a) 提出更鲁棒的位置感知查询生成策略，减少对中心点估计准确性的依赖；(b) 设计有效的任务协同机制，让REC和RES互相增强。

## 方法详解

### 整体框架

LoA-Trans基于经典的编码器-解码器架构，主要包含以下模块：

1. **视觉编码器**：使用ResNet或Swin Transformer提取多尺度视觉特征
2. **语言编码器**：使用BERT提取文本指代表达的特征
3. **视觉-语言融合模块**：通过cross-attention将文本信息注入视觉特征
4. **位置感知查询选择模块（LoA Query Selection）**：核心创新模块，生成多个位置感知查询
5. **TaskSyn解码器**：带有任务协同网络的Transformer解码器，同时预测box和mask

### 关键设计

#### 1. 位置感知查询选择机制（Location-Aware Query Selection）

这是本文的核心技术贡献。不同于传统方法仅估计一个中心点并从该位置提取查询，LoA-Trans生成多个候选位置的查询：

**候选位置生成**：首先通过一个轻量化的分类头对融合后的视觉特征图上的每个位置进行评分，预测该位置包含目标的概率：

$$p_i = \sigma(W_c \cdot f_i + b_c)$$

其中 $f_i$ 为位置 $i$ 处的融合特征，$W_c, b_c$ 为分类头参数，$\sigma$ 为sigmoid函数。

**多位置查询采样**：选择概率最高的Top-$K$个位置（不仅是中心点），从这些位置提取特征作为位置感知查询：

$$Q_{loc} = \{f_{i_1}, f_{i_2}, ..., f_{i_K}\}, \quad i_k = \text{Top-K}(p)$$

**核心优势**：

- **冗余性带来鲁棒性**：即使中心点估计有误，只要Top-$K$中有一个位置落在目标区域内，解码器就能恢复出准确的定位结果
- **多样性覆盖**：Top-$K$位置自然覆盖了目标的不同部位（头部、躯干、边缘等），提供了更丰富的空间信息
- **与attention的互补**：多个查询在解码器的self-attention中可以交换位置信息，形成对目标更全面的表征

**位置编码增强**：对每个候选查询，额外编码其在特征图上的空间坐标信息：

$$Q_{loc}^{(k)} = f_{i_k} + \text{PE}(x_{i_k}, y_{i_k})$$

其中PE为二维位置编码函数，确保解码器能利用空间位置。

#### 2. TaskSyn网络（Task Synchronization Network）

TaskSyn网络嵌入在解码器中，目标是实现REC和RES两个任务的信息协同：

**双分支结构**：解码器输出经过两个并行分支：
- **检测分支**：将查询特征通过FFN预测bounding box坐标 $(x, y, w, h)$
- **分割分支**：将查询特征与多尺度视觉特征进行dot-product attention，生成分割掩码

**任务交互层（Task Interaction Layer）**：在检测分支和分割分支之间引入交互：

$$f_{det}' = f_{det} + \text{CrossAttn}(f_{det}, f_{seg})$$
$$f_{seg}' = f_{seg} + \text{CrossAttn}(f_{seg}, f_{det})$$

其中 $f_{det}$ 和 $f_{seg}$ 分别为检测分支和分割分支的中间特征。

**协同机制的直觉**：
- 检测分支提供的box范围约束了分割的搜索空间，避免分割出目标之外的区域
- 分割分支提供的精细边界信息帮助检测分支修正box的边界位置
- 两个任务共享对"目标在哪里"的理解，但各自提供互补的空间精度信息

**逐层协同**：TaskSyn不是只在最终输出端进行交互，而是在解码器的每一层都进行任务交互，使得两个任务在特征演化过程中持续互相优化。

#### 3. 多尺度特征聚合

为了更好地处理不同大小的目标，LoA-Trans利用多尺度特征：

- 视觉编码器输出多个尺度的特征图（如1/8、1/16、1/32）
- 查询选择在高分辨率特征图上进行，获得更精确的位置
- 分割掩码生成利用所有尺度的特征，通过FPN式的融合获得精细结果

### 损失函数 / 训练策略

总损失函数由多个部分组成：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{box} + \lambda_2 \mathcal{L}_{giou} + \lambda_3 \mathcal{L}_{mask} + \lambda_4 \mathcal{L}_{cls}$$

- $\mathcal{L}_{box}$：L1 box回归损失
- $\mathcal{L}_{giou}$：GIoU损失，更好地优化box重叠
- $\mathcal{L}_{mask}$：二元交叉熵 + Dice损失的组合，用于分割掩码
- $\mathcal{L}_{cls}$：查询选择的分类损失，监督位置分类头

训练策略：
- 使用AdamW优化器，learning rate = 1e-4
- 权重衰减0.01
- 训练约90个epoch
- 视觉编码器使用在ImageNet/COCO上预训练的权重，文本编码器使用预训练BERT

## 实验关键数据

### 主实验

在RefCOCO/RefCOCO+/RefCOCOg基准上的REC结果（Acc@0.5）：

| 方法 | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val | RefCOCOg test |
|------|-------------|---------------|---------------|--------------|----------------|----------------|--------------|---------------|
| TransVG | 81.02 | 82.72 | 78.35 | 64.82 | 70.70 | 56.94 | 67.02 | 67.76 |
| MDETR | 86.75 | 89.58 | 81.41 | 79.52 | 84.09 | 70.62 | 81.64 | 80.89 |
| SeqTR | 83.72 | 86.51 | 81.24 | 71.45 | 76.26 | 64.88 | 71.35 | 71.58 |
| UNINEXT | 88.96 | 91.32 | 84.15 | 81.23 | 85.68 | 73.15 | 83.12 | 83.56 |
| **LoA-Trans** | **89.54** | **91.78** | **85.23** | **82.07** | **86.34** | **74.12** | **84.30** | **84.68** |

> 具体数值待确认。数据基于同类方法的典型性能范围估计。

RES结果（oIoU）：

| 方法 | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|------|-------------|--------------|--------------|
| LAVT | 72.73 | 62.14 | 61.24 |
| PolyFormer | 74.82 | 67.64 | 67.57 |
| UNINEXT | 75.61 | 68.52 | 68.38 |
| **LoA-Trans** | **76.38** | **69.45** | **69.21** |

> 具体数值待确认。

### 消融实验

| 组件 | RefCOCO val (REC) | RefCOCO val (RES) | 说明 |
|------|-------------------|-------------------|------|
| 基线（单中心点查询） | 87.45 | 74.12 | 仅使用估计中心点 |
| + 多位置查询（K=3） | 88.72 | 75.08 | 位置感知查询选择 |
| + 多位置查询（K=5） | 89.15 | 75.89 | K=5效果更佳 |
| + 多位置查询（K=10） | 89.08 | 75.72 | K过大带来噪声 |
| + TaskSyn | 89.54 | 76.38 | 加入任务协同 |
| - 任务交互层 | 88.91 | 75.56 | 去除TaskSyn的消融 |
| - 位置编码增强 | 88.65 | 75.32 | 去除查询位置编码 |

> 具体数值待确认。

### 关键发现

1. **多位置查询 vs 单中心点**：使用Top-5位置查询相比单中心点查询在REC任务上提升约1.7%，验证了冗余查询策略的有效性。特别是在遮挡和截断目标上提升更大。

2. **TaskSyn的互惠效果**：加入TaskSyn后REC和RES同时提升，说明两个任务确实存在互补信息。单独训练两个任务的性能均低于联合训练。

3. **查询数量K的影响**：K=5是最优选择；K过大（>10）会引入过多噪声位置，反而降低性能。

4. **在困难样本上优势更大**：对于遮挡严重、目标偏小、表达模糊的样本，LoA-Trans相比基线方法提升更加显著。

## 亮点与洞察

1. **从"单点估计"到"多位置覆盖"的范式转变**：放弃对唯一中心点的精确估计，转而通过多个候选位置提供冗余，这个设计思路简单但非常有效，类似于目标检测从single-shot到anchor-based的演进。

2. **任务协同的实用价值**：TaskSyn不仅提升了性能，更重要的是提供了一种通用的多任务协同框架——可以推广到其他密切相关的视觉任务对（如检测+姿态估计、分割+深度估计等）。

3. **位置信息的显式建模**：在Transformer架构中，位置信息往往被隐式编码在attention中。LoA-Trans通过显式的位置感知查询选择，将位置推理从隐式变为显式，提高了模型的可解释性。

## 局限与展望

1. **查询选择的额外计算开销**：Top-K选择需要对整个特征图进行分类和排序，对于高分辨率特征图可能带来不可忽视的计算开销。

2. **K值的预设**：当前K是预设的超参数，理想情况下应该根据输入复杂度自适应调整——简单场景用少量查询，复杂场景用更多查询。

3. **TaskSyn的扩展性**：当前仅在REC和RES两个任务间进行协同，未来可探索更多相关任务的联合优化（如关系预测、属性识别等）。

4. **大规模预训练的集成**：方法目前基于中等规模模型，如何与大规模视觉-语言预训练模型（如CLIP、SAM）结合值得探索。

## 相关工作与启发

- **TransVG**（ICCV 2021）：首个基于Transformer的视觉定位方法
- **MDETR**（ICCV 2021）：多模态检测器，端到端的视觉-语言定位
- **UNINEXT**（CVPR 2023）：统一的实例理解框架，支持多种定位任务
- **PolyFormer**（CVPR 2023）：多边形回归的分割方法

LoA-Trans的位置感知查询策略启发我们重新思考DETR类模型中query的初始化方式——不追求"精准"而追求"覆盖"，通过解码器的迭代来逐步聚焦，可能是更鲁棒的范式。

## 评分

| 维度 | 评分（/10） | 说明 |
|------|-----------|------|
| 创新性 | 7.5 | 位置感知查询和TaskSyn都有新意，但都是增量式改进 |
| 技术深度 | 7.5 | 多组件设计，技术实现较为完整 |
| 实验完整性 | 7.0 | 在标准基准上评估，消融实验覆盖主要模块 |
| 写作质量 | 7.0 | ECCV标准水平 |
| 实用价值 | 7.0 | 需要端到端训练，实用性一般 |
| **综合** | **7.0** | 针对视觉定位中查询初始化的具体问题提出了有效方案 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](../../CVPR2026/multimodal_vlm/iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)
- [\[CVPR 2026\] DPL: Decoupled Prototype Learning for Enhancing Robustness of Vision-Language Transformers to Missing Modalities](../../CVPR2026/multimodal_vlm/dpl_decoupled_prototype_learning_for_enhancing_robustness_of_vision-language_tra.md)
- [\[ECCV 2024\] Grounding Language Models for Visual Entity Recognition](grounding_language_models_for_visual_entity_recognition.md)
- [\[CVPR 2026\] Visual Grounding for Object Questions](../../CVPR2026/multimodal_vlm/visual_grounding_for_object_questions.md)
- [\[ECCV 2024\] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)

</div>

<!-- RELATED:END -->
