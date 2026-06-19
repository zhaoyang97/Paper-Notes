---
title: >-
  [论文解读] Language Guided Concept Bottleneck Models for Interpretable Continual Learning
description: >-
  [CVPR 2025][可解释性][持续学习] 本文将语言引导的概念瓶颈模型（CBM）引入持续学习，用 ChatGPT 生成人类可理解的概念、CLIP 编码概念嵌入构建概念瓶颈层，在缓解灾难性遗忘的同时提供透明的决策解释，在 ImageNet-subset 上超越 SOTA 3.06%。 领域现状：持续学习需要模型不断学习新…
tags:
  - "CVPR 2025"
  - "可解释性"
  - "持续学习"
  - "概念瓶颈模型"
  - "CLIP"
  - "灾难性遗忘"
---

# Language Guided Concept Bottleneck Models for Interpretable Continual Learning

**会议**: CVPR 2025  
**arXiv**: [2503.23283](https://arxiv.org/abs/2503.23283)  
**代码**: [https://github.com/FisherCats/CLG-CBM](https://github.com/FisherCats/CLG-CBM)  
**领域**: LLM效率  
**关键词**: 持续学习, 概念瓶颈模型, 可解释性, CLIP, 灾难性遗忘

## 一句话总结
本文将语言引导的概念瓶颈模型（CBM）引入持续学习，用 ChatGPT 生成人类可理解的概念、CLIP 编码概念嵌入构建概念瓶颈层，在缓解灾难性遗忘的同时提供透明的决策解释，在 ImageNet-subset 上超越 SOTA 3.06%。

## 研究背景与动机

**领域现状**：持续学习需要模型不断学习新任务而不遗忘旧知识（灾难性遗忘）。现有方法分为正则化、回放和架构扩展三类，但都是黑盒决策，缺乏可解释性。

**现有痛点**：随着模型不断更新知识，理解其学到了什么、如何保留旧信息变得至关重要。ICICLE 尝试通过原型部分网络提升可解释性，但严重限制了模型可塑性。

**核心矛盾**：可解释性与灾难性遗忘缓解之间存在 trade-off——增加透明性约束往往限制模型适应新任务的能力。

**本文目标**：设计一个同时提升可解释性和持续学习性能的框架。

**切入角度**：概念瓶颈模型天然具有可解释性（中间层对应人类概念），结合 CLIP 的零样本能力和 ChatGPT 的概念生成，可为持续学习提供跨任务泛化的语义概念。

**核心 idea**：用 ChatGPT 为每个类别生成概念词，用 CLIP 文本编码器编码为概念瓶颈层，通过语义一致性对齐实现可跨任务泛化的可解释表示。

## 方法详解

### 整体框架
每当新任务到来：(1) 用 ChatGPT 为新类别生成人类可理解概念；(2) CLIP 文本编码器将概念编码为嵌入向量，构建概念瓶颈层(CBL)；(3) 图像通过 CLIP 视觉编码器提取特征，与 CBL 计算概念得分矩阵；(4) 概念得分向量用于最终分类。语义知识增强原型缓解遗忘。

### 关键设计

1. **语言引导的概念瓶颈层（Language-Guided CBL）**:

    - 功能：在特征提取和分类之间插入人类可理解的概念中间层
    - 核心思路：对每个类别查询 ChatGPT 生成描述性概念词，然后用概念选择模块从候选概念中挑选最具信息量和区分性的概念，构建任务特定的概念池 $\mathcal{C}$。概念激活矩阵 $E_{clip} = f_I(\mathcal{X}) \cdot f_T(\mathcal{C})^\top$ 度量图像与每个概念的对齐程度
    - 设计动机：概念瓶颈的每个神经元对应一个可理解概念，天然提供决策解释

2. **语义增强原型（Semantic-Augmented Prototypes）**:

    - 功能：利用语义知识增强类别原型，缓解灾难性遗忘
    - 核心思路：利用概念得分向量构建类别原型表示，新任务到来时通过语义相似性关联新旧概念，保持旧类别的决策边界稳定
    - 设计动机：传统原型方法仅靠特征距离，语义增强提供更鲁棒的类间区分

3. **概念可视化与解释**:

    - 功能：为模型预测提供人类可理解的解释
    - 核心思路：对每个预测，展示激活最高的概念及其得分，直观解释"模型为什么做出这个分类"
    - 设计动机：持续学习场景下理解模型决策尤为重要

### 损失函数 / 训练策略
交叉熵损失 + Mahalanobis 损失引导语义知识学习，用于概念选择。

## 实验关键数据

### 主实验
在7个基准数据集上超越 SOTA，ImageNet-subset 上最终平均准确率提升 3.06%，同时全程维持可解释性。

### 关键发现
- 语言引导的概念瓶颈不仅提升可解释性，还意外提升了持续学习性能
- 概念的跨任务泛化能力比纯视觉特征更强

## 亮点与洞察
- 首次将 CBM 的可解释性优势系统性地引入持续学习
- ChatGPT + CLIP 的概念生成管线可推广到其他需要可解释性的场景

## 局限与展望
- 概念质量依赖 ChatGPT 生成的准确性
- 随任务数增加，概念瓶颈层维度持续增长

## 相关工作与启发
- **vs ICICLE**: 原型部分网络可解释但限制可塑性。本文的CBM提供更灵活的可解释性
- **vs 标准CLIP-CBM**: 仅处理静态分类。本文扩展到持续学习场景

## 评分
- 新颖性: ⭐⭐⭐⭐ CBM+持续学习的结合角度新颖
- 实验充分度: ⭐⭐⭐⭐ 7个数据集全面验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对可解释AI有实际推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability](albm_attribute_concept_space.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)

</div>

<!-- RELATED:END -->
