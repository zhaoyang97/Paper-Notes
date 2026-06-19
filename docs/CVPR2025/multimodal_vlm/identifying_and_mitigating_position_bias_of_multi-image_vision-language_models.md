---
title: >-
  [论文解读] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models
description: >-
  [CVPR 2025][多模态VLM][多图推理] 本文发现多图大视觉语言模型（LVLM）存在严重的位置偏差——开源模型偏重后置图片、闭源模型忽视中间图片——并提出了一种无需训练的SoFt Attention（SoFA）方法，通过在图像间因果注意力与双向注意力之间做线性插值来缓解该偏差，在多个基准上提升了2~3%的平均准确率。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "多图推理"
  - "位置偏差"
  - "注意力机制"
  - "大视觉语言模型"
  - "无训练方法"
---

# Identifying and Mitigating Position Bias of Multi-image Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2503.13792](https://arxiv.org/abs/2503.13792)  
**代码**: [https://github.com/xytian1008/sofa](https://github.com/xytian1008/sofa)  
**领域**: 多模态VLM  
**关键词**: 多图推理、位置偏差、注意力机制、大视觉语言模型、无训练方法

## 一句话总结

本文发现多图大视觉语言模型（LVLM）存在严重的位置偏差——开源模型偏重后置图片、闭源模型忽视中间图片——并提出了一种无需训练的SoFt Attention（SoFA）方法，通过在图像间因果注意力与双向注意力之间做线性插值来缓解该偏差，在多个基准上提升了2~3%的平均准确率。

## 研究背景与动机

多模态大模型已从单图推理扩展到多图推理，广泛应用于差异检测、图像计数、视频理解等任务。然而，NLP领域中已经发现LLM在处理多文档时存在"位置偏差"（如"lost in the middle"现象），即模型倾向于关注输入序列的首尾位置而忽略中间信息。本文的核心问题是：**这种位置偏差是否也存在于多图LVLM中？** 实验发现，仅仅交换输入图片的顺序，就可能导致约30%的预测发生改变，准确率波动高达10%。这严重影响了模型的鲁棒性和可靠性。作者进一步分析发现，图像间的**因果注意力机制**是位置偏差的根本原因——后置图像能看到前面所有图像，而前置图像则是"孤立"的。核心idea：通过在因果注意力和双向注意力之间做软插值，平滑图像token对位置信息的依赖。

## 方法详解

### 整体框架

作者首先设计了Position-wise Question Answering（PQA）任务来定量识别位置偏差的模式，然后分析了位置偏差的机制来源（因果注意力 vs 位置编码），最后提出SoFA方法来缓解该偏差。SoFA是一种即插即用的无训练方法，只需修改LVLM中图像token之间的注意力掩码即可。

### 关键设计

1. **Position-wise Question Answering（PQA）任务**:
    - 功能：定量评估LVLM在每个图像位置上的推理能力
    - 核心思路：让模型对每张图片独立回答同一个问题（如"每张图片中有多少只猫？"），输出形如 $[3, 2, 0, ...]$ 的列表，从而逐位置统计准确率
    - 设计动机：现有多图基准只评估整体性能，无法区分模型在哪些位置表现好/差。PQA通过从VQAv2中构造位置中立的问题集，实现了细粒度的位置分析

2. **位置偏差的机制分析**:
    - 功能：找到位置偏差的根因
    - 核心思路：影响位置信息的两个因素是位置编码和因果注意力。作者对比了三种图像间注意力机制：(A) 因果attention（默认，单向）、(B) 孤立attention（图像间无交互）、(C) 双向attention（图像间全连接）。实验表明将因果attention改为孤立或双向均可显著缓解偏差，但会导致性能下降
    - 设计动机：直接修改位置编码过于激进（会破坏视频理解等需要时序信息的场景），而因果注意力是更温和的切入点

3. **SoFt Attention（SoFA）**:
    - 功能：在不重新训练的情况下缓解多图位置偏差
    - 核心思路：对图像间的注意力掩码做线性插值：$\mathbf{M}_{\text{soft}} = (1-\sigma)\mathbb{1}_{\text{causal}} + \sigma\mathbb{1}_{\text{bidirectional}}$，其中 $\sigma$ 控制双向注意力的比例。**仅修改图像token间的attention，文本token间保持因果attention不变**
    - 设计动机：完全切换到双向attention会偏离训练分布导致性能下降，而通过插值可以在"准确率"和"鲁棒性"之间找到平衡。SoFA每隔两层部署一次（而非每层都用），以更好地贴合训练框架

### 损失函数 / 训练策略

SoFA是**无训练方法**，不涉及任何参数更新。超参数 $\sigma$ 通过每个任务的32-shot验证集确定最优值。模型使用FP16精度和Flash Attention，禁用sub-image splitting以确保公平性。

## 实验关键数据

### 主实验（位置鲁棒性）

| 模型 | 基准 | 无SoFA预测不一致率 | 有SoFA预测不一致率 | 减少幅度 |
|------|------|-------------------|-------------------|---------|
| Idefics2 | BLINK | 41.55% | 12.36% | -29.19% |
| InternVL2 | MuirBench | 38.65% | 5.16% | -33.49% |
| LLaVA-NeXT | MIRB | 28.56% | 6.96% | -21.60% |

### 整体性能提升

| 模型 | BLINK | Mantis-Eval | MuirBench | MIRB | NLVR2 | MVBench |
|------|-------|-------------|-----------|------|-------|---------|
| InternVL2 | 38.95 | 50.30 | 54.53 | 42.66 | 85.56 | 29.31 |
| InternVL2+SoFA | **43.26** | **51.11** | **57.14** | **46.19** | **88.19** | **32.77** |
| LLaVA-NeXT | 53.34 | 50.83 | 48.22 | 57.15 | 87.28 | 54.26 |
| LLaVA-NeXT+SoFA | **55.92** | **54.51** | **50.43** | **60.67** | **89.45** | **57.71** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 因果attention（默认） | 偏差严重，前端准确率低 | 基线 |
| 孤立attention | 偏差消除但性能大幅下降 | 图像间无交互导致OOD |
| 双向attention | 偏差缓解但性能略降 | 偏离训练分布 |
| SoFA（插值） | 偏差缓解且性能提升 | 最优平衡 |
| 100张图长上下文 | 49.19→55.11%（+5.92%） | SoFA在长上下文优势更大 |

### 关键发现

- 开源模型表现为**近因偏差**（recency bias），即后置图片表现好、前置图片表现差；闭源模型（如GPT-4o）表现为**U型曲线**，中间位置最差
- 随着图片数量增加，偏差加剧：20张图时OpenFlamingo的前后准确率差达14%
- SoFA在视觉检索和类比任务上提升最大（+6.84%和+5.53%），因为这些任务需要充分理解第一张参考图
- 16-shot in-context learning场景中，SoFA将VizWiz准确率从45.35%提升到49.17%

## 亮点与洞察

- **问题定义清晰**：首次系统性研究多图LVLM中的位置偏差，设计了PQA任务实现位置粒度的评估
- **方法简洁有效**：仅修改注意力掩码的插值系数，无需重训练，即插即用
- **机制分析深入**：通过注意力分布可视化清楚展示了SoFA如何将注意力从尾部集中分散到全局

## 局限与展望

- $\sigma$ 需要针对每个任务用验证集调优，增加了使用成本
- 仅缓解偏差而未根除——因果注意力仍有残余影响
- 对于本身需要图片顺序信息的任务（如时序视频理解），SoFA的效果需要谨慎评估
- 未探索与位置编码修改的联合使用

## 相关工作与启发

- 受NLP中"lost in the middle"研究启发，将位置偏差问题迁移到多模态领域
- 与LLM-as-a-judge中的位置偏差（倾向选第一个回答）现象类似
- SoFA的思路启发：对于其他LVLM继承自LLM的固有偏差，可考虑类似的"软修正"策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究LVLM位置偏差并提出PQA评估框架，但方法本身（注意力插值）相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖6个基准、5个开源模型+GPT-4o、多种场景（in-context learning、长上下文、任务类型分析）
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从现象发现→机制分析→解决方案层层递进
- 价值: ⭐⭐⭐⭐ 揭示了LVLM的重要缺陷，SoFA方法实用性强，但2~3%的提升相对有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] What's in the Image? A Deep-Dive into the Vision of Vision Language Models](whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](../../NeurIPS2025/multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[CVPR 2025\] Joint Vision-Language Social Bias Removal for CLIP](joint_vision-language_social_bias_removal_for_clip.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [\[CVPR 2025\] Seeing the Abstract: Translating the Abstract Language for Vision Language Models](seeing_the_abstract_translating_the_abstract_language_for_vision_language_models.md)

</div>

<!-- RELATED:END -->
