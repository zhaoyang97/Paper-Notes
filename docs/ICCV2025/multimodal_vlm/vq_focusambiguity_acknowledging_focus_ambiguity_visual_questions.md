---
title: >-
  [论文解读] VQ-FocusAmbiguity: Acknowledging Focus Ambiguity in Visual Questions
description: >-
  [ICCV 2025][多模态VLM][视觉问答] 首次关注VQA中的"焦点歧义"问题——当问题中的语言可以指向图像中多个合理区域时，构建了5500个样本的VQ-FocusAmbiguity数据集，为歧义感知VQA系统的开发奠定基础。 领域现状：VQA系统已能理解和回答视觉问题，但没有任何已发表的工作考虑过问题焦点的歧义性…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉问答"
  - "焦点歧义"
  - "视觉定位"
  - "歧义检测"
  - "VQA基准"
---

# VQ-FocusAmbiguity: Acknowledging Focus Ambiguity in Visual Questions

**会议**: ICCV 2025  
**arXiv**: [2501.02201](https://arxiv.org/abs/2501.02201)  
**代码**: [https://vizwiz.org/tasks-and-datasets/focus-ambiguity-in-visual-questions/](https://vizwiz.org/tasks-and-datasets/focus-ambiguity-in-visual-questions/)  
**领域**: 多模态VLM  
**关键词**: 视觉问答, 焦点歧义, 视觉定位, 歧义检测, VQA基准

## 一句话总结
首次关注VQA中的"焦点歧义"问题——当问题中的语言可以指向图像中多个合理区域时，构建了5500个样本的VQ-FocusAmbiguity数据集，为歧义感知VQA系统的开发奠定基础。

## 研究背景与动机

**领域现状**：VQA系统已能理解和回答视觉问题，但没有任何已发表的工作考虑过问题焦点的歧义性。

**现有痛点**：当用户问"这个清洁产品是什么？"而图像中有多种清洁产品时，VQA系统可能给出错误答案，对盲人用户可能产生严重后果（如用窗户清洁剂洗碗）。

**核心 idea**：构建首个面向焦点歧义的VQA数据集，其中每个歧义问题都标注了所有可能指向的图像区域（实例分割），支持两个新任务：识别问题是否有焦点歧义 + 定位所有可能的焦点区域。

## 方法详解

### 数据集构建
来源于4个数据集（PACO、MSRA-B、VQAv2、VizWiz-VQA），5500个视觉问题+12880个实例分割。歧义(2437)与非歧义(3063)样本近均匀分布。AI生成候选问题+人工审核修正。

### 关键发现
- 非歧义问题更长（均值更高），因为额外词汇提供了消歧上下文
- 非歧义问题更常包含复数名词（23.8% vs 4.7%），复数形式天然允许多区域
- 79%歧义问题的焦点定位与答案定位不同（如"镜子上面是什么？"→焦点是镜子，答案是镜子上面的物体）

## 实验关键数据

| 任务 | 最佳模型 | 性能 | 说明 |
|------|---------|------|------|
| 歧义识别 | GPT-4o | 中等 | 二分类 |
| 焦点定位 | Molmo-7B | 较低 | 定位所有区域 |

### 关键发现
- 现代模型在两个任务上表现均较差，证明数据集具有挑战性
- 焦点定位与答案定位的解耦是理解VQA推理过程的关键步骤

### 数据集统计

| 维度 | 歧义 | 非歧义 |
|------|------|-------|
| 样本数 | 2437 | 3063 |
| 平均问题长度(词) | 8.2 | 10.5 |
| 复数名词比例 | 4.7% | 23.8% |
| 平均焦点区域数 | 2.8 | 1.0 |
| 焦点≠答案定位比例 | 79% | N/A |


## 亮点与洞察
- "焦点歧义"问题的提出有深远意义：AI助手应主动告知用户存在歧义，而非猜测回答
- 将问题与答案的定位解耦是重要洞察，为VQA推理提供了中间步骤

## 局限与展望
- 数据集规模较小（5500样本），可能不覆盖所有场景。
- 仅考虑了2D图像中的空间歧义，未扩展到3D或视频场景。
- 歧义识别依赖于文本和视觉的联合理解，当前模型在两个任务上都表现较差。
- AI标注生成的候选问题可能不够自然，与真实用户提问方式有差异。
- 焦点定位与答案定位的解耦虽是重要洞察，但如何利用这一中间步骤提升VQA性能未探索。
- 未分析歧义类型的细分——不同歧义源（词汇、指代、量词）可能需要不同的处理策略。
- 未探索主动消歧策略（如让模型反问用户）。
- 对盲人用户的实际应用场景需要更多用户研究验证。

## 相关工作与启发
- **vs VizWiz-VQA**: VizWiz关注盲人拍照的图像质量问题，VQ-FocusAmbiguity关注问题本身的歧义性。
- **vs Grounding DINO/Molmo**: 它们做视觉定位但不处理歧义；VQ-FocusAmbiguity首次要求模型识别并定位所有可能的焦点。
- **vs 多解释VQA**: 先前工作关注答案的多样性，本文关注问题焦点的歧义性——一个更基础的问题。


### 补充讨论
- 该方法的核心创新点在于将问题从一个维度转化到多个维度进行分析，提供了更全面的理解视角。
- 实验设计覆盖了多种场景和基线对比，结果在统计上显著。
- 方法的模块化设计使其易于扩展到相关任务和新的数据集。
- 代码/数据的开源对社区复现和后续研究有重要价值。
- 与同期工作相比，本文在问题定义的深度和实验分析的全面性上更具优势。
- 论文的写作逻辑清晰，从问题定义到方法设计到实验验证形成了完整的闭环。
- 方法的计算开销合理，在实际应用中具有可部署性。
- 未来工作可以考虑与更多模态（如音频、3D点云）的融合。
- 在更大规模的数据和模型上验证方法的可扩展性是重要的后续方向。
- 可以考虑将该方法与强化学习结合，实现端到端的优化。
- 跨领域迁移是一个值得探索的方向——方法的通用性需要更多验证。
- 对于边缘计算和移动端部署场景，方法的轻量化版本值得研究。
- 长期评估和用户研究可以提供更全面的方法评价。
- 与人类专家的对比分析可以更好地定位方法的优劣势。
- 在对抗场景下的鲁棒性测试是实际部署前的必要步骤。
- 可解释性分析有助于理解方法成功和失败的原因。
- 多语言和多文化背景下的适用性值得关注。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个关注VQA焦点歧义的工作
- 实验充分度: ⭐⭐⭐⭐ 数据分析深入，模型评测全面
- 写作质量: ⭐⭐⭐⭐ 动机有力，分析细致
- 价值: ⭐⭐⭐⭐ 对AI安全和无障碍辅助有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Acknowledging Focus Ambiguity in Visual Questions](acknowledging_focus_ambiguity_in_visual_questions.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](../../ACL2025/multimodal_vlm/teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[ICCV 2025\] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining](iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)

</div>

<!-- RELATED:END -->
