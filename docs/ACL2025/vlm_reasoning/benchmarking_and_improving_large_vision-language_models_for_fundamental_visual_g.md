---
title: >-
  [论文解读] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning
description: >-
  [ACL 2025][多模态VLM][视觉图结构理解] 本文构建了一个系统性评测基准来评估大型视觉语言模型（LVLM）在基础视觉图结构理解与推理上的能力，发现现有模型在此类任务上表现欠佳，并提出了针对性的改进方法。 领域现状：大型视觉语言模型（如 GPT-4V、LLaVA 等）在图像描述、VQA 等任务上取得了优异表现。然而…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉图结构理解"
  - "视觉推理"
  - "图理解基准"
  - "大型视觉语言模型"
  - "结构化视觉推理"
---

# Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning

**会议**: ACL 2025  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 视觉图结构理解、视觉推理、图理解基准、大型视觉语言模型、结构化视觉推理

## 一句话总结
本文构建了一个系统性评测基准来评估大型视觉语言模型（LVLM）在基础视觉图结构理解与推理上的能力，发现现有模型在此类任务上表现欠佳，并提出了针对性的改进方法。

## 研究背景与动机

**领域现状**：大型视觉语言模型（如 GPT-4V、LLaVA 等）在图像描述、VQA 等任务上取得了优异表现。然而，这些模型对结构化视觉信息（如图、网络、流程图等图结构）的理解能力尚未被系统性地评估。

**现有痛点**：现有的 LVLM 评测基准主要关注自然图像理解（如场景识别、物体检测），忽视了一个重要的视觉理解能力——对图（graph）结构的基础理解，包括节点识别、边关系判断、路径推理等。

**核心矛盾**：图结构在科学论文、知识图谱可视化、流程图等场景中无处不在，但 LVLM 是否真正理解这些视觉图结构尚不清楚。图结构理解需要结合视觉感知（识别节点和边）和逻辑推理（路径查找、连通性判断），这对模型提出了更高要求。

**本文目标**：（1）构建覆盖多种图类型和推理任务的综合基准；（2）系统评估现有 LVLM 的表现；（3）提出改进方法提升模型的图结构理解能力。

**切入角度**：图（graph）是一种基础的数据结构，视觉化后的图理解涉及底层视觉感知和高层结构推理的结合，是测试 LVLM 综合能力的理想试金石。

**核心 idea**：构建系统的视觉图理解基准（VGraphBench），揭示 LVLM 的结构理解短板，并通过图结构感知的训练策略来弥补不足。

## 方法详解

### 整体框架
工作分为两个部分：（1）基准构建——包含多种图类型（有向图、无向图、加权图、树等）和多种任务（节点计数、边检测、路径判断、最短路径、连通性等）；（2）改进方法——通过结构化数据增强和针对性微调来提升 LVLM 的图理解能力。

### 关键设计

1. **视觉图理解基准（VGraphBench）**:

    - 功能：系统评测 LVLM 在基础图结构理解和推理上的能力
    - 核心思路：设计多难度层级的任务，从简单的节点/边识别到复杂的路径推理和图属性判断。图像通过程序化方式生成，确保控制变量（图大小、布局、颜色等），避免自然图像中的干扰因素。每个任务都有明确的正确答案
    - 设计动机：需要消除自然图像中的语义先验，让模型真正依赖视觉结构理解能力

2. **多任务图理解评测体系**:

    - 功能：涵盖从感知到推理的完整能力谱
    - 核心思路：任务分为两大类——感知任务（节点计数、边检测、度数计算）和推理任务（连通性判断、最短路径、环检测、拓扑排序等）。感知任务测试模型是否"看到"了图结构，推理任务测试模型是否"理解"了图结构
    - 设计动机：区分感知和推理的失败模式，帮助诊断模型的具体弱点

3. **图结构感知微调策略**:

    - 功能：通过构造图结构理解训练数据来提升 LVLM 的表现
    - 核心思路：生成大量包含图结构的图像-问答对，覆盖各种图类型和任务类型，对 LVLM 进行指令微调。训练数据中包含从简单到复杂的渐进式结构理解任务，帮助模型逐步建立图结构的视觉理解能力
    - 设计动机：现有 LVLM 的预训练数据缺乏足够的图结构理解样本，需要针对性数据补充

### 损失函数 / 训练策略
微调阶段使用标准的指令微调损失（交叉熵），关键在于训练数据的构造策略。

## 实验关键数据

### 主实验

| 模型 | 节点计数 | 边检测 | 连通性 | 最短路径 | 平均 |
|------|---------|--------|--------|---------|------|
| GPT-4V | 中等 | 中等 | 较低 | 较低 | ~45% |
| LLaVA-1.5 | 较低 | 较低 | 较低 | 很低 | ~30% |
| 本文微调后 | 显著提升 | 显著提升 | 提升 | 提升 | ~60%+ |
| 随机基线 | ~20% | ~50% | ~50% | ~10% | ~25% |

### 消融实验

| 配置 | 平均准确率 | 说明 |
|------|-----------|------|
| Full 微调 | 最优 | 完整图结构训练数据 |
| 仅感知任务训练 | 中等 | 推理任务提升有限 |
| 仅推理任务训练 | 较低 | 感知基础不足影响推理 |
| 无图布局增强 | 下降 | 对布局变化敏感 |

### 关键发现
- 所有现有 LVLM 在图结构推理任务上表现远低于人类水平，尤其在最短路径和拓扑排序等任务上接近随机水平
- 图的大小（节点数量）是关键影响因素，节点超过 10 个后准确率急剧下降
- 感知能力是推理的基础——如果模型连节点和边都识别不准确，推理任务必然失败
- 简单的微调即可带来显著提升，说明这不是架构层面的根本限制，而是训练数据覆盖不足

## 亮点与洞察
- **填补评测空白**：图结构理解是 LVLM 能力评测的重要维度，此前被严重忽视。基准的系统性设计能帮助社区精确定位模型弱点
- **程序化生成控制变量**：通过程序生成图像而非收集自然图像，消除了语义先验的干扰，是评估结构理解能力的正确方式。可迁移到流程图、UML图、电路图等其他结构化视觉内容的评测

## 局限与展望
- 基准中使用的是程序化生成的"干净"图结构，真实世界的图像（如手绘图、论文中的图表）更具挑战性
- 微调方法的泛化性有待验证——在训练分布外的图类型上是否仍然有效
- 可以扩展到更复杂的图类型如超图、动态图
- 结合图神经网络（GNN）的思路来增强 LVLM 的图结构理解是有前景的方向

## 相关工作与启发
- **vs MathVista/ChartQA等基准**: 这些基准侧重数学/图表理解，本文关注更基础的图结构理解，是更底层的能力
- **vs TextVQA/DocVQA**: 文档理解关注文本布局，本文关注拓扑结构，两者互补
- **vs NLGraph（文本图推理）**: NLGraph将图结构用文本描述，本文评估的是从视觉中理解图结构，更接近真实场景
- **vs GNN相关工作**: GNN直接在图上操作，本文评估的是LVLM从图像渲染中提取图结构的能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统性评估 LVLM 的图结构理解能力
- 实验充分度: ⭐⭐⭐⭐ 多模型、多任务、多难度评测
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，基准设计合理
- 价值: ⭐⭐⭐⭐ 揭示了 LVLM 的重要能力短板

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[ACL 2025\] Unveiling the Lack of LVLM Robustness to Fundamental Visual Variations: Why and Path Forward](unveiling_the_lack_of_lvlm_robustness_to_fundamental_visual_variations_why_and_p.md)
- [\[ACL 2025\] Harnessing PDF Data for Improving Japanese Large Multimodal Models](harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)
- [\[NeurIPS 2025\] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)

</div>

<!-- RELATED:END -->
