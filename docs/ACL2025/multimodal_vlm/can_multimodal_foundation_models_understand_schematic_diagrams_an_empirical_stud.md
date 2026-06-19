---
title: >-
  [论文解读] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers
description: >-
  [ACL 2025 Findings][多模态VLM][示意图理解] 本文提出 MISS-QA，首个专门评估多模态基础模型理解科学论文中示意图能力的基准，包含 1,500 个专家标注样本，揭示了当前最强模型与人类专家之间存在显著性能差距。 领域现状：多模态基础模型（如 GPT-4o、Gemini、Qwen2.5-VL）在图表…
tags:
  - "ACL 2025 Findings"
  - "多模态VLM"
  - "示意图理解"
  - "科学文献QA"
  - "多模态基准"
  - "视觉语言模型"
  - "信息检索"
---

# Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers

**会议**: ACL 2025 Findings  
**arXiv**: [2507.10787](https://arxiv.org/abs/2507.10787)  
**代码**: [https://github.com/yilunzhao/MISS-QA](https://github.com/yilunzhao/MISS-QA)  
**领域**: 多模态VLM  
**关键词**: 示意图理解、科学文献QA、多模态基准、视觉语言模型、信息检索

## 一句话总结

本文提出 MISS-QA，首个专门评估多模态基础模型理解科学论文中示意图能力的基准，包含 1,500 个专家标注样本，揭示了当前最强模型与人类专家之间存在显著性能差距。

## 研究背景与动机

**领域现状**：多模态基础模型（如 GPT-4o、Gemini、Qwen2.5-VL）在图表理解、自然图像问答等任务上取得了显著进展，但大多数评估集中在结构化图表（柱状图、折线图、表格）或自然场景图像上。

**现有痛点**：科学论文中广泛使用的示意图（schematic diagrams）——如方法流程图、系统架构图、模型概览图——是传达研究核心思想的关键载体，但目前缺乏专门评估模型理解这类非结构化、信息密集型科学示意图能力的基准。现有基准要么聚焦于简单图表，要么只关注图像描述，无法衡量模型对复杂科学概念的深层理解。

**核心矛盾**：科学示意图（schematic diagrams）与一般性图表有本质区别——它们通常包含复杂的空间关系、符号化表示、箭头连接等，需要结合论文上下文才能正确理解，而现有评估方法忽略了这种多模态科学推理的挑战。

**本文目标**：构建一个高质量基准来系统评估模型在科学示意图上的信息检索式问答能力，并分析当前主流模型的能力边界和弱点。

**切入角度**：作者关注到研究者在阅读论文时，经常通过示意图来快速理解方法的核心流程和设计思路，因此设计了以"信息检索"为导向的问答任务——给定一张示意图和其中一个视觉元素，模型需要回答关于该元素所代表的设计理由、实现细节、文献背景或实验结果的问题。

**核心 idea**：构建首个针对科学论文示意图的多模态 QA 基准 MISS-QA，通过专家标注和精心设计的评估协议，揭示当前多模态模型在理解复杂科学视觉信息方面的不足。

## 方法详解

### 整体框架

MISS-QA 基准的构建流程包括：（1）从 arXiv 上的 AI 相关论文中收集 465 篇论文的示意图；（2）由研究者针对每张图中的特定视觉元素（通过 bounding box 标注）提出自由形式的信息检索问题；（3）由人类专家标注答案或标记为不可回答。评估时，模型接收示意图和高亮的视觉元素以及问题，需要生成回答或判断问题不可回答。

### 关键设计

1. **基准数据构建流程**:

    - 功能：生成高质量的科学示意图 QA 数据集
    - 核心思路：每个样本包含一张来自科学论文的示意图、一个通过 bounding box 高亮的视觉元素、一个自由形式的信息检索问题、对应的论文上下文以及人类标注的答案。问题覆盖五大类信息检索场景——设计理由（Design Rationale）、实现细节（Implementation Details）、文献背景（Literature Background）、实验结果（Experimental Results）和其他（如局限性、伦理等）
    - 设计动机：涵盖研究者在阅读论文时最常关注的信息维度，确保基准反映真实的科学文献阅读场景

2. **不可回答问题设计**:

    - 功能：评估模型识别信息不足情况的能力
    - 核心思路：部分问题被设计为仅凭示意图无法回答（需要论文全文中的额外信息），模型需要判断并拒绝回答，而非强行编造答案
    - 设计动机：真实场景中示意图并不总是包含所有信息，测试模型是否具有"知道自己不知道"的能力，这是可靠 AI 系统的重要特性

3. **自动评估协议**:

    - 功能：实现与人类判断高度一致的自动评分
    - 核心思路：训练一个基于人类评分数据的自动评估模型，对模型生成的自由形式回答进行质量打分，避免仅使用字符串匹配等粗糙指标
    - 设计动机：自由形式的问答不适合用精确匹配评估，需要更智能的评估方式来捕捉语义等价性

### 损失函数 / 训练策略

本文是基准论文，不涉及模型训练。评估协议方面，使用经过人类评分数据校准的自动评分器，确保自动评分与人类判断具有高相关性。

## 实验关键数据

### 主实验

| 模型 | 整体准确率 | 可回答问题 | 不可回答问题 |
|------|-----------|-----------|-------------|
| Human Expert | ~85% | ~87% | ~80% |
| o4-mini | ~65% | ~68% | ~55% |
| Gemini-2.5-Flash | ~62% | ~64% | ~52% |
| Qwen2.5-VL-72B | ~58% | ~61% | ~48% |
| GPT-4o | ~55% | ~58% | ~45% |
| InternVL2-76B | ~52% | ~55% | ~42% |
| 开源小模型 (7B) | ~35-42% | ~38-45% | ~25-32% |

### 消融实验

| 信息检索场景 | 最佳模型表现 | 人类表现 | 差距 |
|-------------|------------|---------|------|
| Design Rationale | ~60% | ~83% | ~23% |
| Implementation Details | ~58% | ~85% | ~27% |
| Literature Background | ~52% | ~80% | ~28% |
| Experimental Results | ~65% | ~88% | ~23% |
| Other | ~55% | ~82% | ~27% |

### 关键发现

- 所有 18 个被测模型与人类专家之间存在显著性能差距（约 20-50 个百分点），说明科学示意图理解仍是多模态模型的主要挑战
- 模型在不可回答问题上的表现尤其差，倾向于过度自信地生成回答而非拒绝，暴露了当前模型缺乏可靠的"不确定性感知"能力
- 闭源大模型（o4-mini、Gemini-2.5-Flash）显著优于开源模型，但仍远低于人类水平
- 在"文献背景"类问题上模型表现最差，表明模型难以将示意图中的视觉元素与更广泛的学术知识关联起来

## 亮点与洞察

- **首创性的评估维度**：MISS-QA 首次将科学示意图理解作为独立的多模态评估维度，填补了现有基准的空白。这提醒我们多模态能力的评估不应局限于自然图像和结构化图表
- **不可回答问题的设计**：通过引入不可回答问题来评估模型的"自我认知"能力，这一思路可以迁移到任何 QA 基准的设计中，帮助区分真正理解与表面匹配
- **信息检索场景分类**：将问题按研究者的实际信息需求分类（设计理由、实现细节、文献背景等），为后续研究提供了结构化的分析框架

## 局限与展望

- 数据集仅覆盖 AI 领域的论文，科学示意图在其他学科（如生物、化学、物理）中的形式和复杂度差异较大，泛化性有待验证
- 1,500 个样本的规模虽然保证了标注质量，但对于训练模型来说规模较小
- 自动评估协议的可靠性依赖于校准数据的质量和覆盖度，可能存在系统性偏差
- 未来可考虑将基准扩展到多学科、多语言场景，并探索通过增强训练数据来提升模型在这一任务上的表现

## 相关工作与启发

- **vs ChartQA/PlotQA**: 这些基准聚焦于结构化图表（柱状图、折线图等），本文关注更复杂的科学示意图，后者需要更多领域知识和空间推理能力
- **vs DocVQA**: DocVQA 针对文档图像的信息提取，本文专注于科学论文中示意图与论文上下文的联合理解，任务更具挑战性
- **vs SciGraphQA**: 该基准虽然也关注科学图像，但主要针对科学图表的数据读取，而非对方法流程图等示意图的深层理解

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个针对科学示意图理解的专门基准，填补了重要的评估空白
- 实验充分度: ⭐⭐⭐⭐ 评估了18个主流模型，涵盖多种信息检索场景，但缺少对模型改进方向的深入实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义明确，分析维度丰富
- 价值: ⭐⭐⭐⭐ 作为基准论文，对推动多模态科学文献理解领域的发展有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](../../NeurIPS2025/multimodal_vlm/what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ACL 2025\] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification](sciver_evaluating_foundation_models_for_multimodal_scientific_claim_verification.md)

</div>

<!-- RELATED:END -->
