---
title: >-
  [论文解读] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?
description: >-
  [ACL 2025][多模态VLM][LVLM-as-a-Judge] 系统评估了 13 个开源小型 LVLM（≤9B 参数）作为图表理解和推理任务的评判者，发现部分开源模型（如 LLaVA-Critic-7B）可达到接近 GPT-4 水平的评判能力（约 80% 一致率），但位置偏差和长度偏差等问题仍然普遍存在。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "LVLM-as-a-Judge"
  - "Chart Comprehension"
  - "Evaluation Benchmark"
  - "视觉语言"
  - "Bias Analysis"
---

# Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?

**会议**: ACL 2025  
**arXiv**: [2505.08468](https://arxiv.org/abs/2505.08468)  
**代码**: [https://github.com/tahmedge/chart_lvlm_judge](https://github.com/tahmedge/chart_lvlm_judge)  
**领域**: Multimodal & VLM  
**关键词**: LVLM-as-a-Judge, Chart Comprehension, Evaluation Benchmark, Vision-Language Model, Bias Analysis

## 一句话总结

系统评估了 13 个开源小型 LVLM（≤9B 参数）作为图表理解和推理任务的评判者，发现部分开源模型（如 LLaVA-Critic-7B）可达到接近 GPT-4 水平的评判能力（约 80% 一致率），但位置偏差和长度偏差等问题仍然普遍存在。

## 研究背景与动机

图表（chart）是数据可视化的核心载体，相关的下游任务（图表问答、图表描述等）近年来快速发展。大型视觉语言模型（LVLM）在这些任务上展现出潜力，但其质性评估面临几个关键瓶颈：

**人工评估成本高**：开放式回答的评估需要大量人力和时间，传统文本相似度指标（如 BLEU）又无法捕获回答质量。

**隐私与部署限制**：企业不愿将专有数据发送给 OpenAI/Google 等闭源模型；而兼容的开源模型（70B-400B）需要极高计算资源。

**缺乏专门评测**：此前没有系统研究小型开源 LVLM 能否有效评估图表相关任务。

核心研究问题：小型（≤10B 参数）开源 LVLM 能否以低成本替代 GPT-4 作为图表理解任务的自动评估者？

## 方法详解

### 整体框架

设计了一套标准化的"LVLM-as-a-Judge"评估框架，涵盖评判类型（pairwise/pointwise）× 参考类型（有参考/无参考）× 评估维度（事实准确性/信息量/相关性/多维度）的组合矩阵，总计生成约 10 万条由 GPT-4o 和 LLaVA-Critic-70B 产出的参考评判数据。

### 关键设计

1. **评估标准设计（Rubric Design）**: 定义了四个维度的评估标准。对于 pairwise 评估，评判者需在两个回答中选择更优者；对于 pointwise 评估，需在 1-5 的 Likert 量表上打分。每次评判要求附带解释（explanation），因为先前研究表明"解释+判断"模式能提升评判质量。这种设计确保了评估的多角度覆盖和可解释性。

2. **评估数据构建**: 使用三个数据集——OpenCQA（1.1k 开放式 QA 实例）、VisText（L1 结构描述 + L2/L3 洞察描述各 1.2k 实例），以及新提出的 Chart-Instruct-Eval（400 个指令跟随评估实例）。对于前两个数据集，收集了 Gemini-1.0-Pro 和 Claude-3-Haiku 的输出，使用 GPT-4o 和 LLaVA-Critic-70B 计算参考评判分数。Chart-Instruct-Eval 的动机是填补图表领域中指令跟随评估的空白——为每个样本手工准备了一个好/一个差的回答，好回答完全遵循指令，差回答忽视指令但内容相似。

3. **偏差分析框架**: 定义了位置偏差（交换两个回答顺序后评判是否改变）和长度偏差（错误选择是否与回答长度相关）两个指标。这是对评判公平性的系统检验，直接影响评估结果的可靠性。

### 评估指标体系

- Judgment Accuracy：pairwise 场景下评判者与参考答案的一致率
- Error Distance：pointwise 场景下评判者打分与参考打分的平均绝对差
- Positional Bias/Length Bias：衡量评判偏差的百分比
- Format Adherence：输出是否遵循 JSON 格式要求
- Instruction Following Evaluation Accuracy：是否能正确评估其他模型的指令跟随能力

## 实验关键数据

### 主实验（Pairwise 评判准确率，越高越好）

| 模型 | 参数量 | OpenCQA Avg | VisText L1 Avg | VisText L2/L3 Avg |
|------|--------|-------------|----------------|-------------------|
| LLaVA-Critic-7B | 7B | **79.5** | **79.1** | **77.1** |
| LLaVA-Next-Mistral-7B | 7B | 75.9 | 75.1 | 75.1 |
| XGen-MM-Phi3-3.8B | 3.8B | 71.6 | 75.4 | 70.7 |
| Qwen2-VL-7B | 7B | 66.9 | 57.6 | 70.0 |
| InternLM-Xcomposer-7B | 7B | 64.5 | 72.0 | 75.6 |
| PaliGemma-3B | 3B | 0.0 | 0.0 | 0.0 |
| ChartGemma-3B | 3B | 0.0 | 0.0 | 0.0 |
| Idefics-9B | 9B | 20.3 | 20.9 | 24.3 |

### Pointwise 评判（Error Distance，越低越好）

| 模型 | OpenCQA Avg | VisText L1 Avg | VisText L2/L3 Avg |
|------|-------------|----------------|-------------------|
| LLaVA-Critic-7B | **0.5** | **0.5** | **0.6** |
| Qwen2-VL-7B | 0.7 | 0.6 | 0.7 |
| InternLM-Xcomposer-7B | 0.9 | 0.9 | 0.7 |
| PaliGemma-3B | 5.0 | 5.0 | 5.0 |

### 偏差与指令跟随分析

| 模型 | 长度偏差 | 位置偏差 | 指令跟随评估 | 格式遵循 |
|------|----------|----------|-------------|---------|
| Qwen2-VL-7B | **21.5** | **35.8** | **87.0** | 98.6 |
| mPLUG-Owl3-7B | 21.9 | 42.5 | **93.5** | 98.9 |
| LLaVA-Critic-7B | 76.4 | 39.6 | 45.5 | **99.7** |
| LLaVA-Next-Mistral-7B | 71.8 | 77.0 | 27.0 | 98.9 |

### 关键发现

1. **LLaVA-Critic-7B 是最佳评判者**但最偏好长回答（长度偏差 76.4%）——准确性最高但公平性堪忧。
2. **模型大小不决定评判能力**：3.8B 的 XGen-MM 优于 9B 的 Idefics。PaliGemma/ChartGemma 完全失败是因为无法遵循评判指令格式。
3. **指令跟随评估是盲点**：在 pairwise/pointwise 中表现最好的 LLaVA-Critic 在指令跟随评估中仅 45.5%，而 mPLUG-Owl3 达到 93.5%。
4. **参考信息影响不大**：有参考 vs 无参考的评判准确率差异在统计上不显著（p>0.05）。

## 亮点与洞察

- **首次在图表领域系统评估 LVLM-as-a-Judge**：覆盖 13 个模型、3 个数据集、多种评估维度，评测方案设计严谨。
- **"准确但有偏"的悖论**：LLaVA-Critic 准确率最高但长度偏差最严重，提醒我们评判准确率和公平性需要分别考量。
- **Chart-Instruct-Eval 新基准**：填补了图表领域指令跟随评估的空白，揭示了大多数模型在这方面的薄弱。
- **人工评估验证**：两位标注者与 LLaVA-Critic-70B 的相关性高于 GPT-4o，佐证了开源模型作为替代标注者的可行性。

## 局限与展望

- 仅使用 GPT-4o 和 LLaVA-Critic-70B 作为参考评判标准，这些模型本身可能存在偏差。
- 未探索微调小型 LVLM 专门用于图表评判任务的可能性。
- 偏差分析较为表层，未深入分析偏差的根本原因和缓解策略。
- 测试的图表类型和复杂度有限，对更复杂的交互式图表或 3D 图表的评判能力未涉及。

## 相关工作与启发

- 与 Prometheus-VL、LLaVA-Critic 等通用多模态评估模型的工作方向一致，但专注于图表领域的垂直场景。
- 启发：在选择自动评估 judge 时，不仅要关注准确率，还需系统检查偏差；不同任务类型可能需要不同的最优 judge。
- 10 万条评判数据本身可作为训练数据，微调专门的图表评判模型。

## 评分

- **新颖性**: ⭐⭐⭐ — 方法论上属于系统性评测研究而非新方法提出，但在图表领域是首创。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 13 个模型、3 个数据集、多维度分析（准确性/偏差/指令跟随/格式遵循），非常全面。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，实验结果呈现直观，分析深入。
- **价值**: ⭐⭐⭐⭐ — 为图表领域的自动评估提供了实用指南，偏差分析对社区有警示意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Vision-Language Models Evaluate Handwritten Math?](can_vision-language_models_evaluate_handwritten_math.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs](chart-based_reasoning_transferring_capabilities_from_llms_to_vlms.md)
- [\[ACL 2025\] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](chartcoder_chart_to_code.md)

</div>

<!-- RELATED:END -->
