---
title: >-
  [论文解读] Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs
description: >-
  [ACL 2025][多模态VLM][图表问答] 本文提出一种将LLM的推理能力迁移到VLM的方法，通过改进图表表示预训练、构造大规模合成推理数据集和多任务微调，使5B参数的PaLI-3在ChartQA上超越10倍大的模型。 领域现状：视觉语言模型（VLM）在多模态任务上取得了越来越好的表现，但其推理能力——特别是涉及数值计…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "图表问答"
  - "VLM推理"
  - "知识蒸馏"
  - "ChartQA"
  - "推理增强"
---

# Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs

**会议**: ACL 2025  
**arXiv**: [2403.12596](https://arxiv.org/abs/2403.12596)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 图表问答, VLM推理, 知识蒸馏, ChartQA, 推理增强

## 一句话总结
本文提出一种将LLM的推理能力迁移到VLM的方法，通过改进图表表示预训练、构造大规模合成推理数据集和多任务微调，使5B参数的PaLI-3在ChartQA上超越10倍大的模型。

## 研究背景与动机

**领域现状**：视觉语言模型（VLM）在多模态任务上取得了越来越好的表现，但其推理能力——特别是涉及数值计算和多步逻辑的推理——仍然有限。相比之下，大语言模型（LLM）的推理能力通过Chain-of-Thought等技术已有大幅提升。图表问答（Chart QA）是一个典型的需要复杂推理的多模态任务：需要从视觉中提取信息，再进行数值运算或逻辑推理。

**现有痛点**：小型VLM（如PaLI-3 5B）在ChartQA等任务上的推理能力明显弱于大型模型（如PaLI-X 55B），但大型模型的推理成本高昂。Chen et al. (2023c)指出PaLI-3在ChartQA上落后于PaLI-X，可能是由于推理能力不足。同时，现有VLM的训练流程中缺乏针对图表理解的专门预训练任务。

**核心矛盾**：小型VLM的表示能力可能足够理解图表视觉元素，但缺乏多步推理能力；而LLM具备强大的推理能力却无法直接处理视觉输入。如何将两者的优势结合起来？

**本文目标**：设计一套高效的训练方案（recipe），将LLM的推理能力迁移到小型VLM中，使其在图表问答任务上达到或超越大型模型的性能。

**切入角度**：将图表转换为结构化表格作为LLM和VLM之间的桥梁——LLM在表格上生成推理轨迹（reasoning traces），然后用这些轨迹训练VLM。

**核心 idea**：通过"改进预训练+合成推理数据+多任务微调"的三步方案，让小型VLM继承LLM的推理能力。

## 方法详解

### 整体框架
整体pipeline包含三个阶段：（1）继续预训练阶段，使用改进的chart-to-table翻译任务增强VLM对图表的内部表示；（2）合成数据构造阶段，利用LLM基于图表的表格表示生成推理轨迹，构造比原始数据集大20倍的训练集；（3）多任务微调阶段，使用Hsieh et al. (2023)的多任务框架同时训练答案生成和推理过程生成。

### 关键设计

1. **图表表示预训练（Chart Representation Pre-training）**:

    - 功能：提升VLM对图表视觉元素（颜色、线条、位置）与文本内容（图例、单位）之间关联的理解
    - 核心思路：在PaLI-3的预训练阶段追加chart-to-table翻译任务——给定图表图像，模型需要输出对应的结构化表格文本。相比Liu et al. (2023a)的原始版本，改进了表格格式的标准化和错误处理，提升了训练数据质量
    - 设计动机：显式的chart-to-table翻译任务迫使模型学习图表视觉元素到结构化数据的精确映射，为后续的推理提供更好的内部表示

2. **大规模合成推理数据（Synthetic Reasoning Traces）**:

    - 功能：构造包含中间推理步骤的训练数据，用于教会VLM进行多步推理
    - 核心思路：先将图表转换为表格，然后将表格和问题输入LLM（如PaLM-2），让LLM生成详细的推理轨迹（包括信息提取、数值计算、逻辑推断的每一步）。通过这种方式将原始ChartQA训练集扩展了约20倍。推理轨迹以自然语言形式描述每个推理步骤，如"首先从表格中找到X列对应Y行的值为Z，然后计算..."
    - 设计动机：VLM直接学习从图表到答案的映射缺少中间推理监督，合成的推理轨迹提供了显式的推理路径，相当于"教模型怎么想"而非"只告诉答案"

3. **多任务微调框架（Multi-task Fine-tuning）**:

    - 功能：同时训练答案预测和推理过程生成两个任务
    - 核心思路：基于Hsieh et al. (2023)的框架，为每个训练样本设置两个目标——直接输出答案和输出推理过程+答案。两个任务共享模型参数，使用多任务损失 $L = L_{answer} + \lambda L_{rationale}$ 联合训练。推理时模型可以只输出答案（保持与PaLI-3相同的推理速度），也可以先输出推理过程再给出答案
    - 设计动机：相比先蒸馏再预测的串行方法，多任务并行训练更高效，且推理过程作为辅助任务可以正则化答案预测，避免过拟合

### 损失函数 / 训练策略
使用多任务交叉熵损失，答案预测和推理过程生成两个任务按固定权重组合。此外还可以选择性地在推理阶段使用program-of-thought提示来进一步精化数值推理结果。

## 实验关键数据

### 主实验

| 模型 | 参数量 | ChartQA-Human | ChartQA-Aug | PlotQA | FigureQA |
|------|--------|---------------|-------------|--------|----------|
| PaLI-3-5B | 5B | 33.8 | 65.3 | - | - |
| PaLI-X-55B | 55B | 57.6 | 79.9 | - | - |
| Gemini Ultra | - | 63.3 | 80.8 | - | - |
| GPT-4V | - | 60.3 | 78.1 | - | - |
| **ChartPaLI-5B** | 5B | **64.2** | **82.5** | **73.1** | **63.2** |
| ChartPaLI-5B + PoT | 5B | **66.7** | **83.1** | - | - |

### 消融实验

| 配置 | ChartQA-Human | ChartQA-Aug | 说明 |
|------|---------------|-------------|------|
| ChartPaLI完整 | 64.2 | 82.5 | 全部组件 |
| w/o 预训练续训 | 56.8 | 76.3 | 图表表示预训练贡献+7.4 |
| w/o 合成推理数据 | 52.1 | 71.8 | 合成数据贡献最大（+12.1） |
| w/o 多任务框架（只用答案） | 60.3 | 79.6 | 多任务提升+3.9 |
| 原始数据量（1x） | 58.5 | 78.2 | 20x数据量提升+5.7 |

### 关键发现
- 合成推理数据贡献最大（+12.1分），验证了"教模型怎么推理"比"只给答案"有效得多
- 图表预训练续训贡献显著（+7.4分），说明良好的图表内部表示是推理的前提
- 5B参数的ChartPaLI超越了55B的PaLI-X和Gemini Ultra等模型，参数效率提升11倍
- Program-of-thought在数值计算密集的问题上额外提升2.5分，但对定性问题帮助不大

## 亮点与洞察
- 将LLM推理能力迁移到VLM的方法非常巧妙：利用表格作为模态间的桥梁，让LLM在纯文本域生成推理轨迹，再用这些轨迹训练VLM。这种思路可以推广到其他需要推理的多模态任务
- 多任务框架比标准蒸馏更高效，因为推理过程和答案共享底层表示，互相增强
- 20x的数据增强策略说明在训练数据充足的情况下，小模型完全可以达到大模型的水平

## 局限与展望
- 方法依赖PaLI-3和PaLM-2的特定架构，对开源VLM（如LLaVA）的适用性待验证
- chart-to-table的中间步骤引入了信息损失，对复杂图表（如多Y轴、嵌套图表）可能不适用
- 合成推理轨迹的质量上限受限于教师LLM的能力，错误的推理轨迹可能误导学生模型
- 可以探索端到端的推理蒸馏方法，去除显式的chart-to-table中间步骤

## 相关工作与启发
- **vs MatCha (Liu et al., 2023)**: MatCha使用math reasoning预训练增强图表理解，本文进一步引入合成推理轨迹和多任务框架
- **vs DePlot (Liu et al., 2023)**: DePlot专注于chart-to-table翻译，本文在此基础上增加了推理能力迁移
- **vs Hsieh et al. (2023) Distilling Step-by-Step**: 本文首次将其多任务蒸馏框架应用于多模态任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 推理迁移的pipeline设计新颖，但核心组件多为已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多对比、详细消融
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，实验部分特别充实
- 价值: ⭐⭐⭐⭐ 对图表理解和小模型推理增强都有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings](hidden_in_plain_sight_evaluation_of_the_deception_detection_capabilities_of_llms.md)
- [\[CVPR 2026\] VisRes Bench: On Evaluating the Visual Reasoning Capabilities of VLMs](../../CVPR2026/multimodal_vlm/visres_bench_on_evaluating_the_visual_reasoning_capabilities_of_vlms.md)
- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](transferring_textual_preferences_to_vision-language_understanding_through_model_.md)
- [\[ACL 2025\] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?](judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](../../ACL2026/multimodal_vlm/shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)

</div>

<!-- RELATED:END -->
