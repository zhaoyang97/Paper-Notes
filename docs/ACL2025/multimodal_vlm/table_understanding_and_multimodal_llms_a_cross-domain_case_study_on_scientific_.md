---
title: >-
  [论文解读] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables
description: >-
  [多模态VLM] 提出 TableEval 基准（3017 张表格，5 种格式），系统比较了文本 LLM 和多模态 LLM 在科学 vs. 非科学表格理解任务上的表现，发现模型对表格模态（图像/文本）保持鲁棒但在科学表格上性能显著下降。 表格是结构化数据展示中最常见的工具之一，广泛应用于金融、医学、教育和学术研究等领域…
tags:
  - "多模态VLM"
---

# Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables

- **会议**: ACL 2025
- **arXiv**: [2507.00152](https://arxiv.org/abs/2507.00152)
- **代码**: [esborisova/TableEval-Study](https://github.com/esborisova/TableEval-Study)
- **领域**: Multimodal VLM / 表格理解
- **关键词**: 表格理解, 多模态大模型, 跨领域评估, 科学表格, 可解释性分析

## 一句话总结

提出 TableEval 基准（3017 张表格，5 种格式），系统比较了文本 LLM 和多模态 LLM 在科学 vs. 非科学表格理解任务上的表现，发现模型对表格模态（图像/文本）保持鲁棒但在科学表格上性能显著下降。

## 研究背景与动机

表格是结构化数据展示中最常见的工具之一，广泛应用于金融、医学、教育和学术研究等领域。近年来 LLM 在各类 NLP 任务上展现了强大能力，但它们理解表格数据的能力仍然缺乏系统研究。现有工作主要集中以下不足：

**数据源偏向**: 大多数表格理解（Table Understanding, TU）研究只关注非科学来源的表格（如 Wikipedia、金融报告），而对科学文献中的表格关注极少。科学表格包含大量专业术语、复杂缩写和密集数值，需要领域知识和更强的算术推理能力。

**模态影响未知**: 表格可以用图片（PNG/JPEG）或多种文本格式（HTML、XML、LaTeX、Dict）表示，但不同表示模态对模型性能的影响尚不清楚。

**可解释性缺失**: 现有 TU 研究很少从可解释性角度分析模型如何利用表格中的上下文信息做决策。

本文旨在填补以上三个空白——构建一个跨领域、跨模态的表格理解基准，系统评估 LLM/MLLM 的性能，并通过梯度归因方法进行可解释性分析。

## 方法详解

### 整体框架

论文的研究分为三个主要阶段：

1. **构建 TableEval 数据集**: 从已有的科学和非科学数据集收集表格，统一转换为 5 种格式（Image、Dict、HTML、XML、LaTeX）。
2. **模型评估**: 在 TableEval 的每个子集上，使用不同表格表示分别评估各类 LLM 和 MLLM。
3. **可解释性分析**: 对模型输出施加梯度归因方法，生成 token 级别的重要性热图，分析模型如何利用输入中的表格内容。

### 关键设计一：TableEval 基准构建

TableEval 包含 3017 张表格和 11312 个评估实例，由 6 个子集组成：

**科学表格**（3 个子集）：
- **ComTQA (PubTables-1M)**: 来自 PubMed Central 论文的 VQA 任务，由 Gemini Pro 生成标注
- **numericNLG**: 来自 ACL Anthology，聚焦带数值推理的 Table-to-Text 生成
- **SciGen**: 来自 arXiv 和 ACL 论文，面向推理感知的文本生成任务

**非科学表格**（3 个子集）：
- **ComTQA (FinTabNet)**: 来自 S&P 500 企业年度报告的 VQA 任务
- **LogicNLG**: 来自 Wikipedia 的逻辑推理文本生成
- **Logic2Text**: 来自 Wikipedia 的逻辑形式到文本生成

每张表格被转换为 5 种格式：PNG 图片通过 PDFFigure2.0 或 imgkit 生成；HTML 和 XML 从原始来源提取或通过脚本互转；LaTeX 从论文源码提取或由 HTML 生成；Dict 为列表头和单元格值的线性化表示。

### 关键设计二：基于梯度的可解释性分析

论文采用 Inseq 工具包中的 **Input × Gradient** 方法进行特征归因分析。该方法的核心思想是：

$$\text{Attribution}(x_i) = x_i \cdot \frac{\partial y}{\partial x_i}$$

其中 $x_i$ 是输入 token 的嵌入，$y$ 是模型的输出 logit。将梯度与输入相乘后取平均，得到每个 token 对输出的贡献度，以热图形式可视化。

同时，论文还可视化了生成 token 的 **log-probability**，作为模型置信度的代理指标。例如，在 FinTabNet 的一个实例中，Mistral-Nemo 的注意力集中在与问题相关的年份列上并正确回答，而 Llama-3 的注意力更分散且在关键数字 token 上置信度极低。

### 关键设计三：跨模型评估体系

论文评估了 9 个模型配置（4 个 MLLM + 5 个 LLM 配置）：

- **MLLM**: Gemini-2.0-Flash（闭源基线）、LLaVa-NeXT-8B、Qwen2.5-VL-3B/7B、Idefics3-8B
- **LLM**: Llama-3.2-3B、Qwen2.5-3B/14B、Mistral-Nemo-12B

所有实验在 zero-shot 设定下进行，使用统一的 prompt 模板，不在 prompt 中标注文档格式类型，也不对模型输出做后处理。评估指标包括 BertScore.F1、MoverScore、ROUGE-L.F1、METEOR 等。

## 实验关键数据

### 表格一：图像 vs. 文本模态的性能对比（跨模型和数据集平均）

| 指标 | Image | Dict | HTML | XML | LaTeX |
|------|-------|------|------|-----|-------|
| BertScore.F1 | ~0.72 | ~0.69 | ~0.68 | ~0.68 | ~0.69 |
| MoverScore | ~0.28 | ~0.25 | ~0.24 | ~0.24 | ~0.25 |
| ROUGE-L.F1 | ~0.25 | ~0.22 | ~0.21 | ~0.21 | ~0.22 |
| METEOR | ~0.30 | ~0.27 | ~0.27 | ~0.27 | ~0.27 |

图像在所有指标上领先文本约 1-13%，四种文本格式之间差异极小（最大约 4%）。

### 表格二：科学 vs. 非科学表格的性能对比（跨模型和格式平均）

| 指标 | 科学表格 | 非科学表格 | 差距 |
|------|---------|-----------|------|
| BertScore.F1 | ~0.62 | ~0.78 | +16% |
| MoverScore | ~0.17 | ~0.35 | +18% |
| ROUGE-L.F1 | ~0.13 | ~0.33 | +20% |
| METEOR | ~0.17 | ~0.40 | +23% |

非科学表格在所有指标上大幅超过科学表格，最大差距可达 **34%**。其中 LogicNLG 取得最高分，SciGen 最低。

### 表格三：各模型平均预测长度统计

| 模型 | 平均长度 | 最小 | 最大 |
|------|---------|------|------|
| Idefics3-8B | 139 | 0 | 4416 |
| Gemini-2.0-Flash (image) | 207 | 2 | 3097 |
| Qwen2.5-VL-7B | 292 | 4 | 3464 |
| Mistral-Nemo | 303 | 21 | 2941 |
| Llama-3.2-3B | 464 | 22 | 5626 |
| Qwen2.5-14B | 481 | 29 | 4154 |

Idefics3 生成最简洁的回答，Qwen2.5-14B 和 Llama-3 生成的回答最长，响应长度差异是导致 n-gram 重叠类指标（如 BLEU、ROUGE）结果不同的重要因素。

## 关键发现

1. **模态鲁棒性**: 当前 LLM/MLLM 对不同文本格式的表格表示不敏感（HTML/XML/LaTeX/Dict 之间差异仅约 4%），图像模态略优于文本（1-13%），说明模型在预训练中已充分接触过这些格式。
2. **科学表格是瓶颈**: 模型在科学表格上性能显著下降（最多 34%），原因包括科学表格任务更复杂（需要多步推理、整表内容总结）以及预训练数据中科学表格可能较少。
3. **模型排名**: MLLM 中 Gemini-2.0-Flash 和 Idefics3 最优，LLM 中 Gemini-2.0-Flash 和 Mistral-Nemo 最优。开源模型仍落后于闭源 Gemini。模型参数量与准确度不相关——Qwen2.5-3B 与 14B 表现接近。
4. **可解释性洞察**: Input × Gradient 归因揭示了正确模型（Mistral-Nemo）将注意力集中在与答案相关的表格区域，而错误模型（Llama-3）注意力分散且在关键 token 上置信度低。tokenization 差异也影响性能——Llama-3 的三位数 token 切割导致年份数字处理困难。

## 亮点与洞察

- **跨域 + 跨模态的系统评估**: 首次同时比较科学/非科学表格和图像/多种文本格式，填补了多个研究空白。
- **TableEval 基准的实用价值**: 3017 张表格 × 5 种格式 × 2 类任务，公开在 HuggingFace 上，为后续表格理解研究提供了标准化评测平台。
- **可解释性与表格理解结合**: 将梯度归因方法应用于 TU 任务是一个有意义的新方向，能揭示模型在结构化数据上的决策模式。
- **格式无关性的发现对工程有参考价值**: 实际应用中不必过度纠结表格用哪种文本格式输入 LLM——HTML、XML、LaTeX、Dict 效果接近。

## 局限性

1. 所有模型使用相同 prompt 且不做输出后处理，可能导致分数偏低；针对模型定制 prompt 和结构化输出可能改善结果。
2. 仅依赖自动评估指标（BLEU、ROUGE 等），这些指标的局限性已有大量文献记录。
3. 可解释性分析仅针对文本输入，未覆盖多模态输入场景下各模态的贡献分析。
4. 数据集仅限英语，无法评估多语言表格理解能力。
5. 模型规模限制在 3-14B 参数，未评估更大的模型（如 70B+ 或 GPT-4）。

## 相关工作与启发

本文建立在表格理解、LLM 评估和可解释性三个方向的基础上。与之前仅关注 Wikipedia 表格的工作不同（如 TabFact、ToTTo、WikiTables），本文明确将科学表格纳入评测范围。Sui et al. (2024) 和 Singha et al. (2023) 研究了不同格式对 LLM 的影响但结论不一致，本文通过更大规模的实验给出了"格式差异不大"的结论。Deng et al. (2024) 比较了图像和文本输入，本文在更多模型和数据集上验证了类似发现。

**启发**: 科学表格理解仍是一个有价值的研究方向——可以考虑将科学领域的结构化数据加入预训练或做领域适配微调，来缩小科学与非科学表格的性能差距。

## 评分

⭐⭐⭐

中规中矩的 benchmark + 评估类工作。TableEval 基准本身有价值，跨域跨模态的评估视角覆盖面广。但方法层面没有提出新技术，本质是一个实验调研论文。可解释性分析仅做了少量实例的定性展示，深度有限。适合作为表格理解领域的参考基线和数据资源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](../../ICCV2025/multimodal_vlm/hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
