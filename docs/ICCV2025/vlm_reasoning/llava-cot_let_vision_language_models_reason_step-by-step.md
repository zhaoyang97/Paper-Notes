---
title: >-
  [论文解读] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step
description: >-
  [ICCV 2025][多模态VLM][视觉语言模型] LLaVA-CoT 提出了一种让视觉语言模型自主进行多阶段结构化推理的方法——通过构建 LLaVA-CoT-100k 结构化推理标注数据集训练模型依次执行"总结→视觉解读→逻辑推理→结论生成"四个阶段，并提出阶段级回溯搜索（SWIRES）实现测试时缩放，使 11B 模型超越 Gemini-1.5-pro 和 GPT-4o-mini。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "分阶段推理"
  - "思维链"
  - "测试时缩放"
  - "结构化推理"
---

# LLaVA-CoT: Let Vision Language Models Reason Step-by-Step

**会议**: ICCV 2025  
**arXiv**: [2411.10440](https://arxiv.org/abs/2411.10440)  
**代码**: [https://github.com/PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)  
**领域**: 多模态VLM / LLM推理  
**关键词**: 视觉语言模型、分阶段推理、思维链、测试时缩放、结构化推理

## 一句话总结

LLaVA-CoT 提出了一种让视觉语言模型自主进行多阶段结构化推理的方法——通过构建 LLaVA-CoT-100k 结构化推理标注数据集训练模型依次执行"总结→视觉解读→逻辑推理→结论生成"四个阶段，并提出阶段级回溯搜索（SWIRES）实现测试时缩放，使 11B 模型超越 Gemini-1.5-pro 和 GPT-4o-mini。

## 研究背景与动机

**领域现状**：大语言模型通过推理时间缩放（如 OpenAI o1）展现了强大的推理能力。然而，当前的视觉语言模型（VLMs）在处理复杂的视觉问答任务时，仍难以进行系统化和结构化的推理。大多数 VLM 直接从问题跳到答案，缺乏中间推理步骤的显式结构化。

**现有痛点**：Chain-of-Thought（CoT）提示虽然能引导模型"一步一步思考"，但这种方式依赖外部提示工程，模型本身并没有内化系统推理的能力。具体来说：(1) 标准 CoT 的推理步骤是非结构化的自由文本，缺乏明确的阶段划分；(2) 模型无法自主决定何时进行视觉信息提取、何时进行逻辑推导；(3) 现有推理时缩放方法（如 beam search）作用于 token 级别，粒度太细，对长链推理效率低下。

**核心矛盾**：VLM 需要在"快速直觉回答"和"深度结构化推理"之间找到平衡。简单问题不需要深度推理，但复杂推理问题需要明确的认知阶段（先看懂题目、再理解图像、然后推理、最后得结论）。如何让模型自主掌握这种结构化推理流程是关键挑战。

**本文目标**：(1) 训练 VLM 自主进行多阶段结构化推理而非依赖外部提示；(2) 设计高效的推理时缩放方法利用多阶段结构。

**切入角度**：作者观察到人类解决复杂视觉推理问题时自然地经历"审题→看图→推理→结论"的认知过程。与其让模型输出自由格式的思维链，不如用结构化标签将这个过程显式编码，用标注数据教会模型这种推理范式。

**核心 idea**：构建 100k 结构化推理标注数据集让 VLM 学会自主分阶段推理（用 `<SUMMARY>` `<CAPTION>` `<REASONING>` `<CONCLUSION>` 标签），配合阶段级回溯搜索实现高效的测试时缩放。

## 方法详解

### 整体框架

LLaVA-CoT 基于 Llama-3.2-11B-Vision-Instruct 微调。训练数据是 LLaVA-CoT-100k 数据集，包含来自多个视觉问答数据集的样本，每个样本被标注为四阶段结构化推理链。模型被训练为给定图像和问题后，自主依次生成四个阶段的输出，每个阶段用特定标签包裹。推理时，可以直接让模型自主生成完整推理链（标准推理），也可以应用 SWIRES 方法在每个阶段进行多次采样和最优选择（测试时缩放）。

### 关键设计

1. **LLaVA-CoT-100k 结构化推理数据集**:

    - 功能：为 VLM 提供结构化推理的训练信号
    - 核心思路：从多个 VQA 数据源（如 ShareGPT4V、ChartQA、GeoQA+、A-OKVQA、ScienceQA 等）采集样本，利用 GPT-4o 为每个样本生成四阶段结构化标注。四个阶段分别是：(a) **Summary**（`<SUMMARY>`）——概括问题和解题策略；(b) **Caption**（`<CAPTION>`）——描述图像中与问题相关的视觉信息；(c) **Reasoning**（`<REASONING>`）——基于视觉信息进行多步逻辑推理；(d) **Conclusion**（`<CONCLUSION>`）——给出最终答案。所有标注使用结构化标签显式划分，使模型在训练时学到"何时该看图、何时该推理"的元认知能力。数据量仅 100k 但覆盖面广。
    - 设计动机：与传统 CoT 数据不同，这里的推理不是自由格式的，而是有明确阶段划分的。这种结构让模型学到的不仅是"怎么推理"，还有"推理流程怎么组织"。

2. **阶段级回溯搜索（Stage-Wise Retracing Search, SWIRES）**:

    - 功能：在推理时通过阶段级多次采样和回溯实现测试时缩放
    - 核心思路：在每个推理阶段（Summary / Caption / Reasoning / Conclusion），模型生成多个候选输出。然后利用一个评估策略（如自一致性或置信度评分）选择该阶段的最优候选，再将其作为下一阶段的输入继续生成。如果某阶段所有候选都不够好，可以回溯到前一阶段重新采样——这就是"retracing"。与标准 beam search 在 token 级搜索不同，SWIRES 在阶段级搜索，搜索空间大幅缩小（4 个阶段 vs 数百个 token），同时每次搜索单元的语义量更大、评估更可靠。
    - 设计动机：标准 beam search 对长序列生成效率极低，因为搜索空间呈指数爆炸。SWIRES 利用了推理的天然阶段结构，将搜索从 token 级提升到阶段级，在保持效果的同时大幅降低计算量。

3. **自主多阶段推理机制**:

    - 功能：使模型无需外部提示即可自发进行结构化推理
    - 核心思路：通过微调，模型内化了`<SUMMARY>→<CAPTION>→<REASONING>→<CONCLUSION>`的推理流程。在推理时，模型收到图像和问题后，自动首先生成 Summary 标签及其内容，标明解题计划；然后生成 Caption 标签描述视觉观察；接着生成 Reasoning 标签进行多步推理；最终生成 Conclusion 给出答案。这种固定的阶段顺序不是通过 prompt engineering 实现的，而是通过训练数据让模型自主习得的行为模式。
    - 设计动机：外部 CoT 提示不稳定且需要人工设计。将结构化推理内化为模型的自然行为，使得推理过程更可靠、可解释，且每个阶段的输出可以独立评估和改进。

### 损失函数 / 训练策略

使用 Llama-3.2-11B-Vision-Instruct 作为基座模型，在 LLaVA-CoT-100k 上进行全参数微调。训练使用 FSDP（Fully Sharded Data Parallel），8 卡并行，学习率 $10^{-5}$，3 个 epoch，batch size 4/卡。训练框架基于 Meta 的 llama-recipes。

## 实验关键数据

### 主实验

在六个多模态推理基准上的对比（准确率 %）：

| 模型 | MMStar | MMBench | MMVet | MathVista | AI2D | 平均 |
|------|--------|---------|-------|-----------|------|------|
| Llama-3.2-11B (base) | 49.8 | 65.8 | 57.6 | 48.6 | 77.0 | — |
| GPT-4o-mini | 54.8 | 76.9 | — | 52.4 | 77.8 | — |
| Gemini-1.5-pro | 57.6 | 73.9 | — | 57.7 | 79.1 | — |
| Llama-3.2-90B-Instruct | 56.2 | 78.3 | — | 58.3 | 78.9 | — |
| **LLaVA-CoT (11B)** | **57.6** | **73.8** | **60.8** | **54.8** | **85.0** | — |
| **LLaVA-CoT + SWIRES** | **59.2** | **75.1** | **62.3** | **57.2** | **86.4** | — |

LLaVA-CoT 仅 11B 参数，基于 100k 训练数据和 SWIRES 缩放，即超越了 90B 的 Llama-3.2 和闭源的 Gemini-1.5-pro、GPT-4o-mini。相比基座模型平均提升约 9.4%。

### 消融实验

| 配置 | MMStar | MathVista | AI2D | 说明 |
|------|--------|-----------|------|------|
| LLaVA-CoT (Full) | 57.6 | 54.8 | 85.0 | 完整四阶段推理 |
| w/o Summary 阶段 | 55.8 | 52.3 | 83.1 | 缺少审题规划 |
| w/o Caption 阶段 | 56.1 | 53.1 | 82.8 | 缺少视觉解读 |
| w/o Reasoning 阶段 | 50.2 | 47.5 | 78.2 | 缺少逻辑推理（退化最严重） |
| 直接回答（无CoT） | 49.8 | 48.6 | 77.0 | 基座模型水平 |
| Standard beam search | 58.0 | 55.5 | 85.8 | Token级搜索，计算量大 |
| SWIRES | 59.2 | 57.2 | 86.4 | 阶段级搜索，更高效更准确 |

### 关键发现

- Reasoning 阶段是最关键的，去除后性能退化最严重（回落到接近基座模型水平）
- Summary 和 Caption 阶段虽然改进幅度较小但不可或缺——它们为 Reasoning 提供了结构化的输入信息
- SWIRES 相比 standard beam search 在性能和效率上双赢：准确率更高（因为语义级评估更可靠），计算量更少（阶段级搜索空间远小于 token 级）
- 仅 100k 训练数据就能获得显著推理能力提升，说明结构化推理标注的数据效率极高
- 在 MathVista（数学推理）和 AI2D（图表理解）等需要深度推理的题目上提升最显著

## 亮点与洞察

- **结构化推理标签是一个简单但极具洞察力的设计**：仅通过四个标签将自由格式 CoT 组织为有明确职责的阶段。这种方法几乎不增加训练复杂度，但让模型学到了"元认知"能力——知道何时观察、何时推理、何时总结。该方法可以迁移到任何多步推理任务。
- **SWIRES 将搜索从 token 级提升到语义级**：这是对 inference-time scaling 方向的一个重要贡献。传统 beam search 在 token 级别搜索，粒度太细且搜索空间巨大。SWIRES 利用任务的天然阶段结构进行粗粒度搜索，思路简洁但效果突出。
- **100k 数据量的高效微调**：说明"数据质量 >> 数据量"——结构化的高质量推理标注比海量但无结构的数据更有效。这个经验对资源有限的研究者很有启发。

## 局限与展望

- 四阶段的固定顺序可能不适合所有任务——某些问题可能需要"看图→推理→再看图"的迭代过程
- SWIRES 的阶段级评估策略尚较简单（自一致性），更复杂的评估函数可能进一步提升效果
- 基座模型限于 Llama-3.2-11B-Vision，在更大模型或其他架构上的效果有待验证
- 训练数据依赖 GPT-4o 标注，存在标注质量上限和成本问题
- 未探索动态阶段数——某些简单问题可能只需两步，复杂问题可能需要六步
- 可以进一步结合 reward model 进行强化学习，从 SWIRES 的搜索结果中学习更好的推理策略

## 相关工作与启发

- **vs Chain-of-Thought (CoT)**：标准 CoT 通过 prompt 引导模型输出非结构化推理过程，模型本身没有内化推理结构。LLaVA-CoT 通过训练让模型自主产生结构化推理，四个阶段各司其职，更可控、可解释
- **vs OpenAI o1**：o1 展示了推理时间长思考的强大能力，但其内部机制不透明。LLaVA-CoT 提供了一种开源、可解释的结构化推理替代方案，且证明仅 11B 模型就能接近大模型的推理表现
- **vs Llama-3.2-90B-Vision**：8 倍小的模型通过结构化推理训练超越了 8 倍大的基座模型，充分说明"教会模型如何思考"比"堆叠模型参数"更高效

## 评分

- 新颖性: ⭐⭐⭐⭐ 结构化推理阶段和SWIRES搜索都是新颖的，但idea本身简洁清晰而非复杂
- 实验充分度: ⭐⭐⭐⭐ 六个基准全面对比，消融充分，但缺少更多模型规模的验证
- 写作质量: ⭐⭐⭐⭐⭐ 论文写作清晰易懂，demo展示说服力强，开源完善
- 价值: ⭐⭐⭐⭐⭐ 开源多模态推理方向的标杆工作，100k数据+11B模型即超越闭源大模型，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](../../NeurIPS2025/multimodal_vlm/can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)
- [\[NeurIPS 2025\] Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](../../NeurIPS2025/multimodal_vlm/unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)

</div>

<!-- RELATED:END -->
