---
title: >-
  [论文解读] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts
description: >-
  [ACL 2025][多模态VLM][多模态问答] 提出 WikiMixQA 基准，包含 1,000 道需要跨表格和图表进行多模态推理的多选题，评估 12 个 VLLM 后发现闭源模型在提供精确上下文时准确率约 70%，但需从长文档检索时性能骤降，开源模型最高仅 27%，揭示了当前视觉语言模型在长上下文多模态文档理解上的严重不足。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态问答"
  - "文档理解"
  - "表格"
  - "图表"
  - "视觉语言模型"
  - "长上下文"
  - "跨模态推理"
---

# WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts

**会议**: ACL 2025  
**arXiv**: [2506.15594](https://arxiv.org/abs/2506.15594)  
**代码**: [GitHub](https://github.com/negar-foroutan/WikiMixQA)  
**领域**: 多模态VLM  
**关键词**: 多模态问答, 文档理解, 表格, 图表, 视觉语言模型, 长上下文, 跨模态推理

## 一句话总结

提出 WikiMixQA 基准，包含 1,000 道需要跨表格和图表进行多模态推理的多选题，评估 12 个 VLLM 后发现闭源模型在提供精确上下文时准确率约 70%，但需从长文档检索时性能骤降，开源模型最高仅 27%，揭示了当前视觉语言模型在长上下文多模态文档理解上的严重不足。

## 研究背景与动机

文档理解（Document Understanding）是 NLP 的核心任务之一。真实文档不仅包含纯文本，还大量嵌入表格、图表等结构化/视觉化元素，使自动理解变得困难。视觉语言大模型（VLLM）在诸多任务上展现了能力，但在以下方面仍存在显著缺陷：

- **单页局限**：现有 VQA 基准多聚焦于单页文档，无法测试跨页推理能力
- **缺乏跨模态推理**：大多数数据集的问题只涉及单一模态（纯文本或单个图表），不要求模型综合多种模态信息
- **评估粒度不足**：现有数据集缺少对"回答问题所需模态类型"的精确控制，难以细粒度分析模型短板
- **长上下文处理薄弱**：模型在给定精确信息时表现尚可，但需从长文档中定位并提取分散信息时性能大幅下降

WikiMixQA 正是为填补这一空白而设计：以 Wikipedia 为文档来源，构建需要跨表格-表格、图表-图表、表格-图表进行推理的多选问答基准。

## 方法详解

### 1. 数据构建流程（四阶段管线）

**(1) 文档收集**：从 2022 年 3 月 Wikipedia 英文转储（400 万+条目）出发，筛选含 ≥3 个表格的文章（392,223 篇），再下载超百万张图片，用微调的 ViT 模型过滤非图表图片，保留含有效图表的文章（15,164 篇）。利用 Wikipedia 的"Instance of"属性将文档归入 Economy、Geography、History、Politics、Science、Sport、Wikimedia 七大类别，最终保留 7,258 篇文档。

**(2) 模态对选择**：为保证问题质量和多样性，不随机配对而是基于语义相似度选择模态对。对缺少标题的表格，用 Llama-3-8B-Instruct 生成描述；对图表图片，用 GPT-4-turbo 确认是否为图表并提取关键信息。然后使用 BAAI/bge-reranker-v2-m3 跨编码器计算所有模态对的相似度得分，筛选出语义相关的 table-table、chart-chart、table-chart 配对。

**(3) 问题生成**：保留相似度在宏观均值到 0.9 之间的配对，排除字符数 <512 的表格。每个配对生成三类多选题（分别基于两个模态和综合两者），每题包含 4 个选项、正确答案和解释，共用 GPT-4-turbo 生成 3,528 道题。之后用 InternVL2-Llama3-76B 进行两阶段质量筛选：先判断模态信息是否足以回答问题，再验证 GPT-4 给出的答案是否正确。

**(4) 人工标注**：从 3,528 题中选取 2,001 题（含 938 道 AI 筛选通过的和 1,063 道随机采样的），由三名计算机科学硕士生标注。标注分两步——首先判断问题是否必须整合两种模态才能回答（有效性检查），然后评估答案正确性（Correct / Wrong / Small Edit）。通过多数投票，595 题标为 Correct，405 题经修订后加入，最终形成 1,000 道题的基准。

### 2. 三种评估设置

| 设置 | 提供的上下文 | 测试能力 |
|------|-------------|---------|
| **Blind** | 无任何上下文 | 模型内部知识和推理 |
| **Oracle** | 精确的相关表格/图表 | 结构化数据解读与推理 |
| **Wikidoc** | Wikipedia 页面快照 | 长上下文中检索+推理 |

### 3. 数据集特征

- **规模**：1,000 道 MCQ，来自 526 篇唯一 Wikipedia 文档（约 4,000 页）
- **文档平均长度**：24.18 页，1,815 ± 2,825 tokens
- **七大主题**：Economy、Geography、History、Politics、Science、Sport、Wikimedia
- **三种模态组合**：约 50% 为 table-table，其余为 chart-chart 和 table-chart
- **质量保障**：AI 预筛 + 人工两轮标注，确保问题必须跨模态才能回答

## 实验关键数据

### 表1：三种评估设置下模型准确率（%）

| 模型 | Blind | Oracle | Wikidoc |
|------|-------|--------|---------|
| GPT-4o | 33.46 | **71.42** | **55.24** |
| Gemini-2.0-pro | 22.67 | 69.53 | 23.47 |
| Gemini-2.0-flash | 23.27 | 67.52 | 24.47 |
| Claude-3.5-Sonnet | 11.28 | 70.82 | 35.56 |
| InternVL2.5-78B | 3.29 | 27.67 | — |
| Qwen2.5-VL-72B | 0.39 | 23.17 | — |
| Llama-3.2-11B | 10.68 | 14.08 | — |
| 人类专家 | — | 87.50 | — |

**关键发现**：Oracle 设置下闭源模型约 70%，但 Wikidoc 设置下仅 GPT-4o 超过 50%（55.24%），其余闭源模型接近随机水平。开源模型在 Oracle 设置下最高仅 27.67%（InternVL2.5-78B），严重落后于人类的 87.50%。

### 表2：Oracle 设置下按问题类型（模态组合）细分

| 模型 | 2 Charts | 2 Tables | 1 Chart + 1 Table |
|------|----------|----------|-------------------|
| GPT-4o | 71.31 | 71.63 | 71.15 |
| Gemini-2.0-pro | 54.65 | **77.43** | 69.61 |
| Claude-3.5-Sonnet | 66.66 | 73.29 | 70.38 |
| InternVL2.5-78B | 24.41 | 30.02 | 26.53 |
| Qwen2.5-VL-72B | 22.48 | 24.22 | 21.92 |

**关键发现**：GPT-4o 在三种类型上表现均衡（~71%）；Gemini-2.0-pro 在双表格问题上表现最佳（77.43%），但在双图表问题上仅 54.65%，差距显著。图表解读对所有模型而言比表格理解更具挑战性。

### 表3：Oracle 设置下按主题细分（闭源模型）

| 模型 | History | Politics | Geography | Sports | Science | Economy | Wikimedia |
|------|---------|----------|-----------|--------|---------|---------|-----------|
| GPT-4o | 74.12 | 68.61 | 76.32 | 75.96 | 72.94 | 58.25 | 72.48 |
| Claude-3.5-Sonnet | 67.74 | **76.68** | 69.47 | 71.15 | 68.24 | 61.17 | 74.50 |
| Gemini-2.0-pro | 75.81 | 71.75 | 72.11 | 69.23 | 69.41 | 52.43 | 72.48 |

**关键发现**：Economy 主题对所有模型最具挑战（GPT-4o 仅 58.25%），原因在于经济类问题大量涉及柱状图和折线图的比较分析。

## 核心发现

1. **长上下文严重退化**：从 Oracle 到 Wikidoc，GPT-4o 准确率从 71.42% 降至 55.24%（降 16 个百分点），其余闭源模型退化更严重（Gemini-2.0-pro 从 69.53% 降至 23.47%），表明从长文档中定位相关多模态信息是当前 VLLM 的关键瓶颈
2. **开闭源鸿沟**：开源模型在 Oracle 设置下最高仅 27.67%，接近随机水平（25%），与闭源模型差距约 43 个百分点
3. **人机差距显著**：人类专家在 Oracle 设置下达 87.50%，领先最佳模型 16 个百分点，说明跨模态推理对当前模型仍然困难
4. **图表比表格更难**：双图表问题普遍难于双表格问题，可能因为图表需要视觉感知+数值推理的双重能力
5. **Blind 设置验证有效性**：所有模型在 Blind 设置下均低于或接近随机水平，证明 WikiMixQA 的问题确实需要上下文信息，无法靠"猜测"或预训练知识解答

## 亮点

- **系统性管线**：从文档收集、模态配对、AI 生成到人工标注的四阶段构建流程有条理，质量控制严谨
- **跨模态推理设计**：每道题必须综合两种模态（table-table / chart-chart / table-chart）才能回答，填补了现有基准的空白
- **三种评估粒度**：Blind / Oracle / Wikidoc 三种设置精准隔离了模型在知识、推理和检索上的能力
- **广泛模型覆盖**：评估了 12 个 VLLM（4 开源 + 8 闭源），提供了全面的性能画像
- **与人类对比**：引入人类专家基线（87.50%），量化了模型与人类的差距

## 局限与展望

1. **数据源单一**：仅使用 Wikipedia，文档风格和内容分布有限，未覆盖科学论文、财务报告等专业文档
2. **推理深度有限**：当前问题主要需要跨两个模态的单跳或简单多跳推理，缺少更复杂的多步推理链
3. **长上下文评估方式局限**：Wikidoc 设置使用页面截图作为输入，未结合文本表征，可能低估了模型在文本+视觉混合输入下的能力
4. **开源模型评估不完整**：由于计算限制，Wikidoc 设置仅在闭源模型上评估
5. **图表类型定义宽泛**：将地图、和弦图等都归为"chart"，可能引入不同难度的任务混淆
6. **主题覆盖仍可扩展**：七大类别中 Economy 的文档数偏少（80 篇），代表性不足

## 与相关工作的对比

| 基准 | 文档来源 | 题目数 | 跨页推理 | 平均 Tokens | 证据类型 |
|------|---------|--------|---------|------------|---------|
| MMLongBench-Doc | 多源 | 1k | ✓ | 21,214 | 表格/图表/地图 |
| DUDE | 多源 | 41k | ✓ | 1,832 | 表格/图表/地图 |
| MP-DocVQA | 工业文档 | 50k | ✗ | 2,027 | 表格/图表 |
| InfographicsVQA | 信息图 | 30k | ✗ | 288 | 表格/图表/地图 |
| TAT-DQA | 财务报告 | 16k | ✗ | 577 | 表格 |
| **WikiMixQA** | **Wikipedia** | **1k** | **✓** | **1,815** | **表格/图表/地图** |

WikiMixQA 的独特价值在于：(1) 每道题严格要求跨模态推理；(2) 控制了所需模态类型，支持细粒度分析；(3) 提供 Blind/Oracle/Wikidoc 三级评估；(4) 覆盖七大主题，主题多样性优于专业领域基准。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次构建严格要求跨表格/图表推理的文档理解基准，设计合理
- 实验充分度: ⭐⭐⭐⭐ — 12 个模型、三种设置、按主题/模态类型细分，分析全面
- 写作质量: ⭐⭐⭐⭐ — 构建管线描述清晰，图表丰富，附录详实
- 价值: ⭐⭐⭐⭐ — 揭示了 VLLM 在长上下文多模态推理中的关键短板，对社区有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[AAAI 2026\] Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts](../../AAAI2026/multimodal_vlm/format_matters_the_robustness_of_multimodal_llms_in_reviewing_evidence_from_tabl.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)

</div>

<!-- RELATED:END -->
