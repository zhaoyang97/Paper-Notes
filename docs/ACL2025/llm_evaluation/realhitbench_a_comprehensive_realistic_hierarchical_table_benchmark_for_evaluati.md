---
title: >-
  [论文解读] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis
description: >-
  [ACL2025][LLM评测][benchmark] 提出 RealHiTBench——首个全面评估 LLM 对复杂层次化表格理解能力的基准，包含 708 张来自 13 个平台、24 个领域的真实复杂表格和 3,752 道题目，定义了 5 种复杂结构类型和 5 大任务类型，并提出基于树结构的 TreeThinker 推理管线显著提升模型对层次化表头的理解能力。
tags:
  - "ACL2025"
  - "LLM评测"
  - "benchmark"
  - "hierarchical table"
  - "table reasoning"
  - "LLM evaluation"
  - "TreeThinker"
---

# RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis

**会议**: ACL2025  
**arXiv**: [2506.13405](https://arxiv.org/abs/2506.13405)  
**代码**: [cspzyy/RealHiTBench](https://github.com/cspzyy/RealHiTBench)  
**领域**: LLM评测  
**关键词**: benchmark, hierarchical table, table reasoning, LLM evaluation, TreeThinker

## 一句话总结

提出 RealHiTBench——首个全面评估 LLM 对复杂层次化表格理解能力的基准，包含 708 张来自 13 个平台、24 个领域的真实复杂表格和 3,752 道题目，定义了 5 种复杂结构类型和 5 大任务类型，并提出基于树结构的 TreeThinker 推理管线显著提升模型对层次化表头的理解能力。

## 研究背景与动机

**表格数据无处不在**：经济、科学、就业等领域广泛使用表格来组织多维关系数据，表格分析是 LLM 落地的重要场景。

**现有基准过于简单**：TAT-QA、TableBench、InfiAgent-DABench 等主流基准大量使用"扁平表格"（每列一个属性、每行一条记录），无法反映真实应用中的复杂层次结构。

**层次化表格被低估**：虽然 HiTab、SciTab 等考虑了层次表格，但它们要么领域单一（仅科学/航空）、要么输入格式单一（仅图像）、结构不够复杂（层次通常不超两层）、任务类型有限。

**HiTab 的固有缺陷**：HiTab 以有损 JSON 格式预提取树结构，无法真实评估 LLM 能否从原始表格输入中直接理解结构信息，且仅覆盖 3 个领域、任务类型单一、监督信号不完整。

**多模态评估缺失**：现有层次化表格基准要么只做文本要么只做图像，缺乏同时支持 LLM 和 MLLM、多种输入格式（LaTeX/HTML/PNG）的统一评估框架。

**缺乏结构理解专项测试**：现有基准未设计专门考察模型对复杂表格结构（如嵌套子表、隐式多表连接）理解能力的任务类型。

## 方法详解

### 整体框架

RealHiTBench 由两部分组成：（1）基准数据集——从 13 个公开平台收集 708 张真实复杂表格（涵盖 24 个领域），定义 5 种复杂结构类型和 5 大任务类型（含细分子类型），经 GPT-4o 自动标注 + 人工三轮检查生成 3,752 道题目，支持 LaTeX/HTML/PNG 多格式输入；（2）TreeThinker 推理管线——通过树结构组织层次化表头 → 关键词-表头对齐 → 子表定位 → React 风格多轮推理，增强 LLM 对复杂层次结构的理解。

### 关键设计一：5 种复杂表格结构类型定义

- **功能**：系统定义并分类 5 种真实场景中的复杂表格结构——（1）层次化列表头（多层合并单元格）、（2）层次化行表头（缩进或多列分级）、（3）嵌套子表（全宽单元格分割）、（4）多表连接（显式多表 + 隐式多表，即结构相同的子表看似单表实为多表对比）、（5）杂项（表外注释文字、单元格背景色等非结构化信息）。
- **核心思路**：从真实表格中归纳结构复杂性的本质来源，建立一套可量化的结构分类体系，使评估不再停留在"是否层次化"的二元判断上。
- **设计动机**：现有基准对表格复杂性缺乏细致定义，仅用简单的列层级描述无法捕捉嵌套子表、隐式多表等真实场景中的结构挑战；精确的分类体系才能实现对模型结构理解能力的细粒度评估。

### 关键设计二：Structure Comprehending 任务类型

- **功能**：在常规的事实验证（FC）、数值推理（NR）、数据分析（DA）、图表生成（CG）之外，新增"结构理解"（SC）任务——对源表的复杂部分进行交换/变换后生成新表，用相同问题分别询问两张表，评测 LLM 能否感知结构差异并给出正确答案。
- **核心思路**：通过控制变量（仅改变结构、不改变内容）来隔离测试模型对表格结构本身的理解，而非仅考察数据检索能力。
- **设计动机**：传统 TableQA 任务可能被"猜测"或"模式匹配"策略绕过，只有专门设计的结构对比题目才能真实暴露模型在层次化结构理解上的不足。

### 关键设计三：TreeThinker 树结构推理管线

- **功能**：提出三阶段管线——（1）Tree Generation：提示模型将表头编码为元组 $(flag, start, end, content)$ 并组织成树结构；（2）Tree-based Reasoning：将问题分解为关键词，与表头树对齐以定位相关子表；（3）React-Style Refinement：通过多轮 Thought-Action-Result 循环精化答案。
- **核心思路**：用树结构显式表示表头的父子层级关系，消除 LLM 在扁平化文本输入中对层次信息的感知盲区，再通过关键词对齐减少无关信息干扰。
- **设计动机**：实验表明 LLM 在复杂层次表格上表现很差，根因在于模型无法从线性化的表格文本中自动推断层级结构；TreeThinker 通过显式结构注入弥补了这一短板。

## 实验关键数据

### 主实验：25 个模型在 RealHiTBench 上的表现

| 模型 | 输入 | FC-EM | NR-F1 | SC-F1 | DA-GPT | CG-PASS@1 |
|------|------|-------|-------|-------|--------|-----------|
| DeepSeek-R1 | Text | **70.91** | **72.54** | **84.62** | **42.59** | 7.14 |
| GPT-4o | Text | 60.31 | 50.12 | 71.14 | 36.36 | 20.13 |
| GPT-4o+TreeThinker | Text | 64.50 | 65.08 | 75.67 | 37.63 | **39.47** |
| Gemini-1.5-Pro | Text | 59.08 | 43.74 | 69.71 | 36.17 | 9.74 |
| Llama3.3-70B | Text | 53.08 | 48.99 | 68.93 | 27.98 | 24.03 |
| Qwen2.5-72B | Text | 51.93 | 39.23 | 68.34 | 35.90 | 14.29 |
| TableGPT2-7B | Text | 46.10 | 39.81 | 56.68 | 32.47 | 67.53 |
| GPT-4o | Image | 43.39 | 36.68 | 52.89 | 33.10 | 10.39 |

### 消融实验：TreeThinker 各组件贡献

| 配置 | GPT-4o Avg | Δ | Llama3-70B Avg | Δ |
|------|-----------|---|----------------|---|
| TreeThinker (完整) | 63.29 | — | 62.23 | — |
| 去掉 Tree Generation | 55.27 | -8.02 | 54.04 | -8.19 |
| 去掉 Tree-based Reasoning | 60.75 | -2.54 | 54.58 | -7.65 |

### 关键发现

1. **整体表现低**：几乎所有模型的 EM 指标未超过 70，图表生成 PASS@1 普遍低于 30，真实层次化表格仍是 LLM 的重大挑战。
2. **文本优于图像**：GPT-4o 文本输入比图像输入平均高 ~15 分，Gemini-Pro 高 ~10 分；但图像可作为文本的补充（Image+Text 效果最佳）。
3. **DeepSeek-R1 一骑绝尘**：在 FC、NR、SC 三项上均大幅领先，表明强推理能力对层次化表格理解有显著帮助。
4. **TreeThinker 显著提效**：GPT-4o 在图表生成上 PASS@1 从 14.29 提升到 33.55（+134.7%），NR-F1 从 36.68 提升到 49.35；Tree Generation 是最关键组件（去掉后下降 ~8 分）。
5. **表格过长仍是瓶颈**：>20K token 的表格上 GPT-4o 分数从 56.45 骤降至 30.77，视觉模态影响更严重。
6. **表格专用模型过拟合**：TableLlama 等表格微调模型在复杂层次表格上表现反而不如通用 LLM。

## 亮点与洞察

1. **全面性**：同时覆盖 5 种复杂结构、5 大任务类型、文本+图像双模态、13 个数据源 24 个领域，是层次化表格理解领域最全面的基准。
2. **Structure Comprehending 设计巧妙**：通过结构变换后的对比测试，精准隔离了模型的结构理解能力，这一评测范式可推广到其他结构化数据领域。
3. **TreeThinker 方法新颖实用**：用树结构显式编码表头层次并与问题关键词对齐的思路简洁有效，不依赖额外训练即可大幅提升性能。
4. **隐式多表连接的发现**：首次识别并定义了"看似单表实为多表"的隐式连接类型，揭示了一个此前被忽视的理解难点。

## 局限性

1. **数据规模受限**：708 张表格、3,752 道题目相对不大，6 名标注员各投入 540 小时，扩展成本高。
2. **TreeThinker 效率问题**：多轮提示策略显著增加推理成本，存在性能-效率的权衡。
3. **长表格处理未解决**：127 张超长表格无法在单次对话中完整输入，论文仅提出问题未给出解决方案。
4. **标注潜在偏差**：大学生标注员虽经过培训，但在高度专业化领域（如金融、科学）的表格标注中可能引入误差。

## 相关工作与启发

| 对比维度 | RealHiTBench | HiTab | TableBench |
|----------|-------------|-------|------------|
| 表格复杂度 | 高（5种复杂结构，中位数大于其他基准） | 中（预提取JSON树结构，仅2层层级） | 低（以扁平表格为主） |
| 任务类型 | FC/NR/DA/CG/SC（含结构理解） | 仅基础QA | FC/NR/DA/CG（无结构理解） |
| 输入格式 | LaTeX/HTML/PNG（文本+图像） | 有损JSON | 仅文本 |
| 领域覆盖 | 13平台24领域 | 3个领域 | 6平台18领域 |
| 模型评估 | 25模型（LLM+MLLM+表格专用） | 少量模型 | 多模型但仅文本 |

**vs MultiHierTT**：MultiHierTT 虽考虑层次表格但仅做数值推理，来源单一（仅上市公司报告）；RealHiTBench 在任务多样性、领域覆盖和结构复杂度上全面超越。

**vs TableVQA-Bench**：TableVQA 虽支持图像输入但表格结构简单（低层次），且不支持文本输入对比；RealHiTBench 同时支持双模态并专注复杂结构。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个系统定义复杂层次化表格结构并设计 Structure Comprehending 任务的基准，TreeThinker 树推理思路新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 25个模型、多模态多格式、消融完整、长表格分析等多维度评估非常充分
- **写作质量**: ⭐⭐⭐⭐ — 复杂结构定义清晰，图示丰富，TreeThinker 流程描述详细
- **价值**: ⭐⭐⭐⭐ — 为表格理解研究提供了急需的高质量挑战性基准，TreeThinker 具有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](../../ACL2026/llm_evaluation/arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
