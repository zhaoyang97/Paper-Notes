---
title: >-
  [论文解读] FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging
description: >-
  [ICCV 2025][多模态VLM][金融数值推理] 提出 FinMMR，一个双语（中英文）多模态金融数值推理基准，包含 4300 道题目和 8700 张图像，覆盖 14 个金融子领域，要求模型进行多步精确数值计算；评测了 15 个 SOTA MLLM，最好模型在 Hard 子集仅达 53% 准确率，揭示了当前 MLLM 在专业领域多模态推理中的核心瓶颈。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "金融数值推理"
  - "多模态基准"
  - "大语言模型评估"
  - "思维链推理"
  - "知识增强"
---

# FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging

**会议**: ICCV 2025  
**arXiv**: [2508.04625](https://arxiv.org/abs/2508.04625)  
**代码**: 无（提供在线评测平台）  
**领域**: 多模态视觉语言模型  
**关键词**: 金融数值推理, 多模态基准, 大语言模型评估, 思维链推理, 知识增强

## 一句话总结

提出 FinMMR，一个双语（中英文）多模态金融数值推理基准，包含 4300 道题目和 8700 张图像，覆盖 14 个金融子领域，要求模型进行多步精确数值计算；评测了 15 个 SOTA MLLM，最好模型在 Hard 子集仅达 53% 准确率，揭示了当前 MLLM 在专业领域多模态推理中的核心瓶颈。

## 研究背景与动机

大型推理模型（LRM）在代码、数学和科学推理上已取得显著进展，多模态大语言模型（MLLM）也在通用多模态推理上表现优异。然而在金融等高风险专业领域，MLLM 面临的挑战尚未被充分理解：
- 金融分析需要从视觉丰富的文档中提取关键指标，并进行多步精确数值计算
- 现有基准存在局限：FAMMA 来自教科书/考试题，MathVista 不涉及金融知识，MMMU 仅有选择题

FinMMR 的三大优势：
1. **多模态性**：所有表格、图表以图像形式呈现，包含干扰图像
2. **综合性**：覆盖 14 个金融子领域（公司金融、银行业、行业分析等）
3. **挑战性**：要求精确数值答案，消除选择题中的猜测偏差

## 方法详解

### 整体框架

FinMMR 的构建分为两条路径：
1. 从公开文本金融推理基准（MMMU、MMMU-Pro、FinanceMath、CodeTAT-QA、CodeFinQA、DocMath-Eval）中提取题目并转化为多模态形式
2. 从最新中国金融研究报告中全新构建 CRRQA（Chinese Research Report QA，2150 道）
两个来源合并为 FinMMR，每道题均配备可执行 Python 解决方案和精确数值答案。

### 关键设计

1. **多模态转换**：将表格数据渲染为图像，从文本中移除对应表格信息，确保 MLLM 不能仅依赖文本内容。核心创新是引入**干扰图像**——从同一报告中相邻位置选取语义相关但无关的图像，模拟真实场景。

2. **难度分级系统**：基于 Python 解题方案的复杂度指标进行启发式分级，考虑运算符数量 $o$、代码行数 $l$ 和括号对数 $p$：$rc = \ln(\max(o,1)) + \ln(\max(l+p,1))$。据此分为 Easy（1300）、Medium（1500）和 Hard（1500）三个级别。

3. **评估系统**：采用 CoT（思维链）、PoT（程序思维）和 IO（无提示）三种提示方法。PoT 生成 Python 代码并执行，严格数值评估的误差容限为 0.2%，要求精确到单位、百分比和小数位。

### 损失函数 / 训练策略

本文为基准测试，不涉及模型训练。评估策略的核心贡献包括：
- **视觉过滤推理流水线**：先让 MLLM 判断图像相关性，过滤干扰图后再推理
- **知识增强**：构建含 3133 个 Python 函数的金融函数库，通过 MLLM 引导的知识检索增强推理
- **模型协作**：用 GPT-4o 作为视觉解析器将图像转为结构化文本，再由 LRM 进行推理

## 实验关键数据

### 主实验

| 模型 | Extended Thinking | Hard (CoT) | Hard (PoT) | Medium (CoT) | Easy (CoT) | Avg (CoT) |
|---|---|---|---|---|---|---|
| Claude 3.7 Sonnet | ✔ (64K) | 53.00 | 51.40 | 62.50 | 78.50 | 64.00 |
| Claude 3.7 Sonnet | ✘ | 50.80 | 48.50 | 62.25 | 77.00 | 63.35 |
| OpenAI o1 | ✔ | 48.40 | 44.70 | — | — | — |
| GPT-4o | ✘ | 45.40 | 47.80 | 63.33 | 78.00 | 62.24 |
| Llama 4 Maverick | ✘ | 48.70 | 47.80 | 63.25 | 77.83 | 63.26 |
| Qwen2.5-VL-72B | ✘ | 43.30 | 46.20 | 63.42 | 77.42 | 61.38 |
| QVQ-72B-Preview | ✔ | 40.30 | 6.20 | 55.67 | 75.42 | 57.13 |

### 消融实验 / 知识增强效果

| 模型 | PoT 基线 | + 知识增强 (RAG) | 提升 |
|---|---|---|---|
| Gemini 2.0 Flash Thinking | 78.71 | 83.02 | +4.31 |
| GPT-4o | 80.60 | 83.62 | +3.02 |
| Claude 3.7 Sonnet | 81.21 | 85.43 | +4.22 |
| Claude 3.7 Sonnet (64K) | 83.53 | 86.29 | +2.76 |

（以上基于 1160 道表格 QA 实例）

**干扰图像的影响**（Qwen2.5-VL-72B，PoT）：

| 子集 | 标准图像 | 干扰图像 | 降幅 |
|---|---|---|---|
| Hard | 57.18% | 47.23% | ↓9.95 |
| Medium | 73.01% | 61.36% | ↓11.65 |
| Easy | 61.59% | 53.64% | ↓7.95 |

视觉过滤推理流水线在 Medium 验证集上将准确率从 64.73% 提升到 71.56%（+6.83）。

### 关键发现

1. **所有 MLLM 在 FinMMR 上均表现不佳**：最强模型 Claude 3.7 Sonnet (64K思考) 在 Hard 子集仅 53%，远低于 60% 的及格线
2. **PoT 优于 CoT**：PoT 平均准确率 37.64% vs CoT 36.20%，且 token 消耗更少。但 QVQ-72B 因强化学习偏差导致 PoT 仅 6.2%，暴露了训练策略问题
3. **干扰图像严重影响推理**：导致准确率下降超 10%，说明 MLLM 的视觉过滤能力不足
4. **错误分析（100个失败案例）**：视觉感知错误 30%、知识推理错误 38%、数值计算错误 32%
5. **模型协作有效**：GPT-4o 解析 + DeepSeek-R1 推理达 86.72%，优于单模型 Claude 3.7 Sonnet 的 83.53%

## 亮点与洞察

- **基准设计的实践导向**：引入干扰图像模拟真实场景中的信息过载，这是其他基准缺乏的
- **全面的错误归因**：将失败分解为视觉感知、知识推理、数值计算三类，为改进方向提供了清晰路线图
- **知识增强的有效性**：通过结构化金融函数库 + MLLM 引导检索 + MLLM 判断，弱模型可逼近 SOTA 性能
- **关于 Extended Thinking 的洞察**：提升有限（+2.2pp）但 token 消耗增加 12 倍，引出效率与效果的权衡问题

## 局限与展望

- 仅评估了 zero-shot 设置，未探索 few-shot 或微调场景
- CRRQA 部分依赖 Qwen-VL-Max 生成初始问题，可能引入模型偏差
- 金融函数库为手动构建，覆盖范围和可扩展性受限
- 未评估最新的 GPT-o3 等模型
- 干扰图像的构造方式（相邻图像）相对简单，更复杂的干扰模式值得探索

## 相关工作与启发

- 与 MathVista、MMMU 等通用基准相比，FinMMR 在领域深度和推理复杂度上具有显著优势
- 视觉过滤推理流水线（解耦感知与推理）的思路对其他领域也有参考价值
- 模型协作（视觉解析器 + 文本推理器）的框架为多模态系统设计提供了新思路

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个大规模多模态金融数值推理基准，干扰图像和三维错误分析设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 15个模型、3种提示方法、错误分析、视觉过滤、知识增强、模型协作等全面评估
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，RQ 驱动的实验组织方式便于理解
- **价值**: ⭐⭐⭐⭐⭐ 填补了金融领域多模态推理评测的空白，对 MLLM 改进具有明确指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[ACL 2025\] FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](../../ACL2025/multimodal_vlm/finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)
- [\[ACL 2025\] LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating](../../ACL2025/multimodal_vlm/longdocurl_multimodal_long_doc.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ACL 2025\] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](../../ACL2025/multimodal_vlm/fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)

</div>

<!-- RELATED:END -->
