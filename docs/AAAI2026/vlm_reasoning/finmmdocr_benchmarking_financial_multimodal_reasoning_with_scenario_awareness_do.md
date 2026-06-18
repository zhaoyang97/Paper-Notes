---
title: >-
  [论文解读] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation
description: >-
  [AAAI 2026][多模态VLM][金融推理] 本文提出FinMMDocR，一个面向真实金融场景的双语多模态推理基准，包含1200道专家标注的数值推理题目，涵盖12类隐式金融情景、9类长文档（平均50.8页）和平均11步推理链，最强MLLM (o4-mini-high) 仅达58%准确率，揭示现有模型在复杂金融推理中的严重不足。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "金融推理"
  - "多模态基准"
  - "文档理解"
  - "多步计算"
  - "RAG"
---

# FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation

**会议**: AAAI 2026  
**arXiv**: [2512.24903](https://arxiv.org/abs/2512.24903)  
**代码**: [https://bupt-reasoning-lab.github.io/FinMMDocR](https://bupt-reasoning-lab.github.io/FinMMDocR)  
**领域**: 多模态VLM  
**关键词**: 金融推理, 多模态基准, 文档理解, 多步计算, RAG

## 一句话总结
本文提出FinMMDocR，一个面向真实金融场景的双语多模态推理基准，包含1200道专家标注的数值推理题目，涵盖12类隐式金融情景、9类长文档（平均50.8页）和平均11步推理链，最强MLLM (o4-mini-high) 仅达58%准确率，揭示现有模型在复杂金融推理中的严重不足。

## 研究背景与动机

### 领域现状
多模态大语言模型 (MLLM) 和大型多模态推理模型 (LMRM) 近年来取得了显著进展，在视觉常识推理和 VQA 等任务上表现优异。然而，在金融等专业领域的真实推理场景中，它们的能力尚未得到充分检验。

### 现有痛点
现有金融 QA 和文档 QA 基准存在三个关键缺陷：

**缺乏真实金融情景**：传统基准（如FinQA、TAT-QA）仅提取明确陈述的信息，忽略了金融分析师需要根据市场环境做出假设和判断的能力

**多模态文档理解不足**：部分基准仅使用纯文本输入，多模态基准中的图表和表格过于稀疏孤立，长文档基准又缺乏领域特定数值推理题

**忽视精确多步计算**：金融决策需要精确的数值计算，但现有基准忽略单位、百分比和小数问题，或允许过大的误差容限（如1%）

### 核心矛盾
金融分析师在实际工作中需要：(1) 理解市场环境和隐含条件，(2) 从长达几十页的专业文档中提取分散的关键信息，(3) 执行精确的多步数值计算。现有基准无法同时评估这三个维度。

### 切入角度
构建一个"三位一体"的金融多模态推理基准——同时考核情景感知、文档理解和多步计算，配合中英双语、精确评估标准和丰富的 RAG 分析。

## 方法详解

### 整体框架
FinMMDocR 由1200道中英双语（各600题）数值推理问题组成，每题配备真实金融情景、视觉丰富的金融文档、证据页码标注、黄金Python解题程序和精确答案。

### 关键设计

#### 1. **情景感知 (Scenario Awareness)**
- 57.9%的问题包含隐式金融情景（需要推断假设而非给定条件）
- 覆盖12类金融场景（如投资组合管理、金融建模与预测等）
- 每题平均涉及1.9个混合场景
- **设计动机**：真实金融分析中，分析师必须结合当前市场环境做出专业判断，而非简单地从文档中提取已给定的信息

#### 2. **文档理解 (Document Understanding)**
- 837篇中英金融文档，覆盖9个类别（如公司研究、行业研究、期货期权、金融工程等）
- 平均50.8页/文档，38.8k tokens/文档
- 包含专业金融图表（如K线图）
- 65%的问题需要跨页推理（平均2.4个证据页）
- **设计动机**：金融分析师需要从冗长的专业文档中定位和提取关键数据点，这是一个核心的实际技能

#### 3. **多步计算 (Multi-Step Computation)**
- 平均11步推理：5.3步信息提取（1.0步文本+4.3步视觉）+ 5.7步计算
- 严格评估标准：0.2%误差容限，精确考核单位、百分比、小数
- 配备黄金Python解决方案
- **设计动机**：金融决策是高风险领域，计算错误可能导致重大损失，必须要求精确答案

### 数据构建流程

#### 英文数据（600题）
- 从 DocMath-Eval_CompLong 选取600题（300 testmini + 300 test）
- 补充完整解题程序、标准答案和证据页标注
- 将文档每页渲染为图像，移除原始文本输入

#### 中文数据（600题，全新构建）
- 收集385篇中文研究报告
- 基于文档内容构建真实金融场景
- 由 Gemini 2.5 Pro 和 Claude 3.7 Sonnet 辅助生成问题和解答
- 严格质量控制：15名金融硕士+2名CFA专家标注

#### 质量保障
- 对候选标注进行交叉审查
- 初始759题中淘汰159题（21%）
- 保留的600题中494题需要修改（82%），其中451题需修正证据页、80题需调整解答、36题需重新表述问题

## 实验关键数据

### 主实验（图像输入 MLLM）

| 模型 | 规模 | 总体ACC | 有情景 | 无情景 | 文档≤30页 | 文档≥31页 |
|------|------|---------|--------|--------|-----------|-----------|
| OpenAI o4-mini-high | - | **58.00** | 55.72 | 62.34 | 57.02 | 58.95 |
| Doubao-1.5-thinking-pro | - | 38.17 | 39.50 | 35.41 | 43.99 | 32.51 |
| Claude 3.7 Sonnet (Thinking) | - | 37.00 | 35.60 | 39.40 | 41.96 | 32.18 |
| Qwen2.5-VL | 72B | 12.92 | 10.57 | 17.71 | 14.04 | 11.82 |
| Llama 4 Maverick | 400A17B | 2.67 | 3.65 | 0.75 | 1.86 | 3.45 |

### 消融实验（错误分析，o4-mini-high 100个失败案例）

| 错误类型 | 出现次数/100 | 说明 |
|----------|-------------|------|
| 文档理解错误 | 78 | 无法准确定位或提取关键信息 |
| 知识推理错误 | 44 | 公式选择或推理结构错误 |
| 情景感知错误 | 33 | 误解任务意图或上下文约束 |
| 数值计算错误 | 5 | 公式正确但计算精度有误 |

### RAG分析

| 方法类型 | 代表方法 | 关键发现 |
|----------|---------|----------|
| 文本RAG | BM25, Contriever, BGE-M3 | 低于视觉RAG |
| 视觉RAG | ColQwen2.5 | **最佳检索性能** |
| Agentic RAG | ViDoRAG, MDocAgent等5种 | 准确率低于简单ColQwen2.5，但消耗更多tokens和时间 |

### 关键发现
1. **没有模型超过60%准确率**：最强o4-mini-high仅58%，开源模型差距更大
2. **推理增强模型一致优于非推理模型**：前三名均为推理增强模型
3. **视觉理解差异巨大**：MLLM之间视觉能力差距（~30%）远大于LLM文本理解差距（~12%）
4. **信息提取是主要瓶颈**：在PoT设置下，提取错误比计算错误影响更大
5. **复杂Agent反而不如简单RAG**：更长的管线引入错误传播，迭代Agent延迟大但收益小
6. **多情景任务更困难**：情景数量增加时所有模型准确率显著下降

## 亮点与洞察
- **三维度统一设计**：情景+文档+计算的整合评估思路非常贴合金融分析实际工作流
- **中英双语**：打破了金融NLP benchmarks以英语为主的局面
- **严格质量控制**：82%数据修改率体现了极高的标注标准
- **RAG深度分析**：系统比较了6种检索模型和5种Agentic RAG，结论"复杂Agent不如简单RAG"极有实践价值
- **最佳模型配置的发现**：o4-mini-high是唯一图像输入性能超过文本（OCR+LLM）的模型

## 局限与展望
- 数据规模1200题在benchmark中偏小
- 中文数据来源于"授权渠道"的研究报告，可复现性存疑
- 0.2%误差容限虽严格但可能对某些题目过于苛刻（如涉及大量中间计算的场景）
- 仅评估Program-of-Thought (PoT)范式，其他推理范式（如CoT）未充分探索
- Agentic RAG的评估受限于所选底座模型（Doubao-1.5-vision-pro），其他底座可能产生不同结果
- 缺乏人类专家在相同条件下的baseline对比

## 相关工作与启发
- DocMath-Eval (Zhao et al., 2024)：本工作的英文数据基础，但原版只有文本输入
- FinMMR (Tang et al., 2025)、MME-Finance (Gan et al., 2025)：单图像金融推理，难度和场景丰富度不足
- ColPali/ColQwen2.5 (Faysse et al., 2025)：视觉检索方法在金融文档检索中表现突出
- 启发：未来的专业领域基准不仅需要领域知识，更需要模拟真实工作流（情景判断→信息提取→多步推理→精确计算）

## 评分
- 新颖性: ⭐⭐⭐⭐ （三维统一+双语设计新颖，但benchmark工作创新性有限）
- 实验充分度: ⭐⭐⭐⭐⭐ （26种配置+RAG分析+错误分析，非常全面）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，但部分细节在附录较多）
- 价值: ⭐⭐⭐⭐⭐ （填补金融多模态推理评估空白，实用价值高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICML 2026\] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](../../ICML2026/multimodal_vlm/multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[ACL 2025\] LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating](../../ACL2025/multimodal_vlm/longdocurl_multimodal_long_doc.md)
- [\[ACL 2025\] MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration](../../ACL2025/multimodal_vlm/mmboundary_reasoning_step_confidence.md)

</div>

<!-- RELATED:END -->
