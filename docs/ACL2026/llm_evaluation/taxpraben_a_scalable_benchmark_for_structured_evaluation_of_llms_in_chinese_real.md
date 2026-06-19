---
title: >-
  [论文解读] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice
description: >-
  [ACL 2026][LLM评测][税务实践] 本文提出 TaxPraBen，首个面向中国税务实践的 LLM 评测基准，包含 14 个数据集共 7.3K 样本，覆盖税务风险防控、稽查分析和税务筹划三大真实场景，并设计了"结构化解析—字段对齐提取—数值与文本匹配"的可扩展评估范式，评测 19 个 LLM 后发现闭源大模型和中文优化模型表现更优，而税务领域微调模型 YaYi2 改进有限。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "税务实践"
  - "LLM评测基准"
  - "结构化评估"
  - "中文税务"
  - "Bloom认知分类"
---

# TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice

**会议**: ACL 2026  
**arXiv**: [2604.08948](https://arxiv.org/abs/2604.08948)  
**代码**: [https://github.com/Yating-Chen/TaxPraBen](https://github.com/Yating-Chen/TaxPraBen)  
**领域**: 信息检索  
**关键词**: 税务实践, LLM评测基准, 结构化评估, 中文税务, Bloom认知分类

## 一句话总结

本文提出 TaxPraBen，首个面向中国税务实践的 LLM 评测基准，包含 14 个数据集共 7.3K 样本，覆盖税务风险防控、稽查分析和税务筹划三大真实场景，并设计了"结构化解析—字段对齐提取—数值与文本匹配"的可扩展评估范式，评测 19 个 LLM 后发现闭源大模型和中文优化模型表现更优，而税务领域微调模型 YaYi2 改进有限。

## 研究背景与动机

**领域现状**：LLM 在通用 NLP 任务上表现优异，但在税务等高度专业化、知识密集且法规驱动的领域仍存在明显不足。现有领域基准如 FinBen（金融）、MedBench（医学）、LAiW（法律）已覆盖多个垂直领域，但税务方向的评测基准极度匮乏。

**现有痛点**：(1) 已有税务相关评测（如 TaxBen）主要关注孤立的 NLP 任务（文本分类、生成、推理），未能反映真实税务实践中语义理解与数值计算并重的需求；(2) 海外税务数据与中国税务管理实际差异大，难以直接迁移；(3) 部分模型在孤立任务上表现良好，但在需要同时整合语义推理和数值计算的真实场景中却表现不佳，现有排名存在虚高现象。

**核心矛盾**：税务实践要求 LLM 同时具备政策语义理解和精确数值计算能力，而传统 NLP 基准通过分离的任务评估无法捕捉这种复合需求，导致模型实际应用能力被高估。

**本文目标**：构建首个面向中国税务实践的综合性评测基准，涵盖从知识记忆到理解再到应用的完整认知层级，填补税务实践场景评估的空白。

**切入角度**：从 Bloom 认知分类法出发，将税务任务分为知识记忆（KM）、知识理解（KU）和知识应用（KA）三个层级，并引入税务风险防控、稽查分析和税务筹划三个贴近真实工作场景的实践任务。

**核心 idea**：设计一个结构化评估范式，通过"结构化解析→字段对齐提取→数值与文本混合匹配"的流水线，实现对 LLM 税务实践能力的端到端评估，同时该范式可扩展到法律、医疗、金融等其他领域。

## 方法详解

### 整体框架

TaxPraBen 要回答的问题是：一个 LLM 到底能不能胜任真实的中国税务工作？为此它围绕"会背—会懂—会用"的认知阶梯组织了 14 个数据集共 7.3K 实例，既含 10 个传统应用任务，也含 3 个贴近实战的实践场景任务。数据通过三条流水线汇集——书籍 OCR（税务考试指南与筹划案例）、官方文档下载（国家税务总局政策法规）、网络爬取（新闻、风控报告、稽查案例），全部经人工验证并由 ChatGPT 辅助标注。评测端则用一套"结构化解析→字段对齐提取→数值与文本混合匹配"的范式，把模型自由文本压成标准 JSON 后再自动打分。

### 关键设计

**1. Bloom 认知分类法任务体系：从"会背"到"会用"分三层评**

像 TaxBen 那样只盯着孤立 NLP 任务，看不出模型在真实税务工作中到底卡在哪一环；TaxPraBen 借 Bloom 认知分类法把能力拆成三个层级来分别施压。知识记忆（KM）考的是对税法条文、政策法规的准确复述，代表数据集是 TaxRecite；知识理解（KU）考的是从税务材料里识别关键信息、读懂政策含义，包含 TaxSum、TaxTopic、TaxRead；知识应用（KA）则要求模型在真实场景里把法规和计算综合用起来，涵盖 TaxCalc、TaxSCQ、TaxMCQ 等 10 个数据集。这样一条从记忆到应用的完整能力链，正好能暴露"会背但不会算"这类被传统基准掩盖的短板。

**2. 结构化评估范式：语义准确与结构对齐一起评**

真实税务实践的输出从来不是单一答案，而是"解释文本 + 关键数值"的混合体，传统单一指标根本兜不住。TaxPraBen 为此先定一套统一的 JSON 输出协议，再为三类实践场景各定标准化模式——TaxRisk 提取风险点与解决方案、TaxInspect 提取犯罪行为与罪名与处罚结果、TaxPlan 生成筹划策略并算出节税额。由于开源模型输出格式参差不齐，它索性用 ChatGPT-3.5 当结构化解析器，把自由文本先转成规范 JSON 再自动评估，从而用一个轻量解析步骤替掉了脆弱的正则匹配，让评分不再对输出格式敏感。

**3. 多源数据融合与质量控制：让数据集贴近真实而非学术构造**

现有领域基准多依赖公开数据集，离真实应用场景有距离；TaxPraBen 用三条数据流水线分别覆盖考试题目、政策法规和网络案例，尽量还原税务工作者面对的素材分布。ChatGPT 在此只承担结构化信息提取和数据增强的辅助角色，所有数据仍须过人工可用性检查这一关。指令标注则交给税务专业团队，采用 4 分量表评分，并以 Fleiss' Kappa 和 Krippendorff's Alpha 双指标确认标注一致性，保证数据集既贴近实战又足够可靠。

### 损失函数 / 训练策略

TaxPraBen 是评测基准，不涉及模型训练。评估指标按输出类型设计：分类任务用 Accuracy/F1/Macro-F1，生成任务用 BERTScore/BARTScore，结构化预测用 EM Accuracy，混合匹配任务则取 EM Accuracy 与 BERTScore 的加权平均，以同时兼顾数值精确与语义贴合。

## 实验关键数据

### 主实验

**19 个 LLM 在税务认知层级上的零样本表现**

| 模型 | KM (记忆) | KU (理解) | KA (应用) | 类型 |
|------|----------|----------|----------|------|
| ERNIE-3.5 | 0.667 | 0.599 | 0.475 | 闭源中文 |
| Grok3 | 0.519 | 0.579 | 0.482 | 闭源多语 |
| GPT-4o | 0.478 | 0.637 | 0.472 | 闭源多语 |
| ChatGPT | 0.488 | 0.602 | 0.415 | 闭源多语 |
| Qwen2.5 | 0.499 | 0.538 | 0.375 | 开源中文 |
| DeepSeek-R1 | 0.461 | 0.455 | 0.324 | 开源中文 |
| YaYi2 (税务微调) | 0.485 | 0.307 | 0.239 | 领域微调 |
| Mistral-v0.3 | 0.400 | 0.277 | 0.114 | 开源多语 |

### 消融实验

| 维度 | 发现 | 说明 |
|------|------|------|
| 零样本 vs 一样本 | 11/19 模型在一样本下提升 | 部分模型如 GLM4 反而下降 |
| NLP-CLS 任务 | 大多数模型表现差 | 税务分类需要深度领域知识 |
| NLP-REA 任务 | 所有模型困难 | 多步数值推理是瓶颈 |
| NLP-GEN 任务 | 表现相对最好 | ERNIE-3.5 零样本达 0.670 |
| 中文 vs 多语 LLM | 中文模型普遍更优 | 语言优化对税务术语至关重要 |

### 关键发现

- 知识应用（KA）是最难的层级，所有模型在此层级表现最差，反映税务实践对经济活动上下文推理的依赖
- ERNIE-3.5 凭借知识增强预训练策略在 KM 任务上领先，说明领域知识融合的价值
- YaYi2 虽经税务数据微调但表现不如通用 LLM，说明微调数据量和任务覆盖度不足时效果有限
- 推理任务（REA）对所有模型都是挑战，暴露了 LLM 在数值计算和税务逻辑理解方面的系统性弱点

## 亮点与洞察

- 首次将 Bloom 认知分类法系统应用于中文税务领域评测，建立了从记忆到应用的完整能力评估链
- 结构化评估范式具有很强的可迁移性——"结构化解析→字段对齐→混合匹配"的方法可直接应用于法律、金融、医疗等需要"解释文本+关键数值"混合输出的领域
- 用 ChatGPT-3.5 作为轻量级结构化解析器替代脆弱的正则匹配，是一个实用的工程 trick，降低了评估对输出格式的敏感性

## 局限与展望

- 数据来自互联网和公开书籍，存在测试数据泄露风险——LLM 可能在预训练中已接触过部分测试内容
- 评测模型参数量集中在 7B 左右，缺乏对更大开源模型的比较
- 自动语义相似度指标可能无法完全反映人类对答案质量的判断
- 未来可扩展到多轮对话式税务咨询、跨年度政策变化追踪等更复杂场景

## 相关工作与启发

- **vs TaxBen**: TaxBen 关注孤立 NLP 任务，TaxPraBen 增加了三个真实实践场景并提出结构化评估，更贴近实际应用
- **vs FinBen/LAiW**: 金融和法律领域基准已较成熟，TaxPraBen 填补了税务领域的空白，且其混合评估方法可反哺这些领域
- **vs MMLU/C-Eval**: 通用基准缺乏领域深度，无法评估专业数值推理能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个中文税务实践基准，结构化评估范式有通用价值
- 实验充分度: ⭐⭐⭐⭐ 19个模型横评全面，但缺少更大模型的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务分类和数据构建描述详细
- 价值: ⭐⭐⭐⭐ 填补税务评测空白，结构化评估范式可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2025\] RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](../../ACL2025/llm_evaluation/rulearena_rule_guided_reasoning.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)

</div>

<!-- RELATED:END -->
