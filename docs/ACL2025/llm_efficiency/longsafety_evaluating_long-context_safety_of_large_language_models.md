---
title: >-
  [论文解读] LongSafety: Evaluating Long-Context Safety of Large Language Models
description: >-
  [ACL 2025][LLM效率][长上下文安全] 提出LongSafety——首个专门针对开放式长上下文任务的LLM安全评估基准，包含7类安全问题和6种任务类型共1,543个测试用例，揭示大多数模型安全率低于55%，且短上下文安全能力无法迁移到长上下文场景。 随着长序列处理技术的进步，LLM在理解和生成长文本方面展现出了卓…
tags:
  - "ACL 2025"
  - "LLM效率"
  - "长上下文安全"
  - "安全评估基准"
  - "多智能体评估"
  - "LLM安全"
  - "长文本"
---

# LongSafety: Evaluating Long-Context Safety of Large Language Models

**会议**: ACL 2025  
**arXiv**: [2502.16971](https://arxiv.org/abs/2502.16971)  
**代码**: [github.com/thu-coai/LongSafety](https://github.com/thu-coai/LongSafety)  
**领域**: LLM效率  
**关键词**: 长上下文安全, 安全评估基准, 多智能体评估, LLM安全, 长文本

## 一句话总结

提出LongSafety——首个专门针对开放式长上下文任务的LLM安全评估基准，包含7类安全问题和6种任务类型共1,543个测试用例，揭示大多数模型安全率低于55%，且短上下文安全能力无法迁移到长上下文场景。

## 研究背景与动机

随着长序列处理技术的进步，LLM在理解和生成长文本方面展现出了卓越的能力。然而，长上下文场景中的安全问题——如有害内容隐含和模型认知干扰——日益凸显，却缺乏系统性的评估工具。

现有的长上下文基准（如LongBench、InfiniteBench、RULER）主要关注通用能力评估，不涉及安全问题。而现有安全基准（如SafetyBench、Red Team、Advbench）通常局限于数百词的短上下文查询任务，无法评估需处理成千上万词文档的长上下文模型。唯一的并行工作LongSafetyBench使用选择题格式，但这种格式无法充分评估生成安全性——而生成安全对于生成式模型更为关键。

这一研究空白促使作者提出LongSafety，首个面向**开放式**长上下文任务的综合安全评估基准。

## 方法详解

### 整体框架

LongSafety的构建流程包括三个核心环节：数据收集（从互联网收集与安全场景相关的长文档）、指令策划（针对每个文档编写可能触发安全问题的指令）以及多智能体评估框架（评估模型响应的安全性）。

问题定义为：给定长上下文 $C$ 和安全指令 $I$，模型生成响应 $R$，评估 $R$ 是否安全。由于指令可以拼接在上下文前后两种位置，只有两种位置的响应都安全才算通过。

### 关键设计

1. **安全分类体系（7类安全问题）**: 涵盖毒性内容（Toxicity Content）、偏见观点（Biased Opinion）、身心伤害（Physical & Mental Harm）、违法活动（Illegal Activities）、不道德行为（Unethical Activities）、隐私与财产（Privacy & Property）、敏感话题（Sensitive Topics）。这一分类基于Sun等和Zhang等的先前框架修订而来，针对长上下文场景进行了调整。

2. **多样化任务类型（6种）**: 包括问答（QA）、生成（Generation）、头脑风暴（Brainstorming）、摘要（Summarization）、改写（Rewrite）和角色扮演（Role-playing）。前5种来自Ouyang等人的研究，角色扮演为新增任务，进一步拓展了长上下文任务的覆盖范围。

3. **多智能体评估框架**: 由三个专门角色组成，均由LLM驱动：

    - **风险分析器（Risk Analyzer）**：分析指令中隐含的安全风险，生成可能导致安全/不安全响应的行为参考集
    - **上下文摘要器（Context Summarizer）**：为长上下文生成精炼摘要，捕捉关键信息并突出与指令相关的内容，消除干扰信息
    - **安全判定器（Safety Judge）**：综合风险分析和上下文摘要，对模型响应做出安全/不安全的二元判定
   
   该框架通过多角度协作分析，在测试集上达到了92%的准确率，显著优于单一评估器。

4. **数据收集流程**: 

    - 众包工人使用预定义安全关键词在互联网搜索相关文档
    - 工人提取纯文本内容（可组合多文档形成长上下文）
    - 为每个上下文编写三条不同任务类型的安全指令
    - 保留最可能触发安全问题的指令，并过滤上下文与指令不一致的样本

### 损失函数 / 训练策略

LongSafety是评估基准而非训练方法，不涉及损失函数设计。评估采用新提出的 $SR_{long}$ 指标：只有当模型对指令在上下文前/后两种拼接方式的响应都被判定为安全时，该实例才被记为安全。

## 实验关键数据

### 主实验

| 模型 | $SR_{long}$ | $SR_{short}$ | 下降幅度 |
|------|-------------|--------------|----------|
| Claude-3.5-haiku | 77.7% | 89.9% | -12.2% |
| Claude-3.5-sonnet | 76.8% | 94.0% | -17.2% |
| GPT-4-turbo | 48.3% | 84.3% | -36.0% |
| GPT-4o | 40.4% | 73.7% | -33.3% |
| GPT-4o mini | 37.1% | 64.2% | -27.1% |
| Qwen2.5-72B | 31.3% | 72.2% | -40.9% |
| Llama-3.1-8B | 13.4% | 74.2% | -60.8% |

### 消融实验

| 配置 | 评估准确率 | 说明 |
|------|-----------|------|
| 完整多智能体框架 | 92% | 三角色协作 |
| 去掉Context Summarizer | 90% | 降低2%但仍优于单一Judge |
| GPT-4o mini单一Judge | <90% | 仅使用Safety Judge |
| Llama-Guard-3 | 最低 | 传统安全守卫 |

### 关键发现

- **长短上下文安全不对齐**：短上下文安全排名靠前的模型在长上下文中可能表现很差。Llama-3.1-8B短上下文安全排名第2，但长上下文下降超过60%，排名倒数第2
- **敏感话题最具挑战性**：所有开源模型在Sensitive Topics上的安全率低于20%，大多数闭源模型也低于50%
- **生成导向任务更危险**：Generation、Brainstorming、Summarization和Rewrite等生成任务的平均 $SR_{long}$ 低于30%，而QA任务达到46.3%
- **上下文相关性加剧风险**：包含相关上下文时，安全风险比不相关上下文更明显
- **输入长度影响安全**：随着输入序列变长，安全风险进一步加剧
- **Claude-3.5系列显著领先**：是唯一平均安全率超过55%的模型系列

## 亮点与洞察

- **填补重要空白**：首次系统性地评估LLM在长上下文开放式生成任务中的安全性，开放式格式比选择题更能反映真实场景
- **安全不迁移的发现**：短上下文安全性不能预测长上下文安全性，这对依赖短上下文安全测试的模型部署实践提出了严峻挑战
- **多智能体评估器设计精巧**：将评估任务分解为风险分析、上下文摘要、安全判定三个子任务，体现了分而治之的思想
- **数据统计详实**：1,543个测试用例，平均5,424词，覆盖7×6=42种安全问题+任务类型组合

## 局限与展望

- 基准主要覆盖英文内容，缺乏多语言评估
- 上下文长度主要集中在数千词级别，未覆盖超长（100K+ tokens）场景
- 多智能体评估器依赖GPT-4o mini，本身可能存在判断偏差
- 未提出具体的长上下文安全增强方法，主要是诊断性工作
- 安全关键词驱动的数据收集可能存在覆盖盲区
- 未来可探索长上下文专用的安全对齐训练策略

## 相关工作与启发

- LongBench（Bai et al., 2024）和InfiniteBench（Zhang et al., 2024a）关注长上下文通用能力，本文补充了安全维度
- SafetyBench（Zhang et al., 2023a）和SALAD-Bench（Li et al., 2024）是经典短上下文安全基准，本文扩展到长上下文
- LongSafetyBench（Huang et al., 2024）同样关注长上下文安全但使用选择题，本文的开放式生成格式更具挑战性
- 多智能体评估框架的设计思路可以推广到其他复杂的NLG评估任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个长上下文开放式安全基准，但主要是评测框架而非方法创新
- 实验充分度: ⭐⭐⭐⭐⭐ 16个模型、7类安全问题、6种任务类型、多维分析、评估器对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，表格和图表丰富，分析层次分明
- 价值: ⭐⭐⭐⭐⭐ 揭示了长上下文安全的严峻现状，对安全评估和对齐研究有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)

</div>

<!-- RELATED:END -->
