---
title: >-
  [论文解读] TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues
description: >-
  [ACL2025][LLM 其他][时间推理] 提出TReMu框架，通过时间感知记忆化（时间线摘要）和神经符号时间推理（LLM生成Python代码执行时间计算），将GPT-4o在多会话对话时间推理基准上的准确率从29.83%提升到77.67%。 1. 时间推理是LLM的薄弱环节：已有研究（TimeBench、TRAM等）表明…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "时间推理"
  - "多轮对话"
  - "神经符号推理"
  - "记忆增强Agent"
  - "时间线摘要"
---

# TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues

**会议**: ACL2025  
**arXiv**: [2502.01630](https://arxiv.org/abs/2502.01630)  
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: 时间推理, 多轮对话, 神经符号推理, 记忆增强Agent, 时间线摘要

## 一句话总结
提出TReMu框架，通过时间感知记忆化（时间线摘要）和神经符号时间推理（LLM生成Python代码执行时间计算），将GPT-4o在多会话对话时间推理基准上的准确率从29.83%提升到77.67%。

## 背景与动机

1. **时间推理是LLM的薄弱环节**：已有研究（TimeBench、TRAM等）表明，即使是最先进的LLM在时间推理任务上表现仍然不佳，尤其面对复杂的时间关系。
2. **多会话对话的时间挑战被低估**：现有时间推理基准多基于短文本（故事、维基百科），未充分考虑多会话对话中的特殊时间特征。
3. **相对时间表达普遍存在**：在多轮对话中，说话者常使用"上周""昨天"等相对时间表达而非具体日期，模型需从对话上下文中推断实际时间。
4. **跨会话依赖增加推理难度**：不同对话会话中涉及同一实体的事件需要跨会话关联，例如"提到创业想法"和"几个月后真正创业"分布在不同session中。
5. **长对话历史带来"历史噪声"**：随着对话轮数增加，无关历史信息的累积会干扰LLM对关键时间细节的提取，信噪比降低。
6. **缺少多会话对话时间推理评测基准**：现有对话相关基准（TimeDial、LoCoMo等）未显式覆盖相对时间和跨会话依赖这两个关键时间特征。

## 核心问题
如何提升LLM-Agent在多会话长对话中的时间推理能力，特别是处理相对时间表达和跨会话事件依赖？

## 方法详解

### 整体框架
TReMu基于记忆增强LLM-Agent的三阶段流程（记忆化→检索→响应），改进了记忆化和推理两个阶段。

### 组件一：时间感知记忆化（Time-aware Memorization）

**时间记忆写入**：
- 对每个对话session，指导LLM不仅生成摘要，还要提取事件并推断其发生的具体日期
- 关键区分：事件被"提及"的时间 vs 事件"发生"的时间。例如1月28日的对话中提到"上周六做了意大利餐"，记忆中应记录事件发生于1月25日
- 输出为细粒度的 (时间, 事件摘要) 对，而非传统的整体session摘要

**记忆组织**：
- 按时间线格式组织所有记忆条目，同一时间的事件归组
- 基于推断的时间步进行索引，支持基于时间的高效检索
- 与MemoChat的主题式摘要形成对比：MemoChat输出"Hobbies and Daily Rituals"这样的主题摘要，而TReMu输出"01/25/2020: Michelle做了意大利餐"这样的时间锚定记忆

### 组件二：神经符号时间推理（Neuro-symbolic Temporal Reasoning）

- 给定时间问题和检索到的相关记忆，指导LLM生成Python代码作为中间推理过程
- 利用Python的 `datetime` 和 `dateutil` 库进行精确的时间计算
- 提供预定义的辅助函数（如 `weekRange(t)` 返回t所在周的起止日期），支持函数调用
- `dateutil.relativedelta` 可处理"下周五"等相对时间计算
- 代码逐行执行相当于CoT的逐步推理，但借助编程语言的精确性避免自然语言推理的模糊错误
- 代码执行结果作为中间依据，再由LLM给出最终答案

### 基准构建
- 基于LoCoMo数据集（平均304.9轮、19.3个session、9209.2 tokens/对话）增强构建
- 用GPT-4o四步pipeline：事件提取→跨session事件链接→QA生成→人工质控
- 三种问题类型：Temporal Anchoring（264题，推断事件具体时间）、Temporal Precedence（102题，事件先后）、Temporal Interval（234题，事件间隔）
- 含112道不可回答题目，共600题

## 实验关键数据

### GPT-4o结果（Table 5）

| 方法 | TA | TP | TI | 总体准确率 | 不可答F1 |
|------|-----|-----|-----|-----------|---------|
| Standard Prompting | 18.18 | 58.82 | 30.34 | 29.83 | 20.84 |
| CoT | 67.80 | 74.51 | 49.15 | 61.67 | 43.18 |
| MemoChat | 35.23 | 43.14 | 25.21 | 32.67 | 37.02 |
| MemoChat + CoT | 51.14 | 49.02 | 26.50 | 41.67 | 38.00 |
| Timeline + CoT | 83.33 | 78.41 | 58.55 | 71.50 | 52.84 |
| **TReMu** | **84.47** | **81.37** | **68.38** | **77.67** | **64.42** |

### 跨模型对比（总体准确率）

| 方法 | GPT-4o | GPT-4o-mini | GPT-3.5-Turbo |
|------|--------|-------------|---------------|
| SP | 29.83 | 29.00 | 23.83 |
| CoT | 61.67 | 45.67 | 25.83 |
| TReMu | **77.67** | **51.17** | **33.67** |

### 关键发现
- TReMu在三个LLM上均取得最优准确率和F1分数
- 时间感知记忆化贡献显著：MemoChat+CoT→Timeline+CoT，GPT-4o准确率从41.67→71.50（+29.83）
- 神经符号推理进一步提升：Timeline+CoT→TReMu，准确率从71.50→77.67（+6.17）
- 代码执行失败率普遍很低，GPT-4o最低，验证了Python代码推理方案的可靠性
- 对GPT-4o/mini而言，直接用记忆机制反而不如CoT，因为这些模型上下文足够长；但GPT-3.5因输入限制需要记忆机制

## 亮点

1. **时间感知记忆设计精妙**：区分事件"发生时间"与"被提及时间"，用时间线格式组织记忆，解决了相对时间的歧义问题
2. **神经符号推理务实有效**：利用LLM擅长写Python代码的能力，将时间计算外包给精确的符号执行器，避免自然语言推理的模糊错误
3. **基准构建方法可复用**：四步pipeline（提取→链接→生成→质控）为其他领域构建推理评测基准提供了模板
4. **消融实验清晰分离贡献**：通过MemoChat/Timeline/CoT/符号推理的组合对比，清楚展示每个组件的增量收益

## 局限与展望

1. **仅在多选QA设置下评估**：未测试生成式对话场景，多选题可能低估实际应用中的时间推理难度
2. **依赖闭源LLM**：仅测试了GPT系列，开源模型（如Llama-3-70B）因上下文长度限制和指令遵循能力不足而无法适配
3. **基准规模有限**：600题基于LoCoMo的少数对话构建，多样性和泛化性有待验证
4. **记忆化阶段本身需要LLM推理**：时间线摘要的生成依赖LLM的时间推断能力，如果LLM本身推断错误，后续推理将级联出错
5. **Python代码生成存在错误可能**：虽然失败率低，但仍需重新生成机制，未讨论这种重试策略的系统开销

## 与相关工作的对比

### vs MemoChat (Lu et al., 2023)
MemoChat生成主题式的session摘要（如"Hobbies and Daily Rituals"），丢失了细粒度时间信息。TReMu生成时间锚定的事件记忆（如"01/25/2020: Michelle做了意大利餐"），直接支持时间推理。实验显示在GPT-4o上TReMu比MemoChat+CoT高36个百分点。

### vs CoT (Wei et al., 2022)
CoT用自然语言逐步推理，容易在时间计算中犯错（如无法正确判断"上周"对应哪个日期范围）。TReMu用Python代码替代自然语言推理，借助 `dateutil.relativedelta` 和自定义时间函数实现精确计算，是CoT的"符号化升级版"。

### vs Code-based Temporal QA (Li et al., 2023)
Li等人也用代码执行做时间问答，但仅基于短文本，不涉及记忆机制和多会话场景，且不支持函数调用。TReMu的函数调用能力（如 `weekRange`）扩展了可处理的时间推理类型。

## 启发与关联
- 时间感知记忆化的思路可推广到任何需要长期记忆的Agent系统，不限于对话场景
- "LLM生成代码→符号执行→LLM解释结果"的三步推理范式可迁移到数值推理、逻辑推理等其他结构化推理任务
- 基准构建中"提取→链接→QA生成→质控"的pipeline为低成本构建高质量评测集提供了参考

## 评分
- 新颖性: ⭐⭐⭐⭐ — 时间感知记忆+神经符号推理的组合是新的，区分事件发生/提及时间的设计有洞察
- 实验充分度: ⭐⭐⭐⭐ — 3个LLM×6种方法，消融完整，含case study和执行失败率分析
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，Table 3/4的对比直观展示记忆差异，整体结构良好
- 价值: ⭐⭐⭐⭐ — 多会话对话时间推理是实际需求，框架和基准均有贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Controlling Politeness in Multi-Turn Dialogues Through Pre-Phrase Augmentation](controlling_politeness_in_multi-turn_dialogues_through_pre-phrase_augmentation.md)
- [\[ACL 2025\] Temporal Reasoning for Timeline Summarisation in Social Media](temporal_reasoning_for_timeline_summarisation_in_social_media.md)
- [\[ACL 2025\] SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](synapticrag_enhancing_temporal_memory_retrieval_in_large_language_models_through.md)
- [\[ACL 2025\] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)

</div>

<!-- RELATED:END -->
