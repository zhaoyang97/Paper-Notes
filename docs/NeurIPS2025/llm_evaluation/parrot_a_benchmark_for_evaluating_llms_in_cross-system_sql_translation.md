---
title: >-
  [论文解读] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation
description: >-
  [NeurIPS 2025][LLM评测][SQL翻译] 本文提出 PARROT，一个面向跨系统 SQL 翻译（SQL-to-SQL）的实际且真实的基准测试，包含来自 38 个开源基准和真实业务场景的 598 个核心翻译对（扩展到 28,003 对），覆盖 22 种生产级数据库系统，揭示当前最强 LLM 的平均准确率低于 38.53%。
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "SQL翻译"
  - "跨数据库系统"
  - "SQL方言"
  - "LLM评估"
  - "基准测试"
---

# PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation

**会议**: NeurIPS 2025  
**arXiv**: [2509.23338](https://arxiv.org/abs/2509.23338)  
**代码**: [https://github.com/weAIDB/PARROT](https://github.com/weAIDB/PARROT)  
**领域**: LLM评测  
**关键词**: SQL翻译, 跨数据库系统, SQL方言, LLM评估, 基准测试

## 一句话总结

本文提出 PARROT，一个面向跨系统 SQL 翻译（SQL-to-SQL）的实际且真实的基准测试，包含来自 38 个开源基准和真实业务场景的 598 个核心翻译对（扩展到 28,003 对），覆盖 22 种生产级数据库系统，揭示当前最强 LLM 的平均准确率低于 38.53%。

## 研究背景与动机

**领域现状**：LLM 在 Text-to-SQL（自然语言转 SQL）任务上表现日益出色，已成为数据库交互的热门研究方向。然而，另一个密切相关且具有重大实际意义的问题——跨系统 SQL 翻译（Cross-System SQL Translation，即 SQL-to-SQL）——却严重被忽视。这一任务是将一种数据库系统（如 MySQL）的 SQL 查询转换为另一种系统（如 ClickHouse）的等价查询。

**现有痛点**：(1) 现有 SQL 基准（如 Spider、WikiSQL）聚焦于极少数数据库系统（通常仅 SQLite），无法覆盖真实生产环境中的系统多样性；(2) 这些基准使用的 SQL 语法高度标准化，无法捕捉大量系统特定的 SQL 方言差异（如自定义函数、数据类型、语法规则），而这些差异恰恰是 SQL-to-SQL 翻译的核心挑战；(3) 缺乏专门的 SQL-to-SQL 评估基准，导致这一实际需求（如数据库迁移、多数据库系统协作）缺乏系统性研究。

**核心矛盾**：Text-to-SQL 研究繁荣但局限于标准化 SQL 和少数系统，而真实世界中跨系统 SQL 迁移是刚需，涉及大量方言差异，现有基准完全无法评估。

**本文目标** (1) 构建一个覆盖多系统、包含真实方言差异的 SQL-to-SQL 翻译基准；(2) 系统评估当前 LLM 的跨系统 SQL 翻译能力；(3) 提供不同粒度的评估变体以适应不同测试需求。

**切入角度**：从 38 个开源 SQL 基准和真实业务服务中收集翻译对，专门设计数据来挑战系统特定的 SQL 理解能力。提供两种评估指标——方言兼容性（AccEX，语法正确性）和结果一致性（AccRES，执行结果等价性）。

**核心 idea**：构建一个覆盖 22 种生产级数据库系统的实际 SQL-to-SQL 翻译基准，揭示 LLM 在理解系统特定 SQL 方言方面的严重不足。

## 方法详解

### 整体框架

PARROT 的数据构建流程：从 38 个开源数据库基准和真实业务场景中收集 SQL 查询 → 识别跨系统翻译对 → 标注系统特定的方言特征（自定义函数、数据类型、语法规则等）→ 验证翻译等价性 → 组织成核心集和扩展集。最终产出三个变体：PARROT 核心集（598 对，高难度）、PARROT-Diverse（28,003 对，广覆盖）和 PARROT-Simple（5,306 对，聚焦压力测试）。

### 关键设计

1. **系统特定 SQL 方言的全面覆盖**:

    - 功能：确保基准能挑战 LLM 对各数据库系统独特语法特性的理解
    - 核心思路：覆盖 22 种生产级数据库系统（包括 MySQL、PostgreSQL、Oracle、SQL Server、ClickHouse、Hive、Spark SQL、Presto 等），针对每系统的独特特性（如 ClickHouse 的 Array 函数、PostgreSQL 的窗口函数高级用法、Hive 的 UDF 等）设计翻译对。数据来源包括各系统的官方文档测试用例、开源 benchmark 和实际业务 SQL
    - 设计动机：真实的 SQL 迁移场景恰恰难在方言差异上——相同的查询意图在不同系统中可能需要完全不同的函数、类型转换和语法结构

2. **多变体基准设计**:

    - 功能：适应不同粒度的评估需求
    - 核心思路：
        - **PARROT（核心集）**：598 个精心挑选的翻译对，专注于挑战性方言理解，用于核心能力评估
        - **PARROT-Diverse**：28,003 个翻译对，覆盖广泛的语法变体，用于全面的语法能力测试
        - **PARROT-Simple**：5,306 个代表性样本，用于快速聚焦的压力测试
    - 设计动机：不同使用场景需要不同粒度：研究者需要核心集的深度评估，系统开发者需要 Diverse 的广覆盖测试

3. **双维度评估指标**:

    - 功能：从语法和语义两个角度评估翻译质量
    - 核心思路：
        - **AccEX（方言兼容性）**：翻译后的 SQL 能否在目标系统上成功执行（语法正确性）
        - **AccRES（结果一致性）**：翻译后的 SQL 执行结果是否与原始 SQL 等价（语义正确性）
    - 设计动机：一个 SQL 可能在目标系统上语法正确但逻辑错误（如类型转换导致精度丢失），需要两个维度共同评估

### 评估策略

在核心集上评估多种 LLM，涵盖不同许可协议（开源/闭源）、参数规模（7B–671B）和任务范围（通用/代码专用）。提供公开排行榜以推动持续评估。

## 实验关键数据

### 主实验（PARROT 核心集排行榜）

**方言兼容性（AccEX）排行榜：**

| 排名 | 模型 | 参数量 | AccEX |
|------|------|--------|-------|
| 🏆 | GPT-4o | UNK | 53.32% |
| 🥈 | DeepSeek-V3 | 671B | 50.64% |
| 🥉 | Claude 3.7 Sonnet | UNK | 48.09% |
| 4 | DeepSeek-R1 | 671B | 44.42% |
| 5 | DeepSeek-R1-32B | 32B | 41.98% |
| — | o3-mini | UNK | 27.94% |
| — | DeepSeek-R1-7B | 7B | 17.03% |

**结果一致性（AccRES）排行榜：**

| 排名 | 模型 | AccRES |
|------|------|--------|
| 🏆 | o3-mini | 54.23% |
| 🥈 | o1-preview | 48.69% |
| 🥉 | DeepSeek-R1 671B | 40.52% |
| 4 | DeepSeek-V3 671B | 32.65% |
| — | GPT-4o | 21.87% |

### 消融实验

| 分析维度 | 发现 |
|---------|------|
| 整体难度 | 所有 LLM 平均准确率 < 38.53%，远低于人类 DBA 水平（> 90%） |
| AccEX vs AccRES 排名不一致 | GPT-4o 在 AccEX 第一但 AccRES 仅 21.87%，说明语法正确不等于语义正确 |
| 推理模型的分化 | o3-mini 在结果一致性上最强但方言兼容性偏弱，暗示推理能力有助于语义理解但不够了解方言 |
| 规模效应 | DeepSeek-R1 从 7B→32B→671B 性能持续提升，但仍远低于人类 |
| 核心集 vs 扩展集 | 核心集专注高难度方言理解，更能区分模型能力 |

### 关键发现

- **LLM 在 SQL-to-SQL 翻译上严重不足**：最强模型 GPT-4o 的 AccEX 仅 53.32%，而人类 DBA 配合翻译工具可达 >90%
- **两种评估维度揭示不同能力**：AccEX 排名领先的模型在 AccRES 上可能表现一般（GPT-4o 是最突出的例子），说明"写出能运行的 SQL"和"写出结果正确的 SQL"是两种不同能力
- **推理模型的有趣分化**：o3-mini 在 AccRES 上最强但 AccEX 偏弱，暗示推理链有助于理解查询意图但可能导致使用了目标系统不支持的写法
- **小模型严重挣扎**：7B 级别模型（如 DeepSeek-R1-7B）AccEX 仅 17%，说明 SQL 方言知识需要大量参数容量

## 亮点与洞察

- **填补重要空白**：首个专门面向跨系统 SQL 翻译的基准，22 种数据库系统的覆盖在数据库研究中前所未有
- **揭示 Text-to-SQL 繁荣背后的盲区**：学界聚焦 NL-to-SQL 而忽略了实际意义同样重大的 SQL-to-SQL 问题
- **AccEX vs AccRES 的分离现象**：提出了一个有价值的评估框架——语法可执行性与语义等价性是正交的两个维度
- **实际应用价值大**：数据库迁移（如从 MySQL 到 PostgreSQL）和多系统协作是企业中的高频需求
- **持续评估设施**：公开排行榜和 GitHub 仓库支持社区持续贡献和评估

## 局限与展望

- 论文完整内容未能获取（arXiv HTML 不可用），部分细节（如数据构建 pipeline 中的质量保证流程、具体的方言类型分布分析）可能有遗漏
- 核心集仅 598 对，某些数据库系统可能覆盖不够充分
- 评估仅聚焦于准确率，未分析具体的错误类型分布（如函数映射错误 vs 类型转换错误 vs 语法结构错误）
- 人类 DBA 基线的获取方式和规模未详细说明
- 未探索 few-shot、RAG（检索系统文档）等增强策略对性能的影响

## 相关工作与启发

- Spider 和 WikiSQL 是 Text-to-SQL 的经典基准，但仅覆盖 SQLite，本文填补了 SQL-to-SQL 方向
- CrackSQL（同一团队的后续工作）提出了基于 LLM 的混合 SQL 方言翻译系统
- SQL 方言之间的差异是数据库系统领域长期存在的实际挑战，本文首次对其进行系统化的 LLM 评估
- 与 NL2SQL 领域的 BIRD、Dr.Spider 等强调鲁棒性的基准互补

## 评分

⭐⭐⭐⭐ (4/5)

选题精准（SQL-to-SQL 是被严重忽视但实际意义重大的方向），数据规模和系统覆盖面印象深刻（22 种生产级数据库、28K+ 翻译对），双维度评估设计合理。揭示了一个令人警醒的差距：最强 LLM 在 SQL 方言翻译上远不及人类。公开排行榜的设计有助于持续推动该领域进步。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2025\] SwiLTra-Bench: The Swiss Legal Translation Benchmark](../../ACL2025/llm_evaluation/swiltra-bench_the_swiss_legal_translation_benchmark.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](../../ACL2025/llm_evaluation/culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[AAAI 2026\] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](../../AAAI2026/llm_evaluation/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)
- [\[ACL 2025\] JuStRank: Benchmarking LLM Judges for System Ranking](../../ACL2025/llm_evaluation/justrank_llm_judge_system_ranking.md)

</div>

<!-- RELATED:END -->
