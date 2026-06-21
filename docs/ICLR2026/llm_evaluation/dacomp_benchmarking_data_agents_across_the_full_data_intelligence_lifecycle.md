---
title: >-
  [论文解读] DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle
description: >-
  [ICLR2026][LLM评测][数据智能体] DAComp 是一个覆盖企业级"数据智能全生命周期"的 210 任务基准，把数据智能拆成"硬轴"仓库级数据工程（DE）和"软轴"开放式数据分析（DA），分别用执行式多指标和分层 rubric 的 LLM-judge 评测，发现即便 GPT-5 在 DE 上严格成功率也只有 20%、DA 平均不到 50%，暴露出当前数据智能体在整体管线编排和开放式推理上的硬伤。
tags:
  - "ICLR2026"
  - "LLM评测"
  - "数据智能体"
  - "数据工程"
  - "数据分析"
  - "仓库级评测"
  - "LLM-judge"
---

# DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EtzJy9yI5J](https://openreview.net/forum?id=EtzJy9yI5J)  
**代码**: https://da-comp.github.io  
**领域**: LLM 评测 / 数据智能体 / Benchmark  
**关键词**: 数据智能体, 数据工程, 数据分析, 仓库级评测, LLM-judge

## 一句话总结
DAComp 是一个覆盖企业级"数据智能全生命周期"的 210 任务基准，把数据智能拆成"硬轴"仓库级数据工程（DE）和"软轴"开放式数据分析（DA），分别用执行式多指标和分层 rubric 的 LLM-judge 评测，发现即便 GPT-5 在 DE 上严格成功率也只有 20%、DA 平均不到 50%，暴露出当前数据智能体在整体管线编排和开放式推理上的硬伤。

## 研究背景与动机
**领域现状**：LLM 在 text-to-SQL、软件工程、计算机控制等任务上展现出强大的推理与代码生成能力，催生了一批"数据智能体"——希望让 agent 自动完成把原始数据变成可决策洞察的整条流程。但要评估这种 agent 到底行不行，需要能真实反映企业复杂度的基准。

**现有痛点**：现有数据类基准几乎都把复杂工作切成了孤立的小任务。一类（BIRD、Spider 2.0、DA-Code）把数据工程简化成"生成一条 SQL / 一个脚本"，schema 小、代码量只有几十到上百行，完全无法体现真实数据工程要在几十张表、上千列上搭建多层数据管线的工程量；另一类（DSBench、BLADE、DABStep）把数据分析压缩成"有标准答案的确定性问答"，丧失了真实分析中规划、探索、综合洞察的开放性。

**核心矛盾**：真实企业数据智能其实横跨两种本质不同的能力——一边是像数据工程师那样**系统性大规模写代码、并在需求演化下维护管线**（工程现实性，Hard 轴），另一边是像数据分析师那样**面对开放式业务问题做战略规划、迭代探索、综合出可执行建议**（分析开放性，Soft 轴）。现有基准最多只覆盖其中一轴，无法回答"一个 agent 能不能端到端跑通整条数据智能生命周期"。

**本文目标**：构造一个同时沿 Hard 轴和 Soft 轴评测、且贴近真实企业复杂度的基准，覆盖从架构设计、管线实现、系统演化到开放式分析的完整链路，并为这些非平凡任务设计可靠的评测协议。

**切入角度**：作者把数据工程师和数据分析师的真实职责形式化为两类任务族 DAComp-DE 与 DAComp-DA，在工业级 SaaS schema 和真实分析问题上构建任务，并针对"确定性工程"和"开放式分析"两种性质分别设计执行式评测与分层 rubric 评测。

**核心 idea**：用一个 210 任务的基准把"数据智能全生命周期"完整刻画出来——仓库级 DE 用执行式多指标量化、开放式 DA 用经人类验证的分层 rubric LLM-judge 评测——从而精确诊断当前数据智能体的真实瓶颈。

## 方法详解

### 整体框架
DAComp 是一个基准，不是一个模型，所以"方法"是它如何定义任务、如何评测、如何标注。整体上它沿两条轴展开：**Hard 轴 = DAComp-DE（仓库级数据工程）**，**Soft 轴 = DAComp-DA（开放式数据分析）**，共 210 个任务（DE-Arch 30 / DE-Impl 30 / DE-Evol 50 / DA 100），并额外发布了一份高质量中文适配版 DAComp-zh（任务集与英文版完全对应）。

DE 把数据工程建模成在企业 schema 上"从原始数据源经过 staging→core→mart 多层管线、产出语义层数据"的 DAG 构建过程，形式化为 $(S, C^\star) = \pi_{de}(Q_{de}, C_0, B)$，其中 $Q_{de}$ 是高层需求、$S$ 是工程规格（如数据契约）、$C_0$ 是初始仓库、$B$ 是数据库、$C^\star$ 是最终 DE 仓库，并据此切成架构、实现、演化三种任务类型。DA 则把开放式分析建模成 $O = \pi_{da}(Q_{da}, D)$：给定可分析的语义层数据 $D$ 和开放式业务问题 $Q_{da}$，agent 自主写 SQL/Python 聚合计算、解释中间结果、综合出报告、关键洞察、可执行建议和可视化图表，且同一问题允许多条有效分析路径、没有唯一标准答案。

评测随任务性质分流：确定性的 DE-Impl/DE-Evol 走执行式多指标，开放式的 DA 和 DE-Arch 走分层 rubric 的 LLM-judge。整条基准由 8 位专家通过"数据收集 → 任务设计 → 评测构建"的标注流水线打磨而成。

### 关键设计

**1. 双轴任务划分：把数据工程和数据分析当成两种本质不同的能力分别考**

针对"现有基准只覆盖一轴、且把任务切碎"的痛点，DAComp 明确区分 Hard 轴和 Soft 轴。Hard 轴 DAComp-DE 是首个引入**仓库级**数据工程的基准：agent 不是写一条孤立 SQL，而是要在工业 schema（平均 32 张表、412 列）上编排多层数据管线，并分成三类——**DE-Architecture** 给定高层需求 $Q_{de}$ 和初始仓库 $C_0$、产出工程规格 $S$（高层规划）；**DE-Implementation** 给定详细规格 $S$ 和空仓库（$C_0=\varnothing$）、从零实现整个仓库 $C^\star$；**DE-Evolution** 给定既有仓库 $C_0$ 和新规格 $S$、在演化需求下改造成 $C^\star$。Impl 与 Evol 都极其繁重，常需跨 30+ 文件改动 4000+ 行代码，逼近真实工程工作量。Soft 轴 DAComp-DA 则首创真实世界开放式分析：面对复杂业务问题，agent 要规划多步分析、迭代写代码、跨结果综合洞察并自主生成可视化，与以往"确定性答案"的数据科学任务划清界限。之所以分两轴，是因为实验证实工程与分析是**相互独立的能力**——同一模型在两轴上的表现并不一致，合并成单一分数会掩盖真实短板。

**2. 执行式多指标：用递进严格度的三档分数诊断"管线编排"瓶颈**

确定性的 DE-Impl/Evol 输出是可执行的 DAG，作者设计了三个严格度递增的执行式指标来定位瓶颈，而不是只给一个笼统的对错。第一档是**部分得分的 Component Score（CS）**，$\text{CS} = \sum_j w_j s_j$，它**用 gold 上游输入孤立地评估每个 DAG 节点**，衡量纯粹的组件级 SQL 生成能力；第二档是**Cascading Failure Score（CFS）**，沿 DAG 顺序评估、**一旦上游依赖出错就把该节点得分归零**，从而衡量端到端的数据完整性；第三档是最严格的**Success Rate（SR）**，$\text{SR} = \mathbb{I}[\forall j: s_j = 1]$，要求每一个组件都完美无瑕。三档之所以关键，在于 CS 与 CFS/SR 之间的落差正好暴露核心瓶颈：**agent 能写对单个组件，却无法把它们整体编排成正确管线**——这是"整体管线编排"难、而非单纯"代码生成"难的直接证据。报告里还配有 Max-CFS@8、Max-CS@8 等多次采样取最好的变体。

**3. 分层 rubric + GSB 的 LLM-judge：让开放式分析既能多路径打分又不偏袒方法选择**

开放式的 DA 和 DE-Arch 没有唯一答案，作者构造了一个分层 rubric 框架交给 LLM-judge 评。它沿六个维度评分：Completeness、Accuracy、Insightfulness 由分层 rubric 评，Readability、Analytical depth、Visualization 由 Good–Same–Bad（GSB）评。rubric 把一个问题 $Q$ 自顶向下分解成 requirement 和 sub-requirement，每个 sub-requirement 下**枚举多条等价有效的解题路径**，每条路径挂自己的打分项（叶节点）；打分时 judge 为每个 sub-requirement **挑选最匹配的那条路径、只套用该路径的打分项，再自底向上聚合**，从而在不惩罚方法选择的前提下容纳多种正确做法。rubric 得分是满足项的归一化加权和：

$$\text{Score}_{rubric}(O, R) = \frac{\sum_{k=1}^{N} s_k}{\sum_{k=1}^{N} w_k}, \quad s_k = \Lambda(c_k, O) \in [0, w_k]$$

GSB 则只把最终分析结果与五份预置 baseline 报告做对比，按 Good/Same/Bad 计分：

$$\text{Score}_{gsb}(O, O_{base}) = \frac{\max(0, |G| - |B|)}{|G| + |S| + |B|}$$

DA 任务最终分是两者加权：$\text{Score}_{da} = \alpha \cdot \text{Score}_{rubric} + (1-\alpha) \cdot \text{Score}_{gsb}$，论文取 $\alpha=0.6$、用 Gemini-2.5-Flash 当 judge。DE-Arch 评法类似但用非分层的标准 rubric、不含 GSB 项。这个 judge 的可靠性经过严格验证：作者从五个不同 LLM 采样输出、确认枚举路径能覆盖所有观察到的解法，并验证 judge 与人类专家高度一致，从而把"有效但未预料的解法被误判为错"的假阴性风险压到最低。

**4. 8 专家标注流水线：从工业资产里反向工程出可执行、可评测的真实任务**

任务的真实性来自一套严谨的三阶段标注。**数据收集**全部基于宽松许可（Apache-2.0、MIT）资产——DE 侧收集 73 个企业级 SaaS schema 及其数据转换项目（平均 400 列）、灌入大规模且关系一致的合成数据；DA 侧从网络精选 100 个复杂数据库、补上由 DE 转换数据派生的分析建模层。**任务设计**上，DA 每张分析表先草拟 8 个开放问题、五人按真实性与难度投票留 top-2；DE-Evol 由在职数据工程师按企业标准撰写新业务需求；DE-Impl 把选定 SaaS 转换项目反向工程成一份 `data_contract.yaml`、刻画完整 DAG 与语义；DE-Arch 从 Impl/Evol 的分析层出发提 5 个候选需求、由工程师选 1 个可行又有挑战的。**评测构建**上，DA 至少 3 位标注者建分层 rubric 并对齐讨论，路径枚举遵循三原则（路径要是方法学上不同的策略而非增量步骤、确定性输出对照可程序化计算的锚值校验、用方法学软约束公平评价未枚举的有效路径）；DE-Impl/Evol 则编写执行脚本对照 gold 仓库自动校验、在节点/层级给部分得分以捕捉逐步正确性。

## 实验关键数据

### 主实验
作者在 OpenHands（CodeAct-Agent）框架下评测了一批开源与闭源模型，并为 DA 额外开发了通过 Bash/文件系统交互、能跑 Python 和 SQL 的自定义 DA-Agent。DE 侧聚合为 DE Score（Impl/Evol 用 CFS），DA 侧用 $\alpha=0.6$ 聚合 rubric 与 GSB。

DAComp-DE（英文版，Table 3）核心结果：

| 模型 | DE-Arch | Impl CS | Evol SR@8 | DE Score |
|------|---------|---------|-----------|----------|
| GPT-5 | 63.93 | 39.87 | 20.00 | **43.45** |
| Gemini-2.5-Pro | 51.96 | 36.88 | 8.00 | 32.88 |
| Qwen3-Coder | 51.43 | 32.86 | 12.00 | 32.80 |
| DeepSeek-V3.1 | 52.66 | 30.73 | 10.00 | 31.41 |
| Qwen3-235B-A22B | 50.73 | 20.15 | 2.00 | 20.15 |

即便最强的 GPT-5，DE Score 也只有约 43%、严格成功率仅 20%，且开源专精模型（Qwen3-Coder、DeepSeek-V3.1）足以与通用闭源模型 Gemini-2.5-Pro 抗衡。

DAComp-DA（DA-Agent baseline，Table 5）核心结果：

| 模型 | Completeness | Accuracy | Visualization | DA Score |
|------|-------------|----------|---------------|----------|
| GPT-5 | 64.23 | 43.81 | 27.44 | **50.84** |
| Kimi-K2 | 52.31 | 33.56 | 14.40 | 41.89 |
| Gemini-2.5-Pro | 45.43 | 30.30 | 13.40 | 34.70 |
| DeepSeek-V3.1 | 48.74 | 32.97 | 11.45 | 34.33 |
| Qwen3-8B | 9.89 | 4.12 | 0.15 | 4.47 |

DA 任务上多数模型分数跌到 50% 以下，只有少数闭源系统展现较稳健的分析力；可视化维度普遍极低（GPT-5 也才 27.44），说明"把数值结果转成有效图表"是当前共同短板。

### 关键发现
- **管线编排比代码生成更难**：DE 上 CS（孤立组件分，GPT-5 约 39.87）与严格 SR（20%）之间的巨大落差，证明瓶颈在端到端编排而非写单个 SQL，框架优化只能稳定交互、无法突破上限。
- **工程与分析是两种独立能力**：同一模型在 DE 与 DA 上的相对强弱并不一致，支持把基准拆成双轴的设计动机。
- **规模不是万能解药**：Qwen3-235B-A22B 这类大模型在 Impl/Evol 上几乎崩溃（SR 仅 2%），说明仓库级工程对模型是结构性挑战、不靠堆参数解决。
- **中英一致**：DAComp-zh（Table 4/6）与英文版排名和数值高度一致，验证基准的跨语言稳健性。

## 亮点与洞察
- **"全生命周期"的任务设计真正贴近企业**：DE-Arch/Impl/Evol 三连覆盖了从规划、从零搭建到演化维护的完整工程链路，Evol 任务模拟真实维护（平均改 1718 行、跨 13 文件）是以往基准缺失的一环。
- **三档执行指标是诊断利器**：CS/CFS/SR 不只给一个总分，而是把"组件级能力"和"整体编排能力"分离出来，能精确指出 agent 卡在哪——这种"可诊断性"对推动后续方法迭代比单一排行榜更有价值。
- **rubric 多路径打分巧妙解决开放式评测的公平性**：枚举等价路径、judge 选最匹配路径再聚合，既容忍方法多样性又不失严格，且配了五模型采样验证覆盖率来压假阴性，这套思路可迁移到任何"开放式、多解法"的 agent 评测。
- **GSB 与 rubric 互补**：rubric 管 Completeness/Accuracy/Insightfulness 这类可拆解维度，GSB 管 Readability/Analytical depth/Visualization 这类更主观的整体质量，分而治之。

## 局限与展望
- **LLM-judge 仍是潜在偏差源**：尽管做了人类一致性验证，开放式 DA 的最终分依赖 Gemini-2.5-Flash 评判，judge 自身的偏好和能力上限可能影响绝对分数，跨 judge 的稳健性值得进一步考察。
- **合成数据的真实性边界**：DE 任务用大规模合成数据填充工业 schema，虽保证关系一致，但与真实生产数据的脏乱、长尾分布仍有差距，可能让任务难度偏理想化。
- **可视化维度评测偏弱**：可视化分普遍极低且方差不小，说明该维度的 rubric 或 judge 对图表质量的刻画还不够细，未来可引入更专门的图表评估。
- **210 任务规模相对有限**：相比 SWE-Bench（2294）等，DAComp 任务数不算大，且 DE 部分高度依赖人工反向工程，扩展到更多领域 schema 的成本较高。

## 相关工作与启发
- **vs SWE-Bench / WebArena / OSWorld**：这些是通用 agent 基准（软件工程、网页导航、计算机控制），用执行式评测但不涉及数据智能的"分析开放性"；DAComp 专注数据领域且同时覆盖确定性工程与开放式分析两轴。
- **vs BIRD / Spider 2.0 / DA-Code**：它们是 text-to-SQL / 数据科学基准，但都是"非仓库级、单条 SQL/脚本、有确定答案"；DAComp 首次把数据工程提升到仓库级 DAG 编排（平均 2000+ 行代码）。
- **vs DSBench / BLADE / DABStep**：同为数据分析基准，但要么是确定性答案、要么 schema 很小（10~12 列）；DAComp-DA 用分层 rubric 支持多路径开放式评测，并强调自主可视化。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个把数据智能"全生命周期"双轴化、并首创仓库级 DE 评测的基准。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多家开源/闭源模型、双框架、中英双版本，并有 judge 一致性验证。
- 写作质量: ⭐⭐⭐⭐ 任务定义与评测协议清晰，公式与统计表详尽，部分附录细节需查原文。
- 价值: ⭐⭐⭐⭐⭐ 提供了可诊断真实瓶颈的严格 testbed，对推动企业级数据智能体发展有直接意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](detecting_data_contamination_in_llms_via_in-context_learning.md)
- [\[ICLR 2026\] ASIDE: Architectural Separation of Instructions and Data in Language Models](aside_architectural_separation_of_instructions_and_data_in_language_models.md)
- [\[ICML 2026\] Nonparametric LLM Evaluation from Preference Data](../../ICML2026/llm_evaluation/nonparametric_llm_evaluation_from_preference_data.md)
- [\[ICLR 2026\] Rethinking LLM Evaluation: Can We Evaluate LLMs with 200× Less Data?](rethinking_llm_evaluation_can_we_evaluate_llms_with_200_less_data.md)
- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)

</div>

<!-- RELATED:END -->
