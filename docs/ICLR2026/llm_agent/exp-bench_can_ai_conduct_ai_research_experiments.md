---
title: >-
  [论文解读] EXP-Bench: Can AI Conduct AI Research Experiments?
description: >-
  [ICLR 2026][LLM Agent][端到端实验] EXP-Bench 从 51 篇 NeurIPS/ICLR 2024 顶会论文及其开源代码里半自动抽取出 461 个"完整 AI 研究实验"任务，逼着 Agent 走完"提假设→设计实验→写代码→真跑→下结论"全流程，结果发现当下最强 Agent 完整跑通可执行实验的成功率仅 **0.5%**。
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "端到端实验"
  - "研究 Agent"
  - "半自动数据构建"
  - "合取评测"
  - "可执行性验证"
---

# EXP-Bench: Can AI Conduct AI Research Experiments?

**会议**: ICLR 2026  
**代码**: [Just-Curieous/Curie/benchmark/exp_bench](https://github.com/Just-Curieous/Curie/tree/main/benchmark/exp_bench)  
**领域**: LLM Agent / AI 科研自动化 / Benchmark  
**关键词**: 端到端实验、研究 Agent、半自动数据构建、合取评测、可执行性验证  

## 一句话总结
EXP-Bench 从 51 篇 NeurIPS/ICLR 2024 顶会论文及其开源代码里半自动抽取出 461 个"完整 AI 研究实验"任务，逼着 Agent 走完"提假设→设计实验→写代码→真跑→下结论"全流程，结果发现当下最强 Agent 完整跑通可执行实验的成功率仅 **0.5%**。

## 研究背景与动机
**领域现状**：AI 科研本质是数字化的，特别适合被 LLM Agent 自动化；现有 Agent 已经能做文献综述、假设生成、代码生成等单点任务。但真正的实证研究需要的是端到端、可复现的严谨实验，远不止这些孤立能力。

**现有痛点**：已有 benchmark 大多只覆盖科研流程的某个切片——要么测抽象推理/实验设计（BoxingGym、Lab-Bench），要么测代码片段生成或事后数据分析（SciCode、ScienceAgentBench、DiscoveryBench），要么在 Kaggle 这类受限环境里测 ML 调参（MLE-Bench、DSBench）。RE-Bench 只有 7 个手工任务，规模太小；PaperBench、CORE-Bench 虽源自论文，但聚焦"跑已有脚本/标准分析"这类定义良好的子任务，**都没真正捕捉到一条完整、迭代的 AI 研究实验链路，也没给出可规模化构造这类任务的方法**。

**核心矛盾**：要做这种高保真任务就得人工从论文+代码里挖实验细节，但论文只呈现打磨过的最终结论，省略了中间步骤；关键条件、数据预处理又散落在正文、附录、补充材料和庞大代码库里——纯手工策展既费力又无法规模化。

**本文目标**：建一个能逐步评估并引导 Agent 走完整个实验流水线的 benchmark，同时配一套能规模化生产这类任务的管线。**核心 idea**：用**半自动数据构建管线**（多模态抽取 + 轻量人工核验），把"论文叙事 + 开源代码"这对天然耦合的研究产物，反向工程成结构化、可执行、可细粒度打分的实验任务。

## 方法详解

### 整体框架
EXP-Bench 由两部分组成：一个**数据集规格**（每个任务 = 问题陈述 + 来自原始研究产物的 ground-truth 解）和一条**三阶段半自动构建管线**。构建侧从顶会论文筛选源材料，抽出研究任务与对应实现脚本，再在干净 Docker 里执行验证；评测侧用"LLM-as-judge + 代码执行验证器"对 Agent 的设计/实现/执行/结论四个阶段分别打分，并用合取指标聚合。

```mermaid
flowchart LR
    A[顶会论文+开源代码] --> B[Stage1 源筛选<br/>引用数/Star/Fork]
    B --> C[Stage2.1 抽研究任务<br/>OCR+多模态+多轮检索]
    C --> D[Stage2.2 抽实现脚本<br/>工具增强Agent定位代码链]
    D --> E[Stage3 Docker执行验证<br/>Monitor查作弊+对比结论]
    E -->|失败回退| D
    E --> F[461任务/12737子任务]
    F --> G[Agent: 设计→实现→执行→结论]
    G --> H[Judge: D/I/E/C + 合取指标]
```

### 关键设计

**1. 任务三件套规格：把"论文+代码"压成可评测的实验单元。** 每个任务的输入（问题陈述）给 Agent 三样东西——从论文实验中提炼的**研究问题**、指导实验路径的**高层方法描述**、以及部分脚本被遮蔽（mask）的**代码仓库**；对应的 ground-truth 解则包含三部分：指明关键变量/常量/流程的**实验设计**、以 `git diff` 形式核验的**代码改动**、以及直接回答研究问题的**最终结论**。这套"输入三件套 + 答案三件套"既保留了真实科研的完整工作流，又让每一环都能被独立打分。

**2. 两阶段抽取：先挖任务、再挖实现。** Stage 2.1 抽研究任务时先用 OCR 索引 PDF，对表格/图/跨页元素用多模态 LLM 解读成结构化文本，再做多轮抽取——第一轮 RAG 检索找高层研究 takeaway，第二轮在子小节粒度做语义抽取并把每段分类为"实现上下文"或"候选研究问题"，最后回查全文（含附录）补回遗漏的 setup 约束。Stage 2.2 则把任务交给一个工具增强 Agent（能读 PDF、开终端、上网），在完整代码库里做**目标条件搜索**：定位实现指定方法与预期产物的那条脚本链，输出"所需脚本列表 + 运行说明"，再用 AST 追踪把脚本链解析成自然语言的逐步实现要求，作为评测实现正确性的 ground truth。

**3. 执行驱动的验证与遮蔽：杜绝捷径与造假。** Stage 3 先用 LLM Monitor 审查 Agent 日志，检测三类违规——是否直接读了论文 PDF、是否做了 `git checkout`/切分支等操作、是否用了假数据/硬编码/占位结果而非真跑实验；通过后在干净 Docker 容器里从零复现脚本，把输出与原论文结论用 LLM 对比，失败则回退细化。最终任务入库时会遮蔽 README、相关脚本等文件（通过脚本化 git 操作递归处理子模块），逼 Agent 真去推理而非抄答案。

**4. 合取评测指标：拆穿"看似对了"的脆弱性。** 评测用 LLM-judge（o3-mini）给出 **D**（设计正确率）、**I**（实现正确率）、**C**（结论正确率），用代码执行验证器给出 **E**（可执行性）。单看 C 和 E 方差很大——Agent 能编出"听起来合理但无实验支撑"的结论，假实现也可能恰好跑通。于是引入合取指标 $I\cdot E$（实现既对又能跑）、$C\cdot D$（结论建立在合理设计上）、以及 $\text{All}\checkmark$（D/I/C 全对）和 $\text{All}\cdot E\checkmark$（再加可执行）。在只跑执行验证的子集上，单独 Monitor 检查 M 平均得分 20.6%，叠加 D+C 骤降到 3.7%，再加 I 到 0.4%，最后加 E 仅剩 **0.2%**——合取评测把端到端正确性的脆弱暴露无遗。

## 实验关键数据

### 主实验表格
在全部 461 个任务上评测 OpenHands(OH) 与 IterativeAgent(IA) 两个 Agent 框架配多种 LLM（D/I/E/C 单位 %，All·E✓ 为完整可执行成功率）：

| Agent | Model | D | I | E | I·E | C | All✓ | **All·E✓** |
|---|---|---|---|---|---|---|---|---|
| OpenHands | o3-mini | 18.4 | 20.3 | 15.0 | 2.9 | 21.0 | 1.4 | **0.5** |
| OpenHands | Claude-3.7 Sonnet | 16.0 | 35.0 | 33.2 | 14.9 | 13.4 | 0.7 | **0.4** |
| OpenHands | Amazon Nova Pro | 18.2 | 19.5 | 26.8 | 0.0 | 15.7 | 0.0 | **0.0** |
| OpenHands | Claude-3.5 Haiku | 20.6 | 26.2 | 9.3 | 1.3 | 13.8 | 0.0 | **0.0** |
| OpenHands | DeepSeek R1 | 6.8 | 10.0 | 0.7 | 0.0 | 2.4 | 0.0 | **0.0** |
| IterativeAgent | Claude-3.5 Haiku | 6.4 | 20.6 | 25.2 | 5.4 | 2.2 | 0.0 | **0.0** |
| IterativeAgent | Amazon Nova Pro | 0.1 | 10.0 | 18.1 | 0.0 | 0.3 | 0.0 | **0.0** |

最强配置 OH+o3-mini 完整可执行成功率也只有 0.5%，All✓ 仅 1.4%。

### 消融实验表格
合取指标逐级收紧时（仅在执行子集上），平均得分的崩塌轨迹：

| 评测条件 | 平均得分 |
|---|---|
| 仅 Monitor 检查 (M) | 20.6% |
| + 设计与结论 (M·D·C) | 3.7% |
| + 实现正确性 (·I) | 0.4% |
| + 执行验证 (·E) | **0.2%** |

### 关键发现
- **数据规模与覆盖**：461 任务 / 12,737 个可独立打分子任务，源自 51 篇论文（NeurIPS 2024 占 53%、ICLR 2024 占 47%），横跨 CV、NLP、RL、生成模型等多个子领域。
- **三类系统性失败**：16.1% 的设计变量被错分类；39.7% 缺失关键实现组件；执行阶段 29.4% 环境/依赖配错、23.8% 脚本级报错。
- **行为差异**：OH 系列常"早停"——产出看似合理但没真跑实验的回答，拿到虚高的设计/实现分；IA 系列几乎跑满 40 分钟时限但效率低。运行时长/成本与最终表现几乎不相关。
- **RL 是相对亮点**：多个 OH 模型在 RL 类任务的实现正确率 I 可达约 41%（36 任务均值），明显高于其他类别。

## 亮点与洞察
- **把"论文+代码"当成天然 ground truth**：已发表、经同行评审、有开源实现的研究本身就是完整实验工作流的现成范例，反向工程比从零造任务可信得多，也把人工核验降到"轻量一致性检查"。
- **合取评测是这篇的方法论贡献**：单点指标会被"听起来对/恰好能跑"骗过，$I\cdot E$、$C\cdot D$ 这类合取形式显著降低方差、过滤过度给分，给出更稳健可判别的信号——这套思路对任何端到端 Agent 评测都通用。
- **0.5% 这个数字本身极具冲击力**：它清楚地把"做单点任务"和"做完整科研"之间的鸿沟量化出来，也为后续 Agent 改进（规划能力、实现完整度、环境鲁棒性）提供了明确靶点。

## 局限与展望
- **半自动仍需人工**：管线虽把人工降到轻量核验，但仍依赖人审最终任务内容，且构建管线本身是反复手工试错调出来的，迁移到新会议/新领域时这部分成本未必能省。
- **执行评测成本高**：因执行耗时，只有任务子集做了可执行性验证（#E 列各模型从 56 到 420 不等），E 相关指标的统计基础不完全对齐全集。
- **依赖原实现作为唯一真值**：当原代码本身有歧义或多种合理实现时，以单一脚本链为 ground truth 可能低估 Agent 的合理但不同的解法。
- **可执行性 E 的高方差**：假实现/mock 也可能跑通，引入高估偏差——作者靠合取指标缓解，但单独 E 仍不可靠。
- **展望**：作者明确把 EXP-Bench 定位为"训练能自动化核心科研环节的 Agent"的大规模数据来源，半自动管线为后续规模化生产留了口子。

## 相关工作与启发
- **科学推理类**（BoxingGym、AAAR、Lab-Bench）：测抽象推理/静态设计，不做真实验。
- **科学编码类**（SciCode、BLADE、DiscoveryBench、ScienceAgentBench）：聚焦代码片段或事后分析，把编码/分析从迭代实验语境里孤立出来。
- **ML benchmark 类**（DSBench、MLE-Bench、RE-Bench、ML-Gym、Curie、CORE-Bench、PaperBench）：要么受限于 Kaggle 环境/小规模/简化指标，要么只测"跑已有脚本"这类定义良好的子任务。
- **启发**：EXP-Bench 的差异点在于"端到端 + 逐步 + 可规模化构造"三者兼得；其合取评测与执行验证的组合，值得任何想做"真能干活"而非"看起来能干活"的 Agent 评测者借鉴。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个真正覆盖端到端 AI 研究实验、且给出可规模化半自动构造方法的 benchmark，合取评测是扎实的方法论贡献。
- **实验充分度**: ⭐⭐⭐⭐ 461 任务 / 12,737 子任务、多 Agent×多 LLM、含成本-时间与指标稳定性分析；执行评测仅覆盖子集略有遗憾。
- **写作质量**: ⭐⭐⭐⭐ 动机—管线—评测逻辑清晰，图表（Fig.1 流程、Fig.3 分布、Table 1 主表）信息密度高，failure 归因具体。
- **价值**: ⭐⭐⭐⭐⭐ 0.5% 的成功率一锤定音地标定了"AI 自动做科研"的真实差距，为下一代研究 Agent 提供了高保真靶场与训练数据来源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Framework for Studying AI Agent Behavior: Evidence from Consumer Choice Experiments](a_framework_for_studying_ai_agent_behavior_evidence_from_consumer_choice_experim.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](../../ACL2025/llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICLR 2026\] NetArena: Dynamic Benchmarks for AI Agents in Network Automation](netarena_dynamic_benchmarks_for_ai_agents_in_network_automation.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?](../../ACL2025/llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)

</div>

<!-- RELATED:END -->
