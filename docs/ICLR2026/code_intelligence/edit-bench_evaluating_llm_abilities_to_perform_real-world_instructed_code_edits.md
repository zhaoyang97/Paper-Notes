---
title: >-
  [论文解读] EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits
description: >-
  [ICLR 2026][代码智能][指令式代码编辑] EDIT-Bench 把近 500 名真实开发者在自研 VSCode 插件里写下的 in-the-wild 指令式代码编辑请求，转成 540 道带测试用例的难题，评测 40 个 LLM，发现这是一个连 SOTA 都只有 1 个模型过 60% 的硬骨头。
tags:
  - "ICLR 2026"
  - "代码智能"
  - "指令式代码编辑"
  - "真实世界数据"
  - "LLM 评测"
  - "上下文依赖"
  - "多语言"
---

# EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FtL9eEmU6v](https://openreview.net/forum?id=FtL9eEmU6v)  
**代码**: [https://github.com/waynchi/editbench](https://github.com/waynchi/editbench)（Leaderboard: [https://waynechi.com/edit-bench/](https://waynechi.com/edit-bench/)）  
**领域**: 代码智能 / Benchmark  
**关键词**: 指令式代码编辑、真实世界数据、LLM 评测、上下文依赖、多语言  

## 一句话总结
EDIT-Bench 把近 500 名真实开发者在自研 VSCode 插件里写下的 in-the-wild 指令式代码编辑请求，转成 540 道带测试用例的难题，评测 40 个 LLM，发现这是一个连 SOTA 都只有 1 个模型过 60% 的硬骨头。

## 研究背景与动机
- **领域现状**：指令式代码编辑（developer 高亮一段代码、用自然语言让 AI 直接改写）已成为 Copilot、Cursor、Continue 等 AI 编程助手与自动补全、聊天并列的主流交互模式。
- **现有痛点**：几乎没有 benchmark 直接评测这种能力。代码生成基准（HumanEval、MBPP）只测从零写代码；少数编辑相关数据集要么是标注者手写的模板化问题（CanItEdit、EditEval），要么来自 Leetcode/教学式题目（Aider Polyglot），都不反映真实软件开发的多样性。而 Chatbot Arena、Copilot Arena 这类"arena 式"真实评测虽然贴近实际，却要海量人类投票才能给一个新模型排名，成本极高。
- **核心矛盾**：真实编辑请求是**模糊、随意、上下文依赖**的——一句"fix this"配上高亮代码就指望模型读懂意图，而现有 benchmark 都是"修改某个明确函数到明确状态"的良定义任务，二者分布差距巨大。
- **本文目标**：构建一个**根植于真实使用**、且**自动可评（有测试用例）**的指令式代码编辑基准，逼模型像在真实 IDE 里那样综合利用用户指令、完整代码文件、高亮区域、光标位置等多源信息。
- **核心 idea**：**用真实插件采集 in-the-wild 数据**——自研一个模仿 Copilot/Cursor 编辑功能的开源 VSCode 插件，让用户用免费 SOTA 模型换取真实编辑数据，再由人类专家把这些真实请求**逐题打磨成带测试 harness 的难题**，从而兼顾"真实"与"可自动评分"。

## 方法详解

### 整体框架
EDIT-Bench 的构造是一条"真实采集 → 人工筛选 → 测试 harness 化 → 翻译扩充"的流水线：先用插件从 458 名用户收集 2672 条被采纳的真实编辑，逐层过滤到约 470 道有趣且有挑战性的问题，再由专家团队为其中 109 道写出可泛化的测试用例（EDIT-Bench-core），最后翻译到 5 种自然语言扩成 540 道（EDIT-Bench-complete），并在其上评测 40 个 LLM。

```mermaid
flowchart LR
    A[自研 VSCode 编辑插件] -->|458 用户| B[2672 条真实采纳编辑]
    B -->|过滤: 仅 Py/JS / 去重 / 去琐碎| C[~470 道有趣难题]
    C -->|专家写测试 harness + 双人 review| D[EDIT-Bench-core 109 道]
    D -->|GPT-4o 翻译注释到 5 种语言| E[EDIT-Bench-complete 540 道]
    E -->|pass@1| F[评测 40 个 LLM]
```

### 关键设计
**1. 用真实插件采集 in-the-wild 编辑数据：让数据自己"长"出真实分布。** 作者没有雇标注者凭空写题，而是开发了一个以指令式编辑为核心功能的开源 VSCode 插件并招募近 500 名用户：每次编辑时用户高亮一段代码、写一句任务描述，插件就记录下指令、代码上下文（高亮段、光标位置、前缀 prefix、后缀 suffix）以及模型响应和用户是否采纳。用户不拿报酬，而是换取免费 SOTA 模型访问权，这让采集到的指令天然带着真实开发的随意与多样——同一个 bug 可能被写成"fix this"、直接粘贴 error trace、或一段自然语言描述。整个采集经过 IRB 审查并提供隐私控制。

**2. 上下文依赖的问题设计：强迫模型综合多源信息而非只读指令。** EDIT-Bench 是首个把"用户指令 + 待编辑代码文件 + 高亮代码区域 + 光标位置"四要素组合进同一道题的编辑基准。真实指令常常模糊到只看指令无法判断意图，必须靠高亮区域和光标位置等上下文线索来锁定。代码上下文可以很长（≥10k 字符，完整文件中位数约 4.5k tokens），而高亮待改部分中位数仅 138 tokens，所以模型要在长上下文里精准定位并利用注释、高亮等线索。评测时模型被要求**重新生成整个文件**来完成编辑。

**3. 真实请求转可评测题：专家测试 harness + 双人 review。** 原始数据没有测试用例，无法自动评分。作者组建 5 人资深程序员团队（兼通涉及的自然语言与编程语言），为每题编写**环境配置 + 测试用例**：测试要忠于用户意图且能泛化到不同实现，遇到仍然过于模糊的题目直接剔除；用 GPT-4o/Sonnet 3.7 生成示例解供参考、筛查 PII，且每题由第二位标注者复核。值得一提的是，作者原想用 coding agent（如 Claude Code）自动造测试，但发现 agent 常退化成直接 pattern-match 源码这类劣质测试；不过 agent 在环境搭建（尤其 JS 的包/配置）上很可靠，于是只用它配合 conftest.py / jest-config.js 做环境 setup，测试逻辑仍由人写。

**4. 多语言翻译扩充：HumanEval-XL 式注释翻译。** core 的 109 道题原始覆盖 5 种自然语言（英、俄、中、波兰、西），分布不均。作者借鉴 HumanEval-XL，用 GPT-4o 把每题的注释翻译到其余语言，扩成 540 道、5 种自然语言（英、西、俄、中、葡）× 2 种编程语言（Python、Javascript）的 EDIT-Bench-complete，并请母语者（主要中文、西语）抽检翻译质量；对比 GPT-4o-nano/mini 与 Google Translate 后选定 GPT-4o。

## 实验关键数据

### 与现有编辑基准对比

| Benchmark | # Problems | 来源 | # NL | 指令长度 | # PL | 代码上下文长度 | 高亮 |
|---|---|---|---|---|---|---|---|
| CanItEdit | 105 | 标注者 | 1 | 140 ± 105 | 3 | 1309 ± 1116 | 否 |
| EditEval | 194 | 标注者 | 1 | 99.9 ± 49.3 | 1 | 258 ± 185 | 否 |
| Aider Polyglot | 225 | 编程练习 | 1 | 606 ± 885 | 5 | 6184 ± 6452 | 否 |
| **EDIT-Bench** | **540** | **In-the-wild** | **5** | **238 ± 738** | **2** | **5642 ± 7567** | **是** |

EDIT-Bench 是唯一来自真实世界、唯一支持高亮、指令与代码长度方差都最大的基准；Python 题含 74 个 unique imports，是其他基准（15~25）的 3 倍以上。

### 主实验（40 个 LLM 的 pass@1）
- 仅 **1/40** 模型过 60%：**claude-sonnet-4 达 66.67%**，居首。
- 闭源整体强于开源：top 15 里只有 4 个开源，bottom 15 全是开源；最强开源 **glm-4.6 为 56.48%**，kimi-k2-0905、deepseek-chat-v3.1 紧随。
- 反直觉：**gpt-5（medium reasoning）落后于 gpt-5-mini**——失败案例多是缩进格式、边界 case 这类"简单活"。

### 消融实验（上下文信息的影响，top-7 模型族最佳模型）

| Model | Code Only | +Highlight | +Cursor | +Highlight +Cursor |
|---|---|---|---|---|
| claude-sonnet-4 | 62.41 | 64.81 (+2.40) | 63.15 (+0.74) | 64.26 (+1.85) |
| deepseek-chat-v3.1 | 51.48 | 54.26 (+2.78) | 53.15 (+1.67) | 52.78 (+1.30) |
| gemini-2.5-flash | 52.59 | 52.96 (+0.37) | 52.41 (−0.18) | 56.30 (+3.71) |
| kimi-k2-0905 | 54.63 | 56.48 (+1.85) | 52.22 (−2.41) | 58.15 (+3.52) |
| glm-4.6 | 52.96 | 56.48 (+3.52) | 52.22 (−0.74) | 44.81 (−8.15) |
| o3-mini | 60.00 | 56.85 (−3.15) | 59.26 (−0.74) | 55.19 (−4.81) |
| qwen3-coder | 56.48 | 53.89 (−2.59) | 56.48 (+0.00) | 53.89 (−2.59) |

加高亮代码让 5/7 模型提升（最高约影响 8%），但加光标位置结果喜忧参半（glm-4.6 双加反而 −8.15）。

### 关键发现
- **难易差异巨大**：按"≤20 个模型解出"划为 hard，easy↔hard 平均差距 **59.3%**（std 10.6%）；hard 题指令更短（约短 5 倍）但高亮代码略长，逼模型综合推理而非照指令照搬。
- **任务类别差异**：四类功能编辑分布为 addition 43% / modification 27% / fix 22% / optimization 8%；模型在 **bug fixing 最强（avg 52.2%）**，在 **optimization（44.6%）和 feature addition（39.6%）最弱**；不同模型擅长类别不同（claude-sonnet-4 除 optimization 外每类第一）。
- **与现有基准弱相关**：与 Aider Polyglot 仅 r=0.24（p=0.06）、与 Chatbot Arena 编码子集 r=0.11（p=0.01），说明真实数据捕捉到一批独特的难编辑任务。

## 亮点与洞察
- **"真实采集 + 可自动评"两全**：用免费模型访问权换真实数据，再人工 harness 化，绕开了 arena 式评测需海量投票的成本，又保留真实分布——这是数据采集机制上的巧思。
- **明确指出 agent 不擅长写测试、却擅长配环境**，于是把测试逻辑留给人、环境 setup 交给 agent，是很实用的人机分工经验。
- **上下文消融定量化**：高亮代码普遍有用、光标位置却收益不稳，给"该往编辑模型喂什么上下文"提供了实证依据。
- **多语言 + 长上下文 + 高亮**三者组合，是现有编辑基准都没覆盖的真实维度。

## 局限与展望
- **规模偏小**：core 仅 109 道独立题，扩到 540 主要靠注释翻译，独立问题量有限；作者计划随插件持续采集扩充语言与题量。
- **仅 Python/JS**：真实采集中 PHP（18%）、HTML（7%）等都被舍弃，覆盖面受限。
- **代表性存疑**：作者承认不确定这些题在多大程度上涵盖了所有真实用例。
- **翻译引入的扩充**可能让同一题的多语言版本相关性过高，稀释了"独立难题"的有效数量。
- 未来计划探索自动化流程，更顺滑地把真实数据转成 benchmark 题。

## 相关工作与启发
- **静态代码基准**（HumanEval、MBPP）与 **live 基准**（LiveCodeBench）测从零生成，本文测编辑。
- **编辑相关基准**：CanItEdit、EditEval（标注者手写）、Aider Polyglot（编程练习）、CodeEditorBench（竞赛题）——本文以"in-the-wild + 高亮 + 多语言"区别于它们。
- **真实数据评测**：Copilot Arena（真实补全）、Chatbot Arena（真实聊天偏好）、SWE-Bench 系列（真实 issue 但需 agentic 多文件、单一自然语言）——EDIT-Bench 补上"单步指令式编辑"这一真实场景。
- **启发**：把"真实用户交互痕迹"系统化转为可自动评测的 benchmark，是一条可推广到补全、重构、调试等其他交互模式的方法论；人机分工（人写测试、agent 配环境）也值得其他 benchmark 构造借鉴。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首个把 in-the-wild 指令式编辑（含高亮+光标+多语言）转为自动可评 benchmark 的工作，数据采集机制有真创新。
- **实验充分度**: ⭐⭐⭐⭐ 评测 40 个模型、含上下文消融、难易/类别拆分、与两个现有 leaderboard 相关性分析，覆盖全面。
- **写作质量**: ⭐⭐⭐⭐ 动机清晰、构造流程透明、表格与发现解读到位。
- **价值**: ⭐⭐⭐⭐ 切中 AI 编程助手主流交互模式的评测空白，leaderboard 可持续更新，对模型训练与评测都有实际指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning](codesense_a_real-world_benchmark_and_dataset_for_code_semantic_reasoning.md)
- [\[ACL 2026\] AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor](../../ACL2026/code_intelligence/automonitor-bench_evaluating_the_reliability_of_llm-based_misbehavior_monitor.md)
- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](../../ACL2026/code_intelligence/referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](../../ACL2026/code_intelligence/logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2025\] CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](../../ACL2025/code_intelligence/compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)

</div>

<!-- RELATED:END -->
