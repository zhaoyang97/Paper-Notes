---
title: >-
  [论文解读] CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning
description: >-
  [ICLR2026][代码智能][代码推理] CodeSense 是第一个面向真实世界软件工程的细粒度代码语义推理 benchmark：作者从 744 个 Python/C/Java GitHub 项目里跑测试、抓执行轨迹，自动构造出语句级/代码块级/函数级的执行值与程序属性（循环、指针别名、分支）的 ground truth，共 4483 个样本，评测 14 个 SOTA LLM 后发现它们连单条真实语句的算术和 API 调用都常常算不对。
tags:
  - "ICLR2026"
  - "代码智能"
  - "代码推理"
  - "真实世界代码"
  - "执行追踪"
  - "细粒度语义"
  - "LLM 评测"
---

# CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=ehXVDJm0PS](https://openreview.net/forum?id=ehXVDJm0PS)  
**代码**: https://codesense-bench.github.io/  
**领域**: 代码智能 / Benchmark / 代码语义推理  
**关键词**: 代码推理, 真实世界代码, 执行追踪, 细粒度语义, LLM 评测

## 一句话总结
CodeSense 是第一个面向真实世界软件工程的细粒度代码语义推理 benchmark：作者从 744 个 Python/C/Java GitHub 项目里跑测试、抓执行轨迹，自动构造出语句级/代码块级/函数级的执行值与程序属性（循环、指针别名、分支）的 ground truth，共 4483 个样本，评测 14 个 SOTA LLM 后发现它们连单条真实语句的算术和 API 调用都常常算不对。

## 研究背景与动机
**领域现状**：评测代码 LLM 的 benchmark 已经很多，但大致分两类。一类是代码生成类（HumanEval+、LiveCodeBench、BigCodeBench、CodeBenchGen），数据多来自竞赛题或合成片段；另一类是推理类（CruxEval、REval、CodeMind），主要做函数级的输入/输出预测，代码也偏短、偏合成。还有 SWE-Bench、KGym 这类用真实代码的，但只看某个端到端任务（如给 GitHub issue 生成补丁）的成败。

**现有痛点**：代码生成 benchmark 测不出模型是否真的"理解"代码在干什么；输入/输出预测这类粗粒度推理只给一个最终结论，无法定位模型到底在哪一步——某条语句、某个 API、某个循环——理解错了；而 SWE-Bench 这种任务级评测，模型成功了你也说不清它是真懂语义还是靠模式匹配蒙对。换句话说，没有 benchmark 能在**细粒度**上、用**真实世界代码**去探针模型的语义推理能力。

**核心矛盾**：软件工程任务（测试输入生成、漏洞检测、缺陷定位、程序修复）的底层共同需求，是对程序执行行为的细粒度语义理解——比如要触发某段危险代码，得知道算术运算 $n = input \times 23$ 和分支条件 $3465 \ge n \ge 2287$ 的语义，从而反推出 $input=120$。但现有评测既不细、又不真实，无法度量这种能力。

**本文目标**：构造一个覆盖真实项目、多语言、细粒度（语句 / 代码块 / 函数三个层次 + 循环 / 指针 / 分支三类程序属性）的语义推理 benchmark，并配一套能自动抽 ground truth 的执行追踪框架。

**切入角度**：代码语义在编程语言理论里有明确定义——"给定输入，这段代码的输出值是什么"（operational semantics 描述逐步执行，axiomatic semantics 用逻辑断言描述属性）。作者把这套形式化定义落成一系列可自动验证的预测任务：让模型预测某语句/代码块的执行值、循环迭代次数、两个指针是否别名、某分支是否走到。ground truth 不靠人标，而是**真跑代码、记轨迹**得到。

**核心 idea**：用真实项目的动态执行轨迹当 ground truth，把"理解代码"拆成一谱系细粒度的、可自动判分的语义预测任务，从而精确暴露 LLM 在代码推理上到底弱在哪里。

## 方法详解

### 整体框架
CodeSense 不是一个模型，而是一条"数据采集 → 任务定义 → 评测"的 benchmark 构造管线。整体分三步：(1) 从 GitHub 收集 744 个 Python/C/Java 真实项目，用对应语言的工具链把它们**构建、跑测试、记录执行轨迹**（每条语句的变量值、数据类型、函数调用名、内存地址）；(2) 用静态+动态分析从轨迹里**自动抽取 ground truth**，过滤掉无意义的函数，得到 2125 个 Python、876 个 C、875 个 Java 唯一函数，并据此定义一谱系细粒度推理任务，curate 出 4483 个带标准答案的样本；(3) 用自然语言 prompt 把这些任务喂给 14 个 SOTA LLM，用 exact-match accuracy 判分，围绕 6 个研究问题做分析。

由于这是一篇 benchmark/数据集论文，核心价值在"任务怎么定义、ground truth 怎么来"，不存在一个可画框架图的模型 pipeline，故下面按任务谱系和数据构造两条线讲清楚。

### 关键设计

**1. 细粒度语义推理任务谱系：把"理解代码"拆成可自动判分的小题**

针对"粗粒度 I/O 预测定位不到模型哪一步出错"这个痛点，作者按粒度从粗到细设计了四类任务，每类都有明确的输入和可验证的标准答案：

- **Task 1 代码块级语义**：给一段语句块（从函数入口开始、逐步增大到整个函数），给输入预测输出，或给输出预测输入。函数级 I/O 预测只是它的特例。
- **Task 2 语句级语义**：把语句按编程语言语义分成五类——算术、布尔表达式、API/函数调用、变量赋值、常量赋值——随机抽一条语句，给输入预测其输出，从而看模型对哪类语句最弱。
- **Task 3 函数内程序属性**：聚焦三类 SE 关键属性——3-1 循环（预测迭代次数、循环内变量值、循环后变量值）；3-2 指针（给函数和输入，问某程序点两个指针是否别名指向同一内存，只在 C 上做）；3-3 分支（给函数、输入和某条件分支位置，问该分支走 true 还是 false）。
- **Task 4 语义近似**：很多 SE 任务不需要精确值，只需近似——如不必算出 $input=120$，只要说"100–150 之间的整数能触发危险代码"即可。作者为各数据类型定义了一组抽象值（abstract value，从一段具体值区间映射到一个抽象标签），评测模型能否预测对抽象值。

这套谱系的巧妙处在于，每道题都对应 SE 实践里的真实需求（循环次数关系到死循环/DoS 检测，分支预测关系到测试输入生成，指针别名关系到内存安全），又因为答案是真跑出来的，可以 exact-match 自动判分。

**2. 多语言真实项目的执行追踪框架：让 ground truth 来自"真跑"而非人标**

针对"细粒度 ground truth 极难构造"这个痛点，作者为三种语言各自搭了一套构建+追踪工具链，记录每条语句执行时的值、类型、调用名和内存地址：

- **Python**：从 PyPIbugs 数据集的 1489 个仓库里，剔除没测试或四年未更新的，留 544 个；装好依赖后用 `pytest` 跑测试，用 `PySnooper` 追踪。
- **C**：用 OSS-Fuzz 里 100 个项目，在 docker 里用各项目的 fuzzing harness 构建并 fuzz，追踪框架建在 GNU 调试器 GDB 之上。
- **Java**：用 SF110 数据集的 100 个项目，用 EvoSuite 生成并运行测试，追踪工具建在 Java Debugger 之上。

这套框架是论文强调的可复用资产：它把"抓真实代码的细粒度语义"自动化了，意味着 benchmark 可以持续扩充、缓解数据泄漏（因为可以不断纳入新项目），也能为未来的模型后训练提供语义信号 ground truth。

**3. 数据过滤与原子类型约束：保证样本"有意义且可判分"**

从整程序轨迹里，作者按执行轨迹中函数的进入/退出点切出唯一函数，然后过滤掉：只含注释的、长到塞不进上下文的、函数体无实际功能的（如只有 `return 0`、`printf("...")`，或只是包装另一个已测函数的 `void myfunc(){ func(); }`）。最终得到 2125/876/875 个 Python/C/Java 唯一函数，长度从 3 到 516 行不等。

为了让答案可被 exact-match 干净判分，作者把 ground truth 限定在**原子数据类型**（int、float、str、bool、list、pointer、double、dict、tuple 等）——这是探针真实代码语义的第一步。即便只测原子类型，实验也显示模型已经很吃力。最终 curate 出 4483 个样本（Python 3050、C 1359、Java 74），分布见下方实验表。

## 实验关键数据

### 主实验
评测 14 个 SOTA LLM（8 个 reasoning、6 个 non-reasoning，参数 7B–14B，含 GPT-4o-mini、Claude 3.5 Sonnet、Gemini 1.5、Llama-3.1、Qwen2.5、Phi、DeepSeek-R1 蒸馏系列等），用 vLLM 推理，答案放在 `<ans></ans>` 标签内与 ground truth 做 exact match。下表为各任务样本量与代表性结论：

| 任务 | 样本量（Py/C/Java） | 代表性发现 |
|------|------|------|
| Task 1 代码块 I/O | 1860 / 731 / – | Claude 3.5、GPT-4o-mini 在 C 单语句块上准确率 <30%，Python 无一模型超 50% |
| Task 1 函数 I/O | 308 / 94 / 74 | 块从 1→3 句准确率持续下降；Claude 3.5 在 3 句 Python 块仅约 20%、C <10% |
| Task 2/4 语句级 | 545 / 485 / – | 算术、API 调用最难；布尔与常量赋值相对好 |
| Task 3-1/4 循环 | 105 / – / – | 循环后变量值最难，迭代次数最易 |
| Task 3-2 指针别名 | – / 49 / – | 二分类，部分开源模型 <50%（不如随机猜） |
| Task 3-3 分支 | 232 / – / – | 二分类，指针别名比分支执行更好预测 |
| **合计** | **3050 / 1359 / 74 = 4483** | Claude 3.5 综合最佳；输入预测（逆向语义）最难 |

整体结论非常一致：当前 LLM 对细粒度代码语义普遍薄弱，连真实代码里的**单条算术语句和 API 调用**都常常算不对。其中输入预测（给输出反推输入，即逆向 operational semantics）是最难的任务，最强的 Claude 3.5 在小函数上也只有约 12% 准确率。

### 消融实验
作者通过 6 个研究问题（RQ）做了多维度分析，等价于一组"换条件看掉多少点"的消融：

| 配置 / 变量 | 关键发现 | 说明 |
|------|---------|------|
| 代码块增大 1→3 句 (RQ1) | 准确率持续下降 | 模型既难推单语句、又难跨语句跟踪变量状态 |
| 语句类型 (RQ2) | 算术/API 最难，布尔/常量赋值较好 | 给 prompt 补 API 定义也提升有限 |
| 程序属性 (RQ3) | 循环后值最难；指针别名优于分支 | 部分开源模型在二分类上低于随机 |
| Prompting 策略 (RQ4) | shot 越多越好；RAG 式相关示例有效；CoT 帮助有限 | CoT 对循环这类多步任务有些帮助，对语句预测几乎无用 |
| 抽象值 vs 具体值 (RQ5) | 给 3-shot 示例时抽象值预测优于具体值 | 但只给定义不给示例时，模型无法把定义套用到查询 |
| 编程语言 (RQ6) | 输出预测 Java/Python 优于 C | C 更底层、训练数据更少；但 Python 的输入预测最差 |

### 关键发现
- **模型靠模式匹配而非真懂语义**：当代码里出现 `p = q`，模型在零样本下也能答对"p 和 q 是否别名"；遇到 `for i in range(100):` 也能报出循环次数 100。这说明它们把自然语言语义问题和某些表层代码模式做了关联，而非真正执行推理——一旦没有这种显式模式，准确率就崩。
- **逆向语义最难**：输入预测远难于输出预测，揭示 LLM 对"从结果反推原因"这种逆向 operational semantics 几乎无能为力。
- **In-context learning 的有效条件**：当 prompt 定义新概念（如抽象值）或提供高度相关示例（同类型语句、同函数内不同循环的 RAG 式示例）时最有效；纯 CoT "think step by step" 对简单语句预测帮助甚微。

## 亮点与洞察
- **ground truth 来自真跑，而非人标或合成**：用 PySnooper/GDB/Java Debugger 三套追踪框架记录真实项目执行轨迹，把"理解代码"变成可 exact-match 自动判分的小题——这是 benchmark 能做到细粒度又规模化的关键，也天然缓解数据泄漏（可持续纳入新项目）。
- **细粒度定位模型弱点**：相比 SWE-Bench 只给任务成败，CodeSense 能告诉你模型是栽在"算术语句"还是"API 调用"还是"循环后值"上，这种诊断力对指导后训练很有价值。
- **抽象值任务是务实折中**：承认精确值预测过难，转而评测近似语义（区间/抽象标签），既贴合 SE 实践（很多任务只需近似），又给出了一个更可达的中间目标。
- **可迁移思路**：用动态执行轨迹自动生成训练/评测信号的做法，可推广到任何"行为可被运行时观测"的任务——如把循环不变量、数据依赖也纳入轨迹标注，用于代码模型的语义后训练。

## 局限与展望
- **只覆盖原子数据类型**：ground truth 限定在 int/float/str/list/pointer 等原子类型，真实代码里大量复杂对象、自定义类型的语义推理还没被评测，结论的覆盖面有限。
- **语言间样本极不均衡**：Java 仅 74 个样本（且只有函数级 I/O，没有语句/循环/指针任务），C 没有代码块以外的部分任务，跨语言比较的某些结论（如"Java 比 C 易"）建立在不对等的任务子集上，需谨慎解读。
- **exact-match 判分可能偏严**：对浮点、字符串等输出，精确匹配可能把语义正确但格式不同的答案判错，可能低估模型真实能力（作者也部分用抽象值缓解）。
- **依赖测试覆盖**：轨迹来自项目自带或自动生成（EvoSuite/fuzzing）的测试，未被测试覆盖的路径/分支无法采到 ground truth，benchmark 的代码路径分布受限于测试质量。
- **改进方向**：把追踪框架扩到更多语言和复合类型；引入"语义错误归因"指标而非只看 exact match；用这套 ground truth 做模型后训练并回测 benchmark 验证语义信号的增益。

## 相关工作与启发
- **vs CruxEval / CruxEval-X**: 它们在合成 Python（及其 LLM 翻译的多语言版本）上做函数级 I/O 预测；CodeSense 用真实 Python/C/Java 代码，并把推理粒度细化到语句、代码块和循环/指针/分支属性，定位能力更强。
- **vs REval / CodeMind**: REval 在 ClassEval/HumanEval 上做分支预测，CodeMind 在已有 benchmark 上做输出预测和代码合成；两者数据仍偏合成/教学，CodeSense 强调真实世界项目 + 多语言 + 可自动扩充的执行追踪框架。
- **vs SWE-Bench / KGym**: 它们用真实代码但只评端到端任务（补丁生成、内核崩溃修复）的成败，测不出细粒度语义理解；CodeSense 把语义理解本身拆成可验证的子任务，互补性强。

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一个用真实多语言项目执行轨迹构造细粒度语义推理 ground truth 的 benchmark，任务谱系设计扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个模型 × 6 个 RQ × 多语言/多任务，分析维度丰富且结论自洽
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰、动机贴合 SE 实践；图多依赖附录，正文部分结论需查图
- 价值: ⭐⭐⭐⭐⭐ 诊断力强、框架可复用、可持续扩充，对代码模型评测与后训练都有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits](edit-bench_evaluating_llm_abilities_to_perform_real-world_instructed_code_edits.md)
- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](../../ACL2026/code_intelligence/referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](../../ACL2026/code_intelligence/logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2025\] CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](../../ACL2025/code_intelligence/compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)
- [\[ICLR 2026\] Code2Bench: Scaling Source and Rigor for Dynamic Benchmark Construction](code2bench_scaling_source_and_rigor_for_dynamic_benchmark_construction.md)

</div>

<!-- RELATED:END -->
