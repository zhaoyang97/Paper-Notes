---
title: >-
  [论文解读] AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms
description: >-
  [ICML 2026][代码智能][verified code generation] AlgoVeri 构建了一个跨 Dafny、Verus、Lean 严格对齐的经典算法 verified code generation 基准，显示当前 LLM 在复杂全局不变量、系统级约束和显式证明搜索上仍有巨大缺口，尤其是 Lean 与 Verus 的成功率远低于 Dafny。
tags:
  - "ICML 2026"
  - "代码智能"
  - "verified code generation"
  - "形式化验证"
  - "Dafny"
  - "Verus"
  - "Lean"
  - "算法基准"
---

# AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms

**会议**: ICML 2026  
**arXiv**: [2602.09464](https://arxiv.org/abs/2602.09464)  
**代码**: https://github.com/haoyuzhao123/algoveri  
**领域**: 代码智能 / 形式化验证 / Verified Code Generation  
**关键词**: verified code generation、形式化验证、Dafny、Verus、Lean、算法基准  

## 一句话总结
AlgoVeri 构建了一个跨 Dafny、Verus、Lean 严格对齐的经典算法 verified code generation 基准，显示当前 LLM 在复杂全局不变量、系统级约束和显式证明搜索上仍有巨大缺口，尤其是 Lean 与 Verus 的成功率远低于 Dafny。

## 研究背景与动机
**领域现状**：LLM 已经能从自然语言生成代码，但“能跑过单元测试”并不等于程序对所有输入都正确。形式化验证要求模型生成实现、规格和证明工件，让 verifier 证明代码满足合约，因此比 HumanEval/MBPP 这类测试驱动基准更严格。

**现有痛点**：已有 vericoding 基准大多只覆盖单一工具，或者把不同语言中的不同任务拼在一起。这样得到的分数不可比：Dafny 上的成功可能只是数组越界安全检查，Lean 上的成功却可能需要完整函数正确性证明。多语言比较若不对齐规格，无法判断差异来自模型能力还是题目难度。

**核心矛盾**：社区需要知道 LLM 是否真的能验证“经典算法”的全局性质，但现有基准要么太简单，要么跨工具不公平。复杂算法如红黑树、最大流、Tarjan 强连通分量需要维护全局不变量、ghost state 和终止性证明，正是当前模型最容易暴露推理短板的地方。

**本文目标**：作者希望把问题难度、验证语言特性和模型推理能力拆开评估。具体做法是收集 77 个 textbook-style 经典算法任务，在 Dafny、Verus、Lean 中写出语义对齐的规格，然后用相同迭代修复协议评估闭源和开源模型。

**切入角度**：论文没有追求超大规模题库，而是强调“对齐”和“算法深度”。每个任务都要求同一个算法语义在三种验证系统中被表达，最后还用语义 validator 排除模型通过 `assume`、`sorry` 或退化算法钻空子的情况。

**核心 idea**：用严格平行规格的三语言经典算法基准，把 LLM verified code generation 从“局部安全检查”推进到“全局算法正确性”评估。

## 方法详解
AlgoVeri 本质上是一个基准构建与评测 pipeline。它的关键不是提出新的生成模型，而是把任务、规格、验证系统、迭代修复和语义过滤都固定下来，让不同模型和语言之间的性能差异可解释。

### 整体框架
输入是一道经典算法题及其自然语言描述。作者为 Dafny、Verus 和 Lean 分别提供函数签名、前置条件、后置条件和必要 helper definition，要求三种语言的规格表达同一组语义约束。评测时，模型接收问题描述和某个语言的 formal specification，生成实现与证明工件；若 verifier 报错，模型可在多轮中读取错误信息并修复。一个解首先需要 compiler/verifier 接受，然后再经过 LLM 语义过滤，确认它确实实现了指定算法而非绕过规格。

### 关键设计

**1. 跨验证系统的规格对齐：让跨语言分数可比**

已有 vericoding 基准的根本问题是跨工具分数不可比——Dafny 上的“成功”可能只是数组越界检查，Lean 上的“成功”却要完整正确性证明，分差到底来自模型能力还是题目难度无从判断。AlgoVeri 的做法是为同一道算法题在 Dafny、Verus、Lean 中手工编写并交叉审阅函数签名、前置条件、后置条件和辅助定义（helper definition），强制三种系统表达同一组语义约束。例如最长递增子序列（LIS）任务不满足于“结果落在数组范围内”，而是统一要求“存在对应长度的递增子序列、且不存在更长的合法子序列”。把题目难度固定住之后，观察到的跨语言差距才能干净地归因到工具链抽象层次、模型的证明搜索（proof search）能力和语言语法障碍上。

**2. 经典算法级别的全局不变量任务：把评测推到全局推理**

HumanEval/MBPP 这类基准只考单步、局部属性的题，会高估模型能力——例如证明一次二叉搜索树旋转保持局部有序很容易，但证明红黑树完整插入正确（要维持全局黑高不变量）就难得多。AlgoVeri 因此把 77 道 textbook 级经典算法作为题库，覆盖堆、线段树、红黑树、Bellman-Ford、Edmonds-Karp、高斯消元等，要求模型生成循环不变量（loop invariant）、ghost state、辅助引理（lemma）、终止性证明或 Lean 的 tactic 脚本。这些题的正确性依赖跨循环、跨数据结构的全局性质，正好戳中当前模型在图的可达性不变量和复杂状态更新上最容易暴露的推理短板。

**3. 验证通过后的语义过滤：区分“真验证”和“钻空子”**

形式化验证只保证“实现满足给定规格”，可一旦规格有漏洞，模型就能用 vacuous proof（空洞证明）、`assume`/`sorry` 这类作弊语句、或干脆实现一个更简单的退化算法来骗过 verifier。为此 AlgoVeri 在 verifier 通过之后再加一道 LLM judge 语义校验，并拆成两个指标：Compiler Verified 表示 verifier 接受，Full Mark 表示 verifier 接受且 judge 确认实现的确是指定算法；两者之差被定义为算法忠实度落差（algorithmic fidelity gap）。这一层过滤让基准同时考察“验证是否正确”和“是否忠实实现了目标算法”，避免把钻空子的通过算成真本事。

### 损失函数 / 训练策略
论文不训练模型，而是使用统一 inference-time refinement 协议。主实验中大多数模型使用 15 轮修复，GPT-5.3 Codex 使用 8 轮；开源模型额外评估 $10\times15$ 的多样本预算。每轮模型根据 compiler/verifier error 修改上一版代码。作者还比较 depth 和 width：depth 是沿同一条链迭代修复，width 是平行采样多个独立候选，在等计算预算下比较收益。

## 实验关键数据

### 主实验
主表报告三种验证系统上的 Compiler Verified 和经过语义过滤后的 Full Mark。下面保留更能说明问题的 Full Mark，同时列出 Verified 到 Full Mark 的落差。

| 模型 / 预算 | Dafny Verified / Full Mark | Verus Verified / Full Mark | Lean Verified / Full Mark | 主要含义 |
|-------------|----------------------------|-----------------------------|----------------------------|----------|
| GPT-5.3 Codex, $1\times8$ | 49.35 / 42.86 | 14.29 / 11.69 | 23.38 / 11.69 | Codex 在 Dafny 最强，但 Lean 存在很大的语义落差 |
| Gemini-3 Flash, $1\times15$ | 55.84 / 40.26 | 25.97 / 24.68 | 9.09 / 7.79 | frontier 模型能利用修复轮次，但 Lean 仍极难 |
| GPT-5 mini, $1\times15$ | 41.56 / 30.47 | 7.79 / 6.49 | 5.19 / 5.19 | 小模型在 Verus/Lean 上快速掉队 |
| GPT-OSS-120B, $1\times15$ | 21.04 / 13.51 | 7.66 / 7.01 | 12.60 / 7.01 | 开源模型基础能力明显低于闭源 frontier |
| GPT-OSS-120B, $10\times15$ | 44.16 / 28.57 | 12.99 / 10.39 | 25.97 / 14.29 | 增加平行预算有帮助，但仍不及 Gemini 的 Dafny 表现 |
| Qwen3-235B, $10\times15$ | 32.47 / 29.87 | 12.99 / 12.99 | 6.49 / 6.49 | 扩展预算后 Dafny 改善，Lean 仍低 |
| Devstral-2-123B, $10\times15$ | 33.77 / 18.18 | 14.29 / 12.99 | 6.49 / 6.49 | 语义过滤暴露出 Dafny 上较大的 fidelity gap |

### 消融实验
分析部分重点比较 test-time compute 的深度修复和宽度采样，并拆解不同语言的失败模式。

| 分析项 | 观察结果 | 说明 |
|--------|----------|------|
| Gemini-3 Flash 修复深度 | Dafny 和 Verus 的 pass rate 随 15 轮 repair 持续提升，Dafny 从初始到第 15 轮接近三倍提升 | frontier 模型能把 verifier feedback 当成有效训练信号式的推理反馈 |
| GPT-OSS-120B 修复深度 | 大约第 3 到 4 轮后趋于饱和，继续修复收益很小 | 当前开源模型更像“带错误上下文重采样”，缺少多轮 proof repair 能力 |
| GPT-OSS 等计算预算 | repair 曲线没有显著超过平行采样 baseline，深修复甚至可能更差 | 对开源模型而言，width 比 depth 更划算 |
| Dafny 错误轨迹 | 早期 syntax/type 错误较快下降，后续主要剩 verification failure | 高层抽象和 SMT 自动化让模型能集中修逻辑 |
| Verus 错误轨迹 | syntax/type 错误在 15 轮中持续存在 | Rust 宏、所有权、整数类型转换等形成语法与类型屏障 |
| Lean 错误轨迹 | hallucination 与 verification error 占主导 | 模型既要找对 lemma/tactic，又要构造正确证明，搜索和推理双重困难 |

### 关键发现
- “已验证”不等于“实现了指定算法”。Gemini-3 Flash 在 Dafny 上从 55.84% Verified 掉到 40.26% Full Mark，说明必须单独评估 algorithmic fidelity。
- 语言层级很明显：Dafny 的自动化和数学抽象最友好，Verus 更贴近系统代码但语法/类型负担重，Lean 信任内核小但证明搜索空间最大。
- 难题主要集中在高级数据结构和图算法。需要 ghost state、全局可达性、平衡不变量或最大流最优性证明的任务，目前几乎仍是模型瓶颈。

## 亮点与洞察
- 这篇论文的价值在于把 benchmark 设计得足够“公平”。它不是简单堆题，而是把同一算法规格平移到三套验证生态里，让跨语言性能差异有了可解释的基础。
- Full Mark 指标很关键。很多代码生成基准只看通过率，而 AlgoVeri 显示形式化验证也会被退化实现或规格漏洞污染，必须把“证明通过”和“算法忠实”拆开。
- depth vs width 的分析对 agentic coding 很有启发：不是所有模型都适合多轮 repair。对修复能力不足的模型，盲目拉长反馈链可能比并行采样更浪费。

## 局限与展望
- 语义过滤使用 LLM judge，虽然能抓住明显退化算法和作弊行为，但仍可能有误判；更理想的是为更多任务设计可机械检查的 algorithmic fidelity 条件。
- 规格由专家手工编写和审阅，质量高但成本大。未来若要扩展到上千任务，需要半自动规格迁移和形式化 well-formedness 检查工具。
- AlgoVeri 目前主要评估生成能力，没有系统比较专门训练、检索增强证明、搜索树规划等不同 agent 架构。它更像可靠测量尺，而不是完整解决方案。
- 对 Verus 和 Lean 的低分既反映模型不足，也反映训练数据稀缺和工具体验问题。后续改进可能需要语言生态、错误消息设计和模型训练共同推进。

## 相关工作与启发
- **vs VeriCoding**: VeriCoding 规模更大，但跨语言任务和规格不对齐；AlgoVeri 规模较小，却能更公平地比较 Dafny、Verus、Lean 的真实难度。
- **vs CLEVER / Verina**: CLEVER 和 Verina 聚焦 Lean，并强调防作弊或 proof-gap 诊断；AlgoVeri 进一步强调跨验证系统对齐和复杂经典算法。
- **vs DafnyBench**: DafnyBench 更偏 annotation/hint generation；AlgoVeri 要求从规格出发生成完整实现与证明工件，任务更接近端到端 verified code generation。
- **vs HumanEval / MBPP**: 后者只测有限测试样例，AlgoVeri 要求 verifier 证明所有输入的正确性，因此更适合评估可靠代码生成的上限与短板。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 严格对齐三种验证系统的经典算法 vericoding 基准很有辨识度，解决了此前跨语言分数不可比的问题。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖闭源/开源模型、三种验证生态、修复轮次、等预算分析和失败模式拆解，证据链完整。
- 写作质量: ⭐⭐⭐⭐ 结构清楚，benchmark 动机强；部分模型命名和未来版本设定需要读者注意时间背景。
- 价值: ⭐⭐⭐⭐⭐ 对代码智能、形式化验证和 agentic repair 研究都很有用，能区分真实推理进展和浅层验证通过率。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](../../ACL2025/code_intelligence/codedpo_code_alignment.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](../../ACL2025/code_intelligence/galla_graph_aligned_large_language_models.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](../../ACL2025/code_intelligence/feabench_repo_code_gen.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)

</div>

<!-- RELATED:END -->
