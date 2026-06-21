---
title: >-
  [论文解读] Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification
description: >-
  [ICLR 2026][代码智能][形式化验证] 本文提出 **DAFNYCOMP** —— 首个针对「多函数程序组合式规约生成」的形式化验证基准，揭示了前沿 LLM 在单函数 Dafny 验证上能拿到 58%+ 通过率，但一旦把 2–5 个函数拼成调用链，端到端验证率几乎归零（最强模型 Pass@8 仅 2%），证明「局部成功不能组合」。
tags:
  - "ICLR 2026"
  - "代码智能"
  - "形式化验证"
  - "Dafny"
  - "组合推理"
  - "规约生成"
  - "Benchmark"
  - "Pass@k"
---

# Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=y4kAMUBqLq](https://openreview.net/forum?id=y4kAMUBqLq)  
**代码**: 待确认  
**领域**: 代码智能 / 形式化验证 / LLM 评测  
**关键词**: 形式化验证, Dafny, 组合推理, 规约生成, Benchmark, Pass@k  

## 一句话总结
本文提出 **DAFNYCOMP** —— 首个针对「多函数程序组合式规约生成」的形式化验证基准，揭示了前沿 LLM 在单函数 Dafny 验证上能拿到 58%+ 通过率，但一旦把 2–5 个函数拼成调用链，端到端验证率几乎归零（最强模型 Pass@8 仅 2%），证明「局部成功不能组合」。

## 研究背景与动机
**领域现状**：LLM 已大幅改变代码生成，而保证自动合成代码的正确性成为关键问题。形式化验证（如 Dafny）通过前置条件、后置条件、循环不变量等契约给出数学级正确性保证，但「规约瓶颈」——写注解需要专业知识且规约代码量与实现相当——长期阻碍其落地。近期工作开始用 LLM 自动补全规约，并在单函数任务上取得中等成功率。

**现有痛点**：现有基准（DAFNYBENCH、MBPP-DFY）几乎只评估**孤立单函数**内部的注解补全，无法考察真实软件系统所需的**跨组件组合推理**能力——而现实中的正确性恰恰诞生于多个组件的复杂交互。

**核心矛盾**：单函数任务允许纯局部推理就能过关，掩盖了模型在跨函数边界传播契约时的真实缺陷；当一个函数缺失或弱化的后置条件无法满足下游被调函数的前置条件时，错误会沿调用链级联放大，导致端到端证明失败。

**本文目标**：构造一个**强制要求组合推理**的诊断性基准，把「局部正确」与「组合可验证」之间的鸿沟量化出来，并系统刻画失败机理。

**核心 idea**：**「合成可验证的多函数 Dafny 程序 + 剥离契约让模型重建」** —— 把多个独立函数按无环调用图拼成程序，机械验证保证有 ground-truth，再让模型重新生成跨函数契约，以此暴露组合验证的系统性失败。

## 方法详解

### 整体框架
DAFNYCOMP 采用两阶段合成流水线产出 400 个机械验证过的多函数程序：先在 Python 层做**程序组装**（把独立函数按数据流拼成组合程序并验证功能正确），再做**形式化翻译**（增量式 AST 引导地转成可验证 Dafny）。评测时采用「规约重建」任务——只剥离契约子句让模型补回，并用 Pass@k 与 verifier-in-the-loop 多轮自精化两种协议衡量 11 个前沿模型。

```mermaid
flowchart LR
    A[LeetCodeDataset<br/>圈复杂度>5 筛选] --> B[程序组装<br/>链式/DAG 拼接 2-5 函数]
    B --> C[三阶段校验<br/>类型/格式/单测]
    C --> D[增量 AST 翻译<br/>逐片段译+验+重组]
    D --> E[≤10 轮精化<br/>Claude-4-Sonnet]
    E --> F[400 验证程序<br/>300 链 + 100 DAG]
    F --> G[剥离契约<br/>requires/ensures/...]
    G --> H[LLM 重建规约<br/>Pass@k / 多轮精化]
```

### 关键设计

**1. 程序组装：用圈复杂度筛函数、用数据流强制组合。** 作者从 LEETCODEDATASET 中按 McCabe 圈复杂度 $>5$（约前 30%）且代码行数 $\geq 10$ 过滤出 1,847 个候选函数，确保控制流足够复杂（带终止条件的循环、嵌套条件、递归）来给规约生成施压。组合时主要采用**链式组合**——前一函数输出喂给后一函数输入，制造显式数据依赖，并额外用 10 个非链 DAG 拓扑模板扩展分支结构。组合后还要识别共享 import 的最小依赖集合（因为原数据集大量 `import *` 会让 Dafny 无法解析库归属）。组装结果经三阶段校验过滤：(i) 约束传播类型检查，沿组合链传播输入输出的类型与形状约束，拒绝接口不兼容的组合；(ii) 用 Black/isort 做确定性格式标准化，降低后续 prompt 方差；(iii) 跑参考单测验证功能正确，单测由各组成函数输入输出约束的**交集**构造，交集为空则丢弃。最终得到 1,200 个有效 Python 程序。

**2. 形式化翻译：增量 AST 翻译 + 整程序组装互补。** 直接端到端 Python→Dafny 翻译成功率不足 5%，因为 Dafny 要求显式规约、不变量、终止论证，而 Python 完全没有这些语义元素。作者改用**增量翻译**：把每个 Python 程序的 AST 分解为函数级/控制结构级片段，逐片段译成 Dafny 并立即验证以把错误定位到最小单元，再按 AST 层级逐步重组成完整程序。关键洞察是「整程序组装」与「增量翻译」互补——Python 端先完整组装能利用成熟工具链保证可靠性，并提供全局逻辑蓝图，避免增量翻译产出无法组合的孤立片段。每个候选 Dafny 程序还经历 ≤10 轮根据验证器反馈强化规约的精化（加循环不变量、细化后置条件）。整条流水线用 CLAUDE-4-SONNET 合成，1,200 次尝试最终产出 564 个验证程序（47% 成功率），并刻意把 Claude 系模型排除出下游评测以防信息泄漏。

**3. 规约重建任务：只剥契约，隔离组合推理。** 与 DAFNYBENCH 删除全部注解不同，本任务**只剥离方法/函数开括号前的契约子句**（`requires`、`ensures`、`reads`、`modifies`、`decreases`），保留实现，让模型重新生成这些规约以使验证通过。这一设计把「重建跨函数契约」从实现层面隔离出来，专注考察模型能否生成捕捉跨边界涌现正确性的规约。最终基准含 300 链式 + 100 DAG，平均每程序 3.2 个函数、8.4 条跨函数数据依赖，中位数需要 7 个循环不变量与 4 个断言——是 DAFNYBENCH 注解密度的约 $3.5\times$。

**4. 双协议测试时扩展评测。** 评测同时报告两种测试时扩展：**独立采样 Pass@k**（$k\in\{1,2,4,8\}$）衡量多次独立尝试内解出问题的概率；**verifier-in-the-loop 多轮自精化**则在每轮失败时把模型上一次输出连同验证器报错喂回，要求在保持方法签名与声明契约不变的前提下修订，最多 $T=3$ 轮，每轮跑一次 Dafny 验证器。两种协议分别量化「更多采样」与「带反馈迭代」两类计算的边际收益。

## 实验关键数据

### 主实验：链式 split 独立采样 Pass@k（300 题）

| 模型 | Syntax@8 (%) | Verified@1 | Verified@8 |
|---|---|---|---|
| GPT-4O | 99.67 | 0.33 | 0.33 |
| O4-MINI | 99.00 | 0.00 | 0.67 |
| GEMINI-2.5-PRO | 96.00 | 0.00 | **2.00**（最强） |
| DEEPSEEK-R1 | 99.00 | 0.33 | 0.33 |
| QWEN3-CODER-480B | 99.00 | 0.00 | 1.00 |
| QWQ-32B | 91.00 | 0.00 | 0.00 |

Pass@8 全体均值：Syntax $94.36\%$ vs Verified $0.55\%$ —— **Syntax–Verified gap 高达 93.82%**。

### 消融/对比：多轮自精化（代表模型）

| 模型 | 设置 | Verified@T1 | Verified@T3 |
|---|---|---|---|
| DEEPSEEK-R1 | 链式 | 0.33 | **9.67** |
| O4-MINI | 链式 | 0.00 | 9.67 |
| QWEN3-CODER-480B | 链式 | 0.33 | 3.00 |
| DEEPSEEK-V3.1 | DAG | 1.00 | 7.00 |
| GPT-4.1 | DAG | 1.00 | 4.00 |

对比：前沿模型在单函数基准上 Syntax $>99\%$、Verified $>58\%$；DAFNYCOMP 上 Verified 崩到低个位数。

### 失败模式分布（900 个失败案例人工分析）

| 失败模式 | 占比 | 主要机理 |
|---|---|---|
| 规约脆弱（Specification Fragility） | 39.2% | 契约传播失败（多米诺效应） |
| 实现-证明错位（Impl–Proof Misalignment） | 21.7% | 实现与规约被独立生成 |
| 推理不稳定（Reasoning Instability） | 14.1% | 归纳链断裂 |
| 其他（语法/超时等） | 25.0% | 杂项 |

### 关键发现
- **普遍崩溃**：高语法正确（94%+）与近零验证（0.55%）的鸿沟跨越所有模型族、稠密/MoE、通用/代码专用，并非格式问题。
- **组合带来不成比例的退化**：单函数 58%→组合 0.55%，源于跨函数契约的级联失败而非简单的加性难度。
- **采样收益快速饱和**：k=4→k=8 平均仅 +0.27%，11 个模型中 7 个零增益；多轮精化能升到 9.67% 但远未饱和。
- **推理专用模型无明显优势**：QWQ-32B 仍 0% 验证，DEEPSEEK-R1 仍 <1%，说明当前 RL/推理轨迹训练尚不足以解决组合验证。

## 亮点与洞察
- **「局部成功不能组合」这一命题被首次量化**：把一个工程直觉变成了可追踪的诊断指标（Syntax–Verified gap），且数值极端（93.82%），冲击力强。
- **合成流水线设计扎实**：圈复杂度筛选 + 三阶段校验 + 增量 AST 翻译 + 污染分析，保证了基准既新颖又有机械验证的 ground-truth，且回避了同族模型泄漏。
- **失败模式的机理刻画有指导意义**：规约脆弱（多米诺）、实现-证明错位（独立生成）、推理不稳定（归纳链断裂）三类，直接指向「应在训练中强化契约传播与实现-规约互一致性」的改进方向。
- **诚实的 null result**：明确指出推理专用模型也没救，避免了夸大某类方法。

## 局限与展望
- **组合模式受限**：主评测聚焦无环链式（加 100 个 DAG 扩展），循环图、互递归、共享状态密集依赖等更复杂模式仍超出范围，扩展需解决带环依赖的合成与验证可处理性。
- **规约类型单一**：只测功能正确性（前/后置条件、不变量），未覆盖活性、资源界、安全策略等正交属性。
- **合成依赖单一模型**：流水线全程用 Claude-4-Sonnet，虽已排除 Claude 系下游评测，但合成偏置仍可能潜在影响题目分布。
- **未给出解法**：本文是诊断性基准，指出问题但未提供训练/推理层面的解决方案，留给后续工作。

## 相关工作与启发
- **形式化验证基准**：单函数类（DAFNYBENCH、MBPP-DFY，50–60% 成功率）与交互式定理证明类（miniCodeProps、FVAPPS，基于 Lean 但脱离实际编程）；DAFNYCOMP 填补了「组合式规约生成」这一前置能力的空白。
- **动态基准生成**：静态基准受污染与过拟合困扰，本文用受控组合 + 污染分析在保证新颖性的同时维持语义复杂度。
- **LLM 组合推理**：与 Dziri et al. 关于 Transformer 难以组合的工作呼应，形式化验证施加了更严格的「跨组件保不变量」要求；与 Olsson/Elhage 的机制可解释性分析（电路实现局部算法而非全局一致证明）相互印证。
- **启发**：这套「拼装-验证-剥离-重建」的合成范式可推广到智能合约验证、分布式协议等任何需要组合正确性的领域；下一步值得探索把契约传播作为显式训练信号或在推理时引入跨函数约束求解。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个组合式形式化验证基准，「局部成功不能组合」命题被首次系统量化，问题切口独到。
- **实验充分度**: ⭐⭐⭐⭐ 11 个前沿模型 × 双协议 × 链式/DAG × 900 例人工失败分析，覆盖全面；但合成全靠单一模型、未含解法略减分。
- **写作质量**: ⭐⭐⭐⭐ 动机-构造-结果-机理逻辑清晰，三类失败模式带具体反例，takeaway 提炼到位。
- **价值**: ⭐⭐⭐⭐⭐ 给可验证代码生成树立了一个极具挑战性、几乎人人 0 分的诊断标尺，明确指出当前 LLM 的结构性短板与训练改进方向，社区价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification](spotit_evaluating_text-to-sql_evaluation_with_formal_verification.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ICLR 2026\] Evolving Graph Structured Programs for Circuit Generation with Large Language Models](evolving_graph_structured_programs_for_circuit_generation_with_large_language_mo.md)
- [\[ICLR 2026\] CrossPL: Systematic Evaluation of Large Language Models for Cross Programming Language Interoperating Code Generation](crosspl_systematic_evaluation_of_large_language_models_for_cross_programming_lan.md)
- [\[ICLR 2026\] LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models](learnat_learning_nl2sql_with_ast-guided_task_decomposition_for_large_language_mo.md)

</div>

<!-- RELATED:END -->
