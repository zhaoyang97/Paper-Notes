---
title: >-
  [论文解读] CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics
description: >-
  [ICLR 2026][LLM评测][凝聚态物理] 提出 CMPhysBench——520 道研究生级凝聚态物理开放式计算题基准，并配套树编辑距离驱动的 SEED 度量给出细粒度部分分，揭示即便最强的 Grok-4 也只有 36 SEED / 29% 准确率，暴露 LLM 在前沿物理领域的巨大能力缺口。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "凝聚态物理"
  - "计算题基准"
  - "SEED 度量"
  - "表达式编辑距离"
  - "物理推理评测"
---

# CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3d0FRYx0D0](https://openreview.net/forum?id=3d0FRYx0D0)  
**代码**: [https://github.com/CMPhysBench/CMPhysBench](https://github.com/CMPhysBench/CMPhysBench)  
**领域**: LLM 评测 / 科学推理基准  
**关键词**: 凝聚态物理, 计算题基准, SEED 度量, 表达式编辑距离, 物理推理评测  

## 一句话总结
提出 CMPhysBench——520 道研究生级凝聚态物理开放式计算题基准，并配套树编辑距离驱动的 SEED 度量给出细粒度部分分，揭示即便最强的 Grok-4 也只有 36 SEED / 29% 准确率，暴露 LLM 在前沿物理领域的巨大能力缺口。

## 研究背景与动机
**领域现状**：LLM 在奥赛数学、编程乃至科学发现上表现亮眼，物理因其对推理、数学精度与概念理解的三重严苛要求，被视为检验 LLM 是否"真懂世界结构"的理想试金石。

**现有痛点**：早期物理基准（SciQ、ScienceQA）停留在高中难度；近期 PHYBench、UGPhysics 推进到本科水平，但仍偏广覆盖、浅深度，**严重低估当代物理研究的前沿子领域**。凝聚态物理（CMP）作为现代物理主流，融合量子力学、统计物理、固体物理与多体理论，具有跨学科、数据稀缺、数学表述精密的特点，却几乎无专门基准覆盖。

**核心矛盾**：评测方法本身也成问题——选择题忽略中间推理；精确字符串匹配（Accuracy）过于严苛，把"差一个单位/常数"的近似正确解判为全错；而 LLM 输出 LaTeX 噪声大、答案类型多样，使得现有度量要么不可靠要么无法泛化。

**本文目标**：构建一个深度聚焦 CMP、强调开放式计算、配套细粒度结构化打分的基准，诊断 LLM 在前沿物理上的真实能力与失败模式。

**核心 idea**：① **领域深耕的开放式计算题基准**——520 道博士/博后手工编写的研究生级 CMP 计算题；② **SEED 度量**——把多种答案类型统一映射到抽象语法树（AST），用树编辑距离给出非二值的部分分。

## 方法详解

### 整体框架
CMPhysBench 由"数据集 + 数据构建管线 + SEED 度量"三部分组成。数据侧通过四阶段管线（采集→结构化→质控→标注）从 17 本经典 CMP 教材中提炼出 520 道计算题，按 6 大主题与 5 种答案类型双重分类；评测侧把模型输出的 LaTeX 表达式规范化为 SymPy 对象再转 AST，按答案类型走不同对齐规则后计算树编辑距离，输出 0–100 的连续 SEED 分。

```mermaid
flowchart LR
    A[17 本 CMP 教材<br>PDF] -->|MathPix→Markdown| B[采集 Collection]
    B --> C[结构化为标准计算题<br>Structuring]
    C --> D[博士/博后专家审校<br>Quality Control]
    D --> E[520 道题<br>6主题 × 5答案类型]
    E --> F{模型作答}
    F -->|LaTeX 输出| G[规范化→SymPy→AST]
    G --> H[按答案类型对齐<br>+ 树编辑距离]
    H --> I[SEED 分 0-100<br>部分分 + 误差定位]
```

### 关键设计

**1. 研究生级 CMP 开放式计算题集：拒绝选择题，逼模型给完整推导**　520 道题覆盖磁性（115）、超导（55）、强关联系统（15）、半导体（115）四个核心子领域，外加理论基础（晶体学、等离激元、相变、凝聚态场论，110 题）与 Others（量子力学、统计物理、电动力学、量子场论，110 题）两个泛化维度，难度从本科习题跨越到高阶研究生挑战。所有题目均为**计算题**而非选择/判断题，要求模型独立产出含中间步骤的完整解答，从而同时考察概念理解与计算精度。专家进一步按答案类型把题目分为 Tuple、Equation、Numeric、Expression、Interval 五类（Numeric 占比最高约 65.6%），为后续度量的类型分支提供依据。

**2. SEED 度量：把"对错"变成"差多少"的树编辑距离打分**　SEED（Scalable Expression Edit Distance）继承 EED 的核心管线但做了三方面扩展。打分核心是：先把预测与标准答案的表达式各自解析为 AST，计算两树的编辑距离 $d$，再按节点总数 $n$ 归一化得到分数。直观地，完全匹配记 100 分，否则按 $\text{SEED} = \max\!\big(0,\ 60 - 100 \cdot \tfrac{d}{n}\big)$ 给部分分——例如 13 个节点、编辑距离为 1 时 $\text{SEED}=60-100/13\approx52$，编辑距离为 9 时归一化后压到 0。这种"近似正确给中间分、错得离谱归零"的设计，让度量比二值精确匹配更贴近人类专家判断。

**3. 类型感知的答案统一与鲁棒 LaTeX 预处理：让噪声输出也能稳定建树**　SEED 的可扩展性来自两点。一是**答案类型统一**：Expression 直接转 AST；Equation 把所有项移到一侧再比较；Tuple 按位置逐分量算 SEED 后取平均；Interval 用符号表示编码开闭边界；Numeric 则带单位换算、科学计数法解析与容差内取整。同时原生支持矩阵/向量与不等式（统一规范为 $f(\cdot)\ \#\ 0,\ \#\in\{<,\le,>,\ge\}$，并在变号操作下保持语义）。二是**鲁棒预处理**：剥离 `\boxed{}`、去掉 `\left`/`\right`、补全隐式乘法（如 `2x`、`ab`）、统一 Unicode 符号与字体命令、丢弃"Final Answer:"等自然语言噪声、自动配平括号与分式。这套"类型无关 AST + 可插拔物理感知规范化"使 SEED 不仅适用 CMP，还能轻松迁移到其他 STEM 任务。

## 实验关键数据

### 主实验（18 个模型在 CMPhysBench 上的表现）

| 模型梯队 | 代表模型 | SEED 分 | 专家标注准确率 |
|---|---|---|---|
| 领先簇 | **Grok-4** | **36.0** | **28.9%** |
| 领先簇 | o3 / Gemini 2.5 Pro | 30–35 | 23–29% |
| 中间带 | 多数主流模型 | 23–28 | 16–20% |
| 指令微调开源 | Llama-3.x-70B-Instruct 等 | 20–22 | 14–15% |
| 蒸馏/小模型 | R1-Distill-Qwen-32B 等 | 15–17 | 10–12% |

即便最强模型也仅 36 SEED / 29% 准确率，凝聚态物理对当前 LLM 仍是显著能力缺口。

### 子领域与度量对比

| 维度 | 关键发现 |
|---|---|
| 子领域峰值 | Grok-4 领跑磁性(35.30)/超导(43.42)/理论(41.21)；o3 在 Others(46.42) 最强；DeepSeek-R1 在强关联(42.16) 居首 |
| 强项不可迁移 | Grok-4 超导/理论强但强关联弱；Qwen3-32B 理论(35.47)尚可但磁性仅 8.47 |
| 度量与人类相关性 | **SEED ρ=0.90** > EED / GPT-4o(0.56) / xVerify(0.51) / OlympiadBench-Rule(0.41) |

### 错误分析（GPT-4o 自动归因，与专家 98% 一致）

| 错误类型 | 占比与代表 |
|---|---|
| 概念与模型误用 | 最主导，GPT-4o 66.5% / DeepSeek-V3 56.3% / Claude 3.7 Thinking 51.6% |
| 数学或逻辑错误 | 20–30%，o4-mini 31.0% / o3 29.4% |
| 任务理解错误 | 指令微调模型偏高，QwQ-32B 27.0% / Qwen3-32B 24.2%；Gemini 2.5 Pro 仅 7.5% |
| 单位/量纲错误 | 罕见（<2%） |

### 关键发现
- **推理模型未必更强**：Long-CoT 模型在 CMP 难题上不一定胜过通用模型——题目需领域知识，越"想"越容易在中途出错并传播到最终答案。
- **SEED 系统性高于严格准确率 +5~9 分**：大量"近似正确"解（单位、常数、边界条件差错）被精确匹配判全错，SEED 给出部分分更贴近真实能力。
- **数学推理与物理推理存在显著鸿沟**：概念/模型误用占比最高，说明瓶颈在领域物理原理的正确运用而非纯符号计算。

## 亮点与洞察
- **填补前沿子领域空白**：首个深度聚焦凝聚态物理、由领域专家手工编写的研究生级计算题基准，难度与专业度远超既有物理基准。
- **度量即贡献**：SEED 用"AST + 树编辑距离 + 物理感知规范化"把评测从二值对错升级为可解释的连续部分分，ρ=0.90 显著优于 GPT-4o judge 与规则匹配，是面向科学推理评测的通用方法论。
- **可诊断、可证伪**：八类错误归因 + 子领域雷达图把"模型差"拆解成具体失败模式，并提出"微调能减少概念误用、但数学逻辑错误会持续"的可检验假设，把基准变成研究平台。

## 局限与展望
- **规模相对有限**：520 题对覆盖整个 CMP 仍偏少，强关联系统仅 15 题，子领域分布不均可能影响统计稳健性。
- **错误归因依赖 GPT-4o**：虽与专家 98% 一致，但 Grok-4 因不输出中间链被排除分析，归因方法对"思维链可见"有依赖。
- **SEED 的物理等价性边界**：树编辑距离对结构等价敏感，但对深层物理等价变换（如不同坐标系/规范下的等价解）仍可能误判，未来需更强的符号等价引擎。
- **静态基准的污染与时效**：随模型迭代，公开数据集存在被训练数据污染的风险，需持续扩充与动态更新。

## 相关工作与启发
- **通用科学基准**（SciQ、ScienceQA、ARC、SciBench、SciEval）：覆盖广但难度多停留在 K-12 / 入门大学且偏选择题，难以探测深层推理。
- **进阶物理基准**（UGPhysics、GPQA、SuperGPQA、PHYSICS、PHYBench、PhysReason）：引入本科到研究生难度与步骤/表达式感知打分，但仍重广覆盖、轻特定子领域深度。
- **启发**：CMPhysBench 的"窄而深 + 结构化部分分度量"范式可迁移到其他高度专业化的 STEM 子领域评测；SEED 的 AST 树编辑距离思路也为数学/化学等需要符号等价判定的任务提供了可复用的评测组件。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 首个深度聚焦凝聚态物理的研究生级计算题基准，SEED 度量把树编辑距离引入物理评测并做物理感知扩展，立意与方法均有新意。
- **实验充分度**: ⭐⭐⭐⭐ — 18 个开闭源模型、6 子领域、5 答案类型、5 种度量对比 + 八类错误归因，覆盖全面，诊断深入。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图示（SEED 计算流程、错误类型、雷达图）有效，部分排版含 OCR/公式噪声但不影响理解。
- **价值**: ⭐⭐⭐⭐ — 揭示 LLM 在前沿物理的真实缺口并提供可证伪研究平台与可迁移评测方法论，对科学推理评测社区价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CMT-Benchmark: A Benchmark for Condensed Matter Theory Built by Expert Researchers](cmt-benchmark_a_benchmark_for_condensed_matter_theory_built_by_expert_researcher.md)
- [\[ICLR 2026\] PRISM-Physics: Causal DAG-Based Process Evaluation for Physics Reasoning](prism-physics_causal_dag-based_process_evaluation_for_physics_reasoning.md)
- [\[ICLR 2026\] Evaluating Language Models' Evaluations of Games](evaluating_language_models_evaluations_of_games.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ICLR 2026\] RefineBench: Evaluating Refinement Capability of Language Models via Checklists](refinebench_evaluating_refinement_capability_of_language_models_via_checklists.md)

</div>

<!-- RELATED:END -->
