---
title: >-
  [论文解读] FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory
description: >-
  [ICLR 2026][LLM评测][subgoal completion] 本文提出 **FormalML**——首个面向「子目标补全（subgoal completion）」的 Lean 4 基准，用一个自研的 `to_theorem` 翻译策略从机器学习理论（优化 + 概率）的形式化库中自动抽取 4,937 道证明片段题，系统性地暴露了当前 LLM 证明器在复杂上下文、前提利用与效率上的真实短板。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "subgoal completion"
  - "Lean 4"
  - "theorem proving"
  - "ML theory"
  - "premise utilization"
---

# FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wCRZbspSZi](https://openreview.net/forum?id=wCRZbspSZi)  
**代码**: 已开源（Codebase + Dataset，论文内提供链接）  
**领域**: LLM 评测 / 形式化定理证明 / Benchmark  
**关键词**: subgoal completion, Lean 4, theorem proving, ML theory, premise utilization  

## 一句话总结
本文提出 **FormalML**——首个面向「子目标补全（subgoal completion）」的 Lean 4 基准，用一个自研的 `to_theorem` 翻译策略从机器学习理论（优化 + 概率）的形式化库中自动抽取 4,937 道证明片段题，系统性地暴露了当前 LLM 证明器在复杂上下文、前提利用与效率上的真实短板。

## 研究背景与动机
**领域现状**：LLM 在形式化定理证明上进展惊人，DeepSeek-Prover-V2、Kimina-Prover 等借助自然语言辅助的 Long-CoT 已在 miniF2F 等竞赛级基准上拿到银牌级表现。但这些基准的任务都是「从零生成一整条完整证明」，问题来自奥赛或本科教材，上下文简单、不依赖外部前提。

**现有痛点**：当数学家真正用 LLM 当 copilot 去做研究级证明时，效果远不及预期——Equational Theories Project 直言「在大多数难例上 LLM 给不出超出人类参与者已有想法的有用建议」。真实工作流里，专家会先用自然语言把证明骨架搭好、把定理形式化，剩下的是散落在证明中间、用 `sorry` 占位的一个个**短而不平凡的证明义务**。补全这些缺口才是 LLM 最该发挥作用的位置，但这个任务**没有任何专门基准**。

**核心矛盾**：子目标补全与全证明生成是两种不同的难度结构。前者每道题的证明往往很短，但要求模型：(1) 读懂一长串累积下来的复杂假设与目标；(2) 从局部库和 Mathlib 中正确调用相互依赖的前提引理；(3) 不能像做奥赛题那样无脑堆长推理链——「过度思考」在这里既不提分又烧算力。换句话说，好的证明器必须在**准确率与简洁性之间取得平衡**，而这恰恰是现有竞赛导向模型不擅长的。

**本文目标**：构造一个能贴近真实「填空式」证明协作场景的基准，并据此系统评测主流证明器，定位它们的根本瓶颈。

**核心 idea**：**【从过程式证明里"切片"出子目标】** 作者观察到人写 Lean 4 证明虽是过程式（procedural）的 tactic 序列，但**一行 tactic 通常对应作者心里的一个完整推理步**；于是设计 `to_theorem` 策略，记录某行（或某段）tactic 执行前后的证明状态，把「前状态当假设、后状态当目标」自动合成为一道独立定理——原 tactic 序列就是它的标准答案。靠调节切片长度还能造出不同难度的题。

## 方法详解

### 整体框架
FormalML 的构造分三步：先用自研 `to_theorem` 策略把 Lean 4 过程式证明在「行级」切片成独立子目标定理；再从优化库 Optlib 与概率库 FoML 中筛选顶层定理做抽取、保留底层引理作前提库；最后按证明长度划分难度、按是否需要前提标注，得到 4,937 道题。评测端用 Pass@K 在低算力预算下衡量准确率，并辅以前提利用、难度分层、效率（EWA@K）三个维度的专项实验。

```mermaid
flowchart LR
    A[Optlib 优化库<br/>FoML 概率库] --> B[选顶层定理<br/>保留引理作前提库]
    B --> C[to_theorem 策略<br/>行级/段级切片]
    C --> D[4937 道子目标定理<br/>JSON: 语句+前提+证明]
    D --> E[难度分层 L1/L3/L5]
    D --> F[前提标注<br/>Mathlib / 局部库]
    E --> G[Pass@K 主评测]
    F --> H[前提利用专项]
    D --> I[EWA@K 效率 / 专家迭代]
```

### 关键设计

**1. `to_theorem` 翻译策略：把过程式证明步骤具体化为独立定理。** 这是整个基准的技术地基。Lean 4 自带的 `extract_goal` 只能抓取当前证明状态的整体目标，无法捕捉过程式证明中每一步的细粒度中间子目标——而现有库里绝大多数证明都是过程式风格（用 `rw`、`simp` 逐步变换证明状态，几乎不出现 `have` 这种声明式语句）。`to_theorem` 的做法是：选定一行（或连续几行）tactic，记录其执行**前后**的证明状态，把前状态的全部假设塞进定理前提、把后状态作为新目标，自动合成一道新定理；原来的那几行 tactic 直接就是这道题的可验证答案。以九行证明的 `linear_gradient` 引理为例，理论上可切出最多 9 道单行子目标、4 道两行子目标，以此类推——切片长度天然成了难度旋钮。

**2. 双库数据策展：把 ML 理论拆成优化与概率两类。** 作者刻意把范围收窄到机器学习的基础理论，理由是「AI agent 正越来越多地自动化科学发现，而自动定理证明能保证推导结果的可靠性」。优化侧基于 Optlib（覆盖 GD、SubGD、PGD、Nesterov 加速、块坐标下降 BCD、ADMM），概率侧基于 FoML（Rademacher 复杂度的泛化误差界、有界差分不等式、McDiarmid 等），并额外补全了 Hoeffding 引理、Bennett、Bernstein 不等式。最终优化类 2,907 题、概率类 2,030 题。每道题以 JSON 存储完整证明上下文——源位置、形式化语句、所需 import 与 namespace、完整 tactic 序列、相关前提，保证可独立评测。

**3. 难度与前提双重标注：让评测可分层。** 由于证明多为过程式风格，作者直接用**证明长度**作为难度代理，定义长度 1/3/5 三档难度（L1 共 3,924 题，更高难度 1,013 题）。前提维度上，共 1,547 题需要显式调用前提，按来源分为 Mathlib 与局部库两类并统计平均前提数。这套标注支撑了后续「前提利用」「难度衰减」两个核心发现的专项实验。

**4. EWA@K 效率度量：把"啰嗦"算进分数。** 针对子目标补全「短证明却被 Long-CoT 模型过度思考」的痛点，作者提出效率加权准确率 $\text{EWA@}K = \text{Pass@}K \times \frac{100}{\text{Response Length}}$，用响应长度归一化成功率。这把「在只需几行直证的题上烧掉海量 token」的行为直接惩罚进指标，使「准确率高但输出短」的模型（如 STP）凸显出来，而三个 Long-CoT 模型因 token 量巨大反而垫底。

## 实验关键数据

### 主实验（Pass@K，部分模型）

| 类别 | 模型 | Budget | Pass@1 | Pass@32 (All) |
|------|------|--------|--------|---------------|
| 自动化 tactic | aesop | - | - | 43.29 |
| 自动化 tactic | Ensemble（全合并） | - | - | 48.67 |
| BFS 树搜索 | BFS-Prover (7B) | 8×50 | - | 25.31 |
| 全证明生成 | **STP (7B)** | 32 | **26.96** | **63.21** |
| 全证明生成 | DeepSeek-Prover-V2 (noCoT) | 32 | 16.85 | 62.06 |
| 全证明生成 | Leanabell-Prover (7B) | 32 | 24.10 | 58.07 |
| 全证明生成 | DeepSeek-Prover-V2 (CoT) | 32 | 18.43 | 39.50 |

- **Pass@1 全场最高仅 26.96%（STP）**，远达不到实用「数学家助手」门槛；Pass@32 最高 63.21%。
- **BFS 树搜索吃更多算力却不占优**（普遍 <30%），说明逐步交互 Lean 的搜索效率低。
- **aesop 单 tactic 强到反超部分 LLM**（低预算下），但高预算被 LLM 反超；即便所有自动化 tactic 集成仍不敌 LLM。

### 前提利用（Pass@32，相对 M=0 增量）

| 模型 | M=0 | M=* (仅真前提) | M=20 (大候选集) |
|------|-----|----------------|------------------|
| **DeepSeek-Prover-V2 (noCoT)** | 58.37 | **71.86 (+13.49)** | 65.80 (+7.43) |
| Goedel-Prover | 39.04 | 50.10 (+11.06) | 39.50 (+0.46) |
| STP | 57.72 | 61.86 (+4.14) | 56.04 **(-1.68)** |
| DeepSeek-Prover-V1.5 | 52.94 | 56.69 (+3.75) | 49.45 **(-3.49)** |

- **DeepSeek-Prover-V2 前提利用一枝独秀**，给真前提可提升约 10%+；作者归因于它训练时见过大量自然语言推理数据。
- **STP 虽全局强，但 M=10/20 时反而掉分**，因其训练集缺乏此类前提数据——全局最优 ≠ 前提利用最优。

### 关键发现
- **Finding 1**：现有证明器在低算力预算下还不足以当数学家的实用助手，且不同专业子领域（如 McDiarmid vs Hoeffding）表现差异巨大。
- **Finding 2**：候选前提集一大就用不好；即便给真前提多数模型也利用不充分，瓶颈主要在基座模型。
- **Finding 3**：难度升高（L3/L5）性能显著衰减，STP 在 L5 仅 Pass@128 33.36%。
- **Finding 4**：Long-CoT 在子目标补全上**既不提分又烧 token**，EWA@32 全场垫底；STP 输出最短、准确率最高，EWA 最优。
- **Finding 5**：专家迭代（从 mathlib、PFR、scilean 等 5 个库抽 9.2 万题，8.8 万题训练）能显著提分，尤其 Pass@1，是有潜力的方向。

## 亮点与洞察
- **任务定义本身是贡献**：把「子目标补全」从全证明生成里独立出来，精准对准了「LLM 当数学 copilot」的真实落点——比起刷竞赛分，这个中间里程碑更贴近研究级协作。
- **`to_theorem` 是可复用的造数引擎**：不止用于建基准，专家迭代实验里它从 5 个真实 Lean 库一口气抽出 9 万+ 训练题，证明这条「过程式证明→海量子目标题」的管线可大规模放大。
- **EWA@K 戳中 Long-CoT 痛处**：用一个简单的「成功率/响应长度」指标，把竞赛叙事下被忽视的「过度思考」量化出来，揭示长链推理并非万灵药。
- **反直觉结论扎实**：miniF2F 的强者（Long-CoT 系）在这里集体失速，排名与竞赛榜不一致，直指模型对竞赛题的过拟合。

## 局限与展望
- **难度只用证明长度代理**：长度并不完全等于推理难度，过程式风格下的「行数」可能与语义复杂度脱节，难度分层略显粗糙。
- **范围限于 ML 理论的两个库**：Optlib + FoML 虽覆盖优化与概率，但相对整个数学仍偏窄，结论能否外推到其他领域待验证。
- **前提选择阶段被略过**：论文聚焦「前提利用」，把前提检索/选择当作上游已完成；真实场景中检索误差会进一步放大难度。
- **只做了 1 轮专家迭代**：Finding 5 仅初步验证，多轮迭代的收益曲线、是否会过拟合到 `to_theorem` 抽取分布尚未探究。

## 相关工作与启发
- **vs 全证明生成基准**（miniF2F、PutnamBench、ProofNet、FormalMATH、LeanDojo）：这些要么上下文简单、要么不需前提，且全是整条证明生成；FormalML 是唯一同时具备「前提 + 复杂上下文 + 子目标补全」三特性的 Lean 4 基准。
- **vs 证明器方法**：BFS 树搜索（Reprover、BFS-Prover）与全证明生成（STP、Goedel、DeepSeek-Prover、Kimina）两条路线都被纳入评测，结论是当前 SOTA 在「填空式」任务上的优势远不如竞赛任务明显。
- **启发**：(1) 评测要贴近真实工作流——「填中间步」比「从零生成」更能暴露模型短板；(2) 效率应作为一等公民进入证明器评价，而非只看 Pass@K；(3) `to_theorem` 这类「从已有证明反向造题」的思路，对低成本扩充形式化训练数据有普适价值。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首次定义并基准化「子目标补全」任务，`to_theorem` 切片造数与 EWA@K 度量均为原创，填补了真实协作场景的评测空白。
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 BFS + 全证明生成两类共十余个 SOTA 证明器，从准确率、前提利用、难度分层、效率、专家迭代五个维度系统评测，结论扎实。
- **写作质量**: ⭐⭐⭐⭐ 动机清晰、图 1/图 2 把 `sorry` 占位与 `to_theorem` 机制讲得很直观；Lean 代码细节较多，对非形式化背景读者门槛偏高。
- **价值**: ⭐⭐⭐⭐⭐ 为「LLM 当数学 copilot」提供了首个可靠标尺，并直接产出可复用的造数管线与训练信号，对形式化定理证明社区有长期价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] An Open-Ended Benchmark and Formal Framework for Adjuvant Research with MLLM](an_open-ended_benchmark_and_formal_framework_for_adjuvant_research_with_mllm.md)
- [\[ACL 2026\] Can We Predict Before Executing Machine Learning Agents?](../../ACL2026/llm_evaluation/can_we_predict_before_executing_machine_learning_agents.md)
- [\[ICLR 2026\] Characterizing Deep Research: A Benchmark and Formal Definition](characterizing_deep_research_a_benchmark_and_formal_definition.md)
- [\[ICLR 2026\] CMT-Benchmark: A Benchmark for Condensed Matter Theory Built by Expert Researchers](cmt-benchmark_a_benchmark_for_condensed_matter_theory_built_by_expert_researcher.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)

</div>

<!-- RELATED:END -->
