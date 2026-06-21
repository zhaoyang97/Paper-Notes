---
title: >-
  [论文解读] MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes
description: >-
  [ICLR 2026][AI安全][道德推理] MoReBench 提出用专家手写的 23,018 条 rubric 准则去评测推理模型在 1,000 个道德困境上思考过程的结构质量：（而非最终结论对错），并发现 scaling laws 与数学/代码基准都无法预测模型的道德推理能力。 - 领域现状：AI 越来越多地替人或与…
tags:
  - "ICLR 2026"
  - "AI安全"
  - "道德推理"
  - "过程评测"
  - "rubric 评分"
  - "多元价值"
  - "规范伦理学"
  - "推理模型"
---

# MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RMwJXp5Kb1](https://openreview.net/forum?id=RMwJXp5Kb1)  
**代码**: [https://github.com/morebench/morebench](https://github.com/morebench/morebench)  
**数据**: [https://hf.co/datasets/morebench/morebench](https://hf.co/datasets/morebench/morebench)  
**领域**: AI Safety / 道德推理评测 / 过程性评测  
**关键词**: 道德推理, 过程评测, rubric 评分, 多元价值, 规范伦理学, 推理模型  

## 一句话总结
MoReBench 提出用专家手写的 23,018 条 rubric 准则去评测推理模型在 1,000 个道德困境上**思考过程的结构质量**（而非最终结论对错），并发现 scaling laws 与数学/代码基准都无法预测模型的道德推理能力。

## 研究背景与动机
- **领域现状**：AI 越来越多地替人或与人一起做高风险决策，而推理模型同时输出最终答案和（部分透明的）思维链，提供了研究"AI 怎么想"的机会。已有的价值评测从 ETHICS、Delphi 这类共识价值，发展到道德信念、价值偏好、多步案例、利益相关者视角。
- **现有痛点**：这些工作几乎都在评测模型**决定了什么**（what they decide），而非**它如何推理到这个决定**（how they reason）。少数尝试评测推理过程的工作（自动驾驶场景的义务论/后果论分类、哲学家人工对比、训练专门分类器判演绎/溯因）要么范围窄、要么难以规模化。
- **核心矛盾**：数学/代码有客观正确答案、易于自动验证；道德困境恰恰**没有唯一正确答案**——往往存在多个站得住脚的结论，因此无法用"对/错"来评，但又正是人机交互中最核心的能力。
- **本文目标**：构建一个可规模化、自动化、聚焦"推理过程"而非"推理结果"的道德推理评测体系，覆盖 AI 给人建议（Advisor）和 AI 自主决策（Agent）两种角色。
- **核心 idea**：**用专家手写的 rubric 准则替代"标准答案"**——既然好的道德推理无法用单一结论衡量，那就把"一段好推理应当包含/避免哪些要素"拆成大量原子化、带权重的准则，用 LLM-judge 逐条打分再加权求和，把主观的"过程质量"转成可计算的分数。

## 方法详解

### 整体框架
MoReBench 由三块构成：**数据集构建**（53 位道德哲学专家为 1,000 个道德困境写 23,018 条带权 rubric，外加 150 个理论标注的 MoReBench-Theory）、**评测方法学**（选 LLM-judge、设计加权聚合公式、做长度校正）、以及对 rubric 本身的**元评测**（验证判别力与鲁棒性）。模型被评的对象是其**思维链**（thinking trace）和最终回答，逐条对照 rubric 由 judge 模型判"是否满足"。

```mermaid
flowchart LR
    A[道德困境场景<br/>Advisor/Agent 两角色] --> B[专家手写 rubric<br/>20+ 条原子准则/场景<br/>带权 -3~+3]
    B --> C[二次专家 review]
    C --> D[被测模型生成<br/>思维链 + 最终回答]
    D --> E[GPT-oss-120b judge<br/>逐条判是否满足]
    E --> F[加权聚合公式<br/>得到场景分数]
    F --> G[长度校正<br/>Regular / Hard]
```

### 关键设计

**1. 双角色场景 + 专家 rubric：把"好推理"拆成可打分的原子准则**。MoReBench 把场景锚定在 AI 的两类真实角色上——Moral Advisor（给人建议，源自 DailyDilemmas 的日常人际/职场困境）和 Moral Agent（自主行动，源自 AIRiskDilemmas 的 AGI 安全场景，如科研造假举报、教育隐私），再加专家从伦理学文献、辩论案例、应用伦理新闻里改写的案例。每个场景配一套**至少 20 条**的准则，要求每条准则客观、贴合该场景上下文、原子化（只评一个方面），整套准则覆盖所有重要考量且彼此无重叠。准则被归入五个维度：Identifying（识别道德考量）、Clear Process（清晰系统的推理表达）、Logical Process（说明各考量如何整合权衡）、Helpful Outcome（给出可操作路径）、Harmless Outcome（不给违法/有害建议）。为降低个人偏见，每套 rubric 由另一位资深专家 review 增删，再经研究团队复核，最终汇成 1,000 场景 23,018 条准则的"群体共识分布"。

**2. 带符号权重的加权聚合分数**。专家给每条准则赋权 $p_{ij}\in[-3,3]\setminus\{0\}$（+3 critically important 到 -3 critically detrimental），judge 输出满足标记 $r_{ij}\in\{-1,1\}$。一个理想回答应满足所有正权准则、不触犯任何负权准则得 100 分，反之得 0。单场景分数定义为
$$s_i = \frac{\sum_{j=1}^{M_i} \mathrm{sgn}(p_{ij})\cdot r_{ij}\cdot p_{ij}}{\sum_{j=1}^{M_i}|p_{ij}|}$$
分子用 $\mathrm{sgn}(p_{ij})$ 保证"满足正权准则加分、触犯负权准则减分"，分母归一化到权重总量，使不同场景的分数可比；全体平均 $\bar{s}$ 即 **MoReBench-Regular**。

**3. 长度校正惩罚冗长**。基于准则满足度的评测有一个天然漏洞：回答越长、越啰嗦，越容易"碰上"更多准则而虚高得分（HealthBench 已暴露此问题）。MoReBench 用响应长度相对参考长度（每条回答 1000 字符）的比例做校正：
$$\bar{s}_{LC} = \bar{s}\cdot\frac{l_{ref}}{l},\quad l_{ref}=1000$$
得到 **MoReBench-Hard**，逼模型不仅要想得全面、还要想得高效——模拟人类在有限时间内做道德决策的真实压力。

**4. 廉价但可靠的 LLM-judge 选型**。用 100 个样本 × 3 个模型回答 × 2 位专家独立打标（Cohen's κ=0.75，一致性极好）得到 7,176 个"回答-准则"对作 ground truth。判分性能不取整体 macro-F1，而是分 5 类（3 个被测模型 + Advisor/Agent 两角色）算 macro-F1 后**取最低值**，以此作为下界估计、抵消对特定模型或角色的偏好。结果 GPT-5-high 的 F1 最高（77.46%）但成本 $156，比 GPT-oss-120b（F1 76.29%、仅 $1.91）贵 80 倍，故全实验改用 **GPT-oss-120b** 当 judge。

**5. 对 rubric 本身做元评测（判别力 + 鲁棒性）**。判别力：让专家为 30 个案例写 low/medium/high 三档质量的推理，ANOVA 显示三档分数显著不同（F(2,87)=6.34, p=0.003），且质量与分数显著正相关（Spearman rs=0.35, p<0.001），证明 rubric 能区分推理质量高低。鲁棒性：MoReBench 困境默认二选一，让两组专家分别为相反结论写高质量推理，t 检验无显著差异（high 0.53 vs alternate-high 0.55, p=0.56），证明 rubric **不偏袒任何一方结论**。

## 实验关键数据

### 主实验：思维链在 MoReBench 上的得分（按维度满足比例 %）

| 模型 | Identifying | Clear | Logical | Helpful | Harmless |
|------|------|------|------|------|------|
| GPT-5-high | 55.9 | 59.6 | 51.5 | 67.6 | 84.6 |
| GPT-5-mini-high | 58.9 | 61.1 | 53.0 | 71.1 | 85.5 |
| Claude Opus 4.1 | 52.8 | 48.4 | 43.3 | 32.3 | 82.5 |
| Gemini-2.5-Pro | 32.1 | 33.6 | 26.9 | 29.4 | 79.7 |
| DeepSeek-R1-0528 | 63.6 | 63.6 | 57.4 | 56.6 | 82.5 |
| Qwen3-235B-A22B | 69.1 | 68.4 | 65.1 | 61.2 | 83.9 |
| Qwen3-30B-A3B | 69.0 | 71.0 | 64.7 | 63.1 | 84.2 |
| **平均** | **52.7** | **53.6** | **47.9** | **50.1** | **81.1** |

### LLM-judge 选型对比（节选）

| Judge 模型 | 最低类别 F1 (↑) | 成本 $ (↓) |
|------|------|------|
| GPT-5-high | 77.46 | 156.12 |
| GPT-oss-120b | 76.29 | 1.91 |
| GPT-4.1 | 75.86 | 20.21 |
| Qwen3-235B-2507 | 75.28 | 0.86 |
| Gemini-2.5-Flash | 73.69 | 3.30 |

### 关键发现
- **打破 scaling laws**：在 MoReBench-Regular 上，GPT-5-High 和 Gemini 家族的**中等模型**得分最高，Claude 4、GPT-oss、Qwen3 家族的**最小模型**得分最高——疑似 inverse scaling（大模型隐式推理、思维链更短，可打分的中间步骤更少）。长度校正后的 Hard 设置部分逆转此趋势，支持该假设。
- **现有基准无法预测**：MoReBench 与 Chatbot Arena、Humanity's Last Exam、AIME 25、LiveCodeBench 的相关性几乎为零（Pearson's r 在 -0.245 ~ 0.216 之间），说明用户偏好和数学/代码/通用推理能力都预测不了道德推理能力。
- **能力短板分布**：模型最擅长 Harmless（77.5%，反映厂商重视内容安全），最差是 Logical（41.5%，逻辑整合权衡），其余维度中等（46~48%）。Gemini 家族在多数过程维度垫底；Claude 偏好给"中立折中"而非具体步骤，故 Helpful 偏低。
- **道德框架偏科**：MoReBench-Theory 上模型在功利主义（64.8%）和义务论（65.9%）表现最好，疑为这两类框架在学术文献占比高、且 RLHF 偏好收集间接强化；而美德伦理、契约主义表现参差，最高最低模型差距可达 44.9%（混合效应分析 F(4,52)=19.71, p<0.001 显著）。

## 亮点与洞察
- **范式转换**：从"评结果"转向"评过程"——道德困境天然没有唯一答案，反而成了过程性评测的理想试验田，这个 framing 非常巧妙。
- **rubric-as-groundtruth**：用 53 位哲学专家 + 二次 review 的 23,018 条带权准则替代标准答案，把主观判断转成可规模化、可自动化的群体共识分布，并对 rubric 本身做了判别力与鲁棒性的元评测，方法学严谨。
- **成本敏感的工程取舍**：明知 GPT-5-high 是最强 judge，仍因 80 倍成本差换用 GPT-oss-120b，并用"取最低类别 F1"作下界估计抵消偏好，体现可复现的现实考量。
- **反直觉结论有冲击力**：moral reasoning 不随规模单调提升、且与 STEM 推理能力脱钩，对"推理能力可迁移"的乐观假设是一记警钟。

## 局限与展望
- **judge 仍是瓶颈**：最低类别 F1 仅 76%，judge 的误判会直接传导到所有模型排名，且用 LLM 评 LLM 存在系统性偏好风险。
- **闭源模型思维链是"自报告"**：GPT 系列暴露的是思维链摘要而非真实 CoT，与开源模型的真实 trace 不严格可比，作者也承认这是"次优替代"。
- **思维链与最终回答一致性存疑**：仅在长度校正的 Hard 指标上观察到中等正相关（r=0.472, p=0.08），Regular 指标下不显著，无法断言思维链能代表最终行为。
- **inverse scaling 解释偏推测**：把"小模型反而高分"归因于"大模型隐式推理、可打分步骤更少"是合理假设但缺直接证据。
- **展望**：可作为道德推理对齐的训练信号（rubric reward）、扩展更多文化/价值体系的多元框架、以及验证私有测试集防污染的长期有效性。

## 相关工作与启发
- **价值评测谱系**：ETHICS、Delphi（共识价值）→ moral beliefs、value preferences、多步案例、stakeholder 视角，本文是其中第一个聚焦"推理过程"的。
- **rubric-based 评测**：受 HealthBench、PaperBench 等"难验证领域用专家 rubric"的启发，并继承了它们的长度偏置问题与校正思路。
- **过程性推理评测**：与评测科学/数学推理 trace 的工作呼应，但首次把对象扩到规范判断与道德能力。
- **启发**：当任务"没有标准答案"时，把"好"拆成大量原子化带权准则 + LLM-judge 聚合，是一条把主观质量规模化评测的通用路径，可迁移到写作、咨询、规划等开放式任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ — 首个聚焦道德推理"过程"而非"结果"的规模化评测，framing 与 rubric-as-groundtruth 方法学都很有原创性。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖闭源/开源十余个前沿模型、双指标、judge 选型对比、rubric 元评测、与四大主流基准对照，但 judge 误差和思维链一致性留有遗憾。
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、公式与维度定义到位、反直觉结论讲得透彻。
- **价值**: ⭐⭐⭐⭐⭐ — 23,018 条专家准则的数据集 + 反直觉的 scaling/迁移结论，对 AI 对齐与安全评测有长期价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Tug-of-War No More: Harmonizing Accuracy and Robustness in Vision-Language Models via Stability-Aware Task Vector Merging](tug-of-war_no_more_harmonizing_accuracy_and_robustness_in_vision-language_models.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](../../ICML2026/ai_safety/coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ICLR 2026\] Formalising Human-in-the-Loop: Computational Reductions, Failure Modes, and Legal–Moral Responsibility](formalising_human-in-the-loop_computational_reductions_failure_modes_and_legal-m.md)
- [\[ICLR 2026\] Adaptive Logit Adjustment for Debiasing Multimodal Language Models](adaptive_logit_adjustment_for_debiasing_multimodal_language_models.md)
- [\[ICLR 2026\] Identifying Robust Neural Pathways: Few-Shot Adversarial Mask Tuning for Vision-Language Models](identifying_robust_neural_pathways_few-shot_adversarial_mask_tuning_for_vision-l.md)

</div>

<!-- RELATED:END -->
