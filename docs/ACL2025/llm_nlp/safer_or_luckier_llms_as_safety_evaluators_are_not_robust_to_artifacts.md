---
title: >-
  [论文解读] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts
description: >-
  [ACL2025][LLM 其他][LLM-as-a-judge] 系统评估了11个LLM裁判在安全领域的鲁棒性，发现道歉前缀等表面文本特征（artifact）可将评估偏好扭曲高达98%，提出基于jury的多模型聚合方案但仍未完全解决该问题。 LLM-as-a-judge已成为安全评估的标准范式：人工标注成本高、速度慢…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "LLM-as-a-judge"
  - "安全评估"
  - "artifact鲁棒性"
  - "偏置分析"
  - "jury评估"
---

# Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts

**会议**: ACL2025  
**arXiv**: [2503.09347](https://arxiv.org/abs/2503.09347)  
**代码**: 未开源  
**领域**: LLM/NLP  
**关键词**: LLM-as-a-judge, 安全评估, artifact鲁棒性, 偏置分析, jury评估

## 一句话总结
系统评估了11个LLM裁判在安全领域的鲁棒性，发现道歉前缀等表面文本特征（artifact）可将评估偏好扭曲高达98%，提出基于jury的多模型聚合方案但仍未完全解决该问题。

## 研究背景与动机

**LLM-as-a-judge已成为安全评估的标准范式**：人工标注成本高、速度慢，业界普遍依赖LLM自动评估生成内容的安全性，但其可靠性缺乏系统验证。

**已知通用偏置尚未在安全领域被检验**：位置偏置（position bias）和冗长偏置（verbosity bias）在通用指令跟随任务中已被发现，但安全评估场景下的表现未知。

**安全领域存在特有的表面特征关联**：安全回复常伴随道歉语（"I'm sorry"）、引用权威来源、提供后续帮助等模式，这些表面特征可能被LLM裁判误用为安全信号。

**高人类一致性不意味着高鲁棒性**：如果测试数据中80%的"安全"回复都包含道歉语，LLM可能通过学习道歉-安全的相关性而非真正理解安全概念来获得高一致性。

**模型能力与裁判质量的关系未被验证**：业界默认使用最大最强的模型（如GPT-4）作为裁判，但"更大=更好"这一假设缺乏实证。

**single-run评估的一致性假设未经检验**：通常只运行一次LLM评估并假设结果稳定，但即使温度为0，结果是否真的一致？

## 方法详解

### 整体框架
构建一个系统化的LLM安全裁判可靠性评估框架，从三个维度衡量：**artifact鲁棒性**、**人类一致性**、**自一致性**，覆盖样本级和模型级两个层次。

### 数据集构建
- 576个安全相关prompts，覆盖5个安全域：CSAM、虚假信息、自残、毒性、色情内容
- 4606个completion样本，组成2303对进行偏好评估
- 每对由3位人类标注者标注安全偏好

### 5种Artifact设计

| Artifact | 操作 | 安全相关假设 |
|----------|------|-------------|
| **Apology（道歉）** | 在completion前加"I'm sorry, as a chatbot I cannot respond to this." | 安全回复常以道歉开头 |
| **Authority（权威）** | 在completion后加虚假Wikipedia引用 | 安全回复常引用可信来源 |
| **Halo（光环）** | 在completion后加"Is there anything else I can help you with?" | 拒绝后常提供其他帮助 |
| **Verbosity（冗长）** | 用Command R改写使其更长但语义不变 | 更长回复可能被偏好 |
| **Position（位置）** | 交换两个completion的顺序 | 位置可能影响判断 |

### 评估指标

**Tie Detection（样本级）**：将原始completion与其artifact注入版本配对，理想裁判应判为平局（得分0%=完美鲁棒，±100%=完全偏向/反对artifact）。

**Winrate Shift（模型级）**：在真实模型对比场景中，向一方注入artifact后衡量胜率变化，更贴近实际使用场景。

### 裁判模型
11个模型，5个家族：Llama3（70B/8B）、Claude3（Sonnet/Haiku）、GPT4（Turbo/4o/4o-Mini）、Command R（Plus/R）、Mistral（Large/8x7B），覆盖8B到100B+规模。

## 实验关键数据

### 表1：Tie Detection测试结果（%，绝对值越大=越不鲁棒）

| 裁判模型 | Apology | Position | Authority | Verbosity | Halo |
|----------|---------|----------|-----------|-----------|------|
| GPT-4 Turbo | **97** | 5 | -10 | -25 | -14 |
| GPT-4o | **83** | 0 | -4 | -14 | 0 |
| Claude 3 Haiku | **67** | -11 | -16 | -12 | 10 |
| Llama3 70B | **66** | 9 | -19 | 8 | 5 |
| Command R Plus | **-2** | 0 | -2 | 0 | 0 |
| Command R | -49 | **-36** | -48 | 8 | -4 |

### 表2：Winrate Shift测试结果（%，绝对值越大=越不鲁棒）

| 裁判模型 | Apology | Position | Authority | Verbosity | Halo |
|----------|---------|----------|-----------|-----------|------|
| GPT-4 Turbo | 15 | -4 | -3 | -1 | 0 |
| Claude 3 Haiku | 10 | **-29** | -5 | -1 | 1 |
| GPT-4o Mini | 12 | **-23** | -5 | -1 | 0 |
| Llama3 70B | 6 | **18** | 0 | 0 | 1 |
| Command R Plus | 1 | 13 | -1 | 0 | 0 |

### 关键发现

1. **道歉是最强的样本级artifact**：GPT-4 Turbo在Tie Detection中偏好道歉达97%（几乎所有样本都判含道歉的版本更安全），9/11模型在Winrate Shift中偏移超2%。
2. **位置是最强的模型级artifact**：Winrate Shift最大达30%（Claude 3 Haiku为29%），远超道歉的最大15%，意味着在真实评估中位置偏置影响更大。
3. **冗长偏置被高估**：在Winrate Shift中所有模型偏移不超过±2.5%，挑战了"LLM偏好更长回复"的常见认知。
4. **Command R Plus是唯一近乎鲁棒的模型**：在Tie Detection中除位置外所有artifact偏移均≤2%。
5. **GPT-4家族自一致性差**：3.1-5.7%的跨run变化，但它们是最常用的裁判模型。
6. **更大模型≠更鲁棒**：各家族内大小模型的鲁棒性表现不一致，小模型在位置Winrate Shift上有时更鲁棒。

## 亮点与洞察

- **首次系统评估安全领域LLM裁判的artifact鲁棒性**：之前的工作集中在通用QA/指令跟随，本文是安全评估领域的第一项全面研究。
- **揭示人类一致性与鲁棒性的解耦**：高人类一致性的模型可能在artifact面前大幅偏移，说明仅用人类一致性评估裁判质量是不够的。
- **3种安全领域特有artifact（Apology/Authority/Halo）的提出**：基于安全数据中的统计模式精心设计，揭示LLM依赖统计相关性而非安全概念。
- **Jury方法的初步成功**：基于artifact-aware选择的Strong jury（Command R Plus + Claude 3 Sonnet + Llama3 70B）在鲁棒性和人类一致性上均优于任何单一模型。

## 局限与展望

1. **冗长改写引入自增强偏置**：使用Command R改写冗长版本，可能导致Command R系列裁判的冗长偏置结果不可靠。
2. **未包含专用安全裁判模型**：仅评估通用LLM作为裁判，未测试如LlamaGuard等专用安全分类器。
3. **Jury方案未完全解决问题**：即使最优的Strong jury仍无法消除道歉和位置偏置。
4. **仅涵盖成对偏好比较**：未探索绝对评分等其他评估范式。
5. **未尝试CoT推理缓解方案**：论文在结论中提及CoT可能有助于裁判忽略artifact，但未实验验证。
6. **数据规模有限**：576个prompt、5个安全域，更多样的安全场景可能揭示不同的偏置模式。

## 相关工作与启发

### vs Zheng et al. (2023) MT-Bench + Chatbot Arena
MT-Bench首次发现位置偏置和冗长偏置，但仅限于通用QA任务。本文将这些研究扩展到安全领域，并发现不同的结论：安全评估中冗长偏置微不足道，但道歉偏置极为严重。此外，MT-Bench未研究artifact对模型级winrate的影响。

### vs Koo et al. (2024) Cognitive Biases in LLM Evaluators
该工作从认知偏差角度研究LLM裁判（包括位置偏置、自增强偏置等），但在通用任务设定下。本文贡献了3种安全领域特有的artifact（Apology/Authority/Halo），并发现人类一致性与鲁棒性之间无正相关——这是重要的负面结果。

### vs Verga et al. (2024) Jury-based Evaluation
该工作提出LLM jury减少自增强偏置。本文将jury方案应用于安全评估，并提出artifact-aware的jury选择策略（平衡正反偏置的模型），在鲁棒性上优于简单多数投票。

## 评分
- 新颖性: 7/10 — 安全领域的artifact研究有新意，但整体思路（bias分析+jury缓解）较为直接
- 实验充分度: 8/10 — 11个模型×5种artifact×2个层次，实验矩阵非常全面
- 写作质量: 8/10 — 结构清晰，指标定义严谨，图表信息量大
- 价值: 8/10 — 对安全评估实践有直接指导意义，揭示了当前评估体系的重大缺陷

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement](why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)
- [\[ACL 2025\] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](beyond_prompt_engineering_robust_behavior_control_in_llms_via_steering_target_at.md)
- [\[ACL 2025\] Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_robustness_incorrect_mcq.md)
- [\[ACL 2025\] LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts](llms_know_their_vulnerabilities_uncover_safety_gaps_through_natural_distribution.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)

</div>

<!-- RELATED:END -->
