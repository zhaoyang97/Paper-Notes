---
title: >-
  [论文解读] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs
description: >-
  [AAAI2026 Oral][LLM安全][LLM公平性] 提出Gender-ECE指标，系统评估六种开源LLM在性别代词预测任务中的置信度校准与人类偏见对齐程度，发现Gemma-2模型校准最差且存在极端的男女代词校准差异，而训练数据过滤较少的GPT-J-6B反而校准最好。 LLM在招聘、医疗、法律等高风险领域的广泛应用…
tags:
  - "AAAI2026 Oral"
  - "LLM安全"
  - "LLM公平性"
  - "性别偏见"
  - "置信度校准"
  - "Expected Calibration Error"
  - "共指消解"
  - "Gender-ECE"
---

# The Confidence Trap: Gender Bias and Predictive Certainty in LLMs

**会议**: AAAI2026 Oral  
**arXiv**: [2601.07806](https://arxiv.org/abs/2601.07806)  
**作者**: Ahmed Sabir, Markus Kängsepp, Rajesh Sharma (University of Tartu)  
**代码**: [GitHub](https://github.com/ahmedssabir/GECE)  
**领域**: AI安全  
**关键词**: LLM公平性, 性别偏见, 置信度校准, Expected Calibration Error, 共指消解, Gender-ECE  

## 一句话总结

提出Gender-ECE指标，系统评估六种开源LLM在性别代词预测任务中的置信度校准与人类偏见对齐程度，发现Gemma-2模型校准最差且存在极端的男女代词校准差异，而训练数据过滤较少的GPT-J-6B反而校准最好。

## 背景与动机

LLM在招聘、医疗、法律等高风险领域的广泛应用，使得模型的可信度问题日益突出。模型不仅从训练数据中继承性别偏见，还可能放大刻板印象。关键问题不仅在于检测偏见，还在于确保用户能可靠地解读模型预测——尤其是当模型在不同性别群体上表现出不均匀的置信度分布时。

Calibration（校准）是模型可信度的核心维度：一个校准良好的模型，当其预测置信度为80%时，实际正确率也应约为80%。然而，现有关于LLM偏见的研究几乎没有考察模型在偏见预测中的校准质量。如果模型在性别偏见场景中过度自信（overconfident）却频繁出错，部署风险极高。

虽然已有大量工作研究LLM中的偏见与刻板印象，但几乎没有研究考察模型预测的置信度是否与人类标注的偏见判断一致。本文填补了这一空白，聚焦于性别代词共指消解（gendered pronoun resolution）任务，分析LLM的概率校准表现，并提出Gender-ECE——一种专为衡量性别校准差异设计的新指标。

## 核心问题

**LLM在性别代词解析任务中的预测置信度，在多大程度上是校准良好的？** 具体而言：(1) 模型自信的预测是否确实正确？(2) 男性与女性代词的校准误差是否存在系统性差异？(3) 校准指标能否捕捉公平性相关的偏差？

## 方法详解

### 整体框架

对于包含代词的句子 $S = (w_1, w_2, \ldots, w_T)$，提取模型在代词位置 $k$ 的概率：

$$P(w_p \mid w_1, \ldots, w_{k-1}) = \frac{e^{z_{k-1, w_p}}}{\sum_{j=1}^{V} e^{z_{k-1, j}}}$$

其中 $z_{k-1, w_p}$ 是模型在位置 $k$ 对代词 $w_p$ 的logit值。通过比较模型对"him"与"her"的概率分配，评估模型在性别偏见职业（如nurse、developer）语境下的置信度表现。评估流程：确定性前向传播 → 提取代词token概率 → 使用offset mapping精确对齐 → 计算多种校准指标。

### 关键设计：Gender-ECE

标准ECE将预测分入 $M$ 个bin $B_m$，计算每个bin中平均置信度与平均准确率的加权绝对差：

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

但ECE无法揭示模型对男性与女性代词的行为差异。本文提出Gender-ECE：

$$\text{Gender-ECE} = \frac{1}{2} \left( \text{ECE}_{\text{male}} + \text{ECE}_{\text{female}} \right)$$

其中 $\text{ECE}_{\text{male}}$ 和 $\text{ECE}_{\text{female}}$ 分别在模型**预测标签**为男性和女性的子集上计算。与MacroCE按预测正确/错误分组不同，Gender-ECE按预测的性别标签分组，直接反映模型对不同性别代词的校准质量。与cc-ECE按真实标签分组不同，Gender-ECE关注的是模型偏好（predicted label），更能揭示模型的置信度偏见。

### 其他校准指标

- **ICE（Instance Calibration Error）**: $\text{ICE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{p}_i|$，逐实例计算置信度与真实标签差异
- **MacroCE**: 将实例按正确/错误分组，分别计算ICE后取平均
- **Brier Score**: $\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2$，概率预测的均方误差

## 实验关键数据

### 数据集

- **WinoBias**: 3,160句Winograd-style句子，代词在句中位置，测试职业性别偏见
- **Winogender**: 720句模板句，引入gender-neutral "someone"避免刻板印象
- **GenderLex**: 1,676句对，代词置于句末（last cloze），用ChatGPT生成+人工校正

### 模型: GPT-J-6B, Llama-3.1-8B, Gemma-2-9B, Qwen2.5-7B, Falcon3-7B, DeepSeek-8B

### GenderLex上校准结果（Table 1）

| 模型 | ECE↓ | MacroCE↓ | ICE↓ | Brier↓ | Gender-ECE(Group) | G-ECE(M) | G-ECE(F) | Human↑ |
|------|------|----------|------|--------|-------------------|----------|----------|--------|
| GPT-J-6B | **0.076** | 0.453 | 0.374 | 0.432 | **0.076** | 0.085 | 0.066 | 0.715 |
| Llama-3.1-8B | 0.111 | 0.466 | 0.371 | 0.446 | 0.111 | 0.112 | 0.109 | **0.727** |
| Gemma-2-9B | 0.327 | 0.493 | 0.390 | 0.559 | 0.267 | 0.330 | 0.204 | 0.617 |
| Qwen2.5-7B | 0.106 | 0.476 | 0.422 | **0.385** | 0.107 | 0.052 | 0.162 | 0.637 |
| Falcon3-7B | 0.161 | 0.491 | 0.449 | 0.356 | 0.149 | 0.081 | 0.217 | 0.605 |
| DeepSeek-8B | 0.085 | 0.461 | 0.369 | 0.470 | 0.090 | 0.074 | 0.106 | 0.686 |

### 男女代词分别ECE（Table 3）

| 模型 | WinoBias-M | WinoBias-F | GenderLex-M | GenderLex-F |
|------|-----------|-----------|-------------|-------------|
| GPT-J-6B | 0.206 | 0.508 | 0.373 | 0.377 |
| Llama-3.1-8B | 0.197 | 0.559 | 0.396 | 0.333 |
| **Gemma-2-9B** | **0.067** | **0.895** | **0.056** | **0.901** |
| Qwen2.5-7B | 0.130 | 0.596 | 0.426 | 0.416 |
| Falcon3-7B | 0.215 | 0.502 | 0.505 | 0.363 |
| DeepSeek-8B | 0.158 | 0.606 | 0.303 | 0.469 |

Gemma-2-9B在男性代词上ECE仅6-7%，女性代词上高达89-90%，差异极端。

### WinoQueer（LGBTQ+偏见，Table 4）

| 模型 | Gay | Lesbian | Trans | Queer |
|------|-----|---------|-------|-------|
| GPT-J-6B | 0.121 | 0.790 | 0.816 | 0.700 |
| Qwen2.5-7B | 0.189 | **0.898** | **0.919** | **0.788** |
| Gemma-2-9B | **0.026** | 0.221 | 0.586 | 0.182 |
| DeepSeek-8B | 0.277 | 0.838 | 0.258 | 0.910 |

在LGBTQ+任务上，结论反转：Gemma-2-9B反而校准最好，Qwen2.5-7B最差。

### Beta校准后的改善

| 模型 | 校准前准确率 | 校准后准确率 |
|------|------------|------------|
| GPT-J-6B | 69.2% | 76.9% |
| Llama-3.1-8B | 65.8% | 74.9% |
| Qwen2.5-7B | 61.1% | 76.4% |
| Gemma-2-9B | 51.6% | 54.7% |
| DeepSeek-8B | 63.5% | 69.9% |

Beta post-hoc calibration使ECE降低约3倍，同时提升准确率，但不构成偏见缓解策略。

### 模型规模影响（Table 6, WinoBias）

| 模型 | ECE | Gender-ECE(M) | Gender-ECE(F) |
|------|-----|---------------|---------------|
| Gemma-2-9B | 0.429 | 0.438 | 0.156 |
| Gemma-2-27B | 0.366 (↓14.7%) | 0.341 (↓22.1%) | 0.381 (↑144.2%) |

增大模型规模改善了男性代词校准，但女性代词校准误差暴增144.2%。

## 亮点

- **Gender-ECE指标**：首次按预测性别标签分组计算ECE，直接揭示模型对不同性别的置信度偏见，比标准ECE和MacroCE更有针对性
- **反直觉发现**：训练数据过滤最少的GPT-J-6B校准最好，暗示数据增强（如gender swapping）可能反而扰乱模型置信度
- **极端性别差异**：Gemma-2在WinoBias上男性ECE仅7%、女性ECE高达90%，定量揭示了严重的性别校准不对称
- **蒸馏传播偏见**：DeepSeek-8B（从Llama-3.1-8B蒸馏）校准误差更高、人类对齐更低，表明蒸馏过程损害了校准质量
- **规模悖论**：模型增大改善了男性校准但恶化了女性校准（Gemma-2-27B女性ECE↑144%），挑战了"bigger is better"的假设

## 局限与展望

- **仅限英语和二元性别代词**：未涉及非二元性别代词（they/them）和多语言场景
- **仅模板句评估**：Winograd-style控制句与自然语言差距大，Table 8显示在自由文本caption中ECE差异消失
- **校准≠偏见缓解**：Beta calibration改善了置信度可靠性，但并不消除底层偏见
- **样本量敏感性**：Table 7显示样本量50时ECE标准差达0.038，小数据集下校准评估不可靠
- **未涉及闭源模型**：仅评估开源模型，缺少GPT-4、Claude等闭源模型对比
- **人类标注一致性中等**：GenderLex的标注者间Cohen's $\kappa = 0.51$，仅moderate agreement

## 与相关工作的对比

- **Kadavath et al. (2022)**: 通过prompting评估LLM自我置信度 $P(\text{true})$，本文直接使用token logit概率，更底层、更可复现
- **Kapoor et al. (2024)**: 通过fine-tuning校准QA任务置信度，本文聚焦于偏见场景的校准，无需额外训练
- **Zhao et al. (2018) WinoBias**: 提出性别偏见benchmark但未考察校准质量，本文在此基础上增加了calibration维度
- **MacroCE (Si et al. 2022)**: 按正确/错误分组的instance-wise校准，Gender-ECE改为按预测性别分组且采用bin-wise计算，更稳定更可解释
- **Cheng et al. (2023)**: 通过生成persona描述评估LLM刻板印象，方法更开放但难以量化比较

## 启发与关联

- **置信度审计作为部署门控**：在高风险应用部署LLM前，应检查不同人口统计学子群体的校准差异，Gender-ECE可作为标准检查项
- **数据过滤的双刃剑**：GPT-J校准最好的发现提示，过度的数据清洗/增强可能引入新的校准问题，值得在数据工程中注意
- **蒸馏需要校准感知**：蒸馏不仅要保持性能，还需关注校准质量的传递，可考虑在蒸馏loss中加入calibration-aware正则项
- **与AI Safety评估的结合**：可将Gender-ECE扩展为通用的Group-ECE，按任意属性（种族、年龄、国籍）分组评估校准公平性

## 评分

- 新颖性: ⭐⭐⭐⭐ — Gender-ECE指标有新意，将calibration与fairness交叉分析的视角较新，但技术深度有限（ECE的简单变体）
- 实验充分度: ⭐⭐⭐⭐ — 三个benchmark + WinoQueer + 规模消融 + post-hoc calibration实验，覆盖面广；缺少闭源模型和非英语评估
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，RQ驱动，表格和reliability diagram丰富；部分分析重复
- 价值: ⭐⭐⭐⭐ — 实用性强，Gender-ECE可直接用于LLM部署前的公平性审查，Gemma-2的极端发现具警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[AAAI 2026\] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)
- [\[ACL 2026\] ADVICE: Answer-Dependent Verbalized Confidence Estimation](../../ACL2026/llm_safety/advice_answer-dependent_verbalized_confidence_estimation.md)
- [\[NeurIPS 2025\] TRAP: Targeted Redirecting of Agentic Preferences](../../NeurIPS2025/llm_safety/trap_targeted_redirecting_of_agentic_preferences.md)

</div>

<!-- RELATED:END -->
