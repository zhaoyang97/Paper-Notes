---
title: >-
  [论文解读] Navigating Rifts in Human-LLM Grounding: Study and Benchmark
description: >-
  [ACL2025][LLM评测][对话grounding] 系统研究人与 LLM 对话中的 grounding（建立共识）失败问题，发现 LLM 主动澄清的频率仅为人类的 1/3、主动追问的频率仅为 1/16，提出 Rifts 基准（约 1.8K 任务）评测 LLM 的 grounding 能力，并通过 grounding forecaster 实现初步干预。
tags:
  - "ACL2025"
  - "LLM评测"
  - "对话grounding"
  - "人机交互"
  - "对话行为分析"
  - "基准评测"
  - "澄清请求"
  - "指令跟随"
---

# Navigating Rifts in Human-LLM Grounding: Study and Benchmark

**会议**: ACL2025  
**arXiv**: [2503.13975](https://arxiv.org/abs/2503.13975)  
**代码**: [GitHub](https://github.com/microsoft/rifts)  
**领域**: LLM评测  
**关键词**: 对话grounding, 人机交互, 对话行为分析, 基准评测, 澄清请求, 指令跟随

## 一句话总结

系统研究人与 LLM 对话中的 grounding（建立共识）失败问题，发现 LLM 主动澄清的频率仅为人类的 1/3、主动追问的频率仅为 1/16，提出 Rifts 基准（约 1.8K 任务）评测 LLM 的 grounding 能力，并通过 grounding forecaster 实现初步干预。

## 研究背景与动机

**LLM 被训练为指令跟随者**：当前 LLM 通过 RLHF 优化指令跟随，但有效对话需要参与者协作建立共同理解（common ground）。

**Grounding 失败的代价高昂**：从用户沮丧到高风险场景的严重后果（如医疗建议、法律咨询中的误解）。

**LLM 极少主动发起澄清**：面对模糊指令，LLM 倾向于猜测用户意图并直接生成回复，而非通过提问消除歧义。

**早期 grounding 失败会级联恶化**：一次失败后，后续对话继续失败的概率从 12% 跃升至 30%（WildChat 数据）。

**缺乏系统性的 grounding 评测基准**：现有对话评测多关注端到端质量，缺少对离散 grounding 行为的细粒度评估。

**人机 grounding 不对称**：在 WildChat/Bing Chat 中，grounding 的"重活"几乎全由人类承担（修复、澄清、追问），LLM 几乎不参与。

## 方法详解

### 整体框架

定义 grounding 行为分类体系 → 构建 LLM-based 标注器标注真实对话日志 → 分析人与 LLM 的 grounding 不对称性 → 训练 grounding forecaster 预测对话走向 → 基于 forecaster 构建 Rifts 基准 → 提出并验证干预策略。

### 关键设计一：Grounding 行为分类体系

- **功能**：将对话行为分为三大类：Advancing（推进共识：Next Turn、Follow-up、Acknowledgment）、Addressing（修复失败：Reformulation、Repair、Restart）、Disambiguating（消歧：Clarification、Overresponse）。
- **核心思路**：基于 Clark & Schaefer 的经典 grounding 理论，结合 LLM 对话的特殊性，同时覆盖人类和 LLM 发起的行为。每种行为作为 grounding 状态（成功/失败/不确定）的可观测信号。
- **设计动机**：比前人工作更全面——不仅关注人类发起的行为（如追问、澄清），也纳入 LLM 发起的行为（如过度回应 Overresponse）。三层分类直接对应 grounding 的成功、失败和不确定状态。

### 关键设计二：Grounding Forecaster

- **功能**：训练一个模型，仅基于用户的初始消息预测未来对话中的 grounding 行为类别（advancing/addressing/disambiguating）。
- **核心思路**：使用条件训练（conditional training），在每条用户消息后附加一个 grounding 预测 token，微调 Llama-3.1-8B 学习预测。推理时分析预测 token 的 logits 分布来判断对话走向。
- **设计动机**：post-hoc 标注只能事后分析，forecaster 可以在对话发生前预判，从而实现主动干预。这是极具挑战性的任务——需要在看不到 LLM 回复的情况下预测用户的后续行为（相当于对所有可能的助手回复取边际化）。

### 关键设计三：Rifts 基准构建与评测

- **功能**：从 WildChat 中筛选约 1.8K 条真实用户 prompt，按 forecaster 预测的 grounding 类别分层（Advancing/Addressing/Disambiguating/No Grounding），构建标准化评测基准。
- **核心思路**：用 forecaster 过滤出 grounding 困难最大的 prompt（logit 最高的 top-150），再加入不需要 grounding 的 prompt 作为对照。评测函数：Advancing 类任务需 follow-up，Addressing/Disambiguating 类任务需 clarify，No Grounding 类任务不应做额外 grounding。
- **设计动机**：基于真实用户交互（而非人造场景），隐含的假设是某些 prompt 无论 LLM 如何回复，用户都必须来回沟通才能建立共识。基于 forecaster 的筛选比随机采样更具代表性。

### 损失函数

Grounding forecaster 使用标准的因果语言建模目标（causal language modeling loss），即在微调 Llama-3.1-8B 时对包含 grounding token 的序列计算交叉熵损失。

## 实验关键数据

### 主实验：Rifts 基准上各模型表现

| 模型 | Rifts 准确率 |
|------|-------------|
| GPT-4o | 25.26% |
| GPT-4o-mini | 24.48% |
| o3-mini | 25.26% |
| Claude Sonnet 3.5 | 26.95% |
| Claude Opus 3 | 24.57% |
| Llama 3.1 8B | 24.22% |
| Llama 3.1 70B | 23.88% |
| **Llama 3.1 8B + GROUND** | **54.48%** |
| 随机基线 | 33% |

### 消融实验：Grounding 行为统计对比

| 分析维度 | Human-LLM (WildChat/Bing) | Human-Human (MultiWOZ) |
|---------|---------------------------|----------------------|
| 人类发起修复(repair) | 高频 | 低频 |
| 人类发起澄清 vs LLM 发起澄清 | 3:1 | ~1:1 |
| 人类追问 vs LLM 追问 | 16:1 | ~2:1 |
| LLM 过度回应(overresponse) | ~30% 助手轮次 | 人类极少过度回应 |
| Session restart 率 (WildChat) | 高于单轮修复率 | — |

### 关键发现

1. **所有前沿模型在 Rifts 上低于随机基线**（avg 23.23% vs 33%），No Grounding 类准确率高达 96%，但需要主动 grounding 的类别仅 2.22%。
2. **Grounding 失败级联效应**：P(第1轮失败) = 0.12 → P(连续2轮失败) = 0.30 → P(连续3轮失败) 持续上升。
3. **简单干预（+ GROUND prompt）即可提升 32 个百分点**，从 24.22% → 54.48%，说明 LLM 有潜在能力但缺乏触发。
4. **推理模型（o3-mini）未改善 grounding**：经常不验证理解就开始推理。

## 亮点与洞察

1. **经典语言学理论与 LLM 实践的优雅结合**：将 Clark 的 grounding 理论操作化为可量化的对话行为分类，直接适用于 LLM 交互分析。
2. **Forecaster 的创新设计**：仅从用户 prompt 预测 grounding 走向，无需看到 LLM 回复，使主动干预成为可能。
3. **级联效应的量化发现**：首次用数据证明早期 grounding 失败会滚雪球式恶化对话质量。
4. **Rifts 基准的实用价值**：来自真实用户交互，可直接用于评测和改进 LLM 的对话协作能力。

## 局限性

1. Rifts 仅来自 WildChat（OpenAI 模型交互），分布偏向该平台的用户群和任务类型。
2. Grounding 行为标注依赖 GPT-4o-mini，存在标注器偏差，特别是 clarification 和 follow-up 边界模糊。
3. Forecaster 的 ROC AUC 仅为 0.61，预测能力有限。
4. 干预策略仅为简单的 prompt 追加，未探索更精细的对话策略（如多轮澄清、主动确认）。
5. 未考虑系统提示（system prompt）对 LLM grounding 行为的影响（如 Bing Chat 的元提示不可见）。

## 相关工作与启发

### vs Shaikh et al. (2024)

前作同样分析 LLM 的 grounding 行为，但仅关注人类发起的行为子集（追问、确认、澄清）。本文扩展到 LLM 发起的行为（如 Overresponse），并首次构建预测模型和标准化基准。

### vs Decision-Theoretic Dialogue (Horvitz & Paek, 2007)

决策理论方法曾用于口语对话系统中预测 grounding 失败并触发人工转接。本文将该思想迁移到 LLM 对话场景，用 forecaster 替代传统的置信度模型，用 prompt 干预替代人工转接。

### 启发

1. RLHF 训练应纳入 grounding 行为的奖励信号，forecaster 可直接作为 reward model。
2. Rifts 级别的评测应成为 LLM 对话能力的标配基准，补充现有的 instruction-following 评测。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个系统性量化人-LLM grounding 差距的工作，forecaster + benchmark 设计新颖
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多个前沿模型、标注验证完整，但 forecaster 精度有提升空间
- **写作质量**: ⭐⭐⭐⭐⭐ — 概念清晰、层层递进、图表精美、理论与实验结合紧密
- **价值**: ⭐⭐⭐⭐ — 揭示 LLM 对话的重要盲区，Rifts 基准对社区有直接推动价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles](influences_on_llm_calibration_a_study_of_response_agreement_loss_functions_and_p.md)
- [\[ACL 2025\] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States](towards_dynamic_theory_of_mind_evaluating_llm_adaptation_to_temporal_evolution_o.md)
- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)
- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)

</div>

<!-- RELATED:END -->
