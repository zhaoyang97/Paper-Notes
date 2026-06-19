---
title: >-
  [论文解读] Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle
description: >-
  [ICML 2025][时间序列][LLM evaluation] 提出 Daily Oracle——一个每日自动从新闻生成预测性 QA 对的持续评估基准，系统性揭示了 LLM 预测能力随预训练数据过时而平滑退化的规律，TF 题平均降 21.55%、MC 题降 11.33%，且 RAG 也无法完全挽救。
tags:
  - "ICML 2025"
  - "时间序列"
  - "LLM evaluation"
  - "continuous benchmark"
  - "news forecasting"
  - "temporal generalization"
  - "knowledge cutoff"
---

# Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle

**会议**: ICML 2025  
**arXiv**: [2411.08324](https://arxiv.org/abs/2411.08324)  
**代码**: [https://agenticlearning.ai/daily-oracle](https://agenticlearning.ai/daily-oracle)  
**领域**: 时间序列 / LLM评估  
**关键词**: LLM evaluation, continuous benchmark, news forecasting, temporal generalization, knowledge cutoff

## 一句话总结

提出 Daily Oracle——一个每日自动从新闻生成预测性 QA 对的持续评估基准，系统性揭示了 LLM 预测能力随预训练数据过时而平滑退化的规律，TF 题平均降 21.55%、MC 题降 11.33%，且 RAG 也无法完全挽救。

## 研究背景与动机

**领域现状**：现有 LLM 评估基准（如 MMLU、HumanEval）是静态的一次性问题集，一旦发布就固定不变。少数动态基准（如 RealTimeQA、FreshQA）有更新机制，但更新频率低且不聚焦于预测能力。

**现有痛点**：静态基准面临两个根本性问题：(1) 随着 LLM 不断更新，基准的内容可能被泄露到训练数据中，评估结果失真；(2) 缺乏时间维度，无法追踪模型能力随时间的变化轨迹——我们不知道 LLM 的"保鲜期"有多长。

**核心矛盾**：世界在不断变化，但评估基准是静态的。现有预测类数据集（ForecastQA、AutoCast）要么规模小、要么不持续更新、要么依赖人工出题难以扩展。没有一个基准能同时满足"持续更新"和"评估预测能力"两个需求。

**本文目标**：构建一个"永不过时"的持续评估框架，回答核心问题——LLM 的预测能力如何随知识截止日期过后而退化？RAG 能否拯救过时的知识？

**切入角度**：利用每日新闻的天然时间特性——今天的新闻就是昨天的"未来事件"。从新闻中自动生成预测性问题，新闻的发布日期天然作为答案的验证时间点。

**核心 idea**：用每日新闻自动生成的预测性 QA 对作为持续评估探针，在时间轴上追踪 LLM 预测能力的退化曲线。

## 方法详解

### 整体框架

Daily Oracle 的 pipeline：每天抓取新闻文章 → LLM 自动生成预测性 True/False 和 Multiple Choice QA 对 → 7 条原则自动过滤低质量问题 → 用后续实际新闻结果验证答案 → 在 2020.1-2024.12 的时间轴上持续评估各 LLM。当前数据集包含 16,783 个 TF 问题和 14,727 个 MC 问题，日均 17.2 题。

### 关键设计

1. **四步 QA 生成流水线**:

    - 功能：从每日新闻文章自动生成高质量预测性 QA 对
    - 核心思路：(1) Article Summary——用 LLM 提取新闻中的新事件摘要，过滤掉评论性文章；(2) QA Generation——每篇文章生成 2 个 TF 和 2 个 MC 问题，TF 题强制一正一反保证平衡；(3) Misleading Choices——为 MC 题生成 3 个具有迷惑性的错误选项；(4) QA Filtering——按 7 条原则（答案正确性、不可提前回答性、无信息泄露、客观性、含时间要素、公共兴趣、答案非显而易见）逐条打 0/1/2 分，总分≥13 才保留
    - 设计动机：每一步都针对预测性 QA 的特殊需求设计——摘要步骤确保问题关于新事件（才能作为预测题），过滤步骤确保问题在发布日期前确实不可回答（否则不是真正的预测）。人工评估显示 LLM 过滤与人工共识的一致率达 85%

2. **三种评估设置（Closed/Open/Gold）**:

    - 功能：系统性地分离"知识过时"和"推理能力不足"的影响
    - 核心思路：Closed-Book（闭卷）纯测模型内建知识；Constrained Open-Book（有限开卷）用 BM25 检索最多到某个时间截止日的 top-5 新闻作为 RAG 上下文（每篇截断 512 词），引入 RAG Cutoff 概念限制可访问信息的最新日期；Gold Article（金标准）直接给出生成问题的原文，退化为阅读理解任务以验证题目本身可答
    - 设计动机：三个设置形成逐步放开信息量的梯度——闭卷暴露知识过时，开卷测 RAG 补偿效果，金标准验证数据集质量。RAG Cutoff 的设计特别巧妙，防止模型通过检索直接获取答案

3. **时间退化分析框架**:

    - 功能：量化 LLM 性能退化的时间规律
    - 核心思路：按月计算准确率，绘制 5-month moving average 曲线；按年统计平均准确率和 Year-over-Year (YoY) 变化率；特别区分知识截止日期前后（Pre-Cutoff vs Post-Cutoff）的退化速率
    - 设计动机：区分 Pre/Post-Cutoff 是关键——如果退化只发生在 Cutoff 之后，说明问题纯粹是知识过时；如果 Cutoff 之前也有退化，说明有更深层的时间泛化问题

### 损失函数 / 训练策略

纯评估工作，不涉及模型训练。评估指标为准确率，TF 题随机基线 50%，MC 题（4 选 1）随机基线 25%。

## 实验关键数据

### 主实验表格

| 模型 | 知识截止 | 2020 TF/MC | 2024 TF/MC | 平均YoY(TF) | 平均YoY(MC) |
|------|---------|-----------|-----------|------------|------------|
| Claude-3.5-Sonnet | 2024.4 | 81.2/76.9 | 64.3/61.8 | -5.58% | -5.03% |
| GPT-4 | 2023.4 | 69.7/70.6 | 56.9/51.6 | -4.75% | -7.04% |
| GPT-3.5 | 2021.9 | 62.9/50.3 | 56.1/43.1 | -2.84% | -3.08% |
| Llama-3-8B | 2023.3 | 65.1/52.4 | 57.0/47.0 | -2.97% | -2.30% |
| Mixtral-8x7B | 未知 | 57.8/57.4 | 36.0/46.3 | -10.78% | -4.68% |
| Gemma-2-2B | 2024.7 | 58.7/47.9 | 55.8/43.3 | -1.04% | -1.98% |

整体平均：TF 从 64.68% 降至 50.74%（-21.55%），MC 从 58.30% 降至 51.69%（-11.33%）。

### 消融表格（Pre/Post Knowledge Cutoff 退化对比）

| 模型 | Pre-Cutoff YoY(TF) | Post-Cutoff YoY(TF) | Pre-Cutoff YoY(MC) | Post-Cutoff YoY(MC) |
|------|-------------------|--------------------|--------------------|---------------------|
| Claude-3.5-Sonnet | -4.77% | -12.41% | -6.26% | -11.78% |
| GPT-4 | -5.83% | -1.96% | -4.23% | **-18.54%** |
| GPT-3.5 | -4.33% | -3.43% | +0.14% | -0.31% |
| Llama-3-8B | -1.95% | -6.50% | -2.21% | -1.25% |
| Gemma-2-2B | -1.41% | -3.68% | -4.46% | -4.07% |

### 关键发现

- **退化是平滑且持续的**：不是在知识截止日期突然断崖式下降，而是渐进式退化，暗示世界知识在模型中是逐步"过期"的
- **Post-Cutoff 退化显著加速**：GPT-4 在 MC 上 Post-Cutoff YoY 达 -18.54%，是 Pre-Cutoff(-4.23%)的 4.4 倍
- **RAG 提升但不消除退化**：开卷设置下准确率有所回升，但下降趋势依然存在；过时的 RAG 上下文甚至可能拖累表现（Llama-3-8B 在旧 RAG Cutoff 下比闭卷更差）
- **Gemma-2-2B 最稳定**：得益于最晚的知识截止日期（2024.7），YoY 变化最小
- **Mistral/Mixtral 在 TF 上跌破随机基线**：主要原因是模型拒绝回答（"I cannot predict the future"），拒绝率显著上升

## 亮点与洞察

- **"永不过时"的评估范式**：基于每日新闻的自动 QA 生成框架，从结构上保证了基准永远不会过时，这是一种可推广到其他领域的评估思路
- **退化的平滑性**出人意料——暗示 LLM 对世界知识的"遗忘"不是二元的截止切换，而是一种连续的时间衰减
- **RAG 不是万能药**：即使提供最新信息，模型仍需"理解时间上下文"才能做出正确预测，这对 RAG 系统的设计有重要启示

## 局限性

- QA 生成本身依赖 GPT-4/GPT-3.5，可能引入特定偏差（某些类型的问题更容易被生成）
- 仅覆盖新闻领域（政治、商业、体育等），科学发现、技术突破等领域的时间泛化特征可能不同
- 无法精确区分"知识过时"和"推理能力不足"——即使在 Gold Article 设置下，部分问题仍可能因推理困难而被答错
- 数据集规模（31.5K 问题）虽是同类最大，但某些细分类别（如 Healthcare）的样本可能不足

## 相关工作与启发

- **vs ForecastBench**：ForecastBench 双周更新 1000 题，依赖人工提交；Daily Oracle 每天自动生成，规模大 30 倍且无需人工干预
- **vs RealTimeQA**：RealTimeQA 测试模型是否获取了最新知识（搜索引擎式），Daily Oracle 测试模型能否预测未来（预测式），本质上测试的能力不同
- **vs FreshQA**：FreshQA 关注答案随时间变化的事实问题，Daily Oracle 关注预测未来事件——前者是"知识是否更新"，后者是"能否时间外推"

## 评分

- 新颖性: ⭐⭐⭐⭐ 持续评估范式和"日更新"基准设计有独到价值
- 实验充分度: ⭐⭐⭐⭐ 5 年时间跨度、8 个模型、3 种评估设置，分析全面
- 写作质量: ⭐⭐⭐⭐ 图表清晰，分析逻辑严密
- 价值: ⭐⭐⭐⭐ 揭示 LLM 时间泛化退化的系统性规律，对模型更新策略和 RAG 设计有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs](../../NeurIPS2025/time_series/nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](../../ICLR2026/time_series/edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICML 2026\] Embedding Hybrid Systems into Continuous Latent Vector Fields](../../ICML2026/time_series/embedding_hybrid_systems_into_continuous_latent_vector_fields.md)
- [\[ACL 2025\] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models](../../ACL2025/time_series/revisiting_llms_as_zero-shot_time_series_forecasters_small_noise_can_break_large.md)

</div>

<!-- RELATED:END -->
