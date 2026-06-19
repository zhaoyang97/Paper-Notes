---
title: >-
  [论文解读] PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations
description: >-
  [ACL 2025][LLM评测][psychometrics] 提出 PATCH 框架，将心理测量学中的项目反应理论（IRT 3PL/2PL 模型）引入 LLM 基准测试，在 TIMSS 2011 八年级数学测试（88 道题、56 个国家/地区）上对比 GPT-4V、Gemini-Pro-Vision、Qwen-VL 与人类群体的能力值，发现 IRT 能力估计与简单准确率排名显著不同，GPT-4V 与韩国/新加坡/中国台北学生处于同一排名区间；同时发布 4 个高质量数据集（TIMSS 2011 & 2008 数学/科学/物理）。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "psychometrics"
  - "IRT"
  - "TIMSS"
  - "human-LLM comparison"
  - "benchmarking"
---

# PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations

**会议**: ACL 2025  
**arXiv**: [2404.01799](https://arxiv.org/abs/2404.01799)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: psychometrics, IRT, TIMSS, human-LLM comparison, benchmarking

## 一句话总结
提出 PATCH 框架，将心理测量学中的项目反应理论（IRT 3PL/2PL 模型）引入 LLM 基准测试，在 TIMSS 2011 八年级数学测试（88 道题、56 个国家/地区）上对比 GPT-4V、Gemini-Pro-Vision、Qwen-VL 与人类群体的能力值，发现 IRT 能力估计与简单准确率排名显著不同，GPT-4V 与韩国/新加坡/中国台北学生处于同一排名区间；同时发布 4 个高质量数据集（TIMSS 2011 & 2008 数学/科学/物理）。

## 研究背景与动机

**领域现状**：MMLU、GSM8K 等 LLM 学术能力基准被广泛使用，研究者常用简单准确率将 LLM 与"人类水平"进行对比，作为模型选型和发展方向的核心参考。

**测量质量问题**：现有基准的题目质量未经心理测量学验证——题目的难度、区分度完全未知，某些题目可能区分度为零甚至为负。

**评估指标粗糙**：简单准确率将所有题目等权处理——答对 10 道简单题和答对 10 道难题的准确率一样，但反映的能力完全不同。这是心理测量学已解决 50+ 年的经典问题。

**人类参照不明确**：现有基准的人类表现通常来自便利样本（如 MTurk 工人），无法代表任何明确的人类群体，"LLM 超越人类"的结论缺乏严谨意义。

**切入角度**：IRT 是教育测量学的金标准（50+ 年历史），TIMSS 是全球最大规模的标准化国际数学测试之一——两者结合同时解决"测量质量"与"人类参照"两大核心问题。

**核心 idea**：用 IRT 模型估计 LLM 的能力参数 $\theta$，在与 56 个国家/地区人类学生相同的标尺上进行公平、精确的对比。

## 方法详解

### 整体框架
选择高质量标准化测试（TIMSS 2011 八年级数学，88 道公开题目）→ 利用 56 个国家/地区约 30 万学生作答数据拟合 IRT 模型（估计每道题的难度 $b$、区分度 $a$、猜测率 $c$）→ LLM 在完全相同的 88 道题上作答 → 用已校准的 IRT 模型估计 LLM 的能力值 $\theta$ → 在同一标尺上与 56 个人类群体直接对比。

### 关键设计

1. **三参数项目反应理论模型 (3PL-IRT)**

    - 功能：对每道选择题 $j$ 估计三个参数——区分度 $a_j$、难度 $b_j$、猜测率 $c_j$
    - 核心公式：$P(\theta) = c_j + \frac{1-c_j}{1+\exp(-a_j(\theta - b_j))}$
    - 对于开放题使用 2PL 模型（$c_j = 0$），因为开放题不存在随机猜对
    - 设计动机：3PL 是教育测量领域处理多选题的标准模型，猜测参数 $c$ 对低能力模型的估计至关重要

2. **TIMSS 2011 作为评测基准**

    - 功能：使用 TIMSS（Trends in International Mathematics and Science Study）2011 年八年级数学公开题目（88 道）
    - 题目涵盖代数、几何、数据与概率、数论四大数学领域
    - 56 个国家/地区约 30 万学生的作答数据，IRT 参数已由 IEA 专家校准
    - 设计动机：TIMSS 经过严格的跨文化验证和质量控制流程，远超现有 LLM 基准的测量质量

3. **多模态 LLM 评测**

    - 功能：评测 GPT-4V、Gemini-Pro-Vision、Qwen-VL 等多模态模型
    - 核心流程：将题目（含图表/几何图形）输入 LLM → 提取答案 → 嵌入 IRT 模型估计 $\theta$
    - 设计动机：TIMSS 部分题目含图表，需多模态模型才能公平作答

4. **四个高质量数据集发布**

    - TIMSS 2011 数学 + TIMSS 2008 数学 + TIMSS 2011 科学 + TIMSS 2011 物理
    - 每个数据集均含题目原文/图像、标准答案、评分标准和 IRT 参数

## 实验关键数据

### 主实验 -- LLM vs 56 个人类群体

| 模型 | IRT 能力值 | 等价人类水平 | 简单准确率排名 |
|------|---------|------------|----------|
| GPT-4V | 高 | 前 5 国家水平 | 可能不同 |
| GPT-3.5 | 中 | 中等国家水平 | 可能不同 |
| Llama-3 | 中低 | 低于平均 | 可能不同 |

### IRT vs 简单准确率的排名差异

| 对比 | 发现 |
|------|------|
| 模型排名 | IRT 和准确率排名可能显著不同 |
| 人类对比 | IRT 提供更精确的定位 |

### 关键发现
- IRT 能力估计与简单准确率的模型排名显著不同——简单准确率可能误导
- GPT-4V 在 IRT 估计下达到前 5 国家 8 年级学生水平
- 题目难度和区分度对评估结果影响大
- TIMSS 的题目质量远高于现有 LLM 基准

## 亮点与洞察
- **将 50+ 年成熟的心理测量学理论引入 LLM 评估**是重要的方法论贡献
- **56 个国家/地区的明确人类参照**解决了现有基准“与谁比”的问题
- **IRT vs 准确率的排名差异**说明简单指标可能严重误导

## 局限与展望
- 仅测试 8 年级数学，未覆盖更高级别或其他学科
- IRT 假设 LLM 和人类遵循同一测量模型，可能不完全成立
- 改进方向：多学科拓展、自适应 IRT 测试

## 相关工作与启发
- **vs MMLU/GSM8K**：它们用简单准确率，PATCH 用 IRT
- **vs Elo rating**：Elo 用于模型间对比，PATCH 用于模型-人类对比

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将心理测量学引入 LLM 评估是重要创新
- 实验充分度: ⭐⭐⭐⭐ 多模型 x 56 人类群体
- 写作质量: ⭐⭐⭐⭐⭐ 理论基础扎实
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 基准方法论有重大推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)
- [\[ACL 2025\] WebWalker: Benchmarking LLMs in Web Traversal](webwalker_benchmarking_llms_in_web_traversal.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)

</div>

<!-- RELATED:END -->
