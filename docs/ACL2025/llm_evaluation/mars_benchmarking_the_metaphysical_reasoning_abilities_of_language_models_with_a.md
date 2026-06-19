---
title: >-
  [论文解读] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset
description: >-
  [ACL 2025][LLM评测] 本文提出了 **Metaphysical Reasoning（形而上推理）** 的形式化定义，将分布变化下的推理分解为三步判别过程，并构建了首个大规模评估基准 Mars（355K 标注数据），实验表明 20+ 语言模型在该任务上表现均不理想，揭示了 LLM 在理解事件组成要素变化及其因果效应方面的显著短板。
tags:
  - "ACL 2025"
  - "LLM评测"
---

# MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset

**会议**: ACL 2025  
**arXiv**: [2406.02106](https://arxiv.org/abs/2406.02106)  
**代码**: [GitHub](https://github.com/HKUST-KnowComp/MARS)  
**作者**: Weiqi Wang, Yangqiu Song (HKUST)

## 一句话总结

本文提出了 **Metaphysical Reasoning（形而上推理）** 的形式化定义，将分布变化下的推理分解为三步判别过程，并构建了首个大规模评估基准 Mars（355K 标注数据），实验表明 20+ 语言模型在该任务上表现均不理想，揭示了 LLM 在理解事件组成要素变化及其因果效应方面的显著短板。

## 研究背景与动机

- **核心问题**：要让 LLM 成为具有泛化推理能力的 conscious agent，其关键能力之一是理解环境因素或其他智能体动作引发的**分布性情境变化（distributional situational changes）**。例如天气从晴转雨时，驾驶员行为分布随之改变。
- **现有不足**：
  1. 事件中可能发生的变化范围极大，现有知识库无法穷尽覆盖
  2. 分布变化推理缺乏清晰的任务形式化定义
  3. 现有基准（PlanBench、TRAC 等）仅覆盖有限场景和变化类型，且忽略变化引起的后果（transitions）
- **目标**：形式化定义 metaphysical reasoning，构建首个系统评估 LLM 在分布变化推理能力上的大规模 benchmark

## 方法详解

### 1. 事件变化的形式化定义

将事件 e 表示为七类组件的函数：e = f(s, v, o, t, l, n, se)，分别对应主语（subject）、谓语（verb）、宾语（object）、时间量词（temporal）、空间量词（spatial）、数值属性（numerical）和子事件（sub-event）。变化通过替换其中一个组件来实现。

对 s, v, o, se 采用**概念抽象化（conceptualization）**——将实例逐级上升为更抽象的概念；对 t, l, n 采用**数值变异**——逐级增大数值或空间范围。这构建了变化的层次化分布。

### 2. 三步判别过程

| 步骤 | 任务名称 | 核心问题 | 输入与输出 |
|------|----------|----------|-----------|
| Step 1 | Metaphysical Event Discrimination | 变化后的事件在现实中是否合理？ | 原始事件 e + 变化后事件 e' → 二分类 |
| Step 2 | Metaphysical Inference Discrimination | 变化后事件的推断结果是否合理？ | 变化后事件 e' + 推断状态 i → 二分类 |
| Step 3 | Metaphysical Transition Reasoning | 需要什么变化才能让不合理推断变合理？ | 变化后事件 e' + 形而上推断 i + 额外变化 c' → 二分类 |

### 3. 数据构建流程

采用 ChatGPT + 人工标注的流水线：

1. **文本分解与提取**：从 Wikitext 和 BookCorpus 中提取事件，用 few-shot prompt 引导 ChatGPT 分解文本并提取七类组件
2. **组件抽象与变异**：为每个组件生成 3 个逐渐增高抽象度的概念或数值变异
3. **推断生成**：对每个变化后事件分别生成 1 个合理推断和 1 个形而上推断
4. **转换生成**：为形而上推断生成使其变合理的额外变化
5. **人工标注**：通过 AMT 收集 5 票/条标注，IAA = 81%，Fleiss Kappa = 0.56；专家验证准确率 93.67%

## 实验关键数据

### 表1：Mars 各任务数据规模

| 任务 | 文本数 | 事件数 | 训练集 | 测试集 | 总计 | 专家一致率 |
|------|--------|--------|--------|--------|------|-----------|
| Meta. Event | 9,998 | 55,190 | 96,004 | 11,982 | 119,999 | 94.0% |
| Meta. Inference | 9,837 | 35,528 | 96,009 | 11,981 | 120,000 | 96.5% |
| Meta. Transition | 9,677 | 31,447 | 92,495 | 11,560 | 115,618 | 93.5% |

### 表2：主要实验结果（Accuracy %）

| 模型 | 设置 | Event Acc | Inference Acc | Transition Acc |
|------|------|-----------|---------------|----------------|
| DeBERTa-Large | Zero-shot | 48.27 | 47.73 | 50.73 |
| DeBERTa-Large | Fine-tuned | 64.45 | 69.57 | 72.93 |
| VERA 11B | Zero-shot | 51.82 | 60.97 | 61.31 |
| LLaMa-3-70B | Zero-shot | 57.41 | 63.40 | 60.15 |
| LLaMa-3.1-70B | Zero-shot | 59.22 | 63.61 | 61.28 |
| LLaMa-3.1-70B + RAG | Zero-shot | 61.21 | 66.38 | 61.53 |
| LLaMa-3.1-405B | Zero-shot | 60.01 | 64.52 | 61.74 |
| Gemma-2-9B | Fine-tuned | 61.23 | 69.24 | 73.30 |
| GPT-4 | Zero-shot | 53.90 | 51.20 | 49.41 |
| GPT-4 (COT) | Zero-shot | 51.28 | 51.49 | 47.62 |
| GPT-4o-mini + RAG | Zero-shot | 59.99 | 54.54 | 49.39 |

**关键发现**：

- 所有模型在 zero-shot 下表现不佳，最优 LLM（LLaMa-3.1-405B）Event 任务仅 60%
- 微调后最优结果约 74%，仍有大幅提升空间
- GPT-4 系列意外地不如开源 LLM，可能因负例由 ChatGPT 生成与 GPT 内部知识矛盾
- COT、few-shot 等高级 prompting 方法仅带来有限改善

### 表3：概念化知识迁移效果

| 模型 | 训练数据 | Event Acc | Inference Acc | Transition Acc |
|------|----------|-----------|---------------|----------------|
| DeBERTa 435M | Mars | 64.45 | 69.57 | 72.93 |
| DeBERTa 435M | CANDLE + Mars | **64.95** | **71.85** | **74.39** |
| LLaMa-3 8B | Mars | 60.06 | 65.76 | 69.83 |
| LLaMa-3 8B | CANDLE + Mars | **60.93** | **69.13** | **74.09** |

在 CANDLE 概念化知识上预训练再微调 Mars，三个任务均获得一致性提升，表明抽象概念化知识有助于增强形而上推理能力。

## 亮点

1. **新颖的任务形式化**：首次将分布变化推理定义为三步判别过程（事件判别→推断判别→转换推理），覆盖了变化的可行性、后果和动机
2. **大规模高质量基准**：355K 标注数据，3 个任务，7 类变化，专家验证一致率 >93%，远超同类基准规模
3. **系统的错误分析**：对 GPT-4 的错误归因为幻觉（41.7%）、概念与上位词混淆（36.3%）、内部矛盾（17.7%）和标注错误（4.3%），清晰揭示了 LLM 的失败模式
4. **可扩展的解决方案**：发现 CANDLE 概念化知识迁移可提升性能，且 CANDLE 无需人工标注即可自动构建，提供了低成本增强路径

## 局限性

1. **变化类型有限**：仅定义了 7 种组件变化，未涵盖形容词、副词、介词短语等其他可变组件
2. **依赖闭源模型**：数据构建流程依赖 ChatGPT，成本高且可复现性受限
3. **缺少实用方案**：论文聚焦于评估基准构建，未探索增强 LLM metaphysical reasoning 的系统性方法
4. **时空与数值推理仍弱**：分析显示 LLM 对 spatial/temporal/numerical 三类变化的推理最差，CANDLE 预训练对此无实质帮助

## 相关工作

- **分布变化推理**：Propara (Dalvi et al., 2018)、TRAC (He et al., 2023b)、PlanBench (Valmeekam et al., 2023) 等聚焦有限场景的状态变化追踪和逻辑推理，Mars 首次综合覆盖变化的合理性、结果和转换
- **概念抽象化**：AbsATM (He et al., 2024)、AbsPyramid (Wang et al., 2024d) 提供了概念化数据资源；CANDLE (Wang et al., 2024b) 的概念化知识可迁移增强推理
- **LLM 基准测试**：与 commonsense reasoning 基准（ATOMIC、ConceptNet）不同，Mars 关注的是分布外的抽象场景推理，更接近 System II reasoning 的目标

## 评分

- ⭐ 新颖性: 4/5 — 三步形而上推理的形式化定义新颖，首个覆盖分布变化推理的大规模基准
- ⭐ 实验充分度: 5/5 — 20+ 模型、多种设置（zero-shot/fine-tuned/API/RAG/COT）、知识迁移分析、组件级分析、错误分析
- ⭐ 写作质量: 4/5 — 结构清晰，定义严谨，但"metaphysical"一词与哲学传统含义差异较大，可能引起混淆
- ⭐ 综合价值: 4/5 — 揭示了 LLM 在抽象推理上的关键弱点，但缺少提出实用增强方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning](kristeva_close_reading_as_a_novel_task_for_benchmarking_interpretive_reasoning.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)

</div>

<!-- RELATED:END -->
