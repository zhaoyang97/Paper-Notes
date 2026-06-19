---
title: >-
  [论文解读] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty
description: >-
  [ACL 2025][LLM安全][factuality] 构建 ComparisonQA 基准（283K 配对问题），通过让高频和低频实体共享同一抽象问题实现受控对比，结合正确性和不确定性的两轮评估方法发现 LLM（包括 GPT-4o）对低频知识的鲁棒性极差。 领域现状：LLM 的事实性评估是热门方向…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "factuality"
  - "knowledge frequency"
  - "robustness"
  - "uncertainty"
  - "benchmark"
---

# ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty

**会议**: ACL 2025  
**arXiv**: [2412.20251](https://arxiv.org/abs/2412.20251)  
**代码**: [https://github.com/HKUST-KnowComp/ComparisonQA](https://github.com/HKUST-KnowComp/ComparisonQA)  
**领域**: LLM安全  
**关键词**: factuality, knowledge frequency, robustness, uncertainty, benchmark

## 一句话总结
构建 ComparisonQA 基准（283K 配对问题），通过让高频和低频实体共享同一抽象问题实现受控对比，结合正确性和不确定性的两轮评估方法发现 LLM（包括 GPT-4o）对低频知识的鲁棒性极差。

## 研究背景与动机

**领域现状**：LLM 的事实性评估是热门方向，PopQA、SimpleQA 等基准已发现模型在低频实体上表现差。

**现有痛点**：现有对比方法中高频和低频实体使用的问题本身不同（难度、形式均不同），无法排除问题难度差异的干扰。

**核心矛盾**：如何在严格控制变量的条件下，证明知识频率确实是影响 LLM 性能的关键因素？

**本文目标** 构建受控对比基准 + 解决语义捷径（semantic shortcut）问题。

**切入角度**：让配对实体共享同一"抽象问题"（用上位词替代具体实体），确保唯一变量是实体频率。

**核心 idea**：通过共享抽象问题+两轮评估（正确性+不确定性），实现受控且无捷径的事实性鲁棒性评估。

## 方法详解

### 整体框架
从 DBpedia 提取高低频实体对 -> 用 GPT-4 生成共享抽象问题 -> 两轮评估（第一轮测正确性，第二轮用不确定性过滤语义捷径）-> 构建 ComparisonQA-Hard 子集。

### 关键设计

1. **实体对提取**

    - 按 DBpedia 关系数量将实体分为高频（前 1/3）和低频（后 1/3）
    - 配对要求：同一上位词（如都是"城市"），确保可共享问题
    - 设计动机：DBpedia 关系数量与 LLM 训练数据频率高度相关

2. **抽象问题生成**

    - 用上位词替代具体实体名生成 MCQ（如"What is the population of this city?"）
    - 同一问题分别用高频实体和低频实体实例化
    - 设计动机：共享抽象问题确保唯一变量是实体频率

3. **两轮评估方法**

    - 第一轮：标准 MCQ 测试正确率
    - 第二轮：测量模型不确定性（token probability entropy），识别靠语义捷径猜对的问题
    - 设计动机：仅看正确率会高估模型能力

4. **ComparisonQA-Hard 子集**

    - 结合正确率和不确定性自动筛选高质量、无捷径的低频难题（81K）

## 实验关键数据

### 主实验 -- 高频 vs 低频实体正确率对比

| 模型 | 高频正确率 | 低频正确率 | 差距 |
|------|-----------|-----------|------|
| GPT-4o | ~85% | ~55% | -30% |
| Llama-3-70B | ~78% | ~45% | -33% |
| Qwen-2-72B | ~80% | ~48% | -32% |

### 鲁棒性评估（两轮 method 后）

| 模型 | 高频鲁棒率 | 低频鲁棒率 | 说明 |
|------|-----------|-----------|------|
| GPT-4o | ~70% | ~35% | 鲁棒率远低于正确率 |
| 所有模型平均 | ~65% | ~30% | 低频知识鲁棒性极差 |

### 关键发现
- **频率是确定性因素**：受控对比证明低频实体性能下降 30+ 个百分点
- **语义捷径普遍存在**：大量"答对"的题目实际上是通过选项语义线索猜出
- **不确定性是有效过滤工具**：低不确定性+高正确率的组合可有效识别捷径问题
- **GPT-4o 也不例外**：即使最强模型在低频知识上鲁棒性也极差

## 亮点与洞察
- **共享抽象问题**是解决受控对比的优雅方案，确保了因果推断的有效性
- **两轮评估方法**将不确定性引入事实性评估，弥补了纯正确率评估的盲区
- **283K 规模的配对数据集**为系统研究知识频率效应提供了丰富资源

## 局限与展望
- 频率定义依赖 DBpedia 关系数量，与实际预训练数据频率可能不完全对应
- 抽象问题由 GPT-4 生成，可能引入偏差
- 仅评估 MCQ 格式，开放式生成场景未覆盖

## 相关工作与启发
- **vs PopQA**：PopQA 使用不同问题比较不同频率实体，无法排除问题难度差异
- **vs SimpleQA**：SimpleQA 只用对抗性正确率选题，忽略语义捷径问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 受控对比设计和两轮评估方法都有创新
- 实验充分度: ⭐⭐⭐⭐ 283K数据 + 多模型 + 不确定性分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰
- 价值: ⭐⭐⭐⭐ 为事实性评估提供了更严格的方法论

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](../../NeurIPS2025/llm_safety/trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)
- [\[ACL 2025\] Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs](are_the_hidden_states_hiding_something_testing_the_limits_of_factuality-encoding.md)
- [\[ACL 2025\] Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs](fairness_difference_awareness.md)

</div>

<!-- RELATED:END -->
