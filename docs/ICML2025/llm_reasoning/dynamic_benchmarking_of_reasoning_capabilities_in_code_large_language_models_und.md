---
title: >-
  [论文解读] DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination
description: >-
  [ICML 2025][LLM推理][Code LLM benchmarking] 基于蜕变测试思想，将编程问题分解为复杂度相关的算法抽象和复杂度无关的上下文描述，通过四个 LLM Agent 协作自动生成语义等价但文本不同的编程问题变体，有效规避数据污染并评估 Code LLM 的真实推理能力，在 18 个模型上验证了框架的有效性。
tags:
  - "ICML 2025"
  - "LLM推理"
  - "Code LLM benchmarking"
  - "data contamination"
  - "metamorphic testing"
  - "dynamic evaluation"
  - "DyPass"
---

# DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination

**会议**: ICML 2025  
**arXiv**: [2503.04149](https://arxiv.org/abs/2503.04149)  
**代码**: [项目页](https://codekaleidoscope.github.io/dycodeeval.html)  
**领域**: 代码智能  
**关键词**: Code LLM benchmarking, data contamination, metamorphic testing, dynamic evaluation, DyPass

## 一句话总结

基于蜕变测试思想，将编程问题分解为复杂度相关的算法抽象和复杂度无关的上下文描述，通过四个 LLM Agent 协作自动生成语义等价但文本不同的编程问题变体，有效规避数据污染并评估 Code LLM 的真实推理能力，在 18 个模型上验证了框架的有效性。

## 研究背景与动机

**领域现状**：Code LLM（如 DeepSeek-Coder、Qwen2.5-Coder、CodeLlama 等）的评测主要依赖 HumanEval、MBPP 等公开静态基准。模型在这些基准上的 Pass@1 分数被广泛用于衡量代码推理能力。

**现有痛点**：静态基准面临严重的数据污染问题——LLM 训练语料不可避免地包含这些公开基准的数据，导致评测分数虚高、无法反映真实推理能力。现有缓解方案各有不足：LiveCodeBench 从在线平台抓取新题，但仍依赖人工出题且题目语义复杂度不可控；PPM 通过手动定义 lambda 算子生成变体，但人工成本高且变体多样性有限（BLEU-4 仅降至 0.69）。

**核心矛盾**：当 LLM 在新基准上表现下降时，无法区分是模型能力不足还是新基准更难。需要一种在保持语义复杂度不变的前提下生成足够多样评测题目的方法。

**本文目标** (1) 如何自动生成与原题语义等价但文本不同的编程问题？(2) 如何确保生成的变体不改变原题的算法复杂度？(3) 如何提供包含测试用例和标准解的完整评测？

**切入角度**：作者借鉴软件工程中蜕变测试（metamorphic testing）的思想。核心观察是：一道编程题可以分解为"复杂度相关的算法抽象"和"复杂度无关的上下文描述"两部分。修改上下文描述不改变算法逻辑和复杂度，但能产生全新的问题文本，从而破坏记忆化。

**核心 idea**：用 LLM Agent 自动替换编程问题的上下文描述（如将"雨打窗户求滤波"改为"银行交易求滤波"），保持算法抽象不变，从而生成复杂度等价但文本不同的评测变体来检测和规避数据污染。

## 方法详解

### 整体框架

DyCodeEval 接受一个种子编程问题（来自 HumanEval 或 MBPP），经过四个顺序执行的 LLM Agent 生成语义等价的新问题。新问题保留原题的标准解和测试用例（因为算法逻辑不变），但 prompt 文本完全不同。生成过程有内在随机性（场景采样×上下文生成），理论上每次运行都产生不同变体。整体流程：种子题→场景提议→上下文生成→提示重写→验证→输出变体。

### 关键设计

1. **场景提议 Agent（Scenario Proposer）**:

    - 功能：为变体问题生成多样化的应用场景，确保跨次运行的变体不重复
    - 核心思路：维护一个场景池（初始包含银行、医疗、教育等预定义场景），用 LLM 以已有场景为 few-shot 示例迭代生成新场景并加入池中，直到达到预定大小（实验中 50 个场景）。每次生成变体时从池中随机采样一个场景
    - 设计动机：场景的多样性是防止污染的第一道防线。通过 LLM 迭代扩展而非手动定义，确保场景覆盖面广且可持续增长

2. **上下文生成 Agent（Context Generator）**:

    - 功能：为种子问题的每个输入变量赋予场景相关的语义上下文
    - 核心思路：首先通过递归类型推断算法分析 ASSERT 语句中的具体值，推导每个输入变量的数据类型（如 `List[int]`、`Tuple[int|string]`）。然后 prompt LLM 根据选定场景为每个变量分配有意义的上下文名称。例如"推荐系统"场景下，`List[int]` 可能变为"用户博客的阅读量列表"。类型推断算法虽不完备但保证正确性（sound），即收集到的类型一定出现在标准解中
    - 设计动机：Python 无显式类型声明，需要自动推断类型才能为变量赋予合理的语义上下文

3. **提示重写 Agent（Prompt Rewriter）**:

    - 功能：将原始编程问题的 prompt 重写为符合新场景和上下文的版本
    - 核心思路：提供详细的场景描述和变量上下文信息，要求 LLM 执行重写任务（而非从零生成），保持原始问题的核心算法要求不变。重写任务比生成任务更简单可控，配合详细的上下文信息可以确保语义等价性
    - 设计动机：避免了从零生成时可能引入额外算法约束的风险

4. **验证 Agent（Validator）**:

    - 功能：作为概率性预言机，检查重写后的问题是否与原题保持一致
    - 核心思路：从两个角度验证：(1) 对比原始与重写 prompt 确保核心概念和事实准确性；(2) 检查标准解是否能正确解决重写后的问题。两项检查均通过才接受，否则重新生成。Claude-3.5-Sonnet 作为基础模型时一致性率达 95%
    - 设计动机：LLM 重写过程可能无意中改变题意，双重验证确保质量

### 损失函数

本文无传统意义的训练损失。核心评测指标为 Pass@K 和新提出的 DyPass@K。DyPass@K 对种子问题生成 $n$ 个语义变体 prompt，检查模型能否稳定解决所有变体，而非对同一 prompt 生成 $n$ 个候选解。DyPass 扩大了输入空间，能区分模型是记忆了问题上下文还是真正理解了算法逻辑。碰撞概率理论分析表明，50 场景×50 上下文 = 2500 种组合，重复概率极低。

## 实验关键数据

### 主实验表格：手动污染环境下的检测效果（HumanEval）

| 模型 | 泄露 0% Pass@1 | 泄露 100% Pass@1 (静态) | 泄露 100% Pass@1 (DyCodeEval) |
|------|---------------|----------------------|------------------------------|
| Llama-3.2-1B | 4.3 | 28.7 | 4.9 |
| Llama-3.2-3B | 13.4 | 42.1 | 10.4 |
| DeepSeek-Coder-1.3B | 53.0 | 72.0 | 27.4 |

100% 泄露下静态 Pass@1 大幅提升（记忆化），但 DyCodeEval 分数几乎不变甚至下降，证明框架有效抵抗污染。

### Pass@K vs DyPass@K 对比表格

| 模型 | Pass@3 | Pass@5 | Pass@10 | DyPass@3 | DyPass@5 | DyPass@10 |
|------|--------|--------|---------|----------|----------|-----------|
| Llama-3.2-1B | 0.22 | 0.27 | 0.34 | 0.17 | 0.21 | 0.26 |
| Llama-3.2-1B (污染) | **0.82** | **0.83** | **0.85** | 0.13 | 0.15 | 0.17 |
| Llama-3.2-3B | 0.35 | 0.40 | 0.48 | 0.31 | 0.36 | 0.43 |
| Llama-3.2-3B (污染) | **0.88** | **0.88** | **0.89** | 0.24 | 0.27 | 0.29 |

污染模型的 Pass@K 虚高至 0.82–0.89，但 DyPass@K 反而下降（0.13–0.29），对比鲜明。

### 关键发现

- 在野模型评测中，Qwen2.5-Coder-7B 在 HumanEval 和 MBPP 上均落在 95% 置信区间外，强烈暗示存在数据污染
- DyCodeEval 生成问题的外部 BLEU-4 仅 0.17（HumanEval）和 0.02（MBPP），远低于所有基线方法（PPM: 0.69/0.57），证明高度多样化
- 10 次独立运行的 Pass@1 方差极小，证明评测结果具有稳定性和可复现性
- 基础模型从 Claude-3.5-Sonnet 换为 Haiku 后一致性率从 95% 降至 83%，表明高能力模型对质量至关重要

## 亮点与洞察

- 蜕变测试思想从软件工程到 LLM 评测的迁移非常自然：算法抽象/上下文描述的二分法精确对应了蜕变关系的核心理念
- DyPass 指标的提出为 Code LLM 评测提供了污染感知的新标准
- Qwen2.5-Coder-7B 的污染发现具有实际警示价值
- 四个 Agent 分工明确，每步任务简单，降低了 LLM 出错概率

## 局限性

- 仅在 Python 编程问题上验证，多语言场景未探索
- 验证 Agent 是概率性预言机，存在漏检风险（Haiku 时仅 83%）
- 未考虑 CoT 推理模式下的污染检测效果
- 基础模型（Claude-3.5-Sonnet）的计算成本较高，限制了大规模部署

## 相关工作与启发

- **vs LiveCodeBench**：LCB 靠时间戳过滤旧题，DyCodeEval 靠语义变换——后者不依赖新题持续产出
- **vs PPM**：PPM 手动定义算子，多样性有限（BLEU-4=0.69）；DyCodeEval 全自动且多样性高 4 倍（BLEU-4=0.17）
- **启发**：蜕变测试思路可推广到数学推理、NLP 理解等其他 LLM 基准的防污染改造

## 评分

⭐⭐⭐⭐⭐ 蜕变测试到 Code LLM 评测的迁移极具创新性，18 个模型×2 个种子数据集的大规模实验充分，DyPass 指标有理论价值，Qwen2.5-Coder-7B 的污染发现有实际影响力。是 Code LLM 评估方法论的重要贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](../../NeurIPS2025/llm_reasoning/core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)
- [\[NeurIPS 2025\] ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](../../NeurIPS2025/llm_reasoning/chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)
- [\[ICML 2025\] Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](emergent_symbolic_mechanisms_support_abstract_reasoning_in_large_language_models.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](../../NeurIPS2025/llm_reasoning/i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](../../ACL2025/llm_reasoning/glore_long_cot_representation.md)

</div>

<!-- RELATED:END -->
