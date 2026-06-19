---
title: >-
  [论文解读] CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis
description: >-
  [ACL 2025][LLM推理][inference scaling] 提出 CoT-based Synthesizer——一种新的推理扩展策略，通过利用 CoT 推理分析多个候选回答的互补信息来合成更优的最终答案，即便所有候选回答都是错误的也能综合出正确答案，在 MATH500 上对 Llama3-8B 提升 11.8%、对 GPT-4o 提升 10.3%。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "inference scaling"
  - "answer synthesis"
  - "chain-of-thought"
  - "self-consistency"
  - "reward model"
---

# CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis

**会议**: ACL 2025  
**arXiv**: [2501.01668](https://arxiv.org/abs/2501.01668)  
**代码**: [https://github.com/RUCKBReasoning/CoT-based-Synthesizer](https://github.com/RUCKBReasoning/CoT-based-Synthesizer)  
**领域**: LLM推理  
**关键词**: inference scaling, answer synthesis, chain-of-thought, self-consistency, reward model

## 一句话总结
提出 CoT-based Synthesizer——一种新的推理扩展策略，通过利用 CoT 推理分析多个候选回答的互补信息来合成更优的最终答案，即便所有候选回答都是错误的也能综合出正确答案，在 MATH500 上对 Llama3-8B 提升 11.8%、对 GPT-4o 提升 10.3%。

## 研究背景与动机

**领域现状**：LLM 在复杂推理任务上经常一次生成的答案不正确，因此推理扩展（inference scaling）策略被广泛研究，包括 Best-of-N 和 Self-Consistency。

**Best-of-N 的问题**：对每个候选回答独立打分，无法利用候选之间的关系，容易出现 reward hacking；且当所有候选都错时无法纠错。

**Self-Consistency 的问题**：依赖精确匹配的多数票投票，当正确答案在候选中出现频率低时会失败；同样无法在所有候选都错时产出正确答案。

**核心矛盾**：传统方法假设正确答案一定存在于候选集中，但现实中所有候选可能都包含局部错误。

**本文目标** 超越"从候选中选择"的范式，转向"从候选中综合"——利用多个候选的互补信息合成出更好的答案。

**核心 idea**：训练一个轻量级 Synthesizer 模型，用 CoT 推理分析候选回答的优劣，取长补短综合出正确答案。

## 方法详解

### 整体框架
1. **推理阶段**：Policy model 对查询生成 N 个多样候选回答 → 将查询和候选输入 CoT-based Synthesizer → Synthesizer 分析各候选并综合出最终答案
2. **数据生成阶段**：自动化管线构建训练数据（两阶段：合成答案生成 + LLM 修复）
3. **训练阶段**：在生成的数据上用 SFT 训练 Synthesizer-8B

### 关键设计

1. **多样候选生成**

    - 使用较高采样温度 $t=0.9$ 和 Top-P $p=0.9$ 生成多样候选
    - 截断低概率 token 保证生成质量，同时保持多样性

2. **回答分析与综合（Response Analysis and Synthesis）**

    - **分析阶段**：Synthesizer 分析查询与每个候选的关系，考虑回答频率、相关性和准确性，但不过度依赖频率，优先考虑逻辑连贯性和事实准确性
    - **综合阶段**：当存在正确候选时，从其他候选中提取有效推理步骤进一步增强答案；当所有候选都有缺陷时，整合多个候选中的合理成分构建更完整的答案
    - 通过设计特定 prompt 实现结构化推理

3. **两阶段数据生成管线**

    - **阶段一：合成答案生成**
        - Sampling LLM（Llama3-8B-Instruct）生成候选回答集
        - Response LLM（Llama3.1-70B-Instruct）用 CoT 对候选进行分析综合
        - 多次采样 Response LLM（N=50 次），用 gold answer 过滤保留正确的合成答案
    - **阶段二：LLM 修复**
        - 当所有候选都错时,明确告知 Response LLM 所有候选不正确
        - 提示模型反思错误并从错误答案中提取有效推理步骤，综合出修正后的回答
        - 再次用 gold answer 过滤保留正确答案

### 损失函数 / 训练策略

- 使用标准自回归语言建模目标 SFT 训练：$p_\phi(y|x,R) = \prod_{i=1}^T p_\phi(y_i|x,R,y_{<i})$
- 基座模型：Llama3-8B-Instruct
- 训练数据：MATH 12k 样本扩展到 295k，WikiTQ 18k 扩展到 87k
- 数学任务用精确匹配过滤，TableQA 用 CritiqueLLM 打分过滤

## 实验关键数据

### 主实验

**MATH500 数据集平均准确率**（7 个 policy model 平均）：

| 方法 | 平均准确率 |
|------|-----------|
| CoT-prompting | 55.2% |
| Self-Consistency | 60.8% |
| ArmoRM (Best-of-N) | 60.0% |
| Scalar RM | 59.7% |
| **Synthesizer-8B (Ours)** | **62.6%** (+7.4) |

**单模型亮点**：
- Llama3-8B 在 MATH500：24.2% → 36.0%（+11.8%）
- GPT-4o 在 MATH500：62.5% → 72.8%（+10.3%）
- 在 WikiTQ 上平均 83.6%，超越所有基线
- 在 FeTaQA 上平均 86.0%，同样领先

**关键特性**：
- 8B 的 Synthesizer 能增强 70B 甚至 API 模型（GPT-4o）的性能
- 在训练数据中不包含 GSM8k 和 FeTaQA 的情况下，仍在这些任务上表现优异（零样本泛化）

### 消融实验

| 设置 | GSM8k 平均 | MATH500 平均 |
|------|-----------|-------------|
| Synthesizer-8B | 89.3 | 62.6 |
| w/o CoT training | 86.9 (-2.4) | 57.7 (-4.9) |
| w/o training | 84.1 (-5.2) | 57.5 (-5.1) |

- 去掉 CoT 训练后性能下降，且对 Llama3.1-70B 等强模型反而变差，说明 CoT 分析防止了对 Sampling LLM 的过拟合
- 不训练直接用 Llama3-8B 做综合，对强模型基本无效

### 关键发现

1. **数据扩展性**：训练数据量与性能呈 log-linear 关系，持续增长不饱和；而 Scalar RM 数据量增大后性能反而下降（因为重复指令导致过拟合）
2. **推理扩展性**：候选回答从 5 增到 25 个，Synthesizer 性能持续提升；而 Best-of-N 方法在候选增多后出现 reward hacking 下降
3. **零候选正确时的表现**：当 5 个候选全部错误时，Synthesizer 仍能合成出 7 个正确答案（SC 和 ArmoRM 此时正确数为 0）
4. **成本效率**：只需 5 个候选即可达到 SC 用 10+ 个候选才能达到的效果

## 亮点与洞察

1. **颠覆性思路**：从"选最好的"到"取长补短合成新的"，打破了传统推理扩展方法的根本局限
2. **小模型提升大模型**：8B 的 Synthesizer 能显著提升 GPT-4o 等远大于自身的模型，体现了"专精综合"的价值
3. **修复阶段巧妙**：针对所有候选都错的情况设计 LLM Repair 阶段，明确告知模型候选错误并要求反思，有效提高了训练数据覆盖率
4. **强泛化性**：在未见过的数据集和结构不同的模型上都能工作

## 局限与展望

1. 当候选数量大时需要分组迭代综合，增加了推理复杂度
2. 训练数据生成依赖较强的 Response LLM（Llama3.1-70B），成本仍然不低
3. 目前仅在数学推理和表格问答上验证，尚未扩展到代码生成、开放生成等更多任务
4. 每次推理需要多次采样 + 综合模型推理，推理延迟较高

## 相关工作与启发

- **Self-Consistency / USC**：经典推理扩展方法，但依赖"正确答案在候选中"的假设
- **Best-of-N / Reward Model**：另一主流方向，但独立评分无法利用候选间关系
- **LMCOR**：也做答案综合但直接用 gold answer 训练，缺少 CoT 分析过程
- **启发**：综合思路可推广到多 LLM ensemble——不是选最好的模型回答，而是综合所有模型的推理步骤

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| **总评** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)
- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](../../NeurIPS2025/llm_reasoning/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [\[ACL 2025\] Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](test_time_scaling_selective_qa.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [\[ACL 2025\] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)

</div>

<!-- RELATED:END -->
