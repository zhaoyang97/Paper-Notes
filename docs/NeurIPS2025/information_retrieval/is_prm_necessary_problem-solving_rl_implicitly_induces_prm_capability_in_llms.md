---
title: >-
  [论文解读] Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs
description: >-
  [NeurIPS 2025][信息检索/RAG][过程奖励模型] 系统研究表明纯 RL 训练（无需显式 PRM 监督）能隐式诱导出强大的过程判断能力，且现有 PRM 在 DeepSeek-R1/QwQ-32B 等强推理模型上甚至不如简单多数投票有效；提出 Self-PRM 让模型用自身的内部奖励信号重排输出，一致性地优于外部 PRM。
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "过程奖励模型"
  - "强化学习"
  - "Self-PRM"
  - "推理验证"
  - "内省式评估"
---

# Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2505.11227](https://arxiv.org/abs/2505.11227)  
**代码**: 无（使用公开模型 DeepSeek-R1、QwQ-32B）  
**领域**: 信息检索  
**关键词**: 过程奖励模型, 强化学习, Self-PRM, 推理验证, 内省式评估

## 一句话总结

系统研究表明纯 RL 训练（无需显式 PRM 监督）能隐式诱导出强大的过程判断能力，且现有 PRM 在 DeepSeek-R1/QwQ-32B 等强推理模型上甚至不如简单多数投票有效；提出 Self-PRM 让模型用自身的内部奖励信号重排输出，一致性地优于外部 PRM。

## 研究背景与动机

**领域现状**：LLM 推理能力的提升有两条主要路径——RL 训练（如 DeepSeek-R1、QwQ-32B 通过 GRPO/DAPO 等算法用最终答案正确性作为奖励信号训练）和过程奖励模型（PRM，对推理过程中每一步打分，如 PRM800K、Math-Shepherd-PRM 等）。两条路线各有拥趸，但它们之间的关系几乎没有被系统研究过。

**现有痛点**：PRM 面临三个根本性限制：(1) 推理步骤的粒度定义模糊——什么算"一步"没有统一标准；(2) 高质量步级标注成本极高——需要数学专家逐步评判推理正确性；(3) 系统性地容易被 reward hacking——模型学会生成"看起来正确"但实际有缺陷的推理步骤来骗取高分。DeepSeek-R1 的技术报告就明确指出 PRM 在其训练中没有带来显著收益。

**核心矛盾**：一方面，PRM 社区投入大量资源构建步级监督数据和训练专门的奖励模型；另一方面，最强的推理模型（DeepSeek-R1、QwQ-32B）都是纯 RL 训练的。问题在于：是否真的需要显式的过程监督，还是 RL 训练本身已经隐式地赋予了模型过程判断能力？

**本文目标** (1) 系统验证 RL 训练与 PRM 能力之间的关系；(2) 评估现有 PRM 对强推理模型的实际效用；(3) 探索更好的利用模型自身推理能力来进行解答验证的方式。

**切入角度**：如果 RL 训练让模型在解题过程中学会了"什么是好的推理步骤"，那么这种理解力应该也能被用来评价他人的推理步骤——即解题能力和判断能力是同一个硬币的两面。

**核心 idea**：解题能力与过程判断能力在纯 RL 训练中共同进化，外部 PRM 对已经很强的推理模型是多余的。

## 方法详解

### 整体框架

本文是实证研究+方法提出的混合工作。首先在 ProcessBench 上系统评估多类模型的 PRM 能力，然后通过 RL 训练曲线分析能力共演化现象，最后提出 Self-PRM 和 Self-REF 两个实用框架。

### 关键设计

1. **PRM 能力的系统评估**:

    - 功能：揭示不同训练方式对过程判断能力的影响
    - 核心思路：在 ProcessBench 上评估三类模型——(1) 纯 RL 训练的推理模型（DeepSeek-R1、QwQ-32B）作为 Generative PRM 提示后使用；(2) 在 PRM 数据上显式训练的 Discriminative PRM（如 Skywork-PRM-7B、Math-Shepherd-PRM-7B 等）；(3) 指令微调模型（Qwen2.5 系列）。关键发现：纯 RL 模型的 F1 分数（DeepSeek-R1: 83.5, QwQ-32B: 83.7）全面超越所有显式训练的 PRM（最强的 Qwen2.5-Math-PRM-72B 仅 78.3），甚至超过许多远大于自身参数量的 PRM
    - 设计动机：验证"PRM 能力可从 RL 中自然涌现"的核心假说

2. **Self-REF（自参考增强 PRM）**:

    - 功能：利用模型自身生成的解答作为参考信号增强 PRM 判断
    - 核心思路：在评价推理正确性时，先让模型自己生成一个参考解答，然后基于自己的解答来判断待评价解答的正确性。对于未经过 RL 训练的指令微调模型（如 Qwen2.5-32B-Instruct），Self-REF 带来巨大提升（F1: 45.1→63.9, +18.8），但对已有强 PRM 能力的 RL 模型（QwQ-32B: 83.7→83.0）效果中性甚至略有下降——说明 RL 模型已经内化了足够的过程理解能力
    - 设计动机：探索自生成推理作为弱监督信号对不同训练阶段模型的效用

3. **Self-PRM（自身作为 PRM 的重排策略）**:

    - 功能：让强推理模型用自身的奖励信号替代外部 PRM 进行 Best-of-N 选择
    - 核心思路：给定 k 个采样输出，不使用外部 PRM（如 Qwen2.5-Math-PRM-72B）评分，而是让模型自己评价每个输出，选择自评最高分的作为最终答案。在 AIME24/25 和 CNMO24 上，Self-PRM 在大采样量（k=32/64）时一致优于外部 PRM 和多数投票
    - 设计动机：既然模型自身的过程理解更好地对齐了自身的推理行为，那么自评价应该比外部评价更准确

### RL 训练中 PRM 能力的共演化

使用 Qwen2.5-7B-Base + DAPO 从头训练，每 10 步在 ProcessBench 上评估 PRM F1。发现 F1 与解题准确率几乎同步上升，且 F1 的改善往往领先于准确率（尤其在训练初期），表明模型先学会"识别好的推理步骤"，然后才学会"生成好的推理步骤"。卡方检验在所有模型上都显著拒绝了"解题能力与判断能力独立"的零假设。

## 实验关键数据

### ProcessBench 评估（平均 F1）

| 模型类型 | 模型 | 平均 F1 |
|---------|------|---------|
| Discriminative PRM | Math-Shepherd-PRM-7B | 31.5 |
| Discriminative PRM | Skywork-PRM-7B | 42.1 |
| Discriminative PRM | Qwen2.5-Math-PRM-7B | 73.5 |
| Discriminative PRM | Qwen2.5-Math-PRM-72B | 78.3 |
| Generative PRM (RL) | **DeepSeek-R1** | **83.5** |
| Generative PRM (RL) | **QwQ-32B** | **83.7** |

### Self-PRM vs 外部PRM vs 多数投票（QwQ-32B, k=32）

| 策略 | AIME24 | AIME25 | CNMO24 |
|------|--------|--------|--------|
| 多数投票 | 86.7 | 76.7 | 83.3 |
| BoN w/ PRM (72B) | 86.7 | 76.7 | 83.3 |
| **BoN w/ Self-PRM** | **90.0** | **80.0** | **88.9** |

### Self-REF 效果

| 模型 | 原始 F1 | +Self-REF F1 | 变化 |
|------|---------|-------------|------|
| Qwen2.5-32B-Instruct | 45.1 | 63.9 | +18.8 |
| R1-Distill-Qwen-32B | 76.7 | 79.3 | +2.6 |
| QwQ-32B | 83.7 | 83.0 | -0.7 |
| DeepSeek-R1 | 83.5 | 81.1 | -2.4 |

### 关键发现

- 纯 RL 训练的模型在 PRM 任务上全面超越所有专门训练的 PRM，包括 72B 参数的大型 PRM
- 外部 PRM (Qwen2.5-Math-PRM-72B) 对 DeepSeek-R1/QwQ-32B 的 BoN 重排效果与简单多数投票持平甚至更差——用 72B 的 PRM 完全是浪费
- Self-PRM 在 k≥16 时开始显现优势，在 k=32 时达到最佳效果（AIME24: 90.0 vs 86.7）
- Self-REF 对指令微调模型帮助巨大（+18.8 F1），但对 RL 模型无效甚至略有害
- Self-PRM 的致命弱点：难题上精度极低（<10%），经常把错误解答标为正确

## 亮点与洞察

- **挑战了 PRM 的必要性假设**：这是第一篇系统论证"纯 RL 训练可以隐式产生 PRM 能力"的工作，对 PRM 领域大量资源投入的意义提出了重要质疑。核心信息是：与其花大力气标注步级数据训练 PRM，不如直接做更多 RL
- **F1 领先于准确率的发现非常有趣**：模型先学会"判断好坏"后才学会"自己做好"，这暗示过程理解可能是产出能力的前提条件，而非反过来
- **Self-PRM 的实用价值**：简单地让模型自我评估就能超越 72B 参数的外部 PRM，这对工程实践有直接指导意义——部署时不需要额外的 PRM 模型

## 局限与展望

- Self-PRM 在难题上精度极低，可能原因是模型的 reward alignment 不够好——当模型本身解题成功率低时，它的自评价也不可靠
- 分析局限于数学推理任务，对代码生成、逻辑推理等领域是否成立未验证
- 没有探索如何在 RL 训练中主动强化 PRM 能力（目前只是观察到自然涌现）
- Self-REF 对 RL 模型的轻微负面效果说明自生成参考可能引入噪声，需要更智能的参考选择机制

## 相关工作与启发

- **vs PRM800K/ProcessBench**: 这些工作投入大量人力标注步级数据来训练 PRM，本文表明纯 RL 就能达到甚至超过的效果，这可能改变该领域的研究方向
- **vs PRIME**: PRIME 引入隐式奖励机制绕过步级标注，思路与本文发现一致——显式步级监督并非必需
- **vs GenPRM**: GenPRM 用代码验证改善 PRM，是另一种避免人工标注的路径，与 Self-PRM 的思路互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统地挑战了"PRM 必要性"这一广泛接受的假设，Self-PRM 设计直觉合理
- 实验充分度: ⭐⭐⭐⭐ 大量模型对比、统计检验、训练曲线分析、Self-PRM 局限性分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，论证逻辑严密，对 Self-PRM 的局限性很诚实
- 价值: ⭐⭐⭐⭐⭐ 对 PRM 领域有重要的战略性启发——可能整条技术路线需要反思

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving in AI Education](../../ACL2026/information_retrieval/retrieval-augmented_tutoring_for_algorithm_tracing_and_problem-solving_in_ai_edu.md)
- [\[NeurIPS 2025\] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](../../ACL2025/information_retrieval/semantic_outlier_removal_with_embedding_models_and_llms.md)
- [\[ICLR 2026\] FrugalRAG: Less is More in RL Finetuning for Multi-hop Question Answering](../../ICLR2026/information_retrieval/frugalrag_less_is_more_in_rl_finetuning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs](../../ACL2025/information_retrieval/automatic_benchmark_generation_from_scientific_papers_via_retrieval-augmented_ll.md)

</div>

<!-- RELATED:END -->
