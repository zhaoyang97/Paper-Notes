---
title: >-
  [论文解读] Communicating Activations Between Language Model Agents
description: >-
  [ICML 2025][LLM评测][多智能体通信] 提出让 LLM 智能体通过中间层激活（而非自然语言）进行通信的方法——在模型 B 的前向传播中间层注入模型 A 的激活向量，无需额外参数和数据，在多项推理基准上比自然语言通信提升 27%，计算量仅为 1/4。 领域现状：多 LLM 智能体通过自然语言对话（如 debate…
tags:
  - "ICML 2025"
  - "LLM评测"
  - "多智能体通信"
  - "激活空间"
  - "模型嫁接"
  - "LLM推理"
  - "计算效率"
---

# Communicating Activations Between Language Model Agents

**会议**: ICML 2025  
**arXiv**: [2501.14082](https://arxiv.org/abs/2501.14082)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 多智能体通信, 激活空间, 模型嫁接, LLM推理, 计算效率

## 一句话总结
提出让 LLM 智能体通过中间层激活（而非自然语言）进行通信的方法——在模型 B 的前向传播中间层注入模型 A 的激活向量，无需额外参数和数据，在多项推理基准上比自然语言通信提升 27%，计算量仅为 1/4。

## 研究背景与动机

**领域现状**：多 LLM 智能体通过自然语言对话（如 debate）可以提升推理能力，但计算成本随智能体数量和消息长度快速增长。

**现有痛点**：(a) 自然语言通信需要完整的生成+解析循环，计算开销大; (b) 解码过程将丰富的内部表示压缩为单一 token，丢失大量信息; (c) 研究表明模型中间层包含比输出层更丰富的实体表示。

**核心矛盾**：自然语言是为人类设计的通信媒介，未必是 LLM 间最优的通信方式。

**本文目标**：能否让 LLM 用更高效、更高信息密度的方式——直接传递激活向量——来通信？

**切入角度**：Hernandez et al. 发现模型在约一半层处已构建了丰富的实体表示，但到后面层会压缩为下一个 token 的表示——这意味着中间层激活比最终输出携带更多信息。

**核心 idea**：暂停模型 B 在第 j 层的计算，用函数 $f$ 融合模型 A 第 k 层的激活，然后继续 B 的前向传播。

## 方法详解

### 整体框架
1. 模型 A 接收 prompt $x_A$，前向传播到第 $k$ 层得到激活 $h_{A,k}$
2. 模型 B 接收 prompt $x_B$，前向传播到第 $j$ 层得到激活 $h_{B,j}$
3. 用函数 $f$ 融合最后一个 token 的激活：$h_{B,j}^{\text{new}} = f(h_{A,k}, h_{B,j})$
4. 继续 B 的剩余层前向传播，解码输出

### 关键设计

1. **激活融合函数**:

    - 功能：将两个模型的中间激活融合
    - 核心思路：测试了 sum ($a+b$)、mean ($\frac{a+b}{2}$)、replace ($a$) 和可学习线性层四种函数
    - 设计动机：简单函数（如 mean）零额外参数，已能带来显著改进

2. **层选择策略**:

    - 功能：选择最优的注入层 $j$ 和提取层 $k$
    - 核心思路：实验发现中间层（约 40-60% 深度）效果最佳
    - 设计动机：浅层信息太原始，深层信息开始退化为下一 token 预测

### 损失函数 / 训练策略
- 完全免训练（sum/mean/replace 无可学习参数）
- 可学习线性层版本仅需少量任务无关数据训练

## 实验关键数据

### 主实验

| 任务 | 自然语言 Debate | 激活通信 (mean) | 提升 |
|------|---------------|---------------|------|
| GSM8K | 52.3% | 66.5% | +14.2% |
| MMLU (平均) | 49.8% | 53.1% | +3.3% |
| Biographies | 68.2% | 86.7% | +18.5% |
| 协调游戏 | 41.0% | 68.0% | +27.0% |

计算量：<1/4 的自然语言通信

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| sum | 较好 | 信息相加可能过强 |
| **mean** | **最优** | 平衡两模型信息 |
| replace | 中等 | 完全丢弃 B 的信息 |
| learned linear | 略优于 mean | 需少量训练 |
| 浅层注入 (10%) | 差 | 信息太原始 |
| 中间层注入 (50%) | **最优** | 平衡点 |

### 关键发现
- 激活通信在小模型（1-7B）上也有效果，不像自然语言 debate 只在大模型上有效
- 中间层激活比最终输出包含更丰富的信息
- 只需 A 的部分前向传播 + B 的完整前向传播，计算效率极高

## 亮点与洞察
- **LLM 间的"心灵感应"**——跳过语言这个瓶颈，直接传递高维表示，概念上非常优雅
- 中间层是"信息最丰富"层的发现有独立研究价值
- 方法极其轻量——零参数、零训练、即插即用

## 局限与展望
- 要求两个模型有相同隐藏维度（否则需要投影层）
- 仅传递最后一个 token 的激活，未利用序列中所有 token 的信息
- 对模型对的选择敏感——不同组合效果差异大
- 未分析安全性影响（激活注入是否可能导致有害输出）

## 相关工作与启发
- **vs Multi-agent Debate**: 自然语言通信，计算量 4×+，效果差
- **vs CIPHER**: 用 tokenizer 嵌入通信，信息损失大于激活
- **vs CALM**: 用学习的注意力层融合激活，需要训练，本文无需

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 开创性地用激活替代语言进行 LLM 间通信
- 实验充分度: ⭐⭐⭐⭐ 多任务、多融合函数、多模型
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法简洁
- 价值: ⭐⭐⭐⭐⭐ 对多智能体 LLM 系统有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ICML 2026\] The ACUTE Protocol: Operationalizing Language Model Activations for Better Calibration, Utility, and Trust](../../ICML2026/llm_evaluation/the_acute_protocol_operationalizing_language_model_activations_for_better_calibr.md)
- [\[ICML 2025\] EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities](enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln.md)
- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](../../ICML2026/llm_evaluation/hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)

</div>

<!-- RELATED:END -->
