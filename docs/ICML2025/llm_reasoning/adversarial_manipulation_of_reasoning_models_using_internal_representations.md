---
title: >-
  [论文解读] Adversarial Manipulation of Reasoning Models using Internal Representations
description: >-
  [ICML 2025 (R2FM Workshop)][LLM推理][Reasoning Models] 本文发现推理模型（如 DeepSeek-R1-Distill-Llama-8B）在 CoT 生成阶段存在一个线性"谨慎方向"（caution direction），通过消融该方向可有效越狱模型，揭示了 CoT 本身是对抗攻击的新靶点。
tags:
  - "ICML 2025 (R2FM Workshop)"
  - "LLM推理"
  - "Reasoning Models"
  - "Chain-of-Thought"
  - "Jailbreak"
  - "Activation Space"
  - "Caution Direction"
  - "DeepSeek-R1"
---

# Adversarial Manipulation of Reasoning Models using Internal Representations

**会议**: ICML 2025 (R2FM Workshop)  
**arXiv**: [2507.03167](https://arxiv.org/abs/2507.03167)  
**代码**: [GitHub](https://github.com/ky295/reasoning-manipulation)  
**领域**: LLM推理 / AI安全 / 对抗攻击  
**关键词**: Reasoning Models, Chain-of-Thought, Jailbreak, Activation Space, Caution Direction, DeepSeek-R1  

## 一句话总结

本文发现推理模型（如 DeepSeek-R1-Distill-Llama-8B）在 CoT 生成阶段存在一个线性"谨慎方向"（caution direction），通过消融该方向可有效越狱模型，揭示了 CoT 本身是对抗攻击的新靶点。

## 研究背景与动机

推理模型在生成最终回答前会产生 chain-of-thought (CoT) 推理过程。与传统 LLM 不同——后者在 prompt-response 边界做出拒绝决策——推理模型的安全机制嵌入在 CoT 生成过程中。这带来一个关键问题：**CoT 是否引入了新的攻击面？**

现有越狱攻击（如 GCG、representation engineering）主要针对传统 LLM 的 prompt 处理阶段。对于推理模型，CoT 的安全角色尚未被充分理解。作者聚焦以下问题：

1. 推理模型在何处做出拒绝/遵从决策？
2. 是否存在可解释的内部表示控制这一决策？
3. 能否通过干预 CoT 阶段的激活来操纵模型行为？

## 方法详解

### 整体框架

方法基于 **representation engineering** 思路，在推理模型的激活空间中寻找与安全行为相关的线性方向，然后通过消融该方向来破坏模型的安全机制。

### 1. 谨慎方向的发现

在 DeepSeek-R1-Distill-Llama-8B 上，收集模型生成 CoT 时的中间层激活。对比处理有害 prompt 和无害 prompt 时的激活差异，通过 PCA 找到区分拒绝/遵从行为的主方向：

$$\mathbf{d}_{\text{caution}} = \text{PCA}_1\left(\mathbb{E}[\mathbf{h}_{\text{refuse}}] - \mathbb{E}[\mathbf{h}_{\text{comply}}]\right)$$

该方向被称为"谨慎方向"，因为它与 CoT 文本中的谨慎推理模式（如"我需要小心"、"这可能有害"）高度相关。

### 2. 方向消融攻击

将激活向量在谨慎方向上的投影去除：

$$\mathbf{h}' = \mathbf{h} - (\mathbf{h} \cdot \mathbf{d}_{\text{caution}}) \mathbf{d}_{\text{caution}}$$

此操作仅应用于 CoT 生成阶段的 token 激活，不干预 prompt 处理。

### 3. 与 Prompt 攻击的结合

将谨慎方向信息融入 prompt 优化攻击（如 GCG），使对抗 suffix 在激活空间中推动模型远离谨慎方向，提升攻击成功率。

### 4. CoT-only 干预的充分性

关键发现：仅干预 CoT token 的激活就足以控制最终输出，无需修改 prompt 处理阶段的激活。这证明推理模型的安全决策确实发生在 CoT 生成过程中。

## 实验

### 主实验：越狱攻击效果

| 方法 | ASR (Attack Success Rate) | 干预阶段 |
|------|---------------------------|----------|
| 无攻击（基线） | ~5% | — |
| Prompt-only GCG | ~30% | Prompt |
| CoT 谨慎方向消融 | ~75% | CoT |
| GCG + 谨慎方向消融 | ~85% | Prompt + CoT |

### 消融实验：干预位置分析

| 干预阶段 | 效果 |
|----------|------|
| 仅 Prompt 激活 | 较弱，模型仍在 CoT 中恢复谨慎 |
| 仅 CoT 激活 | 强效，直接绕过安全推理 |
| Prompt + CoT 激活 | 最强 |
| 仅最终回答激活 | 无效，决策已在 CoT 中完成 |

### 关键发现

- 谨慎方向在不同有害类别（暴力、欺诈等）间具有通用性
- 方向的线性可分性表明安全机制以简单线性方式编码
- CoT 的前几个 token 中谨慎方向的激活程度最高，表明模型在 CoT 早期就做出安全判断

## 亮点与洞察

- **CoT 是新攻击面**：推理模型将安全决策从 prompt 边界转移到 CoT 内部，但这反而暴露了新的可操纵点
- **线性安全表示**：安全行为被编码为激活空间中的简单线性方向，而非复杂的非线性结构
- **干预的精准性**：仅需干预 CoT 阶段即可控制输出，说明 CoT 是推理模型安全的核心承载
- 对 alignment 研究的启示：RLHF 等对齐方法可能只是在激活空间中创建了浅层的线性边界

## 局限性

- 仅在 DeepSeek-R1-Distill-Llama-8B（8B 参数蒸馏模型）上验证，未测试更大规模原始模型
- 谨慎方向的鲁棒性（如对模型微调后的稳定性）未充分探究
- 防御策略（如检测谨慎方向消融）未讨论
- Workshop 论文，实验规模有限

## 相关工作与启发

- **Representation Engineering (Zou et al., 2023)**：在传统 LLM 中发现安全相关的线性方向
- **GCG 攻击 (Zou et al., 2023)**：基于梯度的通用越狱 suffix 优化
- **DeepSeek-R1**：开源推理模型，CoT 推理能力强
- 本文将 representation engineering 从传统 LLM 扩展到推理模型的 CoT 阶段，开辟了新研究方向

## 评分

⭐⭐⭐⭐ — 简洁有力地揭示了推理模型安全机制的新攻击面，发现具有重要的安全研究意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models](../../AAAI2026/llm_reasoning/chain-of-thought_driven_adversarial_scenario_extrapolation_for_robust_language_m.md)
- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](../../ICML2026/llm_reasoning/floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/llm_reasoning/a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[ICLR 2026\] The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models](../../ICLR2026/llm_reasoning/the_first_impression_problem_internal_bias_triggers_overthinking_in_reasoning_mo.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](../../ACL2026/llm_reasoning/mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)

</div>

<!-- RELATED:END -->
