---
title: >-
  [论文解读] MARGE: Improving Math Reasoning for LLMs with Guided Exploration
description: >-
  [ICML 2025][LLM推理][数学推理] MARGE 提出了一种基于"命中引导探索"（hit-guided exploration）的方法来增强 LLM 的数学推理能力，通过系统地探索自生成解答中的中间推理状态，实现充分探索和更好的信用分配，无需外部标注或额外价值模型，同时提升了单次准确率和探索多样性。
tags:
  - "ICML 2025"
  - "LLM推理"
  - "数学推理"
  - "引导探索"
  - "中间状态"
  - "信用分配"
  - "自生成数据"
---

# MARGE: Improving Math Reasoning for LLMs with Guided Exploration

**会议**: ICML 2025  
**arXiv**: [2505.12500](https://arxiv.org/abs/2505.12500)  
**代码**: [https://github.com/georgao35/MARGE](https://github.com/georgao35/MARGE)  
**领域**: LLM推理 / 数学推理  
**关键词**: 数学推理, 引导探索, 中间状态, 信用分配, 自生成数据

## 一句话总结
MARGE 提出了一种基于"命中引导探索"（hit-guided exploration）的方法来增强 LLM 的数学推理能力，通过系统地探索自生成解答中的中间推理状态，实现充分探索和更好的信用分配，无需外部标注或额外价值模型，同时提升了单次准确率和探索多样性。

## 研究背景与动机

**领域现状**：LLM 在数学推理上展现了强大潜力，但高质量训练数据的短缺限制了进一步提升。当前主流方法通过自生成数据（self-generated data）来扩展训练规模，即让 LLM 自己生成解题过程，然后用正确/错误信号进行强化学习（RL）训练。

**现有痛点**：
   - **虚假相关数据**：现有方法生成的数据中存在大量"虚假正确"的推理路径——最终答案碰巧正确，但中间推理步骤有误
   - **探索不充分**：标准 RL 方法（如 GRPO、ReST）主要从完整解答的最终结果获取反馈，对中间推理步骤的探索不足
   - **信用分配困难**：当一个多步推理最终出错时，难以判断是哪一步出了问题
   - **准确率-多样性权衡**：现有对齐方法（alignment methods）通常在提高准确率的同时降低了探索多样性（pass@k 下降）

**核心矛盾**：为了提升数学推理能力，我们需要更多高质量的训练数据；但现有的数据生成方法只利用最终答案的对错信号，浪费了中间推理步骤中蕴含的丰富信息。

**切入角度**：不仅看最终答案是否正确，而是深入到每个中间推理状态，看从这个状态出发能否到达正确答案——这就是"命中引导探索"。

**核心 idea**：从模型自生成的解答中提取中间推理状态，在这些状态上重新采样后续推理路径来判断该状态的"命中率"（hit rate），然后利用命中率引导探索和训练，使模型不仅学会"什么答案正确"，更学会"哪些推理路径更可靠"。

## 方法详解

### 整体框架
MARGE 的整体流程为迭代式的自我改进循环：
1. **生成阶段**：用当前策略为训练题目生成多条完整解答
2. **中间状态提取**：从生成的解答中提取中间推理状态（reasoning states）
3. **命中估计**：对每个中间状态，重新采样后续推理来估计其命中率
4. **训练数据构建**：利用命中率构建高质量的训练对——优势状态（高命中率）vs 劣势状态（低命中率）
5. **策略更新**：用构建的数据训练模型，提升推理能力
6. 重复上述循环

### 关键设计

1. **中间推理状态探索（Intermediate State Exploration）**:

    - 给定一条自生成的解答轨迹 $\tau = (s_0, a_0, s_1, a_1, ..., s_T)$
    - 在每个中间状态 $s_t$ 处"分叉"——保留到 $s_t$ 的推理前缀，重新采样后续步骤
    - 对每个状态采样 $K$ 条后续路径，计算有多少条最终到达了正确答案
    - 命中率 $h(s_t) = \frac{\text{到达正确答案的路径数}}{K}$
    - 设计动机：命中率反映了中间状态的"好坏"——高命中率说明从该状态容易到达正确答案，是好的推理中间结果；低命中率说明中间推理已经偏离

2. **命中引导的信用分配（Hit-Guided Credit Assignment）**:

    - 传统方法（如 DPO）只用最终结果标记整条轨迹为正/负
    - MARGE 利用命中率对每个中间步骤进行更精细的信用分配
    - 具体地，对于一条最终正确的轨迹，其中命中率低的步骤仍然可能是"虚假的好步骤"（碰巧最终正确）
    - 对于一条最终错误的轨迹，其中命中率高的步骤可能包含好的推理片段
    - 通过对比不同命中率的状态-动作对来构建训练数据
    - 设计动机：解决了传统方法中"整条轨迹一刀切"的粗糙信用分配问题

3. **探索多样性保持**:

    - 标准 RL 训练倾向于让模型坍缩到少数高奖励的固定模式，降低 pass@k
    - MARGE 通过探索中间状态的不同分支，天然保持了更高的探索多样性
    - 训练数据中既包含"从好状态探索成功"也包含"从坏状态纠正过来"的样本
    - 这使得模型学到更丰富的推理策略，而非仅记住一种解题模板
    - 设计动机：在数学推理中，pass@k 的提升意味着模型能用多种方式解决同一问题，这对实际应用（如 majority voting）非常重要

4. **无需外部标注和价值模型**:

    - 不需要人工标注中间步骤的对错
    - 不需要训练额外的 reward model 或 value model
    - 命中率完全通过模型自身的采样来估计
    - 设计动机：降低了方法的实现复杂度和计算开销

### 损失函数 / 训练策略
MARGE 使用类似 DPO 的偏好优化损失，但训练对是基于命中率构建的：高命中率的（状态, 动作）对作为 chosen，低命中率的作为 rejected。具体数据格式为 `{"query": 数学题目, "": guidance hit, "gt": 正确答案}`。训练基于开源数据集（Math-Step-DPO-10K 和 Big-Math-RL-Verified），代码使用 `backwardlearning` 框架。

## 实验关键数据

### 主实验

| 模型 | 方法 | MATH | GSM8K | 其他基准 |
|------|------|------|-------|---------|
| Qwen2.5-Math-7B-Instruct | Base | 基线 | 基线 | 基线 |
| Qwen2.5-Math-7B-Instruct | +MARGE | **显著提升** | **显著提升** | 多数提升 |
| LLaMA-3.1-8B-Instruct | Base | 基线 | 基线 | 基线 |
| LLaMA-3.1-8B-Instruct | +MARGE | 提升 | 提升 | 一致提升 |
| MetaMath-Mistral | Base | 基线 | 基线 | 基线 |
| MetaMath-Mistral | +MARGE | 提升 | 提升 | 跨架构有效 |

### 消融实验

| 配置 | 单次准确率 | pass@k | 说明 |
|------|----------|--------|------|
| 标准 DPO (整条轨迹) | 提升 | 下降 | 准确率-多样性权衡 |
| MARGE (无命中引导) | 小幅提升 | 保持 | 纯探索不够 |
| MARGE (命中引导) | **显著提升** | **提升** | 同时改善两项 |
| 不同采样数 K | 随 K 增大而改善 | - | 但边际递减 |

### 关键发现
- MARGE 是少数能同时提升单次准确率和 pass@k 的方法——打破了常见的准确率-多样性权衡
- 命中引导的信用分配是核心贡献——仅做中间状态探索但没有命中引导，效果不明显
- 在多个骨干模型（Qwen2.5-Math、LLaMA-3.1、MetaMath-Mistral）上均有效，展现了良好的通用性
- 随着自生成数据规模的扩大，MARGE 的优势更加明显——体现了其解锁自生成数据规模化潜力的能力
- 不需要额外的 reward model 或 value model，降低了实现成本

## 亮点与洞察
- **"命中率"是一个优雅的信号**：不需要训练复杂的价值网络，仅通过从中间状态重新采样就能估计每步推理的"水平"，思路简洁有效
- **打破准确率-多样性权衡**：这是 alignment 方法中的一个难题，MARGE 通过中间状态探索自然地解决了这个问题
- **可迁移到其他推理任务**：中间状态探索+命中引导的框架不限于数学推理，可以应用到代码生成、逻辑推理等任何需要多步推理的任务
- **扩展 scaling 的思路**：当训练数据有限时，通过中间状态的分叉探索可以指数级地扩展有效训练信号

## 局限与展望
- 命中率估计需要对每个中间状态采样 K 次，计算成本随 K 和轨迹长度增长
- 中间状态的"切分点"选择可能影响效果——如何确定哪些位置值得探索是一个问题
- 论文主要在数学推理上验证，在其他推理任务（如代码、常识推理）上的效果未充分验证
- 命中率的估计可能有噪声——当 K 较小时，估计不准确可能引入错误信号
- 可考虑将命中率与其他信号（如 ORM 打分）结合，进一步提高信号质量

## 相关工作与启发
- **vs Standard RL (GRPO/PPO)**：标准 RL 只从完整轨迹的最终奖励学习，MARGE 深入到中间步骤，信号更丰富且信用分配更精确
- **vs Process Reward Model (PRM)**：PRM 需要对中间步骤进行人工标注或训练单独的奖励模型，MARGE 不需要额外模型，仅用采样估计命中率
- **vs STaR/ReST**：这些自我改进方法只保留最终正确的轨迹，MARGE 还能从错误轨迹中提取有价值的中间片段

## 评分
- 新颖性: ⭐⭐⭐⭐ 中间状态命中引导的思路有新意，但整体框架与现有自我改进方法有相似之处
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准，消融分析到位，但实验数字细节可以更完整
- 写作质量: ⭐⭐⭐⭐ 动机论述清晰，方法描述有条理
- 价值: ⭐⭐⭐⭐ 同时提升准确率和多样性是有价值的贡献，对数学推理提升方法有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](../../ICLR2026/llm_reasoning/dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ACL 2025\] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](../../ACL2025/llm_reasoning/logicpro_program_guided_reasoning.md)
- [\[NeurIPS 2025\] SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers](../../NeurIPS2025/llm_reasoning/sand-math_using_llms_to_generate_novel_difficult_and_useful_mathematics_question.md)
- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[ACL 2025\] FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving](../../ACL2025/llm_reasoning/finereason_evaluating_and_improving_llms_deliberate_reasoning_through_reflective.md)

</div>

<!-- RELATED:END -->
