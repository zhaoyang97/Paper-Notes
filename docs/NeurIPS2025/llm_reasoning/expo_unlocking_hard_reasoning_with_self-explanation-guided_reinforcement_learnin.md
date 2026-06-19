---
title: >-
  [论文解读] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning
description: >-
  [NeurIPS 2025][LLM推理][自我解释] 提出 Self-Explanation Policy Optimization (ExPO)，一种通过让模型在给定正确答案条件下自主生成推理链（self-explanation）作为正样本的模块化框架，解决 GRPO 等 RL 后训练方法在困难推理任务上因缺乏有效正样本而无法学习（分布锐化）的根本问题——ExPO 生成的自解释样本既在当前策略分布内（in-distribution），又能提供正向学习信号，可无缝集成到 DPO 和 GRPO 中。
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "自我解释"
  - "GRPO"
  - "DPO"
  - "正样本生成"
  - "困难推理"
  - "分布锐化"
---

# ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2507.02834](https://arxiv.org/abs/2507.02834)  
**代码**: [GitHub](https://github.com/HumainLab/ExPO_rl_reasoning_by_explanation)  
**领域**: LLM推理 / 强化学习后训练  
**关键词**: 自我解释, GRPO, DPO, 正样本生成, 困难推理, 分布锐化  

## 一句话总结
提出 Self-Explanation Policy Optimization (ExPO)，一种通过让模型在给定正确答案条件下自主生成推理链（self-explanation）作为正样本的模块化框架，解决 GRPO 等 RL 后训练方法在困难推理任务上因缺乏有效正样本而无法学习（分布锐化）的根本问题——ExPO 生成的自解释样本既在当前策略分布内（in-distribution），又能提供正向学习信号，可无缝集成到 DPO 和 GRPO 中。

## 背景与动机

1. **GRPO 的分布锐化困境**：当前 RL 后训练方法（如 GRPO）依赖模型自身生成正样本，但在困难推理任务（如 MATH level-5）中模型初始正确率极低，所有采样答案均错误时优势项归零、KL 项主导，模型反而退化而非进步。

2. **丢弃难题不是解决方案**：现有工作通过丢弃全部答案错误的训练样本来规避问题，但这仅仅是回避而非解决——模型永远无法学会它当前无法解决的问题。

3. **专家 CoT 的局限性**：虽然使用人工撰写的专家推理链（expert CoT）看似直观，但实验发现其效果常常不如自生成样本——因为专家 CoT 在当前策略下概率极低（out-of-distribution），梯度信号对策略改进的贡献微弱。

4. **只强化已知能力**：GRPO 风格方法本质上是对模型输出分布的"锐化"（distribution sharpening）——强化高概率正确回答的概率，但无法引导模型探索全新的推理路径来解决此前完全无法解决的问题。

5. **正样本稀缺是核心瓶颈**：负样本（错误答案）在 RL 后训练中总是充足的，但在困难任务中有效的正样本极度匮乏，这是制约模型推理能力提升的根本瓶颈。

6. **缺乏理论指导**：此前关于什么样的正样本对 RL 后训练最有效，缺乏系统的理论分析——STaR 等方法虽然实用但其成功原因未被完全理解。

## 核心问题
在 RL 后训练中，当模型初始成功率极低时，如何获取能够有效指导学习的正样本？这些正样本应具备什么性质？

## 方法详解

### 理想正样本的两个性质
通过政策改进的梯度分析，论文严格论证理想正样本需满足：
1. **分布内性质**（In-distribution）：样本在当前策略 $\pi_\theta$ 下应有较高概率——概率太低则对策略改进的梯度贡献（$T_1$ 项）趋近于零
2. **正向学习信号**（Positive learning signal）：样本的 CoT $c_1$ 应使正确答案的条件概率高于其他 CoT，即 $\pi_\theta(a^*|q, c_1) > \pi_\theta(a^*|q, c_2)$

### 自解释生成
核心思想极其简洁：给定问题 $q$ 和正确答案 $a^*$，让模型自己生成推理链：
$$\tilde{c} \sim \pi_\theta(\text{cot} | q, a^*)$$
这种条件生成降低了任务难度（从开放式问题求解转为条件解释），使模型能产生比标准 CoT 质量更高但仍在策略分布内的推理链。

### 为什么自解释优于专家 CoT？
- **分布内**：自解释的 prompt 仅比标准 CoT 多几个 token（正确答案），因此 $\pi(\cdot|q, a^*)$ 与 $\pi(\cdot|q)$ 分布接近，NLL 显著低于专家 CoT
- **正向信号**：Lemma 2 证明自解释平均而言比标准 CoT 更可能导向正确答案
- **天然过程监督**：自解释与错误 CoT 的偏差较小，帮助模型精确定位需要调整的推理步骤

### ExP-DPO 实例化
- **离线版**：初始策略一次性生成所有自解释作为偏好响应，自生成 CoT 作为非偏好响应
- **在线迭代版**：周期性用更新后的策略重新生成自解释，避免分布漂移（distributional drift）导致正样本失效

### ExP-GRPO 实例化
在 GRPO 目标中添加 ExP-SFT 项 $\beta \log \pi_\theta(\tilde{c}, a^* | q)$：当所有采样答案均错误时，ExP-SFT 提供学习信号，启动原本停滞的试错循环；$\beta$ 可使用退火策略逐步减小以避免锁定不完美 CoT。

## 实验关键数据

### 表1：ExP-DPO 结果（Pass@4）

| 设置 | 模型 | 正样本类型 | MATH | GSM8K |
|------|------|-----------|------|-------|
| 离线 | Qwen2.5-3B | ExPO $\tilde{c}$ | **54.3** | **80.1** |
| 离线 | Qwen2.5-3B | 专家 CoT $c_E$ | 43.7 | 69.6 |
| 在线 | Qwen2.5-3B | ExPO $\tilde{c}$ | **60.4** | **85.4** |
| 在线 | Qwen2.5-3B | 专家 CoT $c_E$ | 49.3 | 76.3 |

ExPO 自解释在所有设置下均显著超越专家 CoT（MATH 上 +10.6~11.1%）。

### 表2：MATH 各难度级别准确率分解（Qwen2.5-3B-Instruct）

| 方法 | Level-1 | Level-2 | Level-3 | Level-4 | Level-5 |
|------|---------|---------|---------|---------|---------|
| Base (pass@64) | 97% | 88% | 75% | 32% | 4% |
| GRPO | 91% | 84% | 77% | 39% | 2% |
| GRPO SFT-GT-CoT | 95% | 89% | 83% | 65% | 12% |
| **ExP-GRPO** | **96%** | **91%** | **86%** | **76%** | **23%** |

在最困难的 Level-5 问题上，ExP-GRPO 将准确率从 GRPO 的 2% 提升至 **23%**，提升 11.5 倍。

## 亮点

1. **理论深度与实用性兼备**：通过梯度分析严格论证了理想正样本的两个必要性质，为 STaR 等先前方法的成功提供了理论解释
2. **方法极简优雅**：核心思想仅需在 prompt 中加入正确答案做条件生成，无需外部模型、无需额外标注、无需额外架构
3. **模块化设计**：可无缝嵌入 DPO 和 GRPO 两大主流框架，适配性强
4. **解决了真正的痛点**：不仅提升了样本效率，更让模型能够学会此前完全做不出的困难题目——这是从"强化已知"到"获取新能力"的质变
5. **实验发现反直觉但有说服力**：专家 CoT 效果劣于自生成解释这一发现挑战了"数据质量越高越好"的常识

## 局限与展望

1. **实验仅限数学推理**：虽然论文讨论了代码生成等扩展场景，但实际实验仅在 MATH 和 GSM8K 上进行，其他推理任务的效果待验证
2. **依赖可验证奖励**：需要能够判断最终答案是否正确（outcome verifier），对开放式生成任务不直接适用
3. **模型规模有限**：实验使用 3B 级别模型（Qwen2.5-3B, LLaMA-3.2-3B），对 7B+ 模型是否仍有显著增益未知
4. **自解释质量的上限**：如果模型在给定答案条件下仍无法生成有意义的推理链，方法将失效——作者自己也承认这意味着问题超出了 RL 后训练的能力范围
5. **SFT 项权重 $\beta$ 的调节**：论文提到可用退火策略但实验中似乎未充分探索最优调度

## 与相关工作的对比

- **vs STaR**：STaR 首先提出在给定正确答案条件下重新生成推理链，但仅限于 SFT 训练和直接 prompting；ExPO 将这一思想推广到 RL 框架并提供了理论解释
- **vs GRPO/DeepSeek-R1**：GRPO 在困难任务上存在分布锐化问题，ExPO 通过注入自解释学习信号直接解决此问题
- **vs 基于专家 CoT 的方法**：实验表明专家 CoT（out-of-distribution）在 DPO/GRPO 训练中效果反而不如自解释（in-distribution），这是反直觉但理论可解释的结果
- **vs 自我纠正/自我精炼方法**：这些方法让模型迭代改进自身输出，但在初始完全错误时缺乏有效起点；ExPO 通过条件生成提供了可靠的"冷启动"机制

## 评分
- 新颖性: ⭐⭐⭐⭐ — 将条件生成解释为理想正样本并提供严格理论支撑，将散落的直觉统一为清晰框架
- 实验充分度: ⭐⭐⭐⭐ — DPO/GRPO 双框架验证、难度级别分解分析充分，但模型规模和任务多样性有限
- 写作质量: ⭐⭐⭐⭐⭐ — 从问题定义、理论分析到算法设计再到实验验证，叙事极为流畅且自洽
- 价值: ⭐⭐⭐⭐⭐ — 直击 RL 后训练推理的核心痛点，方法简洁通用，对研究和工程实践均有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](../../ACL2026/llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
