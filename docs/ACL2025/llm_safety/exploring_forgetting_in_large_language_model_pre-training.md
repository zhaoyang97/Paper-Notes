---
title: >-
  [论文解读] Exploring Forgetting in Large Language Model Pre-Training
description: >-
  [ACL2025][LLM安全][catastrophic forgetting] 系统性地探索了 LLM 预训练阶段的灾难性遗忘问题，提出了基于实体记忆的新指标（M_ex、M_in）替代传统 PPL 来检测遗忘，并验证了周期性高强度 memory replay 策略在缓解预训练遗忘中的有效性。 灾难性遗忘（catastro…
tags:
  - "ACL2025"
  - "LLM安全"
  - "catastrophic forgetting"
  - "pre-training"
  - "entity memory"
  - "memory replay"
  - "forgetting curve"
---

# Exploring Forgetting in Large Language Model Pre-Training

**会议**: ACL2025  
**arXiv**: [2410.17018](https://arxiv.org/abs/2410.17018)  
**代码**: -  
**领域**: LLM安全  
**关键词**: catastrophic forgetting, pre-training, entity memory, memory replay, forgetting curve  

## 一句话总结

系统性地探索了 LLM 预训练阶段的灾难性遗忘问题，提出了基于实体记忆的新指标（M_ex、M_in）替代传统 PPL 来检测遗忘，并验证了周期性高强度 memory replay 策略在缓解预训练遗忘中的有效性。

## 研究背景与动机

灾难性遗忘（catastrophic forgetting）是构建全能模型的重大障碍。尽管在 LLM 微调阶段的遗忘问题已有大量研究，但**预训练阶段的遗忘**却鲜少被系统性探索。这一空白尤为关键，原因在于：

**预训练是知识获取的主要阶段**：模型在预训练中获取各种事实性知识，微调阶段主要增强任务能力。如果预训练中发生遗忘，模型会对用户的事实性查询给出不满意的回答

**传统指标的失效**：通用指标如 PPL 被证明对检测预训练遗忘不敏感（Gupta et al., 2023），这使得遗忘问题长期被遮蔽

**检测困难**：预训练数据极其多样，几乎不可能用单一任务指标来反映遗忘

本文提出三个核心研究问题：
- (1) 如何正确识别和量化预训练中的遗忘？
- (2) 简单轻量的 memory replay 方法能否缓解预训练遗忘？
- (3) 模型的遗忘曲线是否与人类学习规律相似？能否指导 replay 策略设计？

## 方法详解

### 1. 遗忘的存在性验证

**A+B 双数据集范式**：为放大遗忘效果，设计了先在数据集 A 上训练、再在数据集 B 上训练的设置。A 较小以避免过拟合，B 较大模拟主流预训练场景。

**PPL 的失败**：实验显示 PPL 在 A→B 过渡时不仅没有升高，反而下降。原因是 PPL 的概率平均特性被常见 token 的高预测准确性所主导，掩盖了低频信息的丢失。

**M(f) 指标的初步成功**：Tirumala et al. (2022) 的 memorization score 通过二值判断（model 的 argmax 预测是否正确）比 PPL 更敏感，在 A→B 过渡时捕获到了微小的遗忘信号。但它仍被不易遗忘的特征所主导，导致**低估遗忘程度**。

### 2. 新的实体相关指标

核心论点：预训练遗忘应聚焦于**实体信息**的遗忘。理由：
- 实体信息在数据中出现频率低，更容易被遗忘
- 用户对遗忘的感知主要通过实体信息（如"某人出生在哪"）
- 相比抽象能力的遗忘，实体遗忘更容易定义和测量

**M_in（内部召回指标）**：
- 输入包含实体的上下文（实体前 32 个 token），让模型贪婪解码 32 个 token
- 计算解码 token 与训练数据真实 token 的逐 token 匹配率
- 衡量模型在给定实体上下文时输出实体相关细节的能力

**M_ex（外部召回指标）**：
- 输入实体之前的 32 个 token（不含实体本身），让模型解码 32 个 token
- 检查生成文本中是否包含目标实体的子串
- 衡量模型从暗示性上下文中回忆实体的能力

另外还采用 **PPL_ent** 和 **M(f)_ent**：在实体相关样本上计算 PPL 和 M(f) 的变体。

### 3. Memory Replay 策略

设计了多种 replay 策略并比较效果：

**关键设计维度**：
- **Replay 频率**：每 100 步 replay 一次，仅 1% 的额外开销
- **存储策略**：全部样本 / 含实体样本 / 高 loss 样本
- **检索策略**：随机采样 vs BM25 相似度检索
- **退出机制**：同一样本最多 replay 5 次，避免过度集中

**核心策略**：

| 策略 | 描述 |
|------|------|
| Vanilla | 标准预训练 |
| Upper Bound | 在测试集上直接训练后立即评估 |
| BM25 | 用 BM25 检索相似已见样本进行 replay |
| BM25 + 仅实体样本 | 仅存储含实体的样本 |
| Focused Stochasticity | 随机采样 + 退出机制 |
| **Intensive Focused Stochasticity** | 每个 replay batch 训练 5 个 epoch |

### 4. 遗忘曲线分析

受人类遗忘曲线（Loftus, 1985）启发，研究两个因素：
- **学习强度的影响**：初始高强度学习是否带来更持久的记忆？
- **周期性复习**：类似人类的定期复习是否能改善遗忘曲线？

## 实验

### 实验设置

- **模型**：GPT-2（受算力限制，预计 1.5B 模型需 ~30,000 GPU 小时）
- **数据集 A**：OpenWebText (~8B tokens) 或 Pile (~13B tokens)
- **数据集 B**：SlimPajama 子集 (~49B tokens)
- **混合预训练**：将 A 和 B 混合打乱为一个完整集合从头训练

### 传统指标 vs 新指标

在 A (Pile) → B (SlimPajama) 的设置下：
- **PPL 和 M(f)**：在 A→B 转换后反而持续改善，显示虚假的"无遗忘"信号
- **PPL_ent 和 M(f)_ent**：在实体数据上显示部分恢复，但仍被不易遗忘元素主导
- **M_ex 和 M_in**：在 A→B 转换时显示**显著的性能下降**，且恢复非常缓慢，更准确地反映了遗忘现象

### Memory Replay 结果

| 方法 | PPL_ent | M(f)_ent | M_ex (×10⁻³) | M_in (×10⁻²) |
|------|---------|----------|-------------|-------------|
| Vanilla 预训练 | 26.03 | 0.4093 | 5.273 | 3.988 |
| Upper Bound | 23.74 | 0.4182 | 14.46 | 4.162 |
| BM25 | 27.95 | 0.4015 | 4.586 | 3.895 |
| BM25 + 仅实体 | 28.09 | 0.4013 | 4.575 | 3.941 |
| Focused Stochasticity | 25.79 | 0.4101 | 5.496 | 3.980 |
| **Intensive Focused** | **25.40** | **0.4121** | **5.450** | **4.003** |

关键发现：
1. **BM25 相似度检索反而不如 baseline**：可能因为检索集中在少数样本上导致不均匀
2. **简单随机 replay 有效**：Focused Stochasticity 优于 baseline
3. **高强度 replay 最优**：Intensive Focused Stochasticity 在所有指标上最好，且仅增加 5% 计算量

### 下游任务验证

| 方法 | HellaSwag | MMLU | Winograd | 平均 |
|------|-----------|------|----------|------|
| Vanilla | 27.46 | 23.20 | 53.47 | 34.71 |
| Intensive Focused | 27.75 | 23.00 | **55.68** | **35.48** |

减少样本级遗忘也改善了通用下游任务性能。

### 遗忘曲线发现

1. **即使相同分布也会遗忘**：后续训练数据与初始数据分布相同时，仍然观察到显著的指标下降
2. **高学习强度 → 更慢遗忘**：与人类学习规律一致，初始高强度学习带来更好的指标，但低强度实验最终会"追上来"
3. **困难数据需要更多训练**：难以记忆的数据从高强度学习中获益更大，维持更持久的差距
4. **周期性高强度 replay 有效**：每 1000 步进行一次 5 epoch 的高强度 replay，不仅提升了上界和下界，且比直接用 100 epoch 训练更节约计算

## 亮点与洞察

1. **揭示了 PPL 作为遗忘指标的严重缺陷**：PPL 被常见 token 的准确预测所主导，无法反映知识丰富但低频的实体信息遗忘。这对整个社区使用 PPL 进行评估提出了重要警示
2. **实体视角的新颖切入**：将预训练遗忘聚焦于实体记忆，既有理论合理性（实体是用户最直接感知的知识），又有实操可行性
3. **人类学习规律的映射**：发现 LLM 的遗忘曲线与 Loftus (1985) 的人类遗忘曲线惊人相似——高强度学习减缓遗忘、周期性复习改善长期记忆
4. **极低的额外开销**：Intensive Focused Stochasticity 仅增加 5% 计算量（T_replay = 1.05·T_0），却带来全面改善

## 局限性

1. **模型规模小**：受算力限制仅在 GPT-2 上实验，虽然 scaling law 暗示结论可推广到大模型，但缺乏直接验证
2. **replay 策略探索有限**：仅测试了简单的 replay 方法，更复杂的策略（如自适应频率、重要性加权）留待未来
3. **集中学习的副作用**：高强度 replay 可能影响模型的泛化性，特定数据子集的强化可能削弱其他任务的能力
4. **与微调遗忘的关系**：预训练遗忘和微调遗忘各有不同指标和缓解方法，二者的连接未被探索

## 相关工作

- **灾难性遗忘**：McCloskey & Cohen (1989), Ratcliff (1990) 的经典工作
- **持续学习方法**：episodic memory replay (de Masson D'Autume et al., 2019), meta-lifelong framework (Wang et al., 2020)
- **样本级遗忘**：Toneva et al. (2018) 定义的 example forgetting
- **预训练遗忘初探**：Tirumala et al. (2022) 的 memorization dynamics, Biderman et al. (2023) 的 emergent memorization
- **continual pre-training**：Gupta et al. (2023) 的 warm-up 策略研究

## 评分

⭐⭐⭐⭐ (4/5)

选题重要且新颖——预训练遗忘长期被忽视但影响深远。新指标设计合理、实验逻辑清晰。但受限于计算资源仅在小模型验证，且 replay 策略较为简单，距离实际大规模预训练的应用还有距离。与人类遗忘曲线的类比为预训练策略设计提供了启发性的新方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Unveiling and Addressing Pseudo Forgetting in Large Language Models](unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)
- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](../../ACL2026/llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)
- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[ACL 2025\] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](../../NeurIPS2025/llm_safety/demystifying_language_model_forgetting_with_low-rank_example_associations.md)

</div>

<!-- RELATED:END -->
