---
title: >-
  [论文解读] EvoLM: In Search of Lost Language Model Training Dynamics
description: >-
  [NeurIPS 2025 Oral][强化学习][training dynamics] 系统训练 100+ 个 1B/4B 参数的 LM（从零开始），透明地研究预训练→续训→SFT→RL 各阶段的训练动态，揭示过度训练的递减收益、灾难性遗忘的缓解策略、以及 SFT/RL 配置的复杂权衡。 现代语言模型训练被分为多个阶段（预…
tags:
  - "NeurIPS 2025 Oral"
  - "强化学习"
  - "training dynamics"
  - "scaling law"
  - "continued pre-training"
  - "reinforcement-learning"
  - "SFT"
---

# EvoLM: In Search of Lost Language Model Training Dynamics

**会议**: NeurIPS 2025 Oral  
**arXiv**: [2506.16029](https://arxiv.org/abs/2506.16029)  
**代码**: 有（模型、数据、训练和评估 pipeline 全部开源）  
**领域**: 强化学习  
**关键词**: training dynamics, scaling law, continued pre-training, reinforcement-learning, SFT

## 一句话总结

系统训练 100+ 个 1B/4B 参数的 LM（从零开始），透明地研究预训练→续训→SFT→RL 各阶段的训练动态，揭示过度训练的递减收益、灾难性遗忘的缓解策略、以及 SFT/RL 配置的复杂权衡。

## 研究背景与动机

现代语言模型训练被分为多个阶段（预训练、续训、SFT、RL），但下游开发者很难评估每个阶段设计选择的影响。现有研究存在几个关键缺陷：

**不透明的分析**：许多 post-training 研究使用现成的 base model，不严格控制模型大小、数据量等关键变量

**中间检查点的可靠性**：依赖中间 checkpoint 评估会因学习率未完整衰减而低估真实能力

**阶段间交互不清**：预训练量如何影响 RL 效果？SFT 和 RL 如何配置数据分配？这些问题缺乏系统研究

本文通过从零开始完整训练 100+ 个模型，每个都完成完整的学习率调度，消除了上述混杂因素。

## 方法详解

### 整体框架

四阶段流水线：
1. **预训练**：在 FineWeb-Edu 上训练，token 预算从 Chinchilla 最优（20x模型参数）到 320B tokens
2. **续训（CPT）**：在 FineMath 上续训 2B-42B tokens，配合数据回放策略减缓遗忘
3. **SFT**：在 GSM8K/MATH 增广数据集上微调，使用模型一致性过滤低质量样本
4. **RL**：使用 PPO + 二元可验证奖励，数据与 SFT 不重叠

模型用 LLaMA-2 架构初始化，1B 和 4B 参数两种规模。所有配置使用完整的学习率调度，只取最终 checkpoint。

### 关键设计

**评估协议**的双维度设计是本文亮点：
- **上游任务**（Upstream）：HellaSwag、Winogrande、PIQA 等 0-shot 准确率 → 衡量语言建模能力
- **下游任务**（Downstream）：GSM8K-Platinum、MATH（域内）+ CRUXEval、BGQA、TabMWP、StrategyQA（域外）→ 衡量推理能力
- **四种采样策略**：Pass@1（贪心）、Maj@16（多数投票）、RM@16（ORM 打分最佳）、Pass@16（任一正确）

**数据回放策略**：CPT 时混入少量预训练数据（FineWeb），最优比例约 5%（8B 回放 + 42B 领域数据）。

### 损失函数 / 训练策略

- 预训练和 CPT 使用标准的 next-token prediction loss
- SFT 使用标准的交叉熵损失
- RL 使用 PPO 算法 + 二元可验证奖励（答案正确为 1，错误为 0）
- 所有模型都完成完整学习率衰减

## 实验关键数据

### 主实验：预训练规模的影响

| 模型 | ID Maj@16 (SFT) | ID Maj@16 (SFT+RL) | OOD Maj@16 (SFT) | OOD Maj@16 (SFT+RL) |
|------|----------------|---------------------|-------------------|----------------------|
| 1B-20BT | ~8% | — | — | — |
| 1B-80BT | ~15% | 21.4% | 24.6% | 31.0% |
| 1B-160BT | 14.2% | 22.5% | 25.6% | 31.6% |
| 1B-320BT | 16.1% | 25.0% | 24.8% | 29.9% |
| 4B-160BT | 26.4% | 34.8% | 26.0% | 33.2% |

### 模型规模与预训练预算交互

| 对比 | ID Greedy (SFT/SFT+RL) | ID Pass@16 (SFT/SFT+RL) |
|------|------------------------|--------------------------|
| 1B-320BT（同 compute） | 14.1/20.1 | 36.0/49.0 |
| 4B-80BT（同 compute） | 11.3/15.7 | 34.2/43.0 |
| 1B-160BT（同 tokens） | 12.8/17.5 | 34.5/45.1 |
| 4B-160BT（同 tokens） | 22.0/27.8 | 47.6/58.4 |

### 消融实验

**CPT 数据回放比例**（1B-160BT 基础，总 50BT CPT）：

| 配置 | GSM8K-Platinum Pass@1 |
|------|-----------------------|
| 无 CPT | 6.04% |
| FineMath 50BT（无回放） | 19.27% |
| FineWeb 1.6BT + FineMath 48.4BT | 16.21% |
| **FineWeb 8BT + FineMath 42BT** | **21.01%** |
| FineWeb 16BT + FineMath 34BT | 15.22% |

**SFT/RL 数据分配**（固定 100K 样本，4 epochs）：
- 分配更多给 SFT → 最大化域内性能（ID Greedy 在 70K SFT 时饱和）
- 分配更多给 RL → 提升域外泛化（OOD 在 10K SFT / 90K RL 时最佳）

### 关键发现

**Takeaway 1**：过度预训练不总能改善下游性能，甚至可能导致退化（80-160x 模型参数时饱和）

**Takeaway 3**：CPT 导致灾难性遗忘，5% 的数据回放可有效缓解

**Takeaway 4-5**：充分的领域 CPT 是 post-training 成功的前提；无 CPT 时 RL 甚至可能降低性能

**Takeaway 7-8**：过度 SFT 提升域内但损害域外性能，且限制 RL 进一步提升

**Takeaway 10**：RL 主要提高已正确输出的采样概率，而非真正提升推理能力（Correct Ratio 上升但 Pass@16 下降）

**Takeaway 12**：ORM score 是可靠的无监督代理指标，8B 奖励模型的评分与 1B 模型的准确率有 0.62-0.84 的 Pearson 相关

## 亮点与洞察

- **端到端透明性**：100+ 模型全部从零训练且完整衰减，消除了中间 checkpoint 的混杂因素（Table 3 实证了中间 checkpoint 显著低估真实性能）
- **RL 的本质洞察**：RL 并非"教会模型新能力"，而是"放大已有正确行为的概率"——Pass@16 下降而 Correct Ratio 上升
- **模型规模的解锁条件**：小模型在预训练不足时反而优于大模型；只有预训练到饱和区域后，模型规模优势才显现
- **ORM score 作为代理指标**的实用价值，尤其在标注困难的任务中

## 局限与展望

- 模型规模仅到 4B，趋势是否推广到更大模型未验证
- 仅关注推理任务的 post-training，安全对齐、指令遵循、代码等目标未探索
- RL 仅使用 PPO + 可验证奖励，未探索 GRPO、DPO 等替代方法
- 数学领域的结论是否迁移到其他专业领域（如法律、医学）不确定

## 相关工作与启发

- **Chinchilla scaling law**：本文在其基础上研究"过度训练"区间的下游表现
- **Springer et al. (overtrained)**：发现过度预训练损害 SFT，本文扩展至 RL 并验证了生成式推理任务的结论
- **Yue et al.**：RL 主要提升置信度而非推理能力的并行发现，本文提供了 epochs 和数据量两个维度的精细 trade-off
- **Zhao et al. (Echo)**：RL 放大预训练模式而非创造新模式，本文从训练动态角度提供补充证据

## 评分

- **新颖性**: 7/10 — 系统性强但单项发现多为已知趋势的量化验证
- **实验充分度**: 10/10 — 100+ 模型的系统控制实验堪称标杆
- **实用性**: 9/10 — 12 条 Takeaway 对 LM 训练实践有直接指导价值
- **写作质量**: 9/10 — 结构清晰，图表丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[ICML 2026\] The Surprising Difficulty of Search in Model-Based Reinforcement Learning](../../ICML2026/reinforcement_learning/the_surprising_difficulty_of_search_in_model-based_reinforcement_learning.md)
- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)
- [\[ICLR 2026\] Masked Skill Token Training for Hierarchical Off-Dynamics Transfer](../../ICLR2026/reinforcement_learning/masked_skill_token_training_for_hierarchical_off-dynamics_transfer.md)
- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](../../ICCV2025/reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)

</div>

<!-- RELATED:END -->
