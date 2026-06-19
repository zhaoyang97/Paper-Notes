---
title: >-
  [论文解读] ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India
description: >-
  [AAAI 2026][强化学习][法律AI] 本文首次将基于PPO的强化学习（RLAIF）应用于印度法律领域的判决预测与摘要生成任务，虽然性能未超越SFT和商业模型，但作为定位论文（position paper）揭示了RL在法律NLP中的关键挑战与未来方向。 问题背景 法律AI在印度司法系统中具有重要应用前景…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "法律AI"
  - "PPO"
  - "RLAIF"
  - "判决预测"
  - "法律文档摘要"
---

# ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India

**会议**: AAAI 2026  
**arXiv**: [2512.18014](https://arxiv.org/abs/2512.18014)  
**代码**: [github.com/ShubhamKumarNigam/ReGal](https://github.com/ShubhamKumarNigam/ReGal)  
**领域**: 强化学习  
**关键词**: 法律AI, PPO, RLAIF, 判决预测, 法律文档摘要

## 一句话总结

本文首次将基于PPO的强化学习（RLAIF）应用于印度法律领域的判决预测与摘要生成任务，虽然性能未超越SFT和商业模型，但作为定位论文（position paper）揭示了RL在法律NLP中的关键挑战与未来方向。

## 研究背景与动机

### 问题背景

法律AI在印度司法系统中具有重要应用前景，主要涉及两个关键任务：

**法庭判决预测与解释（CJPE）**：预测案件结果（接受/驳回）并生成支持性解释

**法律文档摘要**：从冗长的法庭判决书中生成简洁摘要

现有方法主要依赖监督微调（SFT），存在以下局限：
- 依赖大规模标注数据集
- 无法动态整合实时反馈来增强可解释性
- 模型输出与法律推理目标之间存在对齐差距

### 核心动机

RLHF/RLAIF在通用LLM对齐中已展现出强大能力（如ChatGPT），但在法律领域（尤其是印度法律场景）几乎未被探索。作者希望回答一个关键问题：**PPO-based RLAIF能否提升法律NLP任务的推理质量和可解释性？**

这是一项探索性研究，重点不在于刷SOTA，而在于揭示RL应用于法律文本时遇到的根本性挑战。

### 与现有工作的区别

- 此前法律判决预测（ILDC, PredEx, NyayaAnumana）均使用SFT
- 法律摘要领域的RL工作（如SAC-VAE、DQN）主要聚焦抽取式方法，且非针对印度法律
- **本文是首个将RLAIF/PPO同时应用于印度法律判决预测和抽象式摘要的工作**

## 方法详解

### 整体框架

ReGal框架采用两阶段训练流程：
1. **阶段一：监督微调（SFT）** → 使用标注数据微调Llama-2-7B，获得参考策略 $\pi^{SFT}$
2. **阶段二：PPO强化学习** → 使用AI生成的reward信号对SFT模型进行PPO优化

整体架构是任务无关的（task-agnostic），同一PPO训练流程可灵活应用于判决预测和摘要生成两个不同任务。

### 关键设计

#### 1. **基础模型与SFT训练**

选用Llama-2-7B作为基础模型，原因是与先前法律判决预测文献使用同一模型，便于公平对比。在两个任务上分别进行指令微调：
- CJPE任务：使用PredEx数据集训练预测+解释
- 摘要任务：使用IL-TUR数据集训练抽象式摘要生成

微调后的模型记作 $\pi^{SFT}$，作为PPO阶段的参考策略。

#### 2. **任务特定奖励模型（Reward Models）**

针对两个任务分别构建RM：
- **CJPE奖励模型**：微调InLegalBERT进行预测正确性分类。奖励为二值信号：预测正确→1，错误→0
- **摘要奖励模型**：基于n-gram重叠（ROUGE-style）和浅层语义相似度与金标摘要比对，输出标量奖励

这些RM模拟AI反馈（RLAIF），替代昂贵的人工标注。

#### 3. **PPO优化**

将语言模型视为可学习策略 $\pi_\theta$，PPO优化目标为：

$$\mathcal{L}_{PPO}(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_\theta(x)} \left[ \min \left\{ \frac{\pi_\theta(y|x)}{\pi^{SFT}(y|x)} \cdot r(y), \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi^{SFT}(y|x)}, 1-\epsilon, 1+\epsilon\right) \cdot r(y) \right\} \right]$$

关键参数解读：
- **概率比 $\frac{\pi_\theta(y|x)}{\pi^{SFT}(y|x)}$**：衡量当前策略相对于SFT基线的偏离程度
- **裁剪参数 $\epsilon=0.1$**：限制策略更新幅度，保持训练稳定
- **$r(y)$**：任务特定奖励，来自RM的反馈信号
- 取两个term的min确保保守更新，避免过大策略变化导致训练不稳定

### 损失函数 / 训练策略

- 学习率：1.41e-5
- PPO最大epoch数：1
- 训练batch size：4，mini-batch size：2
- 输出长度约束：100-500 tokens
- 使用混精度训练（GradScaler）优化GPU显存利用
- 训练设备：NVIDIA A100 80GB，总计约$100 GPU费用
- 推理时最大新token数限制为500

## 实验关键数据

### 数据集

| 数据集 | 规模 | 用途 | 平均文档长度 |
|--------|------|------|-------------|
| PredEx | 15,222篇 (12,178/3,044) | 判决预测+解释 | ~4,500 tokens |
| In-Abs | 7,130篇 (7,030/100) | 法律摘要 | ~4,377 words |

### 主实验

**判决预测与解释任务（PredEx数据集）：**

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | BERTScore | BLANC |
|------|---------|---------|---------|------|--------|-----------|-------|
| GPT-3.5 Turbo | - | - | - | - | - | - | - |
| LLaMA-2 SFT | **0.50** | **0.43** | **0.44** | **0.25** | **0.36** | **0.69** | **0.28** |
| ReGal (PPO) | 0.19 | 0.04 | 0.12 | 0.01 | 0.10 | 0.50 | 0.02 |

**ILDC Expert数据集：**

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR | BERTScore | BLANC |
|------|---------|---------|---------|------|--------|-----------|-------|
| GPT-3.5 Turbo | **0.54** | **0.43** | **0.45** | **0.28** | **0.47** | **0.73** | **0.34** |
| LLaMA-2 SFT | 0.49 | 0.38 | 0.40 | 0.29 | **0.51** | 0.69 | 0.36 |
| ReGal (PPO) | 0.25 | 0.05 | 0.16 | 0.01 | 0.16 | 0.50 | 0.03 |

### 消融实验

**不同推理策略对比（PredEx & In-Abs）：**

| 推理策略 | R1 (PredEx) | R1 (In-Abs) | BLEU (PredEx) | METEOR (In-Abs) |
|---------|------------|------------|--------------|----------------|
| Vanilla | 0.39 | **0.47** | 0.07 | **0.34** |
| SFT | **0.42** | 0.44 | **0.12** | 0.34 |
| DPO | 0.38 | 0.44 | 0.08 | 0.34 |
| PPO | 0.30 | 0.41 | 0.05 | 0.31 |

**基础模型变体消融：**
- Phi-3 Mini（更小模型）：性能大幅下降，无法处理长而复杂的法律文本
- LLaMA-2-7B无SFT：缺乏法律领域知识适配，质量显著降低

**奖励模型变体消融：**
- 使用InLegalBERT预训练版（未针对任务微调）替代微调RM：PPO性能进一步恶化，输出更加不连贯

### 关键发现

1. **PPO在法律NLP中全面落后于SFT和商业模型**：ReGal在所有指标上均显著低于LLaMA-2 SFT
2. **目标不匹配是核心瓶颈**：SFT策略未充分优化法律推理，导致PPO起点不佳
3. **奖励模型与法律文本对齐困难**：法律语言的细腻性和专业性超出RM的捕获能力
4. **严重的幻觉问题**：PPO模型会编造法律原则、虚构判例引用，尤其在输入不充分时
5. **RM精度直接决定PPO上限**：奖励模型的细粒度与法律推理的复杂度之间存在根本差距

## 亮点与洞察

1. **诚实的定位论文**：不追求SOTA而是坦诚揭示RL在法律AI中的失败原因，为后续研究提供宝贵经验
2. **跨任务验证**：在判决预测和摘要两个结构不同的任务上验证，增强结论可靠性
3. **系统性失败原因分析**：从目标不匹配、RM局限、法律复杂性、数据约束、超参选择等8个维度深入分析
4. **幻觉案例展示**：通过实例对比展示法律AI幻觉的危害性（如编造隐私权条款引用）

## 局限与展望

1. **RM需要更强的法律对齐**：需构建捕获法律推理细微差别的高质量奖励模型
2. **引入人工反馈（RLHF）**：RLAIF不足以指导法律文本生成，需法律专家参与
3. **法律领域预训练**：基础模型需要更深度的法律领域适配（domain-adaptive pretraining）
4. **抗幻觉机制**：需要事实性约束或幻觉感知奖励
5. **扩展至多司法管辖区数据**：单一印度最高法院数据覆盖有限

## 相关工作与启发

- **PredEx, ILDC数据集**：印度法律判决预测的基准数据集
- **InLegalBERT**：印度法律领域预训练语言模型
- **RLAIF (Lee et al., 2023)**：AI反馈替代人工反馈的成本效益方案
- 启发：**法律AI可能需要更精细的RLHF流程**，而非简单套用通用RLAIF框架；奖励信号的质量是RL方法在法律领域成败的决定性因素

## 评分

- 新颖性: ⭐⭐⭐（首次在印度法律场景应用PPO-RLAIF）
- 实验充分度: ⭐⭐⭐⭐（多数据集、消融、失败分析完善）
- 写作质量: ⭐⭐⭐⭐（position paper定位清晰，分析诚恳）
- 价值: ⭐⭐⭐（负面结果同样有价值，为后续研究指明方向）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[ICLR 2026\] When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift](../../ICLR2026/reinforcement_learning/when_sensors_fail_temporal_sequence_models_for_robust_ppo_under_sensor_drift.md)
- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)

</div>

<!-- RELATED:END -->
