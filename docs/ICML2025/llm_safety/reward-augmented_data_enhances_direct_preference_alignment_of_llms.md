---
title: >-
  [论文解读] Reward-Augmented Data Enhances Direct Preference Alignment of LLMs
description: >-
  [ICML2025][LLM安全][DPO] 提出一种奖励增强的数据重标注方法：，通过将偏好对条件化于奖励分数构建扩增数据集，使DPO能感知回复质量全谱，缓解高质量rejected回复被遗忘和低质量chosen回复被盲目学习的问题，在多个基准上一致性大幅提升DPO性能。 直接偏好对齐的三大局限 现有直接对齐算法（如DPO）仅…
tags:
  - "ICML2025"
  - "LLM安全"
  - "DPO"
  - "偏好对齐"
  - "奖励条件化"
  - "数据增强"
  - "RLHF"
  - "LLM 对齐"
---

# Reward-Augmented Data Enhances Direct Preference Alignment of LLMs

**会议**: ICML2025  
**arXiv**: [2410.08067](https://arxiv.org/abs/2410.08067)  
**代码**: [shenao-zhang/reward-augmented-preference](https://github.com/shenao-zhang/reward-augmented-preference)  
**领域**: 信号通信  
**关键词**: DPO, 偏好对齐, 奖励条件化, 数据增强, RLHF, LLM 对齐

## 一句话总结

提出一种**奖励增强的数据重标注方法**，通过将偏好对条件化于奖励分数构建扩增数据集，使DPO能感知回复质量全谱，缓解高质量rejected回复被遗忘和低质量chosen回复被盲目学习的问题，在多个基准上一致性大幅提升DPO性能。

## 研究背景与动机

### 直接偏好对齐的三大局限

现有直接对齐算法（如DPO）仅关注**相对偏好**（chosen vs. rejected），忽略了回复的**绝对质量分数**，导致三大问题：

**高质量rejected被遗忘（Unlearning）**：当chosen与rejected质量差距极小时（如r=9 vs. r=8），DPO仍会最大化隐式奖励差距，导致高质量rejected的概率被不必要地压低

**低质量chosen被盲目学习**：DPO无差别地提升所有chosen回复的概率，即使某些chosen质量很低（如r=1），只要它胜过了更差的rejected（r=0）

**奖励稀疏性（Reward Sparsity）**：最优回复（$r = r_{\max}$）在数据中非常稀疏，DPO无法区分不同质量水平的chosen回复，难以泛化到最优回复

### 核心洞察

RLAIF流程中，judge模型（如GPT-4或奖励模型）已经给出了每个回复的**质量分数**，但DPO完全忽略了这一信息。如果让LLM条件化于目标奖励来生成回复，就能利用全部质量谱信息。

## 方法详解

### 1. 奖励条件化策略（Reward-Conditioned Policy）

定义目标条件化奖励函数：

$$R(x, y, g) = -(g - r(x, y))^2$$

其中 $g$ 是目标奖励，$r(x,y)$ 是judge模型给出的实际质量分数。策略 $\pi(y|x,g)$ 的优化目标为：

$$\min_{\pi} \mathbb{E}_{g, x \sim \mathcal{D}_N, y \sim \pi(\cdot|x,g)} \left[(g - r(x,y))^2\right]$$

### 2. 数据重标注构建奖励增强数据集

对原始偏好数据集 $\mathcal{D}_N = \{(x^i, y_w^i, y_l^i)\}$ 中的每一对，利用两个目标奖励值进行重标注：

- **目标 $g = r_w^i$（chosen的奖励）**：$R(x,y_w,g)=0 > R(x,y_l,g)=-(r_w-r_l)^2$，保持原始排序 $y_w \succ y_l$
- **目标 $g = r_l^i$（rejected的奖励）**：$R(x,y_l,g)=0 > R(x,y_w,g)=-(r_w-r_l)^2$，**反转排序** $y_l \succ y_w$

这样每对原始数据生成两对新数据，数据集大小从 $N$ 扩展到 $2N$。

### 3. 实现方式

- 通过**系统提示词**实现条件化，如 "generate responses of score $g$"
- 推理时设定 $g^* = r_{\max}$（最高奖励，如10），引导模型生成最优回复
- 可直接搭配任意直接对齐算法（DPO、IPO等），**无需修改算法本身**

### 4. 理论保证

论文提供了收敛性证明（Theorem 4.1）：奖励增强DPO的次优性以 $O(N^{-1/2})$ 速率衰减，保证全局收敛到最优策略，优于先前目标条件化RL仅能证明局部改进的结果。

## 实验关键数据

### 指令跟随基准（UltraFeedback + DPO）

| 模型 | AlpacaEval 2.0 LC WR | MT-Bench | Arena-Hard |
|------|----------------------|----------|------------|
| Qwen2-7B-Instruct | 20.93 | 7.90 | 24.3 |
| + DPO (UF) | 21.46 | 8.33 | 21.9 |
| + DPO (**RA**, 本文) | **31.17** | **8.47** | **30.1** |
| Gemma-2-9B-It | 49.20 | 8.54 | 42.8 |
| + DPO (UF) | 50.70 | 8.54 | 35.8 |
| + DPO (**RA**, 本文) | **59.27** | **8.59** | **43.9** |
| SPPO | 55.60 | 8.40 | 47.6 |
| + DPO (UF) | 52.75 | 8.41 | 40.4 |
| + DPO (**RA**, 本文) | **60.97** | **8.73** | **49.0** |

### 学术基准平均分

| 模型 | GSM8K | GPQA | TruthfulQA | 平均 |
|------|-------|------|------------|------|
| Llama-3.1-8B + DPO (UF) | 78.47 | 33.72 | 56.61 | 53.50 |
| Llama-3.1-8B + DPO (**RA**) | **78.77** | 32.55 | **63.32** | **54.37** |
| Gemma-2-9B + DPO (UF) | 83.32 | 34.14 | 65.12 | 59.22 |
| Gemma-2-9B + DPO (**RA**) | **83.62** | **35.74** | **65.27** | **59.75** |

### 关键消融实验

- **Half RA**（仅半数数据做增强，总量与原始相同）仍大幅超过原始DPO，证明提升不仅来自数据量翻倍
- **隐式奖励增强（IRA）**：用DPO模型的隐式奖励重标注，效果甚至超过用GPT-4打分的RA，说明DPO未充分利用数据
- **多属性奖励条件化**：用ArmoRM的5维属性奖励条件化，Llama-3-8B达到SOTA（LC WR 56.57），超过SimPO（53.70）

## 亮点与洞察

1. **极简方法，零算法修改**：只需对数据做重标注+加系统提示，不改DPO算法本身，即插即用
2. **SPPO复用数据仍有效**：已在UF上训练过的SPPO再用原始UF做DPO会退化，但用RA数据反而大幅提升，证明方法真正提取了更多信息
3. **缓解Unlearning**：实验验证高质量rejected（奖励≥5）的log概率下降幅度远小于标准DPO
4. **兼容无奖励分数数据**：可用DPO隐式奖励做重标注，适用于仅有二元偏好的数据集
5. **理论严谨**：提供了全局收敛到最优策略的理论证明，优于先前目标条件化RL的局部保证

## 局限与展望

1. **依赖奖励分数质量**：方法效果取决于judge模型打分的准确性，如果奖励分数噪声大，增强效果可能下降
2. **系统提示实现较粗糙**：用文本提示实现奖励条件化可能不如嵌入层条件化高效，存在优化空间
3. **仅验证了7B-9B模型**：缺少更大规模模型（70B+）的实验验证
4. **推理时需手动设定 $g^*$**：目标奖励需要人工选择最高值，未探索动态或自适应目标
5. **未与PPO/REINFORCE充分对比**：主要对比对象是直接对齐方法，与显式RL方法的对比有限

## 相关工作与启发

- **SteerLM / DPA**：条件化微调的先驱，但关注多属性用户定制，本文聚焦解决DPO本身的局限
- **RPO (Nemotron-4)**：同样关注unlearning问题但需改算法，本文仅需改数据
- **SimPO**：on-policy对齐强基线，本文在相同设置下超越
- **决策Transformer**：类似的条件化序列建模思想，但本文结合DPO而非SFT

## 评分
- 新颖性: ⭐⭐⭐⭐ — 奖励条件化+数据重标注的组合简洁优雅，洞察深刻
- 实验充分度: ⭐⭐⭐⭐⭐ — 5个基座模型×多个基准+丰富消融，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 动机阐述清晰，理论+实验结合好
- 价值: ⭐⭐⭐⭐ — 即插即用的实用方法，对偏好学习社区有直接启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] POPri: Private Federated Learning using Preference-Optimized Synthetic Data](popri_private_federated_learning_using_preference-optimized_synthetic_data.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](../../NeurIPS2025/llm_safety/virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[ICML 2025\] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)
- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)

</div>

<!-- RELATED:END -->
