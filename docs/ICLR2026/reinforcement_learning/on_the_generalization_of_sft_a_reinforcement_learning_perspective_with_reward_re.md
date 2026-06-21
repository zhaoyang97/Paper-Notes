---
title: >-
  [论文解读] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification
description: >-
  [ICLR 2026][强化学习][SFT泛化] 从RL策略梯度视角数学证明SFT梯度隐式编码了逆概率加权(1/π_θ)的病态奖励结构→低概率token梯度过大导致泛化受限，提出DFT(Dynamic Fine-Tuning)仅需一行代码修改(CE loss乘token概率：$-p\log p$)消除逆概率加权→在数学推理/代码生成/多模态任务上大幅超越SFT，离线RL设定下甚至超越GRPO/PPO。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "SFT泛化"
  - "策略梯度"
  - "逆概率加权"
  - "Dynamic Fine-Tuning"
  - "奖励矫正"
---

# On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

**会议**: ICLR 2026  
**arXiv**: [2508.05629](https://arxiv.org/abs/2508.05629)  
**代码**: [GitHub](https://github.com/yongliang-wu/DFT)  
**领域**: 强化学习  
**关键词**: SFT泛化, 策略梯度, 逆概率加权, Dynamic Fine-Tuning, 奖励矫正

## 一句话总结
从RL策略梯度视角数学证明SFT梯度隐式编码了逆概率加权(1/π_θ)的病态奖励结构→低概率token梯度过大导致泛化受限，提出DFT(Dynamic Fine-Tuning)仅需一行代码修改(CE loss乘token概率：$-p\log p$)消除逆概率加权→在数学推理/代码生成/多模态任务上大幅超越SFT，离线RL设定下甚至超越GRPO/PPO。

## 研究背景与动机

**领域现状**：SFT是LLM后训练的标准范式，简单高效地获取类专家行为；RL通过奖励信号探索多样策略从而泛化更好，但需要大量计算、精细超参调节和显式奖励函数。

**核心痛点**：
   - (1) SFT在困难基准(OlympiadBench/AIME24)上甚至出现性能退化：Qwen2.5-Math-1.5B在OlympiadBench上SFT从15.88降到12.63
   - (2) "SFT记忆，RL泛化"现象被多篇工作经验性观察到(Chu et al., 2024)，但缺乏数学层面的根因分析
   - (3) 混合SFT+RL方法(InstructGPT/DPO等)并未改善SFT本身→当只有正样本、无奖励/偏好数据时SFT是唯一选择

**根本问题**：为什么SFT在只有正样本时泛化差？能否从理论上揭示SFT与RL的差异根源？

**切入角度**：通过重要性采样将SFT梯度严格重写为策略梯度形式→发现SFT隐含的"奖励"是逆概率加权的稀疏指示函数→这就是泛化受限的数学根因。

**直觉类比**：Focal Loss $-(1-p)^\gamma\log p$ 强调难样本(欠拟合时代)→DFT $-p\log p$ 弱化难样本(LLM过拟合时代)→目标设计哲学的根本反转。

**实际需求**：在无奖励模型/偏好数据/在线采样条件下（只用正样本），能否让SFT接近RL效果？

## 方法详解

### 整体框架

本文把SFT放进RL的策略梯度框架里审视：通过重要性采样把标准SFT的梯度严格改写成on-policy策略梯度的样子，从而读出SFT在背后隐含的奖励结构究竟是什么。诊断结果是这个隐含奖励既稀疏又被逆概率$1/\pi_\theta$放大，正是泛化差的根因；对应的解法DFT（Dynamic Fine-Tuning）只需在交叉熵上乘一个token概率把逆概率项中和掉，是一行代码的改动。

### 关键设计

**1. SFT梯度的策略梯度改写：把"为什么泛化差"算出来**

以往"SFT记忆、RL泛化"只是经验观察，本文给出数学根因。对标准SFT梯度做重要性采样变换，可以严格写成on-policy策略梯度的形式 $\nabla\mathcal{L}_{SFT} = -\mathbb{E}_{y\sim\pi_\theta}\big[\tfrac{\mathbf{1}[y=y^*]}{\pi_\theta(y|x)} \nabla\log\pi_\theta(y|x)\big]$。把它和标准策略梯度 $\nabla J = \mathbb{E}[\nabla\log\pi_\theta \cdot r(x,y)]$ 对齐，就能读出SFT的隐式奖励由两部分构成：奖励函数 $r(x,y)=\mathbf{1}[y=y^*]$ 只在exact match时非零，因此极度稀疏；而重要性权重 $w=1/\pi_\theta(y|x)$ 在模型概率越低的token上越大，等于把梯度向低概率token疯狂放大。两者叠加，优化就被拽去死磕那些低概率的精确匹配token，过拟合训练集而非学到泛化能力——这也解释了为什么SFT会在OlympiadBench、AIME24这类难基准上不升反降。

**2. DFT矫正损失：用token概率中和逆概率权重**

既然病根是那个$1/\pi_\theta$，最直接的解法就是乘回一个$\pi_\theta$把它消掉。这样得到的token级DFT损失为 $\mathcal{L}_{DFT} = -\sum_{t=1}^{|y^*|} \text{sg}\big(\pi_\theta(y_t^*|y_{<t}^*,x)\big) \log\pi_\theta(y_t^*|y_{<t}^*,x)$，其中$\text{sg}(\cdot)$是stop-gradient算子，让梯度不流过权重项，使权重只起"缩放"作用而不引入额外的优化通路。形式上它就是把标准交叉熵$-\log p$换成$-p\log p$，DFT因此保持了SFT原本的实现样貌，不需要奖励模型、偏好数据、reference模型或在线采样，真正意义上的一行代码改动。

**3. Token级加权与均匀奖励：为什么必须落在token粒度**

加权粒度的选择直接决定方法是否可用。若改用句子级概率$\pi(y|x)=\prod_t\pi(y_t)$，长序列连乘会让权重小到数值下溢、loss几乎无信息，几何均值变体同样收效甚微——实验里句子级加权只有15.75，和不加权的15.92持平，而token级加权能把均值从15.92拉到31.58。从奖励视角看，中和之后每条expert轨迹拿到的实际是均匀奖励1，等价于RLVR给所有正确样本赋同一奖励的做法，避免了把更新过度集中在个别低概率token上，这正是DFT既稳又泛化的来源。

## 实验关键数据

### 表1：数学推理主实验 (Avg@16, NuminaMath-CoT 100K样本)

| 模型 | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Avg |
|------|---------|---------|---------------|--------|-------|-----|
| Qwen2.5-Math-1.5B (base) | 31.66 | 8.51 | 15.88 | 4.16 | 19.38 | 15.92 |
| + SFT | 43.76 | 13.04 | 12.63↓ | 1.87↓ | 18.75↓ | 18.01 |
| + **DFT** | **64.89** | **20.94** | **27.08** | **6.87** | **38.13** | **31.58** |
| Qwen2.5-Math-7B (base) | 40.12 | 14.39 | 17.12 | 6.68 | 27.96 | 21.25 |
| + SFT | 53.96 | 16.66 | 18.93 | 2.48↓ | 26.09↓ | 23.62 |
| + **DFT** | **68.20** | **30.16** | **33.83** | **8.56** | **45.00** | **37.15** |

DFT在Qwen2.5-Math-1.5B上均值提升+15.66，是SFT(+2.09)的**5.9倍**。SFT在AIME24/OlympiadBench上退化，DFT持续正向提升。

### 表2：离线RL对比 (Qwen2.5-Math-1.5B)

| 方法 | 设定 | Math500 | Minerva | OlympiadBench | AIME24 | AMC23 | Avg |
|------|------|---------|---------|---------------|--------|-------|-----|
| DPO | Offline | 46.89 | 11.53 | 22.86 | 4.58 | 30.16 | 23.20 |
| RFT | Offline | 48.23 | 14.19 | 22.29 | 4.37 | 30.78 | 23.97 |
| PPO | Online | 56.10 | 15.41 | 26.33 | 7.50 | 37.97 | 28.66 |
| GRPO | Online | 62.86 | 18.93 | 28.62 | 8.34 | 41.25 | 32.00 |
| **DFT** | **Offline** | **64.71** | **25.16** | **30.93** | **7.93** | **48.44** | **35.43** |

DFT作为离线SFT方法，超越所有离线RL基线(+11.46 vs RFT)，甚至超越在线GRPO(+3.43)。

### 表3：代码生成 (UltraFeedback 10K)

| 模型 | HumanEval | HE+ | MultiPL-E Avg |
|------|-----------|-----|---------------|
| Qwen2.5-Coder-7B base | 62.2 | 53.0 | 57.76 |
| + SFT | 54.9↓ | 48.8↓ | 57.62 |
| + **DFT** | **67.7** | **59.8** | **62.30** |

SFT在Qwen2.5-Coder-7B上全面退化，DFT仍持续提升。

## 关键发现

1. **SFT退化现象的数学解释**：SFT在难基准上退化是因为隐式的1/π_θ逆概率权重导致梯度偏向低概率token→过拟合训练集的exact match→牺牲泛化。

2. **Token概率分布极化效应**：DFT训练后token概率呈双峰分布——高概率的语义关键词概率进一步提升，低概率的连接词/标点('the','let',',','.')概率被压低→模型学会区分语义vs语法token。

3. **收敛效率显著提升**：DFT在前120步内达到峰值，前10-20步即超越SFT的最终精度→梯度更新更具信息量→避免SFT的优化停滞。

4. **跨任务跨模型一致性**：5个基座模型(LLaMA/DeepSeek/Qwen)×4类任务(数学/代码/多模态/RL)一致提升→方法具有强鲁棒性。

5. **事实知识任务的失效**：在Natural Questions数据集上DFT（30.14%）反而低于SFT（36.62%）→DFT强化模型已有信念→当模型缺乏领域知识时反而阻碍学习。

## 亮点与洞察
- **"SFT=带病态奖励的RL"**是深刻的理论洞察：不只是经验观察→数学精确推导→揭示了"SFT记忆 RL泛化"现象的根本原因。
- **一行代码的威力**：$-p\log p$ 替代 $-\log p$ → 极其简洁但效果显著→学术美感与实用价值兼具。
- **无需任何额外资源**：不需要奖励模型、偏好数据、reference模型、大batch或在线采样→是资源受限场景下的最优选。
- **Focal Loss反转折射时代变迁**：CV时代强调难样本(欠拟合)→LLM时代弱化难样本(过拟合)→目标函数设计哲学需要重新审视。

## 局限性
- **事实知识学习受限**：DFT依赖模型自身confidence加权→当模型缺乏先验知识时会强化错误信念（NQ上退化1%+）
- **对hard样本/低资源领域不利**：初始概率低的样本权重被DFT压低→可能丢失稀有但重要的训练信号
- **评估范围有限**：未在更大规模LLM(70B+)和更多任务类型(对话/摘要等)上验证
- **无质量感知**：对所有正样本赋均匀奖励1→未利用示例质量差异→未来可探索非均匀奖励分配

## 相关工作对比

| 对比方法 | 关键差异 |
|----------|----------|
| **DPO** (Rafailov 2023) | DPO需要偏好对(正/负样本)通过隐式奖励直接优化策略→DFT只需正样本一行代码改loss；DFT在离线RL设定下超越DPO +12.23 avg |
| **GRPO** (DeepSeek 2024) | GRPO是在线RL需要采样多个response+reward验证→DFT无需在线采样/reward model；类似规模下DFT离线超GRPO +3.43 avg |
| **iw-SFT** (Qin & Springenberg 2025) | iw-SFT引入基于数据生成策略的importance weighting→需估计未知π_b；DFT直接用当前模型概率作权重→更简洁且不依赖额外假设 |
| **Focal Loss** (Lin 2017) | Focal强调难样本$-(1-p)^\gamma\log p$→DFT反向弱化难样本$-p\log p$→在LLM过拟合场景下更合理 |

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SFT-RL统一理论+一行代码改动的深刻简洁性
- 实验充分度: ⭐⭐⭐⭐⭐ 5模型×5数学基准+代码+多模态+离线RL+消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导与实践完美结合，故事线清晰连贯
- 价值: ⭐⭐⭐⭐⭐ 对LLM SFT训练实践有直接重大影响，极低实施门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT](getting_your_llms_ready_for_reinforcement_learning_with_lightweight_sft.md)
- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[ICLR 2026\] Leveraging Explanation to Improve Generalization of Meta Reinforcement Learning](leveraging_explanation_to_improve_generalization_of_meta_reinforcement_learning.md)
- [\[ICLR 2026\] R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](r1-reward_training_multimodal_reward_model_through_stable_reinforcement_learning.md)
- [\[ICLR 2026\] RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs](rl_squeezes_sft_expands_a_comparative_study_of_reasoning_llms.md)

</div>

<!-- RELATED:END -->
