---
title: >-
  [论文解读] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks
description: >-
  [AAAI 2026][社会计算][强化学习] 提出 CAPO（Curriculum Advantage Policy Optimization），一种基于优势信号的自适应课程机制，通过先模仿（仅正向优势样本）再判别（引入负向信号）的两阶段策略，稳定且显著提升 LLM 在数学推理和多模态 GUI 推理任务上的表现。
tags:
  - "AAAI 2026"
  - "社会计算"
  - "强化学习"
  - "课程学习"
  - "优势函数"
  - "LLM推理"
  - "多模态推理"
---

# From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks

**会议**: AAAI 2026  
**arXiv**: [2512.02580](https://arxiv.org/abs/2512.02580)  
**代码**: 无  
**领域**: 社会计算  
**关键词**: 强化学习, 课程学习, 优势函数, LLM推理, 多模态推理

## 一句话总结

提出 CAPO（Curriculum Advantage Policy Optimization），一种基于优势信号的自适应课程机制，通过先模仿（仅正向优势样本）再判别（引入负向信号）的两阶段策略，稳定且显著提升 LLM 在数学推理和多模态 GUI 推理任务上的表现。

## 研究背景与动机

强化学习（RL）已成为 LLM 后训练的主流范式，显著提升了模型的推理能力。PPO、GRPO 等 RL 算法的核心是**优势函数（advantage）**，它量化一个轨迹是否优于或劣于期望基线，从而提供正向和负向反馈来引导策略更新。

然而，现有方法从训练一开始就**不加区分地混合正负优势信号**，这带来了根本性问题：

**早期混合导致模糊引导**：模型尚未建立稳定基础时就暴露于负向样本，导致梯度噪声大、学习不稳定

**限制了进一步提升空间**：过早的正负混合阻碍模型从简单到困难的渐进学习

这一现象类似于发展心理学中的观察：**儿童首先通过正向模仿学习基本行为，之后才通过纠正性反馈和惩罚推进泛化**。这种分阶段的学习进程自然地将优势信号定位为有效的课程信号。

核心研究问题：**优势信号本身能否作为课程引导指标，实现正负反馈的结构化整合？**

## 方法详解

### 整体框架

CAPO 是一种与优势函数兼容的通用训练机制，可无缝嵌入 GRPO、PPO、RLOO、Reinforce++ 等多种 RL 算法。其核心是将训练分为两个阶段：

- **Phase 1（模仿阶段）**：仅使用正向优势样本 ($A_\tau \geq 0$) 建立稳定基础
- **Phase 2（判别阶段）**：引入全谱优势信号（正+负），培养判别能力，提升泛化性

### 关键设计

#### 1. **Phase 1: 正向模仿阶段**

训练目标选择性地仅更新正向优势轨迹：

$$\mathcal{J}_{\text{phase-1}}(\theta) = \mathbb{E}_\tau\left[\mathbb{I}_{A(\tau)\geq 0}\left(\frac{1}{T}\sum_{t=1}^{T}\min(\rho_t A_t, \hat{\rho}_t A_t) - \beta\mathbb{D}_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right)\right]$$

其中指示函数 $\mathbb{I}_{A_\tau \geq 0}$ 过滤掉负向优势样本，$\beta$ 控制 KL 惩罚强度。这一阶段鼓励模型强化正确的推理行为，同时保持接近参考分布。

**设计动机**：排除负向异常值降低梯度方差 $\text{Var}(\hat{g})$，即使引入偏差，整体 MSE 也会降低，确保稳定提升。

#### 2. **Phase 2: 判别阶段**

一旦稳定基础建立，CAPO 过渡到接纳全谱优势的判别阶段。通过引入负向优势样本，模型不仅学会强化优质推理轨迹，还学会抑制次优轨迹，从而增强泛化能力。

此时策略使用的方差已自然缩小，恢复无偏估计：$\mathbb{E}[\hat{g}_{\text{phase-2}}] = g$，使模型能够实现泛化。

#### 3. **课程调度策略**

采用**硬切换点（hard switch point）**进行阶段过渡，如在总训练步数的 10% 或 20% 处切换。实验发现，简单的硬切换比任何渐进引入方案都更有效且鲁棒，无需精细的超参数调节或任务特定的监控。

### 损失函数 / 训练策略

- Phase 1 使用 PPO/GRPO 等的标准裁剪目标 + KL 正则，但用指示函数过滤负向样本
- Phase 2 恢复标准 RL 目标，使用全部样本
- 理论保证：在 Robbins-Monro 条件下，CAPO 的方差-偏差权衡确保渐近收敛到局部最优

### 理论支撑

作者从方差-偏差权衡角度提供了理论论证：

- Phase 1 通过排除负向异常值降低梯度估计方差，即使有偏，总 MSE 更低
- Phase 2 随着策略改善，优势估计方差自然缩小，使用全部样本恢复无偏性
- 在合理步长条件下，两阶段的 MSE 渐近消失，参数收敛到稳态点

## 实验关键数据

### 主实验

在 Qwen2.5-Math-7B 和 Qwen2.5-Math-1.5B 上，跨四种 RL 算法和七个数学推理基准评估：

| 方法 | AIME24 | AMC | MATH500 | GSM8K | Minerva | Olympiad | Avg. |
|------|--------|-----|---------|-------|---------|----------|------|
| **7B - GRPO** | 16.7 | 52.5 | 75.2 | 86.5 | 29.4 | 36.9 | 48.9 |
| **7B - GRPO+CAPO** | **20.0** | **65.0** | **76.8** | **88.9** | **33.1** | **39.7** | **52.8↑3.9** |
| 7B - PPO | 26.7 | 52.5 | 71.0 | 80.9 | 34.2 | 34.1 | 48.6 |
| 7B - PPO+CAPO | **30.0** | **57.5** | **72.6** | **85.2** | **37.9** | **37.8** | **51.8↑3.2** |
| 7B - RLOO | 30.0 | 55.0 | 73.8 | 82.7 | 35.5 | 36.0 | 50.4 |
| 7B - RLOO+CAPO | **33.3** | **67.5** | **74.8** | **84.6** | **36.0** | **35.6** | **53.3↑2.9** |
| **1.5B - GRPO** | 13.3 | 52.5 | 71.2 | 83.2 | 26.8 | 30.1 | 45.6 |
| **1.5B - GRPO+CAPO** | **23.3** | **62.5** | **71.8** | **83.9** | **32.0** | **32.9** | **49.6↑4.0** |

CAPO 在所有算法和模型规模上均提供 **+1.7 到 +4.0** 的平均提升。

### 消融实验

| 配置 | AIME24 | AMC | MATH500 | GSM8K | Avg. | 说明 |
|------|--------|-----|---------|-------|------|------|
| GRPO（基线） | 16.7 | 52.5 | 75.2 | 86.5 | 49.5 | 标准 RL |
| GRPO+静态课程 | 16.7 | 65.0 | 75.0 | 86.3 | 51.8 | 按难度排序数据 |
| GRPO+ADARFT | 15.8 | 55.0 | 74.4 | 91.0 | 47.8 | 自适应微调 |
| **GRPO+CAPO** | **20.0** | **65.0** | **76.8** | **88.9** | **53.9** | 本文方法 |

**GUI 多模态推理（QwenVL2.5-3B）**：

| 方法 | GUI-Act-Web SR | OmniAct-Web SR | AndroidControl-Low SR | Overall |
|------|---------------|----------------|----------------------|---------|
| GRPO | 70.23 | 70.76 | 63.87 | 70.79 |
| **GRPO+CAPO** | **85.85** | **74.16** | 61.41 | **74.60↑3.81** |

### 关键发现

1. **广泛兼容性**：CAPO 作为即插即用增强，在 GRPO、PPO、RLOO、Reinforce++ 四种算法上均稳定提升
2. **跨模态泛化**：从数学推理成功迁移到多模态 GUI 推理场景，Overall +3.81
3. **最佳切换点**：Phase 1 到 Phase 2 的切换在 20%-30% 训练步数时效果最佳
4. **OOD 鲁棒性**：在 ARC-C 和 GPQA-Diamond 上（纯数学训练 → 通用推理），CAPO 比 GRPO 高 +3.8
5. **训练动态**：Phase 2 引入负向样本后，熵稳步上升表明探索更多样化，而奖励继续提升证实了泛化增强

## 亮点与洞察

- **将优势信号重新定义为课程信号**：这是一个简洁而深刻的洞察，改变了对 RL 中 advantage 的传统理解
- **发展心理学启发**：模仿→判别的两阶段与儿童认知发展过程的类比自然且有说服力
- **理论与实践统一**：从方差-偏差权衡角度的理论分析为两阶段设计提供了严谨支撑
- **极度简洁的实现**：核心修改仅需添加一个指示函数过滤负向样本，然后在某个步数硬切换，工程实现极为简单
- **广泛的即插即用兼容性**验证了方法的通用性

## 局限与展望

- 切换点目前需要预设（10%-30%），未来可探索基于训练动态的自适应切换策略
- 仅验证了数学推理和 GUI 推理两类任务，对代码生成、自然语言推理等其他推理类型的效果未知
- 理论分析假设优势估计噪声独立且有界，实际中可能不完全满足
- 未探索更细粒度的课程调度（如逐步增加负向样本比例），虽然实验表明硬切换最佳
- Phase 1 期间完全丢弃负向样本可能损失一些有价值的反馈信号

## 相关工作与启发

- 与 DeepSeek-R1 的关系：CAPO 可视为对 R1 类 RL 训练流程的一种增强，不改变算法本身而改变样本使用策略
- 与传统课程学习的区别：传统方法依赖外部静态难度指标（如 pass@k），CAPO 使用模型内在的 advantage 信号
- 对 RLHF 的启发：在偏好优化中也可以考虑先学习"好的"偏好，再学习区分"好与坏"
- 对多模态推理的启示：优势课程信号可以跨模态迁移，为统一推理训练提供了新思路

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 advantage 重定义为课程信号的洞察简洁深刻
- 实验充分度: ⭐⭐⭐⭐⭐ — 四种 RL 算法 × 两种模型规模 × 七个数学基准 + 多模态 GUI + OOD + 消融
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，理论推导严谨，实验组织有序
- 价值: ⭐⭐⭐⭐⭐ — 极高实用价值，即插即用，工程友好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)
- [\[ACL 2026\] Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms](../../ACL2026/social_computing/reheat_nachos_for_dinner_evaluating_ai_support_for_cross-cultural_communication_.md)
- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](../../ACL2025/social_computing/taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)

</div>

<!-- RELATED:END -->
