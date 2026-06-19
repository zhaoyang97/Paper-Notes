---
title: >-
  [论文解读] Language Model Distillation: A Temporal Difference Imitation Learning Perspective
description: >-
  [AAAI 2026][强化学习][知识蒸馏] 从模仿学习/逆强化学习的视角重新审视语言模型蒸馏，提出利用教师模型输出分布的稀疏性（top-p token集中了96%以上概率质量），构建top-p MDP进行时序差分（TD）学习，证明了在缩减动作空间中的最优策略具有可界的次优性保证，并以IQL算法为基础实现的Bellman Distill方法在多个模型家族上超越了现有蒸馏方法。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "知识蒸馏"
  - "模仿学习"
  - "时序差分学习"
  - "top-p动作空间"
  - "逆强化学习"
---

# Language Model Distillation: A Temporal Difference Imitation Learning Perspective

**会议**: AAAI 2026  
**arXiv**: [2505.20335](https://arxiv.org/abs/2505.20335)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 知识蒸馏, 模仿学习, 时序差分学习, top-p动作空间, 逆强化学习

## 一句话总结

从模仿学习/逆强化学习的视角重新审视语言模型蒸馏，提出利用教师模型输出分布的稀疏性（top-p token集中了96%以上概率质量），构建top-p MDP进行时序差分（TD）学习，证明了在缩减动作空间中的最优策略具有可界的次优性保证，并以IQL算法为基础实现的Bellman Distill方法在多个模型家族上超越了现有蒸馏方法。

## 研究背景与动机

**LLM蒸馏的IL/IRL视角**：大模型蒸馏已成为标准做法，将大容量教师模型压缩为高效的学生模型。现有蒸馏方法大多可被视为行为克隆（Behavior Cloning, BC）——直接模仿教师在每个step的输出分布。

**BC的根本缺陷**：BC遭受**累积误差（compounding errors）**问题，也称为自回归模型中的**暴露偏差（exposure bias）**——训练时学生看到的是教师生成的上下文，而推理时看到的是自己生成的（可能有误差的）上下文，导致误差随序列长度指数级累积。

**TD学习的潜力**：模仿学习文献提供了丰富的工具来解决BC的不足，特别是时序差分（TD）学习方法。TD学习通过估计动作在给定状态下的长期影响来缓解累积误差。然而，**直接将TD学习应用于LLM面临巨大的动作空间挑战**——词表大小 $|\mathcal{V}|$ 通常为数万至十几万。

**关键观察**：LLM的输出分布高度稀疏——**top-50个token就占据了96%的概率质量，top-7个token贡献≥90%**。这启发了一个自然的想法：在TD学习中，只考虑少量高概率候选动作就够了。

**核心问题**：是否存在蒸馏场景特有的可利用结构？答案是教师模型的分布稀疏性——这是纯IL设定中不具备的（因为通常无法访问专家策略的完整分布）。

## 方法详解

### 整体框架

1. 定义语言模型生成的MDP：状态=prompt+已生成序列，动作=下一个token
2. 利用教师分布稀疏性定义top-p候选集
3. 构建top-p MDP并证明次优性有界
4. 在top-p MDP上实现IQL算法进行蒸馏

### 关键设计

#### 1. **Top-p候选集与Top-p MDP**

给定教师策略 $\pi^\star$ 和状态 $s$，top-p候选集定义为概率质量前 $p$ 的token子集：

$$\mathcal{A}_p^\star(s) = \{a_i : \sum_i \pi^\star(a_i|s) = p; \pi^\star(a_i|s) \geq \pi^\star(a_j|s), \forall i < j\} \subseteq \mathcal{A}$$

基于此定义top-p MDP $\mathcal{M}_p = (\mathcal{S}, \mathcal{A}_p^\star, \mathbb{P}, r, \gamma)$，即将原MDP的动作空间从完整词表 $\mathcal{V}$ 缩减到 $\mathcal{A}_p^\star$。

#### 2. **次优性理论保证**

本文的核心贡献是证明了在缩减动作空间中学到的最优策略与原始最优策略之间的差距可控。

**Top-p Bellman算子**的定义：

$$(B_p^\pi \bar{Q})(s,a) = r(s,a) + \gamma \mathbb{E}_{a' \sim \text{proj}_p(\pi)}[\bar{Q}(s',a') - \log \text{proj}_p(\pi)(a'|s')]$$

其中 $\text{proj}_p(\pi)$ 是策略 $\pi$ 到 $\mathcal{A}_p^\star$ 的投影（重新归一化）。

**Proposition 1**（收缩性）：$\mathcal{B}_p^\pi$ 在 supported $\infty$-范数下是收缩映射。

**Proposition 2**（次优性界）：$\|Q^\star - \bar{Q}^{\text{proj}_p \pi^\star}\|_{\infty, \mathcal{A}_p^\star} \leq \kappa(p)$，其中 $\kappa(p) = -\frac{\gamma}{1-\gamma}\log p$。

**Proposition 3**（夹逼条件）：$\bar{Q}^{\text{proj}_p \pi^\star}(s,a) \leq \bar{Q}_p^\star(s,a) \leq Q^\star(s,a)$

**Proposition 4**（有界次优性，核心定理）：

$$\|Q^\star - \bar{Q}_p^\star\|_{\infty, \mathcal{A}_p^\star} \leq \kappa(p) = -\frac{\gamma}{1-\gamma}\log p$$

当 $p=0.8$ 时，$\kappa(0.8) \approx 0.22 \cdot \gamma/(1-\gamma)$，次优性很小。这意味着可以放心地在大幅缩减的动作空间中运行IRL算法。

#### 3. **基于IQL的Bellman Distill实现**

选择IQL（Inverse soft Q-Learning）作为基础算法：

**Q函数masking**：通过top-p mask $\mathcal{F}_p^\star$ 只更新感兴趣的Q值：

$$(\mathcal{F}_p^\star Q)(s,a) = \begin{cases} Q(s,a) & \text{if } a \in \mathcal{A}_p^\star \\ -\infty & \text{otherwise} \end{cases}$$

**策略投影**：$(\text{proj}_p \pi_Q)(a|s) = \exp Q(s,a) / \sum_{\mathcal{A}_p^\star} \exp Q(s,a)$

最终的优化目标结合了Q masking和策略投影：

$$\max_Q \mathcal{J}^\star(Q) = \mathbb{E}_{\rho^\star}[\phi((\mathcal{F}_p^\star Q)(s,a) - \gamma V^{\text{proj}_p \pi_Q}(s'))] - \mathbb{E}_{\rho^\star}[V^{\text{proj}_p \pi_Q}(s) - \gamma V^{\text{proj}_p \pi_Q}(s')]$$

**关键实现细节**：
- 学生模型的logits直接作为Q值（经常数平移），一个模型同时充当Q函数和策略
- 使用 $\chi^2$ 正则化：$\phi(x) = x - x^2/4\alpha$，$\alpha=0.1$
- Q值裁剪：$Q_{\min} = -10$ 保证数值稳定性
- 离线训练：先从教师生成数据集，再用BD训练

### 损失函数 / 训练策略

- 同时最大化语言建模目标 $\mathcal{J}_{PT}$ 保持NLP基准性能
- 两阶段训练：Phase 1 SFT初始化 → Phase 2 BD蒸馏
- 批大小64，学习率5e-6，$p=0.8$（OPT 1.3B用$p=0.5$）
- 折扣因子 $\gamma=0.99$
- 教师为每个query生成8条response
- 4×A40 GPU训练

## 实验关键数据

### 主实验

**Rouge-L分数（5次随机种子平均）**：

| 方法 | GPT-2 (1.5B→120M) | | | OPT (6.7B→125M) | | | Qwen-2.5 (3B→0.5B) | | |
|------|-------|----------|--------|-------|----------|--------|-------|----------|--------|
| | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna |
| SFT | 23.2 | 9.9 | 14.3 | 23.2 | 9.9 | 14.3 | 24.5 | 16.5 | 17.6 |
| KD | 22.8 | 10.8 | 13.4 | 21.9 | 9.7 | 14.0 | 24.4 | 14.7 | 17.8 |
| SeqKD | 22.7 | 10.1 | 14.3 | 22.0 | 10.1 | 13.7 | 24.7 | 15.3 | 17.4 |
| MiniLLM | 24.6 | 13.2 | 16.9 | 23.8 | 10.2 | 15.3 | 26.7 | 19.1 | 20.5 |
| **BD (Ours)** | **24.7** | **13.3** | 16.2 | **25.1** | **11.4** | 14.8 | **27.8** | **19.5** | **20.7** |

### 消融实验

**Top-p值的影响**：

| p值 | GPT-2 340M | | | OPT 125M | | | Qwen-2.5 0.5B | | |
|-----|----------|----------|--------|----------|----------|--------|----------|----------|--------|
| | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna |
| 1.0 (无mask) | 24.9 | 15.2 | 15.7 | 23.6 | 10.5 | 14.2 | 26.6 | 18.1 | 18.6 |
| 0.8 | 25.8(+0.9) | 15.4(+0.2) | 16.5(+0.8) | 25.1(+1.5) | 11.4(+0.9) | 14.8(+0.6) | 27.8(+1.2) | 19.5(+1.4) | 20.7(+2.1) |
| 0.5 | 25.6(+0.7) | 15.5(+0.3) | 16.1(+0.4) | 24.4(+0.8) | 11.0(+0.5) | 14.7(+0.5) | 27.6(+1.0) | 19.2(+1.1) | 20.2(+1.6) |

**训练时间对比**：

| 模型 | MiniLLM | BD (Ours) | 加速比 |
|------|---------|-----------|--------|
| GPT-2 1.5B→340M | 11.2h | 1.3h | 8.6× |
| OPT 6.7B→350M | 12.8h | 3.5h | 3.7× |
| Qwen-2.5 3B→0.5B | 10.7h | 0.8h | 13.4× |

**$\chi^2$ 正则化消融**：

| 配置 | GPT-2 760M | | | Qwen-2.5 0.5B | | |
|------|----------|----------|--------|----------|----------|--------|
| | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna |
| 有 $\chi^2$ | 26.2 | 16.1 | 17.3 | 27.8 | 19.5 | 20.7 |
| 无 $\chi^2$ | 25.9 | 15.7 | 17.1 | 27.4 | 19.2 | 20.3 |

### 关键发现

1. **Top-p mask一致有效**：$p=0.8$时，在所有三个模型家族上相比$p=1.0$（无mask）都有显著提升，Qwen-2.5上最大提升达2.1个Rouge-L点。这直接验证了利用教师分布稀疏性的价值。

2. **离线训练的效率优势**：BD比在线方法MiniLLM快3.7-13.4倍达到最优验证性能，因为避免了昂贵的在线自回归生成。

3. **GPT-4o-mini评估的模型胜率**：BD在win rate评估中一致优于KD、SeqKD和MiniLLM基线，表明生成质量的提升不仅体现在Rouge-L上。

4. **规模一致性**：随着学生模型增大，BD的Rouge-L持续提升，展现了良好的可扩展性。

5. **$\chi^2$正则化有效**：引入$\chi^2$正则化在所有设置上都带来了0.2-0.5的稳定提升。

## 亮点与洞察

- **视角转换的价值**：将蒸馏问题重新cast为IL/IRL问题，自然引入了TD学习工具箱来解决BC的累积误差问题，是一种优雅的理论创新
- **利用LLM特有结构**：教师分布的稀疏性是LLM-specific的结构性质，将其转化为动作空间缩减的理论保证，连接了RL理论与LLM实践
- **次优性界的实用意义**：$\kappa(p) = -\gamma\log p/(1-\gamma)$ 给出了明确的trade-off——可以量化地决定缩减动作空间到什么程度
- **Q值≈logits的巧妙设计**：学生模型的logits直接作为Q值，无需额外的value网络，一个模型同时服务策略和value function

## 局限与展望

- **共享词表假设**：教师和学生必须共享词表才能进行分布匹配，限制了跨架构蒸馏
- **离线训练的性能上限**：虽然高效，但理论上在线训练能获得更好的最终性能（分布偏移问题未完全解决）
- **环境简单性**：仅在instruction-following任务上评估，未涉及代码生成、数学推理等更复杂的任务
- **MiniLLM在GPT-2上仍有竞争力**：BD对GPT-2家族的优势不如对OPT和Qwen家族明显
- **超参数敏感性**：$p$的选择对性能有影响（$p=0.5$ vs $p=0.8$），缺少自适应$p$的策略
- **理论与实践的gap**：理论分析假设tabular设定，实际使用参数化模型时收缩性等性质不一定严格成立

## 相关工作与启发

- 将SeqKD→KD→MiniLLM系列方法统一到IL/IRL框架下（offline off-policy BC → online mixed-policy BC → policy gradient），提供了清晰的分类学
- IQL的选择理由是它将saddle-point IRL重参数化为简单最大化，避免了复杂的min-max优化
- Top-p候选集与nucleus sampling中的top-p采样概念一致，但用于训练而非推理
- 理论框架具有通用性——任何soft IRL算法都可以通过top-p投影适配到缩减动作空间

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — IL视角+top-p MDP理论框架是全新的贡献
- **实验充分度**: ⭐⭐⭐⭐ — 三个模型家族+多消融，但任务类型单一
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导严谨，motivation清晰，IL/IRL背景介绍到位
- **价值**: ⭐⭐⭐⭐ — 为LLM蒸馏提供了新的理论框架和实践方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](../../ICLR2026/reinforcement_learning/model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)
- [\[ICLR 2026\] Contextual Similarity Distillation: Ensemble Uncertainties with a Single Model](../../ICLR2026/reinforcement_learning/contextual_similarity_distillation_ensemble_uncertainties_with_a_single_model.md)
- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](../../ICLR2026/reinforcement_learning/the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)

</div>

<!-- RELATED:END -->
