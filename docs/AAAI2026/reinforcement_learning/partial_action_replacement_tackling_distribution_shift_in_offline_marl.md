---
title: >-
  [论文解读] Partial Action Replacement: Tackling Distribution Shift in Offline MARL
description: >-
  [AAAI 2026][强化学习][离线多智能体强化学习] 提出部分动作替换（PAR）原理，从理论上证明在分解行为策略下分布偏移随偏离智能体数量线性增长（而非联合动作空间的指数增长），并基于此开发 SPaCQL 算法，通过 Q 函数集成的不确定性动态加权不同 PAR 策略，在 Random 和 Medium-Replay 数据集上显著超越所有基线。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "离线多智能体强化学习"
  - "分布偏移"
  - "部分动作替换"
  - "保守Q学习"
  - "不确定性估计"
---

# Partial Action Replacement: Tackling Distribution Shift in Offline MARL

**会议**: AAAI 2026  
**arXiv**: [2511.07629](https://arxiv.org/abs/2511.07629)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 离线多智能体强化学习, 分布偏移, 部分动作替换, 保守Q学习, 不确定性估计

## 一句话总结

提出部分动作替换（PAR）原理，从理论上证明在分解行为策略下分布偏移随偏离智能体数量线性增长（而非联合动作空间的指数增长），并基于此开发 SPaCQL 算法，通过 Q 函数集成的不确定性动态加权不同 PAR 策略，在 Random 和 Medium-Replay 数据集上显著超越所有基线。

## 研究背景与动机

离线多智能体强化学习（Offline MARL）面临的核心挑战源于多智能体系统的组合爆炸特性：

**联合动作空间的维度灾难**：任何有限数据集对所有可能的动作组合只能提供稀疏覆盖。这迫使学习算法必须对无数分布外（OOD）联合动作进行价值推理。标准 Q-learning 在这种情况下极易失败——函数逼近器（如神经网络）会对未见过的 OOD 动作产生任意高的 Q 值，引导智能体走向发散策略。

**几何直觉**：离线数据集是高维联合动作空间中的一组稀疏已知点。传统 Q-learning 更新需要评估所有智能体都采用新策略的联合动作，这等于在远离任何已知数据点的位置查询 Q 值，迫使 Q 函数进行大幅度且不可靠的外推。

**部分动作替换的核心洞见**：如果只改变一个或少数智能体的动作，而保持其他智能体的动作来自数据集，则查询的联合动作与已知数据只差异一个坐标——仅需小范围的局部外推，而非跳入未知的大范围外推。关键前提是数据由独立行动的智能体收集（分解行为策略），这在实践中很常见（独立人类演示、独立训练的智能体、去中心化系统等）。

**稳定性-协调性权衡**：仅替换单个智能体的动作（ICQL-QS）稳定但可能错过有价值的多智能体协调行为；完全替换所有动作则有效但风险高。需要一种自适应方法来平衡这种权衡。

## 方法详解

### 整体框架

基于部分动作替换原理，本文提出两个算法：
1. **ICQL-QS**（Individual CQL with Q-sharing）：保守基线，每次仅替换一个智能体的动作
2. **SPaCQL**（Soft-Partial Conservative Q-Learning）：主要贡献，自适应地混合不同替换数量的 Bellman 算子

### 关键设计

1. **ICQL-QS：稳定但保守的基线**：为每个智能体 $i$ 定义个体 Bellman 算子 $\mathcal{T}_i^{\text{ind}}$——智能体 $i$ 的下一步动作 $a_i'$ 从其学习策略 $\pi_i$ 采样，其他所有智能体的动作 $a_{-i}'$ 从数据集 $\mathcal{D}$ 采样：

   $\mathcal{T}_i^{\text{ind}} Q(s, \boldsymbol{a}) := \mathbb{E}_{(s', \boldsymbol{a}'_{-i}) \sim \mathcal{D}, a'_i \sim \pi_i(\cdot|s')}[r + \gamma Q(s', a'_i, \boldsymbol{a}'_{-i})]$

   损失函数包含 TD 误差项和 CQL 保守正则化项。虽然看似"短视"（每次只考虑单个智能体偏离），但共享的 Q 函数提供隐式耦合：每次更新都为所有可能的联合动作提供学习信号。**Proposition 1 证明** ICQL-QS 的梯度等价于在集中式平均 Bellman 算子 $\mathcal{T}^{ai} = \frac{1}{n}\sum_{i=1}^{n}\mathcal{T}_i^{\text{ind}}$ 上的随机梯度下降。

2. **SPaCQL：自适应混合部分备份**：核心观察——没有单一固定的备份策略在所有数据质量下都最优。Random 数据集偏好保守的单智能体更新；Expert 数据集可能需要协调的多智能体偏离。

   定义 $n$ 个基 Bellman 算子 $\{\mathcal{T}^{(k)}\}_{k=1}^{n}$，其中 $\mathcal{T}^{(k)}$ 恰好替换 $k$ 个智能体的动作（随机均匀选择）：
   
   $\mathcal{T}^{(k)} Q(s, \boldsymbol{a}) := \mathbb{E}_{s' \sim \mathcal{D}, \boldsymbol{a}'^{(k)}}[r + \gamma Q(s', \boldsymbol{a}'^{(k)})]$

   SPaCQL 算子是这些基算子的凸组合：$\mathcal{T}^{SP} Q = \sum_{k=1}^{n} w_k \mathcal{T}^{(k)} Q$，作为 $\gamma$-收缩的凸组合，$\mathcal{T}^{SP}$ 自身也保证是 $\gamma$-收缩。

3. **基于不确定性的自适应权重**：高集成分歧意味着数据覆盖不足，因此应降权高风险偏离。通过 Q 函数集成方差度量不确定性：

   $u_k = \sqrt{\text{Var}_j[Q_{\theta_j}(s', \boldsymbol{a}'^{(k)})]}$
   
   权重为不确定性的倒数归一化：$w_k = \frac{1/u_k}{\sum_k 1/u_k}$

   这样，不确定性高的（大 $k$，多智能体偏离）配置权重低，不确定性低的配置权重高。

   目标值构造：$Y_{SP} = r + \gamma \sum_{k=1}^{n} w_k \min_j Q_j^{tar}(s', \boldsymbol{a}'^{(k)})$

### 损失函数 / 训练策略

完整损失：$\mathcal{L}(\theta) = \mathbb{E}_\mathcal{D}[(Q_\theta(s, \boldsymbol{a}) - Y_{SP})^2] + \xi_c$

其中保守正则项 $\xi_c = \alpha \sum_{i=1}^{n} \lambda_i (\mathbb{E}_{a_i \sim \pi_i}[Q_\theta(s, a_i, \boldsymbol{a}_{-i})] - \mathbb{E}_\mathcal{D}[Q_\theta(s, \boldsymbol{a})])$，与 CFCQL 相同。

实现：使用 10 个 Q 网络集成，5 个随机种子，所有超参数与 CFCQL 论文一致。

## 理论贡献

本文最大的理论贡献是严格证明了部分动作替换的优越性：

**Lemma 1（线性散度界）**：对任何子集 $S \subseteq \{1,...,n\}$，$W_1(d^{(S)}, d^{(\varnothing)}) \leq \frac{\gamma}{1-\gamma} \sum_{i \in S} \text{TV}(\pi_i, \mu_i)$  
→ 分布偏移随偏离智能体数线性增长，而非联合空间指数增长。

**Theorem 1（紧价值误差界）**：$|V^\pi - \hat{V}^\pi| \leq \varepsilon_{\text{Subopt}} + \varepsilon_{\text{FQI}} + \frac{4\gamma}{(1-\gamma)^2} \sum_{i=1}^{n} \text{TV}(\pi_i, \mu_i)$  
→ 单智能体偏离时第三项减少为 $\frac{4\gamma}{(1-\gamma)^2} \text{TV}(\pi_k, \mu_k)$，严格改进联合 TV 界。

**Theorem 2（推广到相关行为策略）**：引入最大过剩相关度 $\kappa$，误差界增加独立于 $n$ 的常数  $\kappa$，仍然保持线性缩放。

**Theorem 3（SPaCQL 误差界）**：误差随有效偏离数 $k_{eff} = \sum_k w_k \cdot k$ 缩放，算法自适应调节。

## 实验关键数据

### 主实验

在 MPE（Cooperative Navigation, Predator-Prey, World）和 MaMujoco（Half-Cheetah）上评估。

| 任务 | 数据集 | OMAR | MACQL | IQL | CFCQL | DoF | **SPaCQL** |
|------|--------|------|-------|-----|-------|-----|-----------|
| CN | Random | 34.4 | 45.6 | 5.5 | 62.2 | 35.9 | **78.2±14** |
| CN | Med-R | 37.9 | 25.5 | 10.8 | 52.2 | 57.4 | **71.9±13.2** |
| CN | Expert | 114.9 | 12.2 | 103.7 | 112 | **136.4** | 111.9 |
| PP | Random | 11.1 | 25.2 | 1.3 | 16.5 | - | **89.4±13.7** |
| PP | Med-R | 47.1 | 11.9 | 23.2 | 71.1 | 65.4 | **75.0±12.7** |
| World | Random | 5.9 | 11.7 | 2.9 | 68 | 13.1 | **94.3±7.4** |
| World | Med-R | 42.9 | 13.2 | 41.5 | 73.4 | 58.6 | **105.2±11.1** |
| Half-C | Random | 13.5 | 5.3 | 7.4 | 39.7 | - | **43.8±4.9** |
| Half-C | Med-R | 57.7 | 37.0 | 58.8 | 59.5 | - | **66.1±3.4** |

SPaCQL 在**全部 Random 和 Med-R 数据集**上均取得最佳，共赢得 16 个任务中的 10 个。

### 消融实验

| 配置 | CN-Rand | CN-Expert | World-Rand | World-Expert | 说明 |
|------|---------|-----------|------------|-------------|------|
| CFCQL | 62.2 | **112** | 68 | **119.7** | 全联合更新 |
| ICQL-QS | 77.7 | 97.2 | 89.9 | 106.5 | 仅单智能体替换 |
| SPaCQL | **78.2** | 111.9 | **94.3** | 112.3 | 自适应混合 |

ICQL-QS vs CFCQL 验证了核心权衡：
- Random 数据上 ICQL-QS 大幅领先（部分替换优势大）
- Expert 数据上 CFCQL 略优（需要联合更新捕获协调行为）
- SPaCQL 两端都接近最优

### 关键发现

1. **Random/Med-R 数据集上的压倒性优势**：SPaCQL 在 World-Random 上得分 94.3 vs CFCQL 的 68（+38.7%），PP-Random 上 89.4 vs 16.5（+441.8%）。当智能体行为独立/低协调时，部分动作替换的优势极其显著。

2. **Expert 数据集上的可比性能**：在高质量协调数据上，SPaCQL 与最佳方法可比，说明自适应权重能正确增大 $w_n$（更多联合偏离）。

3. **自适应权重的可视化验证**：Random 数据上 $w_1$（单智能体偏离）主导；Expert 数据上 $w_2, w_3$ 增大——完全符合理论预期。

4. **不确定性估计验证**：ICQL-QS 的 Q 值估计不确定性在 Random 数据上始终低于 CFCQL，Expert 数据上两者相似。

5. **线性缩放的理论验证**：部分替换不是权宜之计，而是有严格理论保证——分布偏移 + 价值误差均线性缩放。

## 亮点与洞察

- **理论贡献远超算法**：本文最大的价值在于 Lemma 1 和 Theorem 1-3，形式化证明了"维度灾难在离线 MARL 中可能被高估了"——对整个领域具有方法论意义
- **ICQL-QS 的隐式协调**（Proposition 1）：看似独立更新，实际等价于集中式目标的梯度下降——这个洞见打消了"部分替换无法协调"的直觉担忧
- **推广到相关行为策略**（Theorem 2）：实际数据很少完全独立，引入最大过剩相关度 $\kappa$ 作为可加惩罚项，理论仍成立
- **算法设计的简洁性**：SPaCQL 本质上就是"多个 Bellman 算子的不确定性加权凸组合"——概念清晰、实现简单、理论保证强

## 局限与展望

1. 理论分析假设有限状态-动作空间和 i.i.d. 转移，实际中使用神经网络和轨迹数据时可能不成立
2. Theorem 1 假设 Q 函数是 $2/(1-\gamma)$-Lipschitz 的，对神经网络需要谱归一化等额外技术保证
3. 不确定性估计仅使用 Q 集成方差，可能有更好的度量方式
4. 在所有 Expert 数据集上未能领先，说明完全联合更新在高质量协调数据上仍有优势
5. 仅在简单环境（MPE、MaMujoco）验证，与真实世界多智能体场景的差距较大

## 相关工作与启发

- **CFCQL**（Shao et al. 2023）：CQL 的多智能体扩展，与本文最直接相关——也使用部分替换但仅用于正则化而非目标值计算，且无自适应加权
- **CQL**（Kumar et al. 2020）：保守 Q-learning 的单智能体版本，本文的理论分析基础
- **IQL**（Kostrikov et al. 2022）：隐式 Q-learning，避免 OOD 动作查询
- **DoF**（Li et al. 2025）：基于扩散模型的离线 MARL，在 Expert 数据上强
- **SAC-N / EDAC**（An et al. 2021）：Q 集成方差用于不确定性估计——SPaCQL 的权重设计灵感来源

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （部分动作替换的形式化 + 线性缩放证明 + 自适应混合算子，理论贡献突出）
- 实验充分度: ⭐⭐⭐⭐ （4个任务×4种数据集全面覆盖，权重可视化直观，但环境复杂度有限）
- 写作质量: ⭐⭐⭐⭐⭐ （理论推导严谨清晰，动机阐述到位，Figure 1/2 直观解释几何直觉）
- 价值: ⭐⭐⭐⭐⭐ （为离线 MARL 提供了更乐观的理论视角，方法论层面的贡献）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](../../ICML2026/reinforcement_learning/safety_generalization_under_distribution_shift_in_safe_reinforcement_learning_a_.md)
- [\[NeurIPS 2025\] Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL](../../NeurIPS2025/reinforcement_learning/oryx_a_scalable_sequence_model_for_many-agent_coordination_in_offline_marl.md)
- [\[AAAI 2026\] Efficient Multiagent Planning via Shared Action Suggestions](efficient_multiagent_planning_via_shared_action_suggestions.md)
- [\[AAAI 2026\] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)
- [\[ICLR 2026\] Recurrent Action Transformer with Memory](../../ICLR2026/reinforcement_learning/recurrent_action_transformer_with_memory.md)

</div>

<!-- RELATED:END -->
