---
title: >-
  [论文解读] Bootstrap Off-policy with World Model (BOOM)
description: >-
  [NeurIPS 2025][强化学习][基于模型的强化学习] 提出 BOOM 框架，通过 bootstrap 循环将在线规划器（MPPI）与 off-policy 策略学习紧密结合：策略初始化规划器，规划器反过来通过无似然对齐损失（likelihood-free alignment）引导策略改进，配合 soft Q-weighted 机制优先学习高回报行为，在高维连续控制任务上取得 SOTA。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "基于模型的强化学习"
  - "online planning"
  - "离策略学习"
  - "世界模型"
  - "actor divergence"
  - "behavior alignment"
---

# Bootstrap Off-policy with World Model (BOOM)

**会议**: NeurIPS 2025  
**arXiv**: [2511.00423](https://arxiv.org/abs/2511.00423)  
**代码**: [molumitu/BOOM_MBRL](https://github.com/molumitu/BOOM_MBRL)  
**领域**: 强化学习  
**关键词**: 基于模型的强化学习, online planning, 离策略学习, 世界模型, actor divergence, behavior alignment

## 一句话总结

提出 BOOM 框架，通过 bootstrap 循环将在线规划器（MPPI）与 off-policy 策略学习紧密结合：策略初始化规划器，规划器反过来通过无似然对齐损失（likelihood-free alignment）引导策略改进，配合 soft Q-weighted 机制优先学习高回报行为，在高维连续控制任务上取得 SOTA。

## 研究背景与动机

**在线规划的优势**：Model-based RL 中在线规划（如 MPPI）通过前向模拟未来轨迹，能生成比纯策略网络更高质量的动作，显著提升样本效率和最终性能

**Actor Divergence 问题**：当使用规划器收集数据时，行为策略 β = π + MPPI 与策略网络 π 存在固有分布偏移——replay buffer 中的数据来自 β 而非 π，这打破了 off-policy 学习的分布一致性假设

**价值学习的分布偏移**：价值函数在 β 的状态-动作分布上训练，但在 π 访问但 β 很少覆盖的区域会产生过估计，导致价值估计不准确

**策略更新不可靠**：策略基于有偏的 Q 值更新，可能被误导到错误方向，尤其在高维复杂环境中问题更严重

**规划器分布不可参数化**：MPPI 等采样优化规划器的输出动作分布是非参数的（经过重加权和重采样），无法计算精确似然，传统 KL 散度等度量不可直接使用

**现有方法的不足**：TD-MPC2 将规划与 off-policy 学习结合但未处理 actor divergence；BMPC 仅用简单模仿学习对齐但缺乏价值引导；DreamerV3 在高维任务上样本效率不足

## 方法详解

### 整体框架

BOOM 由三个紧密耦合的组件组成：**策略网络 π**、**在线规划器 MPPI**、**世界模型**（编码器 h、动态模型 f、奖励预测器 R、价值函数 Q）。核心是一个 **bootstrap 循环**：策略为规划器提供初始化，规划器通过模型预测优化产生更高质量动作，再通过行为对齐反向引导策略改进。世界模型以 TD-MPC2 风格联合训练，既支持规划器模拟未来轨迹，又为策略提供价值估计。

### 关键设计 1：Likelihood-Free 对齐损失

- **功能**：让策略 π 对齐规划器 β 生成的动作，缩小两者的分布差距
- **核心思路**：采用 forward KL 散度 KL(β‖π) 而非 reverse KL。forward KL 展开后，与 β 相关的项是常数可丢弃，最终简化为 $\mathcal{L}_{\text{align}} = \mathbb{E}[-\log \pi(a|s)]$，只需计算 π 对规划器动作的对数似然，完全避免了需要 β(a|s) 的问题
- **设计动机**：MPPI 的输出分布是非参数的（经加权平均后不再是简单高斯），似然不可计算。reverse KL 需要 β 的似然，而 forward KL 只需从 β 采样（replay buffer 中已有）并评估 π 的似然，天然适配这种场景

### 关键设计 2：Soft Q-Weighted 机制

- **功能**：对 replay buffer 中的对齐样本按 Q 值加权，优先对齐高回报动作
- **核心思路**：定义 softmax 权重 $w_i = \exp(Q_i/\tau) / \sum_j \exp(Q_j/\tau)$，将对齐损失加权为 $\mathcal{L}_{\text{align}} = \sum_i w_i [-\log \pi(a_i|s_i)]$
- **设计动机**：replay buffer 中历史动作质量参差不齐（早期规划器性能差），直接均匀对齐会引入低质量行为。Q-weighted 机制借鉴了 MPPI 自身的价值引导选择思想，让策略专注于高价值经验，加速学习

### 关键设计 3：Bootstrap 策略目标函数

- **功能**：将对齐损失与标准 Q 值最大化策略目标结合
- **核心思路**：$\mathcal{L}_{\text{policy}} = -[Q(s, \pi(s)) + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{align}}]$，同时追求策略自身Q值最大化和与规划器的行为对齐
- **设计动机**：纯 Q 值最大化在 actor divergence 下不稳定，纯模仿规划器则缺乏策略自主改进能力。两者结合既保留 off-policy RL 的探索优势，又通过对齐约束防止策略偏离数据分布

### 关键设计 4：世界模型的间接改进

- **功能**：bootstrap 对齐通过改善策略-数据分布匹配，间接提升世界模型质量
- **核心思路**：价值学习更准确 → TD 损失梯度更 informative → 编码器、动态模型、奖励预测器一同受益（因为采用联合 TD-style 训练）
- **设计动机**：世界模型质量直接影响规划器性能，形成良性循环：更好的对齐 → 更准的价值 → 更好的世界模型 → 更好的规划

## 损失函数与训练策略

- **世界模型损失**：$\mathcal{L}_{\text{model}} = \sum_{t=0}^{H} \gamma^t [\|f(z_t,a_t) - \text{sg}(h(s'_t))\|^2 + \text{CE}(R_t, r_t) + \text{CE}(Q_t, q_t)]$，联合训练编码器、动态模型、奖励和价值
- **策略损失**：Q 值最大化 + λ_align × Q-weighted forward KL 对齐损失
- **训练流程**：warmup 阶段用随机动作收集数据预训练世界模型；主循环中每步用规划器收集数据 → 从 replay buffer 采样更新世界模型 → 更新策略
- **超参数**：对齐系数 λ_align = dim(A)/1000（DMC）或 dim(A)/50（Humanoid-Bench）；温度 τ = 1；对超参数不敏感（10× 范围内稳定）

## 实验关键数据

### 表 1：DMC Suite 高维运动任务 Total Average Return（3 seeds）

| 任务 | SAC | DreamerV3 (10M) | TD-MPC2 | BMPC | **BOOM** |
|------|-----|-----------------|---------|------|----------|
| Humanoid-stand | 9.0 | 717.0 | 913.3 | 947.9 | **962.1** |
| Humanoid-walk | 173.8 | 755.6 | 884.8 | 935.1 | **936.1** |
| Humanoid-run | 1.6 | 353.5 | 316.2 | 531.2 | **582.8** |
| Dog-stand | 197.6 | 35.4 | 936.4 | 971.3 | **986.8** |
| Dog-walk | 24.7 | 9.1 | 885.0 | 942.9 | **965.4** |
| Dog-trot | 67.1 | 8.4 | 884.4 | 911.3 | **947.9** |
| Dog-run | 16.5 | 4.3 | 427.0 | 673.7 | **820.7** |
| **DMC 平均** | 58.8 | 269.0 | 745.6 | 835.8 | **877.7 (+5.0%)** |

### 表 2：Humanoid-Bench 高维任务 Total Average Return（3 seeds）

| 任务 | SAC | DreamerV3 (10M) | TD-MPC2 | BMPC | **BOOM** |
|------|-----|-----------------|---------|------|----------|
| H1hand-stand | 74.1 | 845.4 | 728.7 | 780.0 | **926.1** |
| H1hand-walk | 27.0 | 744.0 | 644.2 | 672.6 | **935.4** |
| H1hand-run | 14.1 | 622.4 | 66.1 | 236.0 | **682.2** |
| H1hand-sit | 268.4 | 699.1 | 693.7 | 688.2 | **918.1** |
| H1hand-slide | 19.0 | 367.6 | 141.3 | 440.1 | **926.1 (+110.5%)** |
| H1hand-pole | 122.5 | 577.4 | 207.5 | 739.9 | **930.5** |
| H1hand-hurdle | 12.9 | 135.7 | 59.0 | 197.1 | **435.6 (+121.0%)** |
| **H-Bench 平均** | 68.5 | 555.6 | 338.8 | 511.7 | **820.6 (+47.7%)** |

### 消融实验关键结论（Dog-run，223/38 dims）

- Forward KL 显著优于 Reverse KL（后者需近似似然，不准确有害）
- Q-weighted 加速训练且提升最终性能（vs 均匀加权）
- 对齐系数 0.1×~10× 范围内性能稳定，鲁棒性强

## 亮点与洞察

1. **问题定义清晰**：精确刻画了"在线规划 + off-policy RL"范式中 actor divergence 的两个后果（价值偏移和策略更新不可靠），并给出理论分析
2. **Forward KL 的巧妙选择**：在规划器分布不可参数化的约束下，forward KL 是唯一自然的选择，简洁且无需额外近似
3. **理论保证完备**：Theorem 1 证明对齐控制回报差距上界，Theorem 2 证明对齐控制 Q 值偏差上界，理论与实验一致
4. **性能提升显著**：尤其在最难的 Dog-run (+21.8%) 和 H1hand-hurdle (+121.0%) 上优势巨大，说明方法在高维复杂任务中的优越性
5. **实现简单**：核心改动仅在策略损失中加一项 Q-weighted log-likelihood 项，几乎零额外计算开销

## 局限性

1. **仅验证连续控制**：所有实验限于 DMC 和 Humanoid-Bench 的运动控制任务，未在操作、导航等任务类型或离散空间上验证
2. **依赖世界模型质量**：BOOM 以 TD-MPC2 风格世界模型为基础，模型不准确时规划器和对齐的质量都会下降
3. **Forward KL 的 mode-covering 倾向**：forward KL 使策略覆盖规划器分布的所有模式，可能导致策略过于分散，在多模态规划器分布下不一定最优
4. **replay buffer 中动作质量退化**：虽有 Q-weighted 机制缓解，但随训练进行，早期低质量数据仍占据 buffer 空间
5. **缺乏真实机器人实验**：仅在仿真环境中验证，sim-to-real 差距未讨论

## 相关工作与启发

- **TD-MPC2**：BOOM 的直接基线，共享世界模型架构但未处理 actor divergence
- **BMPC**：同样尝试对齐策略与规划器，但使用简单模仿+动作重标注，缺乏价值引导
- **DreamerV3**：imagination-driven MBRL，不使用在线规划，在高维任务上效率受限
- **Offline RL**：CQL、IQL 等处理分布偏移的思路（保守估计、隐式策略）与本文的对齐思路有异曲同工之处
- **启发**：bootstrap 循环思想可推广到其他存在"强执行器+弱学习器"不匹配的场景，如搜索引导的 LLM 训练

## 评分

- **新颖性**: ⭐⭐⭐⭐ — actor divergence 问题虽已知，但 likelihood-free forward KL + Q-weighted 对齐的组合方案简洁有效
- **实验充分度**: ⭐⭐⭐⭐ — 14 个高维任务覆盖全面，消融完整，但缺少真实场景和更多任务类型
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题动机-方法-理论-实验逻辑链条清晰，公式推导严谨
- **价值**: ⭐⭐⭐⭐ — 对 planning + off-policy RL 范式的核心痛点给出实用方案，开源代码，具有较高实践价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[NeurIPS 2025\] A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning](a_unifying_view_of_linear_function_approximation_in_offpolic.md)
- [\[ICML 2025\] Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning](../../ICML2025/reinforcement_learning/log-sum-exponential_estimator_for_off-policy_evaluation_and_learning.md)
- [\[ICLR 2026\] From Observations to Events: Event-Aware World Model for Reinforcement Learning](../../ICLR2026/reinforcement_learning/from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
