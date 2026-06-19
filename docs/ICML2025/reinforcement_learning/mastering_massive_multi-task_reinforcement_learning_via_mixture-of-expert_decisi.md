---
title: >-
  [论文解读] Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer
description: >-
  [ICML2025][强化学习][多任务强化学习] 提出 M3DT 框架，将 MoE 引入 Decision Transformer 实现参数分离——通过任务分组让每个专家只学习一个小任务子集的特定知识，配合三阶段训练机制（骨干→专家→路由器）避免梯度冲突，增加专家数既扩展参数又降低任务负载，成功将离线多任务 RL 扩展到 160 个仿真控制任务。
tags:
  - "ICML2025"
  - "强化学习"
  - "多任务强化学习"
  - "Transformer"
  - "Mixture-of-Experts"
  - "任务可扩展性"
  - "参数可扩展性"
---

# Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer

**会议**: ICML2025  
**arXiv**: [2505.24378](https://arxiv.org/abs/2505.24378)  
**代码**: 待确认  
**领域**: 强化学习  
**关键词**: 多任务强化学习, Decision Transformer, Mixture-of-Experts, 任务可扩展性, 参数可扩展性

## 一句话总结

提出 M3DT 框架，将 MoE 引入 Decision Transformer 实现参数分离——通过任务分组让每个专家只学习一个小任务子集的特定知识，配合三阶段训练机制（骨干→专家→路由器）避免梯度冲突，增加专家数既扩展参数又降低任务负载，成功将离线多任务 RL 扩展到 160 个仿真控制任务。

## 研究背景与动机

**领域现状**：Decision Transformer (DT) 将离线 RL 建模为序列预测问题，高容量 Transformer 架构也被用于多任务 RL (MTRL)。受 LLM 在海量任务上泛化能力的启发，研究者期望训练能掌握大量多样任务的 RL 智能体。

**现有痛点**：
- **任务可扩展性不足**：大多数工作仅处理几十个任务（Atari 或 Meta-World），扩展后性能显著退化。Gato 覆盖 600+ 任务但控制任务表现差。
- **参数扩展效率低**：简单增大 DT 模型尺寸会迅速触及性能天花板。在 20M 参数后继续增大几乎无改善——这与 NLP 中模型越大越好的 scaling law 截然不同。

**核心矛盾**：共享参数面临的梯度冲突随任务数增多而加剧，导致性能退化；而朴素增大共享参数无法缓解——因为根本问题不是参数容量不足，而是所有任务共享参数导致的优化冲突。

**本文目标**：如何在任务数量极大（160 个）时既保持每个任务的学习质量，又实现高效的参数扩展？

**切入角度**：通过系统的实证分析，作者发现两个关键洞察：
- 性能退化在任务数较少（<40）时最剧烈，任务数大后退化趋于平缓 → **反过来看，将每个参数子集需学习的任务数减少到足够小即可显著提升性能**
- 参数扩展在"同时减少任务负载"时效果最好 → **扩展参数的同时必须减少每个参数子集的任务数**

**核心 idea**：用 MoE 实现"参数分离 + 任务分组"，增加专家数量同时扩展参数和降低任务负载，再配合三阶段训练避免梯度干扰。

## 方法详解

### 整体框架

M3DT 的整体 pipeline 如下：

- **输入**：与 Prompt-DT 一致，轨迹序列 $(r^*_{1}, s^*_1, a^*_1, \ldots, \hat{r}_{t-K+1}, s_{t-K+1}, a_{t-K+1}, \ldots, \hat{r}_t, s_t, a_t)$，前缀为 trajectory prompt
- **骨干网络**：5.29M 参数的 Prompt-DT（6层、8头、256维），学习跨任务共享知识
- **MoE 层**：每个 Transformer block 中 FFN 旁并行添加专家 FFN 和路由器 MLP，前馈结果为 $f(x) = x + f_{\text{FFN}}(x) + f_{\text{MoE}}(x)$
- **路由器**：5层 MLP，对隐藏状态做 softmax 得到各专家权重
- **输出**：预测动作 $a_t$，损失为 MSE

### 关键设计

1. **MoE 增强的 DT 架构**

    - 功能：在 DT 每个 Transformer block 的 FFN 旁并行添加多个专家 FFN，保留原始 FFN 以维护骨干知识
    - 核心思路：MoE 输出为 $f_{\text{MoE}}(x) = \sum_{i=1}^N \text{Softmax}(f_{\theta_r}(x))_i \cdot f_{\theta_i}(x)$，各专家结构与原 FFN 相同。原始 FFN 保留不动，MoE 输出作为残差加入，这样骨干的共享知识不会被破坏
    - 设计动机：作者在附录 B.2 中发现，Transformer block 中 MLP 层的梯度冲突远比 Attention 层严重，且单纯扩大 MLP 效果有限。因此选择在 MLP/FFN 侧加入 MoE 进行参数分离，每个专家只处理少量任务，大幅缓解 FFN 内的梯度冲突

2. **任务分组策略**

    - 功能：将所有任务显式分为若干组，每组分配给一个专家独立训练
    - 核心思路：提供两种分组方式——（1）**随机分组**：将任务随机等分，简单有效；（2）**基于梯度的分组**：计算每个任务的 agreement vector $A(\mathcal{T}_i) = g_i \odot \frac{1}{N}\sum_{i=1}^N g_i$（任务梯度与平均梯度的逐元素乘积），反映梯度一致性，再用 K-means 聚类。实验中梯度分组比随机分组在 160 任务上高约 2% (77.89 vs 80.14)
    - 设计动机：标准 MoE 的 top-k 动态路由在 RL 场景下不稳定——实验显示 top-4 routing 无法随专家数扩展，性能随专家增加反而下降。显式分组避免了路由不稳定和负载不均衡问题，同时利用任务结构信息使组内梯度冲突最小化

3. **三阶段训练机制**

    - 功能：将训练分为三个阶段，依次独立优化骨干、专家、路由器
    - 核心思路：
        - **阶段 1 — 骨干训练（400k 步）**：在所有任务上训练 Prompt-DT 骨干，学习跨任务共享表征。关键：在梯度冲突达到峰值时提前停止（约 400k 步），避免参数过拟合到梯度主导的任务
        - **阶段 2 — 专家训练（200k 步）**：冻结骨干，每个专家独立用其任务子集训练。各专家可并行训练（~1.8h/专家）
        - **阶段 3 — 路由器训练（400k 步）**：冻结骨干和专家，在所有任务上训练路由器学习动态权重分配。总训练约 24.2h（RTX 4090）
    - 设计动机：端到端训练 MoE 在 RL 中效果很差——实验显示端到端训练的 MoE 模型梯度冲突甚至比纯 PromptDT 的 MLP 还严重。分阶段训练让每个模块在无干扰环境下学习，骨干的提前停止策略避免了梯度冲突峰值后的性能下降

### 损失函数 / 训练策略

- 动作预测 MSE 损失：$\mathcal{L}_{DT} = \mathbb{E}[\frac{1}{K}\sum_{m=t-K+1}^t (a_{i,m} - \pi(\tau_i^*, \tau_{i,m}))^2]$
- 三个阶段均使用 Adam 优化器，学习率 1e-4，batch size 16
- 不使用额外的 MoE 负载均衡损失（因为显式分组已保证均衡）

## 实验关键数据

### 主实验：不同任务规模下的性能对比

在 Meta-World (50 任务) + DMControl (30 任务) + Mujoco Locomotion (80 任务) 共 160 任务上，统一归一化评分。

| 任务规模 | 方法 | 归一化分数 | 参数量 |
|---------|------|-----------|--------|
| 10 tasks | PromptDT-Small | ~88% | 1.47M |
| 10 tasks | HarmoDT-Large | ~89% | 173.30M |
| 10 tasks | M3DT-Gradient | **89.23%** | 47.87M |
| 80 tasks | PromptDT-Large | ~73% | 173.30M |
| 80 tasks | M3DT-Gradient | **79.58%** (+6.6%) | 98.37M |
| 160 tasks | PromptDT-Large | ~68% | 173.30M |
| 160 tasks | HarmoDT-Large | ~72% | 173.30M |
| 160 tasks | M3DT-Gradient | **80.14%** (+7.5%) | 174.12M |

M3DT 在 160 任务上的分数甚至高于其他 baseline 在 80 任务上的分数。

### 消融实验

| 配置 | 归一化分数 (160 tasks) | 说明 |
|------|----------------------|------|
| M3DT-Random | 77.89 | 随机分组 |
| M3DT-Gradient | **80.14** | 梯度分组，最优 |
| M3DT w/o 3-stage training | ~68 | 端到端训练，退化到 PromptDT 水平 |
| M3DT w/o grouping | 67.34 | 所有专家在所有任务上联合训练，比 baseline 还差 |
| M3DT-R w/o expert freezing | 次优 | 第三阶段同时微调专家，知识被覆盖 |
| M3DT-G w/o expert freezing | 次优 | 同上 |

### 关键发现

- **MoE 结构本身不够**：单纯加 MoE 端到端训练（w/o 3-stage）效果与 PromptDT 相当，梯度冲突甚至更严重。**任务分组 + 三阶段训练才是核心**
- **专家数与性能正相关但有上限**：8→40 专家性能持续提升（160 任务上提升 11.7%），但 40 专家后收益递减。上限由三方面决定：(1) 骨干共享知识有限；(2) 任务子集足够小后再拆收益递减；(3) 路由器在专家数多时分配难度增大
- **骨干提前停止很关键**：最优停止点恰好在梯度冲突达到峰值时（400k 步）。训练太少（200k）骨干知识不充分；训练太多（>400k）参数过拟合到主导任务，后续专家学习也受影响
- **Top-K 路由在 RL 中失败**：Top-4 routing 无法随专家数扩展，验证了 RL 中 sparse gating 的不稳定性。M3DT 需要所有专家的加权组合
- **随机分组已经足够好**：M3DT-Random 已超越所有 baseline，说明"减少任务数"本身就是核心收益；梯度分组额外提升约 2%

## 亮点与洞察

- **"减少任务数"的逆向思维**：从"任务增加→性能退化曲线"的形状中发现退化主要集中在小任务数阶段，反推出核心策略不是让模型更大，而是让每个参数子集看到更少的任务。这个洞察精辟且与直觉相反。
- **MoE 在 RL 中的关键适配**：NLP 中 MoE 用动态 top-k 路由，但 RL 数据分布与 NLP 差异巨大，动态路由不稳定。M3DT 用显式任务分组 + 全专家加权替代 top-k sparse routing，是一个重要的领域适配范式。
- **梯度冲突峰值作为骨干训练的停止信号**：将梯度冲突动态（先升后平）与训练阶段对齐，在峰值处切换到分组训练，兼顾了共享知识学习和冲突规避。

## 局限与展望

- **仅限仿真环境**：所有实验在 Meta-World/DMC/Mujoco 仿真中进行，状态空间为低维连续向量（最大 39 维），未验证在高维视觉输入或真实机器人上的效果。
- **推理成本随专家数线性增长**：所有专家均参与前馈，不同于 NLP 中 sparse MoE 只激活部分专家。作者尝试了 top-k 但效果不好，设计适配三阶段训练的 sparse gating 是重要的工程方向。
- **三阶段训练的超参数**：各阶段训练轮次（400k/200k/400k）需要额外调优，且骨干停止点依赖对梯度冲突的监控。自适应阶段切换机制值得探索。
- **任务分组依赖先验**：梯度分组需要先训练骨干再计算 agreement vector，计算额外开销。对全新任务集合或持续到达的新任务，如何在线更新分组是开放问题。
- **未验证泛化和持续学习**：所有实验中训练和测试使用相同任务集。在 held-out 任务上的泛化能力、以及新任务到来时的持续学习能力未被探索。

## 相关工作与启发

- **vs Multi-Game DT (Lee et al., 2022)**：直接扩大模型参数处理多任务 Atari，作者证明这种朴素扩展在 20M 参数后即饱和。M3DT 通过 MoE 在等量参数下性能显著更优。
- **vs Gato (Reed et al., 2022)**：通用多模态智能体覆盖 600+ 任务但控制任务性能差。M3DT 聚焦控制任务场景，通过任务分组和专家专精化取得更好表现。两者思路互补——MoE 可以集成到 Gato 类框架中。
- **vs HarmoDT (Hu et al., 2024)**：HarmoDT 用 task-specific mask 屏蔽冲突参数来缓解梯度冲突，本质上是"共享参数 + 掩码"范式。M3DT 的"共享骨干 + 独立专家"是更彻底的参数分离方案，在 160 任务上优势明显 (+7.5%)。
- **vs Soft MoE in RL (Obando-Ceron et al., 2024)**：该工作发现 top-k MoE 在深度 RL 中难以扩展，与本文的 top-4 实验结论一致。M3DT 的三阶段训练 + 全专家加权是绕过这一障碍的有效方案。

## 评分

- 新颖性: ⭐⭐⭐⭐ MoE 在 RL 中的应用有创新，核心贡献在于RL场景下的关键适配（显式分组+三阶段训练）和实证洞察（任务数vs性能曲线）
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 10-160 任务规模的系统实验，消融分析极为详尽（分组策略/专家数/骨干停止点/路由器设计/专家结构），domain-wise 分析消除了偏差
- 写作质量: ⭐⭐⭐⭐ 从实证观察到方法设计的逻辑链清晰完整，Figure 2/3 对问题的可视化说服力强
- 价值: ⭐⭐⭐⭐ 对多任务 RL 的可扩展性研究有重要推动，"参数分离 + 任务分组 + 分阶段训练"范式可推广到其他多任务场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)
- [\[NeurIPS 2025\] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](../../NeurIPS2025/reinforcement_learning/modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)
- [\[ICML 2025\] Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](robust_noise_attenuation_via_adaptive_pooling_of_transformer_outputs.md)
- [\[CVPR 2026\] TaskForce: Cooperative Multi-agent Reinforcement Learning for Multi-task Optimization](../../CVPR2026/reinforcement_learning/taskforce_cooperative_multi-agent_reinforcement_learning_for_multi-task_optimiza.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)

</div>

<!-- RELATED:END -->
