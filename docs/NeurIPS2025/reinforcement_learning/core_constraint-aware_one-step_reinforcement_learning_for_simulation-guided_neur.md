---
title: >-
  [论文解读] CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design
description: >-
  [NeurIPS 2025][强化学习][one-step RL] 提出 CORE（Constraint-aware One-step REinforcement learning），一种无 critic 的单步 RL 框架，通过结构化分布采样、scaling-graph 解码器和约束感知的 reward shaping 来高效探索 DNN 加速器的硬件-映射联合设计空间，在 7 个 DNN 模型上取得至少 15× 的 latency 改善。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "one-step RL"
  - "design space exploration"
  - "DNN accelerator"
  - "constraint-aware"
  - "scaling graph"
---

# CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design

**会议**: NeurIPS 2025  
**arXiv**: [2506.03474](https://arxiv.org/abs/2506.03474)  
**代码**: 未开源  
**领域**: 强化学习  
**关键词**: one-step RL, design space exploration, DNN accelerator, constraint-aware, scaling graph  

## 一句话总结
提出 CORE（Constraint-aware One-step REinforcement learning），一种无 critic 的单步 RL 框架，通过结构化分布采样、scaling-graph 解码器和约束感知的 reward shaping 来高效探索 DNN 加速器的硬件-映射联合设计空间，在 7 个 DNN 模型上取得至少 15× 的 latency 改善。

## 研究背景与动机

**领域现状**：基于仿真的设计空间探索（DSE）在硬件-软件协同设计中至关重要。DNN 空间加速器的设计涉及 PE 数量、buffer 大小、映射策略（tiling、loop order、并行维度）等结构化参数，搜索空间巨大且包含复杂的参数依赖约束。例如 tile 大小必须满足跨内存层级的层次化约束 $D_i \leq D_{i+1}$，PE 数量限制并行度上界。

**现有方法的局限**：
   - 遗传算法（GA）和贝叶斯优化在高维/稀疏反馈场景下效率低，缺乏有效的约束编码机制
   - 多步 RL 方法（如 ArchGym）将 DSE 建模为 MDP，需要长 rollout 探索，面临 reward 稀疏/延迟、部分设计状态维护困难，以及启发式 masking 导致约束违反等问题
   - HASCO 等两步法先用贝叶斯优化搜索硬件再用启发/Q-learning 优化映射，忽略了硬件与映射的强耦合性

**核心矛盾**：联合优化硬件和映射是必要的（两者高度耦合），但联合空间太大、约束太复杂，传统方法要么拆开优化损失最优性，要么暴力搜索效率极低。

**核心洞察**：DSE 可以建模为单步 MDP——策略网络一次性生成完整候选配置，无需维护中间状态和 rollout，天然支持批量并行仿真。关键是如何在单步生成中编码参数间的依赖约束。

**核心 idea**：用 scaling graph 编码参数依赖来保证采样可行性，用批次内相对优势替代 critic 来实现高效的单步策略优化。

## 方法详解

### 整体框架

CORE 的 pipeline 分三步循环：(1) 策略网络输出结构化分布参数 → (2) scaling-graph 解码器将采样动作映射为可行的设计配置 → (3) 并行仿真器（MAESTRO）评估 E=32 个设计点，返回 latency/area/power 等指标 → 计算 reward 和优势更新策略。输入是固定上下文 $s_0$（编码 DNN 负载信息），输出是硬件参数+映射策略的完整配置。

### 关键设计

1. **单步 MDP 与条件分布采样**：

    - 功能：将设计探索建模为单步 MDP $\mathcal{M}=(s_0, \mathcal{A}, R)$，策略一次直出完整配置
    - 核心思路：策略网络 $\pi_\theta$ 输出所有设计参数的联合分布，通过条件分解 $\pi_\theta(a_1, \ldots, a_N; s_0) = \prod_{i=1}^{N} f_{i,\theta}(a_i \mid a_{i+1} \ldots a_N; s_0)$。离散参数（如并行维度，6 选 1）用 **Categorical 分布**，大范围离散参数（如 PE 数量，512 个选择）用 **Beta 分布** 再量化为离散值
    - 设计动机：消除多步 RL 中的 reward 传播延迟和中间状态维护问题，且 Beta 分布作为连续松弛减少了策略网络输出维度

2. **Scaling-Graph 解码器**：

    - 功能：通过有向图编码参数依赖关系，按拓扑序解码采样动作为可行配置
    - 核心思路：每个节点代表一个设计变量，有向边表示约束/缩放关系。源参数的解码值动态约束目标参数的范围。对于采样动作 $b \in [0,1]$ 和源参数值 $\{A_i\}$，解码为 $B = B_{low} + \lfloor(\frac{\min_i\{A_i\} - B_{low}}{B_s} + 1) b \rfloor B_s$，用 $\min_i\{A_i\}$ 替换固定上界
    - 具体例子：并行度 $P_1$ 的上界受 PE 数量和对应 tile 大小约束——$P_1 = P_{low} + \lfloor(\min\{N_{pe}, X_2\} - P_{low} + 1) p_1 \rfloor$
    - 设计动机：传统方法要么忽略依赖（大量无效设计），要么用启发式 masking（不可微、覆盖不全）。Scaling graph 在采样阶段就保证可行性，且完全可微，支持端到端训练
    - 与之前方法的区别：ArchGym 用启发式 masking 在多步 MDP 中约束动作空间，容易遗漏复杂依赖；而 scaling graph 是声明式的，拓扑排序自动处理所有传递依赖

3. **约束感知 Reward Shaping**：

    - 功能：三层奖励设计，区分正常设计、约束违反和异常设计
    - 核心思路：正常奖励 $R(\xi_k) = \mathbf{w}^\top U(\xi_k)$；约束违反时加惩罚 $R(\xi_k) = \mathbf{w}^\top U(\xi_k) - \alpha_c h(U(\xi_k))$，其中 $h$ 量化违反程度；无法仿真的异常设计给予低于批次平均的惩罚 $R_t(\xi') = \min(\mathbb{E}[R], \hat{R}_{t-1}) - \alpha_p \mathbb{E}[R]$
    - 设计动机：单纯给无效设计固定负奖励信息量不足，量化约束违反程度能让策略学到"违反了多少"而非仅仅"违反了没有"

### 策略优化

**代理优势函数**使用批次内相对奖励（类似 GRPO）：

$$A_t(\xi_k) = R(\xi_k) - \hat{R}_t, \quad \hat{R}_t = \alpha_r \cdot \frac{1}{E}\sum_{k=1}^E R(\xi_k) + (1-\alpha_r)\hat{R}_{t-1}$$

总目标 $L(\theta_t) = L_{up} + L_r + L_e$：
- **条件更新**：$L_{up} = \mathbb{E}[\frac{\pi_\theta}{\pi_{\theta_t}} A_t(\xi_k)]$（类似 PPO 的重要性采样）
- **KL 正则化**：$L_r = -\beta_r \sum_i D_{KL}(\pi_{i,\theta} \| \pi_{i,\theta_t})$（防止更新过大）
- **熵正则化**：$L_e = \beta_e \sum_i \mathbb{H}(\pi_{i,\theta})$（$\beta_e$ 从 1.0 线性衰减到 0.02，早期鼓励探索、后期聚焦利用）

关键特点：**无 Critic**——不学习价值函数，靠批次内相对奖励计算优势。策略网络为 4 层 MLP（512→4096→4096→4096→输出），Adam 优化器，学习率 $10^{-5}$，训练 2000 episodes。

## 实验关键数据

### 主实验（Latency，log10 cycles，越低越好）

| 模型 | GA | HASCO | CORE |
|---|---|---|---|
| ResNet-18 (Cloud) | 7.28 | 6.80 | **4.62** |
| ResNet-50 (Cloud) | 7.29 | 7.30 | **5.47** |
| MobileNetV2 (Cloud) | 7.01 | 6.79 | **4.33** |
| BERT (Cloud) | 7.31 | 6.85 | **5.60** |
| VGG-16 (Cloud) | 7.85 | 7.43 | **5.05** |

CORE 在所有 7 个 DNN 模型上均大幅领先，Cloud 平台 latency 平均降低 **15× 以上**（log scale 下差 1.5-2.5，即 30×-300×）。Edge 平台同样表现优异，如 MobileNetV2 从 6.74 降至 4.97。

### 消融实验

| 配置 | ResNet-18 Latency | ResNet-50 Latency | 说明 |
|---|---|---|---|
| w/o reward shaping | 6.68 | 7.31 | 约束违反率上升，Edge 平台多个模型找不到可行解（"-"） |
| w/o scaling graph | 5.26 | 6.47 | 参数依赖被忽略，探索效率下降 |
| **CORE (full)** | **4.62** | **5.47** | 两个组件协同贡献 |

### 关键发现
- Scaling graph 贡献更大：去掉后 latency 退化约 0.5-1.0（log scale），说明参数依赖建模对采样质量至关重要
- Reward shaping 对 Edge 平台尤其关键：约束更紧时（面积限制 0.2mm²），无 reward shaping 完全找不到可行解
- 采样效率：固定 40,000 预算下，CORE 在约 2,000 episodes（64,000 samples）内收敛，GA 在相同预算下远未收敛
- 小模型（NCF、DLRM）上 HASCO 有竞争力，因为设计空间较小，两步优化足够

## 亮点与洞察
- **Scaling Graph 的通用性**：这种通过有向图编码参数依赖、动态约束范围的解码方式，不仅适用于 DNN 加速器，还可推广到任何结构化设计问题（如编译器调优、芯片设计）。巧妙之处在于把约束编码从"验证后拒绝"变为"生成时保证"，大幅减少无效采样
- **一步 RL 的优势**：消除了多步 RL 中 reward 传播、中间状态维护、马尔可夫假设等问题。对于 DSE 这类"评估贵但生成配置本身不需要多步决策"的问题，单步建模是更自然的选择
- **与 GRPO/DeepSeek-R1 的思想共通**：批次内相对优势替代 critic，这种 critic-free 设计在仿真昂贵场景下特别合理——学一个准确的 value function 本身就需要大量样本，不如直接用批次内比较。论文作者也注明了与 DeepSeek-R1 的方法独立提出但殊途同归

## 局限与展望
- **静态映射假设**：当前仅考虑编译时固定的映射策略，未支持运行时自适应 dataflow，这在动态负载场景下可能不够优
- **仿真保真度依赖**：性能依赖 MAESTRO cost model 的准确性，实际流片后的结果可能有偏差，未做 FPGA/ASIC 实物验证
- **单步建模的局限**：对于需要多阶段决策的设计问题（如多芯片系统设计、涉及路由和布局的问题），单步建模可能不够表达
- **小模型上优势不明显**：NCF、DLRM 等小模型上 HASCO 仍有竞争力，说明在设计空间较小时，CORE 的结构化建模优势未充分发挥
- **可改进方向**：(1) 将 scaling graph 扩展为条件图，支持运行时依赖；(2) 引入 surrogate model 减少仿真调用；(3) 多目标 Pareto 优化替代加权求和

## 相关工作与启发
- **vs ArchGym**：ArchGym 用多步 RL 建模 DSE，需要长 rollout 和中间状态维护，约束用启发式 masking 处理。CORE 用单步直出+scaling graph 编码约束，更简洁高效，但牺牲了对序列决策问题的适用性
- **vs HASCO**：HASCO 是两步法——先贝叶斯优化硬件再启发/Q-learning 优化映射，忽略硬件-映射耦合。CORE 联合优化两者，且在大模型上优势显著（VGG-16 latency 降低 2.4 个数量级）
- **vs DeepSeek-R1 (GRPO)**：两者独立提出了批次内相对优势替代 critic 的思想，CORE 用于硬件设计、GRPO 用于 LLM 训练。这种 critic-free 范式可能对所有"评估贵、动作空间大"的优化问题都有启发

## 评分
- 新颖性: ⭐⭐⭐⭐ 单步 RL + scaling graph 的组合有新意，但各组件本身不算全新
- 实验充分度: ⭐⭐⭐⭐ 7 个 DNN 模型 × 2 平台 × 2 指标，消融完整，但缺少实物验证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，公式与图配合好，但 background 偏长
- 价值: ⭐⭐⭐⭐ 对硬件 DSE 领域有实际价值，scaling graph 思想可迁移到其他结构化优化问题

## 局限与展望
- 仅在 MAESTRO cost model 上验证，未在真实芯片后端（FPGA/ASIC flow）做端到端验证
- 策略网络是全连接网络，对更复杂的设计空间（如多层异构加速器）可能需要更强的架构
- 单步 RL 的联合分布分解假设了参数的链式条件独立，对非链式依赖结构可能不够灵活
- 固定 batch size=32 限于 CPU 并行数，可探索异步仿真进一步提升效率

## 相关工作与启发
- **vs ArchGym (2023)**：ArchGym 用标准 multi-step RL，需长 rollout + reward shaping；CORE 一步直出，样本效率高 1-2 个数量级
- **vs HASCO (2021)**：HASCO 用两步法（先硬件后映射），CORE 联合优化，充分利用硬件-映射耦合
- **vs DiGamma**：DiGamma 用启发式搜索做联合优化但不可扩展；CORE 用可学习策略，泛化性更强
- **与 LLM 中 GRPO 的连接**：critic-free + 批次内 baseline 的思想与 DeepSeek-R1 的 GRPO 一脉相承，说明这种 paradigm 在不同领域都有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 将单步 RL + scaling graph 组合用于硬件 DSE 是新颖的，但各组件（Beta 分布采样、critic-free 优化）本身不完全新
- 实验充分度: ⭐⭐⭐⭐ 7 个 DNN、2 个平台、清晰的消融实验；但缺少真实芯片验证和与更多 RL 方法的对比
- 写作质量: ⭐⭐⭐⭐ 框架图和算法伪代码清晰，scaling graph 解释直观
- 价值: ⭐⭐⭐⭐ 在硬件 DSE 这一重要问题上取得显著改进，方法具有通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](../../AAAI2026/reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] Adaptive Cooperative Transmission Design for URLLC via Deep RL](adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)

</div>

<!-- RELATED:END -->
