---
title: >-
  [论文解读] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][离线强化学习] 提出SIHD框架，利用历史轨迹中的结构信息（结构熵）自适应构建多尺度扩散层次，用结构信息增益替代局部奖励预测作为条件引导信号，并引入结构熵正则化促进对离线数据中稀疏状态的探索，在D4RL基准上最高提升12.6%的决策性能。 领域现状 离线RL使用预收集数据训练策略…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "离线强化学习"
  - "扩散模型"
  - "层次规划"
  - "结构熵"
  - "长horizon决策"
---

# Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.21942](https://arxiv.org/abs/2509.21942)  
**代码**: [GitHub](https://github.com/SELGroup/SIHD.git)  
**领域**: 强化学习  
**关键词**: 离线强化学习, 扩散模型, 层次规划, 结构熵, 长horizon决策

## 一句话总结

提出SIHD框架，利用历史轨迹中的结构信息（结构熵）自适应构建多尺度扩散层次，用结构信息增益替代局部奖励预测作为条件引导信号，并引入结构熵正则化促进对离线数据中稀疏状态的探索，在D4RL基准上最高提升12.6%的决策性能。

## 研究背景与动机

### 领域现状

离线RL使用预收集数据训练策略，避免了在线交互。扩散模型因其强大的分布建模能力被引入离线RL：(1) 作为策略生成器（Diffusion Policy），直接生成多模态行为；(2) 作为轨迹生成器（Diffuser），以奖励为条件生成高回报序列。后者将离线决策转化为条件生成问题。

### 现有痛点

在长horizon任务中，扩散模型面临两个挑战：

**方差累积**：随轨迹长度增加，价值估计的方差指数级增长

**计算成本**：迭代去噪步骤的计算开销随序列长度线性增长

为此，层次扩散方法（HDMI, Hierarchical Diffuser）将决策分解为高层目标规划 + 低层动作生成。但现有方法存在**两个刚性限制**：
- **固定两层结构**：只有subgoal层和action层，无法适应不同任务的时间结构复杂度
- **单一预定义时间尺度**：使用固定间隔切分轨迹，忽视了不同区域的状态转移规律

### 本文切入角度

核心insight：历史轨迹中蕴含的**状态拓扑结构**可以指导扩散层次的构建。通过在状态空间上构建相似度图并优化结构熵，可以自动发现多层次的状态社区分区，进而自适应地进行轨迹分段和多尺度层次构建。同时，结构信息增益可以替代不可靠的局部奖励预测来引导条件扩散。

## 方法详解

### 整体框架

SIHD包含三个模块：
1. **层次构建模块**：从离线状态的拓扑结构中提取编码树，自适应构建多尺度扩散层次
2. **条件扩散模块**：在每层使用共享扩散模型，以结构信息增益为条件信号
3. **正则化探索模块**：用结构熵正则化鼓励探索稀疏状态，同时约束在低层社区内以避免分布外误差

### 关键设计

1. **多尺度扩散层次构建**：

    - 从离线数据集中提取所有状态 $s \in \mathcal{S}$，基于特征相似度（余弦相似度）建立k最近邻状态图 $\mathcal{G}_s$
    - 通过最大化一维结构熵 $\mathcal{H}^1(\mathcal{G}_s)$ 来选择最优 $k$ 值
    - 应用HCSE优化算法求解高度为 $\mathcal{K}$ 的最优编码树 $\mathcal{T}_s^*$，最小化 $\mathcal{K}$ 维结构熵：

    $\mathcal{H}^{\mathcal{K}}(\mathcal{G}) = \min_{h_\mathcal{T} \leq \mathcal{K}} \mathcal{H}^{\mathcal{T}}(\mathcal{G}), \quad \mathcal{H}^{\mathcal{T}}(\mathcal{G}) = -\sum_{\alpha \in \mathcal{T}, \alpha \neq \lambda}\left[\frac{g_\alpha}{\operatorname{vol}(\lambda)} \cdot \log\frac{\operatorname{vol}(\alpha)}{\operatorname{vol}(\alpha^-)}\right]$

    - 编码树自然定义了多层社区分区：每个节点 $\alpha$ 对应一个状态社区 $\mathcal{V}_\alpha$
    - 根据社区分区对轨迹进行自适应分段：同一社区内的连续状态构成一个segment，segment末尾的状态作为subgoal
    - 不同层级的编码树节点对应不同时间尺度的分段

2. **条件扩散模型（以结构信息增益为引导）**：

   最高层扩散器以累积奖励为条件：$y(\tau_g^{\mathcal{K},1}) = \exp(\sum_{t=0}^T \mathcal{R}(s_t, a_t))$

   **关键创新**：中间层和底层扩散器不使用奖励，而使用**结构信息增益**作为条件信号：

    $y(\tau_g^{h,i}) = \mathcal{H}^{\mathcal{T}_s^*}(\mathcal{G}_s; \alpha) = -\frac{g_\alpha}{\operatorname{vol}(\lambda)} \cdot \log\frac{\operatorname{vol}(\alpha)}{\operatorname{vol}(\alpha^-)}$

   这量化了"在知道转移在高层segment内的前提下，推断它在低层segment内所需的额外信息量"。使用classifier-free guidance整合条件信息：$\hat{\epsilon} = \epsilon_{\theta_h}(\tau_{g,k}^{h,i}, (1-\omega)y(\tau_{g,k}^{h,i}) + \omega\emptyset, k)$

   Theorem 4.1证明了条件生成可以分解为层次扩散过程：

    $p(\tau_0|y(\tau_0)) \propto p(\tau_g^{\mathcal{K},1})y(\tau_g^{\mathcal{K},1}) \cdot \prod_{h=1}^{\mathcal{K}-1}\prod_{i=1}^{l_g^h} p(\tau_g^{h,i})y(\tau_g^{h,i})$

3. **结构熵正则化**：

   基于底层扩散模型估计的状态转移概率构建完全加权图 $\mathcal{G}_s'$，计算其在编码树下的结构熵。Theorem 4.2建立了结构熵与Shannon熵的变分下界关系：

    $\mathcal{H}(S) - \sum_{h=1}^{\mathcal{K}-1}[\eta_h \cdot \mathcal{H}(\mathcal{U}_h)] \leq \mathcal{H}^{\mathcal{T}_s^*}(\mathcal{G}_s') \leq \mathcal{H}(S)$

   训练目标中加入正则项——**最大化** $\mathcal{H}(S)$（鼓励覆盖稀疏状态）的同时**最小化** $\mathcal{H}(\mathcal{U}_h)$（保持层次结构、约束探索范围）：

    $\mathcal{L}(\theta_h) = \mathbb{E}\sum_{i=1}^{l_g^h}\left[\|\hat{\epsilon} - \epsilon\|^2 - \eta\mathbb{I}_{h=1}\left[\mathcal{H}(S) - \sum_{j=1}^{\mathcal{K}-1}[\eta_j \cdot \mathcal{H}(\mathcal{U}_j)]\right]\right]$

### 损失函数 / 训练策略

每层扩散器共享参数，使用标准噪声预测MSE损失加上结构熵正则化（仅在底层 $h=1$ 时生效）。训练时预计算编码树并以字典结构存储社区分区以避免重复计算。

## 实验关键数据

### Gym-MuJoCo标准任务

| 环境+数据集 | Diffuser | HDMI | HD | **SIHD** | 提升 |
|------------|----------|------|-----|---------|------|
| HalfCheetah-Expert | 88.9 | 92.1 | 92.5 | **94.4** | 2.1% |
| Hopper-Medium | 74.3 | 76.4 | 99.3 | **103.1** | 3.8% |
| Walker2D-Replay | 70.6 | 80.7 | 84.1 | **89.7** | 6.7% |
| **平均提升（vs HD）** | — | — | — | **3.2%** | — |

### 长Horizon导航任务（Maze2D + AntMaze）

| 环境 | Diffuser | HDMI | HD | **SIHD** | 提升 |
|------|----------|------|-----|---------|------|
| Maze2D-U | 113.9 | 120.1 | 128.4 | **144.6** | **12.6%** |
| Maze2D-Medium | 121.5 | 121.8 | 135.6 | **148.5** | 9.5% |
| Maze2D-Large | 123.0 | 128.6 | 155.8 | **161.7** | 3.8% |
| AntMaze-U | 76.0 | — | 94.0 | **96.5** | 2.7% |
| AntMaze-Medium | 31.9 | — | 88.7 | **92.2** | 3.9% |
| AntMaze-Large | — | — | 83.6 | **89.4** | 6.9% |

### 消融实验

| 变体 | Hopper-Med | Maze2D-Med | AntMaze-Med | 说明 |
|------|-----------|------------|-------------|------|
| SIHD（完整） | **103.1** | **148.5** | **92.2** | 全部模块 |
| SIHD-DH（去多尺度层次） | ~93 | ~125 | ~80 | **降幅最大**，证明层次是核心 |
| SIHD-CG（去条件引导） | ~98 | ~140 | ~87 | 结构信息引导有帮助 |
| SIHD-ER（去正则化） | ~100 | ~142 | ~85 | 正则化在稀疏奖励中更重要 |
| SIHD-FT（固定间隔分段） | — | 146.8 | — | 结构熵优于固定分段 |

### 计算效率

| 方法 | Maze2D训练时间 | Maze2D规划时间 |
|------|-------------|-------------|
| Diffuser | 48.3s | 4.7s |
| HD | 5.8s | 1.9s |
| SIHD | 6.0s | 1.6s |

### 关键发现

- SIHD在中等和低质量数据集上的优势更明显（Expert 1.6% vs Medium 3.8% vs Replay 3.9%），说明结构熵正则化有效缓解了对高质量数据的依赖
- 在长horizon任务中优势更大（Maze2D平均8.3%），因为更深的扩散层次（$\mathcal{K}=4$）更好地支持长期决策
- 去除多尺度层次（SIHD-DH）的性能下降最大，特别是在长horizon任务中，证明了自适应层次构建是核心贡献
- 即使控制参数总量与HD相同，增加层次层数仍带来一致提升——性能增益来源于结构而非容量
- 计算效率与HD相当，比Diffuser快80%+

## 亮点与洞察

- **数据驱动层次构建**：不再依赖人工设计的时间尺度，而是从状态拓扑结构中自动发现合理的层次分解
- **结构信息增益替代奖励预测**：避免了在局部子轨迹上预测奖励的不可靠性，用拓扑信息量提供更稳定的引导
- **正则化的双重约束**：既鼓励探索稀疏状态（最大化Shannon熵），又限制探索在低层社区内（最小化社区熵），精妙地平衡了探索与保守
- **理论完备**：Theorem 4.1保证了层次分解的可分性，Theorem 4.2建立了正则化的变分界

## 局限与展望

- 大规模离线数据集上结构熵优化的计算开销较大（通过预计算缓解）
- Subgoal约束使用简单的末状态替换策略，可能不够精细
- 编码树的层数 $\mathcal{K}$ 需要根据任务调整（sensitivity分析显示 $\mathcal{K}=3$ 或 $4$ 为佳）
- 正则化系数 $\eta$ 对数据质量敏感（Expert数据用小 $\eta$，Medium数据用大 $\eta$）
- 仅在D4RL基准上验证，未扩展到更复杂的真实世界任务

## 相关工作与启发

- 与HDMI/HD的对比：它们使用固定两层+单一时间尺度，而SIHD自适应多层+多尺度
- 结构熵（Li & Pan 2016）原本用于网络科学和图学习，本文首次将其应用于RL的层次决策
- Evans & Şimşek (2023)展示了多层策略层次在组合长horizon任务中的优势，为SIHD提供了动机
- 启发：结构信息原理可能在其他需要层次分解的领域（如自然语言处理的层次建模）中也有应用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将结构信息理论首次引入层次扩散RL，三个设计点均有创新
- 实验充分度: ⭐⭐⭐⭐⭐ D4RL全面基准+消融+效率分析+参数敏感度+可视化
- 写作质量: ⭐⭐⭐⭐ 框架清晰，理论推导完整，但符号系统较复杂
- 价值: ⭐⭐⭐⭐ 在长horizon稀疏奖励任务中提供了显著且一致的提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](../../ICML2025/reinforcement_learning/divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
