---
title: >-
  [论文解读] Parametrized Multi-Agent Routing via Deep Attention Models
description: >-
  [AAAI2026][优化/理论][组合优化] 提出Deep FLPO框架，将最大熵原理（MEP）的代数结构与permutation-invariant的encoder-decoder神经网络（SPN）融合，解决设施选址与路径联合优化的NP-hard混合整数问题，实现策略推理100倍加速、与Gurobi精确解匹配且快1500倍。
tags:
  - "AAAI2026"
  - "优化/理论"
  - "组合优化"
  - "多智能体路径规划"
  - "最大熵原理"
  - "Transformer"
  - "Facility Location"
---

# Parametrized Multi-Agent Routing via Deep Attention Models

**会议**: AAAI2026  
**arXiv**: [2507.22338](https://arxiv.org/abs/2507.22338)  
**作者**: Salar Basiri, Dhananjay Tiwari, Srinivasa M. Salapaka (UIUC)  
**代码**: [GitHub](https://github.com/salar96/LearningFLPO)  
**领域**: 优化  
**关键词**: 组合优化, 多智能体路径规划, 最大熵原理, Transformer, Facility Location  

## 一句话总结

提出Deep FLPO框架，将最大熵原理（MEP）的代数结构与permutation-invariant的encoder-decoder神经网络（SPN）融合，解决设施选址与路径联合优化的NP-hard混合整数问题，实现策略推理100倍加速、与Gurobi精确解匹配且快1500倍。

## 背景与动机

### 参数化序贯决策问题 (ParaSDM)
传统神经组合优化（NCO）关注固定状态/动作空间的离散优化（如TSP、VRP），但实际应用中存在一类更复杂的问题——**参数化序贯决策**（ParaSDM）：$N$个智能体在$K$步中做离散决策，同时需联合优化一组共享的连续参数$\mathcal{Y} \subset \mathbb{R}^d$（如设施位置）。这导致离散策略和连续参数之间强耦合、高度非凸。

### FLPO问题
ParaSDM的典型子问题是**设施选址与路径优化**（FLPO）：$N$个agent（如无人机）需经过$M$个facility（如充电站）从起点到终点，目标是同时优化离散路径$\eta^i(\gamma)$和连续节点位置$\mathcal{Y}$以最小化总运输代价。即使在$N=2, M=4$的小规模下，解空间也包含大量局部最优。

### MEP方法的可扩展性瓶颈
前驱工作基于最大熵原理（MEP）将硬关联$\eta^i \in \{0,1\}$松弛为软概率$p^i \in [0,1]$，通过退火参数$\beta$平衡探索与利用。MEP的策略变量具有Gibbs分布的解析解：

$$p_\beta^i(\gamma) = \frac{\exp[-\beta \cdot d^i(\gamma)]}{\sum_{\gamma' \in \mathcal{G}^i} \exp[-\beta \cdot d^i(\gamma')]}$$

但每步梯度计算复杂度为$O(NM^4)$（使用Markov DP优化后），对中等规模问题已不可行。

## 方法详解

### Shortest Path Network (SPN)

SPN是一个permutation-invariant的encoder-decoder模型，逼近MEP的Gibbs策略分布。

**Encoder**：采用induced attention机制。将所有agent的起点$S$、终点$\Delta$和设施坐标$Y^t$拼接为$E_0 \in \mathbb{R}^{N \times (M+2) \times d_h}$，经$L_{\text{enc}}$层注意力块处理。关键是使用$M^*$个可学习inducing point $T \in \mathbb{R}^{M^* \times d_h}$将标准自注意力$O(NM^2)$降为$O(NM \cdot M^*)$：

$$\hat{E}_{l-1} = \text{MHA}(T, E_{l-1}, E_{l-1})$$
$$I_l = \text{LN}(E_{l-1} + \text{MHA}(E_{l-1}, \hat{E}_{l-1}, \hat{E}_{l-1}))$$

**Decoder**：通过门控插值融合当前位置和终点信息，生成目标感知的查询向量：

$$\alpha = \sigma([h_X, h_\Delta] W), \quad q = \alpha \odot h_X + (1-\alpha) \odot h_\Delta$$
$$\Pi_k = \text{softmax}(E_{L_{\text{enc}}} \cdot q / \sqrt{d_h})$$

两种decoder变体：DED（直接嵌入）和DCAD（双交叉注意力）。消融实验表明DED+退火监督效果最佳，平均最优性gap约6%。

### 混合采样梯度估计

为近似MEP的退火行为，对每个agent采样$L$条路径：前$b$条由SPN的beam search或采样生成（短路径，利用Gibbs分布赋予高权重），其余$L-b$条来自均匀随机采样（促进低$\beta$时的探索）。Free energy梯度的估计为：

$$\widehat{\nabla_{\mathcal{Y}^t} F_\beta} = \sum_{i=1}^N \rho_i \sum_{q=1}^L \hat{p}_\beta^i(\gamma^q) \nabla_{\mathcal{Y}^t} d^i(\gamma^q)$$

其中$\hat{p}_\beta^i$在采样集上重新归一化Gibbs权重。所有梯度$\nabla_{\mathcal{Y}^t} d^i(\gamma^q)$通过PyTorch自动微分获得。

### 课程学习训练

| Phase | 模式 | 预训练 | Epochs | 节点数$M$ |
|-------|------|--------|--------|----------|
| 1 | 监督学习 (KL散度) | 无 | 10k | 10 |
| 2 | 监督学习 | Phase 1 | 1.5k | 50 |
| 3 | RL (Policy Gradient) | 无/Phase 2 | 50k | 50 |
| 4 | RL (Policy Gradient) | Phase 3 | 10k | 100 |
| FT | RL (Fine-Tuning) | Phase 4 | 1k | {10, 100} |

监督阶段最小化模型策略与Gibbs目标分布的KL散度，强化学习阶段使用REINFORCE + POMO-style baseline。

## 实验关键数据

### 可扩展性加速
- **梯度计算**：$N=200, M=100$时SPN+Sampling比原始ParaSDM快约$200\times$
- **最短路径推理**：$N=M=200$时SPN贪心推理比ParaSDM DP快约$100\times$

### 对比基线

| 方法 | 代价（vs Gurobi） | 速度 |
|------|-------------------|------|
| GA / SA / CEM | 比Deep FLPO高$>10\times$ | 更慢 |
| Gurobi (精确解) | 最优 | 基准 |
| Deep FLPO (高$\beta$) | 仅高2% | 快$\sim 1500\times$ |
| Deep FLPO (退火) | **匹配Gurobi** | 快$\sim 1500\times$ |

### 大规模FLPO场景
- $N=200, M=40$：SPN-only比退火版快$10\times$，退火版代价低17%
- $N=800, M=200$：退火版代价低8%但慢$\sim 3\times$——此规模已超出Gurobi和元启发式的实际可行范围
- SPN泛化测试：训练于10-100节点，在10-300节点上平均最优性gap仅6%

### 消融实验
- DED+退火监督在$M=10\sim300$上平均gap约6%，比DCAD+退火好约4%
- 监督预训练对DED改善约3%gap，DCAD若无监督预训练则RL训练崩溃（400-500% gap）

## 亮点

- **结构先验+神经网络的深度融合**：不是黑盒替代传统方法，而是将MEP的Gibbs分布代数结构嵌入SPN架构
- **首个可扩展的ParaSDM神经求解器**：将原本$O(NM^4)$的DP解法实质性降至$O(NM^2)$且可GPU并行
- **退火+混合采样策略精巧**：低$\beta$时均匀探索避免局部最优，高$\beta$时SPN聚焦短路径实现利用
- **跨规模泛化**：10-100节点训练、300节点测试仍保持6%最优性gap
- **实用价值**：匹配Gurobi精确解同时快1500倍，支持800 agent × 200 node的大规模场景

## 局限与展望

- **仅考虑欧式距离代价**：未涉及道路网络、障碍物等更现实的代价函数
- **设施无容量约束**：实际充电站有容量限制，需扩展模型处理多agent竞争
- **2D场景为主**：虽然formulation支持$d$维，但实验仅在$\mathcal{D} = [0,1]^2$验证
- **SPN训练成本**：4阶段课程学习加起来可能需要大量GPU时间，论文未详细报告
- **固定Gibbs温度训练**：SPN在高$\beta$下训练后推理，但$\beta$的选择对性能有影响

## 与相关工作的对比

- **Attention Model (Kool et al.)**：专为TSP/VRP设计，无法联合优化连续参数$\mathcal{Y}$；SPN在此基础上扩展decoder以适配ParaSDM
- **Pointer Network**：使用LSTM+注意力，SPN用Transformer+induced attention更高效
- **SIL (Self-Improved Learning)**：线性注意力扩展NCO，但不处理参数化动作空间
- **原始MEP (Srivastava & Salapaka)**：理论性强但$O(NM^4)$不可扩展；Deep FLPO保留MEP结构保证同时实现百倍加速
- **Gurobi**：精确但不可扩展到大规模；Deep FLPO在精度匹配的前提下快1500倍

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将MEP结构先验和NCO transformer融合解决ParaSDM，开辟新方向
- 实验充分度: ⭐⭐⭐⭐ — 覆盖可扩展性、对比基线、消融，但真实应用场景验证不足
- 写作质量: ⭐⭐⭐⭐ — 问题formulation严谨，方法推导清楚，但符号较多增加阅读难度
- 价值: ⭐⭐⭐⭐⭐ — 解决了ParaSDM的可扩展性核心瓶颈，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](../../NeurIPS2025/optimization/effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](../../ICML2025/optimization/in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[ICLR 2026\] Conditioned Initialization for Attention](../../ICLR2026/optimization/conditioned_initialization_for_attention.md)
- [\[CVPR 2026\] Beyond Single Solution: Multi-Hypothesis Collaborative Deep Unfolding Network for Image Compressive Sensing](../../CVPR2026/optimization/beyond_single_solution_multi-hypothesis_collaborative_deep_unfolding_network_for.md)
- [\[AAAI 2026\] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)

</div>

<!-- RELATED:END -->
