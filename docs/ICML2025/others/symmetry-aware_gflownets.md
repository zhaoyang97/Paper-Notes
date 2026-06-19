---
title: >-
  [论文解读] Symmetry-Aware GFlowNets
description: >-
  [ICML 2025][GFlowNets] 揭示 GFlowNets 在图生成中因等价动作（不同动作产出同构图）导致的系统性采样偏差——节点生成偏向低对称图、片段生成偏向高对称组件，提出通过终态自同构群大小缩放奖励的简单修正方法 SA-GFN，仅需一次自同构群计算即可实现无偏采样。 领域现状：GFlowNets 是一种学习…
tags:
  - "ICML 2025"
  - "GFlowNets"
  - "graph symmetry"
  - "automorphism group"
  - "equivalent actions"
  - "molecule generation"
---

# Symmetry-Aware GFlowNets

**会议**: ICML 2025  
**arXiv**: [2506.02685](https://arxiv.org/abs/2506.02685)  
**代码**: [https://github.com/hohyun312/sagfn](https://github.com/hohyun312/sagfn)  
**领域**: 其他  
**关键词**: GFlowNets, graph symmetry, automorphism group, equivalent actions, molecule generation

## 一句话总结

揭示 GFlowNets 在图生成中因等价动作（不同动作产出同构图）导致的系统性采样偏差——节点生成偏向低对称图、片段生成偏向高对称组件，提出通过终态自同构群大小缩放奖励的简单修正方法 SA-GFN，仅需一次自同构群计算即可实现无偏采样。

## 研究背景与动机

**领域现状**：GFlowNets 是一种学习按奖励正比概率采样组合对象（如分子图）的生成框架。它通过序列动作构建图，训练目标（TB/DB）依赖于状态转移概率的精确计算。

**现有痛点**：图构建过程中存在"等价动作问题"——不同动作可能产生同构图（例如在 $G_1$ 中节点 2 和 3 是等价的，添加新节点到任一个都得到同构的 $G_2 \cong G_3$），此时转移概率必须累加所有等价动作的概率。但检测等价动作需要计算开销高昂的图同构测试，现有方法要么忽略（引入偏差）要么用近似（不精确）。

**核心矛盾**：忽略等价动作引入的偏差是**系统性的**：节点逐步生成模式下，模型偏向采样低对称图（对称图被欠采样，因子为 $1/|\text{Aut}(G)|$）；片段组装模式下，高对称片段被过度采样（因其提供更多等价前向动作）。ZINC250k 中超 50% 分子有多于 1 个对称性——偏差对分子发现影响严重。

**本文目标**：找到一种精确、高效、且易于实现的方法来消除 GFlowNets 中因图对称性引起的采样偏差。

**切入角度**：观察到等价动作的数量比值可以用自同构群大小的比值来表达（Lemma 4.5），因此沿轨迹的累积修正可以 telescope 为仅依赖终态自同构群大小的单次修正。

**核心 idea**：将奖励乘以终态自同构群大小 $|\text{Aut}(G_n)|$ 即可消除等价动作偏差，无需逐步检测。

## 方法详解

### 整体框架

SA-GFN 的核心修改极其简洁：在现有 GFlowNet 管线中，仅将奖励函数从 $R(G)$ 替换为 $\tilde{R}(G) = |\text{Aut}(G)| \cdot R(G)$（节点生成）或 $\tilde{R}(G) = \frac{|\text{Aut}(G)|}{\prod_i |\text{Aut}(C_i)|} \cdot R(G)$（片段生成）。不修改网络架构、训练流程或推理过程，仅 17 行代码改动。

### 关键设计

1. **等价动作的理论刻画（Theorems 4.3, 4.4）**:

    - 功能：建立轨道等价与转移等价的关系，证明轨道等价足以用于 GFlowNets
    - 核心思路：定义轨道等价——动作 $(G_1, t_1, u_1)$ 和 $(G_2, t_2, u_2)$ 轨道等价当且仅当 $t_1=t_2$ 且存在自同构 $\pi$ 使得 $\pi(G_1)=G_2, \pi(u_1)=u_2$。**Theorem 4.3** 证明轨道等价 $\Rightarrow$ 转移等价（同一轨道上的动作产生同构图）。**Theorem 4.4** 证明轨道等价的 flow 约束成立蕴含转移等价的 flow 约束——即用轨道等价替代转移等价对 GFlowNets 是充分的
    - 设计动机：转移等价需要逐步图同构测试（$O(K \times H)$ 次），轨道等价直接从图结构计算，为后续化简打下基础

2. **自同构群修正定理（Theorem 4.6 + Lemma 4.5）**:

    - 功能：将逐步轨道计数化简为自同构群大小的比值
    - 核心思路：**Lemma 4.5** 证明 $\frac{|\text{Orb}(G,u,v)|}{|\text{Orb}(G',u,v)|} = \frac{|\text{Aut}(G)|}{|\text{Aut}(G')|}$（AddEdge 情况，AddNode 类似）。**Theorem 4.6** 将前向/后向策略的比值分解为：$\frac{p_{\bar{\mathcal{A}}}(a|s)}{q_{\bar{\mathcal{A}}}(a|s')} = \frac{|\text{Aut}(G)|}{|\text{Aut}(G')|} \cdot \frac{p_\mathcal{E}(G'|G)}{q_\mathcal{E}(G|G')}$。沿完整轨迹 telescope，自同构群比值的乘积约掉中间项，仅剩 $|\text{Aut}(G_0)|/|\text{Aut}(G_n)| = 1/|\text{Aut}(G_n)|$（初态自同构群为 1）
    - 设计动机：这一化简将"逐步检测等价动作"的 $O(K \times H)$ 复杂度降至"终态一次自同构群计算"的 $O(1)$，且精确无近似

3. **奖励缩放修正（Corollary 5.1 + Theorem 5.2）**:

    - 功能：将理论修正翻译为极简的实现——仅修改奖励
    - 核心思路：**TB 修正**（Corollary 5.1）：将 TB 损失中的 $R(G_n)$ 替换为 $|\text{Aut}(G_n)| R(G_n)$。不修正则模型按 $p \propto R(G)/|\text{Aut}(G)|$ 采样——系统性欠采样高对称图。**DB 修正**（Theorem 5.2）：定义图级 flow 函数 $\tilde{F}$，同样通过缩放奖励实现修正。**片段修正**（Theorem 5.3）：$\tilde{R}(G) = |\text{Aut}(G)| R(G) / \prod_{i=1}^k |\text{Aut}(C_i)|$——高对称片段因提供更多等价前向动作被过度采样，需除以各片段自同构群大小来惩罚
    - 设计动机：奖励缩放方法同时适用于 TB 和 DB 目标，实现极简，且自同构群可用 bliss 算法高效计算

### 损失函数 / 训练策略

修正后的 TB 损失为 $\mathcal{L}_{\text{TB}}(\tau) = (\log \frac{Z \prod p_\mathcal{E}(G_{t+1}|G_t)}{|\text{Aut}(G_n)| R(G_n) \prod q_\mathcal{E}(G_t|G_{t+1})})^2$。DB 损失中可以在每步乘以对称比值（Flow Scaling），或等价地仅缩放终态奖励（Reward Scaling）。论文还给出了无偏似然估计器：$\bar{p}_\mathcal{A}(x) \approx \frac{1}{M|\text{Aut}(G_n)|} \sum_{i=1}^M \frac{p_\mathcal{E}(\tau_i)}{q_\mathcal{E}(\tau_i|G_n)}$。

## 实验关键数据

### 合成图实验（72296 个终态，TB 目标）

| 方法 | L1 误差 ↓ | 说明 |
|------|----------|------|
| Vanilla | 高（不收敛） | 不处理等价动作 |
| PE (Ma et al. 2024) | 中等 | 近似轨道检测，每步计算 |
| Transition Correction | 极低 | 精确图同构测试，计算代价 $K \times H$ 倍 |
| **Reward Scaling** | **极低** | 仅终态一次计算，与精确方法等效 |

### 分子生成实验

| 生成方式 | 方法 | Diversity ↑ | Top-K div ↑ | Top-K reward ↑ | Unique Frac |
|---------|------|------------|-------------|----------------|-------------|
| Atom | Vanilla | 0.929 | 0.077 | 1.09 | 0.93 |
| Atom | **Reward Scaling** | **0.959** | **0.046** | **1.091** | **1.0** |
| Fragment | Vanilla | 0.877 | 0.153 | 0.941 | 1.0 |
| Fragment | **Reward Scaling (Exact)** | **0.879** | 0.151 | **0.952** | 1.0 |

### 关键发现

- Illustrative 实验（6 节点连通图，112 个终态，均匀奖励）完美验证了理论：Vanilla 模型的终态概率按 $|x|$（同构类大小）聚类，Reward Scaling 后变为均匀
- Vanilla 在片段生成中过度偏好高对称组件：cyclohexane（环己烷，$|\text{Aut}|=12$）出现 5220 次 vs 修正后仅 1042 次
- DB 目标下 Reward Scaling 收敛比 Flow Scaling 稍慢，因为修正信号仅在轨迹末端提供（类似于稀疏奖励 vs 密集奖励），但最终精度相同
- 似然估计器在少量采样（$M \sim 10$）即可给出准确估计，训练后模型比随机模型更容易估计

## 亮点与洞察

- **问题识别精准**：等价动作偏差不仅存在而且是系统性的——节点生成欠采样高对称图，片段生成过采样高对称组件。这两个方向相反的偏差之前未被清晰刻画
- **解决方案的简洁与优雅**：将复杂的"逐步检测等价动作"化简为"终态一次自同构群计算+奖励缩放"，理论推导链优美（Lemma 4.5 → Theorem 4.6 → telescope → Corollary 5.1），实现仅需约 17 行代码
- **同时适用于 TB 和 DB 目标、节点和片段两种生成模式**：通用性强，未来所有 GFlowNet 图生成工作都应考虑此修正

## 局限性

- 理论保证依赖于预定义图动作集的特定结构（AddNode/AddEdge/AddFragment），新动作类型需单独验证
- GNN 的有限表达能力可能使不同轨道的节点获得相同表示，导致策略无法区分非等价动作——这是 GNN 表达力问题而非本文方法的局限
- 主要在分子生成上验证，其他类型图生成（如社交网络、程序图）的效果待测试
- 奖励与对称性负相关时修正效果有限（如 atom 任务中 QM9 数据集奖励与对称性负相关）

## 相关工作与启发

- **vs Ma et al. (2024)**：用位置编码近似检测等价动作，需要每步计算且不精确；SA-GFN 精确且仅需终态一次计算，效率提升 $K \times H$ 倍
- **vs Chen et al. (2021)**：在自回归图生成中讨论了轨道与等价转移的关系，启发了本文；但未给出 GFlowNet 具体的修正方案
- **vs MaxEnt RL（Tiapkin et al. 2024）**：GFlowNets 被发现等价于 MaxEnt RL，但若未处理等价动作则仍有偏差。SA-GFN 的修正使两者在图 DAG 环境中真正一致

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 识别出GFlowNets图生成中的系统性对称偏差，解决方案理论优雅且实现极简
- 实验充分度: ⭐⭐⭐⭐ 合成图完美验证理论，分子任务展示实际价值，但领域覆盖有限
- 写作质量: ⭐⭐⭐⭐⭐ 从问题定义到理论推导到实验验证的逻辑链极其清晰
- 价值: ⭐⭐⭐⭐ 所有GFlowNet图生成工作的必要修正，推荐为默认配置

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](../../NeurIPS2025/others/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[ICML 2025\] Time-Aware World Model for Adaptive Prediction and Control](time-aware_world_model_for_adaptive_prediction_and_control.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[ICML 2025\] Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability](constrained_hamiltonian_systems_on_observation-induced_fiber_bundles_theory_of_s.md)

</div>

<!-- RELATED:END -->
