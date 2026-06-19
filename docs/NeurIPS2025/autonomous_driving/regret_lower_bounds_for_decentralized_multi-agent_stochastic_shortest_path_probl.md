---
title: >-
  [论文解读] Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems
description: >-
  [NeurIPS 2025][自动驾驶][多智能体强化学习] 本文首次为去中心化多智能体随机最短路径问题（Dec-MASSP）在线性函数逼近设定下建立了 $\Omega(\sqrt{K})$ 的 regret 下界，通过构造难以学习的实例族并利用对称性论证识别最优策略结构，证明了该下界与已有上界在 episode 数 $K$ 上达到匹配。
tags:
  - "NeurIPS 2025"
  - "自动驾驶"
  - "多智能体强化学习"
  - "随机最短路径"
  - "Regret下界"
  - "去中心化学习"
  - "线性函数逼近"
---

# Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems

**会议**: NeurIPS 2025  
**arXiv**: [2511.04594](https://arxiv.org/abs/2511.04594)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 多智能体强化学习、随机最短路径、Regret下界、去中心化学习、线性函数逼近

## 一句话总结
本文首次为去中心化多智能体随机最短路径问题（Dec-MASSP）在线性函数逼近设定下建立了 $\Omega(\sqrt{K})$ 的 regret 下界，通过构造难以学习的实例族并利用对称性论证识别最优策略结构，证明了该下界与已有上界在 episode 数 $K$ 上达到匹配。

## 研究背景与动机
**领域现状**: 随机最短路径（SSP）问题是目标导向决策的基础模型。单智能体 SSP 在表格设定和线性函数逼近设定下的学习已被充分研究，上下界均已匹配（tabular: $\Omega(B^*\sqrt{SAK})$；线性: $\Omega(dB^*\sqrt{K})$）。

**现有痛点**: 多智能体系统（机器人集群、交通路由、分布式控制）天然需要去中心化协调，但 Dec-MASSP 的学习理论几乎空白。已有工作 (trivedi2023massp) 仅给出上界 $\widetilde{O}(B^{*1.5}d\sqrt{nK/c_{\min}})$，未建立任何下界。

**核心矛盾**: 缺少 regret 下界导致无法判断现有算法的最优性，也无法量化去中心化多智能体学习的固有困难程度。

**本文目标**: 回答"去中心化 MASSP 学习的基本极限是什么？"这一核心开放问题。

**切入角度**: 构造可证明难以学习的 LM-MASSP（线性混合 MASSP）实例族，利用信息论工具推导 regret 下界。

**核心 idea**: 通过精心设计的两节点 $n$ 智能体实例和新颖的对称性论证，首次证明 Dec-MASSP 的 $\Omega(\sqrt{K})$ regret 不可避免。

## 方法详解

### 整体框架
研究分为三步：(1) 构造难以学习的 MASSP 实例族；(2) 分析最优策略和值函数的结构；(3) 利用信息论推导 regret 下界。

### 关键设计

1. **LM-MASSP 实例构造**:

    - 最小但富表达力的网络：仅两个节点 $\mathcal{V}=\{s,g\}$，$n$ 个智能体
    - 全局状态空间 $\mathcal{S}=\{s,g\}^n$，大小 $2^n$（指数级）
    - 每个智能体动作空间 $\mathcal{A}_i=\{-1,1\}^{d-1}$，全局动作空间 $|\mathcal{A}|=2^{n(d-1)}$
    - 采用均匀代价结构：$c(\mathbf{s},\mathbf{a})=1$ 对所有非目标状态，使最优策略等价于最小化到达目标的期望时间
    - 参数化实例：每个实例由 $(n, \delta, \Delta, \theta)$ 确定，$\theta$ 在指数多个候选中变化

2. **新颖特征设计**:

    - 为转移概率设计了满足有效性条件的线性特征 $\phi(\mathbf{s}'|\mathbf{s},\mathbf{a}) \in \mathbb{R}^{nd}$
    - 核心创新：特征构造使得转移概率分解为依赖于各智能体局部信息的结构化形式
    - 转移概率仅取决于当前在节点 $s$ 的智能体数 $r$ 和转移后留在 $s$ 的智能体数 $r'$
    - 该特征设计对更广泛的 MARL 领域也有参考价值

3. **状态空间分区与对称性利用**:

    - 将指数状态空间 $\mathcal{S}$ 按在节点 $s$ 的智能体数分区为 $\mathcal{S}_0, \mathcal{S}_1, \ldots, \mathcal{S}_n$
    - 证明最优值函数对同类型（相同 $r$）的所有状态取值相同：$V^*(\mathbf{s})=V^*_r$
    - 证明值函数严格单调递增：$0=V^*_0 < V^*_1 < \cdots < V^*_n = B^*$
    - 利用此单调性避免了需要闭式值函数表达式的困难

4. **KL 散度界的推导**:

    - 标准方法（如单智能体情况）中 KL 散度仅含两项，可解析处理
    - 多智能体情况下 KL 散度含指数多项，直接分析不可行
    - 利用 KL 散度非负性和实例对称性，得到上界 $\text{KL}(\mathbb{P}^\pi_\theta \| \mathbb{P}^\pi_{\theta^j}) \leq 3 \cdot 2^{2n} \cdot \frac{\Delta^2}{\delta(d-1)^2} \cdot \mathbb{E}_{\theta,\pi}[N^-]$

### 证明策略
- **定理1（最优策略结构）**: 通过对 $r$ 的数学归纳，证明选择 $\mathbf{a}_\theta$（即 $a_{i,j}=\text{sgn}(\theta_{i,j})$）的策略在所有状态下最优
- **定理2（Regret下界）**: 对所有 $\theta \in \Theta$ 的 regret 求平均 → 分解为截断计数项 → 利用 Pinsker 不等式 + KL 散度上界 → 优化 $\Delta$ 得到最紧界

## 实验关键数据

### 主实验

本文为纯理论贡献，无实验数据。主要理论结果为：

| 结果 | 具体界 | 匹配情况 |
|------|--------|----------|
| Regret 下界（本文） | $\Omega\left(\frac{d\sqrt{KB^*/n}}{2^n}\right)$ | 在 $K$ 上匹配上界 $\widetilde{O}(B^{*1.5}d\sqrt{nK/c_{\min}})$ |
| 单智能体特例($n=1$) | $\Omega(dB^*\sqrt{K})$ | 恢复 min2022learning 的已知下界 |
| 有效条件 | $K > \frac{n(d-1)^2 \cdot \delta}{2^{10} B^* (\frac{1-2\delta}{1+n+n^2})^2}$ | 保证所选参数 $\Delta^*$ 有效 |

### 消融实验（理论层面的参数分析）

| 分析维度 | 结论 |
|----------|------|
| $n$ 的影响 | 下界含 $2^{-n}$ 因子，但不应理解为"更多智能体更容易"——$\Delta$ 的约束 $\Delta < 2^{-n}(\frac{1-2\delta}{1+n+n^2})$ 限制了 $n$ 的范围 |
| 近最优策略数量 | 存在指数多个近最优策略（与最优策略仅在少量状态-动作分量上不同），使学习极困难 |
| $\delta$ 参数范围 | $\delta \in (2/5, 1/2)$，控制从 $s$ 到 $g$ 的基础转移概率 |
| $d$ 维度 | 下界与特征维度 $d$ 成线性关系 |

### 关键发现
1. **$\Omega(\sqrt{K})$ 不可避免**: 无论何种去中心化学习算法（包括可共享参数估计的），都无法避免 $\sqrt{K}$ 量级的 regret
2. **单智能体作为特例恢复**: $n=1$ 时，下界退化为已知单智能体结果 $\Omega(dB^*\sqrt{K})$
3. **最优策略的优雅结构**: 尽管状态空间指数级，最优策略仅取决于参数 $\theta$ 的符号，值函数仅取决于在非目标节点的智能体数

## 亮点与洞察
- **首个 Dec-MASSP 下界**: 填补了去中心化多智能体 SSP 学习理论的重要空白
- **指数状态空间的分析**: 首次处理 MASSP 设定中指数级状态-动作空间的下界分析
- **通用性**: 不限定具体通信协议，结论适用于从完全信息共享到零通信的广泛设定
- **特征设计的独立价值**: 提出的线性特征构造方法对更广泛的 MARL 设定有参考意义
- **对称性论证的优雅性**: 通过状态空间分区和单调性避免了直接求解值函数闭式表达式

## 局限与展望
1. **两节点网络**: 实例构造限于两节点，虽足以证明下界但可能不反映实际网络复杂性
2. **$2^{-n}$ 衰减因子**: 下界中的指数衰减因子使得当 $n$ 较大时界变弱，$n$ 维度上的紧性仍待探讨
3. **线性函数逼近限制**: 结果仅适用于线性设定，非线性函数逼近下的下界仍为开放问题
4. **均匀代价假设**: 实例采用均匀代价结构简化分析，非均匀代价下的情况未覆盖
5. **缺乏实验验证**: 纯理论工作，下界的存在性本质使得实验验证困难
6. **通信影响**: 未分析不同通信水平（如通信图结构）对 regret 界的具体影响
7. **未来方向**: 模型错误规范下的 regret 界、非线性函数逼近、通信协议的影响

## 相关工作与启发
- **单智能体 SSP**: tarbouriech2020no 提出 UC-SSP；rosenberg2020near 给出近最优界；cohen2021minimax 达到 minimax 最优；min2022learning 在线性设定下给出匹配上下界——本文在多智能体维度上扩展了这条线
- **Dec-MASSP 上界**: trivedi2023massp 首先定义 Dec-MASSP 并给出 $\widetilde{O}(B^{*1.5}d\sqrt{nK/c_{\min}})$ 上界——本文补全了理论拼图的下界部分
- **去中心化 MARL**: zhang2018fully, trivedi2022multi 的线性代价逼近和通信图设定为本文提供了形式化框架
- **启发**: 本文的状态空间分区和对称性论证技巧可用于其他多智能体学习的下界分析

## 评分 ⭐4
首次建立 Dec-MASSP regret 下界，理论贡献扎实，证明技术新颖（特征设计+对称性论证），但 $2^{-n}$ 衰减因子限制了结论的完整性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[ICML 2025\] Hybrid Quantum-Classical Multi-Agent Pathfinding](../../ICML2025/autonomous_driving/hybrid_quantum-classical_multi-agent_pathfinding.md)
- [\[CVPR 2026\] Unsupervised Multi-agent and Single-agent Perception from Cooperative Views](../../CVPR2026/autonomous_driving/unsupervised_multi-agent_and_single-agent_perception_from_cooperative_views.md)
- [\[ICML 2025\] R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning](../../ICML2025/autonomous_driving/r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen.md)
- [\[ICCV 2025\] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](../../ICCV2025/autonomous_driving/srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)

</div>

<!-- RELATED:END -->
