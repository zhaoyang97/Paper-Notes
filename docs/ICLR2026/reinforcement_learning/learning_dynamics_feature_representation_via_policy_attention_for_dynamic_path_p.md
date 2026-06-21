---
title: >-
  [论文解读] Learning Dynamics Feature Representation via Policy Attention for Dynamic Path Planning in Urban Road Networks
description: >-
  [ICLR 2026][强化学习][动态路径规划] 针对 RL 解动态路径规划时「全局动态信息完整但太贵、局部动态高效但漏关键信息」的两难，本文用「策略注意力筛任务相关子图 + n-hop 邻域抽节点局部特征」的分层蒸馏，把高维全局路网动态压成紧凑且近似满足马尔可夫性的状态，给任意 RL 主干提速且提质。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "动态路径规划"
  - "状态表示"
  - "策略注意力"
  - "n-hop 邻域"
  - "城市路网"
  - "DQN/PPO"
---

# Learning Dynamics Feature Representation via Policy Attention for Dynamic Path Planning in Urban Road Networks

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1E4Bltg6Xb](https://openreview.net/forum?id=1E4Bltg6Xb)  
**代码**: [https://anonymous.4open.science/r/UrbanDynamicPathPlanning-A59E](https://anonymous.4open.science/r/UrbanDynamicPathPlanning-A59E)  
**领域**: 强化学习 / 动态路径规划  
**关键词**: 动态路径规划, 状态表示, 策略注意力, n-hop 邻域, 城市路网, DQN/PPO  

## 一句话总结
针对 RL 解动态路径规划时「全局动态信息完整但太贵、局部动态高效但漏关键信息」的两难，本文用「策略注意力筛任务相关子图 + n-hop 邻域抽节点局部特征」的分层蒸馏，把高维全局路网动态压成紧凑且近似满足马尔可夫性的状态，给任意 RL 主干提速且提质。

## 研究背景与动机
**领域现状**：城市路网里的动态路径规划（Dynamic Path Planning, DPP）通常把路网建模成有向加权图，边权 $w(v_i,v_j;t)$ 随交通状况实时变化。传统做法先用统计/深度模型预测未来路况，再在预测图上跑 A\*、D\* Lite 这类经典搜索，但规划质量被预测精度死死卡住，遇到事故、封路等突发事件就崩。强化学习提供了另一条路：不显式建模未来动态，而是把动态塞进状态、靠交互学决策，值函数会隐式吸收状态转移的统计规律，对未见事件更鲁棒。

**现有痛点**：RL 解 DPP 的成败几乎全押在「状态里怎么表示交通动态 $f_t$」上。两类做法各有死穴——全局方法把整张图的动态都编码进状态，信息完整但维度爆炸、算力吃不消（GCN 这类图方法的开销随图规模线性增长，大城市实时规划不现实）；局部方法只看智能体周边的部分观测，高效但容易漏掉非局部的关键动态，导致短视、次优的路线。

**核心矛盾**：更要命的是，状态表示不充分会破坏马尔可夫性——当前状态没法概括决策所需的全部信息，训练会不稳、性能会退化。于是动态特征设计被夹在「充分」与「紧凑」之间：既要保住决策相关的关键动态，又得砍掉冗余、控住维度。

**本文目标**：构造一个既计算高效、又信息充分（近似马尔可夫）的状态表示，让现有 RL 主干（DQN/PPO/GCN+DQN）插上即用、收敛更快、规划更优。

**核心 idea（分层蒸馏全局动态）**：先用一个离线预训练的「最短路专家」做策略注意力，从全局图里挑出任务相关的稀疏子图（top-k 最短路覆盖的节点）；再在每个决策步用 n-hop 邻域把这个子图进一步收缩到当前节点的局部上下文。两级筛选把高维全局动态 $W_t$ 逐步精炼成低维、节点相关、且时序可预测的状态向量 $W''_t$。

## 方法详解

### 整体框架
DFR 建立在一个有限步、确定性 MDP $M=(S,A,T,R,\gamma,H)$ 上：状态 $s_t=\{v_t,v_g,f_t\}$ 由当前节点、目标节点和动态特征 $f_t$ 组成，动作是移动到当前节点的某个邻居，奖励为负通行成本加到达目标的常数奖励 $R(s_t,a_t,s_{t+1})=-c(v_t,v_{t+1};W_t)+b\cdot\mathbb{I}(v_{t+1}{=}{=}v_g)$。整套方法的核心就是把这个 $f_t$ 设计好。

DFR（Dynamics Feature Representation）把状态里的动态特征构造形式化成一条三级精炼链 $W_{:T} \xrightarrow{\tau,\Psi} W'_{:T} \xrightarrow{v_t,\Phi} W''_{:T}$：原始全局动态 $W_t$（整张图边权）先经任务级映射 $\Psi$ 压成任务相关子集 $W'_t$，再经节点级映射 $\Phi$ 收缩成当前节点的局部特征 $W''_t=f_t$。$\Psi$ 由策略注意力实例化、$\Phi$ 由 n-hop 邻域实例化；两者都只依赖固定的路网拓扑，可一次性离线预计算并复用，因此在线规划几乎不增加额外开销。

```mermaid
flowchart LR
    A["全局动态 W_t<br/>整张图边权, 高维冗余"] -->|"策略注意力 Ψ<br/>距离最优策略 π*_d<br/>取 top-k 最短路子图"| B["任务相关动态 W'_t<br/>稀疏任务子图 G'=(V',E')"]
    B -->|"n-hop 邻域 Φ<br/>当前节点 v_t 的 n 阶邻居 ∩ V'"| C["节点局部动态 W''_t = f_t<br/>低维状态向量"]
    C --> D["RL 智能体<br/>DQN / PPO / GCN+DQN"]
```

### 关键设计

**1. 策略注意力：用「最短路专家」把任务无关的动态先剪掉。** 全局边权 $W_t$ 维度高且大半与当前 source→goal 任务无关，DFR 先训一个只看距离的最优策略 $\pi^*_d$ 来当硬性、可预计算的注意力先验。具体做法是把 MDP 里的动态分量 $f_t$ 去掉、奖励改成 $R_d(s,a,s')=-d(s,s')$（$d$ 是路段长度），代回 Bellman 最优方程训出 $\pi^*_d$——这等价于一个 RL 版的静态最短路规划器。给定起点 $v_t$ 与终点 $v_g$，$\pi^*_d$ 给出按长度排序的 top-k 最短路，这些路覆盖的节点/边构成稀疏子图 $G'=(V',E')$，其边权即任务相关动态 $W'_t=\Psi(\tau,W_t)$，要求满足充分性 $\pi^*(v_t,v_g;W'_t)\approx\pi^*(v_t,v_g;W_t)$。之所以用「纯距离」而非多目标策略当注意力，是因为距离是 DPP 多目标里最基础的约束、且只要拓扑不变就不随时间变，所以预训练能一次性离线完成、提供稳定可解释的子图参考；参数 $k$ 控制完整性与紧凑性的权衡——$k$ 太小漏关键路、太大引冗余。

**2. n-hop 邻域：把任务子图再收缩到「脚下」的局部上下文。** 即便经过全局筛选，$W'_t$ 在大规模路网里仍可能偏高维。DFR 在每个决策步以当前节点 $v_t$ 为中心，取其 $n$ 阶以内邻居与策略注意力子图节点集的交集 $V_l(v_t)=\bigcup_{i=0}^{n} N^i(v_t)\cap V'$，这些节点间的边权就是局部动态特征 $f_t=W''_t(v_t)=\{w(v_i,v_j;t)\mid v_i,v_j\in V_l(v_t)\}$，并要求 $\pi^*(v_t,v_g;W''_t)\approx\pi^*(v_t,v_g;W'_t)$。$n$ 决定局部视野的空间尺度——小 $n$ 抓高度局部的动态但可能漏更宽的上下文，大 $n$ 覆盖广但维度和算力上涨。因为 n-hop 同样只依赖固定拓扑，可离线预计算，因此整条收缩不影响在线效率。

**3. PSR 理论支撑：保证压缩后状态仍近似马尔可夫。** 为什么两级砍完还能学到近优策略？本文借预测状态表示（Predictive State Representations, PSR）来论证：PSR 认为系统状态可由「给定动作序列下对未来可观测结果的预测」来定义，无需引入隐变量。在此视角下 $W''_t$ 是一个预测性表示——它编码了预测未来动作效果所需的充分信息，从而保证 $\pi^*(v_t,v_g;W''_t)\approx\pi^*(v_t,v_g;W_{:T})$。更进一步，精炼过程作用在序列结构 $W_{:T}$ 而非单帧快照上，通过过滤并聚合时序相邻的表示，DFR 隐式捕获了短时时序相关（如局部拥堵传播、车流连续性），让 $W''_t$ 同时保住空间与时间两侧的决策相关信息，对齐马尔可夫假设、稳住训练并缓解部分可观测下的次优问题。

## 实验关键数据

### 实验设置
- **数据**：OpenStreetMap 抽取的三个中国城市路网子图——南京、北京朝阳、上海浦东，建模为有向加权图，以「通行时间」为动态，最小通行时间路径为最优路径。边权由拥堵因子 $\beta\in[0.1,1.5]$ 调制，$c(v_i,v_j;W_t)=d(v_i,v_j)/(\nu_0\times\beta(v_i,v_j;t))$。
- **主干**：DQN（值）、PPO（策略梯度）、GCN+DQN（图特征）三种 RL 算法，各跑「带 DFR」vs「AD（All Dynamics，用全图动态）」两版对照。
- **指标**：Mean GAP（与动态 Dijkstra 真值路径的相对成本差，越低越好）、Success Rate（到达成功率，越高越好）、Compactness Rate（DFR 后维度 / 原维度，越低越好）、Planning Time（单次规划耗时，越低越好）。
- **训练配置**：Adam（lr $10^{-3}$）、$\gamma=0.99$、约 75,600 episodes（≈200 epoch）、replay buffer $10^6$、batch 32、每 episode 100 步定长，$\epsilon$ 从 1.0 线性衰减到 0.1；网络主体为 64 维嵌入 + 两层 64 维隐层的 MLP。

### 主实验
作者把每个模型的表现画成由 $1-\text{GAP}$、SR、$1-\text{CR}$ 三角形面积来汇总。结论是所有算法设置下，DFR 版三角形面积都明显大于 AD 版：

| 维度 | 现象 |
|------|------|
| 整体表现 | DFR 版三角形面积全面大于 AD 版 |
| GCN 类模型 | AD 下 SR 高但 GAP 也高（对动态不敏感）；加 DFR 后对动态变化的敏感度明显改善 |
| 规划耗时 | 加 DFR 后平均规划时间 DQN/PPO 为 $8.18\pm1.74$ ms、GCN+DQN 为 $27.26\pm6.8$ ms |
| 提速幅度 | 相比 DQN+AD / GCN+DQN+AD / PPO+AD 分别提速 **85.59% / 46.08% / 79.32%** |

### 消融实验
在南京子图上对 $k$（top-100 最短路的选取比例，$-1.0$ 表示关闭策略注意力）和 $n$（邻域阶数，$-1$ 表示无 hop 选择）做网格扫描：

| 配置 | Mean GAP | SR | 说明 |
|------|----------|----|----|
| baseline $(k{=}{-}1.0, n{=}{-}1)$ | 0.170 | 0.884 | DQN+AD，全动态无压缩 |
| $k{=}0.6, n{=}1$ | 0.151 | 0.723 | $n$ 太小，局部上下文不足 |
| $k{=}0.6, n{=}2$ | 0.118 | 0.867 | $n$ 增大后明显改善 |
| $k{=}0.6, n{=}4$ | 0.113 | 0.892 | 继续增 $n$ 收益递减、曲线趋于聚拢 |
| $k{=}0.4, n{=}4$ | 0.095 | 0.908 | CR 在所有 $n{=}4$ 配置下均 <5.7% |

### 关键发现
- DFR 在大幅压缩维度（CR 普遍 <5.7%）的同时，把 Mean GAP 从 0.170 降到 0.095、SR 从 0.884 升到 0.908，证明「策略注意力筛全局相关 + n-hop 抓局部依赖」二者互补。
- $n$ 的影响单调可预测——增大到「聚拢边界」后收益饱和；$k$ 的影响更复杂、非单调（$n{=}4$ 时 $k$ 从 0.4 升到 0.6 反而 GAP 升、SR 降）。因此大规模部署建议「适中 $k$ + 较小 $n$」，先把 $n$ 推到聚拢边界再调 $k$。
- 提速来自「在线只需在小的预计算子图上采集动态」，而非牺牲质量换速度——这点用「同时降 GAP/CR、升 SR」的结果坐实，说明结构表示学习与动态特征压缩是互补而非互斥的。

## 亮点与洞察
- **把「特征选择」当成可离线预计算的拓扑先验**：策略注意力和 n-hop 都只依赖固定路网拓扑，所以两级压缩完全离线、在线零额外开销，这是它能既省又快的关键工程洞察。
- **用静态最短路策略当「硬注意力」**：不同于 Transformer 的软注意力，$\pi^*_d$ 是预计算的硬注意力，提供强且可解释的先验，在 RL 智能体开学之前就把问题维度砍掉一大截，思想上接近知识蒸馏。
- **PSR 视角给压缩状态上了理论保险**：用预测状态表示论证 $W''_t$ 仍是充分统计量，把「为什么砍完还马尔可夫」说圆，而不只是经验调参。
- **即插即用**：DFR 是状态表示层的改造，对 DQN/PPO/GCN+DQN 都通用，不绑定特定 RL 算法。

## 局限与展望
- **$k$、$n$ 手工选**：两个核心超参靠人工网格扫，且 $k$ 影响非单调、调参困难，限制了真实大规模路网的开箱即用性；作者提出未来做自适应机制自动调 $k$、$n$。
- **实验规模偏小**：消融只在单个南京子图上做，三城子图也是「指定中心点+半径」抽的局部图，缺大城市全网级别的验证。
- **距离先验的局限**：策略注意力只用距离构子图，若最优时间路径严重偏离最短距离路径（如某些主干道长期拥堵导致绕行更优），top-k 最短路子图可能漏掉真正关键的边。
- **未对比传统/预测式规划器**：作者声明只研究 DFR 在 RL 范式内的增益，没和「预测+经典搜索」类方法正面比，难判断整体 SOTA 位置。

## 相关工作与启发
- **路径规划**：A\*、D\* Lite 等经典搜索依赖固定成本图，不适配高动态路网；学习增强法（先预测路况再搜索）受预测精度上限制约，对分布漂移和突发事件脆弱。DFR 走 RL 路线、免显式预测。
- **状态表示**：已有 RL-DPP 工作要么用局部视图牺牲全局最优、要么用全局视图算力爆炸；GNN 能编码整图但开销随规模增长。DFR 借鉴特征去相关（feature decorrelation）思路做分层精炼，是这条线的折中解。
- **注意力机制**：相比 Transformer 的软注意力，本文的策略注意力是基于任务结构语义的硬、预计算注意力，给 DPP 提供可解释的降维先验，对「如何把领域结构先验注入 RL 状态」有借鉴价值。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— 「最短路专家做硬注意力筛子图 + n-hop 收缩 + PSR 理论背书」的组合在 RL-DPP 状态表示问题上是清晰且有理论支撑的新解法，虽然各组件（注意力降维、n-hop、PSR）本身不新。
- **实验充分度**: ⭐⭐⭐ —— 三城真实路网 + 三主干对照 + $k/n$ 网格消融较完整，提速数字亮眼；但消融只在单子图、未与传统/预测式规划器正面比、缺大规模全网验证。
- **写作质量**: ⭐⭐⭐⭐ —— 问题动机、三级精炼形式化、PSR 论证层层递进，图示清晰；个别公式与符号（如 $t$ 在图论与 MDP 两处含义）需读者留意。
- **价值**: ⭐⭐⭐⭐ —— 即插即用、在线零开销、提速 46%–86%，对城市物流/即时配送/智能交通的实时动态规划有直接工程价值；自适应 $k/n$ 一旦补上落地潜力更大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GRACE: Generative Representation Learning via Contrastive Policy Optimization](grace_generative_representation_learning_via_contrastive_policy_optimization.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Adaptive Feature Fusion](offline_reinforcement_learning_with_adaptive_feature_fusion.md)
- [\[ICLR 2026\] On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting](on-policy_rl_meets_off-policy_experts_harmonizing_supervised_fine-tuning_and_rei.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[ICLR 2026\] 3D-aware Disentangled Representation for Compositional Reinforcement Learning](3d-aware_disentangled_representation_for_compositional_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
