---
title: >-
  [论文解读] Inter-Agent Relative Representations for Multi-Agent Option Discovery
description: >-
  [ICLR 2026][强化学习][多智能体 RL] 本文提出一种**面向智能体间相对关系**的联合状态抽象：先估计一个让全队对齐成本最小的"Fermat 状态"，再以各智能体到它的逐维时序距离作为新的状态表示，在此之上做图拉普拉斯 eigenoption 分解，从而发现数量更少、协调性更强的多智能体联合 option。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "多智能体 RL"
  - "option discovery"
  - "联合状态抽象"
  - "Fermat 状态"
  - "图拉普拉斯 eigenoption"
  - "时序距离"
---

# Inter-Agent Relative Representations for Multi-Agent Option Discovery

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Fte7TOqnQp](https://openreview.net/forum?id=Fte7TOqnQp)  
**代码**: 待确认  
**领域**: 强化学习 / 多智能体 option 发现  
**关键词**: 多智能体 RL, option discovery, 联合状态抽象, Fermat 状态, 图拉普拉斯 eigenoption, 时序距离  

## 一句话总结
本文提出一种**面向智能体间相对关系**的联合状态抽象：先估计一个让全队对齐成本最小的"Fermat 状态"，再以各智能体到它的逐维时序距离作为新的状态表示，在此之上做图拉普拉斯 eigenoption 分解，从而发现数量更少、协调性更强的多智能体联合 option。

## 研究背景与动机
**领域现状**：在单智能体 RL 中，option 框架（Sutton et al., 1999）用时序扩展动作（temporally extended actions）充当状态空间中远距离区域之间的"捷径"，能显著改善探索与规划。其中基于图拉普拉斯特征分解的 Eigenoption（Machado et al., 2017）因任务无关、带探索保证而广受欢迎。

**现有痛点**：把这套搬到多智能体场景有两道硬伤。一是**联合状态空间随智能体数指数增长**，而 eigenoption 发现的 option 数约为状态数的两倍，于是 option 数量爆炸；二是当前的拉普拉斯特征向量神经逼近器只在估计**少数几个**特征向量时才稳定，容易漏掉真正有用、能在不同时间尺度上促进探索的 option。

**核心矛盾**：已有的多智能体 option 发现方法（如基于 Kronecker 图积的 Chen et al., 2022）为了绕开指数爆炸，往往把联合行为拆成**松耦合或完全独立**的单体行为——协调只发生在"选哪个 option"这一层，option 策略本身并不表达协作。这等于用牺牲协调性换取可计算性，强协调行为的发现问题始终悬而未决。

**本文目标**：在压缩联合状态空间的同时，把发现过程**聚焦到智能体之间关系如何变化**，从而既控制 option 数量，又能涌现出强协调的联合行为。

**核心 idea**：作者押注一个归纳偏置——**在没有显式目标时，智能体状态之间的"同步/对齐"本身就是协调的天然基础**（如球类运动中的传球、站位都可理解为多智能体状态同步）。据此他们用一个"全队最大对齐点"重新定义状态表示，让特征向量天然指向智能体间关系而非绝对位置。

## 方法详解

### 整体框架
方法把"如何表示联合状态"这一被以往工作忽视的环节当作发力点：在做图拉普拉斯特征分解**之前**，先把原始联合状态嵌入到一个**智能体间相对表示**里。整条管线分三步——先用 Fermat 编码器估出全队对齐点并算出逐维时序距离表示，再在这个相对表示上跑 eigenoption 发现得到联合 option，最后把联合 option 接进 MacDec-POMDP 框架以支持去中心化执行。

```mermaid
flowchart LR
    A[联合状态 s_t<br/>因子化为各智能体状态 s_t^i] --> B[Fermat 编码器 φ<br/>估计最大对齐点 φ(s_t)]
    B --> C[时序距离 d_θ<br/>逐特征维输出 n-distance]
    C --> D[CMI 解耦<br/>判别器 D_ψ 防退化]
    D --> E[多维相对表示<br/>重构状态转移图]
    E --> F[ALLO 图拉普拉斯<br/>特征向量 μ]
    F --> G[特征向量作内在奖励<br/>训练联合 option 策略]
    G --> H[MacDec-POMDP<br/>投票发起/同步/终止]
```

### 关键设计
**1. Fermat 状态：用"全队对齐成本最小点"重新锚定联合状态。** 作者借鉴 Fermat n-distance（一组点到某点距离之和最小）的几何思想，把联合状态的"离散程度"（spreadingness）定义为各智能体状态到一个虚拟对齐点的距离之和。对一个单体状态空间 $(S^*, d)$ 与 $N$ 个智能体，Fermat 智能体间距离为 $d_F(s^1,\dots,s^N)=\min_{s\in S^*}\sum_{i=1}^N d(s^i, s)$。这个最小化在大/连续空间里不可解，于是用一个参数化的 **Fermat 编码器** $\phi: S\to S^*$ 去逼近这个对齐点，训练目标是让它到各智能体状态的平方距离之和最小：$L_F(\phi,d)=\mathbb{E}_{\tau\sim\rho_\pi}\big[\tfrac{1}{N}\sum_{i=1}^N d(s_t^i,\phi(s_t))^2\big]$。状态以这个 Fermat 点为中心重新表达后，特征向量对联合状态的变化变得**高度敏感**（re-centring 效应），这正是从原始联合状态上做分解所缺的性质。

**2. 时序距离而非欧氏距离作为底层度量。** 上式里的 $d$ 用什么很关键。作者选用**时序/后继距离**（successor distances，Myers et al., 2024），即两状态间策略所需的期望步数，因为它对特征语义不变、且贴合环境动力学（绕开障碍等欧氏距离感知不到的结构）。时序距离本质是放松对称性的**拟度量**（上山比下山慢），强行对称化会损失表达力，所以作者把 Fermat 状态固定为函数第二个输入，得到一个有向度量，可解读为"全队达到完全对齐所需的期望步数"。Fermat 编码器 $\phi$ 与距离逼近器 $d_\theta$ 联合训练，并在 $\phi$ 的目标里对 $d_\theta$ 的参数施加 stop-gradient。

**3. 逐特征维的多维 n-distance + 互信息解耦，防止表示坍缩。** 若把联合状态压成**单个标量**距离，会抹掉"两个智能体到底在哪一维上没对齐"的信息，并严重扭曲状态转移图拓扑，限制可发现的对齐行为多样性。作者改让距离模块在单体状态的**每一维**上各输出一个距离，$d_F:S^*\times S^*\to\mathbb{R}^F$（$F=\dim(S^*)$），再用线性投影层重建整体距离。但无约束的逐维分解会退化（所有维输出相同、或只用一维），于是引入**条件互信息（CMI）约束**：要求每个特征距离 $Z_f^{i,j}$ 携带的关于其它特征 $S_{-f}^{i,j}$ 的信息，不超过特征本身已蕴含的相关性，即最小化 $I(S_{-f}^{i,j}; Z_f^{i,j}\mid S_f^{i,j})$。实现上沿用 Dunion et al. (2023) 的判别器网络 $D_\psi$，训练它区分真实三元组与打乱 $s_{-f}$ 后的假三元组，再用判别结果作解耦惩罚——逼每一维的距离预测"各管各的特征"。

**4. 联合 option 接入 MacDec-POMDP：投票发起、同步、共识终止。** 拿到相对表示后，作者沿用 ROD 循环 + ALLO 逼近器做特征分解，把特征向量当内在奖励 $r_e(s,s')=e(s')-e(s)$ 训练 option 策略。为在去中心化下正确执行联合 option，他们扩展 MacDec-POMDP 并加两条假设：智能体间存在**信息共享机制**、以及保证最少参与数的**同步机制**。联合 option 定义为定义在联合宏历史上的元组 $W=\langle I_W,\pi_W,\beta_W\rangle$；每个智能体选某 option 视为一次"投票"，联合 option 需 $N$ 票（全队共识）才发起、局部 option 仅需 1 票，原始动作被当作立即终止的局部 option。终止需全体选择终止动作，并设 50 步硬停。

## 实验关键数据

### 实验设置
两个多智能体域：**Level-Based Foraging (LBF)** 与 **Overcooked**（均用 JAX 重实现）。LBF 选 `15x15-4p-3f` / `15x15-4p-5f` 并把每个苹果等级设为全队等级之和（强制合作配置）；Overcooked 选 `Forced Coordination` / `Counter Circuit`。用 50 万条随机策略转移训练 n-distance 编码器与 ALLO；LBF 估前 10 个特征向量（20 个 option）、Overcooked 估 20 个（40 个 option）；option 策略用 IQL 训 100 万步（占总训练 5%/10%）。

### 主实验：联合 option 是否有用（H1）+ 对比其它 option 框架（H2）
报告 10 个 seed 的 IQM 与 95% 置信区间。

| 对比维度 | 结论 |
|---|---|
| IQL+IARO vs 无 option 的 IQL | 一致提升：LBF 吃苹果比例更高、Overcooked 每回合成功递送更多 |
| IQL+IARO vs MAPPO / IPPO / IQL / VDN | 多数场景领先；仅 Forced Coordination 不及 VDN（该场景 VDN 本就强） |
| IQL+IARO vs IQL+Kron（Chen 2022 Kronecker 图积） | IARO 更优；Kron 在 LBF 甚至**拖累**性能 |
| IQL+IARO vs IQL+RJS（直接在原始联合状态上发现） | IARO 更优；RJS 首批特征向量把智能体推向状态空间边缘，对采苹果反而有害 |

作者指出 IARO **不依赖中心化（MAPPO）或值分解（VDN）**即可促成协作，但在训练初期收敛略慢（全局 initiation set 下用 option 训练的已知难题）。

### 消融：多维 vs 标量 n-distance（H3）

| 表示 | LBF（简单状态） | Overcooked（多语义特征） |
|---|---|---|
| IARO-Scalar | 与多维相当 | 较弱 |
| IARO-MultiDim | 与标量相当 | **更优** |

结论：状态特征语义越丰富，逐维解耦表示越能让智能体在**特定子集维度**上对齐，产生更丰富的协作行为。

### option 数量分析
最复杂场景上扫不同 option 数（15 seed、64 评估回合）：LBF 仅用 **2 个** option 就有最大涨幅，Overcooked 需前 **10 个** option 才见明显提升——与 eigenoption 理论一致（前几个特征向量连接图中远端节点、给出强探索行为）。option 过多则收益饱和甚至带来训练不稳定与方差升高。

### 关键发现
- 围绕 Fermat 状态 re-centring 后，特征向量对联合状态变化高度敏感，且**同一组前几个特征向量对 3 个 / 4 个智能体诱导出一致的对齐模式**，说明行为模式对队伍规模有迁移性。
- 第一个特征向量沿某一坐标轴对齐、其负向沿另一轴；后续特征向量促成多轴同时对齐及更复杂同步模式。
- 在 15×15 三智能体网格的可视化中，相对表示（右）相比单体状态（左）和原始联合状态（中）给出的特征向量更能反映"智能体之间相对位置"的变化，而非把智能体推向状态空间边缘。
- Overcooked 中 IQL 容易卡在次优解，而联合 option 让智能体能系统性地扫过各种协调模式去搜寻更优策略，这是 IARO 在该域提升尤其明显的原因。

## 亮点与洞察
- **把"状态表示"当成 option 发现的可控旋钮**：作者敏锐地指出 eigenoption 完全受状态表示支配，于是不去改分解算法，而是在分解前换一套"以智能体间关系为中心"的表示，四两拨千斤地同时解决了 option 数量爆炸与协调性缺失。
- **Fermat 状态是个优雅的统一锚点**：用"全队对齐成本最小点"把 $N$ 个智能体的关系压成一个可学习的中心，既压缩维度又保留"谁和谁、在哪维上没对齐"的关键信息。
- **逐维分解 + CMI 解耦**把"对齐"从一个标量升级成可组合的行为字典，让上层策略能挑选在哪些维度上协调，行为表达力明显更强。
- 用时序距离替代欧氏距离，使度量对特征语义不变、贴合动力学，是把单智能体技能发现经验（METRA 等）迁到多智能体的关键一步。

## 局限与展望
- **全队共识假设偏强**：联合 option 必须全体同意才发起，限制了可表达行为的范围；作者承认发现方法本身支持任意智能体子集的行为，但为简化只做了 team-level option，子集 option 留作未来工作。
- **依赖信息共享 / 同步机制**：方法假设存在智能体间信息共享与同步机制（Overcooked 中隐式、LBF 中显式给出队友相对距离与"靠近苹果"标志），尚未上升到一般的通信协议；作者计划用通信协议替代直接共享观测。
- **训练初期收敛慢**：全局 initiation set 下用 option 训练会拖慢早期收敛，这是 option 学习的老问题。
- **评测域规模有限**：仅在 LBF 与 Overcooked 两个域、4 智能体规模上验证，向更大队伍、更复杂域的可扩展性仍待检验；作者还计划研究更丰富的协调行为拓扑。

## 相关工作与启发
- **状态相似度度量**：双模拟度量（Ferns et al., 2004）按奖励/转移差异衡量相似度，但在稀疏奖励下易表示坍缩；时序/后继距离对表示不变、贴合动力学，在目标条件 RL、内在奖励、技能发现中流行。与 METRA（Park et al., 2024）这类单智能体技能发现最接近，但本文是用时序距离逼近**联合 option** eigenoption 发现的潜在表示。
- **MARL 中的状态表示**：以往多用 GNN 把局部观测聚合成全局表示；Utke et al. (2025) 强调相对信息但靠手工空间关系嵌入边特征——本文则**自动学习**智能体间关系，不限制特征语义。
- **多智能体时序扩展动作**：异步方案（Amato et al., 2019 等）多用单体局部 option，协调只在选 option 层；Chen et al. (2022) 用 Kronecker 图积构造联合行为，但仍主要同步独立行为、抓不住强智能体间依赖——本文正是冲着"发现强协调行为"这个未解问题去的。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ — "把联合状态重表示为以 Fermat 对齐点为中心的逐维时序距离，再做 eigenoption 分解"是一个少见且自洽的切入点，把状态表示、时序距离、互信息解耦、option 框架几条线索缝合得很完整。
- **实验充分度**: ⭐⭐⭐⭐ — 三个假设（H1/H2/H3）逐一验证，覆盖无 option、其它 option 框架、标量 vs 多维、option 数量四组对比，10–15 seed 带置信区间；但仅 2 个域、4 智能体，规模偏小。
- **写作质量**: ⭐⭐⭐⭐ — 动机与归纳偏置（球类比喻）讲得清楚，方法推导（Fermat 距离、CMI 上界 Proposition 1）严谨，配有特征向量与 option roll-out 可视化；细节较多需对照附录。
- **价值**: ⭐⭐⭐⭐ — 给"如何发现强协调多智能体 option"提供了一条不依赖中心化/值分解的新路径，Fermat 状态 + 逐维解耦表示的思路对 MARL 表示学习与技能发现有较好的可迁移启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Multi-Agent Guided Policy Optimization](multi-agent_guided_policy_optimization.md)
- [\[ICLR 2026\] Correlated Policy Optimization in Multi-Agent Subteams](correlated_policy_optimization_in_multi-agent_subteams.md)
- [\[ICLR 2026\] Ego-Foresight: Self-supervised Learning of Agent-Aware Representations for Improved RL](ego-foresight_self-supervised_learning_of_agent-aware_representations_for_improv.md)
- [\[ICLR 2026\] When Is Diversity Rewarded in Cooperative Multi-Agent Learning?](when_is_diversity_rewarded_in_cooperative_multi-agent_learning.md)
- [\[ICLR 2026\] Optimal Robust Subsidy Policies for Irrational Agent in Principal-Agent MDPs](optimal_robust_subsidy_policies_for_irrational_agent_in_principal-agent_mdps.md)

</div>

<!-- RELATED:END -->
