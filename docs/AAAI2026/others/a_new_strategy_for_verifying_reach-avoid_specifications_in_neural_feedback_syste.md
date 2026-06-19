---
title: >-
  [论文解读] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems
description: >-
  [AAAI 2026][neural feedback systems] 提出FaBRe（Forward and Backward Reachability）策略，首次开发了针对ReLU神经网络控制器的后向可达集过近似和欠近似算法（GSS/ICH/LEB），并将其与前向可达性分析结合，构成统一的reach-avoid验证框架，旨在突破纯前向分析的可扩展性瓶颈。
tags:
  - "AAAI 2026"
  - "neural feedback systems"
  - "reach-avoid verification"
  - "backward reachability"
  - "forward reachability"
  - "safety verification"
---

# A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems

**会议**: AAAI 2026  
**arXiv**: [2601.08065](https://arxiv.org/abs/2601.08065)  
**代码**: 无  
**领域**: 形式化验证 / 安全关键系统  
**关键词**: neural feedback systems, reach-avoid verification, backward reachability, forward reachability, safety verification

## 一句话总结

提出FaBRe（Forward and Backward Reachability）策略，首次开发了针对ReLU神经网络控制器的后向可达集过近似和欠近似算法（GSS/ICH/LEB），并将其与前向可达性分析结合，构成统一的reach-avoid验证框架，旨在突破纯前向分析的可扩展性瓶颈。

## 研究背景与动机

**领域现状**：神经反馈系统——由神经网络控制的动力学系统——在机器人、自动驾驶、航空航天等安全关键领域日益广泛。验证这类系统是否满足安全规范（reach-avoid属性：进入目标区域且避开危险区域）是部署前的核心需求。

**现有痛点**：

1. 采样/仿射化方法虽然实用但无法提供形式化保证

2. 不变性分析需要构造Lyapunov/Barrier函数，实际中往往难以求解

3. 可达性分析是主流方法，但几乎完全依赖**前向分析**——通过神经网络的后向传播极为困难，导致现有后向方法可扩展性较差

4. 纯前向分析会在长时间步上累积过近似误差，导致验证精度下降

**核心矛盾**：前向分析的过近似在多步传播中逐步膨胀（维度灾难），而后向分析虽然从目标/避障集反向推导可以缓解这一问题，但缺乏可计算的高效算法。

**本文切入角度**：开发高效的后向可达集近似算法，并将前向和后向分析在时间轴上"拼接"——前 $F$ 步前向分析、后 $B$ 步后向分析——利用两者互补，在不增加过多计算开销的前提下提高验证精度。

## 方法详解

### 整体框架

给定离散时间神经反馈系统 $\mathcal{D} = \langle n, I, F, E, u, \delta, T, G, A \rangle$，其中 $u$ 为ReLU前馈神经网络控制器。FaBRe将 $T$ 步时间轴划分为前向 $F$ 步和后向 $B$ 步（$T = F + B$）：

- **Reach属性验证**：计算前向 $F$ 步的过近似可达集和后向 $B$ 步的欠近似可达集，若前者包含于后者则验证通过

- **Avoid属性验证**：分别计算前向和后向的过近似可达集，若对所有时间步两者不相交则验证通过

### 关键设计

1. **后向可达集过近似算法**

    - 将后向一步映射转化为约束优化问题：对每个状态维度分别求最小/最大值，利用已有的神经网络验证器求解
    - 对每个维度的上下界构成超矩形（hyperrectangle），保证包含真实后向可达集
    - 约束编码系统动力学 $x_i^{t+1} = x_i^t + (f_i(\mathbf{x}^t) + u(\mathbf{x}^t)_i + \epsilon_i) \cdot \delta$ 以及下一步状态必须落在目标集 $\mathcal{T}$ 内

2. **后向可达集欠近似算法（三种方案）**

    - **Golden Section Search (GSS)**：在过近似超矩形内，以 $\vec{c} \pm \rho \vec{r}$（$\rho \in (0,1)$）参数化欠近似，通过黄金分割搜索找最大的 $\rho$ 使前向像仍在目标集内。简单适用但需多次前向查询
    - **Iterative Convex Hull (ICH)**：密集采样过近似区域并前向传播，正样本构成候选欠近似超矩形，过近似查询验证后迭代缩小。用点查询代替昂贵的集合查询
    - **Largest Empty Box (LEB)**：针对高度非凸可达集，求解小规模MILP找排除所有负样本的最大空盒。适用于ICH中正样本分布复杂的情况

3. **FaBRe统一验证策略**

    - 时间轴分割 $F/B$ 作为可调参数，在前向和后向分析之间灵活权衡
    - 通过前后"拼接"减少单方向长步传播的累积误差
    - Reach验证和Avoid验证使用不同的拼接条件（包含 vs. 不相交）

### 损失函数 / 训练策略

本文为验证方法而非学习方法，不涉及损失函数。核心计算是：

- 过近似查询：调用NNV工具求解约束优化问题
- 欠近似查询：GSS使用黄金分割搜索，ICH使用采样+验证迭代，LEB转化为MILP求解
- 系统假设：控制器为全连接ReLU前馈网络，$n$ 个输入/$n$ 个输出

## 实验关键数据

### 主实验

本文为方法提案论文，"Ongoing Work"章节指出实验评估正在进行。计划比较的基线包括：

| 比较对象 | 比较方面 | 参考文献 |
|---------|---------|---------|
| Rober et al. | 后向过近似 | IEEE OJ-CS 2023 |
| BURNS (Sidrane & Tumova) | 后向欠近似 | arXiv 2025 |
| Akinwande et al. | 前向可达性（多面体封装） | arXiv 2025 |

### 理论分析

| 算法 | 保证类型 | 计算特点 | 适用场景 |
|------|---------|---------|---------|
| 过近似（约束优化） | sound overapprox. | 调用NNV，每维2次求解 | 所有情形的基础 |
| GSS | sound underapprox. | 需多次前向查询 | 凸/弱非凸可达集 |
| ICH | sound underapprox. | 采样+验证迭代 | 中等非凸度 |
| LEB | sound underapprox. | 求解小规模MILP | 高度非凸可达集 |

### 关键发现

- 前向和后向分析互补：前向从初始集扩展、后向从目标/避障集收缩，拼接可减少过近似膨胀
- 三种欠近似算法适用于不同的可达集几何形状，从简单（GSS）到复杂（LEB）递进
- 方法对ReLU前馈网络控制器通用，理论保证完整（soundness）

## 亮点与洞察

- 首次系统性地提出前向+后向联合验证策略，填补了后向分析缺乏高效算法的空白
- 三种欠近似算法（GSS/ICH/LEB）覆盖不同复杂度场景，设计思路从搜索→采样→MILP逐步强化
- $F/B$ 分割参数化提供了灵活的精度-效率权衡，具有工程实用性
- 理论框架清晰：前向/后向reach-avoid属性的四条性质定义严谨

## 局限与展望

- 目前缺少实验验证，所有算法的实际性能（精度、效率、可扩展性）仍待确认
- 仅支持ReLU前馈网络，不适用于RNN/Transformer等更复杂的控制器架构
- 超矩形欠近似可能对高度非凸可达集仍然松弛，多面体或联合体表示可能更精确
- 后向过近似每步需调用NNV求解器，计算开销可能随维度迅速增长
- 未讨论连续时间系统和随机扰动的扩展

## 相关工作与启发

- **vs 纯前向分析 (Akinwande et al. 2025)**：FaBRe通过后向分析减少前向长步传播的累积误差，是其自然扩展
- **vs Rober et al. 2023**：现有后向可达性方法可扩展性差，本文提出更高效的近似策略
- **vs BURNS (Sidrane & Tumova 2025)**：BURNS也关注后向欠近似，本文提供了三种不同复杂度的替代方案
- **验证领域启发**：前向+后向"夹逼"的思路可推广到其他验证问题（如概率安全、鲁棒性认证）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 前后向联合验证策略+三种欠近似算法的理论贡献扎实
- 实验充分度: ⭐⭐ 论文无实验数据，核心主张尚未经验证
- 写作质量: ⭐⭐⭐⭐ 数学定义严谨，但作为完整论文缺少实验部分
- 价值: ⭐⭐⭐⭐ 方向重要且理论框架完整，但需实验支撑才能评估实际影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](../../ICLR2026/others/a_new_approach_to_controlling_linear_dynamical_systems.md)
- [\[AAAI 2026\] Expressive Temporal Specifications for Reward Monitoring](expressive_temporal_specifications_for_reward_monitoring.md)
- [\[AAAI 2026\] Private Frequency Estimation via Residue Number Systems](private_frequency_estimation_via_residue_number_systems.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)

</div>

<!-- RELATED:END -->
