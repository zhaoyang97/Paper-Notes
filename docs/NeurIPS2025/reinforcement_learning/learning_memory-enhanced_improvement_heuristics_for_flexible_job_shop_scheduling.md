---
title: >-
  [论文解读] Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling
description: >-
  [NeurIPS 2025][强化学习][flexible job shop scheduling] 提出 MIStar——首个基于深度强化学习 (DRL) 的改进型启发式框架，用于求解柔性作业车间调度问题 (FJSP)。核心创新包括有向异构析取图表示、记忆增强异构图神经网络 (MHGNN) 和并行贪心搜索策略，在合成数据和公开 benchmark 上全面超越手工改进启发式和 SOTA 构造型 DRL 方法。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "flexible job shop scheduling"
  - "improvement heuristics"
  - "图神经网络"
  - "heterogeneous disjunctive graph"
  - "parallel greedy search"
---

# Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling

**会议**: NeurIPS 2025  
**arXiv**: [2603.02846](https://arxiv.org/abs/2603.02846)  
**代码**: 无  
**领域**: 组合优化 / 强化学习 / 调度  
**关键词**: flexible job shop scheduling, improvement heuristics, memory-enhanced GNN, heterogeneous disjunctive graph, parallel greedy search

## 一句话总结

提出 MIStar——首个基于深度强化学习 (DRL) 的改进型启发式框架，用于求解柔性作业车间调度问题 (FJSP)。核心创新包括有向异构析取图表示、记忆增强异构图神经网络 (MHGNN) 和并行贪心搜索策略，在合成数据和公开 benchmark 上全面超越手工改进启发式和 SOTA 构造型 DRL 方法。

## 研究背景与动机

**FJSP 是工业 4.0 核心调度问题**：柔性作业车间调度允许每个工序在多台兼容机器上加工，比经典 JSP 更贴近智能制造中的个性化、动态化生产需求，但一对多的工序-机器关系显著增加了求解复杂度。

**构造型 DRL 方法的固有缺陷**：现有 DRL 方法（如 HGNN、DANIEL）多采用构造策略，逐步将工序分配到机器上构建完整解。但部分解的析取图表示不完整（缺少未调度工序间的边），且难以显式编码加工进度信息（如当前机器负载、作业状态），导致状态表示失真，解的质量受限。

**改进型方法从完整解出发，信息无损**：改进型方法以完整调度解作为 MDP 状态，天然避免了部分解的信息缺失问题。析取图在完整解下能准确编码所有工序间的拓扑和顺序关系，更适合高性能调度场景。

**已有改进型 DRL 仅支持非柔性 JSP**：Zhang et al. 的 L2S 系列工作仅针对 JSP 设计，使用传统析取图（无机器节点）和 N5 邻域结构，无法处理 FJSP 中的机器分配调整。直接扩展面临三大挑战：(a) 状态表示需刻画复杂的工序-机器关系；(b) 动作空间需同时涵盖工序重排和机器重分配；(c) 柔性扩大了解空间，加剧局部最优陷阱风险。

**记忆机制可缓解局部最优**：利用历史搜索轨迹增强当前决策是突破局部最优的有效途径，但已有 MARCO 等方法仅用于简单二值优化问题，其信息聚合方式不适合 FJSP 复杂约束。

## 方法详解

### 整体框架 (MIStar)

给定 FJSP 实例，先用 DANIEL 采样生成初始解，将其转换为有向异构析取图，由 MHGNN 提取状态特征。策略网络从 Nopt2 邻域中采样多个候选动作，并行评估后执行最优动作更新当前解。迭代至预设搜索步数终止，保留全局最优解。

### MDP 建模

- **状态 $s_t$**：完整调度解，由每个工序 9 维特征向量和每台机器 4 维特征向量描述
- **动作 $a_t = [O_m, O_n, M_k]$**：将关键路径上的工序 $O_m$ 从当前机器移除，在机器 $M_k$ 的加工序列中插入到工序 $O_n$ 之前（基于 Nopt2 邻域结构）
- **奖励**：两项组成——$r_{gain} = \max(C_{\max}(s_t^*) - C_{\max}(s_{t+1}), 0)$ 衡量相对历史最优解的提升；$r_{penalty}$ 基于与历史状态的相似度惩罚冗余探索。总奖励 $r_t = r_{gain} - r_{penalty}$，早期以 makespan 改进为主导，后期逐渐转向鼓励多样性。

### 有向异构析取图

传统析取图仅有工序节点，无法捕捉 FJSP 中的机器状态。本文提出 $\overrightarrow{\mathcal{H}} = (\mathcal{O}, \mathcal{M}, \mathcal{C}, \mathcal{E})$：

- **工序节点 $\mathcal{O}$**：包含所有工序和虚拟起止节点
- **机器节点 $\mathcal{M}$**：显式建模每台机器的状态
- **有序弧 $\mathcal{C}$**：作业内工序的前后继约束
- **有向超弧 $\mathcal{E}$**：$E^k = (M_k, O^{k1}, O^{k2}, \ldots, O^{kn_k})$ 编码机器 $M_k$ 上工序的完整加工顺序。统一的有向边使不同调度解可被清晰区分

### 记忆增强异构图神经网络 (MHGNN)

分三部分提取特征：

**1. 工序节点嵌入**：双通道编码
- **拓扑通道 (GIN)**：用图同构网络编码图结构信息——$\mu_{O_{ij}}^l = \text{MLP}^l((1+\epsilon^l) \cdot \mu_{O_{ij}}^{l-1} + \sum_{U \in N(O_{ij})} \mu_U^{l-1})$，GIN 对非同构图有强判别力
- **语义通道 (GAT)**：用图注意力网络捕捉作业前驱和机器前驱邻居的差异化语义——$\tau_{O_{ij}}^l = \text{GAT}^l(\tau_{O_{ij}}^{l-1}, \{\tau_U^{l-1} | U \in N(O_{ij})\})$
- 两个通道的输出拼接得到工序嵌入 $h_{O_{ij}} \in \mathbb{R}^{2q}$

**2. 机器节点嵌入**：用异构 GAT 聚合机器上加工序列中各工序的信息，捕捉机器负载。异构 GAT 对不同类型节点使用独立的线性变换计算注意力系数。

**3. 历史动作嵌入（记忆模块）**：
- 每步存储状态-动作对 $(s_t, a_t)$，状态简化为工序-机器关联矩阵 $\bm{L}_t$（仅保留加工顺序，忽略节点属性）
- 通过 Frobenius 内积 $\omega_{t,t'} = \langle L_t, L_{t'} \rangle_F$ 计算当前状态与历史状态的相似度
- KNN 检索 $K$ 个最相似的历史动作，通过**软投票机制**分维度聚合——分别对 $O_m$、$O_n$、$M_k$ 三个维度按频率加权相似度投票选出最优候选
- 避免了直接加权平均离散索引导致的语义失效问题

### 决策与动作空间压缩

由于 FJSP 解中机器分配已确定（动作 $[O_m, O_n, M_k]$ 中 $O_n$ 必在 $M_k$ 上），第三维可省略，动作空间从 $O(|\mathcal{O}|^2 |\mathcal{M}|)$ 压缩到 $O(|\mathcal{O}|^2)$。策略网络输出工序评分矩阵，通过 Nopt2 邻域掩码不可行动作后 softmax 归一化得到动作概率分布。

### 并行贪心搜索策略

每步采样 $P$ 个候选动作并行评估 makespan 改进量，选择最优动作执行。单次迭代即探索 $P$ 个解，以更少迭代次数获得高质量解，显著减少搜索时间。训练使用 $n$-step PPO 算法。

## 实验关键数据

### 合成数据集

两个分布：SD1（紧凑分布，初始解接近最优）和 SD2（稀疏分布，优化空间大）。训练在 4 个小规模上进行，泛化到大规模。

| 规模 | 数据集 | DANIEL(S) | HGNN(S) | GD-400 | BI-400 | **MIStar-400** | OR-Tools |
|------|--------|-----------|---------|--------|--------|----------------|----------|
| 10×5 | SD2 | 11.96% | 46.94% | 10.40% | 7.18% | **4.98%** | 96%最优 |
| 20×10 | SD2 | 19.01% | 112.76% | 18.34% | 15.27% | **13.21%** | 1%最优 |
| 30×10 | SD2 | 9.28% | 109.95% | 8.87% | 6.82% | **5.27%** (泛化) | 0%最优 |
| 40×10 | SD2 | -4.53% | 94.11% | -5.08% | -6.81% | **-7.26%** (泛化) | 0%最优 |

→ MIStar 在所有规模上一致优于构造型 DRL 和手工改进启发式，运行时间仅为 BI/FI 的 1/3~1/5。

### 公开 Benchmark (Hurink & Brandimarte)

模型在 SD2 的 10×5 上训练，零样本迁移到各 benchmark：

| Benchmark | GD-400 Gap | BI-400 Gap | **MIStar-400 Gap** |
|-----------|------------|------------|--------------------|
| mk (Brandimarte) | 3.72% | 3.44% | **2.96%** |
| la (rdata) | 3.08% | 3.24% | **2.37%** |
| la (edata) | 6.96% | 6.94% | **6.83%** |
| la (vdata) | 0.53% | 0.64% | **0.24%** |

→ 在所有 benchmark 上 MIStar 均取得最小 gap，且运行时间稳定不随机器灵活度增加而膨胀。

### 大规模实例 (最大 1500 工序)

| 规模 | OR-Tools (1h) | MIStar-200 |
|------|---------------|------------|
| 50×15 | 39.05% gap, 60min | 44.60% gap, 10.6min |
| 100×10 | 62.22% gap, 60min | 66.03% gap, 15.9min |
| **50×30** | **不可行** | **有解**, 60min |

→ 在 50×30 超大规模上 OR-Tools 完全无法求解，MIStar 仍能给出可行解，展现强大的可扩展性。

### 消融实验

- 记忆模块 + 并行贪心搜索的组合效果最优
- 并行贪心搜索显著提升搜索效率，缓解早期收敛到局部最优的风险
- 记忆模块优化策略决策，提升解的质量
- 并行规模 $P$ 需在运行时间和解质量间权衡

## 亮点与洞察

- **首个针对 FJSP 的 DRL 改进型启发式框架**：填补了 DRL 改进方法从 JSP 到 FJSP 扩展的空白，且学到的策略具备规模无关的泛化能力
- **有向超弧精确编码机器加工序列**：相比传统析取图的无向边，有向超弧能清晰区分不同调度解，解决了 FJSP 状态表示的核心难题
- **记忆模块设计巧妙**：用工序-机器关联矩阵简化存储、Frobenius 内积高效计算相似度、软投票避免离散动作加权平均的语义失效，适配 FJSP 复杂约束
- **动作空间压缩**：利用 FJSP 约束将空间从 $O(|\mathcal{O}|^2|\mathcal{M}|)$ 降到 $O(|\mathcal{O}|^2)$，加速收敛
- **并行贪心策略兼顾效率和质量**：单次迭代探索 $P$ 个候选解，极大减少所需迭代次数，运行时间仅为规则方法的 1/3~1/5

## 局限与展望

1. **并行规模 $P$ 为超参数**：最优 $P$ 取决于实例特征，未来可设计自适应调整机制
2. **Nopt2 邻域的局部性**：仅在关键路径上做单工序移除-重插入，无法实现大范围重构，对初始解质量仍有一定依赖
3. **初始解依赖 DANIEL 采样**：初始解质量直接影响搜索起点，当初始解已接近最优时（如 SD1），改进空间有限
4. **记忆模块空间开销**：每步存储状态矩阵，长搜索轨迹下存储和检索成本增长
5. **仅优化 makespan**：未考虑多目标调度（如机器利用率、总延迟），实际生产场景需求更复杂
6. **大规模实例仍有较大 gap**：50×15 上 gap 达 44.6%，距离实际部署需求仍有距离

## 相关工作与启发

- **L2S (Zhang et al.)**：JSP 上的 DRL 改进方法先驱，MIStar 的直接前身，使用 N5 邻域 + 传统析取图，本文将其扩展到 FJSP 并全面升级
- **HGNN (Song et al.)**：异构析取图 + DRL 构造方法的代表，MIStar 借鉴了其异构图思想但改用有向超弧
- **DANIEL**：双注意力构造方法，本文用作初始解生成器和 baseline
- **MARCO**：记忆增强 RL 用于组合优化的先驱，但仅处理简单二值问题，本文的软投票聚合机制更适合 FJSP 约束
- **启发**：改进型方法 + 记忆机制 + 异构图表示的组合范式可推广到其他复杂调度/组合优化问题（如车间调度变体、车辆路径问题等）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将改进型 DRL 框架扩展到 FJSP，有向异构析取图 + 记忆增强 GNN 组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 合成 + benchmark + 大规模 + 消融全覆盖，对比方法丰富
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，技术细节充分，图表信息量大
- 价值: ⭐⭐⭐⭐ — 对 FJSP 求解和 DRL 改进型方法社区有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](../../ICLR2026/reinforcement_learning/shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](../../ICLR2026/reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)
- [\[ICLR 2026\] Recurrent Action Transformer with Memory](../../ICLR2026/reinforcement_learning/recurrent_action_transformer_with_memory.md)
- [\[ICML 2026\] Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning](../../ICML2026/reinforcement_learning/tracking_drift_variation-aware_entropy_scheduling_for_non-stationary_reinforceme.md)

</div>

<!-- RELATED:END -->
