---
title: >-
  [论文解读] MarS-FM: Generative Modeling of Molecular Dynamics via Markov State Models
description: >-
  [ICLR 2026][计算生物][Molecular Dynamics] 不再去学固定 lag time 的逐帧 MD 转移密度，而是先用马尔可夫态模型（MSM）把轨迹粗粒化成离散亚稳态，再用 Flow Matching 学"态到态"的跳转分布，从而以两个数量级的加速、更强的稀有大构象变化探索能力来替代分子动力学采样。
tags:
  - "ICLR 2026"
  - "计算生物"
  - "Molecular Dynamics"
  - "Markov State Model"
  - "Flow Matching"
  - "Protein Conformation"
  - "Generative Surrogate"
---

# MarS-FM: Generative Modeling of Molecular Dynamics via Markov State Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jP3HnYXoIp](https://openreview.net/forum?id=jP3HnYXoIp)  
**代码**: [https://github.com/valence-labs/mars-fm](https://github.com/valence-labs/mars-fm)  
**领域**: 计算生物学 / 分子动力学生成建模  
**关键词**: Molecular Dynamics, Markov State Model, Flow Matching, Protein Conformation, Generative Surrogate  

## 一句话总结
不再去学固定 lag time 的逐帧 MD 转移密度，而是先用马尔可夫态模型（MSM）把轨迹粗粒化成离散亚稳态，再用 Flow Matching 学"态到态"的跳转分布，从而以两个数量级的加速、更强的稀有大构象变化探索能力来替代分子动力学采样。

## 研究背景与动机
**领域现状**：分子动力学（MD）是研究蛋白质构象系综、揭示功能机制的"计算显微镜"，长轨迹能给出玻尔兹曼分布的采样。但生物事件常发生在毫秒尺度，而 Langevin 积分步长只有飞秒量级，使长模拟代价极其昂贵。为此，近年涌现了一批用生成流来替代 MD 的代理模型。

**现有痛点**：主流的 **MD Emulators（MD-Emus）** 学习固定 lag time $\tau$ 下的转移密度 $p_\tau(y|x)$——给定当前帧 $x(t)$ 生成未来帧 $x(t+\tau)$，推理时自回归拼成轨迹（如 MDGen 一次生成 $K$ 帧）。这套范式有结构性缺陷：lag 太短则加速有限，lag 太长则会跳过重要亚稳态；更根本的是 MD 轨迹本身存在**数据失衡**——绝大多数采样是"留在同一能量极小值内"的高频但无信息转移，而真正驱动探索的高能垒跨越（如折叠/去折叠）极其稀少。训练 batch 被无信息的态内转移淹没，模型难以学到稀有的大构象变化。

**核心矛盾**：固定 lag 的逐帧建模把生成目标和 MD 的时间动力学**强行绑定**，导致模型既被数据失衡限制，又因自回归而累积误差、且探索效率低。

**本文目标**：让生成模型从时间动力学中**解耦**，专注于学习亚稳态之间的宏观转移，在保证长时统计正确的同时大幅提升稀有构象探索效率与泛化性。

**核心 idea**（**MSM Emulators 新范式**）：MSM 把帧聚类成离散态、用马尔可夫链矩阵 $T$ 描述态间动力学，天然丢弃高频噪声并保证长时统计。作者由此提出一类全新生成模型 **MSM-Emus**：不学逐帧转移，而是学 MSM 诱导的态到态转移分布 $p_T(\cdot|x(t))$，并用 **MarS-FM（Markov Space Flow Matching）** 作为代表实例。

## 方法详解

### 整体框架
MarS-FM 把"采样代理 MD"重构成三步流水线：**离线对每个训练蛋白构建 MSM → 用 Flow Matching 学习 MSM 诱导的态到态转移分布 → 推理时用树状/混合采样并行探索能量地形**。关键在于训练目标从"逐帧转移密度 $p_\tau$"换成"态混合分布 $p_T$"：给定起始帧所属态 $S_i$，按 $T_{ij}$ 抽一个目标态 $S_j$，再从 $S_j$ 的 MD 帧集合里均匀抽一个目标帧——任意"落入相邻 MSM 态的帧对"都能成为训练样本，训练转移的数量和多样性不再受帧保存间隔约束。注意 MSM 只在训练期作为数据预处理，推理时不提供任何 MSM 信息。

```mermaid
flowchart TD
    A[每个训练蛋白的 MD 轨迹] --> B[降维: TICA 或 Rg/二级结构观测量]
    B --> C[k-means 微观态 + PCCA+ 聚成 10 个亚稳态]
    C --> D[在 lag time τ 下估计转移矩阵 T]
    D --> E[训练对: 帧 x∈S_i 与 目标帧 x1∈S_j, j~T_ij]
    E --> F[Flow Matching 学向量场 vθ: 噪声→p_T·|x_t]
    F --> G[推理: 树状采样 / MarS-FM⇒MDGen 混合]
    G --> H[并行生成代理构象系综, 计算 RMSD/Rg/二级结构等观测量]
```

### 关键设计

**1. MSM 构建：把高频时间信号压成离散亚稳态。** 对每个训练域单独离线构建一个 MSM，固定一套超参。先在低维碰撞变量空间定义态：要么用 **TICA**（时间滞后独立成分分析，找最大化 $w_j^\top x(t)$ 自相关的方向，保留累计动力学方差超 95% 的最少坐标），要么直接对回转半径、二级结构占比等物理观测量聚类。再 k-means 聚成 100 个微观态，用 **PCCA+** 谱聚类归并成 10 个亚稳态，最后在指定 lag time（四肽用 100 ps、MD-CATH 蛋白用 50 ns）下按式估计转移矩阵：$T_{ij} = C_{ij}/\sum_k C_{ik}$，其中 $C_{ij}=|\{x(t)\in S_i: x(t+\tau)\in S_j\}|$。诱导的转移密度满足 $\int_{S_j} p_T(y|x(t))\,dy = T_{ij}$，最简形式下态内密度均匀，即 $p_T$ 只依赖态身份而非具体构象。由于学的是"态间插值"而非"复现观测路径"，可以放心选更大的 $\tau$ 而不牺牲训练数据量。

**2. 态到态混合分布作为生成目标：从根上绕开数据失衡。** 对帧 $x(t)\in S_i$，MSM 定义了后继态上的类别分布 $j\mapsto T_{ij}$；作者把 $p_T(\cdot|x(t))$ 解读为一个**混合分布**——先按 $T_{ij}$ 采目标态 $S_j$，再从落入 $S_j$ 的经验 MD 帧系综里采构象。训练时**先在所有态 $S_i$ 上均匀采样**，再在态内条件于具体帧 $x(t)$，这保证稀有态被等概率遇到，而标准 MD-Emus 从轨迹均匀采帧必然偏向高频态内转移。因为每个亚稳态聚合了不同 replica、同一 basin 多次访问的帧，这种构造让模型见到跨 replica、跨多次能垒翻越的转移——即便单条轨迹里这类事件寥寥无几。

**3. Flow Matching 训练 + SE(3) 表示。** 每个残基用 SE(3) 表示 $T_\alpha(t)=(q_\alpha(t), r_\alpha(t), (\cos\chi_k,\sin\chi_k)_{k=1}^7)$（四元数旋转、平移、7 个扭转角），目标构象表示为相对输入的 roto-translation 偏移；网络 $v_\theta$ 沿用 MDGen 的 DiT 块并经 IPA 层条件于序列与当前构象。训练用标准 Flow Matching：源分布 $p_0=\mathcal{N}(0,1)$，目标 $p_1=p_T(\cdot|x(t))$，抽噪声 $x_0$ 与目标帧 $x_1$ 做插值 $x_s$，最小化向量场与条件路径速度的失配：
$$\mathcal{L}_{\text{MarS-FM}}(\theta)=\mathbb{E}\,\lVert v_\theta(s,x_s;x(t))-\dot{x}_s\rVert^2$$
与 MDGen 始终取同轨迹未来帧 $x(t+K\tau)$ 不同，MarS-FM 通过 MSM 转移核 $T$ 采 $x_1$，把同一 FM 目标作用在更多样、态条件化的训练对上。

**4. 推理：树状采样与混合方案。** MSM 信息不进推理，给定未见蛋白的序列 $a$ 与输入帧 $x(0)$，有两种策略：(i) **树状采样**——先并行生成 $n$ 帧 $\{y_i\}\sim p_T(\cdot|x(0))$，再对每个 $y_i$ 并行生成 $p_T(\cdot|y_i)$，按采样预算逐层加深；(ii) **MarS-FM⇒MDGen 混合**——先用 MarS-FM 采出分散的构象，再用 MDGen 从每个点生成短轨迹，对应"从不同态启动短模拟"的 MSM 思想，在需要时间保真度的工作流里补回局部动力学。两者绝大部分构象都能并行生成，既减少自回归调用、抑制累积误差，又因脱离时间动力学而更高效探索目标分布。

## 实验关键数据

### 主实验表格（MD-CATH，450 K 高温 replica，严格 20% 序列相似度过滤，495 个测试域；100/1000 构象）

| 方法 | Pairwise RMSD r ↑ | Per-target RMSF r ↑ | Rg KL ↓ | 二级结构 JSD ↓ | MSM JSD ↓ | ΔG_fold MAE ↓ |
|------|------|------|------|------|------|------|
| MD (Oracle) | 0.65 / 0.89 | 0.77 / 0.92 | 2.19 / 0.32 | 0.22 / 0.05 | 0.49 / 0.12 | 2.40 / 0.80 |
| MDGen-100 | 0.34 / 0.28 | 0.60 / 0.46 | 3.66 / 0.78 | 0.29 / 0.26 | 0.51 / 0.27 | 2.58 / 1.52 |
| MDGen-20 | 0.57 / 0.28 | 0.61 / 0.20 | 2.48 / 1.37 | 0.19 / 0.39 | 0.48 / 0.51 | 1.52 / 2.01 |
| BioEmu | 0.23 / 0.26 | 0.64 / 0.67 | 4.75 / 3.55 | 0.44 / 0.41 | 0.55 / 0.41 | 4.82 / 4.62 |
| **MarS-FM⇒MDGen-20** | 0.59 / 0.64 | 0.79 / 0.84 | 1.99 / 0.77 | 0.18 / 0.11 | 0.43 / 0.23 | 1.38 / 1.02 |
| **MarS-FM** | **0.60 / 0.65** | **0.84 / 0.90** | **1.74 / 0.42** | 0.18 / 0.14 | **0.42 / 0.17** | **1.25 / 1.20** |

MarS-FM 在几乎所有指标上大幅领先 MDGen 与 BioEmu，部分指标（如 per-target RMSF r 在 1000 构象下达 0.90）逼近 MD oracle。

### 消融实验表格（四肽，10^4 构象，JSD ↓）

| 方法 | Torsions(all) | TICA-0 | MSM states | Macrostate MAE ↓ |
|------|------|------|------|------|
| MD (Oracle) | 0.08 | 0.20 | 0.21 | — |
| MDGen-1000 | 0.11 | 0.23 | 0.23 | 1.13 |
| MDGen-200 | 0.12 | 0.24 | 0.27 | 1.12 |
| MarS-FM⇒MDGen-200 | 0.10 | 0.21 | 0.23 | 0.83 |
| **MarS-FM** | **0.10** | **0.21** | **0.22** | **0.63** |

四肽体系化学多样性低、无大构象运动，MD-Emus 与 MSM-Emus 本应接近；即便如此，MarS-FM 在 **Macrostate MAE 上几近腰斩**（1.12→0.63），证明它能更好地采到 TICA 空间中的稀有亚稳态。

### 关键发现
- **大构象探索**：图 5 显示 MarS-FM 并行生成的前 4 个样本二级结构含量差异显著，而 MDGen 样本全落在同一能量极小值；TICA 图（图 4）中 MarS-FM 探索到 MDGen 完全忽略的模式。
- **消融对照充分**：作者专门测了 MDGen-20/100、MDGen "in parallel"（只条件输入帧、去掉自回归）来证明 MarS-FM 的优势既不能靠改 lag time、也不能靠单纯减少自回归调用来复现。
- **加速**：相比隐式/显式溶剂 MD 采样，提速超过两个数量级。

## 亮点与洞察
- **范式级贡献**：提出 MSM-Emus 这一全新生成模型类别，把"学时间动力学"换成"学态间转移"，从根上化解了 MD 数据失衡——这是比单纯换架构更深的洞见。
- **复用成熟工具**：MSM/TICA/PCCA+ 都是分子动力学社区的标准武器，MarS-FM 把它们当作训练数据的离线预处理，几乎零额外推理开销，工程上很干净。
- **严格的泛化评测**：用 MMseqs2 最高灵敏度强制测试集与训练集序列相似度 ≤20%（比 BioEmu 的 40% 更苛刻），并在 450 K 高温下专门考察去折叠等大构象变化，评测协议扎实。
- **混合方案的互补性**：明确承认 MSM-Emus 不擅长复现细粒度态内时序，于是提供 MarS-FM⇒MDGen 把两类模型的长处缝合，体现了对方法边界的清醒认知。

## 局限与展望
- **不保真细粒度动力学**：按构造，态到态跳转优先长时热力学/动力学，牺牲了态内逐帧时序；纯 MarS-FM 不适合需要精确局部动力学的工作流，要靠混合方案补。
- **依赖 MSM 质量**：性能上限受离线 MSM 构建（态数、lag time、降维特征选择）影响，每个域单独建 MSM、超参需按数据集调，跨数据集迁移性未充分讨论。
- **态内均匀假设较粗**：最简形式假设态内密度均匀，对内部结构复杂的亚稳态可能损失精度。
- **评测仍以结构观测量为主**：匹配 RMSD/Rg/二级结构分布，但对动力学速率常数、路径机制等更细的动力学量验证有限。

## 相关工作与启发
- **MD-Emus 谱系**：Timewarp、Two-for-One、MDGen 等学固定 lag 转移密度，本文把它们统一归为受数据失衡所限的一类，MDGen 作主要对比基线。
- **Boltzmann 生成器**：Noé 等用 Normalizing Flow 直接采玻尔兹曼分布，本文转向"采 MSM 诱导分布"这一中间粒度。
- **BioEmu**：直接预测构象系综的大规模模型，作为大体系额外基线。
- **启发**：把"先粗粒化求不变/慢变结构、再在粗粒空间上做生成"的思路推广到其他时序/动力学建模问题（如视频、强化学习中的 option/技能发现），可能同样能绕开高频无信息样本主导训练信号的通病。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 提出 MSM-Emus 全新生成模型范式，把生成目标从逐帧转移密度切换到 MSM 诱导的态间转移，是范式级而非增量创新。
- **实验充分度**: ⭐⭐⭐⭐ MD-CATH 大规模数据 + 严格 20% 序列过滤 + 高温大构象评测 + 充分消融（不同 lag、并行变体、混合方案），仅细粒度动力学量验证略少。
- **写作质量**: ⭐⭐⭐⭐ 动机—痛点—范式转变的逻辑链清晰，图 1/4/5 直观对比，公式与采样流程交代到位。
- **价值**: ⭐⭐⭐⭐⭐ 两个数量级加速 + 显著超越现有方法 + 开源，对蛋白构象系综采样与药物发现有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport](wfr-fm_simulation-free_dynamic_unbalanced_optimal_transport.md)
- [\[ICLR 2026\] BioMD: All-atom Generative Model for Biomolecular Dynamics Simulation](biomd_all-atom_generative_model_for_biomolecular_dynamics_simulation.md)
- [\[ICLR 2026\] Multi-state Protein Sequence Design with DynamicMPNN](multi-state_protein_sequence_design_with_dynamicmpnn.md)
- [\[ICLR 2026\] h-MINT: Modeling Pocket-Ligand Binding with Hierarchical Molecular Interaction Network](h-mint_modeling_pocket-ligand_binding_with_hierarchical_molecular_interaction_ne.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](../../NeurIPS2025/computational_biology/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)

</div>

<!-- RELATED:END -->
