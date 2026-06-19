---
title: >-
  [论文解读] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs
description: >-
  [AAAI 2026][3D视觉][旅行商问题] 本文针对非度量图上的旅行商问题（TSP），提出了关于参数 $p$（违反三角不等式的顶点数）和 $q$（最小违反集大小）的改进FPT近似算法，将 $p$ 参数下的近似比从2.5改进到1.5，$q$ 参数下从11改进到3。 旅行商问题（TSP）是组合优化中的经典NP-hard问题…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "旅行商问题"
  - "参数化近似算法"
  - "非度量图"
  - "固定参数可解"
  - "三角不等式"
---

# Parameterized Approximation Algorithms for TSP on Non-Metric Graphs

**会议**: AAAI 2026  
**arXiv**: [2503.03642](https://arxiv.org/abs/2503.03642)  
**代码**: 无  
**领域**: 理论计算机科学 / 组合优化  
**关键词**: 旅行商问题, 参数化近似算法, 非度量图, 固定参数可解, 三角不等式

## 一句话总结

本文针对非度量图上的旅行商问题（TSP），提出了关于参数 $p$（违反三角不等式的顶点数）和 $q$（最小违反集大小）的改进FPT近似算法，将 $p$ 参数下的近似比从2.5改进到1.5，$q$ 参数下从11改进到3。

## 研究背景与动机

旅行商问题（TSP）是组合优化中的经典NP-hard问题，在物流、制造、电信等领域有广泛应用。对于满足三角不等式的度量图（metric graph），经典的Christofides-Serdyukov算法可以达到1.5的近似比，Karlin等人进一步将其改进到 $1.5 - 10^{-36}$。然而，对于一般图（非度量图），TSP甚至无法在多项式时间内被任何可计算函数 $f(n)$ 近似。

度量图和一般图之间存在巨大的近似性能差距，这促使研究者思考：对于"接近"度量的图，能否获得更好的近似算法？近年来，参数化复杂度领域引入了两个自然参数来衡量图离度量图的"距离"：

- **参数 $p$**：图中参与违反三角不等式的三角形的顶点数
- **参数 $q$**：使图成为度量图所需删除的最少顶点数（最小违反集）

这两个参数的实际动机来自城市旅游巴士路线规划：大多数地标之间满足三角不等式，但少数高知名度地标间的快速直达路线会破坏三角不等式。这些违反通常只涉及少量地标，即 $p$ 和 $q$ 较小。

此前的工作中，Zhou等人给出了 $p$ 参数下的FPT 3-近似算法，Bampis等人改进为2.5-近似；而 $q$ 参数下，Zhou等人的最佳FPT结果为11-近似。本文的核心动机正是回答Bampis等人提出的开放问题：这些近似比是否还能进一步改进？

## 方法详解

### 整体框架

本文针对两个参数分别提出了多个算法，核心思路是：将非度量TSP分解为在"坏顶点"（bad vertices，参与违反三角不等式的顶点）和"好顶点"（good vertices）上分别处理的子问题，利用好顶点的度量性质进行shortcutting操作来降低解的代价。

**关键性质（Property 1）**：任何包含好顶点的三角形都满足三角不等式。这是所有算法设计的基础——通过好顶点进行shortcut操作不会增加路径权重。

### 关键设计

1. **ALG.1（$(α+1)$-近似，参数 $p$）**：最简单的算法。选择一个好顶点 $o$，用动态规划在 $G[V_b \cup \{o\}]$ 上求最优TSP tour $T_b$，再用度量TSP的 $\alpha$-近似算法在 $G[V_g]$ 上求 $T_g$，最后对 $T_b \cup T_g$ 做shortcut得到完整TSP tour。关键观察是 $\text{OPT} \geq w(T_b^*)$ 且 $\text{OPT} \geq w(T_g^*)$，因此总近似比为 $(\alpha + 1)$。运行时间为 $2^{O(p)} + n^{O(1)}$，同时改进了近似比和运行时间。

2. **ALG.2（1.5-近似，参数 $p$）**：更精细的算法。首先猜测最优tour $T^*$ 在坏顶点上的子图（即"坏链" $\mathcal{A}$），再构造约束生成树（CST）$F_{\mathcal{A}}$，然后通过辅助图上的最小权匹配来修正奇度顶点。核心创新在于：

    - 通过收缩坏链构造辅助图 $\widetilde{G}$，在其上找最小生成树
    - 构造辅助图 $G'$ 来处理奇度顶点的匹配问题（因为原图可能非度量，不能直接用最小权匹配）
    - 证明 $w(F_{\mathcal{A}}) \leq \text{OPT}$ 且 $w(\mathcal{M}_{\mathcal{A}}) \leq \frac{1}{2} \text{OPT}$

   通过精巧的shortcutting引理（Lemma 3），最终得到 $\frac{3}{2} \cdot \text{OPT}$ 的近似比。

3. **ALG.3（$(α+\varepsilon)$-近似，参数 $p$ 为常数时）**：将TSP归约到度量 $k$-TSPP（$k$-TSP路径问题），利用Traub等人的 $\Phi$-TSP算法。当 $p = O(1)$ 时，近似比几乎等同于度量TSP。运行时间为 $n^{O(p/\varepsilon)}$。

4. **ALG.4（3-近似，参数 $q$）**：参数 $q$ 下的核心算法。面临的挑战是：与 $p$ 参数不同，包含一个坏顶点和两个好顶点的三角形也可能违反三角不等式，因此Property 1不适用。算法包含三个子过程：

    - **LIMB**：通过势集（potential set）技术猜测锚点（anchors）和肢边（limbs），使得 $w(\mathcal{B}') \leq w(\mathcal{B})$
    - **CONNECT**：通过猜测分区将断开的组件连接起来，复杂度控制在 $2^{O(q \log q)}$
    - **SHORTCUT**：处理Euler图上的shortcut操作

### 损失函数 / 训练策略

本文为纯理论工作，不涉及训练。核心分析技术包括：

- **Shortcutting分析**：利用Property 1证明通过好顶点shortcut不增加权重
- **匹配下界**：将TSP tour分解为两个匹配，证明最小匹配权重不超过 $\frac{1}{2} \text{OPT}$
- **势集技术**：在LIMB子算法中，为每个锚点维护大小为 $O(q)$ 的候选集，确保猜测的锚点代价不超过真实锚点

## 实验关键数据

### 主实验

本文为纯理论工作，无实验数据，但给出了完整的理论结果对比：

| 参数 | 近似比 | 运行时间 | 来源 |
|------|--------|----------|------|
| $p$ | 3 | $2^{O(p \log p)} \cdot n^{O(1)}$ | Zhou et al. |
| $p$ | 2.5 | $2^{O(p \log p)} \cdot n^{O(1)}$ | Bampis et al. |
| $p$ | **$\alpha+1 \approx 2.5$** | $2^{O(p)} + n^{O(1)}$ | **本文 ALG.1** |
| $p$ | **1.5** | $2^{O(p \log p)} \cdot n^{O(1)}$ | **本文 ALG.2** |
| $p$ | **$\alpha + \varepsilon$** | $n^{O(p/\varepsilon)}$ | **本文 ALG.3** |
| $q$ | 3 | $n^{O(q+1)}$ | Zhou et al. |
| $q$ | 11 | $2^{O(q \log q)} \cdot n^{O(1)}$ | Zhou et al. |
| $q$ | **3** | $2^{O(q \log q)} \cdot n^{O(1)}$ | **本文 ALG.4** |
| $q$ | **$\alpha + \varepsilon$** | $n^{O(q/\varepsilon)}$ | **本文** |

### 消融实验

| 算法变体 | 近似比 | 关键改进 |
|----------|--------|----------|
| ALG.1 单独 | $\alpha + 1$ | 最简单，运行时间最优 $2^{O(p)} + n^{O(1)}$ |
| ALG.2 使用CST+匹配 | 1.5 | 避免了额外边集 $E'$，直接构造约束生成树 |
| ALG.4 vs 前人 $q$ 参数 | 3 vs 11 | LIMB+CONNECT技术实现FPT时间内3-近似 |

### 关键发现

- 对于参数 $p$，1.5-近似比已达到度量TSP的最佳已知近似比，说明本文算法在该参数化下几乎最优
- 当 $p$ 或 $q$ 为常数时，$(α+\varepsilon)$-近似比几乎匹配度量TSP的最优近似比
- ALG.4将 $q$ 参数下的FPT近似比从11大幅改进到3，同时保持相同的FPT运行时间

## 亮点与洞察

1. **简洁而有效的分治思想**：ALG.1用一个好顶点连接坏顶点子tour和好顶点子tour，极其简洁却已能改进此前最佳结果
2. **约束生成树（CST）概念**：ALG.2引入的CST避免了前人算法中需要额外边集的瓶颈，是关键技术创新
3. **势集技术**：在ALG.4的LIMB子算法中，通过维护 $O(q)$ 大小的候选集来猜测锚点，巧妙地在FPT时间内完成了搜索
4. **归约到 $\Phi$-TSP**：ALG.3展示了参数化TSP与最新的度量TSP算法之间的优雅联系
5. **统一框架**：本文技术对两个参数 $p$ 和 $q$ 都适用，且可能推广到其他相关问题

## 局限与展望

- 对于参数 $q$，FPT框架下能否将近似比从3进一步改进到1.5？这是一个重要的开放问题
- $q$ 的计算本身是NP-hard的，需要 $O(3^q n^3)$ 的时间，这在 $q$ 较大时可能成为瓶颈
- ALG.3的运行时间 $n^{O(p/\varepsilon)}$ 不是FPT的（参数出现在多项式指数中），仅当 $p$ 为常数时有意义
- 所有算法都是组合算法，缺乏实验评估来验证实际性能
- 论文未讨论这些参数化结果在实际应用中的典型参数范围

## 相关工作与启发

本文处于参数化复杂度与近似算法的交叉领域。关键相关工作包括：

- **$\tau$-三角不等式放松**：Andreae-Bandelt、Bender-Chekuri、Mömke等人的系列工作，从不同角度放松度量条件
- **子群路径规划（Subgroup Path Planning）**：在AI中有机器人抛光等应用，与本文的参数化方向互补
- **$\Phi$-TSP框架**：Traub等人的通用框架，本文ALG.3直接利用了这一成果

对后续研究的启发：本文提出的CST和势集技术可能对其他基于图结构的参数化问题有价值，例如参数化的最小权Hamiltonian路径、参数化的车辆路径问题等。

## 评分

- **创新性**: ⭐⭐⭐⭐ — 多个算法各有巧妙设计，近似比改进幅度显著（11→3）
- **理论深度**: ⭐⭐⭐⭐⭐ — 证明严谨完整，技术含量高
- **实用性**: ⭐⭐⭐ — 理论贡献突出，但缺乏实验验证
- **表达清晰度**: ⭐⭐⭐⭐ — 结构清晰，符号定义明确
- **综合评分**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning to Infer Parameterized Representations of Plants from 3D Scans](../../CVPR2026/3d_vision/learning_to_infer_parameterized_representations_of_plants_from_3d_scans.md)
- [\[NeurIPS 2025\] Fully Dynamic Algorithms for Chamfer Distance](../../NeurIPS2025/3d_vision/fully_dynamic_algorithms_for_chamfer_distance.md)
- [\[CVPR 2026\] The Midas Touch for Metric Depth](../../CVPR2026/3d_vision/the_midas_touch_for_metric_depth.md)
- [\[CVPR 2026\] X-band Radar Non-Line-of-Sight Imaging](../../CVPR2026/3d_vision/x-band_radar_non-line-of-sight_imaging.md)
- [\[CVPR 2026\] RINO: Rotation-Invariant Non-Rigid Correspondences](../../CVPR2026/3d_vision/rino_rotation-invariant_non-rigid_correspondences.md)

</div>

<!-- RELATED:END -->
