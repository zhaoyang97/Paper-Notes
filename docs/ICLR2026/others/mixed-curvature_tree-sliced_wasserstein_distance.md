---
title: >-
  [论文解读] Mixed-Curvature Tree-Sliced Wasserstein Distance
description: >-
  [ICLR 2026][混合曲率空间] 把 Tree-Sliced Wasserstein 框架搬到由欧氏/球面/双曲分量做笛卡尔积的混合曲率空间上，用「跨子空间生长的测地线树」作为投影域，得到一个既保留几何与拓扑结构、又有闭式解、可并行的分布距离 MCTSW。 领域现状：流形假设指出真实数据往往集中在低维曲面上…
tags:
  - "ICLR 2026"
  - "混合曲率空间"
  - "Tree-Sliced Wasserstein"
  - "Radon 变换"
  - "最优传输"
  - "测地线"
---

# Mixed-Curvature Tree-Sliced Wasserstein Distance

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=e439wJl5sT](https://openreview.net/forum?id=e439wJl5sT)  
**代码**: 随附补充材料  
**领域**: 最优传输 / 黎曼几何表示学习  
**关键词**: 混合曲率空间, Tree-Sliced Wasserstein, Radon 变换, 最优传输, 测地线  

## 一句话总结
把 Tree-Sliced Wasserstein 框架搬到由欧氏/球面/双曲分量做笛卡尔积的混合曲率空间上，用「跨子空间生长的测地线树」作为投影域，得到一个既保留几何与拓扑结构、又有闭式解、可并行的分布距离 MCTSW。

## 研究背景与动机
**领域现状**：流形假设指出真实数据往往集中在低维曲面上，单一欧氏几何难以刻画。球面空间适合方向性/周期性数据（文本嵌入、全景图），双曲空间适合层级与图结构。近年的混合曲率空间（Mixed-Curvature Space, MCS）把欧氏、球面、双曲分量做笛卡尔积，用一个异质几何同时承载多种结构，已在 VAE、GNN、决策森林、持续学习中展现优势。

**现有痛点**：能在 MCS 上「比较两个概率分布」的工具却很匮乏。KL 散度不是真正的度量、对不相交支撑失效；标准最优传输（OT）有超立方复杂度，难以扩展。Sliced-Wasserstein（SW）靠把测度投影到一维子空间换来闭式解，但一维切片丢掉了弯曲空间中分布的几何与拓扑复杂度。Tree-Sliced Wasserstein（TSW）用「树度量」替代一维直线、提供更丰富的投影几何且仍保留闭式 $W_1$，但它一直停留在欧氏设定，**没人把树系统真正搬进混合曲率空间**。

**核心矛盾**：混合曲率空间恰恰最需要一个能同时尊重异质几何（不同分量曲率不同）的分布距离，而现有 sliced 方法要么投影到一维丢结构、要么用「乘积空间」把各分量独立处理、无法跨子空间联合传输质量。

**本文目标**：在 MCS 上构造一个保结构、有闭式解、可并行的分布距离。

**核心 idea**：**[树即异质几何的桥]** 在 MCS 上构造由一个公共根发出、每条边只在某一个分量里生长测地线的「混合曲率树」；这棵树的边天然横跨不同曲率的子空间，把质量沿树传输就能让传输路径同时穿过欧氏/球面/双曲区域，从而联合刻画异质几何，而树度量又保证 $W_1$ 有闭式解。

## 方法详解

### 整体框架
MCTSW 把 TSW 的「采样一维方向 → 投影 → 算闭式 $W_1$ → 蒙特卡洛平均」流水线整体抬升到混合曲率流形：先在 MCS 上定义**混合曲率树系统**（一组从公共根 $x$ 沿测地线射线长出的边）及其树度量，再定义一个**面向树的 Radon 变换**把流形上的测度投影到树上，最后把树上的闭式 $W_1$ 对随机采样的树求期望，得到 MCTSW 距离，并用蒙特卡洛估计 + GPU 并行实现。

```mermaid
flowchart LR
    A[MCS 测度 μ, ν] --> B[采样混合曲率树 T<br/>根 x + k 条跨分量测地线射线]
    B --> C[Radon 变换 R_α<br/>投影坐标=到根测地距离<br/>splitting map 按距离 softmax 分质量]
    C --> D[树上闭式 1-Wasserstein<br/>W_dT 按子树质量差求和]
    D --> E[对 L 棵随机树取平均<br/>蒙特卡洛 MCTSW]
```

### 关键设计

**1. 混合曲率树系统：让一棵树横跨多个曲率分量** 给定 MCS $\mathcal{M}=\prod_{j=1}^m C^{d_j}_{K_j}$ 中的根 $x$，定义从 $x$ 沿方向 $y$ 的测地线射线 $r^y_x:=\bigsqcup_{t>0}\mathrm{Exp}^{\mathcal{M}}_x(t\cdot\mathrm{Log}^{\mathcal{M}}_x(y))$，每个点都能用参数对 $(t,r^y_x)$ 唯一表示。取 $k$ 个与根等距 $d_{\mathcal{M}}(x,y_i)=\epsilon$ 的点生成 $k$ 条射线，再用「只在 $t=0$（根）处相等」的等价关系把它们粘成商空间 $T^{y_1,\dots,y_k}_x$。其上的树度量是：同一射线上两点距离 $=|t_i-t_j|$（沿测地线本身），不同射线上两点 $=t_i+t_j$（必须绕回公共根 $x$）。正是「绕回根」这一步把分属不同曲率分量的两段路径连起来——根成了异质几何之间的枢纽，这棵树因此能编码混合曲率结构而非任何单一曲率。

**2. 树上的 Radon 变换：用测地距离投影 + softmax 分质量** 要把流形上的密度搬到树上，需要两件东西：一个投影函数和一个 splitting map。投影函数把任意点 $z$ 映到坐标 $d_{\mathcal{M}}(z,x)$（到根的测地距离），对每条射线都用这个坐标；splitting map 决定 $z$ 的质量如何在 $k$ 条射线间分配，本文取测地距离的 softmax：

$$\alpha(z,T)_i=\frac{\exp(-d_{\mathcal{M}}(z,\overline{r^{y_i}_x}))}{\sum_{j=1}^k\exp(-d_{\mathcal{M}}(z,\overline{r^{y_j}_x}))},$$

其中 $d_{\mathcal{M}}(z,\overline{r^{y_i}_x})$ 是 $z$ 到第 $i$ 条完整测地线的最近距离。于是 Radon 变换 $\left(R_\alpha f\right)_T(t,r^{y_i}_x)=\int_{\mathcal{M}} f(z)\,\alpha(z,T)_i\,\delta(t-d_{\mathcal{M}}(x,z))\,d\sigma(z)$ 把质量按「离哪条射线近就多分一点」摊到树上，作者在附录证明了它的良定义性、线性与单射性（单射性是 MCTSW 成为真正度量的关键）。

**3. 闭式树 Wasserstein + 简化采样换并行** 树度量下的 1-Wasserstein 有标准闭式解 $W_{d_T,1}(\mu,\nu)=\sum_{e\in T} w_e\,|\mu(\Gamma(v_e))-\nu(\Gamma(v_e))|$，即对每条边累加「该边远端子树 $\Gamma(v_e)$ 上两测度质量差」乘边长，无需迭代求解 OT。为了能并行，作者施加两条约束：每条边只在恰好一个分量上与根不同（$(y_i-x)$ 只有一个分量非零）、且所有分量同维 $d_i=d$。这样树空间 $\mathcal{T}^{\mathcal{M}}_k$ 与 $\mathcal{M}\times[m]^k\times(\mathcal{M}_\epsilon)^k$ 一一对应，采样退化为：从 Wrapped Normal 采根 $x$、从 $[m]^k$ 采每条边落在哪个分量、再在球面上采方向。由于投影到根后同一棵树内所有坐标共享一次排序，最终复杂度 $O(Ln\log n+Ldmnk)$（$L$ 棵树、$k$ 条边、$n$ 个支撑点、$m$ 个分量），且因各分量计算相互独立而天然适配 GPU。

## 实验关键数据

### 主实验表格

**梯度流（学习 6 个 WND 混合的目标分布，越低越好）：**

| 方法 | log $W_2$ ↓ |
|------|------|
| SW$_{\text{ambient}}$ | 0.33 |
| Prod-TSW | 0.34 |
| Prod-SW | 0.31 |
| **MCTSW (ours)** | **−3.65** |

**图自监督学习（Cora 测试准确率，越高越好）：**

| 方法 | Accuracy ↑ |
|------|------|
| SSGE | 79.55 ± 0.35 |
| E-TSW-SSGE（欧氏） | 77.85 ± 0.32 |
| H-TSW-SSGE（双曲） | 75.10 ± 0.22 |
| S-TSW-SSGE（球面） | 78.33 ± 0.15 |
| **MCTSW-SSGE** | **79.86 ± 0.45** |

**混合曲率 VAE（CIFAR-10 测试 BCE，越低越好）：**

| 隐空间 | 方法 | 正则项 | Test BCE ↓ |
|------|------|------|------|
| 欧氏 | VAE | KL | 0.6423 ± 0.0008 |
| 欧氏 | SWAE | SW | 0.6043 ± 0.0005 |
| 球面 | S-VAE | KL | 0.6285 ± 0.0004 |
| 球面 | STSW-VAE | STSW | 0.6026 ± 0.0009 |
| 双曲 | H-VAE | KL | 0.6402 ± 0.0005 |
| 双曲 | HSW-VAE | HSW | 0.6012 ± 0.0006 |
| MCS | M-VAE | KL | 0.6419 ± 0.0008 |
| **MCS** | **MCTSW-VAE** | **MCTSW** | **0.6000 ± 0.0002** |

### 消融实验表格
三个任务里都内嵌了「混合曲率 vs 常曲率」的对照消融（同一框架只换隐空间几何）：

| 维度 | 对照 | 结论 |
|------|------|------|
| 距离设计（梯度流） | MCTSW vs Prod-TSW/Prod-SW（乘积空间逐分量独立处理） | 跨子空间联合传输的 −3.65 远优于逐分量独立的 ~0.31 |
| 隐空间几何（Cora） | MCS vs 单一 E/H/S | MCS 79.86 > 球面 78.33 > 欧氏 77.85 > 双曲 75.10 |
| 正则项 + 几何（VAE） | MCTSW-VAE vs M-VAE(KL) 及各常曲率 SW 变体 | MCTSW 0.6000 同时压过 KL 版 MCS 与所有常曲率 sliced 变体 |

### 关键发现
- 梯度流上 MCTSW 把 log $W_2$ 从基线的约 0.31 拉低到 −3.65，量级差距极大，说明在 MCS 上用一维切片或乘积空间会严重低估/误导分布差异，而跨子空间的树传输能真正收敛到目标。
- 两条独立的增益叠加：「换成 MCTSW 距离」和「换成混合曲率隐空间」各自都带来提升，VAE 里 MCTSW-VAE 同时优于 KL 版 MCS（M-VAE）和常曲率 SW 变体，验证二者正交互补。
- Cora 上相对 SSGE 仅小幅提升（79.86 vs 79.55），下游图任务的增益不如几何更纯的梯度流/VAE 任务明显。

## 亮点与洞察
- **几何洞察精准**：把 TSW 的「树作为更丰富投影域」与「混合曲率即异质几何拼接」两个 idea 接在一起——树绕回根的结构恰好对应「跨曲率分量传输必须经过枢纽」，是非常自然的几何匹配。
- **理论闭环完整**：从树度量的度量性、Radon 变换的良定义/线性/单射，到 MCTSW 的度量性，逐项给证明，单射性直接撑起距离的可分性。
- **工程可落地**：通过「每条边只动一个分量 + 等维」两条约束把树空间因式分解，换来一次排序复用与跨分量并行，复杂度对支撑点近线性。

## 局限与展望
- **gyrovector 运算的开销与数值稳定性**：作者在结论里明确承认混合曲率算子（Möbius 加法、指数/对数映射）带来运行时开销与数值不稳，是后续要解决的主要瓶颈。
- **下游增益有限**：图 SSL 上仅微幅领先基线，说明在几何结构不那么显式的任务里收益会被稀释。
- **简化假设较强**：并行化依赖「每条边单分量变化 + 所有分量等维」，放宽这些约束后采样与闭式结构如何保持仍待探索。
- **实验规模偏小**：基准集中在 6-WND 混合、Cora、CIFAR-10，缺少大规模真实异质数据上的验证。

## 相关工作与启发
本文站在三条线的交汇处：Sliced-Wasserstein 及其流形变体（球面切片、双曲切片、Cartan–Hadamard 流形切片），Tree-Sliced Wasserstein（用动态采样的树系统替代固定树度量），以及混合曲率表示学习（MCS 用于 VAE、GNN、持续学习）。启发在于：当数据的几何先验是「多种结构拼接」时，分布距离的投影域也应当是「能横跨多种几何的结构」而非单一直线或单一曲率流形；树这种既灵活又保留闭式 OT 的载体，是连接异质几何的天然选择，这一思路可推广到任意流形乘积乃至更一般的非常曲率流形上。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首次把 tree-sliced 框架与 Radon 变换严格搬到混合曲率空间，几何动机与理论构造都清晰原创。
- **实验充分度**: ⭐⭐⭐ 覆盖梯度流/VAE/图 SSL 三类任务并带几何消融，但数据集规模小、图任务增益有限，缺大规模验证。
- **写作质量**: ⭐⭐⭐⭐ 背景—痛点—构造—理论层层递进，定义/定理与几何直觉解释配合得当。
- **价值**: ⭐⭐⭐⭐ 为「在异质几何隐空间上比较分布」补上了一个有闭式解、可并行的工具，对 MCS 表示学习生态有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling](../../ACL2025/others/s2wtm_spherical_sliced-wasserstein_autoencoder_for_topic_modeling.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](../../ICML2025/others/lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [\[ICLR 2026\] RADAR: Learning to Route with Asymmetry-aware Distance Representations](radar_learning_to_route_with_asymmetry-aware_distance_representations.md)
- [\[ICML 2026\] Decision Tree Learning on Product Spaces](../../ICML2026/others/decision_tree_learning_on_product_spaces.md)
- [\[ICML 2026\] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](../../ICML2026/others/cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)

</div>

<!-- RELATED:END -->
