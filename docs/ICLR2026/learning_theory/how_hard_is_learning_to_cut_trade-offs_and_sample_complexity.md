---
title: >-
  [论文解读] How hard is learning to cut? Trade-offs and sample complexity
description: >-
  [ICLR 2026][learning_theory][学习理论] 本文为"学习割平面（learning-to-cut）"任务首次给出样本复杂度的**下界**，证明无论用 B&C 树规模还是 gap closed 作为评分函数，学习一个把实例映射到割平面的函数类，所需样本量都至少等于（在常数倍意义下）用同一函数类去拟合一个一般目标函数的样本量；下界与已知上界几乎吻合，说明两个评分在可学习性上难度相当，进而用 GNN 实验佐证 gap closed 是逼近 tree size 的好代理。
tags:
  - "ICLR 2026"
  - "learning_theory"
  - "学习理论"
  - "样本复杂度"
  - "学习割平面"
  - "分支定界"
  - "整数规划"
  - "Chvátal-Gomory cut"
  - "VC 维"
  - "fat-shattering 维"
---

# How hard is learning to cut? Trade-offs and sample complexity

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pc3ZW7lmaV](https://openreview.net/forum?id=pc3ZW7lmaV)  
**代码**: 未公开  
**领域**: learning_theory  
**关键词**: 学习理论, 样本复杂度, 学习割平面, 分支定界, 整数规划, Chvátal-Gomory cut, VC 维, fat-shattering 维  

## 一句话总结
本文为"学习割平面（learning-to-cut）"任务首次给出样本复杂度的**下界**，证明无论用 B&C 树规模还是 gap closed 作为评分函数，学习一个把实例映射到割平面的函数类，所需样本量都至少等于（在常数倍意义下）用同一函数类去拟合一个一般目标函数的样本量；下界与已知上界几乎吻合，说明两个评分在可学习性上难度相当，进而用 GNN 实验佐证 gap closed 是逼近 tree size 的好代理。

## 研究背景与动机

**领域现状**：分支定界（branch-and-cut, B&C）是整数规划（ILP/MILP）求解器的基石，其中"选哪个割平面（cut）加入当前 LP 松弛"是一个关键决策。近年大量工作用机器学习（模仿学习、强化学习、神经网络打分）替代手工启发式来做 cut selection。评价一个割好坏有两种主流评分函数：(i) **B&C 树规模**——加入割后树的相对缩小量，最贴近真实运行时间（约等于求解的 LP 个数），但要把问题解到最优才能评估，极其昂贵；(ii) **gap closed**——加入割后 LP 松弛目标值的相对改善量，只需解一个 LP，便宜很多，被整数规划界长期使用，常被当作 tree size 的代理。

**现有痛点**：学习方法绕不开的一个根本问题是"要多少训练样本才能在整个（且未知的）实例分布上泛化"，即**样本复杂度**。但已有理论（Balcan et al. 2021、Cheng et al. 2024）只给了特定算法（受限概念类，如固定 CG 权重或神经网络生成 CG 权重）的**上界**，且都是针对 tree size 评分；从来没有人给出**下界**，也没人把两个评分放在一起做原则性比较。

**核心矛盾**：gap closed 虽是 tree size 的"代理"，但两者其实很难正面比较——一个割可能大幅缩小树却根本不切掉 LP 最优分数解，反之一个切掉分数解的割可能对树规模毫无长期帮助（论文给了 $\max 5x_1+8x_2$ 的反例）。那么这个代理到底有多好？两个评分谁更难学？如果某个评分天生更难学，就会影响学习算法设计与评价标准的选择。

**本文目标**：(1) 理论上给出对一大类函数类（含神经网络、图神经网络）都成立的样本复杂度下界，并与已知上界对比判断紧度；(2) 实践上验证 gap closed 是否真能作为 tree size 的可行代理。

**核心 idea**：**把"学习割平面"归约到"学习一般目标函数"**——通过对概念类施加平移/缩放不变性假设，证明割学习问题的 fat-shattering / VC 维不会低于底层函数类拟合任意目标的难度，从而下界由概念类自身的复杂度（如神经网络的 $WL\log(W/L)$）决定，且对两个评分函数都成立。

## 方法详解

### 整体框架
论文是"理论下界 + 实验佐证"双线结构。理论线把 learning-to-cut 形式化为一个监督回归问题：学一个函数 $f\in F$ 把 ILP 实例 $I$ 映射到一组 CG 割权重，目标最小化 $\mathbb{E}_{(I,s)\sim D}[(h(I,f(I))-s)^2]$，其中 $s$ 是该实例"最佳割"的评分标签。借助 Anthony & Bartlett 的 fat-shattering 下界定理（注意：传统 VC 下界只约束一致收敛 UC，不一定反映真实学习样本复杂度，所以必须按概念类小心分析），通过两条结构性假设把割学习的可打散能力下放到底层函数类，得到与上界几乎匹配的下界；实验线则训练一个 GNN 在 Simplex tableau 割上分别用两种评分打分，看 gap closed 训练出的模型能否在 tree size 上同样有效。

### 关键设计

**1. 把割学习形式化为监督回归，并区分于已有的无监督上界设定**：学习目标是公式 $\min_{f\in F}\mathbb{E}_{(I,s)\sim D}[(h(I,f(I))-s)^2]$，标签 $s$ 与实例 $I$ 联合采样而非确定性函数。论文专门论证（Remark 2.1）这个监督设定的合理性：一是真实 B&C 树规模本身带随机性（求解器 tie-breaking 导致的 performance variability），评分天然服从条件分布；二是即便标签确定但属于足够复杂的函数族，fat-shattering 论证依然成立。这与 Cheng et al. (2024) 纯由评分函数驱动、不看标签的**无监督**上界设定不同——这一区分是后续上下界能否对齐比较的前提。

**2. 两条结构假设把"割学习难度"归约到"底层函数类难度"**：这是下界推导的技术核心。**假设 1（平移与缩放封闭）**要求 $F$ 对输入平移、对输出每个坐标缩放保持封闭——神经网络/图神经网络对任意非恒零激活都满足。**假设 2（按行限制保持打散能力）**要求把 $f$ 限制到只看 $n$ 个变量输入时，每个坐标分量 $F_i[n]$ 的 VC 维是常数，记为 $\mathrm{VCdim}(F[n])$。在此之上引入一个"挤压函数" $\sigma'$ 把网络输出压到 $[0,1]^m$ 当作 CG 权重，构造最终概念类 $F_{s,\sigma'}:=\{I\mapsto s(I,h(I)):h\in F_{\sigma'}\}$。**Theorem 3.2** 由此得到核心下界：对 gap-closed 和 tree size 两种评分，$m_L(\epsilon,\delta)=\Omega\!\left(\frac{\mathrm{VCdim}(F[n])}{\epsilon}\right)$。**Corollary 3.3** 进一步说明这个下界恰好等于"用 $F_n$ 去学一个一般目标函数"的样本复杂度——即学割不比学任意函数更容易，且两评分难度同阶。

**3. 神经网络实例化下界并与上界对齐判定紧度**：把上述抽象下界落到具体 ReLU 网络（$\le L$ 层、$\le W$ 权重、concatenation 编码 $I\mapsto(A,b,c)$）。**Proposition 3.4** 给出 $m_L(\epsilon,\delta)\ge \frac{1}{\epsilon C}\,\overline{W}L\log(\overline{W}/L)$，其中 $\overline{W}=W-w_1(n+1)m$ 是扣掉被忽略的 $(n+1)m$ 个输入后剩余的权重数。对比 Cheng et al. (2024) 的上界 $O\!\left(\frac{1}{\epsilon^2}(LW\log(U+m)+W\log M)\right)$，在网络权重远大于 $n,m$ 的 regime 下，上下界只差 $1/\epsilon$ 量级（忽略 $\log$ 因子），这一 gap 在一般学习框架里本就不可避免——说明下界近乎紧。值得注意下界**不依赖**输入数据幅度 $M$。下界中 $W>CL$ 的结构假设并非来自本文证明技术，而是继承自 Bartlett et al. (2019) 用 bit-extraction 给神经网络 VC 维下界时的限制。

**4. 推广到 Simplex tableau 割的受限割池**：实际求解器只从最优单纯形表（tableau）读取 CG 割（Gomory 把 $u$ 取为基的逆），而非所有 CG 割。论文把概念类改写为"先用 simplex 算出 $m$ 个 tableau 割 $(a_i,b_i)$，再用 $g\in G$ 给每个割打分并取最大者"的复合形式 $\tilde G$。**Theorem 3.6** 证明即便割池受限，样本复杂度仍由底层 VC 维驱动：$m_L(\epsilon,\delta)=\Omega(\mathrm{VCdim}(G[n])/\epsilon)$；**Proposition 3.7** 给出对应神经网络下界 $\frac{1}{\epsilon C}\overline{W}L\log(\overline{W}/L)$（此时 $\overline{W}=W-w_1(n+1)(m+1)$），与上界 $O(WL\log(Um)/\epsilon^2)$ 对比同样近乎紧。这保证理论结论在贴近实际求解器的设定下依然成立。

## 实验关键数据

实验用 GNN 把 (ILP 实例, 割) 对编码为带三维点特征的加权二部图（变量点 + 约束点），在 Simplex 最优 tableau 割上训练打分模型来引导 cut selection，比较"用 tree size 训练" vs "用 gap closed 训练"。数据集为 Set Cover、Facility Location、Knapsack、Vertex Cover 四类经典 ILP（各 1000 个实例随机生成），用 Gurobi 12.0.1（关掉默认割/启发式/presolve）求解，250 个测试实例上报平均树规模。

### 主实验表格（部分结果：250 测试实例上的平均 B&C 树规模，越小越好）

**Set Cover**

| 策略 | B&C tree 训练 | gap closed 训练 |
|------|:---:|:---:|
| Optimal（完美预测器） | – | 4.95 |
| Parallelism | – | 8.29 |
| Efficacy | – | 9.90 |
| Mix | – | 9.30 |
| Random | – | 9.71 |
| **GNN** | **8.27** | **8.65** |

**Facility Location**

| 策略 | B&C tree 训练 | gap closed 训练 |
|------|:---:|:---:|
| Optimal（完美预测器） | – | 86.31 |
| Parallelism | – | 144.09 |
| Efficacy | – | 123.63 |
| Mix | – | 133.72 |
| Random | – | 152.46 |
| **GNN** | **128.85** | **134.61** |

### 消融实验表格（两评分训练得到的 GNN 对比，体现 gap closed 代理质量）

| 问题 | GNN(tree size) | GNN(gap closed) | 与最优差距 |
|------|:---:|:---:|:---:|
| Set Cover | 8.27 | 8.65 | 距 Optimal 4.95 仍有空间 |
| Facility Location | 128.85 | 134.61 | 距 Optimal 86.31 仍有空间 |

### 关键发现
- **GNN 确实学到了东西**：在 facility location、knapsack 上明显优于经典启发式（Parallelism/Efficacy/Mix/Random），在 set cover、vertex cover 上与 SOTA 启发式持平。
- **gap closed 是好代理**：用便宜的 gap closed 训练出的 GNN，其 tree size 表现仅略逊于直接用昂贵 tree size 训练，二者差距小——与理论"两评分难度同阶"互相印证。
- **仍有提升空间**：两种 GNN 距完美预测器 Optimal 都还有可观差距，说明 learning-to-cut 远未饱和。

## 亮点与洞察
- **首个 learning-to-cut 样本复杂度下界**：填补了该领域只有上界、无下界的空白，把两个评分函数首次放在同一理论框架里比较。
- **归约思路漂亮**：用两条对（图）神经网络天然成立的结构假设，把"学割"的难度精确归约到"学一般函数"，使下界与已知上界差距收窄到不可避免的 $1/\epsilon$ 量级。
- **理论与实践闭环**：理论说两评分难度同阶 → 实验验证 gap closed 可作代理 → 反过来为整数规划界长期使用 gap closed 提供了首个原则性辩护。
- **下界不依赖数据幅度**：相比上界含 $\log M$（系数幅度上界），下界纯由网络结构 $W,L$ 决定，更干净。

## 局限与展望
- **未利用"切掉分数解"的额外结构**：作者猜想，由于 gap closed 隐含只需要那些真正切掉 LP 分数解的割（数量受限），应能为 gap closed 推出**更好的上界**，从而打破当前"两评分同阶"的对称结论——这是留给后续的关键开放问题。
- **$W>CL$ 结构假设**：神经网络下界继承了 bit-extraction 技术对网络结构的限制，要去掉需另寻专门针对打散 ILP 的证明路径。
- **下界仅对未知分布成立**：在完全确定且结构简单的设定下，真实样本复杂度可能低于本文下界；下界的意义依赖标签机制足够复杂（求解器随机性或丰富标签族）。
- **简化设定**：理论与实验都只考虑"一次加一个割"，而真实求解器是成轮（round）批量加割；分析批量过程要难得多。
- **实验规模偏小**：四类问题实例规模都不大（如 knapsack 仅 2 背包 16 物品），代码未公开。

## 相关工作与启发
- **学割上界**：Balcan et al. (2021) 给固定 CG 权重的上界，Cheng et al. (2024) 给神经网络生成 CG 权重的多项式-对数上界——本文的下界正是与它们对齐比较。
- **割平面理论**：Gomory (1958)、Chvátal (1973) 的 CG 割是全文的割族基础；Deza & Khalil (2023) 的 survey 是 learning-to-cut 的入口。
- **学割的学习方法**：Paulus et al. (2022) 模仿学习选割、Huang et al. (2022) 学打分函数、Tang et al. (2020) 用 RL 把选割建模为 MDP、Ling et al. (2024) 学"何时停止加割"——这些都是本文样本复杂度分析针对的对象。
- **算法选择 / 带预测的算法设计**：Rice (1976)、Gupta & Roughgarden (2016)、Mitzenmacher & Vassilvitskii (2022) 把"按实例选算法"的样本复杂度纳入更广谱系。
- **学习理论工具**：Anthony & Bartlett (2009) 的 fat-shattering 下界定理、Bartlett et al. (2019) 的分段线性网络 VC 维界、Shalev-Shwartz et al. (2009) 与 Feldman (2016) 关于 UC 与学习样本复杂度之间 gap 的发现，是下界推导的理论支柱。
- **启发**：把"任务特定学习问题"归约到"底层函数类拟合任意目标"的下界技巧，可推广到其他 learning-to-optimize 场景（学分支变量、学启发式）来判断已有上界是否紧。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 首个 learning-to-cut 样本复杂度下界，且首次把两个评分函数放进统一理论框架并行比较，归约思路有原创性。
- **实验充分度**: ⭐⭐⭐ — 实验是理论的佐证而非主体，覆盖四类经典 ILP 但规模偏小、代码未公开，距完美预测器仍有差距。
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、上下界对比规范、用具体反例说明两评分不可正面比较，理论陈述严谨；部分证明细节下放附录。
- **价值**: ⭐⭐⭐⭐ — 为整数规划界长期沿用的 gap closed 代理提供了首个原则性理论辩护，并指明 gap closed 上界可改进的开放方向，对 learning-to-cut 的理论基础有奠基意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Mitigating the Curse of Detail: Scaling Arguments for Feature Learning and Sample Complexity](mitigating_the_curse_of_detail_scaling_arguments_for_feature_learning_and_sample.md)
- [\[ICLR 2026\] T-Tamer: Provably Taming Trade-offs in ML Serving](t-tamer_provably_taming_trade-offs_in_ml_serving.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects](minimax_sample_complexity_of_graph_neural_networks_lower_bounds_and_structural_e.md)
- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)

</div>

<!-- RELATED:END -->
