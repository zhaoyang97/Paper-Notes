---
title: >-
  [论文解读] Diffusion & Adversarial Schrödinger Bridges via Iterative Proportional Markovian Fitting
description: >-
  [ICLR 2026][图像生成][Schrödinger Bridge] 本文揭示实践中用来稳定 IMF 训练的「前向-后向交替」启发式其实暗含 IPF 迭代，从而把 IMF 与 IPF 统一为 **IPMF（Iterative Proportional Markovian Fitting）**，给出首个对双向 IMF 的收敛性证明，并把「起始耦合」变成可调旋钮，在生成质量与输入输出相似度之间灵活权衡。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "Schrödinger Bridge"
  - "Iterative Markovian Fitting"
  - "Iterative Proportional Fitting"
  - "熵正则最优传输"
  - "无配对图像翻译"
  - "扩散桥"
---

# Diffusion & Adversarial Schrödinger Bridges via Iterative Proportional Markovian Fitting

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=38fGCBhFF5](https://openreview.net/forum?id=38fGCBhFF5)  
**代码**: [https://github.com/gregkseno/ipmf](https://github.com/gregkseno/ipmf)  
**领域**: 生成模型 / Schrödinger Bridge / 最优传输  
**关键词**: Schrödinger Bridge, Iterative Markovian Fitting, Iterative Proportional Fitting, 熵正则最优传输, 无配对图像翻译, 扩散桥  

## 一句话总结
本文揭示实践中用来稳定 IMF 训练的「前向-后向交替」启发式其实暗含 IPF 迭代，从而把 IMF 与 IPF 统一为 **IPMF（Iterative Proportional Markovian Fitting）**，给出首个对双向 IMF 的收敛性证明，并把「起始耦合」变成可调旋钮，在生成质量与输入输出相似度之间灵活权衡。

## 研究背景与动机
- **领域现状**：受最优传输与随机过程联系启发的 Schrödinger Bridge（SB）扩散桥模型，已成为无配对域翻译（图像风格迁移、单细胞分析等）的有力工具。求解 SB 主要有两条路线：早期的 **IPF（Iterative Proportional Fitting，即 Sinkhorn 算法）**，以及近年崛起的 **IMF（Iterative Markovian Fitting）**。
- **现有痛点**：两条路线各有「漏」。IPF 从满足最优性的先验过程出发迭代逼近边缘匹配，但近似误差会导致 **prior forgetting**——边缘对上了、最优性却丢了；IMF 反过来从满足边缘匹配的过程出发逐步逼近最优性，但单向拟合的误差累积会让边缘匹配性质丧失。
- **核心矛盾**：为了让 IMF 在实践中能用（如 DSBM 的扩散实现、ASBM 的 GAN 实现），人们不得不引入一个**没有理论解释的启发式**——每轮在「学前向扩散」和「学后向扩散」之间交替。这个修改对稳定训练至关重要，却长期被当成「黑魔法」，缺乏分析，也无法解释它为何能避免误差累积。
- **本文目标**：搞清楚这个双向启发式到底在做什么，为它建立收敛理论，并把对 SB 求解器零散的认识（IPF / IMF / 离散 / 连续 / 在线）统一进一个框架。
- **核心 idea**：**双向 IMF「偷偷」用了 IPF 的投影**。本文证明双向 IMF 的一步等价于「两次 IMF 投影 + 两次 IPF 投影」的组合，因此把它命名为 IPMF，并指出 IPF 和 IMF 都只是 IPMF 在特定起始耦合下的特例。

## 方法详解

### 整体框架
SB 问题（带 Wiener 先验）等价于带二次代价的**熵正则最优传输（EOT）**：在端点边缘为 $p_0,p_1$ 的过程中，找一个与先验 Wiener 过程 KL 散度最小、即同时满足「最优性」（输入输出相似）和「边缘匹配」（域翻译正确）的 Markov 过程。IPF 和 IMF 各自只迭代逼近其中一个性质。本文的关键观察是：实践用的双向 IMF 每一轮交替学前向/后向，恰好在「往最优性走」（IMF 投影 = 倒易投影 $\mathrm{proj}_R$ + Markov 投影 $\mathrm{proj}_M$）和「往边缘匹配走」（IPF 投影 $\mathrm{proj}_0,\mathrm{proj}_1$）之间来回，于是把两条历史路线缝合成统一的 IPMF。

```mermaid
flowchart LR
    A["起始耦合 q0<br/>(任意)"] --> B["projR<br/>倒易投影"]
    B --> C["proj1∘projM<br/>后向参数化 + IPF"]
    C --> D["projR<br/>倒易投影"]
    D --> E["proj0∘projM<br/>前向参数化 + IPF"]
    E -->|"重复 K 步"| B
    E --> F["收敛到 SB 解 q*<br/>(reciprocal 类中的 Markov 过程)"]
```

### 关键设计

**1. 双向 IMF = IPMF：把启发式拆成 IMF 投影 + IPF 投影。** 本文的核心理论拆解在于：实践算法每一轮 $q_{4k+2}$（后向参数化）写成 $p(x_1)\prod_n q_{4k+1}(x_{t_n}|x_{t_{n+1}})$，这其实是先做 Markov 投影 $\mathrm{proj}_M$、再把末端边缘强行替换成 $p_1$ 的 IPF 投影 $\mathrm{proj}_1$ 的复合，即 $q_{4k+2}=\mathrm{proj}_1(\mathrm{proj}_M(q_{4k+1}))$；对称地 $q_{4k+4}=\mathrm{proj}_0(\mathrm{proj}_M(q_{4k+3}))$。于是一个 IPMF 步 = 两次倒易投影 $\mathrm{proj}_R(q)=q(x_0,x_1)p_{W^\epsilon}(x_{\mathrm{in}}|x_0,x_1)$（重新铺 Brownian Bridge）+ 两次 Markov 投影 + 两次 IPF 边缘替换。正因为多了 IPF 那两步「把边缘钉回 $p_0,p_1$」，IPMF 才不会像纯单向 IMF 那样累积误差、丢失边缘——这就是启发式有效的根本原因。框架还自然涵盖特例：起始耦合若已有正确边缘且端点间是 Brownian Bridge，IPMF 退化为 IMF；若起始耦合是 reciprocal 类中的 Markov 过程且有正确初始边缘，则退化为 IPF。

**2. 最优性矩阵：用一个矩阵刻画高斯计划的「最优程度」。** 为了能定量分析收敛，本文给高斯耦合引入了**最优性矩阵** $A(q)$。Theorem 3.1 证明任意二维高斯 $q(x_0,x_1)$（端点边缘 $\mathcal N(\eta,Q)$、$\mathcal N(\nu,S)$）都是某个代价 $c_A(x_0,x_1)=-x_1^\top A x_0$ 下熵正则 OT 问题的唯一解，其中 $A=\Xi(P,Q,S)=S^{-1}P^\top(Q-PS^{-1}P^\top)^{-1}$。这个矩阵把「该耦合在为哪个传输代价做最优」编码了出来：当且仅当 $A(q)=\epsilon^{-1}I_D$ 时，代价退化为 $\epsilon^{-1}\|x_1-x_0\|^2/2$，$q$ 恰好是先验 $W^\epsilon$ 下的静态 SB 解。最优性矩阵到 $\epsilon^{-1}I_D$ 的距离，因此成了「离 SB 解还有多远」的天然标尺。

**3. 双向 IMF 的首个收敛性证明：IPF 不动 copula、IMF 收缩 copula。** 基于最优性矩阵，本文给出双向 IMF（即 IPMF）历史上第一个收敛分析。证明思路是把每步拆成两类作用：**IPF 步只换边缘、不改 copula**（联合分布中与边缘无关的、对 $A_k$ 不变的部分），**IMF 步保持边缘、改变 copula** 并把 $A_k$ 拉向 $\epsilon^{-1}I_D$。Theorem 3.2 证明对高斯（$D=1$ 任意 $\epsilon$，或 $D>1$ 离散时间、$\epsilon\gg0$）有**指数收敛**：$\|A_k-\epsilon^{-1}I_D\|_2\le\beta^{2k}\|A_0-\epsilon^{-1}I_D\|_2$、均值与协方差项也以 $\alpha^k,\alpha^{2k}$（$\alpha,\beta<1$）收缩，从而前向/反向 KL 同时趋零。Theorem 3.3 进一步证明只要 $p_0,p_1$ 支撑有界，离散与连续 IPMF 都**弱收敛**到 $q^*$。与既有结果（Shi et al. 仅证 IPF 起点、IMF 亚线性；De Bortoli et al. 证 IPF）相比，本文首次覆盖**任意起始耦合**，且要求 IMF 朝一个被 IPF 不断挪动的「移动靶」收敛，分析难度本质更高。

**4. 起始耦合作为旋钮：在质量与相似度之间权衡。** 既然 IPMF 从任意起始耦合都收敛（IPF/IMF 只是特例），起始耦合 $q_0(x_0,x_1)$ 就从「必须满足特定形式的约束」变成了**可调超参数**。本文据此设计了多种起始耦合——除经典的 IMF-like（独立耦合 $p_0p_1$）、IPF-like、Identity（$x_1=x_0$）外，还引入由 mini-batch OT 给出的 IMF-OT，以及用 SDEdit（DDPM 或 Stable Diffusion v1.5）预生成的 SD/DDPM SDEdit 耦合。直觉是：起点越「像目标分布」生成质量越好，起点越「贴近输入」输入输出相似度越高，从而让用户按任务需求（更高保真 vs. 更强语义对齐）裁剪模型。

## 实验关键数据

### 主实验表格（SB benchmark，cBW²₂-UVP ↓ %，对比已知真值 SB 解）
覆盖 $\epsilon\in\{0.1,1,10\}$、$D\in\{2,16,64,128\}$。各起始耦合在同一求解器（DSBM 或 ASBM）下结果相近，验证「从任意起点都收敛」的猜想。代表性数值（$\epsilon=1$）：

| 算法 | 类型 | D=2 | D=16 | D=64 | D=128 |
|------|------|-----|------|------|-------|
| Best on benchmark† | Varies | 1.04 | 9.08 | 18.05 | 15.23 |
| DSBM-IMF | IPMF | 0.68 | 0.63 | 5.8 | 29.5 |
| DSBM-IPF | IPMF | 0.29 | 0.76 | 4.05 | 29.59 |
| DSBM-Identity | IPMF | 0.26 | 0.69 | 7.46 | 29.5 |
| ASBM-IMF† | IPMF | 0.19 | 1.6 | 5.8 | 10.5 |
| ASBM-IPF | IPMF | 0.18 | 1.68 | 9.25 | 20.47 |
| ASBM-Identity | IPMF | 0.19 | 2.44 | 8.28 | 11.61 |
| SF2M-Sink† | Bridge Matching | 0.2 | 1.1 | 9 | 23 |

### 消融实验表格（CelebA 64×64 male→female，FID↓ / MSE↓）
同一求解器下换起始耦合，观察质量-相似度权衡：

| 起始耦合 | DSBM FID↓ | DSBM MSE↓ | ASBM FID↓ | ASBM MSE↓ |
|----------|-----------|-----------|-----------|-----------|
| IMF | 35.23 | 0.16 | 14.84 | 0.16 |
| DDPM SDEdit | 28.77 | 0.02 | 22.65 | 0.09 |
| SD SDEdit | 61.56 | 0.02 | 33.11 | 0.04 |
| Identity | 13.65 | 0.00 | 19.32 | 0.17 |

（AFHQ 512×512 cat→wild 上同样观察到质量-相似度权衡：DSBM-IMF-OT FID 53.42 / MSE 0.085，DSBM-Identity FID 65.19 / MSE 0.054。）

### 关键发现
- **收敛实验**：128 维高斯上跑 100 步 IPMF，前向/反向 KL 与 $A_k,\nu_k,S_k$ 三项误差均指数趋零，完全吻合理论。
- **起点无关、终点一致**：在 2D Swiss roll、SB benchmark、Colored MNIST 上，IMF/IPF/Identity/Inverted-7 等各种起点都收敛到定性一致的结果，验证 IPMF 比 IPF/IMF 适用更广。
- **权衡可控**：对 DSBM，SD/DDPM SDEdit、Identity 耦合在**保持生成质量的同时大幅提升相似度**（MSE 从 0.16 降到 0.00–0.02）；对 ASBM 则提升相似度但略降质量——起始耦合确实成了实用旋钮。

## 亮点与洞察
- **把「黑魔法」讲成了数学**：前向-后向交替这个被广泛使用却无人解释的工程技巧，被精确地拆解为 IMF + IPF 的复合，给出了「它为何能防止误差累积」的根本答案（多出的 IPF 步把边缘钉回 $p_0,p_1$）。
- **统一视角**：IPF、IMF 不再是竞争路线，而是 IPMF 在不同起始耦合下的退化特例，离散/连续/在线版本都被纳入同一框架，对理解整个 SB 求解器家族很有指导价值。
- **最优性矩阵**是个漂亮的分析工具：把「一个高斯耦合在为哪个 OT 代价做最优」编码成一个矩阵，让原本难以刻画的「最优程度」变得可度量、可收缩，是收敛证明的技术支点。
- **理论直接产出可用旋钮**：「任意起点都收敛」不只是理论性质，它把起始耦合解放成超参数，给实践者一个在质量与保真间调节的新机制。

## 局限与展望
- **一般收敛仍是猜想**：指数收敛只在高斯（且高维需 $\epsilon\gg0$）严格证明，有界支撑下只证了弱收敛，对一般分布、所有 $\epsilon$ 的指数收敛仍停留在 conjecture，靠实验佐证。
- **不与既有实现正面比拼**：作者论证 IPMF 与既有实践算法只差起始耦合，故主动回避直接性能对比，读者较难定位 IPMF 相对最新 SB 求解器的绝对优劣。
- **高维大 $\epsilon$ 退化**：表 1 中 $\epsilon=10$、$D=128$ 时 DSBM 各起点 cBW²₂-UVP 飙到数百，说明大噪声高维下数值仍不稳定。
- **SDEdit 耦合只是部分验证**：SD/DDPM SDEdit 起点的 FID 偏高（如 SD SDEdit 在 DSBM 上 61.56），「好设计的耦合能同时改善两性质」的假设只得到部分支持。
- **展望**：前向-后向 IPMF 框架有望让 rectified flow 也避免误差累积，将统一思路推广到更广的生成建模与在线/连续时间设定。

## 相关工作与启发
- **IPF / Sinkhorn 路线**：De Bortoli et al. (DSB, 2021)、Vargas et al. (2021) 等，证过 IPF 的亚线性收敛与高斯下几何收敛，但受 prior forgetting 困扰。
- **IMF / Bridge Matching 路线**：Shi et al. (DSBM, 2023)、Peluchetti (2023)、Gushchin et al. (ASBM, 2024) 把 rectified flow 推广到随机过程，是本文双向启发式的直接来源；本文为它们补上了理论。
- **起始耦合工具**：mini-batch OT（Tong et al., Pooladian et al.）、SDEdit（Meng et al.）+ DDPM（Ho et al.）/ Stable Diffusion（Rombach et al.），被本文用来构造各类起始耦合。
- **启发**：当一个工程技巧反复有效却无人能解释时，往往隐藏着两套已知方法的等价/复合关系——把它「翻译」回已有理论框架，既能给出解释，又可能顺手解锁新的可调维度（这里是起始耦合）。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ — 把被当作黑魔法的双向 IMF 启发式精确拆解为 IPF+IMF 复合，并提出统一的 IPMF 框架与首个收敛证明，视角新颖且解释力强。
- **实验充分度**: ⭐⭐⭐⭐ — 从高斯、2D toy、SB benchmark 到 MNIST/CelebA/AFHQ 多尺度验证收敛与权衡，但缺乏与最新 SB 求解器的正面性能对比，且高维大 $\epsilon$ 仍不稳。
- **写作质量**: ⭐⭐⭐⭐ — 理论推导严谨、动机交代清晰，但符号密集、IPF/IMF/投影记号众多，对不熟悉 SB 的读者门槛偏高。
- **价值**: ⭐⭐⭐⭐⭐ — 统一了 SB 求解器家族的认识，解释了关键工程技巧，并把起始耦合变成实用旋钮，对生成建模与最优传输社区都有较强的理论与实践价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport](entering_the_era_of_discrete_diffusion_models_a_benchmark_for_schrödinger_bridge.md)
- [\[ICLR 2026\] Branched Schrödinger Bridge Matching](branched_schrödinger_bridge_matching.md)
- [\[ICML 2026\] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](../../ICML2026/image_generation/geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)
- [\[NeurIPS 2025\] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](../../NeurIPS2025/image_generation/grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)
- [\[NeurIPS 2025\] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](../../NeurIPS2025/image_generation/schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)

</div>

<!-- RELATED:END -->
