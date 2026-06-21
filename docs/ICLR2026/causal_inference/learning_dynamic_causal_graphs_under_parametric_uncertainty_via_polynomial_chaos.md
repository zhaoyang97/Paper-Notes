---
title: >-
  [论文解读] Learning Dynamic Causal Graphs Under Parametric Uncertainty via Polynomial Chaos Expansions
description: >-
  [ICLR 2026][因果推理][因果发现] 把每条因果边的强度从"静态权重"升级成"操作参数 $\xi$ 的函数"，用多项式混沌展开 (PCE) 把这个函数学出来，从而发现随运行工况动态变化的因果结构，并给出可证明的可识别性与收敛保证。 领域现状：因果发现已经从约束法 (PC/FCI)、打分法 (GES/NOTEARS)…
tags:
  - "ICLR 2026"
  - "因果推理"
  - "因果发现"
  - "参数不确定性"
  - "多项式混沌展开 (PCE)"
  - "动态因果图"
  - "工业过程"
  - "不确定性量化"
---

# Learning Dynamic Causal Graphs Under Parametric Uncertainty via Polynomial Chaos Expansions

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4bnCXOtHTm](https://openreview.net/forum?id=4bnCXOtHTm)  
**代码**: 待确认  
**领域**: 因果推断 / 因果发现  
**关键词**: 因果发现, 参数不确定性, 多项式混沌展开 (PCE), 动态因果图, 工业过程, 不确定性量化  

## 一句话总结
把每条因果边的强度从"静态权重"升级成"操作参数 $\xi$ 的函数"，用多项式混沌展开 (PCE) 把这个函数学出来，从而发现随运行工况动态变化的因果结构，并给出可证明的可识别性与收敛保证。

## 研究背景与动机
**领域现状**：因果发现已经从约束法 (PC/FCI)、打分法 (GES/NOTEARS) 到函数式因果模型 (LiNGAM/ANM/PNL) 演化出三大流派，能从观测数据里恢复一张有向无环图 (DAG)。但绝大多数方法的隐含前提是**因果图是静态的**——每条边的强度是个不随上下文变化的固定数。

**现有痛点**：真实工业系统违反这一假设。在化工反应器里，进料温度对产品质量的影响强烈依赖催化剂活性，而催化剂会随时间退化；换热器效率随结垢程度变化，直接改写整个热控回路。也就是说，因果效应本身是**可测量运行参数的函数**。这些参数依赖关系不是噪声，而是过程优化和预测性维护的关键信息。现有的贝叶斯因果发现 (DiBS、BCD Nets) 虽然能量化图和参数的后验不确定性，但它们处理的是有限样本带来的**认知不确定性 (epistemic)**，仍然把每条边当作静态量，不刻画边强如何随工况变化。

**核心矛盾**：静态图假设 vs. 工业因果机制随参数连续漂移的物理现实——前者一旦把不同工况的数据混在一起做边缘独立性检验，符号反转的效应会在边缘上相互抵消，导致检验误判为"独立"，整条因果边被漏掉。

**本文目标**：不替换已有的认知/偶然不确定性建模，而是**补上一个新维度**——让每条因果边显式成为一个低维操作参数向量 $\xi$ 的函数，从观测数据中学出完整的参数化因果结构，并配上可识别性证明与收敛算法。

**核心 idea**：**[函数化因果表示 + PCE 谱投影]** 把无穷维的"学函数 $b_{ij}(\xi)$"问题，通过正交多项式基截断转化为有限维的"估计谱系数 $\theta_{ij,\alpha}$"问题，从而既可学、又可证、还能量化不确定性。

## 方法详解

### 整体框架
PCT-CD 分四个阶段串成一条流水线：先把因果系数写成参数 $\xi$ 的函数构造**参数化结构方程模型 (SEM)**；再用 **PCE** 把这些函数展开成谱系数，化无穷维为有限维；接着用一个新设计的**条件独立性检验**在参数空间上发现初始因果骨架；最后用带**自然梯度**的打分法精修结构并量化每条边的强度与置信区间。

```mermaid
flowchart LR
    A[观测数据 X 和参数 ξ] --> B[参数化 SEM<br/>边权 bij ξ]
    B --> C[PCE 谱展开<br/>bij ≈ Σ θ Ψα]
    C --> D[PCT 条件独立检验<br/>发现因果骨架]
    D --> E[打分法 + 自然梯度<br/>精修结构 + 估系数]
    E --> F[动态因果图<br/>+ 置信区间 + 边存在概率]
```

### 关键设计

**1. 参数化结构方程模型：把因果边写成函数而非常数。** 本文不再假设 $X_i = \sum_{j} b_{ij} X_j + \epsilon_i$ 里的 $b_{ij}$ 是常数，而是让它成为操作参数的函数：$X_i = \sum_{j \in PA_i} b_{ij}(\xi) X_j + \epsilon_i$，其中 $\xi \in \mathbb{R}^d$ 是已知可测的运行工况（环境温度、催化剂年龄、原料品质等），服从已知分布 $\mu_\xi$；$b_{ij}(\xi) \in L^2(\Xi)$ 是平方可积的未知函数。一个关键约定是**边集 $E$ 不随 $\xi$ 变，只有边权 $b_{ij}(\xi)$ 随工况变**——这让"结构发现"和"强度建模"解耦：拓扑是稳定的，变化的只是每条边的"调门"。噪声 $\epsilon_i$ 假设互相独立、中心化、亚高斯，且至多一个是高斯的（沿用 LiNGAM 的非高斯可识别条件）。

**2. PCE 谱表示：用正交多项式把函数学出来。** 难点在于 $b_{ij}(\xi)$ 是无穷维对象。本文借助 Wiener–Askey 体系：对常见的 $\mu_\xi$ 都存在一组适配的正交多项式基 $\{\Psi_\alpha(\xi)\}$（高斯对应 Hermite、均匀对应 Legendre、指数对应 Laguerre）。把函数按这组基展开并截断到总阶数 $N_p$：
$$b_{ij}(\xi) \approx \sum_{\alpha \in A_{N_p}} \theta_{ij,\alpha} \Psi_\alpha(\xi), \quad \theta_{ij,\alpha} = \frac{\langle b_{ij}, \Psi_\alpha \rangle_{L^2}}{\langle \Psi_\alpha^2 \rangle_{L^2}}$$
这一步把"学函数"变成"估有限个谱系数 $\theta_{ij,\alpha}$"。理论上对 $s$ 阶可微函数误差以 $C N_p^{-s}$ 多项式衰减，对解析函数（物理系统常见）以 $C\exp(-\gamma N_p^{1/d})$ 指数衰减。当参数维度 $d$ 很大时基的规模 $P=\binom{N_p+d}{d}$ 会爆炸，于是用**双曲截断 (hyperbolic truncation)** 优先保留低阶交互项压缩基的大小。

**3. PCT 条件独立检验：在参数空间上检验，避免边缘抵消。** 标准 CI 检验作用在 $(X_A, X_B, X_Z)$ 的边缘分布上，碰到符号随 $\xi$ 翻转的效应会被边缘抵消而误判独立。本文改为检验**条件协方差函数** $C_{AB|Z}(\xi) := \mathrm{Cov}(X_A, X_B \mid X_Z, \xi)$ 是否处处为零：定义 PCT 条件独立为 $\|C_{AB|Z}\|^2_{L^2(\mu_\xi)} = \mathbb{E}_\xi[C_{AB|Z}(\xi)^2] = 0$。把这个协方差函数也做 PCE 展开后，零假设等价于"谱系数向量为零"。具体做法是先把 $X_A, X_B$ 对交互特征 $\{X_k \Psi_\alpha(\xi)\}$ 回归取残差，残差乘积 $r_A r_B$ 就是 $\xi^{(t)}$ 处的条件协方差信号；构造 $v^{(t)} := r^{(t)}_{A} r^{(t)}_{B}\, \psi(\xi^{(t)})$ 后，多元 CLT 给出 Wald 统计量
$$T_{PCT} = m\, \hat{C}^\top \hat{\Sigma}^{-1}_{\text{reg}} \hat{C} \xrightarrow{d} \chi^2_{df}$$
这个检验当作 CI 预言机塞进 PC 风格的骨架搜索里。

**4. 打分法 + 自然梯度精修：稳健且单步收敛。** 约束法骨架在有限样本下不稳，于是用它做初始化，再做打分优化。定义 PCT-BIC 分数为最小二乘拟合项加 group sparsity 惩罚 $\lambda \|(E,\Theta)\|_0 = \lambda\sum_{i,j} \mathbb{1}\{\|\theta_{ij}\|_2 > 0\}$（按"整条边的系数组"是否非零计数，鼓励稀疏 DAG）。在 DAG 固定时，惩罚项为常数，对 $\Theta$ 的优化退化为每个节点一个最小二乘。由于回归子依赖 $\xi$ 而互相相关，经验 Fisher 信息 $\hat{F}_i = \frac{1}{\sigma_\epsilon^2 m}\Phi_i^\top \Phi_i$ 一般非对角，本文用 Fisher 预条件（自然梯度）更新：
$$\theta_i \leftarrow \theta_i - \eta (\Phi_i^\top \Phi_i + \varepsilon_{\text{reg}} I)^{-1} \Phi_i^\top (\Phi_i \theta_i - x_i)$$
理论上 $0<\eta<2$ 时线性收敛，且当 $\varepsilon_{\text{reg}}=0, \eta=1$ 时**一步达到最小二乘最优**。配合贪心地增删边（只接受保持无环的改动）与 warm start，整套流程既给出最终图与函数关系，还输出因果强度的置信区间和每条边的存在概率。

理论部分给出三个结论：定理 1（可识别性）证明在非高斯噪声等假设下 DAG $G$ 与函数族 $\{b_{ij}(\xi)\}$ 可从 $(X,\xi)$ 联合分布唯一恢复（把 LiNGAM 推广到参数化设定）；定理 2（样本复杂度）给出恢复截断边集所需样本量 $m \gtrsim \frac{\sigma_\epsilon^2}{\gamma \kappa_{N_p}^2}(sP)\log(2n^2P/\delta)$，刻画了样本量随交互特征数 $sP$、设计良态性 $\gamma$、最弱有效边强 $\kappa_{N_p}$ 的依赖；定理 3 即上面自然梯度的线性/单步收敛。

## 实验关键数据

### 主实验表格
在加拿大 Parkland 炼厂化工反应器网络数据集上验证：10,000 个样本、9 个过程变量、11 条经工程原理验证的真实因果边、3 个参数不确定性源（传热系数 $\xi_1$、反应速率常数 $\xi_2$、产率因子 $\xi_3$）。与 23 个 SOTA 方法横向对比（参数 $N_p=4, \alpha_{sig}=0.05, \lambda=1, B=200$）。

| 方法 | TP | FP | FN | Prec. | Recall | F1 | SHD |
|------|----|----|----|-------|--------|-----|-----|
| ICA-LiNGAM | 1 | 14 | 10 | 0.067 | 0.091 | 0.077 | 24 |
| DirectLiNGAM | 2 | 13 | 9 | 0.133 | 0.182 | 0.154 | 22 |
| LiNGAM | 5 | 10 | 6 | 0.333 | 0.455 | 0.385 | 16 |
| NOTEARS | 5 | 5 | 6 | 0.500 | 0.455 | 0.476 | 11 |
| FCI | 5 | 4 | 6 | 0.556 | 0.455 | 0.500 | 10 |
| GES / GIES | 6 | 5 | 5 | 0.545 | 0.545 | 0.545 | 10 |
| CAM / GraNDAG / SAM | 8 | 6 | 3 | 0.571 | 0.727 | 0.640 | 9 |
| **PCT-CD** | **10** | **1** | **1** | **0.909** | **0.909** | **0.909** | **2** |

PCT-CD 正确识别 11 条真实边中的 10 条，仅 1 个假阳和 1 个假阴，SHD=2，F1=90.9%，相比次优方法 (CAM/GraNDAG/SAM 的 64.0%) 接近翻倍。

### 不确定性量化表
PCT-CD 独有的能力是把每条边的强度表示成 $\xi$ 的连续函数，并给出 95% 置信区间、bootstrap 存在概率与主导参数：

| 边 | 均值 | 95% CI | Boot 概率 | 主导 $\xi$ |
|----|------|--------|-----------|-----------|
| $X_1 \to X_2$ | 0.642 | [0.411, 0.873] | 0.95 | $\xi_1$ (传热) |
| $X_1 \to X_3$ | 0.465 | [0.305, 0.669] | 0.88 | $\xi_2$ (反应) |
| $X_7 \to X_9$ | 0.452 | [0.317, 0.632] | 0.95 | $\xi_3$ (产率) |
| $X_6 \to X_9$ | 0.109 | [0.034, 0.156] | 0.93 | $\xi_2$ (反应) |

### 关键发现
- 最强边 $X_1\to X_2$ 的强度随传热条件变化超过 100%，而弱边变化更受约束——证明静态权重确实丢失了关键信息。
- 三个参数各管一类路径：传热 $\xi_1$ 主导进料与热控通路、反应 $\xi_2$ 控制中间转化、产率 $\xi_3$ 决定产品质量路径，与工程直觉吻合。
- 按方法类别看：约束法 (PC/FCI) 精度尚可但召回低（保守漏边）；打分法 (GES/NOTEARS) 受静态图假设根本性限制；传统 LiNGAM 在参数变化下严重模型误设 (38.5% F1)，ICA-LiNGAM 最差 (7.7%)。

## 亮点与洞察
- **范式转变到位**：从"静态图"到"动态函数"的提法直击工业因果发现的真实痛点，而且不是另起炉灶，而是把 PCE 这套成熟的不确定性量化工具第一次系统地嫁接到因果发现上。
- **理论闭环**：可识别性 + 有限样本复杂度 + 收敛性三个定理齐全，自然梯度在线性高斯视角下单步收敛是个漂亮的工程性质。
- **CI 检验抓住了本质矛盾**：把"边缘独立性检验在符号翻转效应下失效"这个隐蔽 failure mode 显式化，并用条件协方差函数的 $L^2$ 范数干净地解决，是方法里最有洞察力的一步。
- **可解释性强**：输出的不是一张图，而是"每条边随哪个工况怎么变 + 多大置信度存在"，这对安全攸关的工业控制是刚需。

## 局限与展望
- **实验单一**：只在一个化工反应器数据集（9 变量、11 边）上验证，规模偏小、领域单一，缺少跨领域和大规模 ($n\gg 9$) 的实证，泛化性存疑。
- **线性 SEM 假设**：模型核心是边权随参数变化的**线性** SEM，非线性因果机制（变量间本身非线性）仍未覆盖，作者也把它列为未来工作。
- **假设较强**：要求因果充分性（无未观测混杂）、faithfulness、$\xi$ 分布已知或可经验构造正交基、至多一个高斯噪声——工业现场这些条件未必都满足，尤其无混杂假设。
- **维度灾难仍在**：基规模 $P=\binom{N_p+d}{d}$ 随参数维度 $d$ 组合爆炸，双曲截断只是缓解，高维参数空间下样本复杂度 $sP$ 项会很重。
- **展望**：作者提出扩展到未观测混杂、更复杂非线性交互，以及在线自适应控制场景的应用。

## 相关工作与启发
- **因果发现三大流派**：约束法 (PC/FCI/RFCI)、打分法 (GES/FGES/NOTEARS/DAG-GNN/RL-BIC)、函数式因果模型 (LiNGAM/DirectLiNGAM/VAR-LiNGAM/ANM/PNL)——本文定位为对它们的补充而非替代，可识别性证明直接推广自 LiNGAM。
- **贝叶斯因果发现**：DiBS、BCD Nets 量化的是图/参数的后验（认知不确定性），与本文的"参数不确定性"正交互补。
- **时变/上下文相关因果结构**：Song et al.、Huang et al. 的时变因果工作是最近的邻居，但本文用 PCE 显式把边权写成参数函数的做法是新的切入角度。
- **PCE 本身**：Wiener 提出、Xiu & Karniadakis 推广，在工程不确定性量化、敏感性分析、过程监控里成熟，但据作者所知本文是首次系统用于因果发现——"把 A 领域成熟工具嫁接到 B 领域核心难题"是值得借鉴的研究范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首次把 PCE 系统引入因果发现，"因果边=参数函数"的范式提法清晰、切口准，配套可识别性证明，理论与问题动机都新。
- **实验充分度**: ⭐⭐⭐ 对比 23 个 baseline 很全面、提升显著，但只在单个小规模化工数据集上验证，缺合成数据系统性扫参与跨领域大规模实证，说服力受限。
- **写作质量**: ⭐⭐⭐⭐ 动机-方法-理论-实验逻辑顺畅，把"边缘抵消"这类隐蔽 failure mode 讲清楚，公式与定理组织清晰。
- **价值**: ⭐⭐⭐⭐ 对工业过程控制、根因分析、预测性维护这类安全攸关场景价值明确，输出的不确定性感知因果函数是实用刚需，方法范式可迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A Polynomial Chaos Framework for Causal Discovery in Nonlinear Uncertain Systems](../../CVPR2026/causal_inference/a_polynomial_chaos_framework_for_causal_discovery_in_nonlinear_uncertain_systems.md)
- [\[ICLR 2026\] Coarse-to-Fine Learning of Dynamic Causal Structures](coarse-to-fine_learning_of_dynamic_causal_structures.md)
- [\[ICLR 2026\] Theoretical Guarantees for Causal Discovery on Large Random Graphs](theoretical_guarantees_for_causal_discovery_on_large_random_graphs.md)
- [\[ICLR 2026\] Query-Specific Causal Graph Pruning under Tiered Knowledge](query-specific_causal_graph_pruning_under_tiered_knowledge.md)
- [\[ICLR 2026\] Efficient and Sharp Off-Policy Learning under Unobserved Confounding](efficient_and_sharp_off-policy_learning_under_unobserved_confounding.md)

</div>

<!-- RELATED:END -->
