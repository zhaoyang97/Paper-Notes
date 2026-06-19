---
title: >-
  [论文解读] Optimal Transport under Group Fairness Constraints
description: >-
  [ICML 2026 Spotlight][AI安全][群体公平性] 本文把"群体公平性"显式编码为一个 $K_s \times K_w$ 的组间匹配概率目标 $\mathbf{F}$，提出 **FairSinkhorn** 精确求解、**惩罚式 OT** 凸松弛、以及 **双层成本学习** 三种方案，分别给出有限样本复杂度 $O(1/\sqrt{n})$ 和 fairness 偏差界 $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$，在合成与半合成（约会 app）数据集上勾画出"代价 - 公平性"权衡前沿。
tags:
  - "ICML 2026 Spotlight"
  - "AI安全"
  - "群体公平性"
  - "最优传输"
  - "Sinkhorn"
  - "双层优化"
  - "成本学习"
---

# Optimal Transport under Group Fairness Constraints

**会议**: ICML 2026 Spotlight  
**arXiv**: [2601.07144](https://arxiv.org/abs/2601.07144)  
**代码**: https://github.com/LinusBleistein/fair_ot (有)  
**领域**: AI安全 / 算法公平性 / 最优传输  
**关键词**: 群体公平性、最优传输、Sinkhorn、双层优化、成本学习

## 一句话总结
本文把"群体公平性"显式编码为一个 $K_s \times K_w$ 的组间匹配概率目标 $\mathbf{F}$，提出 **FairSinkhorn** 精确求解、**惩罚式 OT** 凸松弛、以及 **双层成本学习** 三种方案，分别给出有限样本复杂度 $O(1/\sqrt{n})$ 和 fairness 偏差界 $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$，在合成与半合成（约会 app）数据集上勾画出"代价 - 公平性"权衡前沿。

## 研究背景与动机

**领域现状**：算法匹配（招生、招聘、约会、肾移植、城市资源分配）越来越多地由中心化算法决定，最优传输（OT）因其在经济学和社会科学中的良好建模性质成为主流工具。给定两个分布 $\mu, \eta$ 和成本 $c$，熵正则化 OT 求解 $\min_\pi \int c(x,y)\,d\pi + \varepsilon \mathbf{KL}(\pi|\mu\otimes\eta)$，并用 Sinkhorn 算法高效迭代。

**现有痛点**：当特征 $X, Y$ 与敏感属性 $S, W$（性别、种族、社会经济地位）强相关时，标准 OT 会产出"块对角"式高度同质的匹配——例如把高收入学生几乎全部分配到精英学校。已有 fairness × matching 文献几乎集中在**个体公平**（相似个体得相似结果）或**特定问题的参与配额**（肾移植 KPD），缺乏一个针对 OT 本身、可由中央规划者灵活指定的"组间匹配概率"框架。

**核心矛盾**：(a) 现有 fair-OT 工作大多把 OT 当作下游公平预测的工具（如 Wasserstein 重心投影），而不是研究**传输计划本身**的公平性；(b) 精确公平往往代价巨大（"price of fairness"），需要一个能精细控制 trade-off 的松弛框架；(c) 把公平目标"印"进匹配里之后，能否在新样本上**复用**也是开放问题。

**本文目标**：(1) 形式化"组间匹配概率 = 目标 $\mathbf{F}$"这一新型群体公平定义；(2) 给出精确解算法 + 两类松弛方法；(3) 给松弛方法配套有限样本理论保证。

**切入角度**：观察到公平约束 $\pi_{SW}(s,w) = \mathbf{F}_{sw}$ 在传输计划 $\mathbf{\Pi}$ 上是**线性**的，因此 Lagrange 对偶仍可写成 Sinkhorn 风格的乘性形式；同时熵正则保证唯一性与可微性，让"通过学成本来诱导公平"的双层优化变成良定问题。

**核心 idea**：用一个"组间概率目标矩阵 $\mathbf{F}$"统一表达限制同源婚配、最低少数族裔配额、4/5 雇佣规则、欧盟董事会 40% 性别配额等真实规则，并提供从精确到松弛的算法谱系。

## 方法详解

### 整体框架

标准熵正则 OT 在敏感属性与特征强相关时会产出"块对角"匹配，把弱势群体几乎全锁死在原属圈层里。本文要做的，是让规划者能用一个组间匹配概率矩阵 $\mathbf{F}$ 直接指定"哪类人该以多大比例匹配哪类资源"，再把它转译成传输计划上的线性约束。输入是两份带敏感属性的样本 $(\mathbf{x}_i, \mathbf{s}_i)_{i=1}^n \sim \mu$、$(\mathbf{y}_j, \mathbf{w}_j)_{j=1}^m \sim \eta$、成本矩阵 $\mathbf{C}_{ij} = c(\mathbf{x}_i, \mathbf{y}_j)$、熵正则 $\varepsilon$ 与公平目标 $\mathbf{F} \in \Pi(\mathbf{p}, \mathbf{q})$；输出是传输计划 $\mathbf{\Pi} \in \mathbb{R}_+^{n\times m}$，行列和满足边际约束，同时让组间质量 $\sum_{i: s_i=s, j: w_j=w} \mathbf{\Pi}_{ij}$ 尽量贴近 $\mathbf{F}_{sw}$。围绕这一目标，作者从"硬约束的精确解"逐步松弛到"可复用的间接几何重塑"，给出三条互补的求解路径。

### 关键设计

**1. FairSinkhorn：把精确公平嫁接进 Sinkhorn 的乘性结构**

第一条路径直接把公平当硬约束求精确解，针对的痛点是"规划者要求一个分毫不差满足 $\mathbf{F}$ 的方案"。约束写成 $\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] = \mathbf{F}_{sw}$，其中 $\mathbf{B}_{sw}$ 是标记样本组对 $(s,w)$ 的 0/1 矩阵。这个约束的关键性质是**线性**的，因此引入 Lagrange 对偶变量 $\mathbf{h} \in \mathbb{R}^{K_s \times K_w}$ 后，最优解仍保持 Sinkhorn 的乘性形式 $\mathbf{\Pi} = \text{diag}(e^{\mathbf{f}/\varepsilon})(\mathbf{K} \odot \mathbf{H}) \text{diag}(e^{\mathbf{g}/\varepsilon})$：$\mathbf{K} = e^{-\mathbf{C}/\varepsilon-1}$ 是标准 Sinkhorn 核，而 $\mathbf{H} = \sum_{sw} e^{h_{sw}/\varepsilon} \mathbf{B}_{sw}$ 是一张按 $(s,w)$ 块常值的"公平系数"矩阵。算法因此只需在标准的行列归一 $\mathbf{u}, \mathbf{v}$ 之外，多插入一步组级再投影 $\mathbf{L}^{(t+1)} \leftarrow \mathbf{F} \oslash \Phi(\mathbf{u}^{(t+1)}, \mathbf{v}^{(t+1)})$ 来更新 $\mathbf{H}$，其中 $\Phi$ 把当前传输的组级质量加总。因为只是给定点迭代加了一步块级归一，复杂度与原 Sinkhorn 同阶、收敛速度几乎不变，可以 drop-in 替换现有 pipeline。它给出的是"perfect fairness"基线，但代价是强行抹平所有组间几何信息，传输成本可能显著上升——这恰恰是后两条松弛路径要解决的问题。

**2. 惩罚式 OT：一个凸的"代价 - 公平性"旋钮**

很多场景并不需要分毫不差的公平，只要"足够接近"即可，于是第二条路径把硬约束换成平方惩罚 $\mathcal{L}_\mathbf{F}(\mathbf{\Pi}) = \sum_{(s,w)} (\text{Tr}[\mathbf{\Pi}^\top \mathbf{B}_{sw}] - \mathbf{F}_{sw})^2$，写进目标 $\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}] + \varepsilon \mathbf{KL}(\mathbf{\Pi}) + \lambda \mathcal{L}_\mathbf{F}(\mathbf{\Pi})$，其中 $\lambda$ 就是规划者手里的"公平强度旋钮"。由于 $\mathcal{L}_\mathbf{F}$ 是凸的，整体仍强凸、唯一解，且 $\mathbf{\Pi}$ 会随 $\lambda$ 沿一条凸曲线从 Sinkhorn（$\lambda=0$）平滑滑向 FairSinkhorn（$\lambda\to\infty$），规划者可按可接受代价就地选点。求解用**广义条件梯度**：每次迭代把惩罚项围绕当前 $\mathbf{\Pi}^t$ 线性化，得到修正成本 $\mathbf{C} + \nabla \mathcal{L}_\mathbf{F}(\mathbf{\Pi}^t)$，用标准 Sinkhorn 解这个子问题作为搜索方向，再线搜索取凸组合保证下降。理论上作者证明 $\mathbb{E}|m^\star(\mu_n, \eta_n) - m^\star(\mu, \eta)| \lesssim 1/\sqrt{n}$，与标准熵正则 OT 同阶——也就是说加 fairness 惩罚**不损失统计效率**；证明思路是把样本最优值 $m_n^\star$ 夹在两个对线性化子问题求值的随机量之间，再接上 rakotomamonjy2015 / genevay2019 / rigollet2022 的工具链。这条路径胜在凸、可复现、可解释，代价是每批新样本都得重解一次。

**3. 双层成本学习：学一个"诱导公平"的几何并能复用**

前两条都在直接雕刻 $\mathbf{\Pi}$，每来一批新数据就要重算；第三条路径换了思路——不动 $\mathbf{\Pi}$，而是学一个参数化成本 $c_\theta$，让它诱导出的熵正则 OT 解 $\mathbf{\Pi}_\varepsilon(c_\theta)$ 自然满足公平目标，学一次便能缓存复用。它写成双层优化 $\min_\theta \mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) + \frac{1}{\lambda} \mathscr{D}(c_\theta, c_\text{base})$，约束内层 $\mathbf{\Pi}_\varepsilon(c_\theta) = \arg\min_\mathbf{\Pi} \text{Tr}[\mathbf{\Pi}^\top \mathbf{C}_\theta] + \varepsilon \mathbf{KL}(\mathbf{\Pi})$，其中 $\mathscr{D}$ 约束学到的成本别偏离基准成本（如平方欧氏）太远。内层熵正则 OT 因强凸而有唯一解，整个双层目标因此良定，梯度通过迭代微分或隐式微分回传。成本有两种参数化：**Mahalanobis** $c_\mathbf{M}(x,y) = (x-y)^\top \mathbf{M} (x-y)$ 可解释，能直接读出原始空间里哪些特征方向被强调或压制（即不公平的"元凶方向"）；**神经成本** $c_\theta(x,y) = \|\phi_{\theta_1}(x) - \phi_{\theta_2}(y)\|_2^2$ 更灵活，能做环状等非线性几何变换。代价是泛化保证较弱：作者证明对参数族任一 $\theta$，新样本上的 fairness 偏差满足 $\sup_\theta \mathbb{E}[|\mathcal{L}_\mathbf{F}(\mathbf{\Pi}_\varepsilon(c_\theta)) - \mathcal{L}_\mathbf{F}(\pi_\varepsilon^\star(c_\theta))|] \lesssim \exp(5R_\Theta/\varepsilon)/\sqrt{n}$，其中 $R_\Theta$ 是成本族上界，该界随 $\varepsilon$ 减小指数爆炸。

### 损失函数 / 训练策略

三条路径的训练负担递增：FairSinkhorn 只需迭代 $T$ 次定点更新，无可学参数；惩罚式 OT 是凸问题，由 $\lambda$ 单调控制 fairness 强度；成本学习版用 SGD/Adam 优化 $\theta$，每步内层跑一次熵正则 Sinkhorn，正则项 $\mathscr{D}(c_\theta, c_\text{base})$ 对 Mahalanobis 取 $\|\mathbf{M} - \mathbf{I}\|_F^2$、对神经成本取网络权重的 $\ell_2$ 范数。$\varepsilon$ 与 $\lambda$ 通过 grid 扫描描出完整的 trade-off 曲线。

## 实验关键数据

### 主实验

**合成实验**：Gaussians（两组学生 / 学校按高斯混合分布生成，privileged 靠近 elite）和 Circles（少数群体在半径 2 圆环上）。目标 $\mathbf{F} = \begin{bmatrix} 0.20 & 0.30 \\ 0.28 & 0.22 \end{bmatrix}$，即 60% 弱势学生匹配精英学校。

| 方法 | 公平违反 $\mathcal{L}_\mathbf{F}$ | 传输代价（相对 Sinkhorn） | 备注 |
|------|------|------|------|
| Sinkhorn (vanilla) | 高（块对角偏差大） | 0（基准） | 完全忽略公平 |
| Sinkhorn + 调大 $\varepsilon$ | 仍高 | 上升 | 仅靠熵正则**无法**逼近 $\mathbf{F}$ |
| FairSinkhorn | $\approx 0$（精确） | 显著上升 | 完美公平，代价最高 |
| Penalized OT（变 $\lambda$） | 平滑插值 | 平滑插值 | 凸曲线，trade-off 最灵活 |
| Cost learning - Mahalanobis | 中（Circles 上 $\ge 10^{-2}$） | 中 | 线性变换不够，受限于几何 |
| Cost learning - MLP | 低 | 中（Gaussians 上与 Penalized 同档） | 非线性几何，Circles 上才追得上 Penalized |

**半合成约会 App 实验**：Kaggle 数据集子采样匹配美国人口统计，敏感属性为 7 档收入，feasible matching 由性取向决定。目标 $\mathbf{F}_{sw} = \mathbb{P}(S=s_i) \mathbb{P}(W=w_j)$（即独立 = 完全打破收入同源婚配）。结果与合成实验一致：trade-off 曲线呈凸形，Penalized 与 Cost Learning 几乎重合，扩展性良好到高维 + 多组。

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| 仅调 $\varepsilon$（vanilla OT） | 公平违反**不收敛**到 $\mathbf{F}$ | 熵正则只能让 plan 更"模糊"，不会自动趋向 $\mathbf{F}$ |
| $\lambda$ 由小到大（Penalized） | 沿 Sinkhorn → FairSinkhorn 的凸曲线 | 验证了 trade-off 可控且连续 |
| Mahalanobis vs MLP（Cost Learning） | Circles 上 MLP 显著更低 | 线性几何对环状分布无能为力 |
| Train cost → reuse on test set | 推理时间比 Penalized 快 1-2 个数量级，公平损失略增（generalization gap MLP > Mahalanobis） | 学到的成本可复用，新样本上 fairness 仍远好于 vanilla OT |

### 关键发现
- **熵正则不能替代公平约束**：再大的 $\varepsilon$ 也不会让 vanilla OT 自动靠近 $\mathbf{F}$，证实显式公平机制的必要性。
- **Penalized 几乎"压倒性"地最灵活**：因为它在所有耦合 $\Pi$ 上搜索，而 Cost Learning 受限于"熵正则 OT 解 + 参数成本族"这一更窄子集；但代价是每批新样本都要重算。
- **Cost Learning 的核心价值是"可迁移"**：学一次、用多次，特别适合实时匹配系统（招聘、约会）；并且 Mahalanobis 矩阵的特征值/向量本身就是公平诊断工具。
- **理论与实验闭环**：Penalized 的 $O(1/\sqrt{n})$ 和 Cost Learning 的 $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$ 都在实验中被"反向验证"——样本量增大时新样本上的 fairness gap 确实收窄。

## 亮点与洞察
- **fairness target 这一抽象非常通用**：4/5 雇佣规则、欧盟 40% 女性董事配额、法国奖学金最低配额都可以表达成一个 $\mathbf{F}$，把法规直接接入算法。这是文中我最欣赏的"概念性贡献"。
- **公平约束是线性的 → Sinkhorn 加一步就够**：FairSinkhorn 的推导异常干净，只在 $\mathbf{u}, \mathbf{v}$ 之外多维护一个块常值矩阵 $\mathbf{T}^{(t)}$，几乎零额外开销，可直接 drop-in 替换现有 OT pipeline。
- **凸惩罚不损失统计效率**：证明加 fairness 惩罚后样本复杂度仍是 $O(1/\sqrt{n})$，这一结论对整个"约束 OT"领域都有独立价值。
- **可迁移的 trick**：双层"学成本以诱导某种 plan 性质"的范式可以推广到 fairness 以外——比如把 $\mathcal{L}_\mathbf{F}$ 换成稀疏性、单调性、域不变性等目标，直接得到一族"约束诱导式 OT"算法。

## 局限与展望
- **只覆盖群体公平**：个体公平（"相似个体得相似匹配"）需要不同形式化，不能简单套用 $\mathbf{F}$。
- **没有讨论 fairness-bias trade-off**：当 $\mathbf{F}$ 本身被 mis-specified 时（规划者对真实人口分布判断错误），三种方法的鲁棒性都未量化。
- **下游效用未建模**：把学生分到大学只是匹配，后续工资 / 学业成就才是真正关心的福利指标；论文把这一步留给未来工作。
- **复杂度依赖 $\exp(5R_\Theta/\varepsilon)$**：成本学习的偏差界随 $\varepsilon$ 减小**指数爆炸**，所以小正则区间下的泛化保证比较弱；实操上需要在 fairness、cost、$\varepsilon$ 三角中谨慎调参。
- **改进思路**：(a) 把 $\mathbf{F}$ 本身做成可学的（让规划者只指定上界 / 不等式约束）；(b) 引入因果图，把 sensitive attribute → feature 的相关性显式建模，得到 counterfactual 版本的公平 OT；(c) 与 stable matching（Gale-Shapley）混合，得到既稳定又群体公平的算法。

## 相关工作与启发
- **vs Gale-Shapley fair variants（karni2021、devic2023）**：他们处理迭代式个体匹配 + 个体公平；本文是连续质量分裂式 OT + 群体配额，二者面向不同应用域（个体偏好驱动的市场 vs 中心规划下的资源分配）。
- **vs KPD fairness（ashlagi2014、dickerson2014、zhang2025）**：KPD 的公平基本上是"参与配额"，仅适合特定 integer programming 结构；本文把公平做成一个通用的 $\mathbf{F}$ 矩阵，并适配 OT 这一更广的连续松弛框架。
- **vs Wasserstein-barycenter fairness（gouic2020、chzhen2020、divol2024）**：他们用 OT 当作公平**预测**的工具（重心投影 → 修正模型输出），公平的是预测；本文公平的是**传输计划本身**，二者方向正交、可以叠加。
- **vs nguyen2024（fair Wasserstein barycenter）**：他们对 sliced-Wasserstein 距离施加近似相等约束；本文施加的是 explicit mass constraint 在 sensitive subgroup 之间，语义更直接、更易和监管规则对应。
- **vs 约束 OT 一脉（courty2016、blondel2018、liu2022、korman2015）**：本文是约束 OT 在"群体公平"这一具体语义上的落地，并贡献了惩罚式公平 OT 的样本复杂度结果，对整个约束 OT 理论补了一块。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"组间匹配概率目标 $\mathbf{F}$"作为一类全新的公平定义引入 OT，并首次系统给出三种求解路径与对应理论保证
- 实验充分度: ⭐⭐⭐⭐ 合成 + 半合成（约会 app）覆盖了二维 / 高维、两组 / 七组，trade-off 曲线与可迁移性都得到验证；但缺少真实关键应用（招聘、招生）的端到端 case study
- 写作质量: ⭐⭐⭐⭐⭐ 故事线（school → fairness target → exact → penalized → cost learning）层层递进，定义、命题、算法、定理交替推进，pacing 几乎完美
- 价值: ⭐⭐⭐⭐⭐ 给"算法治理 + 配额监管"提供了一个数学清晰、计算高效、可复用的工具箱，监管者、平台和研究者都能受益

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Bypassing the Transport Plan: Dynamic Reweighting for Out-of-Distribution Detection with Optimal Transport](../../CVPR2026/ai_safety/bypassing_the_transport_plan_dynamic_reweighting_for_out-of-distribution_detecti.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)
- [\[ICML 2026\] Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)
- [\[ICLR 2026\] Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](../../ICLR2026/ai_safety/optimal_transport-induced_samples_against_out-of-distribution_overconfidence.md)
- [\[AAAI 2026\] Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](../../AAAI2026/ai_safety/truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)

</div>

<!-- RELATED:END -->
