---
title: >-
  [论文解读] Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles
description: >-
  [ICML2026][优化/理论][双层优化] 本文为光滑「非凸-强凸」双层优化在标准（确定性 / 随机）一阶 oracle 下首次给出与条件数 $\kappa$ 强相关的复杂度下界——确定性情形 $\Omega(\kappa^{3/2}\epsilon^{-2})$、随机情形 $\Omega(\kappa^{5/2}\epsilon^{-4})$，证明双层问题在本质上比单层非凸和 min-max 优化更难，并暴露出现有上界与下界之间巨大的 $\kappa$ 幂次缺口。
tags:
  - "ICML2026"
  - "优化/理论"
  - "双层优化"
  - "一阶 oracle"
  - "复杂度下界"
  - "zero-chain"
  - "条件数"
---

# Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles

**会议**: ICML2026  
**arXiv**: [2511.19656](https://arxiv.org/abs/2511.19656)  
**代码**: 无（纯理论论文）  
**领域**: 优化理论 / 双层优化 / 复杂度下界  
**关键词**: 双层优化、一阶 oracle、复杂度下界、zero-chain、条件数

## 一句话总结
本文为光滑「非凸-强凸」双层优化在标准（确定性 / 随机）一阶 oracle 下首次给出与条件数 $\kappa$ 强相关的复杂度下界——确定性情形 $\Omega(\kappa^{3/2}\epsilon^{-2})$、随机情形 $\Omega(\kappa^{5/2}\epsilon^{-4})$，证明双层问题在本质上比单层非凸和 min-max 优化更难，并暴露出现有上界与下界之间巨大的 $\kappa$ 幂次缺口。

## 研究背景与动机

**领域现状**：双层优化（bilevel optimization）形如 $\min_{\mathbf{x}} H(\mathbf{x}):=f(\mathbf{x};\mathbf{y}^*(\mathbf{x}))$，其中 $\mathbf{y}^*(\mathbf{x})=\arg\min_{\mathbf{y}} g(\mathbf{x};\mathbf{y})$，广泛出现在元学习、强化学习、超参数优化和联邦学习中。本文聚焦最常被分析的「非凸-强凸」设定：下层 $g$ 关于 $\mathbf{y}$ 是 $\mu$-强凸且光滑，上层 $f$ 光滑但可能非凸。近年大量工作研究其**上界**（收敛速度），既包括依赖二阶信息（Hessian / Jacobian 向量积）的 AID / ITD 类方法，也包括只用一阶 oracle 的 penalty-based、value-function 等"全一阶"算法。

**现有痛点**：上界已被研究得很透，但**下界**进展严重滞后。难点在于双层结构内在的复杂性——天真构造出的 hard instance 往往得到"平凡"下界，强不过经典的单层下界 $\Omega(\epsilon^{-2})$，无法刻画出对条件数 $\kappa=L_g/\mu$ 和精度 $\epsilon$ 的真实依赖。

**核心矛盾**：已有的几个下界都绕开了"标准一阶 oracle"这个最自然的设定。Ji and Liang (2023) 的下界依赖二阶 oracle，且要求超目标 $H(\mathbf{x})$ 是凸/强凸（对一般双层问题过于受限）；Dagréou et al. (2024) 的有限和下界 $\Omega(n+\sqrt{n}\epsilon^{-2})$ 完全不含 $\kappa$；Kwon et al. (2024) 的下界建立在一种 $\mathbf{y}^*$-aware oracle 上——这种 oracle 能直接返回逼近真解 $\mathbf{y}^*$ 的估计，实际上把问题退化成了类单层优化。**在直接访问 $f,g$ 的标准（随机）一阶 oracle 下，双层优化的真实下界一直是 open problem。**

**本文目标**：在不借助二阶信息、不假设 $H$ 凸性、不依赖 $\mathbf{y}^*$-aware oracle 的前提下，给出依赖 $\kappa$ 和 $\epsilon$ 的非平凡一阶复杂度下界，回答"双层到底比单层 / min-max 难多少"。

**切入角度**：作者借鉴 Nesterov 与 Carmon et al. (2020) 经典的 worst-case 构造思想——用满足 **zero-chain** 性质的 hard instance 强迫任意 zero-respecting 一阶算法只能逐坐标"激活"变量，从而把"必须发现 $T$ 个坐标"翻译成"至少 $T$ 次 oracle 调用"。

**核心 idea**：把强凸侧用三对角 Laplacian 矩阵 $A$ 编码、非凸侧用 Carmon 的 $\Psi/\Phi$ 硬函数编码，**让上层和下层的 zero-chain 串联起来**，使坐标激活速度被强凸下层的条件数额外拖慢，最终把 $\kappa$ 的幂次"叠"进下界。

## 方法详解

这是一篇纯理论论文，"方法"即**hard instance 的构造与下界证明**。整体思路是：先建立一套通用的下界论证框架（zero-chain + zero-respecting 算法类），再为确定性和随机两种 oracle 分别设计满足函数类 $\mathcal{F}(L_f,L_g,\mu,\Delta)$ 的最坏实例，让任意一阶算法在该实例上必须付出指定数量的 oracle 调用。

### 整体框架

下界论证遵循经典三步：**(1) 定义算法类**——把所有现有一阶双层算法（penalty、primal-dual、value-function、barrier 等单/双循环方法）统一抽象为满足 **zero-respecting** 性质的迭代（Definition 4）：第 $t+1$ 步的 $\mathbf{x}^{t+1},\mathbf{y}^{t+1}$ 的支撑集只能落在历史迭代点及其上/下层梯度支撑集的并集里；**(2) 构造 hard instance**——设计 $\{f,g\}$ 使其落在标准函数类 $\mathcal{F}(L_f,L_g,\mu,\Delta)$ 内，同时整体是一条 zero-chain，于是从 $\mathbf{x}=\mathbf{0}$ 出发每次迭代至多激活一个新坐标；**(3) 翻译成下界**——证明要让 $\|\nabla H(\mathbf{x})\|_2<\epsilon$ 必须激活足够多坐标 $T$，而 $T$ 被设计成正比于 $\kappa^{3/2}\epsilon^{-2}$（确定性）或 $\kappa^{5/2}\epsilon^{-4}$（随机），即得下界。

构造的核心数学对象是上层 $f$ 与下层 $g$ 沿坐标块 $\mathbf{y}^{(0)},\dots,\mathbf{y}^{(T)}$ 的耦合：

$$f(\mathbf{x};\widetilde{\mathbf{y}})=\sum_{i=1}^{T}\frac{\lambda^2 L_f}{L}\Big[\Psi\big(-\tfrac{C_l}{\lambda}y^{(i-1)}_n\big)\Phi\big(-\tfrac{C_r}{\lambda}y^{(i)}_1\big)-\Psi\big(\tfrac{C_l}{\lambda}y^{(i-1)}_n\big)\Phi\big(\tfrac{C_r}{\lambda}y^{(i)}_1\big)\Big]$$

$$g(\mathbf{x};\widetilde{\mathbf{y}})=\sum_{i=0}^{T}\Big[\tfrac{L_g n^2}{2(4n^2+1)}(\mathbf{y}^{(i)})^\top\big(\tfrac{1}{n^2}I_n+A\big)\mathbf{y}^{(i)}-L_g(\mathbf{b}_x^{(i)})^\top\mathbf{y}^{(i)}\Big]$$

下层 $g$ 是关于 $\mathbf{y}$ 的二次强凸函数，其最优解 $\mathbf{y}^*(\mathbf{x})$ 通过线性系统隐式依赖 $\mathbf{x}$；上层 $f$ 只通过下层最优解 $\mathbf{y}^*$ 的端点坐标 $y_n^{(i-1)},y_1^{(i)}$ 发生耦合，从而把"求下层最优"这件难事嵌进了 zero-chain 的传播路径。

### 关键设计

**1. Zero-respecting 算法类 + zero-chain 还原：把"迭代次数"逼成"坐标激活次数"**

下界要对*所有*一阶算法成立，因此必须先把算法的能力上限刻画清楚。作者采用 zero-chain（Definition 5）：函数 $f$ 满足当 $\mathrm{supp}(\mathbf{x})\subset\{1,\dots,i-1\}$ 时 $\mathrm{supp}(\nabla f(\mathbf{x}))\subset\{1,\dots,i\}$，即梯度每次最多把支撑集向前推进一个坐标。配合 zero-respecting 算法只能在历史梯度支撑集张成的子空间里更新，从 $\mathbf{x}=\mathbf{0}$ 出发 $t$ 步后必有 $\mathrm{supp}(\mathbf{x}^t)\subset\{1,\dots,t\}$。于是只要证明"达到 $\epsilon$-稳定点需激活至少 $T$ 个坐标"，就立刻得到 $\Omega(T)$ 的 oracle 调用下界。这一步把一个连续优化的复杂度问题，转化成纯组合的"坐标逐个点亮"计数问题。

**2. 三对角 Laplacian 矩阵 $A$：用强凸下层把 $\kappa$ 的幂次"灌"进链条**

下层强凸部分采用一维离散 Laplacian 三对角矩阵 $A$（对角 $2$、次对角 $-1$，首尾对角为 $1$），它半正定且 $\|A\|_2\le 4$。$A$ 的三对角结构天然保持 zero-chain：若 $\mathrm{supp}(\mathbf{x})\subset\{1,\dots,i-1\}$ 则 $A\mathbf{x}\subset\{1,\dots,i\}$，乘一次 $A$ 至多激活一个新坐标。把下层写成 $\tfrac{L_g n^2}{2(4n^2+1)}\mathbf{y}^\top(\tfrac{1}{n^2}I_n+A)\mathbf{y}-L_g\mathbf{b}^\top\mathbf{y}$ 后，求其精确最优解 $\mathbf{y}^*$ 等价于解一个三对角线性系统，而该系统的"信息传播速度"被条件数 $\kappa=L_g/\mu$ 严重拖慢——下层越病态，越需要更多 oracle 调用才能让坐标沿链条传到末端。这正是 $\kappa$ 进入下界的来源，也是双层下界 $\Omega(\kappa^{3/2}\epsilon^{-2})$ 比 min-max 的 $\Omega(\sqrt{\kappa}\epsilon^{-2})$ 多出 $\kappa$ 因子的根本原因。

**3. Carmon 的 $\Psi/\Phi$ 硬函数：在非凸上层制造梯度大、却必须逐坐标推进的耦合**

上层非凸性由 Carmon et al. (2020) 的两个分量函数承担：$\Psi(x)$ 在 $x\le 1/2$ 处恒为 $0$（含各阶导数），$x>1/2$ 时取 $\exp(1-\tfrac{1}{(2x-1)^2})$；$\Phi(x)=\sqrt{e}\int_{-\infty}^x e^{-t^2/2}dt$。它们满足关键引理：当 $x\ge 1,|y|<1$ 时 $\Psi(x)\Phi'(y)>1$（保证未激活坐标处梯度足够大、离稳定点还很远），且 $\Psi(0)=\Psi'(0)=0$（保证 zero-chain：相邻坐标块只有当前一块被"点亮"后才会贡献梯度）。$\Psi,\Psi',\Phi,\Phi'$ 全部非负且有界，这一有界性是让构造合法落进函数类 $\mathcal{F}(L_f,L_g,\mu,\Delta)$（光滑、梯度有界、超目标降幅 $\le\Delta$）的关键。把 $\Psi(\cdot)\Phi(\cdot)$ 作用在下层最优解的端点坐标上，就把"非凸上层的稳定点"和"强凸下层逐坐标解链"焊死成一条整体 zero-chain，最终得到确定性下界 $\Omega(\Delta L_f\kappa^{3/2}\epsilon^{-2})$（Theorem 4.1）。

**4. 随机情形：两个有界超立方体控制方差，把下界推到 $\Omega(\kappa^{5/2}\epsilon^{-4})$**

随机设定下 oracle 返回的是满足 $\mathbb{E}[G_f]=\nabla f$、方差 $\le\sigma^2$ 的无偏估计。作者在确定性构造上叠加随机性论证：用**两个有界超立方体**约束 $\mathbf{x},\mathbf{y}$ 的取值范围，从而控制随机一阶 oracle 的方差预算。这与并行工作 Chen and Zhang (2025) 通过消去耦合变量 $\mathbf{y}$ 降维的做法不同——本文保留 $\mathbf{y}$、用区域有界性约束噪声，构造更简洁（不引入额外辅助变量 $z$）。最终证明任意一阶 zero-respecting 算法要找 $\epsilon$-稳定点至少需 $\Omega(\kappa^{5/2}\epsilon^{-4})$ 次随机 oracle 调用，比单层随机非凸的 $\Omega(\epsilon^{-4})$ 强 $\kappa^{5/2}$ 倍、比 min-max 的相应下界强 $\kappa^{13/6}$ 倍。

### 损失函数 / 训练策略
不适用——本文不训练任何模型，全部为闭式构造与不等式证明。

## 实验关键数据

本文没有数值实验，"结果"是理论下界。下面两张表汇总本文下界与已有相关下/上界的对比。

### 主结果：本文下界 vs. 相关设定的最优下界

| 设定 | 单层非凸 | 非凸-强凸 min-max | **本文：非凸-强凸双层** | 本文相对提升 |
|------|----------|-------------------|------------------------|--------------|
| 确定性一阶 | $\Omega(\epsilon^{-2})$ (Carmon 2020) | $\Omega(\sqrt{\kappa}\epsilon^{-2})$ (Li 2021) | $\Omega(\kappa^{3/2}\epsilon^{-2})$ | $\times\kappa^{3/2}$ / $\times\kappa$ |
| 随机一阶 | $\Omega(\epsilon^{-4})$ (Arjevani 2023) | $\Omega(\kappa^{1/3}\epsilon^{-2})$ (Li 2021) | $\Omega(\kappa^{5/2}\epsilon^{-4})$ | $\times\kappa^{5/2}$ / $\times\kappa^{13/6}$ |

> ⚠️ min-max 随机下界原文写作 $\Omega(\kappa^{1/3}\epsilon^{-2})$，与本文 $\epsilon^{-4}$ 量级不在同一精度刻度，提升因子 $\kappa^{13/6}$ 以原文陈述为准。

### 与已有双层下界 / 上界的缺口对比

| 工作 | 类型 | oracle / 假设 | 复杂度 | 是否含 $\kappa$ |
|------|------|---------------|--------|----------------|
| Ji & Liang (2023) | 下界 | 二阶 oracle，$H$ 凸 | 含 $\sqrt{\kappa}$ gap | 是（受限设定） |
| Dagréou et al. (2024) | 下界 | 有限和一阶 | $\Omega(n+\sqrt{n}\epsilon^{-2})$ | 否 |
| Kwon et al. (2024) | 下界 | $\mathbf{y}^*$-aware 随机 | — | 退化为类单层 |
| **本文** | **下界** | **标准（随机）一阶** | $\Omega(\kappa^{3/2}\epsilon^{-2})$ / $\Omega(\kappa^{5/2}\epsilon^{-4})$ | **是** |
| Chen et al. (2025) | 上界 | 一阶 penalty | $\kappa^4\epsilon^{-2}$（加速后 $\kappa^{3.5}\epsilon^{-2}$） | 是 |
| Kwon et al. (2024) | 上界 | 随机一阶 | $\epsilon^{-6}$ | — |

### 关键发现
- **双层 > min-max > 单层**：确定性下界 $\kappa^{3/2}$ vs. min-max 的 $\sqrt{\kappa}$ vs. 单层的 $\kappa^0$，从理论上严格确认双层优化本质更难，且难度差距恰好量化为 $\kappa$ 的幂次。
- **上下界之间仍有巨大缺口**：确定性情形上界 $\kappa^{3.5}\epsilon^{-2}$ 与本文下界 $\kappa^{3/2}\epsilon^{-2}$ 相差 $\kappa^2$；随机情形上界 $\epsilon^{-6}$ 与下界 $\epsilon^{-4}$ 也未匹配。最优复杂度仍是 open problem。
- **下界在二次下层时依旧成立**：作者指出，即便把下层限制为二次函数这一最简情形，本文下界仍然适用，因此后续闭合缺口的努力可以先从"二次下层"这个干净 regime 入手。

## 亮点与洞察
- **把双层难度"翻译"成 $\kappa$ 的幂次**：最让人"啊哈"的是它用一个统一的 zero-chain 框架，把强凸下层（三对角 $A$）与非凸上层（$\Psi/\Phi$）串成一条链，使条件数 $\kappa$ 自然地以幂次形式叠进下界，给出"双层比 min-max 多一个 $\kappa$ 因子"的干净结论。
- **构造极简**：相比并行工作 Chen & Zhang (2025) 需要引入辅助变量 $z$、随机情形要消去耦合变量降维，本文用"保留 $\mathbf{y}$ + 两个有界超立方体控方差"的更轻量构造拿到更强（甚至更大 $\kappa$ 依赖）的下界。
- **可迁移的工具箱**：把"三对角 Laplacian 编码强凸 + Carmon 硬函数编码非凸 + zero-chain 串联"这套组合，可直接迁移去攻击其他嵌套结构（如三层优化、带约束双层）的下界问题。

## 局限与展望
- **下界未必紧**：上下界仍差 $\kappa^2$（确定性）和 $\epsilon^2$（随机），无法断言这是最优复杂度——可能是下界偏松，也可能是现有算法不够好。
- **仅覆盖非凸-强凸**：构造依赖下层强凸（三对角 $A$ 正定、$\mathbf{y}^*$ 唯一），对下层仅凸或非凸的更一般双层问题不直接适用。
- **只是存在性下界**：给出的是"存在最坏实例"，并未提供能匹配下界的算法；作者建议先在"二次下层"这一简化但仍有意义的 regime 中寻找更紧的上界。

## 相关工作与启发
- **vs Li et al. (2021)（min-max 下界）**：本文构造高度受其启发，但把单层 min-max 的"鞍点链"升级为双层的"下层解链 + 上层稳定点链"两段耦合，因此下界从 $\sqrt{\kappa}$ 抬到 $\kappa^{3/2}$，本质区别在于多了一层"求精确下层最优"的链式代价。
- **vs Kwon et al. (2024)**：他们用 $\mathbf{y}^*$-aware oracle 把双层退化成类单层、上界为 $\epsilon^{-6}$；本文坚持标准一阶 oracle，论证更贴近真实算法能力，并暴露其上界与真实下界间的差距。
- **vs Chen & Zhang (2025)（并行工作）**：两者独立得到一阶双层下界、都发现 $\kappa$ 依赖比单层 / min-max 更大，但构造路线不同（对方引入辅助变量 $z$、随机情形降维消去 $\mathbf{y}$；本文构造更简、保留 $\mathbf{y}$ 用有界超立方体控方差）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在标准（随机）一阶 oracle 下给出依赖 $\kappa$ 的非凸-强凸双层下界，填补 open problem。
- 实验充分度: ⭐⭐⭐⭐ 纯理论无数值实验，但下界证明完整、与多条已有结果系统对比。
- 写作质量: ⭐⭐⭐⭐ 构造与证明思路清晰，hard instance 各组件动机交代到位。
- 价值: ⭐⭐⭐⭐⭐ 量化了双层比 min-max / 单层"难多少"，并明确指出上下界缺口与未来主攻方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] A Fully First-Order Layer for Differentiable Optimization](a_fully_first-order_layer_for_differentiable_optimization.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](../../NeurIPS2025/optimization/a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](../../ICLR2026/optimization/bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](../../ICML2025/optimization/improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)

</div>

<!-- RELATED:END -->
