---
title: >-
  [论文解读] Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality
description: >-
  [ICLR 2026][LLM推理][测试时扩展] 本文把"带验证器的测试时扩展"重新表述成一个最优传输（采样）问题，用一套统一框架精确刻画了**生成器覆盖度、验证器 ROC、采样算法次优性**三者的几何关系，揭示出次优性—覆盖度曲线存在「传输 / 策略改进 / 饱和」三段制，并据此设计与分析了序贯（SRS、SMC）和批量（BRS）两类采样算法。
tags:
  - "ICLR 2026"
  - "LLM推理"
  - "学习理论"
  - "测试时扩展"
  - "验证器"
  - "最优传输"
  - "次优性"
  - "Youden指数"
---

# Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BBDhQJh6GB](https://openreview.net/forum?id=BBDhQJh6GB)  
**代码**: https://github.com/marcellobullo/ot-resampling  
**领域**: LLM推理 / 学习理论  
**关键词**: 测试时扩展, 验证器, 最优传输, 次优性, Youden指数

## 一句话总结
本文把"带验证器的测试时扩展"重新表述成一个最优传输（采样）问题，用一套统一框架精确刻画了**生成器覆盖度、验证器 ROC、采样算法次优性**三者的几何关系，揭示出次优性—覆盖度曲线存在「传输 / 策略改进 / 饱和」三段制，并据此设计与分析了序贯（SRS、SMC）和批量（BRS）两类采样算法。

## 研究背景与动机
**领域现状**：测试时扩展（test-time scaling）是提升 LLM 性能的重要方向，其中"基于验证器"的做法尤其流行——用一个二值奖励机制（单元测试、标准答案、LLM-as-a-judge 等）来筛选生成结果。典型流水线由三部分组成：生成器（参考 LLM）、验证器、采样算法（如 Best-of-N）。最终性能是这三者属性的叠加。

**现有痛点**：已有理论工作大多只刻画了聚合层面的 scaling law（如 pass@N、策略散度），而且普遍假设**验证器是完美的**——这个假设在实践中几乎不成立。少数研究开始考虑验证器的不完美性，但缺一套能把"生成器特性 × 验证器误差 × 采样算法"三者交互关系统一起来、并量化其几何结构的框架。

**核心矛盾**：性能由三个量共同决定——(i) 生成器的**覆盖度** $\beta-1$（最优策略能在多大程度上偏离参考策略而仍被其"覆盖"）；(ii) 验证器的 **ROC**（用真阳率 TPR、假阳率 FPR 刻画）；(iii) 采样算法的**次优性**。这三者并非各自独立，而是耦合在一起：放宽覆盖度未必降低次优性，是否降低取决于验证器够不够准。以往的渐近曲线无法刻画这种精确依赖关系。

**本文目标**：回答"基于验证器的采样到底能多大程度逼近被诱导出的最优策略，以及验证误差如何塑造这种逼近"，需要一个**精确**（而非渐近）的细粒度分析。

**切入角度**：作者观察到，测试时验证本质是一个**采样问题**：给定对参考分布 $\mu$ 的生成访问权，目标是从一个由真值验证器 $r^\star$ 定义的目标分布 $\nu^\star$ 中采样，而我们只能通过一个近似正确的验证器 $\hat r$ 来引导。"把 $\mu$ 搬运到 $\nu^\star$"天然是一个**最优传输**问题。

**核心 idea**：用最优传输的几何语言统一刻画测试时验证——把次优性分解成"最优传输代价 OTC + 策略改进项"，从而把覆盖度、ROC、次优性三者的相互作用一次性讲清楚。

## 方法详解

### 整体框架
本文不提出新的"工程系统"，而是给出一套**分析框架**：把测试时验证形式化为"从代理分布 $\mu$ 到目标分布 $\nu^\star$ 的传输计划设计问题"，并在这套框架下推导算法的次优性与计算复杂度。

具体设定：提示 $x$ 经参考 LLM 生成响应 $y\sim\pi_{\mathrm{ref}}(\cdot\mid x)$，诱导出参考分布 $\mu$。真值验证器把验证建模为**集合成员判定**——存在正确响应集合 $S^\star(x)$，$r^\star(x,y)=\mathbf 1\{y\in S^\star(x)\}$。最优策略被约束在一个被参考策略"充分覆盖"的策略类里（$\ell_1$ 型覆盖约束，等价于 $\chi^2(\mu\Vert\nu)\le\beta-1$）。由于二值奖励结构，本文给出了诱导最优策略的**闭式解**（Theorem 2.1）：目标对参考的 Radon-Nikodym 导数

$$\eta_r(y)=\begin{cases}\dfrac{1}{s_r}\wedge\dfrac{m_\beta(s_r)}{s_r} & y\in S_r\\[1mm] 0\vee\dfrac{1-m_\beta(s_r)}{1-s_r} & y\notin S_r\end{cases},\qquad m_\beta(s)\triangleq s+\sqrt{s(1-s)(\beta-1)},$$

其中 $s_r\triangleq\int_{S_r}r\,\mathrm d\mu$ 是参考策略落在验证集上的质量。整个分析就建立在这个闭式解之上。

采样算法被刻画为 $\mu$ 与 $\nu^\star$ 之间的一个**耦合** $\rho$，传输代价取 Hamming 代价 $C(\rho)=\int\mathbf 1\{y\ne z\}\,\rho(\mathrm dy,\mathrm dz)$（直观上就是"为采到目标样本所需的拒绝比例"）。算法的优劣用**次优性**衡量：

$$\mathrm{SubOpt}(A)\triangleq\int r^\star\,\mathrm d\nu^\star-\int r^\star\,\mathrm d\nu_A.$$

难点在于 $\nu^\star$ 不可直接访问，只能通过不完美验证器 $\hat r$ 的成员判定来引导，于是验证器质量（TPR/FPR/Youden 指数 $J$）必然渗入次优性。下面四个关键设计依次回答：怎么建框架、几何长什么样、序贯怎么采、批量怎么采。

### 关键设计

**1. 测试时验证 = 最优传输：把问题搬到可分析的几何空间**

本文最根本的一步，是识别出测试时验证就是"把参考分布 $\mu$ 传输到目标分布 $\nu^\star$"。若算法接受得太宽松，诱导分布贴近 $\mu$，次优性高；若拒绝得太狠，次优性可降但要付出巨大算力。关键挑战是设计一个在"样本利用率"与"诱导次优性"之间权衡的传输计划。在 Hamming 代价下，作者给出最优传输代价的闭式（Lemma 3.1）：

$$\mathrm{OTC}(\beta)=\bigl(1\wedge m_\beta(s_{r^\star})\bigr)-s_{r^\star},$$

它是"从 $\mu$ 搬到 $\nu^\star$ 所需的最小拒绝概率"，刻画了问题**本身**的内在难度（与算法无关）。这一框架的价值在于：它把原本散落在不同论文里的覆盖度、验证误差、算法效率，统一成同一张传输图上的几何量，使得"精确"分析（而非渐近）成为可能。

**2. 次优性—覆盖度的三段几何：放宽覆盖度未必更好**

把次优性分解为两部分——(i) 最优传输代价 OTC（搬运 $\mu\to\nu^\star$ 的内在困难），(ii) 策略改进项（采样算法在多大程度上抵消这一代价）。直接从 $\nu^\star$ 采样时，策略改进恰好等于传输代价，得到最优方案；但验证器不完美使这一理想不可达，次优性由验证器 ROC（尤其 Youden 指数 $J=\mathrm{TPR}-\mathrm{FPR}$）与覆盖度 $\beta$ 共同支配。作者证明随覆盖约束放宽，曲线呈现三个截然不同的区制：

$$\mathrm{SubOpt}=\mathrm{OTC}(\beta)\cdot(1-\alpha J),\quad\alpha\ \text{随}\ \beta\ \text{分段取值}.$$

- **传输区制**（$\beta$ 小）：次优性以 $O(\sqrt\beta)$ 增长，完全被 OTC 主导，策略改进几乎不起作用；
- **策略改进区制**（中等 $\beta$）：OTC 开始饱和，只要验证器够准（$J>0$ 且 $s_{\mathrm{ver}}\le s_{r^\star}$），次优性可随覆盖度**下降**；若 $s_{\mathrm{ver}}\ge s_{r^\star}$ 则假阳过多、毫无改进；
- **饱和区制**（$\beta$ 大）：OTC 稳定在 $1-s_{r^\star}$，两项都平台化，次优性恒定、再加覆盖度也没用。

这一几何修正了以往"次优性随覆盖度平方根缩放"的笼统结论：覆盖度—次优性的权衡**不是普适的，而是被验证器 ROC 调制的**。

**3. 序贯采样算法：AiC 违约、SRS 与 SMC 合规且等价**

序贯协议下不停生成直到接受。作者对比三个算法。**AiC**（accept-if-correct）就是把 BoN 推广到序贯：采一个、判成员、不过就重采。但它**无视覆盖约束** $\beta$，在低覆盖区会违约（Theorem 3.3：当 $\beta<1/s_{\mathrm{ver}}$ 时 $\pi_{\mathrm{AiC}}\notin\Pi(\beta)$）；其复杂度 $\mathbb E[\tau_{\mathrm{AiC}}]=1/s_{\mathrm{ver}}$、次优性 $\mathrm{OTC}(\beta)(1-\tilde\alpha J)$。

为修正违约，作者提出 **SRS**（序贯拒绝采样）：用 $\hat r$ 对应的 $\eta_{\hat r}$ 作包络做拒绝采样，**构造上永远满足覆盖约束**，即便成员判定失败也可能接受样本。又从"最小化传输代价"出发提出 **SMC**（序贯极大耦合）：先比似然比与均匀数，过了就接受，否则不停从 $\mu$ 重采直到通过成员判定（关键是 Lemma 3.4 给出残差测度的可生成等价表示 $\mu_{\mathrm{res}}=\mu(\cdot\mid S)$，绕开"无法从残差直接采样"的障碍）。令人意外的是（Theorem 3.5–3.6）：SRS 与 SMC **期望提案数完全相同** $\mathbb E[\tau]=\tfrac1{s_{\mathrm{ver}}}(1\wedge m_\beta(s_{\mathrm{ver}}))$、次优性也相同——SMC 看似为省算力而生，却因缺残差访问把潜在收益抵消掉了，两者在复杂度和次优性上等价；且二者相比 AiC 计算复杂度有 $m_\beta(s_{\mathrm{ver}})$ 倍的加速。

**4. 批量采样算法：BoN 有最大批量上限、BRS 全程合规**

批量协议一次并行生成 $N+1$ 个响应、从中挑选。**BoN** 即便用真值验证器也有非零次优性（只能在 $N+1$ 个样本里选），且存在**最大可用批量** $N_{\max}$（Theorem 3.7）：超过它就违反覆盖约束，仅当 $\beta\ge(1-s_{r^\star})/s_{r^\star}$ 时才能无限增大 $N$。其次优性 $\mathrm{SubOpt}(\mathrm{BoN})=(1-s_{r^\star})^{N+1}-(0\vee 1-m_\beta(s_{r^\star}))$ 随 $N$ 减小。为突破 BoN 在低覆盖区的不可行与恒定次优性，作者把 SRS 推广为 **BRS**（批量拒绝采样）：并行抽一批、对其中 $N$ 个做拒绝采样、都不过就用第 $N+1$ 个兜底。BRS **对任意 $N$ 都满足覆盖约束**（Theorem 3.9），次优性随批量**指数衰减** $\mathrm{SubOpt}(\mathrm{BRS})=\mathrm{OTC}(\beta)(1-\tfrac1M)^N$（Theorem 3.10）。总体结论：**低覆盖区用拒绝采样类（SRS/BRS）更优，宽松覆盖区 BoN 类更划算**。

## 实验关键数据

实验围绕两问：(1) 经验次优性曲线与三段制理论预测吻合度如何？(2) 算法对覆盖参数 $s_{\mathrm{ver}}$ 误设的敏感性如何？在 Qwen、Llama、Gemma 上，对覆盖预算 $\beta$ 扫过覆盖三区制、批量设置另扫 $N+1$，每条曲线对 5000 episode 取平均。

### 主实验（序贯协议，次优性与复杂度）

| 现象 | SRS / SMC | AiC | 说明 |
|------|-----------|-----|------|
| 次优性曲线形状 | 完整呈现"$O(\sqrt\beta)$ 上升 → 随 $J$ 下弯 → 平台"三段制 | 仅在饱和区与前者一致，其余区**违约** | 与 Theorem 3.6 吻合 |
| 平台高度 | 由 $s_{r^\star}$ 与 $J$ 决定 | 同饱和区 | 大模型 $s_{r^\star}$ 更高 → 残余次优性更低 |
| 计算复杂度 | $O(\sqrt\beta)$ 后饱和（Qwen3-1.7B 近似验证器≈8 次提案，真值验证器≈6 次） | 常数 | SRS=SMC，证实二者等价 |

### 批量与敏感性分析

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| BRS vs BoN（不完美验证器，Qwen3-14B） | 次优性随 $N$ 增大而下降，理论（Thm. O.1/O.2）与经验高度吻合 | BRS 在中低覆盖区优于 BoN |
| $s_{\mathrm{ver}}=s$ 误设扫描 | $s=1$ 时三算法都退化为 AiC（包络 $M=1$、$\eta_r=\mathbf 1\{y\in S_r\}$） | 验证 AiC 是特例 |
| SRS vs SMC 随假设 $s$ | 真值 $s=0.31$（近似验证器 $s=0.27$）处两者重合；$s$ 低估时 SMC 劣于 SRS、$s$ 高估时 SMC 优于 SRS | 交叉行为：高 $s$ 域用 SMC，低验证器置信度时 SRS 更稳健 |

### 关键发现
- **三段制是真实存在的**：经验曲线在三个 $\beta$ 区制上分别呈上升、下弯、平台，且下弯幅度正比于验证器信息量（$J$ 越大弯得越多），与理论几何一一对应。
- **SRS≡SMC**：尽管推导原理不同（一个从合规拒绝采样、一个从最小化传输代价出发），二者期望提案数与次优性完全相同——"缺残差访问"把 SMC 的理论省算力优势抵消了。
- **模型规模主要影响平台高度**：更大的 Qwen 模型 $s_{r^\star}$（基础正确率）更高，残余次优性更低，而非改变曲线形状。
- **SMC 对 $s$ 误设敏感**：高估/低估验证器准确度会导致它在 SRS 上下穿越，提示在验证器置信度不确定时优先用 SRS。

## 亮点与洞察
- **"采样问题 → 最优传输"的重述非常巧妙**：一旦把测试时验证看成把 $\mu$ 搬到 $\nu^\star$，覆盖度（$\chi^2$ 球半径）、验证误差（ROC/Youden）、算法效率（Hamming 传输代价）就自然落在同一张几何图上，把原本割裂的经验观察统一起来。
- **三段制 + Youden 指数调制**是核心"啊哈"：它纠正了"次优性 $\sim\sqrt{\text{覆盖度}}$"的笼统说法——是否能靠加覆盖度受益，取决于验证器够不够准，存在一个明确的相变。
- **OTC 这个与算法无关的"问题难度下界"可复用**：任何想分析其他筛选/重采样策略的工作，都可以拿 $\mathrm{OTC}(\beta)=(1\wedge m_\beta(s_{r^\star}))-s_{r^\star}$ 当基准刻度。
- **算法选择的工程指引清晰**：低覆盖（算力受限、想贴近最优策略）选 SRS/BRS，宽松覆盖选 BoN，是可直接落地的结论。

## 局限与展望
- **理论假设较强**：分析假定采样算法可获知 $s_{\hat r}$（参考策略落在验证集上的质量），实践中通常不可得——本文虽把它当可调超参 $s$ 做了消融，但真实部署仍需估计或调参。
- **验证建模为二值集合成员**：把验证抽象成 $r^\star=\mathbf 1\{y\in S^\star\}$ 对编程/数学等"客观可判定"任务合适，但对带噪、连续或主观打分的验证器（如多数 LLM-as-a-judge 场景）覆盖有限。
- **批量算法的不完美验证器分析放在附录**：正文主分析（BoN/BRS）以准确验证器为主，近似验证器情形未在正文充分展开。
- **可拓展方向**：把框架推广到非 Hamming 代价、多轮/树搜索式扩展，或验证器本身可学习/自适应的设定。

## 相关工作与启发
- **vs Huang et al. (2025a)**：他们报告次优性随覆盖度平方根缩放并用 $\ell_1$ 覆盖约束；本文沿用其覆盖约束设定，但用最优传输给出**精确**分解，指出该平方根缩放只在传输区制成立、整体被验证器 ROC 调制，是对其结论的细化与修正。
- **vs Dorner et al. (2025)**：他们把 AiC 这类策略称作"拒绝采样"；本文严格区分 AiC 与真正的拒绝采样（SRS），并证明 AiC 在传输区制违反覆盖约束，从而提出合规的 SRS/SMC/BRS 替代。
- **vs 经典 Best-of-N 分析（Brown et al. 2024 等）**：以往多停在 pass@N 这类聚合曲线；本文把 BoN 放进同一传输框架，给出其最大可用批量 $N_{\max}$ 与精确次优性，并指出其在低覆盖区的固有缺陷。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把测试时验证统一重述为最优传输并精确刻画三段几何，视角新且自洽
- 实验充分度: ⭐⭐⭐⭐ Qwen/Llama/Gemma 多模型验证三段制与敏感性，但验证器被简化为二值成员判定
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨、图与定理对应清晰，符号略密集需要耐心
- 价值: ⭐⭐⭐⭐⭐ 给"该用哪种采样算法、加覆盖度有没有用"提供了可操作的理论判据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Slicing Wasserstein over Wasserstein via Functional Optimal Transport](slicing_wasserstein_over_wasserstein_via_functional_optimal_transport.md)
- [\[ICLR 2026\] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling](expressive_power_of_implicit_models_rich_equilibria_and_test-time_scaling.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](../../ICML2026/learning_theory/on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)

</div>

<!-- RELATED:END -->
