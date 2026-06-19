---
title: >-
  [论文解读] A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning
description: >-
  [NeurIPS 2025 (Spotlight, top 3%)][强化学习][temporal difference learning] 首次引入矩阵分裂理论，将线性函数逼近下的TD、FQI和PFQI统一为求解同一目标线性系统 $(\Sigma_{cov} - \gamma\Sigma_{cr})\theta = \theta_{\phi,r}$ 的迭代方法（仅预条件子不同），给出各算法收敛的充要条件，提出rank invariance新概念，并揭示target network的本质是预条件子从常数到数据自适应的连续变换。
tags:
  - "NeurIPS 2025 (Spotlight, top 3%)"
  - "强化学习"
  - "temporal difference learning"
  - "fitted Q-iteration"
  - "matrix splitting"
  - "preconditioning"
  - "convergence analysis"
---

# A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning

**会议**: NeurIPS 2025 (Spotlight, top 3%)  
**arXiv**: [2501.01774](https://arxiv.org/abs/2501.01774)  
**代码**: 无  
**领域**: 强化学习 / 策略评估 / 理论分析  
**关键词**: temporal difference learning, fitted Q-iteration, matrix splitting, preconditioning, convergence analysis

## 一句话总结
首次引入矩阵分裂理论，将线性函数逼近下的TD、FQI和PFQI统一为求解同一目标线性系统 $(\Sigma_{cov} - \gamma\Sigma_{cr})\theta = \theta_{\phi,r}$ 的迭代方法（仅预条件子不同），给出各算法收敛的充要条件，提出rank invariance新概念，并揭示target network的本质是预条件子从常数到数据自适应的连续变换。

## 研究背景与动机
**领域现状**：Off-policy策略评估(OPE)中，TD学习可能发散而FQI通常更稳定。传统理解认为三种算法的区别仅在于对目标值函数的更新次数（TD=1次, FQI=∞次, PFQI=有限次）。

**现有痛点**：(1) 传统观点无法解释为何TD收敛不蕴含FQI收敛（反之亦然）；(2) 现有收敛分析依赖"特征线性独立"等强假设，给的多是充分条件而非充要条件；(3) 文献中存在多处错误结论（如Ghosh 2020声称线性独立即可保证off-policy TD不动点唯一——本文指出还需rank invariance）。

**核心矛盾**：三种看似不同的算法，在数学上是否有统一的本质？各自收敛的精确条件到底是什么？

**切入角度**：借鉴数值线性代数中的矩阵分裂和预条件技术，将三种RL算法重写为同一迭代格式 $\theta_{k+1} = (I - MA)\theta_k + Mb$ 的不同预条件子$M$实例。

## 方法详解

### 整体框架
核心洞察：TD、FQI、PFQI求解的是同一个目标线性系统（即LSTD系统）：$(\Sigma_{cov} - \gamma\Sigma_{cr})\theta = \theta_{\phi,r}$，区别**仅在预条件子M**的选择——TD用 $M = \alpha I$（常数），FQI用 $M = \Sigma_{cov}^{-1}$（数据自适应），PFQI用 $M = \alpha\sum_{i=0}^{t-1}(I-\alpha\Sigma_{cov})^i$（从TD到FQI的过渡）。收敛性完全由目标线性系统的一致性和迭代矩阵 $H = I - MA$ 的半收敛性决定。

### 关键设计
1. **Rank Invariance条件（秩不变性）**:
    - 功能：提出一个新的、温和的条件来替代传统的"特征线性独立"假设
    - 核心思路：定义 $\text{Rank}(\Phi) = \text{Rank}(\Phi^\top \mathbf{D}(I - \gamma\mathbf{P}_\pi)\Phi)$，等价于 $\gamma\Sigma_{cov}^\dagger\Sigma_{cr}$ 没有等于1的特征值。证明rank invariance是目标线性系统对任意奖励函数$R$都有解（universal consistency）的充要条件（Proposition 4.2）；rank invariance + 特征线性独立 = 目标线性系统有唯一解（Proposition 4.5）；在on-policy设定下rank invariance自动成立（Proposition 4.7）
    - 设计动机：特征线性独立是过强假设——实际中over-parameterization普遍存在。Rank invariance更温和且几乎总是满足的，作为分析的新基石可以去掉many prior work的冗余假设

2. **预条件子连续变换与Proper Splitting**:
    - 功能：揭示TD→PFQI→FQI的预条件子转换链，证明rank invariance下FQI享有proper splitting优势
    - 核心思路：$\alpha I \underset{t=1}{\rightleftharpoons} \alpha\sum_{i=0}^{t-1}(I-\alpha\Sigma_{cov})^i \xrightarrow{t\to\infty} \Sigma_{cov}^{-1}$。当rank invariance成立时，$(Sigma_{cov}, \Sigma_{cr})$ 构成proper splitting（Lemma 5.2），FQI收敛条件放松为 $\rho(\gamma\Sigma_{cov}^\dagger\Sigma_{cr}) < 1$，不动点保证唯一。这解释了FQI为何比TD更robust——FQI的预条件子自适应于数据分布
    - 设计动机：target network（DQN中广泛使用）的本质首次被理论刻画：增加target更新间隔$t$等价于从常数预条件子连续过渡到特征协方差逆的自适应预条件子

3. **TD学习率区间与编码器-解码器视角**:
    - 功能：给出TD稳定性的充要条件以及可行学习率的精确刻画
    - 核心思路：TD稳定当且仅当三个条件同时满足——一致性、$A_{LSTD}$正半稳定、$\text{Index}(A_{LSTD}) \leq 1$（Corollary 6.2）。可行学习率形成区间 $\alpha \in (0, \epsilon)$，其中 $\epsilon = \min_{\lambda \in \sigma(A_{LSTD})\backslash\{0\}} \frac{2\Re(\lambda)}{|\lambda|^2}$（Corollary 6.3）。在on-policy设定下，即使特征不线性独立，TD仍然稳定（Theorem 6.4）——放松了Tsitsiklis & Van Roy 1996的经典条件
    - 设计动机：首次证明"大学习率不行时小学习率可能有效"——可行学习率形成连续区间而非孤立点，为实践调参提供理论依据

### 损失函数 / 训练策略
本文是纯理论工作，无训练过程。核心在于证明了三种迭代格式——TD: $\theta_{k+1} = (I - \alpha(\Sigma_{cov} - \gamma\Sigma_{cr}))\theta_k + \alpha\theta_{\phi,r}$, FQI: $\theta_{k+1} = \gamma\Sigma_{cov}^\dagger\Sigma_{cr}\theta_k + \Sigma_{cov}^\dagger\theta_{\phi,r}$, PFQI: 嵌套$t$步TD更新——最终都收敛到目标线性系统的解集 $\Theta_{LSTD}$（如果收敛的话）。关键发现是PFQI增加$t$（target network更新间隔）在特征不线性独立时可能导致发散而非收敛。

## 实验关键数据

### 核心理论结果总结

| 算法 | 收敛充要条件 | 预条件子 | 关键特性 |
|---|---|---|---|
| TD | 一致性 + $H_{TD}$半收敛 | $\alpha I$（常数） | 收敛依赖学习率$\alpha \in (0, \epsilon)$ |
| FQI | 一致性 + $H_{FQI}$半收敛 | $\Sigma_{cov}^{-1}$（自适应） | Rank invariance下条件放松为$\rho < 1$ |
| PFQI | 一致性 + $H_{PFQI}$半收敛 | TD→FQI的过渡 | 增加$t$不保证稳定化 |

### 算法间收敛蕴含关系

| 蕴含关系 | 是否成立 | 关键条件 |
|---|---|---|
| TD收敛 → FQI收敛 | ✗ 不一定 | 构造了反例 |
| FQI收敛 → TD收敛 | ✗ 不一定 | 构造了反例 |
| TD稳定 → PFQI收敛 | ✓ 有条件 | 对任意有限$t$，存在$\epsilon_t$使$\alpha \in (0, \epsilon_t)$时PFQI收敛 |
| 增加$t$ → PFQI更稳定 | ✗ 不一定 | 特征不线性独立时增加$t$可能导致发散 |

### 关键发现
- Rank invariance单独即可保证FQI线性系统非奇异（Proposition 4.6），而目标线性系统需要rank invariance + 特征线性独立（Proposition 4.5）
- On-policy设定下rank invariance自动成立→TD/FQI/PFQI不动点必定存在
- 文献修正：指出Ghosh 2020、Asadi 2024、Xiao 2021等prior work中条件为充分而非充要

## 亮点与洞察
- 统一框架极其优雅——一个预条件子差异统一三种算法，数学简洁且有力
- 首次将数值线性代数的经典工具（matrix splitting, proper splitting, semiconvergent matrices）引入RL收敛分析
- 充要条件而非充分条件——比prior work更sharp，且纠正了多处文献错误
- Target network的理论本质首次被刻画为预条件子从常数到数据自适应的连续变换
- "学习率形成区间"为实践调参提供理论支撑，encoder-decoder视角提供TD收敛的新直觉

## 局限与展望
- 仅限线性函数逼近——虽然神经网络最后一层通常是线性的，但核心理论无法直接推广到非线性
- 仅限策略评估（policy evaluation），未涉及控制（policy improvement/optimization）
- 纯理论工作，缺少实证验证——在实际规模问题上rank invariance等条件是否常见未被经验检验
- 主体分析在expected TD（确定性版本）上，虽声称可推广到stochastic/batch TD但推广较为简略

## 相关工作与启发
- **vs Tsitsiklis & Van Roy (1996)**：经典on-policy TD收敛需特征线性独立，本文Theorem 6.4证明可去掉该假设
- **vs Fellows et al. (2023)**：仅给了PFQI在FQI收敛条件下增加$t$的充分条件，本文给出充要条件并揭示增加$t$可能反而发散
- **vs DQN target network**：本文给出target network的首个理论解释——预条件子变换，为设计更好的target更新策略提供指导
- **启发**：矩阵分裂视角可能启发新RL算法——设计更优预条件子加速收敛；rank invariance可能成为RL理论分析的新标准假设

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 矩阵分裂视角完全是新的，统一三种算法的优雅性极高
- 实验充分度: ⭐⭐⭐ 纯理论工作，仅有反例构造，无实证验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但数学密度极高（附录50+页），需较强线性代数背景
- 价值: ⭐⭐⭐⭐⭐ Spotlight论文，在RL理论领域有重要影响，修正文献多处错误并建立新的分析范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[ICLR 2026\] Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](../../ICLR2026/reinforcement_learning/is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [\[NeurIPS 2025\] Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)

</div>

<!-- RELATED:END -->
