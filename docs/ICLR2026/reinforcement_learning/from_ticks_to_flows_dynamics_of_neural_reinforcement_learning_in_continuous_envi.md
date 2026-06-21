---
title: >-
  [论文解读] From Ticks to Flows: Dynamics of Neural Reinforcement Learning in Continuous Environments
description: >-
  [ICLR 2026][强化学习][连续时间 RL] 把连续控制的深度 RL 建模成一个连续时间随机过程，引入"环境时钟"和"梯度时钟"两个时间尺度，并用 Itô-Taylor 展开 + 无限宽线性化网络，首次推导出在每个梯度步状态分布如何无穷小演化的方程，最终化简为一个只有五个时变量的闭合系统。 领域现状：神经网络极大推动…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "连续时间 RL"
  - "Actor-Critic"
  - "随机微分方程"
  - "无限宽神经网络"
  - "Itô-Taylor 展开"
  - "两时间尺度"
---

# From Ticks to Flows: Dynamics of Neural Reinforcement Learning in Continuous Environments

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TdiRLe3rPA](https://openreview.net/forum?id=TdiRLe3rPA)  
**代码**: 待确认  
**领域**: reinforcement learning / 连续控制理论  
**关键词**: 连续时间 RL, Actor-Critic, 随机微分方程, 无限宽神经网络, Itô-Taylor 展开, 两时间尺度  

## 一句话总结
把连续控制的深度 RL 建模成一个连续时间随机过程，引入"环境时钟"和"梯度时钟"两个时间尺度，并用 Itô-Taylor 展开 + 无限宽线性化网络，首次推导出在每个梯度步状态分布如何无穷小演化的方程，最终化简为一个只有五个时变量的闭合系统。

## 研究背景与动机
**领域现状**：神经网络极大推动了深度 RL（尤其是 Actor-Critic 类算法）在仿真游戏、机器人控制上的成功，监督学习侧也已有成熟理论（NTK、无限宽极限、参数分布演化）解释过参数化网络为何有效。

**现有痛点**：但深度 RL 的成功在理论上几乎仍是"黑箱"——尤其连续状态/连续动作的控制任务，带神经网络函数逼近的理论分析寥寥无几。**核心矛盾**在于：监督学习里数据分布是固定的，可以干净地研究梯度步带来的参数演化；而 RL 里数据分布本身会随着策略（参数）更新而改变，状态分布与参数耦合在一起，没法直接套用监督学习理论。

**本文目标**：刻画一个 Actor-Critic 智能体在学习过程中，状态分布如何随梯度更新而局部变化——不是去描述整条环境时间轨迹的全局演化，而是描述"每一个梯度小步"带来的无穷小改变。

**核心 idea**：**[两时间尺度建模]** 借鉴随机控制中 SDE"描述瞬时变化比描述全局演化更容易"的哲学，把智能体的状态看成同时受两个时钟驱动——快的"环境时钟"（从 0 走到 horizon $T$）和慢的"梯度时钟"（每步只走 $\eta$）。**[无限宽线性化 + Itô-Taylor]** 用线性化的两层无限宽网络把状态写成参数的多项式，结合网络输出的高斯性，推出跨两个时间尺度的非参数演化方程。

## 方法详解

### 整体框架
全文是一条理论推导链：先把连续控制 RL 形式化成一个"可探索的"随机微分方程（SDE）模型，再给出连续时间 Actor-Critic 的梯度更新算法，然后用线性化无限宽网络作为可分析的代理模型，最后借 Itô-Taylor 展开 + 鞅中心极限定理把"一个梯度步内状态/动作/价值如何变化"推成一个闭合方程组。

```mermaid
flowchart LR
    A[可探索 SDE 动力学<br/>Eq.4 单噪声源] --> B[连续时间 Actor-Critic<br/>Episodic 梯度更新]
    B --> C[线性化两层无限宽网络<br/>tanh, η=O 1/√n]
    C --> D[Itô-Taylor 展开<br/>状态写成参数多项式]
    D --> E[鞅 CLT + 条件 LLN<br/>高斯极限]
    E --> F[Thm 6.1 闭合系统<br/>5 个时变量]
```

### 关键设计

**1. 可探索的随机动力学：把策略噪声和环境噪声合成单一噪声源。** 论文在 control-affine MDP 上定义连续时间 RL，状态按 SDE $ds_t = (g(s_t)+h(s_t)a_t)dt + \sigma(s_t)dw_t$ 演化。为了让智能体真正探索，作者引入带策略噪声 $w'_t$ 和环境噪声 $w_t$ 的探索性 SDE，并证明（Lemma 3.1）在 $d_s=d_a=1$ 下它在分布上等价于一个**只有单一噪声源**的动力学 $d\tilde s^\pi_t = (g+h\pi)dt + \sqrt{h(\tilde s)^2+\sigma(\tilde s)^2}\,dw_t$（Eq.4）。这一步关键在于：它修正了 Wang et al. (2020) 的 relaxed-control 表述——后者在确定性环境（$\sigma\equiv0$）下策略的随机性会消失，而本文的探索性动力学即使环境确定也能保持探索（实验里能看到明显的随机跳变，覆盖更多状态-动作对）。同时单噪声源让后续离散数值模拟和神经网络分析都变得可处理。

**2. 连续时间 Episodic Actor-Critic：用时序差分式残差驱动连续时间梯度。** 在探索动力学之上，作者改编 Jia & Zhou 的连续时间策略梯度，给出确定性策略下的 Actor 与 Critic 更新（Eq.5）。核心是一个连续时间的 TD-error 式残差 $\delta = \partial_t v + r - \beta v$（即把贝尔曼残差写成对时间的偏导形式），策略和价值的梯度都是 $\int_0^T e^{-\beta l}\,\partial_\theta(\cdot)\,\delta\,dl$ 形式的折扣积分。算法 1 在离散网格上以单条轨迹做随机更新（episodic RL），结构上类似 coagent network。这套更新把"环境时间上的积分"和"梯度时间上的参数移动"显式分开，为后面两时钟分析铺路。

**3. 线性化两层无限宽网络：让策略/价值成为参数的可分析多项式。** Actor 和 Critic 都用两层 tanh 网络 $F(s;W,C)=\frac{1}{\sqrt n}\sum_\kappa C_\kappa \phi(W_\kappa\cdot s)$，只训练第一层 $W$、固定输出层 $C$，并在初始化 $W^0$ 附近做线性化 $F^{lin}_\pi(s;W)=F_\pi(s;W^0)+\Phi(s;W^0)(W-W^0)$。选 tanh 是因为它对称且光滑（保证动力学可微），学习率取 $\eta=O(1/\sqrt n)$。这一步把策略对 $W$ 变成线性、对 $s$ 与 $W^0$ 非线性，从而能把状态变量展开成参数的多项式——这是整套理论可解析的根基（作者引用先前工作说明线性化网络在大宽度下行为接近真实网络）。

**4. Itô-Taylor 展开 + 鞅 CLT：把单梯度步的变化推成五变量闭合系统（核心结论 Thm 6.1）。** 这是全文主结果。状态随机变量经 Itô-Taylor 展开可写成 $W^\tau-W^0$ 的（无穷）多项式；再对价值、动作及其导数施加鞅中心极限定理与对应的大数定律，得到它们在单梯度步内的变化 $\Delta v_{t,\tau}, \Delta v'_{t,\tau}, \Delta a_{t,\tau}, \Delta a'_{t,\tau}$ 都是**高斯**的（均值为折扣积分、方差正比于 $(\Delta s_{t,\tau})^2$），误差仅 $O(1/\sqrt n)$。状态本身的变化 $\Delta\tilde s_{t,\tau}=\eta Z_{t,\tau}-M_{t,\tau}+G_{t,\tau}+O(1/n)$。**关键洞察**：这些辅助变量的变化都依赖 $\Delta s$ 和 TD-error 式量 $q_{l,\tau}=v'_{l,\tau}+r-\beta v_{l,\tau}$（因为它们的分布是状态分布的 push-forward），而状态的变化又只依赖 $q$ 和这五个变量本身——于是整个系统**闭合**了，单步梯度动力学只由五个时变量（状态、动作、动作导数、价值、价值时间导数）相互决定。值得注意的是 $\Delta s_{t,\tau}$ 不是 $O(\eta)$ 量级，这源于环境随机性带来的发散项（但其随机部分均值为 0）。

## 实验关键数据

这是一篇理论论文，实验目的是"佐证"而非"刷点"，在玩具连续控制任务（LQR）上验证。

### 探索动力学验证

| 设置 | 现象 |
|------|------|
| 加性 Wiener 噪声 $\pi(s_t)+w_t$（确定性环境 $\sigma=0$） | 轨迹光滑，**探索不足**，状态-动作覆盖差 |
| 本文探索动力学（Eq.4） | 出现随机跳变，状态-动作覆盖更好；$\Delta t\to 0$ 时近似均值轨迹平滑收敛 |

### 主实验：Episodic 连续时间 Actor-Critic
LQR 环境：$g(s)=s, h(s)=1, \sigma(s)=0.1$，探索噪声 ×0.05，$\beta=0.1$，$s_0=2.0$，动作范围 $[-5,5]$，$r(s)=-500s^2$（驱动回到原点），$\Delta t=0.02, T=1$（每个 episode 50 步），结果对 20 个随机种子取平均。

| 维度 $d_s$ | 结果 |
|-----------|------|
| 1 / 2 / 8 / 32 | 各维度智能体均学到**近最优策略** |

### 关键发现
- **理论模型 vs 真实算法**：用 Thm 6.1 的理论模型仿真出的曲线（黑色虚线）与在线 episodic Actor-Critic（算法 1）的实测结果**高度吻合**，直接验证了主定理对梯度时间动力学的刻画。
- 探索动力学相比加性噪声方案在确定性环境下仍能保持有效探索。

## 亮点与洞察
- **"两个时钟"的隐喻很有解释力**：把 RL 学习显式拆成快环境时钟 + 慢梯度时钟，干净地分离了"数据分布随参数改变"这个深度 RL 理论的核心难点。
- **"五变量闭合系统"是惊人的简化**：从无限宽过参数化网络出发，最终单梯度步动力学只剩 5 个时变量相互决定，说明复杂神经 RL 背后可能藏着源自概率论/随机过程基本原理的简单结构。
- **首次给出连续 RL 中状态分布的梯度时间演化方程**（在消失学习率下），把随机控制和现代过参数化 RL 真正搭了一座桥。
- 修正了既有探索性动力学（relaxed control）在确定性环境下退化的缺陷，且单噪声源等价表述让理论与离散模拟都可落地。

## 局限与展望
- **维度限制**：核心定理只在 $d_s=d_a=1$ 给出，高维情形会引入 $d_s\times d_s$ 的耦合项，证明复杂度显著上升，留作未来工作。
- **"懒惰区"假设**：线性化 + 无限宽把分析锁在 lazy regime——特征不变、学习率相对太慢，离真实特征学习（feature learning）场景有距离。
- **强假设**：依赖光滑动力学、单隐层、渐近宽度、tanh 等光滑激活；扩展到有限宽、非光滑激活、部分可观测、更丰富的连续控制 benchmark 都是自然的下一步。
- 实验仅限玩具 LQR，缺少对复杂环境（如 MuJoCo）的端到端验证；收敛性分析、regret 分析尚未展开。

## 相关工作与启发
- **连续时间 RL**：建立在 Doya、Jia & Zhou (2022/2023) 的连续时间 RL 框架上，并与 relaxed-control 探索（Wang et al. 2020）形成对比修正。
- **神经网络理论**：借鉴 NTK / 无限宽线性化（Jacot et al. 2018；Lee et al. 2019）、过参数化收敛理论（Allen-Zhu et al. 2019；Arora et al. 2019）把网络写成可解析形式。
- **随机逼近 ODE 视角**：与 Borkar & Meyn 的 ODE 方法、参数空间学习动力学分析互补，但本文创新在于刻画的是**状态分布**在梯度时间上的演化，而非参数轨迹。
- **启发**：这种"把过参数化网络在 RL 中的学习降维成少数随机变量的闭合系统"思路，为后续设计更简单的深度 RL 理论模型、乃至从理论反推算法改进提供了模板。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首次推导连续 RL 中状态分布的梯度时间演化方程，"两时钟 + 五变量闭合系统"框架在深度 RL 理论里是全新视角。
- **实验充分度**: ⭐⭐⭐ 作为纯理论论文实验仅起佐证作用，玩具 LQR 上验证了理论与算法吻合，但缺复杂环境与定量收敛/regret 分析。
- **写作质量**: ⭐⭐⭐⭐ 推导脉络清晰（配有证明流程图），两时钟隐喻直观；但密集的 SDE/Itô-Taylor 记号门槛较高。
- **价值**: ⭐⭐⭐⭐ 为连续控制深度 RL 提供了理论分析的新范式，搭起随机控制与过参数化 RL 的桥梁，对理论社区有较强奠基意义；实践价值需后续工作兑现。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Value Flows](value_flows.md)
- [\[ICLR 2026\] Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning](neuralsymbolic_approaches_for_interpretable_actor-critic_reinforcement_learning.md)
- [\[ICLR 2026\] Flowing Through States: Neural ODE Regularization for Reinforcement Learning](flowing_through_states_neural_ode_regularization_for_reinforcement_learning.md)
- [\[ICLR 2026\] MOBODY: Model-Based Off-Dynamics Offline Reinforcement Learning](mobody_model-based_off-dynamics_offline_reinforcement_learning.md)
- [\[ICLR 2026\] On Predictability of Reinforcement Learning Dynamics for Large Language Models](on_predictability_of_reinforcement_learning_dynamics_for_large_language_models.md)

</div>

<!-- RELATED:END -->
