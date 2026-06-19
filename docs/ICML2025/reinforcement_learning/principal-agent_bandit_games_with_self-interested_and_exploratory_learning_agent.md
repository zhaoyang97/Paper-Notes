---
title: >-
  [论文解读] Principal-Agent Bandit Games with Self-Interested and Exploratory Learning Agents
description: >-
  [ICML 2025][强化学习][principal-agent] 本文研究重复委托-代理赌臂博弈中，代理基于经验均值做决策（而非已知真实均值）且可能随机探索时，如何设计委托人的激励算法使后悔界达到 $\tilde{O}(\sqrt{T})$ 或 $\tilde{O}(T^{2/3})$，显著优于先前 $\tilde{O}(T^{11/12})$ 的结果。
tags:
  - "ICML 2025"
  - "强化学习"
  - "principal-agent"
  - "multi-armed bandits"
  - "incentive design"
  - "regret bound"
  - "exploration"
---

# Principal-Agent Bandit Games with Self-Interested and Exploratory Learning Agents

**会议**: ICML 2025  
**arXiv**: [2412.16318](https://arxiv.org/abs/2412.16318)  
**代码**: 无  
**领域**: 强化学习 / 博弈论  
**关键词**: principal-agent, multi-armed bandits, incentive design, regret bound, exploration

## 一句话总结
本文研究重复委托-代理赌臂博弈中，代理基于经验均值做决策（而非已知真实均值）且可能随机探索时，如何设计委托人的激励算法使后悔界达到 $\tilde{O}(\sqrt{T})$ 或 $\tilde{O}(T^{2/3})$，显著优于先前 $\tilde{O}(T^{11/12})$ 的结果。

## 研究背景与动机
**领域现状**: 委托-代理赌臂博弈建模在线市场场景——委托人（如电商平台）通过激励引导代理（用户）选择特定行为，以间接探索未知环境。现有工作普遍假设代理完全知道各臂真实期望奖励（oracle假设）。

**现有痛点**: 在真实场景中，代理也在学习——只能基于历史经验估计奖励。Dogan et al. (2023a) 放松了假设允许探索，但仍假设不探索时选真实最优臂，且后悔界高达 $\tilde{O}(T^{11/12})$。

**核心矛盾**: 代理经验均值不断更新导致最优激励随时间变化，委托人在不知代理经验均值时必须同时应对双方不确定性。

**本文目标**: (1) 更一般化的代理学习行为建模；(2) 更优后悔界的算法。

**切入角度**: 自利学习代理选"经验最优臂"而非"真实最优臂"，探索概率 $p_t \leq c_0\sqrt{t^{-1}\log(2t)}$。

**核心idea**: 新消除框架+不对称二分搜索适应经验均值波动+对坏臂适度采样稳定估计。

## 方法详解

### 整体框架
分阶段消除框架：维护好臂集合 $\mathcal{A}_m$ 和坏臂集合 $\mathcal{B}_m$。每阶段（1）对坏臂适度采样稳定估计量；（2）二分搜索各好臂的近最优激励；（3）用估计激励均匀探索好臂；（4）在线消除表现差的臂。

### 关键设计

1. **不对称二分搜索 (Algorithm 3)**:

    - 功能：为目标臂 $a$ 搜索近最优激励 $b_{m,a}$
    - 核心思路：跟踪最近成功激励 $y^{\text{upper}}$，失败时立即复验；若复验也失败则停止搜索。每轮仅需 $O(\log T)$ 步
    - 估计误差：$b_{m,a} - \pi_a^\star(t) \in (0, \frac{4}{T} + \frac{\lceil\log_2 T\rceil}{N_a(t)} + \frac{2}{\min_i N_i(t)}]$
    - 设计动机：传统对称检查需 $\log T$ 倍放大；不对称检查省去该对数因子

2. **放大激励**:

    - 将 $b_{m,a}$ 放大为 $\bar{b}_{m,a} = \min\{1+\frac{1}{T}, b_{m,a} + 4C_m + Z_m^{-1}\}$
    - $C_m = \sqrt{\frac{\log(4KT/\delta)}{2T_{m-1}}}$ 基于Hoeffding不等式控制未来波动

3. **坏臂采样**:

    - 每阶段让坏臂被采样 $Z_m = \sqrt{|\mathcal{A}_m| (\max\{1,|\mathcal{B}_m|\})^{-1} T_{m-1}}$ 次
    - 防止经典消除后某臂 $N_i$ 指数小于 $T_m$ 导致线性后悔

4. **在线消除**:

    - 委托人在激励中嵌入自己的估计值 $\hat{\theta}_a(t)$，间接比较联合估计 $\hat{\theta}_a + \hat{\mu}_a$
    - 若 $A_t \neq a$，则存在 $b$ 使得 $(\hat{\mu}_b + \hat{\theta}_b) - (\hat{\mu}_a + \hat{\theta}_a) \geq 3 \times 2^{-m}$

### 训练策略
探索性代理算法（Algorithm 5）加入概率放大：重复搜索消除流程对数多次，用中位数选择精炼活跃臂集合。

## 实验关键数据

### 主实验（理论后悔界）

| 代理行为 | 奖励模型 | 后悔界 |
|----------|----------|--------|
| 自利学习（无探索） | i.i.d. | $O(\sqrt{KT\log(KT)})$ |
| 探索性学习 | i.i.d. | $O(K^{1/3}T^{2/3}\log^{2/3}T)$ |
| Dogan模型简化 | i.i.d. | $O(\log^2(T)\sqrt{KT})$ |
| 自利学习 | 线性 | $O(d^{4/3}T^{2/3}\log^{2/3}T)$ |

### 与先前工作对比

| 方法 | 代理知道真实均值? | 后悔界 |
|------|------------------|--------|
| Dogan et al. (2023a) | 是 | $O(T^{11/12}\sqrt{\log T})$ |
| **本文 Alg.5** | **否** | $O(K^{1/3}T^{2/3}\log^{2/3}T)$ |

### 关键发现
- 后悔界匹配oracle设置下界 $\Omega(\sqrt{KT})$（忽略对数）
- 探索代理后悔从 $T^{11/12}$ 大幅降至 $T^{2/3}$
- 坏臂适度采样是防止线性后悔的关键

## 亮点与洞察
- **不对称检查**精巧利用"失败即复验"策略，快速检测越界，可推广到变化环境中的二分搜索
- **在线消除**巧妙将估计值嵌入激励中间接比较——信息不对称下的优雅方案
- 统一框架涵盖oracle/自利学习/探索代理作为特例

## 局限与展望
- i.i.d.奖励假设较强，未考虑时间相关性
- 线性设置下后悔 $T^{2/3}$ 与下界 $\sqrt{dT}$ 仍有gap
- 假设委托人知道代理选择策略和探索概率上界

## 相关工作与启发
- **vs Scheid et al. (2024b)**: 假设oracle代理，本文放松为自利学习代理，后悔界相当
- **vs Dogan et al. (2023a)**: 更一般模型下大幅改进后悔界至 $T^{2/3}$

## 评分
- 新颖性: ⭐⭐⭐⭐ 代理学习行为推广+新消除框架设计都有创新，不对称搜索方法巧妙
- 实验充分度: ⭐⭐ 纯理论无数值实验，缺少实际场景验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Table 1一目了然，Remark解释充分
- 价值: ⭐⭐⭐⭐ 对委托-代理赌臂问题的重要推进，更接近实际应用场景
- 总体: ⭐⭐⭐⭐ 理论贡献扎实，后悔界大幅改进具有显著意义

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals](learning_to_incentivize_in_repeated_principal-agent_problems_with_adversarial_ag.md)
- [\[ICML 2025\] Optimal and Practical Batched Linear Bandit Algorithm](optimal_and_practical_batched_linear_bandit_algorithm.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](../../ICLR2026/reinforcement_learning/nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)

</div>

<!-- RELATED:END -->
