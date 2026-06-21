---
title: >-
  [论文解读] Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring
description: >-
  [ICLR2026][强化学习][噪声鲁棒探索] 针对内在动机探索里经典的"噪声电视"陷阱，本文提出 **Learning Progress Monitoring (LPM)**：用"模型这一轮比上一轮进步了多少"当内在奖励，而不是用预测误差或新颖度——因为不可学习的随机转移不会带来任何进步，所以天然不会被噪声吸住；在 MNIST、3D 迷宫、Atari 上都比 SOTA 收敛更快、覆盖更多状态、外在回报更高。
tags:
  - "ICLR2026"
  - "强化学习"
  - "噪声鲁棒探索"
  - "Noisy-TV"
  - "学习进度"
  - "内在奖励"
  - "信息增益"
---

# Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=wzm38DRLhC](https://openreview.net/forum?id=wzm38DRLhC)  
**代码**: https://github.com/Akuna23Matata/LPM_exploration  
**领域**: 强化学习 / 内在动机探索  
**关键词**: 噪声鲁棒探索、Noisy-TV、学习进度、内在奖励、信息增益

## 一句话总结
针对内在动机探索里经典的"噪声电视"陷阱，本文提出 **Learning Progress Monitoring (LPM)**：用"模型这一轮比上一轮进步了多少"当内在奖励，而不是用预测误差或新颖度——因为不可学习的随机转移不会带来任何进步，所以天然不会被噪声吸住；在 MNIST、3D 迷宫、Atari 上都比 SOTA 收敛更快、覆盖更多状态、外在回报更高。

## 研究背景与动机
**领域现状**：在稀疏奖励的强化学习里，环境很久才给一次外在奖励，纯随机探索效率极低。主流解法是给智能体加一个**内在奖励** $r_t = r^e_t + \beta r^i_t$：好奇心方法（ICM、RND、Ensemble）用**预测误差/不确定性**当 $r^i_t$，鼓励去访问"模型预测不准"的转移；情节性新颖度方法（EME、EDT 等）则奖励"这一集里没见过的状态"。

**现有痛点**：这一整类方法都栽在一个老问题——**Noisy-TV（噪声电视）**。环境里只要有一个不可学习的随机源（一台播放雪花噪声的电视、一个按了就出随机画面的遥控器），好奇心智能体就会被它牢牢吸住：因为雪花噪声永远"预测不准"，预测误差/新颖度永远很高，智能体宁可一直盯着电视也不去探索真正有用的区域。

**核心矛盾**：噪声电视的本质，是内在奖励**分不清两种不确定性**——**认知不确定性（epistemic）**是模型因为数据少而无知，多收集数据就能消除；**偶然不确定性（aleatoric）**是环境本身的随机性（传感器噪声、雪花），再多数据也消不掉。现有"噪声鲁棒"方法（AMA、EME、EDT）走的路线是先**把这两种不确定性分离开**，再只奖励前者。但分离本身极难，往往需要强先验或海量数据，**早期训练阶段奖励仍被噪声主导**，白白浪费大量样本。

**切入角度**：作者从神经科学的一个发现出发——人类探索时会**监控自己的学习进度**，倾向于去看"让我学到最多东西"的转移。看一个不可学习的转移**不会产生学习进度**，所以这个策略天生对噪声电视免疫，根本不需要先去显式分离两种不确定性。

**核心 idea**：把内在奖励从"预测误差/新颖度"换成"**模型改进量**"——奖励的是"我的世界模型这一轮比上一轮变好了多少"，而不是"我预测得多差"。雪花噪声永远学不会、模型永远不进步，于是它的内在奖励**直接为零**，问题从源头消失。

## 方法详解

### 整体框架
LPM 工作在**基于模型的 RL** 设定下：智能体维护一个动力学模型 $f_\theta$，给定当前观测 $o_t$ 和动作 $a_t$ 预测下一观测 $\hat o_{t+1}$。用 $t$ 表示环境步、$\tau$ 表示模型更新步，每隔 $N$ 个环境步更新一次模型（即 $f^{(\tau)}_\theta$ 是第 $\tau$ 次更新后的模型）。

核心思路是：动力学模型这一轮相比上一轮的**误差下降量**，就是"学到了多少"，应该拿来当内在奖励。直接的做法是把上一轮模型 $f^{(\tau-1)}_\theta$ 存下来在当前样本上重新跑一遍——但本文不这么做，而是另起一个**误差模型 $g_\phi$** 去预测"上一轮模型在这个状态上的**期望误差**"。整条流水线就是：动力学模型预测 → 算当前误差 → 误差模型给出上一轮的期望误差 → 两者相减得到内在奖励 → 与外在奖励合并去更新策略，每 $N$ 步同步更新两个模型。这个"用期望误差而非单点误差"的选择，是后面理论能成立的关键。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["观测 o_t, 动作 a_t"] --> B["动力学模型 f_θ<br/>预测下一观测 ô_{t+1}"]
    B --> C["学习进度奖励<br/>当前误差 ε^(τ)"]
    A --> D["双网络误差模型 g_φ<br/>预测上一轮期望误差"]
    D --> C
    C -->|r_i = g_φ − ε^(τ)| E["合成奖励 r = r_e + β·r_i<br/>更新策略 π"]
    E -->|每 N 步同步更新 f_θ, g_φ| B
```

### 关键设计

**1. 学习进度奖励：奖励"模型变好了多少"，而不是"预测有多差"**

这是全文的范式转变，直接针对 Noisy-TV。先定义第 $\tau$ 轮动力学模型在时刻 $t$ 的对数 MSE 误差：

$$\varepsilon^{(\tau)}_t(o_{t+1}) = \log\left(\frac{1}{\dim(\Omega)}\,\big\|o_{t+1} - f^{(\tau)}_\theta(o_t, a_t)\big\|_F^2\right)$$

直觉上，$\varepsilon^{(\tau-1)}_t - \varepsilon^{(\tau)}_t$ 就刻画了"模型经过第 $\tau$ 次更新后，在这个转移上预测精度提升了多少"。LPM 把这个**误差下降量**当内在奖励。关键区别在于：好奇心方法奖励的是 $\varepsilon^{(\tau)}_t$ 本身（误差大就奖励高），所以雪花噪声误差永远大、永远被奖励；而 LPM 奖励的是**误差的变化量**——雪花噪声不可学习，模型在它上面永远学不会、误差不下降，进步量恒为 0，于是内在奖励自动为零。智能体不需要"判断这是不是噪声"，只要噪声不带来进步，它就自动失去吸引力。

**2. 双网络误差模型 $g_\phi$：用"期望误差"代替"单点旧误差"**

要算误差下降量，就得知道"上一轮模型在当前样本上的误差"。朴素做法是缓存旧模型 $f^{(\tau-1)}_\theta$ 重新推理，但本文改用一个独立的**误差模型** $g_\phi: \mathcal{O}\times\mathcal{A}\to\mathbb{R}$ 去**回归上一轮模型的期望误差**：

$$g^{(\tau)}_\phi(o_t, a_t) \approx \mathbb{E}_D\!\left[\varepsilon^{(\tau-1)}_t(o_{t+1})\right]$$

为此维护一个固定大小 $d$ 的回放队列 $D$，每条记录 $(o_t, a_t, \varepsilon^{(\tau)}_t)$；$g_\phi$ 用 $D$ 里"上一轮模型产生的误差"来训练。最终内在奖励写成：

$$r^i_t = \mathbb{E}_D\!\left[\varepsilon^{(\tau-1)}_t(o_{t+1})\right] - \varepsilon^{(\tau)}_t(o_{t+1}) = g^{(\tau)}_\phi(o_t, a_t) - \varepsilon^{(\tau)}_t(o_{t+1})$$

之所以非要取**期望**而不是用单点旧误差，是因为单点误差本身带有偶然噪声的抖动，会让奖励忽正忽负、甚至在真有信息增益时变成负值（见下方理论）。用 $g_\phi$ 维护一个"期望误差"的平滑估计，奖励才稳定可靠地追踪真实学习进度。算法上每步算当前误差、推入两个缓冲，当 $|D|=d$ 时才给出 $r^i_t$（否则置 0），每 $N$ 步同步更新 $f_\theta$ 和 $g_\phi$。

**3. 与信息增益的单调对应：从理论上证明"进步=信息"**

作者把信息增益定义为后验相对先验的 KL 散度 $\mathrm{IG} := \mathbb{E}_{p(\theta|D)}[\log p(D|\theta)] - \log p(D) = \mathrm{KL}(p(\theta|D)\,\|\,p(\theta))$，在 i.i.d. 高斯观测假设下用 log-MSE 作为似然的代理。**定理 4.1** 证明 LPM 的内在奖励满足 $r^i \ge \tfrac{1}{c}\,\mathrm{IG}$、且 $\mathrm{IG}=0 \Leftrightarrow r^i=0$（在可辨识条件下双向成立）——即 $r^i$ 是信息增益的**零等变（zero-equivariant）单调指示量**：奖励为正当且仅当模型真的学到了新东西，模型啥也没学到（如盯着雪花）时奖励恰为零。**定理 4.2** 进一步说明误差模型的**期望**操作不可省：若改用单点内在奖励 $r^{i,\text{point}} = \log\mathrm{MSE}(\theta) - \log\mathrm{MSE}(\theta_D)$，存在 $\theta$ 使得 $r^{i,\text{point}}<0$ 但 $\mathrm{IG}>0$，单调性被打破。这就从理论上回答了"为什么要双网络/为什么要取期望"。

### 损失函数 / 训练策略
动力学模型 $f_\theta$ 用回放缓冲 $B$ 的 $(o_t,a_t,o_{t+1})$ 拟合下一观测；误差模型 $g_\phi$ 用固定队列 $D$ 拟合"上一轮模型的误差"。两者每 $N$ 个环境步同步更新一次。策略 $\pi$ 用合成奖励 $r_t = r^e_t + \beta r^i_t$ 配任意 RL 算法（如 PPO）训练。仅当 $|D|=d$ 队列填满后才发放内在奖励，避免早期估计不可靠。

## 实验关键数据

实验围绕三个问题展开：LPM 内在奖励是否收敛更快且对偶然不确定性鲁棒？纯探索任务里是否覆盖更多状态？带外在奖励任务里是否拿到更高回报？环境从 Noisy MNIST → MiniWorld 3D 迷宫（160×120 RGB）→ MountainCar 连续控制 → Atari，复杂度递增，每个都设确定性/状态噪声/动作噪声多种条件。

### 主实验：MountainCar 连续控制状态覆盖率（%）

| 方法 | 确定性覆盖 | 随机性覆盖 | 下降幅度 |
|------|-----------|-----------|---------|
| **LPM (本文)** | 76.50 ± 9.08 | **67.04 ± 14.60** | **12.4%** |
| Ensemble | 91.22 ± 2.04 | 61.02 ± 5.03 | 33.1% |
| EDT | 82.16 ± 13.57 | 53.52 ± 10.53 | 34.9% |
| EME | 89.16 ± 3.40 | 32.46 ± 11.31 | 63.6% |
| RND | 45.50 ± 14.53 | 28.00 ± 10.10 | 38.5% |
| AMA | 33.00 ± 12.31 | 13.20 ± 4.62 | 60.0% |
| IDF | 90.92 ± 5.13 | 12.80 ± 3.19 | 85.9% |

关键看点：很多 baseline（Ensemble/EME/IDF）在**确定性**环境覆盖率比 LPM 还高，但一加噪声就崩——IDF 暴跌 85.9%、EME 跌 63.6%；而 LPM 仅跌 12.4%，随机条件下覆盖率反超所有方法。这正是"噪声鲁棒"的直接体现。

### 其他环境对比

| 环境 | 指标 | LPM 表现 | 对比 |
|------|------|---------|------|
| Noisy MNIST | 内在奖励收敛步数 | ≈150 步收敛到 0 | AMA 需 ≈400 步；EDT 始终无法收敛，一直觉得随机转移有趣 |
| MiniWorld 3D 迷宫 | 平均访问状态数 | **1347.6** | 比次优高 95.3 个状态（+7.6%），且确定/状态噪声/动作噪声三档都稳定 |
| Atari（6 局） | 外在回报 | 6 局中 4 局最优 | Space Invader 加噪仅掉 3.9%、UpNDown 掉 4.7%、Ms PacMan 反升 0.3%；EME 在干净 Space Invader 最强但加噪后 100% 崩 |
| Montezuma's Revenge | 拿到非平凡回报所需步数 | 20M 步 | RND 需 50M 步；NGU 跑 50M 步仍为 0 |

### 关键发现
- **范式转变是核心增益来源**：奖励"进步量"而非"误差/新颖度"，让噪声的内在奖励从源头归零，不再需要昂贵的不确定性分离。
- **双网络的期望误差很重要**：理论与实验都表明，用单点旧误差会让奖励抖动甚至变负，破坏与信息增益的单调对应；用 $g_\phi$ 估计期望误差后奖励稳定。
- **鲁棒性体现在"加噪后掉点最少"**：LPM 在确定性环境不一定是覆盖率第一，但加噪后退化幅度远小于所有 baseline，这是它最突出的特征。
- **计算开销可控**：虽是双网络，但与 AMA（双头网络）相当，远低于 Ensemble/EME（多模型）。

> ⚠️ 上述 Atari 的 128 随机种子、MNIST 5 次运行等数字均来自原文图注，不同环境/任务难度不可直接横向比大小。

## 亮点与洞察
- **"奖励进步而非误差"是个极简却深刻的视角转换**：一行公式 $r^i_t = g^{(\tau)}_\phi - \varepsilon^{(\tau)}_t$ 就让 Noisy-TV 失效，不必显式区分认知/偶然不确定性——这是本文最"啊哈"的地方。
- **神经科学到算法的映射很干净**：人类"监控学习进度"的观察，直接对应"奖励模型改进量"，动机具体且可证明（单调对应信息增益）。
- **理论给出了双网络的必要性**：很多方法的辅助网络是工程经验，本文用定理 4.2 证明"取期望"不可省，把设计选择从 trick 提升为有理论支撑的必需品。
- **可迁移性**：这种"用模型改进量当奖励"的思路可推广到任何带可学习世界模型的探索/主动学习场景（主动感知、机器人自监督探索），只要能定义"这一轮比上一轮进步多少"。

## 局限与展望
- **理论依赖 i.i.d. 高斯观测假设**：作者承认在该假设之外，定理 4.1/4.2 的鲁棒性难以分析，目前靠实验补足。
- **基于模型设定**：LPM 需要一个可周期性更新的动力学模型 $f_\theta$ 与误差模型 $g_\phi$，对纯 model-free 流程不是即插即用。
- **更新周期 $N$、队列大小 $d$、权重 $\beta$ 等超参的敏感性**正文着墨不多，跨环境的鲁棒区间仍待系统刻画。
- **极端长程任务仍是边界**：Montezuma 上虽优于 RND/NGU，但拿到的也只是"非平凡"分数，离真正攻克长程稀疏奖励还有距离。

## 相关工作与启发
- **vs 好奇心方法（ICM/RND/Ensemble）**：它们奖励预测误差或模型分歧，噪声让误差恒高所以会被吸住；LPM 奖励误差的下降量，噪声不带来下降故奖励归零。
- **vs AMA（偶然不确定性过滤）**：AMA 显式估计并扣掉偶然不确定性，需要大量数据才能可靠分离，早期仍被噪声主导（MNIST 上 ≈400 步才收敛）；LPM 不做分离，≈150 步即收敛。
- **vs 情节新颖度方法（EME/EDT）**：它们靠"状态新颖/相似度"奖励，随机转移天然"最新颖"，反而最被噪声吸引（EDT 始终无法收敛）；LPM 以学习进度为信号，绕开了新颖度被噪声劫持的问题。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把内在奖励从"误差/新颖度"换成"学习进度"，是对 Noisy-TV 的范式级重述。
- 实验充分度: ⭐⭐⭐⭐ MNIST→迷宫→连续控制→Atari→Montezuma 覆盖全面，但超参敏感性分析略少。
- 写作质量: ⭐⭐⭐⭐ 动机—方法—理论—实验逻辑清晰，理论与设计选择对应紧密；个别表述有小笔误。
- 价值: ⭐⭐⭐⭐⭐ 概念极简、有理论保证、计算开销低，对噪声鲁棒探索很有实用与启发价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Diversity-Incentivized Exploration for Versatile Reasoning](diversity-incentivized_exploration_for_versatile_reasoning.md)
- [\[ICLR 2026\] AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust Exploration](alphasage_structure-aware_alpha_mining_via_gflownets_for_robust_exploration.md)
- [\[ICML 2025\] Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](../../ICML2025/reinforcement_learning/robust_noise_attenuation_via_adaptive_pooling_of_transformer_outputs.md)
- [\[ICML 2025\] Learning Progress Driven Multi-Agent Curriculum](../../ICML2025/reinforcement_learning/learning_progress_driven_multi-agent_curriculum.md)
- [\[ICLR 2026\] Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning](beyond_distributions_geometric_action_control_for_continuous_reinforcement_learn.md)

</div>

<!-- RELATED:END -->
