---
title: >-
  [论文解读] Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions
description: >-
  [ICLR 2026][强化学习][无反向传播学习] 受 Forward-Forward 算法的"goodness 函数"启发，本文提出 **ARQ（动作条件化的均方根 Q 函数）**——把局部 RL 中每个细胞输出的隐向量用"减均值后求均方根（即标准差）"直接读成标量 Q 值，并把动作以 one-hot 拼到模型输入端做条件化，从而摆脱了此前无反向传播方法中"输出维度必须等于动作数"的限制，在 MinAtar 和 DeepMind Control 上既超过 SOTA 局部 RL 方法 AD，又在多数任务上击败了用反向传播训练的 DQN/SAC。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "无反向传播学习"
  - "Forward-Forward"
  - "局部强化学习"
  - "Q-learning"
  - "动作条件化"
  - "时序差分"
---

# Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pi4tbBMLsM](https://openreview.net/forum?id=pi4tbBMLsM)  
**代码**: 待确认  
**领域**: 强化学习 / 生物可塑性学习  
**关键词**: 无反向传播学习, Forward-Forward, 局部强化学习, Q-learning, 动作条件化, 时序差分  

## 一句话总结
受 Forward-Forward 算法的"goodness 函数"启发，本文提出 **ARQ（动作条件化的均方根 Q 函数）**——把局部 RL 中每个细胞输出的隐向量用"减均值后求均方根（即标准差）"直接读成标量 Q 值，并把动作以 one-hot 拼到模型输入端做条件化，从而摆脱了此前无反向传播方法中"输出维度必须等于动作数"的限制，在 MinAtar 和 DeepMind Control 上既超过 SOTA 局部 RL 方法 AD，又在多数任务上击败了用反向传播训练的 DQN/SAC。

## 研究背景与动机
- **领域现状**：反向传播虽是深度学习的基石，但其生物可塑性饱受质疑（需要同步计算与权重对称）。Hinton 的 Forward-Forward（FF）用两次前向传播替代前向+反向，按层贪心地最大化正样本、最小化负样本的"goodness"$G_z=\sum_i z_i^2$，是一种轻量、生物可信的无反向传播范式。
- **现有痛点**：绝大多数无反向传播工作都停留在**监督学习**上找梯度替代品，却忽视了 RL 这一更"天然"的学习信号来源——大脑本身就是奖励驱动进化、疑似在实现 TD 学习的系统。少数把局部学习引入 RL 的工作（如 Artificial Dopamine, AD）用两组映射的点积来产出每个动作的价值估计，但**点积输出维度被死死绑定为动作空间大小 $n_a$**，严重限制了每个细胞的表达力。
- **核心矛盾**：FF 的 goodness 衡量"输入与标签的兼容度"，而 RL 的 value 衡量"状态-动作对的可取程度"——两者本质都是对当前输入"可取性"的度量。能否把这条类比打通，用 goodness 函数从**任意维度**的隐向量里读出价值，从而解放局部 RL 网络的容量？
- **本文目标**：设计一种可即插即用替换标准 Q-learning 公式的局部价值估计机制，既保持无反向传播，又不受动作空间维度束缚。
- **核心 idea**：**【价值即 goodness】** 用隐向量的标准差（减均值后的 RMS）当作标量 Q 值，配合**【动作进输入】** 把动作候选拼到输入端，让网络为每个 state-action 对产出专属表示。

## 方法详解

### 整体框架
ARQ 沿用 AD 的多细胞堆叠结构，每个细胞同时接收下层、上层（带时序跳连）、原始观测以及**动作候选**作为输入，经过一个类注意力机制后得到隐向量 $y$，再对 $y$ 施加"减均值求均方根"的 goodness 函数得到标量 $Q(s,a)$；梯度只在细胞内部回传，整体保持无反向传播。关键改动有二：把输出读数从"点积投影到 $n_a$ 维"换成"任意维隐向量的 RMS"，把动作从"输出端按索引选"换成"输入端拼接条件化"。

```mermaid
flowchart LR
    S[状态 s_t] --> X[拼接输入 X]
    HB[下层 h_t^{l-1}] --> X
    HT[上层 h_{t-1}^{l+1}] --> X
    A[动作候选 a_t<br/>one-hot/二值] --> X
    X --> ATT[类注意力<br/>tanh(XᵀWatt2ᵀWatt1X)·h]
    ATT --> Y[隐向量 y<br/>维度 d 可任意]
    Y --> RMS[goodness: 减均值后 RMS]
    RMS --> Q[标量 Q(s,a)]
```

### 关键设计
**1. 均方根 goodness 函数：把价值读成隐向量的标准差。** ARQ 的灵魂在于不再像 AD 那样用一组线性投影把隐状态点积成 $n_a$ 个标量，而是直接对网络产出的隐向量 $y$ 做统计读数：先求均值 $\mu_y=\mathbb{E}_{y_i\in y}\,y_i$，再算减均值后的均方根 $Q_\theta(s,a)=\sqrt{\mathbb{E}_{y_i\in y}(y_i-\mu_y)^2}$，这恰好等于 $y$ 的标准差。相比 FF 原始的平方和 $\sum_i z_i^2$，先减均值再取 RMS 的设计是为了**防止 goodness 随单元数增多而爆炸**——这样隐向量维度 $d$ 可以自由放大而不影响数值尺度。这个读数过程**不含任何参数**，意味着任意架构产出的中间向量都能即插即用地被解释为价值估计，局部 RL 网络由此获得"输出可任意宽"的自由度。训练目标仍沿用 DQN 的 Bellman 均方误差 $\mathcal{L}_\theta=\big(R_t+\gamma\max_{a'}Q_\theta(S_{t+1},a')-Q_\theta(S_t,A_t)\big)^2$，无需改动现有 Q-learning 流程。

**2. 输入端动作条件化：让表示专属于 state-action 对。** 由于 goodness 函数天生只吐一个标量，把动作放到输入端就成了最自然的选择。ARQ 在拼接输入 $X=\mathrm{concat}(s_t,h_t^{l-1},h_{t-1}^{l+1},a_t)$ 时把动作候选 $a_t$ 一并喂进去（离散任务用 one-hot，连续任务按 Seyde 的 bang-bang 离散化成二值向量），整个网络针对"状态+这个动作"产出一个专属标量。这与 DQN/AD"只吃状态、输出 $n_a$ 维再按动作索引"形成鲜明对比。作者用 PCA 可视化论证了其价值：**不做条件化时，隐激活几乎完全按动作身份聚类、与 Q 值毫无关联**，动作相关方差霸占了表示空间；**加上输入条件化后，表示变得由状态主导、与 Q 值呈现温和正相关**，模型得以把容量分配给与价值真正相关的结构，而非隐式地去推断动作身份。

**3. 在 AD 之上落地：解绑维度以释放类注意力机制的容量。** 为公平对标 SOTA，ARQ 直接实现在 AD 架构上。单个细胞先算 $h_t^l=\mathrm{ReLU}(W_h X)$，再经类注意力 $y_t^l=\tanh(X^\top W_{att2}^\top W_{att1}X)\,h_t^l$，最后对 $y_t^l$ 取 RMS goodness。其中 $Z_1=W_{att1}X$、$Z_2=W_{att2}X$、$h_t^l$ 分别扮演自注意力里 query/key/value 的角色，$Z_2^\top Z_1$ 生成一张**跨特征维度**（而非跨 token）的交互图来重新分配信息。关键区别在于：AD 强制 $Z_2$ 宽度为 $n_a$，导致这张交互图被动作数卡死；而 ARQ 让 $Z_2$ 与 $y$ 的维度 $d$ 可自由选取。作者据此推断，正是"任意隐维度"让 ARQ 能充分吃下每个细胞内的非线性，叠加"输入端动作条件化"带来的 state-action 专属表示，两者合力才把局部 RL 的类注意力机制容量真正榨干。

## 实验关键数据

### 主实验表格
MinAtar（离散控制）与 DeepMind Control（连续控制）上的最终性能，均为 5 个随机种子的 mean ± 95% 置信区间。

| MinAtar | Freeway | Breakout | SpaceInvaders | Seaquest | Asterix |
|---|---|---|---|---|---|
| DQN (w/ BP) | 55.86 | 27.09 | 188.03 | 37.96 | 13.60 |
| AD (w/o BP) | 57.12 | 63.76 | 363.49 | 27.83 | 22.01 |
| **ARQ (Ours)** | **60.74** | **87.84** | **544.99** | **96.45** | **35.32** |

| DMC | Walker Walk | Walker Run | Hopper Hop | Cheetah Run | Reacher Hard |
|---|---|---|---|---|---|
| TD-MPC2 (w/ BP) | 958.80 | 834.07 | 348.55 | 808.46 | 934.84 |
| SAC (w/ BP) | 980.43 | 895.02 | 319.46 | 917.40 | 980.01 |
| AD (w/o BP) | 975.30 | 762.51 | 470.95 | 831.57 | 955.93 |
| **ARQ (Ours)** | 976.33 | 771.15 | **516.23** | 880.61 | 973.66 |

ARQ 在全部 5 个 MinAtar 游戏上一致超过 AD，并出人意料地**在全部游戏上超过 DQN**；DMC 上 ARQ 全面优于 AD，并在 Hopper Hop 等任务上反超用反向传播的 SAC/TD-MPC2。

### 消融实验表格
goodness 非线性选择对比（MinAtar，mean ± 95% CI）。RMS 优于均值/均方/方差等替代读数。

| 非线性函数 | Breakout | SpaceInvaders |
|---|---|---|
| **ARQ (RMS, 默认)** | **87.84** | **544.99** |
| Mean | 79.84 | 500.13 |
| MS (均方) | 82.10 | 434.88 |
| Var (方差) | 81.34 | 416.46 |
| AD | 67.40 | 369.96 |

### 关键发现
- **动作条件化对 ARQ 几乎是决定性的**：在 Breakout 上加输入端条件化把平均回报从约 55 提升到约 85（+50%），而同样的改动对 AD 只带来轻微提升——说明只有 RMS + 动作条件化组合在一起才让 ARQ 生效。
- **游戏机制分析**：ARQ 在 Breakout/SpaceInvaders 上大幅超 DQN，作者归因于 AD 的时序自顶向下连接提供了"连招"所需的时间连贯性；在策略呈双峰（攻击 vs. 补氧）的 Seaquest 上，AD 落后 DQN 而 ARQ 反超，体现动作条件化更能捕捉多模态策略结构。
- **稳定性来源猜想**：局部 TD 更新 + 缩短的梯度路径 + 逐层平均带来的方差削减，三者合力让 ARQ 比全反向传播网络学得更稳更快。

## 亮点与洞察
- **一个无参数读数解开维度枷锁**：把"价值=隐向量标准差"这一观察落成 RMS goodness，零参数、即插即用，却直接拆掉了 AD"输出维度=动作数"的硬约束，思路极简而锋利。
- **打通 FF 与 RL 的概念类比**：goodness（兼容度）↔ value（可取度）的类比不是修辞，而是被工程化成具体读数函数并跑出 SOTA，给"生物可信学习用于 RL"开了一扇门。
- **无反向传播却赢过反向传播**：在低维基准上把 DQN/SAC 这类 BP 方法多数任务掀翻，是该方向少见的"生物可信不再等于性能妥协"的实证。
- **PCA 解释有说服力**：用"激活按动作聚类 vs. 按状态聚类"的可视化把"为什么动作要进输入端"讲得直观可信，而非纯靠刷点。

## 局限与展望
- **仅限低维基准**：实验局限在 MinAtar（10×10 网格）与 DMC 低维观测，未验证在高维原始像素/大动作空间上局部方法是否仍可行。
- **连续控制靠离散化绕过**：连续动作通过 bang-bang 二值离散化处理，本质回避了真正的连续动作条件化，扩展性存疑。
- **对比式训练未用上**：作者明确把"用 replay buffer 采正负样本、按 FF 原始对比方式训练"留作未来工作，目前仍沿用 DQN 的 MSE 目标，FF 的对比精髓尚未完全发挥。
- **理论解释偏猜想**："任意维度释放非线性容量""逐层平均削减方差"等关键论断多以 conjecture/hypothesize 措辞给出，缺乏严格分析。

## 相关工作与启发
- **Forward-Forward（Hinton, 2022）**：本文 goodness 函数与无反向传播范式的直接来源；ARQ 的核心创新可视为"把 FF 的 goodness 从监督迁到 RL 价值估计"。
- **Artificial Dopamine（Guan et al., 2024）**：最直接的对标与实现底座，ARQ 在其类注意力细胞上做改造，针对性解决了其输出维度受限的痛点。
- **DQN / TD 学习（Mnih et al., 2013; Sutton, 1988）**：训练目标与价值估计范式的根基，ARQ 完全复用 Bellman MSE，因此能即插即用。
- **bang-bang 离散化（Seyde et al., 2021）**：为连续控制下的动作条件化提供了可行的离散化桥梁。
- **启发**：当某个模块的"输出读数方式"成为表达力瓶颈时，与其堆参数，不如换一个无参数但维度可自由的统计读数——RMS/标准差这类"尺度无关"的统计量是值得在更多即插即用场景里复用的工具。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把 FF 的 goodness 类比成 RL value 并落成无参数 RMS 读数，配合输入端动作条件化解绑维度，角度新巧；但整体是在 AD 上的精炼改造而非全新框架。
- **实验充分度**: ⭐⭐⭐ — 双基准 + 5 种子 + 动作条件化/非线性双消融 + PCA 分析较扎实，但仅限低维环境、缺高维与大规模验证。
- **写作质量**: ⭐⭐⭐⭐ — 动机链条（FF→RL→维度瓶颈）清晰，算法伪代码对照与图示到位，关键论断诚实标注为猜想。
- **价值**: ⭐⭐⭐⭐ — 为"生物可信无反向传播学习"在 RL 上提供了能超越 BP 基线的实证，方法即插即用、易被后续局部 RL 工作采纳。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Mean Flow Policy with Instantaneous Velocity Constraint for One-step Action Generation](mean_flow_policy_with_instantaneous_velocity_constraint_for_one-step_action_gene.md)
- [\[ICLR 2026\] Multistep Quasimetric Learning for Scalable Goal-Conditioned Reinforcement Learning](scaling_goal-conditioned_reinforcement_learning_with_multistep_quasimetric_dista.md)
- [\[ICLR 2026\] XQC: Well-Conditioned Optimization Accelerates Deep Reinforcement Learning](xqc_well-conditioned_optimization_accelerates_deep_reinforcement_learning.md)
- [\[ICLR 2026\] Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](occupancy_reward_shaping_improving_credit_assignment_for_offline_goal-conditione.md)
- [\[ICML 2026\] Direction-Conditioned Policies via Compositional Subgoal Scoring for Online Goal-Conditioned Reinforcement Learning](../../ICML2026/reinforcement_learning/direction-conditioned_policies_via_compositional_subgoal_scoring_for_online_goal.md)

</div>

<!-- RELATED:END -->
