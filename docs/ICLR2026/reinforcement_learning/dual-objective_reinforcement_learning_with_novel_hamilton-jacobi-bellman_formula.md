---
title: >-
  [论文解读] Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations
description: >-
  [ICLR2026][强化学习][Hamilton-Jacobi-Bellman] 本文把 HJ-RL（Hamilton-Jacobi 视角的强化学习）从单一的「到达 / 避障 / 可达-避免」扩展到两类双目标问题——到达后持续避障（Reach-Always-Avoid, RAA）和先后到达两个目标（Reach-Reach, RR），证明它们的 Bellman 方程可以**精确分解**为已研究过的简单可达/避免子问题的组合，并据此设计了 DOHJ-PPO 算法，在成功率、安全性和速度上全面超越 10 个拉格朗日类与 HJ-RL 基线。
tags:
  - "ICLR2026"
  - "强化学习"
  - "Hamilton-Jacobi-Bellman"
  - "安全强化学习"
  - "可达-避免"
  - "双目标"
  - "值函数分解"
---

# Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=1SdPgRQrr5](https://openreview.net/forum?id=1SdPgRQrr5)  
**代码**: 待确认  
**领域**: 强化学习  
**关键词**: Hamilton-Jacobi-Bellman、安全强化学习、可达-避免、双目标、值函数分解  

## 一句话总结
本文把 HJ-RL（Hamilton-Jacobi 视角的强化学习）从单一的「到达 / 避障 / 可达-避免」扩展到两类双目标问题——到达后持续避障（Reach-Always-Avoid, RAA）和先后到达两个目标（Reach-Reach, RR），证明它们的 Bellman 方程可以**精确分解**为已研究过的简单可达/避免子问题的组合，并据此设计了 DOHJ-PPO 算法，在成功率、安全性和速度上全面超越 10 个拉格朗日类与 HJ-RL 基线。

## 研究背景与动机
**领域现状**：在安全强化学习里，既要完成任务（到达目标）又要满足约束（避开危险）是核心诉求。主流做法是把约束折算进奖励——拉格朗日方法（CPPO、PPO-LAG 等）引入一个乘子，把「安全违规的折扣累积和低于阈值」这类约束变成混合目标来优化。另一条线是 HJ-RL：从动态规划的 Hamilton-Jacobi 视角出发，用**非常规的 Bellman 更新**来求解安全和到达任务，代表性的有安全 Bellman 方程（SBE）和可达-避免 Bellman 方程（RABE）。

**现有痛点**：拉格朗日方法需要精细的奖励工程和乘子调参，而且因为优化的是折扣累积和，约束满足只是「期望意义上」的、缺乏直接可解释性。HJ-RL 这一边虽然漂亮——它的值函数传播的是轨迹上的**极值**（最坏惩罚、最好奖励）而非折扣和，因此一个正的值就等价于「任务一定能完成」——但到目前为止 HJ-RL 只能处理三种原子操作：到达（Reach）、避免（Avoid）、可达-避免（Reach-Avoid），无法表达更复合的任务。

**核心矛盾**：很多实际任务本质上是**双目标耦合**的。比如战机飞进指定空域后还得持续不坠机（到达后还要一直避险），或者机器人要按任意顺序拜访两个目标点。这类任务用折扣和奖励去配平两个竞争目标极难调，而现有 HJ-RL 又缺少对应的 Bellman 形式。时序逻辑（LTL）方法虽然能描述这类复合规约，但通常要构造一个自动机、对状态做乘积扩展，求解路径和 HJ 的极值优化是两套体系，难以直接复用 HJ-RL 的安全性优势。

**本文目标**：为两个互补的双目标问题——RAA（到达一个目标且永远避开危险）和 RR（按任意顺序到达两个目标）——推导出**显式、可解的** Bellman 形式，并能直接套用现成的 HJ-RL 求解器。

**切入角度**：作者观察到 RAA 和 RR 虽然主题不同，但在数学上是**互补的**（只差一个 max/min 操作），而且它们的值函数有一种「可分解」的基本结构——复合目标可以拆成若干个已经会解的原子目标的组合。

**核心 idea**：用「值函数分解」代替「自动机构造」——证明 RAA 的值等于一个 Avoid 子问题 + 一个用「修正奖励」定义的 Reach-Avoid 子问题的组合，RR 的值等于三个 Reach 子问题的组合，从而把复合 HJ-RL 问题归约到已解决的简单问题上。

## 方法详解

### 整体框架
本文要解决的是：在确定性 MDP $M = \langle S, A, f \rangle$ 上，求一个策略最大化「双目标的最坏情况结果」。两个问题的目标函数分别写成轨迹上极值的组合：

$$V^\pi_{RAA}(s) = \min\left\{ \max_t r(s^\pi_t),\ \min_t q(s^\pi_t) \right\}, \qquad V^\pi_{RR}(s) = \min\left\{ \max_t r_1(s^\pi_t),\ \max_t r_2(s^\pi_t) \right\}$$

其中 $r$ 是要最大化的奖励、$p$ 是要最小化的惩罚，为方便记 $q = -p$（于是「最小化最坏惩罚」变成「最大化最小的 $q$」）。RAA 取「最好奖励」与「最坏惩罚」的较小者，逼着智能体在某个时刻进入目标、且全程不踩危险区；RR 取两个「最好奖励」的较小者，逼着智能体把两个目标都访问到。这类值和 RL 里常见的无穷折扣和**根本不同**——它不沿轨迹累加，而是由轨迹上的某些极值点决定，且有直接语义：若 $r$ 只在目标区 $G$ 内为正、$q$ 只在危险区 $H$ 内为负，则 $V^\pi_{RAA}(s) > 0$ 当且仅当策略 $\pi$ 真的完成了 RAA 任务。

整个方法分三步推进：（1）指出朴素策略 $\pi: S \to A$ 在这类多目标无穷时域问题上**本质不足**，必须做状态增广；（2）证明增广后的 RAA/RR 值可以**精确分解**为简单 HJ 子问题的组合（Theorem 1、2），并证明该增广是最优的（Theorem 3）；（3）把确定性的分解结论松弛为随机版本（SRBE/SRABE），设计 DOHJ-PPO 用 PPO 同时学复合表示和分解表示。

### 关键设计

**1. 状态增广：用历史极值变量补全马尔可夫性的缺口**

朴素地找一个只依赖当前状态的策略 $\pi: S \to A$ 在这里是有缺陷的。问题不出在状态转移（转移本身是马尔可夫的），而出在奖励的**非马尔可夫性**：在无穷时域下配平多个目标时，最优动作往往依赖于轨迹历史而不只是当前状态。论文用 Figure 3 给了直观例子——RR 里一个无记忆智能体在中间状态只能把它和某一个固定动作绑定，于是只能到达一个目标；RAA 里机器人能否冒险穿过锥形障碍去够目标，取决于它**之前是否已经到过目标**，光看当前状态决定不了。

解法是把 MDP 增广为 $\bar M$，状态变成 $\mathcal S = S \times Y \times Z$，新增两个辅助变量实时跟踪「到目前为止的最好奖励 $y$ 和最坏惩罚 $z$」。对 RAA：

$$y^{\bar\pi}_{t+1} = \max\left\{ r(s^{\bar\pi}_{t+1}),\ y^{\bar\pi}_t \right\}, \qquad z^{\bar\pi}_{t+1} = \min\left\{ q(s^{\bar\pi}_{t+1}),\ z^{\bar\pi}_t \right\}$$

对 RR 则把 $z$ 的更新也改成 max（因为两个都是要最大化的奖励 $r_1, r_2$）。增广后策略 $\bar\pi: \mathcal S \to A$ 就能利用历史信息。值得强调的是，增广的目的**不是**把奖励变成马尔可夫的（它们本来就是当前状态的确定性函数），而是因为这里用的是 HJ 式的极值优化目标而非折扣和，需要把「已经实现的极值」显式喂给策略。

**2. RAA 分解定理：拆成一个 Avoid + 一个修正奖励的 Reach-Avoid（Theorem 1）**

这是核心理论贡献之一。直接求 RAA 值很难，但 Theorem 1 证明它可以先解一个对应惩罚 $q$ 的避免问题拿到 $V^*_A(s) = \max_\pi \min_t q(s^\pi_t)$，再用一个**修正后的奖励**去解一个标准的可达-避免问题：

$$V^*_{RAA}(s) = \max_\pi \max_t \min\left\{ \tilde r_{RAA}(s^\pi_t),\ \min_{\tau \le t} q(s^\pi_\tau) \right\}, \qquad \tilde r_{RAA}(s) := \min\left\{ r(s),\ V^*_A(s) \right\}$$

直觉是：修正奖励 $\tilde r_{RAA}$ 把「原始奖励 $r$」和「从该状态出发还能保证的避险能力 $V^*_A$」取小——也就是只有当一个状态既在目标里、又处在「之后还能一直避开危险」的安全区时，它才算真正的好状态。这样一来 RAA 就退化成对 $\tilde r_{RAA}$ 的可达-避免问题，可以直接用 Hsu 等人的 RABE 求解器。对应的 Bellman 方程（Corollary 1）为：

$$V^*_{RAA}(s) = \min\left\{ \max\left\{ \max_{a \in A} V^*_{RAA}(f(s,a)),\ \tilde r_{RAA}(s) \right\},\ q(s) \right\}$$

它在结构上和 RABE 同形，只是把奖励项换成了 $\tilde r_{RAA}$。和时序逻辑里对谓词做布尔代数分解不同，这里的分解是为 HJ 式安全最优控制量身定制的（论文附录 L 论证了 TL 的谓词代数不足以胜任）。

**3. RR 分解定理：拆成三个 Reach 问题（Theorem 2）**

RR 的分解与 RAA 对偶。先解两个奖励各自的到达问题，得到 $V^*_{R1}(s) = \max_\pi \max_t r_1(s^\pi_t)$ 和 $V^*_{R2}(s)$，再用一个修正奖励解第三个到达问题：

$$V^*_{RR}(s) = \max_\pi \max_t \tilde r_{RR}(s^\pi_t), \qquad \tilde r_{RR}(s) := \max\left\{ \min\left\{ r_1(s), V^*_{R2}(s) \right\},\ \min\left\{ r_2(s), V^*_{R1}(s) \right\} \right\}$$

$\tilde r_{RR}$ 的含义是：一个状态值得停留，当且仅当它要么「已在目标 1 且之后还能够到目标 2」，要么「已在目标 2 且之后还能够到目标 1」——两条路径取较优。这样 RR 就变成对 $\tilde r_{RR}$ 的纯到达问题，Bellman 方程（Corollary 2）退化为标准 Reach 形式 $V^*_{RR}(s) = \max\{ \max_a V^*_{RR}(f(s,a)),\ \tilde r_{RR}(s) \}$。

**4. 增广最优性保证（Theorem 3）：再多的历史也无用**

光证明能分解还不够，还要确认这个特定的 $(y, z)$ 增广是「够用」的。Theorem 3 给出夹逼式的结论：

$$\max_\pi V^\pi_{RAA}(s) \le \max_{\bar\pi} V^{\bar\pi}_{RAA}(s) = \max_{a_0, a_1, \dots} \min\left\{ \max_t r(s_t),\ \min_t q(s_t) \right\}$$

右端是「能预知全部动作序列时」的理论最优，左端是无记忆策略能达到的上界。定理说明：最优增广策略恰好达到右端的理论最优值（一般 $\ge$ 无记忆策略），而且**再往状态里塞更多历史信息也无法提升**最优策略的表现。这就为只跟踪 $(y, z)$ 两个极值变量提供了严格依据——它既必要又充分。

### 损失函数 / 训练策略
理论分解假设了动态确定、且能预先解出分解子值，这两点在实际 RL 里都受限。DOHJ-PPO 通过两步把理论落地为算法。

**随机松弛（SRBE / SRABE）**：高性能 RL 需要随机策略。沿用 So 等人的随机可达 Bellman 方程（SRBE）的思路，本文把它推广为随机可达-避免 Bellman 方程（SRABE）：

$$\hat V^\pi_{RAA}(s) = \mathbb E_{a \sim \pi}\left[ \min\left\{ \max\left\{ \hat V^\pi_{RAA}(f(s,a)),\ \tilde r_{RAA}(s) \right\},\ q(s) \right\} \right]$$

实际用的是带折扣 $\gamma$ 的版本（保证收缩性），在 $\gamma \to 1^-$ 取极限恢复无折扣值；对应的动作值 $\hat Q^{\gamma,\pi}_{RAA}(s,a)$ 去掉外层期望即得，PPO 的优势函数取 $\hat A^\pi_{RAA} = \hat Q_{RAA} - \hat V_{RAA}$。这个松弛相当于把极值算子和期望算子的顺序对调——二者本不可交换，论文证明它给出的是值的**保守估计**（附录 D），并在随机动态实验里验证其有效性。

**三处对 PPO 的最小改动 + 协同自举**：DOHJ-PPO 在 PPO 上做三处改动。其一，为分解出来的子目标引入额外的 actor 和 critic（按 Theorem 1、2，复合值由更简单的 R/A/RA 值组合而成，于是每个分解项用自己的网络去学）。其二，复合 actor/critic 与分解 actor/critic **同时训练**而非先后训练——每轮迭代里各 actor 各自 rollout，复合表示更新时，分解值从当前的分解 critic 沿复合轨迹自举推断，再通过带特殊 RAA/RR 修正奖励的 GAE 和 target 整合进来。其三，**耦合重置（coupled resets）**：训练分解 actor/critic 的轨迹，其初始状态从复合轨迹采样。这是为了避免「独立先解分解目标」带来的偏差——比如 RAA 里若让 avoid 分解独自收敛，它会只顾避险、跑到一个与奖励无关的安全角落里，得到的值估计虽优却与整体任务错位；用复合轨迹的状态来初始化分解 rollout，就把分解学习锚定在 PPO 的 on-policy 数据分布上。

## 实验关键数据

### 主实验
实验分两层：先用 DQN 在 2D 网格世界做定性演示（Figure 2），验证分解定理诱导出的行为确实不同；再用 DOHJ-PPO 在 Hopper、F16、SafetyGym、HalfCheetah 四个连续控制环境上做定量对比（Figure 4），每个算法在 1000 条轨迹上评估三项指标：**Success**（同时完成两个子目标的轨迹占比）、**Partial Success**（至少完成一个子目标的占比）、以及达成所需的平均步数（速度）。

| 设置 | 对比维度 | DOHJ-PPO 表现 | 说明 |
|------|---------|--------------|------|
| 网格世界 DQN（RAA） | vs 经典 RA | RA 会停在「之后必然碰撞」的区域；RAA 学会到达后停在安全区 | 定性验证 Theorem 1 |
| 网格世界 DQN（RR） | vs 单 Reach | 单 Reach 到一个目标就停；RR 诱导轨迹访问两个目标 | 定性验证 Theorem 2 |
| 4 个连续控制环境 | vs 10 个 SOTA 基线 | 所有任务/环境均居第一或第二；环境越复杂（如 HalfCheetah）领先越明显 | Success 维度全面领先 |

对比的 10 个基线覆盖三类：拉格朗日类（CPPO、PPO-LAG、P2BPO、LOGBAR）、HJ-RL 类（RESPO、RCPPO、RA）、以及 STL/LTL-RL 与 MORL 类（D-STL PPO、SPARSE STL PPO、MORL-PPO）。关键现象是：**几乎所有算法都能拿到很高的 Partial Success，但极少有算法能拿到高 Success**——这说明「完成单个目标」不难，难的是同时配平两个竞争目标，尤其在折扣和奖励框架下。DOHJ-PPO 是唯一在 RAA 和 RR 两类任务上都强、且达成速度最快的算法。

### 消融实验
核心消融是**随机动态鲁棒性**（Figure 5）：在 HalfCheetah 的速度/角速度上注入仿射高斯噪声，强度分为 null、低（0.5）、中（1.0）、高（2.0）四档，与各任务前三名基线比较 256 条轨迹的 Success（即 $V_{RAA} > 0$ 或 $V_{RR} > 0$ 的占比），取三个种子的最大学习曲线。

| 任务 | 噪声档 | DOHJ-PPO 相对第二名 | 说明 |
|------|--------|---------------------|------|
| RAA | 中等 | 峰值性能领先 8%–22% | 且最快到峰值；噪声极高时各算法都退化到同样差 |
| RR | 多数档位 | 峰值性能领先 >30% | 中等噪声档仍领先最佳基线 DSTL >15% |
| RR | 中等噪声 | 比 DSTL 慢到峰值，但峰值约为其 2 倍 | 速度与性能的权衡 |

### 关键发现
- **极值传播是制胜关键**：DOHJ-PPO 的优势源于新 Bellman 更新传播的是轨迹**极值**（最大/最小）而非短期平均（折扣和）。折扣和会把两个竞争目标揉成一个标量去配平，极易顾此失彼；极值形式天然保留「最坏子目标」的语义，因而能稳定地同时满足双目标。
- **Partial vs Full Success 的鸿沟**揭示了双目标 RL 的真正难点：拿到单目标（Partial）对几乎所有方法都容易，区分高下的是能否同时拿下两者（Full）。
- **随机松弛是保守而有效的近似**：SRBE/SRABE 把期望和极值算子对调虽不严格（二者不可交换），但实验证明它在随机策略和一定程度的随机动态下都给出可靠（保守）估计。作者也坦言随机动态下尚无理论保证。

## 亮点与洞察
- **「分解」替代「自动机」是优雅的范式迁移**：LTL-RL 处理复合任务靠构造自动机做状态乘积，本文则证明复合 HJ 值可被**精确**拆成已解决的原子 HJ 子问题。这把复合安全 RL 从「需要新求解器」变成「复用旧求解器 + 组合」，是可复用性极强的思路。
- **修正奖励 $\tilde r$ 的构造很巧**：RAA 里 $\tilde r_{RAA} = \min\{r, V^*_A\}$ 把「是不是目标」和「之后还能不能保证安全」耦合进一个标量奖励，让复合问题退化为标准 RA 问题——这种「用子问题的最优值去重塑奖励」的技巧可迁移到其他需要前瞻性约束的任务。
- **协同自举 + 耦合重置**解决了分解类方法的通病：分开学子目标会让子策略跑偏到与整体任务无关的区域，用复合轨迹的状态初始化分解 rollout，把分解学习钉在正确的状态分布上，这个工程设计对任何「分解-组合」式 RL 都有借鉴意义。
- **值函数的直接可解释性**：正值⇔任务可达成，这让安全 RL 的「约束满足」从期望意义变成了确定性语义，对 mission-critical 应用价值很高。

## 局限与展望
- **随机动态缺乏理论保证**：作者明确承认，由于期望与极值算子不可交换，SRBE/SRABE 在随机**动态**（而非随机策略）下只是近似，目前仅靠一个人工注噪的消融实验佐证鲁棒性，尚无收敛或最优性证明。
- **只支持两个目标的复合**：当前框架针对 RAA / RR 两类双目标问题。作者展望可以迭代地把分层目标（对应更复杂的时序逻辑规约）分解成一张 Bellman 值的图，但这需要为逻辑运算的非平凡组合推导广义分解原理，且求解这张值图需要更高效的表示、收敛性保证机制和采样启发式——这些都留作未来工作。
- **确定性动态假设**：理论分解（Theorem 1–3）建立在确定性转移 $s_{t+1} = f(s_t, a_t)$ 上，向一般随机 MDP 推广时严格性会打折。
- **横向比较的 caveat**：不同环境、不同噪声档的难度不可直接横比，「领先 8%–22%」「>30%」这些数字是在各自设置内相对第二名而言，跨设置不宜直接比大小。

## 相关工作与启发
- **vs 拉格朗日 / CMDP 方法（CPPO、PPO-LAG 等）**：它们把约束折算成期望折扣和并引入乘子，需要精细奖励工程和调参，约束满足只是期望意义的；本文不需要任何乘子或类似超参，把到达和避险当作**硬约束**，且学到的值函数对约束满足有直接解释。
- **vs 时序逻辑 RL（LTL/STL，如 D-STL）**：它们通过构造自动机、对状态做乘积扩展来表达复合规约，求解体系与 HJ 的极值优化不同；本文的状态增广和分解是专为复用 HJ 子问题求解器而设计的，并证明即便在非 NMRDP 设置下仍能（理论上精确、实践中近似地）达到最优策略。
- **vs 已有 HJ-RL（RABE/Hsu et al. 2021、RCPPO/So et al. 2024、RESPO/Ganai et al. 2023）**：它们只覆盖 R、A、RA 三种原子操作，不处理复合任务；本文把 HJ-RL 推广到 RAA、RR，类比了 LTL-RL 文献里从 MDP 到 NMRDP 的进阶，并直接把这些已有求解器当作分解后的子模块来用。
- **vs 多目标 / 奖励分解 RL（MORL、HRA、reward decomposition）**：那些工作追求帕累托最优的折扣和向量奖励、或对折扣和做分解；本文的分解针对的是**极值型**（非累加）目标下的安全最优控制，这正是折扣和分解方法处理不好的场景。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 HJ-RL 推广到 RAA/RR 双目标问题，并给出精确的值分解定理与最优性保证
- 实验充分度: ⭐⭐⭐⭐ 四环境、10 基线、含随机动态消融，覆盖较全；但确定性动态外的理论验证仍偏经验
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰、动机层层递进；定理较密集，附录依赖较重
- 价值: ⭐⭐⭐⭐⭐ 为复合安全 RL 提供了「分解-组合」的范式与可解释值函数，对 mission-critical 控制有实用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RLP: Reinforcement as a Pretraining Objective](rlp_reinforcement_as_a_pretraining_objective.md)
- [\[ICLR 2026\] A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning](a_reward-free_viewpoint_on_multi-objective_reinforcement_learning.md)
- [\[ICLR 2026\] BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management](borearl_a_multi-objective_reinforcement_learning_environment_for_climate-adaptiv.md)
- [\[ICLR 2026\] Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [\[ICLR 2026\] Dual Goal Representations](dual_goal_representations.md)

</div>

<!-- RELATED:END -->
