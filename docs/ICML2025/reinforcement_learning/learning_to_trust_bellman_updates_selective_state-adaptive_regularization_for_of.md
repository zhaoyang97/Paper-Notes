---
title: >-
  [论文解读] Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL
description: >-
  [ICML2025][强化学习][离线RL] 提出选择性状态自适应正则化（SSAR），用神经网络为每个状态动态生成正则化系数，并仅在高质量动作上施加约束，统一了CQL（值正则化）和TD3+BC（策略约束）两大离线RL范式，在D4RL离线和O2O场景均大幅超越基线。 离线RL的核心挑战在于如何平衡 Bellman更新带来的性能…
tags:
  - "ICML2025"
  - "强化学习"
  - "离线RL"
  - "状态自适应正则化"
  - "选择性约束"
  - "值函数正则化"
  - "策略约束"
  - "Offline-to-Online RL"
---

# Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL

**会议**: ICML2025  
**arXiv**: [2505.19923](https://arxiv.org/abs/2505.19923)  
**代码**: [QinwenLuo/SSAR](https://github.com/QinwenLuo/SSAR)  
**领域**: 离线强化学习 (Offline RL)  
**关键词**: 离线RL, 状态自适应正则化, 选择性约束, 值函数正则化, 策略约束, Offline-to-Online RL

## 一句话总结
提出选择性状态自适应正则化（SSAR），用神经网络为每个状态动态生成正则化系数，并仅在高质量动作上施加约束，统一了CQL（值正则化）和TD3+BC（策略约束）两大离线RL范式，在D4RL离线和O2O场景均大幅超越基线。

## 研究背景与动机

离线RL的核心挑战在于如何平衡 **Bellman更新带来的性能提升** 与 **正则化约束带来的安全性**：

**全局固定系数的局限**：现有方法（CQL、TD3+BC等）使用固定的全局正则化系数 $\beta$，但最优系数因任务、训练阶段、数据密度而异——弱正则化导致值过估计，强正则化退化为行为克隆

**训练动态变化**：训练初期策略不可靠需强约束，后期策略接近数据集后应放松约束以释放Bellman更新的潜力

**数据质量差异**：高数据密度的状态可以信任Bellman更新结果，低密度状态需收紧约束

**Offline-to-Online效率**：固定系数导致离线与在线Q值差距大，影响微调效率

## 方法详解

### 核心思想：统一框架

**命题3.1（CQL与策略约束的等价性）**：当策略 $\pi(a|s) \propto \exp(Q(s,a))$ 建模为Boltzmann分布时，CQL的值正则化项等价于数据集动作的负对数似然：

$$\min_Q \beta \, \mathbb{E}_{s \sim D}\left[\log \sum_a \exp Q(s,a) - \mathbb{E}_{a \sim D}[Q(s,a)]\right] \iff \min_\pi \beta \, \mathbb{E}_{(s,a) \sim D}[-\log \pi(a|s)]$$

这一等价性将值正则化和显式策略约束统一到同一框架下，使得状态自适应系数可以同时应用于两类方法。

### 组件一：状态自适应系数

用神经网络 $\beta_\phi(s)$ 生成每个状态的正则化系数，通过以下目标自适应更新：

$$L_\beta(\phi) = \mathbb{E}_{(s,a) \sim D}\left[\log \pi(a|s) - C_n(s)\right] \beta_\phi(s)$$

其中阈值 $C_n(s) = \min\{\log \pi(\mu + n\sigma | s), \log \pi(\mu - n\sigma | s)\}$，$\mu, \sigma$ 为策略的均值和标准差，$n$ 控制信任区域宽度。

**机制**：当数据集动作的对数概率超过阈值时系数降低（信任Bellman更新），低于阈值时系数升高（加强约束）。

### 组件二：分布感知阈值

参数 $n$ 从 $n_{start}$ 线性增加到 $n_{end}$：

$$n \leftarrow n + \Delta n, \quad \Delta n = (n_{end} - n_{start}) \cdot T_{inc} / T$$

当 $\mathbb{E}_{(s,a) \sim D}[\log \pi(a|s) - C_n(s)] > 0$ 时停止更新，确保信任区域随训练进展自适应扩展。

### 组件三：选择性正则化

仅在高质量动作子集 $\hat{D}$ 上施加约束，避免低质量动作误导策略：

- **低方差数据集**：按轨迹回报筛选，$G > G_T$ 的轨迹构成 $\hat{D}$
- **高方差数据集**：用IQL预训练 $Q, V$ 网络，按优势 $Q(s,a) - V(s) > 0$ 筛选

**CQL + SSAR 的Q函数更新**：

$$\min_Q \beta_\phi(s) \, \mathbb{E}_{s \sim \hat{D}}\left[\log \sum_a \exp Q(s,a) - \mathbb{E}_{a \sim D}[Q(s,a)]\right] + \frac{1}{2} \mathbb{E}_{s,a,s' \sim D}\left[(Q - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k)^2\right]$$

**TD3+BC + SSAR 的策略更新**：

$$\max_\pi \mathbb{E}_{s,a \sim D}\left[Q_{norm}(s, \pi(s)) - \mathbb{I}((s,a) \in \hat{D}) \, \beta_\phi(s) (\pi(s) - a)^2\right]$$

### 组件四：Offline-to-Online 微调

固定系数网络参数，线性衰减输出：

$$\beta_{on}(s) = \min\left\{1 - \frac{N}{N_{end}}, 0\right\} \cdot \beta(s)$$

利用离线训练好的系数网络的泛化能力，甚至可丢弃离线数据集仅用在线数据微调。

## 实验关键数据

### 离线性能（D4RL基准）

| 数据集 | TD3+BC Base | TD3+BC+SSAR | CQL Base | CQL+SSAR |
|--------|:-----------:|:-----------:|:--------:|:--------:|
| halfcheetah-m | 48.3 | **56.5** | 47.1 | **63.9** |
| hopper-m | 58.7 | **101.6** | 65.6 | **89.1** |
| walker2d-m | 82.3 | **87.9** | 81.6 | **84.9** |
| halfcheetah-mr | 44.4 | **49.6** | 45.7 | **53.8** |
| hopper-mr | 66.4 | **101.6** | 92.3 | **101.4** |
| walker2d-mr | 81.6 | **93.5** | 79.2 | **94.7** |
| halfcheetah-me | 92.9 | **94.9** | 93.0 | **102.1** |
| hopper-me | 101.4 | **103.8** | 97.8 | **109.6** |
| **Locomotion 总分** | 1000.8 | **1116.7** | 1030.4 | **1139.1** |
| **AntMaze 总分** | 131.8 | **276.0** | 294.1 | **406.8** |

- Locomotion任务：SSAR使TD3+BC提升 **+11.6%**，CQL提升 **+10.5%**
- AntMaze任务（稀疏奖励）：TD3+BC提升 **+109%**，CQL提升 **+38%**

### Offline-to-Online 微调（250k步）

| 数据集 | IQL | SPOT | CQL | TD3+BC(SA) | CQL(SA) |
|--------|:---:|:----:|:---:|:----------:|:-------:|
| halfcheetah-m | 49.7 | 58.6 | 48.0 | **82.9** | **95.3** |
| hopper-m | 75.2 | 99.9 | 63.8 | **103.5** | **99.3** |
| walker2d-m | 80.8 | 82.5 | 82.8 | **101.6** | **105.9** |
| **Locomotion总分** | 1057.4 | 1107.1 | 1068.4 | **1218.6** | **1278.4** |

O2O场景下CQL(SA)总分达1278.4，超越所有基线约10%。

## 亮点与洞察

1. **理论优雅**：通过Proposition 3.1建立CQL与策略约束的等价性，为统一框架提供理论基础
2. **状态级自适应**：用神经网络替代固定系数，同时解决跨任务、跨训练阶段、跨数据密度三个维度的系数选择问题
3. **选择性约束**：仅约束高质量动作的思路简洁有效，尤其在稀疏奖励的AntMaze任务上带来巨大提升（TD3+BC从131.8→276.0）
4. **O2O无缝过渡**：冻结系数网络+线性衰减的简单策略即可高效微调，甚至可丢弃离线数据保护隐私
5. **即插即用**：作为通用模块可同时增强值正则化（CQL）和策略约束（TD3+BC）两类方法

## 局限与展望

1. **额外计算开销**：需要训练系数网络 $\beta_\phi$，高方差数据集还需预训练IQL的Q/V网络
2. **超参数未完全消除**：$n_{start}$、$n_{end}$、$T_{inc}$、选择性阈值 $G_T$ 等仍需调整，虽比单一固定 $\beta$ 更灵活但并非零超参
3. **仅验证连续控制**：实验局限于D4RL的MuJoCo和AntMaze，未验证离散动作空间或真实世界任务
4. **选择性策略依赖数据质量先验**：高/低方差数据集需要不同的筛选策略（回报阈值 vs. 优势函数），实际应用中可能难以预判
5. **O2O阶段线性衰减偏简单**：更复杂的衰减策略或自适应停止可能进一步提升效率

## 相关工作与启发

- **CQL** (Kumar et al., 2020)：值正则化代表，SSAR的直接改进对象
- **TD3+BC** (Fujimoto & Gu, 2021)：极简策略约束方法，SSAR同样显著提升其性能
- **IQL** (Kostrikov et al., 2021)：提供优势函数预训练，用于SSAR的数据筛选
- **Cal-QL** (Nakamoto et al., 2024)：另一种自适应CQL变体，SSAR在O2O场景可与之互补

## 评分
- 新颖性: ⭐⭐⭐⭐ — 统一值正则化与策略约束的视角有理论贡献，选择性+状态自适应的组合设计合理
- 实验充分度: ⭐⭐⭐⭐ — D4RL全系列任务覆盖完整，含离线和O2O两种设置，消融充分
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，理论推导严谨，但公式密度较高
- 价值: ⭐⭐⭐⭐ — 即插即用的通用模块，对离线RL社区有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)
- [\[ICML 2025\] Embedding Safety into RL: A New Take on Trust Region Methods](embedding_safety_into_rl_a_new_take_on_trust_region_methods.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[ICLR 2026\] Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning](../../ICLR2026/reinforcement_learning/beyond_penalization_diffusion-based_out-of-distribution_detection_and_selective_.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)

</div>

<!-- RELATED:END -->
