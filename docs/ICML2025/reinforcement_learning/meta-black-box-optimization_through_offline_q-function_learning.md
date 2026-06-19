---
title: >-
  [论文解读] Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)
description: >-
  [ICML2025][强化学习][Meta-BBO] 提出 Q-Mamba，首个离线 MetaBBO 框架，通过 Q 函数分解 + 保守 Q 学习 + Mamba 架构，在不到在线方法一半训练预算下达到可比甚至更优的 BBO 算法配置性能。 Meta-Black-Box-Optimization (MetaBBO)：是一种双…
tags:
  - "ICML2025"
  - "强化学习"
  - "Meta-BBO"
  - "离线强化学习"
  - "Q函数分解"
  - "Mamba"
  - "动态算法配置"
  - "保守Q学习"
---

# Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)

**会议**: ICML2025  
**arXiv**: [2505.02010](https://arxiv.org/abs/2505.02010)  
**代码**: 已开源  
**领域**: 元学习 / 黑盒优化 (Meta-Learning / Black-Box Optimization)  
**关键词**: Meta-BBO, 离线强化学习, Q函数分解, Mamba, 动态算法配置, 保守Q学习

## 一句话总结

提出 Q-Mamba，首个离线 MetaBBO 框架，通过 Q 函数分解 + 保守 Q 学习 + Mamba 架构，在不到在线方法一半训练预算下达到可比甚至更优的 BBO 算法配置性能。

## 研究背景与动机

**Meta-Black-Box-Optimization (MetaBBO)** 是一种双层"学习优化"范式：用元级神经网络策略 $\pi_\theta$ 动态配置底层 BBO 算法（如差分进化 DE）的超参数，使其在问题分布上取得最优性能。

现有 MetaBBO 方法面临两个核心瓶颈：

**学习有效性**：先进 BBO 算法拥有庞大的配置空间（例如 MadDE 有 10+ 超参数），联合动作空间随维度指数增长（$M^K$），RL 难以学到有效策略

**训练效率**：现有方法均采用在线 RL 范式，底层优化每次需数百步仿真，采样轨迹极其耗时（训练需 25-28 小时）

核心问题：能否用离线 RL 从预收集的轨迹数据中学习 DAC 策略，同时保证学习效果？

## 方法详解

### 整体框架

Q-Mamba 包含三个核心设计：

1. **离线 E&E 数据集收集**：混合利用与探索数据
2. **保守 Q 学习损失**：分解式贝尔曼更新 + 保守正则化
3. **Mamba-based Q-Learner**：基于状态空间模型的长序列 Q 函数学习

### 1. Q 函数分解机制

将联合动作空间 $A = (A_1, \ldots, A_K)$ 分解为逐维顺序决策，对第 $i$ 维的分解 Q 函数：

$$
Q(a_i^t | s^t) \leftarrow \begin{cases}
\max_{a_{i+1}^t} Q(a_{i+1}^t | s^t, a_{1:i}^t), & \text{if } i < K \\
R(s^t, a_{1:K}^t) + \gamma \max_{a_1^{t+1}} Q(a_1^{t+1} | s^{t+1}), & \text{if } i = K
\end{cases}
$$

- 每个超参数 $A_i$ 离散化为 $M=16$ 个 bin
- 用 5-bit 二进制编码表示动作 token（`00000` ~ `01111`），`11111` 为起始 token
- 将指数级空间 $M^K$ 化为线性级 $K \times M$

### 2. 离线 E&E 数据集（Exploration & Exploitation）

收集 $D = 10$K 条轨迹，混合比例 $\mu = 0.5$：

- **Exploitation 数据**（$\mu \cdot D$）：用预训练好的 MetaBBO 方法（RLPSO、LDE、GLEET）在问题分布上 rollout 得到的高质量轨迹
- **Exploration 数据**（$(1-\mu) \cdot D$）：随机策略控制超参数得到的探索轨迹

混合两类数据可同时提供高质量经验和多样性，减少离线学习偏差。

### 3. 保守 Q 学习损失

训练目标为三分支组合损失（对轨迹 $\tau$ 中每个时间步 $t$、每个动作维 $i$、每个 bin $j$）：

$$
J(Q_{i,j}^t | \theta) = \begin{cases}
\frac{1}{2}(Q_{i,j}^t - \max_j Q_{i+1,j}^t)^2, & i < K,\ j = a_i^t \\
\frac{\beta}{2}[Q_{i,j}^t - (r^t + \gamma \max_j Q_{1,j}^{t+1})]^2, & i = K,\ j = a_i^t \\
\frac{\lambda}{2}(Q_{i,j}^t - 0)^2, & j \neq a_i^t
\end{cases}
$$

- **前两支**：标准分解贝尔曼目标（TD 误差），最后一维加权 $\beta=10$ 以锚定反向传播精度
- **第三支**：CQL 保守正则化，将未在轨迹中选择的 bin 的 Q 值压向 0（下界），缓解分布偏移导致的过估计
- $\lambda=1$ 平衡保守项

### 4. Mamba-based Q-Learner 架构

以自回归方式逐维输出 $Q_i^t$：

$$
\mathbb{O}_i^t, h_i^t = \text{mamba\_block}([s^t, \text{token}(a_{i-1}^t)], h_{i-1}^t | W_{mamba})
$$

$$
Q_i^t = \sigma(\text{Linear}(\mathbb{O}_i^t | W_{head}, b_{head}))
$$

- 输入：当前优化状态 $s^t \in \mathbb{R}^9$（6 维种群统计特征 + 3 维时序特征）拼接前一步动作 token
- Mamba 块：通过选择性状态空间模型（SSM）处理，参数 $\bar{B}$、$C$ 为输入的函数（时变），支持灵活学习长短期依赖
- Q-value head：线性层 + Leaky ReLU，输出 16 维 Q 值向量
- 选择 $a_i^t = \arg\max_j Q_{i,j}^t$ 后 token 化送入下一步

**选择 Mamba 而非 Transformer 的原因**：

- MetaBBO 涉及 $T \times K$ 步决策（数千步长序列），Mamba 通过选择性 SSM 灵活处理长短期依赖
- 硬件友好的并行扫描（PrefixSum）算法，与 FlashAttention 同等内存效率
- 训练速度优于 Transformer（13h vs 16h）

训练使用 AdamW，学习率 $5 \times 10^{-3}$，300 epochs，batch size 64。

## 实验关键数据

### 分布内泛化性能（8 个未见测试问题，19 次独立运行，累积性能指标 Perf）

| 方法 | 类型 | Alg0 (K=3) | Alg1 (K=10) | Alg2 (K=16) | 训练/推理时间 |
|------|------|-----------|------------|------------|-------------|
| RLPSO | 在线 | 0.9855 | 0.9953 | 0.9914 | 28h / 11s |
| LDE | 在线 | 0.9563 | 0.9877 | 0.9904 | 28h / 12s |
| GLEET | 在线 | 0.9616 | 0.9938 | 0.9910 | 25h / 13s |
| DT | 离线 | 0.9325 | 0.6764 | 0.8706 | 13h / 10s |
| DeMa | 离线 | 0.9492 | 0.9015 | 0.9159 | 12h / 10s |
| QDT | 离线 | 0.9683 | 0.9917 | 0.9919 | 20h / 12s |
| QT | 离线 | 0.9729 | 0.9955 | 0.9926 | 20h / 12s |
| Q-Transformer | 离线 | 0.9666 | 0.9951 | 0.9895 | 16h / 11s |
| **Q-Mamba** | **离线** | **0.9889** | **0.9973** | **0.9950** | **13h / 10s** |

**关键发现**：Q-Mamba 在三种不同配置空间规模的算法上均取得最优性能，训练时间不到在线方法的一半。

### 分布外泛化（Neuroevolution / MuJoCo 连续控制）

- 仅在合成 BBOB 问题（≤50 维）上训练的 Q-Mamba，零样本迁移到数千参数的 MLP 策略进化任务
- 在 Ant、HalfCheetah、Hopper 等 MuJoCo 任务上达到与在线 MetaBBO 基线可比的性能

### 消融实验

| 设置 | $\lambda=0$ | $\lambda=1$ | $\lambda=10$ |
|------|-----------|-----------|------------|
| $\beta=1$ | 0.9756 | 0.9828 | 0.9855 |
| $\beta=10$ | 0.9833 | **0.9889** | 0.9857 |

- 去除保守正则化（$\lambda=0$）性能明显下降 → 保守项对缓解分布偏移至关重要
- $\beta=10$ 加权最后维度 Q 值 → 通过锚定反向传播提升整体精度
- 数据混合比 $\mu=0.5$（等比例混合）效果最佳

## 亮点与洞察

1. **首个离线 MetaBBO 框架**：打破该领域完全依赖在线 RL 的局面，训练效率提升 ≥2×
2. **Q 函数分解**：巧妙地将指数级联合动作空间化为线性序列决策，降低学习难度
3. **保守 Q 学习 + 分解 Bellman**：组合损失设计精致，$\beta$ 加权最后维度的 trick 有理论支撑（反向传播锚点）
4. **Mamba vs Transformer**：实证 Mamba 在长序列 Q 学习中优于 Transformer（时变 SSM 的选择性记忆优势）
5. **泛化性强**：从 ≤50 维合成问题零样本迁移到数千维 Neuroevolution 场景

## 局限与展望

1. **离线数据依赖预训练基线**：E&E 数据集需要先训练 RLPSO/LDE/GLEET 等在线方法，初始成本未被充分讨论
2. **仅验证进化算法**：未在梯度类优化器或贝叶斯优化等其他 BBO 范式上测试
3. **动作离散化信息损失**：统一离散化为 16 bin 可能丢失连续超参数的精细粒度
4. **OOD 泛化实验有限**：仅测试了 Neuroevolution 一类 OOD 场景，更多现实场景有待验证
5. **Mamba 架构对 CUDA 的依赖**：硬件加速限于特定 GPU，部署灵活性受限

## 相关工作与启发

- **ConfigX**（Guo et al., 2024b）：构建大规模算法空间，本文在此基础上选择底层算法
- **Q-Transformer**（Chebotar et al., 2023）：Q 分解 + Transformer，本文替换为 Mamba 取得提升
- **CQL**（Kumar et al., 2020）：保守 Q 学习思想直接被本文吸收
- **Mamba**（Gu & Dao, 2023）：选择性状态空间模型，本文首次将其应用于 MetaBBO
- **Decision Transformer**（Chen et al., 2021）：条件模仿学习范式的对照基线

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个离线 MetaBBO + Q 分解 + Mamba 的组合具有显著创新性
- 实验充分度: ⭐⭐⭐⭐ — 9 个基线对比 + 消融 + OOD 泛化，较为全面
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，框架图直观，行文流畅
- 价值: ⭐⭐⭐⭐ — 为 MetaBBO 领域引入离线 RL 新范式，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization](../../NeurIPS2025/reinforcement_learning/metabox-v2_a_unified_benchmark_platform_for_meta-black-box_optimization.md)
- [\[NeurIPS 2025\] Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/optimizing_the_unknown_black_box_bayesian_optimization_with_energy-based_model_a.md)
- [\[ICML 2025\] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[ICML 2025\] Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
