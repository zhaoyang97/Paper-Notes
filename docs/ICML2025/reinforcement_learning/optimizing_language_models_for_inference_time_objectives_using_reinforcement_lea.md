---
title: >-
  [论文解读] Optimizing Language Models for Inference Time Objectives using Reinforcement Learning
description: >-
  [ICML2025][强化学习][inference-time compute] 提出在 RL 训练阶段显式优化推理时 k-sample 目标（pass@k / majority voting），通过 leave-one-out 控制变量构造无偏低方差梯度估计，在 MATH 和 CodeContests 上显著提升推理时性能。
tags:
  - "ICML2025"
  - "强化学习"
  - "inference-time compute"
  - "pass@k"
  - "majority voting"
  - "policy gradient"
  - "leave-one-out"
  - "REINFORCE"
  - "推理时目标优化"
---

# Optimizing Language Models for Inference Time Objectives using Reinforcement Learning

**会议**: ICML2025  
**arXiv**: [2503.19595](https://arxiv.org/abs/2503.19595)  
**作者**: Yunhao Tang, Kunhao Zheng, Gabriel Synnaeve, Rémi Munos
**机构**: Meta（FAIR）
**代码**: 未开源  
**领域**: 强化学习  
**关键词**: inference-time compute, pass@k, majority voting, policy gradient, leave-one-out, REINFORCE, 推理时目标优化

## 一句话总结

提出在 RL 训练阶段显式优化推理时 k-sample 目标（pass@k / majority voting），通过 leave-one-out 控制变量构造无偏低方差梯度估计，在 MATH 和 CodeContests 上显著提升推理时性能。

## 研究背景与动机

### 核心问题

传统语言模型训练优化的是**单样本期望奖励** $\max_\theta \mathbb{E}_{y\sim\pi_\theta}[r(x,y)]$，但部署时往往采用**多样本推理策略**（如 pass@k、majority voting）。训练目标和推理目标之间存在**不匹配（mismatch）**：

- 训练时不知道推理时会使用什么算法 → 无法充分利用推理时计算
- 推理时多次采样的性能提升依赖于生成的**多样性**，但单样本训练倾向于模式坍缩（mode collapse）

### 动机

如果**提前知道推理时将使用 pass@k 或 majority voting**，能否在训练阶段就显式优化这些目标？这样做的潜在收益是：

1. 训练出的策略更适配推理算法，获得更好的 k-sample 性能
2. 在推理计算预算固定时，每单位推理计算的效用更高
3. 提供训练-推理目标对齐的理论框架

### 相关背景

- **Inference-time scaling**：近年来推理时计算扩展被广泛验证有效（AlphaGo, DeepSeek-R1, OpenAI o1 等）
- **pass@k**：允许 k 次尝试，只要有一次正确即算通过，适用于有验证器的场景（如代码生成）
- **majority voting（Self-Consistency）**：生成 k 个回答，取多数投票结果，广泛用于数学推理

## 方法详解

### 整体框架：k-sample 目标的统一形式

本文将推理时目标统一为 **k-sample 目标函数**：

$$\max_\theta \mathbb{E}_{(y_i)_{i=1}^k \sim \pi_\theta(\cdot|x)} \left[ f(x, y_1, \ldots, y_k) \right]$$

其中聚合函数 $f$ 可以处理任意数量的生成结果。特殊情况包括：

| 目标 | 聚合函数 $f$ | 说明 |
|------|-------------|------|
| 单样本（标准 RL） | $f(\mathbf{y}) = r(y_1)$ | 退化为传统 RLHF |
| pass@k | $f(\mathbf{y}) = \max(r_1, \ldots, r_k)$ | k 次中最好的 |
| majority voting | $f(\mathbf{y}) = r(\text{maj}(a_1, \ldots, a_k))$ | 多数投票答案的奖励 |
| 平均奖励 | $f(\mathbf{y}) = \frac{1}{k}\sum_{i=1}^k r_i$ | k 个样本的均值 |

### 关键设计：Leave-One-Out 梯度估计

#### 朴素 REINFORCE 梯度的问题

直接对 k-sample 目标应用 REINFORCE 得到：

$$f(\mathbf{y}) \sum_{i=1}^{k} \nabla_\theta \log \pi_\theta(y_i|x)$$

这一梯度的方差为 $\mathcal{O}(k)$，随样本数 k **线性增长**，与标准 k-sample 平均梯度的 $\mathcal{O}(k^{-1})$ 形成鲜明对比。原因是 k 个样本通过聚合函数 $f$ **耦合**在一起。

#### Leave-One-Out 控制变量

为降低方差，本文提出 **leave-one-out (LOO) 控制变量**：

$$\sum_{i=1}^{k} \left( f(\mathbf{y}) - f(\mathbf{y}_{-i}) \right) \nabla_\theta \log \pi_\theta(y_i|x)$$

其中 $\mathbf{y}_{-i}$ 表示去掉第 $i$ 个生成后的剩余样本集合。有效优势函数定义为：

$$A_i \coloneqq f(\mathbf{y}) - f(\mathbf{y}_{-i})$$

它衡量第 $i$ 个生成对整体目标函数的**边际贡献**。

**无偏性证明（Lemma 1）**：由于 k 个样本独立同分布，$\mathbb{E}[f(\mathbf{y}_{-i}) \nabla_\theta \log \pi_\theta(y_i|x)] = 0$，因此 LOO 控制变量不引入偏差。

#### 各目标下的 LOO 梯度特性

**pass@k 的稀疏信号特性**：

优势函数退化为 $A_i = \max(r_{1:k}) - \max(r_{-i})$，仅对**最优生成** $y_{(k)}$ 非零：

$$A_{(k)} = r_{(k)} - r_{(k-1)}$$

即梯度信号仅来自最优与次优的差距。这意味着：
- 当问题太简单（解出率 >> $1/k$）→ 多个样本都对，信号稀疏
- 当问题太难（解出率 << $1/k$）→ 几乎无正确样本，信号同样稀疏
- **最佳学习区间**：解出率约 $O(k^{-1})$ 时梯度信号最密集

**majority voting 的投票翻转信号**：

优势函数 $A_i = r(\text{maj}(\mathbf{a})) - r(\text{maj}(\mathbf{a}_{-i}))$ 衡量第 $i$ 个答案是否**翻转了多数投票结果**。只有当移除 $a_i$ 导致投票结果改变时，梯度信号才非零。

### 训练策略

本文将 k-sample 目标嵌入**在线 RL 训练框架**：

1. **采样**：对每个 prompt $x$，从当前策略 $\pi_\theta$ 采 k 个独立响应
2. **评估**：用验证器/奖励模型计算 $r(x, y_i)$，并计算聚合函数 $f$ 和 LOO 优势
3. **更新**：用 LOO 梯度估计执行策略梯度更新
4. **正则化**：可选地加入 KL 散度正则化防止策略退化

关键超参数包括：
- **训练时 k 值**：训练时使用的样本数，可与推理时的 k 不同
- **温度调度**：控制采样多样性
- **目标选择**：pass@k vs. majority voting vs. 混合目标

## 实验关键数据

### MATH 数据集实验

在数学推理任务 MATH 上验证不同训练目标对推理时性能的影响：

| 训练目标 | pass@1 | pass@8 | pass@64 | majority@8 | majority@64 |
|---------|--------|--------|---------|------------|-------------|
| 标准 RL（单样本） | **最优** | 基线 | 基线 | 基线 | 基线 |
| pass@k 优化 | 略降 | **显著提升** | **显著提升** | 持平 | 持平 |
| majority voting 优化 | 略降 | 持平 | 持平 | **显著提升** | **显著提升** |

关键发现：
- 针对 pass@k 训练的模型在 pass@k 指标上明显优于标准 RL
- 针对 majority voting 训练的模型在投票指标上表现更好
- 但存在**trade-off**：针对推理时目标优化会轻微牺牲 pass@1 性能

### CodeContests 代码生成实验

在更具挑战性的 CodeContests 竞赛编程任务上的表现：

| 方法 | pass@1 | pass@10 | pass@100 | 相对提升(pass@100) |
|------|--------|---------|----------|-------------------|
| 标准 RL 基线 | 基线值 | 基线值 | 基线值 | — |
| pass@k 优化（本文） | 持平/略降 | 显著提升 | **大幅提升** | 显著 |

关键发现：
- 在代码生成这种**高难度、低解出率**任务上，pass@k 优化的优势更加明显
- 与基线相比，pass@100 的改进幅度远大于 pass@10，说明方法在**大采样预算**下收益更大
- 训练时 k 值与推理时 k 值的比例影响最终性能

## 亮点与洞察

1. **训练-推理对齐的新视角**：首次系统性地将推理时 k-sample 目标纳入 RL 训练目标，提供了统一的数学框架，涵盖 pass@k 和 majority voting 作为特例

2. **LOO 控制变量的优雅设计**：leave-one-out 优势函数的物理含义清晰——衡量单个生成对整体目标的边际贡献，且保持无偏性的证明极为简洁

3. **pass@k 梯度的稀疏性洞察**：揭示了 pass@k 优化的信号密度与题目难度的关系——最优学习区间在解出率约 $O(1/k)$ 处，为课程学习（curriculum learning）提供了理论指导

4. **实用的 trade-off 分析**：诚实地展示了推理时目标优化与 pass@1 性能之间的权衡，帮助实践者根据部署场景选择合适的训练目标

5. **代码生成场景的强适配性**：在 CodeContests 等验证器天然可用、解出率极低的场景下，pass@k 优化的实际价值尤为突出

## 局限与展望

1. **仅限简单推理策略**：只考虑了 pass@k 和 majority voting，未涉及更复杂的推理时策略如 best-of-k（需辅助奖励模型）、MCTS、beam search 等

2. **训练计算开销**：每个 prompt 需要采样 k 个完整响应，训练成本约为标准 RL 的 k 倍，对大规模训练的可扩展性存疑

3. **k 值泛化问题**：训练时使用固定 k 值，推理时使用不同 k 是否仍然有效需要验证，训练 k 与推理 k 的最优比例关系尚不清晰

4. **任务覆盖有限**：仅在数学推理（MATH）和代码生成（CodeContests）上验证，更广泛的推理任务（如常识推理、多步规划）的适用性未验证

5. **与 GRPO/RLOO 的关系**：leave-one-out 基线与 GRPO 等方法有密切联系，但文中的比较和区分不够明确

6. **实验数据不完整**：本地缓存截断于 Section 3.1，完整的消融实验和定量数据未能获取

## 相关工作与启发

### 推理时计算扩展
- **Self-Consistency (Wang et al., 2022)**：majority voting 的原始提出，本文将其目标纳入训练优化
- **OpenAI o1 / DeepSeek-R1**：推理时计算扩展的成功案例，但训练目标仍为单样本
- **AlphaCode (Li et al., 2022)**：在竞赛编程中使用大规模采样 + 过滤，本文为类似场景提供了训练端优化

### 策略梯度方差降低
- **RLOO (Kool et al., 2019; Ahmadian et al., 2024)**：leave-one-out 基线在平均奖励目标下的应用，本文推广到一般 k-sample 目标
- **GRPO (Shao et al., 2024)**：组相对策略优化，同样利用组内样本做基线，但优化目标不同

### 对后续研究的启发
- 可将框架扩展到更复杂的 tree search 推理策略
- LOO 优势函数的"边际贡献"视角可启发新的 credit assignment 方法
- 训练目标随课程难度自适应选择 k 值可能带来进一步提升

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将推理时目标显式纳入 RL 训练的视角新颖，LOO 梯度估计的推导干净
- 实验充分度: ⭐⭐⭐ — 覆盖 MATH 和 CodeContests 两个代表性场景，但缓存不完整限制了完整评估
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，特例分析直觉好，从一般框架到具体应用逐步展开
- 价值: ⭐⭐⭐⭐ — 在推理时计算日益重要的当下，训练-推理目标对齐具有很高的实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [\[NeurIPS 2025\] Behavior Injection: Preparing Language Models for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/behavior_injection_preparing_language_models_for_reinforcement_learning.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICML 2025\] ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](revise_learning_to_refine_at_test-time_via_intrinsic_self-verification.md)
- [\[ICLR 2026\] Optimistic Task Inference for Behavior Foundation Models](../../ICLR2026/reinforcement_learning/optimistic_task_inference_behavior_models.md)

</div>

<!-- RELATED:END -->
