---
title: >-
  [论文解读] SafeMIL: Learning Offline Safe Imitation Policy from Non-Preferred Trajectories
description: >-
  [AAAI 2026][强化学习][离线安全模仿学习] 本文提出SafeMIL，通过将代价函数学习建模为多实例学习（MIL）问题，从有限的非偏好轨迹和大量无标签轨迹中学习安全的模仿策略，在不需要逐步reward/cost标注的情况下，实现约束满足性能比最佳基线提升3.7倍。 问题背景 强化学习在现实世界部署面临两大挑战： 在…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "离线安全模仿学习"
  - "多实例学习"
  - "约束MDP"
  - "行为克隆"
  - "代价函数学习"
---

# SafeMIL: Learning Offline Safe Imitation Policy from Non-Preferred Trajectories

**会议**: AAAI 2026  
**arXiv**: [2511.08136](https://arxiv.org/abs/2511.08136)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 离线安全模仿学习, 多实例学习, 约束MDP, 行为克隆, 代价函数学习

## 一句话总结

本文提出SafeMIL，通过将代价函数学习建模为多实例学习（MIL）问题，从有限的非偏好轨迹和大量无标签轨迹中学习安全的模仿策略，在不需要逐步reward/cost标注的情况下，实现约束满足性能比最佳基线提升3.7倍。

## 研究背景与动机

### 问题背景

强化学习在现实世界部署面临两大挑战：

**在线交互风险高**：机器人、自动驾驶等场景中，在线试错成本极高

**奖励函数设计困难**：为复杂任务设计合适的reward函数可能导致意外行为

模仿学习（IL）通过学习专家演示避免了reward设计问题，但传统IL隐式假设所有演示都是安全的。当数据中混有不安全轨迹时，直接模仿可能学到危险行为。

### 核心动机

现实中存在一个实用场景：
- 我们有**少量非偏好轨迹**（如交通事故记录、有毒聊天内容举报）
- 我们有**大量无标签轨迹**（混合了安全和不安全行为）
- 每步的reward和cost信息**不可获得**

**标注一条轨迹为"非偏好"远比标注每步的cost容易**。例如：
- 自动驾驶：知道某段驾驶"闯了红灯"比标注每帧的精确安全成本容易
- 聊天机器人：用户举报有毒内容只需要轨迹级标签

### 与现有工作的差异

| 方法类别 | 是否需要online交互 | 是否需要逐步reward/cost | 数据来源 |
|---------|------------------|---------------------|---------|
| 标准RL | ✓ | ✓ | 在线交互 |
| 离线安全RL | ✗ | ✓ (需cost标注) | 离线数据集 |
| 标准IL | ✗ | ✗ | 专家演示 |
| T-REX/PEBBLE | ✓ (需在线RL) | ✗ (从排名学reward) | 排名轨迹 |
| SafeDICE | ✗ | ✗ | 非偏好+无标签 |
| **SafeMIL（本文）** | ✗ | ✗ | 非偏好+无标签 |

SafeMIL是**首个将MIL引入离线安全IL设定**的工作。

## 方法详解

### 整体框架

SafeMIL分为两个阶段：
1. **代价函数学习**：通过MIL框架从轨迹级标签学习状态-动作级的代价函数 $\hat{c}_\theta(s, a)$
2. **安全策略学习**：利用学到的代价函数筛选/加权无标签数据中的偏好轨迹，通过行为克隆（BC）学习安全策略

### 关键设计

#### 1. **将代价函数学习建模为MIL问题**

**多实例学习（MIL）回顾**：
- 数据以"袋"（bag）为单位：$\mathcal{B} = \{x_1, x_2, \ldots, x_K\}$
- 只有袋级标签 $Y$，没有实例级标签
- 袋为正（$Y=1$）当且仅当至少包含一个正实例
- 袋为负（$Y=0$）当且仅当所有实例为负

**轨迹→MIL的映射**：
- **负袋**：从非偏好轨迹数据集 $\mathcal{D}^N$ 有放回采样 $K$ 条轨迹（确保所有轨迹都是非偏好的）
- **无标签袋**：从无标签数据集 $\mathcal{D}^U$ 有放回采样 $K$ 条轨迹

**关键引理（Lemma 1）**：无标签袋包含至少一条偏好轨迹的概率为：
$$P(\mathcal{B} \cap \mathcal{T}_p \neq \emptyset) = 1 - (1-\alpha)^K$$
其中 $\alpha$ 是无标签数据中偏好轨迹的比例。当 $K$ 足够大时，此概率趋近1，因此无标签袋可视为**正袋**。

#### 2. **基于对称函数的袋评分函数**

基于对称函数基本定理，设计置换不变的袋评分函数：

$$Score(\mathcal{B}) = g\left(\sum_{\tau \in \mathcal{B}} f(\tau)\right)$$

作者选择直观的函数形式：
- $f(\tau) = \frac{1}{K} \sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t)$（轨迹的平均折扣代价）
- $g$ = 恒等函数

最终评分：
$$Score(\mathcal{B}) = \frac{1}{K} \sum_{\tau \in \mathcal{B}} \sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t)$$

**直觉解释**：当 $K \to \infty$，评分收敛为该袋轨迹的期望累积代价。非偏好袋的期望代价应高于无标签袋（因后者包含偏好轨迹）。

**Theorem 1**：$P(Score(\mathcal{B}_n) > Score(\mathcal{B}_u)) = 1 - (1-\alpha)^K$，即负袋评分高于无标签袋评分的概率与包含偏好轨迹的概率相同。

#### 3. **Bradley-Terry损失训练代价函数**

利用负袋评分应高于无标签袋评分的关系，使用Bradley-Terry模型训练代价函数：

$$\mathcal{L}_\theta = -\mathbb{E}_{\mathcal{B}_n \sim \rho^N, \mathcal{B}_u \sim \rho^U} \left[ \log \frac{\exp(Score(\mathcal{B}_n))}{\exp(Score(\mathcal{B}_n)) + \exp(Score(\mathcal{B}_u))} \right]$$

这个损失函数驱动 $\hat{c}_\theta$ 为非偏好行为分配更高代价值。

#### 4. **基于代价函数的策略学习**

学到 $\hat{c}_\theta$ 后，有两种策略学习方式：

**a) 硬阈值筛选**：选取累积代价低于阈值 $\hat{b}$ 的轨迹进行BC：
$$\mathcal{T}_{\hat{c}_\theta} := \{\tau \in \mathcal{D}^U \mid \sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t) \leq \hat{b}\}$$

**b) 软加权BC（默认使用）**：对每条轨迹赋予权重：
$$w(\tau) = \exp\left(-\sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t) / \beta\right)$$

加权BC损失：
$$\min_\pi \sum_{\tau \in \mathcal{D}^U} \left[ w(\tau) \sum_{t=0}^{T-1} \mathcal{L}_\pi(s_t, a_t) \right]$$

$\beta$ 越小，对高代价轨迹的惩罚越强。

#### 5. **部分轨迹扩展**

完整轨迹学习计算昂贵，因此SafeMIL支持使用**部分轨迹**（长度 $H$）构建袋：
- 从非偏好数据集采样的部分轨迹可能展示偏好行为→负袋中部分实例可能被错误标注
- 但当袋大小 $K$ 足够大时，负袋的平均代价仍高于无标签袋，Score关系仍成立

### 损失函数 / 训练策略

- 交替训练代价函数 $\hat{c}_\theta$ 和策略网络 $\pi$
- 每轮采样一对负袋和无标签袋进行代价函数更新
- 同时使用加权BC更新策略
- 训练步数：100万步
- 非偏好轨迹数量：50条
- 无标签轨迹数量：200条

## 实验关键数据

### 实验设置

**环境**：
- MuJoCo速度约束任务：Walker-Velocity, Swimmer-Velocity, Ant-Velocity
- 导航任务：Point-Circle2, Point-Goal1, Point-Button1

**数据**：使用DSRL（Datasets for offline Safe RL）基准，移除所有reward和cost信息。

**评估指标**：
- Normalized Return（0=随机策略, 1=约束RL策略）
- Normalized Cost（0=约束RL策略的cost水平）
- Normalized CVaR@20% Cost（最差20%运行的平均cost）

### 主实验

**速度约束任务（Fig. 1中主要结果）：**

| 方法 | Walker-Vel Cost | Swimmer-Vel Cost | Ant-Vel Cost | 安全性表现 |
|------|----------------|-----------------|-------------|----------|
| BC-Unlabeled | 高 (>0) | 高 (>0) | 高 (>0) | 学到非偏好行为 |
| SafeDICE | 中等 | 中等 | 中等 | 部分约束满足 |
| DWBC-NU | 中等 | 中等 | 中等 | 不稳定 |
| T-REX-WBC | 中等 | 中等 | 中等 | 部分改善 |
| **SafeMIL** | **≈0** | **≈0** | **≈0** | **最佳安全性** |

**导航任务**：SafeMIL在Point-Goal1上最优，在Point-Circle2和Point-Button1上与基线competitive。

**跨所有环境**：SafeMIL的中位安全性能是最佳基线的**3.7倍**。

### 消融实验

**袋大小 $K$ 的敏感性（Swimmer-Velocity）：**

| 袋大小 $K$ | Normalized Cost | Normalized Return | 说明 |
|-----------|----------------|-------------------|------|
| 1 | 较高 | 正常 | 无MIL效果 |
| 8 | 降低 | 正常 | 开始生效 |
| 16 | 进一步降低 | 正常 | 改善明显 |
| 64 | 接近0 | 正常 | 趋于稳定 |
| 128 | ≈0 | 正常 | 最佳安全性 |

符合理论预期：$K$ 越大，无标签袋包含偏好轨迹的概率越高，代价函数学习越准确。

**部分轨迹长度 $H$ 的敏感性（$K=128$, Swimmer-Velocity）：**

| 轨迹长度 $H$ | Normalized Cost | 说明 |
|-------------|----------------|------|
| 1 | ≈0 | 稳定 |
| 5 | ≈0 | 稳定 |
| 10 | ≈0 | 稳定 |

当 $K$ 足够大时，安全性能对轨迹长度不敏感——支持使用部分轨迹以降低计算开销。

**加权方式对比**：
- 轨迹级加权（Eq. 12）vs 状态-动作级加权（Eq. 14）
- 在Swimmer-Velocity和Point-Goal1上两种方式表现相似

### 关键发现

1. **MIL框架有效解决了轨迹级到状态级的标签传递问题**：仅从轨迹级"非偏好"标签学到了精确的状态-动作级代价函数
2. **袋大小 $K$ 是关键超参数**：$K$ 过小时MIL信号不足，$K≥64$ 后趋于稳定
3. **部分轨迹训练计算高效且不损失性能**：实际应用中不需要完整轨迹长度
4. **SafeMIL在速度约束任务上优势明显**：几乎完全恢复了约束RL策略的安全水平
5. **导航任务中表现competitive**：在Point-Goal1上最优，其他导航任务上与基线匹配

## 亮点与洞察

1. **MIL formulation的巧妙应用**：将"轨迹中哪些状态-动作对是危险的"这个弱监督问题自然地映射为MIL中的"袋中哪些实例是正的"问题
2. **仅需50条非偏好轨迹**：极低的标注需求使方法在实际中高度可行
3. **理论保证**：通过Lemma 1和Theorem 1给出了评分函数有效性的概率性保证
4. **简洁的评分函数设计**：不需要复杂的注意力机制或深度嵌套结构，简单的加和评分即可工作
5. **广泛适用性**：方法不依赖特定环境假设，可扩展到任何安全性关键的序贯决策场景

## 局限与展望

1. **偏好轨迹比例 $\alpha$ 未知**：实际中可能需要估计或调参
2. **袋大小选择**：理论上 $K$ 越大越好，但计算开销也增加，需要权衡
3. **非偏好行为的代价同质性假设**：Theorem 1假设非偏好轨迹有相似代价，实际中可能不成立
4. **仅评估了有限的基线**：未与在线安全RL方法或基于偏好的更多方法对比
5. **连续动作空间**：当前主要在MuJoCo和导航任务验证，更复杂的高维任务（如自动驾驶）有待验证
6. **成本阈值 $\hat{b}$ 的设定**：硬阈值方法中需要先验知识设置适当的阈值

## 相关工作与启发

- **SafeDICE (Jang et al., 2023)**：直接估计偏好策略的稳态分布，是最直接的比较基线
- **T-REX (Brown et al., 2019)**：从排名轨迹学reward，但需要在线RL优化
- **DWBC (Xu et al., 2022)**：使用PU学习训练判别器作为BC权重
- **COptiDICE (Lee et al., 2022)**：带约束的离线RL，作为上界参考（使用完整reward+cost信息）
- 启发：**弱监督信号（轨迹级标签）结合恰当的学习框架（MIL）可以有效替代昂贵的逐步标注**

## 评分

- 新颖性: ⭐⭐⭐⭐（MIL用于安全IL的formulation创新）
- 实验充分度: ⭐⭐⭐⭐（6个环境+多维敏感性分析）
- 写作质量: ⭐⭐⭐⭐（问题定义清晰，理论与实验结合好）
- 价值: ⭐⭐⭐⭐（实用性强，50条非偏好轨迹即可学安全策略）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Boundary-to-Region Supervision for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/boundary_to_region_supervision_for_offline_safe_rl.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](../../ICLR2026/reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)

</div>

<!-- RELATED:END -->
