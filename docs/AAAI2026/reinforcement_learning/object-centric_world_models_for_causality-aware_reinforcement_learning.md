---
title: >-
  [论文解读] Object-Centric World Models for Causality-Aware Reinforcement Learning
description: >-
  [AAAI 2026][强化学习][以物体为中心的世界模型] 提出 STICA 框架，通过统一的以物体为中心的 Transformer 架构实现世界模型、策略网络和价值网络，其中世界模型将观测分解为独立物体的隐状态进行 token 级动力学预测，策略和价值网络通过因果注意力机制估计 token 级因果关系实现因果感知决策，在 Safety Gym 和 OCVRL 基准上显著超越 DreamerV3 等 SOTA 方法。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "以物体为中心的世界模型"
  - "因果注意力"
  - "基于模型的强化学习"
  - "注意力机制"
  - "Transformer"
---

# Object-Centric World Models for Causality-Aware Reinforcement Learning

**会议**: AAAI 2026  
**arXiv**: [2511.14262](https://arxiv.org/abs/2511.14262)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 以物体为中心的世界模型, 因果注意力, 基于模型的强化学习, Slot Attention, Transformer

## 一句话总结

提出 STICA 框架，通过统一的以物体为中心的 Transformer 架构实现世界模型、策略网络和价值网络，其中世界模型将观测分解为独立物体的隐状态进行 token 级动力学预测，策略和价值网络通过因果注意力机制估计 token 级因果关系实现因果感知决策，在 Safety Gym 和 OCVRL 基准上显著超越 DreamerV3 等 SOTA 方法。

## 研究背景与动机

深度强化学习（RL）在多个领域取得了成功，但仍需要大量的环境交互，在考虑实时操作和物理设备故障的真实世界任务中成本高昂。基于模型的 RL（MBRL）通过学习世界模型来在"想象"环境中优化策略，从而提高样本效率。

然而，现有世界模型面临以下关键挑战：

**整体表示的局限性**：从 Dreamer 系列到 STORM、IRIS 等最新 Transformer 世界模型，都学习环境的整体（holistic）表示。当环境高维、非平稳且包含多个物体及其复杂交互时，整体表示难以捕捉个体物体之间的重要关系和交互。

**人类认知的启示**：人类通过将环境分解为离散概念（如物体和事件）来感知环境，实现更高效和因果感知的决策。将这种认知机制融入世界模型有望让 RL 智能体在复杂场景中更有效地运作。

**现有物体中心 MBRL 的不足**：已有的物体中心 MBRL 方法要么需要随机探索的 episode（OODP、COBRA）、要么依赖监督学习（FOCUS、OC-STORM）或外部数据集预训练（SOLD），且大多不适用于非平稳环境和部分可观测设置（如第一人称视角）。**STICA 是首个从观测直接提取物体中心表示、无需随机探索/监督/预训练的 MBRL 智能体。**

## 方法详解

### 整体框架

STICA 由三个组件构成：
1. **以物体为中心的世界模型**：Slot-based 自编码器 + Transformer 动力学模型
2. **因果策略网络**：Transformer + 因果注意力
3. **因果价值网络**：Transformer + 因果注意力

训练流程遵循标准 MBRL 范式：从真实环境收集经验 → 训练世界模型 → 在想象轨迹上训练策略和价值网络。

### 关键设计

1. **Slot-based 自编码器**：将观测 $o_t$ 分解为物体中心表示和背景表示。

   **编码器**：使用 Slot Attention（Locatello et al. 2020）从观测中获得 $n$ 个 slot $(s_t^1, ..., s_t^n)$，每个 slot 作为 128 维 logits 定义 16 个 8 类别的 categorical 分布，从中采样得到隐状态 $z_t^i$。另外定义一个可学习的、**时间无关**的背景隐状态 $z_{BG}$，用于表示与时间无关的静态背景信息（如环境布局、智能体身体），使编码器能将动态物体与静态背景分开提取。

   **解码器**：Spatial Broadcast 解码器从隐状态重建个体 RGB 图像 $\hat{o}_t^k$ 和未归一化掩码 $m_t^k$，背景 $\hat{o}_{BG}$ 的掩码填充为 0。通过 softmax 归一化掩码后混合：$\hat{o}_t = \sum_{k=1}^{n} M_t^k \odot \hat{o}_t^k + M_{BG} \odot \hat{o}_{BG}$。

   **损失函数**：$\mathcal{L}_\phi^{ae} = \mathbb{E}_B[\frac{1}{T}\sum_{t=1}^{T}(\mathcal{J}_{rec.}^t + \alpha_1 \mathcal{J}_{ent.}^t + \alpha_2 \mathcal{J}_{cross}^t)]$，包含重建误差、熵正则化（防止分布退化为确定性）和交叉熵项（将提取的隐状态与动力学模型预测对齐）。由于 Slot Attention 提取顺序随机，交叉熵项在计算前需要重排索引以最小化 L1 距离。

2. **Transformer 动力学模型**：以 Transformer-XL 为核心的聚合模型，接收历史奖励 $r_{1:t-1}$、隐状态 $(z_{1:t}^1, ..., z_{1:t}^n)$ 和动作 $a_{1:t}$ 作为 token，通过因果掩码确保不访问未来时间步。创新点在于位置编码**仅依赖时间 $t$，不依赖物体索引 $1,...,n$**，保证输出对隐状态顺序的等变性。

   预测器包括：
    - **隐状态预测器**：$p_\psi^{\hat{z}}(\hat{z}_{t+1}^k | h_t^k)$，categorical 分布
    - **奖励预测器**：$p_\psi^{\hat{r}}(\hat{r}_t | h_t')$，正态分布
    - **折扣因子预测器**：$p_\psi^{\hat{\gamma}}(\hat{\gamma}_t | h_t')$，Bernoulli 变量

   损失函数：$\mathcal{L}_\psi^{dyn} = \mathbb{E}_B[\frac{1}{T}\sum_{t=1}^{T}(\mathcal{J}_{cross}^{t+1} + \beta_1 \mathcal{J}_{rew.}^t + \beta_2 \mathcal{J}_{dis.}^t)]$

3. **因果注意力（Causal Attention）机制**：这是 STICA 的核心创新。

   **动机**：环境包含因果物体（与动作选择和价值估计相关，如目标、障碍物）和非因果物体（无关，如地板）。策略需关注奖励相关物体，价值估计可完全忽略非因果物体。

   **因果图矩阵 $G$**：定义物体间的因果关系，$G_{i,j}$ 表示从物体 $j$ 到物体 $i$ 的因果关系存在性（1为策略/价值，2为因果物体，3为非因果物体）。

   **因果分数 $p_t^k \in [0,1]$**：通过 MLP 估计每个隐状态 $z_t^k$ 代表因果物体的概率。基于此构建权重矩阵 $W_t$，然后通过 $W_t G W_t^\top$ 表示 token 间的因果关系。

   **因果注意力计算**：$\text{CA}_t = \text{Norm}\left(\text{softmax}\left(\frac{Q_t K_t^\top}{\sqrt{d}}\right) \odot W_t G W_t^\top\right) V_t$

   这用因果关系矩阵来缩放标准注意力权重，使策略和价值网络的每一层注意力都考虑因果结构。所有隐状态共享相同位置编码，确保输出对顺序等变。

### 损失函数 / 训练策略

- 策略学习采用 A2C + GAE
- 策略和价值网络各自独立的 Transformer
- 每个 Transformer 处理 $n$ 个隐状态 token + 1个可学习的目标 token
- 端到端学习，无需预训练或监督

## 实验关键数据

### 主实验

**Safety Gym 基准**（8个3D任务，第一人称视角，非平稳目标，多物体交互）：

| 方法 | PointGoal1 | PointGoal2 | PointBtn1 | PointBtn2 | CarGoal1 | CarGoal2 | CarBtn1 | CarBtn2 | 平均 | 归一化平均 |
|------|-----------|-----------|----------|----------|---------|---------|--------|--------|------|-----------|
| PPO | 5.00 | 4.43 | 4.94 | 3.78 | 3.15 | 1.46 | 0.95 | 1.51 | 3.15 | 1.00 |
| TWM | 9.89 | 8.87 | 2.79 | 0.23 | 16.89 | 17.25 | 3.79 | 7.40 | 8.39 | 3.83 |
| DreamerV3 | 19.13 | 13.64 | 4.01 | 4.16 | 16.32 | 15.30 | 5.20 | 3.87 | 10.20 | 4.06 |
| TD-MPC2 | 8.00 | 6.50 | 4.31 | 4.84 | 2.57 | 1.24 | 1.90 | 0.48 | 3.73 | 1.15 |
| **STICA** | 13.63 | **13.64** | **11.52** | **5.97** | **17.09** | **18.27** | **9.65** | **9.25** | **11.90** | **5.49** |

**OCVRL 基准**（成功率）：

| 方法 | Obj. Goal | Obj. Interaction | Obj. Reaching |
|------|-----------|-----------------|---------------|
| TWM | 0.727 | 0.080 | 0.772 |
| DreamerV3 | 0.677 | 0.156 | 0.697 |
| **STICA** | **0.737** | **0.333** | **0.867** |

### 消融实验

| 配置 | PointButton1 | CarButton1 | 说明 |
|------|-------------|-----------|------|
| STICA（完整） | 最优 | 最优 | 全部组件 |
| STICA w/o CA | 显著降低 | 显著降低 | 移除因果注意力 |
| STICA w/o BR | 轻微降低 | 轻微降低 | 移除背景分离 |
| STICA w/o CA+BR | 中等降低 | 中等降低 | 移除两者 |
| STICA w/o CA+BR+TP+TV | 接近 TWM | 接近 TWM | 进一步移除 Transformer 策略/价值网络 |
| TWM | 基线 | 基线 | 无物体中心表示 |

消融结论的优先级排序：
1. **因果注意力**：最大提升（从 "w/o CA" 到完整 STICA）
2. **Transformer 策略/价值网络**：显著提升
3. **背景分离**：适度提升
4. **物体中心世界模型单独使用**：仅边际提升

### 关键发现

1. **Safety Gym 全面领先**：STICA 在8个任务中7个取得最佳，归一化平均分 5.49 vs DreamerV3 的 4.06（+35.2%）
2. **Button 任务优势巨大**：PointButton1 上 STICA 得分 11.52，DreamerV3 仅 4.01——因为 Button 任务需要在多个动态障碍物中识别目标按钮，正是物体中心+因果注意力的用武之地
3. **Obj. Interaction 任务**：STICA 成功率 0.333 vs DreamerV3 的 0.156（+113.5%），因为需要智能体直接与物体交互，物体中心世界模型和因果价值网络协同作用
4. **PointGoal1 例外**：该任务物体少且简单，整体表示足够，STICA 无显著优势
5. **因果注意力可视化**：价值网络几乎只关注奖励相关目标物体；策略网络主要关注目标但也适当关注其他物体——符合直觉

## 亮点与洞察

- **统一框架**：世界模型、策略、价值网络全部使用物体中心 Transformer，架构高度统一优雅
- **因果注意力的显式建模**：不只是用物体中心表示，还通过因果分数显式刻画"哪些物体对决策有因果影响"，且通过注意力权重可视化实现可解释性
- **背景分离的端到端学习**：通过可学习的时间无关背景隐状态 $z_{BG}$，无需预训练即可实现背景与前景分离
- **位置编码设计**：动力学模型中仅用时间位置编码（不用物体索引），保证对 slot 顺序的等变性——这对 Slot Attention 的随机提取顺序至关重要
- **首个无需额外监督的物体中心 MBRL**：不需要随机探索、标注数据或预训练

## 局限与展望

1. 因果注意力中的因果图 $G$ 结构是预定义的（3类：目标、因果物体、非因果物体），灵活性有限
2. 固定的 slot 数量 $n$ 可能不适应物体数量变化的场景
3. 仅在 Safety Gym（3D 第一人称）和 OCVRL（2D/3D 固定视角）上验证，未在更复杂的真实世界环境测试
4. Slot Attention 的随机提取顺序需要重排索引来计算损失，增加了计算开销
5. 因果分数 $p_t^k$ 的估计依赖 MLP，在更复杂场景中是否准确有待验证

## 相关工作与启发

- **Dreamer 系列**（Hafner et al.）：RSSM 世界模型的代表，STICA 直接对标 DreamerV3
- **STORM**（Zhang et al. 2023）：Transformer 世界模型，学习整体表示
- **TWISTER**（Burchi & Timofte 2025）：Transformer + 对比预测编码
- **OCRL**（Yoon et al. 2023）：模型无关的物体中心 RL，将 Transformer 用于策略和价值网络但缺乏世界模型
- **EIT**（Haramati et al. 2024）：Entity Interaction Transformer，类似物体中心策略但不含世界模型
- **SlotFormer**（Wu et al. 2023）：物体中心视频预测模型，是 STICA 动力学模型设计的参照

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （因果注意力 + 物体中心世界模型 + 背景分离的统一框架，设计非常优雅）
- 实验充分度: ⭐⭐⭐⭐ （Safety Gym + OCVRL 双基准测试，消融全面，可视化有说服力，但缺乏真实环境）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，公式严谨，图1的框架图非常直观）
- 价值: ⭐⭐⭐⭐⭐ （在物体中心 MBRL 方向上的重要推进，因果注意力有广泛适用性）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICLR 2026\] From Observations to Events: Event-Aware World Model for Reinforcement Learning](../../ICLR2026/reinforcement_learning/from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)
- [\[CVPR 2026\] Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](../../CVPR2026/reinforcement_learning/specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [\[CVPR 2026\] GeoWorld: Geometric World Models](../../CVPR2026/reinforcement_learning/geoworld_geometric_world_models.md)
- [\[AAAI 2026\] Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization](provably_efficient_multi-objective_bandit_algorithms_under_preference-centric_cu.md)

</div>

<!-- RELATED:END -->
