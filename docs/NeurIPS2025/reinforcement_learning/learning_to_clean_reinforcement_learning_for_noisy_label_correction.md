---
title: >-
  [论文解读] Learning to Clean: Reinforcement Learning for Noisy Label Correction
description: >-
  [NeurIPS 2025][强化学习][噪声标签] 将噪声标签纠正问题建模为强化学习中的马尔可夫决策过程，提出 RLNLC 框架，通过 k 近邻嵌入空间构建策略函数判断哪些标签需纠正，并设计标签一致性奖励和跨子集对齐奖励指导纠正过程，在多个基准数据集上的实例依赖和对称噪声场景中均达到最优性能。 深度神经网络在有噪声标签的数…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "噪声标签"
  - "标签纠正"
  - "策略梯度"
  - "Actor-Critic"
  - "k近邻"
---

# Learning to Clean: Reinforcement Learning for Noisy Label Correction

**会议**: NeurIPS 2025  
**arXiv**: [2511.19808](https://arxiv.org/abs/2511.19808)  
**代码**: 暂无  
**领域**: 强化学习  
**关键词**: 噪声标签, 标签纠正, 策略梯度, Actor-Critic, k近邻

## 一句话总结

将噪声标签纠正问题建模为强化学习中的马尔可夫决策过程，提出 RLNLC 框架，通过 k 近邻嵌入空间构建策略函数判断哪些标签需纠正，并设计标签一致性奖励和跨子集对齐奖励指导纠正过程，在多个基准数据集上的实例依赖和对称噪声场景中均达到最优性能。

## 研究背景与动机

深度神经网络在有噪声标签的数据集上训练时，会先学习干净数据的特征然后逐渐过拟合到噪声数据，导致泛化能力严重退化。现有噪声标签处理方法主要包括：

**样本选择方法**（Co-teaching、MentorNet）：利用训练动态（损失值、预测置信度）筛选可靠样本，但本质上是被动的筛选策略，无法主动纠正标签。

**标签纠正方法**（PLC、SSR）：利用模型预测调整标签，但通常仅做单步贪心纠正，缺乏**长期后果考虑**——某次纠正可能在短期看合理但长期看导致错误累积。

**半监督学习方法**（DivideMix、LongReMix）：将噪声数据视为无标签数据，但依赖固定的分割机制，难以动态适应数据特征变化。

核心洞察：**噪声标签纠正天然适合 RL 的序列决策框架**——纠正是一系列动作，每次纠正改变数据状态，需要最大化长期累积奖励（最终标签质量），且需要探索不同纠正策略以避免局部最优。

## 方法详解

### 整体框架

RLNLC 将问题定义为 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, \mathcal{R}, \gamma)$：
- **状态**：当前数据集及其标签 $\boldsymbol{s}^t = \{(\mathbf{x}_i, \hat{\mathbf{y}}_i^t)\}_{i=1}^N$
- **动作**：对每个样本的二值纠正决策 $a_i \in \{0, 1\}$
- **转移**：确定性地用 k 近邻预测标签替换被选中纠正的标签
- **奖励**：评估纠正后标签质量的复合函数

通过 Actor-Critic 方法学习最优策略，训练完成后部署策略迭代纠正标签，再用"清洗"后的标签微调预测模型。

### 关键设计

1. **基于 k 近邻的策略函数**：策略函数建立在特征提取网络 $f_\theta$ 的嵌入空间上。对每个样本 $\mathbf{x}_i$，通过注意力加权其 k 近邻的标签生成预测标签 $\bar{\mathbf{y}}_i = \sum_{j \in \mathcal{N}(\mathbf{x}_i)} \alpha_{ij} \hat{\mathbf{y}}_j^t$，其中注意力权重通过余弦相似度和温度参数 $\tau$ 计算。纠正概率定义为：

    $\pi_\theta(\boldsymbol{s}^t)_i = \frac{\sum_{j=1}^C \mathbb{1}(\bar{\mathbf{y}}_{ij} > \bar{\mathbf{y}}_{i\hat{y}_i}) \cdot \bar{\mathbf{y}}_{ij}}{\sum_{j=1}^C \mathbb{1}(\bar{\mathbf{y}}_{ij} \geq \bar{\mathbf{y}}_{i\hat{y}_i}) \cdot \bar{\mathbf{y}}_{ij}}$

   这个设计非常精巧：它量化了 k 近邻预测与当前标签的不一致程度——如果当前标签在 k 近邻预测中已是最大概率类，纠正概率为零；偏离越大，纠正概率越高。

2. **双重奖励函数**：

    - **标签一致性奖励 (LCR)**：评估纠正后全局标签的统计平滑性。使用独立的固定骨干 $f_\omega$（与策略网络解耦），计算每个样本标签与其 k 近邻标签的 KL 散度的负均值：$\mathcal{R}_{\text{LCR}} = -\mathbb{E}_{i \in [1:N]}[\text{KL}(\hat{\mathbf{y}}_i^{t+1}, \sum_j \alpha_{ij} \hat{\mathbf{y}}_j^{t+1})]$
    - **噪声标签对齐奖励 (NLA)**：将数据划分为"干净子集"（$a_i=0$，标签未改动）和"噪声子集"（$a_i=1$，标签被纠正），计算噪声子集中每个样本的纠正标签与其在干净子集中 k 近邻标签的 KL 散度负均值。
    - 复合奖励：$\mathcal{R} = \exp(\mathcal{R}_{\text{LCR}} + \lambda \mathcal{R}_{\text{NLA}})$，用指数函数将无界的负 KL 散度映射到 $(0, 1]$，确保奖励有界。

3. **Critic 的高效输入编码**：由于确定性转移机制，用下一状态 $\boldsymbol{s}^{t+1}$ 替代 $(s^t, a^t)$ 作为 Critic 输入。为降低维度（状态维度与数据集大小 $N$ 成正比），采用分箱策略：将每个样本按其标签一致性奖励 $r(\mathbf{x}_i, \hat{\mathbf{y}}_i^{t+1})$ 分到 $N_b$ 个 bin 中（$N_b \ll N$），用各 bin 的样本比例构成长度为 $N_b$ 的向量作为 Critic 输入。

### 损失函数 / 训练策略

- **Actor 更新**：$\theta \leftarrow \theta + \beta_\theta \nabla_\theta \log \pi_\theta(\boldsymbol{a}^t | \boldsymbol{s}^t) Q(\boldsymbol{s}^t, \boldsymbol{a}^t)$
- **Critic 更新**：使用 SARSA 的 TD 误差 $\delta^{t-1} = \mathcal{R}(\boldsymbol{s}^{t-1}, \boldsymbol{a}^{t-1}) + \gamma Q(\boldsymbol{s}^t, \boldsymbol{a}^t) - Q_\phi(\boldsymbol{s}^{t-1}, \boldsymbol{a}^{t-1})$
- **初始状态随机化**：每个训练 epoch 开始前，随机修改 $\boldsymbol{s}_0^0$ 的少量标签生成扰动初始状态 $\boldsymbol{s}^0$，增强探索性
- **训练→部署→微调**：先训练策略网络 500 epochs，再部署策略执行 $T'=25$ 步标签纠正，最后用纠正标签微调预测模型 100 epochs

## 实验关键数据

### 主实验

**CIFAR10-IDN / CIFAR100-IDN 实例依赖噪声（测试准确率 %）**

| 方法 | CIFAR10 20% | CIFAR10 40% | CIFAR10 50% | CIFAR100 20% | CIFAR100 40% | CIFAR100 50% |
|------|-----------|-----------|-----------|------------|------------|------------|
| CE | 75.8 | 62.5 | 39.4 | 30.4 | 21.5 | 14.4 |
| DivideMix | 94.8 | 94.5 | 93.0 | 77.1 | 70.8 | 58.6 |
| SSR | 96.5 | 96.3 | 94.1 | 78.8 | 77.0 | 72.8 |
| **RLNLC** | **97.3** | **96.9** | **95.8** | **80.5** | **78.5** | **74.7** |

**真实世界噪声数据集**

| 方法 | Animal-10N | Food-101N |
|------|-----------|----------|
| SURE | 89.0 | - |
| LongReMix | - | 87.3 |
| **RLNLC** | **90.2** | **89.2** |

### 消融实验

**CIFAR100-IDN 消融研究（测试准确率 %）**

| 配置 | 20% | 30% | 40% | 45% | 50% | 说明 |
|------|-----|-----|-----|-----|-----|------|
| RLNLC (完整) | **80.5** | **80.1** | **78.5** | **77.2** | **74.7** | 全部组件 |
| w/o $\mathcal{R}_{\text{NLA}}$ | 78.4 | 77.9 | 76.2 | 76.3 | 72.0 | 移除跨子集对齐奖励 |
| w/o $\mathcal{R}_{\text{LCR}}$ | 79.3 | 78.5 | 76.1 | 76.1 | 73.9 | 移除标签一致性奖励 |
| w/o 随机化 $s^0$ | 79.9 | 79.5 | 77.8 | 76.8 | 73.1 | 不扰动初始状态 |
| $f_\omega \leftarrow f_\theta$ | 78.4 | 76.9 | 75.2 | 74.3 | 73.8 | 策略和奖励共享特征网络 |

### 关键发现

- **RLNLC 在所有噪声类型和比例上均 SOTA**：在 CIFAR10-IDN 50% 噪声下比 SSR 高 1.7%，在 CIFAR100-IDN 全部 5 个噪声比例下一致超出 SSR 1.5-2.2%。
- **极端对称噪声下优势更明显**：在 CIFAR100 90% 对称噪声下达到 44.2%，远超 DivideMix 的 31%（+13.2%），说明 RL 的探索能力在高噪声场景中至关重要。
- **标签纠正精度随步数稳定提升**：CIFAR10-IDN 在低噪声下 $T'=5$ 步即可达 90%+ 纠正准确率，高噪声下 $T'=10$ 步也能达到 90%。
- **解耦策略和奖励网络至关重要**：$f_\omega \leftarrow f_\theta$ 变体性能大幅下降（50% 噪声下降 0.9%），因为共享特征会导致奖励评估与策略优化的循环偏差。

## 亮点与洞察

- **RL 视角的独特价值**：噪声标签纠正作为序列决策问题的建模非常自然——标签纠正是动作，标签状态会演化，需要非近视的长期优化。相比单步贪心方法，RL 能通过探索发现更好的纠正路径。
- 策略函数的设计巧妙地将 k 近邻的预测不一致性转化为纠正概率，既有可解释性（不一致性越大越可能是噪声）又可微分（通过特征网络 $f_\theta$）。
- Critic 的分箱编码方案简洁高效——将 $N$ 维输入压缩到 $N_b$ 维（默认 100），同时保留了状态分布的全局信息。

## 局限与展望

- 计算成本较高：需要预训练特征网络 + RL 策略学习 500 epochs + 策略部署 25 步 + 微调 100 epochs，整体流程较复杂。
- k 近邻计算在大规模数据集上可能成为瓶颈（需每步重新计算特征距离）。
- 仅在分类任务上验证，未测试检测、分割等其他标签噪声场景。
- 动作空间仅为二值（纠正/不纠正），未考虑"纠正为哪个类"的更精细动作设计——当前依赖 k 近邻预测的自动替换。
- 缺少与其他 RL 方法（如 PPO、SAC）的比较，Actor-Critic 选择的合理性未充分论证。

## 相关工作与启发

- 与 DivideMix 等半监督方法的区别在于：DivideMix 通过 GMM 拟合损失分布做一次性划分，而 RLNLC 通过多步序列决策迭代改进标签，具有自适应性。
- 与 RLHF 的思路有交叉——RLHF 用 RL 优化人类偏好，RLNLC 用 RL 优化标签质量，都是将判别问题转为 RL 决策问题。
- 初始状态随机化的技巧（类似 domain randomization）可推广到其他 RL-for-ML 框架中。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将噪声标签纠正建模为 RL 的视角新颖，策略函数和奖励设计独特
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个数据集、多种噪声类型和比例、全面消融和超参敏感性分析
- 写作质量: ⭐⭐⭐⭐ 数学推导详细，方法表述清晰
- 价值: ⭐⭐⭐⭐ 开辟了 RL 在数据质量提升中的新应用方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [\[NeurIPS 2025\] ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[ICML 2025\] Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets](../../ICML2025/reinforcement_learning/adversarial_cooperative_rationalization_the_risk_of_spurious_correlations_in_eve.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[ICLR 2026\] Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring](../../ICLR2026/reinforcement_learning/beyond_noisy-tvs_noise-robust_exploration_via_learning_progress_monitoring.md)

</div>

<!-- RELATED:END -->
