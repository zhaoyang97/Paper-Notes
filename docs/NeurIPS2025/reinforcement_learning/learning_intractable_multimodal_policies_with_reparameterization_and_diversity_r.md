---
title: >-
  [论文解读] Learning Intractable Multimodal Policies with Reparameterization and Diversity Regularization
description: >-
  [NeurIPS 2025][强化学习][多模态策略] 提出Diversity-regularized Actor Critic（DrAC）算法，通过将不可解析的多模态策略（amortized actor和diffusion actor）统一为stochastic-mapping formulation，利用重参数化技巧直接进行策略梯度优化，并设计基于距离的多样性正则化替代传统熵正则化，在多目标导航和生成式RL等多样性关键任务中展现显著优势。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "多模态策略"
  - "重参数化"
  - "多样性正则化"
  - "扩散策略"
  - "actor-critic"
---

# Learning Intractable Multimodal Policies with Reparameterization and Diversity Regularization

**会议**: NeurIPS 2025  
**arXiv**: [2511.01374](https://arxiv.org/abs/2511.01374)  
**代码**: [GitHub](https://github.com/PneuC/DrAC)  
**领域**: 强化学习  
**关键词**: 多模态策略, 重参数化, 多样性正则化, 扩散策略, actor-critic

## 一句话总结

提出Diversity-regularized Actor Critic（DrAC）算法，通过将不可解析的多模态策略（amortized actor和diffusion actor）统一为stochastic-mapping formulation，利用重参数化技巧直接进行策略梯度优化，并设计基于距离的多样性正则化替代传统熵正则化，在多目标导航和生成式RL等多样性关键任务中展现显著优势。

## 研究背景与动机

现实世界中的决策常常是多模态的——面对同一状态，可能存在多种同样好但本质不同的决策方案。例如迷宫导航中多条可行路径、对战游戏中的多元化策略、生成式任务中的多样性需求等。然而，主流深度RL算法（如SAC、TD3、DDPG）几乎清一色采用确定性或单模态高斯策略，无法表达复杂的多模态决策分布。

**学习多模态策略的核心挑战在于不可解析性**：

**amortized actor**（如SQL中使用的）：将状态和隐变量拼接送入网络直接输出动作，但决策概率$\pi_\theta(a|s)$没有解析形式。

**diffusion actor**（如DACER）：通过扩散过程迭代生成动作，表达力极强但概率同样不可解析。

不可解析意味着：
- 无法直接计算熵$\mathcal{H}(\pi(\cdot|s))$，因此最大熵RL框架不自然适用。
- 现有尝试要么牺牲表达力使用可解析但较弱的多模态模型，要么依赖SVGD等变分推断技术但性能不佳或计算开销大。
- DACER虽通过噪声缩放控制多样性，但并非通过梯度直接优化策略参数来提升多样性。

**本文的核心idea**：不可解析的概率密度不代表不可训练！只要策略可以写成"确定性映射+固定隐分布"的形式，就能通过重参数化技巧绕过概率密度计算，直接进行策略梯度优化。

## 方法详解

### 整体框架

DrAC建立在actor-critic架构之上，包含三个核心创新：
- 将多模态actor统一定义为stochastic-mapping actor
- 通过重参数化实现无需概率密度的策略梯度
- 用基于距离的多样性正则化替代熵正则化

### 关键设计

1. **Stochastic-Mapping Actor统一框架**：将策略定义为$\pi_\theta = \{f_\theta, p_z\}$——参数化映射函数$f_\theta: \mathcal{S} \times \mathcal{Z} \to \mathcal{A}$和固定隐分布$p_z$的组合。采样过程为$a \leftarrow f_\theta(s, z), z \sim p_z$。

    - **Amortized actor**：$f_\theta^{Amort}(s, z) \equiv g_\theta(s \oplus z)$，直接将状态和隐变量拼接送入网络。
    - **Diffusion actor**：$f_\theta^{Diffus}(s, z) \equiv x_0$，其中$x_0$通过$T$步逆扩散过程从纯噪声$z_T$迭代得到。

   这一统一视角揭示了两种actor的共同本质，为统一优化奠定基础。

2. **重参数化策略梯度（PGRT）**：论文证明，对任意stochastic-mapping actor，策略梯度可通过重参数化trick估计：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim d^\pi, z \sim p_z}[\nabla_a Q(s, f_\theta(s, z)) \nabla_\theta f_\theta(s, z)]$$

即直接将Q函数的动作梯度反向传播到$f_\theta$的参数，完全不需要计算$\pi_\theta(a|s)$。这比SVGD更高效且效果更好。

3. **基于距离的多样性正则化**：传统熵正则化需要概率密度，不适用于不可解析actor。本文提出使用成对距离的对数几何均值作为多样性度量：

$$D^\pi(s) = \mathbb{E}_{x,y \sim \pi(\cdot|s)}[\log \delta(x, y)]$$

其中$\delta$为L2距离。使用几何均值（而非算术均值）的关键原因是：算术均值可能高估多样性——当数据形成几个远离的小簇时，平均成对距离大但实际多样性低；几何均值对小值敏感，能避免这种高估。在对数尺度上操作使其更易与奖励平衡。

### 损失函数 / 训练策略

- **Critic损失**（双critic + target网络，融入多样性正则化）：

$$\mathcal{L}_\phi = \mathbb{E}_{s,a,r,s' \sim \mathcal{D}}[\text{MSE}(Q(s,a;\phi_i), r + \gamma(\tilde{V}(s';\hat{\phi}) + \alpha \tilde{D}_\theta(s')))]$$

- **Actor损失**（PGRT + 多样性梯度）：

$$\mathcal{L}_\theta = -\mathbb{E}_{s \sim \mathcal{D}, z \sim p_z}[Q(s, f_\theta(s,z); \phi) + \alpha \tilde{D}_\theta(s)]$$

- **自动系数调整**：借鉴SAC的自动温度调整，设定目标多样性$\hat{D}$后自动优化系数$\alpha$：

$$\mathcal{L}_\alpha = \mathbb{E}_{s \sim \mathcal{D}}[\alpha(\tilde{D}_\theta(s) - \hat{D})]$$

## 实验关键数据

### 多目标迷宫导航

| 算法 | 简单迷宫可达目标数 | 中等迷宫可达目标数 | 困难迷宫可达目标数 | 障碍物鲁棒性 |
|------|-----------------|-----------------|-----------------|------------|
| SAC（单模态） | ~2 | ~2 | ~1 | 低 |
| SQL（amortized） | ~4 | ~3 | ~2 | 中 |
| DACER（diffusion） | ~2 | ~2 | ~1 | 低 |
| **DrAmort（ours）** | **~4** | **~4** | **~3** | **最高** |
| DrDiffus（ours） | ~3 | ~3 | ~2 | 中 |

### 生成式RL（游戏关卡生成）

| 算法 | MarioPuzzle回报 | MarioPuzzle多样性 | MultiFacet回报 | MultiFacet多样性 |
|------|----------------|-----------------|---------------|----------------|
| SAC | 中等 | 低 | 中等 | 低 |
| SQL | 低 | 中等 | 低 | **高** |
| DACER | 中等 | 低 | 中等 | 中等 |
| **DrAmort** | **最高** | **最高** | **最高** | 高 |

### MuJoCo标准基准（6个locomotion任务）

| 算法 | 最优任务数 | 整体表现 |
|------|---------|---------|
| SAC | 2/6 | 基准水平 |
| SQL | 0/6 | 最差 |
| DACER | 1/6 | 与SAC持平 |
| **DrAmort** | **3/6** | **最优或持平** |
| DrDiffus | 0/6 | 与DACER持平 |

### 关键发现

1. **Amortized actor胜出**：在所有实验中，基于amortized actor的DrAmort展现了最强的多模态表达力和最佳整体性能。扩散actor在多模态表达上反而不如预期，且推理和训练速度显著更慢。
2. **多样性带来鲁棒性**：在out-of-distribution测试中（删除目标、添加障碍物），高多样性策略（DrAmort）展现了最佳的少样本鲁棒性，确认多模态策略在实际场景中的价值。
3. **PGRT优于SVGD**：SQL使用SVGD作为策略梯度估计器，DrAmort使用重参数化trick，后者在amortized actor上持续表现更好，表明重参数化是更优的梯度估计方法。
4. **距离正则化优于噪声缩放**：DACER通过缩放额外噪声控制多样性，在多目标迷宫中几乎无法学到多模态行为，而DrDiffus使用距离正则化则可以。

## 亮点与洞察

- 将看似不同的amortized actor和diffusion actor统一到同一理论框架下，揭示了它们本质上都是"确定性映射+随机源"的形式，为统一优化提供了理论基础。
- 用几何均值替代算术均值来度量多样性，这个看似微小的选择在实验中产生了重要影响。
- 为amortized actor正名——在合理的训练算法下，这种简单的模型在多模态RL中表现出令人惊讶的强大能力。

## 局限与展望

- 扩散actor可能需要更深的网络、更多扩散步数和更精细的超参调优才能充分发挥潜力。
- 基于距离的多样性度量可以探索更多的距离函数和聚合方式。
- 目前仅在连续动作空间验证，离散或混合动作空间的适配有待探索。
- 温度调度策略可进一步优化。

## 相关工作与启发

- 与SQL、S2AC等基于SVGD的方法对比，验证了重参数化路线在训练不可解析actor时的优越性。
- 与DACER的对比揭示了"通过噪声缩放控制多样性"与"通过梯度优化多样性"之间的本质差异。
- 为quality-diversity RL领域提供了新的实用工具。

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一框架和多样性正则化有创新，但各组件均有先驱工作
- 实验充分度: ⭐⭐⭐⭐⭐ 多目标导航、生成式RL、MuJoCo三类场景全面评估
- 写作质量: ⭐⭐⭐⭐ 条理清晰，图示直观
- 价值: ⭐⭐⭐⭐ 为多模态RL提供了实用且高效的算法框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](../../ICML2025/reinforcement_learning/robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](../../ICML2025/reinforcement_learning/fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)

</div>

<!-- RELATED:END -->
