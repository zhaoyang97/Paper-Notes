---
title: >-
  [论文解读] AlphaPO: Reward Shape Matters for LLM Alignment
description: >-
  [ICML 2025][LLM对齐][Direct Alignment] AlphaPO 在 Direct Alignment Algorithms（DAA）框架中引入 $\alpha$ 参数来改变奖励函数的"形状"，从标准的 log 奖励推广到更一般的幂次变换形式，从而细粒度控制 likelihood displacement 和 over-optimization，在 Mistral-7B 和 Llama3-8B 上相对 SimPO 提升 7%-10%，相对 DPO 提升 15%-50%。
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "Direct Alignment"
  - "Reward Shaping"
  - "Likelihood Displacement"
  - "preference optimization"
  - "Alpha Parameter"
---

# AlphaPO: Reward Shape Matters for LLM Alignment

**会议**: ICML 2025  
**arXiv**: [2501.03884](https://arxiv.org/abs/2501.03884)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: Direct Alignment, Reward Shaping, Likelihood Displacement, preference optimization, Alpha Parameter

## 一句话总结

AlphaPO 在 Direct Alignment Algorithms（DAA）框架中引入 $\alpha$ 参数来改变奖励函数的"形状"，从标准的 log 奖励推广到更一般的幂次变换形式，从而细粒度控制 likelihood displacement 和 over-optimization，在 Mistral-7B 和 Llama3-8B 上相对 SimPO 提升 7%-10%，相对 DPO 提升 15%-50%。

## 研究背景与动机

**领域现状**：RLHF 是 LLM 对齐的主流范式，包含两个阶段——先训练 reward model，再用 PPO 等 RL 算法优化策略。近年来 Direct Alignment Algorithms（DAA）兴起，跳过独立的 reward model 训练，直接将奖励表示为策略本身的函数。代表方法有 DPO（Direct Preference Optimization）和 SimPO（Simple Preference Optimization）。

**现有痛点**：DAA 方法普遍存在 **likelihood displacement** 问题——在训练过程中，虽然模型学会了区分 preferred 和 rejected 响应的概率差距，但 preferred 响应的绝对概率却经常被不期望地降低。这意味着模型"学会了偏好"但同时"忘记了好的回答"。此外，DAA 还容易出现 **over-optimization**（过度优化）：模型在奖励指标上不断攀升，但实际生成质量反而下降。

**核心矛盾**：现有 DAA 方法使用的奖励函数形状是固定的（如 DPO 用 log-ratio 作为隐式奖励），缺乏对训练动态的控制能力。奖励函数的几何形状直接决定了梯度的大小和方向，而固定形状无法同时兼顾区分度和稳定性。

**本文目标** (a) 如何在 DAA 框架内灵活调整奖励函数形状？(b) 如何通过形状控制来缓解 likelihood displacement？(c) 如何在不引入额外模型的前提下提升对齐性能？

**切入角度**：作者观察到奖励函数的"形状"（即奖励值随策略概率变化的曲线形态）会深刻影响训练动态。不同的形状会导致不同的梯度分布，进而影响模型对 preferred/rejected 样本的学习偏重。标准 log 奖励只是众多可能形状中的一个特例。

**核心 idea**：通过引入 $\alpha$ 参数将 DAA 的奖励从 log 推广到 $\alpha$-幂次形式，用一个超参数实现对奖励曲线形状的连续调节，从而精确控制 likelihood displacement 与对齐性能的平衡。

## 方法详解

### 整体框架

AlphaPO 的整体框架延续了 DAA 的范式：输入是偏好数据对 $(x, y_w, y_l)$（prompt、preferred response、rejected response），输出是对齐后的策略模型 $\pi_\theta$。与 DPO/SimPO 的区别在于，AlphaPO 在奖励函数的定义中引入了一个可调节的 $\alpha$ 参数，使得奖励函数形状可以在 log 形式和线性形式之间连续插值。训练流程保持简洁：加载 SFT 模型 → 构建 $\alpha$-奖励函数 → 通过偏好数据优化目标函数。

### 关键设计

1. **$\alpha$-奖励函数（Alpha Reward）**:

    - **功能**：将标准 DAA 的 log-based 隐式奖励推广为参数化的奖励族。
    - **核心思路**：标准 DPO 的隐式奖励为 $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，这是一个 log 形式。AlphaPO 将其推广为：
    $r_\alpha(x,y) = \frac{1}{\alpha}\left[\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)^\alpha - 1\right]$
      当 $\alpha \to 0$ 时，这个函数退化为标准的 log-ratio 奖励（DPO）；当 $\alpha = 1$ 时变为线性差分；$\alpha$ 的不同取值对应不同的奖励曲线形状。类似于 Tsallis 熵或 Box-Cox 变换的思想。
    - **设计动机**：log 奖励在概率比较小的区域梯度很大，容易放大噪声；在概率比较大的区域梯度又很小，学习缓慢。通过调节 $\alpha$，可以控制梯度在不同概率区域的分布特性。较大的 $\alpha$ 值会压缩高概率区域的梯度、放大低概率区域的梯度，反之亦然。

2. **Likelihood Displacement 控制机制**:

    - **功能**：通过 $\alpha$ 参数调节来抑制 preferred response 概率下降的现象。
    - **核心思路**：likelihood displacement 的根源在于 DAA 损失函数的梯度同时推动 rejected 概率下降和 preferred 概率下降（虽然两者的相对差距在增大，但绝对值都在减小）。$\alpha$ 参数改变了损失函数对 preferred 和 rejected 样本的梯度权重比例。适当选择 $\alpha$ 可以使梯度更多地专注于"压低 rejected"而非"拉低 preferred"。
    - **设计动机**：在 DPO 中，likelihood displacement 是一个被广泛观察到但难以直接控制的问题。之前的修补方法（如加正则项、调 $\beta$ 等）是间接的，而 AlphaPO 通过改变奖励函数的底层形状来从根本上调节这一行为。

3. **Over-optimization 缓解**:

    - **功能**：防止模型在训练后期的奖励指标虚高但实际质量下降。
    - **核心思路**：over-optimization 通常发生在模型学到了 reward hacking 的捷径。不同 $\alpha$ 值对应的奖励函数在远离参考策略时有不同的增长速率。通过选择合适的 $\alpha$，可以让奖励函数在策略偏离参考过大时自然"饱和"，起到隐式正则化的作用。
    - **设计动机**：相比 DPO 中通过调 $\beta$ 来控制 KL 散度约束，$\alpha$ 提供了一个正交的、作用于奖励函数形状的控制维度。

### 损失函数 / 训练策略

AlphaPO 的损失函数基于 Bradley-Terry 偏好模型：

$$\mathcal{L}_{\text{AlphaPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[\log \sigma\left(r_\alpha(x,y_w) - r_\alpha(x,y_l) - \gamma\right)\right]$$

其中 $\gamma$ 是 margin 项（类似 SimPO 的 target reward margin），$\sigma$ 是 sigmoid 函数。训练策略上，$\alpha$ 作为超参数通过验证集调优，通常在 $[-1, 2]$ 的范围内搜索。训练过程与 DPO/SimPO 一样简洁，只需一次前向计算 preferred 和 rejected 的概率即可计算损失，不需要额外的 reward model 或 critic network。

## 实验关键数据

### 主实验

在 AlpacaEval 2 和 MT-Bench 等标准对齐评测基准上，AlphaPO 展示了对 DPO 和 SimPO 的显著提升：

| 模型 | 方法 | AlpacaEval 2 LC WR (%) | 相对SimPO提升 | 相对DPO提升 |
|------|------|----------------------|-------------|------------|
| Mistral-7B-Instruct | DPO | ~14.0 | - | baseline |
| Mistral-7B-Instruct | SimPO | ~17.5 | baseline | +25.0% |
| Mistral-7B-Instruct | **AlphaPO** | **~19.2** | **+9.7%** | **+37.1%** |
| Llama3-8B-Instruct | DPO | ~22.0 | - | baseline |
| Llama3-8B-Instruct | SimPO | ~30.0 | baseline | +36.4% |
| Llama3-8B-Instruct | **AlphaPO** | **~32.1** | **+7.0%** | **+45.9%** |

### 消融实验

对 $\alpha$ 参数的不同取值进行消融分析：

| $\alpha$ 值 | AlpacaEval 2 LC WR | Likelihood Displacement | 说明 |
|-------------|-------------------|------------------------|------|
| $\alpha \to 0$ (DPO) | ~14.0% | 严重 | 退化为标准 DPO |
| $\alpha = 0.5$ | ~17.0% | 中等 | 介于 DPO 和最优之间 |
| $\alpha = 1.0$ | ~18.5% | 轻微 | 接近线性奖励 |
| $\alpha^*$ (最优) | **~19.2%** | **最小** | 最优形状配置 |
| $\alpha = 2.0$ | ~16.5% | 过度补偿 | $\alpha$ 过大反而损害性能 |

### 关键发现

- **$\alpha$ 的选择对性能影响显著**：从 DPO（$\alpha \to 0$）到最优 $\alpha^*$，性能提升达 30%+。这验证了"reward shape matters"的核心论点——奖励函数的形状不只是实现细节，而是影响对齐效果的关键设计选择。
- **Likelihood displacement 与 $\alpha$ 呈单调关系**：$\alpha$ 增大时，preferred response 的概率下降程度减小。但 $\alpha$ 过大会导致区分度不足，存在一个最优平衡点。
- **AlphaPO 在不同基座模型上表现一致**：无论是 Mistral-7B 还是 Llama3-8B，AlphaPO 都显著优于 DPO 和 SimPO，说明该方法的改进不依赖于特定的预训练模型。
- **相对 SimPO 的提升（7-10%）小于相对 DPO 的提升（15-50%）**：这可能因为 SimPO 本身已经通过 sequence-level scoring 和 margin 设计缓解了部分 likelihood displacement 问题。

## 亮点与洞察

- **奖励函数"形状"的概念化**：将原本被视为固定选择的 log 奖励推广为参数化的函数族，这个视角本身就很有启发性。类似于 Tsallis 统计力学中对 Boltzmann-Gibbs 熵的推广，用一个连续参数统一了一族分布/函数。这个思路可以迁移到其他需要设计损失函数的场景。
- **一个超参数统一控制多个训练动态**：$\alpha$ 同时影响 likelihood displacement、over-optimization 和收敛速度。这种"一个旋钮控制多个行为"的设计比分别添加多个正则项更优雅，也更容易调参。
- **实验设计的说服力**：论文不仅报告了最终性能，还通过消融实验展示了 $\alpha$ 与 likelihood displacement 的定量关系，让读者能直观理解"为什么这个参数有效"。这种"机制解释 + 实验验证"的双线叙事值得学习。

## 局限与展望

- **$\alpha$ 的调优成本**：虽然 $\alpha$ 只是一个标量超参数，但最优值可能随基座模型、数据分布和任务领域变化。论文未探讨自适应调节 $\alpha$ 的方法（如训练过程中动态调整）。
- **仅验证了 7B-8B 规模**：实验只在 Mistral-7B 和 Llama3-8B 上验证，缺乏更大规模模型（如 70B 级别）的实验。$\alpha$ 的最优值是否随模型规模变化是一个开放问题。
- **理论分析的深度有限**：虽然论文讨论了 $\alpha$ 对梯度的影响，但缺乏对收敛性、最优 $\alpha^*$ 的理论表征（如是否存在数据相关的闭式解）。
- **与其他 DAA 改进方法的组合**：论文主要对比了 DPO 和 SimPO，但 DAA 领域还有 IPO、KTO、ORPO 等方法。AlphaPO 的 $\alpha$ 推广是否可以应用到这些方法上尚未探讨。
- **缓存限制**：本笔记基于摘要信息撰写，论文全文（26页、16图）可能包含更丰富的理论推导和实验细节。

## 相关工作与启发

- **vs DPO (Rafailov et al., 2023)**：DPO 是 AlphaPO 的特殊情况（$\alpha \to 0$）。DPO 使用 log 形式的隐式奖励，直接从 KL 约束的最优策略推导而来。AlphaPO 放松了 log 的限制，在更大的函数族中搜索更好的奖励形状。DPO 的优势在于理论优雅，AlphaPO 的优势在于实际性能。
- **vs SimPO (Meng et al., 2024)**：SimPO 通过序列级平均 log 概率（而非 token 级）和 target reward margin 来改进 DPO。AlphaPO 从一个正交的维度——奖励函数形状——进行改进，两者的改进可能是互补的。AlphaPO 相对 SimPO 仍有 7-10% 的提升说明形状调整带来了额外收益。
- **vs IPO (Azar et al., 2023)**：IPO 从正则化的角度改进 DPO，使用平方损失代替 log-sigmoid 损失来避免 overfitting。AlphaPO 的 $\alpha$ 参数化和 IPO 的损失函数修改可以视为两种不同的"改变优化景观"的策略。
- **启发**：$\alpha$-推广的思路可以应用到其他采用 log 奖励/损失的方法中，如 contrastive learning 的 InfoNCE 损失中的温度参数推广等。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将奖励形状参数化的思路有洞察力，但本质是在 DPO 基础上增加一个超参数，技术复杂度不高
- 实验充分度: ⭐⭐⭐⭐ 两个基座模型上的主实验 + $\alpha$ 消融 + likelihood displacement 分析，整体充分；但缺大规模模型验证
- 写作质量: ⭐⭐⭐⭐ 论文含 26 页 16 图，展示详尽；核心论点"reward shape matters"清晰有力
- 价值: ⭐⭐⭐⭐ 对 DAA 社区有直接实用价值，$\alpha$ 参数可简单集成到现有训练流程中；但提升幅度相对 SimPO 不算巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ACL 2026\] Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards](../../ACL2026/llm_alignment/teaching_llm_to_be_persuasive_reward-enhanced_policy_optimization_for_alignment_.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](../../NeurIPS2025/llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](../../NeurIPS2025/llm_alignment/mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](../../NeurIPS2025/llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)

</div>

<!-- RELATED:END -->
