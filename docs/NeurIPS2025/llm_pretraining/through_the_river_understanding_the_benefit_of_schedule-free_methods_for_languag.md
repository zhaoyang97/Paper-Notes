---
title: >-
  [论文解读] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training
description: >-
  [NeurIPS 2025][预训练][Schedule-Free优化器] 从 river-valley 损失景观的几何视角，分析了 Schedule-Free (SF) 优化器在语言模型预训练中不需要学习率衰减和权重平均就能持续追踪最优解的原因，并揭示 SF 隐式执行了权重平均，进而提出解耦动量和平均窗口的改进版 SF-AdamW。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "Schedule-Free优化器"
  - "学习率调度"
  - "River-Valley损失景观"
  - "权重平均"
  - "大规模预训练"
---

# Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training

**会议**: NeurIPS 2025  
**arXiv**: [2507.09846](https://arxiv.org/abs/2507.09846)  
**代码**: 无（基于公开 llm-baselines 代码库）  
**领域**: LLM预训练/优化  
**关键词**: Schedule-Free优化器, 学习率调度, River-Valley损失景观, 权重平均, 大规模预训练

## 一句话总结

从 river-valley 损失景观的几何视角，分析了 Schedule-Free (SF) 优化器在语言模型预训练中不需要学习率衰减和权重平均就能持续追踪最优解的原因，并揭示 SF 隐式执行了权重平均，进而提出解耦动量和平均窗口的改进版 SF-AdamW。

## 研究背景与动机

**核心问题**：随着模型和数据规模的不断增大，传统的固定训练预算策略（如 cosine 学习率调度）越来越不适用。主流替代方案各有局限：

1. **WSD (Warmup-Stable-Decay) 调度**：灵活但需要手动触发衰减阶段才能评估模型质量，在稳定阶段无法判断训练进展
2. **权重平均（Weight Averaging）**：消除了显式衰减的需求，但需要额外的内存存储模型参数副本（如 LLaMA-8B 需额外 16GB）

**关键疑问**：是否存在一种方法，既不需要显式学习率衰减，也不需要额外内存开销，就能持续追踪最优解？

**River-Valley 损失景观**：近期研究发现神经网络的损失景观呈现"河谷"结构——陡峭的"山丘"方向（高曲率）和平坦的"河流"方向（低曲率）。好的优化器应当沿着河流方向前进，同时保持在谷底附近。

## 方法详解

### 整体框架

论文围绕 Schedule-Free AdamW (SF-AdamW) 展开，从理论和实验两个层面分析其优化动力学：

1. **经验观察**：证明 SF-AdamW 能自然地追踪河流，无需衰减或平均
2. **理论分析**：揭示 SF 隐式执行权重平均，并在 Edge of Stability (EoS) 状态下运行
3. **改进方案**：提出解耦参数 C，改善动量敏感性和大 batch 性能

### 关键设计

**1. SF 方法的核心更新规则**

SF 方法维护三个迭代序列 $(x_t, y_t, z_t)$：

$$x_t = (1-c_t) x_{t-1} + c_t z_t$$
$$y_t = (1-\beta) z_t + \beta x_t$$
$$z_{t+1} = z_t - \gamma \Delta_t$$

其中 $c_t = 1/t$，$\gamma$ 是学习率，$\beta$ 是动量参数。梯度在 $y_t$ 处计算，$x_t$ 为输出迭代。

**2. SF 追踪河流的关键发现**

- 对 SF-AdamW 的检查点施加短暂 AdamW 衰减，发现几乎没有额外的损失下降——说明 SF 已经在谷底附近
- 线性插值实验显示：AdamW (constant LR) 呈凸形（横跨谷壁），AdamW (decay) 呈单调下降（从谷壁到谷底），SF-AdamW 呈平坦缓降（已在谷底）

**3. $y_t$ 迭代的核心角色**

通过重新推导 SF 的等价形式，发现 $x_t$ 实际上是过去所有 $y_t$ 的加权平均：

$$x_T = \sum_{t=1}^T \alpha_t y_t$$

这意味着 SF 隐式地执行了权重平均，且不需要额外内存。

### 损失函数 / 训练策略

**Edge of Stability 分析**：

SF-GD 在二次目标函数上的稳定性阈值为 $\lambda_1(H) > \frac{2}{(1-\beta)\gamma}$，比标准 GD 的阈值 $2/\gamma$ 放大了 $(1-\beta)^{-1}$ 倍，解释了 SF 可以使用更大学习率的原因。

**Central Flow 分析**：

在 EoS 状态下，$y_t$ 的时间平均轨迹遵循中心流微分方程。理论推导表明：当 $\beta_1$ 增大时，振荡方差 $\sigma^2$ 减小，即更大的动量参数能抑制山丘方向的振荡，使 $y_t$ 更紧密地追踪河流。

**改进版 SF-AdamW**：

引入解耦参数 $C$，重新定义 $c_t = \frac{(1-\beta)C}{t}$，使得平均权重 $\alpha_t$ 仅依赖于 $C$，$\beta$ 独立控制动量。这解决了原始 SF 中 $\beta$ 同时控制动量和平均窗口的耦合问题。

## 实验关键数据

### 主实验 (含表格)

实验设置：124M 参数 LLaMA 风格模型，SlimPajama-6B 数据集，0.5M token batch size。

| 优化器 | 需要 LR 衰减 | 需要权重平均 | 额外内存 | 最终验证 PPL |
|-------|:-----------:|:---------:|:-------:|:----------:|
| AdamW + constant LR | ✗ (次优) | ✗ (次优) | 无 | 较高 |
| AdamW + cosine LR | ✓ | ✗ | 无 | 基线 |
| AdamW + WSD | ✓ | ✗ | 无 | ≈基线 |
| AdamW + EWA | ✗ | ✓ | +16GB/8B | ≈基线 |
| SF-AdamW | ✗ | ✗ | 无 | ≈基线 |
| Refined SF-AdamW (C=200) | ✗ | ✗ | 无 | **优于基线** |

### 消融实验 (含表格)

**动量敏感性消融**：

| $\beta_1$ | 原始 SF $x_t$ | 原始 SF $y_t$ | EWA of $y_t$ | Refined SF $x_t$ (C=50) |
|:---------:|:------------:|:------------:|:------------:|:----------------------:|
| 0.1 | 差（偏离河流） | 良好 | 更好 | **恢复正常** |
| 0.5 | 差（偏离河流） | 良好 | 更好 | **恢复正常** |
| 0.95 | 最优 | 最优 | 无更多收益 | 进一步提升 (C=200) |

**大 batch 实验** (2M token batch)：

| 方法 | $\beta_1$ | 最终验证损失 |
|-----|:---------:|:----------:|
| AdamW + cosine | 0.9 | 基线 |
| 原始 SF-AdamW | 0.98 | 落后于基线 |
| Refined SF-AdamW (C=200) | 0.98 | **追平基线** |

### 关键发现

1. **SF-AdamW 无需衰减和平均**：与 AdamW 不同，SF-AdamW 在整个训练过程中持续追踪河流，衰减和 EWA 不带来额外收益
2. **$y_t$ 比 $x_t$ 更鲁棒**：即使动量参数不优，$y_t$ 仍然能追踪河流，而 $x_t$ 会偏离
3. **隐式权重平均**：$x_t$ 是 $y_t$ 的加权平均，SF 在无额外内存开销下实现了权重平均效果
4. **EoS 行为**：SF 的 $y_t$ 迭代在训练中自然稳定在 Edge of Stability 阈值附近
5. **解耦改进有效**：引入参数 $C$ 后，SF 在动量鲁棒性和大 batch 性能上均有显著提升

## 亮点与洞察

1. **几何直觉清晰**：通过 river-valley 景观给出了 SF 为何有效的直观解释——它自然沿着河流前进而不振荡到谷壁
2. **理论-实验一致性强**：从 2D 玩具模型到 124M 语言模型，观察到的现象高度一致
3. **实用价值高**：SF 方法无需预设训练预算、无需衰减调度、无需额外内存，是大规模持续训练的理想选择
4. **简洁优雅的改进**：仅引入一个参数 $C$ 就同时解决了动量敏感性和大 batch 两个问题

## 局限与展望

1. **实验规模有限**：仅在 124M 参数模型上验证，未在更大规模（如 7B+）上实验
2. **理论简化假设**：central flow 分析依赖简化假设（如 $c_t = 1/t$ 可忽略），在深度学习中的验证仍待完善
3. **参数 $C$ 的选取**：虽然论文声称对 $C$ 不敏感，但最优值仍需 sweep
4. **与其他优化器的结合**：未探讨 SF 与 AdEMAMix 等新优化器的整合可能
5. **单一架构验证**：主要在 LLaMA 和 GPT-2 风格模型上验证，未涵盖其他架构（如 MoE）

## 相关工作与启发

- **Defazio et al., 2024 (Schedule-Free)**：SF 方法的原始论文，本文在其基础上深入分析动力学
- **Wen et al., 2025 (WSD 的 river-valley 解释)**：提供了 WSD 在 river-valley 视角下的理解
- **Cohen et al., 2021/2025 (Edge of Stability / Central Flow)**：EoS 理论框架，本文将其扩展到 SF 方法
- **Zhang et al., 2025 (Exponential Weight Averaging)**：EWA + constant LR 的分析，本文指出 SF 的隐式平均更优
- 对优化器设计的启示：大规模训练中，**解耦不同功能的超参数**是提高鲁棒性的通用策略

## 评分

⭐⭐⭐⭐⭐ (5/5)

理由：论文在理论深度和实验验证方面都做得非常扎实。从直观的几何解释到严格的数学分析（EoS 稳定性阈值、central flow 推导），再到清晰的实验对比，逻辑链条完整且令人信服。改进版 SF-AdamW 的设计简洁优雅、动机充分。虽然实验规模受限于计算资源，但核心发现具有很强的泛化潜力，为大规模 LLM 预训练的优化器选择提供了重要的理论指导。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](../../ICCV2025/llm_pretraining/syncity_training-free_generation_of_3d_worlds.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](../../ACL2025/llm_pretraining/stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)

</div>

<!-- RELATED:END -->
