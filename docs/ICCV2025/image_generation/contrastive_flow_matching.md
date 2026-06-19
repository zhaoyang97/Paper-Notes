---
title: >-
  [论文解读] Contrastive Flow Matching (ΔFM)
description: >-
  [ICCV 2025][图像生成][Flow Matching] 在 Flow Matching 的训练目标中引入对比正则项，强制不同条件的流场互相远离，从而在零额外推理开销下实现 9× 训练加速、5× 更少采样步数、FID 最多降低 8.9。 Flow Matching（FM）是当前生成模型的主流训练范式之一…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Flow Matching"
  - "对比学习"
  - "条件生成"
  - "训练加速"
---

# Contrastive Flow Matching (ΔFM)

**会议**: ICCV 2025  
**arXiv**: [2506.05350](https://arxiv.org/abs/2506.05350)  
**代码**: [https://github.com/gstoica27/DeltaFM.git](https://github.com/gstoica27/DeltaFM.git)  
**领域**: 扩散模型  
**关键词**: Flow Matching, 对比学习, 条件生成, 图像生成, 训练加速

## 一句话总结

在 Flow Matching 的训练目标中引入对比正则项，强制不同条件的流场互相远离，从而在零额外推理开销下实现 9× 训练加速、5× 更少采样步数、FID 最多降低 8.9。

## 研究背景与动机

Flow Matching（FM）是当前生成模型的主流训练范式之一，其核心思想是学习一个速度场将噪声分布映射到数据分布。在无条件设定下，ODE 流具有天然的唯一性保证——不同初始点的流轨迹不会交叉。然而，当引入条件信息（如类别标签）后，**不同条件对应的流场可能在中间状态大量重叠**，这一性质被打破了。

这种重叠带来两个实际问题：

**生成模糊性**：模型在中间时间步无法有效区分不同类别的去噪方向，导致生成结果呈现"平均化"倾向，类别特征不鲜明

**训练效率低下**：模型需要更多的训练迭代才能学会在高度重叠的流场中做出正确的区分

现有方案各有不足：Classifier-Free Guidance（CFG）虽然能改善条件一致性，但推理时需要两次前向传播，计算成本翻倍；REPA 通过对齐外部预训练编码器的表征来改善生成质量，但引入了对额外模型的依赖。作者因此提出一个自然的问题：**能否在训练阶段就直接鼓励不同条件的流场分离，从根本上解决流重叠问题？**

## 方法详解

### 整体框架

ΔFM 的设计哲学极其简洁：在标准 FM 的损失函数上添加一个对比正则项，无需修改任何网络架构，也不增加推理时的额外计算。整个方法可以看作是将对比学习中"拉近正样本、推远负样本"的思想移植到了流场学习中。

具体而言，标准 FM 要求模型预测的速度场尽量接近真实流的方向（正样本匹配），而 ΔFM 额外要求模型预测的速度场同时远离其他样本的流方向（负样本排斥）。

### 关键设计

**对比流匹配损失**（公式 6）：

$$\mathcal{L}^{(\Delta\text{FM})}(\theta) = \mathbb{E}\left[\|v_\theta(x_t, t, y) - (\dot{\alpha}_t \hat{x} + \dot{\sigma}_t \varepsilon)\|^2 - \lambda\|v_\theta(x_t, t, y) - (\dot{\alpha}_t \tilde{x} + \dot{\sigma}_t \tilde{\varepsilon})\|^2\right]$$

- 第一项是标准 FM 回归损失，让模型拟合当前样本 $(\hat{x}, \varepsilon)$ 对应的目标速度
- 第二项是对比项，$(\tilde{x}, \tilde{\varepsilon})$ 来自同一 batch 中随机选取的另一个样本。减去该项意味着最大化模型预测与"错误目标"之间的距离
- $\lambda$ 控制对比强度，实验表明 $\lambda = 0.05$ 在所有设定下最优

**负样本构造**非常高效：直接从当前 batch 中随机抽取另一个样本作为负例，不需要维护额外的 memory bank 或 momentum encoder，也不需要额外的前向传播。这使得每步训练的额外开销几乎可以忽略。

**与 CFG 的兼容性设计**：作者推导了 ΔFM 效果的闭式表达，发现其与 CFG 可能产生冲突（二者都试图增强条件信号，但方式不同）。为此提出修改版 CFG 公式 $\text{CFG}^{\wedge}$，使两者能够协同工作而非互相干扰。

**泛化性**：当 $\lambda = 0$ 时，ΔFM 退化为标准 FM，因此它是 FM 的严格泛化。

### 损失函数 / 训练策略

训练策略保持与标准 FM 完全一致，唯一变化是损失函数。每个训练步骤中：
1. 采样一个 mini-batch 的 $(x, y)$ 对
2. 对每个样本进行正常的噪声注入得到 $x_t$
3. 计算标准 FM 损失
4. 随机配对 batch 内样本构造负例，计算对比损失
5. 以 $\lambda = 0.05$ 的权重组合两部分

这种即插即用的特性意味着任何使用 FM 训练的模型都可以零成本地切换到 ΔFM。

## 实验关键数据

### 主实验

所有实验在 ImageNet-1k 上进行，使用 SiT 系列模型。

| 模型 | 分辨率 | FM (FID↓) | ΔFM (FID↓) | 改进 |
|------|---------|-----------|-------------|------|
| SiT-B/2 | 256×256 | 42.28 | 33.39 | -8.89 |
| SiT-XL/2 | 256×256 | 20.01 | 16.32 | -3.69 |
| SiT-B/2 | 512×512 | — | — | 类似提升 |
| REPA SiT-XL/2 | 256×256 | 11.14 | 7.29 | -3.85 |
| REPA SiT-XL/2 | 512×512 | 11.32 | 7.64 | -3.68 |

文本到图像生成（CC3M 数据集，MMDiT 架构）同样有效：FID 从 24 降至 19（-5）。

与 CFG 叠加使用时，FID 可进一步从 2.09 降至 1.97，证明 ΔFM 与 CFG 互补。

### 消融实验

**λ 值消融**（SiT-XL/2 + REPA，256×256）：

| λ | FID↓ | IS↑ |
|---|------|-----|
| 0（标准 FM） | 11.14 | — |
| 0.01 | ~8.5 | ~120 |
| **0.05** | **7.29** | **129.89** |
| 0.1 | ~8.0 | ~125 |
| 过大 | 退化 | 退化 |

$\lambda$ 过大会导致模型过度关注推远负样本而忽视正确拟合，产生退化行为。$\lambda = 0.05$ 在所有实验设定中都是最优选择，显示出良好的鲁棒性。

**Batch Size 影响**：更大的 batch size 带来更多的负样本选择，提升更加稳定和显著，但即使在较小 batch size 下也能获得一致的增益。

### 关键发现

1. **训练加速 9×**：ΔFM 达到同等 FID 所需的训练步数仅为标准 FM 的约 1/9，这是最令人印象深刻的结果之一
2. **推理加速 5×**：由于流场更加清晰，模型在更少的去噪步数下就能生成高质量结果
3. **更早的类别分化**：可视化（Fig 4）显示 ΔFM 训练的模型在去噪早期阶段就开始进行类别区分，而标准 FM 需要到后期才能逐渐分化
4. **玩具实验验证**（Fig 3）：二维流场可视化清晰展示了 ΔFM 如何将不同条件的流轨迹推向不同区域

## 亮点与洞察

1. **极致的简洁性**：整个方法只改一行损失函数，却带来全方位的改进（FID、训练速度、推理速度），这体现了好的研究直觉——找到问题的本质原因（流重叠）并用最简单的方式解决

2. **推理零开销**：与 CFG 需要两次前向传播不同，ΔFM 的改进完全发生在训练阶段，推理时没有任何额外成本。这在实际部署中极具价值

3. **理论与实践的良好对齐**：从无条件 FM 的流唯一性出发，发现条件 FM 丧失了这一性质，然后用对比学习恢复它——逻辑链条清晰完整

4. **与现有方法的互补性**：ΔFM 可以与 REPA、CFG 叠加使用，分别获得额外增益，说明它捕获的是一个正交维度的改进

5. **对比学习在生成模型中的新应用**：将判别式学习中的核心思想（对比）创造性地应用到生成式训练中，且不引入判别器

## 局限与展望

1. **负样本质量**：当前采用 batch 内随机采样作为负例，这种策略可能不是最优的。探索 hard negative mining 或基于类别关系的负样本选择策略可能带来进一步提升

2. **λ 的自适应调整**：固定 $\lambda = 0.05$ 虽然足够鲁棒，但不同训练阶段可能需要不同的对比强度（早期更大以快速分离，后期减小以精细拟合）。设计 $\lambda$ 的 schedule 可能有收益

3. **多条件场景的扩展**：论文主要验证了类别条件和文本条件，对于更复杂的多条件组合（如类别+风格+布局）场景的效果有待探索

4. **大规模验证不足**：文本到图像实验仅在 CC3M 上进行，未在更大规模数据集（如 LAION）或更大模型（如 Stable Diffusion 级别）上验证

5. **理论分析可以更深入**：虽然直觉清晰，但缺乏对 ΔFM 收敛性的严格理论保证，以及对比项如何影响学到的流场分布的理论分析

## 相关工作与启发

- **Flow Matching (Lipman et al., 2023)**：ΔFM 的基础框架，本文直接在其损失函数上扩展
- **REPA (Yu et al., 2024)**：通过对齐预训练表征改善生成质量，ΔFM 与其正交互补
- **Classifier-Free Guidance**：推理时增强条件信号的标准方法，ΔFM 在训练时实现类似效果但无推理开销
- **对比学习 (SimCLR, MoCo)**：ΔFM 借鉴了对比学习的核心思想，但应用场景和损失形式完全不同
- **SiT (Scalable Interpolant Transformers)**：本文的主要实验骨架网络

**启发**：这项工作提示我们，生成模型训练中可能存在大量类似的"隐性冗余"——不同条件的学习信号互相干扰。对比正则化只是解决这一问题的一种方式，或许还有其他信息论或几何视角的解法值得探索。

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 将对比学习引入流匹配训练，想法自然而有效 |
| 技术深度 | 3.5 | 方法简洁但理论分析偏浅，CFG 兼容性推导有价值 |
| 实验充分性 | 4 | 多模型、多分辨率、多任务验证，消融完整 |
| 实用价值 | 5 | 即插即用、零推理开销、显著提升，工业界直接可用 |
| **总分** | **4** | 简洁高效的改进，实用性极强 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/image_generation/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [\[ICML 2025\] Gaussian Mixture Flow Matching Models](../../ICML2025/image_generation/gaussian_mixture_flow_matching_models.md)
- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](../../ICML2025/image_generation/continualflow_learning_and_unlearning_with_neural_flow_matching.md)

</div>

<!-- RELATED:END -->
