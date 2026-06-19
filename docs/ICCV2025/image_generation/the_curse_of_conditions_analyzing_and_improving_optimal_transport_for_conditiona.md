---
title: >-
  [论文解读] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation
description: >-
  [ICCV 2025][图像生成][最优传输] 本文揭示了条件流匹配中使用标准最优传输（OT）会导致训练-测试不匹配的"条件诅咒"问题——OT 忽略条件信息导致训练时先验分布产生条件偏移，而测试时用的是无偏先验，并提出了 C²OT（Conditional Optimal Transport）通过在 OT 代价矩阵中加入条件权重项来修复此问题。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "最优传输"
  - "条件生成"
  - "流匹配"
  - "条件偏移先验"
  - "ODE求解"
---

# The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation

**会议**: ICCV 2025  
**arXiv**: [2503.10636](https://arxiv.org/abs/2503.10636)  
**代码**: [https://github.com/hkchengrex/C2OT](https://github.com/hkchengrex/C2OT)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 最优传输, 条件生成, 流匹配, 条件偏移先验, ODE求解

## 一句话总结

本文揭示了条件流匹配中使用标准最优传输（OT）会导致训练-测试不匹配的"条件诅咒"问题——OT 忽略条件信息导致训练时先验分布产生条件偏移，而测试时用的是无偏先验，并提出了 C²OT（Conditional Optimal Transport）通过在 OT 代价矩阵中加入条件权重项来修复此问题。

## 研究背景与动机

**领域现状**：流匹配（Flow Matching）是近年来与扩散模型并列的主流生成范式。它通过学习从噪声分布到数据分布的向量场来生成样本。在无条件设置下，小批量最优传输耦合（minibatch OT coupling）是一个被广泛使用的技巧——通过 OT 将噪声样本和数据样本配对，能让流的路径更直，从而在测试时只需更少的 ODE 积分步数就能生成高质量样本。

**现有痛点**：当流匹配扩展到条件生成（如类别条件图像生成）时，标准的 minibatch OT 表现不佳，甚至比不用 OT 的独立耦合还差。这个现象在社区中被观察到但缺乏理论解释，也没有有效的解决方案。很多条件生成的工作因此直接放弃了 OT 耦合。

**核心矛盾**：标准 OT 在计算耦合时只基于噪声和数据的距离，完全忽略了条件信息。这导致了一个严重的训练-测试不匹配：在训练时，由于 OT 倾向于让距离近的噪声-数据对配对，某些条件类别可能集中分配到噪声空间的特定区域，造成先验分布随条件而偏移（conditionally skewed prior）。但在测试时，我们没有办法知道这种偏移，只能从标准的无偏高斯先验中采样，导致测试分布与训练分布不匹配。

**本文目标**：(1) 从理论上分析条件 OT 失效的原因；(2) 提出一个简单而有效的修复方案。

**切入角度**：作者从直觉上理解了问题：OT 是"贪心"的，它把空间上靠近的噪声和数据配对，不管它们的条件标签。如果数据集中不同条件的样本在特征空间中有聚类结构，OT 会让不同条件"瓜分"噪声空间，造成先验偏移。解决方案也很直觉——在 OT 代价矩阵中加入条件信息作为惩罚项，让相同条件的样本更优先配对。

**核心 idea**：在 OT 的代价矩阵中加入条件相似性权重——相同条件的噪声-数据对的运输成本更低，不同条件的成本更高——从而避免条件偏移先验问题。

## 方法详解

### 整体框架

C²OT 的修改非常简洁：仅在流匹配训练中计算 minibatch OT 耦合时，将原始的纯距离代价矩阵替换为一个结合了条件信息的加权代价矩阵。整个框架的推理过程与标准流匹配完全相同——从标准高斯先验采样，通过 ODE 求解器积分得到生成样本。

### 关键设计

1. **条件偏移先验分析（理论贡献）**:

    - 功能：解释为什么标准 OT 在条件生成中失效
    - 核心思路：考虑一个简单例子——两个条件类别 A 和 B，数据分布中 A 在左侧、B 在右侧。标准 OT 会将噪声空间左半部分匹配给 A、右半部分匹配给 B。这意味着训练时 A 类的"先验"实际上不是标准高斯，而是偏向左侧的截断高斯。但测试时生成 A 类图像时，我们从完整高斯采样，可能采到"属于 B"的噪声区域，导致生成质量下降。作者通过 8-gaussians-to-moons 玩具实验可视化了这种偏移。
    - 设计动机：为 C²OT 的设计提供理论依据。理解了问题本质才能设计出正确的解决方案。

2. **条件加权代价矩阵 (C²OT)**:

    - 功能：在 OT 配对时考虑条件信息，避免条件间的噪声空间"抢占"
    - 核心思路：在标准的 OT 代价矩阵 $C_{ij} = \|x_i - z_j\|^2$ 基础上，加入条件惩罚项：$C_{ij}^{C^2OT} = \|x_i - z_j\|^2 + \lambda \cdot d(c_i, c_j)$，其中 $c_i, c_j$ 分别是数据 $x_i$ 和噪声 $z_j$ 的条件信息，$d(\cdot, \cdot)$ 是条件距离。对于离散条件（如类别），$d$ 是指示函数（不同类别则加惩罚）；对于连续条件（如时间戳），$d$ 是 L2 距离。$\lambda$ 控制条件约束的强度。
    - 设计动机：加入条件惩罚后，OT 会更倾向于将相同条件的噪声和数据配对。当 $\lambda \to \infty$ 时退化为独立耦合（完全不跨条件配对）；当 $\lambda = 0$ 时退化为标准 OT。C²OT 在两个极端之间找到了平衡点。

3. **适应连续条件的扩展**:

    - 功能：将 C²OT 推广到连续条件设置（如文本嵌入、时间信息等）
    - 核心思路：对于连续条件，不能简单用指示函数，而是用条件嵌入之间的 L2 距离作为 $d(c_i, c_j)$。作者还讨论了条件维度较高时的归一化策略——将条件距离归一化到与空间距离相同的量级，使 $\lambda$ 的调节更稳定。
    - 设计动机：许多实际应用场景的条件是连续的（如文本引导图像生成中的 CLIP 嵌入），因此需要将离散条件的解决方案平滑推广到连续情况。

### 损失函数 / 训练策略

训练损失与标准流匹配完全一致：$L = \mathbb{E}_{t, (x_1, x_0) \sim \pi^{C^2OT}} \|v_\theta(t, x_t, c) - (x_1 - x_0)\|^2$，其中 $\pi^{C^2OT}$ 是 C²OT 计算的耦合，$x_t = (1-t)x_0 + tx_1$ 是线性插值，$v_\theta$ 是速度场网络。唯一的变化在于耦合 $\pi$ 的计算方式。

## 实验关键数据

### 主实验

不同 NFE（Number of Function Evaluations，ODE 求解步数）下的 FID 比较：

| 方法 | CIFAR-10 (NFE=1) | CIFAR-10 (NFE=5) | ImageNet-32 (NFE=5) | ImageNet-256 (NFE=50) |
|------|-----------------|-----------------|-------------------|---------------------|
| 独立耦合 (IC) | 18.72 | 5.21 | 12.34 | 4.85 |
| 标准 OT | 22.15 | 6.03 | 14.67 | 5.42 |
| C²OT (本文) | **15.83** | **4.52** | **10.89** | **4.21** |

8-gaussians-to-moons 玩具实验的 W2 距离（越低越好）：

| NFE | 独立耦合 | 标准 OT | C²OT |
|-----|---------|---------|------|
| 1 | 0.452 | 0.891 | **0.183** |
| 5 | 0.089 | 0.234 | **0.041** |
| 20 | 0.012 | 0.078 | **0.008** |

### 消融实验

| 配置 | CIFAR-10 FID (NFE=5) | 说明 |
|------|---------------------|------|
| C²OT ($\lambda=1.0$) | **4.52** | 最优 $\lambda$ |
| C²OT ($\lambda=0.1$) | 4.98 | 条件约束太弱，接近标准 OT |
| C²OT ($\lambda=10$) | 4.71 | 条件约束太强，接近独立耦合 |
| C²OT ($\lambda=100$) | 5.15 | 基本等于独立耦合 |
| 标准 OT ($\lambda=0$) | 6.03 | 无条件约束 |
| 独立耦合 ($\lambda=\infty$) | 5.21 | 完全按条件配对 |

### 关键发现

- 标准 OT 在条件生成中确实比独立耦合更差，验证了"条件诅咒"的存在。在 CIFAR-10 上 FID 差 0.82，在 ImageNet-32 上差 2.33
- C²OT 以极小的代价（修改几行代价矩阵计算代码）显著改善了 FID，在低 NFE 下改善尤为明显——NFE=1 时 FID 从 18.72 降到 15.83
- $\lambda$ 的最优值约在 0.5-2.0 之间，过大或过小都不好，说明需要在 OT 的路径拉直效果和条件一致性之间取平衡
- 在连续条件设置下 C²OT 同样有效，说明方法具有通用性
- 条件数量越多（类别数越多），标准 OT 的偏移问题越严重，C²OT 的改善也越明显

## 亮点与洞察

- **问题分析极为清晰**：从玩具实验到理论分析再到大规模实验，层层递进地解释了"为什么条件 OT 不好用"。这种先诊断后治疗的研究范式值得学习。
- **修复方案极其简洁**：只需在 OT 代价矩阵中加一个条件惩罚项，几行代码就能实现。简洁性意味着低实现门槛、可复现性好、容易被社区采用。
- **填补了理论空白**：社区早已观察到条件 OT 不如无条件 OT 好用，但缺乏理论解释。本文提供了清晰的解释和解决方案，对流匹配领域有重要的理论贡献。

## 局限与展望

- $\lambda$ 需要手动调节，不同数据集可能需要不同的最优值。虽然范围不大（0.5-2.0），但自适应选择 $\lambda$ 会更好
- 目前只验证了类别条件和简单连续条件，对于更复杂的条件（如自然语言描述的 CLIP 嵌入），效果尚需验证
- 在非常高的 NFE（如 100+）下，C²OT 与标准 OT 和独立耦合的差距缩小，说明主要优势在低 NFE 下的快速采样场景
- OT 的计算成本本身也是一个考量——在大 batch size 下 OT 求解的三次复杂度可能成为瓶颈

## 相关工作与启发

- **vs 标准 OT-FM (Tong et al., 2024)**: 标准 OT 流匹配在无条件设置下很有效，但在条件设置下有"条件诅咒"。C²OT 是对标准 OT-FM 在条件场景下的直接修复。
- **vs 独立耦合 (IC)**: IC 完全不做 OT 配对，路径不直但没有条件偏移问题。C²OT 比 IC 好是因为它保留了 OT 的路径拉直效果，同时避免了条件偏移。
- **vs Multi-sample Flow Matching**: 其他一些工作尝试用多样本策略来改善 OT 耦合，但没有从条件偏移的角度分析问题。C²OT 直接解决了根本原因。

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题分析新颖且深刻，解决方案虽简单但切中要害
- 实验充分度: ⭐⭐⭐⭐⭐ 从玩具实验到 ImageNet-256 全面覆盖，消融细致
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰，从问题分析到解决方案一气呵成，是教科书级的研究论文写法
- 价值: ⭐⭐⭐⭐ 对流匹配社区有重要理论和实践价值，但影响范围限于用 OT 耦合的条件生成

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](../../NeurIPS2025/image_generation/counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[ICLR 2026\] AlignFlow: Improving Flow-based Generative Models with Semi-Discrete Optimal Transport](../../ICLR2026/image_generation/alignflow_improving_flow-based_generative_models_with_semi-discrete_optimal_tran.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](../../NeurIPS2025/image_generation/pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)

</div>

<!-- RELATED:END -->
