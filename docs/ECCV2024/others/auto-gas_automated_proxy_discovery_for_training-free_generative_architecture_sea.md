---
title: >-
  [论文解读] Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search
description: >-
  [ECCV 2024][生成架构搜索] 本文提出 Auto-GAS，首个面向生成模型（GAN）的免训练架构搜索框架，通过自动发现并优化零成本代理指标来替代传统训练式搜索，实现 110 倍搜索加速，同时保持与训练式方法相当的生成质量。 领域现状：生成对抗网络（GAN）在实时图像生成、图像翻译等场景广泛应用…
tags:
  - "ECCV 2024"
  - "生成架构搜索"
  - "免训练代理"
  - "进化算法"
  - "GAN"
  - "零成本代理"
---

# Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search

**会议**: ECCV 2024  
**代码**: [https://github.com/lliai/Auto-GAS](https://github.com/lliai/Auto-GAS)  
**领域**: 其他  
**关键词**: 生成架构搜索, 免训练代理, 进化算法, GAN压缩, 零成本代理

## 一句话总结

本文提出 Auto-GAS，首个面向生成模型（GAN）的免训练架构搜索框架，通过自动发现并优化零成本代理指标来替代传统训练式搜索，实现 110 倍搜索加速，同时保持与训练式方法相当的生成质量。

## 研究背景与动机

**领域现状**：生成对抗网络（GAN）在实时图像生成、图像翻译等场景广泛应用，但标准 GAN 生成器的推理速度和内存消耗限制了部署效率。生成架构搜索（Generative Architecture Search, GAS）旨在自动寻找最优的 GAN 生成器结构，以实现速度与质量的最佳平衡。现有 GAS 方法主要采用可微分搜索或进化搜索策略，但都需要对候选架构进行完整训练来评估性能。

**现有痛点**：训练式 GAS 方法（如 GAN Compression、EAGAN）搜索代价极为高昂：每个候选架构都需经历完整的训练-评估循环，导致搜索时间通常以 GPU 天为单位计算。这严重限制了 GAS 技术在实际中的采用。一个自然的改进思路是引入免训练搜索（training-free search），即使用零成本代理指标（zero-cost proxy）来快速评估候选架构的潜力，无需训练即可筛选出优质架构。

**核心矛盾**：然而，现有的零成本代理指标（如 SynFlow、NASWOT 等）都是为分类任务设计的，在生成任务上的预测能力显著下降。分类和生成任务的评估标准本质不同——分类关注判别能力，生成关注分布匹配和图像质量——因此需要专门为生成任务设计代理指标，但人工设计这类指标既费时又缺乏理论指导。

**本文目标** (1) 如何自动发现适用于 GAN 生成任务的零成本代理指标？(2) 如何确保发现的代理指标具有足够的预测能力来可靠地指导架构搜索？(3) 如何在显著加速搜索的同时保持生成质量？

**切入角度**：作者观察到，虽然单一手工设计的代理指标在生成任务上效果不佳，但通过组合和变换多种特征统计量（feature statistics），可以构造出预测能力更强的复合代理。因此，可以将代理指标的设计本身视为一个搜索问题，用进化算法来自动发现最优代理。

**核心 idea**：将代理指标的构造本身建模为一个搜索问题，通过进化算法在由特征统计量、变换操作和编码操作组成的搜索空间中自动发现高预测力的生成架构评估代理。

## 方法详解

### 整体框架

Auto-GAS 的整体流程分为两个阶段：(1) **代理发现阶段**——构造一个代理搜索空间，使用进化算法在其中搜索最优的零成本代理指标；(2) **架构搜索阶段**——利用发现的代理指标对 GAN 生成器候选架构进行快速评估和排序，无需训练即可找到最优架构。输入为 GAN 搜索空间定义和少量参考数据，输出为最优生成器架构配置。

### 关键设计

1. **信息感知代理构造（Information-Aware Proxy Construction）**:

    - 功能：从候选 GAN 生成器中提取多种特征统计量，构造代理指标的基本输入
    - 核心思路：不依赖单一指标，而是以网络各层的特征图统计信息（如均值、方差、梯度信息等）作为原始输入。然后定义四类操作——变换（transform）、编码（encoding）、降维（reduction）和增强（augment）——对这些统计量进行组合处理。每个候选代理由一系列操作的组合链表示，类似于一个"代理程序"。这样构成了一个丰富的代理搜索空间，能够表达多样化的候选代理指标。
    - 设计动机：单个统计量（如仅用 Jacobian 范数或 Fisher 信息）无法全面刻画生成能力，但不同统计量的组合与变换可以捕捉更丰富的架构特性。这种设计将代理发现转化为一个组合优化问题，适合用进化算法求解。

2. **进化代理搜索（Evolutionary Proxy Search）**:

    - 功能：在代理搜索空间中自动搜索预测力最强的代理指标
    - 核心思路：初始化一个由随机代理组成的种群，对每个代理计算其与真实架构排名的相关性（使用 Kendall tau 或 Spearman 相关系数）。根据相关性评分对种群进行选择，对优秀个体执行交叉（crossover）和变异（mutation）操作，产生下一代候选代理。经过多轮进化迭代，种群中的代理逐步收敛到高预测力的最优代理。为了计算真实排名作为监督信号，只需对一个小规模子集的候选架构进行训练评价，开销可控。
    - 设计动机：进化算法天然适合处理这类离散组合搜索空间，且不需要代理搜索空间可微。通过在小子集上预计算真实排名来提供评估信号，巧妙地平衡了搜索成本和代理质量。

3. **免训练生成器搜索（Training-Free Generator Search）**:

    - 功能：利用优化后的代理指标，快速评估大量候选生成器架构并选出最优
    - 核心思路：给定 GAN 搜索空间（包括不同的层数、通道数、卷积核大小等配置），对每个候选生成器仅需一次前向传播即可通过代理指标给出评分。然后按评分排序，选择得分最高的架构进行完整训练验证。整个搜索过程无需任何训练，单个架构的评估时间从数小时降至数秒。
    - 设计动机：AutoGAS 的核心优势就在于将搜索阶段的计算开销从训练级降至推理级，使得大规模架构搜索成为可能。

### 损失函数 / 训练策略

代理发现阶段的优化目标为最大化代理评分与真实架构排名之间的秩相关系数（Kendall tau / Spearman correlation）。最终选出的最优架构仍采用标准 GAN 训练流程（对抗损失 + 感知损失等）进行训练，训练策略与基线方法保持一致，确保公平比较。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (Auto-GAS) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| CIFAR-10 | FID ↓ | 与 GAN Compression 相当 | GAN Compression | 搜索速度 110× 提升 |
| 图像翻译 (pix2pix) | FID ↓ | 竞争性结果 | EAGAN | 搜索效率大幅优于训练式方法 |
| 图像翻译 (CycleGAN) | FID ↓ | 竞争性结果 | 训练式 GAS | 搜索时间从 GPU 天降至 GPU 分钟 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 现有分类代理（SynFlow等） | 低秩相关 | 设计用于分类的代理在生成任务上预测力差 |
| 单统计量代理 | 中等秩相关 | 单一特征统计量能力有限 |
| 组合代理（无进化优化） | 较高秩相关 | 手工组合有一定效果但不足 |
| Auto-GAS（完整进化搜索） | 最高秩相关 | 自动发现的代理显著优于所有手工设计代理 |

### 关键发现

- 为分类任务设计的零成本代理（如 SynFlow、NASWOT）在 GAN 生成任务上的秩相关系数显著低于自动发现的代理
- Auto-GAS 发现的代理在不同 GAN 搜索空间和数据集间具有一定的迁移能力
- 搜索加速比可达 110 倍以上（相比 GAN Compression），同时生成质量保持竞争力
- 进化代理搜索过程通常在数百轮内收敛，整体代理发现的开销远小于一次完整的训练式搜索

## 亮点与洞察

- **将代理设计自动化**：首次将"如何评估架构好坏"这个问题本身自动化，是 NAS/GAS 领域的一个重要范式创新
- **跨任务代理差异**的深入分析：明确揭示了分类代理不适用于生成任务的原因，为未来的 task-specific proxy 设计提供了方向
- **搜索空间设计巧妙**：通过 transform-encoding-reduction-augment 操作链构造代理搜索空间，表达能力强且搜索高效
- **实用价值突出**：110 倍的搜索加速使得 GAS 从研究级工具走向实际可用

## 局限与展望

- 代理发现仍需依赖小子集的完整训练来提供真实排名标签，无法完全避免训练开销
- 发现的代理指标的可解释性较弱，难以从中提取对生成模型设计的直觉认识
- 主要在 GAN 上验证，是否适用于其他生成模型（如扩散模型、VAE）尚未探索
- 搜索空间目前限于生成器结构，判别器的联合搜索可能带来进一步提升
- 代理的跨数据集迁移能力虽然有但有限，面对分布差异较大的任务可能需要重新搜索

## 相关工作与启发

- **GAN Compression**（Zhu et al.）: 通道剪枝方式的 GAN 加速，需要完整训练，是 Auto-GAS 的主要基线
- **EAGAN**（ICLR 2024）: 进化式 GAN 架构搜索，搜索效率较高但仍需训练
- **Zero-Cost Proxy for NAS**（SynFlow, NASWOT 等）: 免训练 NAS 代理，但限于分类任务
- **TransNASBench**: 跨任务 NAS 基准，启发了本文跨任务代理分析的思路

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在 GAS 中自动发现代理指标，思路新颖
- 实验充分度: ⭐⭐⭐ 多任务多数据集评估，消融实验较完整，但缺少更多定量对比
- 写作质量: ⭐⭐⭐ 整体清晰，代理搜索空间的描述较抽象
- 价值: ⭐⭐⭐⭐ 110 倍加速的免训练 GAS 有显著实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](attnzero_efficient_attention_discovery_for_vision_transformers.md)
- [\[ICLR 2026\] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](../../ICLR2026/others/enhancing_generative_auto_bidding.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](../../CVPR2025/others/training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](../../CVPR2025/others/subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[CVPR 2026\] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search](../../CVPR2026/others/intrain_intrinsic_trainability_for_zero-cost_neural_architecture_search.md)

</div>

<!-- RELATED:END -->
