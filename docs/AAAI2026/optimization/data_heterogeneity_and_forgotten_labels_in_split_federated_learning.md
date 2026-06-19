---
title: >-
  [论文解读] Data Heterogeneity and Forgotten Labels in Split Federated Learning
description: >-
  [AAAI 2026][优化/理论][联邦学习] 系统研究了 Split Federated Learning 中数据异构导致的灾难性遗忘现象（尤其是 server 端处理顺序造成的 intra-round 遗忘），并提出基于 multi-head 的 Hydra 方法，将 part-2 的最后层分组训练再聚合，显著降低标签间性能差距（PG 最高降低 75.4%）。
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "联邦学习"
  - "灾难性遗忘"
  - "数据异构"
  - "multi-head"
  - "处理顺序"
---

# Data Heterogeneity and Forgotten Labels in Split Federated Learning

**会议**: AAAI 2026  
**arXiv**: [2511.09736](https://arxiv.org/abs/2511.09736)  
**代码**: [GitHub](https://github.com/jtirana98/Hydra-CF-in-SFL)  
**领域**: 优化  
**关键词**: Split Federated Learning, 灾难性遗忘, 数据异构, multi-head, 处理顺序  

## 一句话总结
系统研究了 Split Federated Learning 中数据异构导致的灾难性遗忘现象（尤其是 server 端处理顺序造成的 intra-round 遗忘），并提出基于 multi-head 的 Hydra 方法，将 part-2 的最后层分组训练再聚合，显著降低标签间性能差距（PG 最高降低 75.4%）。

## 背景与动机

### 领域现状

**领域现状**：Split Federated Learning (SFL/SplitFedv2) 将模型分为 part-1（client 端）和 part-2（server 端）。Part-1 像标准 FL 一样在各 client 并行训练后聚合；Part-2 则由 server 按顺序依次处理各 client 的激活值进行训练。

作者发现 SFL 中存在两种灾难性遗忘 (CF)：(1) Part-1 的 **跨轮遗忘**：与 FL 相同，聚合导致模型偏移；(2) Part-2 的 **轮内遗忘 (intraCF)**：由 server 端顺序处理引起，类似持续学习，模型倾向于"记住"最后处理的 client 数据对应的标签，遗忘前面的。实验显示在 CIFAR-10 non-IID 设置下，最后处理的标签精度远高于最先处理的，全局精度仅 43%（IID 下达 80%）。

现有 CL 和 FL 的遗忘缓解方法（如 EWC、Scaffold）因假设不兼容而无法直接应用于 SFL：EWC 基于时序数据假设，Scaffold 需要完整模型。

### 解决思路

**本文目标**：SFL 中 server 端顺序处理导致的 intra-round 灾难性遗忘如何表征和缓解？cut layer 和处理顺序如何影响遗忘程度？

## 方法详解

### 整体框架
Hydra 在 SFL 的 server 端引入 multi-head 架构：将 part-2 进一步分为共享的 part-2a 和多个 head（part-2b）。Part-2a 按顺序更新（保持原 SFL 特性），而各 head 按 client 数据分布分组并行更新，轮末聚合。

### 关键设计
**Client 分组**: 训练前每个 client 发送其标签分布向量（长度 $L$），通过贪心算法将标签分布相似的 client 分入同一组。共 $G$ 个组（head），每组内数据分布较均匀，减小梯度冲突。

**Head 设计**: 每个 head 对应 part-2 的最后若干层。Server 在前传时根据 mapping 将 part-2a 的输出分发到对应 head，反传时反向路由。轮末所有 head 通过 FedAvg 聚合为单一模型——与 CL 中保留多个 head 不同，Hydra 最终输出单 head 模型。

**处理顺序分析**: 作者定义了 cyclic order（按主导标签循环）和 random order 两种方式，发现：
- Cyclic order 整体优于 random order，结构化处理更稳定
- 较小的 $\phi$（同类 client 连续处理数）效果更好，印证了训练序列多样性的重要性

**Cut Layer 影响**: 浅 cut layer → part-2 更大 → intraCF 效应更明显；深 cut layer → part-1 更大 → 类似 FL 的跨轮遗忘更显著。实验证实 intraCF 的危害大于 part-1 的模型偏移。

### 遗忘度量
- **Performance Gap (PG)**: $\mathcal{PG}(r) = \frac{1}{L}\sum_{l=1}^{L}\max_{k}|min(0, A_l^r - A_k^r)|$，衡量标签间精度差距
- **Backward Transfer (BW)**: 衡量峰值精度与最终精度的差异

## 实验关键数据


### 主实验

| 方法 | 全局精度↑ | PG↓ | BW↓ |
|------|:---:|:---:|:---:|
| SFL (baseline) | 43 | 43.4 | 28 |
| SplitFedV1/FL | 55 | 15.7 | 14 |
| MergeSFL | 65 | 21 | 13 |
| MultiHead | 47 | 36.6 | 25 |
| **SFL+Hydra** | **66.5** | **13.4** | **9** |

（MobileNet, CIFAR-10, 80%-DL）

Hydra 效果：
- PG 最高降低 **75.4%**（ResNet101），精度提升最高 **112.2%**
- 在 Sharding、Dirichlet($\alpha$=0.1/0.3) 等不同数据划分下均有效
- 小 head（仅最后 2 层）效果优于大 head，且内存开销仅 $G \times 6$MB

## 亮点与洞察
- 首次系统揭示 SFL 中 intra-round 灾难性遗忘现象，区分于 FL 的跨轮遗忘
- 实验极为扎实：2 模型 × 4 数据集 × 多种数据划分 × 多种处理顺序，每组 ≥10 次重复取中位数
- Hydra 设计巧妙，利用 SFL 的天然分割结构，client 端零开销
- 对 CL 理论在 SFL 中的适用性做了详细验证和对比

## 局限与展望
- Client 分组依赖训练前共享标签分布，可能在隐私敏感场景受限
- Head 数量 $G$ 需预设，自适应分组策略有待探索
- 仅考虑分类任务，生成/回归任务的遗忘模式可能不同
- 未与 SMoFi 等 momentum-based 方法结合，两者可能互补

## 相关工作与启发
- **FL 遗忘方法（EWC/Scaffold）**: 需要完整模型或时序数据假设，不适用 SFL
- **SplitFedV1**: 通过 server 端每 client 一份模型副本避免顺序问题，但等同于 FL，未利用 split 特性
- **MergeSFL**: 合并特征+调整 batch size，精度可达 65% 但 PG 仍为 21，Hydra PG 降至 13.4
- **传统 multi-head (CL)**: 保留多个 head 各自推理，Hydra 聚合为单 head，更适合 SFL 场景

## 相关工作与启发
- IntraCF 的发现对 SFL 研究有重要指导意义：server 端处理策略不容忽视，随机顺序可能带来不稳定性
- Hydra 的 multi-head + 聚合思路可推广到异构联邦学习的 personalization 层面
- 与 SMoFi 互补性强：SMoFi 解决 SFLV1 的 momentum 分歧，Hydra 解决 SFLV2 的顺序遗忘，可探索二者结合

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICML 2026\] Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](../../ICML2026/optimization/learning_locally_revising_globally_global_reviser_for_federated_learning_with_no.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](../../NeurIPS2025/optimization/streaming_federated_learning_with_markovian_data.md)

</div>

<!-- RELATED:END -->
