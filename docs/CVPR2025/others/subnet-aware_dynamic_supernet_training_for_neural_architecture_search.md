---
title: >-
  [论文解读] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search
description: >-
  [CVPR 2025][神经架构搜索] 提出动态超网训练策略（CaLR + MS），通过复杂度感知的学习率调度解决子网训练不公平问题，以及动量分离技术缓解梯度噪声问题，以极低额外开销显著提升 N-shot NAS 的搜索性能。 N-shot NAS 方法使用超网（supernet）包含所有候选子网，通过训练超网来预测子网性能…
tags:
  - "CVPR 2025"
  - "神经架构搜索"
  - "超网训练"
  - "学习率调度"
  - "动量分离"
  - "子网公平性"
---

# Subnet-Aware Dynamic Supernet Training for Neural Architecture Search

**会议**: CVPR 2025  
**arXiv**: [2503.10740](https://arxiv.org/abs/2503.10740)  
**代码**: [项目页面](https://cvlab.yonsei.ac.kr/projects/DYNAS/)  
**领域**: 其他  
**关键词**: 神经架构搜索, 超网训练, 学习率调度, 动量分离, 子网公平性

## 一句话总结

提出动态超网训练策略（CaLR + MS），通过复杂度感知的学习率调度解决子网训练不公平问题，以及动量分离技术缓解梯度噪声问题，以极低额外开销显著提升 N-shot NAS 的搜索性能。

## 研究背景与动机

N-shot NAS 方法使用超网（supernet）包含所有候选子网，通过训练超网来预测子网性能。现有方法使用静态训练策略（所有子网共享相同学习率调度器和优化器），忽略了不同子网的特性差异，导致两个关键问题：

(1) **不公平性问题（Unfairness）**: 高复杂度子网参数更多，需要更多训练迭代才能充分收敛，但静态策略对所有子网等同对待。结果是低复杂度子网过早收敛排名靠前，高复杂度子网训练不充分排名被低估，即使其真实性能更强。

(2) **噪声动量问题（Noisy Momentum）**: 超网中随机采样的子网在每步产生差异大的梯度，这些梯度累积到单一动量缓存中导致动量方向噪声大，训练不稳定。

**核心问题**: 现有方法未考虑子网的个体特性（复杂度和结构），导致超网排名一致性差，进而搜索到次优架构。

## 方法详解

### 整体框架

动态超网训练框架包含两个即插即用组件：(1) 复杂度感知学习率调度器（CaLR）——根据子网复杂度调整 LR 衰减速率；(2) 动量分离（MS）——将结构相似的子网分组，为每组分配独立动量缓存。两者互补，可应用于 SPOS、FairNAS、FSNAS 等多种 NAS 方法。

### 关键设计1: 复杂度感知学习率调度器 (CaLR)

**功能**: 根据子网复杂度调整 LR 衰减速率，平衡不同复杂度子网的训练充分性。

**核心思路**: 使用多项式 LR 调度器 $\eta^t = \eta^0 \cdot (1 - t/T)^{\gamma(\alpha)}$，其中衰减率 $\gamma(\alpha) = \omega \log(\mathcal{C}(\alpha)) + \tau$。$\mathcal{C}(\alpha)$ 为子网参数量。高复杂度子网 $\gamma < 1$（LR 衰减慢，保持更大学习率更长时间），低复杂度子网 $\gamma > 1$（LR 衰减快，避免过训练）。复杂度中等的子网 $\gamma = 1$，等价于线性衰减。

**设计动机**: 高复杂度子网有更多参数需要调整，需要更多有效的训练迭代。直接增加训练步数计算成本高，而调整 LR 衰减速率可以等效地为高复杂度子网提供更多的参数空间探索机会。对数函数确保中等复杂度子网使用标准线性衰减。

### 关键设计2: 动量分离 (MS)

**功能**: 降低超网训练中动量的噪声，稳定训练过程。

**核心思路**: 根据子网在特定边/层上的操作类型进行聚类：$S_i = \{\alpha \in \mathcal{A} | \text{op}(\alpha, e) = o_i\}$。为每个聚类分配独立的动量缓存 $\mu_i$：$\mu_i^t = \beta \cdot \mu_i^{t-1} + g^t$。采样到的子网 $\alpha$ 根据其所属聚类 $S_i$ 更新对应的动量缓存 $\mu_i$。权重仍然在所有子网间共享。

**设计动机**: 基于经验观察——结构相似的子网产生相似的梯度。通过将结构相似的子网聚在一组，组内梯度一致性高，动量更新更稳定。聚类基于单边/层操作类型，分组数等于候选操作数（如7组），动量缓存额外内存可忽略。

### 关键设计3: 评估指标 CB 和 C3

**功能**: 量化超网训练中的不公平性问题。

**核心思路**: 提出 Complexity Bias (CB) 衡量超网排名对低复杂度子网的偏好程度，Complexity-Convergence Correlation (C3) 衡量复杂度与收敛程度的相关性。这两个指标可以直接检测不公平性问题，验证 CaLR 的有效性。

**设计动机**: 之前缺乏量化不公平性问题的指标，仅能通过最终搜索结果间接评估。CB 和 C3 提供了直接的诊断工具。

### 损失函数

使用标准的任务训练损失（如交叉熵），不引入额外损失。CaLR 和 MS 仅修改优化过程（LR 调度和动量更新），不影响损失函数设计。

## 实验关键数据

### ImageNet MobileNet 搜索空间

| 方法 | Params(M) | FLOPs(M) | Top-1(%) | GPU Hours |
|------|-----------|----------|----------|-----------|
| SPOS-L | 4.5 | 471 | 76.6 | 157 |
| **SPOS-L + Ours** | **4.7** | **459** | **77.1** | **159** |
| FairNAS-L | 4.7 | 472 | 76.7 | 364 |
| **FairNAS-L + Ours** | **4.7** | **471** | **77.0** | **369** |
| FSNAS-L | 4.7 | 464 | 76.8 | 740 |
| **FSNAS-L + Ours** | **4.5** | — | **提升** | — |

### NAS-Bench-201 排名一致性 (Kendall's Tau)

| 方法 | CIFAR-10 | CIFAR-100 | ImageNet-16 |
|------|----------|-----------|-------------|
| SPOS | 基线 | 基线 | 基线 |
| SPOS + CaLR | +提升 | +提升 | +提升 |
| SPOS + MS | +提升 | +提升 | +提升 |
| **SPOS + CaLR + MS** | **最优** | **最优** | **最优** |

### 关键发现

1. **一致性提升**: CaLR + MS 在所有 NAS 方法和数据集上均显著提升超网的排名一致性（Kendall's Tau）。
2. **ImageNet Top-1 提升**: SPOS-L 从 76.6%→77.1%（+0.5%），FairNAS-L 从 76.7%→77.0%（+0.3%），仅需增加约 1% 的 GPU 时间。
3. **互补性强**: CaLR 和 MS 解决不同问题（公平性 vs 稳定性），联合使用效果最佳。
4. **通用性好**: 可无缝应用于 one-shot（SPOS、FairNAS）和 few-shot（FSNAS）NAS 方法。
5. **额外开销极低**: 峰值内存增加不到 1%，GPU 时间增加约 1-5%。

## 亮点与洞察

- **问题洞察深刻**: 精准识别了静态超网训练的两个核心问题，并提出针对性解决方案。
- **即插即用**: 两个组件均可独立或联合应用于任意 N-shot NAS 方法，无需修改搜索空间或采样策略。
- **理论与实践结合**: 不公平性问题的可视化分析和 CB/C3 指标提供了清晰的动机支撑。

## 局限与展望

- **复杂度指标选择**: 仅使用参数量作为复杂度评分，未考虑 FLOPs、内存等其他维度。
- **聚类策略简单**: MS 基于单边操作类型聚类，可能无法完全捕捉子网间的梯度相似性。
- **大搜索空间验证不足**: 在 $7^{21}$ 规模的搜索空间中，仅验证了 MobileNet 空间。
- 未来可探索自适应聚类、多维复杂度评分、与非均匀采样策略的结合。

## 相关工作与启发

- **FairNAS**: 通过采样多个子网平衡训练，但未考虑复杂度差异。PPA 从优化过程角度解决公平性。
- **PA&DA**: 通过降低梯度方差改善搜索，但关注采样策略。MS 关注优化器动量。
- **启发**: 超网训练中的"子网个体差异"是一个被忽视的重要因素，动态训练策略是通用解决方案。

## 评分

⭐⭐⭐⭐ — 问题挖掘精准，解决方案简洁有效，即插即用的设计极具实用价值。在多种 NAS 方法上一致提升且开销微乎其微是最大亮点。聚类策略的简单性略显不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[CVPR 2026\] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search](../../CVPR2026/others/intrain_intrinsic_trainability_for_zero-cost_neural_architecture_search.md)
- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)
- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)

</div>

<!-- RELATED:END -->
