---
title: >-
  [论文解读] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming
description: >-
  [NeurIPS 2025][AI安全][图域适应] 提出 GraphRTA 框架，通过模型重编程（基于梯度的权重剪枝）和图重编程（目标图结构与特征优化）双重机制，解决无监督开放集图域适应中已知类分类与未知类识别难题，无需人工设定阈值。 图神经网络在跨域场景中面临分布偏移导致的性能下降问题。无监督图域适应旨在将标签丰富的源图…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "图域适应"
  - "开放集识别"
  - "模型重编程"
  - "图重编程"
  - "对抗学习"
---

# Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming

**会议**: NeurIPS 2025  
**arXiv**: [2510.18363](https://arxiv.org/abs/2510.18363)  
**代码**: [有](https://github.com/cszhangzhen/GraphRTA)  
**领域**: 模型压缩  
**关键词**: 图域适应, 开放集识别, 模型重编程, 图重编程, 对抗学习

## 一句话总结

提出 GraphRTA 框架，通过模型重编程（基于梯度的权重剪枝）和图重编程（目标图结构与特征优化）双重机制，解决无监督开放集图域适应中已知类分类与未知类识别难题，无需人工设定阈值。

## 研究背景与动机

图神经网络在跨域场景中面临分布偏移导致的性能下降问题。无监督图域适应旨在将标签丰富的源图知识迁移到无标签目标图。现有方法主要关注**闭集**设定（源和目标域共享相同标签空间），这在实际应用中并不现实——目标域往往包含源域中不存在的新类别。例如，训练在已知欺诈模式上的检测模型在目标域中可能遇到全新的欺诈手段。

现有开放集方法存在两个核心缺陷：（1）依赖手动设定的熵阈值来区分已知与未知实例，但单一阈值无法适应不同分布；（2）主要关注源域与目标已知组的对齐，忽视了目标未知组的明确分离，导致决策边界不清晰。

GraphRTA 的创新在于：从**模型侧**和**数据侧**双重视角进行重编程，同时消除阈值依赖。

## 方法详解

### 整体框架

GraphRTA 由三个核心模块组成：（1）域无关模型重编程——通过梯度引导的权重剪枝减少源域偏差；（2）分布感知图重编程——修改目标图的结构和节点特征以减少域偏移；（3）三类域对抗学习——将实例分为源、目标已知、目标未知三组进行显式对齐与分离。

### 关键设计

1. **域无关模型重编程（Model Reprogramming）**: 受彩票假说启发，认为只有部分参数对跨域泛化至关重要。引入可微分掩码 $\mathbf{M}^l$ 对每层权重 $\mathbf{W}^l$ 进行剪枝：$\mathbf{Z}^l = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{Z}^{l-1}(\mathbf{W}^l \odot \mathbf{M}^l))$。通过计算梯度绝对值作为重要性得分，将最低 $\rho$% 的权重置零。此设计的核心动机是：梯度较小的权重捕获的是领域特定模式，剪枝后模型可聚焦于可迁移特征。

2. **分类器扩展消除阈值**: 在输出层增加一个额外维度用于未知类：$g_\phi(\mathbf{z}) = [\boldsymbol{\phi}^\top \mathbf{z}, \hat{\boldsymbol{w}}^\top \mathbf{z}]$。扩展后的 logits 通过 softmax 层生成后验概率，最高概率类别即为预测。与传统方法依赖固定阈值不同，此机制基于输入节点表示 $\mathbf{z}$ 动态决策。

3. **分布感知图重编程（Graph Reprogramming）**: 通过变换函数修改目标图的节点特征和结构：$\hat{\mathbf{X}}_t = \mathbf{X}_t + \Delta\mathbf{X}_t$（可学习特征扰动），$\hat{\mathbf{A}}_t = \mathbf{A}_t \oplus \Delta\mathbf{A}_t$（XOR 操作增删边，受预算 $\mathcal{B}$ 约束）。这直接在数据层面减少域偏移，弥补模型重编程无法解决的结构性差异。

4. **三类域对抗学习**: 传统对抗学习只区分源/目标两域，但开放集场景中对齐所有目标样本会导致负迁移。GraphRTA 先通过 Beta 混合模型+EM 算法估计每个目标节点属于已知/未知组的概率（无阈值），然后将域标签设为三类进行对抗训练，显式推开目标未知特征。

### 损失函数 / 训练策略

总体损失 $\mathcal{L} = \mathcal{L}_{adv} + \mathcal{L}_{cls} + \mathcal{L}_{ent}$：

- **对抗损失** $\mathcal{L}_{adv}$：三类域分类交叉熵，通过 GRL 实现极小极大博弈
- **分类损失** $\mathcal{L}_{cls}$：源图标签监督 + 非真值标签对齐到未知类（简化 mixup），训练模型明确将不匹配模式归为未知
- **熵最小化损失** $\mathcal{L}_{ent}$：鼓励目标实例的自信预测 + 未知组的判别性特征生成

## 实验关键数据

### 主实验

**引文数据集节点分类 (Table 2, 选取关键迁移方向)**:

| 方法 | A→C Acc | A→C HS | C→D Acc | C→D HS |
|------|---------|--------|---------|--------|
| GCN | 40.64 | 41.02 | 51.48 | 56.13 |
| GRADE | 57.23 | 59.49 | 61.94 | 64.21 |
| DANCE | 57.77 | 60.94 | 62.97 | 65.42 |
| G2Pxy | 59.75 | 54.47 | 61.42 | 59.13 |
| SDA | 58.23 | 59.97 | **63.55** | **65.53** |
| **GraphRTA** | **66.26** | **66.33** | 63.87 | 65.99 |

**ogbn-arxiv 大规模数据集 (Table 3)**:

| 方法 | I→II Acc | I→II HS | I→III Acc | I→III HS |
|------|----------|---------|-----------|----------|
| SAGE | 44.95 | 37.83 | 42.75 | 38.63 |
| A2GNN | 42.07 | 45.00 | 38.92 | 43.14 |
| DANCE | OOM | OOM | OOM | OOM |
| **GraphRTA** | - | - | - | - |

### 消融实验

根据论文描述，作者在附录 B 中提供了图重编程模块的不同变换函数对比消融。主要发现：

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 无模型重编程 | 性能下降 | 源域偏差未消除 |
| 无图重编程 | 性能下降 | 数据层面域偏移未解决 |
| 无扩展分类器 | 需手动阈值 | 退化为传统方案 |
| GraphRTA (完整) | 最优 | 双重重编程协同互补 |

### 关键发现

- GraphRTA 在 6 个引文网络迁移方向中的 4 个（Acc）和 5 个（HS）取得最优
- 在 WebKB 异质图上同样表现优异，证明方法对异质结构的适用性
- GraphRTA 是架构无关的，可搭配 GCN、GAT、GraphSAGE 等不同 GNN 骨干
- DANCE 在大规模 ogbn-arxiv 上 OOM，而 GraphRTA 可以处理

## 亮点与洞察

- **双重重编程视角新颖**：将"重编程"概念引入图域适应，从模型和数据两侧同时优化，思路清晰
- **无阈值设计实用**：通过 Beta 混合模型+扩展分类器，避免了传统开放集方法对手动阈值的依赖
- **架构无关性好**：可即插即用到不同 GNN 架构中
- Beta 混合模型估计已知/未知组概率的形式化处理比直接熵阈值更优雅

## 局限与展望

- 图重编程中边修改的预算 $\mathcal{B}$ 需要手动设定，仍是一个需调的超参数
- 图重编程需要学习 $\Delta\mathbf{X}_t \in \mathbb{R}^{n_t \times f}$，对大规模目标图内存开销可能较大
- 仅验证了节点分类任务，边级别和图级别任务未涉及
- 未与图基础模型或预训练 GNN 方法对比

## 相关工作与启发

- 模型重编程思路可与 LoRA 等参数高效微调结合
- 图重编程与图结构学习领域 (SLAPS, GSL) 有交叉，可借鉴更高效的结构优化方法
- 三类域对抗学习的框架可推广到更多域的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ (双重重编程在图DA中首次提出)
- 实验充分度: ⭐⭐⭐⭐ (覆盖引文网络、arxiv、WebKB三类数据)
- 写作质量: ⭐⭐⭐⭐ (组织清晰，但部分消融在附录)
- 价值: ⭐⭐⭐⭐ (开放集图DA有实际需求)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring](open-insect_benchmarking_open-set_recognition_of_novel_species_in_biodiversity_m.md)
- [\[ECCV 2024\] Operational Open-Set Recognition and PostMax Refinement](../../ECCV2024/ai_safety/operational_open-set_recognition_and_postmax_refinement.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](../../ICML2026/ai_safety/scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
