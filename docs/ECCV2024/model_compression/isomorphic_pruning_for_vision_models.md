---
title: >-
  [论文解读] Isomorphic Pruning for Vision Models
description: >-
  [ECCV 2024][模型压缩][结构化剪枝] 提出 Isomorphic Pruning，通过将网络子结构建模为图并按图同构性分组，在同构组内独立排序剪枝，解决异构子结构间重要性不可比的问题，在 ViT 和 CNN 上均取得优于专门设计的剪枝方法的效果。 问题背景 结构化剪枝通过移除冗余子结构来降低深度神经网络的计算开销…
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "结构化剪枝"
  - "Transformer"
  - "CNN"
  - "图同构"
  - "子结构分组"
---

# Isomorphic Pruning for Vision Models

**会议**: ECCV 2024  
**arXiv**: [2407.04616](https://arxiv.org/abs/2407.04616)  
**代码**: 有 ([https://github.com/VainF/Isomorphic-Pruning](https://github.com/VainF/Isomorphic-Pruning))  
**领域**: 模型压缩  
**关键词**: 结构化剪枝, Vision Transformer, CNN, 图同构, 子结构分组

## 一句话总结

提出 Isomorphic Pruning，通过将网络子结构建模为图并按图同构性分组，在同构组内独立排序剪枝，解决异构子结构间重要性不可比的问题，在 ViT 和 CNN 上均取得优于专门设计的剪枝方法的效果。

## 研究背景与动机

### 问题背景

结构化剪枝通过移除冗余子结构来降低深度神经网络的计算开销。标准流程是"排序-剪枝"：用重要性准则评估不同子结构的重要性，排序后移除最不重要的部分，最后微调恢复性能。

### 核心问题：异构子结构间的重要性不可比

经典网络（如VGG）由相似的卷积层堆叠而成，全局排序能可靠地揭示相对重要性。但现代视觉模型（Vision Transformer、ConvNext）包含了自注意力、深度卷积、残差连接等多种异构结构：

- 即使在同一个多头注意力层内，也存在两种剪枝方案：剪整个head 和 剪head内维度
- 这些异构子结构的参数规模、权重分布和计算拓扑差异巨大
- **直接做全局排序会导致偏倚剪枝**：例如在DeiT中，MLP层的重要性分数集中在较高值，而注意力层的分数集中在0附近，朴素50%全局剪枝会几乎移除所有注意力参数而保留所有embedding参数

### 关键观察

在同一重要性准则下，异构子结构的重要性分布呈现显著分歧，而同构子结构（拥有相同计算拓扑的结构）则表现出相似的重要性分布模式。这启发了在同构结构组内独立排序的思路。

## 方法详解

### 整体框架

Isomorphic Pruning 分三步：(1) 将网络分解为一组最小独立子结构并建模为图；(2) 通过图同构检测将子结构聚类为同构组；(3) 在每个同构组内独立排序和剪枝。

### 关键设计

#### 1. **子结构的图建模 (Graph Modeling of Sub-structures)**

**功能**：将网络中的可剪枝子结构表示为图，节点是剪枝操作三元组 $(W, d, k)$（参数矩阵、剪枝轴、索引），边表示依赖关系。

**核心思路**：定义剪枝函数 $\hat{w} = g(W, d, k)$，其中 $d \in \{0, 1\}$ 表示剪输出维度或输入维度，$k$ 是第 $k$ 行/列。对相邻层 $W_1, W_2$，$W_1$ 的输出维度与 $W_2$ 的输入维度耦合，必须同时剪枝。依赖建模遵循两条规则：

- 相邻层的相同维度索引存在耦合
- 同一参数矩阵的不同剪枝操作若产生相同切片则耦合（如归一化层）

从种子参数出发递归发现依赖，构建完整子结构图 $G=(V, E)$。

**设计动机**：图表示天然捕获了子结构内部参数间的耦合关系和计算拓扑特征，为后续同构判定提供了形式化基础。

#### 2. **图同构分组 (Ranking with Graph Isomorphism)**

**功能**：判定哪些子结构图是同构的（计算拓扑相同），将同构子结构聚类为组。

**核心思路**：两个图 $G, G'$ 同构的判定条件（边按前向执行顺序排列后）：

$$\text{Isomorphic}(G, G') = \mathbb{1}\{|G|=|G'| \land \text{label}(V_i)=\text{label}(V'_i); \forall i\}$$

其中两个节点标签相同当且仅当来自相同层类型（Linear/Conv）且剪枝维度相同。允许索引 $k$ 不同，因为同一参数矩阵沿同一维度的不同切片是天然同构的。

**设计动机**：同构子结构具有相同的架构设计、参数规模和计算图，因此在组内进行重要性比较是有意义和可靠的。

#### 3. **组内独立排序剪枝 (Grouped Ranking and Pruning)**

**功能**：在每个同构组内独立计算聚合重要性分数，排序后按比例剪枝。

**核心思路**：对同构组 $\{G_1, G_2, \ldots, G_N\}$，计算每个子结构的聚合重要性：

$$I^*(G(V,E)) = \sum_{(W,d,k) \in V} I(W, d, k)$$

其中 $I(W,d,k)$ 可以是任意单层重要性准则（如 L1 范数的 Magnitude Pruning 或基于梯度的 Taylor Pruning）：

$$I(W,d,k) = \left\|\frac{\partial \mathcal{L}}{\partial g(W,d,k)} \cdot g(W,d,k)\right\|_2 \quad (\text{Taylor})$$

在每个同构组内对 $N$ 维重要性向量排序，移除最不重要的 $p\%$。

**设计动机**：同组内参数规模和计算模式一致，聚合重要性分数可以公平比较；跨组比较因拓扑差异会产生偏倚，通过隔离排序避免了这个问题。

### 损失函数 / 训练策略

- **剪枝阶段**：使用 Taylor 准则，随机采样100个 mini-batch（每批64张图），累积梯度做一次性剪枝（one-shot），不做迭代剪枝微调
- **微调阶段**：沿用各模型原始训练协议。DeiT 用 RegNetY 蒸馏 300 epochs，AdamW 优化器，学习率 0.0005，余弦退火，权重衰减 0.05
- ConvNext/ResNet 同样遵循原始训练设置

## 实验关键数据

### 主实验（Vision Transformer 剪枝）

在 ImageNet-1K 上从预训练 DeiT-Base 剪枝得到不同大小的模型：

| 方法 | Params(M) | MACs(G) | Top-1 Acc(%) |
|------|-----------|---------|-------------|
| DeiT-B (原始) | 87.34 | 17.69 | 83.32 |
| DeiT-S (原始) | 22.44 | 4.64 | 81.17 |
| NViT-S | 21.00 | 4.20 | 82.19 |
| SAViT | 25.40 | 5.30 | 81.66 |
| **DeiT-S (Ours)** | **20.69** | **4.16** | **82.41** |
| DeiT-T (原始) | 5.91 | 1.27 | 74.52 |
| NViT-T | 6.90 | 1.30 | 76.21 |
| X-Pruner | - | 2.40 | 78.93 |
| **DeiT-T (Ours)** | **5.74** | **1.21** | **77.50** |
| DeiT-0.6G (Ours) | 3.08 | 0.62 | 72.60 |

从 DeiT-B 剪枝到 DeiT-T 量级，准确率从原始 74.52% 提升至 **77.50%**（+2.98%），MACs 更低。

### 主实验（CNN 剪枝）

| 方法 | Params(M) | MACs(G) | Base Acc | Final Acc | Δ Acc |
|------|-----------|---------|----------|-----------|-------|
| ConvNext-T (原始) | 28.59 | 4.47 | - | 82.06 | - |
| **ConvNext-T (Ours)** | **25.32** | **4.19** | 83.83 | **82.19** | -1.64 |
| ResNet-101 (原始) | 44.55 | 7.85 | - | 77.38 | - |
| **Res101-4.5G (Ours)** | **29.14** | **4.48** | 77.38 | **77.56** | **+0.16** |
| **Res101-3.8G (Ours)** | **24.87** | **3.85** | 77.38 | **77.43** | **+0.05** |
| ResNet-50 (原始) | 25.56 | 4.13 | - | 76.13 | - |
| Res50-2G (Ours) | 15.05 | 2.06 | 76.13 | 75.91 | -0.22 |

ResNet-101 上实现了**无损压缩**（4.5G和3.8G设置下精度不降反升）。

### 消融实验

在DeiT-Base上剪枝不同重要性准则的效果对比：

| 剪枝策略 | 准则 | 精度趋势 | 说明 |
|----------|------|----------|------|
| Global + Taylor | Taylor | 基线 | 传统全局排序 |
| **Isomorphic + Taylor** | Taylor | **一致性提升** | 在所有剪枝率下均优于全局 |
| Global + Hessian | Hessian | 基线 | 二阶信息 |
| **Isomorphic + Hessian** | Hessian | **一致性提升** | 同构分组通用性好 |
| Global + L1 | L1范数 | 基线 | 最简单准则 |
| **Isomorphic + L1** | L1范数 | **一致性提升** | 即使最简单准则也受益 |

Isomorphic Pruning 在所有剪枝准则和所有剪枝率下均一致优于对应的全局/局部剪枝基线。

### 实际延迟分析

| 模型 | GPU延迟(ms) | 加速比 | CPU延迟(ms) | 加速比 | Acc(%) |
|------|------------|--------|------------|--------|--------|
| DeiT-B | 802.97 | 1.00× | 197.73 | 1.00× | 83.32 |
| DeiT-S (Ours) | 230.84 | 3.48× | 44.10 | 4.48× | 82.41 |
| EfficientFormer-L3 | 249.05 | 3.22× | 167.16 | 1.18× | 82.40 |

剪枝DeiT-S 的性能（82.41%）可比 EfficientFormer-L3（82.40%），但CPU延迟仅为后者的26%。

### 关键发现

1. **异构子结构的重要性分布严重分歧**：DeiT 中 MLP 和 Attention 参数的 Taylor 分数分布几乎不重叠，验证了直接全局比较的不可靠性
2. **朴素全局剪枝的偏倚**：50%全局阈值会几乎完全移除注意力层（分数集中在0）而完全保留embedding层
3. **在 ViT 上比 CNN 收益更大**：因为 ViT 内部异构性更强
4. **非均匀架构的内存权衡**：非均匀剪枝精度更高但峰值内存略高于均匀架构

## 亮点与洞察

1. **极其简洁的核心思想**：将"不可比的不要比"这一直觉用图同构形式化，方法实现简单但效果显著
2. **架构无关性强**：同一套方法无需修改即可应用于 ViT、ConvNext、ResNet、MobileNet 等架构
3. **与现有准则正交**：作为排序策略可与 Taylor、Hessian、L1 等任意重要性准则组合
4. **实际加速显著**：不仅理论FLOPs减少，GPU/CPU实测延迟大幅降低

## 局限与展望

1. **剪枝粒度固定**：每个同构组使用相同的剪枝比例 $p\%$，未探索组间自适应分配预算
2. **依赖微调阶段**：需要较长的微调恢复（DeiT 300 epochs），one-shot 剪枝后无微调的精度下降较大
3. **同构判定可能过于严格**：严格图同构可能产生过多组，导致某些组样本量太少影响排序可靠性
4. **未涉及动态/token剪枝**：仅做静态结构化剪枝，未与动态 token 剪枝结合

## 相关工作与启发

- **DepGraph (CVPR 2023)**：同一作者的前序工作，建模依赖关系但未考虑同构分组
- **NViT (NeurIPS 2022)**：通过稀疏训练学习目标结构，计算代价大
- **UPop (ICLR 2023)**：统一剪枝但仍用全局排序
- 启发：图同构分组思想可推广到 NLP Transformer 和多模态大模型的剪枝

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将图同构引入剪枝是新视角，核心思想简洁优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 ViT/CNN 多种架构，有延迟/内存分析、迁移学习、多准则消融
- **写作质量**: ⭐⭐⭐⭐ — 形式化严谨，图示清晰（尤其Fig.2和Fig.4c的重要性分布对比图）
- **价值**: ⭐⭐⭐⭐⭐ — 方法简单通用且性能强，开源代码，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Collaborative Multi-Mode Pruning for Vision-Language Models](../../CVPR2026/model_compression/collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](../../CVPR2025/model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[ECCV 2024\] Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning](token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin.md)
- [\[CVPR 2025\] MDP: Multidimensional Vision Model Pruning with Latency Constraint](../../CVPR2025/model_compression/mdp_multidimensional_vision_model_pruning_with_latency_constraint.md)
- [\[ECCV 2024\] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)

</div>

<!-- RELATED:END -->
