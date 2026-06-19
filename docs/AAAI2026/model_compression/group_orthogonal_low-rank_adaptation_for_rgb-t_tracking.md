---
title: >-
  [论文解读] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking
description: >-
  [AAAI 2026][模型压缩][RGB-T跟踪] 提出 GOLA 框架，通过 SVD 分解量化 LoRA 秩重要性、冻结关键秩保留预训练先验、将冗余秩分组并施加组间正交约束，实现更高效的 RGB-T 跟踪适配。 领域现状 RGB-T（可见光+红外）跟踪通过融合两种模态的互补信息，在低光照、遮挡等复杂场景中显著增强鲁棒性…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "RGB-T跟踪"
  - "LoRA"
  - "低秩适配"
  - "正交约束"
  - "参数高效微调"
---

# Group Orthogonal Low-Rank Adaptation for RGB-T Tracking

**会议**: AAAI 2026  
**arXiv**: [2512.05359](https://arxiv.org/abs/2512.05359)  
**代码**: [GitHub](https://github.com/MelanTech/GOLA)  
**领域**: 视频理解  
**关键词**: RGB-T跟踪, LoRA, 低秩适配, 正交约束, 参数高效微调

## 一句话总结

提出 GOLA 框架，通过 SVD 分解量化 LoRA 秩重要性、冻结关键秩保留预训练先验、将冗余秩分组并施加组间正交约束，实现更高效的 RGB-T 跟踪适配。

## 研究背景与动机

### 领域现状
RGB-T（可见光+红外）跟踪通过融合两种模态的互补信息，在低光照、遮挡等复杂场景中显著增强鲁棒性。近期方法主要采用参数高效微调（PEFT）范式，冻结预训练参数，仅微调少量参数适配下游任务。

### 核心痛点
LoRA 在 RGB-T 跟踪中存在**秩空间严重冗余**问题：
- 训练后对 LoRA 参数矩阵进行 SVD 分解，发现只有**少数秩成为主导**，大量秩几乎不贡献实际信息
- 为了快速收敛，模型倾向于优先整合少数关键秩，**覆盖了预训练先验**
- 剩余冗余秩由于缺乏针对性优化信号而**未被激活**，无法学习细粒度特征
- 最终严重限制了模型处理 RGB-T 跟踪中**多样化挑战**（遮挡、光照变化、形变等）的能力

### 核心 Idea

**保护关键秩 + 激活冗余秩**：(1) 用 SVD 量化秩重要性，冻结关键秩保留预训练先验；(2) 将冗余秩聚类分组；(3) 对组间施加正交约束，迫使各组学习互补且不重叠的特征变换，从而充分利用整个秩空间。

## 方法详解

### 整体框架

GOLA 基于单流跟踪框架，对 backbone 每个线性层应用改进的 LoRA：
- 输入：模板图像 $(I_v^z, I_t^z)$、搜索图像 $(I_v^x, I_t^x)$、在线模板 $(I_v^o, I_t^o)$
- Token 化后拼接：$h = [Z_v; Z_t; X_v; X_t; O_v; O_t]$
- 通过编码器联合特征提取，GOLA 应用于每个线性层：$h' = \mathbf{W}h + \mathbf{BA}h$
- 推理时参数合并：$\mathbf{W}' = \mathbf{W} + \mathbf{BA}$（无额外推理延迟）

### 关键设计

#### 1. **秩分解分区策略（Rank Decomposition Partition）**

- **核心思路**：量化每个秩的重要性，区分关键秩和冗余秩，离线操作不增加训练负担
- **步骤**：
  1. 对矩阵 $\mathbf{B}$ 进行 SVD：$\Sigma, \mathbf{V} \leftarrow \text{SVD}(\bar{\mathbf{B}})$
  2. 选取 Top-k 奇异向量 $V_k$ 和对应奇异值 $\Sigma_k$ 作为参考
  3. 计算每个秩的重要性分数：$\mathbf{S} = \|\bar{\mathbf{B}}^\top \mathbf{V}_k^\top \odot \Sigma_k\|_2$
  4. 按 $\mathbf{S}$ 降序排序，前 k 个为关键秩（**冻结**），其余为冗余秩（**可训练**）
  5. 对冗余秩用受限 k-means 聚类分为 n 组
- **为何选择 $\mathbf{B}$ 做参考**：$\mathbf{B}$ 与任务特异性的关联更强，$\mathbf{A}$ 更像通用特征提取器
- **设计动机**：冻结关键秩保护已学到的泛化能力，避免被新模态适配所覆盖

#### 2. **组间正交约束策略（Inter-Group Orthogonal Constraint）**

- **核心思路**：通过正交损失强制不同秩组的参数保持正交关系，确保各组学习互补特征
- **正交损失**：

$$\mathcal{L}_{orth} = \sum_{i \neq j}\left(\left|\mathbf{A}_{u_i}^\top \mathbf{A}_{u_j}\right| + \left|\mathbf{B}_{u_i}^\top \mathbf{B}_{u_j}\right|\right)$$

- 对 $\mathbf{A}$ 施加**通道正交**约束：增强通用特征的多样性和判别力
- 对 $\mathbf{B}$ 施加**秩正交**约束：确保不同秩携带互补的任务知识
- **效率设计**：每次迭代只随机采样**一对**秩组计算正交损失，降低计算开销
- **设计动机**：避免冗余秩学习重叠的特征空间，使模型能同时应对 RGB-T 跟踪中的多种挑战

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{reg} + \lambda \cdot \mathcal{L}_{orth}$

- $\mathcal{L}_{cls}$: 二元交叉熵分类损失
- $\mathcal{L}_{reg}$: GIoU 回归损失
- $\lambda = 1.4 \times 10^{-3}$
- LoRA rank $r=64$，关键秩数 $k=16$，冗余秩分组数 $n=8$
- 训练 10 epochs，batch_size=128，每 epoch 131,072 图像对
- 在线模板更新阈值 $\tau=0.84$

## 实验关键数据

### 主实验（4 个基准数据集）

| 方法 | GTOT MPR/MSR | RGBT210 PR/SR | RGBT234 MPR/MSR | LasHeR PR/NPR/SR | Speed |
|------|-------------|---------------|-----------------|------------------|-------|
| ViPT | -/- | -/- | 83.5/61.7 | 65.1/-/52.5 | - |
| TBSI | -/- | 85.3/62.5 | 87.1/63.7 | 69.2/65.7/55.6 | 36fps |
| CKD | 93.2/77.2 | 88.4/65.2 | 90.0/67.4 | 73.2/69.3/58.1 | 96fps |
| SUTrack-L384 | -/- | -/- | 93.7/70.3 | 76.9/-/61.9 | 12fps |
| **GOLA-B** | **92.8/78.5** | **90.9/67.0** | **92.2/69.5** | **77.5/73.9/61.6** | **125fps** |
| **GOLA-L** | **95.3/80.9** | **92.0/68.7** | **92.8/71.3** | **78.1/74.5/61.9** | **64fps** |

GOLA-B 在 LasHeR 上以 125fps 的速度超越 SUTrack-L384（12fps），参数量仅 99M（10% 可训练）。

### 微调方法对比

| 方法 | 可训练参数占比 | LasHeR PR/SR | 推理速度 |
|------|-------------|-------------|---------|
| Full Fine-tune | 100% | 72.5/57.9 | 125fps |
| Adapter | 4% | 68.8/54.5 | 78fps |
| VPT | 3% | 70.8/56.3 | 85fps |
| LoRA | 13% | 76.3/60.7 | 125fps |
| DoRA | 13% | 63.7/49.3 | 125fps |
| **GOLA-B** | **10%** | **77.5/61.6** | **125fps** |

GOLA 比 LoRA 提升 1.2%/0.9%（PR/SR）且可训练参数减少 23%。

### 消融实验

| 配置 | PR (%) | SR (%) | 说明 |
|------|--------|--------|------|
| w/o 正交约束 | 76.3 | 60.7 | LoRA 基线 |
| 仅 $\mathbf{A}$ 正交 | 76.8 | 61.2 | 只约束通用特征 |
| 仅 $\mathbf{B}$ 正交 | 76.7 | 61.2 | 只约束任务知识 |
| **$\mathbf{A}$+$\mathbf{B}$ 正交** | **77.5** | **61.6** | 互补效果最佳 |

| 排序 | 聚类 | PR/SR | 说明 |
|------|------|-------|------|
| ✓ | ✗ | 77.0/61.4 | 仅排序 |
| ✗ | ✓ | 76.6/61.0 | 仅聚类 |
| **✓** | **✓** | **77.5/61.6** | 排序+聚类效果最佳 |

| 关键超参 | 最优值 | 说明 |
|---------|--------|------|
| 关键秩数 k | 16 | 过大→冗余秩被冻结过多；过小→预训练先验丢失 |
| 分组数 n | 8 | 过多→每组表达力不足；过少→互补性不够 |
| 正交损失采样对数 | 1 | 1对即可实现正交约束，更多不提升但增加计算 |

### 关键发现

1. **$\mathbf{A}$ 和 $\mathbf{B}$ 正交约束有互补效果**：同时约束两者比单独约束任一个效果更好
2. **秩排序和聚类必须结合**：排序帮助保持泛化能力，聚类促进组内特定知识学习
3. **t-SNE 可视化证实正交约束的效果**：GOLA 的不同组秩特征点在 t-SNE 中呈现明显的聚类和分离
4. **在 19 个属性上几乎全面最优**：特别在 HI（高强度）、HO（遮挡）、SV（尺度变化）等挑战性属性上优势显著

## 亮点与洞察

1. **LoRA 冗余性的深入分析**：通过 SVD 定量揭示了 LoRA 秩空间的冗余问题，为改进 LoRA 提供了理论基础
2. **极简且有效的设计**：冻结关键秩 + 分组 + 正交约束，概念简洁但效果显著
3. **推理无额外开销**：参数合并策略保证推理速度与原始 LoRA 一致（125fps）
4. **强通用性**：GOLA-B 和 GOLA-L 两种规模变体均表现优异，且离线分区策略不影响训练流程

## 局限与展望

1. 关键秩数 k 和分组数 n 需要超参搜索，缺乏自适应机制
2. 聚类基于固定的 constrained k-means，可探索更灵活的动态分组方法
3. 正交约束只在随机采样的一对组间施加，可能存在未充分约束的组对
4. 仅在 RGB-T 跟踪验证，可推广到更多多模态下游任务
5. 离线分区策略依赖模型先训练一轮 LoRA，增加了前期准备工作

## 相关工作与启发

- 与 AdaLoRA 动态调整秩的思路不同，GOLA 保持固定秩但优化秩空间利用
- 组间正交约束类似 MoE 中不同专家的特化思想，但无需路由机制
- 冻结关键秩的策略可推广到 NLP 领域的 LoRA 应用
- t-SNE 可视化分析为 LoRA 变体的评估提供了新视角

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 对 LoRA 冗余性的分析和组正交约束的解决思路新颖实用
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个数据集、19 个属性分析、多种 PEFT 方法对比、详尽消融
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，公式推导完整，可视化分析丰富
- **价值**: ⭐⭐⭐⭐ — 对 LoRA 使用的通用洞察可迁移到多个领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](../../ACL2026/model_compression/polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](../../ACL2026/model_compression/tlora_task-aware_low_rank_adaptation_of_large_language_models.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](../../ACL2025/model_compression/low-rank_interconnected_adaptation_across_layers.md)
- [\[CVPR 2026\] Adaptive Depth Lightweight RGB-T Tracking with Holistic Token Routing](../../CVPR2026/model_compression/adaptive_depth_lightweight_rgb-t_tracking_with_holistic_token_routing.md)

</div>

<!-- RELATED:END -->
