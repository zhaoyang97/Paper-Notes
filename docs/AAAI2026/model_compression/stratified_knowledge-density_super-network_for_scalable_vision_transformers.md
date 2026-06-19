---
title: >-
  [论文解读] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers
description: >-
  [AAAI 2026][模型压缩][Transformer] 提出将预训练 ViT 转化为"分层知识密度超网络"（SKD Super-Network），通过 WPAC（加权 PCA 注意力收缩）和 PIAD（渐进式重要性感知 Dropout）两步实现知识的分层组织，使得任意大小的子网络均可以 O(1) 代价提取，且无需额外微调即可达到或超越 SOTA 压缩方法的性能。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "Transformer"
  - "超网络"
  - "知识密度分层"
  - "PCA"
  - "渐进式dropout"
---

# Stratified Knowledge-Density Super-Network for Scalable Vision Transformers

**会议**: AAAI 2026  
**arXiv**: [2511.11683](https://arxiv.org/abs/2511.11683)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: Vision Transformer, 超网络, 知识密度分层, PCA, 渐进式dropout

## 一句话总结

提出将预训练 ViT 转化为"分层知识密度超网络"（SKD Super-Network），通过 WPAC（加权 PCA 注意力收缩）和 PIAD（渐进式重要性感知 Dropout）两步实现知识的分层组织，使得任意大小的子网络均可以 O(1) 代价提取，且无需额外微调即可达到或超越 SOTA 压缩方法的性能。

## 研究背景与动机

在实际部署 Vision Transformer 时，通常需要为不同资源约束训练和维护多个模型变体，成本极高。现有缩放方案的问题：

**传统剪枝方法**：对每个目标尺寸都需要单独执行剪枝+微调，无法一次性获得多尺度模型

**Learngene 范式**（如 TLEG、WAVE）：从预训练模型中提取核心权重并扩展为不同大小的后代模型，但依赖手工设计的扩展规则，且需要额外的知识蒸馏和长时间微调

**低秩压缩**：将权重矩阵分解为两个低秩矩阵，但分解后的模型信息保留能力有限

作者的核心洞察是：与其为每个目标大小分别压缩，不如**在预训练权重中建立一种分层的知识密度结构**——让重要的知识集中在权重的前几个维度中。这样，提取任意大小的子网络就变成了简单的"截断前 k 个维度"。

## 方法详解

### 整体框架

方法分两个阶段：
1. **WPAC 阶段**：通过加权 PCA 对预训练模型的权重进行**等价变换**，使知识集中到少数关键维度中（初始知识分层）
2. **PIAD 阶段**：通过渐进式重要性感知 Dropout 进一步训练，**强化知识的分层组织**，使小型子网络也能保持良好性能

最终得到的超网络可以按需截断维度来提取子网络，无需额外微调。

### 关键设计

1. **WPAC — 加权 PCA 注意力收缩**

   核心思想：对注意力模块中间特征做 PCA，得到按信息量排序的主成分变换矩阵，然后将其"吸收"到相邻的线性层中，实现**保函数等价变换**。

   **V/O 投影变换**：对 Value 投影输出 $X_v \in \mathbb{R}^{n \times d}$ 计算加权协方差矩阵并做特征分解，得到变换矩阵 $W^{(vo)}_{Trans}$，然后变换权重：
    $W_v \leftarrow W^{(vo)}_{Trans} W_v, \quad b_v \leftarrow W^{(vo)}_{Trans} b_v, \quad W_o \leftarrow W_o (W^{(vo)}_{Trans})^{-1}$

   **Q/K 投影变换**：联合计算 Q 和 K 输出的协方差矩阵（求和后分解），利用正交矩阵的性质 $W^T W = I$ 保证注意力得分不变：
    $\text{sim}_{(i,j)} \equiv (W_q x_i + b_q)^T (W^{(qk)}_{Trans})^T W^{(qk)}_{Trans} (W_k x_j + b_k)$

   **MLP 变换**：由于非线性激活的存在无法直接 PCA，改为按 Taylor 重要性排序维度，构造排列矩阵 $W_{sort}$ 重排权重。

   **加权策略**：传统 PCA 对所有 token 等权处理，但不同 token 对预测的贡献不同。WPAC 使用一阶 Taylor 估计计算 token 级重要性：
    $\Theta_{TE}(h_i) \approx \left|\frac{\delta \mathcal{C}}{\delta h_i} \cdot h_i\right|$
   然后对中心化后的 token 特征按 $\sqrt{\Theta^{\text{token}}_{TE}}$ 加权后再计算协方差矩阵。

2. **PIAD — 渐进式重要性感知 Dropout**

   目标：在 WPAC 的基础上进一步强化知识分层，让小子网络也能表现良好。

   **可丢弃单元**：将 MHSA 的中间维度分为 8 组、MLP 分为 32 组，每组作为一个"可丢弃单元"。

   **重要性评估**分两步：
    - 模块敏感度 $\gamma_m$：跳过模块 $m$ 后代价函数的相对增加
    - 维度级重要性 $I^{(m)}_i = \gamma_m \cdot \alpha^{(m)}_i$，其中 $\alpha^{(m)}_i$ 是模块内归一化的 Taylor 重要性
    - 最终单元重要性按 MACs 归一化：$I_u = \frac{\sum_{i \in u} I^{(m)}_i}{\text{MACs}(u)}$

   **渐进更新**：设目标最大压缩比为 $r$，分 $P_e$ 个 epoch 渐进构建 Dropout List。每个 epoch 开始时追加最不重要的单元，直到列表累积 MACs 达到目标值。

   **子网络采样与训练**：每个 batch 随机采样截断索引 $s$，丢弃 Dropout List 中排名 $s$ 之后的所有单元，训练该子网络并将梯度回传到超网络。

### 损失函数 / 训练策略

- WPAC 阶段无需训练，仅使用 1024 样本的小代理集计算 PCA 变换
- PIAD 阶段：DeiT-B 训练 150 epochs，DeiT-S/Ti 训练 300 epochs
- 渐进构建 Dropout List 的阶段跨 $P_e = 50$ epochs
- 训练设置遵循 DeiT 的标准配置

## 实验关键数据

### 主实验

**与网络扩展方法对比（ImageNet-1k，无微调直接提取子网络）：**

| 方法 | KD | DeiT-B 4:12 | DeiT-B 6:12 | DeiT-S 4:12 | DeiT-S 6:12 | DeiT-Ti 4:12 | DeiT-Ti 6:12 |
|------|-----|------|------|------|------|------|------|
| Albert | 否 | 71.7 | 75.3 | 65.0 | 69.7 | 55.2 | 59.8 |
| WAVE | 是 | 74.5 | 77.5 | 68.9 | 72.7 | 58.6 | 63.2 |
| TLEG | 是 | 71.6 | 76.2 | 63.7 | 69.5 | — | 58.2 |
| **SKD (Ours)** | **否** | **77.0** | **80.4** | **70.6** | **76.2** | **61.4** | **65.8** |

**与网络压缩方法对比（DeiT-S, ImageNet-1k）：**

| 方法 | MACs | Params | Epochs | Top-1 |
|------|------|--------|--------|-------|
| DeiT-S (原始) | 4.26G | 22.05M | — | 79.83 |
| SPViT | 3.30G | 15.90M | 300 | 78.30 |
| RePaViT | 3.20G | 16.70M | 300 | 78.90 |
| WDPruning | 3.10G | 15.00M | 100 | 78.55 |
| **SKD (Ours)** | **3.07G** | **16.03M** | **30** | **79.42** |

### 消融实验

| 配置 | DeiT-S 4:12 | DeiT-S 8:12 | DeiT-Ti 4:12 | DeiT-Ti 8:12 | 说明 |
|------|----------|----------|----------|----------|------|
| Baseline (随机裁剪) | 1.2 | 39.1 | 1.4 | 25.8 | 无分层结构 |
| B + Channel Dropout | 7.3 | 34.4 | 2.0 | 23.1 | 均匀 dropout |
| B + Weighted CD | 34.2 | 64.7 | 29.4 | 49.1 | 加权 dropout |
| B + LayerDrop | 39.7 | 68.6 | 34.5 | 57.1 | 层级 dropout |
| **B + PIAD** | **70.6** | **78.2** | **61.4** | **68.6** | **渐进式重要性感知** |

**WPAC vs 其他剪枝准则（直接评估，保留 50% 维度，DeiT-B→81.8）：**

| 准则 | 1/4 保留 | 2/4 保留 | 3/4 保留 |
|------|---------|---------|---------|
| Random | 0.9 | 24.4 | 74.0 |
| Magnitude | 1.7 | 29.2 | 71.9 |
| Taylor FO | 6.0 | 52.4 | 78.1 |
| Hessian | 5.1 | 52.0 | 78.2 |
| **WPAC** | **41.8** | **76.9** | **81.2** |

### 关键发现

- **WPAC 远超传统剪枝准则**：在 1/4 保留率下，WPAC 达到 41.8% 而 Taylor FO 仅 6.0%，差距高达 35.8 个百分点
- **零微调也能超越需要蒸馏的扩展方法**：SKD 无需额外教师模型和知识蒸馏，直接提取的子网络就优于 WAVE 等需要 KD 的方法
- **极少训练资源**：DeiT-B 上仅需 30 epochs 微调即可匹配或超越需要 100-300 epochs 的压缩方法
- **代理集仅需 1024 样本**即可获得准确的 PCA 投影（图 5 实验验证）
- 加权 PCA 中使用全部 token + 重要性加权的方案最优（表 6），但直接用全部 token 会导致协方差矩阵病态

## 亮点与洞察

- **函数等价变换的巧妙利用**：WPAC 不改变网络行为，只重新组织权重中的知识分布，是一种"免费"的知识集中操作
- **从信息论角度统一理解**：PCA 最大化保留信息 → 知识集中在前 k 维 → 截断即获得最优子网络
- **超极少训练**：WPAC 阶段零训练 + PIAD 阶段少量训练，总成本远低于传统多次压缩
- **统一框架**：同一个超网络覆盖从 1/3 到全尺寸的所有模型，真正做到"一次构建，任意提取"

## 局限与展望

- 仅在 DeiT 和 Swin Transformer 上验证，未扩展到 LLM 或多模态模型
- PIAD 训练仍需要数百个 epoch，对于更大的模型可能成本较高
- 仅支持同构子网络提取（均匀截断维度），未探索异构压缩（不同层不同压缩率）
- 下游任务迁移实验仅覆盖分类任务，未验证检测/分割等下游
- MLP 的维度排序仅用 Taylor 重要性而非 PCA，可能存在信息损失

## 相关工作与启发

- **Once-for-All / Slimmable Networks** 是可缩放网络的经典工作；本文的 SKD 在 ViT 上实现了类似目标但方法更高效
- **Learngene 系列**（TLEG, WAVE, SWS）提供了权重共享和扩展的思路，但本文证明直接从预训练模型建立分层结构更有效
- **低秩压缩**方法（SVD 分解）与 WPAC 思路相关，但 WPAC 在特征空间而非权重空间做 PCA，效果更优
- **Taylor 重要性**被广泛使用，本文将其与 PCA 结合用于加权是有趣的创新

## 评分

- 新颖性: ⭐⭐⭐⭐ — 函数等价 PCA 变换 + 渐进 dropout 的组合是新颖的
- 实验充分度: ⭐⭐⭐⭐⭐ — 大量对比实验、详细消融、跨模型验证
- 写作质量: ⭐⭐⭐⭐ — 数学推导严谨，图示清晰
- 价值: ⭐⭐⭐⭐⭐ — 实用性极强，显著降低多尺度部署成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[AAAI 2026\] EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers](efficientfsl_enhancing_few-shot_classification_via_query-only_tuning_in_vision_t.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
