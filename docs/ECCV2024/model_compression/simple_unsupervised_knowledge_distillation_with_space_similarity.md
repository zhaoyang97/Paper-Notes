---
title: >-
  [论文解读] Simple Unsupervised Knowledge Distillation With Space Similarity
description: >-
  [ECCV 2024][模型压缩][无监督知识蒸馏] CoSS 提出在无监督知识蒸馏中，除了常规的**特征维度余弦相似度**外，额外引入一个**空间维度余弦相似度（Space Similarity）**损失——将特征矩阵转置后在维度方向上对齐，从而弥补 $L_2$ 归一化导致的流形结构信息丢失，以极简的方式在多个 UKD benchmark 上达到 SOTA。
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "无监督知识蒸馏"
  - "空间相似性"
  - "流形学习"
  - "余弦相似度"
  - "同胚映射"
---

# Simple Unsupervised Knowledge Distillation With Space Similarity

**会议**: ECCV 2024  
**arXiv**: [2409.13939](https://arxiv.org/abs/2409.13939)  
**代码**: 无（未提供）  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 无监督知识蒸馏, 空间相似性, 流形学习, 余弦相似度, 同胚映射

## 一句话总结

CoSS 提出在无监督知识蒸馏中，除了常规的**特征维度余弦相似度**外，额外引入一个**空间维度余弦相似度（Space Similarity）**损失——将特征矩阵转置后在维度方向上对齐，从而弥补 $L_2$ 归一化导致的流形结构信息丢失，以极简的方式在多个 UKD benchmark 上达到 SOTA。

## 研究背景与动机

自监督学习（SSL）在大模型上取得了优秀的通用表征，但小模型由于参数有限无法充分从 SSL 中获益。无监督知识蒸馏（UKD）通过将大的自监督 teacher 的知识迁移到小 student 来解决此问题。现有 UKD 方法（SEED、BINGO、DisCo、SMD 等）通常手工构造样本间的相似性关系来蒸馏，但可能忽略 teacher 映射中的其他重要结构信息。

**核心矛盾**：所有现有 UKD 方法都依赖 $L_2$ 归一化后的特征进行操作（用于计算余弦相似度），但 $L_2$ 归一化是**不可逆映射**：它将所有点投影到超球面上，原始流形的丰富结构被破坏。具体来说，从原点出发的同一射线上的所有点会被映射到同一个超球面点，因此归一化后的操作无法恢复原始嵌入流形的拓扑结构。

**数学论证**：$L_2$ 归一化不是同胚映射（homeomorphism），因为它不是连续的、双射的，也没有连续的逆映射。因此仅最小化归一化特征上的目标函数，无法保留原始未归一化的流形结构。

**切入角度**：与其手工构造要保留的样本关系，不如直接让 student 建模 teacher 的嵌入流形。如果两个流形相似，所有样本间关系自然被间接保留。

**核心 idea**：用转置特征矩阵后在空间维度上计算余弦相似度（Space Similarity），恢复归一化丢失的结构信息，与常规特征相似度互补。

## 方法详解

### 整体框架

CoSS 是一个两阶段 UKD 框架：
1. **离线预处理**：用 teacher 模型计算训练集的 $k$-近邻索引
2. **蒸馏训练**：在增强了近邻样本的 mini-batch 上，联合优化特征相似度（Feature Similarity）和空间相似度（Space Similarity）

### 关键设计

1. **离线 k-近邻预处理**:

    - 功能：利用 teacher 编码器为训练集中每个样本计算 $k$ 个最近邻
    - 核心思路：先用 teacher 产生所有训练样本的 $L_2$ 归一化特征，计算相似度矩阵 $S_{ij} = \hat{f}_t(x_i) \cdot \hat{f}_t(x_j)$，然后取 top-k 作为近邻集合 $\Omega_i^k = \arg\max(S_{i\cdot}, k)$
    - 设计动机：标准随机采样的 mini-batch 中，局部邻域信息缺失。通过在 batch 中追加近邻样本，使 student 能捕获流形的局部结构。这对流形建模至关重要——不仅要匹配全局结构，还要保留局部细节。

2. **特征相似度（Feature Similarity / Cosine Similarity）**:

    - 功能：对每个样本，最大化 teacher 和 student 归一化特征向量的余弦相似度
    - 核心思路：

    $\mathcal{L}_{co} = -\frac{1}{bk} \sum_{i=0}^{bk} \text{cosine}(\hat{A}_s^i, \hat{A}_t^i)$

   其中 $\hat{A}^i$ 是样本 $i$ 的 $L_2$ 归一化特征向量。这是 UKD 中广泛使用的标准损失。
    - 设计动机：保证 teacher 和 student 对同一样本的表征方向一致，在归一化流形上做对齐。但单靠它无法恢复归一化前的结构。

3. **空间相似度（Space Similarity）**:

    - 功能：将特征矩阵**转置**，然后在**空间维度**（即特征的每个维度对应的样本响应向量）上计算余弦相似度
    - 核心思路：构造转置矩阵 $Z = A^T$（大小 $d \times bk$），归一化后计算：

    $\mathcal{L}_{ss} = -\frac{1}{d} \sum_{i=0}^{d} \text{cosine}(\hat{Z}_s^i, \hat{Z}_t^i)$

   这里每个 $Z^i$ 是特征空间第 $i$ 个维度在 batch 中所有样本上的响应向量。最小化此损失使 teacher 和 student 的每个特征维度对样本的响应模式一致。
    - 数学保证：在空间维度上的归一化对同一维度下的所有数据点做相同缩放，因此保留了双射性和连续性。当 $\mathcal{L}_{ss}$ 最小时，$f_s(x_i) = \frac{\alpha}{\beta} f_t(x_i)$，其中 $\alpha, \beta > 0$ 是维度级缩放向量，映射连续、双射且可逆，满足同胚条件。
    - 设计动机：恢复 $L_2$ 归一化在特征维度上丢失的结构信息。Feature Similarity 保证样本级对齐方向一致，Space Similarity 保证维度级响应模式一致，两者互补覆盖完整的流形结构。

### 损失函数 / 训练策略

最终损失函数极其简洁：

$$\mathcal{L}_{CoSS} = \mathcal{L}_{co} + \lambda \mathcal{L}_{ss}$$

其中 $\lambda = 1.0$（消融实验显示 $\lambda \in \{0.5, 1.0\}$ 效果一致）。整体损失再乘以 70.0 的缩放因子（观察到不缩放时收敛慢）。

训练设置：$k=15$ 近邻，$N=31$ 个近邻候选，batch size $B=64$，初始学习率 0.03，余弦学习率衰减，蒸馏 25 个 epochs（4 GPU），使用 mocov2_aug 增强策略。Teacher 为 Moco-v2 预训练的 ResNet-50，student 使用 ResNet-18/34 或 EfficientNet-B0，在 student 上添加投影头将输出维度对齐到 teacher 的 2048 维。

**极简设计的优势**：与现有 UKD 方法相比，CoSS **不需要**特征队列、对比学习目标、重型数据增强。

## 实验关键数据

### 主实验

| Student | 指标 | CoSS | DisCo | BINGO | SEED | Moco-v2(baseline) |
|---------|------|------|-------|-------|------|-------------------|
| ResNet-18 | Top-1 | **62.35** | 60.60 | 61.40 | 57.60 | 52.20 |
| ResNet-18 | KNN-10 | **53.78** | 52.03 | 54.16 | 50.12 | 36.70 |
| ResNet-34 | Top-1 | **64.01** | 62.50 | 63.50 | 58.50 | 56.80 |
| EfficientNet-B0 | Top-1 | **67.36** | 66.50 | 63.74 | 61.30 | 42.20 |
| EfficientNet-B0 | KNN-10 | **58.33** | 54.78 | 54.75 | 53.11 | 30.00 |

EfficientNet-B0 仅有 4M 参数（teacher 的 16.3%），top-1 达到 67.36%，仅比 teacher（67.40%）低 0.04%，几乎追平。

### 消融实验

| 配置 | Top-1 (R18) | KNN-10 (R18) | 说明 |
|------|------------|-------------|------|
| $\mathcal{L}_{co}$ only ($\lambda=0$) | ~60.0 | ~51.5 | 等价于 SimReg |
| $\mathcal{L}_{ss}$ only | 略低 | 略低 | 单独空间相似度不够 |
| CoSS ($\lambda=1.0$) | **62.35** | **53.78** | 两者互补效果最佳 |
| $k=0$（无近邻采样） | 下降 | 下降 | 局部信息对流形建模重要 |
| $k=15$（近邻采样） | **62.35** | **53.78** | 最优配置 |

### 关键发现

- **Space Similarity 与 Feature Similarity 互补**：单独使用任一项都不如两者联合，验证了流形结构需要从特征和空间两个方向同时约束
- **近邻采样至关重要**：$k=0$（无近邻）vs $k=15$ 有明显差距，局部流形结构的捕获依赖近邻信息
- **CoSS 超越 SLD（Soft-label Distillation）**：R18 top-1 62.35% vs 59.88%，说明直接建模流形比模仿输出分布更有效
- **跨 teacher 通用**：从 ResNet-101 蒸馏到 ResNet-18 也有效（63.74%），超过 DisCo（62.30%）

## 亮点与洞察

- **极简但有效**：核心创新仅仅是"转置特征矩阵后再做余弦相似度"，实现只需一行代码（transpose），却带来一致性提升
- **理论动机扎实**：从同胚映射的角度严格论证了 $L_2$ 归一化的信息丢失问题，reasoning 链完整
- **不需要特征队列**：SEED/BINGO 等方法需要维护 ~100K 长的特征队列，CoSS 完全不需要，训练更简单高效
- **可迁移的设计原则**：Space Similarity 的思想可以轻松嵌入到任何基于余弦相似度的蒸馏框架中，通用性强

## 局限与展望

- 仅验证了 CNN 架构（ResNet、EfficientNet），未在 ViT 等 Transformer 架构上验证（作者指出 AttnDistill 专注 ViT，但未与其对比）
- Teacher 仅使用了 Moco-v2（较早的 SSL 方法），未验证更强的 teacher（如 DINO、MAE）
- 同胚约束只保证结构"up to a scale"对齐，更强的约束（如等距映射）可能进一步提升
- 近邻预计算需要额外存储全量相似度矩阵，对超大规模数据集可能有瓶颈
- 蒸馏仅 25 epochs，如果增加训练时间是否还能进一步提升未知

## 相关工作与启发

- **vs SEED**：SEED 最小化 teacher-student 在共享嵌入队列上的相似度分布散度，需要大特征队列。CoSS 无需队列，更简单
- **vs BINGO**：BINGO 两阶段（先聚类构造 bag，再对比蒸馏），复杂且引入噪声标签问题。CoSS 单阶段训练
- **vs DisCo**：DisCo 在增强视图间做一致性正则化 + 对比蒸馏，CoSS 不用对比目标也能超越
- **vs SimReg**：SimReg 仅使用 $\mathcal{L}_{co}$，CoSS 加上 $\mathcal{L}_{ss}$ 后一致性超越，证明了空间相似度的增益不是来自超参数调优而是本质性的互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心 idea 简洁但洞察深刻，从流形同胚角度发现了归一化的信息丢失并给出优雅修复
- 实验充分度: ⭐⭐⭐⭐ ImageNet 分类、迁移学习、检测、检索等全面评估，消融详细
- 写作质量: ⭐⭐⭐⭐⭐ 动机论证清晰，从拓扑学概念出发，逻辑链紧密，可读性好
- 价值: ⭐⭐⭐⭐ 实用价值高，设计极简可即插即用，对 UKD 领域有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)
- [\[ECCV 2024\] A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging](a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)
- [\[ICCV 2025\] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](../../ICCV2025/model_compression/soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)
- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](../../ICCV2025/model_compression/cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[ICCV 2025\] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](../../ICCV2025/model_compression/competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)

</div>

<!-- RELATED:END -->
