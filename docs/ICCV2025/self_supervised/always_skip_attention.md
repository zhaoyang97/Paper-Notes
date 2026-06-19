---
title: >-
  [论文解读] Always Skip Attention
description: >-
  [ICCV 2025][自监督学习][注意力机制] 本文从理论上证明了 Vision Transformer 中的自注意力机制是本质上病态的（ill-conditioned），在无 skip connection 时会导致训练崩溃，并提出 Token Graying（TG）方法通过改善输入 token 的条件数来进一步增强 ViT 的训练稳定性和性能。
tags:
  - "ICCV 2025"
  - "自监督学习"
  - "注意力机制"
  - "skip connection"
  - "condition number"
  - "ill-conditioning"
  - "token graying"
  - "Transformer"
---

# Always Skip Attention

**会议**: ICCV 2025  
**arXiv**: [2505.01996](https://arxiv.org/abs/2505.01996)  
**代码**: 暂无公开代码  
**领域**: 自监督学习 / Vision Transformer / 理论分析  
**关键词**: self-attention, skip connection, condition number, ill-conditioning, token graying, Vision Transformer

## 一句话总结

本文从理论上证明了 Vision Transformer 中的自注意力机制是本质上病态的（ill-conditioned），在无 skip connection 时会导致训练崩溃，并提出 Token Graying（TG）方法通过改善输入 token 的条件数来进一步增强 ViT 的训练稳定性和性能。

## 研究背景与动机

**领域现状**：Vision Transformer（ViT）在计算机视觉任务中取得了巨大成功，自注意力（Self-Attention Block, SAB）和前馈网络（FFN）加上 skip connection 构成其核心架构。Skip connection 作为标配组件被广泛使用，但其在 ViT 中起到作用的根本原因一直缺乏深入的理论解释。

**现有痛点**：
   - 作者发现了一个有趣但此前未被报道的经验性现象：在 ViT 中移除 SAB 的 skip connection 会导致**灾难性**的性能崩溃（CIFAR-10 上掉 22%），而移除 FFN 的 skip connection 只有温和的性能下降（掉 2%）
   - 这种"不对等依赖"在 CNN（如 ConvMixer）中不存在——移除 skip connection 对 CNN 性能几乎无影响（±0.2%）
   - 之前唯一尝试无 skip connection 训练 Transformer 的工作需要 5 倍训练时间，且原因未解释清楚

**核心矛盾**：为什么自注意力机制对 skip connection 如此依赖，而其他组件（FFN、卷积）则不然？这背后的根本原因是什么？

**本文目标**
   - 从理论上解释 SAB 对 skip connection 的极端依赖性
   - 理解 skip connection 在 SAB 中的真正作用机制
   - 基于理论洞察提出改善 ViT 训练的新方法

**切入角度**：从矩阵条件数（condition number）的视角分析。条件数衡量矩阵的"病态程度"——条件数越大，Jacobian 矩阵越病态，梯度下降的训练越不稳定。

**核心 idea**：自注意力运算的三次矩阵乘法结构使输出嵌入的条件数上界是输入条件数的**三次方**，导致本质性的病态化，而 skip connection 的作用正是正则化这一条件数。

## 方法详解

### 整体框架

本文的贡献分为两部分：
1. **理论分析**：证明 SAB 输出嵌入本质上病态，以及 skip connection 如何改善条件
2. **Token Graying（TG）方法**：通过预处理输入 token 改善其条件数，作为 skip connection 的补充增强手段

### 关键设计

1. **自注意力的病态性分析（Proposition 4.1）**:

    - 功能：理论推导 SAB 输出嵌入（无 skip connection）的条件数上界
    - 核心思路：对于线性注意力（无 softmax），SAB 的输出 $\mathbf{XW_QW_K^TX^TXW_V}$ 的条件数满足 $\kappa(\text{output}) \leq C \cdot (\sigma_{max}/\sigma_{min})^3$，即输入矩阵条件数的**立方**
    - 设计动机：这解释了为什么 SAB 的输出嵌入在实验中条件数高达 $e^6$，而 FFN 的输出只有 $e^3$。对比 FFN 的乘法结构 $\mathbf{W_{down}W_{up}X}$，其条件数上界仅为输入条件数的**一次方**
    - 虽然理论证明基于线性注意力，但实验表明 softmax 注意力也有相同的病态化问题

2. **Skip Connection 的正则化作用（Proposition 4.2）**:

    - 功能：理论证明 skip connection 显著改善 SAB 输出的条件
    - 核心思路：$\kappa(\mathbf{XM + X}) \ll \kappa(\mathbf{XM})$，添加恒等映射后条件数大幅降低
    - 这为"skip connection 为什么对 SAB 不可或缺"提供了清晰的数学解释：它不仅仅是梯度流的通道，更是条件数的正则化器

3. **Token Graying (TG) — SVD 版本**:

    - 功能：通过 SVD 分解重构输入 token，放大小奇异值以改善条件数
    - 核心思路：对 token 矩阵做 SVD $\mathbf{X = U\Sigma V^T}$，归一化奇异值后用幂指数 $\epsilon \in (0,1]$ 放大（$\tilde{\Sigma} = (\Sigma/\max(\Sigma))^\epsilon$），再重构 $\tilde{\mathbf{X}} = \mathbf{U\tilde{\Sigma}V^T}$
    - 问题：SVD 计算成本过高（训练慢约 6 倍），实用性差

4. **Token Graying (TG) — DCT 版本**:

    - 功能：用离散余弦变换（DCT）近似 SVD 的效果
    - 核心思路：自然图像中 SVD 的主奇异向量通常对应低频内容，而 DCT 本质上也是频域变换——$\hat{\mathbf{X}} = D\mathbf{X}D^T$，在频域放大后 IDCT 重构
    - 计算复杂度 $O(nd\log(nd))$ vs SVD 的 $O(nd\min(n,d))$，训练时间几乎无增加（0.732天 vs 0.723天 基准 vs 4.552天 SVD）

### 损失函数 / 训练策略

方法本身不引入新损失函数，训练仍采用标准交叉熵（有监督）或 MSE（MAE 自监督预训练）。TG 作为前处理步骤在 patch embedding 前执行，超参数只有放大系数 $\epsilon$（默认 0.95）。

## 实验关键数据

### 主实验

在多种 ViT 变体上的 ImageNet-1K 分类结果：

| 模型 | Top-1 Acc (%) | Top-5 Acc (%) | 使用 DCTTG |
|------|---------|---------|------|
| ViT-S | 80.2 | 95.1 | — |
| ViT-S + DCTTG | 80.4 | 95.2 | ✓ |
| ViT-B | 81.0 | 95.3 | — |
| ViT-B + DCTTG | 81.3 | 95.4 | ✓ |
| Swin-S | 81.3 | 95.6 | — |
| Swin-S + DCTTG | 81.6 | 95.6 | ✓ |
| CaiT-S | 82.6 | 96.1 | — |
| CaiT-S + DCTTG | 82.7 | 96.3 | ✓ |
| PVT V2 b3 | 82.9 | 96.0 | — |
| PVT V2 b3 + DCTTG | 83.0 | 96.1 | ✓ |

自监督学习（MAE 预训练 + 微调）：

| 方法 | Top-1 Acc (%) | Top-5 Acc (%) |
|------|---------|---------|
| MAE | 83.0 | 96.4 |
| MAE + DCTTG | 83.2 | 96.6 |

### 消融实验

ViT-B 不同 $\epsilon$ 值的影响（SVD 版本）：

| $\epsilon$ | Top-1 Acc | $\kappa_{in}$ (log) | $\kappa_{out}$ (log) |
|------|---------|------|------|
| — (baseline) | 81.0 | 6.72 | 6.74 |
| 0.9 | 81.2 | 6.64 | 6.66 |
| 0.7 | 81.4 | 6.15 | 6.17 |
| 0.6 | 81.4 | 5.73 | 5.71 |
| 0.5 | 81.0 | 5.29 | 5.25 |

Skip connection 移除实验（ViT-Tiny, CIFAR-10）：

| 配置 | CIFAR-10 Acc | 说明 |
|------|---------|------|
| Standard (SAB+FFN skip) | ~92% | 基准 |
| w/o FFN skip | ~90% | 温和下降（-2%） |
| w/o SAB skip | ~70% | 灾难性崩溃（-22%） |

训练时间对比：

| 方法 | ViT-B 训练时间 (天) |
|------|---------|
| Baseline | 0.723 |
| + SVDTG | 4.552 |
| + DCTTG | 0.732 |

### 关键发现

- **SAB vs FFN 不对等**：移除 SAB 的 skip connection 后条件数从 $e^3$ 飙升到 $e^6$，训练在 30 个 epoch 后发散；移除 FFN 的 skip 影响很小
- **CNN 无此问题**：ConvMixer 移除 skip connection 后性能变化 <±0.2%，证明这是自注意力特有的问题
- **条件数改善与性能正相关**：$\epsilon$ 降至 0.6-0.7 时条件数最优且性能最佳；$\epsilon$ 过低（0.5）虽然条件数好但可能丢失信息
- **DCT 是 SVD 的高效近似**：性能一致但训练开销几乎为零

## 亮点与洞察

- **深刻的理论洞察**：首次从条件数角度揭示了自注意力对 skip connection 的依赖本质——SAB 的三次矩阵乘法结构导致条件数三次方增长，这是一个优雅且可验证的理论解释。
- **简洁实用的改进方法**：DCT Token Graying 不改变模型架构，不增加参数，几乎不增加训练时间（+1.2%），实现极其简单——仅需在 patch embedding 前做 DCT 频域放大。
- **对 ViT 设计的启示**：这一发现暗示了一条新的 ViT 优化路径——与其设计更复杂的注意力机制，不如关注改善自注意力运算中的条件数。可能启发新的注意力归一化或初始化策略。

## 局限与展望

- **性能提升幅度有限**：在 ImageNet-1K 上的提升仅 0.2-0.4%，虽然稳定跨多个架构，但绝对提升不大
- **低精度训练可能有问题**：DCT 涉及大量乘法和求和运算，在低精度（FP16/BF16）下可能对量化误差敏感，论文未验证
- **理论仅覆盖线性注意力**：Proposition 4.1 的严格证明仅针对线性注意力，softmax 注意力只有经验验证
- **可改进思路**：可以探索自适应的 $\epsilon$ 调节策略（不同层、不同训练阶段用不同 $\epsilon$）；也可以将条件数正则化直接作为训练目标的一部分

## 相关工作与启发

- **vs He et al. 2023 (Deep Transformers without Shortcuts)**：他们通过在自注意力中引入归纳偏置实现了无 skip connection 训练，但需要 5 倍训练时间。本文的分析揭示了其成功的根本原因可能正是改善了条件数。
- **vs ResNet skip connections**：在 CNN 中 skip 主要用于缓解梯度消失；在 ViT 中 skip 有更关键的条件数正则化作用。这解释了为什么 VGG（无 skip 的 CNN）仍可训练，但无 skip 的 ViT 不行。
- **vs cosFormer、sigmoid attention 等替代注意力**：这些工作试图用不同激活函数替代 softmax，本文的条件数视角为评估这些替代方案提供了新的理论标准。
- 这个发现对于设计大规模 ViT 很有参考价值——随着模型深度增加，条件数的累积恶化可能是深层 Transformer 训练不稳定的重要原因。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从条件数角度解释 SAB 对 skip 的依赖，理论洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 多架构多任务验证，但绝对提升幅度较小
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验展示系统
- 价值: ⭐⭐⭐⭐ 理论贡献>实际提升，但对 ViT 设计的启示价值很高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[CVPR 2026\] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](../../CVPR2026/self_supervised/cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[ICLR 2026\] Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency](../../ICLR2026/self_supervised/attention_please_revisiting_attentive_probing_through_the_lens_of_efficiency.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](../../ICML2026/self_supervised/limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)

</div>

<!-- RELATED:END -->
