---
title: >-
  [论文解读] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors
description: >-
  [ICML2025][自监督学习][模型再基化] 提出 TransFusion，一种专为 Transformer 设计的两级权重置换方法（头间+头内），实现将旧模型的微调知识（任务向量）免数据免训练地迁移至新版基础模型。 领域现状 领域现状：模型更新问题：预训练模型频繁更新，旧版微调模型过时需重训 解决思路 ：解决思路：任务…
tags:
  - "ICML2025"
  - "自监督学习"
  - "模型再基化"
  - "任务向量"
  - "Transformer"
  - "权重置换"
  - "免数据迁移"
---

# Update Your Transformer to the Latest Release: Re-Basin of Task Vectors

**会议**: ICML2025  
**arXiv**: [2505.22697](https://arxiv.org/abs/2505.22697)  
**代码**: [TransFusion](https://github.com/aimagelab/TransFusion)  
**领域**: 自监督  
**关键词**: 模型再基化, 任务向量, Transformer权重对齐, 权重置换, 免数据迁移

## 一句话总结

提出 TransFusion，一种专为 Transformer 设计的两级权重置换方法（头间+头内），实现将旧模型的微调知识（任务向量）免数据免训练地迁移至新版基础模型。

## 研究背景与动机

### 领域现状

**领域现状**：模型更新问题**：预训练模型频繁更新，旧版微调模型过时需重训

### 解决思路

**解决思路**：任务向量**：$\tau=\theta_A^{ft}-\theta_A$，目标是 $\theta_B^{ft}=\theta_B+\pi(\tau)$

### 现有痛点

**现有痛点**：现有 re-basin 仅适用 MLP/CNN**：多头注意力有"头部污染"问题

### 核心矛盾

**核心矛盾**：残差连接**的置换不一致性

## 方法详解

### Step 1: Inter-Head 对齐

- SVD 奇异值定义置换不变谱距离：$d_{ij}=\|\Sigma_i-\Sigma_j\|$
- 距离矩阵 $D_{ij}=d_{ij}^q+d_{ij}^k+d_{ij}^v$
- 匈牙利算法求最优头配对

### Step 2: Intra-Head 对齐

- 对配对头内行置换最大化内积
- 合成 $P_{attn}=P_{inter}\circ\{P_{intra}^{(h)}\}$
- 对 q/k/v 联合优化

### Step 3: 残差连接处理

- 补偿映射 $\mathcal{I}_i=P_{W_0}P_{in}^\top$ 使两分支共享置换
- 对第一个残差（注意力→加法）和第二个残差（FFN→加法）分别处理
- 保证残差两侧置换一致

### 传输公式

$$\tilde{\theta}_B^{ft}=\theta_B+\alpha\pi(\tau)$$

其中 $\alpha$ 为缩放因子（实验中 $\alpha=1$ 效果最佳）。

**Theorem 3.1**：结构化两级置换保持功能等价性：$O'=OP_{attn}$。

### 计算复杂度

$O(Ld_m^3)$，与 Git Re-Basin 相同（Proposition 3.2）。关键操作：SVD $O(d_k^2 d_m)$ + Hungarian $O(H^3)$ + LAP $O(d_k^3)$。

### 一次匹配多次复用

在多个任务向量场景中，$\theta_A \to \theta_B$ 的置换 $\pi$ 只需计算一次，可复用于所有基于 $\theta_A$ 微调的任务向量。还可在目标端 $\theta_B$ 上进行模型合并。

## 实验关键数据

### 视觉分类（CLIP ViT-B/16 CommonPool→Datacomp）

| 方法 | EuroSAT Task↑ | Supp.↑ | DTD Task↑ | SVHN Task↑ |
|---|---|---|---|---|
| $\theta_B$ zero-shot | 49.02 | 68.73 | 47.50 | 45.97 |
| $\theta_B+\tau$ (naive) | -7.62 | -16.15 | -0.15 | -22.00 |
| Git Re-Basin | +0.95 | -0.48 | -0.91 | +0.79 |
| Optimal Transport | -14.05 | -5.28 | -0.53 | -12.30 |
| **TransFusion** | **+4.95** | **-0.06** | **+0.21** | **+3.64** |

### NLP 任务（QQP/SST2 等）

- TransFusion 同样有效提升迁移性能

### $\alpha$ 敏感性分析

- $\alpha \approx 1$ 时下游任务提升最大
- $\alpha \geq 0.5$ 时泛化能力（支持集）更稳定
- naive transport 在 $\alpha$ 任何值均不稳定

### 消融实验

| 移除组件 | 影响 |
|---|---|
| Inter-head 对齐 | 性能显著下降 |
| Intra-head 对齐 | 中等下降 |
| 残差处理 | 模型崩塌 |
| 谱距离→余弦距离 | 头配对质量下降 |

## 亮点与洞察

1. 免数据免训练：仅需两组权重
2. 谱距离的置换不变性巧妙
3. 功能等价性有数学证明
4. 一次匹配可复用于所有任务向量

## 局限与展望

- LAP 近似无最优性保证
- 仅同架构间迁移
- 头数量多时区分度降低

## 相关工作与启发

- Ainsworth et al. (2023) Git Re-Basin
- Ilharco et al. (2023) 任务向量

## 评分

⭐⭐⭐⭐ — 两级置换策略优雅且有理论保证，问题实用

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)
- [\[ICML 2025\] MTL-UE: Learning to Learn Nothing for Multi-Task Learning](mtl-ue_learning_to_learn_nothing_for_multi-task_learning.md)
- [\[CVPR 2025\] Do Your Best and Get Enough Rest for Continual Learning](../../CVPR2025/self_supervised/do_your_best_and_get_enough_rest_for_continual_learning.md)
- [\[ECCV 2024\] PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer](../../ECCV2024/self_supervised/posformer_recognizing_complex_handwritten_mathematical_expression_with_position_.md)

</div>

<!-- RELATED:END -->
