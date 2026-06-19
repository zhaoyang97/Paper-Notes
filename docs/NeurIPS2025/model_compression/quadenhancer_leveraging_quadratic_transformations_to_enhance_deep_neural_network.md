---
title: >-
  [论文解读] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks
description: >-
  [NeurIPS 2025][模型压缩][二次变换] 提出一种轻量级的二次增强器（QuadEnhancer），通过在每个线性层引入稀疏化的二次交互项，以极少的额外参数和计算开销显著提升现有神经网络架构的性能。 现代深度神经网络的核心构建块是"线性变换 + 非线性激活函数"的组合。尽管这一框架在逼近复杂函数方面取得了巨大成功…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "二次变换"
  - "非线性增强"
  - "轻量级模块"
  - "LoRA微调"
  - "权重共享"
---

# QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2510.03276](https://arxiv.org/abs/2510.03276)  
**代码**: [GitHub](https://github.com/chitar/QuadEnhancer)  
**领域**: 模型压缩  
**关键词**: 二次变换, 非线性增强, 轻量级模块, LoRA微调, 权重共享

## 一句话总结

提出一种轻量级的二次增强器（QuadEnhancer），通过在每个线性层引入稀疏化的二次交互项，以极少的额外参数和计算开销显著提升现有神经网络架构的性能。

## 研究背景与动机

现代深度神经网络的核心构建块是"线性变换 + 非线性激活函数"的组合。尽管这一框架在逼近复杂函数方面取得了巨大成功，但其非线性能力仍有提升空间。现有的增强非线性的路线主要有三条：

**更复杂的激活函数**（如 Swish、GELU、Mish）：这些方法聚焦于逐元素变换，无法捕捉神经元之间的交互。

**设计非线性网络模块**（如 LSTM 门控、注意力机制）：通常是任务特定的，通用性受限。

**多项式变换替代线性操作**（如多项式网络、QuadraNet）：理论上有更强的表达能力，但参数量和计算量的急剧增长限制了其实际应用。

核心痛点在于：标准的二次变换需要 $O(dn^2)$ 的额外参数，这在实际部署中是不可接受的。本文的动机是找到一种方法，在引入二次项提升表达能力的同时，将额外的参数和计算开销控制在可忽略的水平。

## 方法详解

### 整体框架

QuadEnhancer 是一个即插即用的模块，可以附加在任何线性层上。对于标准线性变换 $\tilde{\mathbf{y}} = \mathbf{W}\mathbf{x}$，增强后的输出变为：

$$\mathbf{z} = (\mathbf{\Lambda}\tilde{\mathbf{y}}) \odot \tilde{\mathbf{y}} + \tilde{\mathbf{y}} + \mathbf{b}$$

其中 $\mathbf{\Lambda}$ 是一个稀疏的带状矩阵，$\odot$ 是 Hadamard 逐元素乘积。

### 关键设计

1. **秩-1 分解降参数**: 原始二次变换中每个输出维度对应一个 $n \times n$ 的矩阵 $\mathbf{V}_i$，导致 $O(dn^2)$ 参数。将每个 $\mathbf{V}_i$ 约束为秩-1 矩阵 $\mathbf{V}_i = \mathbf{p}_i\mathbf{q}_i^\top$，可将参数量降至 $O(2dn)$。此时二次变换变为 $\mathbf{z} = (\mathbf{P}\mathbf{x}) \odot (\mathbf{Q}\mathbf{x}) + \mathbf{W}\mathbf{x} + \mathbf{b}$。

2. **权重共享**: 进一步令 $\mathbf{P} = \mathbf{\Lambda}\mathbf{W}$，$\mathbf{Q} = \mathbf{W}$，即复用线性层的权重矩阵 $\mathbf{W}$。这带来两个好处：（a）参数量从 $3dn$ 降至 $dn + d^2$；（b）线性响应 $\tilde{\mathbf{y}} = \mathbf{W}\mathbf{x}$ 只需计算一次，减少计算冗余。

3. **$\mathbf{\Lambda}$ 的稀疏化**: $\mathbf{\Lambda} \in \mathbb{R}^{d \times d}$ 仍有 $O(d^2)$ 参数。通过将其限制为宽度为 $k$ 的带状矩阵（在左下和右上角补充小三角区域形成循环结构），参数量降至 $k \times d$。当 $k=1$ 时，额外参数相对于原始线性层仅为 $O(1/n)$。计算上，$\mathbf{\Lambda}\tilde{\mathbf{y}} = \sum_{r \in \mathcal{K}} \boldsymbol{\lambda}_r \odot \text{Roll}(\tilde{\mathbf{y}}, r)$，其中 Roll 是循环移位操作。

### 损失函数 / 训练策略

- 作者排除了移位 $r=0$ 的情况（即纯平方项 $\tilde{y}_i^2$），因为平方项相比交叉项 $\tilde{y}_i \tilde{y}_j$ 更容易出现数值不稳定（方差为 2 vs. 1，大值概率高数个数量级）。
- 实验中固定 $\mathcal{K} = \{1\}$，即只引入相邻神经元之间的二次交互。
- 训练时无需特殊损失函数，直接使用原任务的标准损失（如交叉熵）进行端到端训练。

## 实验关键数据

### 主实验 — 图像分类

| 模型 | 参数量 | ImageNet | Caltech | CIFAR-10 | CIFAR-100 | Pets | 平均 |
|------|--------|----------|---------|----------|-----------|------|------|
| ViT-M | 2.45M | 63.70 | 87.77 | 96.35 | 80.25 | 91.03 | 82.44 |
| ViT-M+QE | 2.47M | 65.30 | 90.32 | 97.09 | 82.59 | 91.88 | 83.91 |
| ViT-XT | 2.82M | 66.04 | 90.25 | 96.51 | 81.24 | 91.03 | 84.99 |
| ViT-XT+QE | 2.83M | 67.34 | 90.77 | 96.78 | 82.64 | 97.97 | 86.82 |
| ViT-T | 5.37M | 73.96 | 93.07 | 97.97 | 86.13 | 93.87 | 88.57 |
| ViT-T+QE | 5.40M | 75.15 | 94.03 | 98.03 | 86.88 | 94.95 | 89.27 |

### 主实验 — LLM 微调（常识推理）

| 模型 | 方法 | 参数 | BoolQ | HellaSwag | ARC-e | ARC-c | 平均 |
|------|------|------|-------|-----------|-------|-------|------|
| LLaMA-7B | LoRA/32 | 53.5M | 68.90 | 78.10 | 77.80 | 61.30 | 74.73 |
| LLaMA-7B | LoRA/16+QE | 27.6M | 69.69 | 87.11 | 79.41 | 63.99 | 77.85 |
| LLaMA3-8B | LoRA/32 | 54.0M | 70.80 | 91.70 | 84.20 | 71.20 | 80.79 |
| LLaMA3-8B | LoRA/32+QE | 54.7M | 74.92 | 95.02 | 89.85 | 79.60 | 85.46 |

### 消融实验

| 配置 | 参数 | ImageNet | CIFAR-10 | CIFAR-100 | 平均 | 说明 |
|------|------|----------|----------|-----------|------|------|
| ViT-M+QuadraNet | 2.53M | 61.17 | 95.81 | 79.08 | 79.41 | 使用3个独立权重矩阵 |
| ViT-M+SwiGLU | 2.58M | 63.25 | 96.76 | 80.58 | 81.13 | 门控线性单元 |
| ViT-M+QuadEnhancer | 2.47M | 65.30 | 97.09 | 82.59 | 82.40 | 最少参数，最优性能 |

### 关键发现

- QE 在**半参数** LoRA/16 下即可超越全参数 LoRA/32（LLaMA2-7B 提升 2.64%），说明二次交互的表达能力贡献巨大。
- 随模型和数据规模增大，QE 的增益从 0.07% 持续增长到 1.19%，显示出良好的 scaling 特性。
- Pets 数据集上 ViT-XT+QE 取得了 6.94% 的惊人提升，说明对细粒度分类任务效果尤为显著。

## 亮点与洞察

- **极致轻量**：$k=1$ 时额外参数仅为 $d$ 个标量，额外 FLOPs 仅为 $4d$，几乎零成本。
- **即插即用**：可无损应用于 ViT、GPT-2、LLaMA 等不同架构，兼容 LoRA 等参数高效微调方法。
- **数值稳定性设计精巧**：通过排除平方项、仅使用交叉项，避免了 FP16 精度下的溢出问题。

## 局限与展望

- 实验规模偏小（ViT-M/T 是很小的模型，WikiText-2 也是小型语料），需要在更大规模上验证。
- 仅探索了 $k=1$ 的情况，未系统研究不同带宽 $k$ 的影响。
- 缺乏理论分析说明二次增强器为何在何种条件下最有效。
- 训练时间对比显示早期阶段有一定开销，实际部署中的 latency 分析不够充分。

## 相关工作与启发

- 与 QuadraNet 和 SwiGLU 相比，QuadEnhancer 通过权重共享和稀疏化实现了更少参数下的更优效果。
- 这一思路可以推广到任何基于线性变换的神经网络层，包括卷积层（通过将卷积视为矩阵乘法）。
- 对 LoRA 等 PEFT 方法的增强尤为有价值，暗示二次交互可能是提升微调效果的通用手段。

## 评分

- 新颖性: ⭐⭐⭐⭐ 二次变换不是新概念，但通过权重共享+带状稀疏化的参数削减思路新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖三个任务（CV、NLP、LLM微调），有消融和scaling分析
- 写作质量: ⭐⭐⭐⭐⭐ 推导清晰，逐步化简的过程易于理解
- 价值: ⭐⭐⭐⭐ 即插即用的增强方案实用性强，但需要更大规模验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression](../../ICML2025/model_compression/fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](../../ACL2025/model_compression/flipping_kd_small_to_large.md)
- [\[NeurIPS 2025\] Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks](global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](../../ICML2025/model_compression/efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)
- [\[CVPR 2026\] Neural Differentiation in Deep Networks: A Theoretical Framework for Expressivity and Representational Diversity](../../CVPR2026/model_compression/neural_differentiation_in_deep_networks_a_theoretical_framework_for_expressivity.md)

</div>

<!-- RELATED:END -->
