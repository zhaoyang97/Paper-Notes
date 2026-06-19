---
title: >-
  [论文解读] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups
description: >-
  [ICML 2025][物理/科学计算] SOP 提出了一种将任意可微分模型转换为基于分组的自归因神经网络（SANN）的框架，通过端到端学习特征分组实现了在 SANN 中的 SOTA 性能，并从理论上证明了逐特征 SANN 的误差下界和分组 SANN 的零误差可达性。 自解释神经网络（SENN）通过将预测分解为可解释原子的线…
tags:
  - "ICML 2025"
  - "物理/科学计算"
---

# Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups

**会议**: ICML 2025  
**arXiv**: [2310.16316](https://arxiv.org/abs/2310.16316)  
**领域**: 可解释性  

## 一句话总结

SOP 提出了一种将任意可微分模型转换为基于分组的自归因神经网络（SANN）的框架，通过端到端学习特征分组实现了在 SANN 中的 SOTA 性能，并从理论上证明了逐特征 SANN 的误差下界和分组 SANN 的零误差可达性。

## 研究背景与动机

自解释神经网络（SENN）通过将预测分解为可解释原子的线性组合来提供保证的线性解释。自归因神经网络（SANN）是其中一类重要方法，使用特征子集作为可解释原子，预测可忠实地分解为特征子集贡献的线性组合。

然而，现有 SANN 面临严重的**性能-可解释性权衡**：
- **NAM**（Neural Additive Models）：使用单个特征，无法捕获特征相关性
- **BagNet**：依赖固定大小的图像块，缺乏灵活性
- **FRESH**：使用注意力选择单个子集，分组数量受限

这种权衡的根本原因是什么？能否通过更好的分组策略来克服？

## 方法详解

### 理论基础

#### 逐特征 SANN 的误差下界

论文形式化证明了逐特征 SANN 在处理高相关特征数据时存在不可避免的误差下界。

**定理 2.3（二项式的插入误差下界）**：对于 $d$ 维的多线性二项式多项式 $p(x) = \prod_{i \in S_1 \cup S_2} x_i + \prod_{j \in S_2 \cup S_3} x_j$：

$$\sum_{S \subseteq [d]} \text{InsErr}(G, \alpha, S) \geq D_{ins}(\hat{\lambda})$$

其中 $D_{ins}(\hat{\lambda})$ 是通过线性规划对偶可行点计算的下界。**该下界随维度 $d$ 指数增长**。

#### 分组 SANN 的零误差可达性

**定理 2.4（非正式）**：对于任意 $m$ 项多项式 $p$，使用至多 $m$ 个分组的 SANN 即可实现零插入和删除误差。

### SOP 框架

SOP 由三个组件构成：

$$f(x) = \sum_{i=1}^m \underbrace{\theta(\Gamma(x), x)_i}_{\text{分组选择器}} \cdot \underbrace{h(g_i \odot x)}_{\text{骨干预测器}}$$

其中 $\underbrace{g_i \in \Gamma(x)}_{\text{分组生成器}}$。

#### 分组生成器 $\Gamma$

使用多头自注意力模块为特征分配分数，并对每个注意力分布进行阈值截断（保留前 $\tau=20\%$ 的特征）：

$$\Gamma(x) = (g_1, \ldots, g_m) = \text{SoftSelfAttn}_{\tau=20\%}(h_e(x))$$

其中 $h_e$ 是编码器（通常取骨干模型的倒数第二层）。

#### 骨干预测器 $h$

使用预训练的高性能模型（**冻结参数**），对每个分组掩码后的输入做预测：

$$y_i = h(g_i \odot x), \quad i = 1, \ldots, m$$

#### 分组选择器 $\theta$

使用稀疏交叉注意力模块为每个分组分配权重：

$$\theta(\Gamma(x), x) = (c_1, \ldots, c_m) = \text{SparseCrossAttn}(C_h, z)$$

其中 $C_h$ 用目标类别的权重初始化，$z$ 为所有分组的最后隐状态。使用 **sparsemax** 替代 softmax 以产生稀疏的分组权重。

### 关键设计选择

- **二值化分组**：使用 $\{0,1\}$ 掩码避免信息泄露导致的不忠实解释
- **冻结骨干**：保持预训练模型的高性能，仅训练分组生成器和选择器
- **模型无关**：适用于任意可微分模型（ViT、CNN、BERT 等）

## 实验

### 主实验

| 类别 | 方法 | ImageNet-S ViT Err.↓ | IOU↑ | CosmoGrid CNN MSE↓ | MultiRC BERT Err.↓ |
|---|---|---|---|---|---|
| 骨干 | Backbone | 0.097 | - | 0.009 | 0.318 |
| Post-hoc | SHAP-F | 0.306 | 0.391 | 0.028 | 0.455 |
| Post-hoc | FG-F | 0.448 | 0.511 | 0.036 | 0.396 |
| SANN | BagNet | 0.501 | 0.314 | - | - |
| SANN | FRESH | 0.537 | 0.464 | - | 0.386 |
| **SANN** | **SOP** | **0.267** | **0.548** | **0.015** | **0.356** |

SOP 在所有 SANN 中取得最佳性能，且在 ImageNet-S 上甚至优于多数 post-hoc 方法。

### 科学发现应用

在宇宙学 CosmoGrid 数据集上，SOP 的分组和分数揭示了关于星系形成的新洞察。研究者可以通过检查分组来理解模型关注的具体特征（如星系密度、形态等）。

### 模型调试

SOP 可用于检测模型是否依赖正确/错误的特征（如物体 vs 背景），辅助模型调试。

## 亮点

- **理论贡献扎实**：形式化证明了逐特征 SANN 的根本性局限和分组 SANN 的零误差可达性
- **模型无关的框架**：可将任意预训练模型转化为 SANN，无需特定架构
- **端到端学习分组**：无需分组标签监督，分组自动适应数据相关性
- **跨模态验证**：在视觉（ViT）、科学（CNN）和语言（BERT）任务上均表现出色
- **实用价值**：在模型调试和科学发现中展示了实际应用

## 局限性

- 每个分组需要独立的骨干模型前向传播，推理成本随分组数 $m$ 线性增长
- 分组大小固定为 20%，可能不适合所有数据类型
- 二值化分组在梯度传播中需要特殊处理（缩放因子）
- 分组生成器的多头注意力增加了参数量
- 在 Mutag 等化学数据集上 SOP 的解释准确率低于某些 post-hoc 方法

## 评分

⭐⭐⭐⭐ (4/5)

理论与实践结合出色。证明了特征分组对 SANN 的根本性重要性，并提出了一个优雅的模型无关框架。在多个领域的实验验证了方法的通用性和实用性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[ICML 2025\] Compact Matrix Quantum Group Equivariant Neural Networks](compact_matrix_quantum_group_equivariant_neural_networks.md)
- [\[NeurIPS 2025\] AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings](../../NeurIPS2025/physics/astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)
- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](../../NeurIPS2025/physics/exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/physics/learning_fair_representations_with_kolmogorov-arnold_networks.md)

</div>

<!-- RELATED:END -->
