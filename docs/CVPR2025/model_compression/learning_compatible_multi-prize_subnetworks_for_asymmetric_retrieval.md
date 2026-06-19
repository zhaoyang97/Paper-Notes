---
title: >-
  [论文解读] PrunNet: Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval
description: >-
  [CVPR 2025][模型压缩][可剪枝网络] 提出 PrunNet（可剪枝网络），通过为每个权重学习重要性分数并结合冲突感知梯度集成，训练一个可以在任意容量（20%-100%）下产生兼容子网络的统一模型，在 GLDv2 上 46.29 mAP 超越密集网络基线，且所有容量子网络间特征兼容。 领域现状：非对称检索（Asym…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "可剪枝网络"
  - "非对称检索"
  - "子网络兼容性"
  - "冲突感知梯度"
  - "训练后剪枝"
---

# PrunNet: Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval

**会议**: CVPR 2025  
**arXiv**: [2504.11879](https://arxiv.org/abs/2504.11879)  
**代码**: [https://github.com/Bunny-Black/PrunNet](https://github.com/Bunny-Black/PrunNet)  
**领域**: 模型压缩 / 非对称检索  
**关键词**: 可剪枝网络, 非对称检索, 子网络兼容性, 冲突感知梯度, 训练后剪枝

## 一句话总结

提出 PrunNet（可剪枝网络），通过为每个权重学习重要性分数并结合冲突感知梯度集成，训练一个可以在任意容量（20%-100%）下产生兼容子网络的统一模型，在 GLDv2 上 46.29 mAP 超越密集网络基线，且所有容量子网络间特征兼容。

## 研究背景与动机

**领域现状**：非对称检索（Asymmetric Retrieval）将大模型部署在服务器端做离线索引，小模型部署在边端做在线查询。两个模型的特征必须兼容（在同一空间中可以匹配）。

**现有痛点**：现有方法（如 SFSC）需要为每个容量级别分别训练一个兼容模型，N 个容量就需要 N 次训练。新设备上线需要重新训练。

**核心矛盾**：不同容量子网络需要兼容的特征空间，但容量差异导致学习到的表示不同——全局最优和局部最优可能冲突。

**切入角度**：用可学习分数给每个权重标注重要性，贪心剪枝保留 top-c% 的连接。用梯度冲突投影解决不同容量子网络间的优化冲突。

**核心 idea**：可学习重要性分数 + 冲突感知梯度 + 兼容约束 = 一次训练，任意容量子网络。

## 方法详解

### 关键设计

1. **可学习权重重要性分数**：每个权重 $w_{ij}^l$ 关联一个分数 $s_{ij}^l$，剪枝时保留分数 top-$c_i$% 的连接。分数在训练中与权重一起优化

2. **冲突感知梯度集成**：不同容量子网络的损失梯度可能冲突。当 $\mathbf{g}_i \cdot \mathbf{g}_j < 0$ 时，将 $\mathbf{g}_i$ 投影到 $\mathbf{g}_j$ 的正交方向：$\hat{\mathbf{g}}_i = \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{|\mathbf{g}_j|^2}\mathbf{g}_j$

3. **兼容约束**：$\mathcal{L}_{comp} = \|f_{dense}(x) - f_i(x)\|^2$，确保小容量子网络的特征与密集网络对齐

### 损失函数 / 训练策略

$\mathcal{L} = \sum_i \mathcal{L}_{CE}(f_i(x), y) + \lambda \mathcal{L}_{comp}$。迭代剪枝（IP）优于一次性剪枝（OSP）——小网络继承大网络的权重。需要 Adaptive BN 后处理。

## 实验关键数据

### 主实验

GLDv2 地标检索 mAP（20%/40%/60%/80%/100% 容量）：

| 方法 | 20% | 60% | 100% |
|------|-----|-----|------|
| SFSC | 42.45 | 43.72 | 44.47 |
| **PrunNet** | **45.61** | **46.05** | **46.29** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 无冲突投影 | 性能下降 |
| 一次性剪枝 (OSP) | 不如迭代剪枝 (IP) |
| 无兼容约束 | 子网络特征不兼容 |

### 关键发现
- PrunNet 在所有容量级别都超越独立训练的基线——共享训练反而比独立训练更好
- 冲突投影是关键——没有它不同容量的优化互相干扰
- 20% 容量仍达 45.61 mAP（密集 SFSC 的 44.47）——剪枝后甚至超越密集基线

## 亮点与洞察
- **一次训练无限部署**——无需为新设备重新训练，直接按容量剪枝
- **冲突投影的通用性**——这种多任务梯度冲突解决策略可以推广到任何多目标优化

## 局限与展望
- 非结构化剪枝在部分硬件上加速不理想
- 超参数 α 需要调参
- BN 层统计在不同子网络间不同，需要额外的 Adaptive BN

## 评分
- 新颖性: ⭐⭐⭐⭐ 可剪枝网络+冲突投影+兼容约束的组合有效
- 实验充分度: ⭐⭐⭐⭐ GLDv2/In-Shop/VeRi 多数据集
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 对多设备部署场景有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)
- [\[NeurIPS 2025\] Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](../../NeurIPS2025/model_compression/find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)
- [\[CVPR 2025\] Understanding Multi-layered Transmission Matrices](understanding_multi-layered_transmission_matrices.md)
- [\[ICML 2025\] GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration](../../ICML2025/model_compression/gptaq_efficient_finetuning-free_quantization_for_asymmetric_calibration.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](../../ACL2025/model_compression/more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)

</div>

<!-- RELATED:END -->
