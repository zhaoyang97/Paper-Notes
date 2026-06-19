---
title: >-
  [论文解读] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation
description: >-
  [CVPR 2025][模型压缩][后训练量化] 提出 FIMA-Q，通过对角+低秩（DPLR）的 Fisher 信息矩阵近似替代传统对角近似，更准确地捕捉量化误差对输出分布的影响，在 3-bit 极低比特 ViT 量化中大幅超越现有方法（ViT-B 77.63% vs QDrop 74.75%）。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "后训练量化"
  - "Transformer"
  - "Fisher信息矩阵"
  - "对角低秩近似"
  - "低比特量化"
---

# FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation

**会议**: CVPR 2025  
**arXiv**: [2506.11543](https://arxiv.org/abs/2506.11543)  
**代码**: [https://github.com/ShiheWang/FIMA-Q](https://github.com/ShiheWang/FIMA-Q)  
**领域**: 模型压缩 / 量化  
**关键词**: 后训练量化, Vision Transformer, Fisher信息矩阵, 对角低秩近似, 低比特量化

## 一句话总结

提出 FIMA-Q，通过对角+低秩（DPLR）的 Fisher 信息矩阵近似替代传统对角近似，更准确地捕捉量化误差对输出分布的影响，在 3-bit 极低比特 ViT 量化中大幅超越现有方法（ViT-B 77.63% vs QDrop 74.75%）。

## 研究背景与动机

**领域现状**：后训练量化（PTQ）是压缩 Vision Transformer 的实用方案。主流方法 BRECQ 使用 Fisher 信息矩阵（FIM）的对角近似来衡量量化误差的影响，逐块重建最小化 KL 散度。

**现有痛点**：BRECQ 的对角 FIM 近似忽略了输出维度间的相关性——不同输出通道间的量化误差可能相互放大或抵消，对角近似无法捕捉这种交互效应。在极低比特（3-bit）时，量化误差大，这种近似误差被放大，导致性能急剧下降。

**核心矛盾**：精确 FIM 计算代价 $O(d^2)$ 不可接受，对角近似 $O(d)$ 又失去了关键的跨维度信息。

**切入角度**：用对角+低秩（DPLR）分解 $F \approx D + UU^\top$ 来近似 FIM——对角项保留各维度独立重要性，低秩项捕捉跨维度相关性。

**核心 idea**：用 DPLR 近似 FIM 替代纯对角近似 = 在低比特 ViT 量化中捕捉跨维度相关性，大幅提升精度。

## 方法详解

### 关键设计

1. **纠正 BRECQ 的 Hessian 近似误差**:

    - 功能：发现并修正现有方法的理论缺陷
    - 核心思路：揭示 BRECQ 中的 Hessian 对角矩阵实际上与 KL 梯度成线性比例（而非平方），这意味着其对角近似在数学上不自洽。FIMA-Q 回归正确的 FIM 定义重新推导
    - 设计动机：理论纠正带来实践提升——修正后的损失函数更准确反映量化误差的真实影响

2. **对角+低秩 (DPLR) 近似**:

    - 功能：以 $O(d \cdot r)$ 复杂度捕捉跨维度相关性
    - 核心思路：FIM 分解为 $F \approx D + UU^\top$，其中 $D$ 是对角项，$U \in \mathbb{R}^{d \times r}$ 是低秩因子。损失函数 $\mathcal{L}_{DPLR} = \alpha \mathcal{L}_{rank-k} + (1-\alpha) \mathcal{L}_{diag}$，低秩项从量化误差的外积中提取
    - 设计动机：纯对角丢失相关性，纯低秩丢失个体维度重要性，DPLR 结合两者优势

### 损失函数 / 训练策略

$\mathcal{L}_{DPLR} = \alpha \mathcal{L}_{rank-k} + (1-\alpha) \mathcal{L}_{diag}$，秩 $r$ 采用渐进增长策略。使用均匀量化器，1024 张校准图像，Adam 优化器逐块重建。

## 实验关键数据

### 主实验

ImageNet Top-1 准确率（W3A3 = 权重3bit/激活3bit）：

| 模型 | FIMA-Q | QDrop | RepQ-ViT |
|------|--------|-------|----------|
| ViT-B | **77.63** | 74.75 | 70.10 |
| DeiT-B | **76.54** | 72.97 | 75.11 |
| Swin-B | **78.82** | 76.57 | 72.36 |

### 关键发现
- **3-bit 提升最显著**：越低比特量化误差越大，跨维度相关性建模越重要
- **通用性**：在 ViT/DeiT/Swin 三种架构上都consistent提升

## 亮点与洞察
- **理论纠正带来实践突破**——揭示 BRECQ 的数学错误不只是学术意义，直接导致了 2-3% 的精度提升
- **DPLR 可推广**——该近似框架可用于任何需要 FIM 的场景（如持续学习的 EWC 正则化）

## 局限与展望
- 低秩计算增加重建阶段的计算开销
- 依赖校准数据质量
- 仅在视觉任务验证，NLP/语音领域待探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论贡献扎实（纠正前人错误 + DPLR 近似）
- 实验充分度: ⭐⭐⭐⭐ 三种 ViT 架构，多比特配置
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰
- 价值: ⭐⭐⭐⭐ 3-bit ViT 量化的新 SOTA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](../../ECCV2024/model_compression/adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers](l_swag_zero_shot_nas_vision_transformers.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)

</div>

<!-- RELATED:END -->
