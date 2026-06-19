---
title: >-
  [论文解读] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models
description: >-
  [ICML2025][多模态VLM][CLIP] 提出基于核函数的无监督嵌入对齐方法（KUEA），通过在核空间中对齐 CLIP 与 DINOv2 的视觉表示，仅用图像数据微调即可增强 CLIP 的细粒度感知能力，同时保持与文本编码器的兼容性，提升下游 MLLM 性能。 CLIP 通过全局图文对比学习获得了强大的零样本能力…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "CLIP"
  - "DINOv2"
  - "核方法对齐"
  - "视觉表示增强"
  - "零样本分类"
  - "MLLM"
---

# Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models

**会议**: ICML2025  
**arXiv**: [2506.02557](https://arxiv.org/abs/2506.02557)  
**代码**: [peterant330/KUEA](https://github.com/peterant330/KUEA)  
**领域**: 视觉表示 / 多模态VLM  
**关键词**: CLIP, DINOv2, 核方法对齐, 视觉表示增强, 零样本分类, MLLM

## 一句话总结

提出基于核函数的无监督嵌入对齐方法（KUEA），通过在核空间中对齐 CLIP 与 DINOv2 的视觉表示，仅用图像数据微调即可增强 CLIP 的细粒度感知能力，同时保持与文本编码器的兼容性，提升下游 MLLM 性能。

## 研究背景与动机

CLIP 通过全局图文对比学习获得了强大的零样本能力，但其视觉编码器在细粒度感知（颜色、空间关系、计数等）方面存在明显不足，这一缺陷会传递到以 CLIP 作为视觉编码器的下游 MLLM（如 LLaVA、OpenFlamingo）。

DINOv2 等视觉自监督模型在像素级细节捕获上远优于 CLIP，但其特征空间与文本编码器不兼容。已有方法面临两难困境：

- **多编码器融合**（Eagle、Cambrian 等）：计算开销大
- **自监督微调 CLIP**（MaskCLIP 等）：破坏图文对齐，丧失零样本能力
- **重训练 CLIP**：成本极高，且产生的新嵌入与下游模型不兼容

本文的核心洞察是：CLIP 中语义相似但视觉差异大的样本在特征空间中高度聚集，而 DINOv2 能够区分这些样本。如果能在保持特征空间宏观结构的前提下，让 CLIP 学习 DINOv2 的**样本间相对关系**，就能兼得两者优势。

## 方法详解

### 核心思想：核空间对齐

不直接对齐两个模型的特征向量（因维度和空间结构差异极大），而是对齐它们的**核矩阵**——即样本间的相对相似度结构。

### 核函数定义

采用归一化多项式核（normalized polynomial kernel）：

$$k_{\text{poly}(\gamma,c,d)}(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^T \mathbf{y} + c)^d$$

归一化操作：

$$\tilde{k}(\mathbf{x}, \mathbf{y}) = \frac{k(\mathbf{x}, \mathbf{y})}{\sqrt{k(\mathbf{x}, \mathbf{x}) \cdot k(\mathbf{y}, \mathbf{y})}}$$

默认采用 degree=3 的多项式核。DINOv2 侧核参数固定（$\gamma=1/d_{emb}$, $c=1$），CLIP 侧核参数设为可训练。

### 对齐目标

对每对随机采样的图像 $(I_i, I_j)$，最小化两个核函数值的差异：

$$\min_\theta \; \mathbb{E}_{I_i, I_j \sim \mathcal{D}} \left[ \left( k_1(f_\theta(I_i), f_\theta(I_j)) - k_2(g(I_i), g(I_j)) \right)^2 \right]$$

其中 $f_\theta$ 为 CLIP 视觉编码器（可训练），$g$ 为 DINOv2（冻结）。

**关键理论保证**（Proposition 3.1）：基于 Hoeffding 不等式证明梯度的无偏估计可通过有限样本对实现，因此支持随机梯度优化，可扩展到大规模数据。

### 正则化保持图文对齐

为防止微调后的特征偏离原始 CLIP 嵌入过远，加入 L2 正则项：

$$\mathcal{L} = w \cdot \mathcal{L}_{\text{kernel}} + \mathbb{E}_{I_i} \left[ \| f_\theta(I_i) - f_{\theta_0}(I_i) \|_2^2 \right]$$

**理论保证**（Proposition 3.2）：若 $\|f_\theta(I) - f_{\theta_0}(I)\|_2 \leq \lambda$，则图文余弦相似度变化上界为 $2\lambda / \max\{\|f_\theta(I)\|_2, \|f_{\theta_0}(I)\|_2\}$，即正则化直接约束了图文对齐的偏移量。

### 训练配置

- 训练数据：ImageNet-1K 训练集（128 万张图像，**仅图像，无文本**）
- 硬件：2× RTX 4090，ViT-L-14 对齐约 30 小时
- 文本编码器完全冻结，仅微调视觉编码器

## 实验关键数据

### 零样本分类（12 个数据集平均准确率）

| 模型 | 原始 CLIP | Projection | Feature | DIVA | **Kernel (Ours)** |
|------|-----------|------------|---------|------|-------------------|
| ViT-B-16 | 61.22 | 54.17 | 61.41 | — | **62.04** (+0.82) |
| ViT-L-14 | 65.26 | 54.07 | 65.73 | 65.45 | **66.54** (+1.28) |
| ViT-L-14-336 | 66.10 | 54.53 | 66.52 | 65.30 | **67.13** (+1.03) |

Projection 方式导致零样本性能严重下降（-7~12%），直接特征对齐改进微弱，只有核空间对齐能在保持兼容性的同时有效提升。

### 图文检索（Flickr30K / COCO）

对齐后检索性能不降反升。ViT-L-14-336 在 Flickr30K Image→Text R@1 从 83.0 提升至 84.6，Text→Image R@1 从 64.78 提升至 67.08，证明图文对齐被有效保留。

### 细粒度感知任务（Linear Probe）

| 任务 | ViT-L-14 原始 | ViT-L-14 +align | 提升 |
|------|-------------|----------------|------|
| SVHN | 65.20 | **69.39** | +4.19 |
| GTSRB | 72.94 | **74.51** | +1.57 |
| CLEVR Distance | 22.97 | **30.82** | +7.85 |
| CLEVR Counts | 41.25 | **49.67** | +8.42 |

在计数和空间推理任务上提升尤为显著（+5.51% 平均），验证了核对齐有效传递了 DINOv2 的细粒度视觉感知能力。

### 下游 MLLM 提升

**LLaVA-1.5-7B**（ViT-L-14-336）：
- 仅替换视觉编码器（不调 LLM）：平均 60.06→60.88（+0.82）
- 替换 + LoRA 微调 LLM：平均 60.18→**61.70**（+1.52），定位任务平均提升 3.05%

**OpenFlamingo-3B**（4-shot）：
- 平均 45.54→**46.16**（+0.62），无需微调 LLM

## 亮点与洞察

1. **核空间对齐而非特征空间对齐**：绕开了两个模型特征空间结构差异巨大的问题，只对齐样本间相对关系，保持了特征空间的宏观结构
2. **仅需图像数据的轻量微调**：无需图文对、无需重训练文本编码器，在 ImageNet-1K 上 2 块 4090 即可完成
3. **即插即用，兼容下游模型**：对齐后的视觉编码器可直接替换 LLaVA/OpenFlamingo 中的原始 CLIP，无需重做视觉-语言对齐
4. **理论保证完备**：Proposition 3.1 证明了随机优化的可行性，Proposition 3.2 给出了图文对齐偏移的显式上界
5. **泛化性强**：除 CLIP+DINOv2 外，还验证了 SigLIP、DFN、MetaCLIP 等模型对的有效性

## 局限与展望

1. **数据规模受限**：仅在 ImageNet-1K（128 万图像）上微调，更大规模数据（如 DataComp）可能带来更大提升
2. **仅验证小模型**：MLLM 评估限于 LLaVA-7B 和 OpenFlamingo-3B，未扩展到 70B 级别
3. **对齐对象单一**：主要聚焦 CLIP↔DINOv2，对其他视觉基础模型（如 BEiT、MAE）的对齐效果未充分探索
4. **提升幅度相对温和**：零样本分类平均提升约 1%，对于某些应用场景可能不够显著
5. **未探索 token 级对齐**：当前方法仅对齐 CLS token 级嵌入，patch token 级的对齐可能进一步提升密集预测性能

## 评分

- 新颖性: ⭐⭐⭐⭐ — 核空间对齐的视角很新颖，避免了直接特征对齐的困难
- 实验充分度: ⭐⭐⭐⭐ — 覆盖零样本分类/检索/细粒度感知/MLLM 多个维度，消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，动机阐述有力
- 价值: ⭐⭐⭐⭐ — 轻量级、即插即用的方案有很强的实用性，为 MLLM 视觉增强提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ACL 2025\] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](../../ACL2025/multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](../../ICCV2025/multimodal_vlm/multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](../../NeurIPS2025/multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[CVPR 2026\] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](../../CVPR2026/multimodal_vlm/tipsv2_patch_text_alignment.md)

</div>

<!-- RELATED:END -->
