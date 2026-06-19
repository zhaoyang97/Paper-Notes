---
title: >-
  [论文解读] Post-pre-training for Modality Alignment in Vision-Language Foundation Models
description: >-
  [CVPR 2025][多模态VLM][CLIP] 提出 CLIP-Refine，一种介于预训练和微调之间的"后预训练"方法，通过随机特征对齐（RaFA）和混合对比蒸馏（HyCD）两个技术，仅用 1 个 epoch 在小数据集上训练即可缩小 CLIP 的模态间隙并提升零样本性能。 CLIP 预训练的图像-文本编码器存在"模态…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "CLIP"
  - "模态对齐"
  - "后预训练"
  - "知识蒸馏"
  - "特征空间"
---

# Post-pre-training for Modality Alignment in Vision-Language Foundation Models

**会议**: CVPR 2025  
**arXiv**: [2504.12717](https://arxiv.org/abs/2504.12717)  
**代码**: [https://github.com/yshinya6/clip-refine](https://github.com/yshinya6/clip-refine)  
**领域**: 多模态VLM  
**关键词**: CLIP, 模态对齐, 后预训练, 知识蒸馏, 特征空间

## 一句话总结

提出 CLIP-Refine，一种介于预训练和微调之间的"后预训练"方法，通过随机特征对齐（RaFA）和混合对比蒸馏（HyCD）两个技术，仅用 1 个 epoch 在小数据集上训练即可缩小 CLIP 的模态间隙并提升零样本性能。

## 研究背景与动机

CLIP 预训练的图像-文本编码器存在"模态间隙（modality gap）"——图像特征和文本特征聚集在特征空间的不同区域，导致跨模态检索和分类等下游任务性能受限。

现有解决方案有两类缺陷：
- **预训练方法**：需要从头训练，依赖百万级数据集和大量 GPU，计算成本极高
- **微调方法**：虽计算量小，但会退化零样本迁移性能，因为它聚焦于特定目标任务

作者提出"后预训练（post-pre-training）"的新训练阶段，目标是：用轻量计算资源和小数据集（如 COCO Caption + 单GPU），改善现成预训练 CLIP 的模态对齐和零样本泛化能力。

核心挑战在于：直接最小化图文特征距离（$\mathcal{L}_{\text{align}}$）会破坏特征空间的均匀性（uniformity），导致泛化性能下降；而用对比学习则因小 batch size 导致灾难性遗忘。

## 方法详解

### 整体框架

CLIP-Refine 由两个组件组成：RaFA（随机特征对齐）和 HyCD（混合对比蒸馏），通过联合优化目标函数 $\min_{\theta_V, \theta_T} \mathcal{L}_{\text{RaFA}} + \mathcal{L}_{\text{HyCD}}$ 进行后预训练。整个过程仅需 1 个 epoch，可在单张 A100 GPU 上完成。

### 关键设计

1. **随机特征对齐（RaFA）**:
    - 功能：间接缩小模态间隙，同时保持特征空间均匀性
    - 核心思路：不直接最小化图像-文本特征距离，而是让两种模态的特征都向一个共享先验分布 $p(z) = \mathcal{N}(0, I)$ 靠拢。具体做法是对每对图文样本采样一个随机参考向量 $z_{\text{ref}}^i \sim p(z)$，然后最小化 $\mathcal{L}_{\text{RaFA}} = \frac{1}{2B}\sum_{i=1}^B \|z_{\text{img}}^i - z_{\text{ref}}^i\|_2^2 + \|z_{\text{txt}}^i - z_{\text{ref}}^i\|_2^2$
    - 设计动机：直接对齐正样本对（$\mathcal{L}_{\text{align}}$）会破坏均匀性。通过引入共享先验分布的中间桥梁，可以同时实现三个目标：(i) 间接缩小模态间隙（通过共享 $z_{\text{ref}}$），(ii) 引导特征分布趋向均匀分布（标准高斯在超球面上近似均匀），(iii) 随机性防止过拟合

2. **混合对比蒸馏（HyCD）**:
    - 功能：在学习新知识的同时保留预训练模型的旧知识
    - 核心思路：以预训练 CLIP 为教师模型进行自蒸馏，但关键改进是将教师输出与真实标签通过 alpha 混合生成"混合软标签"：$\hat{q}_{i,j}^{I \to T} = \alpha \mathbb{I}_{i=j} + (1-\alpha) q_{i,j}^{I \to T}$，然后最小化学生与混合标签之间的 KL 散度
    - 设计动机：纯蒸馏（Self-KD）会过度约束参数到预训练值，阻碍 RaFA 的学习。混入真实标签（$\alpha=0.5$）让模型在保留教师暗知识的同时，能从正确的图文配对中学到新的跨模态对齐知识

3. **后预训练范式设计**:
    - 功能：定义一个轻量训练阶段
    - 核心思路：以小规模图文数据集（COCO Caption ~118K 对）训练现成预训练模型仅 1 个 epoch，学习率 $1.0 \times 10^{-6}$，batch size 512
    - 设计动机：预训练太贵，微调会退化零样本能力，后预训练填补了这一空白

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{\text{RaFA}} + \mathcal{L}_{\text{HyCD}}$，等权重组合（实验证明等权最优）。

- $\mathcal{L}_{\text{RaFA}}$：L2 距离到随机参考向量
- $\mathcal{L}_{\text{HyCD}} = \frac{1}{2}(\mathcal{L}_{\text{HyCD}}^{I \to T} + \mathcal{L}_{\text{HyCD}}^{T \to I})$，KL 散度形式
- 训练器：AdamW，学习率 $1 \times 10^{-6}$，单 epoch
- 默认先验 $p(z) = \mathcal{N}(0, I)$，$\alpha = 0.5$

## 实验关键数据

### 主实验（零样本分类，12个数据集，ViT-B/32）

| 方法 | 指标 (Avg Top-1 Acc) | ImageNet | Aircraft | Bird | 说明 |
|--------|------|------|----------|------|------|
| Pre-trained | 52.74 | 59.04 | 18.81 | 49.37 | 基线 |
| Contrastive | 45.75 | 52.96 | 13.98 | 40.07 | 灾难性遗忘，性能大跌 |
| m²-mix | 46.48 | 53.58 | 14.64 | 40.76 | 同样退化 |
| Self-KD | 52.94 | 59.06 | 18.96 | 51.52 | 仅保留旧知识 |
| **CLIP-Refine** | **54.69** | **60.93** | **20.77** | **52.72** | **平均提升+1.95** |

### 零样本检索（COCO2017-Val, ViT-B/32）

| 方法 | T→I R@1 | I→T R@1 | 说明 |
|------|---------|---------|------|
| Pre-trained | 30.56 | 33.26 | 基线 |
| Contrastive | 34.88 | 31.86 | T→I提升但I→T退化 |
| **CLIP-Refine** | **37.64** | **38.78** | 双向均大幅提升 |

### 消融实验

| 配置 | Avg ZS Cls. | 说明 |
|------|---------|------|
| HyCD only | 53.13 | 仅蒸馏，小幅提升 |
| HyCD + $\mathcal{L}_{\text{align}}$ | 45.61 | 直接对齐，严重退化 |
| CLIP-Refine (RaFA + HyCD) | **54.69** | 组合最优 |
| RaFA 无随机性 (β=0) | 53.79 | 随机性重要 |
| RaFA 标准高斯 (β=1) | **54.69** | 默认最优 |
| RaFA 过大方差 (β=100) | 53.59 | 方差过大退化 |

### 关键发现

- **对比学习在后预训练中严重退化**：因为小 batch size（512 vs 预训练的 32,768）导致负样本不足，引发灾难性遗忘。图像编码器比文本编码器更容易过拟合
- **直接对齐 + 蒸馏是最差组合**：$\mathcal{L}_{\text{align}}$ 压缩特征空间的均匀性，即使配合蒸馏也无法补救
- **CLIP-Refine 在特征空间分析中同时改善了模态间隙（0.79 vs 1.33）、对齐性（1.28 vs 1.37）和均匀性（0.049 vs 0.089）**
- 数据质量比数据量更重要：COCO Caption 优于规模大 10 倍的 CC3M/CC12M

## 亮点与洞察

- **"后预训练"是一个优雅的训练范式创新**：填补了预训练与微调之间的空白，概念清晰且实用，可与任何预训练模型和微调方法组合
- **RaFA 的间接对齐思想巧妙**：通过引入共享随机参考向量来桥接两种模态，避免了直接对齐的破坏性，并自然维持均匀性
- **HyCD 的标签混合策略简洁有效**：一个 $\alpha$ 参数平衡新旧知识，无需复杂的课程学习或渐进式训练

## 局限与展望

- 目前仅在 CLIP（对比学习模型）上验证，对 SigLIP 等 Sigmoid 损失模型的适用性有待验证
- 后预训练数据集质量敏感，对嘈杂数据集效果受限（除非做额外过滤）
- 仅用 1 个 epoch，是否存在最优训练 epoch 未被探索
- 可以尝试将 RaFA 与更高级的分布匹配方法（如 MMD、Sinkhorn）结合

## 相关工作与启发

- 与 prompt tuning（CoOp、MaPLe）互补：后预训练改善基础模型，prompt tuning 在此基础上做任务适配
- RaFA 的随机特征正则化灵感来自单模态微调领域（R3F, Random feature regularization），本文将其创造性地扩展到跨模态对齐
- 模态间隙问题的理论分析（Qian et al.）证明对比学习无法完全消除模态间隙，为后预训练的必要性提供了理论支撑

## 评分
- 新颖性: ⭐⭐⭐⭐ "后预训练"概念新颖，RaFA 设计巧妙但核心技术（蒸馏+正则）比较成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 12个分类+2个检索数据集，多种预训练模型，详尽的消融和特征空间分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，动机到方法到实验的逻辑连贯，图表精美
- 价值: ⭐⭐⭐⭐ 实用性强，可直接应用于现有 CLIP 模型的改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)
- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)
- [\[CVPR 2025\] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees](smartclip_modular_vision-language_alignment_with_identification_guarantees.md)
- [\[ACL 2025\] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](../../ACL2025/multimodal_vlm/single-to-mix_modality_alignment_with_multimodal_large_language_model_for_docume.md)

</div>

<!-- RELATED:END -->
