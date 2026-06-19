---
title: >-
  [论文解读] Enhancing Target-unspecific Tasks through a Features Matrix
description: >-
  [ICML 2025][多模态VLM][CLIP] 提出 Features Matrix (FM) 方法，利用多个手工 prompt 模板从冻结 CLIP 中提取通用知识构成特征矩阵，通过对齐 unexpected features 与微调视觉特征来增强模型在目标无关任务（如 base-to-novel 泛化、跨数据集泛化、域泛化）上的表现。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "CLIP"
  - "Features Matrix"
  - "提示学习"
  - "Base-to-Novel Generalization"
  - "泛化能力"
---

# Enhancing Target-unspecific Tasks through a Features Matrix

**会议**: ICML 2025  
**arXiv**: [2505.03414](https://arxiv.org/abs/2505.03414)  
**代码**: 未公开  
**领域**: 多模态/视觉语言模型 (VLM), Prompt Learning  
**关键词**: CLIP, Features Matrix, Prompt Learning, Base-to-Novel Generalization, 泛化能力

## 一句话总结

提出 Features Matrix (FM) 方法，利用多个手工 prompt 模板从冻结 CLIP 中提取通用知识构成特征矩阵，通过对齐 unexpected features 与微调视觉特征来增强模型在目标无关任务（如 base-to-novel 泛化、跨数据集泛化、域泛化）上的表现。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：近年来 CLIP 等大规模视觉语言模型在零样本推理上取得了显著成功。Prompt learning 方法（如 CoOp、CoCoOp）通过学习可训练的 prompt 嵌入来适配下游任务，在 base 类上效果好，但在 novel 类上往往不如手工 prompt 的零样本 CLIP。

核心问题在于：**prompt tuning 容易过拟合于下游数据分布，导致模型丢失预训练时获得的通用泛化能力。** 例如 CoOp 在 novel 类上的准确率仅 67.96%，远低于零样本 CLIP 的 74.22%。即使 KgCoOp 等方法引入了正则化约束，其 novel 类表现（72.70%）仍低于零样本 CLIP。

作者认为原因在于：**单一手工 prompt 正则化无法充分挖掘和利用 CLIP 中多样的语义通用知识。**

## 方法详解

### 整体框架

FM 方法是一个通用的即插即用模块，可以兼容 CoOp、CoCoOp、MaPLe、PromptSRC 等现有 prompt learning 框架。

核心思路：
1. 使用 60 个不同的手工 prompt 模板（如 "a photo of one", "a picture of a", "a drawing of a" 等）输入冻结 CLIP 的文本编码器
2. 对每个类别，从所有模板中提取文本特征，形成一个 **Features Matrix**
3. 计算微调后的视觉特征与特征矩阵中各特征的匹配分数，形成 **Scores Matrix**

### 关键设计：Unexpected Features

从 Scores Matrix 中识别两类 "unexpected" 特征：

- **Designated unexpected features** $F^k_{un}$：当前类（标签 $k$）的指定特征中，余弦相似度分数排名靠后的（low-$\beta$）特征
- **Non-designated unexpected features** $F^{\hat{k}}_{un}$：非当前类特征中，余弦相似度分数排名靠前的（top-$\beta$）特征

这些 unexpected features 代表了模型容易混淆或忽略的通用语义信息。

### 损失函数

对比损失 $\mathcal{L}_{CL}$：

$$\mathcal{L}_{CL} = -\log \frac{\exp\{\cos(t_k, v^{tun})\}}{\exp\{\cos(t_k, v^{tun})\} + \exp\{\cos(t_{\hat{k}}, v^{tun})\}}$$

其中 $t_k \in F^k_{un}$，$t_{\hat{k}} \in F^{\hat{k}}_{un}$。

总损失为：

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \gamma \mathcal{L}_{CL}$$

$\mathcal{L}_{CE}$ 为标准的两模态对齐交叉熵损失，$\gamma$ 为超参数。

## 实验关键数据

### 主实验：Base-to-Novel 泛化（11 数据集平均）

| 方法 | Base | Novel | HM |
|------|------|-------|-----|
| CoOp | 82.69 | 63.22 | 71.66 |
| CoOp+DePT | 83.66 | 71.82 | 77.29 |
| **CoOp+Ours** | **81.15** | **74.66** | **77.79** |
| MaPLe | 82.28 | 75.14 | 78.55 |
| MaPLe+DePT | 84.85 | 74.82 | 79.52 |
| **MaPLe+Ours** | **84.45** | **76.53** | **80.32** |
| PromptSRC | 84.26 | 76.10 | 79.97 |
| **PromptSRC+Ours** | **85.70** | **77.35** | **81.32** |

### 关键发现

- FM 在 novel 类上提升显著：CoOp+FM 的 novel 准确率从 63.22% 提升到 74.66%，首次超越 CLIP 零样本水平
- FM 与 MaPLe、PromptSRC 结合后，在所有代表性 baseline 上一致超越 DePT（CVPR2024）
- 在 ImageNet 上，PromptSRC+FM 达到 75.07% HM，超越所有对比方法

## 亮点与洞察

1. **即插即用设计**：FM 作为通用模块可以无缝集成到现有文本或多模态 prompt learning 框架中
2. **重新挖掘手工 prompt 的价值**：利用多个手工 prompt 模板从冻结 CLIP 中深层挖掘语义信息
3. **Unexpected features 的概念巧妙**：通过关注模型容易混淆的特征对进行对比学习，有效保留通用知识
4. **特征矩阵无需训练**：FM 在预训练 CLIP 上一次性提取，不增加额外训练开销

## 局限与展望

- FM 使用 60 个 prompt 模板会增加推理时的计算开销（特征矩阵预计算可缓解）
- $\beta$ 和 $\gamma$ 等超参数需要调优
- 在某些 base 类上可能牺牲少量精度以换取 novel 类泛化
- 仅在分类场景下验证，对于检测/分割等下游任务的效果未知
- 特征矩阵的大小随类别数和模板数线性增长

## 相关工作

- CoOp/CoCoOp（文本 prompt 学习）
- MaPLe/PromptSRC（多模态 prompt 学习）
- DePT（CVPR2024，可迁移 prompt 微调）
- KgCoOp（知识引导的 prompt 约束）

## 评分

⭐⭐⭐⭐ — 方法简单有效，实验全面覆盖 11 个数据集和多个框架，novel 类改进显著。核心 idea 清晰易理解，但技术新颖性相对有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[NeurIPS 2025\] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](../../NeurIPS2025/multimodal_vlm/sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)
- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](../../ICLR2026/multimodal_vlm/enhancing_multi-image_understanding_through_delimiter_token_scaling.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[CVPR 2025\] Mimic In-Context Learning for Multimodal Tasks](../../CVPR2025/multimodal_vlm/mimic_in-context_learning_for_multimodal_tasks.md)

</div>

<!-- RELATED:END -->
