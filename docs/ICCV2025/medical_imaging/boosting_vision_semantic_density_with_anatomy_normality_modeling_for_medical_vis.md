---
title: >-
  [论文解读] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training
description: >-
  [ICCV 2025][医学图像][视觉语言] 提出 ViSD-Boost 方法，通过疾病级视觉对比学习增强视觉语义、以及基于 VQ-VAE 的解剖正常性建模来放大异常信号，解决医学视觉语言预训练中视觉模态语义密度低导致的对齐偏差问题，在 15 个器官 54 种疾病的零样本诊断上达到 84.9% AUC。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "视觉语言"
  - "semantic density"
  - "anatomy normality"
  - "VQ-VAE"
  - "zero-shot diagnosis"
---

# Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training

**会议**: ICCV 2025  
**arXiv**: [2508.03742](https://arxiv.org/abs/2508.03742)  
**代码**: [alibaba-damo-academy/ViSD-Boost](https://github.com/alibaba-damo-academy/ViSD-Boost)  
**领域**: 医学影像 / 视觉语言预训练  
**关键词**: vision-language pre-training, semantic density, anatomy normality, VQ-VAE, zero-shot diagnosis

## 一句话总结

提出 ViSD-Boost 方法，通过疾病级视觉对比学习增强视觉语义、以及基于 VQ-VAE 的解剖正常性建模来放大异常信号，解决医学视觉语言预训练中视觉模态语义密度低导致的对齐偏差问题，在 15 个器官 54 种疾病的零样本诊断上达到 84.9% AUC。

## 研究背景与动机

视觉语言预训练（VLP）在自然图像领域取得了巨大成功，但在医学场景中效果不佳。作者发现核心瓶颈在于**语义密度差距**（semantic density gap）：

- **视觉端信噪比低**：CT 图像包含大量解剖内容，但与诊断相关的内容往往只占图像的极小部分（如膀胱结石可能占不到体积的千分之一）
- **文本端信噪比高**：诊断报告是图像观察的高度浓缩总结，富含诊断相关语义
- **对齐偏差**：将低语义密度的视觉表示与高语义密度的文本表示直接对齐，导致视觉注意力无法聚焦到感兴趣区域

这也解释了为什么大多数医学 VLP 尝试仍限于相对简单的 2D 胸部场景——在复杂的 3D 腹部场景中提取诊断相关视觉线索更加困难。

## 方法详解

### 整体框架

ViSD-Boost 建立在解剖级图像-报告对齐之上，包含两个关键步骤：（1）视觉语义增强——疾病级对比学习区分正常和异常解剖结构；（2）视觉语义密度提升——VQ-VAE 正常性建模放大异常信号。

### 关键设计

1. **解剖级图像-报告对齐（Anatomy-wise Alignment）**:

    - 使用全身分割模型将图像按器官结构解析：$X_i^I \to \{X_{i,j}^I; j=1,...,M\}$
    - 利用 Qwen LLM 将诊断报告分解为解剖级报告：$X_i^R \to \{X_{i,j}^R; j=1,...,M\}$
    - 视觉编码器用 ResNet 提取特征，文本编码器用预训练 BERT
    - 为视觉和文本 token 附加可学习 query token，通过交叉注意力聚合
    - 对比学习目标：$\arg\min -\frac{1}{B \times M}\sum_{i,j} \log \frac{e^{\langle Q_{i,j}^I, Q_{i,j}^R \rangle / \tau}}{\sum_k e^{\langle Q_{i,j}^I, Q_{k,j}^R \rangle / \tau}}$

2. **疾病级视觉对比学习（Disease-level Contrastive Learning）**:

    - 核心思想：正常样本属于同一类别应聚拢，异常样本特征各异应分散
    - 利用 LLM 分析报告自动提取器官级异常标签 $y \in \{0: \text{normal}, 1: \text{abnormal}\}^{B \times M}$
    - 对比损失设计：
        - 异常样本（$y_{i,j}=1$）：仅同一样本的不同增强视图为正样本对（避免不同异常被错误拉近）
        - 正常样本（$y_{i,j}=0$）：同一器官的所有正常样本互为正样本对
    - 使用动量编码器生成正样本对的特征 $Q_{i,j}^{I'}$，防止退化解
    - 在 VLP 训练之前独立执行此步骤

3. **解剖正常性建模（Anatomical Normality Modeling）**:

    - 基于 VQ-VAE 在潜在空间学习每种解剖结构的正常分布
    - 两大关键创新：
        - **多分布学习**：CT 图像包含数十种解剖结构，引入解剖条件 token $A_j$ 指导 VQ-VAE 针对特定解剖结构重建
        - **潜在空间建模**：不在图像空间而在潜在空间训练，提高效率并在高级语义空间编码
    - Transformer-based 编码器 $\varphi_E$ 和解码器 $\varphi_D$，离散码本 $e \in \mathbb{R}^{M \times K \times C}$
    - 仅用正常样本训练：$\mathbb{I}_{y_{i,j}=0} \cdot (\|f_{i,j}^I - \varphi_D(e_{j,k})\|_2^2 + \beta\|\text{sg}[e_{j,k}] - \varphi_E(f_{i,j}^I; A_j)\|_2^2)$
    - 异常样本的分布偏移导致重建质量下降 → 重建误差作为异常性指标

4. **异常语义感知模块（Abnormality Semantic Perception）**:

    - 将原始 embedding $f_{i,j}^I$ 与 VQ-VAE 重建的正常 embedding $q_{i,j}^I$ 拼接
    - 通过 MLP 感知两者差异，放大异常成分
    - 输出 $\hat{f}_{i,j}^I$ 替换原始 embedding 进行 VLP 训练
    - 设计动机：重建误差直接关联诊断相关信息

### 损失函数 / 训练策略

- 三阶段：（1）疾病级对比学习预训练视觉编码器；（2）VQ-VAE 正常性建模训练；（3）VLP 对齐训练
- VQ-VAE 码本通过指数移动平均策略更新
- 超参数 $\beta = 0.25$

## 实验关键数据

### 主实验（CT-RATE 胸部零样本）

| 方法 | Precision | ACC | F1 | AUC |
|------|-----------|-----|-----|-----|
| CT-CLIP | 32.6 | 66.9 | 70.8 | 73.3 |
| BIUD | 33.8 | 68.1 | 71.6 | 71.3 |
| Merlin | 33.7 | 67.2 | 70.9 | 72.8 |
| fVLM | 37.9 | 71.8 | 75.1 | 77.8 |
| **ViSD-Boost** | **38.7** | **73.1** | **75.9** | **79.0** |

腹部场景（MedVL-CT69K，15器官54疾病零样本）：

| 方法 | 平均 SE | 平均 SP | 平均 AUC |
|------|---------|---------|----------|
| Supervised | 62.0 | 76.2 | 73.3 |
| CLIP | 65.5 | 68.0 | 68.4 |
| fVLM | 67.9 | 72.5 | 74.5 |
| **ViSD-Boost** | **72.4** | **74.5** | **78.5** |

### 消融实验

| 配置 | 胸部 AUC | 腹部 AUC | 说明 |
|------|----------|----------|------|
| Baseline（仅对齐） | 77.8 | 74.5 | fVLM 基线 |
| + 疾病级对比学习 | 78.2 | 76.8 | 增强正常/异常区分 |
| + VQ-VAE 正常性建模 | 78.6 | 77.5 | 放大异常信号 |
| + 两者结合 | **79.0** | **78.5** | 完整 ViSD-Boost |

### 关键发现

- 在腹部复杂场景中提升尤为显著（相比 fVLM 提升 4.0% AUC），验证了语义密度提升对 3D 场景的重要性
- 疾病级对比学习的设计至关重要：正常样本聚拢 + 异常样本分散优于传统实例级对比学习
- VQ-VAE 仅用正常样本训练，通过分布偏移自然检测异常，无需异常标注
- 在外部验证集 Rad-ChestCT 上同样表现优异，说明泛化性好

## 亮点与洞察

- **语义密度概念**：首次明确提出医学VLP中视觉语义密度问题，为领域提供了新的理论视角
- **正常性建模巧妙利用分布偏移**：不是直接学习异常特征，而是通过建模"正常是什么样"的分布，让异常通过重建误差自然涌现
- **LLM自动标注**：利用LLM分析报告自动提取器官级正常/异常标签，避免人工标注成本
- **跨场景验证**：在胸部和腹部两个完全不同的CT场景中都取得了SOTA，说明方法的通用性

## 局限与展望

- 依赖全身分割模型的质量，分割错误会影响后续所有步骤
- VQ-VAE 码本大小和解剖条件 token 设计可能需要针对不同场景调优
- 目前仅验证了 CT 模态，X-ray、MRI 等其他模态的效果未知
- 三阶段训练流程较为复杂，端到端训练方案值得探索

## 相关工作与启发

- 与传统视觉表示增强方法不同，ViSD-Boost 从语义密度角度出发，既具有疾病级语义又保持泛化性
- 正常性建模的思路与异常检测（anomaly detection）领域相通，但在 VLP 框架中的应用是全新的
- 多器官联合建模的设计可推广到其他多区域医学影像分析任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 语义密度概念新颖，正常性建模放大异常信号的思路独特
- 实验充分度: ⭐⭐⭐⭐ 胸部+腹部两大场景，54种疾病零样本评估，下游任务迁移验证
- 写作质量: ⭐⭐⭐⭐ 概念阐述清楚，但部分公式较密集
- 价值: ⭐⭐⭐⭐⭐ 15器官54疾病的零样本AUC达84.9%，具有重要临床应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](../../CVPR2025/medical_imaging/multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[ICCV 2025\] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)

</div>

<!-- RELATED:END -->
