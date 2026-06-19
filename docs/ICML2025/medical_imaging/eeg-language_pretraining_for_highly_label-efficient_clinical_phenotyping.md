---
title: >-
  [论文解读] EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping
description: >-
  [ICML 2025][医学图像][EEG-语言模型] 本文首创 EEG-语言模型（ELM），在15000份EEG记录和临床报告上训练，结合时间序列裁剪、文本分割和多实例学习策略，首次实现了EEG的零样本分类和跨模态检索，在低标注场景下病理检测性能显著优于纯EEG自监督方法。 医学神经成像领域（特别是脑电图EEG）在利用深度…
tags:
  - "ICML 2025"
  - "医学图像"
  - "EEG-语言模型"
  - "多模态对齐"
  - "多实例学习"
  - "临床表型"
  - "零样本分类"
  - "病理检测"
  - "自监督学习"
---

# EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping

**会议**: ICML 2025  
**arXiv**: [2409.07480](https://arxiv.org/abs/2409.07480)  
**代码**: 未公开  
**领域**: 脑电信号分析 / 多模态学习  
**关键词**: EEG-语言模型, 多模态对齐, 多实例学习, 临床表型, 零样本分类, 病理检测, 自监督学习

## 一句话总结

本文首创 EEG-语言模型（ELM），在15000份EEG记录和临床报告上训练，结合时间序列裁剪、文本分割和多实例学习策略，首次实现了EEG的零样本分类和跨模态检索，在低标注场景下病理检测性能显著优于纯EEG自监督方法。

## 研究背景与动机

医学神经成像领域（特别是脑电图EEG）在利用深度学习方面严重滞后于其他领域。尽管EEG在癫痫和睡眠障碍等病理检测中广泛应用，但可用的标注数据极为稀缺。

现有挑战包括：

**标注数据稀缺**：自监督学习（SSL）可通过无标注数据预训练来扩大训练规模，但现有方法受限于数据增强设计困难和低信噪比。

**跨模态信息未利用**：计算机视觉和放射学领域已证明自然语言可显著提升表征学习，但EEG-语言预训练尚未被探索。

**EEG报告的异质性**：临床报告通常跨越多个段落，包含与下游任务无关的信息，且不含时间信息标注。

**模态间错位**：EEG为长时间序列，报告为多段落文本，两者之间存在粒度不匹配的问题。

核心动机：将自然语言作为预训练信号引入EEG表征学习，同时解决EEG-文本对的异质性和错位问题。

## 方法详解

### 整体框架

ELM框架包含以下核心组件（Figure 1）：

- **EEG编码器 $f_e$**：残差卷积神经网络，将EEG片段投射为低维向量
- **语言编码器 $f_l$**：预训练MedCPT模型（冻结权重），编码临床文本
- **投影器 $g_e, g_l$**：将两模态映射到共享潜在空间

### 关键设计

#### 1. 子单元对齐策略

面对长时间序列和多段落报告的挑战，提出：
- **时间序列裁剪**：将EEG记录裁剪为多个非重叠片段（60秒/20秒）
- **文本分割**：利用正则表达式将报告按标题分割为临床病史、记录描述、用药和解读四类

#### 2. 三种对齐策略

- **ELM$_{e,l}$**（类CLIP）：EEG和文本分别投射到新的共享空间，使用InfoNCE损失
- **ELM$_l$**（类M-FLAG）：EEG投射到语言模型输出空间，损失包括对齐损失+正交损失：

$\mathcal{L}_{total} = \mathcal{L}_{align} + \mathcal{L}_{orth}$

其中 $\mathcal{L}_{align} = \|\hat{\mathbf{e}} - \hat{\mathbf{l}}\|_2^2$ 最小化嵌入差异，$\mathcal{L}_{orth}$ 促进EEG嵌入维度间的独立性。

#### 3. 多实例学习扩展（ELM-MIL）

核心创新——放松了每个EEG-文本对强对齐的假设：

- 对每个文本样本，采样多个正EEG片段近似 $P(e|l)$ 分布
- 对每个EEG片段，采样多个文本段近似 $P(l|e)$ 分布
- 双向对齐近似 $P(e,l)$

扩展InfoNCE损失至多实例：

$\mathcal{L}^{e|l} = -\frac{1}{B_l}\sum_{k=1}^{B_l} \log \frac{\frac{1}{|P_k|}\sum_{j \in P_k} \exp(s^{e2l}_{j,k}/\tau)}{\sum_{j=1}^{B_e} \exp(s^{e2l}_{j,k}/\tau)}$

温度参数 $\tau = 0.3$，每个主题采样最多 $N=32$ 个EEG片段和 $M=8$ 个文本段。

#### 4. 文本预处理

- 报告按标题分类为四类：临床病史、记录描述、用药信息、解读/印象
- 过滤无关信息（EEG系统信息、技术问题、免责声明等）
- 用Llama-3 8B本地生成单句摘要作为补充

### 损失函数

ELM-MIL最终损失：

$\mathcal{L}^{e,l} = \frac{1}{2}(\mathcal{L}^{e|l} + \mathcal{L}^{l|e})$

## 实验关键数据

### 主实验：TUAB病理检测（线性探测）

| 方法 | ZS(BAcc) | 1%(BAcc) | 10%(BAcc) | 100%(BAcc) | 1%(AUROC) | 100%(AUROC) |
|------|----------|----------|-----------|------------|-----------|-------------|
| Supervised | - | 71.36 | 81.06 | 84.13 | 79.87 | 91.83 |
| TS (EEG-only) | - | 74.99 | 82.16 | 84.10 | 82.51 | 91.50 |
| **ELM-MIL e,l** | **84.31** | 83.10 | 84.21 | **87.11** | **91.56** | **93.91** |
| ELM-MIL e\|l | 79.10 | **83.71** | 84.37 | 85.65 | 92.37 | 93.65 |

关键发现：
- **零样本分类**：ELM-MIL e,l达到84.31%平衡准确率，为该领域首次实现
- **低标注优势**：仅1%标注数据下，AUROC达91.54%，比最佳EEG-only方法（TS 82.51%）提升约9个百分点
- **100%标注**：AUROC 93.91%，显著超越全监督方法（91.83%）

### 跨数据集泛化（NMT数据集）

| 方法 | 1%(AUROC) | 10%(AUROC) | 100%(AUROC) |
|------|-----------|------------|-------------|
| TS | 64.90 | 81.36 | 87.08 |
| **ELM-MIL e,l** | **76.10** | **88.98** | **90.25** |

在来自巴基斯坦、不同采集设备的NMT数据集上，ELM-MIL仅用1%标注即超过TS使用10%标注的性能。

### 检索性能

从437名患者中进行EEG↔报告检索，ELM-MIL在Top-K检索准确率上显著优于其他方法，证明了跨模态对齐的成功泛化。

## 亮点与洞察

1. **首创EEG-语言预训练**：填补了功能性脑数据与文本信息跨模态学习的空白
2. **多实例学习的精妙设计**：放松强对齐假设，更好地处理EEG-文本对中不一致的相关性
3. **文本类型的重要性**：整合多个文本集群（病史+描述+解读+用药）效果最佳，说明不同信息源提供了互补知识
4. **子单元对齐的额外收益**：即使在报告随机打乱的情况下，子单元对齐策略也能促进主体间信息编码，表明该策略具有固有优势
5. **临床实用性**：训练和推理阶段均不需要临床报告，与标准EEG临床实践完全兼容

## 局限性

1. 数据集规模仍小于放射学和计算机视觉领域
2. 报告文本异质性高，信息质量参差不齐
3. 语言编码器冻结可能限制跨模态对齐的上限
4. 评估主要集中在病理检测，尚未扩展到更细粒度的临床任务

## 相关工作

- **EEG自监督学习**：BYOL、VICReg、ContraWR、RP、TS、CPC
- **医学多模态语言建模**：M-FLAG、CLIP、BiomedCLIP
- **多实例学习**：MIL-NCE（视频-文本对齐）

## 评分

⭐⭐⭐⭐ — 开创性地将多模态语言建模引入EEG领域，MIL扩展设计精巧且有效。实验全面覆盖了零样本、低标注、跨数据集等场景。在临床应用上极具价值，尤其是低标注场景下的大幅提升。不足在于数据规模和下游任务的多样性有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)
- [\[CVPR 2026\] From Panel to Pixel: Zoom-In Vision-Language Pretraining from Biomedical Scientific Literature](../../CVPR2026/medical_imaging/from_panel_to_pixel_zoom-in_vision-language_pretraining_from_biomedical_scientif.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2025/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[ICCV 2025\] GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology](../../ICCV2025/medical_imaging/gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)

</div>

<!-- RELATED:END -->
