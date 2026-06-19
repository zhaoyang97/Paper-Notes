---
title: >-
  [论文解读] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray
description: >-
  [NeurIPS 2025][医学图像][视觉语言对齐] 提出 RadZero 框架及核心组件 VL-CABS（基于相似度的视觉语言交叉注意力），在胸部X光上实现可解释的、细粒度的视觉语言对齐，支持零样本分类、定位和分割多任务。 现有痛点 现有痛点：领域现状：多模态模型在放射学中的视觉语言（VL）对齐取得了显著进展…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "视觉语言对齐"
  - "胸部X光"
  - "零样本"
  - "可解释性"
  - "交叉注意力"
---

# RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray

**会议**: NeurIPS 2025

**arXiv**: [2504.07416](https://arxiv.org/abs/2504.07416)

**代码**: [GitHub](https://github.com/deepnoid-ai/RadZero)

**领域**: 医学影像

**关键词**: 视觉语言对齐, 胸部X光, 零样本, 可解释性, 交叉注意力

## 一句话总结

提出 RadZero 框架及核心组件 VL-CABS（基于相似度的视觉语言交叉注意力），在胸部X光上实现可解释的、细粒度的视觉语言对齐，支持零样本分类、定位和分割多任务。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：多模态模型在放射学中的视觉语言（VL）对齐取得了显著进展，但现有方法存在关键不足：

**报告利用不充分**: 放射学报告结构复杂，现有方法难以有效利用其中的细粒度语义信息

**可解释性差**: 传统注意力概率可视化提供的解释有限，临床应用难以接受

**多任务能力弱**: 需要针对分类、定位、分割分别训练不同模型

**零样本泛化不足**: 对未见过的疾病类别泛化能力有限

## 方法详解

### 整体框架

RadZero 包含三个创新组件：(1) VL-CABS 交叉注意力机制;(2) LLM驱动的语义句子提取;(3) 多正样本对比训练。

### 关键设计

**1. VL-CABS (Vision-Language Cross-Attention Based on Similarity)**

- 核心思想: 计算文本嵌入与局部图像patch特征之间的**相似度**,而非注意力概率
- 相似度计算: $S(t, p) = \frac{\text{sim}(f_t, f_p)}{\tau}$, 其中 $f_t$ 是文本嵌入, $f_p$ 是图像patch特征
- 分类: 通过相似度概率进行零样本推理
- 定位/分割: 像素级 VL 相似度图直接提供空间定位信息

**2. LLM 辅助语义提取**

- 使用大语言模型将复杂放射学报告拆解为简洁的语义句子
- 每个句子描述一个独立的医学发现（如 "右肺下叶存在浸润影"）
- 减少冗余信息，提升匹配精度

**3. 多正样本对比学习**

- 一张图像可能对应多个正确的文本描述（多种发现）
- 传统对比学习仅考虑单一正样本对
- 本文使用多正样本 InfoNCE 损失: $\mathcal{L} = -\sum_{k \in P(i)} \log \frac{\exp(s_{ik}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$

**4. 预训练视觉编码器 + 可训练 Transformer 层**

- 冻结预训练视觉编码器（如 BiomedCLIP）
- 添加额外的可训练 Transformer 层处理高分辨率图像
- 高效参数策略，无需全量微调

### 损失函数 / 训练策略

- 多正样本对比损失 + KL散度正则化
- 两阶段训练: 先训练VL对齐，再微调分割头
- 数据: MIMIC-CXR 等公开胸片数据集

## 实验关键数据

### 主实验

零样本分类 (CheXpert 5×200, AUC):

| 方法 | Atelectasis | Cardiomegal. | Consolidat. | Edema | Pl. Effusion | 平均 |
|------|-------------|-------------|-------------|-------|-------------|------|
| BioViL | 72.5 | 85.3 | 78.1 | 82.6 | 88.2 | 81.3 |
| MedCLIP | 74.8 | 87.1 | 79.5 | 84.3 | 89.7 | 83.1 |
| CheXzero | 76.2 | 88.5 | 81.3 | 85.8 | 90.5 | 84.5 |
| RadZero | **79.5** | **90.8** | **84.2** | **88.1** | **92.3** | **87.0** |

零样本定位 (MS-CXR, mIoU / Pointing Game):

| 方法 | mIoU | Pointing Game |
|------|------|-------------|
| GradCAM | 18.5 | 52.3 |
| BioViL-T | 25.3 | 61.8 |
| MedKLIP | 28.7 | 65.2 |
| RadZero | **35.2** | **72.8** |

### 消融实验

各组件的贡献 (CheXpert AUC):

| 模型 | 零样本分类 | 零样本定位 | 零样本分割 |
|------|----------|----------|----------|
| 基础CLIP | 81.3 | 18.5 | 22.1 |
| + VL-CABS | 84.5 | 30.8 | 35.6 |
| + LLM语义提取 | 85.8 | 32.5 | 37.2 |
| + 多正样本对比 | **87.0** | **35.2** | **40.8** |

### 关键发现

1. VL-CABS 提供的相似度图在视觉上比传统注意力图更精确地定位病灶区域
2. LLM 语义提取使文本端特征更集中,分类AUC提升约1.3点
3. RadZero 在零样本分割上远超现有方法(40.8 vs 22.1 mIoU),说明细粒度对齐的优势
4. 模型具备开放词汇语义分割能力,可泛化到训练中未见的疾病描述

## 亮点与洞察

- **可解释性**: VL 相似度图提供了临床可理解的视觉解释，助于医生信任
- **多任务统一**: 一个模型同时支持分类、定位、分割,无需任务特定头
- **高分辨率处理**: 额外 Transformer 层有效利用高分辨率X光图像信息

## 局限与展望

1. 当前仅在胸部X光上验证,其他影像模态（CT、MRI）有待探索
2. LLM 语义提取的质量依赖于 LLM 的医学知识
3. 零样本分割虽优于基线但绝对值仍有提升空间
4. 多正样本策略在极稀少疾病上可能缺乏足够正样本

## 相关工作与启发

- **BioViL/BioViL-T** (Bannur et al.): 生物医学视觉语言预训练
- **CheXzero**: 零样本胸片诊断
- **CLIP**: 对比语言-图像预训练的基础工作

## 评分

- ⭐ 创新性: 8/10 — 基于相似度的交叉注意力设计巧妙
- ⭐ 实用性: 9/10 — 开源代码,直接服务临床场景
- ⭐ 写作质量: 8/10 — 实验全面,定性分析直观

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](../../CVPR2025/medical_imaging/enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)
- [\[NeurIPS 2025\] Sequential Attention-based Sampling for Histopathological Analysis](sequential_attention-based_sampling_for_histopathological_analysis.md)

</div>

<!-- RELATED:END -->
