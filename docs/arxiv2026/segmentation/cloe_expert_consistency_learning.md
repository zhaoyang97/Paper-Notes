# CLoE: Expert Consistency Learning for Missing Modality Segmentation

**会议**: arXiv 2026  
**arXiv**: [2603.09316](https://arxiv.org/abs/2603.09316)  
**作者**: Xinyu Tong, Meihua Zhou, Bowu Fan, Haitao Li
**代码**: 待确认  
**领域**: 语义分割 / 医学图像  
**关键词**: cloe, expert, consistency, learning, missing  

## 一句话总结
多模态医学图像分割在推理时经常面临模态缺失的问题，这会引起模态专家之间的分歧，并使融合不稳定，特别是在小的前景结构上。
## 背景与动机
Multimodal medical image segmentation often faces missing modalities at inference, which induces disagreement among modality experts and makes fusion unstable, particularly on small foreground structures.. We propose Consistency Learning of Experts (CLoE), a consistency-driven framework for missing-modality segmentation that preserves strong performance when all modalities are available.

## 核心问题
多模态医学图像分割在推理时经常面临模态缺失，这会引起模态专家之间的分歧，并使融合不稳定，特别是在小的前景结构上。我们提出了专家一致性学习（CLoE），这是一种一致性驱动的模态缺失分割框架，当所有模态可用时，它可以保持强大的性能。
## 方法详解

### 整体框架
- We propose Consistency Learning of Experts (CLoE), a consistency-driven framework for missing-modality segmentation that preserves strong performance when all modalities are available.
- CLoE formulates robustness as decision-level expert consistency control and introduces a dual-branch Expert Consistency Learning objective.

### 关键设计
待深读后补充。

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 在 BraTS 2020 和 MSD Prostate 上进行的大量实验表明，CLoE 在不完整的多模态分割方面优于最先进的方法，同时表现出强大的跨数据集泛化能力并提高了临床关键结构的稳健性。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
