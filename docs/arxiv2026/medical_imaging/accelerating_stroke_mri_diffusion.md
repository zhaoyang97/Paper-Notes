# Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning

**会议**: arXiv 2026  
**arXiv**: [2603.13007](https://arxiv.org/abs/2603.13007)  
**作者**: Yamin Arefeen, Sidharth Kumar, Steven Warach, Hamidreza Saber, Jonathan Tamir
**代码**: 待确认  
**领域**: 医学图像 / 模型压缩/高效推理  
**关键词**: accelerating, stroke, mri, diffusion, probabilistic  

## 一句话总结
目的：开发一种利用扩散概率生成模型 (DPM) 加速 MRI 重建的数据高效策略，当只有有限的完全采样数据样本可用时，可以加快临床中风 MRI 的扫描时间。

## 背景与动机
Purpose: To develop a data-efficient strategy for accelerated MRI reconstruction with Diffusion Probabilistic Generative Models (DPMs) that enables faster scan times in clinical stroke MRI when only limited fully-sampled data samples are available.. Methods: Our simple training strategy, inspired by the foundation model paradigm, first trains a DPM on a large, diverse collection of publicly available brain MRI data in fastMRI and then fine-tunes on a small dataset from the target application using carefully selected learning rates and fine-tuning durations.

## 核心问题
目的：开发一种利用扩散概率生成模型 (DPM) 加速 MRI 重建的数据高效策略，当只有有限的完全采样数据样本可用时，可以在临床中风 MRI 中实现更快的扫描时间。方法：受基础模型范式的启发，我们的简单训练策略首先在 fastMRI 中的大量、多样化的公开脑部 MRI 数据上训练 DPM，然后使用精心选择的学习率和微调对来自目标应用程序的小数据集进行微调持续时间。

## 方法详解

### 整体框架
- Purpose: To develop a data-efficient strategy for accelerated MRI reconstruction with Diffusion Probabilistic Generative Models (DPMs) that enables faster scan times in clinical stroke MRI when only limited fully-sampled data samples are available.
- When applied to clinical stroke MRI, a blinded reader study involving two neuroradiologists indicates that images reconstructed using the proposed approach from $2 \times$ accelerated data are non-inferior to standard-of-care in terms of image quality and structural delineation.
- The proposed approach substantially reduces the need for large application-specific datasets while maintaining clinically acceptable image quality, supporting the use of foundation-inspired diffusion models for accelerated MRI in targeted applications.

### 关键设计
1. **关键组件1**: Purpose: To develop a data-efficient strategy for accelerated MRI reconstruction with Diffusion Probabilistic Generative Models (DPMs) that enables faster scan times in clinical stroke MRI when only limited fully-sampled data samples are available.
2. **关键组件2**: Methods: Our simple training strategy, inspired by the foundation model paradigm, first trains a DPM on a large, diverse collection of publicly available brain MRI data in fastMRI and then fine-tunes on a small dataset from the target application using carefully selected learning rates and fine-tuning durations.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 该方法通过受控快速 MRI 实验和盲法临床读者研究的临床中风 MRI 数据进行评估。
- 结果：DPM 在大约 4000 名具有非 FLAIR 对比的受试者上进行预训练，并在仅来自 20 名目标受试者的 FLAIR 数据上进行微调，其重建性能与使用更多目标域 FLAIR 数据跨多个加速因子训练的模型相当。
- 实验表明，适度的微调和降低的学习率可以提高性能，而微调不足或过度会降低重建质量。

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
