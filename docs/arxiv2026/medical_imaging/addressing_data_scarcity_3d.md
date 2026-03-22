# Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding

**会议**: arXiv 2026  
**arXiv**: [2603.12514](https://arxiv.org/abs/2603.12514)  
**作者**: Shivam Chaudhary, Sheethal Bhat, Andreas Maier
**代码**: 待确认  
**领域**: 医学图像 / 目标检测  
**关键词**: addressing, data, scarcity, trauma, detection  

## 一句话总结
腹部 CT 扫描中创伤性损伤的准确检测和定位仍然是急诊放射学中的一个关键挑战，这主要是由于带注释的医疗数据严重缺乏。

## 背景与动机
Accurate detection and localization of traumatic injuries in abdominal CT scans remains a critical challenge in emergency radiology, primarily due to severe scarcity of annotated medical data.. This paper presents a label-efficient approach combining self-supervised pre-training with semi-supervised detection for 3D medical image analysis.

## 核心问题
腹部 CT 扫描中创伤性损伤的准确检测和定位仍然是急诊放射学中的一个关键挑战，这主要是由于带注释的医疗数据严重缺乏。

## 方法详解

### 整体框架
- This paper presents a label-efficient approach combining self-supervised pre-training with semi-supervised detection for 3D medical image analysis.
- We employ patch-based Masked Image Modeling (MIM) to pre-train a 3D U-Net encoder on 1,206 CT volumes without annotations, learning robust anatomical representations.
- For detection, semi-supervised learning with 2,000 unlabeled volumes and consistency regularization achieves 56.57% validation mAP@0.50 and 45.30% test mAP@0.50 with only 144 labeled training samples, representing a 115% improvement over supervised-only training.

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
- 对于检测，使用 2,000 个未标记卷和一致性正则化的半监督学习仅使用 144 个标记训练样本即可实现 56.57% 的验证 mAP@0.50 和 45.30% 的测试 mAP@0.50，比仅监督训练提高了 115%。

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
