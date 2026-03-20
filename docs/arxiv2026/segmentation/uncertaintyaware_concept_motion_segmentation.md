# Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos

**会议**: arXiv 2026  
**arXiv**: [2603.00881](https://arxiv.org/abs/2603.00881)  
**作者**: Yu Luo, Guangyu Wei, Yangfan Li, Jieyu He, Yueming Lyu
**代码**: [https://github.com/qimingfan10/SMART](https://github.com/qimingfan10/SMART)  
**领域**: 语义分割 / 视频理解  
**关键词**: uncertainty-aware, concept, motion, segmentation, semi-supervised  

## 一句话总结
从 X 射线冠状动脉造影 (XCA) 序列中分割主冠状动脉对于冠状动脉疾病的诊断至关重要。
## 背景与动机
Segmentation of the main coronary artery from X-ray coronary angiography (XCA) sequences is crucial for the diagnosis of coronary artery diseases.. However, this task is challenging due to issues such as blurred boundaries, inconsistent radiation contrast, complex motion patterns, and a lack of annotated images for training.

## 核心问题
然而，由于边界模糊、辐射对比度不一致、运动模式复杂以及缺乏用于训练的注释图像等问题，这项任务具有挑战性。
## 方法详解

### 整体框架
- To address these challenges, we propose SAM3-based Teacher-student framework with Motion-Aware consistency and Progressive Confidence Regularization (SMART), a semi-supervised vessel segmentation approach for X-ray angiography videos.
- First, our method utilizes SAM3's unique promptable concept segmentation design and innovates a SAM3-based teacher-student framework to maximize the performance potential of both the teacher and the student.
- To address the issue of unreliable teacher predictions caused by blurred boundaries and minimal contrast, we further propose a progressive confidence-aware consistency regularization to mitigate the risk of unreliable outputs.

### 关键设计
1. **关键组件1**: First, our method utilizes SAM3's unique promptable concept segmentation design and innovates a SAM3-based teacher-student framework to maximize the performance potential of both the teacher and the student.
2. **关键组件2**: Second, we enhance segmentation by integrating the vessel mask warping technique and motion consistency loss to model complex vessel dynamics.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 对来自不同机构的三个 XCA 序列数据集进行的广泛实验表明，SMART 实现了最先进的性能，同时需要的注释显着减少，这使其对于标记数据稀缺的现实临床应用特别有价值。
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
