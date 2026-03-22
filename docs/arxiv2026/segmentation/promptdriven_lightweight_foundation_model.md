# Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains

**会议**: arXiv 2026  
**arXiv**: [2603.12624](https://arxiv.org/abs/2603.12624)  
**作者**: Guodong Sun, Qihang Liang, Xingyu Pan, Moyun Liu, Yang Zhang
**代码**: [https://github.com/MVME-HBUT/SAM_FTI-FDet](https://github.com/MVME-HBUT/SAM_FTI-FDet)  
**领域**: 语义分割 / 模型压缩/高效推理  
**关键词**: prompt-driven, lightweight, foundation, model, instance  

## 一句话总结
由于复杂的运行环境、结构重复的组件以及安全关键区域频繁的遮挡或污染，货运列车中准确的视觉故障检测仍然是智能交通系统维护的关键挑战。

## 背景与动机
Accurate visual fault detection in freight trains remains a critical challenge for intelligent transportation system maintenance, due to complex operational environments, structurally repetitive components, and frequent occlusions or contaminations in safety-critical regions.. Conventional instance segmentation methods based on convolutional neural networks and Transformers often suffer from poor generalization and limited boundary accuracy under such conditions.

## 核心问题
由于复杂的运行环境、结构重复的组件以及安全关键区域频繁的遮挡或污染，货运列车中准确的视觉故障检测仍然是智能交通系统维护的关键挑战。

## 方法详解

### 整体框架
- To address these challenges, we propose a lightweight self-prompted instance segmentation framework tailored for freight train fault detection.
- Our method leverages the Segment Anything Model by introducing a self-prompt generation module that automatically produces task-specific prompts, enabling effective knowledge transfer from foundation models to domain-specific inspection tasks.
- Experimental results show that our method achieves 74.6 $AP^{\text{box}}$ and 74.2 $AP^{\text{mask}}$ on the dataset, outperforming existing state-of-the-art methods in both accuracy and robustness while maintaining low computational overhead.

### 关键设计
1. **关键组件1**: Accurate visual fault detection in freight trains remains a critical challenge for intelligent transportation system maintenance, due to complex operational environments, structurally repetitive components, and frequent occlusions or contaminations in safety-critical regions.
2. **关键组件2**: Our method leverages the Segment Anything Model by introducing a self-prompt generation module that automatically produces task-specific prompts, enabling effective knowledge transfer from foundation models to domain-specific inspection tasks.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 实验结果表明，我们的方法在数据集上达到了 74.6 $AP^{\text{box}}$ 和 74.2 $AP^{\text{mask}}$，在准确性和鲁棒性方面优于现有的最先进方法，同时保持较低的计算开销。

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
