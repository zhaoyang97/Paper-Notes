# Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation

**会议**: arXiv 2026  
**arXiv**: [2603.06374](https://arxiv.org/abs/2603.06374)  
**作者**: Jonas Ernst, Wolfgang Boettcher, Lukas Hoyer, Jan Eric Lenssen, Bernt Schiele
**代码**: 待确认  
**领域**: 语义分割 / 3D视觉  
**关键词**: rewis3d, reconstruction, improves, weakly-supervised, semantic  

## 一句话总结
我们提出了 Rewis3d，这是一个框架，它利用前馈 3D 重建方面的最新进展来显着改进 2D 图像上的弱监督语义分割。
## 背景与动机
We present Rewis3d, a framework that leverages recent advances in feed-forward 3D reconstruction to significantly improve weakly supervised semantic segmentation on 2D images.. Obtaining dense, pixel-level annotations remains a costly bottleneck for training segmentation models.

## 核心问题
获得密集的像素级注释仍然是训练分割模型的昂贵瓶颈。
## 方法详解

### 整体框架
- We present Rewis3d, a framework that leverages recent advances in feed-forward 3D reconstruction to significantly improve weakly supervised semantic segmentation on 2D images.
- To address this, we introduce a novel approach that leverages 3D scene reconstruction as an auxiliary supervisory signal.

### 关键设计
1. **关键组件1**: Specifically, a dual student-teacher architecture enforces semantic consistency between 2D images and reconstructed 3D point clouds, using state-of-the-art feed-forward reconstruction to generate reliable geometric supervision.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 我们提出了 Rewis3d，这是一个框架，它利用前馈 3D 重建方面的最新进展来显着改进 2D 图像上的弱监督语义分割。
- 具体来说，双师生架构使用最先进的前馈重建来生成可靠的几何监督，从而强制 2D 图像和重建的 3D 点云之间的语义一致性。
- 大量实验表明，Rewis3d 在稀疏监督方面实现了最先进的性能，比现有方法高 2-7%，且不需要额外的标签或推理开销。
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
