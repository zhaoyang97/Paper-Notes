# 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video

**会议**: arXiv 2026  
**arXiv**: [2603.10125](https://arxiv.org/abs/2603.10125)  
**作者**: Jin Lyu, Liang An, Pujin Cheng, Yebin Liu, Xiaoying Tang
**代码**: 待确认  
**领域**: 视频理解 / 3D视觉  
**关键词**: 4dequine, disentangling, motion, appearance, equine  

## 一句话总结
4D reconstruction of equine family (e.g.

## 背景与动机
4D reconstruction of equine family (e.g.. horses) from monocular video is important for animal welfare.

## 核心问题
In this work, we propose a novel framework called 4DEquine by disentangling the 4D reconstruction problem into two sub-problems: dynamic motion reconstruction and static appearance reconstruction.

## 方法详解

### 整体框架
- In this work, we propose a novel framework called 4DEquine by disentangling the 4D reconstruction problem into two sub-problems: dynamic motion reconstruction and static appearance reconstruction.
- For motion, we introduce a simple yet effective spatio-temporal transformer with a post-optimization stage to regress smooth and pixel-aligned pose and shape sequences from video.
- For appearance, we design a novel feed-forward network that reconstructs a high-fidelity, animatable 3D Gaussian avatar from as few as a single image.

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
- While training only on synthetic datasets, 4DEquine achieves state-of-the-art performance on real-world APT36K and AiM datasets, demonstrating the superiority of 4DEquine and our new datasets for both geometry and appearance reconstruction.

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
