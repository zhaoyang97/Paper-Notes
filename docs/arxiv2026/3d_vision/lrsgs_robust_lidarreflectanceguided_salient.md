# LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction

**会议**: arXiv 2026  
**arXiv**: [2603.12647](https://arxiv.org/abs/2603.12647)  
**作者**: Ziyu Chen, Fan Zhu, Hui Zhu, Deyi Kong, Xinkai Kuang et al.
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: lr-sgs, robust, lidar-reflectance-guided, salient, gaussian  

## 一句话总结
最近的 3D 高斯分布 (3DGS) 方法已经证明了自动驾驶场景重建和新颖视图合成的可行性。
## 背景与动机
Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis.. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting.

## 核心问题
然而，大多数现有方法要么仅仅依赖相机，要么仅使用激光雷达进行高斯初始化或深度监督，而点云中包含的丰富场景信息（例如反射率）以及激光雷达与RGB之间的互补性尚未得到充分利用，导致在具有挑战性的自动驾驶场景（例如具有高自我运动和复杂照明的场景）中出现退化。
## 方法详解

### 整体框架
- Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis.
- To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures.
- In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.

### 关键设计
1. **关键组件1**: In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 最近的 3D 高斯分布 (3DGS) 方法已经证明了自动驾驶场景重建和新颖视图合成的可行性。
- 为了解决这些问题，我们提出了一种用于自动驾驶场景的稳健且高效的激光雷达反射引导显着高斯分布方法（LR-SGS），该方法引入了结构感知的显着高斯表示，从激光雷达提取的几何和反射特征点进行初始化，并通过显着变换和改进的密度控制进行细化，以捕获边缘和平面结构。
- 在 Waymo 开放数据集上进行的大量实验表明，LR-SGS 以更少的高斯和更短的训练时间实现了卓越的重建性能。
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
