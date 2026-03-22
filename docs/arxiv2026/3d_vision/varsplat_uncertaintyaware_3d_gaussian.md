# VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM

**会议**: arXiv 2026  
**arXiv**: [2603.09673](https://arxiv.org/abs/2603.09673)  
**作者**: Anh Thuan Tran, Jana Kosecka
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: varsplat, uncertainty-aware, gaussian, splatting, robust  

## 一句话总结
同步定位与建图 (SLAM) 与 3D 高斯分布 (3DGS) 可以在不同的现实世界场景中实现快速、可微的渲染和高保真度重建。

## 背景与动机
Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes.. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties.

## 核心问题
然而，现有的 3DGS-SLAM 方法隐式地处理测量可靠性，使得姿态估计和全局对齐在低纹理区域、透明表面或具有复杂反射特性的区域中容易发生漂移。

## 方法详解

### 整体框架
- To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance.
- Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.

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
- Replica（合成）和 TUM-RGBD、ScanNet 和 ScanNet++（真实世界）上的实验结果表明，与密集 RGB-D SLAM 的现有研究相比，VarSplat 提高了鲁棒性，并实现了有竞争力或优越的跟踪、映射和新颖的视图合成渲染。

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
