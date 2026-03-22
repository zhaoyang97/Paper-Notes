# InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction

**会议**: arXiv 2026  
**arXiv**: [2603.11298](https://arxiv.org/abs/2603.11298)  
**作者**: Dingqiang Ye, Jiacong Xu, Jianglu Ping, Yuxiang Guo, Chao Fan
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: instanthdr, single-forward, gaussian, splatting, high  

## 一句话总结
高动态范围 (HDR) 新视图合成 (NVS) 旨在从多重曝光低动态范围 (LDR) 图像重建 HDR 场景。

## 背景与动机
High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images.. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization.

## 核心问题
目前的前馈替代方案通过假设曝光不变的外观而忽略了 HDR 问题。

## 方法详解

### 整体框架
- High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images.
- To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass.
- Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping.

### 关键设计
1. **关键组件1**: Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 综合实验表明，我们的 InstantHDR 提供了与最先进的基于优化的 HDR 方法相当的合成性能，同时通过我们的单前向和后优化设置享受 $\sim700\times$ 和 $\sim20\times$ 重建速度改进。

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
