---
title: >-
  [论文解读] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching
description: >-
  [NeurIPS 2025][3D视觉][点云去噪] 提出 U-CAN 无监督点云去噪框架，通过 Noise2Noise 匹配方案和几何一致性约束实现多步去噪路径推断，性能逼近有监督方法，且一致性约束可泛化到 2D 图像去噪。 扫描传感器采集的点云不可避免地包含噪声，严重影响曲面重建和形状理解等下游任务…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "点云去噪"
  - "无监督学习"
  - "Noise2Noise"
  - "一致性约束"
  - "几何重建"
---

# U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching

**会议**: NeurIPS 2025  
**arXiv**: [2510.25210](https://arxiv.org/abs/2510.25210)  
**代码**: 有（Project Page）  
**领域**: 3D视觉 / 点云处理  
**关键词**: 点云去噪, 无监督学习, Noise2Noise, 一致性约束, 几何重建

## 一句话总结

提出 U-CAN 无监督点云去噪框架，通过 Noise2Noise 匹配方案和几何一致性约束实现多步去噪路径推断，性能逼近有监督方法，且一致性约束可泛化到 2D 图像去噪。

## 研究背景与动机

扫描传感器采集的点云不可避免地包含噪声，严重影响曲面重建和形状理解等下游任务。现有方法的痛点：

**有监督方法的局限**: 需要大量"噪声-干净"点云对，标注成本极高

**无监督方法的不足**: 现有无监督方法性能与有监督差距较大

**多步去噪**: 单步去噪难以完全消除噪声，多步方案缺乏有效的优化目标

核心创新：利用多个含噪观测之间的统计关系，通过 Noise2Noise 范式实现无监督学习，同时引入几何一致性约束进一步提升去噪质量。

## 方法详解

### 整体框架

U-CAN 包含三个核心组件：
1. 多步去噪路径推断：神经网络预测每个点的多步去噪轨迹
2. Noise2Noise 匹配损失：利用多个噪声观测间的统计推理
3. 几何一致性约束：确保去噪结果的几何一致性

### 关键设计

1. **多个噪声观测的利用**:

    - 对同一形状/场景获取多个含噪点云观测
    - 不同观测的噪声是独立同分布的
    - 通过 N2N 匹配方案，避免需要干净目标

2. **Noise2Noise 匹配损失**:

    - 将一个噪声观测去噪后与另一个噪声观测匹配
    - 统计意义上，最优去噪结果应最小化两个噪声观测间的期望距离
    - 这等价于去噪到真实的干净曲面

3. **几何一致性约束 (Denoised Geometry Consistency)**:

    - 去噪后的点应保持局部几何结构（如法向量、曲率）
    - 来自同一表面的相邻点去噪后应仍然相邻
    - 约束形式化为去噪结果的局部几何不变量

4. **跨域通用性**:

    - 作者证明几何一致性约束不限于 3D 点云
    - 同样的原理可应用于 2D 图像去噪

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{N2N}} + \lambda_1 \mathcal{L}_{\text{geo}} + \lambda_2 \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{N2N}}$: Noise2Noise 匹配损失（Chamfer 距离变体）
- $\mathcal{L}_{\text{geo}}$: 几何一致性损失（法向量一致性 + 局部曲率保持）
- $\mathcal{L}_{\text{reg}}$: 正则化项（防止退化解）

## 实验关键数据

### 主实验（点云去噪 - PU-Net 数据集）

| 方法 | 监督类型 | CD (×10⁻⁴) ↓ | P2S (×10⁻³) ↓ | 噪声 σ=0.01 CD ↓ | 噪声 σ=0.03 CD ↓ |
|------|---------|-------------|-------------|----------------|----------------|
| PCN | 有监督 | 0.68 | 1.23 | 0.58 | 1.15 |
| Score-based | 有监督 | 0.52 | 0.98 | 0.45 | 0.89 |
| DMR | 有监督 | 0.48 | 0.85 | 0.42 | 0.82 |
| Pointfilter | 无监督 | 1.25 | 2.15 | 1.12 | 2.45 |
| Self2Self | 无监督 | 0.95 | 1.72 | 0.85 | 1.85 |
| **U-CAN** | **无监督** | **0.55** | **0.92** | **0.48** | **0.91** |

### 点云上采样实验

| 方法 | PU1K CD ↓ | PU1K P2S ↓ | PU-GAN CD ↓ | 监督类型 |
|------|----------|-----------|------------|---------|
| PU-Net | 0.72 | 1.35 | 0.85 | 有监督 |
| PU-GAN | 0.65 | 1.18 | 0.72 | 有监督 |
| Dis-PU | 0.58 | 1.05 | 0.65 | 有监督 |
| **U-CAN (去噪+上采样)** | **0.61** | **1.08** | **0.68** | **无监督** |

### 2D 图像去噪实验

| 方法 | SIDD PSNR ↑ | SIDD SSIM ↑ | BSD68 PSNR ↑ |
|------|------------|-----------|------------|
| N2N (baseline) | 38.12 | 0.952 | 29.85 |
| N2V | 37.85 | 0.948 | 29.62 |
| **N2N + 一致性约束** | **38.67** | **0.958** | **30.23** |

### 关键发现

1. U-CAN 是首个在性能上接近有监督方法的无监督点云去噪方法
2. 几何一致性约束贡献约 15-20% 的性能提升
3. 该约束成功迁移到 2D 图像去噪，证明了通用性
4. 多步去噪路径比单步去噪带来稳定的改善

## 亮点与洞察

- **无监督逼近有监督**: 在无标注数据的情况下达到接近有监督的效果，实际应用价值大
- **跨域通用约束**: 几何一致性约束的 2D 图像去噪泛化非常出彩
- **理论支撑**: 对 N2N 方案的统计推理有严格的数学论证

## 局限与展望

1. 需要同一场景的多个噪声观测，单次扫描场景无法直接应用
2. 对于非均匀噪声分布的鲁棒性有待验证
3. 大规模室外场景（如 LiDAR 点云）的实验较少
4. 多步去噪增加了推理时间

## 相关工作与启发

- **Noise2Noise (Lehtinen et al., 2018)**: 无监督去噪的开创性工作
- **Score-based denoising**: 基于分数匹配的点云去噪
- **DMR**: 可微分流形重建去噪方法
- **Self2Self**: 自监督去噪框架

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总体推荐 | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising](../../ICCV2025/3d_vision/noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)
- [\[ICML 2026\] SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising](../../ICML2026/3d_vision/simpc_learning_self-induced_mirror-point_consistency_for_unsupervised_point_clou.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[CVPR 2025\] Consistency-aware Self-Training for Iterative-based Stereo Matching](../../CVPR2025/3d_vision/consistency-aware_self-training_for_iterative-based_stereo_matching.md)
- [\[CVPR 2026\] Topology-aware Feature Propagation for Unsupervised Non-rigid Point Cloud Correspondence](../../CVPR2026/3d_vision/topology-aware_feature_propagation_for_unsupervised_non-rigid_point_cloud_corres.md)

</div>

<!-- RELATED:END -->
