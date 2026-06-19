---
title: >-
  [论文解读] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception
description: >-
  [ICCV 2025][自动驾驶][collaborative perception] 提出 INSTINCT，一种基于 LiDAR 的实例级交互协作感知框架，通过质量感知过滤、双分支检测路由和跨智能体局部实例融合三个核心模块，在多个数据集上实现 SOTA 性能的同时将通信带宽降低至现有方法的 1/264~1/281。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "collaborative perception"
  - "instance-level fusion"
  - "V2X"
  - "LiDAR"
  - "bandwidth efficiency"
---

# INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception

**会议**: ICCV 2025  
**arXiv**: [2509.23700](https://arxiv.org/abs/2509.23700)  
**代码**: [https://github.com/CrazyShout/INSTINCT](https://github.com/CrazyShout/INSTINCT)  
**领域**: 自动驾驶  
**关键词**: collaborative perception, instance-level fusion, V2X, LiDAR, bandwidth efficiency

## 一句话总结

提出 INSTINCT，一种基于 LiDAR 的实例级交互协作感知框架，通过质量感知过滤、双分支检测路由和跨智能体局部实例融合三个核心模块，在多个数据集上实现 SOTA 性能的同时将通信带宽降低至现有方法的 1/264~1/281。

## 研究背景与动机

协作感知系统通过融合多智能体传感数据来克服单车感知在远距离检测和遮挡场景中的局限性。然而，频繁的协作交互和实时性要求（≥10Hz）对通信带宽提出了严苛约束。现有中间融合方法直接传输完整特征图，带宽压力巨大。虽然 query-based 的实例级交互在相机模态上已有研究，但在 LiDAR 协作感知中仍欠发展，且性能落后于 SOTA 方法。INSTINCT 旨在填补 LiDAR 场景下实例级协作感知的空白，同时实现高精度与低带宽的平衡。

## 方法详解

### 整体框架

INSTINCT 的流程分为五步：
1. 单智能体检测器从 LiDAR 数据提取实例特征 $\mathbf{Q}_i$
2. 质量感知过滤模块过滤高质量实例特征 $\tilde{\mathbf{Q}}_i$ 并生成统一空间位置图 $\mathcal{S}_{j \to i}$
3. 双分支检测路由（DDR）将实例分为协作相关 $\mathbf{Q}_i^{coop}$ 和协作无关 $\mathbf{Q}_i^{single}$
4. 跨智能体局部实例融合（CALIF）对协作特征进行跨域适应和基于高斯距离的局部融合
5. 最终检测头输出预测结果

### 关键设计

1. **质量感知过滤（QAF）**: 通过 IoU 惩罚分类损失（MAL Loss）确保传输的实例特征质量。将 MAL 仅用于解码器最后一层，并采用 BEV IoU 避免 3D IoU 的数值不稳定性。同时构建稀疏 2D 相对位置图进行坐标统一和范围过滤，将带宽降低约 94.1%。MAL loss 公式为：$\text{MAL}(p,q,y) = -q^\gamma \log(p) + (1-q^\gamma)\log(1-p)$（当 $q>0$ 时）。

2. **双分支检测路由（DDR）**: 基于直觉——如果一个实例在整个协作场景中没有对应实例，它无法从协作中受益，且可能干扰协作过程。通过拼接所有接收的实例特征为向量表，送入共享参数检测头获取检测框，计算所有实例对之间的 IoU 矩阵 $\mathcal{M}_{iou}$，将 IoU 全低于阈值 $\lambda$ 的实例划入 $\mathbf{Q}_i^{single}$，其余划入 $\mathbf{Q}_i^{coop}$。

3. **跨智能体局部实例融合（CALIF）**: 包含两个子模块：(a) 跨域适应（CDA）通过自注意力机制弥合异构硬件和环境差异导致的域差距，并添加空间位置编码和智能体感知编码；(b) 基于高斯距离的注意力（GDA）通过计算实例检测框外接圆心距离生成注意力权重 $\mathcal{W}_{k,v} = \exp(-\frac{\sqrt{(x_k-x_v)^2+(y_k-y_v)^2}}{\beta r_k^2})$，实现不对称局部交互，远距离或检测偏差大的实例被自然屏蔽。

### 损失函数 / 训练策略

- 分类损失采用 Focal Loss + MAL Loss 组合
- 回归损失使用 L1 Loss
- 引入 Co-GT Sampling 策略：构建跨智能体对象级点云数据库，在训练时采样并验证空间一致性，增加混合实例特征的多样性
- 训练接近收敛时采用 Fade 策略获取更接近真实数据分布的样本
- 优化器：Adam，初始学习率 0.001，OneCycle 调度

## 实验关键数据

### 主实验

| 模型 | 融合类型 | DAIR-V2X AP@0.5/0.7 | V2XSet AP@0.5/0.7 | V2V4Real AP@0.5/0.7 | 带宽(log₂) |
|------|---------|---------------------|-------------------|---------------------|-----------|
| No Fusion | - | 0.635/0.496 | 0.652/0.520 | 0.398/0.220 | 0 |
| V2VNet | 中间 | 0.634/0.423 | 0.827/0.658 | 0.647/0.336 | 24.62/25.10/25.10 |
| Where2comm | 中间 | 0.790/0.665 | 0.926/0.849 | 0.702/0.380 | 21.72/21.19/22.86 |
| CoAlign | 中间 | 0.780/0.655 | 0.929/0.847 | 0.721/0.466 | 24.62/25.10/25.10 |
| **INSTINCT** | **实例** | **0.819/0.753** | **0.923/0.873** | **0.809/0.620** | **13.58/14.16/14.81** |

在真实数据集 DAIR-V2X 和 V2V4Real 上，AP@0.7 分别超越之前最优 13.23% 和 33.08%，带宽低至 $2^{13}$~$2^{14}$ bytes/帧（约 16KB/帧）。

### 消融实验

| QAF | DDR | CDA | GDA | GT | AP@0.5/0.7 | 带宽(log₂) |
|-----|-----|-----|-----|-----|-----------|-----------|
| | | | | | 0.730/0.598 | 17.67 |
| ✓ | | | | | 0.696/0.604 | 13.58 |
| ✓ | ✓ | | | | 0.720/0.632 | 13.58 |
| ✓ | ✓ | ✓ | | | 0.790/0.710 | 13.58 |
| ✓ | ✓ | ✓ | ✓ | | 0.811/0.739 | 13.58 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **0.819/0.753** | 13.58 |

各模块贡献：QAF 降低 94.1% 带宽；DDR 带来 +3.43% AP@0.7；CDA+DDR 带来 +11.23%；CALIF 全部模块带来 +14.16%；Co-GT Sampling 最终 +15.51%。

### 关键发现

- 在姿态噪声鲁棒性测试中，INSTINCT 在所有噪声水平下均保持最优精度，展现出卓越的环境鲁棒性
- 仿真数据集 V2XSet 上提升相对有限（+2.81%），因场景分布均匀导致性能天花板高
- 实例级交互在复杂真实场景中的适应性远强于传统特征级交互

## 亮点与洞察

- 首个在 V2X 场景中实现 LiDAR 实例级交互的协作感知架构
- 通信带宽仅需约 16KB/帧，比特征级方法降低约 1/264~1/281
- 双分支路由策略简洁有效，将协作无关实例分离避免干扰
- 基于高斯距离的局部非对称注意力机制设计巧妙，利用几何先验指导特征交互

## 局限与展望

- 在极稀疏点云场景下检测仍有局限
- DDR 中 IoU 阈值 $\lambda$ 为手动设定的超参数，自适应选取可进一步探索
- Co-GT Sampling 增加了训练复杂度，能否简化值得研究
- 未探讨与相机模态的融合

## 相关工作与启发

- 实例级交互的思路可扩展到其他传感器融合场景（如雷达+相机）
- 质量感知过滤中 MAL Loss 适配 3D 检测的技巧（仅用最后一层+BEV IoU）有通用价值
- Co-GT Sampling 为协作感知场景的数据增强提供了新范式

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个 LiDAR-V2X 实例级协作框架，三模块设计系统且新颖
- **实验充分度**: ⭐⭐⭐⭐ 三个数据集（含真实+仿真）、完整消融、姿态噪声鲁棒性测试
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图示直观
- **价值**: ⭐⭐⭐⭐ 在真实数据集上大幅超越 SOTA，带宽降低显著，工程价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)
- [\[CVPR 2026\] CATNet: Collaborative Alignment and Transformation Network for Cooperative Perception](../../CVPR2026/autonomous_driving/catnet_collaborative_alignment_and_transformation_network_for_cooperative_percep.md)
- [\[ICLR 2026\] SiMO: Single-Modality-Operable Multimodal Collaborative Perception](../../ICLR2026/autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [\[NeurIPS 2025\] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](../../NeurIPS2025/autonomous_driving/sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)

</div>

<!-- RELATED:END -->
