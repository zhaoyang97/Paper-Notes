---
title: >-
  [论文解读] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation
description: >-
  [ICCV 2025][语义分割][无监督时序动作分割] 提出 Skeleton Motion Quantization (SMQ) 方法，通过关节解耦的时序自编码器和骨架运动词量化模块，实现无监督骨架序列时序动作分割，在 HuGaDB、LARa 和 BABEL 三个数据集上大幅超越现有无监督方法。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "无监督时序动作分割"
  - "骨架序列"
  - "运动量化"
  - "时序自编码器"
  - "运动词"
---

# Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation

**会议**: ICCV 2025  
**arXiv**: [2508.04513](https://arxiv.org/abs/2508.04513)  
**代码**: [github.com/bachlab/SMQ](https://github.com/bachlab/SMQ)  
**领域**: 图像分割  
**关键词**: 无监督时序动作分割, 骨架序列, 运动量化, 时序自编码器, 运动词

## 一句话总结

提出 Skeleton Motion Quantization (SMQ) 方法，通过关节解耦的时序自编码器和骨架运动词量化模块，实现无监督骨架序列时序动作分割，在 HuGaDB、LARa 和 BABEL 三个数据集上大幅超越现有无监督方法。

## 研究背景与动机

时序动作分割（Temporal Action Segmentation）旨在将长视频/序列分割为不同动作片段。当前骨架序列的分割方法主要依赖**全监督**学习，标注成本高昂。虽然在 RGB 视频领域已有无监督方法（如 CTE、TOT、ASOT），但它们**未针对骨架数据结构进行优化**，直接应用效果不佳。

作者发现现有无监督视频分割方法在骨架数据上存在两个关键问题：

**使用 Viterbi 解码后处理**的方法（CTE、TOT）假设每个动作在序列中只出现一次，无法识别重复动作

**不使用 Viterbi 的方法**则生成过多碎片化小段，分割质量差

骨架数据具有天然的**关节-时序结构**，现有方法未利用这一先验信息。因此需要一种专门为骨架数据设计的无监督分割方法。

## 方法详解

### 整体框架

SMQ 框架包含三个核心组件：**关节解耦的时序编码器** → **骨架运动量化模块**（含时序分块 + 码本量化） → **时序解码器**。输入骨架序列 $\mathbf{X} \in \mathbb{R}^{N \times C \times T \times V}$，输出每帧的动作类别标签 $\mathbf{Y} \in \mathbb{N}^{N \times T}$。

### 关键设计

1. **关节解耦编码器**: 将输入重塑为 $\mathbf{X}' \in \mathbb{R}^{(N \cdot V) \times C \times T}$，每个关节的时序序列作为独立样本送入 TCN 编码器。通过膨胀残差层（dilation factor 指数增长）捕捉多尺度时序依赖，最终得到 $\mathbf{Z}_{nv} \in \mathbb{R}^{D \times T}$。**关键思想**：保持不同关节信息在嵌入空间中解耦，防止表征被少数关节主导。

2. **时序分块（Temporal Patching）**: 将关节嵌入拼接后 $\mathbf{Z}_{concat} \in \mathbb{R}^{N \times T \times (V \cdot D)}$，划分为 $M = T/P$ 个非重叠时序块 $\mathbf{Z}_p \in \mathbb{R}^{N \times M \times P \times (V \cdot D)}$。相比逐帧量化，基于 patch 的表征能更好捕捉动作的时序连续性，避免过度分割。

3. **骨架运动词量化**: 学习大小为 $K$ 的码本 $\mathcal{C} = \{\mathbf{c}_k\}_{k=1}^{K}$，每个码本元素 $\mathbf{c}_k \in \mathbb{R}^{P \times (V \cdot D)}$ 表示一个"骨架运动词"。量化过程通过最近邻分配实现：
$$\mathbf{q}_i = \mathbf{c}_{k_i} \quad \text{with} \quad k_i = \arg\min_k \|\mathbf{p}_i - \mathbf{c}_k\|_2$$
码本使用 **EMA（指数移动平均）** 更新，衰减因子 $\alpha = 0.5$，提升训练稳定性。

4. **镜像解码器**: 结构与编码器对称，从量化后的表征重建原始骨架序列，同样保持关节解耦。

### 损失函数 / 训练策略

总损失为两项之和：

$$L_{total} = \lambda L_{rec} + L_{commit}$$

- **关节间距离 MSE 重建损失**：计算每帧中每对关节之间的距离差异，而非直接比较坐标。该损失天然具有**平移和旋转不变性**，仅度量姿态差异：
$$L_{rec} = \frac{1}{N \cdot T \cdot V^2} \sum_{n,t,v,w} (dX_{ntvw} - d\hat{X}_{ntvw})^2$$

- **承诺损失（Commitment Loss）**: 鼓励编码器输出靠近已分配的运动词，$\lambda = 0.001$：
$$L_{commit} = \sum_{\mathbf{p}_i} \|\text{sg}[\mathbf{c}_{k_i}] - \mathbf{p}_i\|_2^2$$

## 实验关键数据

### 主实验

| 数据集 | 指标 | SMQ (本文) | ASOT | TOT+Viterbi | CTE+Viterbi |
|--------|------|-----------|------|-------------|-------------|
| HuGaDB | MoF | **42.0** | 33.9 | 33.8 | 39.2 |
| HuGaDB | Edit | **36.1** | 17.4 | 20.8 | 21.7 |
| HuGaDB | F1@50 | **24.3** | 3.0 | 7.5 | 7.5 |
| LARa | MoF | **37.4** | 22.9 | 32.6 | 23.0 |
| LARa | Edit | **39.4** | 23.4 | 17.7 | 17.7 |
| LARa | F1@50 | **16.4** | 5.7 | 3.2 | 1.6 |
| BABEL-S2 | MoF | **49.1** | 43.1 | 35.3 | 42.4 |
| BABEL-S2 | F1@50 | **27.4** | 23.4 | 19.8 | 12.8 |

### 消融实验

**关节解耦 vs 纠缠**:

| 嵌入方式 | MoF (HuGaDB) | Edit | F1@50 |
|----------|-------------|------|-------|
| Entangled | 38.9 | 34.2 | 18.9 |
| **Disentangled** | **42.0** | **36.1** | **24.3** |

**Patch 大小对 LARa 数据集的影响**:

| Patch Size | MoF | Edit | F1@50 | 说明 |
|-----------|-----|------|-------|------|
| 1 (逐帧) | 33.9 | 25.6 | 8.6 | 过度分割 |
| 50 (1秒) | **37.4** | **39.4** | **16.4** | 最优 |
| 100 (2秒) | 30.8 | 36.1 | 15.4 | 分辨率不足 |

**损失函数变体**:

| Commitment | 重建损失 | MoF (LARa) | Edit | F1@50 |
|-----------|---------|-----------|------|-------|
| ✓ | ✗ | 29.9 | 29.9 | 8.8 |
| ✗ | Inter-Joint Dist. MSE | 31.0 | 34.6 | 14.4 |
| ✓ | MSE | 34.6 | 39.4 | 14.4 |
| ✓ | **Inter-Joint Dist. MSE** | **37.4** | **39.4** | **16.4** |

### 关键发现

- SMQ 在所有数据集和指标上**全面超越**现有无监督时序分割方法和自监督骨架表征学习方法
- 关节解耦嵌入对 F1@50 提升高达 **+5.4**（HuGaDB），证明利用骨架结构先验的重要性
- 逐帧量化（patch size=1）导致严重过度分割，1秒 patch 最优
- 关节间距离 MSE 优于标准 MSE，因为具备平移/旋转不变性

## 亮点与洞察

- **运动词（Motion Words）概念新颖**：将时序分割类比为时序聚类，用离散运动词表示原型运动模式，概念直觉且有效
- **关节解耦设计简洁实用**：通过 reshape 操作即可让每个关节独立学习嵌入，无需复杂的图卷积网络
- **关节间距离损失设计精巧**：不需要显式数据增强即可获得旋转/平移不变性

## 局限与展望

- 动作边界只能在 patch 边界处切换，**无法精确检测动作转换点**
- 码本大小 $K$ 需提前指定，与数据集实际动作数量一致（实际场景中动作数未知）
- 仅在运动捕捉/IMU 数据上验证，未扩展到从 RGB 视频估计的骨架上
- 未与监督方法的差距仍然较大（如 MS-GCN 在 HuGaDB 上 MoF=90.4 vs SMQ=42.0）

## 相关工作与启发

- 与 VQ-VAE 的量化机制类似，但这里量化的是时序 patch 而非单帧
- 关节解耦思路可扩展到其他骨架时序任务（如动作识别、运动预测）
- EMA 码本更新策略来自 van den Oord 等人的 VQ-VAE 工作

## 评分

- 新颖性：⭐⭐⭐⭐ — 骨架运动词 + 关节解耦的组合设计新颖
- 实验充分度：⭐⭐⭐⭐ — 三个数据集 + 充分消融
- 实用性：⭐⭐⭐ — 需预知动作数；与监督方法差距大
- 总体：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation](clot_closed_loop_optimal_transport_for_unsupervised_action_segmentation.md)
- [\[ECCV 2024\] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](../../ECCV2024/segmentation/long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[ECCV 2024\] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation](../../ECCV2024/segmentation/un-evimo_unsupervised_event-based_independent_motion_segmentation.md)

</div>

<!-- RELATED:END -->
