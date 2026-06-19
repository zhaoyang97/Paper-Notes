---
title: >-
  [论文解读] BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation
description: >-
  [AAAI2026][视频理解][event camera] 提出双向自适应时序相关性（BAT）框架，将事件相机的时序密集运动线索转化为空间密集线索，实现高精度事件光流估计，在 DSEC-Flow 基准上排名第一。 事件相机以高动态范围和高时间分辨率持续捕捉异步亮度变化，非常适合快速运动和复杂光照下的光流估计…
tags:
  - "AAAI2026"
  - "视频理解"
  - "event camera"
  - "光流"
  - "bidirectional temporal correlation"
  - "注意力机制"
  - "event-based vision"
---

# BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation

**会议**: AAAI2026  
**arXiv**: [2503.03256](https://arxiv.org/abs/2503.03256)  
**代码**: [gangweix/BAT](https://github.com/gangweix/BAT)  
**领域**: 视频理解  
**关键词**: event camera, optical flow, bidirectional temporal correlation, deformable attention, event-based vision

## 一句话总结

提出双向自适应时序相关性（BAT）框架，将事件相机的时序密集运动线索转化为空间密集线索，实现高精度事件光流估计，在 DSEC-Flow 基准上排名第一。

## 背景与动机

事件相机以高动态范围和高时间分辨率持续捕捉异步亮度变化，非常适合快速运动和复杂光照下的光流估计。然而事件数据具有**空间稀疏性**，直接套用基于图像的光流框架（如 RAFT）效果受限。代表方法 E-RAFT 仅在两帧事件表示之间构建相关体（correlation volume），运动线索不足；TMA 虽扩展到多帧前向相关体，但忽略了后向时序运动线索，性能仍有瓶颈。

核心洞察：事件和运动在时间上都是连续且细粒度的，**前向和后向时序运动线索对精确光流估计都至关重要**；同时运动在时间上可能非均匀变化，需要自适应地聚合双向运动线索以保证时序一致性。

## 核心问题

1. 事件数据空间稀疏，构建的相关体运动线索不足，如何获取更丰富的运动信息？
2. 现有方法仅利用单向（前向）时序相关，如何充分利用双向时序线索？
3. 非均匀运动导致线性采样引入不一致运动特征，如何自适应聚合一致运动信息？

## 方法详解

### 整体框架

将参考事件流和目标事件流的 voxel grid 表示各分为 $N$ 组，提取 $2N$ 组特征后进行双向时序相关计算，再通过空间自适应时序运动聚合模块融合运动特征，最终用 ConvGRU 迭代更新光流。

### 1. 事件表示与特征提取

- 事件流转换为 voxel grid $\bm{V} \in \mathbb{R}^{B \times H_0 \times W_0}$，使用双线性核沿时间维离散化
- 两段事件流的 voxel grid 各沿时间维分为 $N$ 组，每组 $B/N$ 个时间 bin
- 共 $2N$ 组输入共享权重的特征提取网络（6 个残差块），得到特征 $\bm{F}_n \in \mathbb{R}^{D \times H \times W}$
- $\bm{F}_N$ 为参考帧，$\bm{F}_{2N}$ 为目标帧

### 2. Bidirectional Temporal Correlation (BTC)

基于当前光流估计 $\bm{f}$ 和线性运动假设，推导相邻组间光流 $\bm{df} = \bm{f}/N$：

- **前向时序相关**：参考帧 $\bm{F}_N$ 与后续帧 $\bm{F}_{N+j}$（$j=1,...,N$）计算相关，warping 后在局部网格内采样做点积
- **后向时序相关**：参考帧 $\bm{F}_N$ 与前序帧 $\bm{F}_{N-j}$（$j=1,...,N-1$）计算相关

共得到 $N$ 组前向 + $N-1$ 组后向相关图。后向相关对处理物体移出目标帧导致的遮挡尤为有效。

### 3. Adaptive Temporal Sampling (ATS)

之前工作的采样半径 $r$ 是固定超参数。BAT 引入可学习缩放因子 $\alpha$，得到自适应采样半径 $lr = \alpha \cdot r$，在训练中自动学习最优采样范围，保持时序一致性。

### 4. Spatially Adaptive Temporal Motion Aggregation (SATMA)

处理非均匀运动导致的不一致运动特征问题：

- 相关特征经 MotionEncoder 编码为运动特征 $\bm{M}_j^{fwd}$ / $\bm{M}_j^{bwd}$
- 目标运动特征 $\bm{M}_N^{fwd}$ 与相邻运动特征拼接后经卷积+Sigmoid 生成空间注意力图 $\bm{A}_{spa}$
- 对相邻运动特征用 **deformable attention** 从目标运动特征中聚合相关信息：生成稀疏采样位置，采样得到 K、V，做注意力计算
- 融合：$\bm{M}_j^{fuse} = \bm{A}_{spa} \odot \bm{M}_j^{agg} + \bm{M}_j^{fwd}$
- 前向和后向融合运动特征共同送入 ConvGRU 更新光流

### 5. 损失函数

沿用 RAFT 的 $l_1$ 损失，对 $K$ 次迭代预测施加指数递增权重：$\mathcal{L} = \sum_{i=1}^{K} \gamma^{K-i} \|\bm{f}^i - \bm{f}^{gt}\|_1$。

## 实验关键数据

| 基准 | 指标 | BAT | 次优方法 | 提升 |
|------|------|-----|----------|------|
| DSEC-Flow | 1PE↓ | **7.715** | IDNet 10.069 | 23.4% |
| DSEC-Flow | EPE↓ | **0.655** | IDNet 0.719 | 8.9% |
| MVSEC dt=4 | EPE↓ | **0.53** | TMA 0.70 | 24.3% |
| MVSEC dt=4 | %Out↓ | **0.71** | TMA 1.08 | 34.3% |
| MVSEC dt=1 | EPE↓ | **0.21** | TMA 0.25 | 16.0% |

消融实验（DSEC-Flow，1PE 指标）：

- Baseline (TMA, N=3): 9.123
- +BTC: 8.279（后向相关显著提升）
- +BTC+ATS: 8.179（自适应采样进一步改善）
- Full (BAT): **7.715**（SATMA 有效抑制不一致运动）

注意力类型比较：deformable attention (7.715) > dense attention (8.049) > spatial-reduction attention (8.731)。

未来光流预测：仅用过去事件，BAT (bwd corr) 的 1PE=33.026 远优于 E-RAFT warm-start 的 85.378。

## 亮点

1. **双向时序相关**思路新颖：将时序密集线索转化为空间密集线索，突破事件数据空间稀疏的核心瓶颈
2. **未来光流预测**能力独特：仅用过去事件预测未来光流，对自动驾驶/无人机等实时场景有重要价值
3. **遮挡处理**：后向时序相关天然有利于处理物体移出目标帧导致的遮挡
4. SATMA 中 deformable attention 的使用既高效又聚焦于相关运动特征
5. 在 DSEC-Flow 和 MVSEC 两个主流基准上均取得大幅领先的 SOTA 结果

## 局限与展望

- 在相机剧烈抖动等**快速运动变化**场景下，后向和前向时序运动差异大，后向线索帮助有限
- 基于线性运动假设推导相邻帧间光流，对非线性运动场景可能不够精确
- 时序分组数 $N$ 为固定超参数（$N=3$），更灵活的动态分组策略或可进一步提升性能
- 未探索无监督/自监督训练范式，依赖 GT 光流标注

## 与相关工作的对比

| 方法 | 时序线索 | 运动聚合 | 未来预测 |
|------|----------|----------|----------|
| E-RAFT | 两帧相关 | 无 | 不支持 |
| TMA | 前向多帧相关 | 简单拼接 | 不支持 |
| IDNet | 无相关体，迭代去模糊 | — | 不支持 |
| **BAT** | **双向多帧相关** | **SATMA (deformable attn)** | **支持** |

与 VideoFlow 的区别：VideoFlow 同时估计多帧双向光流，BAT 则将双向时序运动线索聚合到目标帧，专注于单帧光流估计。

## 启发与关联

- "时序密集→空间密集"的转换思路可推广到其他事件相机任务（如深度估计、场景流）
- 自适应采样半径的设计可借鉴到其他需要相关体的任务中
- SATMA 的 deformable attention + spatial attention 融合机制对处理时序不一致性有通用价值
- 未来光流预测能力值得在下游任务（避障、运动规划）中进一步验证

## 评分

- 新颖性: ⭐⭐⭐⭐ — 双向时序相关与自适应聚合的组合设计有效且新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 两个基准全面对比+详细消融+未来预测实验
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ — DSEC-Flow SOTA，未来光流预测有实际应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation](../../CVPR2026/video_understanding/efficient_all-pairs_correlation_volume_sampling_for_optical_flow_estimation.md)
- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/video_understanding/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](../../ICCV2025/video_understanding/unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[CVPR 2026\] From Contrast to Consistency: Rethinking Event-based Continuous-Time Optical Flow Estimation](../../CVPR2026/video_understanding/from_contrast_to_consistency_rethinking_event-based_continuous-time_optical_flow.md)
- [\[CVPR 2026\] LAOF: Robust Latent Action Learning with Optical Flow Constraints](../../CVPR2026/video_understanding/laof_robust_latent_action_learning_with_optical_flow_constraints.md)

</div>

<!-- RELATED:END -->
