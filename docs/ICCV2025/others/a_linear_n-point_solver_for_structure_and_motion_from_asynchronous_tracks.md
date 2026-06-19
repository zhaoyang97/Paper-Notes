---
title: >-
  [论文解读] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks
description: >-
  [ICCV 2025][运动估计] 本文提出了一种统一的线性 N-point 求解器，能够从具有任意时间戳的 2D 点对应中恢复相机线速度和 3D 点结构，适用于全局快门、滚动快门和事件相机等多种传感器模式。 从点对应中恢复相机运动和场景结构是几何计算机视觉的核心问题。经典的五点算法（5-point）或八点算法（8-poin…
tags:
  - "ICCV 2025"
  - "运动估计"
  - "异步特征轨迹"
  - "线性求解器"
  - "事件相机"
  - "滚动快门"
---

# A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks

**会议**: ICCV 2025  
**arXiv**: [2507.22733](https://arxiv.org/abs/2507.22733)  
**代码**: [GitHub](https://github.com/suhang99/AsyncTrack-Motion-Solver)  
**领域**: 几何计算机视觉  
**关键词**: 运动估计, 异步特征轨迹, 线性求解器, 事件相机, 滚动快门

## 一句话总结
本文提出了一种统一的线性 N-point 求解器，能够从具有任意时间戳的 2D 点对应中恢复相机线速度和 3D 点结构，适用于全局快门、滚动快门和事件相机等多种传感器模式。

## 研究背景与动机
从点对应中恢复相机运动和场景结构是几何计算机视觉的核心问题。经典的五点算法（5-point）或八点算法（8-point）已经非常成熟，但它们有一个基本假设：点对应来自一对同步采集的视图，每个视图代表场景的瞬时捕获。

然而，随着传感器技术的发展，这种同步假设越来越难以满足。滚动快门相机逐行采集图像，不同行的时间戳不同；事件相机更是完全异步的传感器，每个像素独立返回亮度变化事件流，时间分辨率可达微秒级。现有方法要么强制将异步数据聚合为同步帧（丢失了事件相机的核心优势），要么使用基于直线特征的求解器（场景需要有明显的直线结构）。

核心矛盾是：现有的几何求解器无法原生处理来自异步传感器的特征轨迹，而简单地将异步数据同步化又会丢失时间分辨率的优势。本文的切入角度是利用常速度运动模型（constant velocity motion model）和一阶动力学公式，推导出一个对 3D 点和速度都呈线性的入射关系（point incidence relation），从而可以高效地用线性系统求解。

核心 idea：在常速度运动模型下，来自任意时间戳的点观测构成一个对未知量（3D 点位置和线速度）的线性约束系统，可通过 Schur 补高效求解。

## 方法详解

### 整体框架
给定一组带时间戳的 2D 点轨迹 $\{(\mathbf{x}_{ij}, t_{ij})\}$、已知的角速度 $\boldsymbol{\omega}$（来自 IMU 或其他估计），以及相机内参 $\mathbf{K}$，求解器恢复归一化的线速度 $\hat{\mathbf{v}}$ 和 3D 点 $\hat{\mathbf{P}}_i$。首先将观测转换为旋转补偿后的 bearing 向量，构建线性约束系统，然后通过 Schur 补技巧和 SVD 高效求解。

### 关键设计
1. **点入射关系（Point Incidence Relation）**:

    - 功能：建立 2D 观测与 3D 点位置、相机运动之间的几何约束
    - 核心思路：3D 点 $\mathbf{P}_i$ 在时刻 $t_{ij}$ 投影到相机平面上的 bearing 向量 $\mathbf{f}_{ij}'$ 必须与相机坐标系下的 3D 点平行。利用叉积为零的条件：$[\mathbf{f}_{ij}']_\times \mathbf{P}_i - t_{ij}' [\mathbf{f}_{ij}']_\times \mathbf{v} = \mathbf{0}$
    - 设计动机：与经典的对极约束（epipolar constraint）不同，该入射关系直接操作在动力学参数（速度）和 3D 点上，且对时间戳没有同步假设。该关系可以特化为经典的 5-point 算法或最近的直线求解器。

2. **Schur 补高效求解器**:

    - 功能：利用系统矩阵 $\mathbf{A}$ 的稀疏块对角结构，避免对整个大矩阵做 SVD
    - 核心思路：将法方程 $\mathbf{A}^\top \mathbf{A} \mathbf{x} = \mathbf{0}$ 写成块矩阵形式，利用 Schur 补将 $3M+3$ 维问题降为 $3 \times 3$ 的矩阵 $\mathbf{B}$ 的 SVD 问题：$\mathbf{B} = \mathbf{M}_D - \mathbf{M}_B^\top \mathbf{M}_A^{-1} \mathbf{M}_B$。由于 $\mathbf{M}_A$ 是块对角矩阵（每块 $3 \times 3$），其逆的计算复杂度仅为 $O(M)$ 而非 $O(M^3)$
    - 设计动机：直接对 $\mathbf{A} \in \mathbb{R}^{3N \times (3M+3)}$ 做 SVD 在观测数量大时计算代价高昂。Schur 补将计算瓶颈降到 $3 \times 3$ 矩阵的 SVD，极大提升了实际效率。最小情况下仅需 63 μs 即可求解。

3. **退化与解的唯一性分析**:

    - 功能：系统分析求解器在何种条件下产生退化解，以及解的多重性
    - 核心思路：SVD 产生 $\hat{\mathbf{v}}$ 和 $-\hat{\mathbf{v}}$ 两个候选，通过正深度约束 $(\hat{\mathbf{P}}_i)_z > 0$ 选择正确解。退化条件：每条轨迹至少需要 2 个不同时间戳的观测使 $\mathbf{F}_i$ 满秩，且矩阵 $\mathbf{B}$ 的秩至少为 2
    - 设计动机：明确求解器的适用条件和最小采样需求，为 RANSAC 采样策略提供理论指导

### 损失函数 / 训练策略
该方法无需训练，是纯几何求解器。部署时嵌入 RANSAC 循环：每次迭代采样 $M=4$ 条轨迹、每条 $N_i=5$ 个观测，生成速度假设，通过 bearing 向量的角度残差 $\bar{\theta}_i$ 判断内点（阈值 5°），当内点比例超过 0.9 时提前终止。最终用所有内点重新估计速度。

## 实验关键数据

### 主实验

| 传感器类型 | 序列 | eventail (baseline) | 本文 | 本文 (高置信) |
|-----------|------|---------------------|------|------------|
| 全局快门 | desk-normal | 22.7° / 23.4° | 15.1° / 8.5° | 10.2° / 7.3° |
| 全局快门 | shapes_trans | 31.8° / 32.7° | 17.1° / 7.2° | 9.9° / 6.2° |
| 滚动快门 | Seq 4 | 43.8° / 40.8° | 27.5° / 20.1° | 22.6° / 17.4° |
| 滚动快门 | Seq 5 | 45.5° / 44.8° | 24.7° / 17.0° | 19.3° / 13.8° |
| 事件相机 | mountain-normal | 25.2° / 21.4° | 17.1° / 16.1° | 16.9° / 15.8° |
| 事件+全局 | shapes_trans | - | 14.4° / 7.5° | 7.0° / 6.7° |

指标为速度方向的角度误差（均值/中位数，度），越低越好。

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 轨迹数M=3→30 | 误差显著降低 | 空间采样越多越好，但超过30后边际递减 |
| 每轨迹观测n=2→50 | 改善有限 | 时间密度提升对噪声鲁棒性贡献不大 |
| 时间窗口0.05→0.4s | 误差稳定降低 | 更长轨迹能更好地平均高频噪声 |
| 有vs无滚动快门校正 | ~2°差异 | 验证了正确时间戳关联的重要性 |
| 仅事件vs事件+全局快门 | 显著改善 | 多传感器融合是本方法的独特优势 |

### 关键发现
- 点轨迹比直线特征更容易提取，在缺乏明显直线结构的自然场景中优势尤为明显
- 事件相机和全局快门相机的轨迹可以无缝融合，互补提升精度：图像提供高空间分辨率，事件提供高时间分辨率
- 在仅 1 个 3D 点、3 个时间观测的极端情况下就能恢复完整运动方向——这是一个令人惊讶的理论发现

## 亮点与洞察
- **统一理论框架**：从全局快门到事件相机的连续谱上，本方法是第一个真正无假设的异步点求解器
- **巧妙的 Schur 补利用**：充分利用稀疏结构将大规模线性系统降为 $3 \times 3$ SVD，实际运行极快
- **可迁移的 trick**：将运动估计从"恢复相对位姿"重新表述为"恢复一阶动力学"，这种视角转换对其他视觉里程计问题也有启发

## 局限与展望
- 依赖已知的角速度 $\boldsymbol{\omega}$（通常来自 IMU），限制了在纯视觉场景下的独立使用
- 常速度模型假设限制了处理非匀速运动（如急加速/急转弯）的能力
- 高阶导数估计（如加速度）在实际数据上噪声敏感性极高，论文承认这是开放问题
- 仅在相对小规模数据集上验证，未在大规模 SLAM 系统中集成测试

## 相关工作与启发
- **vs Gao et al. (eventail)**: eventail 基于直线特征的速度求解器，本文扩展到点特征，在自然场景中更通用
- **vs 经典 5-point/8-point**: 经典方法假设同步采集，本文通过一阶动力学建模放松了这一假设，是对经典理论的自然推广
- **vs Saurer et al.**: 使用类似的点入射关系但用于已知 2D-3D 对应的绝对位姿估计，本文仅需 2D 观测并同时恢复运动和结构
- **vs 对比最大化方法 (Contrast Maximization)**: CM 框架通过迭代优化事件的参数化图像翘曲函数来估计运动，计算量大且限制于单应变换场景。本方法是闭式线性求解器，效率更高

## 补充说明
- 求解器的最小情况运行时间仅 63 μs（Intel Xeon CPU），适合嵌入 RANSAC 循环的实时应用
- 论文在附录中推导了任意阶 Taylor 展开的一般形式，一阶（线性速度）是本文主要展示的特例
- 在仿真中也测试了加速度估计，但发现实际数据上噪声敏感性太高，留作未来工作
- 约束分析表明：单个 3D 点配合 3 个时间观测即可恢复完整运动方向，这是一个具有理论意义的最小配置结果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为异步点轨迹提供统一的线性求解器，理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实数据、三种传感器模态、多序列验证，但缺少大规模应用
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，理论分析完整，实验逻辑连贯
- 价值: ⭐⭐⭐⭐ 为事件相机时代的几何视觉奠定重要理论基础，具有长远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)
- [\[ECCV 2024\] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](../../ECCV2024/others/improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)
- [\[ACL 2025\] The Harmonic Structure of Information Contours](../../ACL2025/others/the_harmonic_structure_of_information_contours.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/others/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)

</div>

<!-- RELATED:END -->
