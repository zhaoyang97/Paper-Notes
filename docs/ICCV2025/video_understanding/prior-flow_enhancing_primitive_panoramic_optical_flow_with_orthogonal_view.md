---
title: >-
  [论文解读] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View
description: >-
  [ICCV 2025][视频理解][光流] 提出双分支框架 PriOr-Flow，利用正交视图的低畸变先验来补偿 ERP 全景图像极区的严重畸变，从而显著提升全景光流估计精度，在 MPFDataset 和 FlowScape 上分别降低 EPE 30.0% 和 29.6%。 全景光流估计旨在从全景视频的连续帧中估计稠密的像素…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "光流"
  - "equirectangular projection"
  - "dual-branch"
  - "distortion compensation"
  - "orthogonal view"
---

# PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View

**会议**: ICCV 2025  
**arXiv**: [2506.23897](https://arxiv.org/abs/2506.23897)  
**代码**: [GitHub](https://github.com/longliangLiu/PriOr-Flow)  
**领域**: 视频理解 / 光流估计 / 全景视觉  
**关键词**: panoramic optical flow, equirectangular projection, dual-branch, distortion compensation, orthogonal view

## 一句话总结

提出双分支框架 PriOr-Flow，利用正交视图的低畸变先验来补偿 ERP 全景图像极区的严重畸变，从而显著提升全景光流估计精度，在 MPFDataset 和 FlowScape 上分别降低 EPE 30.0% 和 29.6%。

## 研究背景与动机

全景光流估计旨在从全景视频的连续帧中估计稠密的像素运动场，在自动驾驶、视频插帧、3D 重建等领域有重要应用。当前全景图像普遍采用等距柱状投影（ERP），但 ERP 将球面映射到平面时会引入严重的几何畸变，尤其在**极区**（南北极附近）畸变最为剧烈，遵循 cosine 衰减规律。

现有方法可分为三类：

**权重变换方法**（LiteFlowNet360, OmniFlowNet）：通过卷积权重变换适配 ERP，但引入额外计算开销

**切平面方法**（TanImg）：将球面投影到多个切平面上处理，但存在跨平面不连续和运动不一致问题

**ERP-based 方法**（PanoFlow, SLOF, MPF-Net）：直接处理 ERP 图像，使用可变形卷积等技术补偿畸变

**核心问题**：以上所有方法都没有显式针对极区的严重畸变进行处理。极区是 ERP 投影中畸变最大的区域（像素被过度拉伸），导致代价体中噪声严重、光流估计误差大。

**核心洞察**：将全景图像在球面上绕 x 轴旋转 90° 得到**正交视图（Orthogonal View）**，其畸变分布恰好与原始视图互补——原始视图极区高畸变的区域在正交视图中变为低畸变区域。利用这一互补性，可以用正交视图的低畸变信息来补偿原始视图的极区误差。

## 方法详解

### 整体框架

PriOr-Flow 采用**双分支结构**（可集成到 RAFT、GMA、SKFlow 等迭代式光流网络中），以 PriOr-RAFT 为例：

- **原始分支（Primitive Branch）**：处理原始 ERP 帧对，构建原始代价体金字塔
- **正交分支（Orthogonal Branch）**：将 ERP 帧对旋转 90° 得到正交视图，独立提取特征并构建正交代价体金字塔
- **DCCL 算子**：在迭代更新中联合从两个代价体检索相关性信息
- **ODDC 模块**：基于置信度引导的运动特征融合，将正交分支的运动信息补偿到原始分支

### 关键设计

#### 1. 正交视图生成

通过球面旋转操作 $\mathcal{R}$ 实现：
- 将 ERP 像素坐标 $\mathbf{x}$ 映射到 3D 笛卡尔坐标 $P(\mathbf{x})$
- 绕 x 轴旋转 90°：$R_x(90°) \cdot P(\mathbf{x})$
- 重新投影回 ERP 平面，使用双线性插值：$I^o = T_p^o(I^p)$

正交视图的关键特性：其畸变分布与原始视图**互补**，极区变为低畸变，赤道区域变为高畸变。

#### 2. Dual-Cost Collaborative Lookup (DCCL)

传统方法仅从单一代价体检索相关性，极区的畸变噪声会严重影响检索质量。DCCL 在统一的球面上进行关联查找：

- 根据当前光流估计 $\mathcal{F}^p$ 定位对应点 $\hat{\mathbf{x}}^p$（含 mod W 保证水平边界连续）
- 在对应点周围定义局部网格 $\mathcal{N}(\hat{\mathbf{x}}^p)_r^p$
- 分别从原始代价体金字塔索引得到 $\mathcal{C}^p$
- 将局部网格通过球面旋转转换到正交坐标系：$\mathcal{N}(\hat{\mathbf{x}}^p)_r^o$
- 从正交代价体金字塔索引得到 $\mathcal{C}^o$，再转回原始格式 $\mathcal{C}^{o2p}$

两路相关性信息联合输入 ConvGRU 指导光流恢复。

#### 3. Ortho-Driven Distortion Compensation (ODDC)

ODDC 进一步利用正交视图的低畸变先验来补偿极区光流重建：

- **置信度计算**：使用 group-wise 相关性计算原始光流 $G^p$ 和正交光流 $G^{o2p}$ 的置信度图
- **运动特征编码**：包含三个浅层编码器 $\text{En}_c$（相关性）、$\text{En}_g$（置信度）、$\text{En}_f$（光流条件）
- **融合策略**：$m^p = [\text{En}_c(\mathcal{C}^p + \mathcal{C}^{o2p}), \text{En}_g([G^p, G^{o2p}]), \text{En}_f([\mathcal{F}^p, \mathcal{F}^{o2p}]), \mathcal{F}^p, \mathcal{F}^{o2p}]$
- 融合后的运动特征输入 ConvGRU 更新隐状态，解码残差流更新光流

### 损失函数 / 训练策略

- **球面加权 L1 损失**：考虑 ERP 投影的非均匀采样特性，对每个像素的 L1 损失乘以对应球面面积权重 $\omega^j$
- **双分支联合监督**：原始分支用原始 GT 监督，正交分支用旋转后的 GT 监督
- **指数递增权重**：沿迭代方向 $\gamma^{N-i}$（$\gamma=0.8$）递增权重
- **总损失**：$\mathcal{L} = \mathcal{L}_p + \mathcal{L}_o$

训练细节：
- AdamW 优化器，梯度裁剪 [-1, 1]，one-cycle 学习率调度
- 初始学习率 1e-4，使用 RAFT 在 FlyingThings 上的预训练权重
- MPFDataset：batch size 4，训练 60k 步
- FlowScape：batch size 6，训练 100k 步
- 迭代次数：训练和测试均为 12 次

## 实验关键数据

### 主实验

在 MPFDataset 和 FlowScape 两个基准上的对比：

| 方法 | 基线 | MPF-EFT EPE | MPF-EFT SEPE | MPF-City EPE | MPF-All EPE | FlowScape-All EPE | FlowScape-All SEPE |
|------|------|-------------|--------------|--------------|-------------|-------------------|-------------------|
| SphereNet+RAFT | RAFT | 13.2 | 15.7 | 8.28 | 10.7 | 12.9 | 21.0 |
| TanImg+RAFT | RAFT | 4.38 | 9.52 | 3.13 | 3.76 | 18.3 | 25.3 |
| SLOF | RAFT | 4.98 | 8.20 | 1.35 | 3.17 | 7.59 | 5.79 |
| PanoFlow | RAFT | - | - | - | - | 3.38 | 4.78 |
| PanoFlow | CSFlow | - | - | - | - | 3.31 | 4.44 |
| **PriOr-RAFT** | **RAFT** | **3.30** | **6.23** | **1.13** | **2.22** | **2.33** | **3.49** |

- 在 MPFDataset 上 EPE 降低 **30.0%**（vs SLOF），SEPE 降低 **20.9%**
- 在 FlowScape 上 EPE 降低 **29.6%**，SEPE 降低 **21.4%**（vs PanoFlow）

极区性能对比（FlowScape）：

| 方法 | 赤道 EPE | 赤道 SEPE | 极区 EPE | 极区 SEPE | 全区 EPE | 全区 SEPE |
|------|----------|-----------|----------|-----------|----------|-----------|
| PanoFlow | 0.52 | 2.87 | 6.25 | 6.68 | 3.38 | 4.78 |
| PriOr-RAFT | 0.53 | 2.94 | **4.13** | **4.03** | **2.33** | **3.49** |

极区 EPE 提升 **39.7%**，赤道区域性能基本持平。

### 消融实验

#### 模块有效性消融（MPFDataset EFT 场景）

| 模型 | 正交视图 | DCCL | ODDC | 极区 EPE | 极区 SEPE | 全区 EPE | 全区 SEPE |
|------|---------|------|------|----------|-----------|----------|-----------|
| RAFT Baseline | ✗ | ✗ | ✗ | 7.90 | 8.56 | 4.49 | 7.43 |
| +正交+DCCL | ✓ | ✓ | ✗ | 7.56 | 8.16 | 4.32 | 7.20 |
| **PriOr-RAFT (Full)** | ✓ | ✓ | ✓ | **5.57** | **6.47** | **3.30** | **6.23** |

ODDC 模块带来极区 EPE **26.3%** 的提升，是最关键的组件。

#### 通用性消融

| 模型 | EPE | EPE 降幅 | 运行时间 |
|------|-----|---------|---------|
| RAFT | 4.49 | - | 0.07s |
| PriOr-RAFT (4-iter) | 3.89 | ↓13.4% | 0.10s |
| PriOr-RAFT | 3.30 | ↓26.5% | 0.20s |
| GMA | 4.26 | - | 0.07s |
| PriOr-GMA (4-iter) | 3.51 | ↓17.6% | 0.10s |
| PriOr-GMA | 3.25 | ↓23.7% | 0.20s |
| SKFlow | 3.79 | - | 0.12s |
| PriOr-SKFlow (4-iter) | 3.33 | ↓12.1% | 0.13s |
| PriOr-SKFlow | 3.19 | ↓15.8% | 0.30s |

PriOr-GMA 仅用 4 次迭代就实现 17.6% 提升，推理时间几乎不变。

### 关键发现

1. **正交视图选择**：x 轴旋转 90° 效果最佳（EPE 3.30），y 轴旋转会破坏极区连续性（EPE 3.55），45° 旋转低畸变区域较小（EPE 3.41）
2. **迭代次数**：PriOr-RAFT 仅需 3 次迭代即可超越 baseline 的 12 次迭代结果，说明正交分支的运动特征能加速模型收敛
3. **置信度类型**：基于 warp+groupwise 相关性的动态置信度（EPE 3.30）优于固定的畸变图置信度（EPE 3.34），因其能反映实际光流置信度

## 亮点与洞察

1. **互补畸变的巧妙利用**：ERP 投影的极区高畸变问题长期困扰全景视觉，本文通过 90° 球面旋转产生畸变互补的正交视图，思路直觉且优雅
2. **通用性强**：DCCL 和 ODDC 模块可即插即用地集成到 RAFT、GMA、SKFlow 等多种迭代式光流网络中，均带来显著提升
3. **收敛加速**：引入正交分支信息后，仅需更少的迭代次数即可达到甚至超越 baseline 精度，这意味着可以在保持推理速度的前提下获得精度提升
4. **球面统一检索**：DCCL 在统一球面上进行坐标变换和关联查找，确保两个代价体的检索在几何上是对齐的

## 局限与展望

1. **赤道区域的轻微退化**：由于极区和赤道的畸变权衡，PriOr-RAFT 在赤道区域 EPE 略劣于 PanoFlow（0.53 vs 0.52），未来可探索自适应区域加权策略
2. **计算开销**：双分支结构使推理时间约翻倍（0.07s → 0.20s），在实时性要求高的场景中可能成为瓶颈
3. **仅评估合成数据的定量结果**：真实场景评估仅为定性（缺乏真实全景光流 GT 数据集），泛化性需进一步验证
4. **固定旋转角度**：当前只使用单一 90° 旋转，未探索多角度正交视图的潜在增益
5. **未考虑球面上的上下边界处理**：ERP 的上下边界（极点处）的特殊拓扑结构可能需要额外处理

## 相关工作与启发

- **RAFT** (2020)：迭代式光流估计的经典框架，本文直接在其基础上扩展
- **PanoFlow** (2023)：使用可变形卷积和光流畸变增广，但未显式处理极区
- **SLOF** (2022)：使用孪生表征学习和旋转增广，在跨视图光流相似性上添加约束
- **启发**：多视图互补的思想可推广到其他全景视觉任务（深度估计、语义分割等），通过不同投影视图的畸变互补来提升整体性能

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 正交视图互补畸变的思路简洁有效，DCCL+ODDC 的融合机制设计合理
- **技术质量**: ⭐⭐⭐⭐ — 消融全面，多基线集成验证通用性，定量定性结果充分
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集、多基线、多消融维度，但缺少真实场景定量评估
- **实用性**: ⭐⭐⭐⭐ — 即插即用，已开源，对全景光流领域有实际推动作用
- **总评**: ⭐⭐⭐⭐ — 思路优雅，效果显著（30% EPE↓），消融充分，是全景光流估计的扎实工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[CVPR 2026\] FlowFM: Advancing Dark Optical Flow Estimation with Flow Matching](../../CVPR2026/video_understanding/flowfm_advancing_dark_optical_flow_estimation_with_flow_matching.md)

</div>

<!-- RELATED:END -->
