---
title: >-
  [论文解读] Self-Supervised Sparse Sensor Fusion for Long Range Perception
description: >-
  [ICCV 2025][自动驾驶][长距离感知] LRS4Fusion 提出基于稀疏体素表示的长距离LiDAR-Camera融合方法，配合自监督预训练策略（通过稀疏占用和速度场重建），在250米感知范围内实现了目标检测 mAP 提升 26.6%、LiDAR预测 Chamfer Distance 降低 30.5% 的SOTA性能。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "长距离感知"
  - "稀疏体素融合"
  - "自监督预训练"
  - "LiDAR-Camera融合"
  - "深度估计"
---

# Self-Supervised Sparse Sensor Fusion for Long Range Perception

**会议**: ICCV 2025  
**arXiv**: [2508.13995](https://arxiv.org/abs/2508.13995)  
**代码**: [https://light.princeton.edu/LRS4Fusion](https://light.princeton.edu/LRS4Fusion)  
**领域**: 自动驾驶  
**关键词**: 长距离感知, 稀疏体素融合, 自监督预训练, LiDAR-Camera融合, 深度估计

## 一句话总结

LRS4Fusion 提出基于稀疏体素表示的长距离LiDAR-Camera融合方法，配合自监督预训练策略（通过稀疏占用和速度场重建），在250米感知范围内实现了目标检测 mAP 提升 26.6%、LiDAR预测 Chamfer Distance 降低 30.5% 的SOTA性能。

## 研究背景与动机

**领域现状**：当前自动驾驶感知方法主要基于 BEV（Bird's Eye View）表示，处理 50-100 米城市驾驶场景效果良好。主流方法如 BEVFusion、BEVFormer 使用密集 BEV 特征图进行 3D 目标检测、语义占用预测、跟踪和规划。

**现有痛点**：(1) BEV 表示的内存和计算开销随距离二次增长，扩展到 250 米以上极其困难；(2) 城际高速公路驾驶需要至少 250 米感知距离（时速 100km/h 的制动距离要求），而重型卡车（40吨）需要更长的规划horizon——这远超现有方法的 50-100 米范围；(3) 远距离目标极其稀疏（实例数随距离递减），标注成本巨大。

**核心矛盾**：密集 BEV 表示的二次复杂度使其无法扩展到长距离；自监督预训练方法（如 ViDAR、UnO）仅支持单一模态（相机或 LiDAR），无法充分利用多模态信息；且现有数据集 LiDAR 只覆盖 80 米。

**本文目标** (1) 如何在 250 米范围内高效融合多模态数据？(2) 如何在远距离目标极其稀疏的情况下进行自监督学习？(3) 如何在稀疏表示中保持时序信息？

**切入角度**：远距离场景天然稀疏（大部分空间是空的），稀疏体素表示与此天然匹配，避免了密集表示的二次增长。结合 FMCW LiDAR（可测径向速度，覆盖到 400 米）的新传感器能力。

**核心 idea**：用全稀疏体素表示替代密集 BEV，通过精确深度估计投影相机特征到3D后与 LiDAR 特征在稀疏体素空间融合，结合稀疏窗口注意力融合时序信息，并通过占用-速度自监督预训练学习长距离多模态表示。

## 方法详解

### 整体框架

LRS4Fusion 包含：(1) 相机编码器 + 深度估计 → 2D特征提升到3D稀疏体素；(2) LiDAR编码器 → 稀疏体素特征；(3) 稀疏融合模块合并双模态体素；(4) 稀疏后编码器（补全+上下文聚合）处理多尺度特征；(5) 稀疏窗口注意力融合时序信息；(6) 自监督预训练用占用和速度解码器。最终输出包括深度、占用、速度、未来 LiDAR 预测和目标检测。

### 关键设计

1. **精确深度估计模块**:

    - 功能：从 RGBD（RGB + 稀疏LiDAR深度投影）图像精确估计稠密深度图
    - 核心思路：多尺度循环架构，使用 Vim 骨干提取多尺度特征 $F_1^i, F_2^i, F_3^i, F_4^i = f_{img}(I_i^{RGBD})$。使用 Minimal Gated Unit (MGU) 替代 GRU 迭代优化深度图，每次迭代从上下文特征估计深度梯度 $\nabla d_t = F_g(h_t)$，然后融合稀疏 LiDAR 深度和图像梯度进行更新：$d_{t+1} = d_t - \Delta d$，其中 $\Delta d = f_{update}(\nabla d_t - g, (d_t - s_d) \odot M, C_{dg}, C_{inp})$
    - 设计动机：远距离深度估计是长距离感知的关键瓶颈。MGU 只有一个忘记门，比 GRU 减少 1/3 参数和计算量。推理仅 64ms、1.3GB 内存，比 CompletionFormer (188ms, 2.1GB) 和 OGNI-DC (364ms, 2.4GB) 快得多

2. **稀疏多模态融合**:

    - 功能：在统一的稀疏体素空间中融合相机和 LiDAR 特征
    - 核心思路：(1) 相机特征通过预测深度 $D_i$ 和相机矩阵 $K$ 投影到3D：$\mathbf{X}_C = D_i(u,v) K^{-1}(u,v,1)$，转换为稀疏体素 $F_C^i = [\mathbf{F}_C, \mathbf{X}_C]$。(2) LiDAR 通过体素化 PointNet + 稀疏 U-Net 得到 $F_L = [\mathbf{F}_L, \mathbf{X}_L]$。(3) 在稀疏融合模块中拼接两模态特征（单模态为空的体素填零），经批归一化和稀疏卷积融合：$F_{LC} = [\mathbf{F}_{LC}, \mathbf{X}_{LC}]$，总体素数 $Q = M + N - O$（$O$ 为重叠体素数）
    - 设计动机：全稀疏表示使得计算复杂度与占用体素数线性相关而非与空间范围二次相关，这使得扩展到 250 米成为可能

3. **稀疏窗口注意力时序融合**:

    - 功能：在保持稀疏性的同时融合历史帧信息
    - 核心思路：将上一帧体素通过刚体变换和速度校正对齐到当前帧：$\mathbf{X}_q^{t_0'} = (\mathbf{X}_q^{t_{-1}} + \mathbf{v}_q^{t_{-1}} dt) \mathbf{R|T}^{t_{-1} \to t_0}$。然后每个当前时刻的占用体素通过 3D 窗口注意力查询上一帧对齐后的邻近体素：$V_* = \sum_{V^{t_0'} \in J_s} \text{softmax}(\frac{V^{t_0} (V^{t_0'})^T}{\sqrt{d}}) V^{t_0'}$
    - 设计动机：简单拼接过去和当前体素会导致体素数随时间爆炸，丧失稀疏优势。通过以当前帧占用体素为 query、过去帧为 key/value 的窗口注意力，确保输出体素数不增长

4. **自监督预训练**:

    - 功能：从无标注数据中学习强大的多模态时空表示
    - 核心思路：稀疏占用和速度解码器接受 4D 查询点 $(x,y,z,t)$，在体素空间中插值后预测占用 $\hat{o}$ 和速度 $\hat{v}$。对于未来/过去时刻，通过轻量网络 $f_{pose}$ 预测查询点的新位置，然后从两个位置的插值特征预测结果。GT 来自 LiDAR 扫描：占用由点存在性决定，自由空间由 LiDAR 射线穿过确定，速度直接从 FMCW LiDAR 测量
    - 设计动机：远距离目标极其稀疏，标注成本极高。自监督预训练利用 6 万帧无标注数据学习时空表示，大幅减少标注需求

### 损失函数 / 训练策略

三阶段训练：(1) 训练图像特征编码器和深度预测（图像重建 + 深度监督 + 特征蒸馏 loss）；(2) 完整模型自监督训练（过去/当前/未来帧的占用和速度重建）；(3) 在此基础上训练目标检测头（CenterPoint）。

## 实验关键数据

### 主实验 - 长距离目标检测

| 方法 | 模态 | mAP↑ | NDS↑ |
|------|------|------|------|
| PointPillars | L | 39.31 | 41.52 |
| BEVFormer | C | 23.67 | 37.99 |
| BEVFormer (w/ ViDAR预训练) | C | 24.51 | 38.93 |
| BEVFusion | L+C | 40.10 | 48.43 |
| SAMFusion | L+C | 41.55 | 52.44 |
| LRS4Fusion (w/o 预训练) | L+C | 49.58 | 59.12 |
| **LRS4Fusion** | **L+C** | **52.61** | **58.06** |

相比第二好的 SAMFusion 提升 26.6%（+11.06 mAP）！

### 消融实验 - NuScenes LiDAR 预测

| 方法 | 模态 | 1s输入→1s预测 CD↓ | 1s输入→3s预测 CD↓ |
|------|------|-----|------|
| 4DOcc | L | 1.88 | - |
| ViDAR | C | 1.25 | 1.97 |
| **LRS4Fusion** | **L+C** | **0.48** | **1.25** |

在 NuScenes 1s→1s 任务上 CD 改善 61.6%，1s→3s 改善 36.5%。

### 深度估计对比

| 方法 | MAE↓ | RMSE↓ | 延迟(ms)↓ | 内存(GB)↓ |
|------|------|-------|-----------|-----------|
| CompletionFormer | 4.98 | 12.36 | 188 | 2.1 |
| OGNI-DC | 4.76 | 13.16 | 364 | 2.4 |
| **LRS4Fusion** | **3.46** | **9.21** | **64** | **1.3** |

### 关键发现

- **自监督预训练贡献显著**：目标检测 mAP 从 49.58 提升到 52.61（+6.11%），证明从无标注数据学习时空表示的价值
- **相机模态在远距离感知中不足**：BEVFormer 仅 23.67 mAP，因为单目相机在远距离缺乏深度线索。但精确的深度估计+LiDAR融合可以弥补
- **BEV 融合方法在远距离失效**：BEVFusion 仅比纯 LiDAR 的 PointPillars 好 2.01%，LSS 方法在远距离深度估计不准确
- **稀疏体素在全尺度保持稀疏**，进一步降低内存占用，支持更细粒度的离散化
- 在高速场景中 4DOcc 因帧间大位移性能下降（1s历史 CD 16.87 vs 3s历史 23.58），说明时序融合需要考虑运动校正

## 亮点与洞察

- **全稀疏表示是长距离感知的关键**：密集 BEV 在 250 米范围不可行，稀疏体素的复杂度与占用体素数线性相关，天然适配远距离的稀疏场景。这一设计理念可推广到任何需要大范围感知的应用
- **MGU 替代 GRU 的深度估计**：减少 1/3 参数且更快，在长距离场景中 MAE 改善 27%、推理速度提升 3 倍，是高效深度补全的优秀 baseline
- **自监督 LiDAR 预测作为预训练**：将未来 LiDAR 重建作为自监督任务，不需要任何标注数据就能学到有力的 3D 时空表示。这比 ViDAR（只用相机）和 UnO（只用 LiDAR）更通用

## 局限与展望

- 需要 FMCW LiDAR（可测速度），普通 LiDAR 无法直接获取体素级速度，限制了方法的通用性
- 自采数据集非公开，可复现性受限；仅与少数方法对比
- 仅使用一帧历史进行时序融合（$t_0$ 和 $t_{-1}$），未探索更长时序窗口的效果
- 目标检测仍需要标注数据微调（第三阶段），未实现完全无监督
- 稀疏窗口注意力的窗口大小是固定的，未自适应调整

## 相关工作与启发

- **vs BEVFusion/BEVFormer**: 密集 BEV 方法在短距离表现好但无法扩展到 250 米。LRS4Fusion 的稀疏表示在长距离显著优于这些方法
- **vs ViDAR**: 相机自监督预训练方法，在 NuScenes 上 CD 1.25（vs LRS4Fusion 0.48），差距在于缺乏深度线索和多模态信息
- **vs 4DOcc/UnO**: LiDAR 自监督方法，在高速场景（大帧间位移）性能显著下降。LRS4Fusion 的速度校正和多模态融合有效解决了此问题
- **vs SAMFusion**: 基于深度的3D投影方法，在长距离上 mAP 41.55 vs 52.61，说明精确深度+稀疏表示的组合优势

## 评分

- 新颖性: ⭐⭐⭐⭐ 稀疏体素融合+多模态自监督预训练的组合是创新点，但各组件有前人基础
- 实验充分度: ⭐⭐⭐⭐ 在自采长距离数据集和 NuScenes 上详尽验证，深度/检测/预测多任务
- 写作质量: ⭐⭐⭐⭐ 方法描述详细，问题动机清晰
- 价值: ⭐⭐⭐⭐⭐ 将感知距离从 50-100 米扩展到 250 米，对重卡自动驾驶有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[CVPR 2026\] TruckDrive: Long-Range Autonomous Highway Driving Dataset](../../CVPR2026/autonomous_driving/truckdrive_long-range_autonomous_highway_driving_dataset.md)
- [\[NeurIPS 2025\] Availability-aware Sensor Fusion via Unified Canonical Space](../../NeurIPS2025/autonomous_driving/availability-aware_sensor_fusion_via_unified_canonical_space.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](../../NeurIPS2025/autonomous_driving/layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)

</div>

<!-- RELATED:END -->
