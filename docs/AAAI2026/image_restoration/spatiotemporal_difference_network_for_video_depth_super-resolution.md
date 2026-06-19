---
title: >-
  [论文解读] SpatioTemporal Difference Network for Video Depth Super-Resolution
description: >-
  [AAAI 2026][图像恢复][视频深度超分辨率] 基于视频深度超分辨率（VDSR）中空间非光滑区域和时间变化区域呈长尾分布的统计发现，提出 STDNet，通过空间差异分支（学习空间差异表示进行帧内 RGB-D 自适应聚合）和时间差异分支（利用时间差异表示在变化区域进行运动补偿），在 TarTanAir 数据集上 ×16 超分 RMSE 从 112.04cm 降至 96.80cm，平均超越 SOTA 方法 27.6%-32.6%。
tags:
  - "AAAI 2026"
  - "图像恢复"
  - "视频深度超分辨率"
  - "长尾分布"
  - "空间差异"
  - "时间差异"
  - "可变形卷积"
---

# SpatioTemporal Difference Network for Video Depth Super-Resolution

**会议**: AAAI 2026  
**arXiv**: [2508.01259](https://arxiv.org/abs/2508.01259)  
**代码**: [yanzq95/STDNet](https://github.com/yanzq95/STDNet)  
**领域**: 图像复原  
**关键词**: 视频深度超分辨率, 长尾分布, 空间差异, 时间差异, 可变形卷积  

## 一句话总结

基于视频深度超分辨率（VDSR）中空间非光滑区域和时间变化区域呈长尾分布的统计发现，提出 STDNet，通过空间差异分支（学习空间差异表示进行帧内 RGB-D 自适应聚合）和时间差异分支（利用时间差异表示在变化区域进行运动补偿），在 TarTanAir 数据集上 ×16 超分 RMSE 从 112.04cm 降至 96.80cm，平均超越 SOTA 方法 27.6%-32.6%。

## 背景与动机

### 深度超分辨率的发展

深度数据在 3D 重建、虚拟现实、增强现实等领域至关重要。近年来大量深度超分辨率（DSR）方法被提出，从低分辨率（LR）深度图重建高分辨率（HR）深度图。单帧 DSR 方法已取得显著进展，包括滤波方法、多模态融合方法、多任务协作方法和结构导向方法等。视频深度超分辨率（VDSR）通过聚合多帧 RGB-D 特征，进一步提升了重建质量。

### 长尾分布问题

作者对 VDSR 进行了统计分析，发现了跨空间和时间两个维度的长尾分布现象：

- **空间维度**：GT 深度与上采样 LR 深度之间的差异主要集中在非光滑区域（边缘、结构变化处），这些区域在整体数据中仅占少量，但重建难度远高于主导的平滑区域。
- **时间维度**：连续帧和跨帧之间的深度差异主要集中在时间变化区域（动态物体、边缘轮廓、遮挡区域），同样呈现长尾特征。

### 现有方法的不足

现有 VDSR 方法（如 DVSR）虽然引入了多帧特征聚合，但未显式建模这些长尾分布特征。单模态视频 RGB 超分辨率方法在建立多帧、多模态 RGB-D 对应关系方面效果有限。需要一种能专门处理空间和时间长尾区域的框架。

## 核心问题

如何在视频深度超分辨率中针对性地增强空间非光滑区域和时间变化区域这两类长尾分布区域的重建质量，同时保持时间一致性？

## 方法详解

### 整体架构

STDNet 包含两个核心分支：

1. **空间差异分支**：从 LR 深度视频预测空间差异表示 $\boldsymbol{\sigma}$，引导帧内 RGB 特征与深度非光滑区域对齐聚合
2. **时间差异分支**：估计连续帧差异 $\boldsymbol{\varphi}$ 和跨帧差异 $\hat{\boldsymbol{\varphi}}$，优先在时间变化区域进行多帧 RGB-D 聚合

此外引入差异正则化损失来优化时空差异表示的学习。

### 空间差异分支

**空间差异表示**：通过对深度特征进行下采样-上采样操作，捕捉非光滑区域信息：

$$\boldsymbol{\sigma} = |\boldsymbol{F}_d - f_{bu}(f_{bd}(\boldsymbol{F}_d))|$$

**空间差异机制**：

1. 从 $\boldsymbol{\sigma}_t$ 生成滤波核 $\boldsymbol{k}_t = \mathcal{G}(\boldsymbol{\sigma}_t)$，用于对齐 RGB 特征到非光滑深度区域
2. 从 $\boldsymbol{\sigma}_t$ 编码自适应权重 $\boldsymbol{w}_t = \mathcal{E}_w(\boldsymbol{\sigma}_t)$（经过卷积、max、mean、sigmoid）
3. 加权聚合：$\boldsymbol{F}_{sd}^t = f_c(\boldsymbol{F}_d^t, \boldsymbol{w}_t \otimes \boldsymbol{F}_r^t, \mathcal{F}(\boldsymbol{F}_r^t, \boldsymbol{k}_t))$

该机制通过空间差异表示驱动 RGB 信息选择性地传播到深度非光滑区域，有效缓解长尾效应。

### 时间差异分支

**时间差异表示**：

$$\boldsymbol{\varphi}_t = |\boldsymbol{F}_{sd}^t - \boldsymbol{F}_{sd}^{t+1}|, \quad \hat{\boldsymbol{\varphi}}_t = |\boldsymbol{F}_{sd}^t - \boldsymbol{F}_{sd}^{t+2}|$$

分别捕获连续帧和跨帧的时间变化信息。

**时间差异策略**：采用双向迭代方案，包含邻近帧融合和跨帧融合两个阶段：

- **邻近帧融合**：将时间差异 $\boldsymbol{\varphi}_{t-1}$ 通过编码器 $\mathcal{E}_\varphi$ 生成偏移量 $\delta_{t-1}$ 和调制标量 $m_{t-1}$，利用可变形卷积 $\mathcal{D}$ 动态采样时间变化信息。同时用空间差异权重 $\boldsymbol{w}_t$ 缓解跨模态差异：

$$\boldsymbol{F}_f^{t-1,t} = f_c(\boldsymbol{F}_f^t, \mathcal{D}(\boldsymbol{F}_f^{t-1}, \delta_{t-1}, m_{t-1}), \boldsymbol{w}_t \otimes \mathcal{D}(\boldsymbol{F}_r^{t-1}, \delta_{t-1}, m_{t-1}))$$

- **跨帧融合**：类似地使用 $\hat{\boldsymbol{\varphi}}_{t-2}$ 处理 $t-2$ 帧
- 最终融合：$\hat{\boldsymbol{F}}_f^t = \boldsymbol{F}_f^{t-1,t} + \boldsymbol{F}_f^{t-2,t}$

### 差异正则化损失

总损失由重建损失和差异正则化组成：

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \beta \mathcal{L}_{diff}$$

其中 $\mathcal{L}_{rec}$ 使用 Charbonnier 正则化，$\mathcal{L}_{diff} = \alpha_1 \mathcal{L}_{sd} + \alpha_2 \mathcal{L}_{td}$。

- **空间差异损失**：引入不确定性约束，在非光滑区域施加更大的重建误差惩罚：$\mathcal{L}_{sd} = \sum_q (\boldsymbol{\sigma}^q - \min(\boldsymbol{\sigma}^q)) \|\boldsymbol{D}_{GT}^q - \boldsymbol{D}_{HR}^q\|_1$
- **时间差异损失**：约束时间差异表示与 GT 深度的时间变化一致，包含邻近帧和跨帧两项

超参数设置：$\alpha_1 = \alpha_2 = 0.5$，$\beta = 0.01$。

## 实验关键数据

### 表1：TarTanAir 数据集定量对比

| 方法 | 会议 | ×4 RMSE↓ | ×4 MAE↓ | ×8 RMSE↓ | ×8 MAE↓ | ×16 RMSE↓ | ×16 MAE↓ | ×16 TEPE↓ |
|------|------|----------|---------|----------|---------|-----------|----------|-----------|
| DJFR | PAMI'19 | 75.56 | 10.59 | 105.45 | 18.43 | 141.14 | 31.22 | 20.27 |
| DKN | IJCV'21 | 82.69 | 11.73 | 110.10 | 18.78 | 153.56 | 33.21 | 21.93 |
| SGNet | AAAI'24 | 79.40 | 11.36 | 116.33 | 23.15 | 144.17 | 34.34 | 20.14 |
| DORNet | CVPR'25 | 63.38 | 8.60 | 93.75 | 13.96 | 123.24 | 23.59 | 16.40 |
| DVSR | CVPR'23 | 57.72 | 4.40 | 76.96 | 7.74 | 112.04 | 14.39 | 11.06 |
| **STDNet** | - | **50.28** | **3.73** | **72.03** | **6.75** | **96.80** | **12.01** | **8.90** |

在 ×16 超分下，STDNet 相比 DVSR 降低 RMSE 15.24cm、MAE 2.38cm、TEPE 2.16cm。

### 表2：DyDToF 数据集泛化评测

| 方法 | ×4 RMSE↓ | ×8 RMSE↓ | ×16 RMSE↓ | ×16 MAE↓ |
|------|----------|----------|-----------|----------|
| DVSR | 19.53 | 27.63 | 43.55 | 9.80 |
| **STDNet** | **18.23** | **26.87** | **39.24** | **8.72** |

无需微调即在 DyDToF 上超越 DVSR，×16 RMSE 降低 4.31cm。

### 表3：DynamicReplica 数据集泛化评测

| 方法 | ×4 RMSE↓ | ×8 RMSE↓ | ×16 RMSE↓ |
|------|----------|----------|-----------|
| DVSR | 0.37 | 0.58 | 1.25 |
| **STDNet** | **0.32** | **0.53** | **1.10** |

### 消融实验

| 变体 | TarTanAir ×4 RMSE↓ | DyDToF ×4 RMSE↓ |
|------|-------------------|-----------------|
| Baseline（拼接替代 SD+TD）| ~60+ | ~36+ |
| +SD（仅空间差异）| RMSE 降低 3.56cm | — |
| +TD（仅时间差异）| RMSE 降低 14.02cm | — |
| **+SD+TD（完整 STDNet）**| **降低 17.94cm** | — |

去掉差异正则化损失：TarTanAir ×16 RMSE 提升 7.08cm，验证了损失设计的有效性。

### 模型复杂度

与单帧方法相比，STDNet 平均减少 9.23M 参数和 35.82cm RMSE。与多帧 DVSR 相比，推理速度提升 47.35ms，性能提升 4.93cm RMSE，仅增加 4.4M 参数。

## 亮点

1. **从统计分析出发的问题定义**：不是盲目设计网络，而是先对 VDSR 做直方图统计分析，发现空间和时间两个维度的长尾分布特征，据此针对性设计解决方案——这种数据驱动的问题发现方式值得借鉴。
2. **空间差异机制的简洁有效性**：仅通过下采样-上采样差异即可得到空间差异表示（类比 Laplacian 金字塔思想），再用其生成滤波核和权重来引导 RGB-D 聚合，设计简洁但效果显著。
3. **时间差异与可变形卷积的优雅结合**：将时间差异表示转化为可变形卷积的偏移量和调制标量，使得运动补偿自然聚焦于时间变化显著的长尾区域。
4. **全面的跨数据集泛化性**：在 TarTanAir 上训练后，无需微调即在 DyDToF 和 DynamicReplica 上取得一致的性能提升，证明了方法的鲁棒性。
5. **时间一致性的显著改善**：x-t 切片可视化清晰展示了 STDNet 在时间变化区域产生更稳定的深度预测。

## 局限与展望

1. **合成数据训练**：仅在合成数据集（TarTanAir、DyDToF、DynamicReplica）上实验，未验证在真实传感器采集的有噪声深度视频上的效果。
2. **固定的邻近帧数量**：实验发现使用 2 帧（1 邻近+1 跨帧）时性能最佳，但这种经验设定可能不适用于运动剧烈或长时遮挡场景，缺乏自适应帧选择机制。
3. **光流依赖缺失**：空间差异表示通过简单的下采样-上采样差异计算，时间差异通过帧间特征差异计算，均未使用光流进行显式运动估计，在大位移场景下可能受限。
4. **计算开销**：虽然比 DKN 等单帧方法参数更少，但双向迭代+可变形卷积带来的计算量增加，对实时应用仍有挑战。
5. **仅关注深度超分**：未扩展到深度补全、深度去噪等相关任务，方法的通用性未被充分验证。

## 与相关工作的对比

- **DVSR (Sun et al., CVPR 2023)**：首个 dToF-based VDSR 方法，通过多帧融合缓解空间模糊，但未显式处理长尾分布。STDNet 在 TarTanAir ×4/×8/×16 上平均超越 DVSR 32.6%/28.8%/27.6%。
- **DORNet (Wang et al., CVPR 2025)**：最新单帧 DSR 方法，性能显著弱于多帧方法（×16 RMSE 123.24 vs STDNet 96.80），验证了多帧信息的重要性。
- **BasicVSR++ (Chan et al.)**：视频 RGB 超分中的双向循环框架，STDNet 借用了其双向迭代思想，但针对 VDSR 设计了时间差异驱动的聚合策略。
- **SVDC (Zhu et al., 2025)**：视频深度补全框架，通过自适应频率选择融合多帧特征。STDNet 专注于深度超分中的长尾分布问题，设计思路不同。

## 启发与关联

- 长尾分布视角可推广到其他视频恢复任务（如视频去模糊、视频降噪），这些任务中边缘和运动区域同样呈非均匀分布。
- 空间差异表示（下采样-上采样差异）作为非光滑区域检测手段，计算轻量且无需额外监督，可作为注意力权重的生成方式应用到其他多模态融合任务。
- 时间差异驱动的可变形卷积设计可启发视频深度估计和视频光流估计中的运动建模。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 长尾分布视角和时空差异驱动的双分支设计新颖，但单个模块（可变形卷积、双向迭代）均为已有技术的组合应用
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、三个缩放因子、11 种对比方法、详细消融（SD/TD/损失/帧数）、复杂度分析、PCA 可视化，实验全面
- **写作质量**: ⭐⭐⭐⭐ — 统计分析→问题发现→方法设计的叙事逻辑清晰，图表质量高，直方图对比直观展示长尾缓解效果
- **价值**: ⭐⭐⭐⭐ — 在 VDSR 任务上取得显著提升（×16 平均 27.6%），长尾分布视角可推广到其他恢复任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Dual Graph Regularized Deep Unfolding Network for Guided Depth Map Super-resolution](../../CVPR2026/image_restoration/dual_graph_regularized_deep_unfolding_network_for_guided_depth_map_super-resolut.md)
- [\[AAAI 2026\] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment](temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)
- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[AAAI 2026\] Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)
- [\[CVPR 2026\] TextOVSR: Text-Guided Real-World Opera Video Super-Resolution](../../CVPR2026/image_restoration/textovsr_text-guided_real-world_opera_video_super-resolution.md)

</div>

<!-- RELATED:END -->
