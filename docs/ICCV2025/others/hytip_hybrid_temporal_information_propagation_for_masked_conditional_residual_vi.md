---
title: >-
  [论文解读] HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding
description: >-
  [ICCV 2025][视频压缩] 提出 HyTIP 框架，将输出回归（显式缓冲解码帧）和隐状态传播（隐式缓冲潜在特征）两种时序信息传播机制统一到同一学习式视频编码框架中，仅用 SOTA 方法 14% 的缓冲区大小即可达到可比的编码性能。 当前学习式视频编解码器可被理解为沿时间维度运行的循环神经网络（RNN）…
tags:
  - "ICCV 2025"
  - "视频压缩"
  - "时序信息传播"
  - "混合缓冲策略"
  - "RNN"
  - "条件残差编码"
---

# HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding

**会议**: ICCV 2025  
**arXiv**: [2508.02072](https://arxiv.org/abs/2508.02072)  
**代码**: [https://github.com/NYCU-MAPL/HyTIP](https://github.com/NYCU-MAPL/HyTIP)  
**领域**: 视频编码 / 学习式视频压缩  
**关键词**: 视频压缩, 时序信息传播, 混合缓冲策略, RNN, 条件残差编码

## 一句话总结

提出 HyTIP 框架，将输出回归（显式缓冲解码帧）和隐状态传播（隐式缓冲潜在特征）两种时序信息传播机制统一到同一学习式视频编码框架中，仅用 SOTA 方法 14% 的缓冲区大小即可达到可比的编码性能。

## 研究背景与动机

当前学习式视频编解码器可被理解为沿时间维度运行的循环神经网络（RNN），主要有两大时序信息传播范式：

**输出回归（Output-recurrence）**：显式缓冲前一帧解码图像作为参考。直觉简单，但解码帧需同时满足两个目标——既要逼近输入帧（重建质量），又要充当过去信息的充分摘要（时序传播）。这种双重约束导致率失真性能受限。

**隐到隐连接（Hidden-to-hidden）**：隐式缓冲潜在特征。灵活性更强，但通常需要存储 48+ 通道的全分辨率特征图，缓冲区开销巨大。

作者从 RNN 理论角度指出，同时采用两种连接方式的 RNN 在表达能力上最强，由此提出混合方案 HyTIP。

## 方法详解

### 整体框架

HyTIP 是基于帧级时序预测的编码框架，流程为：
1. **运动估计**：计算输入帧 $x_t$ 与参考帧 $\hat{x}_{t-1}$ 之间的光流 $f_t$
2. **运动编码**：将光流编码/解码为 $\hat{f}_t$
3. **帧间编码**：利用解码光流和混合缓冲区中的时序信息生成像素域预测器 $x_c$ 和多尺度特征域预测器 $\{C_1, C_2, C_3\}$，进行条件残差编码
4. **时序缓冲更新**：将当前信息存入混合缓冲区供下一帧使用

帧间编解码器采用掩码条件残差编码（Masked Conditional Residual Coding）：输入信号为 $x_t - m \odot x_c$，其中 $m$ 为逐像素软掩码，在条件编码和条件残差编码之间自适应切换。

### 关键设计

1. **混合缓冲策略（帧间编码）**：混合缓冲区同时存储显式信息（解码帧 $\hat{x}_{t-1}$，3通道）和隐式信息（少量潜在特征 $F_{t-1}$，仅2通道）。显式帧作为主要时序信源，隐式特征提供补充信息。利用"解码帧与当前帧高度相关"这一先验知识，大幅减少对隐式特征的依赖，仅需极小的隐式缓冲即可。相比纯隐式方案的 48+ 通道特征，混合方案仅需 5 通道（3+2），缓冲量减少约 90%。

2. **混合缓冲策略（运动编码）**：对光流编码也采用混合策略。光流信号通常空间上较平滑、时间上变化缓慢，显式缓冲解码光流 $\hat{f}_{t-1}$（2通道）加上少量隐式特征 $F^f_{t-1}$（0.125通道等效），无需运动补偿即可直接用于时序预测。

3. **掩码条件残差编码**：输入信号 $x_t - m \odot x_c = (1-m) \odot x_t + m \odot (x_t - x_c)$，通过软掩码 $m$ 在条件编码和残差编码之间逐像素切换，解决了纯条件编码的瓶颈问题和纯残差编码在遮挡区域的不足。

### 损失函数 / 训练策略

- 率失真优化目标：$R + \lambda D$，其中 $\lambda$ 控制码率和失真的权衡
- PSNR 优化模型：$\lambda \in [227, 2032]$ 随机采样
- MS-SSIM 优化模型：$\lambda \in [7, 46]$ 随机采样
- 训练分两阶段：先在 Vimeo-90K 上进行 5 帧训练，再在 BVI-DVC 上进行 10 帧微调
- 采用可变码率模型，支持单一模型在不同码率下工作

## 实验关键数据

### 主实验

| 方法 | UVG | MCL-JCV | HEVC-B | HEVC-C | HEVC-D | HEVC-E | HEVC-RGB | 平均 BD-rate(%) |
|------|-----|---------|--------|--------|--------|--------|----------|----------------|
| VTM 17.0 (anchor) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| HM 16.25 | +26.3 | +36.8 | +31.8 | +29.8 | +29.6 | +32.5 | +34.0 | +31.5 |
| DCVC-TCM | +16.1 | +27.4 | +29.3 | +59.8 | +18.8 | +61.3 | +24.3 | +33.9 |
| DCVC-HEM | -19.2 | -8.5 | -4.4 | +14.9 | -15.0 | +2.5 | -11.0 | -5.8 |
| DCVC-DC | -29.9 | -21.4 | -16.5 | -9.4 | -30.3 | -28.1 | -29.7 | -23.6 |
| DCVC-FM | -23.9 | -13.4 | -10.9 | -5.4 | -26.9 | -29.2 | -19.7 | -18.5 |
| **HyTIP (Ours)** | **-34.7** | **-25.0** | **-23.7** | **-16.4** | **-35.6** | **-21.2** | **-29.0** | **-26.5** |

HyTIP 以 PSNR-RGB 衡量平均节省 26.5% BD-rate，超过 VTM 17.0，与 DCVC-DC 可比（-23.6%），但缓冲仅需 7.875 通道 vs DCVC-DC 的 55.75 通道。

### 消融实验

| 运动编码缓冲 | 帧间编码缓冲 | 特征图数(显式+隐式) | 平均 BD-rate(%) |
|-------------|-------------|---------------------|----------------|
| Explicit (2+0) | Explicit (3+0) | 5 | 0 (anchor) |
| Implicit (0+4) | Explicit (3+0) | 7 | -12.2 |
| Hybrid (2+4) | Explicit (3+0) | 9 | -16.0 |
| Hybrid (2+0.125) | Explicit (3+0) | 5.125 | -14.7 |
| Hybrid (2+0.125) | Implicit (0+51) | 53.125 | -19.6 |
| Hybrid (2+0.125) | Implicit (0+5) | 7.125 | -15.0 |
| Hybrid (2+0.125) | Hybrid (3+48) | 53.125 | -21.9 |
| Hybrid (2+0.125) | **Hybrid (3+2)** | **7.125** | **-21.5** |

关键发现：Hybrid(3+2) 与 Hybrid(3+48) 性能几乎相同（-21.5% vs -21.9%），但缓冲量减少了 87%。

### 关键发现

1. **隐式缓冲对缓冲大小敏感**：纯隐式方案从 51 通道降到 5 通道时性能从 -19.6% 降至 -15.0%，下降显著
2. **混合缓冲对缓冲大小鲁棒**：混合方案从 48 通道降到 2 通道仅从 -21.9% 降至 -21.5%，基本无损
3. **长序列训练收益**：10 帧训练比 5 帧训练，混合方案额外获得 5.2% BD-rate 增益
4. **复杂度可控**：混合方案的模型大小和计算量与其他方案相当，额外开销可忽略

## 亮点与洞察

- 从 RNN 理论角度统一分析了学习式视频编码中的时序传播机制，框架清晰且有理论支撑
- "显式+隐式"的混合思路简洁高效，利用先验知识（解码帧与当前帧高相关性）大幅减少隐式特征需求
- 仅用 14% 的缓冲区大小即可匹配 SOTA，对硬件部署（片外内存带宽）非常友好
- 混合策略具有良好的可扩展性，可轻松迁移到其他学习式视频编解码器

## 局限与展望

- 目前仅验证在 RGB 域编码，未扩展到 YUV 编码（传统视频编解码器标准工作模式）
- HEVC-E（视频会议序列）上性能略弱于 DCVC-DC，表明静态背景场景下混合策略的优势不明显
- 训练流程与 DCVC-DC/DCVC-FM 不同，无法完全公平比较；若统一训练流程，性能或可进一步提升
- 未探索 B 帧结构（双向预测），仅支持 IPPP 结构

## 相关工作与启发

- DCVC 系列（DCVC-TCM/HEM/DC/FM）：纯隐式缓冲的 SOTA 方法，缓冲开销大
- MaskCRT：纯显式缓冲 + 掩码条件残差编码，本文在此基础上加入隐式缓冲
- 传统编码器（HM/VTM）：通常缓冲 4 帧参考帧，本文混合方案的缓冲量与之相当
- RNN 理论（Hammer, 2000）：兼具 output-recurrence 和 hidden-to-hidden 的 RNN 表达能力最强

## 评分

- **新颖性**: ⭐⭐⭐⭐ 从 RNN 视角统一分析视频编码的时序传播是新颖的切入点，混合策略虽概念简单但效果显著
- **实验充分度**: ⭐⭐⭐⭐⭐ 7 个测试数据集，多种缓冲策略对比，缓冲大小/长序列训练/复杂度全面消融
- **写作质量**: ⭐⭐⭐⭐ 论文结构清晰，RNN 类比恰当，图表丰富
- **价值**: ⭐⭐⭐⭐ 对学习式视频编码的缓冲设计提供了重要参考，实际部署时显著降低内存带宽需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](../../ICML2025/others/residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [\[NeurIPS 2025\] Sheaf Cohomology of Linear Predictive Coding Networks](../../NeurIPS2025/others/sheaf_cohomology_of_linear_predictive_coding_networks.md)
- [\[NeurIPS 2025\] Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](../../NeurIPS2025/others/hybrid-balance_gflownet_for_solving_vehicle_routing_problems.md)
- [\[ACL 2025\] The Harmonic Structure of Information Contours](../../ACL2025/others/the_harmonic_structure_of_information_contours.md)

</div>

<!-- RELATED:END -->
