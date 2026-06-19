---
title: >-
  [论文解读] Information-Bottleneck Driven Binary Neural Network for Change Detection
description: >-
  [ICCV 2025][遥感][二值神经网络] 提出 BiCD，首个专为变化检测设计的二值神经网络，通过信息瓶颈（IB）原理引导的辅助目标模块提升 BNN 的特征表示能力和可分离性，在街景和遥感变化检测数据集上达到 BNN 领域的 SOTA，同时实现 30× 内存压缩和 2.5× 推理加速。 领域现状 变化检测是计算机视觉中…
tags:
  - "ICCV 2025"
  - "遥感"
  - "二值神经网络"
  - "信息瓶颈"
  - "变化检测"
  - "模型压缩"
  - "辅助目标"
---

# Information-Bottleneck Driven Binary Neural Network for Change Detection

**会议**: ICCV 2025  
**arXiv**: [2507.03504](https://arxiv.org/abs/2507.03504)  
**代码**: 无  
**领域**: 遥感 / 变化检测  
**关键词**: 二值神经网络, 信息瓶颈, 变化检测, 模型压缩, 辅助目标

## 一句话总结

提出 BiCD，首个专为变化检测设计的二值神经网络，通过信息瓶颈（IB）原理引导的辅助目标模块提升 BNN 的特征表示能力和可分离性，在街景和遥感变化检测数据集上达到 BNN 领域的 SOTA，同时实现 30× 内存压缩和 2.5× 推理加速。

## 研究背景与动机

### 领域现状

变化检测是计算机视觉中的基础问题，广泛应用于城市地图更新、灾害评估、自动驾驶等领域。现有的深度神经网络方法虽然效果好，但计算和存储开销大，难以部署到边缘设备。

### 现有痛点

网络量化（特别是二值化）是极致的压缩手段，可实现 32× 存储压缩和 58× 计算加速。然而，直接将现有二值化技术应用于变化检测会导致严重的性能退化。根本原因在于：BNN 由于激进的二值化过程，导致输入 $X$ 和隐层特征 $Z$ 之间的互信息 $I(X,Z)$ 显著低于全精度网络，使得网络丧失区分"有意义变化"和"噪声变化"的关键特征粒度。

### 核心矛盾

BNN 需要极致压缩以适配边缘设备，但变化检测任务需要精细的特征表示来区分兴趣变化和噪声变化，这两个需求存在根本性冲突。

### 切入角度

从信息论角度出发，利用信息瓶颈（IB）原理来平衡特征压缩和信息保留，通过辅助目标模块增强 BNN 的特征可分离性，且该模块仅在训练时激活，推理时移除，不增加额外计算开销。

## 方法详解

### 整体框架

BiCD 基于 C-3PO 变化检测框架进行二值化改造。双时态特征金字塔由共享的 1-bit backbone 提取，通过 1-bit 变化生成器合并，经通道平均池化和 1-bit ASPP 模块输出变化掩码。在训练阶段，辅助模块从 1-bit 变化生成器的特征中产生维度对齐的特征，用于计算辅助损失。

### 关键设计

#### 1. **信息瓶颈驱动的辅助目标**

- **功能**：引入基于 IB 原理的辅助目标，增强编码器保留关键输入信息并提升特征可分离性的能力
- **核心思路**：将标准 IB 目标扩展为三项优化：

$$\min I(X, Z(\theta)) - \beta_1 I(Z(\theta), Y) - \beta_2 \Psi$$

其中 $\Psi = I(Z(\theta,\eta)_n, 0) + I(Z(\theta,\eta)_{in}, \Delta X_{in}) + I(X, Z(\theta,\eta))$

三项分别对应：抑制噪声变化、保留兴趣变化、重构原始输入
- **设计动机**：BNN 的 $I(X,Z)$ 天然较低，直接优化 IB 会进一步损害特征质量。通过辅助目标显式提升特征可分离性，同时通过重构损失隐式保留输入信息

#### 2. **辅助目标模块（Auxiliary Module）**

- **功能**：将隐层特征映射到与输入/标签维度对齐的空间，使互信息估计变得可行
- **核心思路**：辅助模块 $\sigma(\cdot, \eta)$ 由四个并行 MLP 分支和卷积输出层组成，将特征 $Z(\theta)$ 转换为维度对齐的 $Z(\theta,\eta)$，然后通过 L1 损失近似互信息
- **设计动机**：由于网络各层特征维度不一致，直接估计互信息不可行。辅助模块作为维度转换器，且仅在训练时使用，推理时移除无额外开销

#### 3. **噪声/兴趣变化分离机制**

- **功能**：利用变化掩码 $Y$ 将对齐后的特征分为"噪声变化" $Z(\theta,\eta)_n$ 和"兴趣变化" $Z(\theta,\eta)_{in}$ 两部分
- **核心思路**：噪声变化部分通过 $\|Z(\theta,\eta)_n\|_1$ 抑制至零，兴趣变化部分通过 $\|Z(\theta,\eta)_{in} - \Delta X_{in}\|_1$ 保留差异信息
- **设计动机**：变化检测的关键在于区分有意义的变化和环境噪声引起的无关变化，通过显式分离和针对性优化来实现

### 损失函数 / 训练策略

最终目标函数：

$$\min \text{Obj} = \beta_1 \|Z(\theta)\|_2 + L_{cd} + \beta_2(\|Z(\theta,\eta)_n\|_1 + \|Z(\theta,\eta) - X\|_1 + \|Z(\theta,\eta)_{in} - \Delta X_{in}\|_1)$$

- $\beta_1 = 1e{-3}$：控制冗余信息抑制率
- $\beta_2 = 0.08$：控制特征可分离性
- Adam 优化器，初始学习率 5e-4，余弦退火，训练 140 epochs
- 辅助模块初始学习率 5e-3，在 epoch 90 和 120 衰减至 1/10

## 实验关键数据

### 主实验

| 数据集 | 框架 | 方法 | Bits | F1-score(%) | vs BNN SOTA |
|--------|------|------|------|-------------|-------------|
| PCD-TSUNAMI | DR-TANet | BiCD | 1 | 85.1 | +2.5 (vs ReActNet 83.4) |
| PCD-TSUNAMI | C-3PO | BiCD | 1 | **86.5** | +2.5 (vs ReActNet 84.0) |
| PCD-GSV | DR-TANet | BiCD | 1 | 67.7 | +2.0 (vs ReActNet 65.7) |
| PCD-GSV | C-3PO | BiCD | 1 | **74.1** | +2.9 (vs ReActNet 71.2) |
| VL_CMU_CD | DR-TANet | BiCD | 1 | 65.9 | +3.3 (vs ReActNet 62.6) |
| VL_CMU_CD | C-3PO | BiCD | 1 | **71.9** | +2.0 (vs ReActNet 69.9) |
| LEVIR-CD | C-3PO | BiCD | 1 | **89.9** | +1.1 (vs ReActNet 88.8) |

值得注意的是，1-bit C-3PO + BiCD 在 TSUNAMI 上的 86.5% 超过了全精度 DR-TANet 的 87.6%，且参数量仅 2.1M（vs 33.4M），OPs 仅 6.6G（vs 28.5G）。

### 消融实验

| 配置 | 辅助目标位置 | F1-score(%) | 说明 |
|------|-------------|-------------|------|
| Baseline | 无 | 84.7 | 无辅助模块 |
| +BiCD (完整Ψ) | backbone | 84.8 | +0.1，Siamese中直接加可分离性效果差 |
| +BiCD (仅重构) | backbone | 85.2 | +0.5，重构损失隐式保留输入信息 |
| +BiCD (完整Ψ) | 1-bit generator | 85.8 | +1.1，可分离性需要与变化特征交互 |
| **+BiCD (最佳)** | **backbone + generator** | **86.5** | **+1.8，重构在backbone + 可分离性在generator** |

### 关键发现

- 可分离性目标必须放在 1-bit 变化生成器中才有效，直接放在 Siamese backbone 中反而有害
- 重构损失放在 backbone 中效果更好，因为需要直接与原始特征对交互
- 最佳配置是两者分离：backbone 做重构，generator 做可分离性
- 在 ARM Cortex-A76 边缘设备上，1-bit 版本延迟 158.4ms vs 全精度 392.8ms，实现 2.5× 加速
- BiCD 不增加推理延迟（辅助模块仅训练时使用）

## 亮点与洞察

1. **首次将 BNN 应用于变化检测**：开辟了新的研究方向，证明 1-bit 网络在变化检测中是可行的
2. **IB 原理的巧妙应用**：不直接优化 IB 目标（因为 BNN 的 $I(X,Z)$ 已经很低），而是引入辅助目标来"补偿"信息损失
3. **训练-推理解耦设计**：辅助模块仅在训练时激活，推理时完全移除，实现零额外开销的性能提升
4. **互信息信息平面分析**：通过信息平面可视化直观揭示 BNN 在变化检测中的瓶颈

## 局限与展望

1. 仅在 ResNet-18 backbone 上验证，未探索更深层次的二值化网络
2. 辅助模块的四分支 MLP 设计缺乏消融验证
3. 超参数 $\beta_1$、$\beta_2$ 需要在每个数据集上分别调优
4. 未与知识蒸馏等其他压缩方法进行结合或对比
5. LEVIR-CD 上相对全精度 SOTA（如 M-CD 的 92.1%）仍有明显差距

## 相关工作与启发

- C-3PO 作为变化检测的强基线框架，其高计算量（222G OPs）正好适合二值化来解决
- 信息瓶颈在模型压缩中的应用（与最小描述长度原理的等价性）提供了理论基础
- 辅助模块思想来自 local learning 框架，在 BNN 场景下的应用是自然的扩展

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个变化检测 BNN，IB 原理的应用有理论深度
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、两个框架、详细消融和边缘设备部署验证
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，信息平面分析直观
- **价值**: ⭐⭐⭐⭐ — 为资源受限场景的变化检测提供了实用方案，有实际部署意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] UniChange: Unifying Change Detection with Multimodal Large Language Model](../../CVPR2026/remote_sensing/unichange_unifying_change_detection_with_multimodal_large_language_model.md)
- [\[ICML 2025\] Neural Augmented Kalman Filters for Road Network Assisted GNSS Positioning](../../ICML2025/remote_sensing/neural_augmented_kalman_filters_for_road_network_assisted_gnss_positioning.md)
- [\[ICCV 2025\] GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration](geoexplorer_active_geo-localization_with_curiosity-driven_exploration.md)
- [\[CVPR 2026\] QuCNet: Quantum Deep Learning Driven Multi-Circuit Network for Remote Sensing Image Classification](../../CVPR2026/remote_sensing/qucnet_quantum_deep_learning_driven_multi-circuit_network_for_remote_sensing_ima.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](../../CVPR2025/remote_sensing/hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)

</div>

<!-- RELATED:END -->
