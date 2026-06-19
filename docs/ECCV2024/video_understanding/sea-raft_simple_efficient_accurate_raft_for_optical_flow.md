---
title: >-
  [论文解读] SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow
description: >-
  [ECCV 2024][视频理解][光流] SEA-RAFT 通过混合拉普拉斯损失(MoL)、直接回归初始光流和刚性流预训练三项改进，在保持简洁架构的同时实现了 SOTA 精度，并比现有方法快 2.3× 以上。 领域现状： 光流估计是低层视觉的基础任务，用于动作识别、视频修复、帧插值、三维重建等下游任务…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "光流"
  - "RAFT"
  - "Mixture of Laplace"
  - "迭代优化"
  - "高效推理"
---

# SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow

**会议**: ECCV 2024  
**arXiv**: [2405.14793](https://arxiv.org/abs/2405.14793)  
**代码**: [https://github.com/princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)  
**领域**: 视频理解 / 光流估计  
**关键词**: 光流, RAFT, Mixture of Laplace, 迭代优化, 高效推理

## 一句话总结

SEA-RAFT 通过混合拉普拉斯损失(MoL)、直接回归初始光流和刚性流预训练三项改进，在保持简洁架构的同时实现了 SOTA 精度，并比现有方法快 2.3× 以上。

## 研究背景与动机

**领域现状**: 光流估计是低层视觉的基础任务，用于动作识别、视频修复、帧插值、三维重建等下游任务。当前 SOTA 方法大多基于 RAFT 架构，通过循环网络迭代精炼光流场。

**现有痛点**:
   - RAFT 类方法需要大量迭代（训练 12 次，推理高达 32 次），导致延迟严重
   - 标准 $L_1$ 损失在遮挡导致的**模糊样本**（ambiguous cases）上表现不佳，这些样本的高误差会主导训练损失
   - 零初始化光流偏离真值过远，收敛慢
   - 原始 RAFT 的自定义编码器和 ConvGRU 设计复杂，不利于扩展

**核心矛盾**: 精度与效率的矛盾——现有高精度方法（如 MS-RAFT+）速度极慢（SEA-RAFT 可达其 24× 速度），而高效方法精度大幅下降。

**本文目标** 在显著提升效率的同时达到或超越 SOTA 精度，并改善跨数据集泛化能力。

**切入角度**: 从损失函数设计（概率回归）、流场初始化策略、预训练数据策略和架构简化四个正交维度同时改进 RAFT。

**核心 idea**: 用混合拉普拉斯分布建模光流的不确定性，让网络学会区分"普通像素"和"模糊像素"，配合直接回归初始流和刚性流预训练，以仅 4 次迭代达到 SOTA。

## 方法详解

### 整体框架

SEA-RAFT 继承了 RAFT 的迭代精炼框架：特征编码器提取特征 → 构建多尺度 4D 相关体 → RNN 迭代精炼光流。关键改进包括三部分：(1) Mixture of Laplace 损失替代 $L_1$；(2) 上下文编码器直接回归初始光流；(3) 在 TartanAir 上做刚性流预训练。

### 关键设计

1. **Mixture of Laplace (MoL) 损失**:

   **功能**: 将光流预测建模为两个拉普拉斯分布的混合，一个处理普通情况，一个处理遮挡等模糊情况。

   **核心思路**: 每个像素预测混合系数 $\alpha$、尺度参数 $\beta_2$ 和均值 $\mu$。关键创新是**固定第一个分量的尺度 $\beta_1 = 0$**，使其等价于标准 $L_1$ 损失：

    $MixLap(x; \alpha, 0, \beta_2, \mu) = \alpha \cdot \frac{e^{-|x-\mu|}}{2} + (1-\alpha) \cdot \frac{e^{-\frac{|x-\mu|}{e^{\beta_2}}}}{2e^{\beta_2}}$

   训练损失为序列加权形式：$\mathcal{L}_{all} = \sum_{i=1}^{N} \gamma^{N-i} \mathcal{L}_{MoL}^i$

   **设计动机**: 
    - 普通像素由 $\alpha$ 接近 1 的第一分量主导，等价于 $L_1$ 损失，与评估指标对齐
    - 模糊像素（如重度遮挡）由第二分量处理，通过大 $\beta_2$ 降低对这些不可预测样本的惩罚
    - 在 log 空间中回归 $\beta$，避免数值不稳定
    - 与关键点匹配中的概率方法不同，光流需要对**每个像素**提供准确对应，因此必须将一个混合分量与 $L_1$ 对齐

2. **直接回归初始光流 (Direct Regression of Initial Flow)**:

   **功能**: 利用上下文编码器 $C$ 接收两帧堆叠输入，直接预测初始光流，替代 RAFT 的零初始化。

   **核心思路**: 将两帧图像堆叠后送入上下文编码器，回归一个初始光流估计及其 MoL 参数。这引入极少的额外计算开销（复用已有编码器）。

   **设计动机**: 零初始化可能与真值偏差很大，需要大量迭代才能收敛。通过 FlowNet 风格的直接回归提供合理的初始估计，可以显著减少所需的迭代次数（从 32 降到 4-12）。

3. **刚性流预训练 (Rigid-Flow Pre-Training)**:

   **功能**: 在 TartanAir 数据集上进行预训练。TartanAir 提供静态场景中由相机运动产生的光流标注。

   **核心思路**: 尽管 TartanAir 的运动多样性有限（仅相机运动），但其场景真实性和多样性远超合成数据集（FlyingChairs/Things），有助于提升泛化能力。

   **设计动机**: 现有训练数据（FlyingChairs、FlyingThings3D）规模和真实性有限。TartanAir 虽然仅有刚性流，但提供了更高的场景真实感，是一种低成本的数据增强策略。

### 架构简化

- **编码器**: 用标准 ImageNet 预训练 ResNet 替代 RAFT 的自定义编码器（不再需要不同的归一化层）
- **RNN**: 用 2 个 ConvNeXt 块替代 ConvGRU，参数更少、训练更稳定
- **迭代次数**: 训练和推理仅需 $N=4$（SEA-RAFT(S/M)），最多 $N=12$（SEA-RAFT(L)）

### 损失函数 / 训练策略

- 预训练: TartanAir 300K 步 → FlyingChairs 100K 步 → FlyingThings3D 120K 步 (即 "C+T")
- 微调: Sintel+Things+KITTI+HD1K 300K 步 ("C+T+S+K+H")
- 针对 Spring/KITTI 的额外微调
- $\gamma < 1$ 指数衰减早期迭代的权重
- $\beta_2$ 上界设为 10 保证训练稳定性

## 实验关键数据

### 主实验 — Spring Benchmark

| 方法 | 额外数据 | 微调 | Spring(test) 1px↓ | Spring(test) EPE↓ | Spring(test) WAUC↑ |
|------|---------|------|-------------------|-------------------|-------------------|
| RAFT | 无 | ✗ | 6.790 | 1.476 | 90.920 |
| FlowFormer | 无 | ✗ | 6.510 | 0.723 | 91.679 |
| MS-RAFT+ | VIPER | ✗ | 5.724 | 0.643 | 92.888 |
| CroCoFlow | CroCo | ✓ | 4.565 | 0.498 | 93.660 |
| **SEA-RAFT(S)** | TartanAir | ✓ | 3.904 | 0.377 | 94.182 |
| **SEA-RAFT(M)** | TartanAir | ✓ | **3.686** | **0.363** | **94.534** |

### 消融实验 — Spring subval

| 实验配置 | 初始流 | TartanAir预训练 | RNN类型 | 损失函数 | EPE |
|---------|--------|--------------|---------|---------|-----|
| SEA-RAFT (w/o Tar.) | ✓ | ✗ | 2×ConvNeXt | MoL ($\beta_1$=0) | 0.187 |
| SEA-RAFT (w/ Tar.) | ✓ | ✓ | 2×ConvNeXt | MoL ($\beta_1$=0) | **0.179** |
| w/o Direct Reg. | ✗ | ✗ | 2×ConvNeXt | MoL ($\beta_1$=0) | 0.201 |
| RAFT GRU | ✓ | ✗ | GRU | MoL ($\beta_1$=0) | 0.189 |
| Naive Laplace | ✓ | ✗ | 2×ConvNeXt | Single Laplace | 0.217 |
| Naive MoL | ✓ | ✗ | 2×ConvNeXt | MoL (两$\beta$均自由) | 0.248 |
| $L_1$ Loss | ✓ | ✗ | 2×ConvNeXt | $L_1$ | 0.206 |
| Mixture of Gaussian | ✓ | ✗ | 2×ConvNeXt | MoG | 0.210 |

### 关键发现

- **MoL 损失的 $\beta_1=0$ 约束至关重要**：自由 $\beta_1$ 的 Naive MoL (0.248) 反而比 $L_1$ (0.206) 更差，而固定 $\beta_1=0$ 的 MoL (0.187) 是最佳
- **直接回归初始流效果显著**：EPE 从 0.201 降至 0.187，仅增加 ~7G MACs
- **迭代瓶颈被消除**：RAFT 的迭代占总延迟 82-86%，SEA-RAFT 仅 26-39%
- **效率优势惊人**：SEA-RAFT(S) 处理 1080p 达 21fps (RTX3090)，比 RAFT 快 3×，比 MS-RAFT+ 快 24×
- Spring 基准 EPE 降低 22.9%（0.363 vs 0.471），1px 降低 17.8%（3.686 vs 4.482）
- KITTI 跨数据集泛化最优：Fl-epe 3.62、Fl-all 12.9

## 亮点与洞察

1. **概率建模与评估指标对齐的巧妙设计**：固定 $\beta_1=0$ 让 MoL 在普通情况退化为 $L_1$，在模糊情况自动宽容——一个简洁而优雅的解决方案
2. **"少即是多"的哲学**：仅 4 次迭代超越 32 次迭代的 RAFT，证明好的初始化比暴力迭代更重要
3. **三项改进的正交性**：损失函数、初始化、数据策略互补且均可独立加入其他 RAFT 变种
4. **架构简化的勇气**：用标准 ResNet 和 ConvNeXt 替代自定义模块，降低复杂度的同时提升性能

## 局限与展望

1. **Sintel Final pass 表现异常**: 在 C+T 设置下 Sintel Final 表现不佳（4.04 vs 竞争方法的 2.40），作者也未能解释原因，加入 KITTI+HD1K 后才改善
2. **MoL 损失的超参数敏感性**: $\beta_2$ 的上界 (10) 和混合分布形式需要实验调优
3. **TartanAir 预训练的局限**: 仅支持刚性流，缺乏独立物体运动
4. **未探索大规模真实数据预训练**: 如 DDVM 使用扩散模型预训练的思路
5. **缺乏不确定性估计的下游应用验证**: MoL 提供了不确定性输出 ($\alpha$, $\beta_2$)，但未在下游任务中验证其价值

## 相关工作与启发

- **RAFT 家族**: GMA、FlowFormer、CRAFT 等聚焦于替换模块（如 Transformer），而 SEA-RAFT 聚焦于损失函数和训练策略——两者正交可互补
- **概率回归**: PDC-Net+ 在关键点匹配中使用 MoL，但其不需要对齐 $L_1$；SEA-RAFT 的 $\beta_1=0$ 约束是针对光流"每像素都需准确"需求的独创适配
- **高效推理**: EMD-L、DIFT 等通过高效实现减少迭代，但精度下降明显；SEA-RAFT 通过初始流回归实现"质量替代数量"

## 评分

- **新颖性**: ⭐⭐⭐⭐ MoL 损失的 $\beta_1=0$ 约束和直接回归初始流的结合虽各自简单，但组合效果出色
- **实验充分度**: ⭐⭐⭐⭐⭐ 在 Spring/Sintel/KITTI 上全面评估，消融实验涵盖损失/初始化/预训练/架构所有维度
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，方法动机阐述充分，数学推导简洁
- **实用价值**: ⭐⭐⭐⭐⭐ 开源代码，2.3×-24× 加速，1080p@21fps 具有实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation](../../CVPR2026/video_understanding/efficient_all-pairs_correlation_volume_sampling_for_optical_flow_estimation.md)
- [\[ECCV 2024\] LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow](layeredflow_a_real-world_benchmark_for_non-lambertian_multi-layer_optical_flow.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](../../ICCV2025/video_understanding/prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)
- [\[CVPR 2026\] FlowFM: Advancing Dark Optical Flow Estimation with Flow Matching](../../CVPR2026/video_understanding/flowfm_advancing_dark_optical_flow_estimation_with_flow_matching.md)

</div>

<!-- RELATED:END -->
