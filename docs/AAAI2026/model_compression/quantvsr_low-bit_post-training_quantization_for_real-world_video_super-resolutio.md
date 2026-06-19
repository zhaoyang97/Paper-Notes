---
title: >-
  [论文解读] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution
description: >-
  [AAAI 2026][模型压缩][视频超分辨率] 提出 QuantVSR，首个面向扩散模型视频超分（VSR）的低比特（4/6-bit）后训练量化框架：通过时空复杂度感知（STCA）机制实现层自适应秩分配，并引入可学习偏置对齐（LBA）模块缓解低比特量化偏差，在 4-bit 设置下将参数量压缩 84.39%、计算量压缩 82.56%，同时保持与全精度模型接近的性能。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "视频超分辨率"
  - "低比特量化"
  - "后训练量化"
  - "扩散模型压缩"
  - "时空复杂度"
---

# QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution

**会议**: AAAI 2026  
**arXiv**: [2508.04485](https://arxiv.org/abs/2508.04485)  
**代码**: [https://github.com/bowenchai/QuantVSR](https://github.com/bowenchai/QuantVSR)  
**领域**: 图像生成  
**关键词**: 视频超分辨率, 低比特量化, 后训练量化, 扩散模型压缩, 时空复杂度

## 一句话总结

提出 QuantVSR，首个面向扩散模型视频超分（VSR）的低比特（4/6-bit）后训练量化框架：通过时空复杂度感知（STCA）机制实现层自适应秩分配，并引入可学习偏置对齐（LBA）模块缓解低比特量化偏差，在 4-bit 设置下将参数量压缩 84.39%、计算量压缩 82.56%，同时保持与全精度模型接近的性能。

## 研究背景与动机

### 领域现状

视频超分辨率（VSR）旨在从低分辨率视频恢复高分辨率细节。基于扩散模型的 VSR 方法（如 MGLD-VSR）凭借强大的生成先验，在真实世界场景下取得了显著的视觉质量提升，远超 GAN 方法。然而扩散模型的推理极其缓慢且资源消耗大，严重制约了在边缘设备上的部署。

### 现有方法的痛点

量化是模型压缩的有效手段，已在图像/视频生成（SVDQuant、ViDiT-Q）和图像恢复（PassionSR）中取得进展。但将量化应用于 VSR 模型面临两个独特挑战：

**时序一致性丧失**：模型量化在不同帧上引入不一致的误差，破坏生成视频的时序连贯性。

**复杂数据分布**：VSR 模型将时序动态嵌入到潜在特征中，导致激活分布更加复杂——需要同时考虑空间和时间两个维度的特征才能弥合全精度与量化模型的性能差距。

### 核心矛盾

低比特量化（如 4-bit）用极有限的整数值表示浮点权重和激活的广泛动态范围，恢复能力急剧下降。现有通用量化方法（如 SVDQuant）采用固定秩的全精度分支来缓解，但存在两个问题：(1) 秩分配策略不够优化，带来不必要的计算开销；(2) 全精度分支可能降低低比特分支的性能，导致整体次优。

### 本文切入角度

**核心 idea**：利用 VSR 输入的时空特性来指导量化——根据每层输入的时间复杂度（帧间差异）和空间复杂度（空间方差）自适应分配全精度分支的秩，复杂层给更高秩以保留信息，简单层用最低秩以节省计算。同时引入可学习偏置对齐来修正低比特量化的系统偏差。

## 方法详解

### 整体框架

QuantVSR 基于 MGLD-VSR 的 UNet 结构，将原始层（Linear、Conv2d、Conv3d）替换为自定义量化层。每个量化层包含：
- **全精度分支（FP Branch）**：低秩矩阵 $L_1 L_2$ 跳过量化
- **低比特分支**：经 Hadamard 变换平滑后的量化计算
- **可学习偏置对齐（LBA）**：修正量化偏差

量化层公式：
$$\boldsymbol{XW} = \underbrace{\boldsymbol{XL_1L_2}}_{\text{FP, STCA}} + \underbrace{Q_A(\boldsymbol{XH})Q_W(\boldsymbol{H}^\top\boldsymbol{R})}_{\text{Low-Bit}} + \underbrace{\boldsymbol{A}_{\text{bias}}}_{\text{LBA}}$$

校准过程分三阶段：分析时空复杂度 → 联合精炼两分支 → 训练 LBA 模块。

### 关键设计

#### 1. **时空复杂度感知机制（STCA）**

**功能**：根据每层输入的时间和空间复杂度为全精度分支自适应分配秩，实现性能与效率的平衡。

**时间复杂度**定义为帧间差异能量：
$$C_t = \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{1}{CHW} \|\boldsymbol{X}_{t+1} - \boldsymbol{X}_t\|_2^2$$

值越高表示帧间运动越剧烈，恢复难度越大。

**空间复杂度**定义为空间方差的均值：
$$C_s = \frac{1}{TC} \sum_{t=1}^T \sum_{c=1}^C \sigma_{h,w}(\boldsymbol{X}_{t,c})$$

空间方差越高表示纹理/边缘等特征越复杂，信息量越丰富。

**秩分配策略**：
- 使用校准集计算每层的时空复杂度分布，设定上下阈值（25th/75th 百分位）
- 若时间和空间复杂度均超过上阈值：秩 +1
- 若均低于下阈值：秩 -1
- 否则保持不变
- 最终约束到 $[r_{\min}, r_{\max}] = [16, 64]$，并取 8 的倍数

**设计动机**：全精度分支的秩决定了计算负担（$r\frac{m+n}{mn}$ 线性增长）。统一高秩浪费计算，统一低秩信息不足。按层复杂度自适应分配使用最少的计算保留最多的信息。实际结果：平均秩仅 24，低于 SVDQuant 的固定秩 32。

#### 2. **双分支联合精炼（Dual-Branch Refinement）**

**功能**：在层自适应秩分配后，联合优化全精度分支和低比特分支以达到整体最优。

**核心思路**：全精度分支改变了低比特分支的数据分布（残差 $R = W - L_1L_2$），可能使量化更困难。两个分支共同贡献最终输出，因此需要联合优化。

- 用 SVD 初始化 $L_1$、$L_2$，提供良好起点避免从零训练的慢收敛
- 在校准集上进行少量训练，以 FP 和量化模型输出的 MSE 为目标

**设计动机**：增强 FP 分支（提高秩）可能反而降低低比特分支性能，联合精炼确保两个分支在整体上达到最优平衡。

#### 3. **可学习偏置对齐模块（LBA）**

**功能**：修正低比特量化中的系统偏差（biased error）。

**问题分析**：量化误差是有偏的，即全精度模型和量化模型的平均输出不同。当权重和激活同时量化时，偏差公式为：
$$\mathbb{E}(\hat{W}\hat{X}) - \mathbb{E}(WX) = \Delta W \mathbb{E}(\hat{X}) + W \mathbb{E}(\Delta X)$$

这个偏差受激活量化误差 $\Delta X$ 的影响，无法简单用统计量修正。

**核心思路**：在量化层输出后添加一个可学习偏置 $A_{\text{bias}}$，与层偏置维度相同，参数量极小（相对整个模型可忽略），收敛快。在推理时可直接融入层偏置，零额外计算开销。

**训练方式**：冻结所有其他参数后独立训练 LBA。

### 损失函数 / 训练策略

- **校准数据**：从 REDS30 的 FP UNet 中在去噪过程中等间隔采样（50 步中取 5 个点），得到 1800 个 input-output 对，每个输入形状 $5 \times 4 \times 64 \times 64$
- **训练目标**：FP 与量化模型输出的 MSE + STE 梯度近似
- **训练设备**：NVIDIA RTX A6000，2 个 epoch
- **学习率**：第一 epoch $1 \times 10^{-3}$，第二 epoch $2 \times 10^{-4}$

## 实验关键数据

### 主实验

在合成数据集（REDS4、SPMCS）和真实世界数据集（MVSR4x）上的结果：

**REDS4 W4A4 量化结果**（最具挑战性的设置）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DOVER ↑ | E*warp ↓ |
|------|--------|--------|---------|---------|----------|
| MGLD-VSR (FP) | 23.27 | 0.6180 | 0.2117 | 0.6761 | 7.24 |
| MaxMin | 16.18 | 0.1995 | 0.6720 | 0.1451 | 52.27 |
| Q-Diffusion | 19.99 | 0.3176 | 0.5279 | 0.4936 | 19.63 |
| SVDQuant | 21.19 | 0.4138 | 0.4718 | 0.5865 | 12.46 |
| **QuantVSR** | **23.31** | **0.6143** | **0.2286** | **0.6822** | **6.88** |

**MVSR4x W4A4（真实世界）**：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DOVER ↑ | E*warp ↓ |
|------|--------|--------|---------|---------|----------|
| MGLD-VSR (FP) | 22.77 | 0.7422 | 0.3571 | 0.6321 | 1.54 |
| SVDQuant | 21.70 | 0.5021 | 0.5780 | 0.4727 | 3.30 |
| **QuantVSR** | **22.90** | **0.7367** | **0.3590** | **0.6219** | **1.40** |

### 消融实验

**SPMCS W4A4 各组件消融**：

| 配置 | SC | LBA | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DOVER ↑ | E*warp ↓ |
|------|-----|-----|--------|--------|---------|---------|----------|
| 无跳连 | ✗ | ✗ | 17.13 | 0.2595 | 0.6480 | 0.0802 | 10.00 |
| 无跳连 | ✗ | ✓ | 21.38 | 0.4996 | 0.3666 | 0.6515 | 3.27 |
| SVDQuant SC | ✓ | ✗ | 18.94 | 0.2820 | 0.5921 | 0.4028 | 6.40 |
| SVDQuant SC | ✓ | ✓ | 22.58 | 0.5783 | 0.3296 | 0.6673 | 1.90 |
| **STCA** | ✓ | ✗ | 22.75 | 0.6071 | 0.2914 | 0.6886 | 1.74 |
| **STCA** | ✓ | ✓ | **22.76** | **0.6075** | **0.2857** | **0.6969** | **1.76** |

**压缩比**：

| 量化设置 | 参数量 (M) | 压缩比 | 计算量 (G) | 压缩比 |
|---------|-----------|--------|-----------|--------|
| W32A32 (FP) | 935 | 0% | 1881 | 0% |
| W8A8 | 263 | 71.87% | 563 | 70.07% |
| W6A6 | 204 | 78.18% | 446 | 76.29% |
| W4A4 | 146 | 84.39% | 328 | 82.56% |

### 关键发现

1. **4-bit 量化几乎无损**：QuantVSR W4A4 的 PSNR（23.31）甚至微超 FP 模型（23.27），这在其他方法中完全不可能（SVDQuant 仅 21.19）。
2. **STCA 大幅优于 SVDQuant 的固定秩策略**：PSNR 从 22.58 提升到 22.75，同时平均秩更低（24 vs. 32），计算更少。
3. **LBA 在性能退化严重时效果最为显著**：无跳连时 PSNR 从 17.13 跃升至 21.38。
4. **时序一致性指标 E*warp 是区分性最强的指标**：QuantVSR 在 REDS4 W4A4 下为 6.88，而 SVDQuant 高达 12.46（劣化近一倍），证明了时空感知设计的必要性。
5. 一些方法在无参考 IQA（如 CLIP-IQA）上得分甚至超过 FP 模型，但其结构性指标（PSNR/SSIM）很差——有噪声的图像也可以获得高无参考分数。

## 亮点与洞察

1. **首次将低比特量化系统性应用于扩散 VSR 模型**：填补了该领域的空白，且方法具有通用性。
2. **时空复杂度驱动的秩分配**：将领域特定知识（视频的时间动态 + 空间纹理）融入量化策略，比通用方案更高效。
3. **LBA 的推理零成本性**：可学习偏置在推理时融入层偏置，完全不增加推理开销。
4. **系统性实验设计**：既评估图像质量（IQA），也评估视频质量（VQA）和时序一致性（E*warp），指标维度全面。

## 局限与展望

1. 仅在 MGLD-VSR（基于 U-Net）上验证，对 DiT 架构的 VSR 模型（如基于视频生成模型的 VSR）的适用性未知。
2. 校准数据来自 REDS30，在跨域数据上的泛化性有待考察。
3. STCA 的秩分配阈值（25th/75th 百分位）是固定的经验值，更精细的搜索可能带来进一步提升。
4. 4-bit 在 MVSR4x 真实世界数据上的 SSIM（0.7367 vs. FP 0.7422）仍有小幅下降，极端精度需求场景可能需要 6-bit。
5. 未探索混合精度量化策略，对不同层使用不同比特宽度可能更优。

## 相关工作与启发

- **SVDQuant** (Li et al.)：通用 4-bit 量化方法，使用 16-bit 并行低秩分支。QuantVSR 的 STCA 是对其固定秩策略的针对性改进。
- **ViDiT-Q** (Zhao et al.)：面向图像/视频生成的量化，但未考虑 VSR 的时空特性。
- **PassionSR** (Zhu et al.)：面向单步扩散超分的量化，但仅限图像 SR。
- **EfficientDM** (He et al.)：低秩量化微调策略的先驱。
- 时空复杂度度量的思想可推广到其他需要帧间一致性的任务（如视频编辑、视频修复）的量化中。

## 评分

- 新颖性: ⭐⭐⭐⭐ — STCA 时空感知秩分配和 LBA 有创新，首次系统性解决扩散 VSR 量化
- 实验充分度: ⭐⭐⭐⭐⭐ — 合成+真实世界数据集，IQA+VQA+时序一致性，消融充分
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰,方法描述详尽
- 价值: ⭐⭐⭐⭐ — 84% 压缩比几乎无损，实用价值明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Gradient Knows Best: Mixed-Precision Quantization via Gradient-Guided Bit Allocation for Super-Resolution](../../CVPR2026/model_compression/gradient_knows_best_mixed-precision_quantization_via_gradient-guided_bit_allocat.md)
- [\[ACL 2025\] PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](../../ACL2025/model_compression/ptq161_low_bit_quantization.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](post_training_quantization_for_efficient_dataset_condensation.md)
- [\[AAAI 2026\] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)
- [\[ICML 2026\] TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization](../../ICML2026/model_compression/twla_achieving_ternary_weights_and_low-bit_activations_for_llms_via_post-trainin.md)

</div>

<!-- RELATED:END -->
