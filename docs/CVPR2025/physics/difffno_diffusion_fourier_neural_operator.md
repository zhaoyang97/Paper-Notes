---
title: >-
  [论文解读] DiffFNO: Diffusion Fourier Neural Operator
description: >-
  [CVPR 2025][物理/科学计算][超分辨率] 提出 DiffFNO，将加权傅里叶神经算子（WFNO）与扩散框架结合用于任意尺度超分辨率，通过模式再平衡（Mode Rebalancing）保留关键高频分量，门控融合机制融合频域和空间域特征，自适应步长 ODE 求解器加速推理，在多个基准上超越现有方法 2-4 dB PSNR。
tags:
  - "CVPR 2025"
  - "物理/科学计算"
  - "超分辨率"
  - "傅里叶神经算子"
  - "扩散模型"
  - "任意尺度"
  - "高频重建"
---

# DiffFNO: Diffusion Fourier Neural Operator

**会议**: CVPR 2025  
**arXiv**: [2411.09911](https://arxiv.org/abs/2411.09911)  
**代码**: 无  
**领域**: 图像修复  
**关键词**: 超分辨率, 傅里叶神经算子, 扩散模型, 任意尺度, 高频重建

## 一句话总结
提出 DiffFNO，将加权傅里叶神经算子（WFNO）与扩散框架结合用于任意尺度超分辨率，通过模式再平衡（Mode Rebalancing）保留关键高频分量，门控融合机制融合频域和空间域特征，自适应步长 ODE 求解器加速推理，在多个基准上超越现有方法 2-4 dB PSNR。

## 研究背景与动机

**领域现状**：图像超分辨率（SR）从 SRCNN 发展到基于注意力机制的深度架构，再到扩散模型用于高保真度生成。任意尺度 SR（可在训练未见过的放大倍率下工作）成为新趋势，神经算子（如 SRNO、HiNOTE）利用函数空间映射实现分辨率无关的上采样。

**现有痛点**：标准 FNO 通过模式截断提高效率，但丢失了 SR 关键的高频分量。MLP 存在频谱偏置（偏好低频）。扩散模型能生成高保真细节但推理速度慢。SRNO 等算子学习方法源于物理模拟，直接应用于真实图像时高频恢复不足。

**核心矛盾**：FNO 的高效全局建模与 SR 对高频细节的依赖之间存在矛盾——模式截断提升了计算效率但牺牲了高频信息；扩散模型能恢复高频但计算代价高昂。

**本文目标**：设计一种既能保留高频信息、又能高效推理的任意尺度 SR 方法。

**切入角度**：不截断傅里叶模式，而是通过可学习权重函数对所有模式进行再平衡，增强高频分量。同时用扩散过程做迭代精炼，并用自适应 ODE 求解器加速。

**核心 idea**：WFNO 保留所有傅里叶模式并用频率相关的可学习权重 $\mathbf{w}(\boldsymbol{\xi}) = 1 + \gamma \cdot \|\boldsymbol{\xi}\|^\alpha$ 增强高频，与注意力算子融合后在扩散框架中迭代精炼。

## 方法详解

### 整体框架
输入 LR 图像通过 CNN 编码器（EDSR-baseline/RDN）提取特征。WFNO 在频域捕获全局依赖并增强高频，AttnNO 在空域用 Galerkin 注意力捕获局部细节。门控融合机制（GFM）自适应结合两路特征，投影到 RGB 空间后通过自适应步长（ATS）ODE 求解器高效完成反向扩散，输出 HR 图像。

### 关键设计

1. **加权傅里叶神经算子（WFNO）及模式再平衡**:

    - 功能：在保留所有傅里叶模式的基础上，自适应增强高频分量
    - 核心思路：在标准 FNO 的频域卷积中引入可学习权重函数 $\mathbf{w}_l(\boldsymbol{\xi}) = 1 + \gamma_l \cdot \|\boldsymbol{\xi}\|^\alpha$（$\alpha = 0.7$），使得更新后的积分算子为 $\mathcal{K}_l \mathbf{v}_l = \mathcal{F}^{-1}(\mathbf{w}_l \cdot \mathbf{P}_l \cdot \mathcal{F}[\mathbf{v}_l])$。当 $\alpha > 0$ 时高频模式获得更高权重，$\gamma_l$ 每层独立学习控制增强强度
    - 设计动机：标准 FNO 截断高频模式以降低计算量，但 SR 任务恰恰需要高频。模式再平衡保留所有模式并让网络学习高频增强策略，比手动截断更灵活

2. **门控融合机制（GFM）**:

    - 功能：自适应融合 WFNO 的频域全局特征和 AttnNO 的空域局部特征
    - 核心思路：将 WFNO 和 AttnNO 的特征图沿通道拼接，通过 $1 \times 1$ 卷积 + sigmoid 生成空间门控图 $\mathbf{G} \in \mathbb{R}^{B \times H \times W \times 1}$，融合结果为 $\mathbf{v}_{fused} = \mathbf{G} \odot \mathbf{v}_{WFNO} + (1 - \mathbf{G}) \odot \mathbf{v}_{AttnNO}$。AttnNO 与 WFNO 共享编码器并行运行，使用 Galerkin 注意力机制
    - 设计动机：WFNO 擅长全局结构和长程依赖，AttnNO 擅长局部纹理和细粒度细节。门控融合在每个空间位置动态平衡两者贡献

3. **自适应步长（ATS）ODE 求解器**:

    - 功能：加速扩散模型的反向采样过程
    - 核心思路：将随机反向扩散重新表述为确定性 ODE，使用自适应步长的数值求解器来求解。根据图像区域的复杂度动态调整积分步长——简单区域用大步长、复杂区域用小步长，在保持质量的同时减少计算开销
    - 设计动机：固定步长的 ODE 求解对所有区域使用相同步数，浪费计算资源。自适应步长在不牺牲质量的前提下显著减少推理时间

### 损失函数 / 训练策略
使用分数匹配损失 $\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0}[\|s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0)\|_2^2]$。前向扩散使用修改的方差保持 SDE，漂移项 $-\frac{1}{2}\beta(t)(\mathbf{x} - \mathbf{Dx})$ 建模下采样导致的高频损失，噪声调度为线性增长（$\beta_{min}=0.1, \beta_{max}=20$）。

## 实验关键数据

### 主实验

| 方法 | 4x SR PSNR | 推理时间 | 任意尺度 |
|------|-----------|---------|---------|
| SRNO | 基线 | 较慢 | 支持 |
| HiNOTE | 中等 | 较慢 | 支持 |
| DiffFNO | **+2-4 dB** | **更快** | 支持 |
| EDSR | 较低 | 快 | 固定尺度 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| FNO (标准截断) | 较低 PSNR | 高频丢失 |
| WFNO (模式再平衡) | 显著提升 | 高频保留是核心 |
| 无 AttnNO | PSNR 下降 | 局部细节损失 |
| 朴素拼接替代 GFM | PSNR 下降 | 自适应融合优于简单拼接 |
| 固定步长 ODE | 推理更慢 | ATS 在等质量下更高效 |

### 关键发现
- 模式再平衡是性能提升的核心——简单的频率相关权重函数就能大幅改善高频重建
- GFM 的空间自适应门控比朴素拼接更有效，因为不同空间位置对全局/局部特征的需求不同
- DiffFNO 不仅在训练分布内的放大倍率上表现好，在训练未见的放大倍率上也保持鲁棒
- ATS ODE 求解器在保持输出质量的同时显著降低了推理时间

## 亮点与洞察
- **频率再平衡的简洁形式**：$1 + \gamma \cdot \|\xi\|^\alpha$ 作为权重函数极其简单，仅多一个标量参数，却能有效解决 FNO 的高频丢失问题。这种"最小干预、最大收益"的设计值得学习
- **扩散框架与神经算子的协同**：WFNO 提供分辨率无关的上采样，扩散过程提供迭代精炼——两者天然互补。前者解决"结构正确"，后者解决"细节逼真"
- **共享编码器的双路设计**：WFNO 和 AttnNO 共享编码器但在不同域（频域/空域）并行处理，高效利用计算资源

## 局限与展望
- 扩散过程即使有 ATS 加速，推理时间仍高于单次前向的方法（EDSR 等）
- 模式再平衡的超参数 $\alpha$ 对不同数据集/任务可能需要调整
- 实验主要在标准 SR 基准上验证，未探索真实世界退化（模糊+噪声+压缩）的场景
- 与最新的基于 Transformer 的 SR 方法（如 SwinIR、HAT）的对比不够完整

## 相关工作与启发
- **vs SRNO**: SRNO 使用标准 FNO + Galerkin 注意力但没有频率再平衡和扩散精炼。DiffFNO 在此基础上增加了 WFNO 和扩散过程，PSNR 提升 2-4 dB
- **vs HiNOTE**: HiNOTE 有自己的编码器但高频重建仍不足。DiffFNO 用模式再平衡直接从频域增强高频
- **vs SRDiff/SR3**: 纯扩散 SR 方法推理极慢且可能不支持任意尺度。DiffFNO 结合了算子学习的分辨率无关性和扩散的高保真度

## 评分
- 新颖性: ⭐⭐⭐⭐ WFNO 的模式再平衡和与扩散框架的结合是新颖的
- 实验充分度: ⭐⭐⭐⭐ 多尺度评估、消融充分、与多种 baseline 对比
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，架构图直观，方法描述详细
- 价值: ⭐⭐⭐⭐ 首次在扩散 SR 中引入傅里叶模式再平衡，为任意尺度 SR 设立新标准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Spatial-Spectral Residuals Informed Diffusion Neural Operator for Pan-sharpening](../../CVPR2026/physics/spatial-spectral_residuals_informed_diffusion_neural_operator_for_pan-sharpening.md)
- [\[ICML 2026\] Generative Neural Operators Through Diffusion Last Layer](../../ICML2026/physics/generative_neural_operators_through_diffusion_last_layer.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/physics/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[NeurIPS 2025\] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](../../NeurIPS2025/physics/adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [\[ICCV 2025\] JPEG Processing Neural Operator for Backward-Compatible Coding](../../ICCV2025/physics/jpeg_processing_neural_operator_for_backward-compatible_coding.md)

</div>

<!-- RELATED:END -->
