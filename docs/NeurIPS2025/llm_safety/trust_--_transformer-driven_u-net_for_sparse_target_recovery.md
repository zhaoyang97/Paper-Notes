---
title: >-
  [论文解读] TRUST -- Transformer-Driven U-Net for Sparse Target Recovery
description: >-
  [NeurIPS 2025][LLM安全][稀疏恢复] 提出 TRUST 架构，将 Transformer 的注意力机制与 U-Net 解码器结合，在感知矩阵未知的条件下同时学习感知算子和重建稀疏信号，在 SSIM 和 PSNR 上显著超越传统方法。 在逆问题 $\mathbf{y} = \mathbf{A}\mathbf{…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "稀疏恢复"
  - "Transformer"
  - "U-Net"
  - "逆问题"
  - "感知矩阵学习"
---

# TRUST -- Transformer-Driven U-Net for Sparse Target Recovery

**会议**: NeurIPS 2025  
**arXiv**: [2506.01112](https://arxiv.org/abs/2506.01112)  
**代码**: 无  
**领域**: 信号处理 / 图像重建  
**关键词**: 稀疏恢复, Transformer, U-Net, 逆问题, 感知矩阵学习

## 一句话总结

提出 TRUST 架构，将 Transformer 的注意力机制与 U-Net 解码器结合，在感知矩阵未知的条件下同时学习感知算子和重建稀疏信号，在 SSIM 和 PSNR 上显著超越传统方法。

## 研究背景与动机

在逆问题 $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{w}$ 中，稀疏恢复通过利用信号的稀疏性来求解欠定系统。然而，现有方法面临以下挑战：

**感知矩阵未知**: 多数传统方法假设 $\mathbf{A}$ 已知，但实际应用中往往未知

**训练数据有限**: 只有少量观测-目标对 $\{(\mathbf{x}, \mathbf{y})\}$ 可用

**幻觉伪影**: 深度学习方法容易产生与真实信号不一致的幻觉

**局部-全局特征**: 纯 U-Net 缺乏全局感受野，纯 Transformer 缺乏多尺度局部特征

## 方法详解

### 整体框架

TRUST 采用混合编码器-解码器架构：
- **编码器**: Transformer 分支，捕获长程依赖并估计稀疏支撑集
- **解码器**: U-Net 风格解码器，通过多尺度特征融合精细化重建
- **跳跃连接**: Transformer 各层与解码器之间的 skip connections

### 关键设计

1. **Transformer 编码分支**:

    - 多头自注意力层捕获输入观测的全局依赖
    - 逐层提取不同抽象层次的特征
    - 估计信号的稀疏支撑（哪些位置非零）

2. **U-Net 解码路径**:

    - 多尺度反卷积逐步恢复空间分辨率
    - 利用 Transformer 各层特征作为引导信息
    - 精细化局部细节恢复

3. **跳跃连接设计**:

    - 不同于传统 U-Net 的对称跳跃连接
    - Transformer 层级 → 解码器层级的非对称连接
    - 让解码器访问不同抽象层次的图像特征

### 损失函数 / 训练策略

$$\mathcal{L} = \|\hat{\mathbf{x}} - \mathbf{x}\|_2^2 + \lambda_1 \|\hat{\mathbf{x}}\|_1 + \lambda_2 \mathcal{L}_{\text{SSIM}}$$

三项损失分别对应：重建误差、稀疏正则化、结构相似性约束。

## 实验关键数据

### 主实验（稀疏信号恢复）

| 方法 | PSNR (dB) ↑ | SSIM ↑ | 重建时间 (ms) ↓ | 幻觉伪影率 (%) ↓ |
|------|------------|--------|---------------|----------------|
| ISTA | 24.3 | 0.712 | 850 | 12.5 |
| LISTA | 27.8 | 0.781 | 120 | 8.3 |
| U-Net | 30.2 | 0.845 | 45 | 15.7 |
| SwinIR | 31.5 | 0.868 | 78 | 9.2 |
| Transformer-only | 31.8 | 0.872 | 92 | 7.8 |
| **TRUST** | **34.6** | **0.921** | **52** | **3.1** |

### 不同压缩率实验

| 压缩率 (M/N) | U-Net PSNR | Transformer PSNR | TRUST PSNR | TRUST SSIM |
|-------------|-----------|-----------------|-----------|-----------|
| 0.1 | 22.1 | 23.5 | **26.8** | 0.785 |
| 0.2 | 26.3 | 27.1 | **30.2** | 0.856 |
| 0.3 | 29.5 | 30.2 | **33.1** | 0.905 |
| 0.5 | 32.8 | 33.5 | **36.2** | 0.945 |

### 消融实验

| 设置 | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| 完整 TRUST | 34.6 | 0.921 |
| 去掉跳跃连接 | 31.8 | 0.875 |
| 去掉 Transformer 编码器 | 30.2 | 0.845 |
| 去掉 U-Net 解码器 | 31.5 | 0.868 |
| 去掉稀疏正则化 | 33.2 | 0.902 |

### 关键发现

1. TRUST 在低压缩率（M/N=0.1）下优势最为明显，说明全局上下文对严重欠采样很重要
2. 跳跃连接贡献约 2.8 dB PSNR 提升，是最关键的组件
3. 幻觉伪影率从 U-Net 的 15.7% 降至 3.1%，证实了混合架构的鲁棒性
4. 推理速度接近纯 U-Net，远快于传统迭代方法

## 亮点与洞察

- **混合架构设计**: Transformer 负责全局推理，U-Net 负责多尺度重建，分工明确
- **幻觉抑制**: 通过稀疏支撑估计指导重建，有效抑制深度学习常见的幻觉问题
- **盲稀疏恢复**: 不需要知道感知矩阵 A，端到端学习，实用性强

## 局限与展望

1. 目前仅在 2D 信号上验证，3D 体积数据的扩展需要研究
2. Transformer 编码器的计算开销在高分辨率信号上仍然较大
3. 稀疏度的超参设置需要领域知识
4. 未与最新的 diffusion-based 重建方法进行对比

## 相关工作与启发

- **LISTA (Gregor & LeCun, 2010)**: 将 ISTA 展开为可学习网络
- **SwinIR**: 基于 Swin Transformer 的图像恢复
- **TransUNet**: 医学图像分割中的 Transformer-UNet 混合架构，本文借鉴了类似思路

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 3 |
| 理论深度 | 3 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总体推荐 | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[ICML 2025\] Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model](../../ICML2025/llm_safety/sorbet_a_neuromorphic_hardware-compatible_transformer-based_spiking_language_mod.md)
- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[ACL 2025\] Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations](../../ACL2025/llm_safety/faithful_and_robust_llm-driven_theorem_proving_for_nli_explanations.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)

</div>

<!-- RELATED:END -->
