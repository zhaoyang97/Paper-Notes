---
title: >-
  [论文解读] Towards Lossless Implicit Neural Representation via Bit Plane Decomposition
description: >-
  [CVPR 2025][音频/语音][隐式神经表示] 发现隐式神经表示（INR）的模型容量上界随比特精度指数增长（$\mathcal{P}(f_\theta) \propto 2^n$），提出比特平面分解——将 n-bit 信号分解为 n 个独立的 1-bit 平面分别训练 INR，首次实现 16-bit 图像的无损（BER=0）隐式神经表示。
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "隐式神经表示"
  - "无损表示"
  - "比特平面分解"
  - "比特偏置"
  - "高精度信号"
---

# Towards Lossless Implicit Neural Representation via Bit Plane Decomposition

**会议**: CVPR 2025  
**arXiv**: [2502.21001](https://arxiv.org/abs/2502.21001)  
**代码**: [https://github.com/WooKyoungHan/LosslessINR](https://github.com/WooKyoungHan/LosslessINR)  
**领域**: 音频语音 / 神经表示  
**关键词**: 隐式神经表示, 无损表示, 比特平面分解, 比特偏置, 高精度信号

## 一句话总结

发现隐式神经表示（INR）的模型容量上界随比特精度指数增长（$\mathcal{P}(f_\theta) \propto 2^n$），提出比特平面分解——将 n-bit 信号分解为 n 个独立的 1-bit 平面分别训练 INR，首次实现 16-bit 图像的无损（BER=0）隐式神经表示。

## 研究背景与动机

### 领域现状

**领域现状**：INR（如 SIREN、FINER）用神经网络将坐标映射到信号值（图像像素/音频振幅），实现连续的信号表示。但现有方法在 8-bit 精度上就已经有明显的位错误率（BER），16-bit 高精度信号更是无法无损表示。

**现有痛点**：INR 存在"比特偏置"现象——高位比特（MSB）学习快而准确，低位比特（LSB）学习慢且不精确。这类似于频谱偏置（低频先学、高频难学），但发生在比特维度。结果是模型的 BER 随精度指数增加。

**核心矛盾**：一个网络同时建模 16 个比特平面的信息——高位的 1 等于低位的 $2^{15}$，信息尺度差异巨大，网络容量被高位"霸占"。

**切入角度**：将 n-bit 信号分解为 n 个独立的 1-bit 平面，每个平面训练一个独立的 INR。1-bit → BER 的理论上界只有 2^1 而非 2^n。

**核心 idea**：信号→比特平面分解→每平面独立INR = 突破精度瓶颈实现无损表示。

### 解决思路

**本文目标**：### 关键设计

1. **比特平面分解**:

    - 功能：将高精度信号分解为多个低精度子信号
    - 核心思路：n-bit 整数值可以分解为 n 个二进制平面 $\{b_0, b_1, ..., b_{n-1}\}$，每个平面是 0/1 值的空间函数。


## 方法详解

### 关键设计

1. **比特平面分解**:

    - 功能：将高精度信号分解为多个低精度子信号
    - 核心思路：n-bit 整数值可以分解为 n 个二进制平面 $\{b_0, b_1, ..., b_{n-1}\}$，每个平面是 0/1 值的空间函数。为每个平面训练独立的小型 INR
    - 设计动机：Theorem 1 证明模型容量上界 $\mathcal{P} \propto 2^n$，分解后每个子问题复杂度仅为 $2^1$

2. **比特偏置现象的发现**:

    - 功能：揭示 INR 学习中被忽视的偏置来源
    - 核心思路：类比频谱偏置（低频先学）——MSB 先学、LSB 后学。这是因为 MSB 的误差在损失函数中贡献 $2^{2(n-1)}$ 倍于 LSB 的误差，梯度自然偏向 MSB
    - 设计动机：理解了偏置原因才能设计正确的解法——分解而非"加大模型"

### 损失函数 / 训练策略

MSE 损失。每个比特平面用目标错误界 $\epsilon(n) = \frac{1}{2(2^n-1)}$ 作为精度参考。可用 SIREN/FINER/Gaussian 等任意激活函数。

## 实验关键数据

| 方法 | 精度 | PSNR | BER |
|------|------|------|-----|
| SIREN (原始) | 16-bit | 47.04% | 0.128 |
| FINER (原始) | 16-bit | 40.64% | 0.164 |
| **Ours (分解)** | 16-bit | **∞** | **0.000** |

### 消融实验

- MSB 先收敛、LSB 后收敛的比特偏置在所有激活函数（ReLU/SIREN/FINER）上一致存在
- 分解后每个比特平面快速收敛（~5000 次迭代），总计算量可控
- 应用：无损压缩、比特深度扩展、三元量化 INR

### 关键发现
- **比特偏置是 INR 精度瓶颈的根因**——不是模型容量不够，而是学习偏向高位比特
- **分解后理论和实践都能达到无损**——BER=0 是首次实现
- **16 个小网络比一个大网络更有效**——因为每个子问题复杂度指数级降低

## 亮点与洞察
- **比特偏置的发现**——与频谱偏置并列的新型偏置现象，对整个 INR 领域有启示
- **极简方案解决根本问题**——不改网络架构/激活函数/训练策略，只改数据表示方式
- **无损 INR 的里程碑**——此前无人实现 16-bit 无损

## 局限与展望
- 16 个独立网络的存储和推理开销
- 方法特定于结构化比特分解，其他分解（如频率分解）未探索
- 主要验证图像/音频，3D 场景未涉及

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 比特偏置的发现和分解解法极具洞察力
- 实验充分度: ⭐⭐⭐⭐ 多激活函数/多精度/多应用
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导优雅
- 价值: ⭐⭐⭐⭐ 为 INR 的精度问题提供了根本解法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[ACL 2026\] Privacy-preserving Prosody Representation Learning](../../ACL2026/audio_speech/privacy-preserving_prosody_representation_learning.md)
- [\[ICML 2026\] Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition](../../ICML2026/audio_speech/towards_understanding_modality_interaction_in_multimodal_language_models_via_par.md)
- [\[NeurIPS 2025\] Slimmable NAM: Neural Amp Models with Adjustable Runtime Computational Cost](../../NeurIPS2025/audio_speech/slimmable_nam_neural_amp_models_with_adjustable_runtime_computational_cost.md)
- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](../../ICLR2026/audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)

</div>

<!-- RELATED:END -->
