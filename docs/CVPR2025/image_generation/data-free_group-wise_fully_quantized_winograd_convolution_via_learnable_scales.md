---
title: >-
  [论文解读] Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales
description: >-
  [CVPR 2025][图像生成][Winograd卷积] 本文提出用组级量化对 Winograd 卷积全流水线进行 8-bit 量化，并通过无数据微调 Winograd 变换矩阵的缩放参数来解决输出变换中的大动态范围问题，在扩散模型上实现近无损图像生成质量和 31.3% 的卷积加速。 领域现状：大规模扩散模型推理成本极高…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "Winograd卷积"
  - "组量化"
  - "无数据微调"
  - "扩散模型加速"
  - "可学习缩放因子"
---

# Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales

**会议**: CVPR 2025  
**arXiv**: [2412.19867](https://arxiv.org/abs/2412.19867)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: Winograd卷积、组量化、无数据微调、扩散模型加速、可学习缩放因子

## 一句话总结
本文提出用组级量化对 Winograd 卷积全流水线进行 8-bit 量化，并通过无数据微调 Winograd 变换矩阵的缩放参数来解决输出变换中的大动态范围问题，在扩散模型上实现近无损图像生成质量和 31.3% 的卷积加速。

## 研究背景与动机

**领域现状**：大规模扩散模型推理成本极高，量化是有效加速手段。Winograd 快速卷积算法可进一步加速卷积层，但在量化域使用 Winograd 会显著增加数值误差。

**现有痛点**：直接对 Winograd 全流水线做量化导致严重质量下降，主要原因是 Winograd 域输出不同位置存在巨大动态范围差异（呈"十字"分布）。之前方法要么需要昂贵的 QAT 来学习变换矩阵，要么需要领域特定数据微调，对基础模型泛化能力有风险。

**核心矛盾**：组量化可以处理 Winograd 域输入变换和逐元素乘法，但对输出变换中的大范围差异无能为力；逐像素量化虽能解决但无法利用高效整数运算核。

**本文目标**：实现完全无数据依赖的 Winograd 全流水线量化，保持基础模型泛化能力。

**切入角度**：Winograd 变换矩阵可从 Vandermonde 矩阵加对角缩放矩阵 $S_A, S_B, S_G$ 导出，这些缩放因子直接控制变换矩阵各行范数，影响 Winograd 域输出的动态范围分布。

**核心 idea**：只微调 Winograd 变换矩阵的约 $n$ 个对角缩放参数，使用随机噪声代替真实数据，即可均衡 Winograd 域输出动态范围，实现全流水线量化。

## 方法详解

### 整体框架
整个 Winograd 流水线（输入变换 $B^TxB$、权重变换 $GwG^T$、逐元素乘法 $W \odot X$、输出变换 $A^TYA$）全部在 8-bit 整数下执行。通过学习缩放因子 $S_B, S_G$ 来均衡各阶段动态范围。

### 关键设计

1. **组级量化全流水线 Winograd**:

    - 功能：对 Winograd 卷积四个阶段全部应用量化
    - 核心思路：将张量划分为大小为 32/64/256 的组，每组独立量化。组大小限制为处理器向量宽度倍数以利于向量化。对输入变换和 Hadamard 乘积组量化已足够，核心挑战在输出变换。
    - 设计动机：组量化比 tensor-wise 或 channel-wise 更细粒度，无需复杂校准流程就能应对扩散模型中激活分布跨时间步剧烈变化的问题。

2. **无数据可学习 Winograd 缩放因子**:

    - 功能：通过微调少量参数解决输出变换的大动态范围问题
    - 核心思路：微调 $S_B$ 和 $S_G$ 各 $n$ 个对角元素，$S_A = (S_B S_G)^{-1}$ 自动确定。优化目标为最小化标准卷积与量化 Winograd 卷积输出差的 SQNR，输入使用随机高斯/均匀噪声。所有卷积层共享一组缩放因子。
    - 设计动机：仅约 $n$ 个参数，不动模型权重也不用真实数据，完全保证泛化能力。

3. **硬件感知优化内核**:

    - 功能：将理论加速转化为实际性能提升
    - 核心思路：开发高度优化的组级量化矩阵乘法内核，充分利用 CPU 向量指令，最大化 MAC 利用率。
    - 设计动机：组量化额外缩放操作可能抵消 Winograd 理论加速，需硬件级优化确保实际效果。

### 损失函数 / 训练策略
SQNR 损失。每次迭代随机选 K=2 个卷积层，用随机噪声输入，SGD 优化缩放因子。

## 实验关键数据

### 主实验
InstaFlow-0.9B 文本生成图像（MS-COCO 2017）：

| 配置 | FID↓ | CLIP↑ |
|------|------|-------|
| FP16 baseline | 23.00 | 30.19 |
| W8A8 组量化 | 23.04 | 30.16 |
| W8A8 Winograd F(6,3) 标准缩放 | 326.96 | 5.95 |
| W8A8 Winograd F(6,3) + 学习缩放 | 26.58 | 29.65 |

Stable Diffusion V1.5：

| 配置 | FID↓ | CLIP↑ |
|------|------|-------|
| FP16 baseline | 21.72 | 31.72 |
| W8A8 Winograd F(6,3) + 学习缩放 | 20.52 | 31.53 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅组量化（无 Winograd） | 近无损 | 组量化质量好 |
| 组量化 + 标准 Winograd | 完全崩溃 | 输出变换范围问题 |
| 组量化 + Winograd + 学习缩放 | 近无损 | 缩放因子有效 |

ResNet ImageNet Top-1：本文 vs BQW 在 ResNet-18 上 68.29% vs 66.67%（+1.62%），ResNet-34 上 71.67% vs 69.11%（+2.56%）。

### 关键发现
- 标准 Winograd 量化导致灾难性质量下降（FID 从 23 涨到 327），学习缩放因子后恢复到 26.58
- 随机噪声训练的缩放因子在不同模型和场景上均有效，验证了无数据方案的泛化能力
- CPU 卷积层加速 31.3%，端到端扩散模型加速 12.8%

## 亮点与洞察
- **无数据微调保证泛化**：只学 Winograd 变换的缩放参数而非模型权重，用随机噪声避免过拟合特定数据分布，设计哲学值得量化领域借鉴。
- **理论驱动的最小参数集**：从 Vandermonde 分解精准定位影响动态范围的参数，大幅缩小搜索空间。
- **首次在大规模扩散模型上展示量化 Winograd 可行性**。

## 局限与展望
- 仅在 CPU 上评估加速，GPU 适用性未讨论
- F(6,3) 恢复后仍有约 3.5 FID 差距
- 仅验证 8-bit，更低 bit + Winograd 组合未探索

## 相关工作与启发
- **vs BQW**: 需要完整训练流水线，本文完全无数据
- **vs PAW+FSQ**: 用训练数据微调可能过拟合，本文在 ResNet-34 上高出 2.2%

## 评分
- 新颖性: ⭐⭐⭐⭐ Vandermonde 分解定位问题+无数据方案，思路优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖扩散和分类两大场景
- 写作质量: ⭐⭐⭐⭐ 理论推导与实验结合紧密
- 价值: ⭐⭐⭐⭐ 为扩散模型部署加速提供新方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](../../ICML2026/image_generation/guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[CVPR 2025\] Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing](beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)
- [\[CVPR 2025\] MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)
- [\[CVPR 2025\] Multi-Group Proportional Representation for Text-to-Image Models](multi-group_proportional_representations_for_text-to-image_models.md)
- [\[CVPR 2025\] Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)](efficient_personalization_of_quantized_diffusion_model_without_backpropagation.md)

</div>

<!-- RELATED:END -->
