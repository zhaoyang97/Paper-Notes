---
title: >-
  [论文解读] FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction
description: >-
  [AAAI 2026][图像生成][Fractal Generative Model] 本文首次将 Hausdorff 维数（HD）引入分形生成模型（FGM），提出可学习的 HD 估计模块、单调动量驱动调度策略（MMDS）和 HD 引导的拒绝采样，在 ImageNet 上实现 39% 的生成多样性提升（Recall），同时保持图像质量。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "Fractal Generative Model"
  - "Hausdorff Dimension"
  - "生成多样性"
  - "动量调度策略"
  - "拒绝采样"
---

# FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction

**会议**: AAAI 2026  
**arXiv**: [2511.08945](https://arxiv.org/abs/2511.08945)  
**代码**: 无  
**领域**: Image Generation  
**关键词**: Fractal Generative Model, Hausdorff Dimension, 生成多样性, 动量调度策略, 拒绝采样

## 一句话总结
本文首次将 Hausdorff 维数（HD）引入分形生成模型（FGM），提出可学习的 HD 估计模块、单调动量驱动调度策略（MMDS）和 HD 引导的拒绝采样，在 ImageNet 上实现 39% 的生成多样性提升（Recall），同时保持图像质量。

## 研究背景与动机
生成模型（GAN、VAE、扩散模型等）虽然能生成高保真图像，但在保持图像质量和多样性之间的平衡仍是根本挑战。分形生成模型（FGM）利用递归自相似性，通过在多个尺度上重复应用紧凑的生成模块来高效生成高质量图像。

**现有痛点**：FGM 的递归自相似结构在保证全局一致性的同时，也导致输出图像出现重复模式，多样性不足。这种固有的自相似性限制了 FGM 捕获复杂数据分布的能力。

**核心矛盾**：FGM 的核心优势（递归自相似）恰恰是其多样性不足的根源；需要在不破坏结构一致性的前提下增强多样性。

**切入角度**：从分形几何中借鉴 Hausdorff 维数（HD）作为结构复杂度的几何指标。HD 量化不同尺度上空间细节的变化，更高的 HD 值通常反映更大的结构丰富性。

**核心 idea**：用 HD 作为训练信号和采样准则来引导 FGM 生成结构更复杂多样的输出。

## 方法详解

### 整体框架
FGM-HD 框架包含三个关键创新：(1) 可学习的 HD 估计网络，从图像嵌入直接预测 HD；(2) 训练阶段使用 MMDS 策略动态调整 HD 损失权重；(3) 推理阶段使用 HD 引导的拒绝采样保留高 HD 输出。

### 关键设计

1. **可学习 HD 估计模块**:

    - 功能：直接从图像嵌入预测 Hausdorff 维数，替代传统的 box counting 方法
    - 核心思路：基于 ResNet152 骨干，替换最后两层为多尺度卷积模块（3×3, 5×5, 7×7 并行卷积），捕获不同尺度的空间信息，最后通过回归层输出 HD 值
    - 设计动机：传统 box counting 方法计算成本高（4.70s/图）且对噪声敏感，本方法仅需 0.32s/图且误差仅 0.005（vs box counting 的 0.002），精度-效率平衡理想

2. **单调动量驱动调度策略 (MMDS)**:

    - 功能：动态调整 HD 损失在混合损失中的权重 λ(t)
    - 核心思路：训练早期图像质量差、HD 估计不可靠，λ 应接近零；随训练进展逐渐增加。通过类似 SGD 动量的累积方案实现：m ← μ·m + (1-μ)·γ·ΔL, λ ← λ + m，其中 ΔL = max(0, L_prev - L_val)
    - 设计动机：直接在混合损失中加固定 HD 权重会导致 (1) 图像质量退化 (2) 多样性改善有限。MMDS 确保模型先关注质量再逐步引入多样性
    - 总损失：L_total = L_gen + λ(t) · L_HD，其中 L_HD = |HD_gen - HD_target|

3. **HD 引导的拒绝采样**:

    - 功能：推理时过滤掉低 HD（结构简单）的生成图像
    - 核心思路：生成一批候选图像后，估计每张的 HD，仅保留 HD 超过阈值 τ 的输出，低 HD 的从头重新生成
    - 设计动机：利用 FGM 的递归结构自然支持并行生成多个候选，后处理不修改模型架构也不增加训练成本

### 损失函数 / 训练策略
- HD 损失：L_HD = |HD_gen - HD_target|，其中 HD_target 是训练集中类别特定的 HD 中位数
- MMDS 参数：μ=0.9, γ=1.0 提供最佳平衡
- 训练在 1000 epoch 后停止：进一步训练不显著改善质量或多样性
- HD 采样阈值：1.55-1.60 提供最佳质量-多样性平衡

## 实验关键数据

### 主实验（ImageNet 256×256，像素级生成）

| 模型 | 类型 | FID↓ | IS↑ | Recall↑ |
|------|------|------|-----|---------|
| StyleGAN-XL | GAN | 2.30 | 265.1 | 0.53 |
| DiffiT | Diffusion | 1.73 | 276.5 | 0.62 |
| RCG | MAGE | 2.15 | 253.4 | 0.53 |
| FGM (baseline) | Fractal | 6.15 | 348.9 | 0.46 |
| **FGM-HD (Ours)** | **Fractal** | **6.21** | **367.1** | **0.64** |

### 消融实验

| 配置 | FID↓ | IS↑ | Recall↑ | LPIPS↑ |
|------|------|-----|---------|--------|
| FGM (baseline) | 6.15 | 348.9 | 0.46 | 0.64 |
| + Fixed HD Loss only | 6.22 | 333.7 | 0.47 | 0.65 |
| + MMDS only | 6.04 | 361.7 | 0.51 | 0.69 |
| + HD Sampling only | 6.78 | 357.9 | 0.58 | 0.73 |
| **+ MMDS & Sampling** | **6.21** | **367.4** | **0.64** | **0.76** |

### HD 估计方法对比

| 方法 | 类型 | 误差↓ | 时间(s)↓ |
|------|------|-------|---------|
| Box Counting | 非学习 | 0.002 | 4.70 |
| Power Spectrum | 非学习 | 0.079 | 3.41 |
| ResNet152 | 学习 | 0.012 | 0.40 |
| **Ours** | **学习** | **0.005** | **0.32** |

### 关键发现
- **固定 HD 损失效果差**：直接加 HD 损失（Recall 仅从 0.46→0.47），同时 IS 从 348.9 降至 333.7，FID 也升高
- **MMDS 和 Sampling 互补**：MMDS 提供训练阶段的渐进优化，Sampling 提供推理阶段的后处理过滤，二者组合效果最佳
- **多尺度卷积的核选择**：3×3+5×5+7×7 组合最优（Loss 0.005），单核效果明显差于多核组合
- **ResNet152 替换层选择**：替换 Stage 4-5 最优（误差 0.005），替换更多层虽更快但精度显著下降
- **MMDS 的 λ 在 epoch 300 开始增大，epoch 800 后趋于平稳**，对应图像质量从混乱到稳定的过程

## 亮点与洞察
- **理论贡献**：首次将分形几何中的 Hausdorff 维数引入生成模型多样性增强，提供了数学上有根据的多样性度量
- **MMDS 的通用性**：动量驱动的损失权重调度策略可推广到任何使用混合损失函数的模型
- **Recall 提升 39%**：在保持图像质量的同时大幅提升生成分布的覆盖率
- **推理阶段无额外训练成本**：拒绝采样仅增加推理时间，不改变模型

## 局限与展望
- FGM 基线的 FID（6.15）仍远高于扩散模型（1.73），整体生成质量差距明显
- HD 估计基于 box counting 标注的数据训练，可能存在标注精度瓶颈
- 仅在 256×256 ImageNet 上验证，未测试更高分辨率或条件生成
- 拒绝采样增加推理时间，高阈值可能导致大量重新生成
- 未探索 HD 与其他多样性度量（如 FID 的模式覆盖指标）的关系
- 未在条件生成和多模态生成模型上验证

## 相关工作与启发
- **Hausdorff GAN**：Li et al. 提出将 HD 用于对齐真假数据的内在维数，本文则专注于 FGM 的多样性增强
- **FGM 的独特价值**：递归自相似结构使 FGM 特别适合需要结构一致性和视觉丰富性的任务
- **多样性与质量的权衡**：本文的方法论（渐进引入多样性信号）可启发其他面临类似权衡的生成任务
- **MMDS 的启发**：对于任何需要平衡多个损失项的训练系统，动量驱动的调度策略提供了比固定指数/线性调度更稳健的方案

## 评分
- 新颖性: ⭐⭐⭐⭐（HD+FGM 组合是全新的，MMDS 策略有通用价值）
- 实验充分度: ⭐⭐⭐⭐（消融充分，HD 估计和调度策略都有详细分析）
- 写作质量: ⭐⭐⭐⭐（结构清晰，方法动机阐述充分）
- 价值: ⭐⭐⭐（FGM 本身的基线性能限制了方法的实际影响力）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Expand and Prune: Maximizing Trajectory Diversity for Effective GRPO in Generative Models](../../CVPR2026/image_generation/expand_and_prune_maximizing_trajectory_diversity_for_effective_grpo_in_generativ.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[NeurIPS 2025\] Boosting Generative Image Modeling via Joint Image-Feature Synthesis](../../NeurIPS2025/image_generation/boosting_generative_image_modeling_via_joint_imagefeature_sy.md)
- [\[CVPR 2025\] Interpretable Generative Models through Post-hoc Concept Bottlenecks](../../CVPR2025/image_generation/interpretable_generative_models_through_post-hoc_concept_bottlenecks.md)

</div>

<!-- RELATED:END -->
