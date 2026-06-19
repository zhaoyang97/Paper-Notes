---
title: >-
  [论文解读] Scaling Backwards: Minimal Synthetic Pre-training?
description: >-
  [ECCV 2024][预训练][合成预训练] 提出 1p-frac——仅用单个分形图像的微小扰动即可实现与 ImageNet-1k 级别可比的预训练效果，挑战了"预训练需要大规模数据集"的常规认知，揭示预训练本质可能更接近权重初始化而非视觉概念学习。 领域现状： 预训练是当前视觉系统的基础技术。主流方法使用大规模真实图像数…
tags:
  - "ECCV 2024"
  - "预训练"
  - "合成预训练"
  - "分形图像"
  - "最小数据集"
  - "视觉表征学习"
  - "ViT"
---

# Scaling Backwards: Minimal Synthetic Pre-training?

**会议**: ECCV 2024  
**arXiv**: [2408.00677](https://arxiv.org/abs/2408.00677)  
**代码**: [GitHub](https://github.com/SUPER-TADORY/1p-frac)  
**领域**: LLM预训练  
**关键词**: 合成预训练, 分形图像, 最小数据集, 视觉表征学习, ViT

## 一句话总结

提出 1p-frac——仅用单个分形图像的微小扰动即可实现与 ImageNet-1k 级别可比的预训练效果，挑战了"预训练需要大规模数据集"的常规认知，揭示预训练本质可能更接近权重初始化而非视觉概念学习。

## 研究背景与动机

**领域现状**: 预训练是当前视觉系统的基础技术。主流方法使用大规模真实图像数据集（ImageNet-1k 的 128 万图，ImageNet-21k 的 1400 万图）进行监督或自监督预训练。合成预训练方向如 FractalDB（100 万分形图像）和 OFDB（1000 张分形图像）已证明无需真实图像也能获得有效表征。

**现有痛点**: 基金模型的规模不断膨胀（从百万到数十亿图像），但预训练的本质仍不清楚——是在发现通用视觉概念，还是仅仅提供了更好的权重初始化？此外，大规模真实数据集存在隐私、版权和公平性问题。

**核心矛盾**: OFDB 已将分形图像减少到 1000 张，但进一步减少类别数会导致性能下降。问题在于"最小的有效预训练数据集到底能有多小？"

**本文目标**: 寻找最小的纯合成预训练数据集，并探究预训练成功的最低要求。

**切入角度**: 不是增加更多图像，而是"向后扩展"（scaling backwards）——用单个分形图像的细微参数扰动构建"类别"，训练模型区分这些人眼不可分的扰动。

**核心 idea**: 预训练的关键不在于数据量的大小，而在于数据生成过程中的结构化多样性——递归自相似结构（分形）配合微小的仿射变换扰动，即可提供足够的预训练信号。

## 方法详解

### 整体框架

1p-frac 由三个核心组件构成：
- 一个分形图像（由迭代函数系统 IFS 定义）
- 局部积分经验分布（LIEP distribution）用于生成扰动图像
- 局部扰动交叉熵损失（LPCE loss）用于预训练

预训练流程：从单个 IFS $\Omega$ 出发，对仿射变换参数施加微小扰动 $\epsilon$，生成 $L$ 个扰动图像作为不同"类别"，训练 ViT 分类这些扰动。

### 关键设计

1. **局部积分经验分布（LIEP Distribution）**: 核心数学工具。

    - 对于单个分形图像 $I$，经验分布退化为 $p_{\text{data}}(x,y) = \delta(x-I)\delta(y)$，直接用交叉熵训练会得到平凡解。
    - LIEP 分布引入扰动参数 $\Delta$，在 $\boldsymbol{\epsilon} \in \mathcal{R}_\Delta = [-\Delta/2, \Delta/2]^{6j}$ 范围内积分：
    $p_\Delta(x,y) = \frac{1}{|\mathcal{R}_\Delta|}\int_{\mathcal{R}_\Delta}\delta(x - I_{\boldsymbol{\epsilon}})\delta(y - \boldsymbol{\epsilon})d\boldsymbol{\epsilon}$
    - 当 $\Delta \to 0$，LIEP 分布收敛到原单图像经验分布
    - **设计动机**: 提供一个连续可控的方式来收窄或扩大数据分布的支撑集，从而精确研究预训练所需的最小分布范围。

2. **局部扰动交叉熵损失（LPCE Loss）**: 预训练目标函数。

    - $\mathcal{L}_\Delta = -\mathbb{E}_{x,y \sim p_\Delta}[\log p_\theta(y|x)]$
    - 实际中通过数值积分近似：均匀采样 $L=1000$ 个扰动点
    - 扰动应用于 IFS 的仿射变换参数：$w_j(\boldsymbol{v}; \boldsymbol{\epsilon}_j) = \left(\begin{bmatrix}a_j & b_j & e_j \\ c_j & d_j & f_j\end{bmatrix} + \boldsymbol{\epsilon}_j\right)\begin{bmatrix}\boldsymbol{v} \\ 1\end{bmatrix}$
    - **设计动机**: 使模型学习区分人眼不可分的微小形状差异，迫使网络关注结构性pattern而非表面特征。

3. **σ-factor 控制分形复杂度**: 使用 Anderson 的 σ-factor 评估 IFS 的复杂度。

    - σ 越小，分形越复杂（类似自然物体的递归结构）
    - σ 过大（如 6.0），分形退化为类似高斯噪声，但仍有正面预训练效果
    - 最优 σ = 3.5
    - **设计动机**: 探索什么样的图像结构对预训练最关键——结论是递归自相似结构比单纯复杂度更重要。

### 损失函数 / 训练策略

- 预训练使用 LPCE loss，超参遵循 DeiT 标准设定
- 数据增强沿用 DeiT（RandomCrop, RandAug, Mixup, CutMix 等）
- 消融发现 RandomCrop 和 Mixup/CutMix 对预训练效果影响最大
- Exploration study 使用 ViT-Tiny，Scaling study 使用 ViT-Base
- 微调数据集包括 CIFAR-10/100、ImageNet-100/1k、Cars、Flowers 等

## 实验关键数据

### 主实验

与不同规模预训练数据集对比（ViT-Tiny，CIFAR-100 微调精度）：

| 数据集 | 图像数 | 类型 | CIFAR-100 | ImageNet-100 |
|--------|--------|------|-----------|--------------|
| Scratch | - | - | 64.2 | 74.9 |
| FractalDB | 1M | FDSL | 81.6 | 88.5 |
| OFDB | 1k | FDSL | 84.0 | 88.6 |
| **1p-frac** | **1** | **FDSL** | **84.2** | **89.0** |
| ImageNet-1k | 1.28M | SL | 85.5 | - |

ViT-Base 在 ImageNet-1k 微调：1p-frac（1 张图）达到 82.1%，超越 ImageNet-21k 预训练的 81.8%。

### 消融实验

| 配置 | CIFAR-100 | 说明 |
|------|-----------|------|
| Δ=0.001 | 1.2 | 扰动太小，预训练崩溃 |
| Δ=0.01 | 19.9 | 开始出现正面效果 |
| Δ=0.05 | 83.0 | 效果接近最优 |
| Δ=0.1 | **84.2** | 最优扰动度 |
| σ=3.5 (最复杂) | **84.2** | 最优分形复杂度 |
| σ=6.0 (类噪声) | 81.3 | IFS 结构仍提供正面效果 |
| Gaussian 噪声 | 1.1 | 完全失败，需要结构化图像 |
| Uniform 噪声 | 2.0 | 同样失败 |
| L=16 采样点 | 78.7 | 少量采样仍有正面效果 |
| L=1000 采样点 | **84.2** | 更多采样更好 |

### 关键发现

- **"反向扩展"成立**: 合成预训练图像从 1M → 1k → 1 时，性能从 81.6 → 84.0 → 84.2 反而提升
- **扰动度存在阈值**: Δ < 0.01 时预训练崩溃，说明分布支撑集需要有最小大小
- **结构化 > 随机性**: 高斯/均匀噪声完全失败，分形的递归自相似结构至关重要
- **真实图像也可以"反向扩展"**: 对灰度化 + Canny 边缘 + 仿射变换的真实图像应用 LPCE loss，也能获得正面预训练效果（C100: 82.2%），这与 1p-frac 的配置几乎等价
- **早期层更受益**: 线性探测实验显示 ViT 前 3 层的 1p-frac 表征质量甚至超过 ImageNet-1k 预训练
- **数据集构建极快**: 1p-frac 仅需 0.04 小时（约 2 分钟），对比 FractalDB 的 19 小时

## 亮点与洞察

- **颠覆性发现**: 单张合成图像预训练可以匹配甚至超越百万级真实图像预训练——这强烈暗示预训练的本质更接近"更好的权重初始化"而非"视觉概念学习"
- **优雅的数学框架**: LIEP 分布和 LPCE loss 提供了一个连续可控的实验工具，可以精确研究预训练的最小需求
- **深刻的反直觉结论**: 人眼无法区分的微小形状差异对模型预训练至关重要——网络学习的"概念"与人类认知的概念存在根本差异
- **实用价值**: 数据集构建时间从数小时压缩到 2 分钟，完全避免真实数据的隐私/版权问题

## 局限与展望

- 仅验证了 ViT 架构，未探索 CNN（如 ResNet）是否有类似"反向扩展"效果
- 当前最优 σ 和 Δ 是通过网格搜索确定的，缺乏理论指导
- 虽然全微调性能可比 ImageNet，但线性探测在深层仍有差距——说明深层语义表征的学习仍需真实数据
- 未探索自监督预训练（如 MAE）在极少合成图像上的表现
- VTAB 上个别任务（如 CLEVR-Count）仍落后于 ImageNet SL 预训练

## 相关工作与启发

- **Asano et al. (2020)**: 单图像自监督学习的先驱，但仅对浅层有效且未使用现代架构
- **OFDB** (Nakamura et al.): 将 FractalDB 压缩到 1000 张，本文进一步压缩到 1 张
- **Visual Atoms**: 另一种 FDSL 数据集，用参数化波函数生成图像，100 万张时优于 FractalDB，但在 1 张时不如 1p-frac
- 启发：预训练可能根本不需要"学习世界的视觉结构"，而是通过分类信号优化了网络权重的几何配置。这一洞察对理解 foundation model 的工作原理有深远意义。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 颠覆性结论，将最小预训练推到极限
- 实验充分度: ⭐⭐⭐⭐⭐ 从探索/超参/扩展/分析/应用五个维度全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，数学形式优雅，实验设计层层递进
- 价值: ⭐⭐⭐⭐⭐ 对预训练本质的深层洞察，且有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] PreLAR: World Model Pre-training with Learnable Action Representation](prelar_world_model_pre-training_with_learnable_action_representation.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/llm_pretraining/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[ICML 2025\] Metadata Conditioning Accelerates Language Model Pre-training](../../ICML2025/llm_pretraining/metadata_conditioning_accelerates_language_model_pre-training.md)
- [\[ICML 2026\] POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation](../../ICML2026/llm_pretraining/poet-x_memory-efficient_llm_training_by_scaling_orthogonal_transformation.md)

</div>

<!-- RELATED:END -->
