---
title: >-
  [论文解读] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment
description: >-
  [CVPR 2025][图像生成][Flow Matching] 提出 Diff2Flow 框架，通过时间步重缩放、插值对齐和速度场推导，实现从预训练扩散模型到 Flow Matching 模型的高效知识迁移，在文生图、深度估计等多任务上以极少微调开销取得优于或持平 SOTA 的性能。 领域现状：扩散模型（如 Stable…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "Flow Matching"
  - "扩散模型"
  - "知识迁移"
  - "参数高效微调"
  - "轨迹对齐"
---

# Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment

**会议**: CVPR 2025  
**arXiv**: [2506.02221](https://arxiv.org/abs/2506.02221)  
**代码**: [https://github.com/CompVis/diff2flow](https://github.com/CompVis/diff2flow)  
**领域**: 扩散模型 / 图像生成  
**关键词**: Flow Matching, 扩散模型, 知识迁移, 参数高效微调, 轨迹对齐

## 一句话总结

提出 Diff2Flow 框架，通过时间步重缩放、插值对齐和速度场推导，实现从预训练扩散模型到 Flow Matching 模型的高效知识迁移，在文生图、深度估计等多任务上以极少微调开销取得优于或持平 SOTA 的性能。

## 研究背景与动机

**领域现状**：扩散模型（如 Stable Diffusion）已在图像生成领域取得巨大成功，同时 Flow Matching（FM）作为替代范式因其更直的采样轨迹和更快的推理速度而受到关注。当前 SOTA 的 FM 基础模型（如 Flux、SDv3）参数量超 8B，微调代价极高。

**现有痛点**：现有基于 FM 的大模型因体量庞大导致微调不切实际，尤其在资源受限环境下。而 Stable Diffusion 架构高效、生态完善，但其扩散训练范式在推理效率上不如 FM。直接将 FM 目标应用于扩散模型会因两者在插值定义、时间步缩放和训练目标上的不匹配而导致收敛缓慢、性能下降。

**核心矛盾**：扩散模型和 Flow Matching 虽然可以统一到同一框架下，但实际实现中在三个关键方面存在错位：(1) 插值公式不同——扩散用 $x_t = \alpha_t x_0 + \sigma_t \epsilon$，FM 用线性插值 $x_t = t x_1 + (1-t) x_0$；(2) 时间步空间不同——扩散是离散 $[0, T]$，FM 是连续 $[0, 1]$；(3) 训练目标不同——扩散预测噪声或 v，FM 预测速度场。

**本文目标**：高效地将预训练扩散模型的知识迁移到 FM 模型，使其兼具扩散模型的先验知识和 FM 的推理优势。

**切入角度**：作者发现扩散模型在非整数时间步上进行推理时仍能生成合理结果，说明其内部的时间步嵌入构成了连续的时间空间。基于这一观察，可以构建扩散与 FM 之间的可逆映射。

**核心 idea**：通过显式构建时间步映射 $f_t$ 和数据点映射 $f_x$，将扩散轨迹"变形"为 FM 线性轨迹，并利用扩散模型的预测推导出 FM 所需的速度估计，从而实现无缝知识迁移。

## 方法详解

Diff2Flow 的核心思想是：不是让模型"忘掉"扩散的参数化方式再重新学习速度预测，而是在数学上精确地将扩散模型的预测转换成 FM 的速度场。整个方法分为轨迹穿越和目标统一两个部分。

### 整体框架

输入为预训练的扩散模型（如 SD 2.1）和训练数据。训练阶段：对每个样本，先按 FM 方式构建线性插值 $x_{t_{FM}}^{FM}$，然后通过逆映射 $f_t^{-1}$ 和 $f_x^{-1}$ 将其转换回扩散空间的数据点和时间步，送入扩散模型获得预测，再通过目标转换公式得到速度估计，最后用标准 FM loss 训练。推理阶段同样做映射后用 Euler 步进采样。

### 关键设计

1. **轨迹穿越（Trajectory Traversal）**:

    - 功能：建立扩散轨迹和 FM 线性轨迹之间的双向可逆映射
    - 核心思路：定义数据点映射 $f_x(x_{t_{DM}}^{DM}) = \frac{1}{\alpha_t + \sigma_t} x_{t_{DM}}^{DM}$，将扩散插值缩放为线性形式；定义时间步映射 $f_t(t_{DM}) = \frac{\alpha_t}{\alpha_t + \sigma_t}$，将离散扩散时间步映射到 FM 的连续 $[0,1]$ 空间。对于非整数时间步，通过分段线性插值扩展到连续域。逆映射通过线性插值从 FM 时间步反求扩散时间步。
    - 设计动机：扩散的方差保持调度下 $\alpha_t^2 + \sigma_t^2 = 1$，使得扩散插值天然可以通过除以 $(\alpha_t + \sigma_t)$ 转化为线性组合形式，与 FM 插值形式对齐。这种映射保证了边界条件一致——噪声端和数据端完全对应。

2. **目标统一（Objective Unification）**:

    - 功能：利用扩散模型的原始预测（如 v-prediction）直接推导出 FM 所需的速度场
    - 核心思路：以 v-参数化为例，扩散模型预测 $v_\theta = \alpha_t \epsilon - \sigma_t x_0$。通过代数推导可以从 $v_\theta$ 恢复出 $\hat{x}_0^{DM}$ 和 $\hat{x}_T^{DM}$ 的估计，进而得到 FM 速度 $\mathbf{v}_\theta(x^{FM}, t_{FM}) = (\alpha_t - \sigma_t)(x_{t_{DM}}^{DM} - v_\theta)$。这意味着模型不需要"重学"一种新的参数化方式，而是在原有预测基础上做确定性变换。
    - 设计动机：先前工作（如 DepthFM）直接让 v-参数化模型预测速度场，迫使模型转换参数化方式，需要更长收敛时间且影响性能。通过目标统一，模型的原始知识被充分保留。

3. **参数高效微调（LoRA 适配）**:

    - 功能：以极少量参数变化完成扩散到 FM 的转换
    - 核心思路：使用 LoRA 方法仅更新权重矩阵的低秩分解 $\Delta W = BA$，冻结主模型权重。关键发现是：直接对扩散模型施加 FM 目标时 LoRA 不工作（因参数化转换需要大幅调整），但结合 Diff2Flow 的对齐策略后 LoRA 可以高效工作。
    - 设计动机：在计算受限的场景下，全量微调不可行。Diff2Flow 的对齐使得模型只需微小参数调整而非完全重新学习，因此 LoRA 的低秩约束不再是瓶颈。

### 损失函数 / 训练策略

使用标准 FM 损失 $\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \|(x_1 - x_0) - \mathbf{v}_\theta(x_t, t)\|^2$。训练流程为：采样 FM 时间步 $t_{FM} \in [0,1]$，构建 FM 插值，逆映射到扩散空间，通过模型获得预测并转换为速度估计，计算 FM loss 进行梯度下降。推理使用标准 Euler 采样。

## 实验关键数据

### 主实验

| 任务 | 方法 | FID ↓ | CLIP ↑ | 美学分 ↑ |
|------|------|-------|--------|----------|
| T2I (SD1.5续训) | SD1.5 | 56.77 | 26.34 | 5.32 |
| T2I (SD1.5续训) | SD1.5 cont. | 56.36 | 26.33 | 5.90 |
| T2I (SD1.5续训) | **Diff2Flow** | **52.80** | **26.54** | **5.99** |

### 消融实验

| 配置 | 说明 |
|------|------|
| FM w/o 对齐 + Full FT | 收敛慢，最终可追上性能 |
| FM w/o 对齐 + LoRA | 性能严重下降，无法收敛到合理水平 |
| Diff2Flow + Full FT | 约 2.5k 步收敛 |
| Diff2Flow + LoRA | 接近 Full FT 性能，远超朴素 FM |

### 关键发现

- 在全量微调条件下，naive FM 和 Diff2Flow 最终能收敛到相近性能，但 Diff2Flow 收敛速度显著更快（约 2.5k 步）
- 在 LoRA 条件下差距更加明显：naive FM 无法缩小与 Diff2Flow 的差距，说明对齐是 PEFT 成功的前提
- Diff2Flow 自然解决了扩散模型的 zero-terminal SNR 问题——转为 FM 后可以正确生成纯黑/纯白图像
- 将 SD1.5 转为 FM 后在同一生成任务上也有性能提升，验证了 FM 轨迹更直的优势
- 应用 Reflow 可以使 SD1.5 仅用 2 步即可生成图像

## 亮点与洞察

- 方法的优雅之处在于数学推导严谨且实现简洁——核心就是两个可逆映射和一个代数变换，无需额外网络组件
- 发现扩散模型在非整数时间步也能正常工作，为正弦位置编码构建连续时间空间提供了实验证据
- 解决了一个实际痛点：大规模 FM base model 微调不起，小但好的扩散模型可以通过 Diff2Flow 获得 FM 的优势
- 方法对不同参数化（epsilon/v）都适用，通用性强

## 局限与展望

- 目前仅在 SD1.5/SD2.1 上验证，对于更大的模型（如 SDXL）的适用性需要进一步探索
- 深度估计实验中训练样本较少（74K），在更大规模数据上的行为需要研究
- 是否可以反过来——从 FM 模型向扩散模型迁移，论文未讨论
- 对于非方差保持调度（如 VE schedule），虽然理论上适用但实验验证有限

## 相关工作与启发

- **DepthFM** 将扩散先验用于 FM 深度估计，但存在对齐缺陷，本文正是针对这一问题提出系统化解决方案
- **Reflow** 通过配对数据拉直采样轨迹，Diff2Flow 可以直接在扩散模型上应用 Reflow
- 与 InstaFlow、一致性蒸馏方向互补——Diff2Flow 关注范式转换而非步骤压缩

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次系统化解决扩散到FM的精确对齐问题 |
| 实验充分度 | 4 | 多任务验证（文生图/深度/reflow），含LoRA消融 |
| 写作质量 | 4 | 数学推导清晰，图示直观 |
| 实用价值 | 5 | 高度实用，让现有扩散模型直接受益于FM优势 |
- 价值: ⭐⭐⭐⭐⭐ 让所有扩散预训练权重可无缝迁移到 FM

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](../../ICLR2026/image_generation/densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_diffusion_alignment.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[CVPR 2025\] Color Alignment in Diffusion](color_alignment_in_diffusion.md)

</div>

<!-- RELATED:END -->
