---
title: >-
  [论文解读] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment
description: >-
  [CVPR 2025][图像生成][扩散模型] 本文提出 DDIM-InPO，通过将扩散模型视为单步生成模型并利用 DDIM 反演技术找到与偏好数据高度相关的潜变量，实现仅需 400 步微调即可达到 SOTA 的高效扩散模型偏好对齐。 领域现状：文本到图像扩散模型（如 SDXL）已具备强大的生成能力，但生成结果与人类偏好的对…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型"
  - "DPO"
  - "DDIM反演"
  - "偏好优化"
  - "人类偏好对齐"
---

# InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment

**会议**: CVPR 2025  
**arXiv**: [2503.18454](https://arxiv.org/abs/2503.18454)  
**代码**: [GitHub](https://github.com/JaydenLyh/InPO)  
**领域**: 图像生成 / 扩散模型对齐  
**关键词**: 扩散模型, DPO, DDIM反演, 偏好优化, 人类偏好对齐

## 一句话总结

本文提出 DDIM-InPO，通过将扩散模型视为单步生成模型并利用 DDIM 反演技术找到与偏好数据高度相关的潜变量，实现仅需 400 步微调即可达到 SOTA 的高效扩散模型偏好对齐。

## 研究背景与动机

**领域现状**：文本到图像扩散模型（如 SDXL）已具备强大的生成能力，但生成结果与人类偏好的对齐仍是重要挑战。LLM 领域的 DPO 方法已成功用于偏好对齐，但扩散模型因长马尔可夫链和反向过程不可解性而难以直接应用。

**现有痛点**：(1) 基于奖励模型的方法（DRaFT、AlignProp）需要通过整个采样过程反向传播，内存开销大且易发生奖励泄漏；(2) RL 方法（DDPO、DPOK）受马尔可夫链长度限制，效率低下；(3) Diffusion-DPO 等直接偏好优化方法将奖励分散到所有去噪步骤，导致稀疏奖励问题，尤其在分布外输入上表现不佳。

**核心矛盾**：现有方法在整个去噪链上分配奖励导致训练信号稀疏，而实际上只有少量潜变量与最终图像质量强相关。

**本文目标**：只微调与目标图像强相关的少量潜变量的输出，实现快速、高质量的偏好优化。

**切入角度**：将扩散模型重新参数化为单步生成框架——在任意时间步 $t$，模型都可以从 $x_t$ 一步估计 $x_0$，据此直接分配隐式奖励。

**核心 idea**：通过 DDIM 重参数化建立潜变量与目标图像的单步映射，再用 DDIM 反演在目标图像空间找到与偏好数据高度相关的潜变量，仅优化这些潜变量的输出。

## 方法详解

### 整体框架

给定偏好数据集 $\{(x_0^w, x_0^l, c)\}$（胜者/败者图像对和对应 prompt），目标是训练模型 $p_\theta$ 对齐人类偏好。方法分三步：(1) 通过 DDIM 重参数化建立任意时间步潜变量到 $x_0$ 空间的单步映射；(2) 用反演技术从偏好图像估计高相关性的潜变量；(3) 仅优化这些潜变量对应的输出。

### 关键设计

1. **DDIM 重参数化的 DPO 奖励分配**:

    - 功能：将扩散模型概念化为时间步感知的单步生成器，直接在任意时间步分配隐式奖励
    - 核心思路：利用 DDIM 的 $x_0(t) = \bar{x}_t - \sigma_t \epsilon_\theta^t(x_t, c)$ 重参数化，将 $x_t$ 与 $x_0$ 空间关联。定义联合奖励 $r_t^c(x_0, x_t)$ 满足 $r(x_0, c) = \mathbb{E}_{p_\theta^c(x_t|x_0)}[r_t^c(x_0, x_t)]$。通过最小化联合 KL 散度 $D_{KL}[p_\theta^c(x_0, x_t) \| p_{ref}^c(x_0, x_t)]$ 作为标准 KL 的上界，推导出适用于任意单步的 DPO 目标
    - 设计动机：相比 Diffusion-DPO 在整个 $x_{0:T}$ 路径上分配奖励，单步映射可以将奖励精确聚焦到与目标图像最相关的变量上

2. **DDIM 反演的潜变量选择**:

    - 功能：找到与偏好数据强相关的潜变量，仅优化这些变量
    - 核心思路：给定偏好图像 $x_0^w$ 和 $x_0^l$，使用 DDIM 反演从 $x_0$ 空间映射回各时间步的 $x_t$。这些反演得到的 $x_t$ 与原始偏好图像高度相关（因为确定性 DDIM 过程保持了图像结构）。只需对这些特定 $x_t$ 计算 DPO 损失即可
    - 设计动机：避免在随机采样的 $x_t$ 上优化（这会导致稀疏奖励），而是精确定位"对生成质量影响最大"的潜变量

3. **高效单步优化目标**:

    - 功能：实现极低计算成本的偏好对齐
    - 核心思路：最终损失简化为：对随机采样的时间步 $t$，用参考模型反演得到 $x_t^w, x_t^l$，计算当前模型和参考模型在该 $x_t$ 上的单步预测 $x_0(t)$ 的对数概率差。每次训练只需一次前向传播和一次反演
    - 设计动机：相比需要多步去噪的方法，单步优化大幅降低内存和计算开销，使 400 步即可完成微调

### 损失函数 / 训练策略

核心损失基于重参数化的 DPO：对随机采样时间步 $t$，用 DDIM 反演获得 $x_t^w, x_t^l$，计算 $\log\sigma(\beta[\log p_\theta^c(x_0^w, x_t^w) / p_{ref}^c(x_0^w, x_t^w) - \log p_\theta^c(x_0^l, x_t^l) / p_{ref}^c(x_0^l, x_t^l)])$。仅需 400 步微调 SDXL-base-1.0。

## 实验关键数据

### 主实验

| 方法 | PickScore ↑ | HPSv2 ↑ | ImageReward ↑ | GenEval ↑ | 训练步数 |
|------|-----------|---------|-------------|----------|---------|
| SDXL (baseline) | 22.00 | 28.48 | 0.88 | 0.55 | - |
| Diffusion-DPO | 22.10 | 28.89 | 1.01 | 0.58 | 2000 |
| D3PO | 22.05 | 28.71 | 0.95 | 0.56 | 2000 |
| DenseReward | 22.12 | 28.93 | 1.05 | 0.59 | 2000 |
| **DDIM-InPO** | **22.25** | **29.12** | **1.18** | **0.62** | **400** |

### 消融实验

| 配置 | PickScore | HPSv2 | 说明 |
|------|---------|-------|------|
| Full InPO (400步) | 22.25 | 29.12 | 完整模型 |
| w/o Inversion (随机 $x_t$) | 22.08 | 28.82 | 去掉反演，性能大幅下降 |
| 全链 DPO | 22.10 | 28.89 | 退化为 Diffusion-DPO |
| InPO (200步) | 22.18 | 29.01 | 200步已有显著提升 |

### 关键发现

- DDIM 反演是性能提升的关键——不用反演而随机采样 $x_t$ 会退化到普通 Diffusion-DPO 的水平
- 400 步微调即超越 2000 步的 Diffusion-DPO，训练效率提升约 5 倍
- 生成图像在视觉美感和 prompt 一致性上都有显著改善
- 方法对 $\beta$ 超参数较为稳健

## 亮点与洞察

- **概念突破：扩散模型=单步生成器**：通过 DDIM 重参数化，将复杂的多步去噪过程简化为单步映射问题，使 DPO 可以直接高效地应用。这个视角非常优雅
- **反演即数据增强**：DDIM 反演本质上是在为偏好数据找到最相关的潜空间表示，相当于一种"对齐感知"的数据增强策略
- **效率革命**：400 步微调 SDXL 就能超越 SOTA，对实际部署极具价值。可迁移到其他需要快速适配的场景（如风格定制）

## 局限与展望

- 依赖 DDIM 的确定性采样假设，对随机采样器的适用性待验证
- 反演过程引入额外计算开销（虽然是一次性的）
- 仅在 SDXL 上验证，对其他架构（如 DiT、Flux）的适用性未知
- 可探索结合 online 偏好数据生成的在线版本

## 相关工作与启发

- **vs Diffusion-DPO**: 在整个去噪链上均匀分配奖励导致稀疏信号；InPO 通过反演精准定位关键潜变量，效率提升 5x
- **vs DRaFT/AlignProp**: 需要通过整个采样过程反传梯度，内存开销大；InPO 只需单步前向传播
- **vs DDPO**: RL 策略梯度方法，需要在线生成大量样本；InPO 可直接在离线偏好数据上训练

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将扩散模型重新概念化为单步生成器并结合反演的思路非常新颖
- 实验充分度: ⭐⭐⭐⭐ 多个评估指标，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 训练效率的巨大提升对实际应用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/image_generation/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)

</div>

<!-- RELATED:END -->
