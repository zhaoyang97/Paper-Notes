---
title: >-
  [论文解读] Remasking Discrete Diffusion Models with Inference-Time Scaling
description: >-
  [NeurIPS 2025][计算生物][离散扩散模型] 提出 ReMDM 采样器，通过在生成过程中允许已解码 token 被重新掩码（remask），赋予离散掩码扩散模型迭代纠错能力，实现推理时计算缩放，在文本、图像和分子设计任务上显著提升采样质量。 扩散模型在图像和视频生成上取得了巨大成功，其核心优势之一是迭代精修：——…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "离散扩散模型"
  - "重掩码采样"
  - "推理时计算缩放"
  - "迭代精修"
  - "可控生成"
---

# Remasking Discrete Diffusion Models with Inference-Time Scaling

**会议**: NeurIPS 2025  
**arXiv**: [2503.00307](https://arxiv.org/abs/2503.00307)  
**代码**: [https://github.com/guanghanwang/remdm](https://github.com/guanghanwang/remdm)  
**领域**: 计算生物  
**关键词**: 离散扩散模型, 重掩码采样, 推理时计算缩放, 迭代精修, 可控生成

## 一句话总结
提出 ReMDM 采样器，通过在生成过程中允许已解码 token 被重新掩码（remask），赋予离散掩码扩散模型迭代纠错能力，实现推理时计算缩放，在文本、图像和分子设计任务上显著提升采样质量。

## 研究背景与动机

扩散模型在图像和视频生成上取得了巨大成功，其核心优势之一是**迭代精修**——在多步生成中反复修正输出、修复错误。然而，当前最先进的离散扩散模型（特别是基于掩码/吸收态的 MDLM）存在一个根本性限制：**一旦 token 被解码，就无法再更新**（failure to remask property）。即使解码时引入了错误，该 token 也会被"锁定"，类似于自回归模型的顺序生成局限。

这个限制带来三个问题：（1）并行解码时token间的独立性导致不一致错误（如"She sell"而非"She sells"或"They sell"）；（2）无法通过增加采样步数来有效提升质量（推理时计算缩放受限）；（3）可控生成能力受限。

本文的核心 idea 是：设计一个新的后验分布，允许已解码的 token 以一定概率被重新掩码，从而实现迭代纠错。关键的巧妙之处在于，这个新后验保持了与 MDLM 相同的边际分布，因此可以直接复用预训练的 MDLM 权重，无需重新训练。

## 方法详解

### 整体框架

ReMDM 在 MDLM 的基础上，引入一个参数 σ_t 控制重掩码概率。当 σ_t=0 时退化为标准 MDLM；当 σ_t>0 时，已解码的 token 有概率被重新掩码，从而获得迭代精修的能力。整个方法可以直接作为采样器应用于预训练的 MDLM 模型上。

### 关键设计

1. **重掩码后验（Remasking Posterior）**：对于已解码的 token z_t ≠ m，后验为 q(z_s|z_t=x, x) = (1-σ_t)x + σ_t·m，即以 σ_t 的概率重新掩码。对于仍被掩码的 token，后验经过精心设计以保持与 MDLM 相同的边际分布 q(z_t|x)。这是一个核心定理（Theorem 3.1），保证了可以复用预训练权重。ReMDM 是一个非马尔可夫过程（类似于 DDIM 之于 DDPM 的关系）。

2. **σ_t 的设计策略**：作者提出了多种策略：

    - **Max-Capped (ReMDM-cap)**：将 σ_t 上限截断为常数 η_cap
    - **Rescaled (ReMDM-rescale)**：σ_t = η_rescale · σ_t^max，通过缩放因子控制重掩码强度
    - **Confidence-Based (ReMDM-conf)**：根据模型对每个 token 预测的置信度分配重掩码概率——置信度低的 token 更可能被重掩码，这是一个非常直觉的设计

3. **开关策略（Turn On/Off）**：

    - **Switch**：在 t_switch 之后才开启重掩码，先用标准 MDLM 生成初步候选
    - **Loop**：分三个阶段——(1) 标准 MDLM 解码；(2) 保持 α 不变，在循环中反复重掩码和预测（纠错阶段）；(3) 用标准 MDLM 解码剩余 token。这是最强的策略，直觉是先生成草稿、再反复修改

4. **与预测器-校正器的关系**：作者证明 ReMDM 更通用——FB 校正器和 DFM 校正器都是 ReMDM 的特例（Proposition 4.2, 4.3），且 ReMDM 可以处理 α_t 为常数的情况（Proposition 4.4），这是 DFM 无法做到的。

### 损失函数 / 训练策略

ReMDM 的 NELBO 为 MDLM 目标的一个重加权版本，仅多了 (1-σ_t) 因子。由于 σ_t=0 时退化为 MDLM，且实验验证训练目标相似时性能可比，**推荐方案是直接复用预训练 MDLM 权重**，仅在推理时使用 ReMDM 采样器，无需重训。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | ReMDM | 之前SOTA | 提升 |
|------------|------|-------|---------|------|
| OWT 文本生成 (T=4096) | MAUVE ↑ | 0.656 | 0.269 (DFM) | 2.4× |
| OWT 文本生成 (T=1024) | MAUVE ↑ | 0.403 | 0.254 (DFM) | 1.6× |
| OWT 快速采样 (T=512) | MAUVE ↑ | 0.350 | 0.211 (DFM) | 1.7× |
| ImageNet 生成 (T=64) | FID ↓ | 4.45 | 4.69 (MDLM) | 更优 |
| ImageNet 生成 (T=64) | IS ↑ | 209.45 | 196.38 (MaskGiT) | +6.7% |
| LLaDA Countdown | pass@1% | 46.1 | 45.2 (LLaDA) | +0.9 |

### 消融实验

| 配置 | MAUVE (OWT T=4096) | 说明 |
|------|-------------------|------|
| MDLM (baseline) | 0.035 | 无重掩码 |
| + 64-bit precision | 提升 | 避免多样性受限 |
| + Nucleus sampling | 提升 | Top-p=0.9 关键质量提升 |
| + ReMDM remasking | 大幅提升 | 最大贡献来源 |
| + Loop strategy | 0.656 | 完整 ReMDM |

### 关键发现
- ReMDM 在推理时计算缩放方面表现优异：增加步数 T 能持续提升质量，而 MDLM 和校正器方法会饱和
- 在分子设计任务中，ReMDM 将可控生成的 novelty-property Pareto 前沿推向更优区域
- 将 ReMDM 应用于 LLaDA 8B 等大型 dLLM 时，也能显著提升下游任务表现

## 亮点与洞察
- **零额外训练成本**：直接复用预训练 MDLM 权重，仅改变采样策略即可获得显著提升，这极大降低了使用门槛
- **理论优雅**：通过保持边际分布不变，构建了与 MDLM 兼容的非马尔可夫过程，同时统一了 FB、DFM 等校正器为特例
- **DDIM 的离散版本**：ReMDM 之于 MDLM 就像 DDIM 之于 DDPM，为离散扩散提供了更灵活的采样空间

## 局限与展望
- 重掩码策略（σ_t schedule、loop 参数）的超参搜索空间较大，需要针对不同任务调参
- 重掩码增加了采样步数的有效使用，但相应地需要更多计算步来达到最佳效果
- 目前主要验证在文本、离散化图像和分子上，更大规模语言模型和连续信号上的扩展有待探索

## 相关工作与启发
- **vs DDIM**: DDIM 在连续域通过非马尔可夫过程实现灵活采样；ReMDM 将这一思想迁移到离散域的吸收态扩散
- **vs FB/DFM correctors**: ReMDM 更通用，能处理 αt 为常数的情况（loop策略），这些校正器做不到
- **vs MaskGiT**: MaskGiT 基于模型置信度解码但不允许重掩码；ReMDM的confidence-based schedule 融合了两者优势

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从概率模型角度优雅地解决了离散扩散的核心限制，理论推导完整
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖文本、图像、分子设计三个领域，加上LLaDA大模型验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、理论推导严谨、实验展示充分
- 价值: ⭐⭐⭐⭐⭐ 零训练成本即可提升离散扩散模型性能，对 dLLM 社区有直接影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[ICLR 2026\] DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](../../ICLR2026/computational_biology/driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[NeurIPS 2025\] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)

</div>

<!-- RELATED:END -->
