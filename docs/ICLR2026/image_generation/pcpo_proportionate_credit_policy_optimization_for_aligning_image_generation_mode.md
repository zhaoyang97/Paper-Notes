---
title: >-
  [论文解读] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models
description: >-
  [ICLR 2026][图像生成][策略梯度] 提出 PCPO，通过稳定目标重构和原则性时间步重加权，修正扩散/流模型策略梯度中固有的不成比例信用分配问题，显著加速收敛并缓解模型崩溃。 - GRPO 成为 T2I 模型对齐的 SOTA 框架，但训练不稳定且易出现模型崩溃 - 根因分析： 1. 标准目标易受数值精度误差影响 2…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "策略梯度"
  - "信用分配"
  - "扩散模型"
  - "流匹配"
  - "模型崩溃"
---

# PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models

**会议**: ICLR 2026  
**arXiv**: [2509.25774](https://arxiv.org/abs/2509.25774)  
**代码**: [GitHub](https://github.com/jaylee2000/pcpo/)  
**领域**: 图像生成  
**关键词**: 策略梯度, 信用分配, 扩散模型, 流匹配, 模型崩溃

## 一句话总结

提出 PCPO，通过稳定目标重构和原则性时间步重加权，修正扩散/流模型策略梯度中固有的不成比例信用分配问题，显著加速收敛并缓解模型崩溃。

## 研究背景与动机

- GRPO 成为 T2I 模型对齐的 SOTA 框架，但训练不稳定且易出现模型崩溃
- **根因分析**：
  1. 标准目标易受数值精度误差影响
  2. 采样器数学结构导致**不成比例的信用分配**——不同时间步的梯度贡献被任意缩放

## 方法详解

### 整体框架

PCPO 把 GRPO 式 T2I 对齐看成一个去噪轨迹上的策略梯度问题，先诊断出标准目标的两大病灶——数值上易爆的策略比率、以及采样器数学结构强加的非均匀时间步权重——再针对性地换掉目标函数并校正每一步的信用权重。整套修正不动模型结构和采样流程，只改训练目标和方差/重加权调度，因此可以直接套在 DDPO、DanceGRPO、Flow-GRPO 等现有框架上。

### 关键设计

**1. 问题诊断：揪出非均匀的原生权重 $w(t)$**

不稳定到底从哪来，作者用 Proposition 1 给出了 DDIM 采样下 log 策略比率的精确分解：

$$\log \rho_t = -\left[w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)}) \cdot \boldsymbol{\epsilon}_{\text{old}}^{(t)} + \frac{1}{2}\|w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)})\|^2\right]$$

关键在于原生权重 $w(t) = C(t)/\sigma_t$ 在不同时间步上跨越数个量级，导致同一条轨迹里某些步的梯度被任意放大、另一些被压没。把它对照标准 REINFORCE 来看就更清楚：REINFORCE 里各动作的贡献是均匀缩放的，而扩散采样器的梯度公式形式相同却多出了这个非均匀的 $w(t)$——它是采样器数学的副产物，而非任何刻意设计的信用分配策略。诊断到这一层，后面的修正就有了明确靶点。

**2. 稳定的 log-hinge 目标：把易爆的 $\rho_t-1$ 换成 $\log\rho_t$**

标准目标里出现的 $\rho_t - 1$ 在数值精度有限时容易被误差主导，作者改用更平滑的 $\log\rho_t$，二者在小更新下差异极小（Taylor 近似误差 < 1.2%），却显著降低了裁剪比率。重构后的基础目标为

$$\mathcal{L}_{\text{PCPO-base}}(\theta) = \mathbb{E}\left[\sum_{t=1}^T \max\{0, \xi|A| - A\log\rho_t\}\right]$$

其中 $A$ 是优势、$\xi$ 控制 hinge 边界。这一步只换目标形式不引入额外开销，是后续成比例信用分配能稳定生效的前提。

**3. 成比例信用分配：强制每步权重对齐**

诊断出 $w(t)$ 非均匀是病根后，PCPO 直接把它拉平。对扩散模型，重新设计 DDIM 的方差调度 $\tilde{\sigma}_t$，使 $w(t) = w^*$ 成为常数，并把 $w^*$ 缩放到与原始权重的均值匹配，从而既消除非均匀又不改变整体梯度量级；对流模型，Proposition 2 给出更直接的做法——重加权训练目标让每步信用与其积分区间成正比，$w(t_i) = \zeta \Delta t_i$。两种修正殊途同归地恢复了正确的信用分配，这也是 FLUX 这类时间步偏移更剧烈、原生权重更不均匀的模型上加速最明显的原因。

## 实验关键数据

### 训练效率

| 基线 | 奖励 | 目标水平 | 基线 Epochs | PCPO Epochs | 加速 |
|------|------|---------|------------|------------|------|
| DDPO | Aesthetics | 6.90 | 147 | 118 | **24.6%** |
| DDPO | BERTScore | 0.52 | 191 | 146 | **30.8%** |
| DanceGRPO (SD1.4) | HPS | 0.370 | 236 | 188 | **25.5%** |
| DanceGRPO (FLUX) | HPS | 0.360 | 209 | 148 | **41.2%** |

### 图像质量（匹配奖励水平下）

| 模型 | 方法 | FID(↓) | FD_DINO(↓) | LPIPS(↑) |
|------|------|--------|-----------|----------|
| SD1.5 (batch=512) | 基线 | 24.09 | 451.19 | 0.6321 |
| SD1.5 (batch=512) | **PCPO** | **22.06** | **391.30** | **0.6525** |
| FLUX | 基线 | 46.23 | 539.83 | 0.5736 |
| FLUX | **PCPO** | **40.38** | **438.88** | 0.5708 |

### 关键发现

1. PCPO 在所有设定下均显著降低裁剪比率（clipping fraction），直接改善收敛
2. FLUX 上加速最显著（41.2%），因为时间步偏移使原生权重更不均匀
3. LMM 统计分析证实 PCPO 对 FID 的改善具有统计显著性（p=0.047）
4. 在 MSCOCO-2017 和 MJHQ-30K 未见提示上泛化良好

## 亮点与洞察

1. **根因分析精准**：将训练不稳定归因于采样器数学结构导致的不成比例信用分配
2. **统一处理扩散和流模型**：针对不同采样器提供了原则性的修正方案
3. **实现简单但效果显著**：仅需修改方差调度或重加权目标，即插即用
4. **缓解模型崩溃**：不仅加速收敛，还维持了图像多样性和保真度

## 局限性

- $\log \rho_t \approx \rho_t - 1$ 的 Taylor 近似假设小策略更新，大步长更新时可能失效
- 方差调度修改可能改变采样轨迹特性（尽管实验表明影响可忽略）
- 与 TempFlow-GRPO、MixGRPO 等并发工作的深入比较有限
- 仅在 Aesthetics、BERTScore、HPSv2.1 三种奖励下验证

## 相关工作

- **T2I 对齐**：DDPO, DPO-Diffusion, DanceGRPO, Flow-GRPO
- **LLM 对齐**：PPO, GRPO, DPO
- **模型崩溃**：Shumailov et al. (2024), 奖励黑客

## 评分

- 新颖性：⭐⭐⭐⭐ — 诊断精准但修正方法相对简单
- 技术深度：⭐⭐⭐⭐ — 从 REINFORCE 视角的分析深刻，数学推导完整
- 实验完整性：⭐⭐⭐⭐⭐ — 多框架多奖励验证 + LMM 统计验证
- 实用价值：⭐⭐⭐⭐⭐ — 即插即用，对现有对齐流程有直接提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PCPO: Proportionate Credit Policy Optimization for Preference Alignment of Image Generation Models](pcpo_proportionate_credit_policy_optimization_for_preference_alignment_of_image_.md)
- [\[ICLR 2026\] STORK: 通过同时解决刚性与结构依赖来加速扩散与流匹配采样](stork_faster_diffusion_and_flow_matching_sampling_by_resolving_both_stiffness_an.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICLR 2026\] Mitigating Noise Shift in Denoising Generative Models with Noise Awareness Guidance](mitigating_noise_shift_in_denoising_generative_models_with_noise_awareness_guida.md)
- [\[ICLR 2026\] Reinforcing Diffusion Models by Direct Group Preference Optimization](reinforcing_diffusion_models_by_direct_group_preference_optimization.md)

</div>

<!-- RELATED:END -->
