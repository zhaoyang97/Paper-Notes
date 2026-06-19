---
title: >-
  [论文解读] DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation
description: >-
  [ECCV2024][扩散模型] 提出 DC-Solver，通过动态补偿（Dynamic Compensation）缓解 predictor-corrector 扩散采样器中的 misalignment 问题，仅需 10 个数据点即可优化补偿比率，并通过级联多项式回归（CPR）实现对未见 NFE/CFG 配置的即时泛化。
tags:
  - "ECCV2024"
  - "扩散模型"
  - "Fast Sampling"
  - "ODE Solver"
  - "Predictor-Corrector"
  - "Dynamic Compensation"
---

# DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation

**会议**: ECCV2024  
**arXiv**: [2409.03755](https://arxiv.org/abs/2409.03755)  
**代码**: [wl-zhao/DC-Solver](https://github.com/wl-zhao/DC-Solver)  
**领域**: 其他  
**关键词**: diffusion model, Fast Sampling, ODE Solver, Predictor-Corrector, Dynamic Compensation

## 一句话总结

提出 DC-Solver，通过动态补偿（Dynamic Compensation）缓解 predictor-corrector 扩散采样器中的 misalignment 问题，仅需 10 个数据点即可优化补偿比率，并通过级联多项式回归（CPR）实现对未见 NFE/CFG 配置的即时泛化。

## 背景与动机

扩散概率模型（DPMs）在视觉生成中表现出色，但采样需要多次前向评估去噪网络，计算开销大。现有快速采样器如 DPM-Solver++、DEIS、UniPC 通过求解扩散 ODE 显著减少了所需的函数评估次数（NFE），其中 UniPC 引入了 predictor-corrector 范式来进一步提升采样质量。然而，predictor-corrector 框架存在一个固有的 **misalignment 问题**：corrector 步骤修正了中间结果 $\tilde{x}_{t_i}^c$，但缓冲区 $Q$ 中存储的模型输出 $\epsilon_\theta(\tilde{x}_{t_i}, t_i)$ 仍然是基于修正前的 $\tilde{x}_{t_i}$ 计算的，两者之间存在不一致。这种不对齐在使用大 classifier-free guidance（CFG）时会被进一步放大。

虽然直接重新计算 $\epsilon_\theta(\tilde{x}_{t_i}^c, t_i)$ 可以解决该问题，但这会引入额外的网络前向传播，使计算量翻倍。因此，如何在不增加 NFE 的前提下缓解这一 misalignment，是本文的核心动机。

## 核心问题

1. **Misalignment 问题**：predictor-corrector 采样器中，corrector 修正后的中间结果与缓冲区中复用的模型输出之间存在不一致，导致后续步骤的采样精度下降
2. **CFG 放大效应**：classifier-free guidance 的线性组合 $\bar{\epsilon}_\theta = s \cdot \epsilon_\theta(x_t, t, c) + (1-s) \cdot \epsilon_\theta(x_t, t, \varnothing)$ 会放大 misalignment 的负面影响
3. **NFE/CFG 泛化**：优化得到的补偿比率是特定于某一 NFE 和 CFG 配置的，用户调整配置时需要重新搜索

## 方法详解

### 1. 动态补偿（Dynamic Compensation）

核心思想是利用缓冲区 $Q$ 中已有的历史模型输出，通过 Lagrange 插值来近似 $\epsilon_\theta(\tilde{x}_{t_i}^c, t_i)$，而不需要额外的网络前向传播。

给定补偿比率 $\rho_i$，定义新的时间步 $t_i' = \rho_i t_i + (1 - \rho_i) t_{i-1}$，然后对缓冲区中的 $K+1$ 个历史模型输出做 Lagrange 插值：

$$\hat{\epsilon}^{\rho_i}(\tilde{x}_{t_i}^c, t_i) = \sum_{k=0}^{K} \prod_{\substack{0 \le l \le K \\ l \ne k}} \frac{t_i' - t_{i-l}}{t_{i-k} - t_{i-l}} \epsilon_\theta(\tilde{x}_{t_{i-k}}, t_{i-k})$$

- 当 $\rho_i = 1.0$ 时，估计值退化为原始的 $\epsilon_\theta(\tilde{x}_{t_i}, t_i)$（即不做补偿）
- 通过调整 $\rho_i$，可以在插值/外推之间切换，找到更好的近似
- 实验表明 $K=2$（二阶 Lagrange 插值）效果最佳

### 2. 补偿比率搜索

将每一步的 $\rho_i$ 视为可学习参数，通过最小化采样轨迹与真实轨迹之间的 $\ell_2$ 距离来优化：

$$\rho_i^* = \arg\min_{\rho_i} \mathbb{E} \| \tilde{x}_{t_{i+1}}^c(\tilde{x}_{t_i}^c, Q^{\rho_i}) - x_{t_{i+1}}^{GT} \|_2^2$$

关键设计：

- 真实轨迹由 999 步 DDIM（条件采样）或 200 步 DDIM（无条件采样）生成
- 仅需 $N=10$ 个初始噪声数据点
- 使用 AdamW 优化 $L=40$ 次迭代，单 GPU 上约 5 分钟即可完成
- 前两步跳过补偿（$\rho_0 = \rho_1 = 1.0$），因为缓冲区中历史数据不足

### 3. 级联多项式回归（Cascade Polynomial Regression, CPR）

作者观察到最优补偿比率随 NFE 和 CFG 的变化是近似连续的，因此提出 CPR 来即时预测任意配置下的补偿比率，无需重新搜索：

$$\hat{\rho}_i^* = f_3^{(p_3)}(i \mid \phi^{(3)})$$

其中 $\phi^{(3)}$ 的系数通过 $\text{CFG}$ 的多项式 $f_2^{(p_2)}$ 确定，而 $f_2$ 的系数又通过 $\text{NFE}$ 的多项式 $f_1^{(p_1)}$ 确定，形成三层级联结构。超参数取 $p_1 = p_2 = 2$，$p_3 = 4$。只需在少量 NFE/CFG 配置上搜索补偿比率，即可用 `scipy.curve_fit` 拟合 CPR 参数，然后泛化到未见配置。

### 4. 即插即用特性

动态补偿不仅适用于 predictor-corrector 采样器（如 UniPC），还可以作为即插即用模块提升 predictor-only 采样器（如 DDIM、DPM-Solver++）的性能。

## 实验关键数据

### 无条件采样（FID↓）

| 方法 | FFHQ NFE=5 | FFHQ NFE=10 | LSUN-Church NFE=5 | LSUN-Bedroom NFE=5 |
|------|-----------|------------|-------------------|-------------------|
| UniPC | 18.66 | 6.99 | - | - |
| **DC-Solver** | **10.38** | **6.82** | - | - |

DC-Solver 在 NFE=5 时将 FFHQ 上的 FID 从 18.66 降至 10.38（降幅 8.28）。

### 条件采样（MSE↓，SD2.1，CFG=7.5）

| 方法 | NFE=5 | NFE=10 |
|------|-------|--------|
| DPM-Solver++ | 0.443 | 0.370 |
| DEIS | 0.436 | 0.368 |
| UniPC | 0.434 | 0.373 |
| **DC-Solver** | **0.394** | **0.294** |

### 跨模型泛化（NFE=5，各模型默认 CFG）

| 模型 | DPM-Solver++ | UniPC | DC-Solver |
|------|-------------|-------|-----------|
| SD1.4 (512²) | 0.803 | 0.813 | **0.760** |
| SD2.1 (768²) | 0.443 | 0.434 | **0.394** |
| SDXL (1024²) | 0.745 | 0.718 | **0.689** |

### 消融实验要点

- **补偿方式**：动态补偿（$\rho_i = \rho_i^*$）显著优于静态补偿（$\rho_i \equiv 1.1$），后者又优于无补偿（$\rho_i \equiv 1.0$）
- **数据点数量**：$N=10$ 已足够，增加到 20/30 提升极小
- **插值阶数**：$K=2$ 最佳，$K=1$ 或 $K=3$ 均更差
- **优化迭代次数**：40 次迭代即可收敛，每步约 22.2 秒

## 亮点

1. **优雅的问题分析**：清晰地识别并形式化了 predictor-corrector 中的 misalignment 问题，解决方案简洁高效
2. **极低的搜索成本**：仅需 10 个数据点、40 次迭代、单 GPU 5 分钟，远优于 DPM-Solver-v3（1024 数据点、8 GPU 11 小时、125MB 存储）
3. **CPR 实现零成本泛化**：三层级联多项式回归可即时预测任意 NFE/CFG 下的补偿比率，无需重新搜索
4. **广泛适用性**：在 SD1.4/1.5/2.1/SDXL 上均有效，支持 512² 到 1024² 分辨率，兼容不同参数化方式（$\epsilon$-prediction、$v$-prediction）
5. **即插即用**：动态补偿可同时提升 predictor-only 和 predictor-corrector 两类采样器

## 局限与展望

1. **需要真实轨迹**：搜索阶段依赖 999 步 DDIM 生成的真实轨迹，虽然只需一次但仍有前置成本
2. **ODE 采样器限定**：方法专注于 ODE 求解器，对 SDE 采样器的适用性未探讨
3. **CPR 泛化边界**：级联多项式回归在极端 NFE/CFG 配置下的外推能力可能有限
4. **与蒸馏方法的结合**：未探索与 consistency distillation 等训练式加速方法的互补性
5. **领域分类为 others**：实际上是扩散模型加速的通用方法，在图像生成领域有广泛价值

## 与相关工作的对比

| 方法 | 类型 | 需训练 | 额外NFE | 泛化NFE/CFG | 搜索成本 |
|------|------|--------|---------|-------------|----------|
| DDIM | predictor-only | 否 | 否 | 天然 | 无 |
| DPM-Solver++ | predictor-only | 否 | 否 | 天然 | 无 |
| UniPC | predictor-corrector | 否 | 否 | 天然 | 无 |
| DPM-Solver-v3 | 学习式参数化 | 是 | 否 | 需重新训练 | 8GPU×11h |
| **DC-Solver** | predictor-corrector+DC | 轻量搜索 | 否 | CPR即时预测 | 1GPU×5min |

DC-Solver 的核心优势在于以极低成本获得显著的采样质量提升，同时通过 CPR 保持了对不同配置的灵活泛化能力。

## 启发与关联

1. **Lagrange 插值思路**：利用历史采样点进行插值/外推来近似目标函数值，这一思路可推广到其他需要减少函数评估次数的数值求解场景
2. **轻量级搜索范式**：仅需少量数据点和迭代即可找到接近最优的超参数，对比 DPM-Solver-v3 的重量级训练，展示了"少即是多"的设计哲学
3. **级联回归的实用性**：CPR 将离散搜索结果泛化为连续预测，这种方法论可应用于其他需要在多维超参数空间中快速预测最优配置的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ — misalignment 的形式化分析和动态补偿方案设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖多模型、多分辨率、多配置，消融全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题动机阐述充分
- 价值: ⭐⭐⭐⭐ — 即插即用的实用加速方法，对扩散模型社区有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adjoint Schrödinger Bridge Sampler](../../NeurIPS2025/others/adjoint_schrödinger_bridge_sampler.md)
- [\[CVPR 2025\] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](../../CVPR2025/others/tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[ECCV 2024\] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](../../ICCV2025/others/a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)

</div>

<!-- RELATED:END -->
