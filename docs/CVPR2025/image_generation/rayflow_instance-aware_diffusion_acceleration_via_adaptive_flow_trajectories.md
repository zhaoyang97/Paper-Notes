---
title: >-
  [论文解读] RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories
description: >-
  [CVPR 2025][图像生成][扩散模型加速] 提出 RayFlow 扩散框架，为每个样本设计独特的扩散路径（指向实例特定目标分布），并通过 Time Sampler 重要性采样优化训练，在最小化采样步数的同时保持生成多样性和稳定性。 扩散模型生成速度慢仍是核心挑战。现有加速方法存在以下问题： 1. 传统扩散的三个问题：…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型加速"
  - "流匹配"
  - "实例感知路径"
  - "重要性采样"
  - "采样稳定性"
---

# RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories

**会议**: CVPR 2025  
**arXiv**: [2503.07699](https://arxiv.org/abs/2503.07699)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 扩散模型加速, 流匹配, 实例感知路径, 重要性采样, 采样稳定性

## 一句话总结

提出 RayFlow 扩散框架，为每个样本设计独特的扩散路径（指向实例特定目标分布），并通过 Time Sampler 重要性采样优化训练，在最小化采样步数的同时保持生成多样性和稳定性。

## 研究背景与动机

扩散模型生成速度慢仍是核心挑战。现有加速方法存在以下问题：

1. **传统扩散的三个问题**：(1) 不同时间步的去噪期望不同，步数压缩必然损失质量；(2) 所有样本收敛到同一标准高斯，路径重叠导致采样随机性大；(3) 临近采样点的结果可能差异巨大，采样不稳定
2. **Rectified Flow 的不足**：虽然用直线 ODE 采样，但路径与实际 ODE 路径差距大，严重限制生成多样性，且缺乏理论最优性证明
3. **蒸馏方法的局限**：计算开销大、训练复杂、难以保持引导能力

本文提出每个样本沿独特路径扩散到实例特定的目标分布 $\mathcal{N}(\epsilon_\mu, \sigma^2 I)$，而非统一的标准高斯。

## 方法详解

### 整体框架

RayFlow 修改扩散过程的目标分布：从标准高斯 $\mathcal{N}(0, I)$ 变为实例特定分布 $\mathcal{N}(\epsilon_\mu, \sigma^2 I)$，其中 $\epsilon_\mu = \mathbb{E}_t[\mathbb{E}[\bar{\epsilon}_t]]$ 是预训练模型的统一噪声期望，$\sigma \to 0$。这使得每个样本的扩散路径不重叠，反向采样更稳定。

### 关键设计

**1. RayFlow 前向/反向过程**

- **功能**：构建每个样本的独特扩散路径，最大化路径概率
- **核心思路**：前向过程 $\psi_t(\cdot|\epsilon) = \sqrt{\bar{\alpha}_t} x_0 + (1-\sqrt{\alpha_t})\epsilon_\mu + \sqrt{1-\bar{\alpha}_t}\epsilon$，在传统 VP 基础上增加了 $(1-\sqrt{\alpha_t})\epsilon_\mu$ 偏移项。理论证明最优参数为 $\epsilon_\mu^* = \mathbb{E}_t[\mathbb{E}[\bar{\epsilon_t}]]$，$\sigma^* \to 0$，即目标分布方差趋近于零
- **设计动机**：让所有时间步共享统一的噪声期望，解决传统扩散中不同步期望不一致的问题。目标分布方差趋零意味着路径几乎确定性，最大化路径概率

**2. Time Sampler 重要性采样**

- **功能**：在训练中自适应选择关键时间步，减少计算冗余
- **核心思路**：最优采样分布 $q^*(t|x_0, \epsilon_\mu) \propto \xi_t(x_0, \epsilon_\mu) p(t)$，其中 $\xi_t$ 衡量模型在时间步 $t$ 的预测误差。用基于 Stochastic Stein Discrepancies (SSD) 的神经网络近似这个最优分布
- **设计动机**：均匀采样时间步导致大量计算浪费在模型已经学好的时间步上。重要性采样聚焦于预测误差大的关键时间步，降低训练损失的方差

**3. 快速一步采样变体**

- **功能**：支持单步生成，实现最快推理
- **核心思路**：由于 RayFlow 中每个样本的路径更加确定（目标分布方差趋零），单步从目标均值 $\hat{\epsilon}_\mu^*$ 直接还原 $x_0$ 成为可能：$x_0 \approx \frac{\hat{\epsilon}_\mu - (1-\sqrt{\bar{\alpha}_T})\epsilon_\mu}{\sqrt{\bar{\alpha}_T}}$
- **设计动机**：路径不重叠 + 统一期望 = 单步采样质量大幅提升

### 损失函数

基于 Flow Matching 框架的条件损失：

$$\mathcal{L}_{CFM} = \mathbb{E}_{t, p(x_t|\epsilon), p(\epsilon)} [\|v_\theta(x_t, t) - u(x_t|\epsilon)\|_2^2]$$

等价于加权的噪声预测损失，权重由信噪比决定。

## 实验关键数据

### 文本到图像生成（SDXL backbone）

| 方法 | FID↓ | 步数 | CLIP Score↑ |
|------|------|------|-----------|
| SDXL (原始) | 23.4 | 50 | 0.32 |
| Rectified Flow | 28.1 | 4 | 0.30 |
| Lightning | 25.6 | 4 | 0.31 |
| **RayFlow** | **22.8** | **4** | **0.32** |
| **RayFlow (1-step)** | **25.1** | **1** | **0.31** |

### 消融实验

| 组件 | FID↓ |
|------|------|
| Baseline (RF) | 28.1 |
| + Instance-aware target | 25.4 |
| + Time Sampler | 23.6 |
| + Full RayFlow | **22.8** |

### 关键发现

- RayFlow 4 步超越原始 SDXL 50 步（FID 22.8 vs 23.4），且保持可控性
- 1 步生成的 FID 仅 25.1，远优于其他加速方法
- Time Sampler 贡献约 ~2 点 FID 提升
- 实例感知路径设计有效避免路径重叠，减少采样随机性

## 亮点与洞察

1. **理论分析充分**：从路径概率最大化推导出最优参数，不是直觉式设计
2. **统一期望的简洁性**：用预训练模型计算 $\epsilon_\mu$，无需额外训练即可获得
3. **Time Sampler 通用性强**：基于 SSD 的重要性采样方法可推广到其他扩散训练

## 局限与展望

- $\epsilon_\mu$ 的计算依赖预训练模型，不同模型的 $\epsilon_\mu$ 不同
- Time Sampler 引入额外的神经网络训练开销
- 「路径概率最大化」的假设在高维空间的实际效果需更多验证

## 相关工作与启发

- **Rectified Flow**：直线采样的先驱，但路径约束过强
- **Consistency Models**：另一种少步生成方法，但需要复杂训练
- **SD-Lightning/Turbo**：蒸馏方法，计算开销大

## 评分

⭐⭐⭐⭐ — 理论推导扎实，实例感知路径设计新颖。在 4 步甚至 1 步生成上取得了出色性能，Time Sampler 也是实用贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] SADA: Stability-guided Adaptive Diffusion Acceleration](../../ICML2025/image_generation/sada_stability-guided_adaptive_diffusion_acceleration.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[CVPR 2025\] ILIAS: Instance-Level Image Retrieval At Scale](ilias_instance-level_image_retrieval_at_scale.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[ECCV 2024\] AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](../../ECCV2024/image_generation/adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)

</div>

<!-- RELATED:END -->
