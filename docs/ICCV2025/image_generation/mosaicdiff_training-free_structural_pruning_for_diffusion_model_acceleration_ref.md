---
title: >-
  [论文解读] MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics
description: >-
  [ICCV 2025][图像生成][结构化剪枝] 本文提出 MosaicDiff，一种免训练的扩散模型结构化剪枝方法，通过将推理过程按预训练学习速度动态分为三个阶段并对各阶段应用不同稀疏度的子网络，实现了在 DiT 和 SDXL 上的显著加速而不牺牲生成质量。 领域现状：扩散模型生成能力出色但计算开销巨大…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "结构化剪枝"
  - "训练免费加速"
  - "预训练动态"
  - "SNR感知"
  - "二阶剪枝"
---

# MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics

**会议**: ICCV 2025  
**arXiv**: [2510.11962](https://arxiv.org/abs/2510.11962)  
**代码**: [https://github.com/bwguo105/MosaicDiff](https://github.com/bwguo105/MosaicDiff)  
**领域**: 扩散模型 / 模型加速  
**关键词**: 结构化剪枝, 训练免费加速, 预训练动态, SNR感知, 二阶剪枝

## 一句话总结

本文提出 MosaicDiff，一种免训练的扩散模型结构化剪枝方法，通过将推理过程按预训练学习速度动态分为三个阶段并对各阶段应用不同稀疏度的子网络，实现了在 DiT 和 SDXL 上的显著加速而不牺牲生成质量。

## 研究背景与动机

**领域现状**：扩散模型生成能力出色但计算开销巨大，社区主要通过减少采样步数（DDIM、DPM-Solver）、知识蒸馏、结构化剪枝、量化和特征缓存等手段加速推理。然而，现有加速方法普遍忽视了扩散模型预训练过程中固有的学习速度差异。

**核心观察**：扩散模型的预训练存在"慢-快-慢"三阶段学习特征——早期阶段模型学习缓慢（主要处理高噪声），中间阶段学习速度骤增（快速捕获粗粒度特征），后期阶段速度再次变慢（精细化细节）。这一关键洞察在现有工作中被完全忽视。

**现有方法局限**：
   - Diff-Pruning 使用单一剪枝模型处理所有时间步，忽略了步级重要性差异
   - EcoDiff 使用可训练掩码但同样缺乏步级自适应性
   - 现有方法在高稀疏度下性能急剧下降

## 方法详解

### 整体框架

MosaicDiff 遵循"Divide → Prune → Conquer"三阶段流程：

1. **Divide**：根据预训练动态的定量分析将推理轨迹分为三个阶段
2. **Prune**：对每个阶段使用 SNR 感知校准数据执行二阶结构化剪枝
3. **Conquer**：合并不同稀疏度的子网络完成最终采样

### 关键设计一：阶段划分与重要性评分

通过监控中间潜在表示 $\hat{x}_t$ 与最终输出 $\hat{x}_0$ 之间的 MSE 变化来刻画学习动态：

$$\text{MSE}(t) = \frac{1}{d}\|\hat{x}_t - \hat{x}_0\|_2^2$$

作者推导了 MSE 期望和梯度的闭式解（Theorem 1）：

$$\mathbb{E}[\text{MSE}(t)] = \frac{1}{d}\left[(1-\sqrt{\bar{\alpha}_t})^2\|\hat{x}_0\|_2^2 + (1-\bar{\alpha}_t)\|\mathbf{I}\|_2^2\right]$$

$$\mathbb{E}[\text{Grad}(t)] = \frac{1}{d}\left[(\delta_t + 2(\sqrt{\bar{\alpha}_{t-1}} - \sqrt{\bar{\alpha}_t}))\|\hat{x}_0\|_2^2 - \delta_t\|\mathbf{I}\|_2^2\right]$$

综合 SNR 信息得到最终重要性评分：

$$score(t) = \mathbb{E}[\text{Grad}(t)] + \lambda \ln \text{SNR}(t)$$

通过阈值 $threshold = M \cdot \max_t(score(t))$ 自动将采样步骤划分为三个阶段。

### 关键设计二：SNR 感知校准数据集

为每个阶段构建专门的校准数据：从标准数据集（如 ImageNet-1K）中选取图像编码为潜在表示，在对应阶段的时间步范围内随机选取 $t$ 并按公式 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 添加噪声，确保校准数据准确反映该阶段的 SNR 特性。对于使用 CFG 的模型，还需提供无条件（null-label）校准样本。

### 关键设计三：二阶结构化剪枝

给定线性层权重 $\mathbf{W} \in \mathbb{R}^{m \times n}$ 和校准输入 $\mathbf{X} \in \mathbb{R}^{b \times n}$，目标是：

$$\arg\min_{\widehat{\mathbf{W}}} \|\mathbf{X}\widehat{\mathbf{W}}^\top - \mathbf{X}\mathbf{W}^\top\|_2^2$$

通过计算 Hessian 矩阵 $\mathbf{H} = \mathbf{X}^\top\mathbf{X}$ 并扩展 OBS 公式获得列级显著性评分：

$$\arg\min_{\mathbf{M}} \sum_{i=0}^{m-1} \mathbf{W}_{i,\mathbf{M}} \cdot (\mathbf{H}_{\mathbf{M},\mathbf{M}}^{-1})^{-1} \cdot \mathbf{W}_{i,\mathbf{M}}^\top$$

剪枝后对剩余权重进行补偿更新 $\delta = -\mathbf{W}_{:,\mathbf{M}} \cdot (\mathbf{H}_{\mathbf{M},\mathbf{M}}^{-1})^{-1} \cdot \mathbf{H}_{\mathbf{M},:}^{-1}$，进一步减小误差。

### 稀疏度分配策略

各阶段稀疏度与平均重要性评分成反比：$s_i \propto 1 - \overline{score}_i$。快速学习阶段（中间阶段）保留更多参数，慢速学习阶段（早期和晚期）允许更激进的剪枝。

## 实验

### 主实验结果

| 方法 | Steps | MACs(T) | 加速比 | IS↑ | FID↓ | Precision↑ |
|------|-------|---------|--------|-----|------|------------|
| Vanilla DiT-XL/2 | 50 | 5.72 | 1.00× | 238.6 | 2.26 | 80.16 |
| Diff-Pruning-0.3 | 50 | 4.10 | 1.29× | 4.68 | 180.76 | 7.24 |
| Learning-to-Cache | 50 | 4.36 | 1.27× | 244.1 | 2.27 | 80.94 |
| **MosaicDiff-0.33** | **50** | **3.92** | **1.32×** | **267.8** | **2.24** | **82.01** |
| Vanilla DiT-XL/2 | 20 | 2.29 | 1.00× | 223.5 | 3.48 | 78.76 |
| **MosaicDiff-0.30** | **20** | **1.64** | **1.28×** | **266.7** | **3.20** | **81.13** |

### SDXL 上的剪枝对比

| 方法 | 稀疏度 | FID↓ | CLIP↑ | SSIM↑ |
|------|--------|------|-------|-------|
| Diff-Pruning | 10% | 108.96 | 0.22 | 0.31 |
| EcoDiff | 10% | 33.75 | 0.31 | 0.53 |
| **MosaicDiff** | **10%** | **23.18** | **0.32** | **0.67** |
| Diff-Pruning | 20% | 404.87 | 0.05 | 0.26 |
| **MosaicDiff** | **20%** | **23.79** | **0.32** | **0.64** |

### 关键发现

- MosaicDiff 在 50 步 DDIM 下以 33% 稀疏度达到 FID 2.24，优于无压缩基线（2.26），同时 MACs 减少 31%
- 在高稀疏度下优势更显著：Diff-Pruning 30% 稀疏度 FID 达 180.76，而 MosaicDiff 仅为 2.24
- 在 SDXL 上同样有效：10% 稀疏度时 FID 改善至 23.18（vs EcoDiff 的 33.75）
- 闭式 MSE/梯度曲线与实验观测高度吻合，验证了理论分析的正确性

## 亮点与洞察

1. **首次将预训练学习动态与后训练加速对齐**：这一视角新颖且有理论支撑，闭式解消除了对实际预训练的依赖
2. **无需训练且无需微调**：利用 Hessian 信息的二阶剪枝无需任何再训练
3. **通用性强**：同时适用于 Transformer (DiT) 和 U-Net (SDXL) 架构
4. **SNR 感知校准**保证了阶段特异性剪枝的精确性

## 局限性

- 阈值 $M$ 和 $\lambda$ 需要超参数调优（虽然作者给出了推荐值）
- 三阶段划分是否对所有噪声调度都最优尚不明确
- Hessian 计算需要一定的校准数据和计算开销

## 相关工作

- **采样加速**: DDIM, DPM-Solver, Consistency Models
- **结构化剪枝**: Diff-Pruning, EcoDiff
- **缓存方法**: DeepCache, Learning-to-Cache
- **训练压缩**: DiP-GO, 知识蒸馏

## 评分

- 新颖性：⭐⭐⭐⭐ — 预训练动态视角新颖
- 技术深度：⭐⭐⭐⭐ — 闭式理论分析有深度
- 实验充分度：⭐⭐⭐⭐ — DiT + SDXL 覆盖全面
- 实用价值：⭐⭐⭐⭐⭐ — 训练免费、即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](../../NeurIPS2025/image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](../../CVPR2026/image_generation/just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)

</div>

<!-- RELATED:END -->
