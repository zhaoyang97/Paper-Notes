---
title: >-
  [论文解读] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model
description: >-
  [CVPR 2025][图像生成][失真-感知权衡] 本文提出方差缩放反向扩散过程，通过一个参数 $\lambda \in [0,1]$ 控制反向采样的方差大小，从而用单个预训练 score 网络灵活遍历 distortion-perception tradeoff 曲线的最优解，并在条件高斯分布下证明了其最优性。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "失真-感知权衡"
  - "扩散模型"
  - "逆问题"
  - "方差缩放"
  - "MMSE估计"
---

# Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model

**会议**: CVPR 2025  
**arXiv**: [2503.20297](https://arxiv.org/abs/2503.20297)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 失真-感知权衡, 扩散模型, 逆问题, 方差缩放, MMSE估计

## 一句话总结

本文提出方差缩放反向扩散过程，通过一个参数 $\lambda \in [0,1]$ 控制反向采样的方差大小，从而用单个预训练 score 网络灵活遍历 distortion-perception tradeoff 曲线的最优解，并在条件高斯分布下证明了其最优性。

## 研究背景与动机

**领域现状**：图像复原领域存在一个根本性的冲突——失真（distortion，如 MSE/PSNR）和感知质量（perception，即图像看起来是否自然）之间存在不可调和的 tradeoff。现有算法要么优化 MSE（如 regression 方法），要么追求感知质量（如 GAN），分别位于 DP 平面的两个极端。

**现有痛点**：(1) 基于 CGAN 的方法（PSCGAN）可以通过多样本平均或调节注入噪声来遍历 tradeoff，但训练依赖成对数据，不同噪声水平需要重新训练；(2) 线性组合两个极端估计器的方法需要部署两个模型，灵活性差；(3) 两种方法都无法证明达到最优 DP 曲线。

**核心矛盾**：现有方法要么不灵活（需要重训练），要么不最优（无法保证在 DP 曲线上），且都无法用单个模型适用于不同类型的逆问题。

**本文目标**：用单个预训练的 score-based 扩散模型，在推理时通过调节单个参数 $\lambda$，灵活且最优地遍历从 MMSE 点到完美感知点的整个 DP tradeoff 曲线。

**切入角度**：score-based 扩散模型的反向过程有两个关键性质——(1) 当反向方差设为 0 时，均值传播收敛到 MMSE 估计；(2) 当使用真实后验方差时，采样来自后验分布（完美感知）。这两点恰好对应 DP tradeoff 的两个极端。

**核心 idea**：通过缩放反向过程的方差 $\lambda \mathbf{C}_k$（$\lambda=0$ 对应 MMSE，$\lambda=1$ 对应后验采样），连续地在两个极端之间插值，并证明在条件高斯分布下这恰好是 DP tradeoff 的最优解。

## 方法详解

### 整体框架

方法的核心是修改标准的条件反向扩散过程。给定预训练的 score 网络 $s_\theta(\mathbf{x}_k, k)$ 和观测 $\mathbf{y}$，在每一步反向采样中使用缩放后的方差 $\lambda \mathbf{C}_k$ 替代原始方差。用户只需选择 $\lambda \in [0,1]$：$\lambda=0$ 得到最小失真（MMSE），$\lambda=1$ 得到最佳感知质量。整个过程使用同一个 score 网络，不需要任何重训练。

### 关键设计

1. **方差缩放反向扩散过程**:

    - 功能：通过缩放反向过程中每一步的方差来控制采样的随机性，从而调节 distortion 和 perception 之间的平衡
    - 核心思路：定义缩放后的联合推断分布 $p_\lambda(\mathbf{x}_k|\mathbf{x}_{k+1}, \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_k(\mathbf{x}_{k+1}, \mathbf{y}), \lambda \mathbf{C}_k)$，其中均值 $\boldsymbol{\mu}_k$ 保持不变（仍由条件 score 决定），仅缩放方差。当 $\bar{\alpha}_T \to 0$ 时，最终边际分布的均值收敛到 MMSE 估计 $\mathbb{E}[X_0|\mathbf{y}]$，方差收敛到 $\lambda \cdot \text{Cov}[X_0|\mathbf{y}]$
    - 设计动机：这一设计的优雅之处在于保持了均值不变——无论 $\lambda$ 取何值，均值始终趋向 MMSE，只是采样的"散布度"不同。这确保了重建结果不会偏离最优均值估计

2. **条件高斯最优性证明 (Theorem 2)**:

    - 功能：证明方差缩放采样在条件高斯分布下达到了 DP tradeoff 的理论最优
    - 核心思路：对于条件高斯分布 $p_{X|Y}(\mathbf{x}|\check{\mathbf{y}}) \sim \mathcal{N}(\boldsymbol{\mu}_{\check{\mathbf{y}}}, \boldsymbol{\Sigma}_{\check{\mathbf{y}}})$，使用 MSE 和 Wasserstein-2 距离度量时，最优 DP 函数为 $D(P) = \text{Tr}(\boldsymbol{\Sigma}) + (\sqrt{\text{Tr}(\boldsymbol{\Sigma})} - P)^2$（当 $P \leq \sqrt{\text{Tr}(\boldsymbol{\Sigma})}$）。完美感知（$P=0$）时最佳 MSE 恰好是 MMSE 的两倍
    - 设计动机：理论保证给出了方差缩放方法的坚实基础，同时量化了感知质量改善的代价——追求完美感知至多使 MSE 翻倍

3. **基于 DPS 的条件 Score 近似**:

    - 功能：实现方差缩放反向采样的实际计算
    - 核心思路：采用 Denoising Posterior Sampling (DPS) 框架近似条件 score：$\nabla_{\mathbf{x}_k} \log p(\mathbf{x}_k|\mathbf{y}) \approx s_\theta(\mathbf{x}_k, k) + \frac{1}{\sigma_n^2} \nabla_{\mathbf{x}_k} \|\mathbf{y} - \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_k))\|_2^2$。使用 Tweedie 公式估计 $\hat{\mathbf{x}}_0$。引入超参数 $\zeta_{k,\lambda}$ 控制条件 score 的权重
    - 设计动机：DPS 框架天然适合本文需求——它只需要一个预训练的无条件 score 网络就能处理任意测量算子 $\mathcal{A}$，无需为不同逆问题重新训练

### 损失函数 / 训练策略

本方法不涉及额外训练。使用现成的预训练 score 网络。推理时只需在 Algorithm 1 中设置 $\lambda$ 值：每步执行 $\mathbf{x}_{k-1} = \frac{1}{\sqrt{\alpha_k}}(\mathbf{x}_k + (1-\alpha_k)(\hat{s} + \zeta_{k,\lambda} \hat{c}(\hat{\mathbf{x}}_0))) + \lambda \tilde{\sigma}_k \mathbf{z}$，其中 $\lambda$ 是唯一需要调节的参数。

## 实验关键数据

### 主实验

混合高斯分布实验：
- $\lambda=0$：轨迹确定性收敛到 MMSE 点
- $\lambda$ 增大：重建分布逐渐趋近真实后验
- $\lambda=1$：重建分布与真实后验匹配，MSE 约为 $\lambda=0$ 时的两倍（符合理论）

FFHQ 256×256 人脸数据集（高斯模糊 + 超分辨率）：

| 方法 | 灵活性 | 是否需重训练 | DP 曲线覆盖度 |
|------|--------|-----------|-------------|
| PSCGAN-N | 有限 | 是(每个噪声级别) | 部分 |
| PSCGAN-z | 有限 | 是 | 部分 |
| DiffPIR | 固定点 | 否 | 单点 |
| **本文方法** | **完全灵活** | **否** | **完整曲线** |

### 消融实验

2D 数据集（Pinwheel, S-curve, Moon）上的 DP tradeoff：

| $\lambda$ | MSE 趋势 | Wasserstein-2 趋势 | 行为 |
|-----------|---------|------------------|------|
| 0 | 最小 | 最大 | 确定性，收敛到 MMSE |
| 0.3 | 略增 | 显著降低 | 轻微随机性 |
| 0.8 | 中等增 | 接近零 | 较强随机性 |
| 1.0 | ~2×MMSE | 零 | 完美感知，匹配后验 |

### 关键发现

- 方差缩放方法在所有 2D 数据集上实现了比 PSCGAN 更大范围的 DP tradeoff 覆盖
- 在 FFHQ 上，单个 score 网络可以同时处理高斯模糊和超分辨率两种逆问题（不同测量算子），而 PSCGAN 每种问题都需要单独训练
- $\lambda=1$ 时的 MSE 约为 $\lambda=0$ 时的 2 倍，实验结果与理论预测高度一致
- 超参数 $\zeta_{k,\lambda}$ 对不同 $\lambda$ 可能需要调整，但整体方法对其不太敏感

## 亮点与洞察

- **理论与实践的优美统一**是本文最大亮点。Theorem 1 和 Theorem 2 提供了坚实的理论基础，同时 Algorithm 1 的实际修改极其简单——只需在标准反向采样中缩放噪声的标准差
- **"2 倍 MMSE"法则**是一个重要的理论洞察——追求完美感知质量的代价至多是 MSE 翻倍，这为实际应用中的 tradeoff 决策提供了量化指导
- **单模型多任务**的灵活性极具实用价值：同一个 score 网络 + 不同的 $\lambda$ + 不同的测量算子 = 全面覆盖各种逆问题在各种 DP 权衡下的解

## 局限与展望

- 理论最优性仅在条件高斯分布下严格成立，对于真实图像分布（非高斯）只是近似最优
- DPS 框架中条件 score 的近似（Jensen gap）在高噪声环境下可能不够精确
- 超参数 $\zeta_{k,\lambda}$ 需要启发式调节，缺少理论指导
- 采样过程需要完整的 $T$ 步反向扩散，计算成本较高。DiffPIR 等加速方法可用更少步数完成，但无法灵活遍历 DP tradeoff
- 仅验证了高斯模糊和超分辨率两种逆问题，对修复、压缩等其他任务的效果还需验证

## 相关工作与启发

- **vs PSCGAN**: PSCGAN 通过多样本平均或调节生成器噪声来遍历 DP tradeoff，但需要成对训练数据且每种噪声级别需重训练。本文方法完全在推理时控制，且理论证明最优
- **vs [Ohayon 2023]**: 该工作在 Wasserstein 空间中理论分析 DP tradeoff，提出线性组合两个极端估计器。但需要分别训练/部署两个模型。本文用单模型通过方差缩放自然实现连续遍历
- **vs DiffPIR**: DiffPIR 基于 DDIM 加速采样，在 DP 平面上只能达到固定的一个点。本文方法可以通过 $\lambda$ 达到曲线上任意一点

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从理论上证明单个扩散模型可以最优遍历 DP tradeoff
- 实验充分度: ⭐⭐⭐⭐ 理论验证充分，但真实图像实验仅在 FFHQ 256×256 上
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，但论文偏理论导向，实验部分可更丰富
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型在逆问题中的应用提供了统一的理论框架和实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](../../ICML2026/image_generation/stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[CVPR 2025\] CustAny: Customizing Anything from A Single Example](custany_customizing_anything_from_a_single_example.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)

</div>

<!-- RELATED:END -->
