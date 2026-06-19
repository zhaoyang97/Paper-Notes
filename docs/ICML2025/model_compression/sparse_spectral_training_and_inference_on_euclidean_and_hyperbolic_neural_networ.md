---
title: >-
  [论文解读] Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks
description: >-
  [ICML2025][模型压缩][稀疏谱训练] 提出 Sparse Spectral Training (SST)，通过在谱域上每步更新全部奇异值、按奇异值大小多项式采样选择性更新奇异向量，并周期性 re-SVD 保持正交性，实现接近全秩训练的预训练效果，同时显存开销与 LoRA 相当。 大模型预训练对 GPU 显存需求巨大…
tags:
  - "ICML2025"
  - "模型压缩"
  - "稀疏谱训练"
  - "SVD"
  - "低秩适配"
  - "预训练"
  - "双曲神经网络"
---

# Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks

**会议**: ICML2025  
**arXiv**: [2405.15481](https://arxiv.org/abs/2405.15481)  
**代码**: [GitHub](https://github.com/biomedical-cybernetics/sparse-spectral-training)  
**领域**: 稀疏训练 / 参数高效预训练  
**关键词**: 稀疏谱训练, SVD, 低秩适配, 预训练, 双曲神经网络

## 一句话总结

提出 Sparse Spectral Training (SST)，通过在谱域上每步更新全部奇异值、按奇异值大小多项式采样选择性更新奇异向量，并周期性 re-SVD 保持正交性，实现接近全秩训练的预训练效果，同时显存开销与 LoRA 相当。

## 研究背景与动机

大模型预训练对 GPU 显存需求巨大，现有低秩训练方法存在明显不足：

- **LoRA**：将权重增量限制在固定低秩子空间 $\Delta W = BA$，由 Eckart-Young-Mirsky 定理，低秩近似误差 $\|\Delta W^* - \Delta W\|_F \ge \sqrt{\sigma_{r+1}^2 + \cdots + \sigma_m^2}$，在预训练等复杂任务中 $\sigma_i (i > r)$ 不可忽略，性能受限严重。
- **ReLoRA / COLA / PLoRA**：通过周期性合并低秩矩阵到基础权重来突破秩限制，但每次合并后 $B$ 重置为零导致 $\frac{\partial \mathcal{L}}{\partial A} = \mathbf{0}^T \frac{\partial \mathcal{L}}{\partial W} = \mathbf{0}$，产生**鞍点问题**，收敛缓慢。
- **GaLore**：投影梯度到低秩空间，但投影矩阵仅基于单 batch 梯度的 SVD，低秩较小时不稳定（OPT-350M 训练时 loss 骤增）。

SST 的动机是在谱域同时兼顾**利用 (exploitation)** 已有主导方向和**探索 (exploration)** 新方向，以接近全秩训练的学习动态。

## 方法详解

### 核心框架：稀疏谱层

SST 将每个线性层的权重矩阵 $W \in \mathbb{R}^{m \times n}$ 用 SVD 分解替代：

$$\mathbf{h} = W\mathbf{x} = U \Sigma V^T \mathbf{x}, \quad [U, \Sigma, V^T] = \text{SVD}(W)$$

其中 $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times m}$, $V^T \in \mathbb{R}^{m \times n}$。关键区别：**SST 保留全秩**，原始 $W$ 从网络中移除。

### 更新策略

**1. 全部奇异值每步更新**：$\Sigma$ 简化为 $m$ 维向量，每步都更新，开销极低：

$$\Sigma^{t+1} = \max(\Sigma^t - \eta \nabla \mathcal{L}_\Sigma, 0)$$

$\max$ 函数确保奇异值非负。

**2. 多项式采样选择性更新奇异向量**：对 $U$ 和 $V^T$ 按奇异值大小做多项式采样，每次迭代选 $r$ 个向量更新：

$$S \subseteq \{1, \ldots, m\}, \quad S \sim \text{Multinomial}(r, \Sigma)$$

被选中的向量做梯度下降后归一化，保持单位范数：

$$U_{\cdot i}^{t+1} = \frac{U_{\cdot i}^t - \eta \nabla \mathcal{L}_{U_{\cdot i}}}{|U_{\cdot i}^t - \eta \nabla \mathcal{L}_{U_{\cdot i}}|}, \quad i \in S$$

**3. 增强梯度**：默认梯度 $\nabla \mathcal{L}_{U_{\cdot i}} = \frac{\partial \mathcal{L}}{\partial W} V_{\cdot i} \Sigma_i$ 将方向和幅度耦合。SST 提出增强梯度，去耦合 $\Sigma_i$：

$$\tilde{\nabla} \mathcal{L}_{U_{\cdot i}} = \frac{\partial \mathcal{L}}{\partial W} V_{\cdot i}$$

使小奇异值对应的向量也能获得充分梯度。

**4. 周期性 re-SVD**：训练中 $U, V^T$ 的正交性逐渐丧失，周期性重新执行 SVD 恢复正交性：

$$[U^{t+1}, \Sigma^{t+1}, {V^{t+1}}^T] = \text{SVD}(U^t \Sigma^t {V^t}^T)$$

防止学习退化到低秩子空间。

### 显存高效实现

将 $U$ 和 $V^T$ 分为 active（$m \times r$，存优化器状态）和 frozen 两段。每次采样后新选的向量与 active 段中未被选中的向量交换，类似**分时操作系统**的资源调度。可训练参数比：

$$\Gamma_{\text{SST},r} = \frac{r(m+n)+m}{mn}$$

略高于 LoRA 同秩但低于 LoRA 秩 $r+1$。

## 实验关键数据

### LLM 预训练（OPT / LLaMA，OpenWebText）

| 模型 | r/d | Full | LoRA | ReLoRA* | **SST** |
|------|------|------|------|---------|---------|
| OPT-125M | 64/768 | 23.50 | 34.23 | 35.80 | **26.98** |
| OPT-350M | 64/1024 | 21.78 | 34.26 | 39.21 | **27.72** |
| OPT-1.3B | 64/2048 | 15.10 | 17.16 | 29.52 | **22.31** |
| LLaMA-130M | 64/768 | 20.04 | 29.71 | 31.33 | **23.35** |
| LLaMA-1.3B | 128/2048 | 14.54 | 16.50 | 17.32 | **14.59** |

- LLaMA-1.3B 上 SST 仅用 **18.7% 可训练参数**（秩为嵌入维度 6%），将低秩方法与全秩训练的 perplexity 差距缩小 **97.4%**。

### 机器翻译（IWSLT'14，Euclidean & Hyperbolic Transformer）

| 维度 | r | Full (E) | SST (E) | Full (H) | SST (H) |
|------|---|----------|---------|----------|---------|
| 64 | 8 | 24.27 | 22.28 | 25.69 | 23.40 |
| 128 | 16 | 25.79 | **25.12** | 24.70 | **25.22*** |
| 256 | 32 | 23.92 | **23.97*** | 19.94 | **25.04*** |

- 带 * 表示 SST **超过全秩**。在双曲 Transformer 上 SST 大幅超越全秩训练。
- 机器翻译上 SST 平均缩小 BLEU 差距 **66.7%**。
- 双曲图神经网络：节点分类差距缩小 **73.7%**，链接预测差距缩小 **82.5%**。

### 梯度动态对比

SST 梯度范数与全秩训练的相关性为 **0.85**，ReLoRA* 仅 **0.58**。ReLoRA* 在每次合并后出现周期性鞍点（梯度范数骤降至 0），SST 无此问题。

## 亮点与洞察

1. **谱域利用-探索均衡**：LoRA 只利用固定 top-r 方向（纯 exploitation），ReLoRA* 每次重启丢弃已学方向（纯 exploration），SST 通过奇异值加权采样实现二者平衡。
2. **避免鞍点**：SVD 初始化和 re-SVD 使低秩矩阵始终沿权重矩阵的主方向更新，消除了 ReLoRA* 零初始化导致的鞍点问题。
3. **增强梯度解耦**：将方向学习与幅度学习分离，小奇异值对应的方向也能获得有效梯度。
4. **通用性强**：首次将参数高效预训练引入双曲空间，在欧几里得和双曲神经网络上均有效。
5. **LLaMA-1.3B 上接近全秩**：14.59 vs 14.54 perplexity，差距仅 0.05，实际可用。

## 局限与展望

1. **SVD 开销**：虽然仅在初始化和周期性 re-SVD 时执行，但对超大模型（>10B）可能成为瓶颈。
2. **秩选择敏感**：不同任务和模型对 $r$ 的最优值不同，缺乏理论指导。
3. **仅验证到 1.3B**：未在更大规模模型（7B+）上验证效果。
4. **GaLore 对比不充分**：主实验未包含 GaLore（因其不稳定），对比说服力有限。
5. **采样策略固定**：多项式采样权重直接用奇异值，未探索自适应或任务相关的采样策略。

## 相关工作与启发

- **LoRA / ReLoRA / PiSSA / GaLore**：SST 统一了这些方法的视角——都是在谱域不同策略下的权重更新。
- **Eckart-Young-Mirsky 定理**：提供了低秩近似的理论下界，是分析 LoRA 局限性的关键工具。
- **双曲神经网络 (HyboNet)**：SST 是首个在双曲空间中应用参数高效预训练的工作。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 谱域采样+全奇异值更新+增强梯度的组合新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 LLM 预训练、翻译、图网络和双曲空间，消融充分
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，与已有方法对比分析到位
- 价值: ⭐⭐⭐⭐ — LLaMA-1.3B 上逼近全秩效果，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICML 2025\] Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias](eigenspectrum_analysis_of_neural_networks_without_aspect_ratio_bias.md)
- [\[ICLR 2026\] Cannistraci-Hebb Training on Ultra-Sparse Spiking Neural Networks](../../ICLR2026/model_compression/cannistraci-hebb_training_on_ultra-sparse_spiking_neural_networks.md)
- [\[ICML 2025\] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression](fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)

</div>

<!-- RELATED:END -->
