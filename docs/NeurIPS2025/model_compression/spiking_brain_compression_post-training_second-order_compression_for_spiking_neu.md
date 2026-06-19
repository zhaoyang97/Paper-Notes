---
title: >-
  [论文解读] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks
description: >-
  [NeurIPS 2025][模型压缩][脉冲神经网络] 提出 Spiking Brain Compression（SBC），一种基于 Van Rossum 距离的二阶后训练一次性压缩框架，专为脉冲神经网络（SNN）设计，通过替代膜电位（SMP）Hessian 实现高效的模块级剪枝和量化，在 ImageNet 规模下首次压缩 SEW-ResNet152 和 Spike-Driven Transformer。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "脉冲神经网络"
  - "后训练压缩"
  - "Hessian矩阵"
  - "非结构化剪枝"
  - "量化"
---

# Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2506.03996](https://arxiv.org/abs/2506.03996)  
**代码**: 暂无  
**领域**: 模型压缩  
**关键词**: 脉冲神经网络, 后训练压缩, Hessian矩阵, 非结构化剪枝, 量化

## 一句话总结

提出 Spiking Brain Compression（SBC），一种基于 Van Rossum 距离的二阶后训练一次性压缩框架，专为脉冲神经网络（SNN）设计，通过替代膜电位（SMP）Hessian 实现高效的模块级剪枝和量化，在 ImageNet 规模下首次压缩 SEW-ResNet152 和 Spike-Driven Transformer。

## 研究背景与动机

脉冲神经网络（SNN）以离散脉冲进行通信，天然适合部署在 TrueNorth、Loihi、SpiNNaker 等神经形态芯片上。但这些芯片的内存和算力有限，需要对 SNN 进行压缩。

**现有方法的痛点**：

**训练中剪枝（如 LTH、UPF、STDS）**需要多轮迭代的压缩-训练循环，对大型预训练 SNN（如 SEW-ResNet152、Spiking Transformer）计算成本过高。

**ANN 后训练压缩方法直接移植到 SNN 效果差**：Optimal Brain Compression（OBC）等方法的损失函数基于电流域（current-based），而 SNN 的实际输出是脉冲序列（spike train），两者存在根本性的"目标函数断裂"。

**SNN 的量化后训练（PTQ）发展滞后**，不像 ANN 领域已有 GPTQ 等成熟方法。

核心动机：需要一种**一次性后训练**压缩方法，既利用二阶优化的精度优势，又正确反映 SNN 的脉冲动态特性。

## 方法详解

### 整体框架

SBC 在模块级（Module-wise）工作：每个模块包含 Linear/Conv(+BN) → LIF 层。将 BatchNorm 和 Conv 折叠为单一线性映射后，每个模块变为 Linear(W) → LIF。对输入脉冲张量 $X \in \{0,1\}^{T \times d_{\text{in}}}$，模块产生输出脉冲序列 $S = f(X, W) \in \{0,1\}^{T \times d_{\text{out}}}$。压缩目标：

$$\arg\min_{\hat{W}} E_X[\mathcal{L}(f(X,W), f(X,\hat{W}))], \text{ s.t. } \mathcal{C}(\hat{W}) > C$$

### 关键设计

1. **Van Rossum 距离（VRD）损失函数**: 由于脉冲序列是离散的 0/1 信号，简单的 L2 范数忽略了时间维度上的脉冲距离。SBC 使用 VRD 的平方作为损失：$\mathcal{L}(W) = \|MS - M\hat{S}\|_2^2$，其中 $M$ 是由衰减核 $k[t] = (1 - 1/\tau_m)^t \cdot (1/\tau_m)$ 产生的卷积矩阵。关键性质：VRD 可按每个 LIF 神经元独立分解 $\|MS - M\hat{S}\|_2^2 = \sum_{j=1}^{d_{\text{out}}} \|MS_{:,j} - M\hat{S}_{:,j}\|_2^2$，使 Hessian 的计算可以按神经元并行化。

2. **替代膜电位（SMP）Hessian**: 脉冲函数 $S[t] = \Theta(U[t] - V_{th})$ 不可微，受代理梯度（surrogate gradient）启发，用常数函数 $g(u) = c$ 替代 Heaviside 函数的导数。由于 OBS 框架只关注 Hessian 的相对大小，$c$ 可以约掉（取 $c=1$）。此时二阶项 $h'' = g' = 0$，得到 SMP Hessian：
    $\mathbf{H}_{\text{SMP}} = E_X[2(MX)^T MX]$
   这恰好是无脉冲情况下膜电位最小二乘损失 $\|MXw - MX\hat{w}\|_2^2$ 的精确 Hessian。相比 OBC 的 $\mathbf{H}_{\text{OBC}} = E_X[2X^TX]$，SMP Hessian 多了卷积矩阵 $M$，更好地捕捉了脉冲时间动态。

3. **SBC 剪枝算法**: 

    - **自适应稀疏度分配**：用 LAMPS 从全局目标稀疏度确定每模块的剪枝比例。
    - **权重排序**（Step 1）：对每个神经元的权重执行 OBS，按损失从小到大记录剪枝顺序，每次批量剪 $B_{\text{in}}$ 个权重并用 Woodbury 公式更新逆 Hessian。时间 $O(d_{\text{in}}^3 / B_{\text{in}})$。
    - **权重剪枝**（Step 2）：根据排序结果生成掩码，用分组 OBS 公式一次性更新剩余权重：$\delta_{\mathbb{P}} = -\mathbf{H}^{-1}_{:,\mathbb{P}}((\mathbf{H}^{-1})_{\mathbb{P}})^{-1}\mathbf{W}_{:,i}$。

### 损失函数 / 训练策略

- SBC 是**无需训练**的一次性后训练方法，仅需少量校准数据估计 Hessian。
- 量化算法沿用 GPTQ 框架，将层级 Hessian 替换为 $\mathbf{H}_{\text{SMP}}$，支持对称均匀量化。
- 校准集消融显示：仅 1-10 样本/类 即可获得稳定性能（标准差 ≤0.5%）。

## 实验关键数据

### 主实验 — 一次性后训练剪枝（与 ExactOBS、MBP 对比）

| 数据集/模型 | 稀疏度 | SBC 精度损失 | ExactOBS 精度损失 | MBP 精度损失 |
|------------|--------|------------|-----------------|-------------|
| N-MNIST / 2FC | 97% | -1.59% | -45.07% | 更差 |
| DVS128-Gesture / 5Conv2FC | 97% | -1.74% | -9.38% | 更差 |
| CIFAR-100 / VGG16-SNN | 75% | — | SBC 优 +7.47% | — |
| ImageNet / SEW-ResNet152* | 多级 | SOTA | 不可及 | 不可及 |
| ImageNet / Spike-Driven Transformer* | 多级 | SOTA | 不可及 | 不可及 |

（*首次被压缩的大型 SNN 架构）

### 量化实验

| 数据集/架构 | 位宽 | SBC | ExactOBS | RTN |
|------------|------|-----|----------|-----|
| N-MNIST / 2FC | 4-bit | 98.14% | 98.16% | 97.58% |
| N-MNIST / 2FC | 2-bit | **92.40%** | 64.29% | 20.34% |
| CIFAR10-DVS / 4Conv2FC | 3-bit | **69.64%** | 67.56% | 52.70% |
| DVS128-Gesture / 5Conv2FC | 2-bit | **64.79%** | 62.78% | 53.82% |

### 与迭代剪枝方法对比

| 数据集/方法 | 稀疏度 | 精度 | 剪枝时间 |
|------------|--------|------|---------|
| CIFAR-100 / LTH-IMP | 68.30% | 基线 | ×100 |
| CIFAR-100 / SBC | 68.30% | +2.02% | **1×** |
| ImageNet / UPF | — | 基线 | ×1000 |
| ImageNet / SBC+FT | — | ≈UPF | **1×** |

### 消融实验 — 校准集大小

| 数据集/模型 | 校准样本数 | 精度 |
|------------|----------|------|
| CIFAR10-DVS / 4Conv2FC | 10 | 49.64% (±1.27) |
| CIFAR10-DVS / 4Conv2FC | 90 | 61.90% (±0.21) |
| CIFAR10-DVS / 4Conv2FC | 900 | 63.84% (±0.49) |
| ImageNet / SEW-ResNet18 | 100 (0.1/类) | 49.35% (±0.19) |
| ImageNet / SEW-ResNet18 | 1000 | 55.06% (±0.12) |
| ImageNet / SEW-ResNet18 | 50000 | 55.89% (±0.04) |

### 关键发现

- 在 2-bit 极端量化下，SBC 比 ExactOBS 高出 **28.11 个百分点**（N-MNIST），说明 SMP Hessian 在低精度场景优势巨大。
- SEW-ResNet152 是迄今为止被成功剪枝的最大最深的 SNN 模型，SBC 是唯一能做到的方法。
- 校准集可以极小（<1 样本/类），这对数据稀缺的神经形态应用场景非常有价值。
- 每层独立并行的模块化设计使得 SBC 的空间复杂度仅为 $O(B_{\text{out}} \cdot d_{\text{in}}^2)$。

## 亮点与洞察

- **VRD 损失的引入恰当且优雅**：将计算神经科学中度量脉冲序列相似性的经典工具引入模型压缩，完美弥合了 SNN 脉冲输出与 ANN 连续输出之间的鸿沟。
- **SMP 的"常数代理梯度"简化令人意外地有效**：看似粗糙的 $g(u) = 1$ 近似，却因 OBS 只关注 Hessian 相对大小而成立。
- **首次压缩大型 SNN**的里程碑意义：为未来大规模 SNN 的部署开辟了道路。

## 局限与展望

- 量化中存在权重补偿导致的越界问题（后量化权重被推到网格范围外），需要研究更好的量化网格选择。
- 目前只使用了最简单的常数代理梯度，更精细的代理梯度函数可能进一步提升性能。
- 自定义 Hessian 的空间复杂度 $\Theta(d_{\text{out}} \cdot d_{\text{in}}^2)$ 在 SEW-ResNet152 的大层上可达 43.5GB，限制了精细化的空间。
- 量化实验仅在小型神经形态数据集上进行，未扩展到 ImageNet 规模。

## 相关工作与启发

- SBC 将 OBC/GPTQ 框架从 ANN 推广到 SNN，核心创新在于损失函数和 Hessian 的 SNN 化。
- 与 UPF（训练时剪枝）互补：SBC 适合已有大型预训练 SNN 的场景，UPF 更适合训练中渐进压缩。
- 对 ANN 压缩也有启示：当输出不是连续值而是某种离散/结构化信号时，需要设计匹配的压缩目标。

## 评分

- 新颖性: ⭐⭐⭐⭐ VRD 损失和 SMP Hessian 是针对 SNN 特性的精巧设计
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 7 个数据集、多种架构（CNN+Transformer）、剪枝+量化，有校准集消融
- 写作质量: ⭐⭐⭐⭐ 推导严谨，实验全面，但部分公式符号较密集
- 价值: ⭐⭐⭐⭐⭐ 填补 SNN 后训练压缩的空白，首次压缩大型 SNN，对神经形态计算社区有即时价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[NeurIPS 2025\] S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection](s2m-former_spiking_symmetric_mixing_branchformer_for_brain_auditory_attention_de.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](../../ICML2025/model_compression/efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](../../AAAI2026/model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)

</div>

<!-- RELATED:END -->
