---
title: >-
  [论文解读] Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation
description: >-
  [ICML2025][ANN-SNN转换] 首次将并行脉冲计算与 ANN-SNN 转换结合，建立数学等价映射关系，在超低时间步（4步）下实现 ImageNet Top-1 72.90%，推理速度加速 19~38 倍。 SNN 的两大主流训练方法各有痛点： - STBP（时空反向传播）：可在极低时延（≤4~6步）下获取 SNN…
tags:
  - "ICML2025"
  - "ANN-SNN转换"
  - "并行脉冲计算"
  - "低时延推理"
  - "量化激活函数"
  - "无训练转换"
---

# Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation

**会议**: ICML2025  
**arXiv**: [2412.13610](https://arxiv.org/abs/2412.13610)  
**代码**: [GitHub](https://github.com/hzc1208/Parallel_Conversion)  
**领域**: SNN (脉冲神经网络)  
**关键词**: ANN-SNN转换, 并行脉冲计算, 低时延推理, 量化激活函数, 无训练转换

## 一句话总结

首次将并行脉冲计算与 ANN-SNN 转换结合，建立数学等价映射关系，在超低时间步（4步）下实现 ImageNet Top-1 72.90%，推理速度加速 19~38 倍。

## 研究背景与动机

SNN 的两大主流训练方法各有痛点：

- **STBP（时空反向传播）**：可在极低时延（≤4~6步）下获取 SNN，但训练开销巨大（速度慢、显存大），难以扩展到大规模网络
- **ANN-SNN 转换**：训练负担小、性能上限高，但转换后的 SNN 需要极高的推理时延才能逼近 ANN 精度；且基于 IF 神经元的串行计算进一步放大了时延问题
- **并行脉冲神经元**：已有工作（Fang et al., NeurIPS 2023）提出并行计算方案，但仅限于 STBP 训练场景，且忽略了先前脉冲序列对当前步的影响（$\lambda^l=1$ 时偏差显著）

本文的核心洞察：并行计算更适合与高时延的转换方法结合，而非受限于 STBP 训练。

## 方法详解

### 1. 并行转换矩阵的构建

核心思想：在 $T$ 步并行推理中，第 $x$ 步判断总脉冲发放数是否 $\geq T-x+1$。

**前提控制矩阵** $\Lambda_{\text{pre}}^l = \frac{1}{T} \cdot \mathbf{1}$：将非均匀输入电流投影为均匀分布。

**后验转换矩阵**：每行的缩放系数为 $c^{l,x} = \frac{T}{x(T-x+1)}$。

两者融合（重参数化）得到最终**并行转换矩阵**：

$$\Lambda_{\text{pc}}^l = \begin{bmatrix} \frac{1}{T} & \frac{1}{T} & \cdots & \frac{1}{T} \\ \frac{1}{T-1} & \frac{1}{T-1} & \cdots & \frac{1}{T-1} \\ \vdots & & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix}$$

### 2. 最优偏移量与无损性证明（Theorem 4.1）

对应 QCFS 函数中的 shift 项 $\psi^l$，推导出逐步最优偏移：

$$\mathbf{b}^l = \left[\frac{\psi^l}{T}, \cdots, \frac{\psi^l}{T-x+1}, \cdots, \psi^l\right]^\top$$

- 当 $T = \tilde{T}$（模拟步数=实际步数）：**无损转换**，$\mathbf{r}^{l,T} = \mathbf{r}_{\text{QCFS}}^{l,\tilde{T}}$
- 当 $T \neq \tilde{T}$ 且 $\psi^l = \theta^l/2$：**期望无损**，$\mathbb{E}(\mathbf{r}^{l,T} - \mathbf{r}_{\text{QCFS}}^{l,\tilde{T}}) = \mathbf{0}$

### 3. 分布感知误差校准（DA-QCFS）

针对实际数据分布非均匀、通道间分布差异大的问题，引入逐通道可学习参数 $\psi_{\text{DA}}^l, \phi_{\text{DA}}^l \in \mathbb{R}^C$：

$$\mathbf{r}_{\text{DA}}^{l,\tilde{T}} = \frac{\theta^l + \phi_{\text{DA}}^l}{\tilde{T}} \text{Clip}\left(\left\lfloor \frac{(\mathbf{W}^l \mathbf{r}^{(l-1),\tilde{T}} + \psi_{\text{DA}}^l)\tilde{T} + \psi^l}{\theta^l} \right\rfloor, 0, \tilde{T}\right)$$

采用贪心思想逐层校准：先算通道均值误差 $\mathbf{e}_{\text{pre}}^l$、$\mathbf{e}_{\text{post}}^l$，用动量 $\alpha$ 更新参数。

### 4. 训练无关（Training-Free）转换三阶段

1. **ReLU → ClipReLU**：记录每层各通道的历史最大激活值作为 $\theta^l$
2. **ClipReLU → DA-QCFS**：用校准数据集做逐层误差校准
3. **DA-QCFS → 并行脉冲神经元**：合并偏移项到 bias，设置 pre/post 双阈值实现等价映射

### 5. 排序性质与二分搜索加速

由于并行推理的脉冲序列具有**排序性**（若第 $x$ 步发放脉冲，则第 $x+1$ 到 $T$ 步也一定发放），可用二分搜索在 $O(\log T)$ 内找到首次发放时刻 $t_{\text{fir}}$。结合 Hadamard 乘积优化，充电阶段复杂度从 $O(T^2)$ 降至 $O(T)$。

## 实验关键数据

### 与 SOTA 对比（QCFS 预训练 ANN）

| 数据集 | 方法 | 网络 | 时间步 T | SNN 精度 |
|---|---|---|---|---|
| CIFAR-10 | QCFS | VGG-16 | 4 | 93.96% |
| CIFAR-10 | **Ours** | VGG-16 | **4** | **95.50%** |
| CIFAR-100 | QCFS | ResNet-20 | 8 | 55.37% |
| CIFAR-100 | **Ours** | ResNet-20 | **8** | **69.62%** |
| ImageNet | QCFS | VGG-16 | 16 | 50.97% |
| ImageNet | **Ours** | VGG-16 | **8** | **73.92%** |
| ImageNet | COS | ResNet-34 | 10 | 72.66% |
| ImageNet | **Ours†** | ResNet-34 | **4** | **72.90%** |

### Training-Free 转换（ImageNet）

| 方法 | 网络 | T=16 | T=32 | T=64 |
|---|---|---|---|---|
| TBC | ResNet-34 | — | 59.03% | 70.47% |
| **Ours** | ResNet-34 | **68.04%** | **72.46%** | **73.03%** |
| **Ours** | ResNet-101 | 73.86% | 76.42% | 77.01% |

### 推理速度

并行推理相比串行 IF 神经元实现 **19~38 倍加速**（$T \geq 32$）。

## 亮点与洞察

1. **首创性结合**：首次将并行脉冲计算引入 ANN-SNN 转换，开辟 SNN 监督学习的"第三条路径"
2. **理论严谨**：证明了无损转换性质、排序性质和最优偏移量，不是经验性方法
3. **统一框架**：QCFS（$\tilde{T}=T$ 或 $\tilde{T}\neq T$）和 ReLU 三种场景用同一套框架，区别仅在是否需要阈值记录和误差校准
4. **实用加速**：二分搜索 + Hadamard 乘积优化使推理开销从 $O(T^2)$ 降至 $O(T)$ 充电 + $O(\log T)$ 发放
5. **4 步超越 STBP**：ImageNet ResNet-34 仅用 4 步就达到 72.90%，超过 6 步 STBP 方法（Dspike 68.19%）

## 局限与展望

1. **仅验证分类任务**：所有实验限于图像分类（CIFAR/ImageNet），未在检测、分割等下游任务验证
2. **网络结构限制**：仅测试 VGG 和 ResNet，未涉及 Transformer、MobileNet 等现代架构
3. **并行计算的硬件适配**：并行转换矩阵的计算模式与现有神经形态芯片（如 Loihi）的串行架构不完全匹配，实际部署的能效优势待验证
4. **校准数据依赖**：DA-QCFS 和 Training-Free 转换均需校准数据集，完全零样本场景未覆盖
5. **排序性质的局限**：二分搜索优化依赖排序性质，若扩展到 LIF（$\lambda < 1$）等更一般的神经元模型，该性质可能不成立

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首次将并行脉冲计算与转换结合，理论新颖）
- 实验充分度: ⭐⭐⭐⭐（多数据集多网络多场景，但缺下游任务和现代架构）
- 写作质量: ⭐⭐⭐⭐（理论推导清晰，符号体系完整）
- 价值: ⭐⭐⭐⭐⭐（为 SNN 高效部署提供全新范式，具有重要实践意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[CVPR 2026\] When AVSR Meets Video Conferencing: Dataset, Degradation, and the Hidden Mechanism Behind Performance Collapse](../../CVPR2026/others/when_avsr_meets_video_conferencing_dataset_degradation_and_the_hidden_mechanism_.md)
- [\[CVPR 2025\] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](../../CVPR2025/others/towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](../../ACL2025/others/attention_entropy_parallel_encoding.md)

</div>

<!-- RELATED:END -->
