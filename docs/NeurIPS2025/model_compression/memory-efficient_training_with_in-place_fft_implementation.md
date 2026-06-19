---
title: >-
  [论文解读] Memory-Efficient Training with In-Place FFT Implementation
description: >-
  [NeurIPS 2025][模型压缩][FFT] 提出 rdFFT——首个真正原地（in-place）的实数域快速傅里叶变换框架，通过隐式复数编码方案消除中间缓冲区，实现训练时零额外内存开销的 FFT/IFFT 计算，内存效率最高提升 1500 倍以上。 领域现状： FFT 广泛用于深度学习中降低计算和内存开销…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "FFT"
  - "原地计算"
  - "内存优化"
  - "循环矩阵"
  - "参数高效微调"
---

# Memory-Efficient Training with In-Place FFT Implementation

**会议**: NeurIPS 2025  
**arXiv**: [2511.01385](https://arxiv.org/abs/2511.01385)  
**代码**: [PyTorch Issue #171022](https://github.com/pytorch/pytorch/issues/171022) (讨论上游合并中)  
**领域**: Model Compression / 高效训练  
**关键词**: FFT, 原地计算, 内存优化, 循环矩阵, 参数高效微调

## 一句话总结

提出 rdFFT——首个真正原地（in-place）的实数域快速傅里叶变换框架，通过隐式复数编码方案消除中间缓冲区，实现训练时零额外内存开销的 FFT/IFFT 计算，内存效率最高提升 1500 倍以上。

## 研究背景与动机

**领域现状**: FFT 广泛用于深度学习中降低计算和内存开销，如频域微调方法 FourierFT 和 Block Circulant Adapter (BCA) 等都依赖 FFT 进行频域参数变换。

**现有痛点**: 标准 FFT 将 $N$ 个实数输入映射为 $N$ 个复数输出（占 $2N$ 实数空间），rFFT 优化后仍需 $N+2$ 实数空间，与输入的 $N$ 实数空间不匹配，无法实现真正原地计算。

**核心矛盾**: 输入输出的维度/内存不一致导致必须分配额外内存缓冲区，这在大规模模型训练中产生显著内存开销，且现有库（FFTW、cuFFT）不支持 bfloat16 数据类型。

**本文目标**: 设计一种输入输出内存空间完全一致的实数域 FFT，使前向和反向传播都能在原始实数缓冲区内完成，消除中间张量的内存分配。

**切入角度**: 利用蝴蝶运算的对称性和实值信号频谱的共轭性质，将 $N+2$ 压缩到 $N$ 个实数空间。

**核心idea**: 通过巧妙的内存布局设计（将 $y_k$ 的实部存于索引 $k$，虚部存于共轭对称索引 $r-k$），使整个 FFT 计算完全在 $N$ 个实数空间内原地完成。

## 方法详解

### 整体框架

rdFFT 基于 Cooley-Tukey 算法，在递归分解的每个阶段保持共轭对称性，通过新的内存布局和蝴蝶运算方案实现完全原地的 FFT 和 IFFT 计算。

### 关键设计

1. **将 $N+2$ 压缩到 $N$ (Squeeze)**: 对于 $r$ 点 FFT 的实值输入，输出满足 $y_0, y_{r/2} \in \mathbb{R}$（始终为实数），其余 $r-2$ 个复数值中只需存储 $y_1, \dots, y_{r/2-1}$ 的实部和虚部，其余可通过共轭重构。→ 这是因为共轭对称性 $y_{r-k} = \overline{y_k}$。→ 实现方式：$y_0$ 存于索引 0，$y_{r/2}$ 存于索引 $r/2$，$y_k$ 的实部存于索引 $k$，虚部存于索引 $r-k$。→ 与标准 rFFT 的区别：不需要额外 2 个实数空间。

2. **四元素对称群 (Proposition 1)**: 在 Cooley-Tukey 递归中，每级的共轭对称对和其蝴蝶配对形成一个四元素对称群 $\{a_1, b_1, a_2, b_2\}$，关于中心索引 $c$ 对称。→ 这保证了蝴蝶运算在每级都能保持共轭对称性并完全原地执行。→ 例如：8-FFT 中索引 (2,6) 的共轭对在 16-FFT 中与 (10,14) 配对，结果对 (2,14) 和 (6,10) 仍然对称。

3. **原地 IFFT 设计**: IFFT 接收共轭对称复数输入，子 IFFT 输出不保证为实值或共轭对称。→ 利用 FFT 的线性性，反转前向 FFT 的计算图结构，在蝴蝶图反向方向上恢复实值信号。→ 关键公式：$x_2 = \frac{y_2 + y_{10}}{2}$，$x_{10} = \frac{y_2 - y_{10}}{2W_N^2}$，将每个共轭对分解为对称和反对称分量。

4. **循环矩阵训练适配**: 循环矩阵 $\mathbf{y} = \text{IFFT}(\text{FFT}(\mathbf{c}) \odot \text{FFT}(\mathbf{x}))$ 中逐元素乘法保持对称性（因为 $\overline{A \cdot B} = \overline{A} \cdot \overline{B}$），所以 IFFT 输入始终满足对称复数条件。→ 梯度计算同理满足对称性。→ 支持 bfloat16 数据类型，兼容现代神经网络训练。

### 损失函数 / 训练策略

- 通过循环矩阵结构替换线性层权重矩阵
- 前向传播和反向传播全部在实数域原地执行
- 支持自动微分框架集成

## 实验关键数据

### 主实验：单层内存效率（D=4096, B=256）

| 方法 | 峰值内存 (MB) | 相对全微调压缩倍数 |
|------|:-:|:-:|
| Full Fine-tuning | 164.25 | 1× |
| LoRA (r=64) | 39.38 | 4.17× |
| FFT (p=512) | 164.50 | 1.00× |
| rFFT (p=512) | 156.66 | 1.05× |
| **rdFFT (p=512)** | **20.13** | **8.16×** |
| FFT (p=4096) | 52.05 | 3.16× |
| rFFT (p=4096) | 44.04 | 3.73× |
| **rdFFT (p=4096)** | **20.02** | **8.21×** |

### 全模型训练内存（LLaMA2-7B）

| 方法 | 峰值内存 (GB) |
|------|:-:|
| Full Fine-tuning | 26.90 |
| LoRA (r=32) | 18.96 |
| FFT (p=1024) | 19.20 |
| rFFT (p=1024) | 19.17 |
| **rdFFT (p=1024)** | **17.92** |
| **rdFFT (p=4096)** | **17.91** |

### 消融实验：运算精度

| 方法 | 绝对误差 | 相对误差 |
|------|:-:|:-:|
| rFFT (p=1024) | 1.92e-07 | 0.0001 |
| rdFFT (p=1024) | 5.75e-07 | 0.0005 |

### 关键发现

- rdFFT 在单层训练中极端情况下（D=1024, B=1, p=1024）实现 **1014 倍**内存压缩。
- 大 batch size 下标准 FFT/rFFT 的中间张量开销急剧增加，而 rdFFT 保持恒定。
- 数值误差在浮点噪声级别，数学等价性完全保持。
- 在 LLaMA2-7B 全模型训练中比 LoRA 节省约 1GB 内存。
- 原生支持 bfloat16，而标准 FFT/rFFT 实现不支持。

## 亮点与洞察

- 从算子级别（arithmetic operator level）解决内存问题，思路独特且优雅。
- 四元素对称群的数学证明精妙，将共轭对称性的递归保持严格化。
- 正在讨论上游合并到 PyTorch，若成功将惠及所有使用 FFT 的深度学习应用。
- 与 LoRA 等参数高效方法正交——可以结合使用进一步降低内存。

## 局限与展望

- CUDA 实现在大尺寸（4096）时因线程块同步限制导致运行时开销增加。
- 目前仅验证了 1D FFT，2D FFT 扩展（如图像域应用）尚未实现。
- 下游任务验证仅在 NLU 任务（MRPC）上进行，缺少更广泛的任务覆盖。
- 循环矩阵微调在模型性能上与 LoRA 仍有差距，方法的适用场景需进一步明确。
- bfloat16 支持虽为优势，但 float16 场景下的表现未充分讨论。

## 相关工作与启发

- LoRA 通过低秩矩阵分解减少可训练参数，但仍需存储中间激活；rdFFT 从算子层面消除中间张量。
- BCA (Block Circulant Adapter) 利用循环矩阵结构进行高效微调，rdFFT 为其提供更高效的 FFT 后端。
- FourierFT 在频域进行微调，可直接受益于 rdFFT 的内存优化。
- 原地操作（in-place operation）在 BN+ReLU 融合、异常检测等场景已有先例，本文将其扩展到 FFT。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个真正原地的实数域 FFT，理论贡献扎实
- 实验充分度: ⭐⭐⭐⭐ 内存效率验证全面，但下游任务覆盖有限
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，图示清晰易懂
- 价值: ⭐⭐⭐⭐ 若合并入 PyTorch 将有广泛影响，但当前适用场景较窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](../../ICCV2025/model_compression/msq_memory-efficient_bit_sparsification_quantization.md)
- [\[AAAI 2026\] Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing](../../AAAI2026/model_compression/towards_test-time_efficient_visual_place_recognition_via_asymmetric_query_proces.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](../../ICML2025/model_compression/mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)

</div>

<!-- RELATED:END -->
