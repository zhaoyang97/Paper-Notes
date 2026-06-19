---
title: >-
  [论文解读] Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model
description: >-
  [ICML 2025][LLM安全][spiking neural network] 提出 Sorbet，首个完全兼容神经形态硬件的 Transformer 脉冲语言模型，通过两项关键创新——基于位移的 PTsoftmax 和 Bit Shifting PowerNorm (BSPN)——替代传统的 softmax 和层归一化，在 GLUE 基准上实现与 BERT 可比的性能，同时节省 27.16 倍能耗。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "spiking neural network"
  - "neuromorphic hardware"
  - "Transformer"
  - "energy efficiency"
  - "binary weight"
---

# Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model

**会议**: ICML 2025  
**arXiv**: [2409.15298](https://arxiv.org/abs/2409.15298)  
**代码**: [github.com/Kaiwen-Tang/Sorbet](https://github.com/Kaiwen-Tang/Sorbet)  
**领域**: AI安全 / 模型效率 / 脉冲神经网络  
**关键词**: spiking neural network, neuromorphic hardware, transformer, energy efficiency, binary weight

## 一句话总结

提出 Sorbet，首个完全兼容神经形态硬件的 Transformer 脉冲语言模型，通过两项关键创新——基于位移的 PTsoftmax 和 Bit Shifting PowerNorm (BSPN)——替代传统的 softmax 和层归一化，在 GLUE 基准上实现与 BERT 可比的性能，同时节省 27.16 倍能耗。

## 研究背景与动机

- 边缘设备上运行LLM存在隐私和能效需求，脉冲神经网络（SNN）因事件驱动、无乘法特性而极具前景。
- 现有基于 Transformer 的 SNN（如 SpikeLM、SpikingBERT）虽然替换了矩阵乘法，但仍依赖 **softmax** 和 **层归一化**——这两种操作涉及指数运算、除法和开方，**无法在神经形态硬件上实现**。
- SpikFormer 通过用卷积+批归一化规避了这些问题，但仅适用于视觉任务。
- **核心挑战**：如何在保持NLP性能的同时，将 Transformer 中所有操作替换为位移和加法？

## 方法详解

### 整体框架

Sorbet 基于 BERT 架构，通过三步实现完全SNN兼容：
1. 用 **PTsoftmax** 替代 softmax
2. 用 **BSPN** 替代层归一化
3. 所有权重量化为1比特、激活量化为4比特，通过脉冲神经元编码

### Bit Shifting PowerNorm (BSPN)

层归一化需要计算均值和方差（涉及除法和开方），BSPN 的设计思路：

**Step 1: 分组缩放**
- 计算输入的 L1 范数：$\mathcal{S} = \frac{1}{n}\sum_{i=1}^{n}|X_i|$
- 将 L1 范数近似为最近的2的幂次：$k = \lceil\log_2(\mathcal{S})\rceil$
- **通过右移操作实现除法**：$X \leftarrow X \gg k$

**Step 2: PowerNorm 归一化**
- 使用运行时方差 $\psi^2$（指数移动平均），推理时直接用存储的 $\psi$ 归一化
- 缩放因子 $\gamma/\psi$ 可进一步量化为2的幂次

**理论保证**：
- 定理 4.2：BSPN 保持有界梯度，$\|\partial\mathcal{L}_{BSPN}/\partial \tilde{X}_{:,i}\| \leq C$
- 引理 4.3：$\Phi(X)$ 为 1-Lipschitz 映射
- 引理 4.4：BSPN 的 Lipschitz 常数不大于 PowerNorm，且通常小于 BN

### Power-of-Two Softmax (PTsoftmax)

标准 softmax 涉及指数和除法，PTsoftmax 的近似思路：

$$\text{PTsoftmax}(z_i) = \frac{2^{\lceil z_i \rceil}}{\sum_j 2^{\lceil z_j \rceil}} \approx 2^{\lceil z_i \rceil - k}$$

其中 $k = \lceil \log_2(\sum_j 2^{\lceil z_j \rceil}) \rceil$。核心操作：
- $2^{z_i}$ 通过**左移**实现
- 除以 $2^k$ 通过**右移**实现
- 完全避免了指数和除法运算

**理论保证**（引理 4.5）：$\frac{1}{2\sqrt{2}} F_2(x_i) \leq \text{PTsoftmax}(x_i) \leq 2\sqrt{2} F_2(x_i)$，近似误差在常数因子内。

### 训练过程

采用多步蒸馏策略（Algorithm 3）：
1. 将 BERT 量化为 1-bit 权重/4-bit 激活
2. 用 PTsoftmax 替换 softmax → 蒸馏
3. 用 BSPN 替换 LN → 蒸馏
4. 转换为 SNN（通过脉冲神经元）

损失函数结合logits蒸馏（KL散度）和中间层激活蒸馏：
$$L = L_{\text{logits}} + L_{\text{reps}} = \text{KL}(p, q) + \sum_i \|r_i^s - r_i^t\|^2$$

## 实验关键数据

### GLUE 基准

| 模型 | 大小 | QQP | MNLI-m | SST-2 | QNLI | RTE | MRPC | STS-B |
|------|:----:|:---:|:-----:|:----:|:----:|:---:|:----:|:-----:|
| BERT_base | 418M | 91.3 | 84.7 | 93.3 | 91.7 | 72.6 | 88.2 | 89.4 |
| BiT (1-bit) | 13.4M | 82.9 | 77.1 | 87.7 | 85.7 | 58.8 | 79.7 | 71.1 |
| SpikeLM | * | 87.9 | 76.0 | 86.5 | 84.9 | 65.3 | 78.7 | 84.3 |
| **Sorbet** | **13.4M** | **86.5** | **77.3** | **90.4** | **86.1** | **60.3** | **79.9** | **78.1** |

Sorbet 在4个任务上达到SNN SOTA，与同大小的ANN量化模型BiT性能可比。

### 能效分析

| 模型 | FP32能耗 (mJ) | FP16 | 1-Bit |
|------|:-----------:|:----:|:-----:|
| BERT | 51.41 | 15.21 | - |
| SpikeLM | 3.98 | 1.77 | - |
| **Sorbet** | - | - | **0.65** |

- **相比BERT节省 27.16×**，相比SpikeLM节省 3.16×。
- PTsoftmax 比标准softmax节能 27.62×，BSPN 比 LN 节能 12.4×。
- 平均脉冲发射率仅 0.13-0.15，大量神经元处于静默状态。

### 消融实验

| 配置 | SST-2准确率 | 与基线差异(δ) |
|------|:--------:|:---------:|
| Softmax + LN (4-bit) | 91.5 | - |
| PTsoftmax + LN | 90.8 | -0.7 |
| Softmax + BSPN | 91.2 | -0.3 |
| PTsoftmax + BSPN | 90.9 | -0.6 |

PTsoftmax 和 BSPN 各自带来的性能损失很小（<1%），主要精度下降来自权重量化和脉冲生成过程。

## 亮点与洞察

1. **首个完全NLP可用的神经形态兼容模型**：解决了SNN在NLP领域的"最后一公里"问题——softmax和LN的替代。
2. **位移近似的优雅设计**：用2的幂次近似连续运算，理论上有界（常数因子误差），实践中性能损失极小。
3. **BSPN 的理论基础扎实**：证明了梯度有界性和 Lipschitz 常数不增，确保训练稳定性。
4. **多步蒸馏策略实用**：逐步替换组件并蒸馏，避免一次性替换导致的灾难性性能崩溃。

## 局限性

- 没有在实际神经形态芯片（如 Intel Loihi）上部署验证，仅通过 Lava 框架和 Verilog 仿真估算。
- 模型规模受限于 BERT-base 级别（13.4M量化后），未探索更大规模模型。
- PTsoftmax 不严格满足归一化条件（概率和不严格为1），虽然实验影响不大但理论上不够完美。
- 与当前主流的大语言模型（如 DeepSeek、Llama）差距较大，更适合边缘推理场景。

## 相关工作

- **Transformer SNN**：Spikformer (Zhou et al., 2024)、Spike-driven Transformer (Yao et al., 2024) 用于视觉；SpikeBERT (Lv et al., 2023)、SpikeGPT (Zhu et al.) 用于NLP但依赖LN/softmax。
- **量化BERT**：BinaryBERT (Bai et al., 2021)、BiT (Liu et al., 2022) 实现二值化但保留复杂运算。
- **简化Transformer**：I-BERT (Kim et al., 2021) 用整数近似激活函数但仍需乘除法。

## 评分

⭐⭐⭐⭐ — 解决了一个明确且重要的工程+理论问题，设计简洁优雅，理论保证充分。但缺乏实际硬件验证和对更大模型的扩展探索。SNN+NLP 方向的重要里程碑。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[NeurIPS 2025\] TRUST -- Transformer-Driven U-Net for Sparse Target Recovery](../../NeurIPS2025/llm_safety/trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)
- [\[ICML 2025\] SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int.md)
- [\[ICML 2025\] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)

</div>

<!-- RELATED:END -->
