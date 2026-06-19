---
title: >-
  [论文解读] FlatQuant: Flatness Matters for LLM Quantization
description: >-
  [ICML2025][模型压缩][量化] 提出 FlatQuant，通过可学习仿射变换（Kronecker 分解）使权重和激活分布更平坦，在 W4A4 量化下首次在 LLaMA-3-70B 上实现 ≤1% 精度损失，同时 prefill 加速 2.3×、decoding 加速 1.7×。 量化中的核心挑战 LLM 推理的核心…
tags:
  - "ICML2025"
  - "模型压缩"
  - "量化"
  - "affine transformation"
  - "Kronecker product"
  - "W4A4"
  - "flatness"
---

# FlatQuant: Flatness Matters for LLM Quantization

**会议**: ICML2025  
**arXiv**: [2410.09426](https://arxiv.org/abs/2410.09426)  
**代码**: [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant)  
**领域**: LLM量化 / 模型压缩  
**关键词**: post-training quantization, affine transformation, Kronecker product, W4A4, flatness

## 一句话总结

提出 FlatQuant，通过可学习仿射变换（Kronecker 分解）使权重和激活分布更平坦，在 W4A4 量化下首次在 LLaMA-3-70B 上实现 ≤1% 精度损失，同时 prefill 加速 2.3×、decoding 加速 1.7×。

## 研究背景与动机

### 量化中的核心挑战
LLM 推理的核心瓶颈是显存和计算开销。量化（将 FP16 参数降为 INT4）是最有效的压缩手段之一。量化误差的大小取决于权重/激活分布的**平坦度（flatness）**——分布越尖锐、离群值越多，均匀量化点的误差越大。

### 现有方法的不足
- **逐通道缩放**（SmoothQuant）：平衡了激活离群值，但代价是权重分布变陡峭，且无法将离群值分散到非离群通道。
- **Hadamard 变换**（QuaRot, SpinQuant）：正交旋转可重分布离群值，但所有层共享同一变换矩阵，无法适应各层特性；且对 pivot token（序列起始 token 中的大量离群值）效果有限。
- 以上方法变换后的分布仍然存在陡峭和分散的问题。

### 关键观察
作者通过可视化 LLaMA-3-8B/70B 各层的权重/激活分布发现：即使经过现有变换，通道幅值包络线仍不平坦。此外，量化误差沿层数方向累积传播，且在 pivot token 处尤为严重。FlatQuant 在这两个维度上均显著优于 baseline。

## 方法详解

### 核心思想
对每个线性层学习一个最优的可逆仿射变换矩阵 $\mathbf{P}^*$，使得变换后的权重和激活更平坦，量化更友好：

$$\mathbf{P}^* = \arg\min_{\mathbf{P}} \|\mathbf{Y} - \mathcal{Q}(\mathbf{X}\mathbf{P})\mathcal{Q}(\mathbf{P}^{-1}\mathbf{W}^\top)\|_F^2$$

直接维护 $n \times n$ 的完整矩阵 $\mathbf{P}$ 会使计算和存储翻倍，因此关键在于高效参数化。

### Kronecker 分解（核心创新）
将 $\mathbf{P}$ 分解为两个小矩阵的 Kronecker 积：

$$\mathbf{P} = \mathbf{P}_1 \otimes \mathbf{P}_2, \quad \mathbf{P}_1 \in \mathbb{R}^{n_1 \times n_1},\ \mathbf{P}_2 \in \mathbb{R}^{n_2 \times n_2},\ n = n_1 n_2$$

利用向量化技巧，矩阵乘法转化为两次小矩阵乘法：

$$\mathcal{Q}(\mathbf{P}_1^\top \times_1 \tilde{\mathbf{X}} \times_2 \mathbf{P}_2) \times \mathcal{Q}(\mathbf{P}_1^{-1} \times_1 \tilde{\mathbf{W}} \times_2 (\mathbf{P}_2^{-1})^\top)^\top$$

- **存储节省**：最高 $n/2$ 倍（当 $n_1 = n_2 = \sqrt{n}$）
- **计算节省**：$\sqrt{n}/2$ 倍
- 实际选取 $n_1, n_2$ 使 $n_1 + n_2$ 最小，例如 $n=8192$ 时取 $(64, 128)$

### 额外可学习组件
1. **逐通道缩放** $\text{diag}(\mathbf{c})$：在仿射变换之前引入，可融合到前一层的 LayerNorm 或线性层中，零额外推理开销。
2. **可学习裁剪阈值** $\alpha_w, \alpha_a \in (0,1)$：对权重和激活各层独立学习裁剪边界，优于网格搜索。

### 训练目标（逐块 MSE 最小化）
对第 $l$ 个 Transformer block：

$$\min_{\Theta} \|\mathcal{F}_l(\mathbf{X}) - \hat{\mathcal{F}}_l(\mathbf{X}; \Theta)\|_F^2, \quad \Theta = \{\mathbf{P}, \mathbf{c}, \alpha_a, \alpha_w\}$$

使用 128 条 WikiText-2 句子校准，AdamW 优化器，初始学习率 5e-3，训练 15 epochs。LLaMA-3-8B 约 0.9 小时/单卡。

### Transformer 集成
- **Self-Attention**：4 个变换矩阵 $\mathbf{P}_a$（Q/K/V 输入）、$\mathbf{P}_o$（输出投影输入）、$\mathbf{P}_h$（Key cache 逐头）、$\mathbf{P}_v$（Value cache 逐头）。其中仅 $\mathbf{P}_a, \mathbf{P}_o$ 做 Kronecker 分解。
- **FFN**：2 个变换矩阵 $\mathbf{P}_{ug}$（FFN 输入）、$\mathbf{P}_d$（down projection 输入），均做分解。
- **LayerNorm**：保留原始 LayerNorm（不改为 RMSNorm），允许各层不同的仿射变换，增强表达力。

### 高效 Kernel 设计
将仿射变换 + 量化融合为单个 Triton kernel：将 $\mathbf{P}_1, \mathbf{P}_2$ 加载到 SRAM，每个 thread block 切分一个 tiling block $\bar{\mathbf{X}} \in \mathbb{R}^{n_1 \times n_2}$，执行 $\mathbf{P}_1 \bar{\mathbf{X}} \mathbf{P}_2$ 并即时量化，所有中间结果驻留 SRAM，消除冗余显存访问。

## 实验关键数据

### W4A4 语言建模（PPL↓）

| 模型 | 方法 | WikiText-2 | C4 |
|------|------|-----------|-----|
| LLaMA-3-70B FP16 | - | 2.86 | 7.17 |
| LLaMA-3-70B | QuaRot+RTN | 55.44 | 79.48 |
| LLaMA-3-70B | SpinQuant+RTN | 7.58 | 15.39 |
| **LLaMA-3-70B** | **FlatQuant+RTN** | **3.78** | **7.86** |
| LLaMA-3-8B FP16 | - | 6.14 | 9.45 |
| LLaMA-3-8B | SpinQuant+RTN | 7.96 | 13.45 |
| **LLaMA-3-8B** | **FlatQuant+RTN** | **6.98** | **11.13** |

### W4A4 零样本推理准确率（Avg↑）

| 模型 | 方法 | Avg (6 tasks) |
|------|------|--------------|
| LLaMA-3-70B FP16 | - | 79.95 |
| LLaMA-3-70B | SpinQuant+RTN | 65.66 |
| **LLaMA-3-70B** | **FlatQuant+RTN** | **79.01** |
| LLaMA-3-8B FP16 | - | 73.23 |
| LLaMA-3-8B | SpinQuant+RTN | 66.98 |
| **LLaMA-3-8B** | **FlatQuant+RTN** | **71.23** |

**核心结论**：

- LLaMA-3-70B W4A4 量化精度损失仅 0.94%（79.01 vs 79.95），远超 SpinQuant（14.29% 损失）。
- 即使用最简单的 RTN 量化器，FlatQuant 也优于 SpinQuant+GPTQ 组合。
- 推理加速：W4A4 下相比 FP16，prefill 加速 2.3×，decoding 加速 1.7×。
- 额外开销极低：仿射变换仅占 2.61% FLOPs，额外存储仅 3.41MB（LLaMA-2-7B）。

## 亮点与洞察

1. **Flatness 视角的统一性**：将量化问题归结为分布平坦度优化，提供了直观且可量化的优化目标（kurtosis/MSE landscape）。
2. **Kronecker 分解的精妙平衡**：用 $(n_1, n_2)$ 分解避免了完整 $n \times n$ 矩阵的开销，存储/计算 trade-off 极佳。
3. **RTN 就够了**：在足够好的变换下，最简单的 round-to-nearest 即可接近 GPTQ 效果，说明变换本身比量化策略更重要。
4. **保留 LayerNorm 的选择**：不同于 QuaRot/SpinQuant 将 LayerNorm 改为 RMSNorm 以共享变换，FlatQuant 保留 LayerNorm 让各层独立学习，表达力更强。
5. **Triton kernel 融合设计**：将 memory-bound 的仿射变换和量化合并，避免中间结果落地显存。

## 局限与展望

1. **校准成本**：虽然称"轻量"，但仍需逐块训练 15 epochs（LLaMA-3-8B 约 0.9 小时）。对于更大模型（如 405B），校准时间可能更长。
2. **W4A4 是最佳甜点**：论文主要聚焦 W4A4 设置，对更激进的 W2/W3 或纯权重量化的效果未深入探讨。
3. **模型覆盖有限**：主要验证 LLaMA 系列，对 Mixtral 等 MoE 架构、多模态模型的适用性未知。
4. **逐层独立变换的局限**：Kronecker 分解虽高效，但仍是线性变换，对非线性分布模式的处理能力受限。
5. **对生成质量的评估不足**：仅评估了 PPL 和零样本分类，缺少长文本生成、对话、指令跟随等下游任务评估。

## 相关工作与启发

- **SmoothQuant**（Xiao et al., 2023）：逐通道缩放的先驱，FlatQuant 将其作为组件之一并扩展到全仿射变换。
- **QuaRot**（Ashkboos et al., 2024）：Hadamard 旋转 + 融合 kernel，FlatQuant 的直接竞争对手。
- **SpinQuant**（Liu et al., 2024）：学习正交矩阵旋转，但受限于共享变换。FlatQuant 放松正交约束为一般可逆矩阵。
- **AffineQuant**（Ma et al., 2024）：同样用仿射变换但无 Kronecker 分解，推理开销大。FlatQuant 可视为其高效版本。

## 评分
- 新颖性: ⭐⭐⭐⭐ — Kronecker 分解仿射变换的思路简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 多模型多任务全面比较，含推理速度和消融实验
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ — W4A4 首次达到实用精度，推理加速显著，有代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](../../ACL2025/model_compression/moqae_mixed_precision_kv_cache.md)
- [\[ICML 2025\] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference](blockdialect_block-wise_fine-grained_mixed_format_quantization_for_energy-effici.md)
- [\[ICLR 2026\] The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](../../ICLR2026/model_compression/the_geometry_of_llm_quantization_gptq_as_babais_nearest_plane_algorithm.md)
- [\[CVPR 2026\] What Matters in Practical Learned Image Compression](../../CVPR2026/model_compression/what_matters_in_practical_learned_image_compression.md)
- [\[AAAI 2026\] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning](../../AAAI2026/model_compression/beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)

</div>

<!-- RELATED:END -->
