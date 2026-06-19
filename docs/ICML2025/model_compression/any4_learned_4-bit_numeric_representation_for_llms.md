---
title: >-
  [论文解读] any4: Learned 4-bit Numeric Representation for LLMs
description: >-
  [ICML 2025][模型压缩][量化] 提出 any4——一种通过 k-means 聚类学习每行权重矩阵的最优 4-bit 非均匀量化码本的方法，无需权重/激活预处理，在 Llama 2/3、Mistral、Mixtral 上均优于 int4/fp4/nf4，且仅用单个校准样本即可。 领域现状：LLM 部署需要权重压缩…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "量化"
  - "4-bit"
  - "LLM推理"
  - "非均匀量化"
  - "查找表"
---

# any4: Learned 4-bit Numeric Representation for LLMs

**会议**: ICML 2025  
**arXiv**: [2507.04610](https://arxiv.org/abs/2507.04610)  
**代码**: [https://github.com/facebookresearch/any4](https://github.com/facebookresearch/any4)  
**领域**: 模型压缩  
**关键词**: 量化, 4-bit, LLM推理, 非均匀量化, 查找表

## 一句话总结
提出 any4——一种通过 k-means 聚类学习每行权重矩阵的最优 4-bit 非均匀量化码本的方法，无需权重/激活预处理，在 Llama 2/3、Mistral、Mixtral 上均优于 int4/fp4/nf4，且仅用单个校准样本即可。

## 研究背景与动机

**领域现状**：LLM 部署需要权重压缩，4-bit 量化是主流方案。现有数值格式包括 int4（均匀）、fp4（对数/线性）、nf4（匹配高斯分布）。

**现有痛点**：int4 均匀量化与权重的高斯分布不匹配导致精度损失；fp4 和 nf4 使用固定分布，无法适应每一层/每一行权重的独特分布特征。AWQ/GPTQ 等需要复杂的权重/激活预处理。

**核心矛盾**：先验假设的数值分布（均匀/高斯/浮点）不一定最优——为什么不直接从数据中学习最佳的量化码本？

**本文目标**：设计一种无需预处理的自适应 4-bit 数值表示。

**切入角度**：对权重矩阵每行使用 k-means 聚类，学习 16 个最优量化值（查找表），开销仅为每个权重元素额外 ~0.06 位。

**核心 idea**：将量化问题回归经典信号处理——Lloyd-Max 量化器的 LLM 版本。

## 方法详解

### 整体框架
对权重矩阵 $W$ 的每一行：
1. 运行 k-means 聚类，得到 16 个聚类中心（即 4-bit 码本）
2. 每个权重值映射到最近的聚类中心（4-bit 索引）
3. 存储每行 16 个 bf16 值的查找表（LUT），推理时通过 LUT 反量化

### 关键设计

1. **逐行 k-means 码本学习**:

    - 功能：为权重矩阵每行学习独立的 16 个量化值
    - 核心思路：k-means 最小化量化重建误差 $\sum_i \|w_i - c_{\text{nearest}}\|^2$
    - 设计动机：不同行的权重分布可能差异很大，逐行学习比全矩阵共享更精确
    - 开销：每行 16 × 2 字节 = 32 字节，分摊到 4096 列约 0.0625 bits/element

2. **分组量化兼容**:

    - 功能：在 k-means 码本之上叠加分组 scale 和 zero-point
    - 核心思路：group size=128 时每元素额外 0.25 bits，总计 4.3125 bits/element
    - 设计动机：标准 int4 分组量化已 4.25 bits，any4 仅多 0.06 bits 的 LUT 开销

3. **单样本校准**:

    - 功能：用一个精心选择的多样化样本替代传统的数百个校准样本
    - 核心思路：k-means 聚类完全基于权重自身分布，不依赖激活分布
    - 设计动机：极大简化校准流程

### 损失函数 / 训练策略
- 纯 PTQ（训练后量化），无需微调
- k-means 聚类计算成本低
- GPU 实现通过高效 LUT 策略（tinygemm 库）

## 实验关键数据

### 主实验
Llama-3 系列 perplexity 对比：

| 模型 | FP16 | int4 | fp4 | nf4 | **any4** |
|------|------|------|-----|-----|----------|
| Llama-3-8B | 6.14 | 6.82 | 6.65 | 6.48 | **6.35** |
| Llama-3-70B | 2.85 | 3.02 | 2.98 | 2.95 | **2.90** |

### 消融实验

| 配置 | PPL | 说明 |
|------|-----|------|
| any4 (逐行LUT) | 6.35 | 最佳 |
| any4 (全矩阵LUT) | 6.52 | 共享LUT精度下降 |
| nf4 | 6.48 | 固定高斯分布 |
| any4 + AWQ 预处理 | 6.28 | 与预处理方法正交可叠加 |

### 关键发现
- any4 在所有测试模型和规模上一致优于 int4/fp4/nf4
- 与 AWQ/GPTQ 竞争力相当但不需要权重/激活预处理
- any2 和 any3 在更低位宽也表现出竞争力
- tinygemm 库实现了延迟优化的 GPU 矩阵乘法

## 亮点与洞察
- **回归经典**——将信号处理中的 Lloyd-Max 量化器应用于 LLM，思路简洁有力
- 每行独立码本的开销极低（0.06 bits），但精度提升显著
- 与预处理方法（AWQ/GPTQ）正交，可叠加使用
- 开源 tinygemm 库是实际贡献

## 局限与展望
- 逐行 k-means 增加离线量化时间
- LUT 实现可能在某些硬件上不如原生 int4/fp4 高效
- 未讨论 any4 在 KV cache 量化中的应用

## 相关工作与启发
- **vs nf4 (QLoRA)**: nf4 假设高斯分布，any4 从数据学习，更灵活
- **vs AWQ/GPTQ**: 它们预处理权重/激活，any4 不需要，但两者可结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 经典方法的精妙应用
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型系列、多规模、完整对比
- 写作质量: ⭐⭐⭐⭐ 背景清晰，方法简洁
- 价值: ⭐⭐⭐⭐⭐ 实用且已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](../../AAAI2026/model_compression/specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)
- [\[NeurIPS 2025\] Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment](../../NeurIPS2025/model_compression/q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)
- [\[ICML 2025\] Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks](generalization_bounds_via_meta-learned_model_representations_pac-bayes_and_sampl.md)
- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](../../ICML2026/model_compression/neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)
- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](../../ICCV2025/model_compression/msq_memory-efficient_bit_sparsification_quantization.md)

</div>

<!-- RELATED:END -->
