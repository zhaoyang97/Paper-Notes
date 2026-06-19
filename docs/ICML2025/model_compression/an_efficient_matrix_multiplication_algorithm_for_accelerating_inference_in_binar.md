---
title: >-
  [论文解读] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks
description: >-
  [ICML 2025][模型压缩][二值/三值网络] 提出 RSR/RSR++ 算法——通过预处理固定的二值/三值权重矩阵构建分桶排列索引，实现 $O(n^2/\log n)$ 复杂度的向量-矩阵乘法，比标准 $O(n^2)$ 方法快最高 29× 的矩阵乘法、6× 的内存节省，并在 1.58-bit LLM 推理中实现 5.24× 加速。
tags:
  - "ICML 2025"
  - "模型压缩"
  - "二值/三值网络"
  - "矩阵乘法"
  - "推理加速"
  - "对数因子改进"
  - "1.58-bit LLM"
---

# An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks

**会议**: ICML 2025  
**arXiv**: [2411.06360](https://arxiv.org/abs/2411.06360)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 二值/三值网络, 矩阵乘法, 推理加速, 对数因子改进, 1.58-bit LLM

## 一句话总结
提出 RSR/RSR++ 算法——通过预处理固定的二值/三值权重矩阵构建分桶排列索引，实现 $O(n^2/\log n)$ 复杂度的向量-矩阵乘法，比标准 $O(n^2)$ 方法快最高 29× 的矩阵乘法、6× 的内存节省，并在 1.58-bit LLM 推理中实现 5.24× 加速。

## 研究背景与动机

**领域现状**：DNN 和 LLM 的推理效率是部署的关键瓶颈。权重量化（特别是二值/三值量化）已证明可保持较好精度——如 1.58-bit LLM（BitNet b1.58）使用三值权重 $\{-1, 0, 1\}$ 且准确率接近全精度模型。

**现有痛点**：即使权重被量化到 1-2 bit，推理速度的提升主要来自硬件级（如 XNOR 操作），算法层面的优化空间被忽视。矩阵乘法仍然是 $O(n^2)$ 的主要瓶颈。

**核心矛盾**：权重矩阵训练后不再变化——这是一个被忽视的结构性特性。可以通过预处理（离线）换取推理时（在线）的加速。

**本文目标**：利用权重矩阵的固定性，设计亚二次时间复杂度的推理算法。

**切入角度**：将二值权重矩阵的列分块，按字典序排列行→同一块中相同模式的行共享计算→用聚合值（aggregate values）批量更新结果向量。

**核心 idea**：预处理权重矩阵为"分桶排列索引"→推理时利用这些索引使每个字典序分组内的行共享计算，实现对数因子加速。

## 方法详解

### 整体框架
分为两阶段：
1. **预处理（离线）**：将三值矩阵转为二值矩阵对 → 列分块 → 行按字典序排列 → 构建分桶索引
2. **推理（在线）**：利用索引进行高效向量-矩阵乘法

### 关键设计

1. **三值→二值分解**:

    - 功能：将三值矩阵 $W \in \{-1, 0, 1\}^{n \times n}$ 分解为两个二值矩阵 $W^+, W^- \in \{0, 1\}^{n \times n}$
    - 核心思路：$W = W^+ - W^-$，其中 $W^+_{ij} = \mathbb{1}[W_{ij} = 1]$，$W^-_{ij} = \mathbb{1}[W_{ij} = -1]$
    - 设计动机：二值矩阵结构更简单，便于利用 0/1 模式的重复性

2. **RSR 算法（Reduction-Sharing-Reconstruction）**:

    - 功能：将列分成宽度为 $b$ 的块，每块内按字典序排列行
    - 核心思路：每块有 $n$ 行，但只有 $2^b$ 种可能的 0/1 模式（对于宽度 $b$ 的块）。对每种模式计算一个聚合值（输入向量对应位置的和），然后将聚合值分配给所有匹配的行
    - 复杂度：选择 $b = \lfloor \log n \rfloor$ 时，每块最多 $n$ 个不同模式但可共享计算，总复杂度 $O(n^2 / (\log n - \log \log n))$
    - 设计动机：对数宽度的块恰好使得模式数（$2^b \approx n$）与行数（$n$）同数量级，最大化共享收益

3. **RSR++ 算法（改进版）**:

    - 功能：进一步优化聚合值写入结果向量的过程
    - 核心思路：利用排列后的连续性，将聚合值的分配从 $O(n)$ 优化为 $O(n/\log n)$
    - 复杂度：$O(n^2 / \log n)$——严格的对数因子改进
    - 设计动机："最后一公里"优化，消除了 RSR 中 $\log \log n$ 项

4. **内存压缩（对数因子）**:

    - 功能：用索引替代原始权重矩阵
    - 核心思路：$n \times n$ 二值矩阵原始存储 $O(n^2)$ bits，用排列索引存储 $O(n^2 / \log n)$
    - 设计动机：索引编码了行的字典序分组信息，隐含了矩阵内容

### 损失函数 / 训练策略
- 纯推理优化，不影响训练过程
- 预处理在模型加载时一次性完成
- 与任何产生二值/三值权重的训练方法兼容

## 实验关键数据

### 基础矩阵乘法（$n \times n$ 矩阵）

| 矩阵大小 | 标准乘法 | RSR++ | 加速比 | 内存节省 |
|---------|---------|-------|--------|---------|
| 1024 | 1.0× | 0.18× | 5.6× | 4.2× |
| 4096 | 1.0× | 0.07× | 14.3× | 5.1× |
| 8192 | 1.0× | 0.05× | 20× | 5.8× |
| 16384 | 1.0× | 0.034× | **29×** | **6×** |

### 1.58-bit LLM 推理（BitNet b1.58）

| 模型 | 标准推理延迟 | RSR++ 延迟 | 加速比 |
|------|-----------|-----------|--------|
| BitNet-700M | 1.0× | 0.31× | 3.2× |
| BitNet-3B | 1.0× | 0.19× | **5.24×** |

### 消融实验

| 配置 | 加速比 (n=8192) | 说明 |
|------|---------------|------|
| RSR（基础版） | 15× | $O(n^2/(\log n - \log \log n))$ |
| **RSR++** | **20×** | $O(n^2/\log n)$ |
| 调优块宽度 $b$ | 22× | 理论最优 $b = \log n$ 附近 |
| 二值矩阵（直接） | 25× | 比三值稍快（不需分解） |
| 三值矩阵 | 20× | 需要分解为二值对 |

### 关键发现
- 加速比随矩阵大小增长——因为对数因子的改进在大矩阵上更显著
- 在 LLM 大小的矩阵（$n \geq 4096$）上加速超过 10×
- 内存节省也是对数因子（4-6× 对当前矩阵大小）
- 预处理时间仅为推理时间的一小部分（模型加载时完成）
- 对 1.58-bit LLM 的端到端加速验证了方法的实际价值

## 亮点与洞察
- **"权重矩阵不变"**这一简单观察蕴含强大的优化潜力——训练后预处理是几乎免费的
- 理论结果漂亮：$O(n^2/\log n)$ 是二值矩阵向量乘法的接近最优复杂度
- 与 1.58-bit LLM（BitNet b1.58）的结合具有现实意义——使这些模型在消费级设备上运行更快
- 方法完全与训练无关——任何现有的二值/三值训练方法都可以直接受益
- 内存和速度的双重改进使方法更有吸引力

## 局限与展望
- 仅适用于严格的二值/三值权重——不适用于更高精度（如 4-bit）的量化
- 预处理索引需要额外的一次性计算
- 硬件级实现（如 FPGA/ASIC）可能进一步放大优势但论文未探索
- 实际加速在 GPU 并行场景中可能受限（GPU 已经有很好的矩阵乘法优化）
- 块宽度 $b$ 的最优选择取决于硬件缓存大小

## 相关工作与启发
- **vs XNOR-Net**: XNOR 在硬件级优化二值乘法，RSR++ 在算法级优化——两者互补
- **vs BitNet b1.58**: BitNet 定义了三值权重的训练方法，RSR++ 发现了这些模型推理的新加速方式
- **vs any4/GPTQ**: 这些是 4-bit+ 量化方法，RSR++ 专注于更极端的 1-2 bit 场景
- **启发**：权重矩阵的固定性在其他量化级别中也可能有未被利用的结构——如行重复模式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论优美（对数因子改进），简单观察蕴含深刻洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 从矩阵乘法到 LLM 端到端，多尺度验证
- 写作质量: ⭐⭐⭐⭐ 算法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 对二值/三值模型的部署有重大实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ICLR 2026\] BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training](../../ICLR2026/model_compression/bep_a_binary_error_propagation_algorithm_for_binary_neural_networks_training.md)
- [\[ACL 2025\] Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](../../ACL2025/model_compression/scaling_laws_and_efficient_inference_for_ternary_language_models.md)
- [\[ICML 2025\] Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks](sparse_spectral_training_and_inference_on_euclidean_and_hyperbolic_neural_networ.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)

</div>

<!-- RELATED:END -->
