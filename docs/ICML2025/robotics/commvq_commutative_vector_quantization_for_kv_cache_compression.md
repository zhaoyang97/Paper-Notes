---
title: >-
  [论文解读] CommVQ: Commutative Vector Quantization for KV Cache Compression
description: >-
  [ICML 2025][机器人][KV缓存压缩] 提出 CommVQ——通过可加向量量化压缩 KV cache，创新性地设计与 RoPE 可交换的码本并用 EM 算法训练，在 2-bit 下几乎无损、1-bit 下仍保持可用精度，使 LLaMA-3.1 8B 在单张 RTX 4090 上支持 128K 上下文。
tags:
  - "ICML 2025"
  - "机器人"
  - "KV缓存压缩"
  - "向量量化"
  - "RoPE交换性"
  - "长上下文推理"
  - "1-bit量化"
---

# CommVQ: Commutative Vector Quantization for KV Cache Compression

**会议**: ICML 2025  
**arXiv**: [2506.18879](https://arxiv.org/abs/2506.18879)  
**代码**: [https://github.com/UMass-Embodied-AGI/CommVQ](https://github.com/UMass-Embodied-AGI/CommVQ)  
**领域**: 机器人/模型压缩  
**关键词**: KV缓存压缩, 向量量化, RoPE交换性, 长上下文推理, 1-bit量化

## 一句话总结
提出 CommVQ——通过可加向量量化压缩 KV cache，创新性地设计与 RoPE 可交换的码本并用 EM 算法训练，在 2-bit 下几乎无损、1-bit 下仍保持可用精度，使 LLaMA-3.1 8B 在单张 RTX 4090 上支持 128K 上下文。

## 研究背景与动机

**领域现状**：LLM 上下文长度不断增长（128K+），KV cache 成为 GPU 内存主要瓶颈——LLaMA-3.1 8B 在 128K 上下文、batch size 2 时 KV cache 需 88GB。

**现有痛点**：现有 KV cache 量化方法（如 KVQuant）逐标量独立量化，在 2-bit 以下精度严重下降；对 key 中 RoPE 位置编码的处理不够优化。

**核心矛盾**：逐标量量化在极低位宽下信息损失太大，需要向量级量化来保留更多信息。

**本文目标**：高效的向量级 KV cache 压缩。

**切入角度**：将每个 token 的 key/value 向量作为整体进行可加向量量化，减少量化误差。

**核心 idea**：设计与 RoPE 矩阵可交换的码本，使解码过程可高效嵌入注意力计算——中间结果可对码本预计算并复用。

## 方法详解

### 整体框架
1. 用可加量化（Additive Quantization）将 key/value 向量编码为多个码字之和
2. 码本设计为与 RoPE 可交换（$C \cdot R = R \cdot C$），预计算 $Q \cdot C$ 复用于所有 token
3. EM 算法训练码本

### 关键设计

1. **可加向量量化**:

    - 功能：将 KV 向量量化为多个码本中码字的加权和
    - 核心思路：$v \approx c_{i_1} + c_{i_2} + \ldots + c_{i_M}$，每个码字索引只需 $\log_2 K$ bits
    - 设计动机：向量级量化比逐标量量化在相同位宽下误差更小

2. **RoPE 可交换码本**:

    - 功能：设计码本使得 $\text{Decode}(\text{RoPE}(\text{Encode}(k))) = \text{RoPE}(\text{Decode}(\text{Encode}(k)))$
    - 核心思路：码本中的码字在 RoPE 旋转下不改变量化码本结构→可预计算 $Q \cdot R \cdot C$ 并复用
    - 设计动机：避免逐 token 解码+RoPE 应用的 $O(N \cdot d)$ 开销，降为 $O(K \cdot d)$（$K$ 为码本大小）

3. **EM 算法码本训练**:

    - 功能：交替执行 E 步（分配码字）和 M 步（更新码本中心）
    - 核心思路：在保持 RoPE 可交换约束下最小化量化重建误差
    - 设计动机：经典的向量量化训练方法，有收敛保证

### 损失函数 / 训练策略
- 量化重建误差 + RoPE 可交换性约束
- Triton 内核实现实际内存节省

## 实验关键数据

### 主实验
LLaMA-3.1 8B 长上下文基准：

| 方法 | 位宽 | LongBench | InfiniteBench | 内存节省 |
|------|------|-----------|--------------|---------|
| FP16 | 16-bit | 42.1 | 22.8 | 1× |
| KVQuant | 2-bit | 38.5 | 18.2 | 8× |
| **CommVQ** | **2-bit** | **41.8** | **22.1** | **8×** |
| KVQuant | 1-bit | 28.3 | 11.5 | 16× |
| **CommVQ** | **1-bit** | **36.2** | **17.8** | **16×** |

### 消融实验

| 配置 | LongBench | 说明 |
|------|-----------|------|
| 逐标量量化 2-bit | 38.5 | 基线 |
| 向量量化 2-bit（无 RoPE 交换） | 40.9 | 向量量化的优势 |
| 向量量化 2-bit（+RoPE 交换） | **41.8** | 完整方法 |

### 关键发现
- 2-bit 下几乎无损（42.1→41.8），优于所有基线
- 1-bit 首次实现可用精度（36.2 vs FP16 的 42.1）
- 使 LLaMA-3.1 8B 在单张 RTX 4090 (24GB)上运行 128K 上下文

## 亮点与洞察
- **RoPE 可交换性设计**是核心创新——将位置编码的数学性质融入量化方案设计
- 从逐标量到向量级量化的范式转变在极低位宽下尤其关键
- 1-bit KV cache 使长上下文 LLM 在消费级 GPU 上可用，实用意义巨大

## 局限与展望
- 码本训练需要校准数据
- RoPE 可交换性要求特定码本结构，可能限制表达能力
- 仅测试 LLaMA 系列模型

## 相关工作与启发
- **vs KVQuant**: 逐标量量化，CommVQ 向量级量化更优
- **vs any4 (权重量化)**: any4 用 LUT 做权重量化，CommVQ 用向量量化做 KV cache 量化，互补
- 对所有使用 RoPE 的 Transformer 推理优化有启发

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ RoPE 可交换码本设计极具创意
- 实验充分度: ⭐⭐⭐⭐ 多基准、1-bit/2-bit、内存分析
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰
- 价值: ⭐⭐⭐⭐⭐ 使长上下文 LLM 在消费级 GPU 上可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization](star_learning_diverse_robot_skill_abstractions_through_rotation-augmented_vector.md)
- [\[CVPR 2026\] AstraNav-Memory: Contexts Compression for Long Memory](../../CVPR2026/robotics/astranav-memory_contexts_compression_for_long_memory.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](../../NeurIPS2025/robotics/vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/robotics/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ECCV 2024\] Decomposed Vector-Quantized Variational Autoencoder for Human Grasp Generation](../../ECCV2024/robotics/decomposed_vector-quantized_variational_autoencoder_for_human_grasp_generation.md)

</div>

<!-- RELATED:END -->
