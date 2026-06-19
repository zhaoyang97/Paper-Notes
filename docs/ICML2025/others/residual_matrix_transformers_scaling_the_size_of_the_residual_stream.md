---
title: >-
  [论文解读] Residual Matrix Transformers: Scaling the Size of the Residual Stream
description: >-
  [ICML2025][Residual Stream] 用外积记忆矩阵替换 Transformer 的残差流向量，使残差流大小可独立于模型参数量和 FLOPS 扩展，在相同 loss 下节省 58% FLOPS、25% 参数和 41% 训练 token。 - 残差流是 Transformer 的核心通信通道：Elhage e…
tags:
  - "ICML2025"
  - "Residual Stream"
  - "Outer Product Memory"
  - "参数效率"
  - "计算效率"
  - "方差传播"
---

# Residual Matrix Transformers: Scaling the Size of the Residual Stream

**会议**: ICML2025  
**arXiv**: [2506.22696](https://arxiv.org/abs/2506.22696)  
**代码**: [bmac3/residual-matrix-transformer](https://github.com/bmac3/residual-matrix-transformer)  
**领域**: Transformer架构  
**关键词**: Residual Stream, Outer Product Memory, 参数效率, 计算效率, 方差传播

## 一句话总结
用外积记忆矩阵替换 Transformer 的残差流向量，使残差流大小可独立于模型参数量和 FLOPS 扩展，在相同 loss 下节省 58% FLOPS、25% 参数和 41% 训练 token。

## 研究背景与动机

- **残差流是 Transformer 的核心通信通道**：Elhage et al. (2021) 指出残差流（residual stream）充当"记忆总线"，各层在其上存取特征。残差流的维度 $D$ 决定了可存储特征的数量（即带宽）。
- **扩展残差流代价高昂**：在标准 Transformer 中，增大 $D$ 会线性增加所有参数矩阵的大小，导致参数量和 FLOPS 同步增长。例如，残差流大小翻倍 → 参数量翻倍、FLOPS 也近似翻倍。
- **类比 MoE 的思路**：Mixture of Experts（Fedus et al., 2022）通过"稀疏参数轴"实现了模型规模独立于计算量的扩展（7 倍效率提升）。本文提出在"残差流大小"这一新轴上做类似的解耦扩展。
- **核心问题**：能否在不增加计算和参数的前提下，扩大残差流容量以提升性能？

## 方法详解

### 核心思想：外积记忆矩阵替换残差流向量

标准 Transformer 中每个 token 的残差流是一个 $D$ 维向量 $\mathbf{x} \in \mathbb{R}^D$。RMT 将其替换为一个 **外积记忆矩阵** $\mathbf{M} \in \mathbb{R}^{D_k \times D_v}$，其中残差流的"大小"为 $D_k \times D_v$，而 $D_v$ 对应标准 Transformer 的注意力头维度 $D_h$。

外积记忆矩阵的构建与检索遵循经典联想记忆（Kohonen, 1972; Anderson, 1972）：

$$\mathbf{M} = \text{Norm}\left(\sum_{p=1}^{N} \mathbf{q}^{(p)} \otimes \mathbf{x}^{(p)}\right)$$

检索时通过键向量张量缩并（tensor contraction）取回数据向量：

$$\mathbf{x}^{(r)} \approx \mathbf{q}^{(r)} \cdot_1 \mathbf{M}$$

### RMT 各层改造

**Embedding 层**：用 $R$ 个键向量 $\mathbf{w}_E^{(h)} \in \mathbb{R}^{D_k}$ 和对应嵌入矩阵 $\mathbf{W}_E^{(h)} \in \mathbb{R}^{D_v \times V}$ 的外积之和构建初始残差矩阵：

$$\text{E}(\mathbf{S}) = \sum_{h=1}^{R} \mathbf{w}_E^{(h)} \otimes \mathbf{W}_E^{(h)} \mathbf{S}$$

**Attention 层**：将标准 Transformer 中的 QKV 投影矩阵 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D_h \times D}$ 替换为键向量 $\mathbf{r}_Q, \mathbf{r}_K, \mathbf{r}_V \in \mathbb{R}^{D_k}$，通过张量缩并从残差矩阵检索特征：

$$\mathbf{Q}^{(h)} = \mathbf{r}_Q^{(h)} \cdot_1 \mathbf{X}, \quad \mathbf{K}^{(h)} = \mathbf{r}_K^{(h)} \cdot_1 \mathbf{X}, \quad \mathbf{V}^{(h)} = \mathbf{r}_V^{(h)} \cdot_1 \mathbf{X}$$

输出通过键向量 $\mathbf{w}_O^{(h)}$ 外积写回残差矩阵：

$$\text{MHA}(\mathbf{X}) = \sum_{h=1}^{R} \mathbf{w}_O^{(h)} \otimes \text{SHA}(\mathbf{Q}^{(h)}, \mathbf{K}^{(h)}, \mathbf{V}^{(h)})$$

**FeedForward 层**：核心 FFN 操作 $\tilde{\text{FF}}(\mathbf{X}) = \mathbf{W}_2 \text{GeLU}(\mathbf{W}_1 \mathbf{X})$ 保持不变（因 FFN 权重存储事实知识），仅在输入/输出端添加键向量适配器进行读取和写回。

**Unembedding 层**：用键向量检索数据后乘以 unembedding 权重得到 logits。

### 关键设计决策

| 组件 | Transformer 参数 | RMT 参数 | 变化 |
|------|-----------------|---------|------|
| QKV 投影 | $\mathbf{W} \in \mathbb{R}^{D_h \times D}$ | $\mathbf{r} \in \mathbb{R}^{D_k}$（键向量） | 矩阵 → 向量 |
| 输出投影 | $\mathbf{W}_O \in \mathbb{R}^{D \times D_h}$ | $\mathbf{w}_O \in \mathbb{R}^{D_k}$（键向量） | 矩阵 → 向量 |
| FFN 权重 | $\mathbf{W}_1, \mathbf{W}_2$ | 保持不变 + 适配器 | 保留以存储知识 |

## 实验关键数据

### 效率对比（核心结果）

| 指标 | RMT 相对 Transformer | 说明 |
|------|---------------------|------|
| FLOPS 节省 | **58%** | 达到相同 loss 所需计算量 |
| 参数节省 | **25%** | 达到相同 loss 所需参数量 |
| 训练 token 节省 | **41%** | 达到相同 loss 所需数据量 |
| 下游评测 | **优于 Transformer** | 具体评测任务未详述 |

### 资源扩展特性

- Transformer：残差流大小 ×2 → 参数量 ×2，FLOPS ≈ ×2
- RMT：残差流大小 ×2 → 参数量和 FLOPS **几乎不变**（近似常数）
- 这是因为 RMT 用 $D_k$ 维键向量替换了 $D_h \times D$ 的矩阵

### 方差传播理论分析（GPT2-medium 配置）

| 层 | 操作 | 模型 | 前向方差比 $\sigma^2_{out}/\sigma^2_{in}$ | 反向方差比 $\sigma^2_{g_{in}}/\sigma^2_{g_{out}}$ |
|----|------|------|----|----|
| Attn | 存储 | RMT | 0.4 | 1.6 |
| Attn | 存储 | Transformer | 1.0 | 1.0 |
| Attn | 检索 | **RMT** | **1.14** | 0.86 |
| Attn | 检索 | Transformer | 0.5 | 1.5 |
| FF | 存储/检索 | **RMT** | **1.0** | **1.0** |
| FF | 存储/检索 | Transformer | 0.4/1.6 | 1.6/0.4 |

**解读**：在 FFN 层，RMT 的前向和反向方差比均为理想值 1.0；Transformer 存在显著的方差衰减/放大问题。在 Attention 检索中 RMT 也更接近 1.0。

## 亮点与洞察

1. **新的扩展维度**：继 MoE 的稀疏参数轴之后，RMT 引入了"残差流大小"作为独立于计算和参数的新扩展轴，概念简洁而有力。
2. **经典联想记忆的回归**：将 1970 年代的外积记忆（Kohonen/Anderson）与现代 Transformer 结合，用向量检索替代矩阵乘法，是一种优雅的参数降维。
3. **FFN 保留不变的设计洞察**：基于 FFN 存储事实知识（Geva et al., 2021; Meng et al., 2022）的认知，选择性地保留 FFN 权重而非全部替换，体现了对 Transformer 内部机制的深入理解。
4. **理论与实验双重验证**：既有方差传播的闭式分析（Table 1-2），又有实际训练效率的实证结果。

## 局限与展望

1. **实验规模偏小**：理论分析基于 GPT2-medium（~350M），尚未在 7B+ 规模验证扩展性是否仍然成立。
2. **仅验证了语言模型**：缺少多模态、视觉等其他模态上的实验。
3. **残差矩阵的存储开销**：虽然参数和 FLOPS 降低，但每个 token 的中间激活从 $D$ 维向量变为 $D_k \times D_v$ 矩阵，推理时的内存占用可能增加。
4. **与现有优化技术的兼容性**：未讨论与 GQA、FlashAttention、KV cache 压缩等主流推理优化的兼容性。
5. **缓存文件截断**：论文实验部分（§4）的详细结果未完全覆盖，58%/25%/41% 数据来自摘要。

## 相关工作与启发

- **Mixture of Experts**（Fedus et al., 2022）：启发了"独立扩展某一轴"的思路
- **外积联想记忆**（Kohonen, 1972; Anderson, 1972）：RMT 核心机制的理论基础
- **残差流作为记忆总线**（Elhage et al., 2021）：motivate 了对残差流容量的研究
- **FFN 存储事实知识**（Geva et al., 2021; Meng et al., 2022）：指导了 FFN 层保留完整权重的设计

## 评分
- 新颖性: ⭐⭐⭐⭐ — 残差流外积记忆替换是新颖且有理论根基的思路
- 实验充分度: ⭐⭐⭐ — 效率提升显著但规模偏小，缺少大模型验证
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，架构描述详尽
- 价值: ⭐⭐⭐⭐ — 若能在大规模验证，将是 Transformer 架构改进的重要方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](../../ACL2025/others/rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)
- [\[ICML 2025\] Softmax is not Enough (for Sharp Size Generalisation)](softmax_is_not_enough_for_sharp_size_generalisation.md)
- [\[ICCV 2025\] HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding](../../ICCV2025/others/hytip_hybrid_temporal_information_propagation_for_masked_conditional_residual_vi.md)
- [\[ICML 2025\] DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](dsp_dynamic_sequence_parallelism_for_multi-dimensional_transformers.md)

</div>

<!-- RELATED:END -->
