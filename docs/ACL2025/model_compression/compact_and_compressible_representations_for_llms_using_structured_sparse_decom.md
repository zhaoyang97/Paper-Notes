---
title: >-
  [论文解读] Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition
description: >-
  [ACL 2025][模型压缩][结构化稀疏分解] 本文提出一种结构化稀疏分解方法，将LLM权重矩阵分解为低秩部分和结构化稀疏部分的组合，实现高压缩比的同时保持模型性能，使大模型在资源受限环境下高效部署成为可能。 领域现状：大语言模型（LLM）的参数量已达数百亿甚至万亿级别，部署时面临巨大的存储和计算开销…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "结构化稀疏分解"
  - "LLM压缩"
  - "权重矩阵分解"
  - "低秩近似"
  - "稀疏表示"
---

# Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition

**会议**: ACL 2025  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 结构化稀疏分解, LLM压缩, 权重矩阵分解, 低秩近似, 稀疏表示

## 一句话总结
本文提出一种结构化稀疏分解方法，将LLM权重矩阵分解为低秩部分和结构化稀疏部分的组合，实现高压缩比的同时保持模型性能，使大模型在资源受限环境下高效部署成为可能。

## 研究背景与动机

**领域现状**：大语言模型（LLM）的参数量已达数百亿甚至万亿级别，部署时面临巨大的存储和计算开销。现有的模型压缩技术主要包括量化、剪枝、知识蒸馏和低秩分解等方法。

**现有痛点**：传统低秩分解（如SVD）虽然能有效压缩模型，但在高压缩比下性能退化严重，因为权重矩阵中并非所有信息都可以用低秩结构捕获。非结构化稀疏方法虽然保留精度更好，但稀疏模式不规则，难以在硬件上高效执行。量化方法在极低位宽下同样面临性能崩塌问题。

**核心矛盾**：高压缩比与性能保持之间存在根本性trade-off。低秩近似忽略了权重矩阵中的稀疏异常值（outlier），而这些outlier对模型输出有不成比例的重大影响。

**本文目标**：设计一种同时利用低秩结构和结构化稀疏模式的混合分解方案，在更高压缩比下实现更好的性能保持。

**切入角度**：作者观察到LLM权重矩阵可以自然地分解为一个"平滑"的低秩成分和一个"尖锐"的稀疏成分，其中稀疏成分主要集中在少数关键通道上，呈现出结构化的分布模式。

**核心 idea**：用结构化稀疏分解（Structured Sparse Decomposition）将权重矩阵表示为低秩矩阵加上按列/行组织的结构化稀疏矩阵，既能保留低秩近似的压缩优势，又能精确捕获对性能至关重要的稀疏异常值，且结构化的稀疏模式允许高效的硬件执行。

## 方法详解

### 整体框架
给定一个LLM的权重矩阵 $W \in \mathbb{R}^{m \times n}$，本方法将其分解为 $W \approx L + S$，其中 $L$ 是秩为 $r$ 的低秩矩阵（$L = UV^T$，$U \in \mathbb{R}^{m \times r}$，$V \in \mathbb{R}^{n \times r}$），$S$ 是结构化稀疏矩阵。整个流程分为三个阶段：（1）初始低秩近似获取残差；（2）对残差施加结构化稀疏约束选择关键元素；（3）联合优化低秩和稀疏两个成分使总重建误差最小化。

### 关键设计

1. **自适应秩选择机制**:

    - 功能：为每一层权重矩阵自动确定最优的低秩近似秩 $r$
    - 核心思路：基于SVD奇异值的能量分布，计算累积能量占比达到阈值 $\tau$ 时对应的秩。不同层的权重矩阵信息分布不同，因此采用逐层自适应而非全局统一的秩设定。通过校准数据集上的输出误差作为反馈信号，微调阈值 $\tau$ 使得总体压缩比满足目标约束
    - 设计动机：固定秩的分解对所有层一视同仁，但实际上注意力层和FFN层的秩分布差异很大。自适应秩分配可以把"预算"集中在信息密度高的层上

2. **结构化稀疏残差捕获**:

    - 功能：从低秩近似的残差 $R = W - L$ 中提取结构化稀疏成分
    - 核心思路：将残差矩阵按列分组（对应输出通道），计算每组的 $\ell_2$ 范数，选择范数最大的 top-$k$ 组保留，其余置零。组内元素进一步通过阈值筛选，保留幅值超过组内均值的元素。这样得到的稀疏矩阵 $S$ 具有列级别的结构化模式（column-wise structured sparsity），非零元素集中在少数列上
    - 设计动机：非结构化稀疏虽然灵活但硬件不友好；按列组织的结构化稀疏可以利用现代GPU的向量化操作高效执行，且列稀疏模式与Transformer中权重矩阵的异常值分布自然吻合（异常值通常集中在特定输出通道）

3. **联合迭代优化**:

    - 功能：交替优化低秩和稀疏两个成分，最小化总重建误差
    - 核心思路：采用交替最小化（Alternating Minimization）策略：固定 $S$ 优化 $L$（等价于对 $W - S$ 做截断SVD），固定 $L$ 优化 $S$（等价于对 $W - L$ 做结构化稀疏选择）。迭代收敛后，两个成分互相适配，比单次分解获得更低的重建误差。在校准数据集上使用输出激活值的MSE作为优化目标，而非简单的权重MSE
    - 设计动机：一次性的分解无法处理低秩和稀疏成分之间的耦合关系。迭代优化让两者相互调整，实现全局最优

### 损失函数 / 训练策略
优化目标为最小化校准数据上的输出重建误差：$\min_{L,S} \|WX - (L+S)X\|_F^2$，其中 $X$ 是校准数据的输入激活值。整个过程无需反向传播或梯度计算，完全基于矩阵运算，处理单层仅需数分钟。分解完成后可选择性地进行少量epoch的微调以恢复端到端性能。

## 实验关键数据

### 主实验

| 模型 | 压缩比 | 方法 | WikiText PPL | C4 PPL | 平均任务准确率 |
|------|--------|------|-------------|--------|--------------|
| LLaMA-2-7B | 2× | SVD | 8.92 | 11.34 | 61.2% |
| LLaMA-2-7B | 2× | ASVD | 7.84 | 10.21 | 63.8% |
| LLaMA-2-7B | 2× | 本文 | **7.12** | **9.53** | **66.4%** |
| LLaMA-2-7B | 4× | SVD | 25.6 | 31.2 | 48.3% |
| LLaMA-2-7B | 4× | GPTQ-4bit | 8.35 | 10.89 | 63.1% |
| LLaMA-2-7B | 4× | 本文 | **7.96** | **10.12** | **64.7%** |
| LLaMA-2-13B | 2× | 本文 | 6.38 | 8.71 | 69.2% |

### 消融实验

| 配置 | WikiText PPL | 说明 |
|------|-------------|------|
| Full model (低秩+稀疏+迭代) | 7.12 | 完整模型 |
| 仅低秩（无稀疏） | 8.92 | 退化为标准SVD，PPL +1.80 |
| 低秩+非结构化稀疏 | 7.45 | 非结构化稀疏有效但硬件效率低 |
| 低秩+结构化稀疏（无迭代） | 7.58 | 单次分解不如迭代优化 |
| 固定秩（全局统一） | 7.89 | 自适应秩选择贡献显著 |

### 关键发现
- 结构化稀疏残差捕获贡献最大，去掉后PPL增加1.80点，说明异常值对模型性能至关重要
- 迭代优化相比单次分解额外降低0.46 PPL，收敛通常只需3-5轮迭代
- 在4×高压缩比下，本方法与4-bit量化的GPTQ性能接近，但两种方法可以正交组合进一步压缩
- 自适应秩分配发现注意力层需要更高的秩，而FFN层可以用更低秩+更多稀疏元素近似

## 亮点与洞察
- 低秩+结构化稀疏的混合分解思路非常优雅，巧妙利用了LLM权重矩阵的内在结构特性，将两种互补的压缩范式统一在一个框架中
- 无需反向传播的逐层分解方案使得压缩过程极其高效，数小时内即可完成70B模型的压缩
- 结构化稀疏的设计使得压缩后的模型可以利用现有GPU的稀疏矩阵运算加速库，真正实现推理加速而非仅减小存储

## 局限与展望
- 分解过程依赖校准数据集，校准数据的领域分布可能影响不同下游任务的性能
- 结构化稀疏的粒度（列级别）可能不是所有架构的最优选择，行级别或块级别稀疏在某些层可能更合适
- 与量化方法的组合策略值得深入探索，低秩+稀疏+量化的三重压缩潜力巨大
- 目前仅在decoder-only模型上验证，encoder-decoder架构的效果待验证

## 相关工作与启发
- **vs ASVD**: ASVD也是自适应SVD分解，但未考虑残差中的稀疏结构，本文通过引入结构化稀疏成分显著提升了高压缩比下的性能
- **vs SparseGPT**: SparseGPT是纯稀疏剪枝方法，不利用低秩结构，在相同有效参数量下本文的混合分解表现更优
- **vs GPTQ**: GPTQ是量化方法，与本文的分解方法正交，两者可以组合使用实现更极致的压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ 低秩+结构化稀疏的混合分解思路有新意，但各组件本身不算全新
- 实验充分度: ⭐⭐⭐⭐ 多模型多压缩比验证，消融实验充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机推导合理
- 价值: ⭐⭐⭐⭐ 对LLM高效部署有实际参考价值，方法可与其他压缩技术正交组合

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)
- [\[ACL 2025\] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)
- [\[ACL 2026\] Two-Stage Regularization-Based Structured Pruning for LLMs](../../ACL2026/model_compression/two-stage_regularization-based_structured_pruning_for_llms.md)
- [\[ACL 2025\] Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)

</div>

<!-- RELATED:END -->
