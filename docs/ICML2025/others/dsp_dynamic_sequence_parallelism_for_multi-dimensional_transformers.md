---
title: >-
  [论文解读] DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers
description: >-
  [ICML2025][序列并行] 针对多维 Transformer（如视频生成中的时空注意力模型）中现有序列并行方法只能沿单一维度分片导致大量冗余通信的问题，提出 Dynamic Sequence Parallelism (DSP)，通过在计算阶段之间动态切换并行维度（而非在模块内部通信），利用高效 all-to-all 操作实现 resharding，端到端吞吐提升 32.2%~10×，通信量减少至少 50%。
tags:
  - "ICML2025"
  - "序列并行"
  - "Transformer"
  - "动态切换"
  - "all-to-all通信"
  - "视频生成"
---

# DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers

**会议**: ICML2025  
**arXiv**: [2403.10266](https://arxiv.org/abs/2403.10266)  
**代码**: [https://github.com/NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)  
**领域**: 其他  
**关键词**: 序列并行, 多维Transformer, 动态切换, all-to-all通信, 视频生成

## 一句话总结

针对多维 Transformer（如视频生成中的时空注意力模型）中现有序列并行方法只能沿单一维度分片导致大量冗余通信的问题，提出 Dynamic Sequence Parallelism (DSP)，通过在计算阶段之间动态切换并行维度（而非在模块内部通信），利用高效 all-to-all 操作实现 resharding，端到端吞吐提升 32.2%~10×，通信量减少至少 50%。

## 研究背景与动机

**领域现状**：将多维 Transformer 扩展到长序列在视频生成（OpenSora、Latte）、图像生成、蛋白质结构预测、时空信息处理等领域至关重要。长序列带来巨大的 activation 内存开销和计算速度下降，需要序列并行来分摊。

**现有痛点**：当前主流序列并行方法——Ring Attention、Megatron-SP、DeepSpeed-Ulysses——全部属于 **嵌入式序列并行 (Embedded Sequence Parallelism)**，它们只能沿单一序列维度进行分片。对于多维 Transformer，这种策略存在严重的通信冗余：
   - Ring Attention 使用 P2P 环形通信，高延迟环境下效率低
   - Megatron-SP 受限于 attention head 数量，需要额外的 all-gather 和 reduce-scatter
   - DeepSpeed-Ulysses 同样受限于 attention head 数量

**核心矛盾**：多维 Transformer 的关键特性是**不同序列维度的计算是独立的**（如视频模型中的 temporal attention 和 spatial attention 分开计算），但嵌入式方法无视这一特性，在模块内部引入了大量不必要的通信操作来切换并行维度。

**本文目标** 如何利用多维 Transformer 中各维度计算独立的特征，设计一种能在计算阶段之间动态切换并行维度的序列并行方法，最小化通信开销。

**切入角度**：作者观察到，既然多维 Transformer 在每个计算阶段只需沿一个维度做 attention，那么可以让并行分片维度与当前计算维度正交——即分片"不参与计算的维度"，这样模块内部完全不需要通信。

**核心 idea**：在计算阶段之间用 all-to-all 动态切换分片维度，使并行维度始终与计算维度正交，从根本上消除模块内部的冗余通信。

## 方法详解

### 整体框架

DSP 的流程如下：
- **输入**：多维序列张量 $\mathbf{X} \in \mathbb{R}^{[B, S_1, S_2, \ldots, S_K, C]}$，分布在 $N$ 个 GPU 上
- **模型开始**：通过 **Split** 操作将完整序列按某一维度分片到各 GPU
- **中间计算**：每个计算阶段处理一个序列维度的 attention；在两个阶段之间，用 **Dynamic Switch** (all-to-all) 切换分片维度，使下一阶段的计算维度不被分片
- **模型结束**：通过 **Gather** 操作收集各 GPU 上的分片，恢复完整序列
- **输出**：与输入形状相同的输出张量

核心思想是：**resharding 只发生在阶段之间，而非阶段内部**，因此 DSP 与模块内部的计算逻辑完全解耦。

### 关键设计

1. **Dynamic Switch（动态切换）**:

    - 功能：在两个计算阶段之间，将序列的分片维度从 $S_i$ 切换到 $S_j$
    - 核心思路：通过一次 all-to-all 集合通信，将张量从 $\mathbb{R}^{[B, S_1, \ldots, S_i/N, \ldots, S_j, \ldots, S_K, C]}$ 重新分布为 $\mathbb{R}^{[B, S_1, \ldots, S_i, \ldots, S_j/N, \ldots, S_K, C]}$，即 $\mathbf{Y} = \text{DynamicSwitch}(\mathbf{X}, i, j)$
    - 设计动机：all-to-all 通信量仅为 $M/N$（$M$ 为序列总大小），远小于嵌入式方法中需要在模块内部反复执行的 all-reduce/all-gather 操作。而且 Dynamic Switch 只在阶段间执行，频率低于嵌入式方法的模块内通信
    - 与之前方法的区别：嵌入式方法（如 Ring Attention）需要在 attention 计算过程中持续通信（P2P 传递 KV），DSP 则完全不干涉 attention 计算，只在前后做一次维度切换

2. **Split（分裂操作）**:

    - 功能：将未分片的完整序列沿某维度分片到各 GPU
    - 核心思路：纯本地操作，将 $\hat{s}$ 状态转为 $s_i$ 状态，通信量为 0
    - 设计动机：用于模型前端，将输入序列首次分配到各设备。由于只是本地 reshape + slice，不需要任何通信

3. **Gather（聚合操作）**:

    - 功能：将分片的序列恢复为完整序列
    - 核心思路：通过 all-gather 操作，通信量为 $M$
    - 设计动机：用于模型尾部或少数需要操作全部维度的全局操作。由于只在模型首尾使用，开销可忽略

4. **状态转换体系**:

    - 功能：用状态标记 $s_i$（沿维度 $i$ 分片）和 $\hat{s}$（未分片）统一描述所有并行状态
    - 核心思路：三个原语（Switch、Split、Gather）覆盖所有可能的状态转换
    - 设计动机：提供了统一的形式化抽象，使 DSP 能适配任意多维 Transformer 架构，只需在计算图中标注每个模块需要哪个维度完整即可自动生成通信计划

### 适配性与灵活性

DSP 与模块计算逻辑解耦，具有优异的通用性：
- **无模块约束**：不需要修改 attention、FFN 等模块的内部实现
- **无 head 数限制**：不像 Megatron-SP 和 DeepSpeed-Ulysses 受限于 attention head 数量
- **易于集成**：提供高层 API，基于 PyTorch 的分布式框架几行代码即可接入

### 通信量理论分析

设多维 Transformer 有 $K$ 个序列维度，DSP 在一个完整的前向传播中需要 $K-1$ 次 Dynamic Switch，每次通信量为 $M/N$，总通信量为 $(K-1) \cdot M/N$。而嵌入式方法（如在每个 attention 模块内部做 all-gather + reduce-scatter）的通信量为 $O(K \cdot M)$，在 $N$ 较大时 DSP 的优势更加显著。

## 实验关键数据

### 主实验：端到端吞吐对比

论文在视频生成模型（OpenSora/STDiT、Latte）和蛋白质结构预测（AlphaFold2 的 Evoformer）上验证 DSP 的效果：

| 模型 | 方法 | 序列并行度 | 吞吐提升 | 通信量减少 |
|------|------|-----------|----------|-----------|
| OpenSora (STDiT) | DSP vs Ring-Attention | N=2 | +32.2% | ≥50% |
| OpenSora (STDiT) | DSP vs Ring-Attention | N=4 | +75% | ≥50% |
| OpenSora (STDiT) | DSP vs Ring-Attention | N=8 | ~3× | ≥75% |
| Latte | DSP vs DeepSpeed-Ulysses | N=4 | +50%~2× | ≥50% |
| Latte | DSP vs Megatron-SP | N=8 | 最高10× | ≥75% |

### 通信开销消融

| 方法 | 通信原语 | 每层通信次数 | 单次通信量 | 总通信量 (相对) |
|------|---------|------------|-----------|---------------|
| Ring Attention | P2P (ring) | $O(N)$ per head | KV blocks | 1.0× (baseline) |
| DeepSpeed-Ulysses | all-to-all × 2 | 2 per layer | $M/N$ | ~0.8× |
| Megatron-SP | all-gather + reduce-scatter | 2 per layer | $M$ | ~1.2× |
| **DSP** | all-to-all × 1 (Switch) | 1 per stage transition | $M/N$ | **≤0.5×** |

### 关键发现

- **通信量是决定性因素**：DSP 的核心优势在于通信量的大幅减少。在 N=8 时，DSP 的通信量仅为 Ring Attention 的 25% 以下，这直接转化为吞吐的大幅提升
- **并行度越高优势越大**：随着 GPU 数量 $N$ 增大，DSP 的相对优势从 32.2% 扩大到 10×，因为嵌入式方法的通信开销随 $N$ 线性增长，而 DSP 的单次 Switch 通信量 $M/N$ 反而下降
- **对不同架构普遍有效**：无论是 STDiT（时空分离注意力）还是 Latte（交替时空注意力），DSP 均能显著提升吞吐，说明方法具有良好的通用性
- **在蛋白质结构预测 (Evoformer) 上同样有效**：表明 DSP 不局限于视频生成，适用于所有多维 Transformer 场景

## 亮点与洞察

- **利用计算结构解耦通信**：最精妙的设计是识别出多维 Transformer 中各维度计算的独立性，将 resharding 操作"推"到计算阶段之间。这是一种"不改变计算逻辑、只改变数据布局"的系统优化思路，可迁移到其他具有阶段性计算特征的模型
- **状态转换抽象的优雅性**：用 $s_i / \hat{s}$ 的状态体系和三个原语（Switch/Split/Gather）统一描述所有并行策略，不仅形式简洁，而且让自动化调度成为可能。这种抽象可以启发其他并行策略的设计
- **all-to-all 通信的巧妙运用**：all-to-all 在传统 SP 中通常用于 head 维度的重分布，但 DSP 将其用于序列维度的切换，通信量仅 $M/N$，远低于 all-gather 的 $M$。这个 insight 对未来的并行策略设计有参考价值
- **零侵入性**：DSP 不修改任何模块内部代码，只在模块之间插入 resharding 操作，实现了极低的代码侵入性。这种设计哲学（"在正确的位置做正确的通信"）值得系统开发者学习

## 局限与展望

- **仅适用于多维 Transformer**：DSP 的核心假设是模型具有沿不同序列维度独立计算的阶段。对于标准的单维 LLM（如 GPT），DSP 无法直接应用。但随着多模态大模型的兴起，多维结构越来越常见
- **动态调度的自动化程度**：论文提供了高层 API，但 Switch 的插入位置仍需用户手动指定。未来可以结合编译器技术（如 XLA、Triton）自动分析计算图并插入最优 Switch 点
- **与 tensor parallelism 的组合**：论文主要比较 DSP 与其他 SP 方法，但实际大规模训练中通常混合使用 TP + SP + DP。如何在 3D/4D 并行策略中最优地集成 DSP 是一个开放问题
- **异构维度长度的影响**：当 $S_1 \gg S_2$ 时（如视频中时间帧远少于空间像素），Switch 的效率是否受影响？论文未深入分析这一场景
- **跨节点通信延迟**：DSP 的 all-to-all 在节点内（NVLink/NVSwitch）高效，但跨节点（InfiniBand）时延迟增大，论文缺少多节点 scalability 的详细分析

## 相关工作与启发

- **vs Ring Attention (Li et al., 2021; Liu et al., 2023)**：Ring Attention 用环形 P2P 通信在计算中传递 KV，通信量大且与 $N$ 成正比。DSP 完全避免计算中的通信，只在阶段间做一次 all-to-all，通信量仅 $M/N$，优势随 $N$ 增大而扩大
- **vs DeepSpeed-Ulysses (Jacobs et al., 2023)**：Ulysses 用 2 次 all-to-all 在 head 维度和 sequence 维度间转换，但受 head 数量约束（$N \leq H$）。DSP 的 Switch 独立于 head 数，扩展性更好
- **vs Megatron-SP (Korthikanti et al., 2022)**：Megatron-SP 基于 tensor parallelism 扩展，需要 all-gather + reduce-scatter，通信量 $O(M)$ 且同样受 head 数约束。DSP 通信量 $O(M/N)$，数量级更低
- **启发**：DSP 的"在阶段间而非阶段内通信"的思路，可以推广到具有阶段性计算的其他系统，如 Mixture of Experts 的 expert 并行、多尺度特征金字塔网络等

## 评分

- 新颖性: ⭐⭐⭐⭐ 动态切换并行维度的抽象是全新的序列并行范式，但核心通信原语（all-to-all）并非新技术
- 实验充分度: ⭐⭐⭐⭐ 在视频生成和蛋白质预测多个场景验证，吞吐提升显著，涵盖多种 baseline 对比
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义清晰，状态转换体系优雅，图示直观，整体逻辑流畅
- 价值: ⭐⭐⭐⭐ 对多维 Transformer 的分布式训练/推理具有很高的实用价值，尤其在视频生成和多模态大模型快速发展的当下

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[ICML 2026\] AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training](../../ICML2026/others/amdp_asynchronous_multi-directional_pipeline_parallelism_for_large-scale_models_.md)
- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[ECCV 2024\] Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](../../ECCV2024/others/superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)
- [\[ACL 2025\] Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference](../../ACL2025/others/bregman_conditional_random_fields_sequence_labeling_with_parallelizable_inferenc.md)

</div>

<!-- RELATED:END -->
