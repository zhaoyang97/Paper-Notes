---
title: >-
  [论文解读] Liger: Linearizing Large Language Models to Gated Recurrent Structures
description: >-
  [ICML2025][物理/科学计算][LLM线性化] Liger 将预训练 Transformer LLM 无额外参数地转换为门控线性循环结构，利用 Key 投影矩阵复用构建门控机制，仅需 0.02% 预训练 token 即可恢复原模型 93% 的性能，同时获得线性时间推理和恒定显存开销。 Transformer 的 so…
tags:
  - "ICML2025"
  - "物理/科学计算"
  - "LLM线性化"
  - "门控线性循环"
  - "线性注意力"
  - "LoRA"
  - "滑动窗口注意力"
---

# Liger: Linearizing Large Language Models to Gated Recurrent Structures

**会议**: ICML2025  
**arXiv**: [2503.01496](https://arxiv.org/abs/2503.01496)  
**代码**: [OpenSparseLLMs/Linearization](https://github.com/OpenSparseLLMs/Linearization)  
**领域**: 模型压缩  
**关键词**: LLM线性化, 门控线性循环, 线性注意力, LoRA, 滑动窗口注意力

## 一句话总结
Liger 将预训练 Transformer LLM 无额外参数地转换为门控线性循环结构，利用 Key 投影矩阵复用构建门控机制，仅需 0.02% 预训练 token 即可恢复原模型 93% 的性能，同时获得线性时间推理和恒定显存开销。

## 研究背景与动机

Transformer 的 softmax 注意力在序列长度上具有 $O(T^2)$ 复杂度，KV Cache 随序列长度线性增长，严重制约长序列推理的速度和显存。线性循环模型（Linear Attention、GLA、Mamba 等）提供 $O(T)$ 训练和 $O(1)$ 推理显存，但从零预训练代价高昂。

**线性化（Linearization）** 是一个新兴方向：将已预训练的 Transformer 权重迁移到线性循环架构，以极低成本获得高效模型。然而现有方法存在两个核心问题：

**架构开销**：SUPRA、LoLCATs 等需要引入额外的特征映射或门控模块，这些模块无法复用预训练权重，必须从零训练

**优化脆弱性**：LoLCATs 采用两阶段训练（先注意力迁移、再 LoRA 微调），流程复杂且无法端到端优化

此外，现有线性化方法忽略了 SOTA 线性循环模型中的**门控机制**（gating mechanism），而门控对于控制记忆保留/遗忘、提升序列建模表达力至关重要。

## 方法详解

### 核心思想：Key 投影复用构建门控

LLM 的参数空间具有结构冗余性。Liger 的核心洞察是：**将 Key 投影矩阵 $\mathbf{W_K}$ 赋予双重角色**——既执行原始的线性变换，又通过无参数的 Pooling 操作派生出门控信号：

$$\mathbf{G}_t = f(\boldsymbol{k}_t) = f(\boldsymbol{x}_t \mathbf{W}_K)$$

其中 $f(\cdot)$ 为 Pooling 操作（如均值池化），无需额外可训练参数。这种参数共享策略确保与预训练权重兼容，同时避免引入辅助门控模块。

### 统一的门控线性循环框架

Liger 将各种门控线性循环结构统一表示为：

$$\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \phi(\boldsymbol{k}_t^\top) \boldsymbol{v}_t$$
$$\boldsymbol{o}_t = \phi(\boldsymbol{q}_t) \mathbf{S}_t$$

其中 $\mathbf{G}_t$ 通过 Pooling($\boldsymbol{k}_t$) 生成，$\phi(\cdot)$ 直接使用 Softmax 归一化（而非学习特征映射）。所有参数 $\mathbf{W_Q}, \mathbf{W_K}, \mathbf{W_V}$ 均继承自预训练 LLM，无需额外模块。

通过不同的门控参数化方式，Liger 可适配多种门控线性循环结构（GLA、Mamba2、mLSTM、HGRN2、RWKV6 等）。

### Liger Attention：层内混合注意力

Liger 提出将**门控循环建模（GRM）**和**滑动窗口注意力（SWA）**在层内加权混合：

$$\boldsymbol{o}_t = \alpha \cdot \text{GRM}(\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t) + \beta \cdot \text{SWA}(\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t)$$

其中 $\alpha + \beta = 1$（默认各 0.5），SWA 窗口大小 $w=64$。GRM 负责全局长程建模，SWA 保留局部 softmax 非线性，总复杂度为 $O(TWD + TD^2)$，保持线性级别。

### Liger 整体架构

- **层内混合**：每层使用 Liger Attention（GRM + SWA）
- **层间混合**：每隔若干层（如 7 层）Liger 块后插入一层标准 softmax 注意力块
- 保留 Pre-Norm、MLP、残差连接等标准组件
- 使用 LoRA（rank=8, alpha=8）对 $\mathbf{W_Q}, \mathbf{W_K}, \mathbf{W_V}$ 进行端到端微调，仅训练 0.085% 参数
- 训练数据：50K 条清洗后的 Alpaca 指令数据（约 0.02B token），训练 2 个 epoch

## 实验关键数据

### 线性化方法对比（Llama-3-8B）

| 模型 | 训练Token(B) | PiQA | ARC-e | ARC-c | HellaSwag | WinoGrande | MMLU | Avg |
|------|------------|------|-------|-------|-----------|------------|------|-----|
| Llama-3-8B (原始) | 15000 | 79.4 | 80.1 | 53.2 | 79.2 | 72.9 | 65.3 | 71.7 |
| SUPRA | 20 | 78.9 | 75.1 | 46.5 | 71.7 | 65.8 | 40.9 | 63.2 |
| LoLCATs (两阶段) | 0.04 | 80.1 | 80.4 | 53.5 | 63.4 | 72.9 | 42.1 | 65.4 |
| **Liger-GLA (Ours)** | **0.02** | **80.3** | **81.1** | **52.5** | **76.3** | **72.0** | **43.4** | **67.6** |

### 与预训练线性模型对比

| 模型 | 训练Token(B) | Avg (no MMLU) |
|------|------------|---------------|
| Mamba-7B | 1200 | 71.0 |
| RWKV-6-7B | 1420 | 69.4 |
| Griffin-7B | 300 | 71.1 |
| Zamba2-7B (Hybrid) | 2100 | 75.3 |
| **Liger-GLA-8B** | **0.02** | **72.4** |

Liger 仅用 0.02B token 就超越了从零训练数百B token 的线性模型。

### 可扩展性分析

| 模型规模 | Llama-3 | Liger-GLA | 恢复率 |
|---------|---------|-----------|--------|
| 1B | 59.9 | 59.0 | 98.5% |
| 3B | 68.1 | 66.5 | 97.7% |
| 8B | 73.0 | 72.4 | 99.2% |

模型越大，性能恢复越好（1B→8B gap 从 4.8% 缩小至 1.8%）。

### 消融实验

| 变体 | PPL↓ | Avg (no MMLU)↑ |
|------|------|----------------|
| Liger-GLA (完整) | 2.96 | 72.4 |
| 随机初始化Gate | 3.16 | 68.8 |
| 去掉SWA | 3.75 | 60.2 |
| 去掉LoRA | 3.23 | 68.1 |
| 纯线性注意力(无Gate) | 3.00 | 71.5 |
| 额外特征映射模块 | 9.04 | 40.2 |

SWA 贡献最大（去掉后性能下降 12.2 点）；引入额外特征映射反而严重损害性能。

## 亮点与洞察

- **零额外参数**：利用 Key 投影的 Pooling 构建门控，完全复用预训练权重，是最简洁的线性化方案
- **极低成本**：仅需 0.02B token（0.02% 预训练量），单卡 A800 即可完成线性化
- **统一框架**：一套方法可适配 GLA、HGRN2、GSA 等多种门控循环结构
- **线性推理**：线性化后的模型具有 $O(T)$ 解码延迟和恒定显存，32K 序列时优势显著
- **Liger Attention 的巧妙设计**：SWA 保留了局部 softmax 非线性信息，GRM 负责全局建模，二者互补

## 局限与展望

- 虽然 Avg(no MMLU) 恢复到 99%，但 **MMLU 恢复率较低**（43.4 vs 65.3），知识密集型任务仍有明显差距
- 滑动窗口大小 $w=64$ 为固定超参数，未探索动态或自适应窗口
- 仅在 Llama-3 和 Mistral 上验证，未涉及更新的架构（如 Qwen、Gemma）
- 训练数据仅用 Alpaca 指令集，数据质量和多样性可能限制恢复上限
- 长上下文任务（如 128K）的实际表现未充分评估

## 评分
- 新颖性: ⭐⭐⭐⭐ — Key投影复用构建门控的思路简洁优雅，避免额外参数
- 实验充分度: ⭐⭐⭐⭐ — 多模型规模、多结构变体、效率分析、消融齐全
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，框架图直观
- 价值: ⭐⭐⭐⭐ — 为LLM部署提供低成本线性化路径，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](large_language_models_to_diffusion_finetuning.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](../../ICML2026/physics/softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](../../NeurIPS2025/physics/encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)
- [\[ICML 2025\] Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)

</div>

<!-- RELATED:END -->
