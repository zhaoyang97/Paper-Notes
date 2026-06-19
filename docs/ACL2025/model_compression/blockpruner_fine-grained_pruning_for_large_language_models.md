---
title: >-
  [论文解读] BlockPruner: Fine-grained Pruning for Large Language Models
description: >-
  [ACL 2025][模型压缩][结构化剪枝] 提出 BlockPruner，将 Transformer 层分解为 MHA 和 MLP 两个最小残差块，基于困惑度评估块重要性并通过迭代搜索进行细粒度剪枝，实现比层级剪枝更优的压缩效果。 1. 领域现状 大语言模型（LLM）规模持续增长，部署成本高昂。模型压缩技术（知识蒸馏、量…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "结构化剪枝"
  - "LLM压缩"
  - "块级冗余"
  - "困惑度"
  - "迭代搜索"
---

# BlockPruner: Fine-grained Pruning for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2406.10594](https://arxiv.org/abs/2406.10594)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 结构化剪枝, LLM压缩, 块级冗余, 困惑度, 迭代搜索  

## 一句话总结

提出 BlockPruner，将 Transformer 层分解为 MHA 和 MLP 两个最小残差块，基于困惑度评估块重要性并通过迭代搜索进行细粒度剪枝，实现比层级剪枝更优的压缩效果。

## 研究背景与动机

### 1. 领域现状

大语言模型（LLM）规模持续增长，部署成本高昂。模型压缩技术（知识蒸馏、量化、剪枝）成为实际部署的重要手段。近期研究发现 LLM 中存在大量冗余层，移除这些层对整体性能影响有限。

### 2. 现有痛点

- 当前层级剪枝方法（如 ShortGPT、LaCo）以整个 Transformer 层为最小剪枝单位，粒度过粗
- 非结构化剪枝虽然能保持较高压缩比下的性能，但需要专用硬件支持，实际加速困难
- 结构化剪枝方法（如 LLM-Pruner、SliceGPT）通常需要剪枝后重新训练模型

### 3. 核心矛盾

层级剪枝忽视了层内部更细粒度的冗余——MHA 和 MLP 块的冗余程度不同，整层移除可能同时丢弃了重要子模块。

### 4. 本文目标

如何在不需要重训练的前提下，以比层级更细的粒度进行 LLM 结构化剪枝？

### 5. 切入角度

利用 Transformer 层的残差结构特性，将每层分解为 MHA 和 MLP 两个最小残差块，分别评估并移除冗余块。

### 6. 核心 idea

将 Transformer 层拆分为 MHA/MLP 块，用困惑度衡量块重要性，通过迭代贪心搜索逐块移除最不重要的块。

## 方法详解

### 整体框架

BlockPruner 包含三个核心步骤：

1. **最小残差块分解**：将每个 Transformer 层拆分为 MHA 和 MLP 两个独立残差块
2. **块重要性评估**：通过困惑度度量每个块的重要性
3. **迭代搜索剪枝**：逐步移除最不重要的块

### 关键设计

#### 最小残差块（Minimal Residual Block）

每个 Transformer 层的计算可分解为两个残差连接：

$$X_i' = \text{MHA}(\text{LN}(X_{i-1})) + X_{i-1}$$

$$X_i = \text{MLP}(\text{LN}(X_i')) + X_i'$$

两个子模块都遵循 $f(x) + x$ 的残差形式，因此可以独立地被"掩码"掉而不破坏信息流。对于 $L$ 层的模型，总共有 $2L$ 个可剪枝的块。

#### 块重要性度量——困惑度

与 ShortGPT 使用的 Block Influence（BI）等**局部**度量不同，BlockPruner 采用**全局**度量——困惑度：

$$\text{PPL} = \exp\left(-\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(w_i|w_{<i})\right)$$

具体做法：逐一掩码每个块 $B_i$，在校准数据集上计算掩码后模型的困惑度 $P_i$。困惑度越低，说明该块越冗余。

#### 迭代搜索算法

不同于一次性移除所有低重要性块，BlockPruner 采用**迭代贪心搜索**：

1. 对所有 $2L - j + 1$ 个剩余块，分别掩码并计算困惑度
2. 移除困惑度最低的块
3. 重复直到达到目标剪枝数 $K$

这种迭代方式考虑了块之间的交互效应。消融实验表明，去掉迭代搜索后性能严重下降（Baichuan2-7B 下降 30.80%）。

### 损失函数/训练策略

BlockPruner 是 **training-free** 方法，不需要任何微调或重训练。仅使用 256 个 Alpaca 数据样本作为校准集。

## 实验关键数据

### 主实验

| 模型 | 方法 | 剪枝比 | PPL↓ | Avg Score |
|------|------|--------|------|-----------|
| Llama2-7B | Dense | 0% | 5.47 | 68.96 |
| Llama2-7B | ShortGPT | 21.02% | 18.45 | 58.18 |
| Llama2-7B | SliceGPT | 21.45% | 30.74 | 57.83 |
| Llama2-7B | **BlockPruner** | 21.99% | **11.51** | **60.17** |
| Llama2-13B | Dense | 0% | 4.89 | 71.72 |
| Llama2-13B | ShortGPT | 24.37% | 20.06 | 62.60 |
| Llama2-13B | **BlockPruner** | 25.12% | **8.16** | **64.53** |
| Qwen1.5-14B | Dense | 0% | 7.44 | 69.07 |
| Qwen1.5-14B | ShortGPT | 22.25% | 1237.21 | 44.72 |
| Qwen1.5-14B | **BlockPruner** | 23.72% | **15.67** | **60.45** |

### 消融实验

| 模型 | 方法 | 剪枝比 | Avg Score |
|------|------|--------|-----------|
| Llama2-7B | BlockPruner | 21.99% | 60.17 |
| Llama2-7B | - 迭代搜索 | 20.95% | 55.89 (-7.11%) |
| Llama2-7B | - 块级→层级 | 21.02% | 58.63 (-2.56%) |
| Baichuan2-7B | BlockPruner | 22.45% | 56.08 |
| Baichuan2-7B | - 迭代搜索 | 22.39% | 38.81 (-30.80%) |
| Qwen1.5-14B | BlockPruner | 23.72% | 60.45 |
| Qwen1.5-14B | - 迭代搜索 | 22.98% | 40.80 (-32.51%) |

### 关键发现

1. **MHA 比 MLP 更冗余**：剪枝比 <17% 时，仅剪 MHA 的性能损失更小；随后 MHA 性能急剧下降
2. **模型越大，冗余越多**：Llama2-13B 比 7B 在相同剪枝比下保持更好性能
3. **Alpaca 比 Wikitext2 更适合做校准集**：指令跟随数据更贴近下游任务分布
4. **256 样本即可**：增加样本数对剪枝效果无显著提升
5. BlockPruner 剪枝后困惑度与下游任务性能正相关

## 亮点与洞察

- **简洁有效的设计**：利用 Transformer 的残差结构天然地将层拆分为两个可独立移除的块，无需引入额外结构
- **全局度量 + 迭代搜索的组合**：困惑度虽非完美的局部度量，但配合迭代搜索可捕捉块间交互，效果远超一次性剪枝
- **MHA/MLP 冗余不对称性**：揭示了 MHA 模块存在更多冗余的现象，对模型设计和压缩策略均有启发
- **无需训练**：相比需要微调的 LLM-Pruner 等方法，BlockPruner 完全免训练，实用性更强

## 局限与展望

1. 困惑度计算的迭代搜索开销较大——每轮需遍历所有剩余块，时间复杂度为 $O(K \cdot 2L)$
2. 未探索与量化等其他压缩技术的组合效果
3. 实验仅覆盖 7B-14B 模型，更大规模模型（70B+）的表现未知
4. 块重要性度量仍有优化空间——困惑度是全局指标，可能存在更精准的块级度量

## 相关工作与启发

- **ShortGPT / LaCo**：层级剪枝方法，BlockPruner 在其基础上进一步细化粒度
- **SliceGPT**：通过计算不变性的权重矩阵列/行剪枝，思路不同但互补
- **FINERCUT**（并行工作）：类似的块级剪枝，但使用 logits 相似度而非困惑度作为重要性度量
- **启发**：残差连接是实现细粒度剪枝的关键结构特性，未来可探索更细粒度（attention head 级别）的剪枝

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 块级剪枝的 idea 直觉清晰，虽非首创但通过困惑度+迭代搜索组合出了有效方案
- **实验充分度**: ⭐⭐⭐⭐ — 6 个模型、5 个基准、详尽的消融实验和分析
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，从初步实验的观察自然引出方法
- **综合价值**: ⭐⭐⭐⭐ — 实用性强，思路简洁，对 LLM 压缩领域有实际参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](../../ICML2025/model_compression/slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ICML 2025\] Instruction-Following Pruning for Large Language Models](../../ICML2025/model_compression/instruction-following_pruning_for_large_language_models.md)
- [\[ICML 2025\] Olica: Efficient Structured Pruning of Large Language Models without Retraining](../../ICML2025/model_compression/olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)

</div>

<!-- RELATED:END -->
