---
title: >-
  [论文解读] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models
description: >-
  [ICML2025][模型压缩][KV Cache 压缩] 提出梯形（ladder-shaped）KV 缓存模式，在不同层保留不同 token 范围的 KV 状态，从而在固定缓存预算下扩展可捕获的上下文跨度，并通过迭代压缩机制支持无限长度的连续生成。 LLM 自回归解码时需缓存所有历史 token 的 Key-Value 状…
tags:
  - "ICML2025"
  - "模型压缩"
  - "KV Cache 压缩"
  - "长上下文推理"
  - "梯形缓存模式"
  - "无训练优化"
  - "LLM 高效推理"
---

# LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

**会议**: ICML2025  
**arXiv**: [2507.14204](https://arxiv.org/abs/2507.14204)  
**代码**: [GATECH-EIC/LaCache](https://github.com/GATECH-EIC/LaCache)  
**领域**: 模型压缩  
**关键词**: KV Cache 压缩, 长上下文推理, 梯形缓存模式, 无训练优化, LLM 高效推理  
**作者**: Dachuan Shi, Yonggan Fu, Xiangchi Yuan, Zhongzhi Yu, Haoran You, Sixu Li, Xin Dong, Jan Kautz, Pavlo Molchanov, Yingyan (Celine) Lin

## 一句话总结

提出梯形（ladder-shaped）KV 缓存模式，在不同层保留不同 token 范围的 KV 状态，从而在固定缓存预算下扩展可捕获的上下文跨度，并通过迭代压缩机制支持无限长度的连续生成。

## 研究背景与动机

LLM 自回归解码时需缓存所有历史 token 的 Key-Value 状态，内存开销随序列长度线性增长，在长上下文场景下极易触发 OOM。现有 KV cache 压缩方法在**长距离建模能力**和**连续生成（不 OOM）**之间难以兼顾：

- **Recency-based（StreamingLLM）**：固定滑动窗口 $\mathcal{O}(1)$ 内存，可无限生成但丢失远距离信息，准确率下降严重。
- **Retrieval-based（Quest）**：保留全部 KV cache 再按需检索，准确率高但内存 $\mathcal{O}(T)$，长序列最终 OOM。
- **Importance-based（H2O）**：依赖 attention map 选择重要 token，与 FlashAttention 不兼容，实际推理吞吐量有限。

核心矛盾：**在不访问 attention map 的前提下，如何用固定大小的缓存尽可能保留更多历史 token 信息？**

## 方法详解

LaCache 包含两个核心组件：**梯形 KV 缓存模式** 和 **迭代压缩机制**。

### 1. 梯形 KV 缓存模式（Ladder-Shaped Pattern）

**关键洞察**：近期 token 的信息虽重要，但其 KV 状态不必在所有层都保留——不同层可以维护**不同 token 集合**的 KV cache。这样在同样总预算下可覆盖更多 token。

具体做法：在浅层保留较早 token 的 KV 状态，在深层逐步切换到较新 token，形成阶梯状结构：

- 第 1~S 层保留 token 1~N₁
- 第 S+1~2S 层保留 token N₁-O+1~N₂
- 以此类推，逐级右移

两个关键超参数：

| 参数 | 含义 | 影响 |
|------|------|------|
| **Span $S$** | 同一 token 被保留的连续层数 | $S$ 越大→同一 token 信息在更多层流动→准确率↑，缓存开销↑ |
| **Overlap $O$** | 相邻阶梯段的 token 重叠数 | $O$ 越大→相邻段过渡更平滑→信息保持更稳定，存储效率↓ |

**理论分析**：

- 梯形模式通过将覆盖范围尽可能均匀分配到每一层，提升了**信息保留的下界**（最坏情况下，若重要 token 出现在覆盖最少的层，不均匀分配会导致更大精度损失）。
- 相邻 token 在自然语言中语义相关性高，阶梯式的平滑过渡与部分重叠保证了较老 token 的信息渐进淡出，而非突然截断。

作者随机采样 1500+ 种缓存模式，验证了 LaCache 的梯形模式位于 PPL-Cache Size 的 Pareto 最优边界。

### 2. 迭代压缩机制（Iterative Compaction）

当已压缩的 KV cache 填满后，再次对其施加梯形模式压缩：

1. 旧 token 的 KV 状态被更激进地丢弃（位于阶梯左侧，优先被裁剪）
2. 新 token 的 KV 状态保留更多（位于阶梯右侧）
3. 释放的空间分配给新入 token

这相当于基于 token 距离的动态压缩：**越老的 token 被压缩越多、越新的 token 被压缩越少**。整体内存保持恒定 $\mathcal{O}(1)$，支持任意长度的连续生成。

### 与 FlashAttention 兼容

LaCache 不依赖 attention map，物理上只是对 KV tensor 做裁剪和重排，可直接嵌入 FlashAttention 流水线，无需修改 attention kernel。

## 实验设置与主要结果

### 实验设置

- **模型**：Llama2-7B/13B, Llama2-7B/13B-Chat, Llama3-8B, Llama3.2-3B-Instruct, SmolLM2-1.7B-Instruct, LongChat-7b-v1.5
- **基线**：Full Cache, StreamingLLM, H2O, TOVA, PyramidInfer, SnapKV
- **数据集**：Wikitext-2, PG19, LongBench (21 个子集), Needle-In-A-Haystack, RULER

### 语言建模（Wikitext-2, PPL↓）

| 模型 | 方法 | Cache | 1K | 2K | 4K | 8K |
|------|------|-------|-----|-----|-----|-----|
| Llama2-7B-Chat | Full | 100% | 4.94 | 5.32 | 6.52 | nan |
| | StreamingLLM | 512 | 6.67 | 7.41 | 7.95 | 8.97 |
| | **LaCache** | 512 | **5.20** | **6.01** | **7.06** | **8.35** |
| Llama3-8B | Full | 100% | 4.28 | 4.39 | 5.82 | 6.16 |
| | StreamingLLM | 512 | 5.46 | 5.33 | 6.73 | 6.99 |
| | **LaCache** | 512 | **4.61** | **4.89** | **6.40** | **6.78** |

在 Llama2-7B-Chat 上 cache=512 时，LaCache 相对 full cache 仅降低约 5% PPL，而 StreamingLLM 降低约 35%。

### 超长序列（PG19, 600K~10M tokens）

- Llama3-8B full cache 在 160K 后 OOM；LaCache 支持到 600K 且 PPL 保持合理。
- 在完整 PG19（1000 万 token）上，LaCache 始终优于 StreamingLLM。

### 极端小缓存（Llama3-8B, cache=80, 即 1% 训练长度）

| 解码长度 | 1K | 4K | 16K | 64K | 128K |
|----------|-----|-----|------|------|------|
| StreamingLLM | 7.28 | 8.31 | 8.88 | 9.94 | 15.68 |
| **LaCache** | **7.13** | **7.99** | **8.46** | **9.53** | **15.08** |

即使只用 80 个 KV slot，LaCache 仍稳定优于 StreamingLLM。

### 长上下文理解（LongBench, 21 个子集平均分）

| 模型 | Full | StreamingLLM 50% | LaCache 50% | StreamingLLM 25% | LaCache 25% |
|------|------|-------------------|-------------|-------------------|-------------|
| Llama2-7B-Chat | 29.08 | 26.56 | **27.34** | 25.41 | **25.68** |
| Llama2-13B-Chat | 30.69 | 28.30 | **29.22** | 26.82 | **27.04** |

50% 预算下 LaCache 比 StreamingLLM 平均高 ~0.8 分；在 HotpotQA、DuReader、MultiFieldQA 等需要长距离检索的子集上优势更明显（HotpotQA: 32.62 vs 29.98）。

### 与 FlashAttention 的效率对比

由于 H2O 需要显式计算 attention map 无法与 FlashAttention 集成，LaCache 在相同精度下实测吞吐量显著更高。

## 亮点与洞察

1. **跨层差异化缓存**的思路很巧妙——打破了"所有层保留相同 token"的隐含假设，用阶梯错位覆盖更多历史 token，是一种免训练的"以空间换跨度"策略。
2. **迭代压缩**设计优雅：将同一个梯形模式重复施加到已压缩的 cache 上，自动实现"越老压越狠"的效果，无需额外状态管理。
3. **不依赖 attention map**是关键工程优势，使其可无缝接入 FlashAttention 等高效推理框架，实际部署价值高。
4. 1500+ 随机模式搜索验证梯形位于 Pareto 前沿，为设计选择提供了经验支撑。

## 局限与展望

1. **信息保留仅基于位置启发式**：梯形模式假设较新 token 更重要，但某些任务（如多跳推理）中关键信息可能分布在序列任意位置，纯位置策略可能漏掉远距离关键 token。
2. **超参数 $S$ 和 $O$ 需要校准**：不同模型/任务的最优配置可能不同，论文未提供自动调参方案。
3. **评测以 PPL 和 LongBench 为主**，缺少对 generation quality（如摘要的 ROUGE、对话的人类评估）的直接度量。
4. **Batch size 固定为 1**，未展示多 batch 场景下的内存-吞吐量权衡。
5. **与量化/稀疏化等正交技术的组合效果**仅简要提及，未做深入实验。
6. **仅验证了 decoder-only 架构**，对 encoder-decoder 或 MoE 模型的适用性未知。

## 相关工作与启发

- **StreamingLLM**（Xiao et al., 2023）：LaCache 的直接基线，梯形模式可视为 StreamingLLM 滑动窗口的跨层扩展。
- **H2O**（Zhang et al., 2024）：importance-based 方法，精度高但与 FlashAttention 不兼容。
- **Quest**（Tang et al., 2024）：retrieval-based，保留全部 cache 再检索，内存 $\mathcal{O}(T)$。
- **SnapKV / PyramidInfer**：其他 KV cache 压缩方案，各有侧重。
- **启发**：梯形模式的核心洞察——"不同层关注不同 token"——可能可以与 layer-wise pruning 或 early exit 结合，实现更细粒度的计算-精度权衡。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 跨层差异化缓存思路新颖，梯形模式直觉清晰
- 实验充分度: ⭐⭐⭐⭐ — 模型/数据集覆盖广，含极端小 cache 和超长序列实验；缺 generation quality 评估
- 写作质量: ⭐⭐⭐⭐ — 图示清晰，动机-方法-实验组织合理
- 价值: ⭐⭐⭐⭐ — 无训练、兼容 FlashAttention、代码开源，部署门槛低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ICML 2025\] Olica: Efficient Structured Pruning of Large Language Models without Retraining](olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](../../ACL2025/model_compression/efficient_long_context_language_model_retrieval_with_compression.md)

</div>

<!-- RELATED:END -->
