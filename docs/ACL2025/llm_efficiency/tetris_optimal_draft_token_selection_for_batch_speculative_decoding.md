---
title: >-
  [论文解读] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding
description: >-
  [ACL 2025][LLM效率][推测解码] Tetris 提出了一种在批量推测解码场景下，跨请求动态选择最优草稿token的方法，通过贪心选择累积接受概率最高的token来最大化有限计算资源下的推理吞吐量。 推测解码（Speculative Decoding, SD）是加速LLM推理的有效方法，通过小型草稿模型快速生成候…
tags:
  - "ACL 2025"
  - "LLM效率"
  - "推测解码"
  - "批量推理"
  - "吞吐量优化"
  - "草稿token选择"
  - "LLM推理加速"
---

# Tetris: Optimal Draft Token Selection for Batch Speculative Decoding

**会议**: ACL 2025  
**arXiv**: [2502.15197](https://arxiv.org/abs/2502.15197)  
**代码**: [GitHub](https://github.com/ZhaoxuanWu/Tetris)  
**领域**: LLM Efficiency  
**关键词**: 推测解码, 批量推理, 吞吐量优化, 草稿token选择, LLM推理加速

## 一句话总结

Tetris 提出了一种在批量推测解码场景下，跨请求动态选择最优草稿token的方法，通过贪心选择累积接受概率最高的token来最大化有限计算资源下的推理吞吐量。

## 研究背景与动机

推测解码（Speculative Decoding, SD）是加速LLM推理的有效方法，通过小型草稿模型快速生成候选token，再由大型目标模型并行验证。然而，现有方法存在以下问题：

1. **固定窗口大小的局限性**：传统SD为每个请求使用相同的草稿窗口大小，但不同请求在不同解码步骤的最优窗口大小差异很大（呈长尾分布），固定窗口无法自适应调整。
2. **单请求优化的不足**：大多数工作仅针对单个请求优化草稿token选择，未考虑多请求批量处理场景下的资源分配问题。
3. **资源浪费问题**：在SD中，一旦某个token被拒绝，其后所有token都必须丢弃（级联失败），这在固定窗口大小下会造成大量计算资源浪费。
4. **服务提供商的实际需求**：LLM推理服务商需要在有限计算容量下最大化总吞吐量，需要跨请求全局优化资源分配。

## 方法详解

### 整体框架

Tetris 在草稿模型和目标模型之间引入一个"管理器"（Manager），该管理器从草稿模型生成的额外草稿token中，贪心选择累积接受概率最高的token组合，送入目标模型验证。核心思想是利用序列内token的级联失败特性与序列间token的独立性，动态为"容易"的请求分配更长的草稿窗口，为"困难"的请求分配更短的窗口。

### 关键设计

1. **跨请求贪心选择算法**：定义每个token (i,j) 的累积接受概率为 ∏p_{i,t}，使用最大堆数据结构，每步从堆中取出累积接受概率最高的token加入验证集合 D*，直到填满计算容量 C。算法时间复杂度为 O(C log N)，并通过GPU的scatter_max操作实现，额外开销不到0.3ms。

2. **序列内级联与序列间独立性**：同一请求内的token具有顺序依赖关系（前一个被拒则后续全部无效），而不同请求间的token相互独立。Tetris利用这一特性，在级联失败风险高时优先选择其他请求的token（并行token），在风险低时继续扩展当前请求的窗口（顺序token）。

3. **额外草稿token机制**：让草稿模型生成超出服务器容量的额外token（通常1-2个），为Tetris提供更多选择空间。实验表明，增加额外草稿token能持续提升验证成功率（VSR）。

### 损失函数 / 训练策略

Tetris 不涉及额外训练，是一种纯推理阶段的优化策略。在实际实现中，由于无法获取真实接受率，使用草稿模型的输出概率作为替代度量。理论上证明了：

- **逐步最优性**（Theorem 1）：给定真实接受率，算法在每个解码步骤产生最优吞吐量。
- **全局最优性**（Theorem 2）：在所有同一序列位置的token具有相同接受率的假设下，Tetris实现全局最优吞吐量。

## 实验关键数据

### 主实验

实验在三种配置下进行：Vicuna-68M/33B、Llama-1B/70B、Llama-1B/405B。

| 数据集 | 指标 | Tetris vs 最佳基线 | Tetris vs 标准SD | 说明 |
|--------|------|-------------------|-----------------|------|
| ShareGPT (Setting 1) | 吞吐量 | +3.50% | +6.70% | Vicuna-68M → 33B |
| Arena (Setting 1) | 吞吐量 | +5.17% | +7.47% | Vicuna-68M → 33B |
| Tough (Setting 3) | 吞吐量 | +5.25% | +5.25% | Llama-1B → 405B |
| Tough (Setting 1) | 端到端延迟 | +5.47% | +9.32% | 延迟降低最显著 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 额外0个草稿token | VSR基线 | 等同标准SD |
| 额外1个草稿token | VSR提升~2-4% | 性价比最优 |
| 额外2个草稿token | VSR进一步提升 | 收益递减 |
| 额外3个草稿token | VSR略有提升 | 因当前顺序pipeline，额外起草时间抵消部分收益 |

### 关键发现

1. 每步最优草稿窗口大小的分布是扁平且长尾的（图2），验证了固定窗口的次优性。
2. 在并行化pipeline下，Tetris的TER指标可实现高达12.04%的吞吐量提升（Setting 3, Tough数据集），远超当前顺序pipeline下的结果。
3. DSD（动态推测解码）在实践中并不总是优于标准SD，可能因为条件接受率估计不准确。
4. Tetris对不同草稿窗口大小的选择具有鲁棒性，在所有配置下一致优于基线。

## 亮点与洞察

- **Tetris形状的直觉**：选出的token呈阶梯状（类似俄罗斯方块），"容易"请求获得更长草稿窗口，形象地展示了资源的自适应分配。
- **理论保证**：提供了逐步和全局最优性的理论证明，而非仅凭经验。
- **即插即用**：无需额外训练，可与现有SD方法和框架（如vLLM）无缝集成。
- **前瞻性设计**：文章指出在并行化pipeline下（如PEARL、Minions），Tetris的草稿和选择时间可完全被隐藏，潜在收益更大。

## 局限与展望

1. **依赖顺序pipeline**：当前vLLM采用顺序pipeline，额外草稿token的生成时间无法隐藏，限制了Tetris的潜力。
2. **接受率估计精度**：使用草稿模型概率作为替代指标，当草稿模型质量较差时可能不够准确。
3. **假设的简化**：全局最优性的理论证明依赖于"同一序列位置的所有token具有相同接受率"的假设。
4. **未考虑请求优先级**：当前方法仅最大化总吞吐量，未考虑不同请求的优先级或公平性。

## 相关工作与启发

- **EAGLE-2 和 MDSD** 也采用了基于草稿模型概率的贪心token选择，但仅在单请求层面操作，Tetris将其推广到批量层面。
- **DSD（刘等, 2024d）**为批量中所有请求自适应确定单一草稿窗口，而Tetris为每个请求独立优化窗口大小，粒度更细。
- 并行化pipeline（Minions, PEARL）的出现将进一步释放Tetris的潜力，是一个值得关注的方向。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 4 |
| 实验充分性 | 4 |
| 工程价值 | 5 |
| 写作质量 | 4 |
| 总分 | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)
- [\[ACL 2025\] A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)
- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](sam_decoding_speculative_decoding_via_suffix_automaton.md)
- [\[ICML 2026\] MineDraft: A Framework for Batch Parallel Speculative Decoding](../../ICML2026/llm_efficiency/minedraft_a_framework_for_batch_parallel_speculative_decoding.md)
- [\[ACL 2025\] CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)

</div>

<!-- RELATED:END -->
