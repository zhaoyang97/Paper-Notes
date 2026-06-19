---
title: >-
  [论文解读] AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism
description: >-
  [ICML 2025][LLM推理][自适应层并行] AdaDecode 通过在中间层训练轻量级 LM Head 实现高置信度的 token 早期预测，将后续层的 KV cache 计算延迟并行化执行，在保证与标准自回归解码完全一致输出的同时，实现最高 1.73× 的解码吞吐量加速。 领域现状：大模型的自回归解码是逐 tok…
tags:
  - "ICML 2025"
  - "LLM推理"
  - "自适应层并行"
  - "早期退出"
  - "轻量级LM Head"
  - "推测解码"
  - "解码加速"
---

# AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism

**会议**: ICML 2025  
**arXiv**: [2506.03700](https://arxiv.org/abs/2506.03700)  
**代码**: [https://github.com/weizhepei/AdaDecode](https://github.com/weizhepei/AdaDecode)  
**领域**: LLM推理加速 / 高效解码  
**关键词**: 自适应层并行, 早期退出, 轻量级LM Head, 推测解码, 解码加速

## 一句话总结
AdaDecode 通过在中间层训练轻量级 LM Head 实现高置信度的 token 早期预测，将后续层的 KV cache 计算延迟并行化执行，在保证与标准自回归解码完全一致输出的同时，实现最高 1.73× 的解码吞吐量加速。

## 研究背景与动机

**领域现状**：大模型的自回归解码是逐 token 串行生成的，每个 token 必须等前一个生成完毕才能开始，这严重限制了现代 GPU 并行计算能力的发挥。随着模型规模膨胀（数十亿到万亿参数）和长链式推理（long CoT）等应用的兴起，解码延迟成为关键瓶颈。

**现有痛点**：目前主流加速方法有两类：
   - **推测解码（Speculative Decoding）**：需要额外的 drafter 模型，增加显存开销，且 drafter 必须与主模型共享 tokenizer 和词表，实际部署受限。
   - **跳层（Layer Skipping）**：跳过部分层减少计算，但跳过的层不会计算 KV cache，导致后续 token 预测出现不一致，输出质量下降。

**核心矛盾**：加速解码的需求与输出一致性保证之间的矛盾——现有方法要么需要额外模型资源，要么牺牲输出质量。

**本文目标**
   - 如何在不引入辅助模型、不修改原模型参数的前提下加速解码？
   - 如何保证加速后的输出与标准自回归解码完全一致？

**切入角度**：作者观察到许多 token（特别是简单、高可预测性 token）在中间层就能被准确预测，后续层对预测结果几乎不产生改变。问题在于原始 LM Head 无法直接利用中间层表示做预测（因为它只在最后一层训练过）。

**核心 idea**：在中间层引入轻量级 LM Head 实现高置信度早期预测 token，将剩余层的计算延迟到后续 token 处理时并行完成，并通过验证步骤确保输出一致性。

## 方法详解

### 整体框架

AdaDecode 的整体 pipeline 如下：

- **输入**：用户 prompt，先通过标准 prefill 阶段初始化 KV cache
- **解码循环**：对每个要生成的 token，从第 1 层开始逐层处理：
  1. 在每个装有轻量级 LM Head 的中间层，计算当前 token 的下一 token 预测概率
  2. 若概率超过阈值 γ，则立即接受预测并开始处理下一个 token，当前 token 的剩余层计算被延迟
  3. 延迟的计算在后续 token 到达该层时**并行**执行（多个 token 在同一层一起算）
  4. 当所有延迟计算完成（到达最后一层），对所有早期预测的 token 做**验证**
  5. 验证通过则保留，验证失败则回退并从正确分布重采样
- **输出**：与标准自回归解码一致的 token 序列

关键洞察是：通过"纵向"在层维度上并行化多个 token 的计算，打破了自回归解码的串行瓶颈。

### 关键设计

1. **轻量级中间层 LM Head**

    - 功能：在选定的中间层（如第 8、16、24 层）引入可训练的 LM Head，用于预测下一个 token
    - 核心思路：将中间层 LM Head 的权重矩阵 $E^{(i)}$ 分解为 $E^{(i)} = E^* \cdot T^{(i)}$，其中 $E^*$ 是最后一层冻结的 LM Head 权重，$T^{(i)} \in \mathbb{R}^{d \times d}$ 是可学习的变换矩阵。由于 $d \ll |V|$（模型维度远小于词表大小），学习 $T^{(i)}$ 比直接学习完整 LM Head 参数高效得多。在 Llama3.1-8B 上，每个轻量级 LM Head 仅 16M 参数，而完整 LM Head 需要 0.5B 参数，节省 31×。
    - 训练目标：最小化中间层预测分布与最终层预测分布之间的 KL 散度 $\mathcal{L}(\theta^{(i)}) = \text{KL}(p^*(t|h^*) \| p_{\theta^{(i)}}(t|h^{(i)}))$，原模型参数完全冻结
    - 设计动机：中间层表示已包含充分的 token 预测信息，只是原始 LM Head 无法提取。轻量级变换矩阵可有效"对齐"中间层表示到最终层的预测空间，同时保持极低参数量。3 个 LM Head 总共仅 48M 参数。

2. **自适应层并行（Adaptive Layer Parallelism）**

    - 功能：根据中间层预测置信度动态决定在哪一层提前预测 token，并将未完成的计算延迟并行化
    - 核心思路：在中间层 $l^{(i)}$ 处，如果预测概率 $p_{\theta^{(i)}}(t_{i+1}|h^{(i)}) > \gamma$（阈值超参数），则立即接受该预测，开始处理下一个 token $t_{i+1}$。当前 token $t_i$ 在 $l^{(i)}$ 之后的所有层的 KV cache 计算被加入延迟队列 $\mathcal{P}$，在后续 token 到达这些层时一并并行计算
    - 与固定深度早期退出的区别：不同 token 可以在不同层退出（自适应），允许更灵活的层并行调度。例如 $t_2$ 在第 16 层退出而 $t_3$ 在第 8 层退出，它们的剩余计算可以在更深层合并并行处理
    - 设计动机：自适应机制确保简单 token（如停用词、常见短语补全）尽早退出，而复杂 token 仍走完整模型层，平衡了速度和预测质量

3. **验证与回退机制**

    - 功能：在所有延迟计算完成后，验证早期预测的 token 是否与标准自回归解码结果一致
    - 核心思路：使用修改版拒绝采样方案——对于早期预测的 token $t$，以概率 $\min\{1, \frac{p^*(t'|h^*)}{p_{\theta^{(i)}}(t'|h^{(i)})}\}$ 接受。被拒绝的 token 从调整后的分布 $\text{Normalize}(\max(0, p^* - p_{\theta^{(i)}}))$ 重采样替换，并清除其后所有 token 的 KV cache 从该 token 重新开始生成
    - 设计动机：验证步骤确保 AdaDecode 的输出与标准自回归解码完全一致（output parity），实际上拒绝率约 5-6%（$\gamma=0.85$ 时），计算浪费极少

### 训练策略

- 训练数据：使用目标任务的训练集（domain-specific），实验证明领域特定的 LM Head 比混合领域训练效果更好
- 训练过程：仅训练 3 个轻量级变换矩阵 $T^{(i)}$（共 48M 参数），原模型 8B 参数完全冻结
- 优化目标：KL 散度损失，收敛快且稳定

## 实验关键数据

### 主实验：解码吞吐量对比

| 模型 | 方法 | 文本摘要 (XSum) | 代码生成 (HumanEval) | 数学推理 (GSM8K) |
|------|------|:-:|:-:|:-:|
| Llama3.1-8B-Inst | Vanilla | 33.31 tok/s (1.00×) | 32.58 tok/s (1.00×) | 33.13 tok/s (1.00×) |
| | SpecDecode (w/ Llama3.2-1B) | 35.64 (1.07×) | 46.26 (1.42×) | 45.38 (1.37×) |
| | **AdaDecode** | **38.09 (1.14×)** | **49.21 (1.51×)** | **49.17 (1.48×)** |
| CodeLlama-13B-Inst | Vanilla | 27.80 (1.00×) | 27.55 (1.00×) | 28.25 (1.00×) |
| | SpecDecode | 26.97 (0.97×) | 29.75 (1.08×) | 28.53 (1.01×) |
| | Self-SpecDecode | 28.63 (1.03×) | 31.40 (1.14×) | 31.64 (1.12×) |
| | LookAhead | 33.08 (1.19×) | 41.04 (1.49×) | 38.42 (1.36×) |
| | SWIFT | 30.02 (1.08×) | 36.64 (1.33×) | 30.51 (1.08×) |
| | **AdaDecode** | **37.99 (1.37×)** | **46.78 (1.69×)** | **44.28 (1.57×)** |
| CodeLlama-34B-Inst | Vanilla | 17.68 (1.00×) | 18.91 (1.00×) | 19.16 (1.00×) |
| | SpecDecode (w/ 7B) | 19.09 (1.08×) | 26.66 (1.41×) | 24.14 (1.26×) |
| | LookAhead | 20.15 (1.14×) | 26.28 (1.39×) | 27.01 (1.41×) |
| | SWIFT | 21.92 (1.24×) | 26.47 (1.40×) | 25.29 (1.32×) |
| | **AdaDecode** | **24.35 (1.38×)** | **32.78 (1.73×)** | **30.68 (1.60×)** |

### 消融实验

| 配置 | 新增 Head 数 | 新增参数 | 一致性比率 | 加速比 |
|------|:-:|:-:|:-:|:-:|
| **AdaDecode (完整)** | 3 | 48M | 0.996 | **1.51×** |
| w/o verification（去掉验证） | 3 | 48M | 0.652 | 1.64× |
| w/ fixed-layer（固定层退出） | 1 | 16M | 0.996 | 1.37× |
| w/ original LM head（原始头） | 0 | 0M | 0.995 | 0.84× |
| w/ mixed-domain head（混合域训练） | 3 | 48M | 0.998 | 1.29× |
| w/ full-parameterized head（完整参数头） | 3 | 1.5B | 0.997 | 1.49× |

### 关键发现

- **验证步骤不可或缺**：去掉验证后虽然加速比从 1.51× 升至 1.64×，但一致性比率从 0.996 暴跌至 0.652，输出完全不可靠
- **自适应层退出优于固定层退出**：自适应机制 (1.51×) 明显优于固定层 (1.37×)，说明不同 token 需要不同深度的处理
- **原始 LM Head 用于中间层预测反而减速**：因为原始头在中间层只能产生极低置信度预测，几乎无法触发早期退出，反而增加了计算开销 (0.84×)
- **领域特定训练效果更好**：混合域训练的 LM Head (1.29×) 显著弱于领域特定训练 (1.51×)
- **轻量级头 vs 完整参数头**：轻量级头 (48M) 与完整参数头 (1.5B) 的加速效果几乎相同 (1.51× vs 1.49×)，证明参数分解策略的有效性
- **阈值 γ 鲁棒性好**：在 0.4-0.9 范围内加速比变化不大，γ=0.85 时早期预测拒绝率仅约 6%

## 亮点与洞察

- **轻量级 LM Head 的参数分解设计**：通过 $E^{(i)} = E^* \cdot T^{(i)}$ 将中间层 LM Head 分解为固定嵌入 × 可学习变换，参数量减少 31×，效果却与完整参数化头持平。这个设计利用了一个深刻洞察：中间层到最终层之间的映射本质上是低维变换，不需要重新学习整个词表嵌入。
- **"纵向并行化"范式**：不同于推测解码的"横向"（跨时间步）加速，AdaDecode 在"纵向"（跨层）实现并行化，多个 token 在不同层的计算可以合并为一次批处理，充分利用 GPU 的并行计算能力。这种思路可以迁移到任何深层序列模型的加速场景。
- **零修改保证输出一致**：不改模型参数、不改架构、不需辅助模型，仅添加极少额外参数就能保证与标准解码一致输出，这在工程部署上非常友好。

## 局限与展望

- **领域特定训练限制泛化性**：消融实验显示混合域训练的 LM Head 效果显著下降，意味着每换一个下游任务就需要重新训练中间层头部，增加部署和维护成本
- **加速比瓶颈**：最高 1.73× 的加速比相比推测解码的某些场景（如 Medusa 的 2-3×）仍有差距，尤其在文本摘要等较难预测的任务上加速比仅 1.14×-1.38×
- **模型规模范围**：仅在 8B-34B 模型上验证，对于更大规模（70B+）或更小规模（3B 以下）模型的效果未知
- **与 tree-based speculative decoding 的结合**：作者在附录中初步探索了与树结构推测解码的结合，但未深入，这是一个有潜力的方向
- **LM Head 层数的选择**：论文选择了 3 个中间层放置 LM Head，但最优的层数和位置选择策略可进一步研究

## 相关工作与启发

- **vs Speculative Decoding**：推测解码需要额外的 drafter 模型，内存开销大且要求 tokenizer 兼容；AdaDecode 无需辅助模型，但加速比上限可能较低。两者的加速方向正交（横向 vs 纵向），理论上可以组合。
- **vs LayerSkip**：LayerSkip 通过跳过固定层来加速但无法保证输出一致性；AdaDecode 虽然也利用中间层预测但会延迟完成所有层的计算并验证，保证了输出一致。
- **vs EESD**：EESD 也结合了早期退出和验证，但仅支持固定深度退出且需要完整 LM Head（0.7B 参数），而 AdaDecode 支持自适应深度退出且仅需 48M 参数。
- **vs Medusa/EAGLE**：Medusa 和 EAGLE 通过多头/自回归方式实现横向加速（一次预测多个 token），与 AdaDecode 的纵向加速互补，可结合使用。

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应层并行是对早期退出和推测解码的巧妙融合，轻量级 LM Head 分解设计有新意
- 实验充分度: ⭐⭐⭐⭐ 覆盖 3 种任务、3 种模型规模、详细消融和超参数分析，但缺少更大模型和更多任务的验证
- 写作质量: ⭐⭐⭐⭐⭐ 方法动机清晰，与相关工作的对比表格直观，图示说明到位
- 价值: ⭐⭐⭐⭐ 实用性强、部署友好，但加速比上限和领域特定训练的限制降低了一些通用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](../../ICLR2026/llm_reasoning/fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)
- [\[NeurIPS 2025\] ARM: Adaptive Reasoning Model](../../NeurIPS2025/llm_reasoning/arm_adaptive_reasoning_model.md)
- [\[NeurIPS 2025\] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification](../../NeurIPS2025/llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)
- [\[ACL 2026\] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](../../ACL2026/llm_reasoning/reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](../../NeurIPS2025/llm_reasoning/sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)

</div>

<!-- RELATED:END -->
