---
title: >-
  [论文解读] Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models
description: >-
  [ICLR 2026][LLM效率][推理模型] 通过 1700+ 组实验系统证明：非推理模型上"4-bit 量化是显存最优"的结论在推理模型上失效——显存最优策略由模型的**有效尺寸**（参数量×位宽）决定，存在"8-bit 4B"这一临界点，小模型应把显存花在更大权重上、大模型应花在更长生成/更多采样上。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "推理模型"
  - "KV cache 压缩"
  - "权重量化"
  - "test-time scaling"
  - "显存-精度权衡"
  - "Pareto 前沿"
---

# Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=b6qQmQ2F13](https://openreview.net/forum?id=b6qQmQ2F13)  
**代码**: [https://github.com/krafton-ai/not-all-bits-are-equal](https://github.com/krafton-ai/not-all-bits-are-equal)  
**领域**: LLM 推理效率 / 显存优化  
**关键词**: 推理模型, KV cache 压缩, 权重量化, test-time scaling, 显存-精度权衡, Pareto 前沿  

## 一句话总结
通过 1700+ 组实验系统证明：非推理模型上"4-bit 量化是显存最优"的结论在推理模型上失效——显存最优策略由模型的**有效尺寸**（参数量×位宽）决定，存在"8-bit 4B"这一临界点，小模型应把显存花在更大权重上、大模型应花在更长生成/更多采样上。

## 研究背景与动机

**领域现状**：以往针对非推理 LLM 的显存-性能权衡研究几乎只关注**压缩权重**，因为权重显存远大于 KV cache，由此得出"4-bit 量化是跨尺度的显存最优选择"这一近乎普适的结论（Dettmers & Zettlemoyer 2023）。

**现有痛点**：现代推理模型生成的 token 量大幅增加，KV cache 随生成长度线性膨胀，反而可能成为显存瓶颈。论文给出一个反直觉的数字：Qwen3-4B 用 4-bit 权重只占 2.49 GB，但其 32k token 生成的 KV cache 需要 4.42 GB（≈1.8× 权重）；在 batch 推理下权重被摊薄，聚合 KV cache 更是主导项。

**核心矛盾**：当 KV cache 成为显存主要成分时，为非推理模型建立的那套量化经验是否还成立？推理模型额外引入了三个显著影响显存-精度权衡的新维度——**生成长度、并行采样、KV cache 压缩**，但它们与传统的模型尺寸、权重精度如何联合权衡，无人系统研究过。

**本文目标**：在**固定显存预算**下，回答"如何在 模型尺寸 / 权重精度 / token 预算 / 并行采样组大小 / KV cache 压缩 这五个旋钮之间权衡，以最大化推理性能"。

**核心 idea**：**【经验律 + 临界点】** 显存最优策略不是 scale-agnostic 的，而是被模型**有效尺寸**（effective size = 参数量 × 每权重位宽，正比于权重显存）主导，存在一个"8-bit 4B（≈4.2 GB）"的临界阈值，跨越它策略发生质变。

## 方法详解

### 整体框架
本文不提新方法，而是一个**受控经验研究**：把推理模型的显存开销拆成权重和 KV cache 两部分 $M = M_{\text{weights}}(N, P_W) + M_{\text{kv}}(N, \pi_{kv}, T, G)$，然后系统扫描五个因子——参数量 $N$、权重精度 $P_W \in \{4,8,16\}$、token 预算 $T$（2k–30k）、采样组大小 $G$（1–16，多数投票）、KV 压缩策略 $\pi_{kv}$（驱逐 / 量化）——在四个 benchmark 上画出"精度 vs 总显存"的 Pareto 前沿，从前沿上各配置的构成反推出部署准则。主力模型是 Qwen3 全家族（0.6B–32B），并用 DeepSeek-R1-Distill、OpenReasoning-Nemotron 验证泛化，共覆盖 1700+ 场景。

### 关键设计

**1. 有效尺寸（effective size）作为统一标尺：把"参数量"和"位宽"折成一个量。** 论文刻意区分两个概念：`model size` 指参数量 $N$，而 `effective size` / `scale` 指权重显存 $M_{\text{weights}} \approx N \cdot P_W$。所有 scale-dependent 结论都以"8-bit 4B（≈4.2 GB）的有效尺寸"为临界点表述，而非以参数量表述——这样 4B-16bit、8B-8bit、14B-4bit 等不同配置能被放到同一根轴上比较，使"小模型 vs 大模型"的策略分界有了一个可计算、可跨配置迁移的判据。

**2. 串行 scaling 下的显存分配律：小模型加权重、大模型加 token。** 固定 $G=1$、KV 全保留，用 **budget forcing**（模型想停时追加 `Wait` 续写，到预算时注入 `**Final Answer**`）控制生成长度，扫 $T$ 从 2k 到 30k。结论：对有效尺寸**低于** 8-bit 4B 的模型，把省下的显存投到更大权重比投到更长生成更划算——例如 1.7B-8bit 用 6k token 就胜过 0.6B-8bit 用 18k token，4B-4bit 用 10k token 超过 1.7B-8bit 用 18k token；而且大有效尺寸的配置端到端**延迟还更低**（延迟由 token 预算主导），是严格占优。对有效尺寸**达到/超过** 8-bit 4B 的模型则相反——延长生成到饱和才是更省显存的提精度方式（>10 GB 预算下前沿配置的 token 预算稳定在 20k 以上）。

**3. 权重精度的最优值由任务性质决定，4-bit 不再普适。** 论文发现"4-bit 最优"只在**知识密集型**任务（GPQA-Diamond）成立——这类任务靠参数容量装知识，大 4-bit 模型更省显存。但在**数学推理 / 代码生成**（AIME25、LiveCodeBench、MATH500）上，4-bit 持续显存低效：8B-8bit 一致胜过 14B-4bit，32B-4bit 被 14B-8bit 和 8B-16bit 双重严格压制——与 Dettmers & Zettlemoyer (2023) 的结论直接相反。直觉解释是数学推理依赖权重里的数值精度，激进 4-bit 会破坏它。

**4. 并行 scaling 只对大模型划算，且最优组大小随预算增长。** 引入并行轴 $G>1$（batch 推理下 KV cache 正比于 $G$ 膨胀，换取多数投票精度）。结果同样 scale-dependent：有效尺寸低于 8-bit 4B 时并行 scaling 的所有配置都落在串行 Pareto 前沿之下（不划算）；高于阈值时并行 scaling 推进前沿，且全局前沿上的**显存最优组大小 $G$ 随预算单调增大**（16.4–28.9 GB 区间 $4 \le G < 8$ 最优，>28.9 GB 时 $G \ge 8$）。此外用外部 PRM 做 Best-of-N（ActPRM-X，7B/13.28 GB 固定开销）几乎总是显存低效，自包含的多数投票更优。

**5. KV cache 压缩是必需项，驱逐 vs 量化的选择也看有效尺寸。** 仅压权重不足以达到显存最优——在所有权重精度下，KV 驱逐（R-KV，目标预算 2k/4k/8k）和 KV 量化（HQQ 后端，per-channel 对称 2/4/8-bit，group=64，保留 128 token 全精度残差 buffer）都把 Pareto 前沿推到无压缩 baseline 之外，<10 GB 低显存区收益尤其明显。两者形态不同：量化降低每 token 显存、曲线左移（伴随精度损失）；驱逐设定 KV 显存上限、曲线呈"竖直"上升（显存不变、精度涨）。选择准则：有效尺寸**小于** 8-bit 4B 时驱逐的显存权衡更好，大模型则两者势均力敌。

## 实验关键数据

### 实验规模与设置

| 维度 | 取值 |
|------|------|
| 模型 | Qwen3 0.6B–32B（主力）+ DeepSeek-R1-Distill + OpenReasoning-Nemotron |
| 任务 | AIME25、MATH500（数学）、GPQA-Diamond（知识）、LiveCodeBench（代码） |
| 权重量化 | GPTQ 4/8-bit（+ AWQ、FP8 复现验证） |
| token 预算 | 2k–30k（4k 步长，budget forcing） |
| 并行采样 | 多数投票，$G$ 最多 16 |
| KV 压缩 | 驱逐 R-KV / StreamingLLM；量化 HQQ |
| 场景总数 | 1700+ |
| 评测 | 每实例 32 次生成取平均，temperature 0.6 |

### Qwen3 权重 vs KV cache 显存对比（节选 Table 1，GB）

| 模型 | 4-bit 权重 | 8-bit 权重 | 16-bit 权重 | KV@30k | KV@30k×16 |
|------|-----------|-----------|------------|--------|-----------|
| Qwen3-0.6B | 0.50 | 0.71 | 1.40 | 3.20 | 51.27 |
| Qwen3-4B | 2.49 | 4.19 | 7.49 | 4.12 | 65.91 |
| Qwen3-8B | 5.68 | 8.94 | 15.26 | 4.12 | 65.91 |
| Qwen3-32B | 18.01 | 32.66 | 61.02 | 7.32 | 117.19 |

> 关键观察：小模型的 30k KV cache（3–4 GB）已逼近甚至超过其权重；并行 ×16 时 KV cache（50–117 GB）彻底压倒权重，印证"KV cache 才是推理模型瓶颈"。

### 关键发现（论文 5 条 Finding）
- **Finding 1**：权重 vs KV 的分配策略 scale-dependent——有效尺寸 < 8-bit 4B 加权重，≥ 则加 token 直到饱和。
- **Finding 2**：知识密集任务 4-bit 广泛最优；数学/代码需更高精度，小模型 8-bit 最优、大模型 8/16-bit 都行。
- **Finding 3**：并行 scaling 只对有效尺寸 ≥ 8-bit 4B 划算，且最优 $G$ 随预算增大。
- **Finding 4**：只量化权重不够，压 KV cache 在所有精度下都推进前沿。
- **Finding 5**：有效尺寸 < 8-bit 4B 时 KV 驱逐优于 KV 量化。

## 亮点与洞察
- **把一条被广泛默认的"普适律"证伪并替换成可操作的临界点**：从"4-bit 永远最优"升级为"看有效尺寸跨没跨 8-bit 4B"，是这篇 empirical paper 最大的价值。
- **"effective size"这个标尺很巧**：用 参数量×位宽 折成权重显存，把异构配置统一到一根轴上，使跨配置比较和迁移成为可能。
- **"延长小模型生成是 false economy"的反直觉结论**：常被当作"以延迟换显存"的手段，实测下不仅显存不省、延迟还更高（延迟由 token 预算主导），因此被大有效尺寸严格占优。
- **多旋钮联合而非单点优化**：同时把 模型尺寸/精度/生成长度/采样数/KV 压缩 放进一个 Pareto 框架，给部署者一份"按预算和任务类型查表"的准则，而非单一 trick。

## 局限与展望
- **本质是经验研究、不给普适处方**：作者自陈结论不为所有任务/模型给出具体配置，只给"原则"；临界点"8-bit 4B"是在 Qwen3/R1-Distill/Nemotron 上观察到的，是否对架构差异更大的模型（MoE、不同 attention 变体）严格成立待验证。
- **任务覆盖偏数学/代码/知识三类**：长文档、多轮 agent、工具调用等场景的显存动态可能不同。
- **延迟/吞吐只在附录分析**：正文以显存为唯一优化目标，实际部署还受 latency、throughput、batch 调度约束，主结论在这些约束下是否保持需进一步研究。
- **KV 压缩方法固定**：驱逐用 R-KV、量化用 HQQ，更激进或更新的 KV 压缩方法可能移动临界点。

## 相关工作与启发
- **权重量化**：GPTQ、AWQ、FP8——本文用它们做权重压缩并交叉验证结论不依赖具体方案。
- **KV cache 压缩**：驱逐线（StreamingLLM 滑窗+attention sink、R-KV 冗余感知保留）与量化线（HQQ 在线 per-channel 量化）。
- **test-time scaling**：budget forcing（Muennighoff et al. 2025）做串行延长、多数投票/self-consistency（Wang et al. 2022）做并行扩展、PRM/Best-of-N 做外部验证。
- **启发**：(1) 任何"普适最优配置"结论都该按部署场景（推理 vs 非推理、任务类型、batch 与否）重新审视；(2) 部署推理模型时应先估算"有效尺寸"落在临界点哪一侧，再决定把显存预算投向权重还是 test-time compute；(3) 数学/代码类高精度敏感任务慎用激进 4-bit。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— 不提新方法，但把一条被广泛默认的普适律证伪并替换成可操作的 scale-dependent 临界点，问题框定和 effective-size 标尺都有原创价值。
- **实验充分度**: ⭐⭐⭐⭐⭐ —— 1700+ 场景、3 个模型家族、4 类任务、5 个旋钮联合扫描，并用 AWQ/FP8 交叉验证结论稳健性，经验研究里属于非常扎实的。
- **写作质量**: ⭐⭐⭐⭐ —— 5 条 Finding 提炼清晰、图表围绕 Pareto 前沿组织、结论可直接落地为部署准则，可读性强。
- **价值**: ⭐⭐⭐⭐ —— 给推理模型部署提供了"按预算+任务查表"的实用指南，对工业界显存预算分配有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models](not_all_models_suit_expert_offloading_on_local_routing_consistency_of_mixture-of.md)
- [\[ICLR 2026\] ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](thinkv_thought-adaptive_kv_cache_compression_for_efficient_reasoning_models.md)
- [\[ICLR 2026\] Mixture-of-Experts Can Surpass Dense LLMs Under Strictly Equal Resource](mixture-of-experts_can_surpass_dense_llms_under_strictly_equal_resource.md)
- [\[ICLR 2026\] Sparse Attention Adaptation for Long Reasoning](sparse_attention_adaptation_for_long_reasoning.md)
- [\[ICLR 2026\] Stacked From One: Multi-Scale Self-Injection for Context Window Extension](stacked_from_one_multi-scale_self-injection_for_context_window_extension.md)

</div>

<!-- RELATED:END -->
