---
title: >-
  [论文解读] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity
description: >-
  [ICLR2026][LLM效率][scaling laws] 系统对比 xLSTM 与 Transformer 的 scaling law，证明 xLSTM 在训练损失-算力 Pareto 前沿、过训练 regime 和推理速度上全面优于同规模 Transformer，且优势随上下文长度增大而增长。
tags:
  - "ICLR2026"
  - "LLM效率"
  - "scaling laws"
  - "xLSTM"
  - "线性复杂度"
  - "Transformer"
  - "推理效率"
---

# xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity

**会议**: ICLR2026  
**arXiv**: [2510.02228](https://arxiv.org/abs/2510.02228)  
**代码**: [NX-AI/xlstm_scaling_laws](https://github.com/NX-AI/xlstm_scaling_laws)  
**领域**: LLM效率  
**关键词**: scaling laws, xLSTM, 线性复杂度, Transformer对比, 推理效率  

## 一句话总结
系统对比 xLSTM 与 Transformer 的 scaling law，证明 xLSTM 在训练损失-算力 Pareto 前沿、过训练 regime 和推理速度上全面优于同规模 Transformer，且优势随上下文长度增大而增长。

## 背景与动机
- Scaling law 是 LLM 设计的核心指导工具（Kaplan 2020, Chinchilla 2022），但现有研究几乎全部聚焦在 Transformer 架构
- xLSTM 等线性复杂度架构已在十亿参数级别展现竞争力（xLSTM 7B），但缺乏与 Transformer 的系统性 scaling 对比
- 传统 FLOP 近似 $C(N,D)=6ND$ 忽略了注意力机制的计算量，无法公平比较线性/二次复杂度模型
- 对推理效率（TTFT、step time）与上下文长度的交互影响也缺乏系统分析

## 核心问题
1. **训练效率**: 给定算力预算，xLSTM 与 Transformer 谁的 loss 更低？
2. **过训练 regime**: xLSTM 在高 token/parameter 比下是否仍保持稳定的 power-law 指数？
3. **上下文长度**: 线性 vs 二次复杂度如何影响 compute-optimal 模型大小？
4. **推理**: 在 TTFT 和 step time 上两种架构如何随上下文长度缩放？

## 方法详解

### 整体框架

本文不提出新模型，而是搭建一套覆盖训练与推理两端的 scaling law 测量管线，回答"线性复杂度的 xLSTM 在 scaling 上到底能不能、什么时候打过 Transformer"。整条管线分三步：先在统一的数据、tokenizer 和**精确算力口径**下，把 Llama-2 风格 dense Transformer 与 xLSTM 7B 架构（纯 mLSTM 层 + MLP）放到 80M–7B 参数、2B–2T tokens 的同一坐标系里大规模扫描；再用**带自由指数的 scaling law 拟合 + IsoFLOP 外推**定位每个算力预算下的 compute-optimal 配置，读出并对比两类架构的 power-law 指数；最后用**基于 roofline 的推理时延建模**把上下文长度对 TTFT 和 step time 的影响纳入同一比较框架。整个研究跑了 672 次训练（292 个 Transformer + 380 个 xLSTM），数据全部来自 DCLM-Baseline 高质量过滤网页文档、用 GPT-NeoX tokenizer、默认序列长度 8192，总算力达 $3.2 \times 10^{23}$ FLOPs。

> 这是一篇 scaling-law/分析型论文，方法是"测量 + 拟合 + 外推"三件分析工具而非可串联的模型流水线，按 skill 规范**跳过框架图**；下面三个关键设计与上面整体框架点名的三步同序对应（精确 FLOP 口径 → scaling law 拟合与外推 → roofline 推理建模）。

### 关键设计

**1. 精确 FLOP 口径：让线性与二次复杂度模型能被公平比较**

传统 scaling 研究用 $C(N,D)=6ND$ 近似算力，这个公式把模型当成纯前馈、完全忽略注意力的二次项。对 Transformer 和 xLSTM 的对比来说这是致命的——前者算力随上下文长度二次增长、后者线性增长，$6ND$ 会系统性低估 Transformer 的真实开销，让比较失真。本文改用逐算子的精确 FLOP 公式，把注意力的二次计算与前馈分开统计，并对 xLSTM 的递归更新、mLSTM 矩阵运算单独建模，从而给两类架构提供一个真正同口径的算力横轴，后续所有 Pareto 前沿和 compute-optimal 结论才站得住。

**2. 带自由指数的 scaling law 拟合与 IsoFLOP 外推：定位每个算力预算下的最优模型**

损失拟合采用 $\hat{L}(N,D) = E + (A N^{-\alpha} + B D^{-\beta})^{\gamma}$，其中 $E$ 是不可约损失、$N$ 是参数量、$D$ 是 token 数，相比 Chinchilla 的固定形式额外引入自由指数 $\gamma$ 来提升对两种架构的拟合质量。在此之上用 IsoFLOP 方法：固定算力预算 $H$，沿不同 $N$/$D$ 组合采样并拟合二阶多项式，找到该预算下的最优 $N^*(H)$、$D^*(H)$，再以幂律 $\hat{N}^*(H)=A'\cdot H^{a}$、$\hat{D}^*(H)=B'\cdot H^{b}$ 外推到更大算力。这套流程让"给定预算该把算力分给更大模型还是更多数据"的问题对两种架构都有可比答案，也使得过训练（高 token/parameter 比）regime 下的 power-law 指数能被直接读出并对比。

**3. 基于 roofline 的推理时延建模：把上下文长度的影响纳入比较**

训练 scaling 之外，推理时延被建模为算力受限或访存受限两种极限：$\tau = \text{FLOPs}_{\text{algo}} / \alpha_{\text{eff}} + \epsilon$ 或 $\tau = \text{Bytes}_{\text{mem}} / \beta_{\text{eff}} + \epsilon$，其中 $\alpha_{\text{eff}}$、$\beta_{\text{eff}}$ 是实测的有效算力/带宽、$\epsilon$ 是固定开销，用 roofline model 判断当前到底卡在计算还是内存上。prefill 与 generation 两阶段分开分析，正是为了刻画 Transformer 的 KV cache 随上下文线性膨胀、而 xLSTM 状态恒定这一结构差异——这让 TTFT 和 step time 对上下文长度的依赖能被理论预测并与实测对照。

## 实验关键数据

### 训练 Scaling

| 发现 | 细节 |
|------|------|
| **Pareto 支配** | xLSTM 在近 5 个数量级算力范围内严格 Pareto-dominate Transformer |
| **过训练指数** | xLSTM 的 power-law 指数 $\eta$ 在 $M=22$ 到 $M=2200$ 范围内保持恒定，与 Transformer 一致 |
| **Compute-optimal 大小** | 相同算力下，xLSTM 最优模型更大（线性运算更便宜→更多参数分配给深度/宽度） |
| **上下文长度影响** | Transformer 在 2048→16384 时最优模型大小显著下降；xLSTM 保持稳定 |

### 推理性能

| 指标 | 16k prefill 结果 |
|------|-----------------|
| **TTFT** | xLSTM 比同尺寸 Transformer 低 30–50% |
| **Step time** | xLSTM 与 prefill 长度无关（常数）；Transformer 线性增长 |
| **极端对比** | 16k prefill 下最大 xLSTM 的 step time < 最小 Transformer 的 step time |

### 通用规律
- Compute-optimal 模型的"loss vs 模型大小"关系在 xLSTM 和 Transformer 间近似落在同一条线上——暗示性能与模型大小存在跨架构的普适关系

## 亮点
- **全面系统性**: 672 次训练覆盖 5 个数量级算力，同时考量训练 + 推理 + 上下文长度
- **精确 FLOP 计算**: 告别 $6ND$ 近似，为线性/二次架构对比提供公平基准
- **实用指导**: 证明 xLSTM 在过训练 regime 下指数稳定，支持"小模型 + 大数据"的实际部署策略
- **推理建模**: 基于 roofline 的理论模型与实测高度吻合

## 局限与展望
- 仅考虑 cross-entropy loss，未评估下游任务（推理、代码、多语言等）
- 未涉及 MoE 或 Attention+xLSTM 混合架构
- 推理实验限于单 GPU，未考量多 GPU 分布式推理场景
- 训练数据仅用 DCLM-Baseline，未验证数据分布变化的影响
- 未探讨 xLSTM 在超长上下文（>16k）下的实际质量表现（如 recall 能力）
- 未与 Mamba、RWKV 等其他线性架构做横向对比

## 与相关工作的对比
- **Chinchilla (Hoffmann 2022)**: 本文复现了 Transformer 的 compute-optimal 指数，并扩展到 xLSTM
- **Gadre 2024 / Sardana 2024**: 本文在过训练 regime 分析上与之一致，但增加了跨架构维度
- **Shen 2024**: 展示线性模型与 Transformer "on par"，本文更进一步证明 xLSTM "优于" Transformer
- **Poli 2024**: 混合架构优于纯 Transformer；本文证明纯线性架构也能胜出
- **Porian 2024**: 本文复现了其 Transformer power-law 指数 $a$

## 启发与关联
- xLSTM 的 Pareto 支配性意味着在同等算力下可获得更好的预训练模型，对资源受限场景特别有价值
- 上下文长度对 compute-optimal 模型大小的影响是一个被广泛忽略的维度，值得在其他架构（Mamba、RWKV 等）中验证
- 推理优势随上下文增长而扩大，暗示在长上下文推理（如 CoT、文档理解）中线性架构潜力巨大
- 跨架构的"模型大小 vs loss"普适关系是一个值得深入研究的理论问题
- 精确 FLOP 计算方法论可直接复用于评估 Mamba、RWKV、RetNet 等其他线性架构的 scaling 行为
- 过训练 regime 指数恒定这一发现为 "小模型多数据" 部署策略提供了理论保障

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个系统性线性复杂度 vs Transformer scaling law 对比
- 实验充分度: ⭐⭐⭐⭐⭐ — 672 次训练、多维度分析、理论+实测推理建模
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表专业
- 价值: ⭐⭐⭐⭐ — 为线性复杂度架构的工程部署提供了重要的 scaling 指导
- 综合: ⭐⭐⭐⭐ — 实验扎实、结论清晰，对架构选型有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs](scaling_laws_meet_model_architecture_toward_inference-efficient_llms.md)
- [\[NeurIPS 2025\] Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](../../NeurIPS2025/llm_efficiency/tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)
- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[ICLR 2026\] Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](towards_greater_leverage_scaling_laws_for_efficient_mixture-of-experts_language_.md)
- [\[ICLR 2026\] Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention for Test-Time Regression](local_linear_attention_an_optimal_interpolation_of_linear_and_softmax_attention_.md)

</div>

<!-- RELATED:END -->
