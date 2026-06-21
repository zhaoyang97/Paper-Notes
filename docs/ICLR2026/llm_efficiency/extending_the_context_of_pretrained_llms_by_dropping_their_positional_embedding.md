---
title: >-
  [论文解读] Extending the Context of Pretrained LLMs by Dropping Their Positional Embedding
description: >-
  [ICLR 2026][LLM效率][位置编码] RoPE 在预训练时是加速收敛的关键归纳偏置，却也是阻碍长度外推的根源；本文提出 DroPE：——预训练完成后直接删掉所有位置编码、再用极少 token 短暂"重校准"，即可让 LLM 零样本泛化到远超训练长度的序列，无需任何长上下文微调。 领域现状：Transformer…
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "位置编码"
  - "RoPE"
  - "NoPE"
  - "零样本上下文扩展"
  - "长上下文"
  - "预训练后处理"
---

# Extending the Context of Pretrained LLMs by Dropping Their Positional Embedding

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RlPVSeKjoc](https://openreview.net/forum?id=RlPVSeKjoc)  
**代码**: 已随投稿开源（论文声明，链接待确认）  
**领域**: llm_efficiency / 长上下文外推  
**关键词**: 位置编码, RoPE, NoPE, 零样本上下文扩展, 长上下文, 预训练后处理  

## 一句话总结
RoPE 在预训练时是加速收敛的关键归纳偏置，却也是阻碍长度外推的根源；本文提出 **DroPE**——预训练完成后直接删掉所有位置编码、再用极少 token 短暂"重校准"，即可让 LLM 零样本泛化到远超训练长度的序列，无需任何长上下文微调。

## 研究背景与动机

**领域现状**：Transformer 注意力的二次复杂度使得直接在长序列上预训练代价高昂，因此"零样本上下文扩展"（不做长上下文微调就能用超过训练长度的序列）成为下一代基座模型的核心诉求。主流位置编码 RoPE 通过对 query/key 做相对旋转注入位置信息，已成事实标准。

**现有痛点**：当推理长度超过预训练长度时，RoPE 的旋转相位会落到训练时从未见过的区间（OOD），性能急剧下降。为此涌现了一批 RoPE 频率缩放法（PI、NTK-RoPE、YaRN、LongRoPE2），但它们仍需昂贵的长上下文微调，且开箱即用时无法真正跨远距离检索信息。另一条路线是从头训练无位置编码的 NoPE 架构，但 NoPE 全程性能逊于 RoPE，始终未能普及。

**核心矛盾**：位置编码身处一个根本性的两难——它在预训练阶段提供强归纳偏置、显著加速收敛（好处），但模型对这种显式位置信息的"过度依赖"恰恰是阻碍其外推到未见长度的元凶（坏处）。RoPE 缩放法试图缝合这道裂缝，却必然要压缩低频，从而扭曲负责语义匹配的注意力头。

**本文目标**：能否只在预训练阶段享用位置编码的归纳偏置，事后再把它丢掉，从而同时拿到"训得快"和"外推好"两边的好处？

**核心 idea（先用后弃）**：本文给出肯定回答。RoPE 是一种**瞬态但关键的训练归纳偏置**——预训练时靠它快速建立位置感知，训练完成后把它整体删除，再用一小撮原长度 token 做短暂重校准，模型既不丢失原有能力，又解锁了强大的零样本长度泛化。

## 方法详解

### 整体框架
DroPE（Dropping Positional Embeddings）不改变常规训练流程，而是在一个已经用 RoPE 预训练好的 Transformer 之上做"摘除 + 重校准"：拿到 RoPE checkpoint → 从每一层删除位置编码（退化为 NoPE 形式的纯因果注意力）→ 在原训练上下文长度上继续训练一小段（重校准）→ 推理时配合 softmax 温度缩放外推到更长序列。整套逻辑由三个递进的观察支撑：位置编码训练时有益（观察一）、RoPE 缩放注定外推失败（观察二）、位置编码可在训练后安全移除（观察三）。

```mermaid
flowchart LR
    A[RoPE 预训练<br/>享受归纳偏置加速收敛] --> B[删除所有层的<br/>位置编码 → NoPE 形式]
    B --> C[原长度上短暂重校准<br/>少量 token]
    C --> D[推理: softmax 温度缩放<br/>零样本外推到 2×~8× 长度]
```

### 关键设计

**1. 诊断病根：位置编码是"训练快"与"外推好"不可兼得的根源。** 论文先用注意力位置偏置泛函 $A_c(\alpha)=\frac{1}{T}\sum_i\sum_{j\le i}c_{ij}\alpha_{ij}$ 量化注意力头的非均匀程度（如对角头把质量压在当前 token、即为 $A_c$ 的最大化解）。理论上（Theorem 3.4）NoPE 在初始化时嵌入近乎均匀，这种均匀性会逐层传播，导致 $A_c$ 及其梯度被一个与序列长度无关的小常数 $C\varepsilon$ 界住——位置非均匀性发展得很慢，因此 NoPE 训练慢、性能差。一个极端例子（Prop 3.2）是常数序列下 NoPE 所有注意力权重恒为 $\alpha_{ij}=1/i$、query/key 梯度全为零；而 RoPE（Prop 3.3）即便在常数序列上也能产生非零 $A_c$ 梯度。这解释了为什么 RoPE 训得快——但显式位置信息一旦写死，外推时就成了枷锁。

**2. 揭示 RoPE 缩放为何注定失败：压低频必然移位语义头。** 在每个 $(2m,2m{+}1)$ 子空间里相对距离 $\Delta$ 的 RoPE 相位为 $\phi_m(\Delta)=\omega_m\Delta$；要把 $\Delta$ 从 $C_{train}$ 扩到 $C_{test}=sC_{train}$ 而相位不越界，任何缩放法都必须取 $\gamma_m\le 1/s$ 来压缩低频。问题在于：高频主要被**位置头**使用（对角/前一 token 等基于相对位置的模式），低频则被**语义头**使用（基于内容匹配的检索）。YaRN/PI/NTK 几乎不动高频却狠压低频，于是位置头基本不受影响、语义头却被严重移位，且距离越远 $\phi_m(\Delta)$ 越大、$1/s$ 的扭曲越剧烈。结果就是：困惑度看似维持（位置头还在），但远距离检索失败（语义头被搬走）——论文实测 YaRN 的零样本行为几乎等价于"把序列裁剪回训练长度"，能保困惑度却看不见窗外的信息。这从机理上说明 RoPE 缩放的外推失败是结构性的、无法回避的。

**3. DroPE 的"先用后弃"：训练后摘除 + 短重校准 + 推理温度缩放。** 既然位置编码训练有益、外推有害，最干净的做法就是只在预训练时留着它、训练完整体删掉。具体地，取 RoPE checkpoint，从每层移除位置旋转（变成 NoPE 形式，仍靠因果掩码隐式编码位置），在**原训练长度**上继续训练一小段做重校准——关键在于此时模型已经从 RoPE 阶段继承了成熟的位置感知，不必像 NoPE 那样从零慢慢长出来，因此只需极少 token 就能恢复原有能力。从头训练时这一步可"零额外成本"嵌入（如把 16K 步 RoPE 训练的最后 2K 步换成 DroPE）；对已用数千亿 token 预训练好的现成模型（SMOLLM、LLaMA2-7B），只需 0.5%~2% 预训练预算的重校准即可。为支持高学习率重校准，DroPE 后会补一个 QKNorm（不改变模型容量），推理时再对 NoPE/DroPE 模型施加 softmax 温度缩放以稳定外推。

## 实验关键数据

### 主实验表格

零样本 NIAH（2× 训练上下文，500 次试验成功率）：

| 方法 | Multi-Query | Multi-Key | Multi-Value |
|------|------------|-----------|-------------|
| RoPE transformer | 0.0 | 0.0 | 0.0 |
| RoPE + PI | 0.0 | 0.0 | 0.0 |
| RoPE + NTK | 21.1 | 19.4 | 16.5 |
| RoPE + YaRN | 17.8 | 0.5 | 14.6 |
| ALiBi | 5.2 | 0.0 | 1.1 |
| NoPE | 9.2 | 36.2 | 21.4 |
| RNoPE-SWA | 5.2 | 25.6 | 20.6 |
| **DroPE** | **28.0** | **41.6** | **23.3** |

LongBench 长上下文任务（SMOLLM，预训练 2048 上下文）：

| 方法 | MultiFieldQA | MuSiQue | GovReport | LCC | NIAH | Avg. |
|------|------|------|------|------|------|------|
| SMOLLM (base) | 4.03 | 0.4 | 4.48 | 5.99 | 0.0 | 2.98 |
| + PI | 13.68 | 2.45 | 5.67 | 11.52 | 0.0 | 6.66 |
| + NTK | 18.87 | 4.89 | 23.71 | 8.26 | 29.84 | 17.11 |
| + YaRN | 20.78 | 4.77 | 15.03 | 10.87 | 48.25 | 19.94 |
| **SMOLLM-DROPE** | **29.33** | **7.93** | 21.87 | **18.56** | **74.92** | **30.52** |

DroPE 把基座 SMOLLM 的平均分提升 **10 倍以上**。

### 消融实验表格

长 NIAH 不同外推倍数下的成功率：

| 方法 | 2× | 4× | 8× |
|------|------|------|------|
| SMOLLM + NTK | 29.84 | 14.37 | 7.19 |
| SMOLLM + YaRN | 48.25 | 25.62 | 12.18 |
| SMOLLM + LongRoPE2 | 44.20 | 26.20 | 16.45 |
| **SMOLLM-DROPE** | **74.92** | **55.00** | **52.20** |

更大模型上的长度泛化（LongBench 平均分）：

| 模型 | Base | NTK | YaRN | DroPE |
|------|------|------|------|------|
| SMOLLM-1.7B | 3.11 | 18.53 | 16.23 | **21.49** |
| LLaMA2-7B | 20.03 | 21.88 | 19.14 | **26.08** |

### 关键发现
- **零额外成本嵌入预训练**：把 16K 步 RoPE 训练的最后 2K 步换成 DroPE，最终原长度困惑度即可追平全程 RoPE，并优于全程 NoPE 基线。
- **快速恢复**：现成 SMOLLM 上，DroPE 仅用 <5B token（原预算的 0.8%）就恢复了 95%+ 的原始性能；最长 120B 重校准甚至超过原模型。
- **倍数越大优势越大**：在 8× 外推下，YaRN/LongRoPE2 崩到 12~16，DroPE 仍有 52.2，差距随外推倍数拉大。
- **可扩展到大模型与现成 LLM**：SMOLLM-1.7B（2% 预算）、LLaMA2-7B（0.5% 预算）重校准后均全面超过 SOTA RoPE 缩放法。

## 亮点与洞察
- **观点反直觉且干净**：业界默认位置编码是必需品、外推靠"修补 RoPE"，本文反其道把它当成"用完即弃的训练脚手架"，一删了之反而最优。
- **理论与机理双线坐实**：用 $A_c$ 泛函的梯度界（Theorem 3.4）解释 NoPE 为何训练慢，用"压低频必移语义头"解释 RoPE 缩放为何注定外推失败——两个观察一正一反，逻辑闭环精彩。
- **实用价值高**：DroPE 可零成本塞进现有训练管线，也能用极小预算（<1%）就地升级任意现成 LLM，工程友好。

## 局限与展望
- **仍需重校准**：虽然成本极低，但"完全无需任何额外训练"尚未达到；删 PE 后必须配 QKNorm 并做短训练。
- **推理需温度缩放**：NoPE/DroPE 外推依赖 softmax 温度缩放这一额外推理技巧，其鲁棒性与超长（>8×）下的边界仍待更广验证。
- **最大规模有限**：实验止于 LLaMA2-7B 量级，对数百亿参数前沿模型与更复杂长上下文 agent 任务的效果有待检验。
- **NoPE 隐式位置容量**：删除显式 PE 后模型完全依赖因果掩码隐式编码位置，极端长度下隐式位置信息是否足够仍是开放问题。

## 相关工作与启发
- **RoPE 缩放系**：PI、NTK-RoPE、YaRN、LongRoPE2 是本文主要对手，论文从机理上论证了它们外推失败的必然性。
- **RoPE 变体与中间路线**：p-RoPE、RNoPE-SWA、SWAN-GPT 等占据 RoPE 与 NoPE 之间的中间地带；本文则是彻底回到 NoPE 但靠"先 RoPE 后摘除"破解 NoPE 训练慢的老问题。
- **NoPE 理论**：Kazemnejad、Haviv 等证明 NoPE 表达力足够、可用因果掩码重建位置；本文补上"为何 NoPE 训练慢"的训练动力学解释，是对该线的重要延伸。
- **启发**：归纳偏置不必"从一而终"——可以设计成阶段性的训练脚手架，在合适时机移除以换取更好的泛化，这一思路或可推广到其他显式先验。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ "预训练后丢弃位置编码"是对位置编码角色的根本性重新认识，配以一正一反两个理论观察，原创性极强。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖从头训练与现成模型两种场景、多模型（SMOLLM-0.5B/1.7B、LLaMA2-7B）、2×~8× 多倍数、NIAH+LongBench 多任务，较扎实；唯模型规模上限偏小。
- **写作质量**: ⭐⭐⭐⭐⭐ 三观察递进、机理论证与图表紧密配合，逻辑闭环清晰，可读性强。
- **价值**: ⭐⭐⭐⭐⭐ 零成本嵌入训练 + <1% 预算升级现成 LLM，对长上下文基座模型的训练范式有直接落地意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context](the_pensieve_paradigm_stateful_language_models_mastering_their_own_context.md)
- [\[ACL 2025\] Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](../../ACL2025/llm_efficiency/mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)
- [\[ICLR 2026\] STEM: Scaling Transformers with Embedding Modules](stem_scaling_transformers_with_embedding_modules.md)
- [\[ICLR 2026\] Let's (not) just put things in Context: Test-time Training for Long-context LLMs](lets_not_just_put_things_in_context_test-time_training_for_long-context_llms.md)
- [\[ICLR 2026\] Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs](tactic_adaptive_sparse_attention_with_clustering_and_distribution_fitting_for_lo.md)

</div>

<!-- RELATED:END -->
