---
title: >-
  [论文解读] Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling
description: >-
  [ICLR 2026][LLM效率][投机采样] 本文把投机采样重新刻画成"在受控散度约束下最大化接受率"的约束优化问题，并由此提出 Cactus——只需读取候选 token 一个概率值、给它加一个由 $q$ 和散度预算 $\delta$ 决定的"奖励"，在保证与验证器分布偏差可控的前提下显著提高接受率与吞吐。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "投机采样"
  - "约束优化"
  - "KL 散度"
  - "接受率"
  - "无损解码"
  - "大模型推理"
---

# Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lpUIkCAy9p](https://openreview.net/forum?id=lpUIkCAy9p)  
**代码**: [https://github.com/MANGA-UOFA/Cactus](https://github.com/MANGA-UOFA/Cactus)  
**领域**: LLM 推理加速 / 投机采样  
**关键词**: 投机采样, 约束优化, KL 散度, 接受率, 无损解码, 大模型推理  

## 一句话总结
本文把投机采样重新刻画成"在受控散度约束下最大化接受率"的约束优化问题，并由此提出 Cactus——只需读取候选 token 一个概率值、给它加一个由 $q$ 和散度预算 $\delta$ 决定的"奖励"，在保证与验证器分布偏差可控的前提下显著提高接受率与吞吐。

## 研究背景与动机
**领域现状**：投机采样（Speculative Sampling, SpS）用一个小的草稿模型自回归地一次提议 $m$ 个 token，再用大验证器模型并行校验，从而每次大模型前向能吐出多个 token，缓解访存瓶颈、提升解码吞吐。SpS 的核心卖点是"无损"——严格保证生成分布与验证器分布 $q$ 完全一致。

**现有痛点**：这种严格的分布等价其实"过度保守"。验证器分布本来就常被 top-k、temperature 等采样手段修改，轻微偏离 $q$ 在实际应用中完全可接受；SpS 却会把"正确但概率略低"的 token 也拒掉，白白损失接受率。Typical Acceptance Sampling（TAS）想缓解这点，用基于熵的启发式接受更多提议——但本文指出 TAS 会**扭曲验证器分布**，当 $q$ 携带关键信息（如细粒度决策信号）时会引发语义漂移、掉点（在 GPQA 上 TAS 甚至跌破 $-1\sigma$ 显著退化线）。

**核心矛盾**：接受率（吞吐）与对验证器分布的保真度（质量）之间存在张力。SpS 站在"保真"极端牺牲吞吐；TAS 站在"激进接受"一端牺牲质量，且**没有对偏离量的显式控制**。

**本文目标**：在"偏离 $q$ 多少"这件事上给出一个**硬约束、可调、有保证**的旋钮，让用户能在质量-吞吐之间安全地权衡，而不是被启发式牵着走。

**核心 idea**：**把投机采样写成约束优化**——目标是最大化候选 token 的接受率，约束是动态构造的目标分布 $h$ 与验证器 $q$ 的 $f$-散度不超过预算 $\delta$。求解这个问题得到一个简洁闭式修正，即 Cactus。

## 方法详解

### 整体框架
作者先把 draft-and-verify 流程抽象成一个统一算法（接受率函数 $\phi$ + 恢复分布 $g$ 两个可插拔组件），证明只要把目标分布设成任意 $h$、并按对应公式定义 $\phi$ 和 $g$，算法就能"精确产生 $h$ 且接受率最优"。SpS 只是其中 $h=q$ 的特例。Cactus 的关键转念是：**不必死守 $h=q$，而是在"离 $q$ 不远"的约束下，动态挑一个能让候选 token 更容易被接受的 $h$**。整条链路是：约束优化建模 → 闭式解 $h$（Theorem 2）→ KL + 二阶 Taylor 近似得到极简打分（Corollary 5）→ 反推 $\phi, g$ 套回统一算法。

```mermaid
flowchart LR
    A[草稿模型 p<br/>提议候选 token n] --> B[约束优化<br/>max 接受率<br/>s.t. D_f h‖q ≤ δ]
    B --> C[闭式解 h<br/>给 token n 加 bonus]
    C --> D[KL + 二阶 Taylor<br/>γ* = q + √2δq1-q]
    D --> E[反推 φ, g<br/>套回 draft-verify]
    E --> F[更高接受率<br/>+ 散度可控]
```

### 关键设计

**1. 把投机采样建模为约束优化：用散度预算 $\delta$ 显式买接受率。** 这是全文的理论支点。在每个时间步、给定草稿模型采到的 token 索引 $n$，作者把目标分布 $h\in\mathbb{R}^{|V|}$ 当作优化变量，求解 $\max_h \min\{h_n/p(n|x_{<t}),1\}$，约束是 $h$ 为合法分布且 $D_f(h\|q(\cdot|x_{<t}))\le\delta$。这里 $\delta\ge0$ 就是那个"允许偏离验证器多远"的旋钮：$\delta=0$ 退化回无损 SpS，$\delta$ 越大越激进。这套表述的价值在于把过去靠启发式（TAS 的熵阈值）拍脑袋决定的"接受多一点"变成了一个有显式目标、有显式约束的凸优化，从而可以谈最优性、可以谈保证。

**2. 闭式最优解：只抬高候选 token，其余按比例缩放。** 对上述凸优化，Theorem 2 给出最优 $h$ 的闭式：候选 token 拿到 $h_n=\gamma^\*$，其余 token $i$ 取 $h_i=\frac{1-\gamma^\*}{1-q(n|x_{<t})}q(i|x_{<t})$，其中 $\gamma^\*$ 是散度约束方程在 $[q(n),+\infty)$ 区间内的根（再夹到 $[q(n),1]$）。直观上就是"把候选 token 的概率从 $q(n)$ 抬到 $\gamma^\*\ge q(n)$，把这部分多出来的质量从其它 token 等比例扣回去"，使候选更容易过验证。由于解依赖采到的 $n$，整体算法的有效分布 $h_{\text{alg}}$ 可能并非恰好偏离 $\delta$，作者用 Theorem 3 兜底：$D_f(h_{\text{alg}}\|q)\le\min\{\Gamma(\delta),\,D_f(p\|q)\}$，其中 $\Gamma$ 连续、单调、$\Gamma(0)=0$——也就是说**整体偏离仍被 $\delta$ 隐式锁住**，调 $\delta$ 就能稳定换取质量-吞吐的不同工作点。

**3. 选 KL 散度 + 二阶 Taylor，得到极简打分 $\gamma^\*$。** 框架里 $f$-散度可任选，但作者论证应选 KL（$f(t)=t\log t$）而非 TAS 隐式用的交叉熵。原因在 Eq.(10) 的分解 $H(h,q)=D_{\mathrm{KL}}(h\|q)+H(h)$：交叉熵里那个熵项 $H(h)$ 会鼓励 $h$ 坍缩成确定性分布，于是 TAS 总把 $h$ 压成熵为 0、并至少增加 $H(q)$ 的散度——当 $q$ 高熵（信息丰富）时偏离尤其严重，这正是 TAS 掉点的根因。纯 KL 没有这个坍缩项。但 KL 约束方程含 $x\log x$ 是超越方程、无闭式根，作者在 $\gamma_0=q(n)$ 处做二阶 Taylor 展开（因 $\delta$ 通常很小、$\gamma^\*$ 离 $q(n)$ 不远），得到 Cactus 的最终打分：

$$\gamma^\* = \min\left\{\, q(n|x_{<t}) + \sqrt{2\delta\, q(n|x_{<t})\big(1-q(n|x_{<t})\big)},\ 1 \,\right\}$$

即"给候选 token 的概率加一个由 $q(n)$ 和 $\delta$ 共同决定的 bonus"，bonus 在 $q(n)=0.5$ 时最大。Corollary 6 进一步证明：当验证器对该 token 不太自信（$\gamma^\*\le0.5$）时，这个近似**严格满足** $D_{\mathrm{KL}}(h\|q)\le\delta$，保证了在最该小心的场景下约束不被违反。

**4. 工程上更省：只读一个概率，降低访存。** 相比 TAS 的判据需要遍历整个词表算熵，Cactus 只需读取候选 token $n$ 处的概率 $q(n)$。在动辄 10 万+ 词表的现代 LLM 上，这进一步削减了访存开销——而投机采样本身要解决的就是访存瓶颈，所以这个"少读一整行 logits"的好处与方法目标天然契合。

## 实验关键数据

### 主实验
在 GSM8K / IFEval / GPQA 三个基准上、用 Qwen3 系列做验证器+草稿对（0.6B 草稿 + 8B/14B 验证器），报告 strict-match 准确率、接受长度 $AL_m$（草稿 $m$ 个时被接受的期望数，越大越快）、以及相对 SpS 的拒绝 token 数 Rej（越负越好）。

| 设置 (m=20, Qwen3-14B 验证 + 0.6B 草稿) | GSM8K Score↑ / AL↑ / Rej↓ | GPQA Score↑ / AL↑ / Rej↓ |
|---|---|---|
| Verifier（参考上界） | 91.71 / – / – | 40.07 / – / – |
| SpS（无损基线） | 91.89 / 5.11 / Ref | 40.91 / 3.84 / Ref |
| TAS | 92.87 / 6.78 / −32% | 40.40 / 6.41 / −46% |
| Cactus 0.75 | 92.15 / 7.15 / −36% | **45.46** / 6.46 / −46% |
| Cactus 1.0 | 92.87 / 7.00 / −34% | **45.46** / 6.74 / −50% |

Cactus 在三个基准上普遍取得最高接受率，同时保持或提升准确率；尤其在硬基准 GPQA 上，TAS 相对 SpS 掉点（如 8B 验证器 m=20 时 42.93→38.89），而 Cactus 反而把 GPQA 抬到 45.46，吞吐与质量同时拿下。

### 消融 / 深入分析

| 分析 | 关键结果 |
|---|---|
| 散度控制的必要性（vs 线性插值） | 同样约 90% 接受率下，Cactus 得分 >86，简单插值 $\alpha=0.9$ 仅 <72；即便 96.3% 接受率，Cactus 仍保持 >80，证明"有保证的散度控制"显著优于"简单混合 $p,q$"。 |
| 准确率-接受率权衡（Fig.1，以 $\sigma$ 度量） | TAS 在 GPQA(8B, m=20) 跌破 $-1\sigma$ 显著退化线；Cactus 始终保持在验证器置信区间内甚至超出。 |
| 墙钟吞吐（A100 40GB + vLLM, GPQA） | 0.6B+14B、m=10 时 Cactus 1.0 相对单验证器近 **1.9× 加速**，同时 GPQA 得分最高；TAS 几乎在所有设置下略逊 Cactus。 |
| 跨模型系列泛化 | 在 Gemma(2B+9B)、DeepSeek-R1(1.5B+7B)、LLaMA(1B+8B) 以及 Qwen3-32B+1.7B 上均验证有效，并把 top-k 接受作为有损基线对比。 |

### 关键发现
- TAS 的掉点不是偶然，而是其隐式用交叉熵导致 $h$ 坍缩为确定性分布的**结构性缺陷**；Cactus 用纯 KL 规避了这一点。
- 在"验证器不自信"的 token 上，Cactus 的近似解被证明严格满足散度约束，即风险最高处反而最稳。
- 把"接受多一点"从启发式变成可证明的约束优化，使得单一旋钮 $\delta$ 就能平滑覆盖从无损到激进的整条工作曲线。

## 亮点与洞察
- **理论统一**：用约束优化的视角重写投机采样，既涵盖 SpS（$\delta=0$），又给出 TAS 的全新解释（Prop.4：TAS 等价于把 $f$-散度换成交叉熵的变体），把两条看似无关的路线放进同一框架。
- **诊断到位**：通过 $H(h,q)=D_{\mathrm{KL}}+H(h)$ 的分解，干净地指出 TAS 掉点的数学根因是熵项导致的分布坍缩——这是比"实验上 TAS 不行"更深一层的解释。
- **简洁可落地**：最终方法只是给候选 token 加一个解析 bonus、且只读一个概率值，几乎零额外开销，易于嵌进现有 SpS/vLLM 流水线。

## 局限与展望
- $\Gamma(\delta)$ 无闭式表达，$\delta$ 与"实际整体偏离"之间的映射只能靠经验调参，缺少直接按目标质量损失反解 $\delta$ 的手段。
- 二阶 Taylor 近似仅在 $\delta$ 较小、$\gamma^\*$ 接近 $q(n)$ 时严格成立；Corollary 6 只覆盖 $\gamma^\*\le0.5$ 的情形，更激进的大 $\delta$ 区间缺乏同等强度的约束保证。
- 实验聚焦推理/数学/指令跟随类基准与较小草稿模型，对开放式长文本生成、超大规模草稿-验证器配比下的质量漂移仍待更系统验证。

## 相关工作与启发
- **投机采样谱系**：从 SpS（Chen et al., 2023; Leviathan et al., 2023）到 Medusa/TAS（Cai et al., 2024），本文把"无损 vs 有损接受"统一进约束优化，给后续设计新的接受准则提供了一个原则性的模板——换 $f$-散度就是换接受策略。
- **对采样研究的呼应**：与 nucleus / typical sampling（Holtzman et al., 2020; Meister et al., 2023）一脉相承，承认"精确 $q$ 并非必需"，但把偏离从启发式上升为可控约束。
- **启发**：任何"为了加速而偏离参考分布"的场景（不止解码，也包括蒸馏、采样近似），都可借这套"最大化收益 s.t. 散度预算"的框架，把偏离量变成可证明、可调的硬约束，而非事后才发现质量塌了。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把投机采样重写为约束优化、并据此统一解释 SpS/TAS，是一个干净且有深度的理论贡献。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖三基准、四个模型系列、多种验证器-草稿配比，含墙钟吞吐与插值对照消融，证据链完整。
- **写作质量**: ⭐⭐⭐⭐ — 从动机、定理到近似推导层层递进，TAS 缺陷的分解分析尤其清晰。
- **价值**: ⭐⭐⭐⭐ — 几乎零开销、易落地的有损投机采样改进，并提供了一个可复用的"散度预算"设计范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Optimization](ni_sampling_accelerating_discrete_diffusion_sampling_by_token_order_optimization.md)
- [\[ICLR 2026\] Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization](global_resolution_optimal_multi-draft_speculative_sampling_via_convex_optimizati.md)
- [\[ICLR 2026\] CONCUR: A Framework for Continual Constrained and Unconstrained Routing](concur_a_framework_for_continual_constrained_and_unconstrained_routing.md)
- [\[ICLR 2026\] vAttention: Verified Sparse Attention via Sampling](vattention_verified_sparse_attention_via_sampling.md)
- [\[ICLR 2026\] Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding](overcoming_joint_intractability_with_lossless_hierarchical_speculative_decoding.md)

</div>

<!-- RELATED:END -->
