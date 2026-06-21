---
title: >-
  [论文解读] The Softmax Bottleneck Does Not Limit the Probabilities of the Most Likely Tokens
description: >-
  [ICLR 2026][学习理论][softmax 瓶颈] 这篇论文从理论上重新审视"softmax 瓶颈"：它证明即使是随机初始化的输出投影矩阵，也能为相当多（GPT-2 量级约 26、实测约 95，Llama2 实测超 1000）的最高概率 token 指定任意接近真实的概率，从而质疑 softmax 瓶颈是否真的在现实场景里显著限制了 LLM 的能力。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "LLM 表达能力分析"
  - "softmax 瓶颈"
  - "输出投影矩阵"
  - "随机矩阵"
  - "符号秩"
  - "嵌入维度"
---

# The Softmax Bottleneck Does Not Limit the Probabilities of the Most Likely Tokens

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DgJqQk6y19](https://openreview.net/forum?id=DgJqQk6y19)  
**代码**: https://github.com/ronenbasri/The-Softmax-Bottleneck-Does-Not-Limit-the-Probabilities-of-the-Most-Likely-Tokens  
**领域**: 学习理论 / LLM 表达能力分析  
**关键词**: softmax 瓶颈, 输出投影矩阵, 随机矩阵, 符号秩, 嵌入维度

## 一句话总结
这篇论文从理论上重新审视"softmax 瓶颈"：它证明即使是随机初始化的输出投影矩阵，也能为相当多（GPT-2 量级约 26、实测约 95，Llama2 实测超 1000）的最高概率 token 指定任意接近真实的概率，从而质疑 softmax 瓶颈是否真的在现实场景里显著限制了 LLM 的能力。

## 研究背景与动机

**领域现状**：Transformer 在最后一层用一个 $N \times d$ 的输出投影矩阵（OPM，也叫 unembedding / readout 矩阵）把 $d$ 维嵌入 $x$ 线性映射成 $N$ 维 logits $y = Ax$（$N$ 是词表大小，远大于 $d$），再经 softmax 得到下一个 token 的分布。

**现有痛点**：一系列工作（Yang 等 [26]、Chang & McCallum [2]、Grivas 等 [7] 等）指出，由于 $d \ll N$，logits 被限制在 $d$ 维线性子空间里，对数概率空间也只能落在一个低维子空间上——这就是所谓 **softmax 瓶颈**。他们论证自然语言的对数概率分布是高维的，这个瓶颈让 LLM 无法精确拟合下一个 token 的真实统计，从而损害性能，并提出 Mixture of Softmaxes、multi-facet softmax、Linear-Monotonic-Softmax 等各种"补救"方案。

**核心矛盾**：这些工作衡量的是"能否表达**任意**完整分布"。但作者指出一个被忽略的事实：在推理时用 nucleus sampling（核采样）或 beam search，**只有最高概率的少数 token 真正影响生成**，那些极不可能 token 的精确概率根本无所谓。所以真正该问的不是"能否表达任意分布"，而是"能否准确指定 **top-$m$** 个最可能 token 的概率"。这是一个弱得多、也现实得多的要求。

**本文目标**：拆成两个子问题——(1) 给定任意 $m$ 个 token，OPM 能否产生一个让它们成为最可能 token、且概率任意指定（和接近 1）的分布？这个 $m$ 能有多大？(2) 最好的（训练后的）OPM 在多大的 $m$ 上能做到这件事，即所需的最小嵌入维度是多少？

**切入角度**：作者从"top-$m$ 视角"重新框定问题，并分别用**随机矩阵理论**（给性能下界）和**符号秩（sign rank）**（给最优矩阵的上界）两套工具去夹逼这个 $m$，再用线性规划做实证。

**核心 idea**：softmax 瓶颈确实让 LLM 只能表达全体分布中一个测度为零的子集，但这个子集对"准确刻画最可能的若干 token"几乎没有约束——因此瓶颈在现实设置下大概率不是真正的性能瓶颈。

## 方法详解

### 整体框架

整篇论文是一个分析性的理论工作，不涉及新模型，目标是给"OPM 能精确表达 top-$m$ 概率的最大 $m$"找上下界并实证。整体分三条腿走：

第一条腿（第 4 节，**性能下界**）：把 OPM 当成随机高斯矩阵 $A \sim \mathcal{N}(0,1)$，问"随机初始化时就能指定多大的 $m$"。先解决"让 top-$m$ 的 logits 都最大"（4.1），再加一步缩放把它升级成"让 top-$m$ 拿到任意指定、且和接近 1 的精确概率"（4.2）。

第二条腿（第 5 节，**最优上界**）：抛开随机矩阵，问"存在的最好 OPM 能做到多大的 $m$"，即最小嵌入维度 $d^*$。作者把这个组合问题精确归约到一个 $\{-1,+1\}$ 指示矩阵的**符号秩**问题，借 Alon 等人的结果算出 $d^* \approx 2m$。

第三条腿（第 6 节，**实证**）：对 GPT-2、GPT2-XL、TinyLlama、T5-Large、Llama2 的训练后 OPM 和同尺寸随机矩阵，用线性规划（LP）实际求解可行的嵌入 $x$，测出真实的 $m$，并发现训练后的矩阵和随机矩阵能力几乎一样。

下面三个关键设计依次对应这三条腿。

### 关键设计

**1. 随机 OPM 的 top-$m$ 下界：用逆 Wishart 近似估出"其余 logit 都更小"的概率**

针对"随机初始化时能指定多大 $m$"这个问题，作者构造一个只落在 $A_m$（$A$ 的前 $m$ 行）行空间里的嵌入 $x_\parallel = A_m^T w$，使前 $m$ 个 logit 取目标值 $y_m$（由 $w=(A_m A_m^T)^{-1}y_m$ 唯一确定），然后估计剩下 $N-m$ 个 logit 中"最大者仍小于 top-$m$ 最小者"的概率。关键技术是：由于 $A$ 元素 i.i.d. 高斯，$(A_m A_m^T)^{-1}$ 服从逆 Wishart 分布 $W_m^{-1}(I,d)$，据此可算出向量 $a_j^T A_m^T (A_m A_m^T)^{-1}$ 的元素近似 i.i.d.、零均值、方差为

$$v \approx \frac{d(d-1)}{(d-m)(d-m-1)(d-m-3)}$$

于是底部 logit 近似 $y_j \sim \mathcal{N}(0, \|y_m\|^2 v)$。这给出下界（Proposition 1）

$$P\big(\exists x,\ y_{m+1},\dots,y_N < y_1,\dots,y_m\big) \gtrsim \Phi^{N-m}\!\left(\frac{\min_{i\in[m]} y_i}{\|y_m\|\sqrt{v}}\right)$$

其中 $\Phi$ 是标准正态 CDF。Corollary 1 进一步证明这个概率在 **top-$m$ 的 logits 全相等且为正**（$y_m = c\mathbf{1}_m$）时最大，此时下界简化为 $\Phi^{N-m}(1/\sqrt{mv})$。把 22 个常见模型代入，作者得到下界从 GPT-2 的 $m\approx 26$ 一直到 GPT3-175B 的 $m\approx 418$——说明现代大模型随机初始化就能指定相当多的 top token。⚠️ $v$ 的推导用了"$a_j^T A_m^T$ 各分量与 $(A_m A_m^T)^{-1}$ 统计独立"的近似，作者在附录 B 验证其精度，细节以原文为准。

**2. 从 top-$m$ logits 升级到精确概率：靠尺度缩放把概率质量挤进 top-$m$**

光让 top-$m$ logit 最大还不够，因为 softmax 下每个 token 的概率依赖全体 logit。作者要的是让 top-$m$ 拿到**任意指定的比值** $p_i/p_m$、且这 $m$ 个概率之和趋近 1（其余之和 $\le \delta$）。难点在于"指定比值"和"和接近 1"看似冲突。解法是引入一个尺度因子 $s$，令 $x_s = s\bar{x}_s$，其中 $\bar{x}_s$ 通过

$$\bar{y}_s(i) = \frac{1}{s}\log\!\left(\frac{p_i}{p_m}\right) + 1$$

构造，保证任意 $s$ 下 top-$m$ 的概率比值都恰好等于目标。随着 $s\to\infty$，所有 $\bar{y}_s(i)$ 趋于 1（变均匀），而只要其余 logit 满足 $\bar{y}_s(j) < 1-\epsilon$，缩放就会把概率质量指数级地压到 top-$m$ 上：取 $s \ge -\frac{1}{\epsilon}\log\frac{m\delta}{N}$ 即可保证 $\sum_{j>m} p_j < \delta$（Proposition 2）。当 $\epsilon,\delta \to 0$，这个事件的概率收敛到设计 1 里 logits 全相等时的下界 $\Phi^{N-m}(1/\sqrt{mv})$（Corollary 2）。结论很漂亮：**只要随机 OPM 能让某 $m$ 个 token 成为最可能的，它就能让它们拿到任意（和接近 1 的）精确概率**——这正是 nucleus sampling 下唯一需要的东西。

**3. 最优 OPM 的上界：把最小嵌入维度归约到符号秩，得到 $d^* \approx 2m$**

前两个设计给的是随机矩阵的下界，那"最好的训练后矩阵"能做到多大？作者问：要让**任意**大小为 $m$ 的 token 子集都能被某个嵌入选为 top-$m$，最小嵌入维度 $d^*$ 是多少？他们把所有"完整 top-$m$ logit 矩阵"（每列让一个不同的 $m$-子集 logit 最大）的最小秩，归约到一个 $\binom{N}{m}$-指示矩阵 $S\in\{-1,+1\}^{N\times\binom{N}{m}}$ 的符号秩：每列恰有 $m$ 个 $+1$ 对应一个 $m$-子集。Proposition 4 证明 $|d^* - \text{signrank}(S)| \le 1$（因为减去一个秩-1 的阈值外积矩阵就把 logit 矩阵变成 $S$ 的符号模式）。再借 Alon 等人 [1] 的引理（用列上符号变化次数上界、用对偶符号秩下界），Proposition 5 给出当 $N \ge 3m+1$ 时 $\text{signrank}(S) = 2m+1$，从而（Corollary 3）

$$2m \le d^* \le 2m+2$$

这意味着存在嵌入维度仅约 $2m$ 的 OPM 就能选出任意 $m$-子集。代入 GPT-2（$d=768$）得最优可达 $m=383$，远超随机矩阵的 $26$；TinyLlama 量级可达 $1023$。注意这个上界只保证"能选为 top-$m$"，要进一步指定它们的精确顺序或概率可能需要更大维度。

### 一个完整示例

以 **GPT-2**（$N=50257$，$d=768$）走一遍三条腿的结论，能直观看到差距：

- **随机矩阵理论下界（设计 1/2）**：$m \approx 26$——把 $A$ 当随机高斯矩阵、且把嵌入限制在 $A_m$ 行空间内时，能近乎确定地指定的 top token 数。
- **线性规划实测（设计 3 之外的实验）**：放开嵌入到全空间（加上正交分量 $x_\perp$ 去压低其余 logit）后，随机矩阵和训练后的 GPT-2 矩阵都能把 $m$ 推到约 **95**；要求精确概率（$\epsilon=0.05,\delta=0.1$）时略降。
- **符号秩最优上界（设计 3）**：存在的最好 OPM 可达 $m = 383$。

三个数 26 → 95 → 383 分别是"随机矩阵保守下界 / 随机矩阵实际能力 / 理论最优"，层层递进。而关键观察是：训练后的 GPT-2（solid blue 曲线）和同尺寸随机矩阵（dotted blue）几乎重合——**训练既没显著提升也没削弱 OPM 指定 top-$m$ 的能力**。

## 实验关键数据

### 主实验

实验对 GPT-2、GPT2-XL、TinyLlama-1.1B、T5-Large、Llama2-7B 的训练后 OPM 及同尺寸随机矩阵，用线性规划求解可行嵌入 $x$（满足 $A_m x = y_m$ 且其余 logit 更小），统计在 $m$ 个随机 token 上成功的比例。下表汇总各模型在不同口径下能可靠指定（成功概率接近 1）的 $m$：

| 模型 | $N$ | $d$ | 理论下界 $m$ (式 4) | LP 实测 $m$ | 符号秩最优 $m$ |
|------|-----|-----|------|------|------|
| GPT-2 | 50257 | 768 | 26 | ~95 | 383 |
| TinyLlama-1.1B | 32000 | 2048 | 71 | ~400 | 1023 |
| Llama2-7B | 32000 | 4096 | 143 | ~1070 | — |
| GPT3-175B | 50257 | 12288 | 418 | — | — |

随机矩阵下界随模型增大单调上升（GPT-2 的 26 → GPT3-175B 的 418），而 LP 实测值通常是下界的数倍——因为下界只用了 $A_m$ 行空间内的嵌入，加上正交分量 $x_\perp$ 后能力大幅提升。

### 经验规律与分析

| 实验 | 关键发现 | 说明 |
|------|---------|------|
| 训练 vs 随机矩阵 | 两条曲线几乎重合 | 训练既不改善也不损害 top-$m$ 指定能力 |
| 精确概率 vs 仅 top-$m$ | 精确概率的可达 $m$ 略低 | 由 $\epsilon=0.05,\delta=0.1$ 的有限取值导致，符合理论预期 |
| $m$ 随 $d,N$ 变化（Fig. 3） | $d\ll N$ 时 $m \approx d/5$ | 给出"需要精确表达多少 token 就需要多大嵌入"的简单经验公式 |

### 关键发现
- **训练后的 OPM 和随机矩阵能力相当**：这是最反直觉的结果——大量先前工作分析训练涌现出的嵌入/OPM 统计规律，但就"指定 top-$m$ 概率"而言，训练带来的差别可忽略。
- **下界很保守、实际能力强得多**：把嵌入放开到全空间后，LP 实测 $m$（GPT-2 约 95、Llama2 超 1000）远超理论下界，说明现实模型能精确刻画的高概率 token 数量非常可观。
- **$m \approx d/5$ 的经验定律**：在 $d \ll N$ 时近似成立，且随 $N$ 增大缓慢下降，给工程上"嵌入维度该取多大"提供了直接参考。
- 结合附录 A 的证据，作者猜想推理时（nucleus sampling 下）top-1000 之后 token 的概率几乎不影响性能，进一步弱化了 softmax 瓶颈在现实中的意义。

## 亮点与洞察
- **重新框定问题本身就是最大的贡献**：从"能否表达任意分布"转到"能否准确表达 top-$m$"，一下子把一个看似严重的限制变成了几乎不构成约束的问题——切中了推理时只有高概率 token 重要的实际机制。
- **两套工具夹逼**：随机矩阵理论（逆 Wishart + 高斯近似）给下界、符号秩给上界，再用线性规划在中间实测，三者互相印证，论证非常完整。
- **符号秩归约很巧妙**：把"任意 $m$-子集能否被选为 top-$m$"这个组合问题精确翻译成 $\binom{N}{m}$-指示矩阵的符号秩，借现成的 $\text{signrank}=2m+1$ 结论直接得到 $d^*\approx 2m$，是把组合几何工具引入 LLM 表达力分析的漂亮一手。
- **$m\approx d/5$** 这样可落地的经验公式，对设计模型的嵌入维度有直接参考价值。

## 局限与展望
- **只覆盖"绝对概率"而非"相对统计的全貌"**：论文证明的是能精确指定 top-$m$ 的概率，但语言建模可能还在意 top token 之外尾部的相对结构；作者也承认 softmax 瓶颈是否在训练时（而非推理时）造成影响仍是开放问题。
- **理论下界依赖近似**：$v$ 的推导用了统计独立性近似（虽有附录验证），且整套随机矩阵分析建立在 i.i.d. 高斯假设上，真实训练后矩阵未必严格满足。
- **"训练不改变能力"是基于 top-$m$ 这个特定指标**：这不代表训练对 OPM 没价值，只是说在这个狭义指标上看不出差别；换个指标结论可能不同。
- **上界只保证"可被选为 top-$m$"**：要进一步指定精确顺序或精确概率，可能需要比 $2m$ 更大的维度，作者未给出这部分的紧界。

## 相关工作与启发
- **vs Yang 等 [26]（原始 softmax 瓶颈 / Mixture of Softmaxes）**：他们论证对数概率空间是高维、低维 OPM 无法覆盖，因而提出混合 softmax 补救；本文反驳说"覆盖任意分布"不是正确的衡量标准，在 top-$m$ 视角下低维 OPM 已经够用。
- **vs Chang & McCallum [2]（multi-facet softmax）**：他们用 king/queen/woman/man 平行四边形说明无法同时给两个 token 最高概率；本文则给出"任意 $m$-子集都能被选为 top-$m$"的构造性上界，说明这类反例并不普遍限制能力。
- **vs Grivas 等 [7]**：他们实证降低 OPM 秩会掉点、训练损失上升；本文呼应 Bamler [20] 的观点，认为这些改进可能源于隐式正则等其他因素，而非瓶颈本身。
- **vs 关于 $m=1$ 的工作 [6, 9, 3]**：先前用线性规划判断单个 token 能否被选为最可能；本文把同样的 LP 框架推广到任意 $m\ge 1$，是直接的能力扩展。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 重新框定问题 + 把符号秩引入 LLM 表达力分析，视角和工具都新
- 实验充分度: ⭐⭐⭐⭐ 覆盖 5 个真实模型 + 同尺寸随机矩阵，理论与 LP 实证互相印证，但未直接验证对下游性能的影响
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，证明结构完整，理论与实验衔接自然
- 价值: ⭐⭐⭐⭐ 对"softmax 瓶颈是否值得补救"给出有力反证，对模型嵌入维度设计有实际指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Language Identification in the Limit with Computational Trace](language_identification_in_the_limit_with_computational_trace.md)
- [\[ICLR 2026\] Softmax Transformers are Turing-Complete](softmax_transformers_are_turing-complete.md)
- [\[ICLR 2026\] Testing Most Influential Sets](testing_most_influential_sets.md)
- [\[ICLR 2026\] To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking](to_augment_or_not_to_augment_diagnosing_distributional_symmetry_breaking.md)
- [\[ICLR 2026\] FlowNIB: An Information Bottleneck Analysis of Bidirectional vs. Unidirectional Language Models](flownib_an_information_bottleneck_analysis_of_bidirectional_vs_unidirectional_la.md)

</div>

<!-- RELATED:END -->
