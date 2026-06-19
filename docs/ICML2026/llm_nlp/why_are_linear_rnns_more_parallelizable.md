---
title: >-
  [论文解读] Why Are Linear RNNs More Parallelizable?
description: >-
  [ICML2026][LLM 其他][Linear RNN] 这篇论文用电路复杂度严格解释了为什么 Linear RNN 比传统非线性 RNN 更容易像 Transformer 一样并行：LRNN 可落在近似 log-depth 的算术电路类中，而非线性 RNN 能表达更难并行的 logspace / polynomial-time 完全问题，二者形成表达力与并行性的基本权衡。
tags:
  - "ICML2026"
  - "LLM 其他"
  - "Linear RNN"
  - "并行化"
  - "电路复杂度"
  - "表达能力"
  - "长上下文架构"
---

# Why Are Linear RNNs More Parallelizable?

**会议**: ICML2026  
**arXiv**: [2603.03612](https://arxiv.org/abs/2603.03612)  
**代码**: https://arg-git.informatik.uni-kl.de/pub/LinearRNN  
**领域**: LLM效率 / 序列模型理论 / 并行计算  
**关键词**: Linear RNN, 并行化, 电路复杂度, 表达能力, 长上下文架构  

## 一句话总结
这篇论文用电路复杂度严格解释了为什么 Linear RNN 比传统非线性 RNN 更容易像 Transformer 一样并行：LRNN 可落在近似 log-depth 的算术电路类中，而非线性 RNN 能表达更难并行的 logspace / polynomial-time 完全问题，二者形成表达力与并行性的基本权衡。

## 研究背景与动机
**领域现状**：长上下文 LLM 架构正在重新关注 RNN 和 state-space / linear attention 类模型。Mamba、RWKV、DeltaNet 等 Linear RNN 变体希望兼具递归状态的长度泛化和类似 Transformer 的高并行吞吐，因此“线性递归为什么容易并行”不再只是理论问题，而直接关系到长序列模型设计。

**现有痛点**：大家知道传统 RNN 是顺序更新，Transformer 可以并行；也知道某些 LRNN 能通过 scan 并行。但这只是算法直觉，还没有清楚回答两个更细的问题：第一，非线性 RNN 是否存在不可避免的并行化障碍；第二，不同 LRNN 变体之间是否只差工程实现，还是表达能力上也有严格层级。

**核心矛盾**：模型越表达力强，往往越像通用顺序计算，也越难压缩到浅层并行电路；模型越容易并行，可能又会牺牲某些算法任务的表达力。LRNN 恰好处在中间地带：它比 Transformer 的某些简单类别更强，但似乎没有传统非线性 RNN 那么难并行。

**本文目标**：论文要给 RNN/LRNN 建立一张复杂度地图：非线性 RNN 能表达哪些复杂度类，LRNN 上界在哪里，DPLR、PD、Mamba 等不同线性更新参数化之间又有什么细粒度差异。

**切入角度**：作者把神经网络识别语言的问题映射到电路复杂度与自动机理论。非线性 RNN 通过 counter machine / stack machine 展示顺序计算能力；LRNN 通过矩阵乘法和算术电路展示可并行性；不同 LRNN 参数化则对应不同 weighted finite automata 能力。

**核心 idea**：线性状态更新可以被写成矩阵乘积和求和，因此能由 log-depth 算术电路并行模拟；非线性递推可以模拟更强的顺序机器，所以除非复杂度理论发生重大塌缩，否则无法同等高效并行。

## 方法详解
这篇论文不是提出一个新模型，而是给现有 RNN 家族做理论分类。理解它的关键是：作者把“是否容易并行”转化成“能否用浅层 bounded fan-in 电路模拟”，把“表达能力”转化成“能解决哪个复杂度类的完全问题”。

### 整体框架
论文先定义两大类序列层。非线性 RNN 的状态更新是 $h_t=f(h_{t-1},x_t)$，其中 $f$ 可包含 ReLU/MLP 等非线性。Linear RNN 的状态更新是 $S_t=A_t(x_t)S_{t-1}+b_t(x_t)$，也就是每一步对前一状态做线性变换再加输入相关项。实际多层模型可以像 Transformer 一样交替堆叠 recurrent sublayer 和 feedforward sublayer。

随后作者引入复杂度类：Transformer 和简单 LRNN 常落在 $\mathsf{TC}^0$ 或 $\mathsf{NC}^1$ 附近；LRNN 的一般上界是 $\mathsf{PNC}^1$，也就是 log-depth 算术电路加 positivity check；非线性 RNN 在 log precision 下能解决 $\mathsf{L}$-complete 问题，在 polynomial precision 下甚至能解决 $\mathsf{P}$-complete 问题。最后，论文用两个合成任务验证理论预测：sorted deterministic graph connectivity 和 iterated $3\times3$ matrix multiplication。

### 关键设计

**1. 用电路复杂度把“能不能并行”变成可证明的深度上界/下界**
以往“RNN 慢、Transformer 快”只是工程经验，说不清是实现落后还是本质障碍。论文的核心方法是把一个序列层“识别某种语言”的能力对应到标准复杂度类：如果它能被 $O(\log n)$、或近似 $O(\log n\log^* n)$ 深度的 bounded fan-in 电路模拟，就说明它能像 Transformer 一样压进对数深度、天然好并行；如果它能表达某个复杂度类的 complete 问题，那么在标准复杂度猜想下（如 $\mathsf{PNC}^1\neq\mathsf{L}$、$\mathsf{NC}\neq\mathsf{P}$）就不可能压到同样浅的电路，必然更顺序。这一步是整篇论证的“坐标系”：它把架构比较从模糊的快慢经验，提升成可证明的渐近并行深度差异。之所以重要，是因为上下文长度到 64K–1M 时 $\log n\approx16$–$20$，而 $\log^2 n$ 可达 256–400——理论上的深度差会直接转化成硬件上的顺序时间差。

**2. 非线性 RNN 的表达力下界：它能模拟更强的顺序机器，所以反而难并行**
要解释“为什么偏偏是传统 RNN 难并行”，作者给非线性递推证了一条表达力下界：log precision 的 MLP RNN 可以模拟 counter machine，因此能解决 sorted deterministic graph connectivity 这个 $\mathsf{L}$-complete 问题；放宽到 polynomial precision，它甚至能模拟 multi-stack machine，识别 $\mathsf{P}$-complete 语言。关键洞察是：非线性递推把递归状态当成一块可任意读写更新的顺序存储器，这正是它算法表达力更强的来源；但反过来，要“完全并行模拟”这种本质顺序的计算，在 $\mathsf{PNC}^1\neq\mathsf{L}$、$\mathsf{NC}\neq\mathsf{P}$ 的假设下就必须付出更深的电路代价（log precision 下约需 $\Omega(\log^2 n)$ 深度，比 Transformer 多一个 $O(\log n)$ 因子）。表达力和并行性的权衡由此被钉死。

**3. LRNN 不是铁板一块：DPLR 严格强于 PD**
论文进一步把“线性 RNN”内部拆开，指出参数化选择会改变表达力上界。一般 LRNN 的状态更新 $S_t=A_t S_{t-1}+b_t$ 可展开成矩阵乘积与求和，因此语言识别整体落在 $\mathsf{PNC}^1$；但具体参数化决定了它在这个上界里能走多远——diagonal-plus-low-rank（DPLR）的变体如 RWKV-7、DeltaNet 能表达 iterated $3\times3$ matrix multiplication，达到 $\mathsf{PNC}^1$-complete，是线性可并行范围内表达力最强的一档；而 permutation-diagonal（PD）参数化的矩阵乘积始终保持置换-对角结构，被限制在 $\mathsf{NC}^1$（虽仍能识别 $\mathsf{NC}^1$-complete 语言）。作者还为每一类 RNN 配上一个能被它模拟的自动机模型——LRNN 对应 weighted finite automaton（WFA）、PD 对应其确定性版本 DWFA——从自动机视角佐证这条层级。对架构设计而言，这是一把“尺子”：DPLR 比 PD、Mamba/S4 更能表达迭代代数计算，却仍保持接近 Transformer 的对数并行深度，是表达力与并行性之间一个很有吸引力的中间点。

### 损失函数 / 训练策略
理论部分没有训练损失；实验部分用合成算法任务做二分类或逐步分类，所有模型用 AdamW、BCEWithLogitsLoss、batch size 128、梯度裁剪 1.0，最长训练 60K steps。比较对象包括 nonlinear RNN、Transformer、Mamba、RWKV-7、DeltaNet。训练长度范围为 $[1,100]$，测试还包括 $[101,200]$ 和 $[201,300]$ 的长度外推。

## 实验关键数据

### 主实验
主结果首先是理论分类表。它回答的不是某个 benchmark 分数，而是不同模型家族的“最强能表达什么”和“最浅能并行到什么程度”。

| 模型类别 | 复杂度定位 | 并行深度含义 | 代表模型/任务 | 结论 |
|----------|------------|--------------|---------------|------|
| Transformer / 简单 LRNN | 约 $\mathsf{TC}^0 \subseteq \mathsf{NC}^1$ | $O(\log n)$ bounded fan-in 深度 | Transformer、Mamba 类简单结构 | 最容易并行，但表达力有限 |
| 一般 LRNN | $\mathsf{PNC}^1$ 上界 | 可由 $O(\log n\log^*n)$ 深度模拟 | 线性状态更新家族 | 比 Transformer 多很小并行开销 |
| DPLR LRNN | $\mathsf{PNC}^1$-complete | 接近 LRNN 上界 | RWKV-7、DeltaNet | 在线性可并行范围内表达力最强 |
| PD LRNN | $\mathsf{NC}^1$-complete | log-depth | permutation-diagonal LRNN | 强于简单有限状态，但弱于 DPLR |
| log-precision nonlinear RNN | $\mathsf{L}$-complete | 可能需要 $\Omega(\log^2 n)$ 深度 | MLP RNN 解图连通 | 表达力强，但并行成本更高 |
| poly-precision nonlinear RNN | $\mathsf{P}$-complete | 标准假设下不能 polylog-depth 并行 | MLP RNN 模拟多栈机 | 最强但最顺序 |

合成实验验证理论预测。训练集和验证集长度在 $[1,100]$，测试额外看 $[101,200]$ 和 $[201,300]$。图中报告的是不同长度桶准确率趋势，论文结论可概括如下。

| 任务 | 理论预期 | 表现最强模型 | 表现较弱模型 | 观察 |
|------|----------|--------------|--------------|------|
| Sorted deterministic graph connectivity | $\mathsf{L}$-complete，非线性 RNN 可解，LRNN 难以完全表达 | nonlinear RNN | Transformer、RWKV-7、Mamba、DeltaNet 在 OOD 长度退化 | 所有模型 ID 可学，只有 nonlinear RNN 长度外推接近完美 |
| Iterated matrix multiplication over $\mathbb{Z}_m$ | DPLR LRNN 与 nonlinear RNN 应更强 | RWKV-7、DeltaNet、nonlinear RNN | Transformer、Mamba | DPLR 和非线性 RNN ID 近完美，OOD 只中等退化 |
| Iterated matrix multiplication over $\mathbb{Z}$ | 无模数整数增长，更考验代数状态 | RWKV-7、DeltaNet、nonlinear RNN | Transformer 明显退化，Mamba 低于 top models | DPLR 的线性代数结构非常适合矩阵乘积 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| nonlinear RNN on graph connectivity | OOD 长度仍接近满分 | 符合 $\mathsf{L}$-complete 能力分析 |
| LRNN/Transformer on graph connectivity | 长度越长退化越明显 | 理论上难以覆盖该顺序可达性问题 |
| RWKV-7 / DeltaNet on IMM | ID 和 OOD 都强 | DPLR 能表达 $\mathsf{PNC}^1$-complete 的矩阵乘积 |
| Mamba on IMM | 明显弱于 RWKV-7/DeltaNet | 简单线性参数化表达力不足 |
| Transformer on IMM | 训练内也不稳定，长度外推更差 | attention 的浅并行优势不等于代数递推能力 |
| 统一训练设置 | AdamW、60K steps、batch 128 | 让差异主要来自架构 inductive bias |

### 关键发现
- LRNN 更容易并行的根本原因是线性递推可归约为矩阵乘积 / scan，而矩阵乘积有 log-depth 算术电路实现。
- 非线性 RNN 的“难并行”不是实现落后，而是因为它能模拟更强的顺序计算模型；这种表达力在复杂度意义上会带来深度代价。
- DPLR 是一个很有吸引力的中间点：它比 Mamba/S4 等简单结构更能表达 iterated algebraic computation，但仍保持接近 Transformer 的并行深度。
- 实验虽然小，但和理论预测方向一致，说明这些复杂度结果不只是抽象分类，也能反映可训练模型在合成算法任务上的长度泛化行为。

## 亮点与洞察
- 论文最强的贡献是把 RNN 架构讨论从“经验上快/慢”提升到“复杂度类与 complete problem”的层面。对长上下文架构设计来说，这种理论坐标非常有用。
- 它给出了一个清晰 trade-off：如果想要 nonlinear RNN 的顺序算法能力，就要接受更深的并行模拟；如果想要接近 Transformer 的并行效率，就需要把状态更新限制在线性/可 scan 的形式。
- DPLR 与 PD 的区分很有启发。许多论文会把“线性 RNN”合成一类，但这篇说明低秩项、置换对角结构等参数化选择会改变表达力上界。
- 合成任务选得比较贴理论：graph connectivity 分离 nonlinear RNN 和 LRNN，iterated matrix multiplication 分离 DPLR 和更简单架构，实验不是随便跑语言建模分数。

## 局限与展望
- 复杂度分析依赖精度、uniformity、bounded fan-in 等形式化假设。它解释的是渐近并行深度，不直接等价于 GPU kernel、显存带宽或训练稳定性。
- 实验是合成算法任务，能验证表达力倾向，但不能直接说明大规模语言建模质量。DPLR 在真实 LLM 预训练中的收益还要看数据、优化和硬件实现。
- 理论主要讨论精确模拟和语言识别能力；实际神经网络可以近似计算，也可能用多层 hybrid 结构绕开单层分类的局限。
- 论文指出 nonlinear RNN 的额外表达力可能需要 $\Theta(\log n)$ 并行代价，但是否值得在真实任务中付出这个代价，仍是开放问题。

## 相关工作与启发
- **vs Transformer复杂度分析**: 既有工作认为 Transformer 落在较浅的 $\mathsf{TC}^0/\mathsf{NC}^1$ 区间；本文把 LRNN 放到相邻的 $\mathsf{PNC}^1$，解释其少量并行开销和额外表达力。
- **vs Mamba/S4理论**: 简单 state-space/linear RNN 往往表达力更受限；本文说明 DPLR 结构如 RWKV-7、DeltaNet 能达到更高复杂度类。
- **vs 传统RNN理论**: 早期 RNN 可模拟 stack/counter machine 的思想在这里被重新用于解释 LLM 架构的并行化边界。
- **vs 并行化非线性RNN方法**: 近期 Newton 类方法可把非线性 RNN 并行到 $O(\log^2 n)$；本文的理论结果说明这与 logspace complete 的预期深度基本一致。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用复杂度理论系统解释 LRNN 并行性，并细分 DPLR/PD 表达力，理论贡献很强。
- 实验充分度: ⭐⭐⭐☆☆ 实验和理论匹配度高，但规模较小，主要是合成任务验证。
- 写作质量: ⭐⭐⭐⭐☆ 论文结构清楚，但复杂度符号密集，对非理论读者门槛较高。
- 价值: ⭐⭐⭐⭐☆ 对长上下文 LLM 架构选择很有指导意义，尤其能帮助理解为什么 DPLR 类线性递推值得关注。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](../../ACL2025/llm_nlp/lm_graph_search_supervision.md)
- [\[NeurIPS 2025\] Composing Linear Layers from Irreducibles](../../NeurIPS2025/llm_nlp/composing_linear_layers_from_irreducibles.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] Weight Decay may matter more than μP for Learning Rate Transfer in Practice](../../ICLR2026/llm_nlp/weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)
- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](../../NeurIPS2025/llm_nlp/linear_transformers_implicitly_discover_unified_numerical_algorithms.md)

</div>

<!-- RELATED:END -->
