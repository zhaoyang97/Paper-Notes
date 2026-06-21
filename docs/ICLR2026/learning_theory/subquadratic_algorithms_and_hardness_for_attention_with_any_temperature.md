---
title: >-
  [论文解读] Subquadratic Algorithms and Hardness for Attention with Any Temperature
description: >-
  [ICLR 2026][学习理论][注意力机制] 本文回答了"任意温度下注意力能否被快速计算"这一基础问题：在头维度 $d=O(1)$ 时给出首个对熵界 $B$ 仅呈对数依赖的真·亚二次算法 $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$，并用 Max-IP / OV 归约证明在 $d=2^{\Omega(\log^* n)}$ 及 $d=\mathrm{poly}(n)$ 两个区间内标准算法本质最优，从而几乎完整刻画了注意力计算的复杂度图谱。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "注意力复杂度"
  - "注意力机制"
  - "亚二次算法"
  - "细粒度复杂度"
  - "多项式方法"
  - "SETH"
---

# Subquadratic Algorithms and Hardness for Attention with Any Temperature

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PSaJZktut7](https://openreview.net/forum?id=PSaJZktut7)  
**代码**: 无（理论论文）  
**领域**: 学习理论 / 注意力复杂度  
**关键词**: 注意力机制, 亚二次算法, 细粒度复杂度, 多项式方法, SETH

## 一句话总结
本文回答了"任意温度下注意力能否被快速计算"这一基础问题：在头维度 $d=O(1)$ 时给出首个对熵界 $B$ 仅呈对数依赖的真·亚二次算法 $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$，并用 Max-IP / OV 归约证明在 $d=2^{\Omega(\log^* n)}$ 及 $d=\mathrm{poly}(n)$ 两个区间内标准算法本质最优，从而几乎完整刻画了注意力计算的复杂度图谱。

## 研究背景与动机

**领域现状**：Transformer 的注意力机制要对 $Q,K,V\in\mathbb{R}^{n\times d}$ 计算 $\mathrm{softmax}(QK^\top)V$，标准做法显式构造 $n\times n$ 的注意力矩阵 $A=\exp(QK^\top)$，因此需要 $O(n^2 d)$ 时间，在长上下文 $n$ 上呈二次增长。由于输入输出本身只有 $O(nd)$ 大小，人们自然追问：注意力能否在不显式构造 $A$ 的前提下亚二次甚至近线性地算出来？

**现有痛点**：此前 Alman & Song (2024) 给出的刻画是：当 $d=\Theta(\log n)$ 时，注意力可在 $n^{1+o(1)}$ 时间内算出，**当且仅当**输入元素界 $B=o(\sqrt{\log n})$；否则在 SETH 假设下需要 $n^{2-o(1)}$ 时间。但他们的快速算法对 $B$ 呈**指数依赖** $2^{O(B^2)}$，一旦 $B$ 稍大就连多项式时间都达不到。

**核心矛盾**：$B$ 小等价于 softmax 用**高温度** $T$（$A=\exp(QK^\top/T)$，高温让分布趋于均匀、近乎对所有 value 取平均）。也就是说现有快速算法只在"高温、近乎不做选择"的退化情形下成立；而温度恰恰是机器学习里影响精度与表达力的关键超参（低温/可学习温度在 NLP、视觉、对比学习中都被证明有用，且高温 Transformer 已被证明表达力更弱）。低熵（低温）情形下没有任何亚二次算法。更糟的是，让运行时随输入数值大小爆炸本身就是坏性质——类比 Knapsack / APSP，人们想要的是随数值**长度**（即 $\log B$）多项式增长的算法。

**本文目标**：① 是否存在对 $B$ 只呈 $\mathrm{polylog}$ 依赖的"真·亚二次"算法（区别于随数值多项式增长的"伪亚二次"）？② 在 $d$ 超常数乃至 $d=\mathrm{poly}(n)$ 时，标准 $O(n^2 d)$ 算法是不是已经最优？

**切入角度**：作者注意到 softmax 分布高度集中——对每个 query，只有内积 $Q_i\cdot K_j$ 接近最大值的少数键贡献非负权重，其余"无关键"可安全丢弃；丢弃后相关键的内积被压缩到一个长度 $O(t)$ 的窄区间内，而 $\exp$ 在有界区间上能被**低次多项式**精确逼近，这正好为细粒度复杂度里的"多项式方法"打开了入口。

**核心 idea**：先用半空间几何把每个 query 的相关键筛出来，再在这个窄区间上用低次多项式逼近 $\exp$ 并把多项式**解耦**成只依赖 $q$ 和只依赖 $k$ 的因子，借助单纯形范围搜索数据结构把求和摊到 $\tilde{O}(n^{1-1/d})$ 每次查询——把对 $B$ 的指数依赖换成对数依赖。

## 方法详解

### 整体框架

论文要解决的核心计算是：对每个 $i$，算出 $O_{i,t}=\sum_j p_{i,j}V_{j,t}$，其中 $p_{i,j}\propto\exp(Q_i\cdot K_j)$ 是 softmax 概率。整篇工作分两条线：**算法侧**给出 $d=O(1)$ 时的真·亚二次算法，**下界侧**证明这个上界几乎无法再改进。

算法侧的流水线是三步串行：给定 $Q,K,V\in[-B,B]^{n\times d}$，**第一步**对每个 query $Q_i$ 用半空间筛掉"无关键"，只留下内积接近最大值的相关键（丢弃造成的误差可证明被控制）；**第二步**在相关键内积被压缩到的长度 $O(t)$ 区间上，用低次多项式 $P(x)\approx\exp(x)$ 替换指数，并把 $P(Q_i\cdot K_j)$ 解耦成"只含 $q_i$ 的项 × 只含 $k_j$ 的项"；**第三步**因为求和只在相关键（落在某半空间内的点）上进行，用 Matoušek 的单纯形范围搜索数据结构做预处理，使每个 query 的求和在 $\tilde{O}(n^{1-1/d})$ 时间内完成，对全部 $n$ 个 query 累计得到 $\tilde{O}(n^{2-1/d})$。低秩矩阵情形通过分解把有效维度从 $d$ 降到 $r=\min(\mathrm{rank}(Q),\mathrm{rank}(K))$。

下界侧则独立地把（双色）Max-IP 与 OV 问题归约到注意力计算，从而在 $d=2^{\Omega(\log^* n)}$ 和 $d=\mathrm{poly}(n)$ 两个区间里给出与标准算法相匹配的硬度。

这是一篇纯理论论文，方法是一串数学构造与归约，不含可视化 pipeline，故不画框架图。

### 关键设计

**1. 丢弃无关键：把 softmax 收缩到一个半空间内**

针对"必须遍历全部 $n$ 个键导致二次时间"这一痛点，作者证明对每个 $Q_i$ 只需保留少数"相关键"。定义 $s^{(i)}_{\max}$ 为使半空间 $\{x:Q_i\cdot x\ge s\log(1+\varepsilon)\}$ 至少含一个 $K_j$ 的最大整数；称 $j$ 为**无关**当 $Q_i\cdot K_j < s^{(i)}_{\max}\log(1+\varepsilon)-\log(n/\varepsilon)$，否则**相关**。直觉是：softmax 概率 $\propto\exp(Q_i\cdot K_j)$ 随内积指数衰减，落后最大值超过阈值 $t=\Theta(\log(n/\varepsilon))$ 的键，其归一化概率已可忽略。把概率重新归一化到相关键上得 $p^{(r)}_{i,j}$，对应输出 $O^{(r)}_{i,t}$，论文证明 $\big|O^{(r)}_{i,t}-O_{i,t}\big|\le 3\varepsilon B$（Lemma 3.2），即丢弃无关键只引入可控的加性误差。关键在于"相关键集合 = 落在半空间 $\{x:Q_i\cdot x\ge\max_j Q_i\cdot K_j - t\}$ 内的 $K_j$"——把"按每个 query 排序"的 $n^2$ 操作，转化成一个可用范围搜索高效回答的几何查询。

**2. 相关区间上的多项式逼近 + 解耦：把指数换成可分离的低次多项式**

筛出相关键后，它们的内积 $Q_i\cdot K_j$ 全部落在长度 $O(t)$ 的窄区间里，这正是多项式方法能用的前提：$\exp$ 只在 $|x|\le p$ 时才能被 $p$ 次多项式逼近好。引用 Aggarwal–Alman 的结果，存在次数 $g=\Theta\!\big(\max\{\tfrac{\log(1/\varepsilon)}{\log(\log(1/\varepsilon)/B)},\,B\}\big)$ 的多项式 $P$，使得 $|P(x)-\exp(x)|<\varepsilon\exp(x)$ 对区间内所有 $x$ 成立。作者令 $c_i=\max_j Q_i\cdot K_j-O(t)$，把 $\exp(Q_i\cdot K_j)$ 写成正比于 $\exp(Q_i\cdot K_j-c_i)$（移到 0 附近）再用 $P$ 逼近，得近似概率 $\hat p_{i,j}\propto P(Q_i\cdot K_j-c_i)$，输出 $\hat o_i=\sum_j\hat p_{i,j}v_j$。

为什么这步是关键：和 $\exp(Q_i\cdot K_j)$ 不同，多项式 $P(Q_i\cdot K_j-c_i)$ 可以**解耦**成只依赖 $q_i$ 的因子与只依赖 $k_j$ 的因子之积——于是 $\sum_j P(Q_i\cdot K_j-c_i)v_j$ 中关于 $k_j$ 的部分可以预先在 $\tilde{O}(n)$ 时间内处理好，每个 query 只需 $\tilde{O}(1)$ 次查询。这与 Alman & Song 直接用 $2^{O(B^2)}$ 秩近似 $\exp(QK^\top)$ 的路线不同：本文只在相关索引上做多项式逼近，因而对 $B$ 仅 $\mathrm{polylog}$ 依赖而非指数依赖。

**3. 单纯形范围搜索把算法推到高维与低秩：$\tilde{O}(n^{2-1/d})$ 的来源**

一维时相关键就是"足够大的 $k_j$"，用区间数据结构即可；但 $d>1$ 时不同 $Q_i$ 对应不同的"大"，逐个 query 排序又回到 $n^2$。作者的关键观察是：相关键集合恰好是半空间 $\{x:Q_i\cdot x\ge\max_j Q_i\cdot K_j-t\}$ 内的点，可用 Matoušek (1992) 的**单纯形范围搜索数据结构**处理——用点集 $\{K_j\}$ 初始化（$O(n\log n)$ 预处理），对每个 $Q_i$ 用半空间查询、返回区域内权重之和，单次查询 $\tilde{O}(n^{1-1/d})$。由于高维下多项式 $P$ 的单项式数随 $d$ 指数增长，需要实例化并查询 $2^{\Omega(d)}$ 个 Matoušek 结构，但在常数 $d=O(1)$ 下这只是亚多项式因子。对全部 $i$ 累加即得主定理 $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$（Theorem 1.1）。低秩推广（Theorem 1.2）则把 $Q=U_Q V_Q^\top,\,K=U_K V_K^\top$ 分解，对 $Q'=U_Q$、$K'^\top=V_Q^\top U_K V_K^\top$ 套用主算法，得 $\tilde{O}(nd+n^{2-1/r}\cdot\mathrm{polylog}(B/\varepsilon))$，$r=\min(\mathrm{rank}(Q),\mathrm{rank}(K))$。此外，梯度计算可被一般性地归约到 $O(d)$ 次注意力调用加 $O(nd^2)$ 开销（Theorem 1.3），从而在 $d=O(1)$ 时整个 LLM 训练过程也是真·亚二次的。

**4. 配套下界：用 Max-IP / OV 归约证明标准算法几乎最优**

针对"上界还能不能更快"，作者从两个区间给出硬度。其一（Theorem 1.4）：把（双色）Max-IP 归约到注意力——任何快速 $\mathrm{AttC}(n,d,B,\varepsilon)$ 算法都能算 $\max_{a\in A,b\in B}a\cdot b$；而 Chen (2018) 证明 Max-IP 在 $d=2^{\Theta(\log^* n)}$ 时于 SETH 下需 $n^{2-o(1)}$，故注意力在 $d=2^{\Omega(\log^* n)}$、$B=\mathrm{poly}(n)$ 时也需 $n^{2-o(1)}$，把此前 $d=\Omega(\log n)$ 的硬度大幅推前。其二（Theorem 1.5）：在 $d=\mathrm{poly}(n)$ 时，标准算法用快速矩阵乘需 $O(T_{\mathrm{MUL}}(n,d,n))$；作者在"广义高维 OV 假设"下证明 $\mathrm{AttC}$ 需 $T_{\mathrm{MUL}}(n,d,n)^{1-o(1)}$，即标准算法在此区间**条件最优**，补上了"$O(n^2 d)$ 上界 vs 仅排除 $n^{2-o(1)}$ 的旧下界"之间的巨大缺口。两条下界与上界合在一起，除 $1\ll d\ll 2^{\Theta(\log^* n)}$ 这一狭窄区间外（该区间的运行时恰好匹配 Max-IP 最优已知算法），完整刻画了 $B=\mathrm{poly}(n)$ 时注意力的复杂度。

### 一个例子：一维舍入算法的直觉

以 $d=1$、$q_i=1$ 为例。给定键 $k_1,\dots,k_8$ 与值 $v_1,\dots,v_8$，先按阈值 $q_i k_{\max}-t$ 砍掉内积过小的键（如 $k_1$ 被丢弃）。对保留下来的相关键，把每个 $k_j$ 舍入到 $\bar k_j$ 使 $q_i k_j\le q_i\bar k_j\le q_i k_j+\log(1+\varepsilon)$，于是 $e^{q_i\bar k_j}$ 是 $e^{q_i k_j}$ 的 $(1+\varepsilon)$ 乘性近似；落入同一宽度 $\log(1+\varepsilon)$ 区间的键（如 $\{k_2,k_3\}$、$\{k_6,k_7,k_8\}$）被视为等价、合并计算。由于相关键落在长度 $t$ 的区间内、每段宽 $\log(1+\varepsilon)$，只需考虑 $\tilde{O}(1/\varepsilon)$ 个区间；再用一个 $\tilde{O}(n)$ 预处理、$\tilde{O}(1)$ 查询的前缀和结构对每个 query 求"区间内值之和"，总时间 $\tilde{O}(nB/\varepsilon)$。这套舍入法已在 $B=o(n)$ 时亚二次，但对 $B$ 仍多项式依赖；把"舍入"升级为"相关区间上的多项式逼近 + 解耦"，才得到对 $B$ 仅对数依赖的真·亚二次算法。

## 实验关键数据

本文为纯理论论文，无实验。其核心结论是一张刻画注意力计算复杂度（在 $B=\mathrm{poly}(n)$、$\varepsilon=1/\mathrm{poly}(n)$ 下，略去亚多项式因子）的结果表：

| 头维度 $d$ | 上界（此前） | 上界（本文） | 下界（此前） | 下界（本文） |
|---|---|---|---|---|
| $O(1)$ | $n^2$ | $n^{2-1/d}$ (Thm 1.1) | $n$ | — |
| $2^{\Theta(\log^* n)}$ | $n^2$ | — | $n$ | $n^{2-o(1)}$ (Thm 1.4) |
| $\Theta(\log n)$ | $n^2$ | $n^{2-o(1)}$* | — | $n^{2-o(1)}$ (Thm C.7) |
| $\mathrm{poly}(n)$ | $T_{\mathrm{MUL}}(n,d,n)$ | — | $n^{2-o(1)}$* | $T_{\mathrm{MUL}}(n,d,n)^{1-o(1)}$ (Thm 1.5) |

（标 * 者来自 Alman & Song (2024)；其 $d=\Theta(\log n)$ 下界需 $B=\Omega(\sqrt{\log n})$，本文则在 $B\ge\log 2$ 时即成立。）

关键复杂度结论：

- **主算法**：$d=O(1)$ 时 $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$，首次对 $B$ 仅对数依赖（此前最优为对 $B$ 指数依赖 $2^{O(B^2)}$）。
- **低秩**：$\tilde{O}(nd+n^{2-1/r}\cdot\mathrm{polylog}(B/\varepsilon))$，$r=\min(\mathrm{rank}(Q),\mathrm{rank}(K))$。
- **梯度/训练**：$O(d)$ 次注意力调用 + $O(nd^2)$ 开销 → $d=O(1)$ 时整训练 $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$。
- **下界**：$d=2^{\Omega(\log^* n)}$ 时 SETH 下需 $n^{2-o(1)}$；$d=\mathrm{poly}(n)$ 时广义高维 OV 假设下需 $T_{\mathrm{MUL}}(n,d,n)^{1-o(1)}$，标准算法条件最优。

## 亮点与洞察

- **把"温度/熵界"从假设变成可分析的几何量**：以往快速注意力算法默默假设高温（小 $B$），本文用"丢弃无关键 + 半空间范围搜索"显式处理任意温度，第一次给出对 $B$ 仅 $\mathrm{polylog}$ 依赖的真·亚二次算法，摆脱了"伪亚二次"。
- **多项式方法 + 解耦是核心技巧**：把 $\exp$ 在窄区间上换成可分离的低次多项式，让"每 query 求和"从 $O(n)$ 摊到 $\tilde{O}(n^{1-1/d})$，是从 $n^2$ 跌到 $n^{2-1/d}$ 的关键。
- **上下界几乎吻合**：除 $1\ll d\ll 2^{\Theta(\log^* n)}$ 这一窄缝外，完整刻画了注意力计算复杂度，且把该窄缝的改进与 Max-IP 最优算法的突破画上等号——告诉后来者"想更快得先突破 Max-IP"。
- **训练全程亚二次**：通过把梯度计算一般性归约到注意力计算，结论自动延伸到完整 LLM 训练（$d=O(1)$），而非只对前向。

## 局限性 / 可改进方向

- **指数依赖维度 $d$**：算法需 $2^{\Omega(d)}$ 个 Matoušek 结构，仅在常数 $d$ 下亚多项式；$d=\omega(1)$ 超常数时退化为 $n^{2-o(1)}$，实践中 $d$ 常达几十上百，理论加速难直接落地。
- **逐元素近似而非范数近似**：本文给的是 $\ell_\infty$ 加性保证，与 Performer/Reformer 等线性时间近似（只保证算子范数）不可直接比较；其下界恰说明线性时间不可能给出如此强的逐元素保证。
- **纯理论、无实证**：常数与亚多项式因子可能很大，是否在真实长序列上跑赢 FlashAttention 类工程实现未验证。
- **窄缝未闭合 + 依赖未证假设**：$1\ll d\ll 2^{\Theta(\log^* n)}$ 区间仍开放；高维下界依赖"广义高维 OV 假设"这一较新、尚未被广泛检验的假设。

## 相关工作与启发

- **细粒度复杂度下的注意力**：延续 Alman & Song (2024a;b)、Keles et al. (2023)、Alman & Yu (2025) 的硬度路线，把基于 SETH 的 OV/Max-IP 归约系统用到注意力，并推进到 $d=2^{\Omega(\log^* n)}$ 与 $d=\mathrm{poly}(n)$。
- **近似注意力工程线**：Performer、Reformer、Longformer、BigBird、Nyströmformer 等（Choromanski 2020、Kitaev 2020、Beltagy 2020、Zaheer 2020、Xiong 2021）追求线性时间但只保证矩阵范数近似，本文从理论上界定了它们无法达到逐元素强近似。
- **多项式方法与范围搜索**：借用 Williams、Abboud 等的多项式方法，以及 Matoušek 单纯形范围搜索、Yao/Agarwal/Matoušek 的 Max-IP 算法，把几何数据结构引入注意力计算，是跨子领域的工具迁移范例。
- **温度的重要性**：呼应对比学习与 Transformer 中温度对精度、表示与表达力的研究（Chen 2020、Wang & Liu 2021、Alman & Song 2025b），从计算复杂度角度解释了"为何低温更难算"。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Poly-attention: a general scheme for higher-order self-attention](poly-attention_a_general_scheme_for_higher-order_self-attention.md)
- [\[ICLR 2026\] Quantum Machine Learning Advantages Beyond Hardness of Evaluation](quantum_machine_learning_advantages_beyond_hardness_of_evaluation.md)
- [\[ICLR 2026\] Parameterized Hardness of Zonotope Containment and Neural Network Verification](parameterized_hardness_of_zonotope_containment_and_neural_network_verification.md)
- [\[ICLR 2026\] Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods](understanding_in-context_learning_on_structured_manifolds_bridging_attention_to_.md)
- [\[ICLR 2026\] Critical Attention Scaling in Long-Context Transformers](critical_attention_scaling_in_long-context_transformers.md)

</div>

<!-- RELATED:END -->
