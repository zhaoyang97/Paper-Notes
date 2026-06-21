---
title: >-
  [论文解读] Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression
description: >-
  [ICLR 2026][学习理论][缩放律] 本文在线性回归 + 多轮 SGD 的可解析设定下，定义并刻画了"有效复用率" $E(K,N)$——把同一份 $N$ 样本数据训练 $K$ 轮，等价于一次过训练多大的数据集——并证明 $E(K,N)$ 不只依赖轮数 $K$，还随数据集规模 $N$ 增大而增大（强凸下饱和点 $\Theta(\log N)$，Zipf 下为 $N$ 的幂），即"**数据集越大，越能多次重复**"，从而修正了 Muennighoff et al. (2023) 中 $E(K,N)\approx K$（与 $N$ 无关）的隐含假设。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "数据缩放律"
  - "缩放律"
  - "多轮训练"
  - "数据复用"
  - "SGD"
  - "线性回归"
  - "强凸"
  - "Zipf 分布"
---

# Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0CXjpAxHUE](https://openreview.net/forum?id=0CXjpAxHUE)  
**代码**: 待确认  
**领域**: 学习理论 / 数据缩放律  
**关键词**: 缩放律, 多轮训练, 数据复用, SGD, 线性回归, 强凸, Zipf 分布  

## 一句话总结
本文在线性回归 + 多轮 SGD 的可解析设定下，定义并刻画了"有效复用率" $E(K,N)$——把同一份 $N$ 样本数据训练 $K$ 轮，等价于一次过训练多大的数据集——并证明 $E(K,N)$ 不只依赖轮数 $K$，还随数据集规模 $N$ 增大而增大（强凸下饱和点 $\Theta(\log N)$，Zipf 下为 $N$ 的幂），即"**数据集越大，越能多次重复**"，从而修正了 Muennighoff et al. (2023) 中 $E(K,N)\approx K$（与 $N$ 无关）的隐含假设。

## 研究背景与动机

**领域现状**：缩放律（Kaplan、Chinchilla）已成为刻画 LLM 预训练性能的核心框架，但它们几乎都建立在"一次过"(one-pass) 范式上——每条数据至多用一次。随着模型对数据的需求从 GPT-2 的不到 10B token 暴涨到 Qwen3 的 36T token，公开高质量数据预计 2028 年前后耗尽，"在同一份数据上多训几轮"(multi-epoch) 成为现实中绕不开的应对手段。

**现有痛点**：多轮训练对缩放律的影响在理论上几乎空白。Muennighoff et al. (2023) 给出经验近似 $N'(K,N)=\big[1+R^*(1-e^{-(K-1)/R^*})\big]\cdot N$（拟合常数 $R^*\approx15.39$），结论是"前 4 轮重复数据几乎和新鲜数据一样好"。但这个公式有两处可疑：一是它与真实曲线仍存在明显 gap；二是它隐含 $E(K,N)=N'/N$ **只依赖 $K$、与 $N$ 无关**，即无论数据集多大，重复 $K$ 次的收益都一样。

**核心矛盾**："重复数据的边际收益何时枯竭"这个问题，被经验公式简化成了一个只关于 $K$ 的常数曲线；但直觉上，更大的数据集应该能撑住更多轮重复才饱和。到底 $N$ 是否进入了有效复用率的刻画，缺乏严格回答。

**本文目标**：在足够简单（线性回归）却仍可精确分析（SGD 非渐近）的设定下，把"训练 $K$ 轮 $N$ 样本 ≈ 一次过训练多少样本"这一比值 $E(K,N)$ 解析地算出来，揭示其对 $K$ 和 $N$ 的双重依赖。

**核心 idea**：**(数据规模进入复用率)** 定义有效复用率 $E(K,N):=\frac1N\min\{N'\ge0:\bar R^*(1,N')\le \bar R^*(K,N)\}$（最优学习率下，匹配 $K$ 轮性能所需的一次过样本数相对 $N$ 的倍数），并证明它存在从"线性增益"到"饱和"的相变，而**相变点随 $N$ 增大而推后**。

## 方法详解

### 整体框架
论文是纯理论分析，主线是：先固定一个标准线性回归 + 多轮随机重排 SGD 的设定，在"最优学习率"口径下定义有效复用率 $E(K,N)$；然后在两类典型数据谱（强凸 / Zipf 幂律）下分别给出 $\bar R^*(K,N)$（最优超额风险）的小-$K$ 与大-$K$ 双区间渐近展开，再由 $\bar R^*(1,N')=\bar R^*(K,N)$ 反解出 $E(K,N)$；最后用合成数据与 LLM 预训练实验交叉验证理论相变。

```mermaid
flowchart LR
    A[线性回归 + K轮重排SGD<br/>初始 w0=0, 学习率 η] --> B[定义最优超额风险<br/>R*(K,N)=min_η E[R(w_KN)]]
    B --> C[定义有效复用率<br/>E(K,N)=N'/N]
    C --> D1[强凸谱<br/>λd≥μ]
    C --> D2[Zipf幂律谱<br/>pi∝i^-α]
    D1 --> E1["R* 小K: ~log(KN)/KN<br/>大K: ~1/N"]
    D2 --> E2["R* 小K: (KN)^-(a-1)/a<br/>大K: N^-(a-1)/(a-b)"]
    E1 --> F1["E(K,N): K 或 Θ(log N)"]
    E2 --> F2["E(K,N): K 或 Θ(N^b/(a-b))"]
    F1 --> G[相变点随 N 增大而推后<br/>→ 数据越大越能多重复]
    F2 --> G
```

### 关键设计

**1. 有效复用率 $E(K,N)$：把"重复 vs 新鲜"翻译成一个可比的倍数。** 多轮训练好不好，本质上是在问"重复利用旧数据，能顶多少新鲜数据"。作者在 MSE 线性回归里定义超额风险 $R(w):=L(w)-\frac12\sigma^2$（扣掉不可约噪声），并把 $K$ 轮 $N$ 样本 SGD 的最优超额风险记为 $\bar R^*(K,N):=\min_{\eta\in(0,1/D^2]}\mathbb E_{w\sim W_{K,N,\eta}}[R(w)]$。关键在于让一次过和多轮训练都各自调到最优学习率后再比，从而排除学习率选择带来的混淆。由此 $E(K,N)=\frac1N\min\{N':\bar R^*(1,N')\le\bar R^*(K,N)\}$ 就是一个干净的、可被解析公式刻画的量——"想达到 $K$ 轮的损失，一次过需要把数据放大多少倍"。

**2. 强凸情形的双区间缩放律与相变。** 在强凸假设 $\lambda_d\ge\mu$、参数先验 $w^*_i\ne0$、以及计算可行轮数 $K=O(N^{0.1})$ 下，作者先精确刻画最优超额风险（Theorem 4.1）：

$$\bar R^*(K,N)=\begin{cases}\dfrac{\sigma^2\mathrm{tr}(H)}{8\lambda_d}\,(1+o_N(1))\cdot\dfrac{\log(KN)}{KN}, & K=o(\log N)\\[2mm]\dfrac{\sigma^2 d}{2}\,(1+o_N(1))\cdot\dfrac{1}{N}, & K=\omega(\log N).\end{cases}$$

直觉是：当 $K\ll\log N$，风险按 $\Theta(\log T/T)$（$T=KN$）下降，多走一轮约等于多过一遍新鲜数据；当 $K\gg\log N$，风险退化为 $\Theta(1/N)$，与 $K$ 无关——再怎么重复也无用。把它代入定义即得 $E(K,N)$（Theorem 4.2）：小 $K$ 时 $E(K,N)=K(1+o(1))$（**有效复用区**），大 $K$ 时 $E(K,N)=\frac{\mathrm{tr}(H)}{4\lambda_d d}(1+o_N(1))\cdot\log N$（**有限复用区**）。两段在 $\lim_{N\to\infty}K/\log N$ 由 $0$ 变 $\infty$ 处发生相变，饱和值封顶在 $\Theta(\log N)$。

**3. "数据越大越能多重复"：相变点随 $N$ 推后。** 这是论文标题所指的核心洞察。固定数据分布，能停留在"有效复用区"的最大轮数随 $N$ 增大而增大——因为相变发生在 $K\sim\log N$ 处。于是收集到 $N$ 个样本后做多轮训练，可逼近一次过训练 $\Theta(N\log N)$ 个样本的性能，**对 $N$ 是超线性的**。这直接戳破了 Muennighoff et al. (2023) 把"有效轮数"当成跨数据规模均匀常数的假设：他们的 $K\le4$ 时 $E\approx K$ 只是"$N$ 不够大、还没到相变点"的特例，真正的临界轮数取决于数据规模与分布。

**4. Zipf 幂律谱：饱和点从 $\log N$ 升级为 $N$ 的幂。** 自然数据常呈长尾，作者进一步分析 Zipf 分布的 one-hot 数据（采样概率 $p_i=c\,i^{-\alpha}$，无标签噪声）。在幂律谱假设（$p_i=ci^{-(a-b)}$、$\Lambda_i=i^{-b}$、$a-b>1$）下，$\bar R^*(K,N)$ 在小 $K$ 区按 $(KN)^{-(a-1)/a}$ 衰减、大 $K$ 区按 $N^{-(a-1)/(a-b)}$ 衰减且与 $K$ 无关（Theorem 5.1），从而 $E(K,N)$ 在 $K=o(N^{b/(a-b)})$ 时 $\approx K$，在 $K=\omega(N^{b/(a-b)})$ 时饱和到 $\Theta(N^{b/(a-b)})$（Theorem 5.2）。同样的"线性增益 → 饱和"结构再现，但**饱和点是 $N$ 的幂而非对数**，幂指数由 Hessian 特征值衰减率与参数范数衰减率共同决定；若改用对数幂律谱（Assumption 5.3），饱和点又变为 $\Theta(\log^b N)$。这说明饱和点的具体阶取决于数据分布形状，是一个"分布相关"的量。

**5. 服务于分析的技术副产物：多轮 SGD 的最优学习率。** 为了把上述风险算到 $o(1)$ 相对误差，作者推导了多轮 SGD 在线性回归下的最优学习率（Lemma 4.4）及对应的期望超额风险近似公式（Lemma G.1，含显式控制的高阶余项）。这些结果本身具有独立价值；实验中合成数据也据此取 $\eta\propto\log(KN)/KN$ 并网格搜索常数系数 $c^*$。此外作者指出 $\log N$ 这个因子与"恒定学习率"绑定：若改用 $\eta_t=\eta_0/(1+bt^\alpha)$ 这类衰减学习率，多轮训练的有效复用率最多为 $O(\kappa_H)$（$\kappa_H=\lambda_1/\lambda_d$ 为条件数），是一个值得后续研究的不同标度行为。

## 实验关键数据

### 主实验（LLM 预训练验证，Section 6.3）

| 设定 | 配置 |
|------|------|
| 模型 | 0.3B 参数，改自 Qwen2.5-0.5B 架构 |
| 数据 | DCLM 子集，共 200B token；新鲜数据规模 0.2B/0.5B/0.8B/1.0B/2B，各训 100 轮 |
| 对照 | 200B 新鲜 token 一次过 |
| 学习率 | 恒定调度（对齐理论，规避调度混淆） |
| 饱和点判据 | $E(K,N)=\lambda K$，取 $\lambda=0.75$ |

| 现象 | 观测结果 | 对应理论 |
|------|----------|----------|
| 小 $K$（$\lesssim5$） | $E(K,N)$ 随 $K$ 近似线性增长（贴黑色虚线 $E=K$） | 复现 Muennighoff $K\le4$ 结论 |
| 大 $K$ | $E(K,N)$ 随新鲜数据规模增大而升高并饱和 | 推翻"有效轮数跨规模均匀"假设 |
| 饱和点 $K(\lambda,N)$ vs $N$ | 线性拟合 $y=0.80\log x+5.21$，$r=0.97$ | 印证强凸下饱和点 $\Theta(\log N)$ |

### 合成数据消融（Section 6.1–6.2）

| 设定 | 关键结果 |
|------|----------|
| 强凸（$d=100$，$\sigma=0.1$，$\eta\propto\log(KN)/KN$） | $E(K,N)$ 随 $\log N$ 先近线性增、后在 $E\approx K$ 处饱和，与 Theorem 4.2 相变完全吻合 |
| Zipf 幂律（$d=10^5$，$a=4.5$，$b=1$） | 大-$K$ 区幂律拟合指数 $0.279\approx b/(a-b)=2/7$，与 Theorem 5.2 一致 |

### 关键发现
- 有效复用率存在清晰的"有效复用区（$E\approx K$）→ 有限复用区（饱和）"相变，相变点随 $N$ 增大而推后。
- 强凸下饱和点 $\Theta(\log N)$、Zipf 下 $\Theta(N^{b/(a-b)})$，理论预测的标度在合成与真实 LLM 实验上都得到验证。
- 实践含义：收集到 $N$ 数据后多轮训练可逼近 $\Theta(N\log N)$ 一次过数据的性能，对数据稀缺时代的训练预算分配有直接指导意义。

## 亮点与洞察
- **把模糊的经验问题钉成一个可解析的量**：$E(K,N)$ 的定义干净且可比（双方都取最优学习率），让"重复 vs 新鲜"第一次有了严格的非渐近刻画。
- **指出被忽略的 $N$ 依赖**：核心结论"数据越大越能多重复"既反直觉又有用，直接修正了被广泛引用的 data-constrained 缩放律的隐含假设。
- **理论与 LLM 实验闭环**：$0.80\log N$ 的拟合（$r=0.97$）与强凸 $\Theta(\log N)$ 理论高度一致，从 100 维线性回归一路打通到 0.3B LLM 预训练，说服力强。
- **技术副产物有独立价值**：多轮 SGD 最优学习率与风险展开公式（Lemma 4.4 / G.1）可被其他分析复用。

## 局限与展望
- **模型简化**：核心结论建立在线性回归 + MSE 上，虽用 LLM 实验佐证，但深网/交叉熵下的 $E(K,N)$ 形式仍是开放问题。
- **恒定学习率假设**：$\log N$ 饱和因子与恒定学习率绑定；作者初步指出衰减学习率下复用率可能降为 $O(\kappa_H)$，但未给完整刻画。
- **轮数上界**：强凸分析要求 $K=O(N^{0.1})$（为控误差），实际超大轮数区间未覆盖；指数 0.1 也非紧。
- **饱和点分布依赖**：饱和点阶（$\log N$ / $N$ 幂 / $\log^b N$）随数据谱变化，意味着真实数据上的具体临界轮数仍需逐分布估计，难以给出统一常数。

## 相关工作与启发
- **数据复用经验研究**：Muennighoff et al. (2023)、Xue et al. (2023) 等就"前几轮重复 ≈ 新鲜数据"与"多轮退化"展开争论；本文从理论给出何时成立、何时枯竭的精确边界。
- **缩放律**：Kaplan / Chinchilla 建立一次过范式下的标度，本文补上多轮范式缺失的一块。
- **最相关理论工作 Lin et al. (2025)**：同样分析线性回归数据复用，但只给出小 $K$ 时 $E(K,N)=\Theta(K)$；本文进一步给出 $o(1)$ 相对误差的显式损失刻画与跨多种数据谱的完整复用率标度。
- **启发**：在数据稀缺、预算受限的预训练中，"采更多数据 + 适度多轮"可能比"死磕一次过"更划算，且最优轮数应显式建模数据规模与分布，而非套用固定常数。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首次把多轮训练有效复用率 $E(K,N)$ 解析刻画到 $o(1)$ 精度，并揭示其对 $N$ 的依赖，修正被广泛引用的经验假设。
- **实验充分度**: ⭐⭐⭐⭐ 合成数据（强凸 + Zipf）+ 0.3B LLM 预训练交叉验证，$0.80\log N$ 拟合 $r=0.97$；但受限于理论设定，恒定学习率、轮数上界等区间未全覆盖。
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，定理分区间陈述、相变叙述直观，理论与实验对应明确。
- **价值**: ⭐⭐⭐⭐⭐ 对数据耗尽时代的预训练预算分配与缩放律建模有直接、可操作的指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)
- [\[ICLR 2026\] Learning under Quantization for High-Dimensional Linear Regression](learning_under_quantization_for_high-dimensional_linear_regression.md)
- [\[ICLR 2026\] Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias](closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression.md)
- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICLR 2026\] Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis](robustness_of_probabilistic_models_to_low-quality_data_a_multi-perspective_analy.md)

</div>

<!-- RELATED:END -->
