---
title: >-
  [论文解读] Quantitative Bounds for Length Generalization in Transformers
description: >-
  [ICLR 2026][学习理论][Transformer] 本文给出 transformer "长度泛化"（在短序列上训练、在任意长序列上仍保持性能）所需**最小训练序列长度的首个定量上界**，核心论证是：只要 transformer 在长序列上的内部行为能被某个长度受限的短序列"模拟"出来，长度泛化就会发生，而这个"短序列上界 $N$"随参数范数、位置周期 $\Delta$、局部性 $\tau$、词表大小 $|\Sigma|$ 和逆误差 $\varepsilon^{-1}$ 多项式（或指数）增长。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "Transformer 理论"
  - "长度泛化"
  - "Transformer"
  - "模拟论证"
  - "有限精度注意力"
  - "训练长度下界"
---

# Quantitative Bounds for Length Generalization in Transformers

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TLSUIyBIfs](https://openreview.net/forum?id=TLSUIyBIfs)  
**代码**: 待确认  
**领域**: 学习理论 / Transformer 理论 / 长度泛化  
**关键词**: 长度泛化, 极限 transformer, 模拟论证, 有限精度注意力, 训练长度下界

## 一句话总结
本文给出 transformer "长度泛化"（在短序列上训练、在任意长序列上仍保持性能）所需**最小训练序列长度的首个定量上界**，核心论证是：只要 transformer 在长序列上的内部行为能被某个长度受限的短序列"模拟"出来，长度泛化就会发生，而这个"短序列上界 $N$"随参数范数、位置周期 $\Delta$、局部性 $\tau$、词表大小 $|\Sigma|$ 和逆误差 $\varepsilon^{-1}$ 多项式（或指数）增长。

## 研究背景与动机
**领域现状**：长度泛化（length generalization, LG）是 LLM 训练里的核心难题——模型只在短序列上训练，却希望它能外推到更长的输入。已有大量经验工作发现 LG 的成败高度依赖任务：有的算法任务能轻松外推，有的（如带重复 token 的字符串复制）则彻底失败。理论侧，Zhou et al. (2023) 提出 RASP-L 猜想（能被"简单"RASP-L 程序表达的任务才能 LG），Huang et al. (2025) 进一步用"极限 transformer（limit transformer）"把这个猜想部分证明出来，说明 C-RASP 可表达的任务在**某个有限训练长度**之后就能 LG。

**现有痛点**：Huang et al. (2025) 的结论是**渐近性**的，用的是"极限中辨识（identification in the limit）"那一类论证——只能说"存在某个有限阈值 $N$ 使得 LG 发生"，但完全没说这个 $N$ 到底多大。对实践者而言这等于没给指导：到底训练上下文要拉到多长才够？跟任务复杂度、模型规模是什么关系？

**核心矛盾**：渐近"存在性"证明与实际"要训多长"之间隔着一道定量鸿沟。要把它补上，必须找到一个能同时刻画**模型复杂度**和**任务结构**的量，并把它和最小训练长度 $N$ 显式联系起来。

**本文目标**：沿用 Huang et al. (2025) 的极限 transformer 框架，给出定量界——求最小的 $N$，使得"两个极限 transformer $f,g$ 在所有长度 $\le N$ 的输入上近似一致"能推出"它们在**任意长度**输入上也近似一致"。

**切入角度**：作者注意到 transformer 前向传播的输出，本质上只依赖序列的一些**充分统计量**（如各 token 在注意力里的经验频率、$\tau$-后缀的局部模式）。如果能用一个短序列 $z$ 把长序列 $x$ 的这些统计量近似复现出来，那么 $f(x)\approx f(z)$、$g(x)\approx g(z)$，再加上"短序列上 $f,g$ 一致"的假设，就能夹出 $f(x)\approx g(x)$。

**核心 idea**：把"长度泛化"归约为"**用有界长度的短串模拟长串前向传播**"，并对不同设定（有限/无限精度、单层/双层、最坏/平均误差）显式算出这个模拟所需的短串长度上界。

## 方法详解

### 整体框架
本文不是提出新模型，而是给一个**纯理论命题**配上定量界，因此"方法"即论证框架。所有结果都建立在 Huang et al. (2025) 的**极限 transformer**之上：它有无限上下文长度和广义位置编码，但被三条假设约束——$\Delta$-周期（$p_i = p_{i+\Delta}$）、平移不变（$\phi_{l,h}(j,i)=\phi_{l,h}(j+t,i+t)$）、$\tau$-局部（$i>j+\tau$ 时 $\phi_{l,h}(j,i)=0$）。这三条让"无限长模型"可以被有限信息刻画。

全文的统一主线是一个高层的**模拟论证（simulation argument）**：给定两个极限 transformer $f,g$ 和一个任意长输入 $x$，构造一个长度 $\le N$ 的短串 $z$，使 $f(x)\approx f(z)$ 且 $g(x)\approx g(z)$；若已知 $f,g$ 在所有长度 $\le N$ 的输入上一致，则 $f(z)\approx g(z)$，三步夹逼得 $f(x)\approx g(x)$。整篇论文的技术核心就是**针对不同设定，构造出这个 $z$ 并算出它需要多长**。作者把问题切成两大精度制度，再在每个制度内区分误差类型与层数：

- **有限精度（第 4 节）**：匹配 Huang et al. (2025)。当序列足够长时 softmax 退化成 hardmax，注意力变成对 argmax token 的均匀平均。此制度下给出单层 transformer 在 $\ell_\infty$ 误差（定理 4.1）和分布平均误差（定理 4.2）下的界。
- **无限精度（第 5 节）**：所有内部计算用无限精度，更适合多层 transformer（深层 token 表示是离散 token 的连续混合）。此制度下给出双层 transformer 的界（定理 5.2）。

两个制度的技术细节迥异，但"用短串模拟长串"的根本思路完全一致，且对数据需求与各参数关系的定性结论也吻合，并被实验验证。

### 关键设计

**1. 极限 transformer + 模拟论证：把"长度泛化"归约为"短串可模拟长串"**

长度泛化最难处理的地方在于"任意长"——你没法对每个长度逐一验证。作者的破题点是把它转成一个**可构造的归约**：只要对任意长串 $x$ 都能造出长度 $\le N$ 的短串 $z$ 满足 $f(x)\approx f(z)$ 且 $g(x)\approx g(z)$，那"$f,g$ 在 $\le N$ 上一致"就足以推出它们在任意长度上一致。

$$f(x)\approx f(z)\approx g(z)\approx g(x)$$

中间那步 $f(z)\approx g(z)$ 来自"短串上一致"的假设，两端来自模拟的保真度。构造 $z$ 的关键是让它**近似保留前向传播真正依赖的充分统计量**：有限精度下是 hardmax 注意力里各 token 类型的经验频率比例，无限精度下是 $(\tau\text{-后缀}, \text{经验直方图})$ 的联合分布。定理 4.1 用**确定性**构造（显式让 $z$ 逼近每个 token 的经验频率），定理 5.2 用**随机**构造（从特制分布采样 $z$ 再用概率方法证存在性）。这一归约是全文复用的"骨架"，把一个关于无限多长度的全称命题压成一个关于有限长度短串的存在性问题。

**2. 有限精度下的 hardmax 退化与 logit margin：单层界 $N\sim \tau\Delta L^2/\varepsilon^2$**

有限精度（$p$ 比特，绝对值 $\le 2^{-p}$ 的量被舍入为 0）带来一个关键化简：当序列够长时 softmax 行为等价于 hardmax。作者定义 transformer 的 **logit margin** $\gamma(f)$ 为"最大注意力 logit 与任何非最大 logit 之间的最小非零间隙"：

$$\gamma(f) := \min_{y\in\Sigma,\, i\in\mathbb{Z}/\Delta}\ \min_{\substack{a,a'\in\mathcal{A}_{(y,i)}\\ a-a'>0}} (a-a')$$

其中 $\mathcal{A}_{(y,i)}$ 收集了处理 token $y$ 时所有可能观测到的注意力 logit。当 $N=|x|\ge 2^{p}/\gamma(f)$ 时，任何严格次大的 logit 经指数化后 $s_j\le \exp(-p\log 2 / \gamma\cdot\gamma)=2^{-p}$ 被舍入为 0，于是注意力只对 argmax token 做均匀平均——这就把连续 softmax 变成了一个**离散的"token 频率比例"问题**，模拟 $z$ 只要复现这些比例即可。

定理 4.1（$\ell_\infty$ 误差）给出
$$N = O\!\left(\max\Big\{2^{p}/\gamma,\ \tfrac{L^2\Delta^7|\Sigma|^6\tau^2}{\varepsilon^2}\Big\}\right),$$
其中 $L$ 是 $f,g$ 的参数范数综合量。一旦超过 $2^p/\gamma$ 这道"进入 hardmax 制度"的门槛，所需训练长度就随周期 $\Delta$、参数范数 $L$、词表 $|\Sigma|$、逆精度 $\varepsilon^{-1}$ **多项式**增长；论文还指出 hardmax 行为可能在更小长度就发生，此时 $N$ 只需随 $\tau\Delta L^2/\varepsilon^2$ 增长。定理 4.2 把 $\ell_\infty$ 换成更实用的**分布平均误差**：token 从 Dirichlet 先验 $\mathrm{Dir}((\alpha_s))$ 抽取的分布上取期望，得到 $N_0=\tilde O(\varepsilon^{-2-2\alpha_0^{-1}})$（$\alpha_0=\min_s\alpha_s$），且平均误差控制有个有趣的代价——结论从 $O(\varepsilon)$ 弱化到 $O(\varepsilon^{1/2})$。作者强调这种平均设定**必须**有某种分布正则性假设：若短序列只支撑在 $\Sigma_{\text{short}}\subsetneq\Sigma$、长序列支撑在补集上，切换点可任意大，则界根本不存在。

**3. 无限精度下的 $\tau$-后缀对数缩放与复杂度度量：双层界 $N\lesssim C(f)\,\varepsilon^{-\max(\gamma^{-1},3)}$**

无限精度更适合多层 transformer，但带来一个新难题：在无限精度且权重有界时，$\tau$-后缀（局部信息）对计算的影响会随序列长度发散而衰减到 0，这会排除掉 induction head 这类 transformer 经验上明明能学的函数。作者的修法是**只对 $\tau$-后缀 logit 乘以对数因子** $\log i$：

$$a_{i,j}^{(l,h)} = (y_j^{(l-1)})^\top K_{l,h}^\top Q_{l,h}\, y_i^{(l-1)} + \log i\cdot \phi_{l,h}(j,i)$$

这个缩放让局部信息与全局信息的相对重要性可调，依 $\max_{t\le\tau}\phi_{1,h}(i-t,i)$ 与 1 的大小关系产生三种制度：**Token 主导**（$<1$，后缀贡献随 $i\to\infty$ 衰减）、**平衡**（$=1$，后缀与全局贡献同阶）、**位置主导**（$>1$，后缀主宰，注意力集中在最大化 $\phi$ 的局部 token）。论文给出**统一分析**同时覆盖三种制度。

双层界的核心量是**复杂度度量** $C(f)$（定义 5.1），它随第一层权重算子范数**指数**增长、随其余量多项式增长；这种指数依赖是不可避免的，因为把第一层 softmax 看作概率测度空间上的函数时，其 Lipschitz 常数本身就随 $\|K_{1,h}^\top Q_{1,h}\|_{\mathrm{op}}$ 指数增长，而 LG 要求一个与序列长度 $T$ 无关的统一界。配套定义**位置 margin** $\gamma(f)=\min_h(\max P_h - \text{次大})$（$P_h=\{\phi_{1,h}(i-t,i)\}\cup\{1\}$）。定理 5.2 给出
$$N\lesssim \max(C(f),C(g))\,\varepsilon^{-\max(\gamma(f)^{-1},\gamma(g)^{-1},3)}.$$
证明的技术内核是**关键模拟引理**（引理 5.3）：第一层输出 $Y_i^{(1)}$ 可写成 $\tau$-后缀 $x_{i-\tau:i}$ 与经验直方图 $\mu(x_{\le i})=\frac1i\sum_{j\le i}e_{x_j}$ 的 Lipschitz 有界函数，于是只要短串 $z$ 近似保留 $\{(x_{i-\tau:i},\mu(x_{\le i}))\}$ 的联合分布即可。作者用一个 $\{0,1\}$ 上的马尔可夫链采样 $z$ 的下标子集 $I$（让 $z$ 包含 $x$ 的大块连续片段以保留后缀分布），证得存在 $I$ 使 $|z|$ 接近目标长度且统计量误差 $\lesssim (G+L)(\tau+1)/n^{1/3}$。作为应用，in-context $k$-gram 任务（induction head 的推广）可用复杂度 $C(f)=\varepsilon^{-\Theta(k^2)}$、$\gamma(f)\ge 1$ 的双层 transformer 近似，故训练长度 $\varepsilon^{-\Theta(k^2)}$ 足以 LG——说明复杂任务确实需要更丰富（更长）的训练数据。

## 实验关键数据

实验目的不是刷指标，而是**定性验证理论预言的标度关系**：训练长度 $N$ 越大、测试损失的"平台值"越低；任务越复杂（$\omega$/$\Delta$/$|\Sigma|$/$k$ 越大），所需 $N$ 越大。

### 单层 transformer（验证定理 4.1 / 4.2）

| 任务 | 可表达性 | 变化的复杂度参数 | 观察到的标度 |
|------|----------|------------------|--------------|
| SimpleTask（$f^*=\sin(\omega\cdot\frac{c_0-c_1}{c_0+c_1})$） | 单层、无位置编码、$L=\Theta(\omega)$ | 频率 $\omega$ | 平台损失随训练长度单调下降、随 $\omega$（即 $L$）单调上升 |
| ModPTask（位置 $\equiv k\bmod p$ 的 token 均值） | 单层、$\Delta=p$、$L=\Theta(1)$ | 周期 $\Delta$ | 平台损失随训练长度下降、随 $\Delta$ 上升 |

两个任务的左图都显示：固定训练长度时，**测试损失随测试长度增大而趋于一个有限平台**（而非发散），印证"存在有限 $N$ 即可在所有长度达到 $\varepsilon$ 误差"。右图则定量印证 $N$ 随参数范数 $L$（SimpleTask）和周期 $\Delta$（ModPTask）增长。

### 双层 transformer（验证定理 5.2）

在 in-context $k$-gram 任务上训练双层 transformer：测试损失同样随测试长度趋于平台；固定 $k=2$ 时平台损失随词表 $S$ 上升，固定 $S$ 时随局部性 $k$ 上升——与复杂度 $C(f)$ 对 $S$、$\tau$ 的定性依赖一致。

### 关键发现
- **hardmax 假设被实测证实**：ModPTask（$p=5$）训练后，位置 $\not\equiv k\bmod p$ 的 token 注意力概率近 0，位置 $\equiv k\bmod p$ 的获得几乎相同的概率（接近均匀），说明足够训练长度下模型确实进入定理 4.1 依赖的 hardmax 制度。
- **"更复杂任务需要更长训练数据"被定量验证**：所有任务里，平台损失对复杂度参数（$\omega,\Delta,S,k$）单调递增，与理论中 $N$ 随这些量增长一致。
- **margin 依赖待解**：$N$ 对逆 margin $1/\gamma$ 呈指数依赖（定理 4.1 的 $\exp(\gamma^{-1})$、定理 5.2 的 $\varepsilon^{-\gamma^{-1}}$），但 margin 究竟是 LG 的真实瓶颈还是分析的人为产物，论文留作未来工作。

## 亮点与洞察
- **把"任意长度"全称命题压成"有限短串"存在性命题**：模拟论证是全文最漂亮的一招——它把验证无限多长度的不可能任务，转成"构造一个长度 $\le N$ 的短串"，从而让定量界成为可能。这个"用短串模拟长串充分统计量"的思路可迁移到其他无限长输入的泛化分析（如 SSM、GNN）。
- **logit margin / 位置 margin 作为统一的"难度旋钮"**：两个制度都用一个 margin 量刻画"注意力分得有多开"，并都对 $1/\gamma$ 呈指数依赖，给出了一个跨设定的难度坐标。
- **$\tau$-后缀对数缩放是个巧妙的建模修补**：无限精度下局部信息本会被冲淡，乘 $\log i$ 既救回了 induction head 这类局部依赖函数的可表达性，又自然衍生出 token 主导/平衡/位置主导三种制度，且能被一套分析统一覆盖。
- **理论标度被合成实验逐项对上**：$L,\Delta,|\Sigma|,k$ 每一个参数都有对应的损失曲线印证，把抽象的界变成可观察的现象，这种"理论—实验逐项对账"很有说服力。

## 局限与展望
- **仅覆盖单层与双层**：定理 4.x 是单层、定理 5.2 是双层，作者明确把"扩展到更大深度"列为最重要的未来方向；深层 transformer 的 LG 定量界仍是开放问题。
- **界可能不紧**：$N$ 对 $1/\gamma$ 的指数依赖、定理 5.2 复杂度对第一层权重的指数依赖，是否反映真实瓶颈尚不清楚；作者自承 margin 依赖可能是分析的人为产物。
- **分布假设较强**：平均误差结果（定理 4.2）依赖 Dirichlet 先验等正则性假设，且平均设定下误差从 $O(\varepsilon)$ 弱化为 $O(\varepsilon^{1/2})$；作者指出寻找 LG 在平均情形成立的**最小**分布条件（如马尔可夫链的平稳分布集中）是有趣的开放问题。
- **极限 transformer 的理想化**：分析对象是有无限上下文、$\Delta$-周期、$\tau$-局部、平移不变的理想化模型，与真实有限上下文、带 RoPE/ALiBi 的架构之间仍有距离。

## 相关工作与启发
- **vs Huang et al. (2025)**：本文直接沿用其极限 transformer 框架，但把"存在某个有限 $N$"的**渐近存在性**升级为 $N$ 随各参数显式标度的**定量上界**——这是从"能不能"到"要多大"的实质推进。
- **vs RASP-L 猜想 (Zhou et al. 2023) / Yang et al. (2025)**：RASP-L 用程序长度刻画哪些任务可 LG（定性分层），本文用 margin 与复杂度度量给出可 LG 任务内部的"难度连续谱"和定量数据需求，是对这条线的细化。
- **vs Chen et al. (2025)**：表面上最接近（同样给 LG 的非渐近界），但他们针对变长输入的**一般计算模型**而非 transformer，提供的是互补视角；本文紧扣 transformer 架构本身的注意力机制。
- **vs Wang et al. (2024) / Golowich et al. (2025)**：这些工作分析特定任务（稀疏 token 选择）或自注意力头抽象的 LG，本文的模拟论证框架更通用，覆盖了有限/无限精度、单/双层、最坏/平均误差多种设定。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 transformer 长度泛化的定量训练长度上界，把渐近存在性推进到显式标度。
- 实验充分度: ⭐⭐⭐⭐ 合成任务逐项验证了理论标度，但只在小规模玩具任务上、未触及真实 LLM。
- 写作质量: ⭐⭐⭐⭐ 模拟论证主线清晰、两制度对照工整，但理论密度高、margin 等量需要细读才懂。
- 价值: ⭐⭐⭐⭐⭐ 为"训练上下文该拉多长"提供了理论指导，并建立了 LG 难度的可量化坐标。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Queue Length Regret Bounds for Contextual Queueing Bandits](queue_length_regret_bounds_for_contextual_queueing_bandits.md)
- [\[ICLR 2026\] Transformers Are Inherently Succinct](transformers_are_inherently_succinct.md)
- [\[ICLR 2026\] Efficient Turing Machine Simulation with Transformers](efficient_turing_machine_simulation_with_transformers.md)
- [\[ICLR 2026\] Probability Distributions Computed by Autoregressive Transformers](probability_distributions_computed_by_autoregressive_transformers.md)
- [\[ICLR 2026\] Understanding and Relaxing the Limitations of Transformers for Linear Algebra](understanding_and_relaxing_the_limitations_of_transformers_for_linear_algebra.md)

</div>

<!-- RELATED:END -->
