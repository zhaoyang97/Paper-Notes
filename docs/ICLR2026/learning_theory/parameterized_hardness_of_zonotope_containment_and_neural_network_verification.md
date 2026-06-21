---
title: >-
  [论文解读] Parameterized Hardness of Zonotope Containment and Neural Network Verification
description: >-
  [ICLR 2026][学习理论][参数化复杂性] 本文证明了 2 层 ReLU 网络的正性/满射性判定、zonotope 非包含、以及 Lipschitz 常数计算等一系列验证相关问题在以输入维度 $d$ 为参数时都是 W[1]-hard 的，从而在指数时间假设（ETH）下排除了固定参数可解性，并说明朴素的「枚举线性区域」暴力算法在对 $d$ 的依赖上已基本最优。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "计算复杂性"
  - "神经网络验证"
  - "参数化复杂性"
  - "W[1]-hard"
  - "ReLU 网络验证"
  - "zonotope 包含"
  - "Lipschitz 常数"
  - "指数时间假设"
---

# Parameterized Hardness of Zonotope Containment and Neural Network Verification

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=y8N45EEW05](https://openreview.net/forum?id=y8N45EEW05)  
**代码**: 无（纯理论论文）  
**领域**: 学习理论 / 计算复杂性 / 神经网络验证  
**关键词**: 参数化复杂性, W[1]-hard, ReLU 网络验证, zonotope 包含, Lipschitz 常数, 指数时间假设

## 一句话总结
本文证明了 2 层 ReLU 网络的正性/满射性判定、zonotope 非包含、以及 Lipschitz 常数计算等一系列验证相关问题在以输入维度 $d$ 为参数时都是 W[1]-hard 的，从而在指数时间假设（ETH）下排除了固定参数可解性，并说明朴素的「枚举线性区域」暴力算法在对 $d$ 的依赖上已基本最优。

## 研究背景与动机

**领域现状**：带 ReLU 激活的神经网络在安全攸关场景（自动驾驶、控制）中越来越多，因此「验证」——给定输入集合 $X$，问网络输出是否一定落在目标集合 $Y$ 内——成为核心任务。一批经典问题（网络验证、估计 Lipschitz 常数、zonotope 包含判定）都早已被证明是 (co)NP-hard 的。

**现有痛点**：NP-hardness 只说明在最坏情况下问题难，但没回答一个更精细的问题：实践中输入数据常被假设近似落在低维流形上，那么当**输入维度 $d$ 很小**时，这些问题是否就变得可解（即固定参数可解，FPT）？这正是参数化复杂性关心的事。已知 2 层 ReLU 网络的「单射性」检查可以在 $(d+1)^d \cdot n^{O(1)}$ 时间内做到（关于 $d$ 是 FPT 的），这给了人们「低维或许可解」的希望。

**核心矛盾**：然而网络「正性/满射性」判定以及 Lipschitz 常数的参数化复杂性，被 Froese 等人在 COLT '25 上作为公开问题留下——到底是像单射性那样 FPT，还是本质上随 $d$ 指数爆炸，没有人知道。如果是后者，那「低维假设能救场」的直觉就站不住脚。

**本文目标**：彻底厘清这一族问题关于参数 $d$ 的参数化复杂性，回答 Froese 等人的公开问题，并给出在 ETH 下的运行时间下界。

**切入角度**：所有这些问题都能在 $n^{O(d)} \cdot \mathrm{poly}(N)$ 时间内用「枚举 zonotope 顶点 / 枚举网络线性区域」的暴力法解决（即都在复杂性类 XP 中）。作者要做的是给出一个**让输入维度随团大小线性增长**的归约，从而把它们钉死在 W[1]-hard。

**核心 idea**：从 MULTICOLORED CLIQUE 出发构造一个 2 层 ReLU 网络，使其输入维度恰好等于团的颜色数 $k$（加 1），让「图中存在 $k$-彩色团」等价于「网络取到正输出」；再借助「同质化」去偏置和「ReLU 网络↔zonotope 对偶」两个工具，把这一硬度结论传导到整个问题族。

## 方法详解

### 整体框架

这是一篇纯计算复杂性论文，没有可训练的「方法 pipeline」，其「方法」是一条**参数化归约链**：以 W[1]-hard 的 MULTICOLORED CLIQUE 为源头，一次归约打通整个验证问题族。整体逻辑是：

输入是一个 MULTICOLORED CLIQUE 实例（图 $G=(V=V_1 \dot\cup \cdots \dot\cup V_k, E)$，问是否存在每种颜色恰好一个点的 $k$-团）；输出是一个 2 层 ReLU 网络 $f:\mathbb{R}^{k} \to \mathbb{R}$，其最大值满足

$$\max_{x \in \mathbb{R}^k} f(x) = k + \binom{k}{2} \iff G \text{ 含 } k\text{-彩色团},$$

否则 $\max f \le k + \binom{k}{2} - 1$。由于网络输入维度 $d = k$（去偏置后为 $k+1$）只随团大小**线性**增长，这条归约就把 CLIQUE 的 W[1]-hardness 与 ETH 下界原样搬到网络问题上。随后两步「桥接」把结论扩散：用**同质化**把硬度强化到无偏置网络，用**对偶**把无偏置网络的正性等价转成 zonotope 非包含，再各自延伸到逼近最大值、Lipschitz 常数等问题。

整条链可概括为：MULTICOLORED CLIQUE $\to$（spike+penalty 归约）$\to$ 2 层 ReLU 正性 $\to$（同质化）$\to$ 无偏置正性 $\to$（对偶）$\to$ zonotope 非包含 / Lipschitz 常数。

### 关键设计

**1. 维度可控的 MULTICOLORED CLIQUE 归约：用 spike 与 penalty 函数把团编码进低维网络**

标准 NP-hardness 归约允许输入维度无限增长，但要证 W[1]-hardness，归约后实例的参数（这里是 $d$）必须被源参数（团大小 $k$）的函数界住——这正是难点。作者让网络的每个输入坐标 $x_c$ 专门负责一种颜色 $c$，于是输入维度恰好 $= k$。为每个节点 $v_{c,i}$ 赋一个唯一标签 $\omega_{c,i} \in \mathbb{N}$，并用 Sidon 集保证每条边 $\{v_{r,i}, v_{l,j}\}$ 的标签 $\omega_{r,i,l,j} := \omega_{r,i} + \omega_{l,i}$ 也互不相同。然后构造两类分段线性函数：对每对颜色 $(r,l)$ 造一个 **spike 函数** $s_{r,l}:\mathbb{R}^2 \to [0,1]$，它几乎处处为零，只在「两个输入之和等于某条边标签 $\omega_{r,i,l,j}$」时冲到 1（每个 spike 用 $3|E_{r,l}|$ 个神经元实现）；对每种颜色 $c$ 造一个 **penalty 函数** $p_c:\mathbb{R}\to[0,1]$，在每个节点标签 $\omega_{c,i}$ 附近有一个窄尖峰、别处为零（用 $3 n_c$ 个神经元）。把它们并行计算再求和得到

$$f(x_1,\dots,x_k) = \sum_{(r,l)\in\binom{[k]}{2}} s_{r,l}(x_r, x_l) + \sum_{c\in[k]} p_c(x_c).$$

要让 $f$ 逼近上界 $k+\binom{k}{2}$，必须所有 penalty 与 spike 同时取正：penalty 取正逼着每个 $x_c$ 贴近某个节点标签（选出一个点），spike 取正逼着这些被选节点两两相邻——合起来正好是一个 $k$-彩色团。整个网络只有 $3(|V|+|E|)$ 个 ReLU 神经元，构造在多项式时间内完成。

**2. 同质化：把含偏置的硬度实例改写成无偏置网络，扫清向下游传导的障碍**

上面的归约依赖偏置（penalty/spike 的「平移尖峰」需要偏置定位）。但下游若干结论（尤其是 zonotope 对偶）只对**正齐次**、无偏置的网络成立，因此必须证明「即使强制所有偏置为零，2 层 ReLU 正性仍是 W[1]-hard」。作者引入**同质化**（Definition 4.1）：给网络加一个额外输入变量 $y$，把第一隐层每个神经元的偏置 $b$ 换成 $y\cdot b$，把输出神经元的偏置 $b$ 换成 $|y|\cdot b$（用两个额外神经元实现 $|y|$）。这样得到的网络全部偏置为零，却在 $y=1$ 的切片上复原了原函数，且把它延拓成正齐次函数（见原文 Figure 3 的几何示意）。先把设计 1 网络的输出偏置设为 $1-k-\binom{k}{2}$，使「正输出 $\iff$ 存在 $k$-团」，再同质化即得到无偏置的参数化归约。由于输入维度变成 $d=k+1$，任何 $\rho(d)\cdot N^{o(d)}$ 时间的算法都会推出 MULTICOLORED CLIQUE 的 $\rho(k)\cdot|V|^{o(k)}$ 算法，与 ETH 矛盾——这就是所有结论共享的 ETH 下界来源。同质化后「满射性」也顺带搞定：正齐次函数满射 $\iff$ 既有正点又有负点，而构造里 $f(0,1)<0$ 恒成立，故满射 $\iff$ 有正点，于是 2 层 ReLU 满射性同样 W[1]-hard（回答了 Froese 等人关于满射性 FPT 与否的问题，答案是否定的）。

**3. ReLU 网络与 zonotope 的对偶：把网络正性的硬度搬到 zonotope 非包含**

正齐次凸 CPWL 函数 $\mathbb{R}^d\to\mathbb{R}$ 与多胞形 $\mathbb{R}^d$ 之间存在对偶：任一 $f$ 可写成 $f(x)=\max\{a_1^\top x,\dots,a_k^\top x\}$，其 Newton 多胞形为 $\mathrm{Newt}(f)=\mathrm{conv}(\{a_1,\dots,a_k\})$，$f$ 就是该多胞形的支撑函数；映射 $\varphi:f\mapsto\mathrm{Newt}(f)$ 是双射且把函数加法/取 max 对应到 Minkowski 和/凸包并。对一个无偏置 2 层网络 $g(x)=\sum_{i\in I_+}\max\{0,w_i^\top x\}-\sum_{i\in I_-}\max\{0,w_i^\top x\}$，把正、负两部分各自对偶成 zonotope $Z_+=\sum_{i\in I_+}\mathrm{conv}(\{0,w_i\})$ 和 $Z_-=\sum_{i\in I_-}\mathrm{conv}(\{0,w_i\})$。则 $Z_+\subseteq Z_-$ 当且仅当 $g\le 0$ 处处成立；反之若存在 $y_*\in Z_+\setminus Z_-$，就有分离超平面使 $g(x)>0$。因此 **2 层 ReLU 正性 $\iff$ zonotope 非包含**。由设计 2 的硬度立得：ZONOTOPE NON-CONTAINMENT 关于 $d$ 是 W[1]-hard，且不存在 $\rho(d)\cdot N^{o(d)}$ 算法（ETH 下），说明枚举一个 zonotope 顶点的 $O(n^{d-1}\cdot\mathrm{poly}(N))$ 算法本质最优。这条对偶让计算几何里独立感兴趣的问题（机器人、控制中常见）也被一并钉死。

**4. Lipschitz 常数的硬度与正面结果：缩放技巧打通难度，ICNN 与子空间嵌入给出可解出口**

对 Lipschitz 常数，作者先证一个桥梁等式：对正齐次 CPWL 函数 $f$，$L_p(f)=\max_{\|x\|_p\le 1}|f(x)|$，即 Lipschitz 常数等于单位球上的最大绝对值。再把设计 2 同质化网络的所有 $y$ 系数缩放一个充分小的 $\varepsilon$ 得到正齐次的 $h$，使 $L_p(h)\approx L:=\max_{x}h(x,1)$——因为最优点 $x^*$ 的分量被缩得足够小，可用略小于 1 的 $y^*$ 把 $(x^*,1)$ 拉进单位球而几乎不损失值。于是计算 $L$ 的硬度转移到 $L_p(h)$：对所有 $p\in(0,\infty]$，**2 层 ReLU $L_p$-Lipschitz 常数都是 NP-hard、W[1]-hard**（把 Virmaux & Scaman 仅对 $p=2$ 的 NP-hardness 推广到全 $p$）；给 3 层网络加一个单 ReLU 神经元，则**任意倍数逼近 3 层 $L_p$-Lipschitz 常数**也 W[1]-hard。作为正面出口，对**输入凸网络（ICNN）**，由于 $f(x)=\max_i\{a_i^\top x+b_i\}$ 是凸函数、其 $L_p$-Lipschitz 常数等于各线性区域梯度对偶范数的最大值 $\max_i\|a_i\|_q$，借助 Hertrich & Loho 给出的 $\mathrm{Newt}(f)$ 小型扩展表示，可证 $L_1(f)$ 多项式时间可算、$L_\infty(f)$ 在 $O(2^d\,\mathrm{poly}(N))$ 时间内可算（关于 $d$ 是 FPT，把 $\max_{y\in Q}\|\pi(y)\|$ 化成 $2d$ 或 $2^d$ 个 LP）。最后基于子空间嵌入给出 zonotope 范数最大化的随机 FPT-近似：用 $\ell_1$ 子空间嵌入做 zonotope 阶约简，把 $n$ 个生成元压到 $r\in O(d\log d\,\varepsilon^{-2})$ 个并满足 $(1+\varepsilon)^{-1}Z(A')\subseteq Z(A)\subseteq(1+\varepsilon)Z(A')$，再枚举约简后 zonotope 的 $O(r^{d-1})$ 个顶点取范数最大值，得到任意范数的 $(1\pm\varepsilon)$-近似。

### 损失函数 / 训练策略

不适用——本文为纯理论论文，无训练目标或优化过程，全部为复杂性归约与可解性证明。

## 实验关键数据

本文无数值实验，「关键数据」是其建立的复杂性结论与运行时间界。下表汇总各问题的硬度结果与对应的最优算法。

### 主结果：硬度与紧匹配的算法界

| 问题（参数 $d$=输入维度） | 本文结论 | 暴力算法运行时间 | ETH 下界 |
|--------|------|------|------|
| 2 层 ReLU 正性（=满射） | W[1]-hard（即便零偏置） | $O(n^{d}\cdot\mathrm{poly}(N))$ | $\rho(d)N^{o(d)}$ 不可解 |
| 逼近 2 层 ReLU 最大值（任意倍数） | W[1]-hard | $O(n^{d}\cdot\mathrm{poly}(N))$ | 同上 |
| 判定 3 层 ReLU 是否恒零 | W[1]-hard | $O(n^{2d}\cdot\mathrm{poly}(N))$ | 同上 |
| ZONOTOPE 非包含 | W[1]-hard | $O(n^{d-1}\cdot\mathrm{poly}(N))$ | 同上 |
| 2 层 $L_p$-Lipschitz，$p\in(0,\infty]$ | NP-hard + W[1]-hard | $O(n^{d}\cdot\mathrm{poly}(N))$ | 同上 |
| 逼近 3 层 $L_p$-Lipschitz，$p\in[0,\infty]$ | W[1]-hard | $O(n^{2d}\cdot\mathrm{poly}(N))$ | 同上 |

核心信息：所有这些问题的暴力枚举算法在「对 $d$ 的指数依赖」上都已基本最优，无法做到 FPT（除非 W[1]=FPT）。

### 正面（可解）结果

| 受限设定 | 结论 | 复杂度 |
|------|------|------|
| ICNN 的 $L_1$-Lipschitz 常数 | 多项式时间可算 | $\mathrm{poly}(N)$ |
| ICNN 的 $L_\infty$-Lipschitz 常数 | FPT（关于 $d$） | $O(2^d\,\mathrm{poly}(N))$ |
| 任意范数在 zonotope 上的最大化 | 随机 FPT-$(1\pm\varepsilon)$ 近似 | $O((cd\log d/\varepsilon^2)^{d-1}T+\mathrm{poly}(n))$ |

### 关键发现
- **维度线性受控是整条证明的命门**：要的不是「随便造个难实例」，而是让输入维度 $d$ 严格随团大小 $k$ 线性增长，否则只能得到 NP-hardness 而非 W[1]-hardness。spike+penalty 的「一种颜色一个坐标」编码正是为此设计。
- **同质化是硬度传导的总开关**：去掉偏置看似是技术细节，实则是后续 zonotope 对偶、Lipschitz 缩放论证都依赖正齐次性的前提；没有它，硬度无法扩散到整个问题族。
- **难度对「稀疏小权重」也免疫**：构造里每个 ReLU 神经元只用常数个、且多项式有界的非零权重，说明「假设网络权重稀疏且小」并不能绕开这些硬度——这对实践中「靠结构假设求可解」的思路是个警示。
- **2 层 / 3 层是清晰分水岭**：判定网络是否恒零，2 层多项式可解、3 层 W[1]-hard，复杂性在层数上发生质变。

## 亮点与洞察
- **把团编码进低维网络的 spike/penalty 构造很巧**：penalty 负责「从每种颜色选一个点」、spike 负责「检验所选点两两相邻」，最大值 $k+\binom{k}{2}$ 恰好对应「$k$ 个点 + $\binom{k}{2}$ 条边」，几何直觉与组合目标严丝合缝。
- **同质化把「带偏置」难题免费转成「正齐次」难题**，这个去偏置技巧可复用到任何需要把仿射构造提升到齐次/对偶世界的复杂性论证中。
- **一条对偶把机器学习问题和计算几何问题钉在一起**：2 层 ReLU 正性 = zonotope 非包含，使得验证、控制、机器人三个社区关心的问题共享同一个 W[1]-hardness，结论的「辐射面」很广。
- **难/易边界给出了可操作的实践指引**：既然一般情形无法绕开硬度，那要么用 ICNN 这类结构受限网络（可证可解），要么在训练中引入「保证线性区域数量小/易枚举」的技巧——论文等于从理论侧论证了启发式与特殊架构的必要性。

## 局限与展望
- 作者明确声明这是纯理论的**最坏情况**结果：实际网络未必触发这些最坏实例，且指数中的常数或许还能压低（如做到 $O(n^{cd})$，$c<1$）。
- 可解性强依赖很强的结构假设（如 ICNN）；什么样的结构假设「既自然、真实数据训练出的网络又满足」仍不清楚。
- 多个公开问题悬而未决：$p\in(1,\infty)$ 时 zonotope 上 $L_p$-最大化是否 FPT（等价于仅正输出权重的 2 层 $L_p$-Lipschitz）、2 层 ReLU 计算 $L_0$ 的复杂性、以及 2 层 ReLU 正性是否属于 W[1]（若是则参数化复杂性可完全定型）。
- 子空间嵌入给出的随机 FPT-近似最坏运行时间偏大，能否去随机化做出确定性 FPT-近似仍是开放方向。

## 相关工作与启发
- **vs Froese et al. (COLT '25 / 2025b)**：他们建立了 2 层 ReLU 正性的 NP-hardness、单射性的 FPT 算法，并把正性/Lipschitz 的参数化复杂性留作公开问题；本文正面回答——这些问题是 W[1]-hard，不可能像单射性那样 FPT，从而把这一族问题的参数化复杂性几乎完全定型。
- **vs Virmaux & Scaman (2018) / Jordan & Dimakis (2020)**：前者证 2 层 $L_2$-Lipschitz 的 NP-hardness、后者证 3 层 Lipschitz 逼近的 NP-hardness；本文把硬度从单一 $p$ 推广到全 $p\in(0,\infty]$ 并加强到 W[1]-hard，给出更强的参数化下界。
- **vs Baburin & Pyatkin (2007) / Shenmaier (2018)**：他们研究 zonotope 上范数最大化（最长向量和）的多项式可解/FPT 与不可逼近性；本文的 Proposition 6.3 推广了其 $L_\infty$ 多项式、$L_1$ FPT 结论，并用 Theorem 5.1 指出一般 zonotope 范数最大化关于 $d$ 是 W[1]-hard。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 解决 COLT '25 公开问题，给出该问题族迄今最强硬度结果。
- 实验充分度: ⭐⭐⭐⭐⭐ 纯理论论文，归约链与算法界完整、相互印证，无需数值实验。
- 写作质量: ⭐⭐⭐⭐ 结构清晰、归约脉络明确，但密度高、需复杂性背景方能读透。
- 价值: ⭐⭐⭐⭐⭐ 同时影响 ML 验证、计算几何、控制/机器人三个社区，并为「为何必须用启发式/特殊架构」提供理论依据。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Training-Free Determination of Network Width via Neural Tangent Kernel](training-free_determination_of_network_width_via_neural_tangent_kernel.md)
- [\[ICLR 2026\] Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks](tractability_via_low_dimensionality_the_parameterized_complexity_of_training_qua.md)
- [\[ICLR 2026\] DeepWeightFlow: Re-Basined Flow Matching for Generating Neural Network Weights](deepweightflow_re-basined_flow_matching_for_generating_neural_network_weights.md)
- [\[ICLR 2026\] Subquadratic Algorithms and Hardness for Attention with Any Temperature](subquadratic_algorithms_and_hardness_for_attention_with_any_temperature.md)
- [\[ICLR 2026\] Toward Practical Equilibrium Propagation: Brain-Inspired Recurrent Neural Network with Feedback Regulation and Residual Connections](toward_practical_equilibrium_propagation_brain-inspired_recurrent_neural_network.md)

</div>

<!-- RELATED:END -->
