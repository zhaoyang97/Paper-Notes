---
title: >-
  [论文解读] Flow Matching with Semidiscrete Couplings
description: >-
  [ICLR 2026][图像生成][Flow Matching] 把 OT 引导的流匹配从"每个 batch 现算 n×n 最优传输"换成"一次性拟合一个 N 维对偶势向量、训练时用一次最大内积搜索把噪声分配给数据点"，在去掉 OT-FM 对 batch 大小 n 的二次依赖的同时，跨多个数据集、有条件/无条件、乃至 mean-flow 单步生成上全面超过 FM 和 OT-FM。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "Flow Matching"
  - "Optimal Transport"
  - "Semidiscrete OT"
  - "Generative Models"
  - "MIPS"
---

# Flow Matching with Semidiscrete Couplings

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4EGjzT6w80](https://openreview.net/forum?id=4EGjzT6w80)  
**代码**: [OTT-JAX (ott-jax/ott)](https://github.com/ott-jax/ott)  
**领域**: 图像生成 / 流匹配 / 最优传输  
**关键词**: Flow Matching, Optimal Transport, Semidiscrete OT, Generative Models, MIPS  

## 一句话总结
把 OT 引导的流匹配从"每个 batch 现算 n×n 最优传输"换成"一次性拟合一个 N 维对偶势向量、训练时用一次最大内积搜索把噪声分配给数据点"，在去掉 OT-FM 对 batch 大小 n 的二次依赖的同时，跨多个数据集、有条件/无条件、乃至 mean-flow 单步生成上全面超过 FM 和 OT-FM。

## 研究背景与动机

**领域现状**：流匹配（FM）通过采样噪声/数据对 $(x_0, x_1)$ 并让速度场 $v_\theta(t, x_t)$ 对齐方向 $x_1 - x_0$ 来训练生成模型，其中 $x_t = (1-t)x_0 + t x_1$。FM 里一个关键自由度是"噪声×数据"的耦合方式：默认的独立耦合 $\pi_I = \mu_0 \otimes \mu_1$ 简单廉价，但产生的 ODE 路径曲率高，推理时需要大量函数求值（NFE）才能积分出好样本。

**现有痛点**：为了拉直流，OT-FM（Pooladian 2023、Tong 2024）提出用最优传输耦合替代独立耦合——每个 batch 采 $n$ 个噪声和 $n$ 个数据点，用 Hungarian/Sinkhorn 求最优匹配再喂给 FM 损失。但实践中 OT-FM 收益有限。Zhang et al. (2025) 指出根因是 $n$ 太小：小样本上的最优匹配本身不稳定，无法逼近大 $n$ 才出现的真实匹配（维度灾难）。他们的解法是把 $n$ 从 256 拉到约 $10^6$ 并用多 GPU Sinkhorn，但这又带来 $O(n^2/\varepsilon^2)$ 的预计算开销——而且越好的结果越要大 $n$、小 $\varepsilon$，等于"越想用就越贵"。

**核心矛盾**：OT-FM 的理论承诺（拉直流 → 低 NFE 高质量）只有在巨大 batch 上才兑现，但巨大 batch 的 Sinkhorn 成本随 $n^2/\varepsilon^2$ 爆炸，使得这条路线在工程上无法普及。本质问题在于它一直把 OT 当成"反复匹配 $n$ 个 i.i.d. 噪声与数据样本"的离散问题。

**本文目标**：在不依赖 batch 大小 $n$ 的前提下兑现 OT-FM 的好处。

**核心 idea（半离散视角）**：注意到训练时数据集本身是有限的 $N$ 个点，于是把 OT 问题写成"连续噪声 → 离散数据"的**半离散最优传输（SD-OT）**。它只需用 SGD 拟合一个大小为 $N$ 的对偶势向量 $g$（一次性预计算）；训练时每个新采样的噪声只需对数据集做一次成本 $O(N)$ 的**最大内积搜索（MIPS）**即可分配到数据点。这样就彻底去掉了 OT-FM 对 $n/\varepsilon$ 的二次依赖。

## 方法详解

### 整体框架
SD-FM 分两步：**预计算阶段**在噪声分布 $\mu_0$ 和有限数据 $\mu_1 = \sum_j b_j \delta_{y_j}$ 之间求解 SD-OT，用 SGD 拟合并存下对偶势 $g^\star \in \mathbb{R}^N$；**FM 训练阶段**对每个新采样噪声 $x_0$，按 $k \sim s_{\varepsilon, g^\star}(x_0)$ 把它配到某个数据点 $x_1^{(k)}$，其余 FM 流程不变。

```mermaid
flowchart LR
    subgraph 预计算["预计算（一次性，SGD K 步）"]
        A[噪声 μ0 + 数据 μ1] --> B["求解半离散 OT<br/>拟合对偶势 g* ∈ R^N"]
        B --> C["χ² 收敛准则<br/>监控 m(g)≈b"]
    end
    subgraph 训练["FM 训练循环"]
        D[采样噪声 x0~N] --> E["MIPS: k*=argmax g*_k + ⟨x0, x1^k⟩"]
        E --> F[配对 (x0, x1^k*)]
        F --> G["FM 回归损失<br/>‖x1-x0 - vθ(t,xt)‖²"]
    end
    C -.存储 g*.-> E
```

### 关键设计

**1. 半离散对偶 + SGD 拟合势向量：把 batch-OT 换成一次性预计算**。SD-OT 利用目标测度有限这一事实，把熵正则 OT 写成完全由 $N$ 维势向量 $g$ 参数化的凹的半对偶问题 $\max_{g} F_\varepsilon(g) = \mathbb{E}_{X\sim\mu}[f_{g,\varepsilon}(X)] + \langle b, g\rangle$，其中软 c-变换在 $\varepsilon=0$ 时退化为 $f_{g,\varepsilon}(x) = -\max_j [g_j - c(x, y_j)]$。沿用 Genevay et al. (2016) 的随机优化，用平均 SGD 求解 $g^\star$。这一步把"每个 batch 都要现算 $n\times n$ 匹配"的反复开销，压缩成训练前**只做一次**、且只依赖数据集本身的固定预计算（在 8×H100 上约 12 小时即可收敛），复杂度对比见表 1：OT-FM 的训练每对要额外付 $O(dn/\varepsilon^2)$，而 SD-FM 每对只付 $dN$ 的查表代价，且 $\Theta$（FM 损失梯度本身）主导一切。

**2. 无偏 $\chi^2$ 收敛准则：让大规模 SD-OT 可监控**。已有 SD-OT 文献没有收敛判据——$g$ 是最优解当且仅当第二边缘 $m(g) = b$。直接用 TV 距离 $\frac{1}{2}\|m(g)-b\|_1$ 监控会因 $m(g)$ 里的期望被范数包住而产生偏差，要消偏需要随 $N$ 线性增长的样本量，代价过高。本文改用 $\chi^2$ 散度 $\chi^2(p\|q) = \sum_j (p_j/q_j)^2 q_j - 1$，证明它可写成对 $x, x'$ 的双重积分形式（Fact 1），从而得到一个 $O(NB)$ 时间、用一个 batch 就能算的**无偏估计**：$\hat\chi^2 = \frac{1}{B(B-1)}\sum_j \frac{1}{b_j}\big[(\sum_i [s_{\varepsilon,g}(x_i)]_j)^2 - \sum_i [s_{\varepsilon,g}(x_i)]_j^2\big] - 1$。实验（图 2、图 3）证实 $\hat\chi^2$ 越小，下游 FID 越低、曲率越小，且收益在 $\hat\chi^2 \approx 0.05$ 后饱和，给了"该预计算到多准"一个明确停表点。

**3. 覆盖 $\varepsilon=0$ 的收敛分析：支撑全程使用无正则**。Genevay et al. (2016) 只分析了 $\varepsilon>0$，但 SD-FM 里最有用的恰是 $\varepsilon=0$（此时配对退化为精确 MIPS，可用快速检索，且 $\varepsilon>0$ 的 $N$ 维 softmax 类别采样太贵）。本文在 $\varepsilon=0$ 时引入额外正则假设（点距 $\delta = \min_{j\ne j'}\|y_j - y_{j'}\| > 0$、$\mu$ 有密度、表面积有界），定义 $L_0 = C_\mu^{\max}/\delta$，证明定理 2：对任意 $\varepsilon\ge 0$，SGD 迭代满足 $\mathbb{E}[\chi^2(m(g_t)\|b)] \lesssim \frac{1}{\min_j b_j}\sqrt{L_\varepsilon \Delta / K}$，且熵传输成本以 $O((1/K)^{1/4})$ 逼近最优。这统一了 $\varepsilon=0$ 与 $\varepsilon>0$ 两种情形，从理论上支持了全文一律用 $\varepsilon=0$ 的选择。

**4. 广义 Tweedie 公式：让 SD 耦合也能做 score 估计与 guidance**。独立耦合下，流匹配速度场和分数有 Tweedie 关系 $\nabla\log\rho_t(x) = \frac{t v_t(x) - x}{1-t}$，但一般耦合下该关系依赖 $X_0, X_1$ 独立而失效。命题 3 给出广义版本 $\nabla\log\rho_t(x) = \frac{t v_t(x) - x + (1-t)\delta_\varepsilon}{(1-t)^2}$，其中校正项 $\delta_\varepsilon$ 在 $\varepsilon\to\infty$（退化为独立耦合）**和** $\varepsilon\to 0$ 时都消失（$\|\delta_\varepsilon\| \lesssim e^{-1/\varepsilon}/\varepsilon$，直觉是 $\varepsilon\to0$ 时 $X_1$ 几乎被 $X_0$ 决定）。这意味着用 $\varepsilon=0$ 的 SD-FM 模型仍可直接从速度场恢复分数，进而支持 CFG/autoguidance 这类需要 $\nabla\log\rho_t$ 的修正采样（命题 4 给出基于权重重采样的修正器）。

## 实验关键数据

### 主实验：ImageNet-64 / PetFace 无条件 + 类条件生成（FID ↓）

| 数据集 | 方法 | Euler 4 | Euler 8 | Euler 16 | Dopri5 | 类条件 Euler 4 | 类条件 Dopri5 |
|---|---|---|---|---|---|---|---|
| ImageNet-64 | I-FM | — | 79.95 | 37.90 | 9.10 | 34.51 | 3.91 |
| ImageNet-64 | SD-FM (PCA500, ε=0) | **45.62** | **23.75** | **15.02** | **8.42** | 26.04 | **3.63** |
| PetFace | I-FM | — | 56.53 | 26.85 | 1.26 | 47.66 | 1.09 |
| PetFace | SD-FM (full d=12k, ε=0) | **20.54** | **12.77** | **7.50** | 1.26 | **19.10** | **1.05** |

低 NFE（Euler 4/8）下提升最显著——这正是 OT 拉直流要兑现的"省推理预算"承诺。

### 消融与代价对比

| 维度 | 发现 |
|---|---|
| 势向量质量 → FID（图 3） | $\hat\chi^2$ 越小，FID 与曲率同步下降，$\hat\chi^2\approx0.05$ 后饱和 |
| 配对耗时（图 4，表 1） | SD-FM 每对配对开销相对 FM 梯度 $\Theta$ 可忽略；OT-FM 大 $n$ 在 64×64 上要 >10 天 |
| $\varepsilon \in \{0, 0.01, 0.1\}$ | 三者性能差异极小，推荐 $\varepsilon=0$（可用 MIPS、配对更快） |
| CelebA 超分（表 3，连续条件） | 4× SR PSNR 21.17→21.41、8× SR PSNR 17.52→17.94，SD-FM 均优于 I-FM |
| Mean-Flow 单步（ImgN-256 latent，图 6） | SD-MF 在低 NFE 下 FID 优于 I-MF，证明 SD 耦合收益超出标准 FM |
| Guidance（图 5） | 用广义 Tweedie 做修正，增大重采样数 $r$ 提升 precision（牺牲 recall） |

### 关键发现
- 把预计算花在"把势向量 $g$ 拟合得更准"上，能稳定换来更低的 FID 和更直的流，且有明确的饱和点。
- SD-FM 用可忽略的配对开销，在所有推理预算、多种数据集、有条件/无条件、甚至 mean-flow 上一致超过 I-FM，并比 OT-FM 便宜数个量级。
- $\varepsilon=0$ 在性能几乎不损的同时能用 MIPS 加速，是工程上的推荐配置。

## 亮点与洞察
- **换问题而非换算法**：不再去优化"如何更快地做大 batch Sinkhorn"，而是直接换成半离散表述，把"每 batch 现算"转成"一次性预计算 + 训练时查表"，从根上去掉对 $n$ 的二次依赖。
- **无偏 $\chi^2$ 准则很实用**：第一次给随机 SD-OT 一个能用一个 batch 无偏估计、且与下游 FID 强相关的收敛判据，让"预计算到多准就够"变得可度量、可停表。
- **$\delta_\varepsilon$ 在两端都消失**很优雅：$\varepsilon\to0$ 和 $\varepsilon\to\infty$ 都让 Tweedie 校正项归零，意味着最便宜（MIPS 可用）的 $\varepsilon=0$ 配置反而保留了 score 估计能力，guidance 不用额外代价。
- **复杂度表（表 1）讲清了价值**：$\Theta$ 主导一切的前提下，SD-FM 的边际配对代价 $dN$ 几乎免费，而 OT-FM 的 $dn/\varepsilon^2$ 是真金白银。

## 局限与展望
- **预计算随 $N$ 增长**：若数据集到十亿级 $N$，拟合 $g$ 仍会成为挑战（虽然仍远小于 FM 训练本身），可用 batching/momentum/$\varepsilon$-tempering 等手段缓解。
- **$\varepsilon>0$ 的类别采样贵**：$N$ 维 softmax 采样开销大，这也是作者一律推荐 $\varepsilon=0$ 的原因。
- **MIPS 目前为精确求解**：用近似 MIPS 进一步加速配对留作未来工作。
- **与更复杂方法的交互未探索**：SD 耦合与 Reflow 等更高级/正交方法（以及 dataloader 视角的"从数据点反查 Laguerre cell 内噪声"）的组合留待后续。

## 相关工作与启发
- **OT-FM 谱系**：Pooladian (2023)、Tong (2024) 提出 batch-OT 耦合；Davtyan (2025) 的 LOOM-CFM 缓存 Hungarian 配对但每 batch $O(n^3)$；Zhang et al. (2025) 用超大 $n$ + 多 GPU Sinkhorn 揭示"$n$ 越大越好"——本文正是对这条线的成本破局。
- **半离散 OT**：Oliker-Prussner、Mérigot (2011)、Cuturi-Peyré (2018)、An (2020, AE-OT) 的 SD-OT 理论，与 Genevay et al. (2016) 的随机对偶优化，是本文方法的直接基石。
- **流/扩散统一**：Albergo (2023) 的随机插值、Lipman (2023) 的 FM、以及 score-velocity 的 Tweedie 桥梁，是广义 Tweedie 公式的来源。
- **并发工作**：Kong et al. (2026) 同样用 SD-OT 改进流模型（聚焦无条件/类条件），本文的差异化贡献是无偏收敛准则、覆盖 $\varepsilon=0$ 的收敛分析、以及可用于 guidance 的广义 Tweedie。
- **启发**：当"把某个子问题做大做准"成本爆炸时，不妨检查能不能换一个等价但结构更友好的表述（这里是利用数据有限性把连续-连续 OT 降成连续-离散），往往能把"每步现算"摊销成"一次性预计算"。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把半离散 OT 引入流匹配并配上无偏收敛准则 + 广义 Tweedie，视角清晰、破局点扎实；扣分因 SD-OT 本身是已有工具，且有并发工作。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖无条件/类条件/连续条件超分/guidance/mean-flow 单步，多数据集多 solver，并直接对比配对耗时；像素空间高分辨率上 OT-FM 因太贵无法全面对照略有遗憾。
- **写作质量**: ⭐⭐⭐⭐ — 图 1 三种耦合对比 + 复杂度表 1 把动机和价值讲得很透，理论与实验衔接顺畅。
- **价值**: ⭐⭐⭐⭐ — 让 OT 引导的 FM 真正变得便宜可用，对追求低 NFE 高质量生成的工程实践有直接意义，且已并入 OTT-JAX 开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Delay Flow Matching](delay_flow_matching.md)
- [\[ICLR 2026\] Carré du champ Flow Matching: 用几何感知噪声改善生成模型的质量-泛化权衡](carré_du_champ_flow_matching_better_quality-generalisation_tradeoff_in_generativ.md)
- [\[ICLR 2026\] Edit-Based Flow Matching for Temporal Point Processes](edit-based_flow_matching_for_temporal_point_processes.md)
- [\[ICLR 2026\] MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference](meancache_from_instantaneous_to_average_velocity_for_accelerating_flow_matching_.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
