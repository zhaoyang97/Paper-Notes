---
title: >-
  [论文解读] Cautious Weight Decay
description: >-
  [ICLR2026][优化/理论][权重衰减] 本文提出 Cautious Weight Decay（CWD），一行代码、与优化器无关的改动：只在「优化器更新方向」与「参数符号」一致的坐标上施加权重衰减，从而保留原始损失目标（不再隐式优化一个被正则化/约束的代理目标），并在到达驻点流形后产生滑模动力学、趋向局部 Pareto 最优的小范数解；在 ADAMW / LION / MUON 上不加新超参即可一致降低语言模型预训练和 ImageNet 的最终 loss 与提升精度。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "权重衰减"
  - "优化器"
  - "隐式正则化"
  - "滑模动力学"
  - "Lyapunov 分析"
---

# Cautious Weight Decay

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=Gwe6gbGng5](https://openreview.net/forum?id=Gwe6gbGng5)  
**代码**: 待确认  
**领域**: optimization  
**关键词**: 权重衰减, 优化器, 隐式正则化, 滑模动力学, Lyapunov 分析

## 一句话总结
本文提出 Cautious Weight Decay（CWD），一行代码、与优化器无关的改动：只在「优化器更新方向」与「参数符号」一致的坐标上施加权重衰减，从而保留原始损失目标（不再隐式优化一个被正则化/约束的代理目标），并在到达驻点流形后产生滑模动力学、趋向局部 Pareto 最优的小范数解；在 ADAMW / LION / MUON 上不加新超参即可一致降低语言模型预训练和 ImageNet 的最终 loss 与提升精度。

## 研究背景与动机
**领域现状**：现代大模型训练几乎都用 decoupled weight decay（解耦权重衰减，Loshchilov & Hutter 2019）。它把衰减项直接作用在参数上，更新规则写成 $x_{t+1} = (1-\eta_t\lambda)x_t - \eta_t u_t$，其中 $u_t$ 是优化器构造的（常被符号归一化的）更新向量。ADAMW、LION、MUON 等 SOTA 优化器都建立在这套机制上，它能稳定训练、改善泛化。

**现有痛点**：解耦权重衰减对「更新方向 $u_t$ 与参数 $x_t$ 是否同向」完全不敏感。当某个坐标上 $u_t$ 和 $x_t$ 同号时，衰减把参数往零拉、起正则作用是好的；但当二者异号时，优化器本来想把这个参数推向最优，衰减却反方向把它往零拽，**主动抵消了有益的更新**。

**核心矛盾**：更深一层的问题是——解耦权重衰减其实在**隐式地改写目标函数**。论文复述了已有结论：SGD 加解耦衰减等价于在 $\ell_2$ 正则目标 $f(x)+\tfrac{\lambda}{2}\|x\|_2^2$ 上做 SGD；LION-K 收敛到正则目标 $f(x)+\tfrac1\lambda K^*(\lambda x)$ 的驻点，LION / MUON 对应到 $\|x\|_\infty \le 1/\lambda$、$\|X\|_{op}\le 1/\lambda$ 的**约束优化**；ADAMW 也近似在解一个 box-约束问题。也就是说，你以为在最小化 $f$，实际上最小化的是一个**依赖 $\lambda$ 的代理目标**，最优解被 $\lambda$ 拉偏了。

**本文目标**：能否保留权重衰减的好处（正则、训练加速、更小的参数范数），同时让优化器真正去最小化**原始** $f$，而不是被 $\lambda$ 扭曲的代理目标？

**切入角度**：作者注意到「衰减有害」只发生在 $u_t$ 与 $x_t$ 异号的坐标上。那就**只在同号坐标施加衰减**——这恰好把「会抵消有益更新」的那部分关掉，保留「起正则作用」的那部分。

**核心 idea**：用一个逐坐标的符号门控 $\mathbb{I}(u_t \odot x_t \ge 0)$ 去乘权重衰减项——同号才衰减、异号就跳过，一行代码搞定，不引入任何新超参。

## 方法详解

### 整体框架
CWD 把标准解耦权重衰减的更新规则改成：

$$x_{t+1} = x_t - \eta_t\big(u_t + \lambda\,\mathbb{I}(u_t \odot x_t \ge 0)\odot x_t\big),$$

其中 $\odot$ 是逐元素乘法、$\mathbb{I}(\cdot)$ 是逐坐标的示性函数（同号取 1、异号取 0）。对比标准式 $x_{t+1}=x_t-\eta_t(u_t+\lambda x_t)$，唯一区别就是给衰减项 $\lambda x_t$ 套了一个符号门 $\mathbb{I}(u_t x_t \ge 0)$。$u_t$ 可以是任意优化器的更新向量（ADAMW 的 $D_t^{-1}\hat m_t$、LION-K 的 $-\nabla K(\tilde m_t)$ 等），所以 CWD 是**优化器无关**的 drop-in 改动。

它带来两层质变：(1) **无偏优化**——只要基础优化器（不加衰减时）会收敛，CWD 的每个聚点 $x^\star$ 都满足 $\nabla f(x^\star)=0$，即收敛到**原始损失**的驻点而非正则代理的解；(2) 到达驻点流形后产生**滑模动力学**，沿流形滑行、尽量缩小参数范数，最终停在局部 Pareto 最优点。下面分别讲清这个「为什么无偏」和「滑模怎么来的」。

### 关键设计

**1. 符号选择性门控：只在同号坐标衰减**

这是 CWD 的全部机制。痛点在于标准衰减项 $\lambda x_t$ 不分青红皂白地把每个参数往零拉，在 $u_t$ 与 $x_t$ 异号的坐标上会对抗优化器的有益更新。CWD 给衰减加一个逐坐标开关：当 $u_t$ 与 $x_t$ 同号（$u_t x_t \ge 0$，衰减与更新方向一致、起正则作用）时施加衰减；当异号（衰减会顶着优化器走）时把衰减置零、自动停用。

之所以有效，是因为这个门控让衰减永远「只帮忙不添乱」——它只在不与主目标冲突时起作用。注意符号约定：对 ADAMW/SGD 写成 $\mathbb{I}(u_t x_t \ge 0)$，而对 LION-K 因为 $u_t=\nabla K(m_t)$ 的方向约定不同，门控写成 $\mathbb{I}(m_t x_t \le 0)$，本质是同一件事（衰减方向与更新方向一致才施加）。整个改动就是 Algorithm 1 里多写的那个示性函数，没有新超参、不用重新调参。

**2. 无偏性的 Lyapunov 证明：CWD 不改写损失地形**

要论证「CWD 真的在最小化原始 $f$ 而不是某个代理」，作者用 Lyapunov 函数分析连续时间动力学。以 SGD+CWD 为例，其 ODE 为 $\dot x_t = -\nabla f(x_t) - \lambda\,\mathbb{I}(\nabla f(x_t)x_t\ge 0)x_t$，取 $H(x)=f(x)$ 作 Lyapunov 函数，则

$$\frac{dH}{dt} = -\|\nabla f(x_t)\|_2^2 - \lambda\big\|(\nabla f(x_t)x_t)_+\big\|_1 \le 0,$$

其中 $(\cdot)_+=\max(0,\cdot)$。关键在于第二项是**非正**的：门控保证只有同号坐标贡献衰减、且该贡献恒非负，于是被减掉。由 LaSalle 不变性原理，轨迹的聚点落在 $\{x\mid \nabla f(x)=0\}$——也就是**原始损失**的驻点集。这正是与标准衰减的根本区别：标准衰减让 $H=f$ 不再单调下降（衰减项会反向贡献），所以它收敛到的是正则代理的解；而 CWD 因为「冲突即停用」，使 $f$ 始终是合法的 Lyapunov 函数，损失地形保持无偏。作者把这套论证推广到 SGDM、LION-K、ADAM（Table 1 给出各自的 Lyapunov 函数，如 SGDM+CWD 用 $H(x,m)=\beta f(x)+\tfrac12\|m\|^2+\lambda\|(mx)_+\|_1$），动量方法则收敛到 $\{(x,m)\mid \nabla f(x)=0, m=0\}$。

**3. 滑模动力学：衰减退化为沿流形的「降范数」副目标**

无偏只说明 CWD 不偏离驻点流形，但它和「干脆不加衰减」有什么不同？区别在进入流形之后。不加衰减时，动量 $m$ 衰减到零、动力学就停在流形上某个普通点；CWD 则不然——进入驻点流形 $M=\{x\mid\nabla f(x)=0\}$ 后，残余动力学变成

$$\dot x_t = -\lambda\, s_t \odot x_t,\quad s_t\in[0,1]^d,$$

其中 $s_t$ 是 Filippov 意义下示性函数在切换面 $\{[\nabla f]_i=0\}$ 上取的选择子（同号取 1、异号取 0、边界取 $[0,1]$）。直观说，衰减项不再影响损失（已在驻点），转而**沿流形滑行、逐坐标地把参数往零缩**，直到无法在所有坐标上同时再减小为止。这就是滑模（sliding mode）：轨迹被约束在流形上、却仍持续运动。终点是流形的**局部 Pareto 前沿** $P$——以「每个坐标都更小」为偏序时无法再被支配的点。换言之，在所有等价的零梯度解里，CWD 偏好范数更小的那个。Figure 2/3 的 toy 例子直观展示：ADAM 停在流形上随机一点，ADAMW 收敛到 box 约束 $\max\{x,y\}\le 1/\lambda$ 的解（被约束拉偏），而 ADAM+CWD 滑到 Pareto 前沿。作者还在 Appendix E 给出离散时间 ADAM+CWD 在光滑非凸下的收敛率。

### 损失函数 / 训练策略
CWD 不改训练目标，仅改优化器的参数更新一行。实验里对 baseline（ADAMW/LION/MUON）网格搜索 batch size、学习率、weight decay $\lambda$、warmup 比例等超参，然后 **CWD 直接复用 baseline 已调好的设置、不再调参**——这是它「即插即换」卖点的关键验证方式。

## 实验关键数据

### 主实验
语言模型用类 Gemma 的 Transformer（338M / 986M / 2B），按 Chinchilla 计算最优（20 tokens/参数）在 C4 上训练；另在 OLMo codebase 上做 OLMo-1B、100B tokens（100 TPP）的 over-training。全部实验约 20,000 H100 GPU 小时。CWD 在所有规模、所有优化器上一致降低最终验证 loss、提升下游精度。

ImageNet（300 epochs，标准增强）验证集 Top-1 精度（%）：

| 模型 | 优化器 | Base | +CWD |
|------|--------|------|------|
| ViT-S/16 (22M) | ADAMW | 78.84 | 79.45 |
| ViT-S/16 | LION | 79.29 | 79.82 |
| ViT-S/16 | MUON | 79.35 | 79.91 |
| ResNet-50 (25.6M) | ADAMW | 76.30 | 76.68 |
| ViT-B/16 (86.6M) | ADAMW | 80.15 | 80.71 |
| ViT-B/16 | MUON | 80.83 | 81.04 |

OLMo-1B（100B tokens）下游 zero-shot 精度（节选）：ADAMW 在 ARC-Easy 0.50→0.53、PIQA 0.67→0.69、MMLU 0.23→0.25；MUON 在 PIQA 0.68→0.71、ComQA 0.30→0.33——CWD 普遍 +1~3 个百分点。

### 消融实验
OLMo-1B（100B tokens）验证 loss，对比不同的「选择性衰减」掩码策略（越低越好）：

| 优化器 | Baseline（标准衰减，调过 λ） | Ours（更新掩码 $\mathbb{I}(ux\ge0)$） | Random（同稀疏率随机掩码） | Gradient（$\mathbb{I}(gx\ge0)$） | No WD（λ=0） |
|--------|------|------|--------|----------|-------|
| ADAMW | 2.65 | **2.56** | 2.82 | 2.75 | 2.70 |
| MUON | 2.51 | **2.42** | 2.73 | 2.74 | 2.62 |

### 关键发现
- **不是「少衰减」的功劳，而是「衰减得有结构」**：用同等稀疏率的随机 Bernoulli 掩码替换 CWD 掩码，loss 反而大幅恶化（ADAMW 2.56→2.82、MUON 2.42→2.73），说明单纯降低衰减频率没用，关键是**按符号选择**。
- **用更新方向 $u$ 选择 > 用梯度方向 $g$ 选择**：把门控里的 $u$ 换成原始梯度 $g$（$\mathbb{I}(gx\ge0)$）也会变差，说明该和**优化器实际更新方向**对齐，而非裸梯度。
- **正则本身有用，CWD 只是用得更好**：$\lambda=0$（完全不衰减）依然劣于调过的衰减，CWD 是「更选择性地用正则」而非「关掉正则」。训练动态上，CWD 全程 loss 更低、收尾参数范数居中（$\lambda=0$ 范数最大、收敛更早停滞；标准 ADAMW 范数最小）。
- **随规模不退化**：111M→2B，CWD 相对 ADAMW 的 loss 优势保持稳定甚至略微扩大；同时 CWD 给出更低的 RMS 归一化梯度范数。
- **指令微调对比 SPD**：在 Alpaca GPT-4 上微调 TinyLlama / Mistral-7B，逐元素 CWD 在 MMLU/AGIEval/WinoGrande 多数指标上匹配或超过 SPD 与「内积版」CWD。

## 亮点与洞察
- **一行代码改写了「优化器到底在优化什么」**：把 $\lambda x_t$ 换成 $\lambda\mathbb{I}(ux\ge0)x_t$，看似微小，却把隐式正则代理目标拉回到了原始损失——这是「形式极简、语义深刻」的典范。
- **把权重衰减重新理解为「副目标」而非「主目标的修改」**：标准衰减改写了你要最小化的函数；CWD 让衰减退居二线，只在不影响主目标时做「降范数」这件副业，并用 Pareto 最优的语言精确刻画终点。这个视角可迁移到其他正则项——凡是「方向可能与主目标冲突」的正则，都可以考虑加符号门控变成无冲突的副目标。
- **理论与工程罕见地对齐**：Lyapunov + LaSalle + Filippov 滑模这套连续时间分析直接解释了 toy 例子里三条轨迹的差异，且预言的「更小范数 / 更低梯度范数」在大模型实验里被观测到。
- **零调参迁移**：直接复用 baseline 的 $\lambda$ 就能涨点，且最优 $\lambda^\star$ 基本不变，落地成本几乎为零。

## 局限与展望
- 滑模终点 $P$ 一般不是单点，最终落在哪个 Pareto 点**依赖初始化和离散化方案**，理论上不唯一、不可控；论文承认精确极限点「intricately 依赖」这些因素。
- ADAMW 的无偏性分析缺一个严格的 Lyapunov 函数（原文坦言 ADAMW 因此「无法建立收敛」），离散时间收敛率只对 ADAM+CWD 在额外假设下给出。
- 增益幅度偏温和（loss 降几个百分点、精度 +0.3~1%），是「免费午餐」而非颠覆性提升；是否在更大规模（>2B）、更长训练、或非 Transformer 架构上同样稳定，仍需更多验证。
- 「为什么更小参数范数对应更好泛化」本文用实验关联（更低梯度范数 + 更低 loss）佐证，但没给出泛化理论上的因果解释。

## 相关工作与启发
- **vs 标准 decoupled weight decay（ADAMW / LION-K / MUON）**：它们隐式优化依赖 $\lambda$ 的正则/约束代理目标，最优解被 $\lambda$ 拉偏；CWD 通过符号门控保持原始损失无偏，同等 $\lambda$ 下 loss 更低、最优 $\lambda^\star$ 不变。
- **vs SPD（Tian et al. 2024）**：同属「选择性/结构化衰减」思路，但 CWD 给出了 Lyapunov/滑模的理论刻画，并在指令微调上以逐元素版匹配或超过 SPD。
- **vs 完全去掉衰减（λ=0）**：消融证明 λ=0 中期下降快但早停滞、终点 loss 更高、范数最大；CWD 不是关掉正则而是更选择性地用正则。
- **vs 随机/梯度掩码**：随机 Bernoulli 掩码或用裸梯度 $\mathbb{I}(gx\ge0)$ 选择都明显更差，凸显「按优化器更新方向逐坐标选择」这一具体结构才是收益来源。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 一行代码的极简改动，却用滑模/Pareto/Lyapunov 给出全新且自洽的理论解释。
- 实验充分度: ⭐⭐⭐⭐ LLM 预训练 + ImageNet + 指令微调三类任务、111M~7B 多尺度、多优化器、掩码消融到位；增益幅度温和、>2B 验证有限。
- 写作质量: ⭐⭐⭐⭐⭐ 动机→机制→理论→实验逻辑清晰，toy 例子与大模型结果互相印证。
- 价值: ⭐⭐⭐⭐⭐ 零调参 drop-in、优化器无关、可直接落地到现有训练栈，对大模型训练有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_µp_for_learning_rate_transfer_in_practice.md)
- [\[ICLR 2026\] Cautious Optimizers: Improving Training with One Line of Code](cautious_optimizers_improving_training_with_one_line_of_code.md)
- [\[ICLR 2026\] Incorporating Expert Priors into Bayesian Optimization via Dynamic Mean Decay](incorporating_expert_priors_into_bayesian_optimization_via_dynamic_mean_decay.md)
- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](../../ICML2026/optimization/limits_of_convergence-rate_control_for_open-weight_safety.md)

</div>

<!-- RELATED:END -->
