---
title: >-
  [论文解读] Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions
description: >-
  [ICML2026][学习理论][敏感度谱] 本文从**参数空间几何**的角度解释了 Transformer 为何学不会 Parity 这类"敏感"布尔函数：它证明随机初始化的 Transformer 几乎必然会计算出"含大量零敏感度字符串"的函数，而像 Parity、First 这种缺乏零敏感度字符串的函数所对应的参数只占整个参数空间的一个**勒贝格测度零**子集，随机初始化几乎必然错过，因而可证不可学。
tags:
  - "ICML2026"
  - "学习理论"
  - "Transformer 表达力与可学习性"
  - "形式语言"
  - "敏感度谱"
  - "参数空间几何"
  - "测度论"
  - "可学习性"
  - "Parity"
  - "layer norm"
---

# Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions

**会议**: ICML2026  
**arXiv**: [2606.08768](https://arxiv.org/abs/2606.08768)  
**代码**: 待确认  
**领域**: 学习理论 / Transformer 表达力与可学习性 / 形式语言  
**关键词**: 敏感度谱, 参数空间几何, 测度论, 可学习性, Parity, layer norm

## 一句话总结
本文从**参数空间几何**的角度解释了 Transformer 为何学不会 Parity 这类"敏感"布尔函数：它证明随机初始化的 Transformer 几乎必然会计算出"含大量零敏感度字符串"的函数，而像 Parity、First 这种缺乏零敏感度字符串的函数所对应的参数只占整个参数空间的一个**勒贝格测度零**子集，随机初始化几乎必然错过，因而可证不可学。

## 研究背景与动机

**领域现状**：理解 Transformer 能力边界的主流工具是**表达力（expressivity）**——存不存在某组参数能让 Transformer 计算某函数。已有大量工作刻画了 Transformer 在表达力上的上下界。

**现有痛点**：表达力不等于**可学习性（learnability）**。即使"正确的参数设置存在"，训练也未必能到达它。最经典的反例就是 Parity（判断输入中 1 的个数奇偶）：带 layer norm 的 Transformer 表达力上完全能表示它，实践中却学不会。更一般的经验/理论发现是：Transformer 学不会**敏感**函数——那些"翻转单个输入位很可能改变输出"的函数。

**核心矛盾**：人们已知 Transformer 偏好**低平均敏感度**的函数，但"平均敏感度"与"优化到底能找到什么"之间的精确机制一直缺失。而且平均敏感度太粗：First（只看第一位的 dictator 函数）平均敏感度只有 1，却同样学不会——单看平均敏感度根本解释不了。

**本文目标**：用**测度论**刻画不同类布尔函数在参数空间里占据的"体积"，把"能不能学会"翻译成"对应参数区域的勒贝格测度是不是零"。

**切入角度**：作者把视角从"平均敏感度"升级到**敏感度谱（sensitivity profile）**——即各个敏感度取值在所有输入字符串上的分布，比单个均值携带细得多的信息。核心物理量是 layer norm 带来的"放大率（blowup）"：layer norm 分母里那个稳定常数 $\epsilon$ 决定了单个参数扰动 / 单比特翻转能被放大多少。

**核心 idea**：证明随机初始化的 Transformer 的敏感度谱**必有非空的低端尾巴**（至少多项式个字符串敏感度很低）；等价地说，"编码极少零敏感度字符串的函数"的参数构成测度零集，随机初始化几乎必然落不进去——于是缺乏零敏感度字符串的函数可证不可学。

## 方法详解

### 整体框架
论文把 Transformer 视为一个**识别器**：它把二进制串 $\mathbf{x}\in\{0,1\}^N$（末尾接 eos）分类为属于 / 不属于某语言，并采用带间隔 $\xi$ 的识别定义——输出 logit $T(\mathbf{x})=\boldsymbol{w}^\top\boldsymbol{y}_{N+1}^L+\omega$ 须 $\ge\xi$ 或 $\le-\xi$ 才算判定。关键的"提升"动作是：把 Transformer 看成**参数与输入的二元函数** $T:\Theta\times\{0,1\}^*\to\mathbb{R}$，其中参数空间 $\Theta\subset\mathbb{R}^M$ 是紧集，从而可以谈"某类函数对应的参数子集占多大体积"。

全文按 layer norm 稳定常数 $\epsilon$ 的两种设定分两条腿走：$\epsilon>0$（热身，分母有下界、整个网络对参数一致 Lipschitz）与 $\epsilon=0$（主战场，分母可任意小、放大率不再一致有界）。两种情形最终都导向同一个反直觉结论——Transformer 识别器**表达力惊人地弱**。下面四个关键设计依次给出：度量工具（敏感度谱 + 放大率）、$\epsilon>0$ 的强不可表达结论、$\epsilon=0$ 的测度论主定理、以及由此推出的不可学习清单。

### 关键设计

**1. 敏感度谱 + 放大率：把"可学性"翻译成可计算的几何量**

痛点是平均敏感度太粗，盖不住 First 这种"低均值却学不会"的反例。作者引入两件度量工具。其一是**敏感度谱**：函数 $f$ 在长度 $N$ 上对每个可能敏感度值 $n$ 都记录"恰有 $K_n(N)$ 个字符串的敏感度等于 $n$"，其中单串敏感度 $s_N(\mathbf{x},f)=\sum_{n=1}^N|f(\mathbf{x})-f(\mathbf{x}^{\oplus n})|^2$ 数的是"翻转某一位会改变输出"的邻居个数。这比平均敏感度 $\mathrm{as}_N(f)=2^{-N}\sum_\mathbf{x}s_N(\mathbf{x},f)$ 精细——尤其它关心**零敏感度字符串的数量** $K_0(N)$。其二是**放大率（blowup）**：单层放大率 $\tau^\ell=\max_n(1+z_n^\ell)$ 量化 layer norm 输出能把输入变化放大多少（$z_n^\ell=1/\sqrt{\sigma^2(\boldsymbol{d}_n^\ell)+\epsilon}$ 是归一化系数），累积放大率 $\beta^L=\prod_\ell\tau^\ell$。Lemma 2.9 把"翻转第 $n$ 位对输出的影响"上界成放大率的乘积 $\mathrm{I}_n\le C_{\text{infl}}\beta^L(\boldsymbol{\theta},\mathbf{x})\beta^L(\boldsymbol{\theta},\mathbf{x}^{\oplus n})(\frac{1}{N}+\delta_{m,n})$。这把"输出对单比特敏不敏感"完全转化成"放大率有没有界"，是后面所有论证的杠杆。

**2. $\epsilon>0$ 情形：一致 Lipschitz 直接逼出"渐近只能算常数函数"**

当 $\epsilon>0$ 时分母有下界 $z_n^\ell\le1/\sqrt{\epsilon}$，于是累积放大率被死死压住 $\beta^L\le(1+1/\sqrt{\epsilon})^L$（Lemma 3.2），与输入长度 $N$ 无关。代入 Lemma 2.9，翻转任一非 eos 位对最终 eos 位激活的影响只有 $\mathcal{O}(1/N)$（Cor. 3.3）。后果非常强：只要函数在某串某位上会翻转输出（即非常数，$\mathrm{maxs}_N\ge1$），当 $N$ 足够大时这点 $\mathcal{O}(1/N)$ 的变化就跨不过间隔 $\xi$，于是**任何非常数布尔函数都无法被识别**（Cor. 3.4），即 $|\mathcal{F}_N|\to2$。更进一步，因为 $T$ 对参数一致 Lipschitz（Lemma 3.1，常数不依赖输入串），用一个 packing 论证可数出可识别函数数量被一个**与 $N$ 无关**的常数封顶：$|\mathcal{F}_N|\le(1+C_{\text{Lip}}\mathrm{diam}(\Theta)/\xi)^M$（Prop. 3.5），而总共有 $2^{2^N}$ 个布尔函数。结论：$\epsilon>0$ 的 Transformer 作为计算模型深度地不可表达。正因如此，作者才转向 $\epsilon=0$ 看能否恢复表达力。

**3. $\epsilon=0$ 的测度论主框架：随机 Transformer 几乎必然有零敏感度字符串**

$\epsilon=0$ 时放大率原则上可任意大，没有 Lemma 3.2 那样的确定性上界——这正是表达力可能更强的来源，也是分析最难的地方。作者的破解办法是把"确定性有界"换成"**高概率有界**"，分三层递进。第一层（Lemma 4.2）：固定串 $\mathbf{x}$，对参数施加一个均匀小扰动 $\boldsymbol{\Delta}\sim\mathrm{Unif}(B_\infty^M(\rho))$，则**以高概率**存在大小为 $k$ 的位集 $S$，使得放大率在 $\mathbf{x}$ 及其 $S$-汉明邻域上都被 $\beta^L=\mathcal{O}(N^{\zeta/2})$ 控住——直观理由是"危险区"（方差 $\sigma^2(\boldsymbol{d}_n^\ell)$ 太小、触发大放大率的参数）只占参数空间一小块体积。第二层（Cor. 4.3）：放大率有界 ⇒ 翻转 $S$ 中任一位对输出影响 $\mathcal{O}(N^{\zeta-1})$，跨不过间隔。第三层（Lemma 4.4）用测度论把单点邻域的结论扩到整个 $\Theta$：用足够小的立方体覆盖参数空间、在每个立方体内套用上一步，得到对 $\boldsymbol{\theta}\sim\mathrm{Unif}(\Theta)$，高概率有 $s_N(\mathbf{x},T(\boldsymbol{\theta},\cdot))\le N-k$。最后对所有足够大的 $N$ 用第一 Borel–Cantelli 引理聚合，得到**主定理 Thm. 4.5**：以概率 1，随机 Transformer 对所有足够大的 $N$ 都至少有 $N^{\frac{D-1}{2L}-5}$ 个零敏感度字符串。换言之，"编码极少零敏感度字符串的函数"的参数是测度零集。

**4. 从敏感度谱到不可学习清单：Parity / First 可证不可学，Majority 反而幸免**

主定理的推论 Cor. 4.6 给出干净的判据：若函数族满足 $K_0(N)<N^{\frac{D-1}{2L}-5}$（零敏感度字符串太少），则几乎所有参数的 Transformer 在 $N$ 足够大时都无法以间隔 $\xi$ 识别它。把它代入具体函数就得到一串不可学习结论：**Parity** 每个输入敏感度都是 $N$、零敏感度字符串为 0，故几乎必然学不会（Cor. 5.1）；$m$-Sparse Parity（最小敏感度 $m$）同理（5.2）；**dictator 函数 / First**（最小敏感度 1）也学不会（5.3, 5.4）——注意 First 平均敏感度只有 1，是平均敏感度分析完全抓不到、但敏感度谱能抓到的关键案例。反过来，**Majority** 有指数多个零敏感度字符串（只有"接近平衡"的输入才敏感），轻松越过多项式门槛 $N^{\frac{D-1}{2L}-5}$，本定理**管不到它**——这与"Transformer 能学好 Majority"的经验一致。作者据此提出 Conjecture 5.5：Majority 对应的参数子集具有正勒贝格测度，作为不可学习图景的正向补充。

## 实验关键数据

实验目的是**验证理论预测**：随机初始化（及训练后）Transformer 的敏感度谱是否真的严重偏向零。

### 主实验：随机初始化的敏感度谱重度偏零
对四种初始化方案（uniform、Gaussian、Xavier uniform、Xavier Gaussian）各随机采样大量超参配置的模型，计算敏感度谱。

| 设置 | 观察到的敏感度谱 | 与理论的关系 |
|------|------------------|--------------|
| 随机初始化（四种方案） | 一致地重度偏向 0，甚至比 Thm. 4.5 预测的更强 | 证实敏感函数占参数空间极小区域 |
| 训练后 | 低敏感度偏置仍然持续 | 说明初始化施加的归纳偏置约束了梯度优化能到达的函数 |
| 均匀随机布尔函数（对照） | 存在零敏感度串的概率 ≈ $1-1/e\approx0.63$ | 随机 Transformer 该概率为 1，确认是真实偏置而非"低敏感函数本就更多" |

### 训练实验：哪些函数能学、哪些不能
在 Parity、Majority、First 及若干 $m$-Sparse 变体上训练 Transformer。

| 目标函数 | 学习结果 | 敏感度谱表现 | 对应理论 |
|----------|----------|--------------|----------|
| Parity | 失败 | 达不到正确（全 $N$）敏感度谱 | 印证 Cor. 5.1 不可学 |
| First | 中等长度可学 | 敏感度正确聚集在 1 | 不矛盾：不可学是渐近的 |
| Majority | 可靠学会 | 敏感度聚在 0 与半串长 | 为 Conj. 5.5 提供经验支撑 |
| $m$-Sparse Parity/Majority | 镜像各自非稀疏版本 | 与对应母函数模式一致 | 与推论一致 |

### 关键发现
- **低敏感度偏置在训练后依然存在**，不是初始化的暂时假象——这暗示初始化的归纳偏置实质性地约束了泛化方向（作者用 PAC-Bayes 与 Bayesian 视角论证：后验绝对连续于先验，必继承其测度零集）。
- **渐近 ≠ 实际长度**：First 在中等长度能学会，与"渐近不可学"并不冲突——失败案例必然存在，但可能出现在比真实应用更长的输入上。
- **偏置是真实的**：对照"均匀随机布尔函数有零敏感度串的概率仅 0.63"vs"随机 Transformer 为 1"，证明这是 Transformer 的内在偏置，而非"低敏感函数数量更多"的统计假象。

## 亮点与洞察
- **从"平均敏感度"升级到"敏感度谱"是关键一招**：它一举抓住了 First 这种"平均敏感度低却学不会"的案例，是旧分析框架完全覆盖不了的，体现了"看分布而非看均值"的威力。
- **把可学习性问题几何化 / 测度化**：用"对应参数子集的勒贝格测度是否为零"来定义"几乎必然学不会"，给"表达力 vs 可学习性"的鸿沟一个干净的数学刻画，思路可迁移到其他架构的归纳偏置分析。
- **$\epsilon>0$ 与 $\epsilon=0$ 双线推进**：先用一致 Lipschitz 在 $\epsilon>0$ 拿到"渐近只能算常数"的强结论，再在更难的 $\epsilon=0$ 用"高概率有界 + 立方体覆盖 + Borel–Cantelli"的测度论组合拳，论证结构本身很值得学习。
- **正负对称的图景**：不仅证 Parity/First 不可学（负向测度零），还猜想 Majority 占正测度（正向），让结论不止于"Transformer 什么都学不会"的悲观叙事。

## 局限与展望
- **结论是渐近的**：不可学习只在 $N$ 足够大时成立，实际输入长度下 First 等仍可学。作者坦承"零敏感度串必然出现的阈值"可能超出真实应用长度，这是分析的根本局限。
- **bound 可能不紧**：理论给的是零敏感度串数量的**多项式**下界，但实验暗示真实增长可能是**指数**的；证明指数 scaling 需要不同技术，仍 open。
- **Majority 只有猜想**：Conj. 5.5（Majority 占正测度）只有经验支撑，尚无证明；"零敏感度串充足是否保证可学"这一逆命题也未解决。
- **理想化的识别器模型**：分析基于带间隔的形式语言识别器、固定 / 排除位置编码等假设，与真实工程 Transformer（可学习位置编码、不同 layer norm 放置）之间仍有距离（作者论证后者不影响结论，但仍是简化）。
- 从"随机初始化的测度零"到"训练后的不可达"靠 PAC-Bayes / Bayesian 论证桥接，训练本身并非严格 Bayesian 推断。

## 相关工作与启发
- **vs Hahn & Rofin (2024)**：本文沿用其 blowup 框架与 Transformer 形式化，但把分析对象从**平均敏感度**推广到**整条敏感度谱**，从而能覆盖 First 等平均敏感度无法刻画的函数，是对其结果的推广与加强。
- **vs Bhattamishra 等 (2023)、Abbe 等 (2023)**：他们经验/理论上观察到 Transformer 难学敏感 / 稀疏布尔函数；本文给出"为什么"——把现象归因于参数空间里敏感函数占测度零的几何事实。
- **vs Chiang & Cholak (2022)**：他们证明带 layer norm 的 Transformer 表达力上能表示 Parity；本文正是去填"能表示却学不会"的鸿沟，从可学习性而非表达力角度回答。
- **vs Buzaglo 等 (2024)、Dziugaite & Roy (2025)**：他们用 PAC-Bayes 把"随机插值网络的泛化"与"目标函数兼容参数体积"挂钩；本文借此把"初始化测度零"推到"训练后难以泛化"，两者互补。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用参数空间测度几何 + 敏感度谱给"表达力 vs 可学习性鸿沟"一个全新且可证的刻画，覆盖了旧分析的盲区。
- 实验充分度: ⭐⭐⭐⭐ 四种初始化 + 多函数训练系统地验证了敏感度谱偏零的核心预测，但本质是验证性、规模有限。
- 写作质量: ⭐⭐⭐⭐⭐ 双线（$\epsilon>0/\!=0$）结构清晰，对"渐近 vs 实际""bound 紧不紧""Majority 仅为猜想"等局限交代得非常诚实。
- 价值: ⭐⭐⭐⭐ 为理解 Transformer 归纳偏置提供了坚实的理论基底，概念上很有启发；但渐近性与理想化假设限制了对实际训练的直接指导。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Generalizing Analogical Inference from Boolean to Continuous Domains](../../AAAI2026/learning_theory/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)
- [\[ICML 2026\] Task-Restricted Symmetries in Recurrent Weight Space](task-restricted_symmetries_in_recurrent_weight_space.md)
- [\[ICLR 2026\] A Faster Parameter-Free Regret Matching Algorithm](../../ICLR2026/learning_theory/a_faster_parameter-free_regret_matching_algorithm.md)
- [\[ICLR 2026\] Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity](../../ICLR2026/learning_theory/barriers_for_learning_in_an_evolving_world_mathematical_understanding_of_loss_of.md)
- [\[ICML 2026\] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)

</div>

<!-- RELATED:END -->
