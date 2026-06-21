---
title: >-
  [论文解读] Statistical and Structural Identifiability in Representation Learning
description: >-
  [ICLR 2026][表示学习理论][表示学习] 本文把"表示稳定性"拆成两个独立概念——统计可辨识性（多次重训得到一致表示）与结构可辨识性（表示对齐到真实生成因子），提出带误差容忍 $\epsilon$ 的"近可辨识"定义，证明了一类带非线性解码器模型（MAE、监督学习器、GPT 中间层）的统计 $\epsilon$-近可辨识，并指出用线性 ICA 后处理潜空间即可消除剩余线性不确定性，得到一个极简的解耦"配方"，在合成解耦基准上用 vanilla autoencoder 就达到 SOTA，在细胞显微的基础模型上把生物变异与批次效应分开。
tags:
  - "ICLR 2026"
  - "表示学习理论"
  - "可辨识性"
  - "表示学习"
  - "ICA"
  - "解耦"
  - "近等距"
---

# Statistical and Structural Identifiability in Representation Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Wa3cfE3Iay](https://openreview.net/forum?id=Wa3cfE3Iay)  
**领域**: 表示学习理论 / 可辨识性  
**关键词**: 可辨识性、表示学习、ICA、解耦、近等距

## 一句话总结
本文把"表示稳定性"拆成两个独立概念——统计可辨识性（多次重训得到一致表示）与结构可辨识性（表示对齐到真实生成因子），提出带误差容忍 $\epsilon$ 的"近可辨识"定义，证明了一类带非线性解码器模型（MAE、监督学习器、GPT 中间层）的统计 $\epsilon$-近可辨识，并指出用线性 ICA 后处理潜空间即可消除剩余线性不确定性，得到一个极简的解耦"配方"，在合成解耦基准上用 vanilla autoencoder 就达到 SOTA，在细胞显微的基础模型上把生物变异与批次效应分开。

## 研究背景与动机

**领域现状**：各种自监督模型尽管模态、任务、数据迥异，却在内部表示上呈现惊人的稳定性——不同模型似乎在收敛到一组共享的世界表示。研究这一现象的经典工具是"可辨识性"（identifiability）：在似然推断里它指数据足以唯一确定模型参数，在神经网络里则被放松为"无限数据足以把训练后模型的表示确定到某个等价类（如线性变换）"。

**现有痛点**：已有可辨识性结果有两类局限。一是要么对数据生成过程下很强的假设（如对比学习要求增广分布在潜空间满足各向同性，这在没有真值因子时根本无法验证），要么假设表示与损失之间是线性关系（Roeder 等只能处理把表示线性映射到损失的倒数第二层）。二是文献普遍**没有区分**两种本质不同的"稳定"：表示在多次重训之间一致，和表示对齐到某个真实的生成因子，是两码事，却常被当作同一个性质混谈。

**核心矛盾**：现代模型（如 MAE）真正有用的往往是**中间层**表示，而这些表示是被**非线性解码器/头**映射到损失的，落在已有理论的覆盖范围之外；同时"完美逐点可辨识"对现代大模型本就不现实，理论需要一个能容忍小误差的松弛版本。

**本文目标**：(1) 给出统计可辨识性与结构可辨识性的清晰、模型无关的近可辨识定义；(2) 证明带非线性解码器的一大类模型的中间层表示也是统计近可辨识的；(3) 给出从统计可辨识走到结构可辨识（即解耦）的最小假设与实用算法。

**切入角度**：作者引入一个"松弛量" $\epsilon$，把可辨识性从"完全相等"放宽成"相差一个简单变换群 $H$ 加一点小扰动 $\epsilon$"，并把 $\epsilon$ 的大小直接挂到解码器的**局部双 Lipschitz 常数**上——这是一个可以被常见正则手段（趋向"动态等距"）所控制的量。

**核心 idea**：用"近等距/双 Lipschitz 的解码器"这一**模型侧温和假设**替代"对数据生成过程的强假设"，证明中间表示在刚性变换下近可辨识，再用线性 ICA 把剩余线性不确定性消到带符号置换，从而得到"自编码器 + 潜空间 ICA"这一极简解耦配方。

## 方法详解

### 整体框架

本文是一篇理论论文，主线是把"表示为何稳定"这一现象用三条递进的定理串起来，再用四组实验验证。整体逻辑是：先重新定义"可辨识"（引入 $\epsilon$ 松弛与变换群 $H$），区分**统计**与**结构**两层；然后证明统计层（Theorem 1：中间层表示在刚性变换 $H_{\text{rigid}}$ 下近可辨识，$\epsilon$ 由解码器双 Lipschitz 常数控制）；再用 ICA 把线性不确定性收紧（Theorem 2：从 $H_{\text{linear}}$ 经白化降到 $H_{\text{rigid}}$、再经 ICA 降到带符号置换 $H_\sigma$）；最后在数据生成过程也满足双 Lipschitz 时把统计可辨识升级为结构可辨识（Theorem 3：自编码器 + ICA 能近似恢复真实潜因子 $g^{-1}$）。

记数据分布 $P(x)$、模型 $M=\{L_\theta:\theta\in\Theta\}$，$F:\theta\mapsto f_\theta$ 把参数映射到表示函数 $f_\theta:X\to\mathbb{R}^D$，$S\subset\Theta$ 为期望损失 $\mathbb{E}_{x\sim P}[L_\theta(x)]$ 的最小值点集合。三个核心变换群：$H_{\text{linear}}$（可逆线性变换）、$H_{\text{rigid}}$（旋转/反射/平移构成的刚性变换，主不确定性可看成 $SO(D)$ 里的旋转）、$H_\sigma$（带符号置换，因为潜变量本就没有天然的排序和符号，这一层一般无法也无需消除）。

### 关键设计

**1. 统计与结构可辨识的 $\epsilon$-近可辨识定义：把"稳定"拆成"一致"与"正确"两件事**

已有工作把表示稳定当成单一性质，本文指出它其实包含两个层次。统计可辨识刻画的是"一致"：优化同一个模型多次得到的表示彼此只差一个简单变换。形式上（Definition 1），若对最小值集合 $S$ 中任意 $\theta,\theta'$ 都存在 $h\in H$ 使 $\lVert f_\theta - h\circ f_{\theta'}\rVert \le \epsilon$，则称模型在群 $H$ 下统计 $\epsilon$-近可辨识；范数取 $L^\infty$（关于 $P$ 的本质上确界）。$\epsilon=0$ 时退化为精确可辨识，这就把数理统计里的经典可辨识性推广了。结构可辨识则刻画"正确"：需要先假设存在某个不可观测的生成因子 $u$（$P(u)$、$P(x\mid u)$，且 $u(x)=\arg\sup_u P(x\mid u)$ 几乎处处良定），若对所有 $\theta\in S$ 有 $\lVert h\circ f_\theta - u\rVert\le\epsilon$，则称模型 $\epsilon$-近地辨识出结构 $u$（Definition 2）。直觉上，统计可辨识是"每次都一样"，结构可辨识是"每次都对"，后者严格更强——解耦正是 $P(u)$ 各分量独立的特例。引入 $\epsilon$ 的意义在于：现代大模型不可能逐点精确可辨识，但"近可辨识"是可达且可度量的，这是本文区别于以往逐点结果的关键。

**2. Theorem 1——非线性解码器下中间表示的刚性近可辨识：把可辨识性从最后一层推广到任意中间层**

以往如 Roeder 等的结果只覆盖那些被**线性**映射到损失的倒数第二层表示（指数族负对数似然形式 $L_\theta(x,y)=-\eta_\theta(x)^\top t_\theta(y)+A_\theta(x)$，$\eta_\theta$ 即表示，可辨识到 $H_{\text{linear}}$）。但很多模型真正关心的是被**非线性**解码器/头映射到损失的更早层表示。本文把端到端网络拆成 $H:\theta\mapsto g_\theta\circ f_\theta$（编码器 $f_\theta$ 接解码器 $g_\theta$），假设整体输出 $g_\theta\circ f_\theta$ 是统计可辨识的（如 MSE 损失下网络学到条件均值这一最优函数），则编码器表示 $f_\theta$ 的近可辨识程度由解码器 $g_\theta$ 的**局部双 Lipschitz 常数**决定。具体地，若 $1+L$ 是 $g_\theta$ 的局部双 Lipschitz 上界，则 $(P,\Theta,L_\theta,F)$ 在 $H_{\text{rigid}}$ 下统计 $\epsilon$-近可辨识，

$$\epsilon = c_D\sqrt{2L+L^2}\,\Delta$$

其中 $c_D$、$\Delta$ 是与模型无关的常数。直觉是：双 Lipschitz 约束控制解码器"扭曲距离"的程度，当潜变量的小变化只引起输出的小变化时常数接近 1、$\epsilon$ 就小。这是目前已知最一般的中间层可辨识性量化结果，覆盖 MAE、下一 token 预测器、监督学习器。它把强假设从"数据生成过程"挪到了"模型类"上——而双 Lipschitz/动态等距恰好是谱归一化、ReZero、零初始化残差等正则手段在实践中本就趋近的状态（Jacobian 奇异值集中在 1 附近）。

**3. Theorem 2——ICA 消除剩余线性不确定性：把刚性歧义收紧到带符号置换**

Theorem 1 留下的是刚性（或线性）不确定性，要把表示真正用起来还得把这层歧义消掉。本文不提出新的 ICA 可辨识性理论，而是证明 $\epsilon$-近可辨识这一松弛对下游 ICA"不添乱"。设 $(P,\Theta,L_\theta,F)$ 在 $H_{\text{linear}}$ 下统计 $\epsilon$-近可辨识，对潜表示先做白化、再做基于对比函数的 ICA，得到的新模型 $(P,\Theta',L'_\theta,F')$ 在带符号置换 $H_\sigma$ 下统计 $\epsilon'$-近可辨识，且

$$\epsilon' = K\epsilon + K'\epsilon^2$$

其中 $K,K'$ 是与 $\epsilon$ 无关、取决于表示协方差谱与 ICA 对比函数性质的常数。机制是分两步收紧：白化把线性不确定性降为刚性，ICA（若收敛充分）再把刚性降为带符号置换，每步的"近性"以新常数的形式被保留。这一步是连接理论与实用的关键——它说明只要表示线性可辨识，一个**纯无监督**的 ICA 后处理就能逼近"标准化"的潜坐标。

**4. Theorem 3——从统计可辨识到结构可辨识：双 Lipschitz 数据生成过程下自编码器 + ICA 恢复真因子**

要从"每次一样"升到"每次都对"，必须对数据生成过程也加假设。本文假设真因子分布 $P(u)$ 是各分量独立的非高斯多元分布，数据 $P(x)$ 由一个 $(1+\delta)$-双 Lipschitz 的光滑微分同胚 $g$ 推前生成。则对一个达到完美重构（即整体 $g_\theta\circ f_\theta$ 在极限下结构地辨识出恒等函数）的编码器-解码器模型，$f_\theta$ 就在 $H_{\text{rigid}}$ 下 $\epsilon$-近地辨识出结构 $g^{-1}$；再叠加白化 + ICA，则在 $H_\sigma$ 下 $\epsilon'$-近地辨识出 $g^{-1}$。关键在于这里对数据生成过程下的假设（双 Lipschitz）与对模型下的假设**同源**，所以结构可辨识只是 Theorem 2 的一个相当直接的推论。代价是 Theorem 3 要求**完美重构**（自编码型），而 Theorem 2 只要输出可辨识、更一般。作者还用 dSprites 风格的连续松弛图像论证：平移一个白方块是局部等距（1-双 Lipschitz），$\lVert f'(p)\rVert_2=2r$ 为常数，从而几何距离 $\propto|p_1-p_0|$ 被保持——说明真实图像流形被等距近似并非空想，由此把解耦实际归约为"vanilla autoencoder + 潜空间线性 ICA"这一极简配方。

### 损失函数 / 训练策略

本文不引入新损失。实验里用的都是标准目标：autoencoder 的重构损失、对比/掩码自监督模型的原始损失；唯一被刻意调节的是控制解码器双 Lipschitz 常数的因素——如 LeakyReLU 的 leak 参数 $\alpha$（3 层解码器双 Lipschitz 上界约 $1/\alpha^K$，$K=3$；$\alpha=1$ 为线性网络，$\alpha=0$ 为 ReLU 网络）与权重衰减（已知对 vanilla autoencoder 足以正则化解码器 Lipschitz 常数）。ICA 作为**训练后**的潜空间后处理（白化 + 对比函数 ICA），不参与模型训练。

## 实验关键数据

### 主实验

四组实验分别对应四条主张。Table 1 验证统计近可辨识与 ICA 的消歧能力：在相同架构/损失/数据、仅不同初始化的模型对之间，估计各种最优变换并报告按潜空间直径归一化的平均 $\ell_2$ 误差，ICA 效率定义为相对刚性变换的 $\ell_2$ 误差下降百分比。

| 模型对 | Permutation | Supervised Rigid | Supervised Linear | ICA (% eff.) |
|--------|-------------|------------------|-------------------|--------------|
| Pythia-160M-0 → Pythia-160M-1 | 0.219 | 0.150 | 0.131 | 0.202 (25%) |
| MAE-timm → MAE-original | 0.197 | 0.109 | 0.036 | 0.145 (59%) |
| CheXpert-small → CheXpert-base | 0.218 | 0.104 | 0.048 | 0.175 (38%) |
| ResNet-18-fc-1 → ResNet-18-fc-2 | 0.382 | 0.206 | 0.175 | 0.312 (40%) |

GPT 类（Pythia）表现出优秀的线性对齐（符合 Roeder 理论），MAE 则如本文理论所预测表现出刚性对齐（含一例跨模型尺寸）；所有情形下无监督 ICA 都消除了大量线性不确定性，MAE 上 ICA 的效率接近全监督最优刚性变换的 60%。

Table 2 验证结构可辨识（解耦）：在四个合成数据集上，vanilla autoencoder + 潜空间 ICA 与专门的解耦网络比较 InfoMEC 三指标（模块度 InfoM、显式度 InfoE、紧致度 InfoC，前两者更重要）。

| 模型 | aggregated (InfoM InfoE InfoC) | Shapes3D | MPI3D | Falcor3D | Isaac3D |
|------|-------------------------------|----------|-------|----------|---------|
| AE | (0.39 0.76 0.25) | (0.34 0.99 0.16) | (0.42 0.40 0.31) | (0.37 0.83 0.20) | (0.41 0.80 0.34) |
| β-VAE* | (0.59 0.81 0.55) | (0.59 0.99 0.49) | (0.45 0.71 0.51) | (0.71 0.73 0.70) | (0.60 0.80 0.51) |
| β-TCVAE* | (0.58 0.72 0.59) | (0.61 0.82 0.62) | (0.51 0.60 0.57) | (0.66 0.74 0.71) | (0.54 0.70 0.46) |
| BioAE* | (0.54 0.75 0.36) | (0.56 0.98 0.44) | (0.45 0.66 0.36) | (0.54 0.73 0.31) | (0.63 0.65 0.33) |
| AE + ICA (ours) | (0.65 0.83 0.40) | (0.79 0.99 0.52) | (0.44 0.66 0.31) | (0.71 0.83 0.33) | (0.64 0.82 0.43) |

仅调权重衰减一个超参的 AE + ICA 在聚合指标上**平均优于所有**专门解耦模型（带 * 的结果引自 Hsu 等 2023，未复现）。

### 消融实验

Table 3/4 是真实世界的细胞显微基础模型 OpenPhenom（大型 MAE）上的去混杂消融：对潜空间分别用原始嵌入（Base）、白化（PCA）、白化 + ICA（PCA + ICA）、白化 + 随机旋转（PCA + Rand）做扰动分类（control vs perturbed），跨批次评估泛化。

| 配置 | 平均 AUROC (↑) | Hoyer 稀疏度 (↑) | 生物变异在 top 25% 特征的集中度 (↑) | 说明 |
|------|---------------|-----------------|-----------------------------------|------|
| Base | ~0.66–0.80 | 较低 | 0.163 | 未变换嵌入 |
| PCA | 提升 | 提升 | 0.332 | 仅白化 |
| PCA + ICA | **最高** | **最高** | **0.386** | 白化 + ICA 旋转 |
| PCA + Rand | 介于中间 | 中等 | 0.287 | 白化 + 随机旋转，对照 |

例如 EIF3H 基因上 AUROC 从 Base 0.682 → PCA 0.724 → PCA+ICA 0.749；PCA+Rand 0.725 明显不如 ICA，说明增益来自 ICA 的特定旋转而非单纯白化或任意旋转。

### 关键发现
- **双 Lipschitz 常数确实预测可辨识性**：warmup 实验（MNIST，调 LeakyReLU 的 $\alpha$）显示经验估计的 $\sqrt{L+L^2}$ 项能预测刚性对齐的 $\ell_2$ 误差，与 Theorem 1 的比例关系吻合——这是理论"可被实验直接验证"的少数可辨识性结果之一。
- **ICA 的增益是"对的旋转"带来的**：PCA + Rand 这一对照表明，把潜空间随机旋转无法复现 ICA 的提升，ICA 找到的特定旋转把生物信号集中到少数特征（稀疏度、集中度同步上升），从而把生物变异从技术批次效应中分离出来。
- **架构差异符合理论分型**：GPT 类线性对齐、MAE 类刚性对齐，与各自损失把表示映射到损失的方式（线性 vs 非线性解码）一致。

## 亮点与洞察
- 把"表示稳定"清晰拆成统计可辨识（一致）与结构可辨识（正确），并用 $\epsilon$ 松弛让"近可辨识"第一次成为可度量、可验证的通用概念——以往的逐点可辨识在现代大模型上根本无法成立。
- 最巧的一步是"换假设位置"：把强假设从无法验证的数据生成过程，挪到可被常见正则手段（趋向动态等距/双 Lipschitz）控制的模型类上，让理论能覆盖 MAE、监督学习器、GPT 中间层这些真实模型。
- 给出一个近乎"免调参"的解耦配方——vanilla autoencoder + 潜空间线性 ICA，在合成基准上击败专门的 β-VAE/β-TCVAE/BioAE，可直接迁移到任何已训练好的表示模型做后处理。
- 真实生物应用（cell painting 去批次效应）展示了纯无监督分离技术变异与生物变异、并实质提升跨批次 OOD 泛化，把抽象的可辨识性理论落到了 drug discovery 的实际痛点上。

## 局限与展望
- 局部双 Lipschitz 条件在实践中难以直接检验，作者只能借"动态等距"类正则手段间接论证，$\epsilon$ 的绝对大小无法精确测量。
- Theorem 3 的结构可辨识依赖**完美重构**与"数据生成过程是双 Lipschitz 微分同胚 + 真因子独立非高斯"这一组较强假设，真实复杂数据未必满足；图像 isometry 论证主要建立在 dSprites 这类过于简化的合成流形上。
- 所有定理都是"无限数据极限"下的近可辨识，有限样本下 $\epsilon$ 的统计估计误差与 ICA 收敛性如何影响结论，文中未给出有限样本保证。
- 结构可辨识完全无监督、靠归纳偏置而非干预，这与因果表示学习路线互补但也意味着无法保证恢复的因子在因果意义上"正确"。

## 相关工作与启发
- **vs Roeder 等 (2021)**：他们只覆盖被线性映射到损失的倒数第二层、可辨识到 $H_{\text{linear}}$；本文用解码器双 Lipschitz 假设把结论推广到任意中间层、非线性映射，且统一进 $\epsilon$-近可辨识框架。
- **vs Zimmermann 等 (2021)（对比学习 InfoNCE）**：他们要求增广分布在潜空间各向同性这一强且不可验证的数据生成假设才得到 $H_{\text{rigid}}$ 可辨识；本文只对模型类下温和假设，假设更少更可控。
- **vs Buchholz & Schölkopf (2024)（非线性 ICA）**：Theorem 2 与其 Theorem 3.1 形式相近，但本文用 $L^\infty$ 双 Lipschitz 给出逐点约束，而对方用 $L^2$ 平均意义的近等距换取更弱的 Jacobian 约束。
- **vs Horan 等 (2021)（等距学习 + ICA）**：本文是其完美可辨识情形的推广，证明 $\epsilon$-近性对下游 ICA 不增加额外复杂度，并把它从合成设定推进到基础模型规模的真实生物数据。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次清晰区分统计与结构可辨识，并给出通用 $\epsilon$-近可辨识定义与最一般的中间层可辨识性定理。
- 实验充分度: ⭐⭐⭐⭐ 从 MNIST 受控验证、预训练模型测量、合成解耦基准到真实细胞显微基础模型，覆盖完整，但有限样本/绝对 $\epsilon$ 量化偏弱。
- 写作质量: ⭐⭐⭐⭐ 理论分层清晰、定理配直觉解释，但定理只给非正式版、细节全在附录，阅读门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 既统一了可辨识性理论，又给出可直接复用的无监督解耦配方与真实生物应用，理论与实用兼具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)
- [\[ICLR 2026\] Learning Correlated Reward Models: Statistical Barriers and Opportunities](learning_correlated_reward_models_statistical_barriers_and_opportunities.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICLR 2026\] Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects](minimax_sample_complexity_of_graph_neural_networks_lower_bounds_and_structural_e.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)

</div>

<!-- RELATED:END -->
