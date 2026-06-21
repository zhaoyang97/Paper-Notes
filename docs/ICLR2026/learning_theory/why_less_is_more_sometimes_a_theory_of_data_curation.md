---
title: >-
  [论文解读] Why Less is More (Sometimes): A Theory of Data Curation
description: >-
  [ICLR 2026][学习理论][缩放定律] 本文用高维二分类 + 随机矩阵理论，给"保留难样本 / 保留易样本 / 全量训练"这几种数据筛选策略推出了**精确的测试误差缩放曲线**，并证明"less is more"（小而精的数据更好）只在「数据充足 + 生成器足够强」这一个象限里成立，从理论上统一解释了 LIMO / s1 等方法为何有效、又为何在最难题上反而"more is more"。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "数据筛选"
  - "高维统计"
  - "缩放定律"
  - "相变分析"
  - "模型崩溃"
  - "随机矩阵理论"
---

# Why Less is More (Sometimes): A Theory of Data Curation

**会议**: ICLR 2026  
**论文**: [OpenReview](https://openreview.net/forum?id=why_less_is_more_sometimes_a_theory_of_data_curation)（⚠️ 链接以原文为准）  
**代码**: 无  
**领域**: 学习理论 / 数据筛选 / 高维统计  
**关键词**: 数据筛选, 缩放定律, 相变分析, 模型崩溃, 随机矩阵理论

## 一句话总结
本文用高维二分类 + 随机矩阵理论，给"保留难样本 / 保留易样本 / 全量训练"这几种数据筛选策略推出了**精确的测试误差缩放曲线**，并证明"less is more"（小而精的数据更好）只在「数据充足 + 生成器足够强」这一个象限里成立，从理论上统一解释了 LIMO / s1 等方法为何有效、又为何在最难题上反而"more is more"。

## 研究背景与动机

**领域现状**：经典缩放定律（Kaplan、Hoffmann 等）告诉大家"more is more"——数据越多、泛化越好，于是训练基础模型动辄要几千亿 token。但训练里所有样本被一视同仁地塞进 loss，不管它有没有信息量。

**现有痛点**：近期一批工作（LIMO = Less Is More、s1）反着来：只挑一小撮"又对又难"的样本，反而把推理性能拉得更高，几千条胜过几十万条。这和"more is more"直接冲突，但大家只有经验观察，没有理论说清**什么时候该 less、什么时候该 more**。更糟的是同一个数据集上还出现自相矛盾的现象：AIME 平均分上"少而精"赢，可一旦只看最难的题，又变成数据越多越好。

**核心矛盾**：数据筛选到底是有害还是有益，取决于一组互相纠缠的因素——**生成训练标签的"生成器"有多准、做筛选的"oracle"有多准、二者方向是否对齐、以及数据量多大**。缺了对这几个量的精确刻画，"less vs more"就只能靠拍脑袋。

**本文目标**：不再提出又一个启发式筛选 trick，而是建一个能**解析求解测试误差**的理论框架，把"何时 prune、何时 scale"变成可计算的相变条件。

**切入角度**：把问题压缩到一个既够丰富、又解析可解的最小模型——**高维各向同性二分类**，配上一个按"难度 + 正确性"做筛选的 pruning oracle，然后用随机矩阵理论（RMT）在 $n,d\to\infty,\ d/n\to\phi$ 的比例极限下求出测试误差闭式解。

**核心 idea**：用三个几何对齐度量 $(\rho,\rho^*,\rho_g)$ 刻画"生成器质量 / oracle 质量 / 二者对齐"，证明最优筛选策略随生成器质量 $\rho$ 发生**相变**——强生成器该"保留难样本"，弱生成器该"保留易样本"。

## 方法详解

本文是纯理论/分析型工作，没有可训练的 pipeline，核心是一套数学建模 + 三条定理。因此不画框架图，用文字 + 公式把"建模—度量—定理"这条链讲清。

### 整体框架

设定一个**师生式的高维分类**问题。数据由生成分布 $P_g=P_{w_g,C_g}$ 产生：$x\sim N(0,C_g)$，标签 $y=\mathrm{sign}(x^\top w_g)$；而真实测试分布是 $P_*=P_{w_*,\Sigma}$。一般情况下 $w_g\neq w_*$（标签漂移）、$C_g\neq\Sigma$（协变量漂移），正文主结果聚焦各向同性、只保留标签漂移的情形（$C_g=\Sigma=I_d$）。

学习器解一个带样本掩码的岭回归（平方损失）：

$$\min_{w}\ \frac1n\sum_{i=1}^n p_i\,\ell(x_i^\top w; y_i)+\frac{\lambda}{2}\|w\|^2,\quad \ell(z;y)=\tfrac12(z-y)^2,$$

其中 $p_i\in\{0,1\}$ 表示样本 $i$ 是否被筛选保留，最优解有闭式 $\hat w=RX^\top DY/n$，$R=(S+\lambda I_d)^{-1}$，$S=X^\top DX/n$，$D=\mathrm{diag}(p_i)$。下游分类器是 $x\mapsto\mathrm{sign}(x^\top\hat w)$，研究对象是测试误差 $E_{\text{test}}(\hat w)=P(\mathrm{sign}(x^\top\hat w)\neq y)$。

关键约束（desiderata）：学习器**看不到** oracle 的方向 $w_o$，也看不到 oracle 标签 $y_o$，只能提交样本 $(x_i,y_i)$、拿到一个比特 $p_i$。整条理论就是要回答：在这种"黑盒筛选"下，挑哪些样本、挑多少，能让 $E_{\text{test}}$ 最小。

### 关键设计

**1. 把"数据筛选"统一成两类 pruning oracle**

痛点是现实中的筛选规则五花八门（按 margin 难度、按标签是否正确、LIMO 式"够难够多样才留"），缺一个统一刻画。本文用一个 oracle 方向 $w_o$ 和对称函数 $q$ 把它们收成两类：

- **标签无关（label-agnostic）**：只看特征，$p_i=q(x_i^\top w_o)$。取 $q(t)=\mathbf 1[|t|\ge\alpha]$ 就是"保留易样本"（KE，离决策边界远、大 margin），取 $q(t)=\mathbf 1[|t|\le\alpha]$ 就是"保留难样本"（KH，贴近边界）。阈值 $\alpha$ 控制保留比例。
- **标签感知（label-aware）**：更贴近 LIMO/s1，样本要同时"oracle 认为标签对"且"够有意思"才留：$p_i=1\iff y_i=y_i^o\ \text{且}\ q(x_i^\top w_o)=1$，其中 $y_i^o=\mathrm{sign}(x_i^\top w_o)$。

保留比例 $p:=P(p_i=1)$ 称作 pruning ratio，$p$ 小=激进剪枝，$p\to1$=全量。这套建模的好处是：后面所有结论都只通过几个标量进入误差公式，从而能解析比较不同 $q$。

**2. 三个对齐度量量化"生成器有多准、oracle 有多准"**

这是全文的灵魂。痛点是"less vs more"之争其实是在比生成器和真值、oracle 和真值的相对质量，但之前没人把它量化。本文用 Mahalanobis 内积定义三个余弦 + 一个余切：

$$\rho=\frac{w_g^\top C w_*}{\|w_g\|_C\|w_*\|_C},\quad \rho^*=\frac{w_o^\top C w_*}{\|w_o\|_C\|w_*\|_C},\quad \rho_g=\frac{w_o^\top C w_g}{\|w_o\|_C\|w_g\|_C},\quad \tau=\frac{\rho_g}{\sqrt{1-\rho_g^2}}.$$

其中 $\rho$=**生成器质量**（训练标签器 $w_g$ 与真值 $w_*$ 的夹角余弦），$\rho^*$=**oracle 质量**，$\rho_g$=oracle 与生成器的对齐度。它们直接决定各自的"baseline"误差：$E_{\text{test}}(w_g)=\frac1\pi\arccos\rho$，$E_{\text{test}}(w_o)=\frac1\pi\arccos\rho^*$。$\rho\to1$ 是强生成器，$\rho<1$ 是弱生成器（标签漂移）。整套理论的相变都挂在这三元组 $(\rho,\rho_g,\rho^*)$ 上。

**3. 用随机矩阵理论求出精确测试误差闭式解（Theorem 1）**

痛点：高维下 $\hat w$ 是个随机矩阵的函数，直接算误差不可行。本文为 resolvent $R$ 及 $R^2$ 构造确定性等价（deterministic equivalents），把任意筛选策略 $q$ 的全部影响压缩成四个标量：

$$p=\mathbb E[q(G)],\ \gamma=\mathbb E[q(G)G^2],\ \beta=2\mathbb E[q(G)\varphi(\tau G)],\ \tilde\beta=2\mathbb E[q(G)\Phi(\tau G)G],$$

其中 $G\sim N(0,1)$，$\varphi,\Phi$ 为标准正态的 PDF/CDF。于是测试误差在比例极限下收敛到

$$E_{\text{test}}(\hat w)\to\frac1\pi\arccos\!\big(|m_0|/\sqrt{\nu_0}\big),$$

$m_0,\nu_0$ 由上面四个标量、$\lambda$、以及 $\rho,\rho_g,\rho^*$ 的组合给出（$m$ 是被剪枝"形变"过的 Marchenko–Pastur 律的 Stieltjes 变换）。这条公式就是分析任何 $q$ 的"机器"——给定一个筛选策略，代入四个标量就能算出整条误差曲线。标签感知情形（Theorem 3）只需把这四个标量换成式 (13) 的修正版（用 $f_i=p_iy_i$ 的导数定义 $\beta,\tilde\beta$），误差公式形式不变。

**4. 最优筛选策略随生成器质量发生相变（Theorem 2）**

这是把公式翻译成可操作结论。在数据充足、无正则的极限里定义误差泛函 $F(q)=\lim_{\phi\to0}\lim_{\lambda\to0}\lim E_{\text{test}}(\hat w)$，固定保留比例 $p$、设 $\rho_g>0$，则：

- **(A) 生成器强（$\rho\to1$）+ oracle 强（$\rho^*\to1$）**：唯一最优是 **Keep Hard（保留难样本）**。直觉：模型已经把任务学得差不多，再喂简单样本是浪费，要靠难样本"精修"——这正是 LIMO 式"less is more"。
- **(B) 生成器弱（$\rho<1$）+ oracle 强（$\rho^*\to1$）**：唯一最优反转为 **Keep Easy（保留易样本）**。直觉：弱模型连基础都没打牢，难样本噪声太大反而误导，要先用干净的易样本把地基搭好。这一支特别适合缓解**模型崩溃**——自训练时模型成了自己的弱生成器。

一句话：**最优策略不是普适的，它随"生成器相对任务的强弱"在 KH 和 KE 之间相变**。

### 一个完整示例

用 LLM 数学推理把相变讲活。设基座 LLM（产生推理轨迹的 $w_g$）在**大多数 AIME 题**上是强生成器（高 $\rho$）：此时落在 Theorem 2(A)，最优是激进剪枝、保留难样本，于是 LIMO/s1 用 1k 条精选样本（56.7%）就胜过 59k（53.3%）甚至 114k（50.2%）——"less is more"。

但把镜头切到**只看最难的 AIME 题**：同一个基座 LLM 相对这批超纲题变成了弱生成器（低 $\rho$），落到"more is more"那一侧，于是性能随数据量单调上涨：1k→28.4%，2k→35.4%，10k→52.1%，1M→64.9%。同一个模型、同一个数据池，只因"任务难度切换"导致 $\rho$ 高低翻转，最优策略就从 less 翻到 more——这正是本文理论一次性解释清楚的"悖论"。

## 实验关键数据

### 主实验

合成实验在 2×2 网格里验证理论：横轴生成器质量（强 $\rho=1$ / 弱 $\rho<1$），纵轴数据量（小 $n=100$ / 大 $n=5000$）。实测（带误差棒）与理论曲线高度吻合，结论是 **"less is more" 只在「数据充足 + 强生成器」一个象限成立**，其余三个象限都是 $p=1$（全量）最优。

| 数据/生成器象限 | 最优保留比例 $p$ | 结论 |
|------|------|------|
| 小数据 + 强生成器 | $p=1$ | more is more |
| 小数据 + 弱生成器 | $p=1$ | more is more |
| 大数据 + 弱生成器 | $p=1$ | more is more |
| **大数据 + 强生成器** | $p\ll1$（激进剪枝） | **less is more** |

LLM 数学推理上的两张真实数据表印证相变（聚合自 LIMO/s1/Sun et al.）：

| 设定 | 训练数据量 | 指标 | 表现 | 体现原则 |
|------|------|------|------|------|
| AIME 平均（强生成器） | 0 / 114k / 59k / **1k** | Pass@1 | 16.5 / 50.2 / 53.3 / **56.7** | less is more |
| AIME 最难题（弱生成器） | 0 / 1k / 2k / 10k / **1M** | Avg@8 | 1.0 / 28.4 / 35.4 / 52.1 / **64.9** | more is more |

### 消融实验

ImageNet 上用预训练模型同时当生成器和 pruner，靠初始训练集大小 $n$ 调节生成器强弱，观察 KE/KH 的交叉点：

| 配置 | 现象 | 说明 |
|------|------|------|
| 小数据（160K，弱生成器） | Keep Easy 更优 | 印证 Theorem 2(B) |
| 大数据（1.2M，强生成器） | Keep Hard 更优，逼近真值标签性能 | 印证 Theorem 2(A) |
| 迭代自训练 + 全量 | 性能逐轮退化（模型崩溃） | 全量是错的 |
| 迭代自训练 + Keep Hard | 跨轮稳定、避免崩溃 | 策略性剪枝救场 |

### 关键发现
- **相变点真实存在**：ImageNet 上随 $n$ 增大，最优策略从 KE 平滑切换到 KH，与理论预测的 $\rho$ 驱动相变一致。
- **数据筛选能阻止模型崩溃**：在伪标签自训练循环里，全量训练越练越差，而每轮只留"又难又对"的样本能稳住性能——把数据筛选从"预处理 trick"提升为"稳定性工具"。
- **同一模型可同时是强/弱生成器**：取决于评估数据切片的难度，这解释了为何同一基准上 less/more 结论会打架。

## 亮点与洞察
- **把口水仗变成可计算的相变**：用三个余弦 $(\rho,\rho^*,\rho_g)$ 就把"less vs more"之争压成 Theorem 2 的一个条件判断，优雅且可证。
- **RMT 给出精确闭式误差**，而非渐近上下界——这让"保留多少、保留哪类"能被定量优化，而不是调参试错。
- **一个反直觉但可迁移的洞察**：弱模型（含自训练初期）该保守地"保留易样本"，强模型才该激进地"保留难样本"。这条原则可直接指导 RLHF/自蒸馏的数据配比与课程设计。
- **统一解释 LIMO/s1 悖论**：同数据集上 less/more 结论相反，根因是任务难度切片改变了 $\rho$ 高低——这是本文最"啊哈"的地方。

## 局限与展望
- **模型极简**：主结果建立在各向同性高维**线性二分类** + 平方损失上，真实 LLM 是非线性、序列、交叉熵，定量外推到 LLM 需谨慎（⚠️ 论文主要靠"现象一致"做桥接，非严格等价）。
- **只保留标签漂移**：正文设 $C_g=\Sigma=I_d$，把协变量漂移推到附录，实际数据两种漂移都存在。
- **oracle 质量被当成可控旋钮**：现实中 $\rho^*$（pruner 准不准）本身难估计，框架假设 oracle 强（$\rho^*\to1$）才有干净结论，弱 oracle 情形的策略更复杂。
- **改进思路**：把理论扩到多类/非线性特征、给出 $\rho/\rho^*$ 的可估计代理量、并在真实 LLM 上验证相变点的可预测性。

## 相关工作与启发
- **vs Sorscher et al. (2022)**：他们经验性地展示 margin 剪枝能"掰弯"神经缩放曲线；本文给出了**精确的解析条件**，说清何时掰得动、何时掰不动。
- **vs LIMO / s1（Ye et al., Muennighoff et al.）**：它们是经验方法（小而精数据胜大数据）；本文为其提供理论依据，并指出其成立的前提（强生成器 + 充足数据）和失效区（最难题切片）。
- **vs Feng et al. / Firdoussi et al. (2024/2025)**：他们的"只判标签对错"的 oracle 是本文式 (6) 在 $q\equiv1$ 时的特例；本文推广到"同时判难度 + 正确性"的更一般 oracle。
- **vs 模型崩溃理论（Dohmatob et al., Shumailov et al.）**：以往强调崩溃的不可逆与缩放定律失效；本文反过来证明**策略性筛选可主动避免崩溃**，给出了崩溃/稳定的相界。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"less vs more"之争首次化为可解析的相变条件，视角独到。
- 实验充分度: ⭐⭐⭐⭐ 合成 + ImageNet + LLM 文献三方印证，但缺真实 LLM 上的直接相变实验。
- 写作质量: ⭐⭐⭐⭐ 理论与直觉穿插，定理可读，但 LLM 桥接稍显类比。
- 价值: ⭐⭐⭐⭐⭐ 给数据筛选 / 自训练 / 防崩溃提供了可操作的原则性指导。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Admissible Heuristics for A*: Theory and Practice](learning_admissible_heuristics_for_a_theory_and_practice.md)
- [\[ICLR 2026\] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation](why_we_need_new_benchmarks_for_local_intrinsic_dimension_estimation.md)
- [\[ICLR 2026\] Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression](larger_datasets_can_be_repeated_more_a_theoretical_analysis_of_multi-epoch_scali.md)
- [\[ICML 2026\] Performative Learning Theory](../../ICML2026/learning_theory/performative_learning_theory.md)
- [\[ICLR 2026\] High-dimensional Analysis of Synthetic Data Selection](high-dimensional_analysis_of_synthetic_data_selection.md)

</div>

<!-- RELATED:END -->
