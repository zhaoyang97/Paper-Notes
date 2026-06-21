---
title: >-
  [论文解读] On Universality of Deep Equivariant Networks
description: >-
  [ICLR 2026][学习理论][不变网络] 这篇论文为深度不变 / 等变网络建立了"在分离约束下的万能逼近"定理，指出深度和读出层（readout）是达成万能性的决定性机制，并为等变情形引入了比标准分离更细的"逐分量分离"（entry-wise separability）判据，统一并推广了此前局限于浅层或特定架构的结论。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "等变网络"
  - "表达能力"
  - "不变网络"
  - "万能逼近"
  - "分离能力"
  - "逐分量分离"
---

# On Universality of Deep Equivariant Networks

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q2D1PI6zY1](https://openreview.net/forum?id=Q2D1PI6zY1)  
**代码**: 无（纯理论论文，无实验代码）  
**领域**: 学习理论 / 等变网络 / 表达能力  
**关键词**: 等变网络、不变网络、万能逼近、分离能力、逐分量分离

## 一句话总结
这篇论文为深度不变 / 等变网络建立了"在分离约束下的万能逼近"定理，指出深度和读出层（readout）是达成万能性的决定性机制，并为等变情形引入了比标准分离更细的"逐分量分离"（entry-wise separability）判据，统一并推广了此前局限于浅层或特定架构的结论。

## 研究背景与动机

**领域现状**：对称性已成为深度学习的核心组织原则，等变网络（CNN、GNN、PointNet、SE(3)-Transformer 等）通过保证"输入变换 → 输出同步变换"来编码对称性归纳偏置，在分子、点云、图、流形等任务上广泛成功。但人们一直担心：这种归纳偏置除了对称性之外，会不会还偷偷施加了**额外的、不想要的**表达力限制？

**现有痛点**：刻画等变网络表达能力的两条主流路线各有硬伤。第一条直接研究**万能性**（universality，即能否逼近所有与对称性兼容的目标函数），但 Ravanbakhsh、Maron 等人的结果要么要求隐藏层用 regular 表示、要么用高阶张量表示，导致中间表示维度随群规模爆炸，**完全不实用**。第二条研究**分离能力**（separation power，即能区分多少对输入），在图学习里通过 Weisfeiler–Leman 测试被研究得很透，但它只是逼近的**必要条件**。

**核心矛盾**：分离对逼近是必要的，却不一定充分。Pacini et al. (2025b) 给出了反例：两个浅层不变架构**分离能力相同、逼近能力却不同**，说明"分离能力相等"并不能推出"能逼近的函数类相等"。这与经典神经网络理论形成鲜明对比——在经典设定下，深度只影响参数效率、不改变可逼近函数类。

**本文目标**：在不变与等变两种情形下，搞清楚**深度和读出层**到底如何修复"分离能力 = 逼近能力"这件事，并给出一个统一框架，超越此前一篇一个架构、各自为政的结论。

**切入角度**：Zaheer、Qi、Segol & Lipman 等人观察到，给受限架构**加全连接读出层或增加深度**就能把它们变成"分离约束下万能"的模型。作者顺着这条线索深挖：深度和读出层是不是分离约束万能性的通用机制？

**核心 idea**：把"万能性"重新表述为**在分离关系 $\rho$ 约束下逼近整个 $C_\rho$ 函数类**；对不变网络证明"加一层全连接读出即可万能"；对等变网络发现标准分离太粗，于是引入更锐利的**逐分量分离**，并证明"深度足够"或"特定读出层"两条路都能达成万能。

## 方法详解

### 整体框架

整篇论文是一套关于等变网络表达能力的逼近理论，没有训练流程或网络管线，因此不画 pipeline 图，而是按"先不变、后等变"的逻辑链层层推进。

作者先把所有研究对象统一在**置换表示 + 层空间（layer space）**的语言下。一个有限群 $G$ 作用在有限集 $X$ 上，得到置换表示 $\mathbb{R}^X$；一个**层空间** $M \subseteq \mathrm{Aff}_G(V, \mathbb{R}^X)$ 是一族满足等变约束的仿射映射（线性层、不变层 $I$、卷积层 $C$、PointNet 层 $P$ 都是它的特例）。把多个层空间用逐点激活 $\tilde\sigma$ 串起来就得到**神经空间**；让中间宽度自由变化、再取一致收敛闭包，就得到一个架构的**万能类** $U_\sigma(M_1,\dots,M_d)$。

接着定义评判标准。一族函数 $U$ 诱导一个等价关系 $\rho(U)$（分不开的输入对），若 $U$ 恰好能逼近所有"尊重 $\rho$"的连续函数 $C_\rho$，就称它**分离约束万能**。论文的三块核心内容就是：(1) 不变网络靠全连接读出达成 $C_\rho$ 万能；(2) 指出等变情形下 $C_\rho$ 这个目标本身刻画不准，需要换成逐分量版本 $C_{\boldsymbol\rho}$；(3) 用两种不同手段（深度 / 卷积读出）把等变网络打到 $C_{\boldsymbol\rho}$ 万能。这三块对应下面三个关键设计，论文叙述顺序与之一致。

### 关键设计

**1. 分离约束万能性框架 + 不变网络的全连接读出定理：把"分离≠逼近"的病态用一层读出修好**

针对的痛点是 Pacini et al. (2025b) 揭示的怪现象：分离能力相同的不变架构，逼近能力却可能严格不等（如 $U_\sigma(C,I) \subsetneq U_\sigma(P,I) \subsetneq C_{S_n}(\mathbb{R}^n,\mathbb{R})$，三者分离能力却完全一样）。作者先把"万能"形式化为分离约束下的相等：一族函数 $U$ 的分离关系为

$$\rho(U) = \{(\alpha,\beta)\in V\times V \mid f(\alpha)=f(\beta)\ \forall f\in U\},$$

而目标函数类是所有尊重该关系的连续函数 $C_\rho(V,W)=\{f\in C(V,W)\mid f(\alpha)=f(\beta)\text{ whenever }(\alpha,\beta)\in\rho\}$。**定理 1** 证明：对任意不变层 $I$ 收尾的网络，只要再接一层普通全连接读出 $L$（即 $L=\mathrm{Aff}(\mathbb{R},\mathbb{R})$ 的并联），就有

$$U_\sigma(M_1,\dots,M_d,I,L) = C_\rho(V),\qquad \rho=\rho\big(U_\sigma(M_1,\dots,M_d,I)\big).$$

证明的巧思在于：全连接读出 $L$ **不改变**网络的分离关系 $\rho$（前面层已把分离能力定死），却能把任意一组分量函数 $f_1,\dots,f_h$ 自由地非线性组合，从而补齐"分离够、但逼近不够"的缺口。这一步直接吸收并推广了 Joshi et al.、Chen et al. 的不变万能性结论。

**2. 逐分量分离（entry-wise separability）：标准分离对等变情形太粗，必须逐个输出坐标看**

把视线转到等变情形时，作者发现一个反例（**Example 3**）：取宽度为 1 的卷积层空间 $C$，深度 $d\ge 2$ 的纯卷积网络满足

$$U_\sigma^d(C) = \{(x_1,\dots,x_n)\mapsto(f(x_1),\dots,f(x_n))\mid f\in C(\mathbb{R})\}\ \subsetneq\ C_{S_n}(\mathbb{R}^n,\mathbb{R}^n).$$

由于恒等映射在 $U_\sigma^d(C)$ 里，它的分离关系 $\rho$ 是**平凡的**（什么都能分开），于是标准目标 $C_{S_n,\rho}=C_{S_n}(\mathbb{R}^n,\mathbb{R}^n)$；但上式说明无论深度多大都达不到这个目标——**标准分离约束万能在等变情形根本无法成立**。问题出在：等变函数的"分离能力"是把所有输出坐标揉在一起看的，而真正限制它的是**每个输出坐标各自能分开什么**。

为此作者引入**逐分量分离**（Definition 6）。设 $\pi_x:\mathbb{R}^X\to\mathbb{R}$ 是投到第 $x$ 个坐标的投影，对神经空间 $N$ 定义每个坐标的分离关系

$$\rho_x(N) = \{(\alpha,\beta)\in V\times V \mid \pi_x f(\alpha)=\pi_x f(\beta)\ \forall f\in N\},$$

并把它们打包成一个**关系族** $\boldsymbol\rho(N)=(\rho_{x_1}(N),\dots,\rho_{x_n}(N))$，对应的目标函数类 $C_{\boldsymbol\rho}$ 要求每个坐标各自尊重自己的 $\rho_x$。由于 $\rho(N)=\bigcap_x \rho_x(N)$，逐分量分离**蕴含**标准分离，且可能严格更强；在不变情形（$G$ 平凡作用于 $\mathbb{R}$）或所有 $\rho_x$ 相等时它退化回标准分离。Proposition 2 进一步证明 Example 3 的卷积网络恰好满足 $U_\sigma^d(C)=C_{\boldsymbol\rho}(\mathbb{R}^n,\mathbb{R}^n)$——新判据精确刻画了原来刻画不了的类。

**3. 两条达成等变万能的路径：深度稳定（定理 2）与卷积读出（定理 3）**

有了逐分量分离这把尺子，作者给出两个殊途同归的等变万能定理。**定理 2（深度路径）**：设输出层空间 $M$（含恒等映射）反复堆叠，当深度达到使**逐分量分离稳定**的阈值后——即

$$\rho := \rho\big(U_\sigma(M_1,\dots,M_f,\underbrace{M,\dots,M}_{d})\big)=\rho\big(U_\sigma(M_1,\dots,M_f,\underbrace{M,\dots,M}_{d+1})\big),$$

则再多堆一层就达成逐分量分离约束万能 $U_\sigma(\dots,\underbrace{M,\dots,M}_{d+1})=C_{\boldsymbol\rho}(V_0,\mathbb{R}^X)$。配合 Pacini et al. (2025a) "分离在有限深度后稳定"的结论，Corollary 1 保证存在阈值 $D$，深度超过它后万能类**饱和**，再加深也不变——这从理论上排除了"无限加深无限变强"的可能，给出了实践上"在有限深度就触顶"的保证。

**定理 3（读出路径）**：若把输出层换成宽度为 1 的卷积滤波 $C$，则**无需任何深度条件**就有 $U_\sigma(M_1,\dots,M_f,C)=C_{\boldsymbol\rho}(V)$。关键区别在于：加 $C$ 层**不改变**模型的逐分量分离能力（它扮演不变情形里全连接读出的"等变替身"），而堆 $M$ 层则可能**提升**分离能力——定理 2 正是为后一种效应付出"等到稳定"的代价。当 $C$ 退化到一维时 $C=L$、$M=I$，定理 3 恰好特化为不变情形的定理 1，两套结果在此合龙。Remark 1 据此直接复现了 Segol & Lipman (2020) 的 PointNet 万能性 $U_\sigma(C,P,C)=U_\sigma(P,P,P)=C_{S_n}(\mathbb{R}^n,\mathbb{R}^n)$，并指出定理 2 的深度阈值只是充分而非必要条件。

### 损失函数 / 训练策略
本文为纯理论工作，不涉及损失函数、优化或训练；所有结论以定理 / 命题 / 推论形式给出，完整证明在正文与附录中（作者在可复现性声明中强调无实验、无数据集，所有断言可通过检验证明来验证）。

## 实验关键数据

本论文无实验。下面用两张表归纳其理论结果，替代常规实验表格。

### 主要定理一览

| 定理 / 结论 | 设定 | 达成万能的机制 | 结论形式 |
|------|------|------|------|
| 定理 1 | 不变网络 | 加全连接读出 $L$ | $U_\sigma(M_1,\dots,M_d,I,L)=C_\rho(V)$ |
| Example 3 / Prop. 2 | 等变（纯卷积） | —（标准分离失效） | $U_\sigma^d(C)=C_{\boldsymbol\rho}\subsetneq C_{S_n}$ |
| 定理 2 + Cor. 1 | 等变网络 | 深度达到分离稳定阈值 | $U_\sigma(\dots,M^{d+1})=C_{\boldsymbol\rho}$，有限深度饱和 |
| 定理 3 | 等变网络 | 宽度 1 卷积读出 $C$ | $U_\sigma(M_1,\dots,M_f,C)=C_{\boldsymbol\rho}(V)$ |

### 对此前结果的统一与推广

| 此前工作 | 原结论 | 本文如何统一 |
|------|------|------|
| Pinkus (1999) | 经典万能逼近 | $U_\sigma(L,L)=C(\mathbb{R},\mathbb{R})$，作为 $\rho$ 平凡的特例 |
| Segol & Lipman (2020) | 3 层 PointNet 万能 | 由定理 3 + Remark 1 直接复现 |
| Joshi et al. (2023) | 区分 $G$-轨道后接头部即不变万能 | 被定理 1 涵盖 |
| Chen / Geerts / Maron | GNN 万能 ↔ Weisfeiler–Leman 分离 | 纳入分离约束万能框架 |
| Pacini et al. (2025b) | 分离相同但逼近不同 | 定理 1 / 逐分量分离给出修复与精确刻画 |

### 关键发现
- **深度和读出层是万能性的决定性机制**：在分离能力不变的前提下，二者改变了可逼近函数类——这与经典网络中"深度只省参数、不扩函数类"形成根本对比。
- **等变需要更细的分离判据**：标准分离把所有输出坐标揉在一起，对等变情形会严重高估能力；逐分量分离逐坐标审视，才精确刻画万能类。
- **两条路径不等价**：堆深度（定理 2）可能提升分离、需等稳定；卷积读出（定理 3）不改分离、可直接达成——后者还揭示定理 2 的深度阈值只是充分条件，常常偏保守。

## 亮点与洞察
- **把"万能性"从"逼近一切"重写成"逼近分离约束下的 $C_\rho$"**，让原本只能逐架构证明的结论有了统一语言，这是全文最关键的视角转换。
- **逐分量分离是一个可迁移的概念工具**：凡是输出是结构化对象（每个坐标受不同稳定子群约束）的等变模型，都可以用"逐坐标投影 + 各自分离关系"来分析表达能力，而不必诉诸更重的微分算子刻画（Remark 2）。
- **"读出层 = 不变情形全连接读出的等变替身"** 这个类比很漂亮：它解释了为什么一维退化时定理 3 自动变成定理 1，把不变与等变两套理论缝在了一起。
- **有限深度饱和**（Corollary 1）对实践有指导意义：它告诉工程师"加深到某个阈值后表达力触顶"，避免盲目堆深。

## 局限与展望
- **仅限逐点激活 + 置换表示**：推广到其他类型表示或更一般的非线性可能需要全新方法（作者明确承认）。
- **结论是渐近的、无定量速率**：没有给出逼近速率或样本复杂度界，而这些对理解实际表达力很关键。
- **未触及可训练性**：深度足以保证万能，但这样的深网何时能被高效训练仍是开放问题——理论上的"存在"不等于优化上的"可达"。
- 个人观察：定理 2 的深度阈值依赖"分离稳定"这一难以一般性判定的条件，Remark 1 也指出该阈值充分不必要，实际中如何估计稳定深度仍缺乏可操作的判据。

## 相关工作与启发
- **vs Ravanbakhsh (2020) / Maron et al. (2019b)**：他们用 regular 表示 / 高阶张量证明等变万能，代价是隐藏维度随群规模爆炸、不实用；本文用置换表示 + 分离约束，刻画的是**实际架构**（CNN/GNN/PointNet）的万能类，更贴近落地。
- **vs Joshi et al. (2023) / Chen et al. (2019)**：他们建立了"分离到 WL / 轨道即不变万能"的初步结果，但局限于不变情形；本文的定理 1 涵盖之，并把战线推进到等变情形。
- **vs Pacini et al. (2025b)**：他们发现"分离相同、逼近不同"的病态并用微分算子刻画；本文对等变情形用逐分量分离给出**不依赖微分算子**的精确刻画，并用读出层 / 深度直接修复该病态。
- **vs 经典深度理论（Telgarsky 2016, Yarotsky 2017/2018）**：经典结论里深度只改善参数效率、不扩函数类；本文揭示在等变 / 分离约束设定下，深度反而能扩张可逼近函数类，是一个值得注意的反差。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 逐分量分离是真正新的判据，统一视角填补了等变万能性理论的空白
- 实验充分度: ⭐⭐⭐ 纯理论论文无实验，但定理证明完整、自洽，按理论标准是充分的
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰、与既有结果对照到位，但概念密度高、对非理论读者门槛较陡
- 价值: ⭐⭐⭐⭐ 为等变网络表达能力分析提供了通用框架和可迁移工具，长期价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Quasi-Equivariant Metanetworks](quasi-equivariant_metanetworks.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] Variational Deep Learning via Implicit Regularization](variational_deep_learning_via_implicit_regularization.md)

</div>

<!-- RELATED:END -->
