---
title: >-
  [论文解读] Explainable K-means Neural Networks for Multi-view Clustering
description: >-
  [ICLR 2026][可解释性][多视图聚类] 把多视图聚类拆成"线性聚类 → 非线性聚类 → 多视图融合"三层优化子问题，每层都由 K-means / 核 K-means 目标实现，组装成一个每层作用都可解释：的 EKNN 网络，从而在效果、效率、完整性、一致性四个维度上同时取得平衡。 领域现状：多视图聚类要把来自不同视…
tags:
  - "ICLR 2026"
  - "可解释性"
  - "多视图聚类"
  - "K-means"
  - "核 K-means"
  - "可解释神经网络"
  - "子空间学习"
  - "三层优化"
---

# Explainable K-means Neural Networks for Multi-view Clustering

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ljM1HTSH9c](https://openreview.net/forum?id=ljM1HTSH9c)  
**代码**: 待确认  
**领域**: 可解释性 / 多视图聚类 / 表示学习  
**关键词**: 多视图聚类, K-means, 核 K-means, 可解释神经网络, 子空间学习, 三层优化  

## 一句话总结
把多视图聚类拆成"线性聚类 → 非线性聚类 → 多视图融合"三层优化子问题，每层都由 K-means / 核 K-means 目标实现，组装成一个**每层作用都可解释**的 EKNN 网络，从而在效果、效率、完整性、一致性四个维度上同时取得平衡。

## 研究背景与动机
**领域现状**：多视图聚类要把来自不同视图（不同特征/模态）的数据点聚成簇。经典 K-means 计算成本低，但只能处理线性可分簇；核 K-means、谱聚类等非线性方法靠计算所有点对相似度来识别任意形状簇，效果好却空间/时间开销大，在大规模数据上吃不消。

**现有痛点**：① 多数方法只盯着 effectiveness（聚类质量），效率提升不明显；② 非线性方法用全部数据点表示一个簇，开销高；用"多中心"近似又对中心选择敏感；③ 多视图场景下，一致性（consistency）与完整性（completeness）是公认两大原则，但现有工作**无法同时兼顾效果、效率、一致性、完整性这四件事**。

**核心矛盾**：非线性可分簇要好效果就得算全部点对相似度（贵），要效率就得用线性近似（不准），二者在多视图设定下还要叠加跨视图一致性/完整性，四个目标互相牵制。

**本文目标**：在一个**可解释**框架内同时平衡效果、效率、完整性、一致性，做大规模数据上的非线性多视图聚类。

**核心 idea**：**【关键观察】** 复杂数据集里不同簇在**全局几何空间非线性可分，但在局部几何空间线性可分**——即"若干个小的线性可分簇拼成一个非线性可分簇"。**【三层分解】** 基于这一假设，把多视图聚类显式拆成三个子问题：1) 对原始数据点做线性聚类（降数据规模、保效率）；2) 在线性簇集合上做非线性聚类（保效果）；3) 跨视图整合各自的划分矩阵做多视图聚类（保完整性+一致性）。**【K-means 即卷积】** 每个子问题都用 K-means / 核 K-means 目标定义，K-means 在网络里扮演 CNN 里卷积的角色，由此构造出 EKNN，每一层物理含义已知，整体可解释。

## 方法详解

### 整体框架
EKNN 把多视图聚类建模成**三层优化问题**，对每个视图 $v$ 串起三层 K-means 风格的层：先用自表示得到子空间表示 $\Theta^v$，再依次做线性聚类层、非线性聚类层、多视图聚类层，最后用一个**共享划分矩阵 $H$** 把所有视图的结果绑在一起。三层联合目标用 K-means 式的迭代算法交替求解，每层的输入、变量、损失都有明确物理含义，所以网络可解释。

```mermaid
flowchart LR
    X["多视图数据 X^v"] --> SR["自表示子空间<br/>Θ^v (X^v=X^vΘ^v)"]
    SR --> L["线性聚类层 (K-means)<br/>f_L = ‖Θ^v − W^v V^v‖²"]
    L --> N["非线性聚类层 (核 K-means)<br/>f_N = Tr(K^v) − Tr(Û^vᵀ K^v Û^v)"]
    N --> M["多视图聚类层 (K-means)<br/>f_M = ‖W^v Û^v − H G^v‖²"]
    M --> H["共享划分矩阵 H<br/>(完整性 + 一致性)"]
    H --> R["最终聚类结果"]
```

### 关键设计

**1. 线性聚类层：用局部线性簇压缩数据，把"效率"做进框架。** 对每个视图先做自表示子空间学习，约束 $X^v = X^v\Theta^v$ 得到能揭示底层簇结构的子空间表示 $\Theta^v$，再在其上做 K-means：$f_{Lv} = \lVert \Theta^v - W^v V^v \rVert_F^2$，其中 $W^v \in \mathbb{R}^{n\times p}$ 是数据点到 $p$ 个线性簇的划分矩阵，$V^v$ 是簇中心矩阵。关键约束是**线性簇数 $p$ 远大于真实簇数 $k$**——这一步把 $n$ 个点压成 $p$ 个局部线性簇（"几个线性簇拼成一个非线性簇"），后续非线性聚类只需在 $p$ 个簇上算，而不是 $n$ 个点上算，效率由此而来。作者指出这一层其实是两子层：一层负责 $X^v = X^v\Theta^v$，一层负责重构损失。

**2. 非线性聚类层：在线性簇集合上做核 K-means，把"效果"做进框架。** 第二层对线性簇中心 $V^v$ 做核 K-means：$f_{Nv} = \lVert \Phi(V^v) - U^v Z^v \rVert_F^2 = \mathrm{Tr}(K^v) - \mathrm{Tr}\big((\hat U^v)^\top K^v \hat U^v\big)$，其中 $U^v \in \mathbb{R}^{p\times k}$ 把 $p$ 个线性簇划到 $k$ 个非线性簇，$K^v$ 是 $V^v$ 的高斯核矩阵，$\hat U^v = U^v (D^v)^{-1/2}$ 是归一化划分。因为核运算只作用在 $p$ 个线性簇而非全部 $n$ 个点上，既保留了识别任意形状簇的非线性能力，又把核 K-means 高昂的 $n\times n$ 相似度开销降到 $p\times p$，这是"既要效果又要效率"的关键折中。

**3. 多视图融合层 + 共享 $H$：用重构把"完整性"和"一致性"做进框架。** 第三层把每个视图的线性+非线性划分 $W^v\hat U^v$ 通过线性映射 $G^v$ 对齐到一个**所有视图共享的划分矩阵 $H \in \mathbb{R}^{n\times k}$**：$f_{Mv} = \lVert W^v\hat U^v - HG^v \rVert_F^2$。当 $W^v\hat U^v$ 固定时这一项就退化成经典 K-means。"强制各视图都重构出同一个 $H$"间接实现了**一致性**，而 $H$ 聚合了各视图信息则带来**完整性**。三层加权组合成总目标 $\min_H F = \sum_v \alpha f_{Lv} + \beta f_{Nv} + \gamma f_{Mv}$，并加上自表示重构项 $\eta\lVert X^v - X^v\Theta^v\rVert_F^2$ 与核范数正则 $\mu\lVert\Theta^v\rVert_*$（抑噪 + 保类内同质性），用 K-means 式迭代交替优化。

**4. 可解释性与子空间扩展：把"explainable"落到模型设计而非事后解释。** 不同于多数 XAI 用 post-hoc（可视化/文本/特征相关性）解释黑盒，EKNN 的可解释来自**模型设计本身**——表达为两层透明性：**模型可分解性**（输入即原始数据、学到的变量都有清晰物理含义、损失就是 K-means/核 K-means）与**算法透明性**（动态行为与误差曲面可数学推导）。K-means 与核 K-means 本身是 EKNN 的特例。作者进一步把 EKNN 扩展到**多视图子空间学习**：引入基于 $H$ 的共享子空间表示 $\Theta^C$（约束 $H = H\Theta^C$），用 ADMM 引入辅助变量 $R'$ 与乘子 $Y'$ 求解，最后对学到的 $\Theta^C$ 跑谱聚类得到结果——更复杂的映射带来更好的效果。

## 实验关键数据

### 主实验表格（ACC% ± STD%）
7 个数据集（COIL20/ORL/BBCSport/Reuters/Football/ANIMAL/NoisyMNIST），对比 COMIC、DMF、MVEC、BMVC、DiMSC、RMSL、SSSL-M。

| 数据集 | RMSL | SSSL-M | Ours (EKNN) | Ours (扩展) |
|---|---|---|---|---|
| COIL20 | 82.19 | 84.36 | 80.17 | **85.22** |
| ORL | 88.10 | 91.56 | 86.28 | **91.73** |
| BBCSport | **97.61** | 93.25 | 92.00 | 94.21 |
| Reuters | 54.04 | 59.46 | 42.17 | **60.55** |
| Football | **92.29** | 89.78 | 80.15 | 90.25 |
| ANIMAL | 73.19 | **77.49** | 70.76 | 79.00 |
| NoisyMNIST | 84.28 | 85.91 | 72.30 | **86.20** |

扩展版（EKNN for 多视图子空间学习）在 7 个数据集中有 5 个取得最高 ACC；F-score、RI 指标趋势一致（扩展版多数最优）。

### 消融 / 分析
- **线性簇数 $p$**：在 $\{20,\dots,200\}$ 扫描，ACC/NMI 随 $p$ 增大而上升、到一定值后趋平，但计算成本同步上升；最终取 $p=160$。
- **收敛性**：在 COIL20/ORL/BBCSport 上目标函数 ~50 次迭代内单调下降并收敛。
- **运行时间**：得益于"先线性聚类压缩数据"，EKNN 及扩展版耗时远低于 RMSL 和 SSSL-M。

### 关键发现
- **扩展版 > 基础版**：引入潜在表示 + 共享子空间表示后效果稳定更好，说明更复杂的映射带来更优结果。
- **三层分解有效**：把线性聚类作为其中一层能在一定程度上**减少最终聚类的过拟合**，间接提升性能。
- **效率来自结构**：先用线性层把 $n$ 个点压成 $p$ 个簇，再让昂贵的核运算只作用在 $p\times p$ 而非 $n\times n$ 上，是耗时远低于 RMSL/SSSL-M 的根本原因。
- **鲁棒性**：在多种数据类型与多种指标（ACC/F-score/RI/NMI）上都稳定，验证三层分解的普适性。

## 亮点与洞察
- **"局部线性、全局非线性"假设很漂亮**：把"非线性簇 = 若干线性簇的拼装"这一几何直觉直接转成可计算的三层结构，效率（线性层压缩）和效果（核层识别形状）在结构层面就解耦了。
- **可解释来自设计而非事后**：EKNN 不是给黑盒补解释，而是每层损失/变量都有物理含义，K-means 与核 K-means 是它的特例，这种"白盒"可解释在实际部署中更可信。
- **K-means ≈ 卷积的类比**：把 K-means 组件类比 CNN 卷积、组件连接类比网络连接，给"用经典聚类搭神经网络"提供了一个统一叙事。

## 局限与展望
- **基础版 EKNN 竞争力一般**：在多个数据集上基础版 ACC 明显低于 RMSL/SSSL-M（如 Reuters 仅 42.17 vs 59.46），真正打得过 SOTA 的是带子空间学习+谱聚类的**扩展版**，说明三层框架本身的增益部分依赖额外的子空间/谱聚类组件。
- **超参数多**：$\alpha,\beta,\gamma,\eta,\mu,\eta',\mu'$ 以及线性簇数 $p$ 都需调，框架虽可解释但调参负担不轻。
- **求解与复杂度细节在附录**：迭代算法与复杂度分析未在正文展开，离散约束（$w,u,h\in\{0,1\}$）下的优化质量值得进一步审视。
- **可扩展性**：虽强调效率，但实验数据集规模有限，超大规模/在线多视图场景的表现仍待验证。

## 相关工作与启发
- **K-means / 核 K-means / 谱聚类**：本文把这些经典目标当作"网络层"复用，延续了 LMSC、DiMSC、RMSL、SSSL-M 等子空间多视图聚类路线，但用三层分解重组。其中核 K-means 提供识别任意形状簇的非线性能力，谱聚类则在扩展版里对学到的共享子空间表示 $\Theta^C$ 做最终划分。
- **自表示子空间聚类**（Elhamifar & Vidal）：用自表示系数矩阵 $\Theta^v$ 揭示簇结构，约束 $X^v = X^v\Theta^v$，是线性层得以在"精炼后的数据点"而非原始噪声特征上做聚类的基础。
- **多视图一致性 / 完整性原则**：现有工作多通过"最大化视图间相关性"实现一致性，本文换成"强制各视图重构同一个共享 $H$"来间接达成一致性，并让 $H$ 聚合各视图信息以保完整性——是对两大经典原则的一种新实现方式。
- **可解释 AI（XAI）**：作者明确区分 post-hoc 解释（可视化/文本/特征相关性）与 design-time 透明性，主张后者（模型可分解性 + 算法透明性）更实用——对"可解释聚类/可解释表示学习"是一个清晰的方法论参考。

## 一句话评价
一个把"局部线性拼非线性"几何直觉工程化为三层 K-means 网络的多视图聚类框架：思路漂亮、可解释性扎实，真正的竞争力来自加了子空间学习与谱聚类的扩展版。

## 评分
- **新颖性**: ⭐⭐⭐⭐ "局部线性拼非线性"的三层优化分解 + "每层即一种 K-means、整体可解释"的网络化叙事是有新意的视角。
- **实验充分度**: ⭐⭐⭐ 覆盖 7 数据集 × 多指标 + $p$/收敛/耗时分析较完整，但基础版多处弱于对比方法，亮点主要靠扩展版撑。
- **写作质量**: ⭐⭐⭐ 框架与动机讲得清楚，但公式密集、求解细节挪到附录，部分表述与排版略粗糙。
- **价值**: ⭐⭐⭐⭐ 给多视图聚类提供了"可解释 + 效率/效果解耦"的结构化框架，对追求白盒聚类的应用场景有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bayesian Neural Networks for Functional ANOVA Model](bayesian_neural_networks_for_functional_anova_model.md)
- [\[ICLR 2026\] FAME: Formal Abstract Minimal Explanation for Neural Networks](fame_formal_abstract_minimal_explanation_for_neural_networks.md)
- [\[ICLR 2026\] Discovering Alternative Solutions Beyond the Simplicity Bias in Recurrent Neural Networks](discovering_alternative_solutions_beyond_the_simplicity_bias_in_recurrent_neural.md)
- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)

</div>

<!-- RELATED:END -->
