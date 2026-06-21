---
title: >-
  [论文解读] Conformal Prediction for Long-Tailed Classification
description: >-
  [ICLR2026][学习理论][长尾分类] 针对长尾分类中"小集合却漏掉稀有类 vs 覆盖好但集合巨大"的两难，本文提出两种保留边际覆盖保证的保形预测方法：一个新打分函数 PAS（按类先验校正 softmax，最优权衡集合大小与宏覆盖），一个新过程 INTERP-Q（线性插值 Classwise 与 Standard 的分位阈值，用一个参数滑动权衡），在 Pl@ntNet-300K（1081 类）和 iNaturalist-2018（8142 类）上显著改善了集合大小与类条件覆盖的折中。
tags:
  - "ICLR2026"
  - "学习理论"
  - "不确定性量化"
  - "保形预测"
  - "长尾分类"
  - "类条件覆盖"
  - "预测集"
  - "覆盖-集合大小权衡"
---

# Conformal Prediction for Long-Tailed Classification

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=8L83ZbFDjk](https://openreview.net/forum?id=8L83ZbFDjk)  
**代码**: https://github.com/tiffanyding/long-tail-conformal  
**领域**: 学习理论 / 不确定性量化 / 保形预测  
**关键词**: 保形预测, 长尾分类, 类条件覆盖, 预测集, 覆盖-集合大小权衡

## 一句话总结
针对长尾分类中"小集合却漏掉稀有类 vs 覆盖好但集合巨大"的两难，本文提出两种保留边际覆盖保证的保形预测方法：一个新打分函数 PAS（按类先验校正 softmax，最优权衡集合大小与宏覆盖），一个新过程 INTERP-Q（线性插值 Classwise 与 Standard 的分位阈值，用一个参数滑动权衡），在 Pl@ntNet-300K（1081 类）和 iNaturalist-2018（8142 类）上显著改善了集合大小与类条件覆盖的折中。

## 研究背景与动机
**领域现状**：保形预测（Conformal Prediction, CP）是一类无分布假设、给单点预测加"预测集"的不确定性量化方法——它不再吐一个可能出错的标签，而是给一个**大概率包含真标签的集合**，让人类（如植物识别 App 的用户）在候选里逐个核对。CP 的核心是用一个校准集为打分函数 $s(x,y)$ 选阈值 $q$，从而保证覆盖率。最基础的 **Standard CP** 只保证**边际覆盖** $P(Y\in C(X))\ge 1-\alpha$；面向类条件的 **Classwise CP**（Mondrian conformal 的实例）则保证每类都有 $P(Y\in C(X)\mid Y=y)\ge 1-\alpha$。

**现有痛点**：现实世界很多分类任务（植物识别、动物识别、疾病诊断）是**极端长尾**的——常见类有上千张图，稀有类只有寥寥几张。更棘手的是，我们往往**更在乎稀有类**（濒危物种、罕见恶性肿瘤），但绝大多数稀有类样本都被拿去训练了，留给校准的 holdout 样本极少甚至为零。在这种设定下，现有 CP 方法逼着实践者做二选一：Standard CP 给出**很小但稀有类覆盖很差**的集合（Pl@ntNet 上 1081 类里有 421 类覆盖低于 50%）；Classwise CP 能让每类覆盖都好但集合**巨大到没法用**（平均 780 个候选）；Clustered CP 对无法归簇的稀有类则退化回 Standard CP。

**核心矛盾**：长尾设定下，**小集合大小**与**类条件覆盖**之间存在本质的 trade-off，且稀有类校准样本稀缺让"逐类保证覆盖"几乎不可行。

**本文目标**：在**保住边际覆盖保证**的前提下，给出一条比 Standard / Classwise 更优、且可调的"集合大小 ↔ 类条件覆盖"折中曲线。

**切入角度**：作者从两个角度切入。一是**放松目标**——不强求每类覆盖，而是优化"平均类条件覆盖"（macro-coverage），并从理论上推导出最优权衡集合大小与宏覆盖的预测集形态。二是**从 Classwise 往回退**——把 Classwise 的强目标用一个插值参数逐步软化到 Standard，让用户自己选落在折中曲线的哪一点。

**核心 idea**：用先验校正后的 $\hat p(y\mid x)/\hat p(y)$ 替代 $\hat p(y\mid x)$ 来打分（PAS），即可在固定集合大小下最优化宏覆盖；同时用阈值线性插值（INTERP-Q）把 Classwise 与 Standard 连成一条可调曲线。

## 方法详解

### 整体框架
所有方法都套同一个 CP 元算法（Algorithm 1）：把预测集写成**阈值集** $C(X;q)=\{y\in\mathcal{Y}: s(X,y)\le q_y\}$，即对每个类 $y$ 用一个阈值 $q_y$ 卡打分。不同方法的区别只在两处——**打分函数 $s$** 和**阈值函数 $\hat q$**。

- Standard CP：任意打分 + 全类统一阈值 $\hat q_{\text{STAND}}=(\hat q,\dots,\hat q)$，$\hat q$ 取所有校准点分数的 $(1-\alpha)$ 经验分位数，给出边际覆盖 $1-\alpha$。
- Classwise CP：任意打分 + 逐类阈值 $\hat q_y^{\text{CW}}$（只用类 $y$ 的 $n_y$ 个校准点算分位数），给出逐类覆盖 $1-\alpha$。
- 本文 **APPROACH I**：换打分函数。保持 Standard 的阈值方式，但把打分从 softmax 换成 **PAS / WPAS**，从而在不损失边际覆盖的前提下优化（加权）宏覆盖。
- 本文 **APPROACH II**：换阈值函数。打分仍用 softmax，但阈值取 Standard 与 Classwise 的**线性插值** $\hat q^{\text{IQ}}$，得到一族可调过程 **INTERP-Q**。

基线打分用 softmax 分数 $s_{\text{softmax}}(x,y)=1-\hat p(y\mid x)$（即 LAC 分数），分数越小表示 $(x,y)$ 越"合群"。

### 关键设计

**1. PAS 打分：用类先验校正 softmax，把"最优集合"从边际覆盖换到宏覆盖**

要解决的痛点是：Standard CP 优化的是"固定集合大小下最大化边际覆盖"，而边际覆盖是**按类先验加权**的类条件覆盖之和 $\text{MarginalCov}(C)=\sum_y p(y)\,\text{CondCov}(C,y)$——这等于把权重压在高频类上，稀有类被系统性忽略。本文转而盯住**宏覆盖**（macro-coverage），即**不加权**的平均类条件覆盖 $\text{MacroCov}(C)=\frac{1}{|\mathcal{Y}|}\sum_y \text{CondCov}(C,y)$。

作者先解一个总体最优化问题：在宏覆盖约束下最小化期望集合大小（及其对偶——集合大小约束下最大化宏覆盖）。Proposition 1 给出最优集合形态是对**密度比**做阈值：

$$C^*(x)=\{y\in\mathcal{Y}: p(y\mid x)/p(y)\ge t\}.$$

关键洞察是：经典的"最小集合 + 边际/类条件覆盖"最优解是对 $p(y\mid x)$ 做阈值（Neyman-Pearson），而**宏覆盖**的最优解是对 $p(y\mid x)/p(y)$ 做阈值——即把后验除以类先验。直觉上，稀有类 $p(y)$ 小，除法把它的分数"抬上来"，让稀有类更容易进集合。实践中真密度未知，就用估计值 $\hat p(y\mid x)$（分类器 softmax）和 $\hat p(y)$（训练标签经验分布）替代，定义打分

$$s_{\text{PAS}}(x,y)=-\hat p(y\mid x)/\hat p(y),$$

PAS 即 prevalence-adjusted softmax（先验校正 softmax）。再把它套进 Standard CP 的阈值 $\hat q$，预测集 $\hat C(x)=\{y:\hat p(y\mid x)/\hat p(y)\ge t\}$ 就**继承了 Standard 的边际覆盖保证**，同时（近似）最优地权衡集合大小与宏覆盖。要强调：PAS 是去**更好地处理这个 trade-off**，并不直接给宏覆盖一个保证。

**2. WPAS：把"等权宏覆盖"推广到"加权宏覆盖"，定向保护用户关心的类**

宏覆盖默认各类等权，但有时我们更在乎某些类（如濒危物种、恶性肿瘤）。给定一组和为 1 的用户类权重 $\omega(y)$，定义加权宏覆盖 $\text{MacroCov}_\omega(C)=\sum_y \omega(y)\,\text{CondCov}(C,y)$——取 $\omega(y)=|\mathcal{Y}|^{-1}$ 退回宏覆盖，取 $\omega(y)=p(y)$ 退回边际覆盖。Proposition 2 表明此时最优集合形态变成对 $\omega(y)\cdot p(y\mid x)/p(y)$ 做阈值，于是打分相应改为

$$s_{\text{WPAS}}(x,y)=-\omega(y)\,\hat p(y\mid x)/\hat p(y),$$

同样套 Standard 阈值即可。在濒危物种实验里，作者把至危类权重设为普通类的 $\lambda\ge 1$ 倍，$\lambda$ 越大、至危类的类条件覆盖提升越多，代价只是集合大小**温和增大**、对非至危类覆盖几乎无影响。

**3. INTERP-Q：线性插值 Classwise 与 Standard 的分位阈值，用一个参数滑动整条折中曲线**

APPROACH II 不动打分而是动阈值。INTERP-Q（"interpolated quantile"）把每类阈值取成 Standard 全局阈值 $\hat q$ 与 Classwise 逐类阈值 $\hat q_y^{\text{CW}}$ 的加权平均：

$$\hat q_y^{\text{IQ}}=\tau\,\hat q_y^{\text{CW}}+(1-\tau)\,\hat q,\quad \tau\in[0,1].$$

对于校准样本太少、$\hat q_y^{\text{CW}}=\infty$ 的类，先把它替换成 1（softmax 分数的最大可能值）再插值。$\tau=0$ 即 Standard，$\tau=1$ 即 Classwise，中间值则连续滑动。Proposition 3 给出它的边际覆盖下界为 $1-2\alpha$，且理论上几乎紧（能构造出覆盖 $1-2\alpha+\alpha^2$ 的病态例子）；但实测中真实数据不出现那种"各类分数分布差异极大的离散病态"，所以 INTERP-Q 的经验覆盖接近 $1-\alpha$。

它的实用价值在于一个有趣的**非线性**现象：阈值线性插值并不导致集合大小线性插值——$\tau=1$ 时集合巨大（Pl@ntNet 上 780、iNaturalist 上 7430），但只把 $\tau$ 降到 0.99，平均集合大小就骤降到 7.6 和 55.8。原因是稀有类的 softmax 分数高度偏向 1（分类器几乎给它们零概率），分位数对 $\tau$ 极其敏感。这让用户能用一个旋钮在保住边际覆盖的同时，灵活选自己想要的"小集合 ↔ 类条件覆盖"位置。

### 一个完整示例
以 Pl@ntNet-300K 上目标 90% 边际覆盖为例，走一遍各方法的表现：**Standard** 平均集合只有 1.57 个候选，但 1081 个植物物种里有 **421 个**覆盖率低于 50%（稀有类被系统性漏掉）；**Classwise** 把"覆盖低于 50% 的类"压到 **0 个**，但平均集合大小爆炸到 **780**（用户根本没法核对）。本文方法给出中间地带：**Standard + PAS** 平均集合只比 Standard 略大（**2.57**），却把"覆盖低于 50% 的类"从 421 **腰斩到 180**；**INTERP-Q** 表现类似，额外好处是这个折中可以通过调 $\tau$ 自由滑动。这个案例直观说明了两种方法如何把原本二选一的极端，变成一条可操作的折中曲线。

## 实验关键数据

### 主实验
数据集：Pl@ntNet-300K（1081 类）、iNaturalist-2018（8142 类），均为长尾。基模型 ResNet-50（交叉熵训练，附录另有 focal loss 结果）。校准集取 val 的 70%。评估指标见下表。

| 指标 | 定义 | 方向 |
|--------|------|------|
| FracBelow50% | 覆盖率 $\le 50\%$ 的类占比 | 越小越好 |
| UnderCovGap | 平均欠覆盖缺口 $\frac{1}{\|\mathcal{Y}\|}\sum_y \max(1-\alpha-\hat c_y,0)$ | 越小越好 |
| MacroCov | 平均类条件覆盖 $\frac{1}{\|\mathcal{Y}\|}\sum_y \hat c_y$ | 越大越好 |
| MarginalCov | 边际覆盖（须 $\ge 1-\alpha$） | 达标即可 |
| Average set size | 平均集合大小 | 越小越好 |

Pl@ntNet-300K（目标 90% 边际覆盖）核心对比：

| 方法 | 平均集合大小 | 覆盖<50% 的类数（/1081） | 说明 |
|--------|------|----------|------|
| Standard | 1.57 | 421 | 集合最小但稀有类大面积失守 |
| Classwise | 780 | 0 | 逐类覆盖完美但集合不可用 |
| **Standard + PAS** | **2.57** | **180** | 集合仅略增，覆盖<50% 的类腰斩 |
| INTERP-Q | 随 $\tau$ 可调 | 随 $\tau$ 可调 | 用一个参数滑动折中 |

### 消融实验

| 配置 / 现象 | 关键指标 | 说明 |
|------|---------|------|
| INTERP-Q, $\tau=1$ | 集合 780 / 7430 | 等价 Classwise，集合巨大 |
| INTERP-Q, $\tau=0.99$ | 集合 7.6 / 55.8 | 阈值微调，集合骤降（非线性） |
| Standard + WPAS（$\lambda\uparrow$） | 至危类覆盖↑ | 上调至危类权重，覆盖明显提升，集合仅温和增大 |
| Standard + PAS（Pareto） | 各 $\alpha$ 下均最优 | 任意边际覆盖水平下集合-覆盖不被任何方法同时超越 |

### 关键发现
- **直接优化目标 trade-off 比绕道边际覆盖更有效**：靠调 Standard 的 $\alpha$ 来间接逼近类条件覆盖，无法最优导航"集合大小 ↔ 类条件覆盖"，而显式优化宏/类条件覆盖的本文方法一致更优。
- **Classwise 应尽量避免**：用本文方法能以小得多的集合达到相当的类条件 / 宏覆盖。
- **Standard + PAS 是 Pareto 最优**：任意边际覆盖水平下，没有方法能同时取得更小集合和更好的类条件/宏覆盖，且实现极简，是实践者的好起点。
- **INTERP-Q 的非线性最有意思**：阈值从 $\tau=1$ 微降到 0.99，集合大小骤降两个数量级，源于稀有类 softmax 分数高度偏向 1。

## 亮点与洞察
- **一个除法换掉整套目标**：把后验除以类先验（$\hat p(y\mid x)/\hat p(y)$）就把 CP 的最优目标从边际覆盖切换到宏覆盖——这个改动有 Neyman-Pearson 式的理论最优性背书，却只是给打分函数加一行，实现成本极低。
- **保证与优化解耦**：本文巧妙之处在于"打分负责优化、阈值负责保证"。PAS 只改打分、复用 Standard 阈值，于是**白嫖**了 Standard 的边际覆盖保证，同时近似最优化一个全新目标——这套"换打分不换阈值"的范式可迁移到任何想优化新覆盖目标的 CP 任务。
- **阈值插值是个简单却好用的旋钮**：INTERP-Q 不需要重新设计打分或重训模型，只在两套现成分位阈值间线性插值，就得到一族连续可调的方法，且非线性特性让"几乎 Classwise 的覆盖 + 接近 Standard 的集合"成为可能。
- **WPAS 的可定向性**：通过类权重 $\omega(y)$ 把"我更在乎濒危物种/罕见癌症"这种领域偏好直接编码进打分，是把业务优先级注入不确定性量化的干净接口。

## 局限与展望
- **PAS 不直接给宏覆盖保证**：它只保证边际覆盖，宏覆盖只是被"近似优化"，依赖 $\hat p(y\mid x)$ 和 $\hat p(y)$ 估计质量；分类器或先验估计偏差会影响实际宏覆盖效果。
- **INTERP-Q 理论保证偏弱**：最坏情况只有 $1-2\alpha$ 边际覆盖（理论几乎紧），虽然实测接近 $1-\alpha$，但这依赖"真实数据不出现病态分数分布"的经验假设，缺乏对何时会失效的刻画。
- **评估依赖截断数据集**：长尾数据集测试集本身也长尾，稀有类测试样本太少无法可靠估类条件覆盖，作者只能造截断版（每类 100 测试样本）来评估，这与真实部署分布存在一定 gap。
- **可改进方向**：把两种方法结合（附录已初探）、给 PAS 设计直接的宏覆盖保证、研究先验/后验估计误差对覆盖的理论影响，都是自然延伸。

## 相关工作与启发
- **vs Standard CP**: Standard 优化"集合大小 + 边际覆盖"，对 $p(y\mid x)$ 阈值，集合最小但稀有类覆盖差；本文 PAS 改优化"集合大小 + 宏覆盖"，对 $p(y\mid x)/p(y)$ 阈值，稀有类覆盖大幅改善，集合仅略增。
- **vs Classwise CP（Mondrian conformal）**: Classwise 逐类保证覆盖但长尾下集合巨大、稀有类校准样本不足时退化；INTERP-Q 把它与 Standard 用阈值插值软化，以远小的集合换取相当的类条件覆盖。
- **vs Clustered CP (Ding et al., 2023)**: Clustered 通过给相似分数分布的类分簇来逼近类条件覆盖，但对无法归簇的稀有类退回 Standard；本文方法不依赖分簇，对极稀有类同样有效。
- **vs APS / RAPS / SAPS 等**: 这些打分主要针对 X-条件覆盖、且集合更大，不在本文范围；本文聚焦 softmax(LAC) 分数下的长尾类条件覆盖这一未被充分研究的目标。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把宏覆盖引入 CP 并推导密度比最优集合 + 阈值插值过程，角度新且理论扎实
- 实验充分度: ⭐⭐⭐⭐ 两个真实大规模长尾数据集 + 多基线 + 濒危物种案例，但主要靠截断数据集评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰、理论与方法层层递进、案例直观
- 价值: ⭐⭐⭐⭐ 长尾不确定性量化的实用工具，PAS 实现极简、易落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)
- [\[ICLR 2026\] Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting](conformal_prediction_with_corrupted_labels_uncertain_imputation_and_robust_re-we.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICML 2026\] Enhancing Conformal Prediction via Class Similarity](../../ICML2026/learning_theory/enhancing_conformal_prediction_via_class_similarity.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)

</div>

<!-- RELATED:END -->
