---
title: >-
  [论文解读] Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting
description: >-
  [ICLR2026][学习理论][损坏标签] 针对训练标签被噪声/缺失损坏、且关键特征在测试时不可得（特权信息）的场景，本文先证明了已有的特权保形预测（PCP）在权重估计不准时仍可能有效的精确条件，再提出一种不依赖权重、改靠"带不确定性地回填标签"的新方法 UI，最后把 Naive CP / PCP / UI 三者取并集得到只要有一个假设成立就有效的三重稳健（TriplyRobust）校准方案。
tags:
  - "ICLR2026"
  - "学习理论"
  - "保形预测"
  - "不确定性量化"
  - "损坏标签"
  - "特权信息"
  - "加权保形"
  - "不确定填补"
---

# Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=ztEKLEUNKS](https://openreview.net/forum?id=ztEKLEUNKS)  
**代码**: https://github.com/Shai128/ui  
**领域**: 学习理论 / 保形预测 / 不确定性量化  
**关键词**: 保形预测、损坏标签、特权信息、加权保形、不确定填补

## 一句话总结
针对训练标签被噪声/缺失损坏、且关键特征在测试时不可得（特权信息）的场景，本文先证明了已有的特权保形预测（PCP）在权重估计不准时仍可能有效的精确条件，再提出一种不依赖权重、改靠"带不确定性地回填标签"的新方法 UI，最后把 Naive CP / PCP / UI 三者取并集得到只要有一个假设成立就有效的三重稳健（TriplyRobust）校准方案。

## 研究背景与动机
**领域现状**：保形预测（Conformal Prediction, CP）是给任意预测模型套一层"统计保证"的通用工具——它用一组留出校准样本算非一致性分数，取分位数当阈值，从而给测试点构造一个能以用户指定概率 $1-\alpha$（如 90%）覆盖真实标签的预测集合。这套保证的前提是校准数据和测试数据**可交换（i.i.d.）**。

**现有痛点**：现实里训练标签常被损坏——要么有噪声、要么直接缺失。一旦标签损坏，能用来算分数的就只剩"干净样本"，而干净样本的分布 $P_{X,Y\mid M=0}$ 和测试分布 $P_{X,Y}$ 不一致，可交换性被破坏，Naive CP（只拿观测到的标签做 CP）会**欠覆盖**。论文用 MEPS 医疗数据演示：随机按特征相关的概率删标签后，Naive CP 达不到 90% 覆盖。

**核心矛盾**：损坏导致的分布漂移本质是一种**协变量漂移**，标准做法是加权保形（WCP）——按似然比 $w(z)=\mathrm{d}P_{\text{test}}/\mathrm{d}P_{\text{train}}(z)$ 给分数重新加权使训练/测试看起来可交换。但 WCP 要求测试时能拿到全部特征来算权重。可现实中那些解释了"为什么这条标签会损坏"的特征（如收入、种族、标注者水平）恰恰因隐私或缺失，**在测试时拿不到**——这类只在训练时可见的特征就是**特权信息（Privileged Information, PI）**。已有的 PCP（特权保形预测）能在没有测试 PI 的情况下给出有效预测集，但它依赖一个苛刻前提：**权重 $w$ 必须是真值**。MEPS 实验里，一旦用估计权重，PCP 也欠覆盖了。

**本文目标**：在"标签损坏 + 测试时缺失特权特征"这个设定下，回答两件事——(1) PCP 对权重误差到底有多敏感？(2) 如果权重根本估不准，能不能完全绕开权重估计？

**切入角度**：作者发现 PCP 的有效性并不是"权重越准越好"这么单调，而是和 Naive CP 本身是过覆盖还是欠覆盖强相关；另一方面，如果 PI 不是用来解释"损坏指示变量 $M$"、而是能很好地**预测标签 $Y$ 本身**，那就可以换一条路：直接把损坏标签回填掉。

**核心 idea**：用"保留不确定性的标签回填（UI）"替代"权重重加权（PCP）"来对抗损坏标签的分布漂移，并把两条路连同 Naive CP 一起做成只要有一个假设成立就有效的三重稳健并集。

## 方法详解
论文按"PI 扮演什么角色"把问题劈成两种情形：情形一 PI 是损坏指示 $M$ 的解释变量（走 PCP + 鲁棒性分析），情形二 PI 是标签 $Y$ 的代理（走 UI），最后两者加 Naive CP 合成 TriplyRobust。

### 整体框架
设训练样本为 $\{(X_i, \tilde{Y}_i, Z_i, M_i)\}_{i=1}^n$：$X_i$ 是观测特征，$\tilde{Y}_i$ 是可能被损坏的观测标签，$Z_i$ 是特权信息，$M_i\in\{0,1\}$ 是损坏指示（$M_i=0$ 表示 $\tilde{Y}_i=Y_i$ 干净，$M_i=1$ 表示损坏，缺失场景下 $\tilde{Y}_i=$'NA'）。测试时只给 $X_{\text{test}}$，目标是构造满足边缘覆盖 $P(Y_{\text{test}}\in C(X_{\text{test}}))\ge 1-\alpha$ 的预测集。核心难点是干净样本分布 $P_{X,Y\mid M=0}$ 与测试分布 $P_{X,Y}$ 之间的漂移。全程的关键假设是 $(X,Y)\perp M\mid Z$：即给定 PI 后，损坏与否独立于真实数据（PI 解释了损坏的来源）。

本文沿三条线推进：① 对依赖权重的 PCP 做**鲁棒性刻画**，给出"权重错了多少还能保住覆盖"的精确区间；② 提出不碰权重、改回填标签的 **UI**；③ 把 Naive CP / PCP / UI 取**并集**成 TriplyRobust，只要三套假设里有一套成立就有效。

### 关键设计

**1. PCP 鲁棒性刻画：把"权重要多准"翻译成与 Naive CP 覆盖状态绑定的精确区间**

PCP 本身的流程是：在校准集上把每个校准点当测试点、用未损坏校准样本跑 WCP 子程序得到各自阈值 $Q(Z_i)$，再取这些阈值的 $(1-\beta)$ 经验分位数当最终阈值 $Q_{\text{PCP}}$（$\beta\in(0,\alpha)$，如 0.05），从而绕开对测试 PI 的需求。定理 1 保证它在**真权重**下有效。本文真正的贡献是追问：权重不准会怎样？作者先看常数误差 $\tilde{w}_i := w_i+\delta$（定理 2），结论出人意料地非单调——**PCP 是否有效取决于 Naive CP 处于过覆盖还是欠覆盖**：若 Naive CP 过覆盖（$Q_{\text{CP}}>Q_{\text{WCP}}$），则即便权重估得很差（$\delta\ge 0$ 或 $\delta$ 足够负）PCP 仍能保覆盖；但若 Naive CP 欠覆盖（$Q_{\text{CP}}<Q_{\text{WCP}}$），$\delta$ 必须落进狭窄区间 $\left(-\tfrac{W_{n+1}}{n+1},\,0\right)$ 才行，给出的覆盖率为 $P(Y_{\text{test}}\in C_{\text{PCP}})\ge 1-\alpha-\varepsilon$。随后定理 3 把分析推广到**逐样本变化的误差** $\tilde{w}_i=w_i+\delta_i$（$\delta_i\in[\delta_{\min},\delta_{\max}]$），有效区间由真权重 $w_i$ 与归一化误差分布 $\tilde{\delta}_i$ 共同决定：过覆盖时有效域是两个条件的 XOR、欠覆盖时正好是其补集 NXOR。

这套刻画和过去只做最坏情况（worst-case）分析的工作不同——它说明权重即使有显著误差，PCP 也可能照样有效，因此把"权重必须很准"这个直觉，替换成"看 Naive CP 的覆盖状态 + 误差是否落进可计算区间"。合成数据上分别构造 Naive CP 过覆盖/欠覆盖两套，扫描 $\delta$ 得到的经验有效区间与定理边界吻合，也解释了 MEPS 上 PCP 失败正是因为误差掉出了那个窄区间。

**2. 不确定填补 UI：用 PI 把损坏标签"带噪回填"，彻底绕开权重估计**

当 PI 是标签 $Y$ 的强代理（如训练时有高分辨率影像、详细临床报告）时，UI 不再去估权重，而是去估标签。流程上把数据分成训练集 $I_1$、校准集 $I_2$、参考集 $I_3$，并训两个模型：只用 $X$ 的 $\hat{f}(x)$（和标准 CP 一样）与同时用 $X,Z$ 的 $\hat{g}(x,z)$。在参考集上算 $\hat{g}$ 的残差 $E_i = Y_i-\hat{g}(X_i,Z_i)$，并按 $z$ 收集条件残差池 $\mathcal{E}(z)=\{E_i: i\in I_3, Z_i=z, M_i=0\}$。回填时，干净样本保留真标签，损坏样本则填成预测值**加一个从残差池随机抽的误差**：

$$\bar{Y}_i = \begin{cases} Y_i & M_i=0\\ \hat{g}(X_i,Z_i)+E(Z_i) & M_i=1\end{cases}$$

之所以要加随机残差 $E(Z_i)$ 而非直接填均值，是这里的精髓：直接填均值（Naive Imputation）会**人为压低标签的方差**，让预测区间过窄而欠覆盖；而"保留不确定性的回填"复刻了真实标签的离散程度，借助"CP 对色散型（增大标签变异）噪声鲁棒"的已有结论，使回填后的非一致性分数 $\bar{S}_i=S(X_i,\bar{Y}_i;\hat{f})$ 仍能给出有效阈值 $Q_{\text{UI}}$。定理 4 证明：只要 $\hat{g}$ 足够准（残差在给定 $Z$ 下独立于 $\hat{g}$ 的预测和预测集）、且 $Y\mid X,Z$ 的密度峰都落在预测区间内，UI 就满足 $P(Y_{\text{test}}\in C_{\text{UI}})\ge 1-\alpha$。直观上，**当给定 $X,Z$ 后 $Y$ 比较好预测时，UI 比 PCP 更管用**，且它在 PCP 失效（权重估不准）时仍能有效。

**3. 三重稳健 TriplyRobust：取三套预测集的并集，假设上"或"而非"且"**

PCP 与 UI 依赖的是**互补**的假设——PCP 要 $M\mid Z$ 估得好，UI 要 $Y\mid Z$ 估得好；而 Naive CP 在底模 $\hat{f}$ 理想时本就有效。既然三者各管一摊，作者直接把三个预测集取并集：

$$C_{\text{TriplyRobust}}(X_{\text{test}}) = C_{\text{Naive CP}}(X_{\text{test}}) \cup C_{\text{PCP}}(X_{\text{test}}) \cup C_{\text{UI}}(X_{\text{test}})$$

定理 5 保证：只要三套假设里**至少一套**成立（$Y\mid X$、$M\mid Z$、或 $Y\mid Z$ 三者之一被估准），TriplyRobust 就达到名义覆盖。取并集天然只会让区间变大、覆盖只增不减，所以"或"逻辑成立；而合成实验进一步显示这个并集**并不过度保守**——只要有一个组件是 oracle，覆盖就回到名义水平且区间不会无谓膨胀。这把"押注单一假设"换成了"三选一兜底"，在实践中明显更稳。

### 损失函数 / 训练策略
方法本身是校准层（calibration）方案，不引入新的训练损失。实验统一用 CQR（Conformalized Quantile Regression）作为非一致性分数，目标覆盖 $1-\alpha=90\%$；训练集拟合模型、验证集早停、校准集做校准（UI 再从校准集里切出参考集采残差），结果在 30 次随机划分上平均。

## 实验关键数据

### 主实验
作者用三类实验验证：合成数据测 TriplyRobust 鲁棒性、合成数据测"权重难估"时 UI 对 PCP 的优势、5 个真实回归基准（Facebook1/2、Bio、House、Meps19）做"缺失响应"。统一目标 90% 覆盖、CQR 分数、30 次划分平均。

| 实验 | 设定 | 关键现象 |
|------|------|----------|
| 合成·权重难估 | $Z$ 强预测 $Y$，缺失机制刻意难估 | PCP 因权重不准达不到 90%；UI 改靠 $(X,Z)\to Y$ 的估计，稳定命中 90% |
| 真实·缺失响应 | 5 基准，人工删 20% 标签、把与 $Y$ 最相关特征设为 PI | Naive CP / Naive Imputation 区间过窄、欠覆盖；PCP（真/估权重）与 UI 都稳定达到 90% |
| 合成·TriplyRobust | QR / PCP / UI 各取 degenerate 或 oracle 变体 | 三者全 degenerate 时欠覆盖；只要任一为 oracle，并集即达名义覆盖且不过度保守 |

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| Naive CP（只用观测标签） | 欠覆盖 | 干净样本分布≠测试分布，可交换性破坏 |
| Naive Imputation（填均值） | 区间过窄、欠覆盖 | 直接填均值压低标签方差 |
| UI（填预测值 + 随机残差） | 达 90% | 保留不确定性的回填复刻真实色散，覆盖恢复 |
| TriplyRobust 全 degenerate | 欠覆盖 | 三套假设都不成立 |
| TriplyRobust 任一 oracle | 达 90% 且不过度保守 | "或"逻辑兜底，并集不膨胀 |

### 关键发现
- **Naive Imputation vs UI 的对比是 UI 设计的核心证据**：两者都做回填，区别只在"是否加随机残差"，而正是这一步决定了覆盖是否成立——直接填均值会欠覆盖，加残差才有效。
- **PCP 的有效性与 Naive CP 的覆盖状态绑定**：真实实验里 PCP 用估计权重仍有效，反推出权重误差恰好落进定理 2/3 的有效区间，与理论自洽。
- **TriplyRobust 的并集不过度保守**：直觉上三区间取并应该很宽，但实验显示只要有一个组件准，覆盖就回到 90% 而非远超，说明这是个"廉价的保险"。

## 亮点与洞察
- **把"权重要多准"问题重述为"看 Naive CP 过覆盖还是欠覆盖"**：这是反直觉且实用的洞察——它解释了为什么有时权重明显有误差，方法却照样有效，避免了一刀切地追求精确权重。
- **"保留不确定性的回填"是可迁移的小技巧**：在任何"缺失值填补 + 下游需要可靠不确定性"的任务里，填均值/点估计都会人为收缩方差；改填"点估计 + 从条件残差池随机抽样"能保住分布的色散，这个思路可迁移到带缺失的回归校准、半监督等场景。
- **假设上的"或"而非"且"**：TriplyRobust 用并集把三套互补假设组合，是一种把"押注单一建模假设"变成"多假设兜底"的稳健性设计范式。

## 局限与展望
- **作者承认**：WCP/PCP 的有效性条件虽有理论支撑，却依赖真权重，实践中拿不到，如何从数据估出这些条件是开放问题。
- **UI 的强假设**：要求给定 PI 后特征与响应独立于损坏指示（类似因果推断里的强可忽略性 strong ignorability），且标签变异只依赖 PI、$Y$ 能从 $(X,Z)$ 准确估出——这些在高维/连续 $Z$ 下可能站不住，论文用聚类（clustering）缓解参考集 $\mathcal{E}(z)$ 为空的问题。
- **展望**：把理论保证推广到多标注者（multiple-annotator）设定，以及把特权信息融入"模糊感知（ambiguity-aware）"校准方法。

## 相关工作与启发
- **vs WCP（Tibshirani et al. 2019）**: WCP 用似然比加权纠正协变量漂移，但要求测试时能算权重（即拿到全部特征）；本文设定下关键特征是测试不可见的 PI，WCP 无法直接用。
- **vs PCP（Feldman & Romano 2024）**: PCP 把 WCP 当子程序、绕开测试 PI，但依赖真权重；本文一方面给出 PCP 对权重误差的精确鲁棒性边界，另一方面用 UI 完全绕开权重估计。
- **vs 噪声标签下的 CP（Einbinder et al. 2023; Sesia et al. 2024）**: 那条线证明 CP 对色散型对称加性噪声鲁棒；UI 正是借用这一结论，故意"带噪回填"而非精确填补，让回填后的分数仍有效。
- **vs 最坏情况权重分析（Lei & Candès 2021; Bhattacharyya & Barber 2024 等）**: 已有工作多做 worst-case 刻画；本文的定理 2/3 给出与 Naive CP 覆盖状态挂钩的更细致有效区间，揭示权重显著出错时方法仍可能有效。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ PCP 鲁棒性的非单调刻画 + 不确定填补 UI + 三重稳健并集，三点都新且互相咬合
- 实验充分度: ⭐⭐⭐⭐ 合成 + 5 个真实基准 + degenerate/oracle 消融充分，但任务集中在回归/缺失响应，分类等场景未覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 理论与直觉穿插、用 MEPS/Figure 1 贯穿动机，定理假设解释清楚
- 价值: ⭐⭐⭐⭐ 给"损坏标签 + 测试缺失特征"下的可靠不确定性量化提供了可落地且有保证的方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)
- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICML 2026\] Enhancing Conformal Prediction via Class Similarity](../../ICML2026/learning_theory/enhancing_conformal_prediction_via_class_similarity.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)

</div>

<!-- RELATED:END -->
