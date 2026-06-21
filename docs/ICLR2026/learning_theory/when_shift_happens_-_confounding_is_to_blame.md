---
title: >-
  [论文解读] When Shift Happens - Confounding is to Blame
description: >-
  [ICLR 2026][学习理论][分布偏移] 这篇论文从因果与信息论的视角给出理论解释：在「隐藏混杂偏移」下，单纯学习不变表征不够，反而需要学习**环境特定**的关系，这正是为什么朴素的 ERM/XGBoost 常能打平甚至超过专门的 OOD 泛化方法、以及为什么把非因果但有信息量的协变量也加进来能提升泛化——并用 8 个真实表格数据集与合成数据加以验证。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "OOD 泛化"
  - "分布偏移"
  - "隐藏混杂"
  - "因果不变性"
  - "预测信息分解"
  - "协变量选择"
---

# When Shift Happens - Confounding is to Blame

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=sFjxg8cyJS](https://openreview.net/forum?id=sFjxg8cyJS)  
**代码**: https://github.com/gautam0707/Confounding_is_to_Blame  
**领域**: 学习理论 / OOD 泛化  
**关键词**: 分布偏移, 隐藏混杂, 因果不变性, 预测信息分解, 协变量选择

## 一句话总结
这篇论文从因果与信息论的视角给出理论解释：在「隐藏混杂偏移」下，单纯学习不变表征不够，反而需要学习**环境特定**的关系，这正是为什么朴素的 ERM/XGBoost 常能打平甚至超过专门的 OOD 泛化方法、以及为什么把非因果但有信息量的协变量也加进来能提升泛化——并用 8 个真实表格数据集与合成数据加以验证。

## 研究背景与动机

**领域现状**：OOD（out-of-distribution）泛化的主流共识是「学因果不变表征」——IRM、VREX、Group DRO 等方法都假设存在跨环境不变的统计关系（典型是 $P(X\mid Y)$ 不变的 label shift 假设，或 $P(Y\mid X)$ 不变的 covariate shift 假设），只要把这些不变关系抓住，模型就能跨环境泛化。

**现有痛点**：近年一系列实证研究给出反直觉结论：(i) 在认真做模型选择的前提下，标准 ERM 常能与 SOTA 的 OOD 方法打平甚至胜出；(ii) Nastl & Hardt（2024）在 16 个真实表格数据集上发现，把**所有**可用协变量（哪怕是非因果的）都喂进去预测，反而比只用因果子集泛化更好。这与「学因果不变就够了」的信条直接冲突，但缺乏理论解释。

**核心矛盾**：现有方法依赖的不变性（label shift / covariate shift 假设）建立在「未观测变量 $U$ 只导致 $X$ 或只导致 $Y$」的因果结构上。但现实里 $U$ 往往**同时导致 $X$ 和 $Y$**（即隐藏混杂 confounder），且 $P(U)$ 在不同环境间发生偏移。此时 label shift、covariate shift、conditional covariate shift、concept shift 会**同时**出现，所有单一不变性假设全部破裂。

**本文目标**：在隐藏混杂偏移下，理论刻画「究竟该学什么才能泛化」，并解释两个实证谜团——为什么 ERM 能赢、为什么加非因果协变量有帮助。

**切入角度**：作者不去提出新方法，而是把「预测信息」$I(Y;\hat Y)$（预测 $\hat Y$ 对真实标签 $Y$ 的信息量）作为统一目标，用因果图上的 d-separation 推理把它分解成若干可解释的偏移项，看在混杂结构下哪些项会失效、哪些项才是关键。

**核心 idea**：用一个 **predictive information 分解** 把各类 OOD 方法统一进同一框架；并证明在隐藏混杂下分解会**坍缩**成「条件信息量 − 残差」，从而说明泛化的关键不是抹掉环境信息、而是显式学习环境特定关系。

## 方法详解

### 整体框架

这是一篇**理论解释**型论文，没有可训练的 pipeline，方法即一套围绕「预测信息」$I(Y;\hat Y)$ 的因果-信息论分析框架。整篇推导可看成三步递进：先用通用因果结构把预测信息拆成 6 个带物理含义的偏移项（Theorem 4.1），把现有各派 OOD 方法都映射成「最大化/最小化其中某一项」；再代入隐藏混杂这一具体结构（$U\to X$、$U\to Y$，且 $X\to Y$ 或 $Y\to X$），用 d-separation 证明大部分偏移项相互抵消，分解坍缩成只剩「条件信息量 − 残差」（Theorem 4.2），由此得出「需要学环境特定关系」的结论；最后分析往输入里加入非因果但有信息量的协变量 $X_I$（作为隐藏混杂 $U$ 的代理）会如何系统性地改善这些项（Proposition 4.1）。

记 $\hat Y=(f\circ\phi)(X)$，$f$ 是分类器、$\phi$ 是特征提取/变换。整个框架用互信息度量各类偏移：$I(X;E)$ 量化 $P(X)$ 偏移、$I(Y;E)$ 量化 label shift、$I(X;E\mid Y)$ 量化 conditional covariate shift、$I(Y;E\mid X)$ 量化 concept shift，其中 $E$ 是刻画 $P(U)$ 跨环境变化的环境变量。这套度量是连接「因果图结构」和「可观测数据偏移」的桥梁。

### 关键设计

**1. 预测信息的通用分解：把所有 OOD 方法装进同一个公式**

针对「各派 OOD 方法各执一词、缺乏统一视角」的痛点，作者在通用因果结构 $X\leftrightarrow Y$（即一部分协变量导致 $Y$、一部分被 $Y$ 导致）下，证明预测信息可分解为六项（Theorem 4.1）：

$$I(Y;\hat Y)=\underbrace{I(\phi(X);Y\mid E)}_{\text{条件信息量}}-\frac{\overbrace{I(\phi(X);E\mid Y)}^{\text{variation}}}{2}+\frac{\overbrace{I(Y;E)}^{\text{label shift}}}{2}+\frac{\overbrace{I(\phi(X);E)}^{\text{feature shift}}}{2}-\frac{\overbrace{I(Y;E\mid \phi(X))}^{\text{concept shift}}}{2}-\underbrace{I(\phi(X);Y\mid \hat Y)}_{\text{residual}}$$

这个分解的价值在于它把现有方法都翻译成「操纵某一项」：IRM 强制分类器跨环境不变，等价于最大化 $-I(\phi(X);E\mid Y)$（即抑制 variation）；DANN 与公平分类的独立性准则最小化 $I(\phi(X);E)$（feature shift）；CDAN 对齐联合分布 $P_e(\phi(X),Y)$，等价于把 concept shift $I(Y;E\mid\phi(X))$ 驱到 0。关键洞察是：从公式可见，**盲目最小化 $I(\phi(X);E)$（抹掉环境信息）会直接减小 $I(Y;\hat Y)$**——这解释了为什么「学得越不变、有时反而预测越差」。

**2. 隐藏混杂下分解坍缩：泛化需要环境特定关系，而非纯不变性**

这是全文理论核心，针对「为什么 ERM 能赢、不变性方法反而吃亏」。作者代入隐藏混杂的具体因果结构：$U\to X$、$U\to Y$，并分别讨论 $X\to Y$ 与 $Y\to X$ 两种情形。借助 d-separation 推理（如对 $Y$ 取条件会在 collider 节点打开 $\phi(X)\leftarrow X\to Y\leftarrow U\leftarrow E$ 这条路径，反之对 $\phi(X)$ 取条件会部分阻断），可得若干不等式（如 $X\to Y$ 时 $I(\phi(X);E\mid Y)\ge I(\phi(X);E)$、$I(Y;E)\ge I(Y;E\mid\phi(X))$）。把它们代回 Theorem 4.1，variation、label shift、feature shift、concept shift 四项两两抵消，预测信息坍缩为（Theorem 4.2）：

$$I(Y;\hat Y)=\underbrace{I(\phi(X);Y\mid E)}_{\text{条件信息量}}-\underbrace{I(\phi(X);Y\mid \hat Y)}_{\text{residual}}$$

这意味着：在隐藏混杂偏移下，能最大化泛化的唯一杠杆是**条件信息量 $I(\phi(X);Y\mid E)$**——即在**每个环境内部**抓住 $\phi(X)$ 对 $Y$ 的信息，也就是学习**环境特定**关系。这恰好解释了 Simpson 悖论（图 2）：只用 $X$ 的线性回归会学到与真实趋势相反的关系，而引入环境特定的统计量（均值/标准差/分位数）作为混杂的代理（类似因果效应估计里的 backdoor 调整）就能恢复正确的 $X$–$Y$ 关系。它还为 Prashant 等（2025）的 MoE 模型（每个专家对应一个隐藏混杂取值）提供了理论依据——MoE 本质就是在最大化 $I(\phi(X);Y\mid E)$。这也说明为什么强调不变性的 IRM/VREX/GDRO 在这类场景里集体失手。

**3. 加入非因果信息协变量：用代理变量同时抬高有用项、压低有害项**

针对「为什么把非因果协变量都加进来反而更好」。作者定义**信息协变量** $X_I$：与 $Y$ 无因果关系（既非祖先也非后代），但给定其他因果协变量 $X$ 和环境 $E$ 后仍与 $Y$ 不独立，即 $Y\not\perp X_I\mid X,E$。由于 $U\to Y$，任何对 $U$ 有信息的变量也对 $Y$ 有信息，于是 $X_I$ 可充当隐藏混杂 $U$ 的代理。Proposition 4.1 证明：把 $X_I$ 并入输入后，条件信息量 $I(\phi_2(\{X\cup X_I\});Y\mid E)$ 上升、feature shift 上升、concept shift $I(Y;E\mid\phi_2)$ 下降，但 variation 也会被放大。换言之加代理变量是一把双刃剑——好处是提升条件信息量并降低 concept shift，代价是抬高 variation。实验观察到的净效应是前者主导：concept shift 的下降带来的收益显著大于 variation 上升的损失，所以「加更多信息协变量」总体有利于泛化。这给出了「有原则的协变量选择」的理论落点：该收集的是对 $U$ 或 $Y$ 有信息量的代理变量。

## 实验关键数据

实验目的有三：(i) 验证真实数据里隐藏混杂偏移确实普遍存在；(ii) 验证分解中「条件信息量 − 残差」与精度正相关；(iii) 验证加入信息协变量能提升泛化。数据取自 TableShift 基准的 8 个真实表格数据集（Food stamps、Readmission、Income、Public coverage、Unemployment、Diabetes、Hypertension、ASSISTments），互信息用 NPEET 工具箱的 KSG 估计器估算。对比方法含两个 ERM 派（XGBoost、MLP）、两个域泛化派（IRM、VREX）和一个鲁棒学习派（Group DRO）。

### 主实验

各类偏移在真实数据中均显著非零（对零均值做单样本 t 检验，$p\approx 0$），佐证隐藏混杂偏移普遍存在：

| 数据集 | 条件协变量偏移 $I(X;E\mid Y)$ | label shift $I(Y;E)$ | covariate shift $I(X;E)$ | concept shift $I(Y;E\mid X)$ |
|--------|------|------|------|------|
| Readmission | 0.107 | 0.068 | 0.097 | 2.032 |
| Food stamps | 0.126 | 0.030 | 0.108 | 2.118 |
| Public coverage | 0.231 | 0.412 | 0.222 | 1.945 |
| ASSISTments | 0.293 | 0.260 | 0.306 | 0.367 |

「条件信息量 − 残差」与精度的 Spearman 秩相关：对 ID 测试精度 $\rho=0.93$、对 OOD 测试精度 $\rho=0.80$（5 方法 × 8 数据集），强支持 Theorem 4.2 的预测目标。

### 消融实验

按 Nastl & Hardt（2024）把协变量分成三个嵌套子集：causal (C) ⊆ arguably causal (AC) ⊆ all (A)，用 sign consistency 度量（某项随精度提升而朝其「有益方向」变化的频率）和分子集精度评估「加协变量」的效果：

| 方法 | CI 一致性 | CS 一致性 | OOD-Test (C) | OOD-Test (AC) | OOD-Test (A) |
|------|------|------|------|------|------|
| XGB | 0.92 | 0.79 | 64.35 | 72.80 | 72.90 |
| MLP | 0.65 | 0.85 | 62.03 | 67.92 | 66.93 |
| GDRO | 0.69 | 0.90 | 61.95 | 66.64 | 65.87 |
| IRM | 0.71 | 0.88 | 61.14 | 61.18 | 62.75 |
| VREX | 0.52 | 0.85 | 60.40 | 65.57 | 65.21 |

> 注：sign consistency 越高说明该项越可靠地随精度同向变化（CI = 条件信息量，正向项；CS = concept shift，负向项）。

### 关键发现
- **条件信息量是分水岭**：concept shift 的 sign consistency 对所有方法都高（说明「加协变量降 concept shift」是普遍现象），但真正决定胜负的是条件信息量——只有 XGB 把 CI 一致性做到 0.92，对应它在 OOD 上明显领先（A 子集 72.90% vs IRM 62.75%）。这与 Theorem 4.2「关键是最大化条件信息量」完全吻合。
- **加协变量普遍有益、但收益递减/偶有回退**：从 C→AC→A，多数方法 OOD 精度上升（如 XGB 64.35→72.80→72.90），印证 Proposition 4.1；但 MLP/GDRO 在 A 上略回退，说明 variation 被放大的副作用在某些方法上会显现。
- **不变性方法整体吃亏**：IRM 的 ID/OOD 精度全面偏低（OOD 在 C 子集仅 61.14%），符合「在隐藏混杂下纯不变性次优」的理论判断。
- **合成数据验证**：在已知因果结构 $U\to X,\;U\to Y,\;X\to Y,\;U\to X_I$ 下，随代理变量数 $|X_I|$ 增多，MSE 下降、条件信息量与 feature shift 上升、concept shift 下降（图 4），与理论一致。

## 亮点与洞察
- **一个分解统一一片江湖**：把 IRM / DANN / CDAN / GDRO 等方法都还原成「操纵预测信息分解里的某一项」，这种「用一个公式解释一类方法在干嘛」的视角非常可复用——遇到新 OOD 方法可以直接问它在最大化/最小化哪一项。
- **「坍缩」是最漂亮的一步**：通过 d-separation 让六项分解在隐藏混杂下精确坍缩成两项，把「该学什么」从模糊的工程直觉变成可证明的结论（学环境特定关系，而非抹环境信息），同时一举解释了 ERM 能赢、MoE 合理、不变性方法吃亏三件事。
- **为「反直觉实证」正名**：把「加非因果协变量更好」从看似违反因果直觉的现象，解释为「代理变量降低 concept shift、抬高条件信息量」的必然结果，给协变量选择提供了可操作的判据（找对 $U$/$Y$ 有信息量的代理）。
- **可迁移的思维**：把「环境特定统计量当作隐藏混杂的代理、做 backdoor 调整」这一招，可迁移到任何存在未观测混杂的预测任务——不必去显式建模 $U$，只要喂入足够的环境/代理信息即可恢复正确关系。

## 局限与展望
- **只解释、不解决**：作者明确说本文目标是解释现象，不提供应对隐藏混杂偏移的具体算法；如何据此设计新方法仍是开放问题。
- **理论依赖结构假设**：Theorem 4.2 的干净坍缩依赖 $X\to Y$ 或 $Y\to X$ 的明确单向结构。真实数据里 $X\leftrightarrow Y$ 混合（虽然作者指出 16 个基准中 11 个是 $X\to Y$ 主导），更一般的纠缠结构下结论强度会打折。
- **互信息估计的脆弱性**：所有结论建立在 KSG 互信息估计之上，高维/小样本下估计偏差可能影响 sign consistency 等度量的可靠性，论文未深入讨论估计误差对结论的影响。
- **代理变量假设的现实性**：Proposition 4.1 需要 $X_I$ 是 $U$/$Y$ 的有效代理且满足条件独立性，现实中很难验证「是否已收集到足够有信息量的协变量」。作者也把「不依赖不可检验代理假设地处理纠缠偏移」列为未来工作。
- **改进思路**：可沿作者列的方向——量化「获取非因果协变量的成本 vs 精度收益」、把 variation 放大的副作用纳入显式正则、以及把对未观测混杂的不确定性建模进 OOD-鲁棒的新范式。

## 相关工作与启发
- **vs Nastl & Hardt (2024)**：他们实证发现「用全部协变量在 ID/OOD 上 Pareto 占优」，但只给现象、无理论；本文用预测信息分解 + Proposition 4.1 给出了为什么——加协变量降 concept shift、提条件信息量。
- **vs IRM / VREX / Group DRO**：这些方法各自最大化/最小化分解中的某一项（如 IRM 抑 variation、GDRO 抗 label shift），本文证明在隐藏混杂下这些项相互抵消、不再是正确目标，从而解释了它们为何常被 ERM 超越。
- **vs Prashant et al. (2025)**：他们提出 MoE（每个专家对应一个混杂取值）应对混杂偏移；本文证明 MoE 等价于最大化条件信息量 $I(\phi(X);Y\mid E)$，为其提供理论依据，同时指出其「混杂支撑重叠 + 离散代理」假设偏强，呼唤更一般的方法。
- **vs anchor regression (Rothenhäusler et al., 2021) / Eastwood et al. (2023)**：前者在「全协变量」与「纯因果协变量」间做线性权衡；后者论证不稳定协变量在条件独立时能帮提升。本文把这些视角统一到「隐藏混杂下加代理变量如何改变各偏移项」的框架里。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用一个预测信息分解统一解释「ERM 为何能赢」「加非因果协变量为何有用」两大反直觉实证谜团，视角新颖。
- 实验充分度: ⭐⭐⭐⭐ 8 个真实数据集 + 合成数据 + 5 类方法系统验证，但互信息估计误差与更一般因果结构验证略欠。
- 写作质量: ⭐⭐⭐⭐ 因果-信息论推理严谨、动机清晰；但 d-separation 与多项分解对非专业读者门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 为 OOD 泛化与协变量选择提供可操作的理论指南（学环境特定关系、收集有信息量的代理变量），影响面广。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] When Bias Meets Trainability: Connecting Theories of Initialization](when_bias_meets_trainability_connecting_theories_of_initialization.md)
- [\[ICLR 2026\] Neyman-Pearson Classification under Both Null and Alternative Distributions Shift](neyman-pearson_classification_under_both_null_and_alternative_distributions_shif.md)
- [\[ICLR 2026\] Know When to Abstain: Optimal Selective Classification with Likelihood Ratios](know_when_to_abstain_optimal_selective_classification_with_likelihood_ratios.md)
- [\[ICLR 2026\] Why Ask One When You Can Ask k? Learning-to-Defer to the Top-k Experts](why_ask_one_when_you_can_ask_k_learning-to-defer_to_the_top-k_experts.md)
- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](../../ICML2026/learning_theory/when_sample_selection_bias_precipitates_model_collapse.md)

</div>

<!-- RELATED:END -->
