---
title: >-
  [论文解读] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach
description: >-
  [AAAI 2026][加权模型计数] 本文将加权模型计数 (WMC) 的权重视为具有方差的随机变量，提出在 structured d-DNNF 表示上多项式时间计算 WMC 方差的算法，同时证明了在 structured DNNF、d-DNNF 和 FBDD 上该问题不可解（除非 P=NP），并将其应用于贝叶斯网络推理中参数不确定性的量化。
tags:
  - "AAAI 2026"
  - "加权模型计数"
  - "知识编译"
  - "方差计算"
  - "贝叶斯网络"
  - "结构化d-DNNF"
---

# Variance Computation for Weighted Model Counting with Knowledge Compilation Approach

**会议**: AAAI 2026  
**arXiv**: [2601.03523](https://arxiv.org/abs/2601.03523)  
**代码**: [https://github.com/nttcslab/variance-wmc](https://github.com/nttcslab/variance-wmc)  
**领域**: 知识编译 / 概率推理 / 加权模型计数  
**关键词**: 加权模型计数, 知识编译, 方差计算, 贝叶斯网络, 结构化d-DNNF

## 一句话总结

本文将加权模型计数 (WMC) 的权重视为具有方差的随机变量，提出在 structured d-DNNF 表示上多项式时间计算 WMC 方差的算法，同时证明了在 structured DNNF、d-DNNF 和 FBDD 上该问题不可解（除非 P=NP），并将其应用于贝叶斯网络推理中参数不确定性的量化。

## 研究背景与动机

**领域现状**：知识编译是将命题公式编译为紧凑可追踪形式的核心技术，其中最重要的查询是加权模型计数 (WMC)——计算布尔函数满足赋值的加权总数。WMC 已广泛应用于贝叶斯网络、因子图、概率编程等概率推理任务。

**现有痛点**：在实际场景中，概率模型的参数通常由数据学习获得，当训练数据不充分时，参数本身就带有不确定性。然而，传统的 WMC 推理方法将权重视为固定实数，完全忽略了参数不确定性，导致推理结果可能不可靠——我们无法判断推理输出到底有多大的波动空间。

**核心矛盾**：贝叶斯统计方法可以通过为参数引入分布，将推理结果视为随机变量并计算其方差来度量不确定性。但 WMC 方差计算的可追踪性（tractability）在各种知识编译表示上几乎完全未知。此前仅有 Nakamura et al. (2022) 在 OBDD 这一最受限表示上、针对网络可靠性分析这一特殊场景做了初步工作。

**本文目标** (a) 在更通用的知识编译表示上，WMC 方差是否可以多项式时间计算？(b) 在哪些表示上该问题变得不可追踪？(c) 如何将方差计算应用于贝叶斯网络推理以量化不确定性？

**切入角度**：作者从知识编译图 (knowledge compilation map) 的视角出发，系统地研究 VC（方差计算）和 CVC（协方差计算）查询在各种 NNF 子类上的可追踪性边界。

**核心 idea**：利用 vtree 引导的变量分解结构，通过递归分解协方差公式并缓存中间结果，在 structured d-DNNF 上实现多项式时间 WMC 方差计算，同时通过规约证明更宽松表示上的不可追踪性。

## 方法详解

### 整体框架

输入是以 structured d-DNNF 表示的布尔函数 $f$，以及每个布尔变量 $x$ 的正权重 $P_x$ 和负权重 $N_x$ 的期望、方差和协方差。输出是 WMC $W_f$ 的方差 $\mathrm{V}[W_f]$。

整体流程分三个阶段：
- **预处理**：构建 LCA 数据结构（$O(|\mathcal{V}|)$），预计算辅助函数 ADJEXP 和 ADJCOV 所需的数据（$O(|\mathcal{V}|^2)$）
- **期望计算**：用标准 WMC 算法预计算每个节点的期望值 $\mathrm{E}[W_{f_\gamma}]$（$O(|\alpha|)$）
- **协方差递归计算**：通过 Algorithm 1 递归分解并缓存 $\mathrm{Cov}[W_{f_\alpha}, W_{f_\beta}]$，最终方差 $\mathrm{V}[W_f] = \mathrm{Cov}[W_f, W_f]$

总复杂度为 $O(|\alpha|^2 + |\mathcal{V}|^2)$。

### 关键设计

1. **WMC 方差的形式化定义 (VC/CVC 查询)**:

    - 功能：将每个变量的权重 $(P_x, N_x)$ 视为随机变量，定义方差计算查询 VC 为计算 $\mathrm{V}[W_f^\mathcal{V}]$，协方差计算查询 CVC 为计算 $\mathrm{Cov}[W_f^\mathcal{V}, W_g^\mathcal{V}]$
    - 核心思路：假设不同变量的权重对 $(P_x, N_x)$ 与 $(P_y, N_y)$（$x \neq y$）互相独立，但同一变量的 $P_x$ 与 $N_x$ 可以相关。这使得期望 $\mathrm{E}[W_f]$ 等价于普通 WMC，方差则是需要新算法来计算的新查询
    - 设计动机：独立性假设对贝叶斯网络的参数独立性自然成立（parameter independence assumption），且使分解公式可推导

2. **基于 vtree 的协方差递归分解 (Algorithm 1)**:

    - 功能：递归计算任意两个 st-d-DNNF 节点 $\alpha, \beta$ 的 WMC 协方差
    - 核心思路：根据 $\mathsf{d}(\alpha)$ 和 $\mathsf{d}(\beta)$ 在 vtree 中的关系，分三种情况分解：
        - **Case I**（无祖先-后代关系）：利用乘积独立性公式 $\mathrm{Cov}[AX, BY] = \mathrm{Cov}[A,B]\mathrm{Cov}[X,Y] + \mathrm{Cov}[A,B]\mathrm{E}[X]\mathrm{E}[Y] + \mathrm{E}[A]\mathrm{E}[B]\mathrm{Cov}[X,Y]$，将协方差分解到 vtree 的左右子树
        - **Case II**（$\alpha$ 是 $\vee$-节点）：利用确定性(determinism)保证，按协方差的可加性 $\mathrm{Cov}[A+B, C] = \mathrm{Cov}[A,C] + \mathrm{Cov}[B,C]$ 分解到子节点
        - **Case III**（$\alpha$ 是 $\wedge$-节点）：利用结构化分解性(structured decomposability)，将 $\beta$ 也分解到 vtree 的左右子树后用乘积公式
    - 设计动机：vtree 提供了变量集的层次化分解结构，使得分解过程中可以精确追踪变量集的变化——这是相比 Nakamura et al. (2022) 在 OBDD 上的算法的关键技术差异
    - 缓存机制：为避免重复递归调用，所有已计算的 $\mathrm{Cov}[\alpha, \beta]$ 值存入 $\mathtt{c}[\alpha, \beta]$，保证总复杂度 $O(|\alpha||\beta|)$

3. **辅助函数 ADJEXP 和 ADJCOV（变量集调整）**:

    - 功能：在递归过程中调整 WMC 的变量集，使不同子问题的变量集一致
    - 核心思路：由于 $W_f^\mathcal{V}$ 的值随 $\mathcal{V}$ 变化而变化，当节点的作用域小于当前 vtree 节点的作用域时，需要用 true 函数"补全"缺失变量的贡献。预处理阶段计算 $\prod_{x \in S}(\mu_{P_x} + \mu_{N_x})$ 等辅助量，使调整操作在 $O(1)$ 时间内完成
    - 设计动机：这是将算法从 OBDD 推广到 st-d-DNNF 的核心困难——OBDD 的变量次序是线性的，而 vtree 是树形的，变量集管理更加复杂

### 理论分析

**可追踪性 (Theorem 7)**：当 $f, g$ 以共享同一 vtree 的 st-d-DNNF $\alpha, \beta$ 给出时，CVC 可在 $O(|\alpha||\beta| + |\mathcal{V}|^2)$ 时间内求解；VC 可在 $O(|\alpha|^2 + |\mathcal{V}|^2)$ 时间内求解。

**不可追踪性 (Theorem 11)**：当布尔函数以 st-DNNF、d-DNNF 或 FBDD 表示时，VC 和 CVC 均不可追踪（除非 P=NP）。证明策略：
- 对 st-DNNF：将模型计数 (CT) 规约到 VC——通过特殊权重设置使得 $|A_f| = \lceil V[W_f]/(4^n - 1)\rceil$
- 对 d-DNNF/FBDD：将语句蕴含 (SE) 规约到 CVC，再利用 Lemma 15 将 CVC 进一步规约到 VC

这意味着 d-DNNF 和 FBDD 虽然支持多项式时间 WMC，但不支持多项式时间方差计算——这是一个令人意外且有趣的可追踪性分离结果。

## 实验关键数据

### 主实验：70 个二值贝叶斯网络上的方差计算

从 bnRep 数据库获取 70 个二值贝叶斯网络（3 至 122 个随机变量），用 ENC2 编码为 CNF 后编译为 SDD（st-d-DNNF 的子集）。用 Beta 分布建模参数不确定性（$\theta=10$）。

| 网络名称 | 随机变量数 | SDD 大小 | 编译时间 (s) | 方差计算时间 (s) |
|----------|-----------|---------|-------------|----------------|
| projectmanagement | 26 | 3888 | 0.500 | 0.025 |
| GDIpathway2 | 28 | 2755 | 0.784 | 0.021 |
| grounding | 36 | 3397 | 2.387 | 0.017 |
| engines | 12 | 1804 | 0.240 | 0.011 |
| windturbine | 122 | 2043 | 1.380 | 0.009 |

**关键结论**：SDD 编译完成后，方差计算最多仅需 0.025 秒；即使加上 SDD 编译时间，最耗时的网络也仅需约 10 秒。

### 参数影响分析：algalactivity2 网络

计算 $\Pr(\text{Chl\_a}=0)$ 的方差，并逐一将每个参数的方差降至原来的 1/10，考察对边际概率方差的影响。

| 被降方差的参数 | 降方差后的边际方差 |
|--------------|-----------------|
| DO\|pH₀,Te₀ | 0.002887 |
| Chl_a\|C₁,DO₀,N₀,Te₁ | 0.003532 |
| Te\|P₀ | 0.003554 |
| pH\|Te₀ | 0.003592 |
| Chl_a\|C₁,DO₁,N₁,Te₀ | 0.003674 |
| (无降方差) | 0.003904 |

### 关键发现

- **方差计算极其高效**：在 SDD 编译后，70 个网络的方差计算均在毫秒级完成，验证了算法的实用可追踪性
- **参数影响非显而易见**：对 $\Pr(\text{Chl\_a}=0)$ 的方差影响最大的参数不仅包括 Chl_a 自身的条件概率，还包括 DO、Te、pH 等其他变量的条件概率，说明不实际计算方差就无法识别哪些参数对推理不确定性影响最大
- **实际决策价值**：$\Pr(\text{Chl\_a}=0)$ 的均值为 0.5281，标准差为 0.0625，当需要判断 $\Pr(\text{Chl\_a}=0) \leq 0.55$ 时，这一量级的方差会实质性影响决策

## 亮点与洞察

- **知识编译图的优雅扩展**：将 VC/CVC 查询嵌入知识编译图的可追踪性框架中，得到了一个"恰到好处"的可追踪性边界——st-d-DNNF 可追踪、更宽松的表示均不可追踪。这个 sharp boundary 非常优美
- **d-DNNF/FBDD 的意外分离**：虽然 d-DNNF 和 FBDD 支持多项式时间 WMC，但 VC 不可追踪——这说明"计算方差"比"计算期望"在本质上更难，需要结构化分解性这一额外约束。这一理论发现本身就很有价值
- **从 OBDD 到 st-d-DNNF 的非平凡推广**：vtree 引导的变量集管理是关键技术贡献。线性变量序（OBDD）到树形分解（st-d-DNNF）的推广，不仅让算法适用面更广，还顺带改进了网络可靠性分析的已知结果（从 pathwidth 到 treewidth）
- **实用的不确定性量化工具**：方差计算使得"哪些参数学得不准会最影响推理结果"可以量化回答，这对指导数据收集和参数学习有直接的实际价值

## 局限与展望

- **CVC 要求共享 vtree**：协方差计算要求两个 st-d-DNNF 尊重同一 vtree，这限制了某些应用场景。作者推测不共享 vtree 时 CVC 不可追踪，但未给出证明
- **仅支持方差（二阶矩）**：高阶矩（如偏度、峰度）的可追踪性未被探讨，而这些在某些决策场景中也很重要
- **编译瓶颈**：SDD 编译本身可能成为瓶颈（最耗时的网络编译需 10 秒），且 SDD 大小可能指数级增长。方差计算虽然多项式，但前提是布尔函数已被编译
- **参数独立性假设**：虽然在贝叶斯网络中合理，但在其他应用（如概率编程）中可能不成立
- **仅验证二值贝叶斯网络**：虽然理论上通过 Appendix B 推广到多值变量，但实验仅在二值网络上验证

## 相关工作与启发

- **vs Nakamura et al. (2022)**：前者仅在 OBDD 上处理网络可靠性分析的特殊场景，本文将算法扩展到 st-d-DNNF（严格超集），将问题推广到一般 WMC，且理论上改进了 pathwidth 到 treewidth 的结果
- **vs 概率电路 (Probabilistic Circuits)**：概率电路可以表示随机变量 $X$ 的分布并计算其矩，但本文需要计算的是概率值 $\Pr(X=a)$ 本身作为随机变量的方差——这是一个不同层面的问题，概率电路目前无法解决
- **vs 凭据网络 (Credal Networks)**：凭据网络通过计算边际概率的上下界来处理不确定性，但即使对 constant treewidth 的网络，推理也是 NP-hard；而本文的方差计算在同等条件下是多项式时间

## 评分

- 新颖性: ⭐⭐⭐⭐ 将方差计算查询系统地嵌入知识编译图是新的视角，但整体思路是对已有框架的自然扩展
- 实验充分度: ⭐⭐⭐⭐ 70 个真实贝叶斯网络 + 参数影响分析 case study，覆盖面好，但缺少与 Monte Carlo 等基线的对比
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义清晰，证明严谨，案例演示直观，整体结构非常专业
- 价值: ⭐⭐⭐⭐ 理论贡献扎实（可追踪性边界），应用前景明确（贝叶斯网络不确定性量化），但受众较窄（知识编译/概率推理社区）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[CVPR 2025\] Uncertainty Weighted Gradients for Model Calibration](../../CVPR2025/others/uncertainty_weighted_gradients_for_model_calibration.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)

</div>

<!-- RELATED:END -->
