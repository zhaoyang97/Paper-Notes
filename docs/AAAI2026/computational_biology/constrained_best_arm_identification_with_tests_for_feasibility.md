---
title: >-
  [论文解读] Constrained Best Arm Identification with Tests for Feasibility
description: >-
  [AAAI 2026][计算生物][Best Arm Identification] 提出带可行性约束的最优臂识别新框架，允许决策者分别测试臂的性能或可行性约束，设计了渐近最优算法，可自适应地选择通过性能或可行性中更容易的方式淘汰次优臂。 最优臂识别（BAI）的广泛应用：BAI 旨在从 $K$ 个臂中通过随机采样找到期望回报…
tags:
  - "AAAI 2026"
  - "计算生物"
  - "Best Arm Identification"
  - "feasibility constraints"
  - "fixed confidence"
  - "sample complexity"
  - "asymptotic optimality"
---

# Constrained Best Arm Identification with Tests for Feasibility

**会议**: AAAI 2026  
**arXiv**: [2511.09808](https://arxiv.org/abs/2511.09808)  
**代码**: 无  
**领域**: 计算生物  
**关键词**: Best Arm Identification, feasibility constraints, fixed confidence, sample complexity, asymptotic optimality

## 一句话总结

提出带可行性约束的最优臂识别新框架，允许决策者分别测试臂的性能或可行性约束，设计了渐近最优算法，可自适应地选择通过性能或可行性中更容易的方式淘汰次优臂。

## 研究背景与动机

**最优臂识别（BAI）的广泛应用**：BAI 旨在从 $K$ 个臂中通过随机采样找到期望回报最高的臂，在药物发现、A/B 测试、众包、数据库调优等领域有广泛应用。

**实际场景中需要约束**：在药物发现中，找到的最优药物不仅要药效最强，还需要毒性和溶解性低于安全阈值；数据库调优中，要最小化延迟的同时保证系统故障风险够低。

**性能和约束可以分开测试**：在药物发现中，药效、溶解性、毒性的实验可以独立进行，不需要每次都同时测试所有指标。但现有约束 BAI 工作（Katz-Samuels and Scott 2018, 2019）假设每次拉臂时同时观察所有指标，这不贴合实际且采样效率低。

**朴素策略都不够好**："先测可行性再比性能"对性能差距大但可行性难判的臂浪费采样；"先比性能再测可行性"对明显不可行但性能接近最优的臂浪费采样。

**核心洞察**：对每个次优臂应选择更容易的方式淘汰——性能远差就用性能淘汰，明显不可行就用可行性淘汰，算法需要自适应地做出选择。

## 方法详解

### 整体框架

给定 $K$ 个臂，每个臂关联 $N+1$ 个分布：$\nu_{i,0}$ 为性能分布，$\nu_{i,\ell}$（$\ell \in [N]$）为第 $\ell$ 个可行性约束分布。臂 $i$ 可行当且仅当 $\mu_{i,\ell} < \frac{1}{2}$ 对所有 $\ell \in [N]$ 成立。在每轮中，决策者选择 $(i, \ell)$ 得到一个来自 $\nu_{i,\ell}$ 的样本，目标是以至少 $1-\delta$ 的概率找到可行集合中性能最高的臂，并最小化总采样数。

算法基于 LUCB 框架扩展，维护如下关键集合：

- $S$：存活臂集合（未淘汰的臂）
- $P$：焦点集合（$S$ 中性能置信区间较高的臂）
- $F$：已确认可行的臂集合
- $I$：已确认不可行的臂集合
- $H_i$：臂 $i$ 尚未确认可行的约束子集

### 关键设计一：双维度复杂度与臂分类

为每个臂 $i$ 定义可行性复杂度 $\theta_i$ 和性能复杂度 $\phi_i$。对于不可行臂，只需找到**一个**违反阈值的约束即可淘汰，因此 $\theta_i$ 取最大偏离阈值约束的逆平方；对于可行臂，需验证**所有**约束都满足，因此 $\theta_i$ 为所有约束偏离阈值逆平方之和：

$$\theta_i = \begin{cases} (\max_{\ell \in [N]} \mu_{i,\ell} - \frac{1}{2})^{-2}, & i \notin \mathcal{F} \\ \sum_{\ell=1}^{N} (\mu_{i,\ell} - \frac{1}{2})^{-2}, & i \in \mathcal{F} \end{cases}$$

性能复杂度则衡量与最优臂 $i^\star$ 的差距——性能优于 $i^\star$ 但不可行的臂无法通过性能淘汰（$\phi_i = \infty$），只能通过可行性淘汰：

$$\phi_i = \begin{cases} \infty, & i < i^\star \\ (\mu_{i^\star,0} - \mu_{i,0})^{-2}, & i > i^\star \end{cases}$$

由此将次优臂划分为 $\mathcal{I}$（通过不可行性淘汰更容易，$\theta_i < \phi_i$）和 $\mathcal{W}$（通过性能比较淘汰更容易），问题总复杂度 $\mathcal{H} = \sum_{i=1}^{K} \mathcal{H}_i$，对最优臂需同时验证其可行性和性能领先。

### 关键设计二：SAMPLE-FEASIBILITY 子程序

当需要测试臂 $i$ 的可行性时，需在 $N$ 个约束中智能选择一个。受 Kano et al. (2019) 启发，选择使 UCB 最大的约束：

$$\ell_t = \arg\max_{\ell \in H_i} \tilde{\mu}_{i,\ell}(t), \quad \tilde{\mu}_{i,\ell}(t) = \hat{\mu}_{i,\ell}(t) + \sqrt{\frac{2 \log M_i(t)}{N_{i,\ell}(t)}}$$

其中 $M_i(t)$ 为臂 $i$ 总可行性测试次数。直觉上优先采样最可能超过阈值 $\frac{1}{2}$ 的约束。采样后：若该约束的下置信界 $> \frac{1}{2}$ → 确认臂不可行，加入 $I$；若上置信界 $< \frac{1}{2}$ → 该约束通过验证，从 $H_i$ 移除；若 $H_i$ 清空 → 臂确认可行，加入 $F$。

### 关键设计三：LUCB 风格的主循环采样

每个 epoch 选择最多两个臂测试性能：

- $a_t = \arg\max_{i \in P} \hat{\mu}_{i,0}(t)$：焦点集中经验均值最高者
- $b_t = \arg\max_{i \in P \setminus \{a_t\}} \bar{\mu}_{i,0}(t)$：焦点集中性能上界最高的次优者

测试完性能后，若 $a_t$ 或 $b_t$ 未在 $F$ 中，则各做一次 SAMPLE-FEASIBILITY 调用。当 $P$ 退化为单元素 $\{i\}$ 时：若 $i \in F$ 则返回 $i$；否则持续测其可行性。当 $S = \emptyset$ 时返回"无可行臂"。

epoch 末尾的更新策略实现了双重淘汰：(1) 从 $S$ 中移除所有不可行臂 $I$；(2) 若 $F \neq \emptyset$，淘汰性能上界低于已知可行臂性能下界的臂。焦点集 $P$ 更新为性能上界高于存活集中性能下界最大值的臂。

### 理论结果

**下界**：对任何 $\delta$-正确算法，期望采样复杂度满足 $\mathbb{E}[\tau] \geq 2\mathcal{H} \log \frac{1}{2.4\delta}$，通过构造精妙的替代 bandit 实例证明。

**上界**：算法的期望停止时间在 $\delta \to 0$ 时满足 $\limsup_{\delta \to 0} \frac{\mathbb{E}[\tau]}{\log(1/\delta)} \in \widetilde{\mathcal{O}}(\mathcal{H})$，渐近匹配下界。上界证明的核心是对停止时间 $\tau$ 按淘汰方式和选择规则做多层分解。

## 实验关键数据

### 合成实验（$K=5, N=3$, 高斯分布, $\delta=0.1$）

以本文算法采样数为 1.0 基准，各算法相对采样量：

| 算法 | 类型 a（不可行臂多） | 类型 b（可行但性能差的臂多） | 类型 c（混合） |
|------|--------|--------|--------|
| **Ours** | **1.0** | **1.0** | **1.0** |
| F-first | 0.58 | 3.13 | 4.00 |
| P-first | 3.09 | 0.84 | 3.47 |
| TF-LUCB-C | 1.77 | 1.98 | 2.78 |
| Naive | 1.81 | 4.75 | 4.06 |

F-first 在类型 a 更优（0.58x），P-first 在类型 b 更优（0.84x），但只有本文算法在所有类型中都接近最优，尤其在混合类型 c 中大幅领先。

### 扩展性实验

- **约束数 $N$ 从 2 增加到 64**：本文算法采样量几乎不随 $N$ 增长，因为不可行臂只需找到一个违反约束；其他同时采样所有约束的方法线性增长。
- **臂数 $K$ 从 4 增加到 64**：本文算法始终优于所有基线。
- **$\delta$ 从 $10^{-1}$ 降至 $10^{-9}$**：采样量随 $\log(1/\delta)$ 线性增长，符合理论预测。

### 药物发现真实数据

数据来自类风湿性关节炎的 secukinumab 临床试验（Genovese et al. 2013），5 个不同剂量为 5 个臂，3 个指标（药效、不良事件概率、感染概率），需找到药效最高且安全的剂量。本文算法在所有 $\delta$ 值下均优于基线。

## 亮点与洞察

1. **新颖的问题形式化**：将性能测试和可行性约束测试分离，准确建模了药物发现、数据库调优等场景中实验可独立进行的特点。
2. **精巧的自适应淘汰**：通过 $\theta_i$ 和 $\phi_i$ 的比较自动决定每个臂的最优淘汰策略，无需预先知道问题结构。
3. **匹配的上下界**：同时给出信息论下界和算法上界，在 $\delta \to 0$ 时渐近最优，理论贡献完整。
4. **约束数可扩展**：采样复杂度不随 $N$ 线性增长，因为判定不可行只需找到一个违反约束。

## 局限与展望

1. **非渐近间隙**：当 $N > 1$ 且 $\delta \not\to 0$ 时，$\mathcal{I}$ 中臂的上下界存在间隙，源于探索成本无法完全消除，缩紧此间隙仍是开放问题。
2. **同一阈值假设**：假设所有约束阈值均为 $\frac{1}{2}$，虽可通过缩放处理不同阈值，但更一般的非均匀阈值设定更自然。
3. **分布假设较强**：假设 1-sub-Gaussian 分布，推广到重尾或非参数分布需额外工作。
4. **实验规模有限**：合成实验 $K=5$ 规模偏小，虽然扩展性实验到了 64，但缺少超大规模场景验证。
5. **未利用约束间相关性**：各约束独立测试，未考虑约束之间可能存在的统计相关性。

## 相关工作与启发

1. **LUCB 算法**（Kalyanakrishnan et al. 2012）：提供了本文主循环的采样策略基础。
2. **阈值 bandit**（Locatelli et al. 2016）：识别所有均值超过阈值的臂，本文只需找到一个违反约束即可判定不可行。
3. **Good Arm Identification**（Kano et al. 2019）：启发了 SAMPLE-FEASIBILITY 中的约束选择策略。
4. **贝叶斯约束优化**（Gardner et al. 2014）：同样允许分别评估目标和约束，但处理连续搜索空间而非 $K$-armed 设定。
5. **启发**：分离测试的思想可推广到多阶段临床试验（先安全性筛选再有效性测试）、多目标 A/B 测试等序贯决策场景。

## 评分

⭐⭐⭐⭐ (4/5)

**优势**：问题形式化新颖且贴合实际，理论结果完整（渐近最优上下界匹配），自适应双重淘汰机制设计精巧。

**不足**：实验规模偏小，sub-Gaussian 假设限制了适用范围，非渐近情形的最优性仍为开放问题。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Constrained Discrete Diffusion](../../NeurIPS2025/computational_biology/constrained_discrete_diffusion.md)
- [\[ICLR 2026\] Verifier-Constrained Flow Expansion for Discovery Beyond the Data](../../ICLR2026/computational_biology/verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)
- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](../../ICML2026/computational_biology/constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)
- [\[ACL 2025\] Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification](../../ACL2025/computational_biology/retrieve_to_explain_drug_target_identification.md)
- [\[AAAI 2026\] On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples](on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini.md)

</div>

<!-- RELATED:END -->
