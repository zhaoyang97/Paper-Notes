---
title: >-
  [论文解读] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications
description: >-
  [ICML2025][学习理论][转导学习] 提出转导局部复杂度（TLC）框架，将经典的局部 Rademacher 复杂度扩展到转导学习设定，获得了与归纳学习几乎一致的超额风险界（仅差对数因子），并解决了十年未决的开放问题。 转导学习（transductive learning）中，学习器同时拥有带标签的训练数据和无标签的测…
tags:
  - "ICML2025"
  - "学习理论"
  - "转导学习"
  - "泛化界"
  - "局部Rademacher复杂度"
  - "VC维"
  - "核学习"
  - "集中不等式"
  - "无放回抽样"
---

# Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications

**会议**: ICML2025  
**arXiv**: [2309.16858](https://arxiv.org/abs/2309.16858)  
**代码**: 无  
**领域**: 学习理论  
**关键词**: 转导学习, 泛化界, 局部Rademacher复杂度, VC维, 核学习, 集中不等式, 无放回抽样

## 一句话总结

提出转导局部复杂度（TLC）框架，将经典的局部 Rademacher 复杂度扩展到转导学习设定，获得了与归纳学习几乎一致的超额风险界（仅差对数因子），并解决了十年未决的开放问题。

## 研究背景与动机

转导学习（transductive learning）中，学习器同时拥有带标签的训练数据和无标签的测试数据，目标是预测测试数据的标签。获取紧致的泛化界是统计学习理论的核心问题。

**归纳学习中的经典结果**：局部 Rademacher 复杂度（LRC）能给出归纳学习中经验风险最小化器的尖锐超额风险界：

$$\mathcal{E}(\hat{f}) \leq \Theta\left(r^* + \frac{x}{n}\right)$$

其中 $r^*$ 是某经验过程的不动点，$n$ 是训练数据量。LRC 已在非参数回归、分类等任务中实现极小极大速率。

**转导学习中的困难**：已有最优结果 [Tolstikhin et al., 2014] 给出的超额风险界为：

$$\mathcal{E}(\hat{f}) \leq \Theta\left(\frac{n}{u}r_m^* + \frac{n}{m}r_u^* + \frac{1}{m} + \frac{1}{u}\right)$$

其中 $m, u$ 分别为训练和测试数据量，$n = m + u$。该界在 $m$ 或 $u$ 远小于 $n$ 时可能发散，与归纳学习界存在本质差距。根本原因在于缺乏有效的无放回抽样经验过程上确界的集中不等式。

**核心开放问题**：能否构建一种基于局部复杂度的转导学习框架，使超额风险界匹配或接近归纳学习的结果？

## 方法详解

### 1. 问题设定

给定全样本 $\mathbf{X}_n = \{\vec{\mathbf{x}}_i\}_{i=1}^n$，训练特征 $\mathbf{X}_m$ 从 $\mathbf{X}_n$ 中均匀无放回抽取 $m$ 个，剩余 $u$ 个为测试特征 $\mathbf{X}_u$。定义测试-训练过程（test-train process）：

$$g(\mathcal{H}) \coloneqq \sup_{h \in \mathcal{H}} \left(\mathcal{L}_u(h) - \mathcal{L}_m(h)\right)$$

### 2. 转导复杂度（TC）

定义四种转导复杂度，以 $\mathfrak{R}_u^+$ 为例：

$$\mathfrak{R}_u^+(\mathcal{H}) \coloneqq \mathbb{E}\left[\sup_{h \in \mathcal{H}} R_u^+ h\right], \quad R_u^+ h \coloneqq \mathcal{L}_u(h) - \mathcal{L}_n(h)$$

关键性质：TC 可被归纳 Rademacher 复杂度上界控制（Theorem 3.1）：

$$\max\{\mathfrak{R}_u^+(\mathcal{H}), \mathfrak{R}_u^-(\mathcal{H})\} \leq 2\mathfrak{R}_u^{(\text{ind})}(\mathcal{H})$$

### 3. 测试-训练过程的集中不等式（核心技术贡献）

**Theorem 4.1**：对有界函数类 $\mathcal{H}$（$|h(i)| \leq H_0$），以概率至少 $1 - e^{-x} - \delta$：

$$g(\mathcal{H}) \leq \mathbb{E}[g(\mathcal{H})] + 4\sqrt{\frac{10rx}{N_{u,m,\delta}}} + 2\sqrt{2}\inf_{\alpha > 0}\left(\frac{\mathfrak{R}_{\min\{u,m\}}^+(\mathcal{H}^2)}{\alpha} + \frac{\alpha x}{N_{u,m,\delta}}\right) + \frac{4H_0^2 x}{N_{u,m,\delta}}$$

其中 $N_{u,m,\delta} = \frac{\min\{u,m\}}{\log_2(4\min\{u,m\}/\delta)}$。

**证明核心创新**：

- 发现测试-训练过程的**组合性质**（Lemma 5.4）：每个损失函数 $h$ 的测试-训练损失变化始终是一对元素之差
- **两次应用指数 Efron-Stein 不等式**的新证明策略：先推导上方差 $V_+(g)$ 的上界，再用该上界与 $\mathcal{H}^2$ 上的经验过程协同推导集中不等式
- 通过 RANDPERM 算法将无放回抽样分解为独立随机变量的采样

### 4. TLC 超额风险界（主定理）

**Theorem 4.11**（核心结果）：在标准假设下，以概率至少 $1 - 3e^{-x} - 3\delta$：

$$\mathcal{E}(\hat{f}_m) \leq c_0 r_{u,m} + \frac{4Bc_2 r^*}{K_0} + \frac{c_3 x}{N_{u,m,\delta}}$$

其中 $r_{u,m}$ 和 $r^*$ 是子根函数的不动点，在标准学习模型下均以快速率收敛至 0。代理方差算子定义为：

$$\tilde{T}_n(h) \coloneqq \inf_{f_1,f_2 \in \mathcal{F}: \ell_{f_1} - \ell_{f_2} = h} 2B\mathcal{L}_n(\ell_{f_1} - \ell_{f_n^*}) + 2B\mathcal{L}_n(\ell_{f_2} - \ell_{f_n^*})$$

## 理论结果

| 结果 | 已有最优界 | 本文界 | 改进 |
|------|-----------|--------|------|
| 通用超额风险界 | $\Theta(\frac{n}{u}r_m^* + \frac{n}{m}r_u^* + \frac{1}{m} + \frac{1}{u})$（可发散） | $\Theta(r_{u,m} + r^* + \frac{\log_2(\min\{u,m\}/\delta) \cdot x}{\min\{u,m\}})$（始终收敛） | 消除 $n/u, n/m$ 因子 |
| 有限 VC 维可实现转导学习 | $\Theta(\frac{d^{(\text{VC})} \log(ne/d^{(\text{VC})})}{m})$（$\log n$ 间隙） | $\Theta(\frac{d^{(\text{VC})} \log(me/d^{(\text{VC})})}{m})$（$\log m$ 间隙） | $\log n \to \log m$，解决十年开放问题 |
| 极小极大下界匹配 | 与下界 $\Theta(d^{(\text{VC})}/m)$ 差 $\log n$ | 与下界差 $\log m$ | 几乎最优 |
| 转导核学习 | 在 $m = o(\sqrt{n})$ 或 $u = o(\sqrt{n})$ 时发散 | 无此限制，始终收敛 | 显著改进 |
| 集中不等式（Corollary 4.3） | [Tolstikhin 2014] 的两版均在某些 $u/n$ 比例下发散 | 在标准模型下始终收敛至 0 | 统一优于两版已有结果 |

## 亮点与洞察

1. **解决十年开放问题**：将有限 VC 维可实现转导学习的对数间隙从 $\log n$ 缩小到 $\log m$，与极小极大下界仅差 $\log m$ 因子
2. **统一框架**：TLC 框架下的结果涵盖了 [Yang, 2022] 在不平衡体制（$m \gg u^2$ 或 $u \gg m^2$）下的结果作为特例，且不受此限制
3. **技术创新深刻**：两次应用指数 Efron-Stein 不等式的策略是处理无放回抽样复杂度的全新方法
4. **与归纳学习的桥梁**：Theorem 3.1 建立了转导复杂度与归纳 Rademacher 复杂度的对称化不等式关系
5. **代理方差算子**的引入为 peeling 策略提供了恰当的局部化度量

## 局限与展望

1. **对数间隙未完全消除**：与极小极大下界仍差 $\log m$ 因子，最优界的完全紧致性仍待解决
2. **纯理论工作**：缺乏实验验证，未在实际转导学习算法上测试界的紧致性
3. **技术条件**：需要 $H_0 \geq 2\sqrt{2}$ 的技术假设（虽可通过重定义 $H_0$ 满足）
4. **Assumption 1 的适用范围**：虽为标准假设，但要求损失函数的特定凸性和 Lipschitz 条件
5. **计算可行性**：不动点 $r_{u,m}, r^*$ 的实际计算在复杂函数类上可能困难

## 相关工作与启发

- **[Tolstikhin et al., 2014]**：转导核学习的局部复杂度分析，本文的直接改进对象
- **[Bartlett et al., 2005]**：归纳学习中 LRC 的经典工作，本文将其扩展到转导设定
- **[Darnstädt et al., 2013]**：有限 VC 维转导学习的现有最优界，本文改进其对数因子
- **[Boucheron et al., 2005]**：指数 Efron-Stein 不等式，本文的核心技术工具
- **[El-Yaniv and Pechyony, 2009]**：转导学习的 Rademacher 复杂度框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 全新的 TLC 框架，解决十年开放问题
- 实验充分度: ⭐⭐ — 纯理论工作，无实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，主要结果摘要明确，但证明细节较重
- 价值: ⭐⭐⭐⭐⭐ — 学习理论领域的重要推进，建立转导与归纳学习的紧密联系

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](improved_and_oracle-efficient_online_ell_1-multicalibration.md)
- [\[ICLR 2026\] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs](../../ICLR2026/learning_theory/an_improved_model-free_decision-estimation_coefficient_with_applications_in_adve.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/learning_theory/kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[NeurIPS 2025\] The Structural Complexity of Matrix-Vector Multiplication](../../NeurIPS2025/learning_theory/the_structural_complexity_of_matrix-vector_multiplication.md)
- [\[NeurIPS 2025\] The Parameterized Complexity of Computing the VC-Dimension](../../NeurIPS2025/learning_theory/the_parameterized_complexity_of_computing_the_vc-dimension.md)

</div>

<!-- RELATED:END -->
