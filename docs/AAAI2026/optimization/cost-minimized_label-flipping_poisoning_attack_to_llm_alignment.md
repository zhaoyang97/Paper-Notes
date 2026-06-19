---
title: >-
  [论文解读] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment
description: >-
  [AAAI 2026][优化/理论][数据投毒攻击] 首次从理论上分析了在 RLHF/DPO 对齐过程中，通过翻转偏好标签来引导 LLM 策略走向攻击者目标所需的最小成本，将其形式化为凸优化问题并推导了成本的上下界，进而提出 PCM（Poisoning Cost Minimization）后处理方法，可在保持投毒效果的同时显著减少标签翻转数量。
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "数据投毒攻击"
  - "RLHF/DPO"
  - "标签翻转"
  - "凸优化"
  - "LLM 对齐安全"
---

# Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment

**会议**: AAAI 2026  
**arXiv**: [2511.09105](https://arxiv.org/abs/2511.09105)  
**代码**: [https://github.com/akimotolab/PoisoningCostMinimization](https://github.com/akimotolab/PoisoningCostMinimization)  
**领域**: 优化  
**关键词**: 数据投毒攻击, RLHF/DPO, 标签翻转, 凸优化, LLM 对齐安全

## 一句话总结

首次从理论上分析了在 RLHF/DPO 对齐过程中，通过翻转偏好标签来引导 LLM 策略走向攻击者目标所需的最小成本，将其形式化为凸优化问题并推导了成本的上下界，进而提出 PCM（Poisoning Cost Minimization）后处理方法，可在保持投毒效果的同时显著减少标签翻转数量。

## 研究背景与动机

随着 LLM 在现实系统中的广泛部署，理解其脆弱性对安全使用至关重要。LLM 的多阶段训练流程（预训练→SFT→RLHF/DPO 对齐）使其容易受到数据投毒攻击。

现有研究已经在经验上证明了 RLHF/DPO 阶段的投毒攻击可行性，但存在关键空白：**理论基础几乎未被探索**。具体来说：
- 攻击者作为标注者只能翻转偏好标签 $w$（从"preferred"变为"not preferred"或反之），不能修改上下文 $x$ 或候选输出 $(y, z)$
- 核心问题是：**将 LLM 的最优策略引导到攻击者目标策略所需的最小标签翻转数量是多少？**
- 理论理解对于确定受害者的最坏情况至关重要，这是经验研究无法完全揭示的

从防御视角看，量化这些最小成本可以指导设计更稳健的 RLHF/DPO 管道，检测或缓解低成本投毒攻击。

## 方法详解

### 整体框架

将标签翻转投毒攻击形式化为一个带线性约束的凸优化问题，推导最小攻击成本的上下界。基于此理论分析，提出 PCM 后处理方法：给定任何现有攻击生成的恶意数据集，通过求解凸优化问题找到需要更少标签翻转但产生相同投毒效果的替代数据集。

### 关键设计

#### 1. **问题形式化**: 将投毒成本最小化转化为凸优化问题

**核心设定**：受害者拥有未标注数据集 $\mathcal{D}_U = \{(x_i, y_i, z_i)\}_{i=1}^N$，标签 $w_i$ 由外部标注者提供。奖励模型采用线性形式 $r(x,y) = \mathbf{r}^\top \phi(x,y)$，其中 $\phi$ 是预训练 LLM 去掉最后嵌入层后的特征提取器。

攻击者的成本定义为翻转标签的数量，用 $\ell_1$ 范数度量：

$$\sum_{i} |\eta(x_i,y_i,z_i) - \eta_A(x_i,y_i,z_i)|$$

攻击者的优化问题：

$$\min_\theta \|\theta - \theta_O\| \quad \text{s.t.} \quad \arg\min_r \mathcal{L}(r;\theta) \subseteq \mathcal{R}(\pi_{r_A}), \quad \|2\theta - \mathbf{1}\|_\infty \leq 1$$

**Theorem 1**：在 Assumption 1（特征矩阵列空间覆盖条件）下，上述问题等价为凸优化问题：

$$\min_\zeta \|\zeta\| \quad \text{s.t.} \quad \Phi\zeta = \Phi(\theta_A - \theta_O), \quad -\theta_O \leq \zeta \leq (\mathbf{1} - \theta_O)$$

其中 $\Phi$ 是 $n \times N$ 维矩阵，第 $i$ 列为 $\phi(x_i,y_i) - \phi(x_i,z_i)$。当使用 $\ell_1$ 成本时，可进一步转化为**线性规划**问题。

**设计动机**：不同的奖励函数可能导致相同的最优策略（$r(x,y)$ 和 $r(x,y)+R(x)$ 对任意 $R: \mathcal{X} \to \mathbb{R}$ 具有相同最优策略），这意味着攻击者可以利用特征空间中的冗余来减少攻击成本。

#### 2. **成本上下界推导**: 揭示 RLHF/DPO 管道的基本脆弱性

**下界（Theorem 2）**：

$$\text{最小成本} \geq \frac{\|(\Phi^\dagger\Phi)(\theta_A - \theta_O)\|_2^2}{\|(\Phi^\dagger\Phi)(\theta_A - \theta_O)\|_*}$$

**上界（Theorem 3）**：

$$\text{最小成本} \leq \min\left\{\left\|\left(\frac{\alpha^* I + \bar{\alpha}\Phi^\dagger\Phi}{\alpha^* + \bar{\alpha}}\right)(\theta_A - \theta_O)\right\|, \|\theta_A - \theta_O\|\right\}$$

**关键洞察（Proposition 4）**：特征维度 $n$ 越大的模型，对标签翻转攻击的鲁棒性越强。当 $n \ll N$（特征维度远小于数据量）时，攻击成本可以被大幅降低。

直觉解释：$\Phi^\dagger\Phi$ 将 $\mathbb{R}^N$ 投影到 $\Phi$ 行空间（维度至多为 $n$）的子空间。当 $n$ 远小于 $N$ 时，投影后的向量远小于原始向量，意味着攻击者可以利用数据冗余以极低成本实现目标。

#### 3. **自适应嵌入情况的扩展分析**: 考虑整个模型（包括嵌入层）被训练

当嵌入 $\phi_\omega$ 也被训练时（如 DPO 全参数微调），攻击者有更多自由度：

$$\min_\zeta \|\zeta\| \quad \text{s.t.} \quad \mathbf{r}_A^\top\Phi_{\omega_A}\zeta = \mathbf{r}_A^\top\Phi_{\omega_A}(\theta_A - \theta_O), \quad -\theta_O \leq \zeta \leq (\mathbf{1} - \theta_O)$$

**Theorem 6**：如果嵌入的表达能力足够强（存在 $\bar{\omega}$ 使得 $\text{col}(\Phi_{\bar{\omega}}) = \{\mathbf{r}_A^\top\Phi_{\omega_A}\}$），问题简化为**只有一个线性等式约束**的优化，攻击成本可能被进一步降低。

这揭示了一个重要的安全隐患：自适应嵌入场景下的攻击不会比固定嵌入更难，甚至可能更容易。

### PCM 后处理方法

**实际操作流程**：

1. 给定目标偏好概率向量 $\theta_A$（由任何现有攻击生成）
2. 求解凸优化问题 (10) 获得成本最小化向量 $\theta_A^*$
3. 将 $\theta_A^*$ 离散化（四舍五入到 $\Theta_m$ 中的值）
4. 按照离散化后的向量翻转偏好标签

PCM 与生成 $\theta_A$ 的方式无关，可以叠加在任何标签翻转攻击之上。

## 实验关键数据

### 合成数据实验

实验设置：从标准正态分布生成嵌入 $\phi$，原始标签全为偏好（$\theta_O = \mathbf{1}$），比较随机翻转攻击和 RLHFPoison 攻击在 PCM 后处理前后的成本。

关键发现：
- 当 $N \gtrsim 5n$（数据量大于 5 倍特征维度）时，PCM 开始显著降低攻击成本
- 随机翻转攻击的成本降低效果与数据量 $N$ 成线性关系
- 离散化粒度 $m$ 对成本影响不大，但更大的 $m$ 可实现更小的性能损失率
- 即使 $m=1$（每个数据点仅一个标注），当 $N$ 足够大时性能损失率可低至 0.1

### 主实验 - 公开 LLM 和数据集

| 配置 | RLHFPoison 输出长度增加率 | RLHFPoison+PCM 输出长度增加率 | 成本降低 |
|------|--------------------------|-------------------------------|----------|
| PKU-SafeRLHF + Phi-3.5-mini | 0.44±0.01 | 0.40±0.01 | -13.4% |
| PKU-SafeRLHF + Llama-2-7b | 0.29±0.02 | 0.29±0.01 | -10.6% |
| PKU-SafeRLHF + Llama-2-13b | 0.25±0.01 | 0.37±0.01 | -8.2% |
| HH-RLHF + Phi-3.5-mini | 0.55±0.02 | 0.27±0.02 | -30.4% |
| HH-RLHF + Llama-2-7b | 1.08±0.36 | 0.87±0.05 | -29.8% |
| HH-RLHF + Llama-2-13b | 1.63±0.55 | 1.27±0.15 | -20.0% |

### 消融分析

| 因素 | 观察 | 说明 |
|------|------|------|
| 数据集大小 $N$ | $N$ 越大，成本降低越多 | HH-RLHF(N=160K)比PKU(N=73K)效果更好 |
| 特征维度 $n$ | $n$ 越小，攻击越容易 | Phi-3.5-mini(n=3072)比Llama-2-13b(n=5120)更脆弱 |
| $n > N$ 的情况 | 无成本降低 | social-reasoning-rlhf(N=3820)上无效 |
| 固定 vs 自适应嵌入 | PCM 在 DPO 训练下仍然有效 | 尽管理论假设固定嵌入 |

### 关键发现

1. **数据冗余是攻击成本降低的根本原因**：当 $N \gg n$ 时，特征空间存在大量冗余，攻击者可以找到多种标签翻转方案来实现相同的投毒效果
2. **理论下界与实际成本差距为 3-4 倍**，证明界的紧致性
3. **PCM 在自适应嵌入场景下仍有效**：虽然理论框架在固定嵌入假设下推导，但 DPO 全参数训练的实际场景中成本降低仍然显著
4. **更大的模型更鲁棒**：特征维度更大的模型（Llama-2-13b vs Phi-3.5-mini）需要更多标签翻转才能被攻击

## 亮点与洞察

1. **理论贡献突出**：首次为 RLHF/DPO 标签翻转攻击建立严格的理论框架，将攻击成本最小化问题转化为凸/线性规划
2. **实用价值双重**：既从攻击者角度提供了更高效的攻击（红队测试），又从防御者角度量化了安全风险
3. **关键不等式 $n \ll N$ 的安全启示**：揭示了现实 RLHF 管道中的根本脆弱性——由于偏好数据集规模通常远大于奖励模型的特征维度
4. **后处理方法 PCM 的通用性**：可以叠加在任何现有攻击方法之上，不依赖特定攻击策略

## 局限与展望

1. **理想化假设**：假设最优奖励模型恢复、攻击者精确知道奖励函数结构、直接修改偏好概率——实际中这些条件难以完全满足
2. **自适应嵌入分析不完整**：Theorem 6 的结果保守且过于悲观；现实中攻击成功与否依赖于受害者的优化算法和初始化
3. **未考虑防御策略**：虽然提到了安全风险，但未提出具体的防御方案
4. **标签翻转攻击模型限制**：只考虑翻转偏好标签，不考虑注入或修改数据三元组的更强攻击者
5. **实验规模有限**：最大模型仅 13B 参数，未在更大规模 LLM 上验证

## 相关工作与启发

- 与 wu2024preference、rlhfpoison 等经验攻击工作互补：为已知攻击提供了理论保证和成本优化
- 建立了标签翻转攻击（传统机器学习分类问题）与偏好学习投毒之间的联系
- 提出了一个重要的安全启示：在 RLHF 数据标注中，**允许不受信任的标注者标注 $k$ 个数据点**所带来的安全风险可能远大于直觉上 $k$ 个错误标签的影响

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个为 RLHF/DPO 投毒攻击建立理论框架的工作
- 实验充分度: ⭐⭐⭐⭐ — 合成数据验证理论 + 真实 LLM/数据集验证实用性，但规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导严谨，问题定义清晰
- 价值: ⭐⭐⭐⭐⭐ — 对 LLM 对齐安全有重要理论和实践意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/optimization/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](../../ICML2026/optimization/cost-aware_stopping_for_bayesian_optimization.md)
- [\[AAAI 2026\] Co-Layout: LLM-driven Co-optimization for Interior Layout](co-layout_llm-driven_co-optimization_for_interior_layout.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[NeurIPS 2025\] Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment](../../NeurIPS2025/optimization/clean_first_align_later_benchmarking_preference_data_cleaning_for_reliable_llm_a.md)

</div>

<!-- RELATED:END -->
