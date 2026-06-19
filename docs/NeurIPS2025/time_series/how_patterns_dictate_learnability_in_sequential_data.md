---
title: >-
  [论文解读] How Patterns Dictate Learnability in Sequential Data
description: >-
  [NeurIPS 2025][时间序列][预测信息] 提出基于预测信息（predictive information）的信息论框架来量化序列数据中时间模式的强度，推导出将预测信息与最小可达风险联系起来的理论界，从而区分"模型不够好"还是"数据本身就不可预测"。 领域现状： 自回归模型在时间序列和 NLP 领域广泛应用…
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "预测信息"
  - "互信息"
  - "可学习性"
  - "序列数据"
  - "信息论"
---

# How Patterns Dictate Learnability in Sequential Data

**会议**: NeurIPS 2025  
**arXiv**: [2510.10744](https://arxiv.org/abs/2510.10744)  
**代码**: [https://github.com/EkMeasurable/Learnability_Ipred](https://github.com/EkMeasurable/Learnability_Ipred)  
**领域**: 时间序列  
**关键词**: 预测信息, 互信息, 可学习性, 序列数据, 信息论

## 一句话总结

提出基于预测信息（predictive information）的信息论框架来量化序列数据中时间模式的强度，推导出将预测信息与最小可达风险联系起来的理论界，从而区分"模型不够好"还是"数据本身就不可预测"。

## 研究背景与动机

**领域现状**: 自回归模型在时间序列和 NLP 领域广泛应用，性能依赖于捕获数据中的时间模式。但识别这些模式仍严重依赖人类专家经验。

**现有痛点**: 
   - 当模型在某数据集上表现不佳时，无法区分是模型能力不足还是数据本身熵太高（不可预测）
   - EvoRate 等度量只关注数值变化趋势，其绝对值难以解释
   - 现有泛化误差界（Rademacher 复杂度等）主要关注经验风险和真实风险的差距，而非最小可达风险

**核心矛盾**: 好的预测需要两个条件——(1) 数据包含可利用的模式；(2) 模型能识别和利用这些模式。现有框架缺乏区分这两者的工具。

**本文目标**: (i) 序列数据的最小可达风险是什么？(ii) 能否区分性能差是因为模型不好还是数据不可预测？

**切入角度**: 从**预测信息** $\mathbf{I}(X_{\text{past}}; X_{\text{future}})$——过去与未来之间的互信息入手，建立信息论学习曲线。

**核心 idea**: 用预测信息的离散导数（通用学习曲线 $\Lambda(k)$）作为"增加一步历史能减少多少不确定性"的度量，将其与自回归模型的最小可达风险联系起来。

## 方法详解

### 整体框架

定义预测信息:

$$\mathbf{I}_{\text{pred}}(k, k') = \mathbf{I}(\mathbf{X}_{t-k+1}^{t}; \mathbf{X}_{t+1}^{t+k'})$$

通用学习曲线:

$$\Lambda(k) = \ell(k) - \ell_0 = H(X_t | \mathbf{X}_{t-k+1}^{t-1}) - \lim_{k \to \infty} H(X_t | \mathbf{X}_{t-k+1}^{t-1})$$

$\Lambda(k)$ 衡量观测 $k$ 步历史相比观测无穷历史多出的不确定性。

### 关键设计

#### 1. 预测信息作为 EvoRate 的推广

**功能**: 将 EvoRate（只看过去 $k$ 步与下一步的互信息）推广为过去 $k$ 步与未来 $k'$ 步的互信息。

**核心关联** (Proposition 4.1): $\mathbf{I}_{\text{pred}}(k+1, k') - \mathbf{I}_{\text{pred}}(k, k') \to \Lambda(k)$ 当 $k' \to \infty$，即预测信息的离散差分收敛到通用学习曲线。

**设计动机**: EvoRate 的绝对值难以解释，学习曲线提供了更有原则的替代——它是预测信息的"离散导数"。

#### 2. Markov 过程的闭式表达（Proposition 4.2）

对 $m$ 阶 Markov 过程：当 $k \geq m$ 时 $\Lambda(k) = 0$，即学习曲线在真实阶数处归零。

**意义**: 可用于识别 Markov 阶数——比 AIC/BIC 更有信息论原则性。

#### 3. 参数模型的渐近行为（Theorem 4.3）

对 $p$ 维参数族生成的平稳弱依赖过程：

$$\mathbf{I}_{\text{pred}}(k, k') \underset{k \to \infty}{=} \frac{p}{2} \ln(k) + \frac{1}{2} \ln \det(F) + \mathcal{O}(1)$$

推论 (Corollary 4.4): $\Lambda(k) \sim \frac{p}{2k}$，因此 $\dim \Theta \approx 2k \Lambda(k)$。

**应用**: 可从学习曲线衰减速率估计参数空间维度。

#### 4. 学习曲线与最小可达风险的联系（Proposition 4.5, 核心结果）

对任意阶 $k$ 的模型 $Q \in \mathcal{H}_k$:

$$\mathcal{R}^\infty(Q^*) \leq \mathcal{R}^k(Q) - \Lambda(k)$$

其中 $\mathcal{R}^\infty(Q^*)$ 是最优预测器的最小可达风险。

**含义**: 
- 除非 $\Lambda(k)$ 可忽略，否则有限记忆模型无法达到最优风险
- $\Lambda(k)$ 量化了因有限上下文导致的不可约不确定性
- 可用经验风险减去估计的 $\Lambda(k)$ 来估计最优风险

#### 5. Oracle 估计器（Corollary 4.7）

给定不同阶 $k$ 的训练模型 $Q_k$:

$$\hat{\mathcal{R}}^\infty(Q^*) = \min_{1 \leq k \leq M} \{\hat{\mathcal{R}}^k(Q_k) - \Lambda(k)\}$$

同时可推断最优回归阶数 $k^*$。

### 损失函数/训练策略

使用 MLE 损失 $\mathcal{L}_{\text{mle}}^k = -\mathbb{E} \ln Q(X_{t+1} | \mathbf{X}_{t-k+1}^t)$ 作为风险衡量。

$\mathbf{I}_{\text{pred}}$ 的估计使用 InfoNCE、MINE、SMILE 等神经网络互信息估计器。

## 实验关键数据

### 高斯过程上的 $\mathbf{I}_{\text{pred}}$ 估计

| 维度 | 核类型 | InfoNCE 偏差 | MINE 偏差 | SMILE 偏差 |
|------|--------|-------------|----------|-----------|
| $d \leq 20$ | 各类核 | 较小 | 较小 | 较小 |
| $d > 20$ | Periodic | 增大 | 增大 | 显著低估 |
| $d > 20$ | RQ | 增大 | 增大 | 显著低估 |

低维设定下各方法估计准确，高维复杂核下性能退化。

### AR 过程的学习曲线

| 真实阶数 $p$ | 设定 | $\hat{\Lambda}(k)$ 归零位置 | 与理论曲线吻合 |
|------------|------|--------------------------|-------------|
| $p = 5$ | 3维 VAR | $k \approx 5$ | 良好 |
| $p = 10$ | 3维 VAR | $k \approx 10$ | 良好 |

学习曲线成功识别了正确的 Markov 阶数。

### Ising 自旋序列上的最小风险估计

| 块大小 $M$ | EvoRate(10) | $\hat{\mathcal{R}}^\infty_{\text{lstm}}$ | $\min \hat{\mathcal{R}}_{\text{lstm}}$ | 比值 |
|-----------|------------|----------------------------------------|--------------------------------------|------|
| 10,000 | 0.276 | 0.372 | 0.485 | ~1.3 |
| 100,000 | 0.286 | 0.368 | 0.468 | ~1.3 |
| 1,000,000 | 0.327 | 0.357 | 0.380 | ~1.06 |
| 10,000,000 | 0.476 | 0.070 | 0.073 | ~1.04 |

### 关键发现

1. $M = 10^7$ 时过程退化为一阶 Markov 链，EvoRate 最高（结构最强），模型风险接近 oracle 估计
2. $M$ 较小时 $\hat{\mathcal{R}}/\hat{\mathcal{R}}^\infty \approx 1.3$，说明模型未达最优——存在未利用的预测性结构
3. Oracle 估计在 LSTM 和 MLP 之间一致性好，仅有微小差异
4. 参数维度估计 $\hat{p} = 2k\Lambda(k) \approx 0.96$（真值为1），恢复准确

## 亮点与洞察

- **解决一个根本问题**: 区分"模型不行"和"数据就是不可预测"，这对实际建模决策极有价值
- **信息论与机器学习的桥梁**: 将 Bialek & Tishby 的理论物理概念（predictive information）转化为机器学习的实用诊断工具
- **参数维度估计**: $\dim \Theta \approx 2k\Lambda(k)$ 这个推论非常优雅，直接从学习曲线读出参数空间大小
- **与 EvoRate 的联系**: 清晰地展示了预测信息如何推广 EvoRate 并赋予其更强的理论基础

## 局限与展望

1. 实验仅在合成数据上验证，未在真实时间序列基准（如 Exchange、ETTh 等）上测试
2. 高维下互信息估计不稳定（MINE、SMILE 等），限制了框架的实际应用范围
3. Oracle 估计器是模型依赖的，不同模型族可能给出不同的最优风险估计
4. 负的 $\Lambda(k)$ 估计值（如 $M = 10^7$ 时）源于估计不稳定性，需要更好的估计器
5. 要求平稳性假设 $(\mathbf{H}_0)$，非平稳序列需要扩展

## 相关工作与启发

- EvoRate (Zeng et al., 2025) 是本文的直接前驱，本文给出了更有原则的推广
- ForeCA (Goerg, 2013) 从频域角度衡量可预测性，与本文互补
- 前瞻学习框架 (Silva et al., 2025) 从不同角度定义了可学习性
- 思路可扩展到多步预测、非平稳过程、因果推断等方向

## 评分

⭐⭐⭐⭐

信息论视角新颖且有理论深度，核心命题（最小风险与学习曲线的联系）有较强指导意义。但实验局限于合成数据，高维互信息估计的不稳定性限制了实用性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)
- [\[ICML 2026\] DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](../../ICML2026/time_series/distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)

</div>

<!-- RELATED:END -->
