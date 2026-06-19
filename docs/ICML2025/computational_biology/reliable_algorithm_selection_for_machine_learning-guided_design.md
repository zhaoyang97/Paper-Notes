---
title: >-
  [论文解读] Reliable Algorithm Selection for Machine Learning-Guided Design
description: >-
  [ICML 2025][计算生物][算法选择] 提出一种设计算法选择方法，通过将候选设计算法配置的成功判定形式化为多重假设检验问题，结合预测驱动推断（Prediction-Powered Inference）技术校正预测误差，以高概率保证选出在未标注设计分布上满足用户定义成功准则的算法配置。 - ML引导设计的实际难题：蛋白…
tags:
  - "ICML 2025"
  - "计算生物"
  - "算法选择"
  - "ML引导设计"
  - "预测驱动推断"
  - "多重假设检验"
  - "蛋白质设计"
---

# Reliable Algorithm Selection for Machine Learning-Guided Design

**会议**: ICML 2025  
**arXiv**: [2503.20767](https://arxiv.org/abs/2503.20767)  
**代码**: [GitHub](https://github.com/clarafy/design-algorithm-selection)  
**领域**: 生物序列设计 / 统计推断  
**关键词**: 算法选择、ML引导设计、预测驱动推断、多重假设检验、蛋白质设计

## 一句话总结

提出一种设计算法选择方法，通过将候选设计算法配置的成功判定形式化为多重假设检验问题，结合预测驱动推断（Prediction-Powered Inference）技术校正预测误差，以高概率保证选出在未标注设计分布上满足用户定义成功准则的算法配置。

## 研究背景与动机

- **ML引导设计的实际难题**：蛋白质/RNA设计中，研究者需选择设计算法（如AdaLead、CbAS等）及其超参数和预测模型，这些选择直接决定设计效果
- **预测不可靠**：设计算法输出的序列通常偏离训练分布，预测模型在此分布外误差可能很大——高预测值不等于高真实标签
- **标注昂贵**：湿实验验证设计序列成本高昂（合成+测量），需在标注前就做出可靠的算法选择
- **现有方法不足**：
    - 仅看预测值：被预测误差误导
    - 校准预测：聚焦单个设计的不确定性，非直接服务于算法选择决策
    - 贝叶斯优化：目标是逐轮逼近全局最优，非保证当前轮次设计满足标准

**核心问题**：如何在不获取设计标签的前提下，选出满足用户定义成功准则的设计算法配置？

## 方法详解

### 整体框架

**将算法选择转化为多重假设检验**（Algorithm 1）：

1. 对菜单 $\Lambda$ 中每个配置 $\lambda$，生成 $N$ 个设计并用其预测模型获取预测值
2. 定义零假设 $H_\lambda$：配置 $\lambda$ 不成功，即 $\theta_\lambda := \mathbb{E}_{Y \sim P_{Y;\lambda}}[g(Y)] < \tau$
3. 计算每个零假设的 p 值（基于PPI）
4. Bonferroni校正后输出 $\hat{\Lambda} = \{\lambda \in \Lambda: p_\lambda \le \alpha/|\Lambda|\}$

**高概率保证**：$\mathbb{P}(\theta_\lambda \ge \tau, \forall \lambda \in \hat{\Lambda}) \ge 1-\alpha$

### 关键设计

**1. 成功准则的灵活定义**

$$\theta_\lambda = \mathbb{E}_{Y \sim P_{Y;\lambda}}[g(Y)] \ge \tau$$

- 均值设计标签：$g(y)=y$
- 超越率：$g(y)=\mathbf{1}[y \ge \gamma]$（如至少10%设计标签超过野生型）
- 用户可根据实际需求自定义 $g$ 和 $\tau$

**2. 预测驱动 p 值**（Algorithm 2）

核心思想：用留出标注数据校正仅基于预测的估计偏差

$$\hat{\theta} = \underbrace{\frac{1}{N}\sum_{i=1}^N g(\hat{y}_i^\lambda)}_{\text{预测部分 }\hat{\mu}} + \underbrace{\frac{1}{n}\sum_{j=1}^n w_j(g(y_j) - g(\hat{y}_j))}_{\text{偏差校正 }\hat{\Delta}}$$

其中 $w_j = p_{X;\lambda}(x_j)/p_{\text{lab}}(x_j)$ 是设计分布与标注数据分布的密度比

标准误：$\sigma^2 = \frac{\hat{\sigma}^2_{\text{pred}}}{N} + \frac{\hat{\sigma}^2_{\text{err}}}{n}$

p 值：$P = 1 - \Phi\left(\frac{\hat{\theta}-\tau}{\sigma}\right)$

**3. 密度比的角色**

- 留出标注数据需通过密度比重新加权，因为设计分布和标注数据分布之间存在协变量偏移
- **已知密度比**：如序列来自已知的组合文库（NNK library）或自回归生成模型
- **未知密度比**：用多类别 logistic 回归估计密度比（MDRE）

**4. 有限样本保证**（Algorithm 3 + Theorem 3.1）

通过 Hoeffding 不等式替代正态近似，获得非渐近有效的 p 值，保证定理3.1成立。

### 损失函数 / 训练策略

- 预测模型：岭回归、全连接NN集成、CNN集成（取决于任务）
- 训练数据：5k-10k标注序列
- 留出数据：5k标注序列用于PPI校正
- 设计序列：每配置采样 $N=50k$-$1M$ 个

## 实验关键数据

### 主实验

**实验一：蛋白质GB1结合亲和力设计**

- 设计空间：4个位点 × 20种氨基酸 = $20^4=160,000$ 个变体（全标注可用）
- 菜单：101个温度超参数 $\lambda \in [0.2, 0.7]$
- 成功准则：均值设计标签 $\ge \tau$

| 方法 | 错误率控制 | 选择率 |
|------|-----------|--------|
| Prediction-only | 100% 错误率 | 高 |
| CalibratedForecasts | 100% 错误率 | 高 |
| GMMForecasts(q=0) | 0% 错误率 | 过于保守，很多τ下不选 |
| **本方法** | **<α=10%** | **τ∈[0,1]时100%** |

- 本方法在所有 $\tau$ 下错误率都 $< \alpha$，同时对广泛的成功准则保持高选择率
- Prediction-only方法被预测误差严重误导，所选配置的真实均值远低于 $\tau$

**实验二：RNA结合能设计**

- 菜单：78个配置（5种设计算法 × 多超参数 × 3种预测模型）
- 密度比需估计（MDRE）

| 方法 | τ<0.32时错误率 | 选择率 |
|------|---------------|--------|
| Prediction-only | 100% | 高 |
| **本方法** | **≈0%** | **合理** |
| GMMForecasts(q=0) | 0% | 非常保守 |

- 即使密度比需估计，本方法的错误率仍远低于替代方法
- 即使偶尔选错，不成功配置的真实标签仍接近 $\tau$（后果温和）

### 消融实验

- **密度比已知 vs 估计**：已知时严格满足理论保证；估计时错误率略高于 $\alpha$ 但仍远优于基线
- **菜单规模**：从78增加到249个配置，Bonferroni校正导致选择率下降约10-20%
- **不同成功准则**：超越率准则 $g(y)=\mathbf{1}[y\ge 1]$ 下结果一致

### 关键发现

- 仅依赖预测的方法几乎总是选错（100%错误率），因为设计分布已偏离训练分布
- 预留标注数据用于PPI校正（即使减半训练数据）比将所有数据用于训练更有价值
- 方法本质是回答"我们在设计空间的哪些区域缺乏统计证据"——若设计分布与标注分布偏离太大，返回空集是合理行为

## 亮点与洞察

1. **问题形式化精确**：将实践中的"选哪个设计算法"问题严格形式化为统计假设检验，提供概率保证
2. **PPI技术巧妙应用**：用少量标注数据校正预测偏差，而非试图校准每个预测——直接瞄准下游决策
3. **密度比的双重角色**：既校正协变量偏移，又自然刻画"走多远会超出能力范围"——密度比方差大意味着证据不足
4. **面向实际需求**：用户定义的成功准则灵活实用（均值、超越率等），贴近生物设计实践
5. **理论+实践结合**：有限样本保证（Theorem 3.1）+ 实际RNA/蛋白质设计验证

## 局限与展望

- Bonferroni校正在菜单很大时过于保守，可考虑层次化或相关性感知的多重检验校正
- 密度比估计的质量直接影响保证的可靠性，高维序列空间中的估计仍具挑战
- 仅考虑单轮设计，未扩展到多轮迭代设计（但框架原则上可逐轮应用）
- 留出标注数据减少了训练数据量，需权衡训练模型质量和校正能力
- 未考虑设计多样性等辅助目标——实际中可能需在多个成功配置中进一步筛选
- 保形预测方法作为替代被证明过于保守（从未选出任何配置），可进一步研究更强大的替代

## 相关工作与启发

- **Angelopoulos et al. (2023)**：Prediction-Powered Inference，本文的技术基础
- **Zhu, Brookes & Busia et al. (2024)**：蛋白质文库设计方法（本文第一个实验的设计算法）
- **Wheelock et al. (2022)**：基于Forecast混合模型的设计标签分布建模（GMMForecasts基线）
- **Sinai et al. (2020)**：AdaLead设计算法
- **Brookes et al. (2019)**：CbAS条件采样设计算法
- **Angelopoulos et al. (2021)**：Learn Then Test框架（多重检验保证）

**启发**：ML引导设计中，预测不确定性的量化应服务于具体的**下游决策**（选哪个算法），而非仅追求通用的不确定性估计。PPI框架提供了一种优雅的"预测+少量标注→可靠决策"范式。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次形式化ML引导设计的算法选择问题并提供概率保证
- 实验充分度: ⭐⭐⭐⭐ — 蛋白质+RNA两个生物设计任务，已知/估计密度比两种场景
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，方法描述精准，Figure 1提供极佳的直觉引导
- 价值: ⭐⭐⭐⭐⭐ — 解决生物设计领域的核心实践痛点，具有广泛推广潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[ICML 2025\] Multivariate Conformal Selection](multivariate_conformal_selection.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/computational_biology/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICML 2026\] Active Timepoint Selection for Learning Measure-Valued Trajectories](../../ICML2026/computational_biology/active_timepoint_selection_for_learning_measure-valued_trajectories.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)

</div>

<!-- RELATED:END -->
