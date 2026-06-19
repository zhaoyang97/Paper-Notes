---
title: >-
  [论文解读] Distributional Adversarial Attacks and Training in Deep Hedging
description: >-
  [NeurIPS 2025][AI安全][对抗训练] 本文首次将分布对抗攻击引入深度对冲框架，提出基于 Wasserstein 球的可计算对抗训练方法（WPGD 和 WBPGD），显著提升了对冲策略在分布偏移和真实市场数据下的鲁棒性与样本外表现。 领域现状：Deep Hedging（Buehler+ 2019）用神经网络参数…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "对抗训练"
  - "分布鲁棒优化"
  - "深度对冲"
  - "Wasserstein距离"
  - "金融衍生品"
---

# Distributional Adversarial Attacks and Training in Deep Hedging

**会议**: NeurIPS 2025  
**arXiv**: [2508.14757](https://arxiv.org/abs/2508.14757)  
**代码**: [github.com/Guangyi-Mira/Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging](https://github.com/Guangyi-Mira/Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging)  
**领域**: AI安全  
**关键词**: 对抗训练, 分布鲁棒优化, 深度对冲, Wasserstein距离, 金融衍生品  

## 一句话总结
本文首次将分布对抗攻击引入深度对冲框架，提出基于 Wasserstein 球的可计算对抗训练方法（WPGD 和 WBPGD），显著提升了对冲策略在分布偏移和真实市场数据下的鲁棒性与样本外表现。

## 研究背景与动机

**领域现状**：Deep Hedging（Buehler+ 2019）用神经网络参数化对冲策略并通过最小化风险度量来训练，已在业界广泛采用。训练数据来自随机模型模拟或历史数据。

**现有痛点**：训练分布与实际部署分布之间存在模型误设（model misspecification），小的分布偏移即可导致对冲策略性能严重退化。已有鲁棒方法仅扰动终端分布或随机化模型参数，缺乏系统的分布级鲁棒性分析。

**核心矛盾**：深度对冲的损失函数（含神经网络策略 + 风险度量）是高度非凸的，标准 Wasserstein DRO 的可计算性结果依赖凸性假设，不适用于此场景。

**本文目标** (a) 量化 deep hedging 对分布偏移的脆弱程度 (b) 设计可计算的分布对抗攻击方法 (c) 通过对抗训练提升鲁棒性

**切入角度**：将逐点对抗攻击（FGSM/PGD）推广到分布层面，利用 Wasserstein DRO 的灵敏度分析（Bartl+ 2021）获得可计算的一阶近似。

**核心 idea**：通过 Wasserstein 球上的投影梯度下降实现分布对抗攻击，并将对抗样本融入训练循环提升 deep hedging 的鲁棒性。

## 方法详解

### 整体框架
基于 OCE（Optimized Certainty Equivalent）风险度量的 deep hedging：$\min_{\tilde{\theta}} \mathbb{E}_{\mathbf{I} \sim \mu}[l_{\text{DH}}(\tilde{\theta}, \mathbf{I})]$，其中 $l_{\text{DH}}(\tilde{\theta}, \mathbf{I}) = \omega + \ell(-\text{PnL}(\theta, \mathbf{I}) - \omega)$，$\omega$ 为可训练参数。DRO 版本为 $\min_{\tilde{\theta}} \max_{\eta \in B_\delta(\mu)} \mathbb{E}_{\mathbf{I} \sim \eta}[l_{\text{DH}}(\tilde{\theta}, \mathbf{I})]$。

### 关键设计

1. **分布对抗攻击的可计算重构（Theorem 3.3）**

    - 功能：将 Wasserstein 球上的无穷维优化问题近似为有限维样本扰动问题
    - 核心思路：基于 DRO 灵敏度分析，当 $\delta \to 0$ 时，最优扰动分布 $\eta_\delta$ 可表示为对每个样本 $X_n$ 的逐点扰动：$\hat{X}_n = X_n + \delta \cdot h(\nabla_x l(\theta; X_n)) \|\nabla_x l(\theta; X_n)\|_*^{q-1} \Upsilon^{1-q}$
    - 其中 $\Upsilon = (N^{-1} \sum_{n=1}^N \|\nabla_x l(\theta; X_n)\|_*^q)^{1/q}$，$h(x) = \text{sign}(x)|x|^{q-1}$ 对应 $\ell_p$ 范数下的对偶
    - 设计动机：将分布扰动分解为样本级扰动，每个样本的扰动量与其梯度大小成正比，梯度大的路径获得更多扰动预算

2. **WPGD（Wasserstein PGD，Algorithm 1）**

    - 功能：分布版本的 PGD 攻击
    - 核心思路：迭代执行 (a) 按灵敏度公式更新每条路径 $\hat{S}_n \leftarrow \hat{S}_n + \beta \cdot \text{sign}(\nabla_x l_{\text{DH}}(\theta; \hat{S}_n)) \|\nabla_x l(\theta; \hat{S}_n)\|_*^{q-1} \hat{\Upsilon}^{1-q}$，(b) 投影回 Wasserstein 约束集 $\hat{B}_\delta(\mu)$
    - 投影方式：$\hat{S}_n \leftarrow S_n + \max(1, \delta/\text{dist})(\hat{S}_n - S_n)$，整体缩放保证约束满足

3. **WBPGD（Wasserstein Budget PGD，Algorithm 2）**

    - 功能：将扰动分解为预算和方向两个变量独立优化
    - 核心思路：令 $\hat{S}_n = S_n + \text{budget}_n \times \text{direction}_n$，其中 $\text{budget}_n \in \mathbb{R}_{\ge 0}$ 控制扰动幅度，$\text{direction}_n \in [-1,1]^{T+1}$ 控制方向
    - 更新规则（Lemma 4.1）：$\text{budget}_n \leftarrow \text{budget}_n + \beta \cdot (g_n^b)^{q-1} \hat{\Upsilon}^{1-q}$，$\text{direction}_n \leftarrow \text{direction}_n + (\beta/\delta) \cdot \text{sign}(g_n^d)$
    - 优势：预算分配和方向优化解耦，更充分地探索对抗空间

4. **Heston 模型扩展（Corollary 4.2）**

    - 功能：处理价格+波动率双序列输入
    - 核心思路：定义加权距离 $d((S,v),(\hat{S},\hat{v})) = (\|S-\hat{S}\|_\infty^p + (\lambda \|v-\hat{v}\|_\infty)^p)^{1/p}$，通过 $\lambda$ 平衡不同量纲
    - 等价于对 $S$ 和 $\lambda v$ 分别独立施加 $\ell_\infty$ 扰动

### 对抗训练策略
训练损失 $\mathcal{L}_{\text{adv}}(\theta) = \alpha \sum_n l_{\text{DH}}(\theta; X_n) + \sum_n l_{\text{DH}}(\theta; \hat{X}_n)$，$\alpha$ 平衡干净样本和对抗样本的权重，交替执行对抗攻击和参数更新。

## 实验关键数据

### Heston 模型对抗攻击测试（经典 deep hedging，CVaR 风险度量）

| 攻击方法 | $\delta=0$ | $\delta=0.01$ | $\delta=0.05$ | $\delta=0.1$ | $\delta=0.3$ | $\delta=0.5$ |
|---------|-----------|--------------|--------------|-------------|-------------|-------------|
| S-WBPGD | 1.928 | 1.964 | 2.136 | 2.447 | 4.577 | 8.075 |
| SV-WBPGD | 1.928 | 1.966 | 2.145 | 2.466 | 4.590 | 7.739 |
| S-WPGD | 1.928 | 1.964 | 2.134 | 2.437 | 4.515 | 7.541 |

### 对抗训练 vs 经典训练的样本外表现

| 模型/方法 | Clean Loss | Attacked Loss ($\delta=0.1$) | 改善 |
|----------|------------|---------------------------|------|
| BS 经典 | 基线 | 大幅退化 | - |
| BS 对抗训练 | 略高 | 显著改善 | 攻击后损失降低 30-50% |
| Heston 经典 | 1.928 | 2.447 | - |
| Heston 对抗训练 | ~2.0 | ~2.1 | 攻击后损失接近基线 |

### 关键发现
- 即使 $\delta=0.01$ 的微小分布扰动也使 CVaR 损失增加约 2%，$\delta=0.5$ 时损失翻 4 倍，证明经典 deep hedging 极其脆弱
- WBPGD 一致性优于 WPGD，尤其在大扰动下——预算-方向解耦能更好地利用攻击资源
- 协方差矩阵的 Frobenius 距离在 $\delta < 0.1$ 时很小（< 1.0 vs 基线 ~386），说明统计意义上的"小偏移"即可造成大损失
- 对抗训练策略在真实市场数据上也保持有效，在市场剧变期间表现尤为突出

## 亮点与洞察
- **分布攻击与逐点攻击的统一视角**：当 $p \to \infty$ 时分布约束退化为 $L_\infty$ 逐点约束，WPGD/WBPGD 包含 FGSM/PGD 为特例。这构建了从 DRO 到对抗训练的完整理论桥梁。
- **预算-方向解耦**：WBPGD 将"给每条路径分配多少扰动量"和"沿哪个方向扰动"拆成独立优化，比直接在路径空间梯度上升更灵活，可迁移到其他时序 DRO 问题。
- **OCE 风险度量的妙用**：将 $\omega$ 作为可训练参数后，deep hedging 的 DRO 转化为标准期望损失最小化形式，直接适配 DRO 文献的理论工具。

## 局限与展望
- 理论结果（Theorem 3.3, Lemma 3.4）本质是 $\delta \to 0$ 的一阶近似，大 $\delta$ 时近似质量未保证
- 假设损失对输入 Lipschitz 连续，但实际金融场景中可能存在不连续（如障碍期权的击穿事件）
- 对抗训练需要额外的梯度计算和投影步骤，训练时间约增 2-3 倍
- 未考虑交易成本和市场冲击对鲁棒性的影响

## 相关工作与启发
- **vs Lutkebohmert+ 2022（参数随机化鲁棒 DH）**：他们通过随机化模型参数引入不确定性，本文直接在数据分布层面扰动，更加灵活且理论基础更扎实
- **vs Wu+ 2023（终端分布鲁棒）**：仅扰动终端 payoff 分布，不触及中间路径；本文扰动整条价格路径，覆盖更广的分布偏移类型
- **vs Madry+ 2018（逐点 PGD）**：经典图像领域对抗训练，本文将其推广到分布层面并适配金融时序数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 将分布对抗攻击引入 deep hedging 是首次，WBPGD 的预算-方向解耦设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 BS/Heston/General Affine Diffusion 模型，真实市场数据验证，消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论和实验结合好，符号一致
- 价值: ⭐⭐⭐⭐ 对金融 ML 实践有直接意义，框架可推广到其他数据驱动决策问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Identifying and Understanding Cross-Class Features in Adversarial Training](../../ICML2025/ai_safety/identifying_and_understanding_cross-class_features_in_adversarial_training.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](../../ICML2026/ai_safety/sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)
- [\[ICLR 2026\] Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks](../../ICLR2026/ai_safety/benchmarking_stochastic_approximation_algorithms_for_fairness-constrained_traini.md)
- [\[CVPR 2026\] Taming the Long Tail: Rebalancing Adversarial Training via Adaptive Perturbation](../../CVPR2026/ai_safety/taming_the_long_tail_rebalancing_adversarial_training_via_adaptive_perturbation.md)

</div>

<!-- RELATED:END -->
