---
title: >-
  [论文解读] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains
description: >-
  [ICML2025][因果推理][演化域泛化] 提出时间感知结构因果模型 (time-aware SCM) 和 SYNC 方法，通过同时学习静态与动态因果表示并建模因果机制漂移，在演化域泛化 (EDG) 任务中有效消除虚假相关，实现优越的时序泛化性能。 演化域泛化 (EDG)：旨在从时序上连续变化的源域中学习…
tags:
  - "ICML2025"
  - "因果推理"
  - "演化域泛化"
  - "因果表示学习"
  - "时间感知结构因果模型"
  - "变分自编码器"
  - "互信息"
  - "虚假相关"
---

# Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains

**会议**: ICML2025  
**arXiv**: [2506.17718](https://arxiv.org/abs/2506.17718)  
**代码**: GitHub（论文中标注有 GitHub 链接，具体地址待确认）  
**领域**: 因果推理  
**关键词**: 演化域泛化, 因果表示学习, 时间感知结构因果模型, 变分自编码器, 互信息, 虚假相关

## 一句话总结

提出时间感知结构因果模型 (time-aware SCM) 和 SYNC 方法，通过同时学习静态与动态因果表示并建模因果机制漂移，在演化域泛化 (EDG) 任务中有效消除虚假相关，实现优越的时序泛化性能。

## 研究背景与动机

**演化域泛化 (EDG)** 旨在从时序上连续变化的源域中学习，使模型能泛化到未来的未见域。与传统域泛化 (DG) 假设域间无时序关系不同，EDG 需要捕捉数据分布随时间演变的模式。

现有 EDG 方法（如 LSSAE、SDE-EDG）的核心问题是：**仅建模数据与标签之间的统计相关性，容易受虚假相关的干扰**。论文以 Caltran 交通监控数据集为例说明：白天图片大多含车辆，夜晚图片大多不含车辆，导致模型学习了"光照→有无车辆"的虚假捷径，而非关注车辆本身的因果特征。

传统因果方法在动态场景中面临两大挑战：

**因果因子随时间变化**：静态 SCM 不能建模时变的类别相关特征

**因果机制漂移**：从因果因子到标签的映射 $P(Y|Z_c)$ 不再跨域不变

## 方法详解

### 整体框架：SYNC (Static-DYNamic Causal Representation Learning)

SYNC 基于新设计的时间感知 SCM，在 sequential VAE 框架中同时学习静态和动态因果表示，并通过信息论目标进行因果因子的解耦。

### 时间感知结构因果模型 (Time-aware SCM)

传统 DG 的 SCM 将因果因子 $Z_c$ 和虚假因子 $Z_s$ 均视为时间不变。本文提出更精细的分解：

- **静态因果因子** $Z_c^{st}$：跨域稳定的类别相关信息（如车辆的形状特征）
- **动态因果因子** $Z_c^{dy}$：随域变化的类别相关信息（如不同时期车辆外观差异）
- **静态虚假因子** $Z_s^{st}$ 和 **动态虚假因子** $Z_s^{dy}$
- **漂移因子** $Z^d$：建模因果机制自身的演化

数据生成过程为 $X \leftarrow (Z^{st}, Z^{dy})$，标签生成为 $Y \leftarrow (Z_c^{st}, Z_c^{dy}, Z^d)$。全局变量 $G$ 和局部变量 $L$ 分别生成静态和动态因子，$L$ 同时作为时间混杂因素通过后门路径引入虚假相关。

### 演化模式学习

采用 sequential VAE 框架分别建模静态和动态表示：

- 静态编码器 $q_\psi(z_t^{st}|x_t) = \mathcal{N}(\mu(z_t^{st}), \sigma^2(z_t^{st}))$，先验 $p(z_t^{st}) = \mathcal{N}(0, I)$
- 动态编码器 $q_\theta(z_t^{dy}|z_{<t}^{dy}, x_t)$，通过 LSTM 建模先验 $p(z_t^{dy}|z_{<t}^{dy})$ 以捕捉时序依赖

演化模式损失 $\mathcal{L}_{fp}$ 包含重构项和两个 KL 散度项：

$$\mathcal{L}_{fp} = -\sum_{t=1}^{T} \mathbb{E}[\log p(x_t|z_t^{st}, z_t^{dy})] + \sum_{t=1}^{T} D_{KL}(q_\psi \| p(z_t^{st})) + \sum_{t=1}^{T} D_{KL}(q_\theta \| p(z_t^{dy}|z_{<t}^{dy}))$$

### 静态-动态解耦

最小化静态与动态表示之间的互信息 $\mathcal{L}_{MI} = \sum_{t=1}^{T} I(z_t^{st}; z_t^{dy})$，使用 mini-batch weighted sampling (MWS) 估计熵项。

### 因果表示挖掘

核心理论基础是 Proposition 1：在合理的熵不等式条件下，同类因果因子比虚假因子具有更高的条件互信息。

**静态因果表示**：通过跨域对比学习（Eq.6）最大化 $I(\Phi_c^{st}(X_t); \Phi_c^{st}(X_{t-1})|Y)$，即拉近相邻域中同类样本的静态因果表示。使用 Gumbel-Softmax 生成 0-1 mask 选择因果维度（mask 比例 $\kappa$）。

**动态因果表示**：以学到的静态因果因子为锚点，通过域内对比学习（Eq.9）最大化 $I(\Phi_c^{dy}(X_t); Z_{c,t}^{st}|Y)$，拉近同域内同类样本的动态因果表示与静态因果表示。

### 因果机制漂移建模

引入漂移因子 $Z^d$，通过 RNN 编码器 $q_\zeta(z_t^d|z_{<t}^d, y_t)$ 学习，输出为 categorical 分布。分类损失 $\mathcal{L}_{mp}$ 基于 $(z_{c,t}^{dy}, z_{c,t}^{st}, z_t^d)$ 联合预测标签。

### 总损失

$$\mathcal{L}_{SYNC} = \mathcal{L}_{evolve} + \alpha_1 \mathcal{L}_{MI} + \alpha_2 \mathcal{L}_{causal}$$

其中 $\mathcal{L}_{evolve} = \mathcal{L}_{fp} + \mathcal{L}_{mp}$，$\mathcal{L}_{causal} = \mathcal{L}_{stc} + \mathcal{L}_{dyc}$。

### 理论保证

- **Theorem 1**：优化 $\mathcal{L}_{evolve}$ 可学习训练域的联合分布 $p(x_{1:T}, y_{1:T})$
- **Theorem 2**：优化 $\mathcal{L}_{SYNC}$ 可在每个时域获得最优因果预测器，即基于 $(Z_{c,t}, Z_t^d)$ 的预测器满足 Definition 2 的最优条件

## 实验关键数据

### 数据集

2 个合成数据集（Circle 30 域、Sine 24 域）+ 5 个真实数据集（RMNIST 19 域、Portraits 34 域、Caltran 34 域、PowerSupply 30 域、ONP 24 域），按 1/2:1/6:1/3 划分源域/验证域/目标域。

### 主实验结果（Table 1，Overall Wst/Avg）

| 方法 | Overall Wst | Overall Avg |
|------|-----------|-----------|
| ERM | 51.9 | 63.9 |
| MMD-LSAE (EDG SOTA-) | 58.1 | 70.9 |
| SDE-EDG (EDG SOTA-) | 55.4 | 71.9 |
| iDAG (因果 DG 最佳) | 56.1 | 63.7 |
| **SYNC (本文)** | **63.4** | **73.1** |

- SYNC 在所有 7 个数据集上 Overall 均最优
- 相比最佳因果 DG 方法 (iDAG)：Wst +7.3%, Avg +9.4%
- 相比最佳 EDG 方法 (MMD-LSAE/SDE-EDG)：Wst +5.3%, Avg +1.2%
- Circle 数据集提升最显著：Wst 67.0% vs 54.0% (MMD-LSAE)，Avg 84.7% vs 81.5% (SDE-EDG)

### 消融实验（Table 2，RMNIST）

| 变体 | 组件 | Wst | Avg |
|------|------|-----|-----|
| A (基础) | 仅 $\mathcal{L}_{evolve}$ | 40.5 | 44.1 |
| B | + $\mathcal{L}_{MI}$ 解耦 | 41.9 | 45.7 |
| C | + 静态因果 $Z_c^{st}$ | 44.1 | 48.7 |
| D | + 动态因果 $Z_c^{dy}$ | 42.9 | 49.2 |
| **SYNC** | **全部** | **45.8** | **50.8** |

**关键发现**：

1. 解耦 (B vs A) 带来稳定提升（+1.5%），说明分离静态/动态信息有价值
2. 静态因果表示 (C) 对 Wst 提升更大（+2.2%），保证了长期稳定泛化
3. 动态因果表示 (D) 对 Avg 提升更大（+3.5%），适应当前分布变化
4. 两者互补：SYNC 联合学习效果最优

### 其他分析

- **解耦可视化**：SYNC 的静态-动态互信息下降更快更稳定（vs LSSAE）
- **时序鲁棒性**：SYNC 在时间轴后期域上仍保持高精度，而 SDE-EDG 在 Circle 后期性能下降
- **决策边界可视化**：SYNC 的决策边界最接近 ground truth

## 亮点与洞察

1. **因果视角切入 EDG 问题**：首次将因果表示学习引入演化域泛化，设计了精细的时间感知 SCM，区分静态/动态因果与虚假因子
2. **互补的双重因果表示**：静态因果保证长期稳定性（改善 worst-case），动态因果适应时变分布（改善 average），二者互补
3. **理论完备性**：证明了方法可获得每个时域的最优因果预测器（Theorem 2）
4. **elegant 的因果因子提取方式**：利用 Gumbel-Softmax mask + 对比学习的组合，避免了传统因果干预的困难

## 局限与展望

1. **域数量假设**：需要足够多的时序源域来学习演化模式，少域场景效果不明
2. **线性时序假设**：LSTM 建模的先验隐含"近期域更相关"假设，对非单调或周期性分布漂移的适用性存疑
3. **mask 比例 $\kappa$ 敏感性**：因果维度选择比例需要调参，论文未充分讨论其敏感性
4. **计算开销**：VAE 重构 + 对比学习 + 互信息估计的联合优化，训练代价较高
5. **仅验证分类任务**：未扩展到回归、检测等任务类型
6. **Proposition 1 条件**：虽然作者声称"条件易满足"，但这一声明缺乏实证支撑

## 相关工作与启发

- **EDG 基线**：LSSAE/MMD-LSAE (Qin et al., 2022/2023) 是基于 sequential VAE 的代表方法，SYNC 在此基础上增加了因果学习组件
- **SDE-EDG** (Zeng et al., 2023a)：用随机微分方程建模连续演化轨迹，平均性能强但 worst-case 不稳定
- **因果 DG**：IRM、IIB 等方法聚焦不变性学习但忽略动态因果信息，在非平稳环境中效果受限
- **DRAIN** (Bai et al., 2023)：贝叶斯框架 + 动态图生成网络参数，思路与本文互补
- **启发**：将因果表示学习与演化模式学习解耦的思路可推广到其他时序分布漂移问题（如持续学习、在线学习）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 时间感知 SCM 设计精巧，静态-动态因果分解是有意义的新贡献
- 实验充分度: ⭐⭐⭐⭐ — 7 个数据集 + 20 个基线 + 完整消融 + 可视化分析，但缺少大规模数据集验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论推导严谨，图示辅助理解效果好
- 价值: ⭐⭐⭐⭐ — 为 EDG 领域引入因果视角，方法论贡献扎实，实用性有待更多场景验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization](../../ECCV2024/causal_inference/integrating_markov_blanket_discovery_into_causal_representation_learning_for_dom.md)
- [\[ICML 2025\] E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time](e-lda_toward_interpretable_lda_topic_models_with_strong_guarantees_in_logarithmi.md)
- [\[ICML 2025\] Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes](classifier_reconstruction_through_counterfactual-aware_wasserstein_prototypes.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)

</div>

<!-- RELATED:END -->
