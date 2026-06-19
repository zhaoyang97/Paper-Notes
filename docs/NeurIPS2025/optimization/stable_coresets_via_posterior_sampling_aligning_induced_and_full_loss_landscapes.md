---
title: >-
  [论文解读] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes
description: >-
  [NeurIPS 2025][优化/理论][Coreset Selection] 提出基于后验采样的 coreset 选择框架，通过在 BatchNorm 层上采样权重扰动来平滑损失曲面，保证 coreset 与全数据集的损失景观对齐（包含 Hessian 和 Newton 步的近似），在高标签噪声下显著优于现有方法。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "Coreset Selection"
  - "Posterior Sampling"
  - "Loss Landscape Alignment"
  - "Label Noise Robustness"
  - "SGD Convergence"
---

# Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes

**会议**: NeurIPS 2025  
**arXiv**: [2511.17399](https://arxiv.org/abs/2511.17399)  
**代码**: [https://github.com/changwk1001/stable-coreset.git](https://github.com/changwk1001/stable-coreset.git)  
**领域**: 高效训练 / 数据选择  
**关键词**: Coreset Selection, Posterior Sampling, Loss Landscape Alignment, Label Noise Robustness, SGD Convergence

## 一句话总结

提出基于后验采样的 coreset 选择框架，通过在 BatchNorm 层上采样权重扰动来平滑损失曲面，保证 coreset 与全数据集的损失景观对齐（包含 Hessian 和 Newton 步的近似），在高标签噪声下显著优于现有方法。

## 研究背景与动机

Coreset 选择旨在找到训练数据的小规模代表性子集来加速训练。基于梯度匹配的方法（Craig、GradMatch 等）通过选择梯度方向与全数据集对齐的子集来实现这一目标。

**核心失效模式**：在实际条件下（标签噪声、对抗扰动、极小子集预算），梯度匹配方法倾向于选择梯度幅值大的样本，这恰恰包括了错标或异常样本。结果是 coreset 产生的**损失曲面与全数据集的损失曲面严重不对齐**——模型在子集上的优化轨迹偏离了全数据集的最优方向。实验发现在 20%-50% 标签噪声下，许多梯度匹配方法的性能灾难性下降。

**二阶方法的困境**：尝试匹配 Hessian 的方法（如 Crest）虽然能一定程度改善对齐，但 Hessian 计算/近似极其昂贵（有时甚至抵消了 coreset 带来的加速），内存开销巨大（可达全数据训练的 3 倍），且在噪声标签下仍不稳定。

**新视角**：与其显式计算 Hessian，不如通过后验采样隐式实现损失曲面对齐——在当前参数的高斯邻域内采样权重扰动，用蒙特卡洛平均的梯度信息进行 coreset 选择，自动获得平滑且几何对齐的选择准则。

## 方法详解

### 整体框架

将标准 coreset 选择问题中的域 $W$ 上的最大化替换为 $(\sigma, \epsilon, \bar{w})$-稳定性约束。每个 epoch 进行 $P$ 次子抽样，每次选择 $m$ 个数据点构成影子数据集 $S_t$，再在 $S_t$ 上做 batched SGD。

### 关键设计

1. **$(\sigma, \epsilon, \bar{w})$-稳定性定义**：
   子集 $S'$ 是稳定的当且仅当：
    $E_{w \sim \mathcal{N}(\bar{w}, \sigma I)} \|\nabla l_{S'}(w) - \nabla l(w)\|_2^2 \leq \epsilon$
   
   即在参数 $\bar{w}$ 的高斯邻域内，子集梯度与全数据梯度的平方距离的期望足够小。这比仅在当前点 $\bar{w}$ 匹配梯度更鲁棒——它要求子集在整个邻域内都是好的代表。

2. **后验采样 = 隐式 Hessian 对齐（Theorem 3.2）**：
   如果子集 $S'$ 是 $(\sigma, \epsilon, w)$-稳定的，设 Hessian 差为 $\mathcal{E} = H_{S',w} - H_{S,w}$，则：
    - $\|\mathcal{E}\| \leq O(\epsilon^{1/2})$：谱范数有界
    - $\text{tr}(\mathcal{E}^2) \leq O(\epsilon/\sigma)$：Frobenius 范数有界
    - $\|H_{S'}^{-1}\nabla l_{S'} - H_S^{-1}\nabla l_S\| \leq O(\epsilon^{1/2})$：Newton 步近似
   
   这意味着通过简单的高斯采样，不需要显式计算 Hessian，就能保证 coreset 的损失曲面（包括曲率信息）与全数据集对齐。

3. **BatchNorm 层采样**：
   对全模型参数采样计算开销大。受近期研究启发（BN 层对 sharpness 控制和优化性能至关重要），作者仅在 BatchNorm 层上施加高斯扰动。实验验证（Table 2 Left）BN 层采样在所有噪声比下都优于 All-layer 和 FC-layer 采样。仅需 4 个采样模型，$\sigma$ 从 $\{0.1, 0.01, 0.001\}$ 中交叉验证选择。

### 损失函数 / 训练策略

**收敛分析（Theorem 3.3）**：
- **绝对噪声**：收敛率 $O(1/\sqrt{T})$，但关键项改进了 $O(1/M)$ 因子（$M$ 为采样数）
- **乘法噪声**：收敛率从之前的 $O(1/\sqrt{RT})$ 改进为 $O(1/\sqrt{MRT})$（$R$ 为 mini-batch 大小）

设置 $\sigma_2^2 d = 1/(M\sqrt{T})$，学习率 $\eta = \min\{1/\sqrt{T}, 1/\beta\}$。

## 实验关键数据

### 主实验（Table 1，大规模实验）

| 数据集 | 噪声率 | 本文方法 | Random | Crest (SOTA) |
|--------|--------|---------|--------|------|
| SNLI | 0.0 | **0.9132** | 0.9046 | 0.9098 |
| SNLI | 0.5 | **0.6062** | 0.5316 | 0.5104 |
| TinyImageNet | 0.0 | 0.5732 | 0.5520 | 0.5609 |
| TinyImageNet | 0.5 | **0.3644** | 0.2857 | 0.3567 |
| ImageNet-1k | 0.0 | 0.7091 | 0.7074 | **0.7136** |
| ImageNet-1k | 0.5 | **0.6388** | 0.5939 | 0.6051 |

### 消融实验（Table 2，采样层选择 + 极端噪声）

| 配置 | CIFAR-10 (0.5 noise) | CIFAR-100 (0.5 noise) | 说明 |
|------|---------------------|----------------------|------|
| BN 层采样 | **0.7318** | **0.5014** | 最佳选择 |
| All 层采样 | 0.7288 | - | 开销大但效果略差 |
| FC 层采样 | 0.7232 | - | 效果最差 |
| 本文 (0.8 noise) | **0.3701** | **0.1680** | 极端噪声下仍有效 |
| Crest (0.8 noise) | 0.1520 | 0.1613 | 严重退化 |

### 关键发现

- 在 SNLI 50% 噪声下，本文方法比次优方法（Random）高 **7% 绝对精度**
- 在 ImageNet-1k 零噪声下 Crest 略优，但其训练时间是本文的 **2 倍**（42h vs 20h），内存是 **3 倍**
- 跨 6 个数据集、多种架构（LeNet/ResNet/ViT/RoBERTa）一致领先
- 时间到精度（time-to-accuracy）比 Crest 快 **20%-200%**
- 梯度匹配误差可视化显示本文方法的梯度估计始终优于 Craig

## 亮点与洞察

- **后验采样 = 免费的曲率信息**：这是核心洞察——高斯扰动下的梯度平均隐式包含了 Hessian 信息，无需显式计算
- **BN 层是 coreset 稳定性的密钥**：仅在 BN 层扰动就够了，这与 BN 层控制模型 sharpness 的研究一致
- **Random baseline 的意外强势**：在高噪声下 Random 采样反而优于多数精心设计的方法，说明复杂选择策略容易被噪声误导。本文方法是首个在此场景下稳定超越 Random 的 coreset 方法
- **损失曲面平滑效果**：可视化清楚展示了标准 Craig 的 coreset 产生尖锐不规则曲面，本文方法产生平滑且与全数据曲面对齐的曲面

## 局限与展望

- 对音频、视频等非标准数据类型的泛化未验证
- 后验分布的选择（球形 vs Hessian-逆 vs 集成）的理论最优性未完全建立
- 在零噪声 + 大数据集场景下优势较小（如 ImageNet-1k 0% noise 略逊于 Crest）
- 理论分析中 BN 层采样的特殊效果缺乏完整形式化解释

## 相关工作与启发

- 与 Craig (2020)、GradMatch (2021)、Crest (2023) 形成 coreset 方法的演进线
- 后验采样平滑的思路借鉴了 SAM (Sharpness-Aware Minimization) 和 Bayesian learning
- Shin et al. (2023) 的 loss curvature matching 是直接动机，本文用采样替代显式曲率计算更高效
- 可启发将后验采样应用于其他数据选择问题（如主动学习、课程学习）

## 评分

- 新颖性: ⭐⭐⭐⭐ 后验采样替代 Hessian 计算的视角简洁而有力，理论联系建立完整
- 实验充分度: ⭐⭐⭐⭐⭐ 6+ 数据集、4+ 架构、多噪声级别、时间/内存效率分析、后验选择消融
- 写作质量: ⭐⭐⭐⭐ 动机阐述生动（损失曲面可视化），理论与实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 实用性极强——简单、高效、鲁棒，几乎无额外开销即可提升 coreset 质量

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Exploring Landscapes for Better Minima along Valleys](exploring_landscapes_for_better_minima_along_valleys.md)
- [\[NeurIPS 2025\] FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling](fedrts_federated_robust_pruning_via_combinatorial_thompson_sampling.md)
- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](../../ICML2026/optimization/convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)
- [\[NeurIPS 2025\] Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules](gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)
- [\[ICML 2026\] SVRG and Beyond via Posterior Correction](../../ICML2026/optimization/svrg_and_beyond_via_posterior_correction.md)

</div>

<!-- RELATED:END -->
