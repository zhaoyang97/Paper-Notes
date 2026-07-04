---
title: >-
  [论文解读] Enhancing DPSGD via Per-Sample Momentum and Low-Pass Filtering
description: >-
  [AAAI2026][AI安全][differential privacy] 提出 DP-PMLF，通过逐样本动量（per-sample momentum）降低裁剪偏差，同时利用低通滤波器（low-pass filter）抑制高频 DP 噪声，首次同时从两个方向缓解 DPSGD 的精度退化问题。 差分隐私随机梯度下降（DPS…
tags:
  - "AAAI2026"
  - "AI安全"
  - "differential privacy"
  - "DPSGD"
  - "Per-Sample Momentum"
  - "Low-Pass Filtering"
  - "Privacy-Utility Trade-off"
---

# Enhancing DPSGD via Per-Sample Momentum and Low-Pass Filtering

**会议**: AAAI2026  
**arXiv**: [2511.08841](https://arxiv.org/abs/2511.08841)  
**代码**: 待确认  
**领域**: AI安全  
**关键词**: differential privacy, DPSGD, Per-Sample Momentum, Low-Pass Filtering, Privacy-Utility Trade-off

## 一句话总结
提出 DP-PMLF，通过逐样本动量（per-sample momentum）降低裁剪偏差，同时利用低通滤波器（low-pass filter）抑制高频 DP 噪声，首次同时从两个方向缓解 DPSGD 的精度退化问题。

## 背景与动机
差分隐私随机梯度下降（DPSGD）通过梯度裁剪和噪声注入为深度学习提供形式化隐私保证，但精度损失严重，根源在于两个相互矛盾的因素：

1. **DP 噪声**：保护隐私需要向聚合梯度注入校准过的高斯噪声，噪声尺度与裁剪阈值 $C$ 成正比——$C$ 越大噪声越多。
2. **裁剪偏差（clipping bias）**：裁剪单样本梯度范数会引入偏差——$C$ 越小偏差越大。

现有方法大多只解决其中一个：
- **LP-DPSGD**（Zhang et al.）用低通滤波器降低 DP 噪声，但引入了额外偏差项，在 clipping bias 占主导时反而性能更差。
- **InnerOuter**（Xiao et al.）用内外动量减少裁剪偏差，但对 DP 噪声缺乏抑制、外层动量无归一化会累积噪声，在 $\epsilon=1$（强隐私）时性能严重退化。

作者观察到 clipping bias 不仅与阈值 $C$ 有关，还和采样方差 $\sigma_{SGD}$ 成正比，因此存在同时降低噪声和偏差的空间。

## 核心问题
如何在不消耗额外隐私预算的前提下，**同时**缓解 DPSGD 中的 DP 噪声和裁剪偏差？

## 方法详解

### 整体框架：DP-PMLF
DP-PMLF 由两个互补模块组成，依次作用于梯度处理流程：

### 1. 逐样本动量（Per-Sample Momentum）
对每个样本 $\xi$ 维护一个动量项，对过去 $k$ 轮的梯度做指数衰减加权平均：

$$v_t^{(\xi)} = \sum_{i=t-k+1}^{t} \hat{\beta}^{t-i} \nabla f^{(\xi)}(x_i)$$

其中 $\hat{\beta}^{t-i} = \beta^{t-i}/c_\beta$，$c_\beta$ 是归一化常数确保系数和为 1。

**作用**：在裁剪之前平滑梯度估计，降低采样方差 $\sigma_{SGD}$。理论上方差可降低 $\rho^2$ 倍，其中：

$$\rho = \sqrt{\frac{(1+\beta)(1-\beta^k)}{(1-\beta)(1+\beta^k)}}$$

当 $\beta \to 1$ 时 $\rho^2 \to k$，即等权平均。归一化也避免了动量系数过大产生的噪声累积问题（InnerOuter 的短板）。

### 2. 低通滤波器（Low-Pass Filter）
裁剪后聚合梯度并加入高斯噪声，然后施加线性低通滤波器：

$$m_t = -\sum_{r=1}^{n_a} a_r m_{t-r} + \sum_{r=0}^{n_b} b_r \bar{v}_{t-r}$$

滤波器系数满足 $-\sum a_r + \sum b_r = 1$ 保证信号均值不变。

**原理**：DP 噪声在所有频率上均匀分布，而真实梯度信号集中在低频段，低通滤波器可保留梯度信号并抑制高频噪声。由于滤波器只是对已加噪输出做后处理，根据 DP 后处理性质（post-processing lemma），**不消耗额外隐私预算**。

### 3. 初始化偏差修正
通过递推计算归一化常数 $c_{m,t}$，输出 $\hat{m}_t = m_t / c_{m,t}$ 修正滤波器初始阶段的暂态偏差。

### 算法流程
1. 采样 mini-batch $\mathcal{B}_t$
2. 对每个样本计算逐样本动量 $v_t^{(\xi)}$
3. 裁剪 $\tilde{v}_t^{(\xi)} = \text{clip}(v_t^{(\xi)}, C)$
4. 聚合并加噪：$\bar{v}_t = \frac{1}{B}\sum \tilde{v}_t^{(\xi)} + w_t$
5. 低通滤波 + 偏差修正 → $\hat{m}_t$
6. 更新模型：$x_{t+1} = x_t - \eta \hat{m}_t$

### 理论保证
- **收敛性**：在 $L$-光滑、有界方差/梯度等标准假设下，收敛上界为

$$\mathcal{O}\!\left(\frac{f(x_0)-f^*}{\eta T} + L\eta C^2 + \frac{L\eta \, d\sigma_{DP}^2}{\Gamma_{DP}} + \frac{\sigma_{SGD}^2}{\rho^2 \Gamma_{SGD}}\right)$$

其中 $\Gamma_{DP}$ 和 $\Gamma_{SGD}$ 分别反映低通滤波器对 DP 噪声和裁剪偏差的抑制倍数。相比 vanilla DPSGD，clipping bias 项额外除以 $\rho^2$，DP 噪声项额外除以 $\Gamma_{DP}$。

- **隐私性**：利用高斯机制 + 子采样隐私放大 + moments accountant，满足 $(\epsilon, \delta)$-DP。

## 实验关键数据

### 图像分类（ViT，无预训练）

| 方法 | CIFAR-10 ($\epsilon$=1) | CIFAR-10 ($\epsilon$=8) | CIFAR-100 ($\epsilon$=1) | CIFAR-100 ($\epsilon$=8) |
|---|---|---|---|---|
| DPSGD | 35.74 | 47.74 | 7.52 | 18.27 |
| LP-DPSGD | 35.84 | 48.37 | 7.55 | 18.52 |
| InnerOuter | 11.55 | 33.53 | 1.13 | 13.93 |
| **DP-PMLF** | **40.96** | **51.47** | **11.40** | **23.15** |

- CIFAR-10 上 $\epsilon=1$ 时超越最优 baseline 约 **5%**，CIFAR-100 上约 **4%**。
- InnerOuter 在 $\epsilon=1$ 时因噪声累积严重退化（CIFAR-10 仅 11.55%）。

### 句子分类（RoBERTa-base 微调，GLUE）

| 方法 | MNLI ($\epsilon$=1) | QNLI ($\epsilon$=1) | QQP ($\epsilon$=8) | SST-2 ($\epsilon$=8) |
|---|---|---|---|---|
| DPSGD | 51.36 | 65.59 | 80.38 | 90.83 |
| **DP-PMLF** | **56.81** | **72.38** | **83.42** | **90.39** |

- MNLI 上超过 baseline 4%+，QNLI 上超过近 3%（$\epsilon=1$）。

### 多模型架构（CIFAR-10，$\epsilon=1$）
- CNN-5：DP-PMLF ~47%，超越最优 baseline 约 9%
- ResNet-18：DP-PMLF ~50%，超越约 1-2%
- ViT：DP-PMLF ~31%，超越约 8%

### 消融实验
- 去掉 per-sample momentum → 性能一致下降，验证其对降低裁剪偏差的有效性。
- 去掉低通滤波器 → 在 $\epsilon \leq 6$（噪声较大时）性能下降；但在 $\epsilon > 6$ 时，过度平滑反而丢失少量真实梯度信息（约 0.5-0.7%）。

## 亮点
1. **首次同时处理两大退化源**：巧妙地将 per-sample momentum（降方差/偏差）和低通滤波器（降噪声）组合，覆盖了 LP-DPSGD 和 InnerOuter 各自失效的场景。
2. **零隐私代价的后处理降噪**：利用 DP 的 post-processing 性质，低通滤波不消耗隐私预算。
3. **严格的理论保证**：提供了完整的收敛分析和隐私证明，收敛上界清晰展示了两个模块各自的贡献项。
4. **跨模态泛化**：在图像分类（CNN/ResNet/ViT）和句子分类（RoBERTa）上均有效。

## 局限与展望
1. **超参数敏感性**：$\beta$、$k$、滤波器系数 $\{a_r\}, \{b_r\}$ 需要手动调整，作者未提出自适应选择方法。
2. **逐样本历史存储开销**：需要为每个样本维护最近 $k$ 轮梯度，内存开销随数据集规模和 $k$ 增长。
3. **强隐私下绝对精度仍低**：$\epsilon=1$ 时 CIFAR-10 最高仅 40.96%，离实用仍有距离。
4. **理论假设较强**：需要有界梯度（Assumption 3）和梯度自相关（Assumption 4）等假设，未扩展到更一般的非凸条件（如 PL 条件、$(L_0, L_1)$-smoothness）。
5. **过度平滑风险**：消融实验显示当 DP 噪声较小（$\epsilon > 6$）时，低通滤波器反而略微损害性能。

## 与相关工作的对比

| 方法 | 降 DP 噪声 | 降裁剪偏差 | 理论保证 | 额外隐私开销 |
|---|---|---|---|---|
| LP-DPSGD | ✓ | ✗（反而增加偏差） | 有（含额外偏差项） | 无 |
| InnerOuter | ✗（累积噪声） | ✓ | 无 | 无 |
| DiceSGD | ✗（需更多噪声） | ✓（误差反馈） | 有 | 有 |
| Clipless DPSGD | ✓ | ✓（无裁剪） | 有 | 无 | 需特定网络架构 |
| **DP-PMLF** | **✓** | **✓** | **有** | **无** |

## 启发与关联
- **方差降低是通用工具**：per-sample momentum 的核心是降采样方差，这一思路可推广到其他 DP 优化器（如 DP-Adam）。
- **频域视角值得进一步探索**：低通滤波器利用了梯度信号和噪声的频谱差异，高阶或自适应滤波器可能进一步提升效果。
- **与联邦学习的结合**：联邦学习同样面临噪声 + 梯度压缩的精度损失，类似的动量平滑 + 频域滤波策略可能有效。

## 评分
- 新颖性: 7/10 — 两个已知组件的巧妙组合，核心贡献在于"同时处理"的洞察和理论分析
- 实验充分度: 7/10 — 覆盖多数据集/多模型/多模态，但缺乏大规模模型和更多 DPSGD 变体对比
- 写作质量: 8/10 — 动机清晰、理论和实验组织良好
- 价值: 7/10 — 在 DP 训练领域提供了实用且有理论支撑的改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FedMOP: Achieving Enhanced Privacy and Performance in Federated Learning via Momentum Orthogonal Projection](../../CVPR2026/ai_safety/fedmop_achieving_enhanced_privacy_and_performance_in_federated_learning_via_mome.md)
- [\[ICLR 2026\] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity](../../ICLR2026/ai_safety/prior-based_noisy_text_data_filtering_fast_and_strong_alternative_to_perplexity.md)
- [\[NeurIPS 2025\] Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](../../NeurIPS2025/ai_safety/spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)
- [\[CVPR 2025\] Joint Out-of-Distribution Filtering and Data Discovery Active Learning](../../CVPR2025/ai_safety/joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)
- [\[CVPR 2026\] SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning](../../CVPR2026/ai_safety/seba_sample-efficient_black-box_attacks_on_visual_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
