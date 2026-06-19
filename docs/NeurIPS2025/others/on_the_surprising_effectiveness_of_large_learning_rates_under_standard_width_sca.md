---
title: >-
  [论文解读] On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling
description: >-
  [NeurIPS 2025 Spotlight][infinite-width limit] 揭示在标准参数化(SP)下，cross-entropy 损失函数使得"不稳定"区间实际分为灾难性不稳定和受控发散两个子区间：在受控发散区间（学习率 $\eta_n = \Theta(n^{-1/2})$）logits 发散但梯度和激活保持稳定，从而首次为 SP 提供了一个实用的、具有特征学习能力的无穷宽极限。
tags:
  - "NeurIPS 2025 Spotlight"
  - "infinite-width limit"
  - "standard parameterization"
  - "learning rate scaling"
  - "cross-entropy loss"
  - "feature learning"
  - "controlled divergence"
---

# On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.22491](https://arxiv.org/abs/2505.22491)  
**作者**: Moritz Haas, Sebastian Bordt, Ulrike von Luxburg, Leena Chennuru Vankadara (Tübingen, UCL Gatsby)
**领域**: 其他  
**关键词**: infinite-width limit, standard parameterization, learning rate scaling, cross-entropy loss, feature learning, controlled divergence

## 一句话总结

揭示在标准参数化(SP)下，cross-entropy 损失函数使得"不稳定"区间实际分为灾难性不稳定和受控发散两个子区间：在受控发散区间（学习率 $\eta_n = \Theta(n^{-1/2})$）logits 发散但梯度和激活保持稳定，从而首次为 SP 提供了一个实用的、具有特征学习能力的无穷宽极限。

## 研究背景与动机

### 核心矛盾
无穷宽理论（Tensor Program 等）预测：在标准参数化（He 初始化 + 全局学习率）下，学习率大于 $\mathcal{O}(1/n)$ 时训练不稳定，而 $\mathcal{O}(1/n)$ 时特征学习消失（退化为 kernel regime）。然而实践中：
- 最优学习率的衰减速度远慢于 $\mathcal{O}(1/n)$，通常约 $\Omega(1/\sqrt{n})$
- 网络即使在很大宽度下仍能稳定训练并学到有意义的特征
- 这一差距在浅层（2层）MLP 的**单步**更新中就已显著存在，排除了深度累积和多步有限宽效应作为主要解释

### 已有解释的不足

**有限宽效应**：差距在 2 层 MLP 单步更新中更显著（Figure F.15），排除了深度/训练时间累积

**Catapult 机制**：在 SP 下线性模型分析表明，catapult 动态需要 $\eta_n = \mathcal{O}(n^{-1})$ 才能避免 sharpness 发散，无法解释更大学习率的稳定性

**对齐假设失效**：Everett et al. (2024) 假设权重与激活之间缺乏对齐可解释差异，但本文通过 Refined Coordinate Check (RCC) 验证无穷宽对齐预测在中等宽度即成立

### 核心问题
为什么 SP 在大学习率下仍然稳定且有效？是否存在一个更贴近实际有限宽网络行为的无穷宽极限？

## 方法详解

### 理论框架：受控发散区间

考虑 $(L+1)$ 层 MLP，宽度 $n$，SP 初始化，SGD 全局学习率 $\eta_n = \eta \cdot n^{-\alpha}$。

**关键观察**：损失函数的选择决定了 logit 发散的后果。
- **MSE 损失**：损失-logit 梯度 $\chi_t = f_t(\xi_t) - y_t$，logit 发散直接导致梯度发散，进而引发灾难性不稳定
- **CE 损失**：损失-logit 梯度 $\chi_t = \sigma(f_t(\xi_t)) - y_t$，softmax 将 logit 映射到 $[0,1]$，logit 发散仅使 $\sigma(f)$ 趋向 one-hot 预测，梯度保持有界

### Proposition 2：SP 下的三个渐近区间（CE 损失）

| 区间 | 学习率指数 $\alpha$ | Logits $\|f_t\|_{RMS}$ | 激活 $\|x_t^l\|_{RMS}$ | 梯度 $\|\chi_t\|_{RMS}$ |
|---|---|---|---|---|
| 稳定区间 | $\alpha \geq 1$ | $\mathcal{O}(1)$ | $\Theta(1)$ | $\mathcal{O}(1)$ |
| **受控发散** | $\frac{1}{2} \leq \alpha < 1$ | $\Theta(n^{1-\alpha}) \to \infty$ | $\Theta(1)$ | $\mathcal{O}(1)$ |
| 灾难性不稳定 | $\alpha < \frac{1}{2}$ | $\to \infty$ | $\to \infty$ | $\to \infty$ |

对比 MSE 损失：$\alpha < 1$ 时直接灾难性不稳定，**不存在受控发散区间**。

### Proposition 4：受控发散区间的特征学习

在 CE 损失 + SP + $\eta_n = \eta \cdot n^{-\alpha}$，$\frac{1}{2} \leq \alpha < 1$ 下：
- 输入层特征学习消失：$\|\Delta x_t^1\|_{RMS} = \Theta(n^{-1/2 - \alpha})$
- **隐藏层特征学习非零**：$\|\Delta x_t^l\|_{RMS} = \Theta(n^{1/2 - \alpha})$，$l \in [2, L]$
- 特别地，当 $\alpha = 1/2$ 时，所有隐藏层的特征学习与宽度无关：$\|\Delta x_t^l\|_{RMS} = \Theta(1)$

这是 **SP 在实用特征学习区间的首个无穷宽极限**。

### 不同优化器和架构的最大稳定学习率

不同层类型的最大稳定学习率指数不同：
- **输出层**（logits）：$\eta_n = \mathcal{O}(n^{-1})$（但 CE 损失下可以超越）
- **隐藏层**：$\eta_n = \mathcal{O}(n^{-1/2})$
- **输入层/LayerNorm/Embedding**：$\eta_n = \mathcal{O}(1)$

对于 Adam：梯度归一化进一步稳定训练，$\eta_n = \Theta(n^{-1})$ 即可达到宽度无关更新（类似 $\mu$P）。Transformer 中可训练 LayerNorm 参数起到关键稳定作用，将最大稳定学习率从 $n^{-1}$ 提升到 $n^{-1/2}$。

### Refined Coordinate Check (RCC)

提出将有效更新 $(\Delta W_t^l) x_t^{l-1}$ 和传播更新 $W_0^l (\Delta x_t^{l-1})$ 分开测量的诊断工具，可在中等宽度（$n \leq 512$）准确估计 width-scaling 指数。

## 实验关键数据

### 实验设置
- **架构**：MLP（2-8层）、Pythia-GPT（至 1.4B 参数）
- **数据**：CIFAR-10、MNIST、Fashion-MNIST、DCLM-Baseline 语言数据
- **优化器**：SGD、Adam、AdamW
- **宽度范围**：至 16384（MLP）、4096（GPT）
- **参数化**：SP、$\mu$P、SP-full-align

### Table: CE vs MSE 损失下最大稳定学习率指数预测 vs 实测

| 参数化 | 损失函数 | 理论预测 max-stable $\alpha$ | 实测 max-stable $\alpha$ | 实测 optimal $\alpha$ |
|---|---|---|---|---|
| SP | MSE | $-1$ | $\approx -1$ | $\approx -1$ |
| SP | CE | $-0.5$ | $\approx -0.5$ | $\approx -0.5$ |
| SP-full-align | CE | $0$ | $\approx 0$ | $\approx 0$（语言）/ 递减（视觉） |

理论预测与实验高度一致。在深层非线性网络中，最大稳定学习率通常主导最优学习率。

### Table: $\mu$P 下 CE vs MSE 性能对比（8 层 MLP，SGD，最优训练准确率）

| 数据集 | SP + CE | SP + MSE | $\mu$P + CE | $\mu$P + MSE |
|---|---|---|---|---|
| MNIST | **~98%** | ~85% | **~98%** | ~97% |
| CIFAR-10 | **~50%** | ~28% | **~52%** | ~50% |
| Fashion-MNIST | **~87%** | ~65% | **~88%** | ~86% |

关键发现：
- SP 下 CE 显著优于 MSE（因 MSE 在大宽度下丧失特征学习）
- $\mu$P 下两种损失性能接近（$\mu$P 保证层间平衡的特征学习）
- 这为 CE 损失在深度学习中的实践主导地位提供了 width-scaling 理论解释

### 关于 SP-full-align 的新发现
- Everett et al. (2024) 推荐的 SP-full-align（SP + $\mu$P 学习率）在**视觉数据集上学习率迁移失败**
- 原因：$W_0^{L+1}$ 与 $\Delta x_t^L$ 之间存在宽度无关的对齐，导致 logit 随宽度发散
- 语言数据上迁移成功是因为输出维度 $d_{\text{out}} \gg n$，使初始算子范数近似宽度无关

## 亮点

- **首个实用的 SP 特征学习无穷宽极限**：在 CE 损失、$\eta_n = \Theta(n^{-1/2})$ 下，所有隐藏层具有宽度无关的特征学习，解决了长期存在的理论-实践差距
- **精细的不稳定区间刻画**：将之前笼统归为"不稳定"的区间分为两个本质不同的子区间，揭示 CE 损失的受控发散是关键机制
- **CE 损失主导地位的 scaling 理论解释**：首次从宽度缩放角度解释为什么 CE 损失在深度学习中远优于 MSE——SP 下 CE 允许更大学习率并保留特征学习，而 $\mu$P 下两者差距消失
- **实用诊断工具 RCC**：将有效更新和传播更新分离测量的 Refined Coordinate Check，可在中等宽度准确估计 scaling 指数，已开源
- **可训练 LayerNorm 的重要性**：从 scaling 角度解释了为什么现代架构几乎都使用可训练 LayerNorm 参数——它们将 Transformer 的最大稳定学习率从 $n^{-1}$ 提升到 $n^{-1/2}$

## 局限性

- **仅分析单 epoch 训练**：多 epoch 设置下的动态行为（如过拟合交互）留作未来工作
- **最优学习率指数的精确预测仍困难**：理论仅预测最大稳定指数，最优指数还依赖架构和数据分布的强假设
- **数值精度限制**：在标准浮点精度下，logit 发散可能在中等宽度超出数值范围，需要特殊实现来减少宽度依赖因子的累积
- **输入层特征学习始终消失**：$\alpha = 1/2$ 的 SP 仍无法恢复输入层特征学习，完整的宽度无关训练仍需 $\mu$P
- **CE + SP 的 logit 发散可能导致过度自信**：logit 快速增长可能恶化模型校准（calibration），$\mu$P 下模型可能更好校准

## 相关工作

- **无穷宽极限**：NTK (Jacot et al., 2018)、Mean-field (Mei et al., 2018; Chizat & Bach, 2018)、Tensor Program (Yang & Hu, 2021) → 本文将 TP 理论扩展到 SP 的受控发散区间
- **$\mu$P 与超参迁移**：Yang et al. (2022) 提出 $\mu$P 实现宽度无关超参数 → 本文解释为何 SP 在实践中也近似有效（CE 损失的功劳）
- **Edge of stability**：Cohen et al. (2021) 发现训练趋向 $2/\eta$ 的 sharpness → 本文分析 catapult 机制在 SP 下的局限
- **大学习率的益处**：Andriushchenko et al. (2023b) 大步长学习稀疏特征、Cai et al. (2024) margin 改进 → 本文从宽度缩放角度提供互补理论
- **SP-full-align**：Everett et al. (2024) 推荐 SP + $\mu$P 学习率作为最佳实践 → 本文指出其在视觉数据上迁移失败，并给出修正初始化方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次识别 CE 损失在宽度缩放中的关键作用，受控发散区间的发现极具洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ — MLP+GPT、SGD+Adam、CE+MSE、SP+μP 全面交叉验证，理论预测与实验高度一致
- 写作质量: ⭐⭐⭐⭐ — 逻辑严谨，主要结论清晰表述；但数学符号密集，对非理论背景读者有一定门槛
- 价值: ⭐⭐⭐⭐⭐ — 解决了无穷宽理论与实践的长期矛盾，对大规模模型训练的超参数选择有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SMRS: Advocating a Unified Reporting Standard for Surrogate Models in the Artificial Intelligence Era](smrs_advocating_a_unified_reporting_standard_for_surrogate_models_in_the_artific.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](../../ICML2025/others/residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[CVPR 2026\] Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes](../../CVPR2026/others/multi-view_crowd_tracking_transformer_with_view-ground_interactions_under_large_.md)

</div>

<!-- RELATED:END -->
