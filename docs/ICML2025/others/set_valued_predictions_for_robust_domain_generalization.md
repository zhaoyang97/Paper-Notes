---
title: >-
  [论文解读] Set-Valued Predictions for Robust Domain Generalization
description: >-
  [ICML2025][集值预测] 提出集值预测器（set-valued predictor）解决域泛化（DG）中的鲁棒性问题：输出标签子集而非单一标签，使预测在尽可能多的未见域上满足预定义的覆盖率要求，同时最小化预测集大小。 - 域泛化的核心挑战：模型在训练分布上表现好，但在未见测试分布（域）上性能下降严重…
tags:
  - "ICML2025"
  - "集值预测"
  - "域泛化"
  - "鲁棒性"
  - "共形预测"
  - "VC维"
  - "最坏情况保证"
---

# Set-Valued Predictions for Robust Domain Generalization

**会议**: ICML2025  
**arXiv**: [2507.03146](https://arxiv.org/abs/2507.03146)  
**代码**: [ront65/set-valued-ood](https://github.com/ront65/set-valued-ood)  
**领域**: LLM评测  
**关键词**: 集值预测, 域泛化, 鲁棒性, 共形预测, VC维, 最坏情况保证

## 一句话总结

提出集值预测器（set-valued predictor）解决域泛化（DG）中的鲁棒性问题：输出标签子集而非单一标签，使预测在尽可能多的未见域上满足预定义的覆盖率要求，同时最小化预测集大小。

## 研究背景与动机

- **域泛化的核心挑战**：模型在训练分布上表现好，但在未见测试分布（域）上性能下降严重。在医疗等高风险场景中，需要最坏情况下的性能保证，而非仅仅平均性能。
- **现有方法的局限**：
    - 基于不变特征的方法（如 IRM、DRO）追求跨域稳定性，但常以性能下降为代价——一个在所有域上只有 60% 准确率的"稳定"预测器并无实用价值。
    - 所有现有方法依赖**单值预测**（singleton prediction），在不同域之间不存在单一最优预测时鲁棒性天然受限。
- **核心洞察**：不同域诱导 $X$ 与 $Y$ 之间不同的概率关系，期望单一预测器在所有域上最优是不切实际的。集值输出可以捕获从训练域学到的一系列潜在关系，从而增强鲁棒性。
- **关键权衡**：预测集不能太大（退化为预测整个标签空间则毫无价值），因此目标是**达到预定义覆盖率的同时最小化预测集大小**。

## 方法详解

### 问题形式化

设特征 $X \in \mathcal{X}$，标签 $Y \in \mathcal{Y}$（有限集），域 $e \in \mathcal{E}$ 对应分布 $P_e$。训练阶段观测 $m$ 个域，每个域 $e$ 有 $n_e$ 个样本。测试时面对未见域 $e_{test}$。

集值预测器定义为 $h: \mathcal{X} \to \mathcal{P}(\mathcal{Y})$，输出标签的子集。

### 学习目标

定义域级损失为最坏标签召回率：

$$\mathcal{L}_{recall}(h, e) = \max_{y \in \mathcal{Y}} P_e[y \notin h(X) | Y = y]$$

定义性能指示器 $\mathbb{1}_{\mathcal{L},\gamma}(h,e) = \mathbb{1}[\mathcal{L}(h,e) \geq \gamma]$，优化目标为：

$$\min_{h \in \mathcal{H}} P_{e \sim D}[\mathcal{L}(h,e) \geq \gamma]$$

即最小化预测器未能在域上达到 $1-\gamma$ 召回率的概率。

### 理论贡献：层级 VC 维

- 将经典 VC 维推广到**多域层级框架**：衡量假设集 $\mathcal{H}$ 在域空间 $\mathcal{E}$ 上的打散能力。
- **定理 3.4**：$\mathcal{H}$ 具有一致收敛性 ⟺ $VCdim_{\mathcal{L},\gamma}(\mathcal{H}) < \infty$，所需域数量 $m = \Theta\!\left(\frac{d + \log(1/\delta)}{\epsilon^2}\right)$。
- **负面结果（定理 3.6）**：当特征维度 $d > 1$ 时，线性假设集在无约束域空间上 VC 维为无穷大——域泛化比经典学习更难。
- **正面结果（定理 3.7）**：若域满足条件高斯假设（协方差结构共享至缩放因子），线性假设集 VC 维有限，所需训练域数 $m = \Theta\!\left(\frac{|\mathcal{Y}|^2(d + \log(|\mathcal{Y}|/\delta))}{\epsilon^2}\right)$。

### SET-COVER 算法

约束优化问题：最小化预测集大小，约束每个训练域每个标签的召回率 $\geq 1-\gamma$：

$$\min_\theta \sum_{i \in S} |h(x_i)| \quad \text{s.t.} \quad \frac{1}{|G_{e,y}|}\sum_{i \in G_{e,y}} \mathbb{1}[y \notin h(X_i)] \leq \gamma, \; \forall e, y$$

将集值预测器分解为每标签二分类器 $h_y$，引入参数化 $h_y^\theta$ 和 hinge 松弛后得到可微优化问题。

通过拉格朗日乘子 $C_{e,y}$ 求解：
- 对 $\theta$ 做梯度下降（最小化预测集大小 + 满足约束）
- 对 $C$ 做梯度上升（违反约束时增大乘子），每隔若干 batch 更新一次

最终损失函数：

$$L_y(\theta, C) = \sum_{i \notin G_y} \max\{0, 1+h_y^\theta(X_i)\} + \sum_{e} \mathbb{1}_{i \in G_{e,y}} C_{e,y} \max\{0, 1-h_y^\theta(X_i)\}$$

### 基线：Robust Conformal Prediction

在合并训练数据上训练分类器获取 logits 作为共形分数，**分域分标签**校准阈值 $t_{e,y}$，测试时取所有训练域阈值的最松准则（任一域通过即纳入预测集）。

## 实验关键数据

### 合成数据

| 设置 | 方法 | Min-Recall | Avg Set Size |
|------|------|-----------|-------------|
| d=50 | ERM | 低（不达标） | 1.0 |
| d=50 | Robust Conformal | 达标 | 较大 |
| d=50 | **SET-COVER** | **达标** | **更小** |

### WILDS 真实数据（目标 min-recall ≥ 90%）

| 数据集 | 方法 | Median Min-Recall | Median Avg Size | ≥90% 域比例 |
|--------|------|-------------------|-----------------|------------|
| Camelyon | ERM | 0.93 | 1.0 | 63% |
| Camelyon | Robust Conformal | **0.98** | 1.79 | 93% |
| Camelyon | **SET-COVER** | 0.96 | **1.05** | 71% |
| FMoW | ERM | 0.76 | 1.0 | 8% |
| FMoW | Robust Conformal | 0.89 | 1.17 | 43% |
| FMoW | **SET-COVER** | **0.91** | **1.10** | **72%** |
| iWildCam | ERM | 0.99 | 1.0 | 71% |
| iWildCam | Robust Conformal | 0.99 | 2.00 | 86% |
| iWildCam | **SET-COVER** | 0.99 | **1.01** | 82% |
| Amazon | ERM | 1.0 | 1.0 | 68% |
| Amazon | Robust Conformal | 1.0 | 4.68 | **99%** |
| Amazon | **SET-COVER** | 1.0 | **2.43** | 96% |

### 关键发现

- SET-COVER 和 Robust Conformal 均显著优于 ERM 和 Pooling-CDFs 基线。
- SET-COVER 在达到相近覆盖率的同时，预测集大小**远小于** Robust Conformal（如 Amazon: 2.43 vs 4.68；Camelyon: 1.05 vs 1.79）。
- 在 FMoW 上 SET-COVER 的 ≥90% 域比例（72%）远超 Robust Conformal（43%），说明端到端优化比后处理校准更有效。
- 协方差矩阵跨域随机变化时，SET-COVER 性能依然稳定（附录实验）。

## 亮点与洞察

1. **范式转换**：从"追求单一最优预测"到"输出小而鲁棒的预测集"，为域泛化提供了全新视角。
2. **理论严谨**：首次定义多域层级框架下的 VC 维并证明泛化界，揭示了域泛化本质上比经典学习更困难（$d>1$ 时线性假设 VC 维无穷）。
3. **实用性强**：SET-COVER 算法通过拉格朗日对偶优化，可与标准神经网络架构无缝结合，无需额外校准数据。
4. **集大小优势显著**：相比 Robust Conformal 这一强基线，SET-COVER 在 4 个 WILDS 数据集上一致保持更小的预测集。

## 局限与展望

- **理论-实践差距**：定理 3.7 仅对线性预测器在条件高斯域上严格成立，实验中用神经网络缺乏理论保证。
- **标签数量扩展性**：当 $|\mathcal{Y}|$ 很大时（如 ImageNet 级别），每标签独立训练二分类器的方案计算开销大。
- **域信息需求**：训练阶段需要域标签来分组约束，在某些场景中域边界不明确。
- **未考虑条件覆盖**：当前目标是域级 min-recall，但未讨论条件覆盖（给定 $X$ 的覆盖率保证）。
- **实验规模有限**：仅在 2-3 类子集上测试 FMoW 和 iWildCam，全类别场景未验证。

## 相关工作与启发

- **共形预测**（Conformal Prediction）：本文的 Robust Conformal 基线是 CP 在多域场景的自然扩展，但 CP 本身只提供边际覆盖保证。
- **Pooling CDFs**（Dunn et al., 2018）：针对回归任务的 DG 集值方法，优化平均覆盖而非最坏情况。
- **QRM**（Eastwood et al., 2022）：最小化域间 α-分位数风险的单值预测方法，受限于单值输出的鲁棒性天花板。
- **不变学习**（IRM, DRO 等）：追求不变特征的稳定性，但常忽略有价值的域相关特征。
- **启发**：集值预测的思路可推广到其他 OOD 场景（如 covariate shift、label shift），VC 维的层级推广还可用于 meta-learning 的理论分析。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 集值预测 + 域泛化的结合视角新颖，层级 VC 维理论框架有原创性
- 实验充分度: ⭐⭐⭐⭐ — 合成 + 4 个 WILDS 数据集，含消融和多种基线对比，但标签数受限
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，问题动机阐述充分，结构合理
- 价值: ⭐⭐⭐⭐ — 为域泛化提出了实用且有理论支撑的新范式，对高风险场景有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](../../CVPR2025/others/gradient-guided_annealing_for_domain_generalization.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/others/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](../../ICCV2025/others/adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[CVPR 2026\] Bridging Domain Expertise and Generalization for Performance Estimation](../../CVPR2026/others/bridging_domain_expertise_and_generalization_for_performance_estimation.md)

</div>

<!-- RELATED:END -->
