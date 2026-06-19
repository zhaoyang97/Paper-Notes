---
title: >-
  [论文解读] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models
description: >-
  [NeurIPS 2025][AI安全][machine unlearning] 提出 DecoRemoval 框架，通过判别性保持的因子去相关（基于随机傅里叶特征的空间映射+自适应权重）和平滑损失扰动两大模块，在不重训的前提下实现数据移除，尤其在分布外（OOD）场景下显著优于现有方法。 领域现状：数据移除（machine…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "machine unlearning"
  - "certified removal"
  - "factor decorrelation"
  - "OOD robustness"
  - "random Fourier features"
---

# Factor Decorrelation Enhanced Data Removal from Deep Predictive Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.23443](https://arxiv.org/abs/2509.23443)  
**代码**: [GitHub](https://anonymous.4open.science/r/DecoRemoval-770220/)  
**领域**: AI Safety / 机器遗忘  
**关键词**: machine unlearning, certified removal, factor decorrelation, OOD robustness, random Fourier features

## 一句话总结
提出 DecoRemoval 框架，通过判别性保持的因子去相关（基于随机傅里叶特征的空间映射+自适应权重）和平滑损失扰动两大模块，在不重训的前提下实现数据移除，尤其在分布外（OOD）场景下显著优于现有方法。

## 研究背景与动机

**领域现状**：数据移除（machine unlearning）是隐私合规的刚性需求。认证式移除（Certified Removal）通过梯度更新和校准噪声注入实现高效的参数调整，是目前最具理论保障的方法。

**现有痛点**：现有数据移除方法假设移除前后的数据分布相似，但实际中持续的移除请求会导致分布漂移（distributional shift）。在 OOD 场景下，特征表示与标签的内在相关性发生变化，现有遗忘机制的移除精度和泛化能力都会下降。

**核心矛盾**：特征降维是处理 OOD 的有效策略，但可能丢弃判别性特征；损失函数若不适配降维后的表示空间，梯度方向会偏离优化目标，导致训练不稳定和泛化退化。

**本文目标**：(1) 在移除过程中降低特征冗余和相关性以对抗分布漂移；(2) 在移除过程中保证隐私安全，防止信息泄露。

**切入角度**：借用 StableNet 的随机傅里叶特征（RFF）实现特征空间映射和去相关，同时通过随机线性扰动平滑损失函数，使局部参数调整能准确近似重训效果。

**核心 idea**：特征去相关+损失扰动 → 在分布漂移下仍能用局部参数更新替代全量重训。

## 方法详解

### 整体框架
DecoRemoval 包含两个主要模块：
- **输入**：训练好的模型 $A(D)$、待移除数据 $(x_n, y_n)$
- **模块一**：判别性保持的因子去相关（DP-FD）——用 RFF 映射降维去相关
- **模块二**：平滑数据移除（Smoothed Removal）——损失扰动+Newton 更新移除
- **输出**：移除后的模型参数 $w_{clf}^-$

### 关键设计

1. **随机傅里叶特征映射（RFF Mapping）**:

    - 功能：将输入特征映射到高维空间以实现去相关
    - 核心思路：$Z_i = \sqrt{2} \cdot \cos(\omega X_i + \phi)$，其中 $\omega \sim \mathcal{N}(0, I)$, $\phi \sim \text{Uniform}(0, 2\pi)$
    - 该变换基于核函数的傅里叶变换，在不直接计算核函数的前提下实现特征嵌入
    - 设计动机：线性计算复杂度实现特征降维和去相关，减少特征冗余

2. **样本加权去相关（Feature Decorrelation via Sample Weighting）**:

    - 功能：通过优化样本权重最小化变换后特征间的统计依赖
    - 核心思路：基于交叉协方差算子 $\Sigma_{AB}$ 度量特征依赖性，用 Frobenius 范数 $I_{AB} = \|\hat{\Sigma}_{AB}\|_F^2$ 量化
    - 加权协方差矩阵：$\hat{\Sigma}_{AB;w} = \frac{1}{n-1} \sum_{i=1}^n [(w_i u(Z_i) - \bar{w u})^T \cdot (w_i v(Z_i) - \bar{w v})]$
    - 最优权重：$w^* = \arg\min_{w \in \Delta_n} \sum_{1 \leq i < j \leq m_Z} \|\hat{\Sigma}_{Z_{:,i}Z_{:,j};w}\|_F^2$
    - 设计动机：单纯 RFF 可能分散判别信息，通过自适应权重保留类别区分性

3. **损失扰动与认证式移除（Loss Perturbation + Certified Removal）**:

    - 功能：在训练损失中注入随机线性项以混淆梯度信号，然后用 Newton 更新实现移除
    - 扰动损失：$L_{\mathbf{p}}(w_{clf}; D) = \sum_{i=1}^n L(w_{clf}^\top x_i, y_i) + \mathbf{b}^\top w_{clf}$
    - Newton 更新移除：$w_{clf}^- = w_{clf}^* + H_{w_{clf}^*}^{-1} \nabla L(w_{clf}^*; (x_n, y_n))$
    - 其中 $H_{w_{clf}^*} = \nabla^2 L(w_{clf}^*; D')$ 是 Hessian 矩阵
    - 关键性质：线性扰动 $\mathbf{b}^\top w_{clf}$ 使梯度偏移常数 $\mathbf{b}$ 但不改变 Hessian，因此移除更新仍然有效
    - 设计动机：$\mathbf{b}$ 混淆了特定样本的梯度信号，降低信息泄露风险

### 认证式移除保证
核心定义：移除机制 $M$ 满足 $\epsilon$-CR 当：

$$e^{-\epsilon} \leq \frac{P(M(A(D), D, x) \in S)}{P(A(D \setminus x) \in S)} \leq e^{\epsilon}$$

即移除单个数据点对模型输出的影响不超过指数因子 $\epsilon$。

## 实验关键数据

### 主实验：OOD 场景下 ACC(%) 和 F1 对比

| 数据集 | 移除量 | Retrain | CR | SISA | DP-SGD | SSD | CU | **DR (Ours)** |
|--------|--------|---------|-----|------|--------|-----|-----|-------------|
| MNIST | 1K | 51.75 | 43.13 | 43.64 | 45.78 | 45.45 | 47.35 | **48.97** |
| MNIST | 10K | 51.02 | 41.87 | 42.96 | 44.87 | 45.03 | 46.53 | **48.34** |
| CIFAR-10 | 1K | 50.76 | 43.09 | 43.21 | 45.30 | 45.06 | 46.84 | **48.56** |
| CIFAR-10 | 10K | 50.01 | 41.76 | 42.51 | 44.37 | 44.51 | 46.11 | **47.83** |
| SST-2 | 1K | 91.76 | 89.71 | 89.98 | 90.45 | 89.98 | 89.94 | **90.45** |
| ESS | 1K | 55.43 | 48.61 | 48.64 | 50.47 | 50.15 | 51.34 | **54.97** |
| CGSS | 1K | 51.60 | 41.24 | 43.52 | 46.76 | 47.77 | 48.77 | **50.82** |

### F1 Score 对比（移除 10K 样本）

| 数据集 | Retrain | CR | SSD | CU | **DR (Ours)** |
|--------|---------|-----|-----|-----|-------------|
| MNIST | 0.495 | 0.390 | 0.450 | 0.450 | **0.473** |
| CIFAR-10 | 0.491 | 0.389 | 0.443 | 0.444 | **0.469** |
| SST-2 | 0.839 | 0.796 | 0.816 | 0.805 | **0.821** |
| ESS | 0.530 | 0.390 | 0.478 | 0.470 | **0.510** |
| CGSS | 0.502 | 0.454 | 0.468 | 0.477 | **0.495** |

### 关键发现
- DecoRemoval 在所有 5 个数据集、3 种移除规模下一致性地最接近全量重训（Retrain）结果
- 在社会调查数据（ESS/CGSS）上提升明显：比次优方法（CU）ACC 高 3-4 个百分点，因为这类数据特征相关性高、RFF 去相关效果显著
- 大规模移除（10K）时优势更大：其他方法性能快速退化，DecoRemoval 保持稳定
- SST-2 文本数据上改进相对较小（~0.5%），因文本特征经过预训练已较好去相关

## 亮点与洞察
- **RFF→去相关→数据移除**的组合路线新颖：首次将特征去相关引入 unlearning，解决了 OOD 下移除效果退化的根本问题
- **损失扰动的双重作用**：既作为正则化平滑优化景观（使 Newton 更新有效），又作为隐私保护手段（混淆梯度信号）
- 线性扰动不改变 Hessian 的理论性质保证了移除机制在扰动下的**正确性不受影响**
- 实验设计中 OOD 构造方式（随机选 10% A 类样本放入 B 类测试集）简单但有效

## 局限与展望
- OOD 设置较为人工：仅随机交换 10% 样本构造分布偏移，真实场景的分布漂移可能更复杂
- 只评估分类任务（MLP backbone），缺少在 Transformer 或更深网络上的验证
- RFF 映射的维度 $m_Z$ 和核函数选择对性能的影响未充分消融
- 认证式移除的 $\epsilon$ 保证在非凸深度网络上理论基础较弱，论文将网络拆分为特征提取+线性分类器来规避此问题
- 推理开销未充分报告：权重优化和 Hessian 计算的实际时间成本

## 相关工作与启发
- **vs Certified Removal (Guo et al.)**: CR 是基础但假设分布不变；DecoRemoval 通过 RFF 去相关扩展到 OOD
- **vs SSD (Foster et al.)**: SSD 用 Fisher 信息矩阵选择性抑制参数，关注遗忘集的参数灵敏度；DecoRemoval 从特征空间角度出发
- **vs StableNet**: StableNet 将 RFF 用于 OOD 分类稳定性；DecoRemoval 将其引入 unlearning 场景，是方法迁移的典型案例

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将特征去相关引入数据移除，RFF+损失扰动的组合有创意
- 实验充分度: ⭐⭐⭐⭐ 5个数据集、3种移除规模、6种基线对比，覆盖图像/文本/结构化数据
- 写作质量: ⭐⭐⭐ 行文可读但部分公式符号不一致，OOD 设置描述较简略
- 价值: ⭐⭐⭐⭐ 解决了 unlearning 在分布漂移下的实际痛点，方法路线值得关注

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](preserving_task-relevant_information_under_linear_concept_removal.md)
- [\[NeurIPS 2025\] Distributional Adversarial Attacks and Training in Deep Hedging](distributional_adversarial_attacks_and_training_in_deep_hedging.md)
- [\[AAAI 2026\] Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data](../../AAAI2026/ai_safety/privacy_on_the_fly_a_predictive_adversarial_transformation_network_for_mobile_se.md)
- [\[NeurIPS 2025\] Provable Watermarking for Data Poisoning Attacks](provable_watermarking_for_data_poisoning_attacks.md)

</div>

<!-- RELATED:END -->
