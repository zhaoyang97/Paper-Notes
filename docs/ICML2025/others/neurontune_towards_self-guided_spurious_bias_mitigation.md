---
title: >-
  [论文解读] NeuronTune: Towards Self-Guided Spurious Bias Mitigation
description: >-
  [ICML2025][虚假相关性] NeuronTune 提出一种无需组标签：的自引导去偏方法：通过对比模型隐空间中正确/错误预测样本的神经元激活差异，识别受虚假偏差影响的维度并将其置零，再重训最后一层分类器，从而显著提升 worst-group accuracy。 - 虚假相关性问题：ERM 训练的模型容易依赖非因果特征（…
tags:
  - "ICML2025"
  - "虚假相关性"
  - "偏差缓解"
  - "神经元剪枝"
  - "最后一层重训练"
  - "worst-group accuracy"
---

# NeuronTune: Towards Self-Guided Spurious Bias Mitigation

**会议**: ICML2025  
**arXiv**: [2505.24048](https://arxiv.org/abs/2505.24048)  
**代码**: [GitHub](https://github.com/gtzheng/NeuronTune)  
**领域**: 鲁棒性/去偏 (Robustness / Debiasing)  
**关键词**: 虚假相关性, 偏差缓解, 神经元剪枝, 最后一层重训练, worst-group accuracy

## 一句话总结

NeuronTune 提出一种**无需组标签**的自引导去偏方法：通过对比模型隐空间中正确/错误预测样本的神经元激活差异，识别受虚假偏差影响的维度并将其置零，再重训最后一层分类器，从而显著提升 worst-group accuracy。

## 研究背景与动机

- **虚假相关性问题**：ERM 训练的模型容易依赖非因果特征（如水鸟分类中依赖水背景而非鸟本身），在缺乏这些虚假关联的测试数据上表现很差
- **现有方法的局限**：
    - **有监督方法**需要组标签 $(y, a)$ 标注哪种虚假属性与哪个类别关联，标注成本高
    - **半监督方法**（JTT, DFR, AFR 等）仍需验证集上的组标签做模型选择
    - **样本级方法**无法直接干预模型内部决策机制，控制精度有限
- **核心动机**：能否从模型内部自动发现受虚假偏差影响的神经元，直接干预决策过程，做到**完全无组标签**的去偏？

## 方法详解

### 整体流程

NeuronTune 是一个 **post hoc** 方法，应用于已训练好的 ERM 模型 $f_\theta = h_{\theta_2} \circ e_{\theta_1}$（特征提取器 + 线性分类头），分三步：

**Step 1：提取嵌入与预测结果**

对识别数据集 $\mathcal{D}_{\text{Ide}}$（默认用验证集），提取每个样本的隐层嵌入 $\mathbf{v} = e_{\theta_1}(\mathbf{x}) \in \mathbb{R}^M$ 及其预测是否正确的标记 $o$。

**Step 2：识别偏差维度**

对每个类别 $y$ 和每个嵌入维度 $i$，将激活值按预测正确/错误分成两组，计算**虚假性得分**：

$$\delta_i^y = \text{Med}(\bar{\mathcal{V}}_i^y) - \text{Med}(\hat{\mathcal{V}}_i^y)$$

其中 $\bar{\mathcal{V}}_i^y$ 是第 $i$ 维上**错误预测**样本的激活值集合，$\hat{\mathcal{V}}_i^y$ 是**正确预测**样本的激活值集合。

- $\delta_i^y > 0$：该维度激活高时反而导致误分类 → 受虚假偏差影响
- $\delta_i^y < 0$：该维度激活高时有助于正确分类 → 核心特征

识别偏差维度集合：$\mathcal{S} = \{i \mid \delta_i^y > \lambda,\; \forall y \in \mathcal{Y}\}$，默认阈值 $\lambda = 0$。

**Step 3：抑制偏差维度 + 重训最后一层**

冻结特征提取器 $e_{\theta_1}$，将嵌入中偏差维度的激活**置零**得到 $\tilde{\mathbf{v}}$，用类别均衡采样在 $\mathcal{D}_{\text{Tune}}$（默认用训练集）上重训分类头：

$$\theta_2^* = \arg\min_{\theta_2} \mathbb{E}_{\mathcal{B} \sim \mathcal{D}_{\text{Tune}}} \ell(h_{\theta_2}(\tilde{\mathbf{v}}), y)$$

### 理论保证

- **Proposition 4.1**：当 $\gamma^T \mathbf{w}_{\text{spu},i} < 0$ 时，模型在虚假相关性失效时仍依赖虚假特征 → 应抑制该神经元
- **Theorem 4.2**：证明 $\delta_i^y \approx -2\mu \gamma^T \mathbf{w}_{\text{spu},i}$，即虚假性得分正值对应需抑制的偏差神经元
- **Theorem 4.3**：NeuronTune 产生的模型参数比原始 ERM 模型更接近无偏最优解

### 模型选择：Spuriousness Fitness Score (SFit)

无组标签时用 SFit 选模型：$\text{SFit} = \sum_{m=1}^{M} \sum_{y \in \mathcal{Y}} |\delta_m^y|$，SFit 越高表示偏差/非偏差维度更可分，模型更适合去偏。

## 实验关键数据

### 图像数据集（Waterbirds / CelebA）

| 方法 | 组标签 | Waterbirds WGA↑ | CelebA WGA↑ |
|------|--------|----------------|-------------|
| ERM | - | 72.6 | 47.2 |
| JTT | 半监督 | 86.7 | 81.1 |
| DFR† | 半监督 | 92.4 | 87.0 |
| BAM (无监督) | 无 | 89.1 | 80.1 |
| **NeuronTune** | **无** | **92.2** | **83.1** |
| **NeuronTune†** | **无** | **92.5** | **87.3** |

### 文本数据集（MultiNLI / CivilComments）

| 方法 | 组标签 | MultiNLI WGA↑ | CivilComments WGA↑ |
|------|--------|--------------|-------------------|
| ERM | - | 67.9 | 57.4 |
| AFR | 半监督 | 73.4 | 68.7 |
| DFR† | 半监督 | 70.8 | 81.8 |
| **NeuronTune** | **无** | **72.1** | **82.4** |
| **NeuronTune†** | **无** | **72.5** | **82.7** |

### ImageNet-9 → ImageNet-A 分布偏移

| 方法 | ImageNet-9 Acc | ImageNet-A Acc | Acc Gap↓ |
|------|---------------|---------------|---------|
| ERM | 90.8 | 24.9 | 65.9 |
| LWBC | 94.0 | 36.0 | 58.0 |
| **NeuronTune** | 93.7 | **37.3** | **56.4** |

### 消融：完全抑制 vs 部分抑制（CelebA）

完全置零（masking=0）WGA **87.3%**；部分抑制（masking=0.2~1.0）WGA 仅 71~73% → 必须完全抑制才有效。

## 亮点与洞察

- **完全无组标签**：不同于 DFR/AFR 等需要验证集组标签做模型选择，NeuronTune 通过 SFit 实现自引导选择
- **神经元级干预**：从样本级操控升级到神经元级，提供更精准的去偏控制
- **理论完备**：从数据模型到选择指标到去偏保证，层层推导，理论支撑扎实
- **轻量 post hoc**：仅重训最后一层，计算开销极低，可适配任意预训练模型
- **跨模态通用**：在视觉（ResNet）和文本（BERT）数据上均有效

## 局限与展望

- **Avg Acc 会轻微下降**：去偏带来 WGA↑ 的同时整体准确率有 1~3% 的损失（平均-最差 accuracy 的 tradeoff）
- **识别数据选择敏感**：用训练集做 $\mathcal{D}_{\text{Ide}}$ 效果差（模型已记忆），必须用独立的验证集
- **线性假设**：理论分析基于线性数据模型和线性回归，实际深度网络中特征纠缠更复杂
- **阈值固定**：$\lambda = 0$ 在所有数据集上统一使用，未探索自适应阈值策略
- **仅操控最后一层**：特征提取器冻结不变，若特征本身已严重纠缠则改善空间有限

## 评分

⭐⭐⭐⭐ — 理论推导严谨、方法简洁实用、无需额外标注即可获得接近半监督方法的去偏效果，是一篇兼顾理论与实践的扎实工作。局限在于线性假设与实际深度网络的 gap，以及 avg acc 的损失。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Causal Estimation of Tokenisation Bias](../../ACL2025/others/causal_tokenisation_bias.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](../../NeurIPS2025/others/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[CVPR 2026\] Bias In, Bias Out? Finding Unbiased Subnetworks in Vanilla Models](../../CVPR2026/others/bias_in_bias_out_finding_unbiased_subnetworks_in_vanilla_models.md)
- [\[ICML 2025\] Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)
- [\[ECCV 2024\] Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](../../ECCV2024/others/rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)

</div>

<!-- RELATED:END -->
