---
title: >-
  [论文解读] Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels
description: >-
  [ECCV2024][社会计算][noisy label] 提出 DaSC 框架，通过分布感知的类中心估计（DaCC）和置信度感知的对比学习（SBCL + MIDL），同时解决长尾分布和噪声标签的联合问题，在 CIFAR 和真实噪声数据集上达到 SOTA。 现实数据往往同时存在长尾分布：和标签噪声：两个问题…
tags:
  - "ECCV2024"
  - "社会计算"
  - "noisy label"
  - "long-tailed learning"
  - "对比学习"
  - "sample selection"
  - "半监督学习"
---

# Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels

**会议**: ECCV2024  
**arXiv**: [2407.16802](https://arxiv.org/abs/2407.16802)  
**代码**: [GitHub](https://github.com/JaesoonBaik1213/DaSC)  
**领域**: 社会计算  
**关键词**: noisy label, long-tailed learning, contrastive learning, sample selection, semi-supervised learning

## 一句话总结
提出 DaSC 框架，通过分布感知的类中心估计（DaCC）和置信度感知的对比学习（SBCL + MIDL），同时解决长尾分布和噪声标签的联合问题，在 CIFAR 和真实噪声数据集上达到 SOTA。

## 背景与动机
现实数据往往同时存在**长尾分布**和**标签噪声**两个问题。单独解决其中一个已有较多工作，但联合处理（NL-LT）仍然具有挑战性：

- **长尾分布**：类别样本量严重不均衡，模型偏向头部类
- **噪声标签**：部分样本标注错误，模型会记忆噪声导致泛化性能下降
- 两者叠加使得准确识别真实数据分布更加困难

现有方法（如 RoLT、PCL、SFA）使用基于特征的噪声样本选择策略，通过计算每个类的中心来判断样本是否为噪声。但存在三个关键局限：

1. **仅使用目标类内高置信度样本**估计类中心——尾部类样本不足导致中心不可靠
2. **对所有样本赋予相等权重**——忽略了部分样本可能是错误标注的事实
3. **缺乏主动提升表示质量的机制**——尾部类的表示质量更需要增强

## 核心问题
如何同时应对长尾分布和噪声标签？具体来说：

- 如何在样本量不足且存在噪声的条件下，准确估计每个类的特征中心？
- 如何在噪声标签环境下学到平衡且鲁棒的特征表示？

## 方法详解

### 整体框架 DaSC
DaSC 包含四个核心组件：

1. **DaCC**（Distribution-aware Class Centroid Estimation）：分布感知的类中心估计
2. **噪声样本选择**：基于 GMM 区分干净/噪声样本
3. **SBCL**（Semi-supervised Balanced Contrastive Loss）：针对高置信度样本的半监督平衡对比损失
4. **MIDL**（Mixup-enhanced Instance Discrimination Loss）：针对低置信度样本的 Mixup 增强实例判别损失

模型架构包含共享特征提取器 $f$、常规分类器 $g^c$、平衡分类器 $g^b$（使用 Balanced Softmax）、伪标签生成器和 MLP 投影头 $q$。

### DaCC：分布感知的类中心估计
与传统方法仅使用目标类内样本不同，DaCC 利用**所有类的样本**来估计每个类的中心，关键是根据模型预测的置信度分配权重：

$$c_k = \text{Norm}\left(\sum_{x(i) \in \mathcal{D}^I} \hat{p}_k^c(i) \cdot z'(i)\right)$$

其中权重通过温度缩放（temperature scaling）的 softmax 获得：

$$\hat{p}_k^c(i) = \frac{\exp(p_k^c(i) / \tau_T)}{\sum_{k=1}^K \exp(p_k^c(i) / \tau_T)}$$

- 温度参数 $\tau_T = 0.1$，使得高置信度样本权重更大，低置信度样本权重被抑制
- 仅使用两个分类器（常规 + 平衡）预测均超过阈值 $\tau$ 的样本集 $\mathcal{D}^I$
- 阈值 $\tau$ 随训练逐步增长：$\tau = \phi^t \hat{\tau}$，$\phi=1.005$，$\hat{\tau}=1/K$

计算出中心后，通过样本特征与中心的余弦相似度，使用 GMM 拟合分配概率，将样本分为干净样本和噪声样本。

### SBCL：半监督平衡对比损失
对高置信度样本（伪标签最大概率 $> \tau_c$）应用改进的监督对比学习：

- 在对比损失的分母中，对每个类的贡献进行**均匀化**（取类内平均），避免头部类因为样本多而主导对比损失
- 利用可靠的伪标签信息来惩罚头部类，促进学习均衡表示

### MIDL：Mixup 增强实例判别损失
对低置信度样本（伪标签不可靠，不适合使用标签信息）应用自监督方式：

- 对同一样本生成两个增强视图 $x^1, x^2$ 作为 query 和 positive key
- 通过 Mixup 生成 $x^{mix} = \lambda x^1 + (1-\lambda) x^2$ 作为**硬负样本**
- Mixup 样本被放入 memory bank，增加负样本的多样性
- 通过实例判别任务提升低置信度样本的表示质量

### 总损失
$$\mathcal{L} = \mathcal{L}_{MixMatch} + \mathcal{L}_{BMixMatch} + \lambda_{SBCL} \mathcal{L}_{SBCL} + \lambda_{MIDL} \mathcal{L}_{MIDL}$$

训练分两个阶段：
- **Warmup 阶段**（前 30 epochs）：使用交叉熵 + Balanced Softmax + SBCL（10 epochs 后引入）
- **主训练阶段**：DaCC 样本选择 + SSL（MixMatch）+ SBCL + MIDL

推理时使用常规分类器和平衡分类器的预测平均值。采用 co-training 框架对抗确认偏差。

## 实验关键数据

### CIFAR-10（长尾 + 对称噪声，不平衡率 0.1）

| 噪声率 | SFA | TABASCO | **DaSC** |
|--------|-----|---------|----------|
| 0.4 | 86.81 | 85.53 | **89.04** |
| 0.6 | 82.89 | 84.83 | **87.12** |

### CIFAR-100（长尾 + 对称噪声，不平衡率 0.1）

| 噪声率 | SFA | TABASCO | **DaSC** |
|--------|-----|---------|----------|
| 0.4 | 56.70 | 56.52 | **61.85** |
| 0.6 | 47.71 | 45.98 | **54.40** |

### CIFAR-10（长尾 + 非对称噪声，不平衡率 0.1）

| 噪声率 | SFA | TABASCO | **DaSC** |
|--------|-----|---------|----------|
| 0.2 | 87.70 | 82.10 | **89.89** |
| 0.4 | 78.13 | 80.57 | **88.85** |

在所有配置下均大幅超越前 SOTA，尤其在高噪声率和非对称噪声场景下提升显著。

## 亮点
1. **DaCC 的跨类加权中心估计**：打破了传统方法只用类内样本的限制，利用温度缩放的预测概率作为权重，实现了对所有样本的有效利用，对尾部类中心估计尤为关键
2. **置信度感知的双路对比学习**：高置信度样本用监督对比（SBCL）利用标签信息实现均衡表示，低置信度样本用自监督对比（MIDL）避免噪声标签干扰——策略设计合理
3. **Mixup 作为硬负样本**：将 Mixup 生成的样本作为 memory bank 的负样本，同时增强了表示学习和数据多样性
4. **实验全面**：涵盖对称/非对称噪声、不同不平衡率、真实噪声数据集

## 局限与展望
1. **计算开销**：DaCC 需要对所有样本做前向传播来估计中心，且使用 co-training 框架（两个网络），计算成本较高
2. **超参数较多**：$\tau_T, \tau_c, \tau_s, \tau_m, \lambda_{SBCL}, \lambda_{MIDL}$ 等参数需要调优
3. **分类领域假设**：方法主要面向分类任务，向检测/分割等任务的迁移性未探讨
4. **DaCC 的阈值策略**：置信度阈值 $\tau$ 的增长策略（$\phi=1.005$）较为经验性
5. **仅在小规模数据集验证**：最大实验规模为 Red mini-ImageNet，大规模数据集上的效果未知

## 与相关工作的对比
- **vs RoLT/PCL**：仅用目标类内高置信样本估计中心；DaSC 用所有样本加权估计，尾部类中心更准确
- **vs SFA**：用高斯分布建模类中心不确定性；DaSC 直接用预测概率加权，更简洁
- **vs TABASCO**：用两种指标 + GMM 选择干净样本；DaSC 的 DaCC 中心质量更高，且增加了对比学习增强表示
- **vs DivideMix/UNICON**：仅处理噪声标签不考虑长尾；DaSC 通过 Balanced Softmax 和 SBCL 显式处理类别不均衡
- **vs BCL/PaCo**：仅处理长尾不考虑噪声；DaSC 通过 DaCC + GMM 选择干净样本

## 启发与关联
- DaCC 的跨类加权估计思想可推广到其他需要估计类别原型的场景（如 few-shot learning、open-set recognition）
- 对高/低置信度样本分别采用不同学习策略的思路，可应用于半监督学习和主动学习
- Mixup 作为硬负样本的方法具有通用性，可扩展到其他对比学习框架

## 评分
- 新颖性: 7/10（DaCC 的跨类加权中心估计和双路对比学习设计有新意，但各组件技术并非全新）
- 实验充分度: 8/10（多种噪声类型、多数据集、完整消融实验）
- 写作质量: 7/10（结构清晰，但公式较多，可读性一般）
- 价值: 7/10（在 NL-LT 这个特定问题上有明确贡献，但实用场景的验证还不够）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [\[ICLR 2026\] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](../../ICLR2026/social_computing/functional_embeddings_enable_aggregation_of_multi-area_seeg_data_for_robust_bci.md)
- [\[ICML 2025\] Learning Survival Distributions with the Asymmetric Laplace Distribution](../../ICML2025/social_computing/learning_survival_distributions_with_the_asymmetric_laplace_distribution.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](../../NeurIPS2025/social_computing/visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)

</div>

<!-- RELATED:END -->
