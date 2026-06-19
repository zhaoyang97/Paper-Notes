---
title: >-
  [论文解读] Compositional Scene Understanding through Inverse Generative Modeling
description: >-
  [ICML2025][图像生成][compositional generation] 本文提出逆生成建模（IGM）框架，将场景理解任务转化为在组合式生成模型中寻找最优条件参数的反演问题，通过将多个小型扩散模型组合来表示复杂场景，实现了强分布外泛化能力，并可直接利用预训练文生图模型进行零样本多目标感知。
tags:
  - "ICML2025"
  - "图像生成"
  - "compositional generation"
  - "inverse generative modeling"
  - "扩散模型"
  - "场景理解"
  - "object discovery"
---

# Compositional Scene Understanding through Inverse Generative Modeling

**会议**: ICML2025  
**arXiv**: [2505.21780](https://arxiv.org/abs/2505.21780)  
**代码**: [energy-based-model/compositional-inference](https://energy-based-model.github.io/compositional-inference)  
**领域**: 图像生成  
**关键词**: compositional generation, inverse generative modeling, diffusion models, scene understanding, object discovery  

## 一句话总结

本文提出逆生成建模（IGM）框架，将场景理解任务转化为在组合式生成模型中寻找最优条件参数的反演问题，通过将多个小型扩散模型组合来表示复杂场景，实现了强分布外泛化能力，并可直接利用预训练文生图模型进行零样本多目标感知。

## 研究背景与动机

传统的场景理解任务由判别式模型主导，这些模型学习从输入图像到视觉属性的直接映射。然而，大量研究表明判别式模型在面对测试分布偏移时表现显著下降，即使是微小的分布变化也会导致性能大幅衰减。

生成式模型长期以来被认为具有更好的泛化潜力，但直到最近扩散模型的出现，生成式模型才在视觉推理任务上展现出有竞争力的结果。然而现有的生成式推理方法主要集中在单标签分类任务上，如何在比训练集复杂得多的场景上执行更广泛的场景理解任务（如目标发现、多目标分类）仍然是一个开放问题。

本文的核心动机来源于费曼的名言"What I cannot create, I do not understand"——通过生成来理解。作者认为，如果一个生成模型能够很好地重建场景，那么生成模型的条件参数就包含了对场景的理解。更关键的是，通过组合式建模，可以用训练时见过的简单场景片段来理解测试时遇到的更复杂场景。

## 方法详解

### 整体框架

本文将场景理解形式化为**逆生成建模**问题。给定一张图像 $\boldsymbol{x}$，目标是推断一组视觉概念 $\{c^1, c^2, \cdots, c^K\}$ 来描述该图像。整体框架分为两个阶段：

1. **组合式生成建模**：构建由多个小型生成模型组合而成的生成模型
2. **逆向推理**：通过优化找到最佳拟合给定图像的条件参数

### 关键设计1：组合式生成模型

为了建模条件概率分布 $p(\boldsymbol{x}|c^1, c^2, \ldots, c^K)$，作者采用乘积分解近似：

$$p(\boldsymbol{x}|c^1, \ldots, c^K) \propto \prod_{k=1}^{K} p(\boldsymbol{x}|c^k)$$

每个 $p(\boldsymbol{x}|c^k)$ 用能量模型（EBM）参数化为 $e^{-E_\theta(\boldsymbol{x}|c^k)}$，乘积分布变为能量之和：

$$p(\boldsymbol{x}|c^1, \ldots, c^K) \propto e^{-\sum_{k=1}^{K} E_\theta(\boldsymbol{x}|c^k)}$$

利用扩散模型的去噪函数 $\epsilon_\theta(\boldsymbol{x}^t, t | c^k)$ 近似表示 $\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x}|c^k)$，组合去噪函数为：

$$\epsilon_\theta^{\text{comb}}(\boldsymbol{x}^t, t) = \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k)$$

### 关键设计2：联合训练组合分数函数

与先前工作在测试时才组合独立训练的去噪函数不同，本文提出直接训练组合分数函数：

$$\mathcal{L}_\theta = \mathbb{E}_{\boldsymbol{x}, \epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k) \right\|^2$$

这种联合训练使得各去噪函数在组合时表现更准确，同时在测试时仍可添加更多项来构建更复杂场景。

### 关键设计3：逆向推理

场景理解被形式化为最大化给定图像的对数似然：

$$\hat{c}^1, \ldots, \hat{c}^K = \arg\min_{c^1, \ldots, c^K} \mathbb{E}_{\epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k) \right\|^2$$

- **离散概念推理**：枚举所有可能的概念配置，选择去噪误差最小的
- **连续概念推理**：采用随机梯度下降，结合多重随机初始化策略避免局部最优

### 关键设计4：概念数量推断

通过在不同 $K$ 值下求解优化问题，选择使似然最大化（去噪误差最小）的 $\hat{K}$：

$$\hat{K} = \arg\min_{K \in [K_{min}, K_{max}]} \left\{ \min_{c^1, \ldots, c^K} \mathbb{E}_{\epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k) \right\|^2 \right\}$$

### 损失函数

训练阶段使用标准的去噪扩散目标，但对组合去噪函数进行端到端训练：

$$\mathcal{L}_\theta = \mathbb{E}_{\boldsymbol{x}, \epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t, c^k) \right\|^2$$

推理阶段优化概念参数时，使用随机梯度下降（SGD），每步仅需单次采样 $\epsilon_n, t_n$，将采样复杂度从 $N$ 降至 1。

## 实验关键数据

### 主实验1：目标发现（CLEVR数据集）

训练集包含3-5个物体的图像，测试分为分布内（3-5）和分布外（6-8）：

| 模型 | 分布内感知率↑ | 分布内估计误差↓ | 分布外感知率↑ | 分布外估计误差↓ |
|------|-------|---------|-------|---------|
| ResNet-50 | 5.3% | 19.4e-2 | 2.9% | 19.7e-2 |
| SlotAttn | 80.4% | 8.7e-4 | 53.3% | 1.3e-3 |
| DINOSAUR | 82.5% | 8.4e-4 | 59.0% | 1.2e-3 |
| GC | 82.2% | 6.0e-4 | 58.7% | 1.2e-3 |
| **IGM (Ours)** | **94.7%** | **1.4e-4** | **85.3%** | **3.5e-4** |

### 主实验2：面部属性预测（CelebA数据集）

训练集仅包含女性面部，分布外测试集为男性面部：

| 模型 | 分布内准确率 | 分布外准确率 |
|------|---------|---------|
| ResNet-50 | 79.6% | 62.2% |
| GC | 79.1% | 61.7% |
| **IGM (Ours)** | **80.8%** | **65.6%** |

### 主实验3：零样本多目标感知

使用预训练Stable Diffusion，无需额外训练：

| 模型 | 准确率↑ |
|------|------|
| Diffusion Classifier | 70.4% |
| DC Variant | 73.2% |
| **IGM (Ours)** | **87.3%** |

### 消融实验

| 变体 | 分布内感知率 | 分布外感知率 |
|------|---------|---------|
| IGM w/o 多重初始化 | 72.8% | 68.0% |
| IGM w/ 多重初始化 | 94.7% | 85.3% |

多重随机初始化策略带来了约22%的分布内提升和17%的分布外提升。

### 关键发现

1. 组合式建模在分布外泛化上优势显著：分布外目标发现任务上比最佳基线高出26.6%
2. 概念数量推断有效：ground truth数量一致地产生最低去噪误差
3. 零样本场景：直接利用预训练Stable Diffusion即可实现87.3%的多目标感知准确率
4. 多重初始化是连续概念推理成功的关键

## 亮点与洞察

1. **优雅的问题转化**：将场景理解这一判别性任务转化为生成模型的逆问题，利用了生成模型固有的组合泛化能力
2. **组合即泛化**：通过将复杂场景分解为简单概念的组合，自然实现了向更复杂场景的泛化，无需见过复杂场景
3. **统一框架**：同一框架可处理离散概念（分类）、连续概念（坐标）和预训练模型（零样本），展现了极强的灵活性
4. **去噪误差作为似然代理**：巧妙利用扩散模型的去噪误差近似对数似然，无需显式计算配分函数
5. **最小训练开销**：零样本方案直接复用预训练模型，无需额外训练

## 局限性

1. **推理效率低**：离散概念推理需要枚举所有可能配置，当概念数量大时计算量呈指数增长 $O(M^K)$
2. **概念独立性假设**：假设场景中各概念独立，忽略了物体间的交互关系（如遮挡、空间关系），在真实场景中可能产生误差
3. **连续推理对初始化敏感**：虽然多重初始化缓解了局部最优问题，但增加了计算成本
4. **评估规模有限**：零样本多目标感知仅在3类动物的小数据集上验证，缺乏大规模真实场景评估
5. **乘积近似偏差**：将联合分布近似为边际分布的乘积是有偏估计，可能在强相关场景下失效

## 相关工作

- **生成式分类器**：Li et al. (2024) 利用扩散模型的去噪误差进行单标签分类，本文将其扩展到多概念组合推理
- **组合式生成模型**：Du & Kaelbling (2024) 提出组合生成建模范式，本文在此基础上将组合用于场景理解而非生成
- **Slot Attention**：Locatello et al. (2020) 通过注意力机制发现物体，但泛化能力有限
- **DINOSAUR**：Seitzer et al. (2022) 结合自监督特征进行目标发现

## 评分

⭐⭐⭐⭐ (4/5)

创新性强，方法优雅，组合泛化的动机令人信服。分布外目标发现的提升显著。但推理效率和评估规模是明显短板，零样本实验设置过于简单。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)
- [\[ICML 2025\] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional](action-minimization_meets_generative_modeling_efficient_transition_path_sampling.md)
- [\[ICML 2025\] Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[ICML 2026\] Compositional Generative Modeling from Decentralized Data](../../ICML2026/image_generation/compositional_generative_modeling_from_decentralized_data.md)

</div>

<!-- RELATED:END -->
