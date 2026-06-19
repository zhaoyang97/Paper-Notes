---
title: >-
  [论文解读] Unsupervised Learning for Class Distribution Mismatch (UCDM)
description: >-
  [ICML2025][图像生成][类别分布不匹配] 提出 UCDM，利用扩散模型从无标注数据中合成正负样本对来训练分类器，在不依赖标注数据的情况下解决训练集与目标任务之间的类别分布不匹配（CDM）问题，在 closed-set 和 open-set 任务上均大幅超越现有半监督方法。 类别分布不匹配（CDM）：是指训练数据的类…
tags:
  - "ICML2025"
  - "图像生成"
  - "类别分布不匹配"
  - "无监督学习"
  - "扩散模型"
  - "伪标签"
  - "开集识别"
---

# Unsupervised Learning for Class Distribution Mismatch (UCDM)

**会议**: ICML2025  
**arXiv**: [2505.06948](https://arxiv.org/abs/2505.06948)  
**代码**: 待确认  
**领域**: 分布偏移  
**关键词**: 类别分布不匹配, 无监督学习, 扩散模型, 伪标签, 开集识别

## 一句话总结

提出 UCDM，利用扩散模型从无标注数据中合成正负样本对来训练分类器，在不依赖标注数据的情况下解决训练集与目标任务之间的类别分布不匹配（CDM）问题，在 closed-set 和 open-set 任务上均大幅超越现有半监督方法。

## 研究背景与动机

**类别分布不匹配（CDM）** 是指训练数据的类别分布与目标任务需求不一致的实际问题。现有方法主要依赖半监督学习（SSL），需要标注数据来定义「已知类」，并将标注数据中未出现的类别归为「未知类」。这些方法存在两个核心问题：

**对标注数据的强依赖**：需要人工标注，在无标签场景下不可用，且标注成本高昂

**有限标注导致性能瓶颈**：基于 one-vs-all 分类器的 SCDM 方法（如 OpenMatch）在已知类上表现尚可，但在将未知类/新类归为统一的「other」类时表现极差

作者提出的核心问题：能否在完全无标注的情况下，仅给定已知类的类名，就能训练出同时处理 closed-set 和 open-set 任务的分类器？

## 方法详解

### 整体框架

UCDM 框架包含三个核心组件：（1）正样本生成管线；（2）负样本生成管线；（3）基于置信度的伪标签机制。

### 1. 正样本生成（Positive Instance Generation）

利用 text-to-image 扩散模型生成属于已知类的正样本，需满足三个性质：

- **无域偏移**：从训练集中随机抽取种子样本，通过前向扩散添加噪声（而非随机噪声向量），保留种子样本信息
- **多样性**：在反向过程中设置 $\sigma_t = 1$，在每步引入随机噪声
- **类别明确**：通过类别 prompt $\mathcal{C}_y$ = "A photo of a [CLASS]" 引导条件生成

前向扩散过程：

$$x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

### 2. 负样本生成（Negative Instance Generation）

**核心创新**：通过条件 DDIM 反演（Conditional DDIM Inversion）从图像中**擦除**指定语义类别。

**定理 3.1**（条件 DDIM 反演）表明，条件反演过程使噪声向量逐步远离类别 $y$ 的语义方向：

$$x_t = \sqrt{\alpha_t} \, x_0 - \sum_{i=0}^{t-1} \left[ \nabla_{x_i} \log p_\theta(x_i)^{s_i} + \nabla_{x_i} \log p_\theta(y | x_i)^{s_i} \right] + \text{残差项}$$

其中 $-\nabla_{x_i} \log p_\theta(y | x_i)$ 表示降低样本属于类别 $y$ 概率的梯度方向。

条件反演的实际计算公式：

$$x_t = \sqrt{\frac{\alpha_t}{\alpha_{t-1}}} \, x_{t-1} + \sqrt{\alpha_t} \, \psi(\alpha_t, \alpha_{t-1}, 0) \, \epsilon_\theta(x_{t-1}, t, \mathcal{C}_y)$$

**定理 3.2**（无条件 DDIM 反向）表明，用无条件 DPM 反向生成的图像 $\tilde{x}_0$ 保留了原图 $x_0$ 的视觉特征，仅近似去除了类别语义：

$$\tilde{x}_0 \approx x_0 - \frac{1}{\sqrt{\alpha_t}} \sum_{i=0}^{t-1} \nabla_{x_i} \log p_\theta(y | x_i)^{s_i}$$

### 3. 基于置信度的伪标签机制

结合两种置信度视角为真实图像分配伪标签：

**Other-probability-driven**：通过 $K$ 个二分类器的输出计算属于「other」类的概率：

$$p(y \in \mathcal{Y}_{\text{other}} | x) = \prod_{j=1}^{K} [1 - p(j | x)]$$

**Known-probability-driven**：结合闭集分类器 $\hat{p}(j|x)$ 和开集分类器 $p(j|x)$ 的预测：

$$\tilde{q}_j = \hat{p}(j|x) \times p(j|x), \quad j = 1, \dots, K$$

当两种视角的最高置信类别一致且得分超过阈值 $\delta$ 时，分配伪标签。

### 4. 总训练损失

$$\mathcal{L} = \mathcal{L}_{\text{gen}}^{(\mathcal{D}_P, \mathcal{D}_N)} + \mathcal{L}_{\text{gen}}^{(\mathcal{D}_{\text{known}}, \mathcal{D}_N')} + \mathcal{L}_{\text{gen}}^{(\mathcal{D}_P', \mathcal{D}_{\text{unknown}})}$$

三项分别对应：生成数据训练、已知类真实数据训练、未知类真实数据训练。

## 实验关键数据

### Closed-set 任务（已知类分类准确率）

| 方法 | CIFAR-10 (60%) | CIFAR-100 (60%) | Tiny-ImageNet (60%) |
|------|:-:|:-:|:-:|
| DS³L | 66.6 | 23.4 | 26.3 |
| UASD | 79.3 | 22.8 | 5.3 |
| CCSSL | 95.7 | 45.6 | 25.8 |
| T2T | - | 50.6 | 41.7 |
| OpenMatch | 68.5 | 10.3 | 10.9 |
| IOMatch | 89.8 | 31.1 | 32.8 |
| **UCDM (Ours)** | **95.6** | **50.9** | **32.3** |

### Open-set 任务（60% mismatch，Tiny-ImageNet）

| 方法 | Known Acc | Unknown Acc | New Acc | Balance Score |
|------|:-:|:-:|:-:|:-:|
| OpenMatch | 10.8 | 3.5 | 5.9 | 3.0 |
| IOMatch | 0.0 | 100.0 | 100.0 | 8.9 |
| **UCDM (Ours)** | **15.8** | **94.9** | **95.4** | **22.9** |

### 核心数字

- Tiny-ImageNet 60% mismatch 下，UCDM（无标注）超过 OpenMatch（每类 40 标注）：已知类 +5.0%、未知类 +91.4%、新类 +89.5%
- CIFAR-10 上 open-set 任务，UCDM 在所有 mismatch 比例下的 balance score 均 > 91，远超所有 baseline

## 亮点与洞察

1. **无监督设定的首次探索**：首次在 CDM 问题上提出纯无监督方案，仅需已知类名即可训练，突破了 SSL 方法对标注数据的依赖
2. **扩散模型的创新应用**：利用条件 DDIM 反演实现语义擦除，理论证明（定理 3.1/3.2）该操作确实沿着降低类别似然的方向移动潜变量
3. **双视角置信度标签机制**：巧妙融合 other-probability-driven 和 known-probability-driven 两种置信度，比单一视角更稳健
4. **在 open-set 任务上优势巨大**：现有 SSL 方法在 unknown/new 类上几乎完全失效（大多为 0%），而 UCDM 在保持已知类性能的同时能准确识别未知类

## 局限与展望

1. **已知类名的先验假设**：虽然不需要标注，但仍需预先给定已知类的类名列表，在类名未知的场景下不适用
2. **依赖预训练扩散模型**：方法的有效性依赖于高质量的 text-to-image 扩散模型，对于扩散模型未见过的细粒度类别可能效果有限
3. **Tiny-ImageNet 上 known-class 准确率偏低**：open-set 任务中 known-class 准确率仅 15-22%，说明在细粒度分类场景下生成的正样本质量有限
4. **计算开销**：每个训练样本都需要扩散模型的前向/反向过程来生成正负样本对，推理成本显著高于普通 SSL 方法
5. **仅在小规模数据集验证**：未在 ImageNet 全集等大规模数据集上实验

## 相关工作与启发

- **SSL under CDM**：UASD、CCSSL、T2T（closed-set）; OpenMatch、IOMatch（open-set）— 均需标注数据
- **扩散生成增强**：DPT、DWD — 需重训扩散模型且假设分布匹配
- **Score-based 生成模型**：DDIM 反演理论为语义擦除提供了理论基础

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在 CDM 场景提出纯无监督方案，条件反演擦除语义的思路有理论支撑
- 实验充分度: ⭐⭐⭐ — 三个标准数据集、多种 mismatch 比例、消融实验完备，但缺少大规模数据集验证
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，定理证明严谨，图示说明直观
- 价值: ⭐⭐⭐⭐ — 打开了 CDM 无监督学习的新方向，对开集识别和分布偏移领域有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Autoencoder-Based Hybrid Replay for Class-Incremental Learning](autoencoder-based_hybrid_replay_for_class-incremental_learning.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICML 2025\] Out-of-Distribution Detection Methods Answer the Wrong Questions](out-of-distribution_detection_methods_answer_the_wrong_questions.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](../../CVPR2025/image_generation/generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[ICCV 2025\] Pretrained Reversible Generation as Unsupervised Visual Representation Learning](../../ICCV2025/image_generation/pretrained_reversible_generation_as_unsupervised_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
