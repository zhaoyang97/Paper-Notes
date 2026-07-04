---
title: >-
  [论文解读] Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes
description: >-
  [ICML2025 Spotlight][图像生成][扩散模型] 通过对数概率密度的 Hessian 曲率（sharpness）建立扩散模型记忆化的几何分析框架，提出可在生成初始阶段检测记忆化的新指标，并设计无需重训练的 SAIL 初始噪声优化策略来缓解记忆化。 扩散模型（Diffusion Models）在图像生成领域取得…
tags:
  - "ICML2025 Spotlight"
  - "图像生成"
  - "扩散模型"
  - "记忆"
  - "sharpness"
  - "Hessian"
  - "score function"
  - "隐私"
---

# Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes

**会议**: ICML2025 Spotlight  
**arXiv**: [2412.04140](https://arxiv.org/abs/2412.04140)  
**代码**: [GitHub](https://github.com/Dongjae0324/sharpness_memorization_diffusion)  
**作者**: Dongjae Jeon, Dueun Kim, Albert No
**领域**: 图像生成  
**关键词**: 扩散模型, 记忆, sharpness, Hessian, score function, 隐私

## 一句话总结

通过对数概率密度的 Hessian 曲率（sharpness）建立扩散模型记忆化的几何分析框架，提出可在生成初始阶段检测记忆化的新指标，并设计无需重训练的 SAIL 初始噪声优化策略来缓解记忆化。

## 研究背景与动机

扩散模型（Diffusion Models）在图像生成领域取得了巨大成功，但存在严重的**记忆化（memorization）**问题——模型有时会直接复制训练样本而非生成新颖输出。这在模型使用敏感数据训练时引发严重的隐私风险，已有多起相关诉讼案例。

现有记忆化分析方法存在以下不足：

**LID（Local Intrinsic Dimensionality）方法**（Ross et al., 2024）：仅能在生成最终步（$t \approx 0$）检测，无法提前预警

**基于训练数据比对的方法**（Carlini et al., 2023）：需要访问训练集，计算开销大，不适合实时检测

**Wen et al. (2024) 的 score 差异指标**：虽然有效但缺乏理论解释

**Prompt 修改策略**：缓解记忆化的同时会损害生成质量

因此，本文的核心动机是：**能否从概率密度的几何特性出发，建立一个统一的理论框架来解释、检测并缓解记忆化？**

## 方法详解

### 核心思想：记忆化 = 概率景观中的尖锐区域

作者提出一个关键观察：记忆化样本对应于学习到的对数概率密度 $\log p(\mathbf{x})$ 中的**尖锐峰值（sharp peaks）**。这种尖锐性可以通过 Hessian 矩阵的特征值来量化：

$$H(\mathbf{x}_t) \coloneqq \nabla^2_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

- **大的负特征值** → 尖锐、孤立的概率峰 → 记忆化
- **小的/正的特征值** → 平滑、宽广的分布 → 良好泛化

与 LID 方法的关键区别在于：sharpness 分析可以在**所有时间步**进行，而非仅限于最终步。

### Score 范数作为 Sharpness 度量（Lemma 4.1）

直接计算高维 Hessian 特征值计算量巨大。作者证明了在高斯假设下，score 函数范数与 Hessian 迹的等价关系：

$$\mathbb{E}[\|s(\mathbf{x})\|^2] = -\text{tr}(H(\mathbf{x}))$$

其中 $H(\mathbf{x}) \equiv -\boldsymbol{\Sigma}^{-1}$。这意味着 **score 范数**可以作为 sharpness 的高效近似——无需进行昂贵的特征值分解。该关系在扩散过程的中高噪声阶段尤其准确，因为此时潜变量分布接近高斯。

### 为 Wen's Metric 提供理论基础（Lemma 4.2）

Wen et al. (2024) 提出的记忆化检测指标为条件与无条件 score 的差异范数：

$$\|s_\theta^\Delta(\mathbf{x}_t)\| \coloneqq \|s_\theta(\mathbf{x}_t, c) - s_\theta(\mathbf{x}_t)\|$$

作者证明了该指标本质上度量的是条件与无条件 Hessian 的**特征值差异**：

$$\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x}|c)}[\|s(\mathbf{x},c) - s(\mathbf{x})\|^2] = \sum_{i=1}^d \frac{(\lambda_i - \lambda_{i,c})^2}{\lambda_{i,c}}$$

（在协方差矩阵可交换且均值相同的条件下）。这为 Wen's metric 提供了 sharpness 层面的理论解释。

### Hessian 放大的新 Sharpness 指标（Section 4.4）

Wen's metric 在生成初始阶段（$t = T-1$）灵敏度不足。作者提出通过 Hessian-score 乘积来放大高曲率方向：

$$\|H_\theta(\mathbf{x}_t, c) \cdot s_\theta(\mathbf{x}_t, c)\|^2 \quad \text{近似} \quad -\text{tr}(H_\theta(\mathbf{x}_t, c)^3)$$

该三阶矩统计量对大的负特征值更加敏感（$\lambda^3$ 放大效应），从而在初始采样步即可有效区分记忆化与非记忆化样本。

### SAIL：基于 Sharpness 感知的初始噪声优化

**Sharpness-Aware Initialization for Latent Diffusion (SAIL)** 是本文提出的缓解策略。核心思路是在推理时优化初始噪声 $\mathbf{z}_T$，使其避开高 sharpness 区域：

- 用提出的 sharpness 指标作为正则项
- 在初始噪声空间中搜索具有低 sharpness 的初始化点
- 引导扩散过程朝向平滑概率区域采样

**关键优势**：
- **无需重训练**：仅调整推理时的初始噪声
- **保留 prompt 不变**：不像 prompt 修改策略那样改变用户输入
- **模型无关**：适用于各种扩散模型架构

## 实验关键数据

### 实验设置

- **数据集**：2D toy dataset、MNIST、Stable Diffusion v1.4 / v2.0
- **记忆化类别**：Exact Memorization (EM)、Partial Memorization (PM)、Non-memorized
- **MNIST 实验**：通过重复单张 "9" 图片诱导记忆化，以 "3" 类作为通用对照
- **Hessian 特征值**：通过 Arnoldi 迭代法近似计算

### 记忆化检测性能对比（AUC / TPR@1%FPR）

| 方法 | Steps | SD v1.4 AUC | SD v1.4 TPR@1%FPR | SD v2.0 AUC | SD v2.0 TPR@1%FPR |
|------|-------|-------------|-------------------|-------------|-------------------|
| Tiled $\ell_2$ (Carlini) | 50 | 0.908–0.94 | 0.088–0.232 | 0.792–0.907 | 0.114 |
| LE (Ren et al.) | 1 | 0.832–0.846 | 0.116–0.124 | 0.848–0.853 | 0 |
| AE (Ren et al.) | 50 | 0.598–0.628 | 0 | 0.809–0.82 | 0 |
| Score norm (本文) | 1 | **有竞争力** | — | — | — |
| Hessian-score (本文) | 1 | **初始步最优** | — | — | — |

### 核心实验发现

1. **特征值分布差异**：记忆化样本在初始和最终步均表现出显著更多、更大的负特征值；非记忆化样本特征值接近零或为正
2. **早期检测**：Hessian-score 乘积指标在 $t = T-1$（第一步）即可检测记忆化，远优于 LID 的最终步限制
3. **条件/无条件 Hessian 差异**：记忆化样本的条件与无条件 Hessian 特征值差距显著，非记忆化样本差异接近零
4. **SAIL 缓解效果**：SAIL 在降低记忆化的同时保持了生成质量（FID / CLIP score 无显著下降）

## 亮点与洞察

1. **理论优雅**：将记忆化问题归结为概率景观的曲率分析，连接了 score function、Hessian、LID 等看似不同的概念
2. **实用价值**：仅需 1 步推理即可检测记忆化，计算开销极低
3. **Wen's metric 的理论基础**：为已有的启发式方法提供了严格的数学解释
4. **三阶矩放大技巧**：Hessian-score 乘积利用 $\lambda^3$ 放大效应，巧妙解决了初始步灵敏度不足的问题
5. **SAIL 设计理念**：不改模型、不改 prompt，仅调初始噪声——这种最小侵入式的缓解策略在实际部署中更可行

## 局限与展望

1. **高斯假设的局限**：核心理论依赖高斯假设（Lemma 4.1, 4.2），虽然在扩散过程中后期分布偏离高斯时 score norm 仍经验性有效，但理论保证减弱
2. **SAIL 计算开销**：初始噪声优化需要额外的前向传播来评估 sharpness，在大规模部署时的效率需要进一步评估
3. **记忆化定义的范围**：主要关注 text-conditional 场景，对 unconditional 或其他条件生成的适用性未充分讨论
4. **缺乏大规模实验**：主要在 SD v1.4/v2.0 上验证，未涉及更新的 SDXL 或 Flux 等模型
5. **Hessian 近似精度**：Arnoldi 迭代仅给出部分特征值，对于 16,384 维潜空间的近似质量有待更深入分析
6. **SAIL 的生成多样性影响**：虽然声称保持质量，但系统性地避开高 sharpness 区域是否会减少生成多样性值得研究

## 相关工作与启发

- **LID 方法**（Ross et al., 2024; Kamkari et al., 2024）：sharpness 是 LID 的泛化，可在任意时间步检测
- **Wen et al. (2024) 的 score 差异指标**：本文为其提供了理论基础
- **Manifold 学习猜想**（Fefferman et al., 2016）：精确记忆 = 零维流形上的数据点
- **SAM 优化**（Foret et al., 2021）：sharpness-aware 思想从训练迁移到推理
- **差分隐私训练**：SAIL 提供了一种互补的、无需重训练的隐私保护方案

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Hessian sharpness 视角统一了多个已有概念，Hessian-score 放大技巧和 SAIL 均有创新
- **实验充分度**: ⭐⭐⭐ — 从 toy 到 SD 的层级验证合理，但缺少更新模型和大规模实验
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，从观察到理论到应用的逻辑链条完整
- **价值**: ⭐⭐⭐⭐ — 为记忆化研究提供了统一的几何框架，SAIL 有实际部署潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Localizing and Mitigating Memorization in Image Autoregressive Models](localizing_and_mitigating_memorization_in_image_autoregressive_models.md)
- [\[ICLR 2026\] Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](../../ICLR2026/image_generation/detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[CVPR 2025\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](../../CVPR2025/image_generation/mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](../../ICCV2025/image_generation/understanding_flatness_in_generative_models_its_role_and_benefits.md)

</div>

<!-- RELATED:END -->
