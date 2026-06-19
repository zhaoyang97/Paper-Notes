---
title: >-
  [论文解读] Data-free Universal Adversarial Perturbation with Pseudo-Semantic Prior
description: >-
  [CVPR 2025][AI安全][UAP] 提出 PSP-UAP，一种无需训练数据的通用对抗扰动生成方法，通过从 UAP 自身提取伪语义先验、输入变换增强和样本重加权策略，在白盒平均 89.95% 愚弄率、黑盒也大幅超越现有方法，且无需任何训练数据。 领域现状：通用对抗扰动 (UAP) 是一种可以攻击任意输入图像的 ima…
tags:
  - "CVPR 2025"
  - "AI安全"
  - "UAP"
  - "adversarial perturbation"
  - "data-free"
  - "pseudo-semantic"
  - "transferability"
---

# Data-free Universal Adversarial Perturbation with Pseudo-Semantic Prior

**会议**: CVPR 2025  
**arXiv**: [2502.21048](https://arxiv.org/abs/2502.21048)  
**代码**: [https://github.com/ChnanChan/PSP-UAP](https://github.com/ChnanChan/PSP-UAP)  
**领域**: AI安全  
**关键词**: UAP, adversarial perturbation, data-free, pseudo-semantic, transferability

## 一句话总结
提出 PSP-UAP，一种无需训练数据的通用对抗扰动生成方法，通过从 UAP 自身提取伪语义先验、输入变换增强和样本重加权策略，在白盒平均 89.95% 愚弄率、黑盒也大幅超越现有方法，且无需任何训练数据。

## 研究背景与动机

**领域现状**：通用对抗扰动 (UAP) 是一种可以攻击任意输入图像的 image-agnostic 扰动。传统 UAP 方法需要大量训练数据来优化扰动，但数据依赖限制了实际攻击场景的适用性。

**现有痛点**：(1) 有数据的 UAP（如 UAP、SGA-UAP）需要完整训练集，隐私和版权限制可能无法获得。(2) 无数据的 UAP（如 AT-UAP-U、TRM-UAP）通常从随机噪声出发优化，缺乏语义信息导致攻击效果有限。(3) 黑盒迁移性差。

**核心矛盾**：无数据设定下缺乏真实图像的语义信息来引导 UAP 优化，但 UAP 本身在训练过程中会逐渐积累来自目标模型的语义信息。

**本文目标** 如何在完全无数据的条件下生成高效的通用对抗扰动？

**切入角度**：关键观察——UAP 在优化过程中自身就包含了丰富的语义信息（不同区域对应不同的语义模式），可以将 UAP 裁剪、缩放后作为"伪语义样本"来引导自身的进一步优化。

**核心 idea**：从 UAP 自身提取伪语义先验作为训练样本，结合输入变换（旋转/缩放/打乱）和 KL 散度引导的样本重加权，无需任何训练数据即可生成高效 UAP。

## 方法详解

### 整体框架
从随机初始化的 UAP $\delta$ 出发，迭代优化。每步从当前 UAP 中随机裁剪+缩放生成 N 个伪语义样本，对每个样本应用输入变换增强鲁棒性，通过样本重加权聚焦困难样本，最大化模型的特征激活来更新 UAP。

### 关键设计

1. **Pseudo-Semantic Prior (PSP)**

    - 功能：从 UAP 自身提取伪语义样本作为训练数据
    - 核心思路：$p_x = \{z + \delta_t | z \in p_z\}$，其中 $z$ 为随机噪声。通过随机裁剪和缩放从 UAP 生成 $x_n = C(x; N)$，N=10 个样本
    - 设计动机：UAP 包含目标模型的语义结构——不同区域触发不同特征图的激活，裁剪后的局部区域各自携带不同的语义片段

2. **Input Transformations**

    - 功能：增强 UAP 的鲁棒性和迁移性
    - 核心思路：三种变换的组合——旋转 $\alpha \in [-6°, 6°]$、缩放 $\beta \in [0.8, 4.0]$、$2\times2$ 块随机打乱
    - 设计动机：变换增强了优化过程中的输入多样性，使 UAP 不依赖于特定的空间配置

3. **Sample Reweighting**

    - 功能：对困难样本（模型反应不明显的样本）给予更大权重
    - 核心思路：$w_n = D_{KL}(P(x_n) || Q(x_n))^{-1}$，$P$ 为干净预测，$Q$ 为加扰动后预测。KL 散度小的样本更难攻击，权重更大
    - 设计动机：不同的伪语义样本对模型的影响不同，优先优化那些攻击效果差的样本可以提升整体性能

### 损失函数 / 训练策略
- 总损失：$L = -\mathbb{E}[\sum_{n=1}^N \sum_{i=1}^l \log(w_n ||\mathcal{A}_i^f(T(x_n + \delta_t))||_2)]$
- 扰动约束：$\epsilon = 10/255$（$\ell_\infty$ 范数）
- 最大迭代：T = 10,000
- 饱和阈值：0.001%

## 实验关键数据

### 主实验

**白盒攻击愚弄率 (%)：**

| 模型 | AT-UAP-U | TRM-UAP | PSP-UAP |
|------|----------|---------|---------|
| VGG16 | 94.50 | 94.30 | **96.26** |
| VGG19 | 92.85 | 91.35 | **94.65** |
| ResNet152 | 73.15 | 67.46 | **85.65** |
| GoogleNet | 82.60 | 85.32 | 81.43 |
| 平均 | 87.95 | 86.39 | **89.95** |

**扩展模型黑盒平均攻击率：**

| 模型 | TRM-UAP | PSP-UAP |
|------|---------|---------|
| ResNet50 | 56.57% | **64.13%** |
| DenseNet121 | 42.91% | **59.95%** |
| MobileNet-v3 | 45.76% | **61.42%** |
| Inception-v3 | 59.96% | **62.67%** |

### 消融实验

| 组件 | 效果 |
|------|------|
| PSP alone | 超越随机先验 |
| + Sample Reweighting | 显著提升 |
| + Input Transformation | 进一步改善 |
| Full method | 最佳性能 |

**vs 有数据方法（VGG16 黑盒平均）：** PSP-UAP 75.65% vs SGA-UAP 69.27%

### 关键发现
- **无数据方法超越有数据方法**：PSP-UAP (无数据) 在黑盒设定下超过 SGA-UAP (有数据)，说明伪语义先验比真实数据在某些场景下更有效
- ResNet152 上提升最大（67.46%→85.65%），可能因为 ResNet 的残差结构对语义扰动更敏感
- 温度参数 $\tau$ 对不同模型最优值不同（1.0-10.0），需要针对性调整
- 三种输入变换的组合比单独使用效果好，但边际收益递减

## 亮点与洞察
- **UAP 自身即数据源**：从扰动自身提取训练信号的自举式思路非常巧妙，避免了数据依赖的根本问题
- **超越有数据方法的无数据方法**：打破了"有数据一定更好"的直觉
- **KL 散度引导的重加权**：将注意力聚焦到困难样本上的策略在其他优化问题中也可借鉴
- 伪语义先验的思路可迁移到其他需要无数据优化的场景（如无数据蒸馏）

## 局限与展望
- GoogleNet 上表现不如 TRM-UAP，可能因为 Inception 模块的多尺度结构对伪语义样本不够敏感
- 未在 Vision Transformer 等现代架构上验证
- 扰动约束 $\epsilon=10/255$ 相对较大，更严格约束下的表现未知
- 迭代 10,000 步的计算开销需要评估

## 相关工作与启发
- **vs AT-UAP-U**: 同为无数据方法，但 AT-UAP-U 从均匀噪声出发缺乏语义引导
- **vs SGA-UAP**: 有数据方法在白盒上略强，但黑盒迁移性不如 PSP-UAP
- **vs TRM-UAP**: 使用文本引导的语义，但受限于 CLIP 的语义空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 从 UAP 自身提取语义先验的自举思路很新颖
- 实验充分度: ⭐⭐⭐⭐ 白盒+黑盒+扩展模型+有数据对比
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰
- 价值: ⭐⭐⭐⭐ 无数据 UAP 在实际攻击场景中更有意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](../../ICCV2025/ai_safety/semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[CVPR 2025\] DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging](deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md)
- [\[CVPR 2025\] Towards Source-Free Machine Unlearning](towards_source-free_machine_unlearning.md)
- [\[ICML 2026\] Semantic Router: On the Feasibility of Hijacking MLLMs via a Single Adversarial Perturbation](../../ICML2026/ai_safety/semantic_router_on_the_feasibility_of_hijacking_mllms_via_a_single_adversarial_p.md)
- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
