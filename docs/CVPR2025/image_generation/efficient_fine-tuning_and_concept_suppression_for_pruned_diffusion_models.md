---
title: >-
  [论文解读] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models
description: >-
  [CVPR 2025][图像生成][扩散模型剪枝] 提出一种双层优化框架，将剪枝扩散模型的微调恢复（下层：蒸馏+扩散损失最小化）和不良概念遗忘（上层：引导模型远离目标概念）统一为单一阶段优化，解决了"先微调再遗忘"两阶段方法中微调最优点不等于遗忘最优初始化的循环依赖问题，在风格去除上 CSD 指标降低 27%。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型剪枝"
  - "知识蒸馏"
  - "概念遗忘"
  - "双层优化"
  - "安全部署"
---

# Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.15341](https://arxiv.org/abs/2412.15341)  
**代码**: [GitHub](https://github.com/rezashkv/bilevel-pruning)  
**领域**: 图像生成 / 模型压缩  
**关键词**: 扩散模型剪枝, 知识蒸馏, 概念遗忘, 双层优化, 安全部署

## 一句话总结

提出一种双层优化框架，将剪枝扩散模型的微调恢复（下层：蒸馏+扩散损失最小化）和不良概念遗忘（上层：引导模型远离目标概念）统一为单一阶段优化，解决了"先微调再遗忘"两阶段方法中微调最优点不等于遗忘最优初始化的循环依赖问题，在风格去除上 CSD 指标降低 27%。

## 研究背景与动机

**领域现状**：扩散模型（如 Stable Diffusion）生成质量持续提升但参数量庞大，在移动端等资源受限场景部署困难。模型剪枝+知识蒸馏是主流压缩方案：先按重要性去除部分参数，再用蒸馏从基础模型恢复剪枝模型的生成能力。

**现有痛点**：蒸馏在加速收敛的同时也会将基础模型的不良属性（如版权风格、NSFW内容）传播到剪枝模型中——即使微调数据集中完全不包含这些内容，蒸馏仍会把教师模型的这些能力"教"给学生模型。一个朴素的修复方案是"先蒸馏微调，再概念遗忘"（两阶段方法），但这存在根本性的循环依赖问题。

**核心矛盾**：微调得到的最优参数 $\hat{\theta}$ 不一定是概念遗忘的最优初始化点。从 $\hat{\theta}$ 出发做遗忘得到 $\theta'$，$\theta'$ 可能偏离微调的最优流形，导致生成质量下降。反之，为遗忘优化的参数也可能影响微调质量。两个目标相互耦合，顺序优化无法达到联合最优。

**本文目标** 如何在剪枝模型的微调过程中同时实现生成能力恢复和不良概念去除？

**切入角度**：将问题形式化为双层优化——下层做蒸馏微调以恢复生成能力，上层做概念遗忘以去除不良内容，两者通过参数共享和梯度交互实现联合优化。

**核心 idea**：用双层优化将剪枝微调与概念遗忘从"先后两阶段"变为"内外层交替"，下层蒸馏恢复生成质量提供约束，上层遗忘在满足约束下最大化概念去除效果。

## 方法详解

### 整体框架

给定剪枝后的扩散模型 $\epsilon_{\theta_{pruned}}$，本方法将微调和概念遗忘统一为双层优化问题：$\min_{\theta} \mathcal{L}^{CU}(\theta)$，s.t. $\theta \in \arg\min \mathcal{L}^{ft}(\vartheta)$。通过惩罚法转化为极小极大问题 $\min_\theta \max_\vartheta G_\lambda(\theta, \vartheta)$，用双循环算法求解：内循环固定 $\theta$ 做 K 步蒸馏微调（等价于标准微调，无额外开销），外循环固定 $\vartheta$ 用 $G_\lambda$ 梯度更新 $\theta$。框架可作为即插即用模块配合任意剪枝方法和概念遗忘方法使用。

### 关键设计

1. **双层优化形式化**:

    - 功能：解决微调与遗忘之间的循环依赖，找到两个目标的联合最优解
    - 核心思路：将约束优化问题 $\min \mathcal{L}^{CU}$ s.t. $\mathcal{L}^{ft}(\theta) - \inf_\vartheta \mathcal{L}^{ft}(\vartheta) \leq 0$ 通过惩罚法和变量分离转化为极小极大问题。核心公式 $G_\lambda(\theta, \vartheta) = \mathcal{L}^{CU}(\theta) + \lambda(\mathcal{L}^{ft}(\theta) - \mathcal{L}^{ft}(\vartheta))$。$\lambda$ 控制微调约束的强度，$\lambda$ 越大则 $\theta$ 越接近微调最优流形。
    - 设计动机：经典双层优化需要二阶导信息（Hessian），计算和内存开销巨大。近年的一阶双层优化框架（如 penalty method）只需梯度信息，使其在扩散模型这种大模型上可行。内循环的 $\max_\vartheta$ 等价于 $\min_\vartheta \mathcal{L}^{ft}$，就是标准微调——零额外开销。

2. **下层：蒸馏微调恢复生成能力**:

    - 功能：在约束层面恢复剪枝模型的生成质量
    - 核心思路：下层损失 $\mathcal{L}^{ft} = \mathcal{L}^{Diff} + \lambda^{OutKD}\mathcal{L}^{OutKD} + \lambda^{FeatKD}\mathcal{L}^{FeatKD}$，包含标准去噪损失、输出蒸馏（学生教师噪声预测匹配）和特征蒸馏（各层特征匹配）。内循环中对 $\vartheta$ 做 K 步梯度下降。
    - 设计动机：作者首先通过实验量化了蒸馏对收敛速度的影响——加入蒸馏后 FID 收敛显著快于纯扩散损失训练，且剪枝模型好于随机初始化（即使都用蒸馏）。这确认了蒸馏不可或缺但也会传播不良属性的双面性。

3. **上层：概念遗忘引导**:

    - 功能：在不破坏生成质量的前提下引导剪枝模型远离目标概念
    - 核心思路：上层采用类 ESD 的负引导策略——给定目标概念 $c$（如"Van Gogh 风格"）和锚定概念 $c'$（如"painting"），最小化 $\|\epsilon_\theta(x_t, t, c') - \epsilon_{\theta_{pruned}}(x_t, t, c)\|^2$，使模型在被提示目标概念时生成锚定概念的效果。外循环用 $G_\lambda$ 的梯度更新 $\theta$，该梯度同时包含遗忘损失和微调约束的信息。
    - 设计动机：与两阶段方法不同，双层优化使上层遗忘步骤中的梯度包含了下层微调约束的信息（通过 $\lambda \cdot \nabla_\theta \mathcal{L}^{ft}(\theta)$ 项），不会偏离微调最优流形太远，从而避免了两阶段方法中遗忘后生成质量退化的问题。

### 损失函数

- **下层微调损失**：$\mathcal{L}^{ft} = \mathcal{L}^{Diff} + \lambda^{OutKD}\mathcal{L}^{OutKD} + \lambda^{FeatKD}\mathcal{L}^{FeatKD}$
- **上层遗忘损失**：$\mathcal{L}^{CU} = \mathbb{E}\|\epsilon_\theta(x_t,t,c') - \epsilon_{\theta_{pruned}}(x_t,t,c)\|^2$
- **联合目标**：$G_\lambda(\theta, \vartheta) = \mathcal{L}^{CU}(\theta) + \lambda(\mathcal{L}^{ft}(\theta) - \mathcal{L}^{ft}(\vartheta))$，$\lambda = 100$

## 实验关键数据

### 风格去除实验（去除 Monet, Picasso, Van Gogh）

| 方法 | CLIP↓ | CP Score↑ | CSD↓ | COCO FID↓ | COCO CLIP↑ |
|:--|:--|:--|:--|:--|:--|
| Stable Diffusion 2.1 | 34.44 | 44.0 | 87.91 | 15.11 | 31.60 |
| 蒸馏模型（无遗忘） | 34.34 | 0.0 | 100.0 | 22.19 | 29.44 |
| 蒸馏 + ESD | 30.78 | 84.0 | 61.45 | 30.38 | 29.02 |
| 蒸馏 + UCE | 30.48 | 82.66 | 65.09 | 26.63 | 29.28 |
| 蒸馏 + ConceptPrune | 29.96 | 91.3 | 53.19 | 27.86 | 28.94 |
| **Bilevel (Ours)** | **26.28** | **97.6** | **39.04** | **22.24** | **29.19** |

### NSFW内容去除（MMA + Ring-a-Bell 对抗提示）

| 方法 | MMA去除率↑ | Ring-a-Bell去除率↑ | COCO FID↓ | COCO CLIP↑ |
|:--|:--|:--|:--|:--|
| 蒸馏 + ESD | 93.70 | 77.27 | 32.47 | 28.57 |
| 蒸馏 + ConceptPrune | 94.12 | 97.72 | 29.56 | 29.45 |
| **Bilevel (Ours, ESD)** | **91.60** | **94.32** | **26.80** | **29.94** |

### 关键发现

- **风格去除大幅领先**：CSD（专门衡量风格相似度的指标）上，双层方法比最佳两阶段基线低 27%（39.04 vs 53.19），同时 COCO FID 更优（22.24 vs 27.86）
- **蒸馏确实传播不良属性**：蒸馏模型的 CSD 甚至达到 100（比原始 SD 还高），说明蒸馏强化了风格学习
- **剪枝优于随机初始化**：即使使用蒸馏，剪枝初始化的收敛速度仍远优于随机初始化
- **双层方法计算开销零增加**：总计 20000 次迭代（内+外循环），与标准微调相同
- **NSFW去除方面持平或略优**：在 NSFW 任务上双层方法与两阶段基线相当，但生成质量（FID/CLIP）更优

## 亮点与洞察

1. **问题定义精准**：首次系统性地揭示并量化了蒸馏会传播不良属性的问题，给出了"蒸馏模型的 CSD=100"这样的直观证据
2. **理论优雅实践简洁**：从约束优化→惩罚法→极小极大→双循环SGD，数学推导完整但最终算法极其简单（交替微调+遗忘步即可），且无额外计算开销
3. **即插即用**：框架与剪枝方法无关（可用 SPDM、BK-SDM、APTP 等），也与遗忘方法无关（可用 ESD、UCE 等），组合灵活
4. **循环依赖的深刻分析**：通过图3的分析清晰展示了为什么两阶段方法会陷入次优——$\hat{\theta}$ 在微调损失面上是最优但在遗忘方向上不是良好的起点

## 局限性与可改进方向

1. **超参数 $\lambda$ 和 K 的选择**：$\lambda=100$、K=20 是实验选定值，不同概念/剪枝率下可能需要调整
2. **遗忘不完全**：从定性结果看，部分"残余泄漏"仍然存在（如基线生成Van Gogh画作中的女性形象）
3. **实验限于 SD 2.1**：未在更大的扩散模型（如 SDXL、SD3）上验证
4. **持续遗忘场景未深入**：论文承认先遗忘 $c_1$ 再遗忘 $c_2$ 的连续遗忘场景仍需进一步研究
5. **遗忘鲁棒性**：在对抗提示（MMA）上的去除率(91.60%)略低于 ConceptPrune(94.12%)

## 相关工作与启发

- **APTP (arXiv 2024)**：动态提示感知剪枝方法，本文的主要剪枝基础
- **ESD (ICCV 2023)**：基于负引导的概念擦除方法，本文上层优化中采用
- **BK-SDM (WACV 2022)**：通过去除U-Net冗余block的结构化剪枝方法
- **UCE (WACV 2024)**：通过修改注意力层 token 嵌入实现概念编辑
- **启发**：蒸馏虽然是压缩的利器，但其"忠实传递"特性也意味着不良属性的传播。任何涉及蒸馏的压缩流程都应考虑附带的安全性问题——这个洞察可推广到 LLM 蒸馏等其他领域

## 评分

⭐⭐⭐⭐ — 问题定义新颖且实用，理论推导优雅但实现简单，框架灵活可组合；但 NSFW 去除效果相比风格去除优势不大，且实验限于 SD 2.1。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[CVPR 2025\] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward](reward_fine-tuning_two-step_diffusion_models_via_learning_differentiable_latent-.md)

</div>

<!-- RELATED:END -->
