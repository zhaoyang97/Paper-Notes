---
title: >-
  [论文解读] DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture
description: >-
  [CVPR 2025][模型压缩][无数据知识蒸馏] 本文提出 DKDM 范式，首次实现扩散模型的无数据知识蒸馏——利用预训练教师模型的反向去噪过程替代真实数据分布，配合动态迭代蒸馏策略高效生成多样化训练知识，支持任意架构学生模型，在完全不接触原始数据的情况下实现与有数据训练相当甚至更优的生成性能。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "无数据知识蒸馏"
  - "扩散模型"
  - "动态迭代蒸馏"
  - "跨架构蒸馏"
  - "生成模型压缩"
---

# DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture

**会议**: CVPR 2025  
**arXiv**: [2409.03550](https://arxiv.org/abs/2409.03550)  
**代码**: [https://github.com/qianlong0502/DKDM](https://github.com/qianlong0502/DKDM)  
**领域**: 扩散模型  
**关键词**: 无数据知识蒸馏, 扩散模型, 动态迭代蒸馏, 跨架构蒸馏, 生成模型压缩

## 一句话总结

本文提出 DKDM 范式，首次实现扩散模型的无数据知识蒸馏——利用预训练教师模型的反向去噪过程替代真实数据分布，配合动态迭代蒸馏策略高效生成多样化训练知识，支持任意架构学生模型，在完全不接触原始数据的情况下实现与有数据训练相当甚至更优的生成性能。

## 研究背景与动机

**领域现状**：扩散模型在图像、视频等生成任务中表现出卓越能力，但训练依赖海量数据。例如，Stable Diffusion 需要数十亿图像-文本对，DALL·E 2 训练数据量高达 56 亿。各大组织已开源了大量预训练扩散模型，这些模型本身已经"记住"了训练数据的分布。

**现有痛点**：现有扩散模型蒸馏方法存在三方面限制：(1) 几乎所有方法都要求访问原始训练数据或其子集；(2) 学生模型通常必须使用与教师相同的架构并从教师权重初始化，严重限制架构灵活性；(3) 一种直觉的无数据方案是先用教师生成完整合成数据集再训练学生，但生成并存储数十亿合成图像的时间和空间成本完全不可接受。

**核心矛盾**：无数据条件下扩散模型的训练目标依赖真实数据 $x^0$——标准扩散训练的 KL 散度目标 $D_{KL}(q(x^{t-1}|x^t,x^0) \| p_\theta(x^{t-1}|x^t))$ 中的后验分布 $q$ 和前向加噪过程 $q(x^t|x^0)$ 都需要 $x^0$。如何消除对 $x^0$ 的依赖同时保持有效的训练信号，是核心挑战。

**本文目标**：提出完整的 Data-Free Knowledge Distillation for Diffusion Models (DKDM) 范式，满足三个严格要求：完全无数据、支持任意架构、知识传递高效。

**切入角度**：作者观察到，相比最终生成的"干净样本"，扩散过程中各时间步的"含噪样本"与扩散优化目标更直接相关。因此知识的形式应是教师模型去噪过程本身的中间状态，而不是最终生成结果。学生从教师的每一步去噪中学习，而非从最终输出中学习。

**核心 idea**：用教师扩散模型的反向过程分布 $p_{\theta_T}$ 替代真实后验 $q$，利用教师的生成能力替代前向加噪过程，设计理论上等价的 DKDM 目标函数，并通过动态迭代蒸馏将知识获取的时间复杂度从 $O(Tb)$ 降至 $O(b)$。

## 方法详解

### 整体框架

DKDM 框架包含一个预训练教师模型 $\theta_T$ 和一个随机初始化的任意架构学生模型 $\theta_S$。训练流程：(1) 从高斯噪声采样构建初始扩大批次集合；(2) 对批次中每个样本执行随机次数的教师去噪（shuffle denoise），使时间步分布均匀化；(3) 每次训练迭代从扩大集合中随机抽取子批次，教师和学生同时对该批次做前向推理，计算 DKDM 损失更新学生；(4) 去噪后的样本替换扩大集合中对应的旧样本，实现动态更新。整个过程不需要访问任何真实数据。

### 关键设计

1. **DKDM 优化目标**:

    - 功能：在无数据条件下为学生模型提供等效于标准扩散训练的优化信号
    - 核心思路：分两步消除对真实数据 $x^0$ 的依赖。第一步消除后验：由于训练良好的教师模型的条件分布 $p_{\theta_T}(x^{t-1}|x^t)$ 近似于真实后验 $q(x^{t-1}|x^t,x^0)$，直接用教师分布替代后验，将原始 KL 散度目标改为 $D_{KL}(p_{\theta_T}(x^{t-1}|x^t) \| p_{\theta_S}(x^{t-1}|x^t))$，噪声预测的 MSE 损失也相应改为 $\|\epsilon_{\theta_T}(x^t,t) - \epsilon_{\theta_S}(x^t,t)\|^2$。第二步消除前向先验：利用教师模型从纯噪声 $\epsilon$ 出发执行反向去噪链 $G_{\theta_T}(T-t)$ 生成含噪样本 $\hat{x}^t$，替代需要 $x^0$ 的前向过程 $q(x^t|x^0)$。最终 DKDM 目标为 $L_{DKDM} = L'_{simple} + \lambda L'_{vlb}$
    - 设计动机：通过三角不等式可证明，最小化 $D_{KL}(p_{\theta_T} \| p_{\theta_S})$ 间接最小化了 $D_{KL}(q \| p_{\theta_S})$，理论上保证了学生模型能逼近真实数据分布。这种设计不涉及架构内部结构，天然支持跨架构蒸馏

2. **动态迭代蒸馏策略**:

    - 功能：将构建训练批次的时间复杂度从 $O(Tb)$ 降至 $O(b)$，同时保证训练数据的多样性
    - 核心思路：分三个递进层次解决效率和多样性问题。**基础迭代蒸馏**：不再为每个训练迭代从头执行完整去噪链，而是让教师持续去噪——每次只执行一步 $g_{\theta_T}(x^t, t)$，输出直接作为下一步的输入，从 $t=T$ 迭代到 $t=1$ 后重新采样噪声重启，形成无限数据流。**Shuffle Denoise**：初始批次中每个样本被随机去噪不同次数，使批次内时间步 $t_i$ 服从均匀分布，避免同批次 $t$ 值相同导致的训练不稳定。**动态机制**：构建大小为 $\rho T |\hat{B}^s|$ 的扩大批次集合 $\hat{B}^+$，每次随机抽取子集 $\hat{B}^s$ 训练，去噪后的样本替换 $\hat{B}^+$ 中对应位置，打破样本始终"绑定"在同一批次的问题
    - 设计动机：朴素实现中获取 $\hat{x}^t = G_{\theta_T}(T-t)$ 需要 $T-t$ 步去噪，一个批次最坏需要 $O(Tb)$ 次教师推理，是标准训练的 $T$ 倍。三层递进策略在保持 $O(b)$ 复杂度的同时，从时间步分布和样本组合两个维度增强多样性

3. **跨架构蒸馏**:

    - 功能：将 CNN 教师模型的生成能力蒸馏到 ViT 学生模型，或反之
    - 核心思路：DKDM 目标只比较教师和学生在相同含噪输入 $\hat{x}^t$ 上的噪声预测和方差输出，不涉及中间特征匹配或权重初始化，因此天然支持任意架构组合。实验验证了 CNN→CNN、CNN→ViT、ViT→CNN、ViT→ViT 四种组合
    - 设计动机：现有方法（如 Xie et al.）发现学生模型随机初始化时蒸馏性能大幅下降，这是因为它们的蒸馏目标隐式依赖于相同的架构/权重。DKDM 从目标函数设计上消除了这一约束

### 损失函数 / 训练策略

最终损失 $L_{DKDM}^{\star} = L_{simple}^{\star} + \lambda L_{vlb}^{\star}$。$L_{simple}^{\star} = \mathbb{E}_{(\hat{x}^t,t) \sim \hat{B}^+}[\|\epsilon_{\theta_T}(\hat{x}^t,t) - \epsilon_{\theta_S}(\hat{x}^t,t)\|^2]$ 指导均值学习，$L_{vlb}^{\star} = \mathbb{E}_{(\hat{x}^t,t) \sim \hat{B}^+}[D_{KL}(p_{\theta_T}(\hat{x}^{t-1}|\hat{x}^t) \| p_{\theta_S}(\hat{x}^{t-1}|\hat{x}^t))]$ 指导方差学习。潜空间实验中因教师模型未训练方差故仅用 $L_{simple}^{\star}$。像素空间用 50 步 Improved DDPM 采样，潜空间用 200 步 DDIM 采样。缩放因子 $\rho = 0.4$，$\lambda = 0.001$。

## 实验关键数据

### 主实验

**像素空间**：

| 数据集 | 方法 | IS↑ | FID↓ | sFID↓ |
|--------|------|-----|------|-------|
| CIFAR10 32² | 有数据训练 | 8.73 | 7.84 | 7.38 |
| CIFAR10 32² | 无数据训练 (0%) | 8.28 | 12.06 | 13.23 |
| CIFAR10 32² | 数据有限 (20%) | 8.49 | 9.76 | 11.30 |
| CIFAR10 32² | **DKDM** | **8.60** | **9.56** | **11.77** |
| CelebA 64² | 有数据训练 | 3.04 | 5.39 | 7.23 |
| CelebA 64² | **DKDM** | **2.91** | **7.07** | **8.78** |

**潜空间**：

| 数据集 | 方法 | FID↓ | sFID↓ |
|--------|------|------|-------|
| CelebA-HQ 256² | 有数据训练 | 9.09 | 12.10 |
| CelebA-HQ 256² | 无数据训练 | 15.36 | 17.56 |
| CelebA-HQ 256² | **DKDM** | **8.69** | **12.50** |
| FFHQ 256² | 有数据训练 | 8.91 | 8.75 |
| FFHQ 256² | **DKDM** | **11.53** | **10.29** |

### 消融实验

动态迭代蒸馏各组件贡献（CIFAR10 FID）：

| 配置 | FID↓ |
|------|------|
| 基础迭代蒸馏 | ~13.5 |
| + Shuffle Denoise | ~11.2 |
| + Dynamic (ρ=0.4) | **9.56** |

跨架构蒸馏（CIFAR10 FID）：

| 教师→学生 | Data-Free | DKDM |
|-----------|-----------|------|
| CNN→CNN | 9.64 | **6.85** |
| ViT→CNN | 44.62 | **13.17** |
| CNN→ViT | 17.11 | **17.11** |
| ViT→ViT | 63.15 | **17.86** |

### 关键发现

- DKDM 在 CelebA-HQ 256 上 FID (8.69) 优于有数据训练 (9.09)，说明从教师的去噪过程学习可能比从原始数据学习更容易——预训练模型"平滑"了数据分布中的噪声和异常值
- 动态迭代蒸馏的三个组件层层递进，每步都有明确的 FID 提升
- 跨架构蒸馏中 DKDM 提升巨大（ViT→CNN 从 44.62 降至 13.17），证明知识传递的形式设计比单纯生成合成数据重要得多
- 方法仅增加少量 GPU 显存开销，在潜空间中训练速度甚至更快

## 亮点与洞察

- 开创性地定义了DKDM这一全新范式，在理论和实践上都有完整的解决方案
- "从生成过程学习"的核心洞察非常精妙——含噪中间状态比干净生成结果对扩散模型训练价值更大，本质上是让学生直接模仿教师的"推理过程"而非"推理结果"
- 动态迭代蒸馏的设计展现了出色的工程直觉——通过打乱时间步 + 扩大采样池 + 随机组合，在 $O(b)$ 预算内最大化训练数据的多样性
- 某些实验超越有数据训练的结果表明：预训练模型可能编码了比原始数据更"友好"的学习信号

## 局限与展望

- 学生模型上界受教师质量约束——如果教师模型生成质量不佳，蒸馏效果也受限
- 实验主要在中低分辨率（最高 256²）验证，缺少对高分辨率和大规模条件生成（如 text-to-image Stable Diffusion）的验证
- 动态迭代蒸馏的超参数 ρ 需要针对不同数据集调优
- 未来方向：(1) 应用到条件扩散模型的蒸馏；(2) 结合模型压缩同时减少参数和数据需求；(3) 探索在数据隐私受限场景的应用

## 相关工作与启发

- 与传统知识蒸馏（Hinton et al.）的核心区别：DKDM 不是用教师的软标签训练学生，而是用教师的反向过程分布替代数据分布
- 与扩散模型加速（Progressive Distillation, Consistency Models）的区别：后者关注减少采样步数，要求相同架构；DKDM 关注无数据训练，支持任意架构
- 启发：预训练模型作为"数据代理"的思路可推广到 GAN、Flow Model 等其他生成模型，在数据版权/隐私敏感场景有广泛应用前景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次定义并解决扩散模型无数据蒸馏问题，理论推导严谨完整
- **实验充分度**: ⭐⭐⭐⭐ — 像素/潜空间、多数据集、跨架构、消融全面，但缺少大规模实验
- **写作质量**: ⭐⭐⭐⭐ — 从问题定义到解决方案层层递进，逻辑清晰
- **价值**: ⭐⭐⭐⭐ — 对数据受限和模型架构迁移场景有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] When Data-Free Knowledge Distillation Meets Non-Transferable Teacher: Escaping Out-of-Distribution](../../ICML2025/model_compression/when_data-free_knowledge_distillation_meets_non-transferable_teacher_escaping_ou.md)
- [\[CVPR 2026\] DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models](../../CVPR2026/model_compression/dmgd_train-free_dataset_distillation_with_semantic-distribution_matching_in_diff.md)
- [\[CVPR 2026\] ManifoldGD: Training-Free Hierarchical Manifold Guidance for Diffusion-Based Dataset Distillation](../../CVPR2026/model_compression/manifoldgd_training-free_hierarchical_manifold_guidance_for_diffusion-based_data.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](what_makes_a_good_dataset_for_knowledge_distillation.md)
- [\[CVPR 2025\] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)

</div>

<!-- RELATED:END -->
