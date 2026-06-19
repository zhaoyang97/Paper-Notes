---
title: >-
  [论文解读] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation
description: >-
  [ICCV 2025][LLM安全][视频数据隐私] 本文首次研究防止视频数据被深度跟踪器未授权使用的问题，提出基于 DiT 的生成式框架生成时序不可学习样本（TUE），通过时间对比损失使跟踪器依赖扰动噪声进行时序匹配而非学习真实数据结构，实现了跨模型、跨数据集和跨任务的强可迁移性。 随着社交媒体的兴起…
tags:
  - "ICCV 2025"
  - "LLM安全"
  - "视频数据隐私"
  - "不可学习样本"
  - "视觉目标跟踪"
  - "生成式扰动"
  - "对比学习"
---

# Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation

**会议**: ICCV 2025  
**arXiv**: [2507.07483](https://arxiv.org/abs/2507.07483)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 视频数据隐私, 不可学习样本, 视觉目标跟踪, 生成式扰动, 对比学习

## 一句话总结
本文首次研究防止视频数据被深度跟踪器未授权使用的问题，提出基于 DiT 的生成式框架生成时序不可学习样本（TUE），通过时间对比损失使跟踪器依赖扰动噪声进行时序匹配而非学习真实数据结构，实现了跨模型、跨数据集和跨任务的强可迁移性。

## 研究背景与动机
随着社交媒体的兴起，大量用户上传的视频被收集用于训练商业视觉目标跟踪（VOT）模型，如 TrackingNet、LaSOT、GOT-10k 等大规模 VOT 数据集主要由这些用户视频组成。这引发了严重的**隐私和版权问题**——个人视频被用于跟踪模型训练而未获同意，而跟踪数据涉及敏感轨迹（如个人行踪、车辆路线、军事目标）。

**现有方法的局限**：不可学习样本（UE）在图像分类领域已被广泛研究——通过添加不可察觉的扰动使训练数据"不可学习"。然而，将图像 UE 直接扩展到视频面临三个挑战：

**效率问题**：视频数据分辨率高、帧数多，传统迭代优化（EM 方法）的 UE 生成计算密集。在 GOT-10k 上，EM 方法需要 33 小时和 3.4GB 参数存储

**有效性问题**：VOT 依赖跨帧的时序匹配（而非图像分类的语义识别），目标物体在不同帧中尺度变化很大，设计支持尺度不变匹配的 UE 更加困难

**迁移性问题**：现有 UE 对特定分类模型优化，难以跨不同跟踪器架构、数据集和任务迁移

**核心 idea：训练一个轻量生成器（而非逐样本优化），生成目标感知的时序扰动，结合时间对比学习使跟踪器学习到的是扰动噪声的"捷径匹配"而非真实视觉结构。**

## 方法详解

### 整体框架
四阶段流程：(1) 使用代理跟踪器 SiamFC 和跟踪数据集训练 TUE 生成器；(2) 用训练好的生成器为用户视频生成保护性扰动；(3) 未经授权方用添加了 TUE 的数据训练跟踪器；(4) 在干净测试数据上评估——受保护的数据应使跟踪器性能严重退化。

### 关键设计
1. **基于 DiT 的 TUE 生成器**:

    - 功能：生成目标感知的不可察觉扰动，添加到视频帧中使跟踪器训练失效
    - 核心思路：使用 DiT-S/8 架构（12层，6头注意力，隐藏维度384），以目标裁剪图像和归一化边界框状态 $\tilde{b}_i$ 作为输入，单次前向传播生成扰动：
    $\hat{z} = z + G_w(z, \tilde{b}_i), \quad \hat{x} = \Phi(x, G_w(c(x, b_j), \tilde{b}_j), b_j)$
      优化目标为最小化跟踪损失（令跟踪器"容易学到"扰动的捷径）：
    $\min_w \mathcal{L}(f_\theta(\hat{z}) * f_\theta(\hat{x}), y)$
    - 设计动机：相比 EM 方法逐视频迭代优化扰动（需 T 步 PGD），生成器方法**一次训练，到处生成**。参数量从 3.4GB 降至 124MB（28 倍压缩），训练时间从 33 小时降至 7 小时（4 倍加速）。更关键的是，生成器可以**零样本迁移**到未见过的数据集

2. **目标状态条件化**:

    - 功能：将目标的边界框信息作为条件注入生成器，使扰动适应不同尺度的目标
    - 核心思路：通过全连接层将归一化的目标状态 $\tilde{b} \in \mathbb{R}^4$（包含归一化的左上角坐标、宽和高）映射到隐藏空间，作为 DiT 的条件输入
    - 设计动机：VOT 中目标在不同帧中尺度变化很大。消融实验表明，去除目标条件（"- Condi."）后保护性能明显下降（AO: 22.5 vs 16.1），证明动态适应目标状态对 TUE 学习至关重要

3. **时间对比损失（Temporal Contrastive Loss, TCL）**:

    - 功能：进一步强化跟踪器对 TUE 噪声的依赖，增大干净样本和 TUE 样本的分布差距
    - 核心思路：以模板 TUE $\hat{z}$ 为 anchor，搜索区域中的 TUE $\hat{e}$ 为正样本，同一视频和其他视频的干净模板/目标为负样本：
    $\min_w [\mathcal{L}(f_\theta(\hat{z}) * f_\theta(\hat{x}), y) + \lambda \mathcal{L}_{cl}(f_\theta(\hat{z}), f_\theta(\hat{e}))]$
      其中 $\lambda = 0.05$
    - 设计动机：仅靠误差最小化可能不够，TCL 通过拉近 TUE 特征、推远干净特征，创造了更大的分布间隔。跟踪器在 TUE 数据上训练时只能学到噪声驱动的匹配模式，在干净测试数据上自然失效

4. **上下文感知扰动**:

    - 功能：同时对目标区域和上下文区域添加扰动
    - 核心思路：现有跟踪器不仅利用中心目标，还利用周围上下文区域进行模板匹配。仅扰动目标区域不够有效（AUC: 39.6 vs 29.5），需要同时腐化上下文信息
    - 设计动机：跟踪器可以从上下文区域学到足够的时序匹配线索。消融实验证实添加上下文噪声将 OTB AUC 从 39.6 进一步降至 29.5

### 损失函数 / 训练策略
- 交替优化：每次迭代先优化生成器 $G_w$（内循环），再优化代理跟踪器 $f_\theta$（外循环）
- 生成器训练 50 个 epoch，学习率 $5 \times 10^{-6}$，batch size 16
- 代理跟踪器：SiamFC（选择原因：训练高效，且生成器泛化性好）
- 扰动约束：$L_\infty$ 范数 $\leq 8/255$，人眼不可察觉
- 训练数据：GOT-10k，7 小时完成（单 RTX 4090）

## 实验关键数据

### 主实验
不同跟踪器在受保护数据上训练后的测试性能（越低表示保护越强）：

| 跟踪器 | 方法 | GOT-10k AO↓ | GOT-10k SR0.5↓ | OTB AUC↓ | LaSOT AUC↓ |
|--------|------|-------------|----------------|----------|-----------|
| SiamFC | Clean | 35.5 | 39.0 | 58.6 | 34.0 |
| SiamFC | EM | 21.4 | 18.2 | 29.5 | 17.6 |
| SiamFC | **TUE+TCL** | **12.1** | **9.0** | **11.4** | **9.5** |
| OSTrack-256 | Clean | 71.0 | 80.4 | 67.4 | 62.3 |
| OSTrack-256 | EM | 26.3 | 24.0 | 48.8 | 29.9 |
| OSTrack-256 | **TUE+TCL** | **18.0** | **15.1** | **30.5** | **22.0** |
| SeqTrack-256 | Clean | 74.7 | 84.7 | 68.1 | 63.6 |
| SeqTrack-256 | **TUE** | **2.3** | **0.7** | - | - |
| DropTrack-384 | Clean | 75.9 | 86.8 | 69.4 | 66.5 |
| DropTrack-384 | **TUE** | **17.1** | **12.9** | **36.7** | **25.2** |

### 消融实验

| 配置 | GOT-10k AO↓ | OTB AUC↓ | 说明 |
|------|-------------|----------|------|
| EM baseline (仅目标) | 27.0 | 39.6 | 传统方法 |
| EM + Context | 21.4 | 29.5 | 添加上下文噪声 |
| TUE Generator | 16.1 | 17.6 | 生成式方法 |
| TUE - Condition | 22.5 | 19.7 | 无目标条件 |
| **TUE + TCL** | **12.1** | **11.4** | 完整方法 |

模型复杂度对比：

| 方法 | 训练时间 | 可学习参数量 |
|------|---------|------------|
| EM + Context | 33 小时 | 3.4GB |
| **TUE Generator** | **7 小时** | **124MB** |

### 关键发现
- **生成式方法远优于迭代优化**：TUE Generator 在效率（4× 加速、28× 参数压缩）和效果上全面超越 EM baseline
- **TCL 是重要的增强因素**：在 SiamFC 上，添加 TCL 将 GOT-10k AO 从 16.1 进一步降至 12.1
- **跨模型迁移性强**：在简单的 SiamFC 上训练的生成器，能有效降低复杂跟踪器（OSTrack、SeqTrack、DropTrack）的性能
- **跨数据集零样本迁移**：GOT-10k 上训练的生成器可以直接用于其他数据集的 TUE 生成，无需重训
- **跨任务迁移**：TUE 不仅破坏 VOT，还能迁移到 VOS（视频对象分割）和长期点跟踪等任务
- 仅使用 25% 的 GOT-10k 视频（约 2300 条）即可训练出有效的 TUE 生成器

## 亮点与洞察
- **问题定义有社会意义**：首次关注 VOT 领域的视频数据隐私问题，在大模型时代具有现实紧迫性
- **生成式替代迭代优化的思路巧妙**：用 DiT 架构的单次前向取代 PGD 的多步迭代，效率提升而效果更好
- **时间对比学习的设计直觉清晰**：通过拉近 TUE 对、推远干净样本，最大化"捷径匹配"和"真实匹配"的分布差距
- **轻量代理+强大迁移**：在最简单的 SiamFC 上训练，但能保护免受最先进跟踪器的利用，说明 TUE 捕获了跟踪任务的通用弱点

## 局限与展望
- 扰动仅添加在目标/上下文区域，对全帧扰动场景（如动作识别保护）需要新设计
- $L_\infty \leq 8/255$ 的约束在高压缩视频（如 YouTube 上传后）中可能被破坏
- 对基于对抗训练或扰动去除的防御方法（如 adversarial training、图像净化）的鲁棒性未充分讨论
- SeqTrack 等大模型训练 epoch 较多时保护效果有轻微波动，长训练稳定性需关注
- 实际部署中，需要用户有能力生成和应用 TUE，可用性有待提升

## 相关工作与启发
- **不可学习样本从图像到视频的扩展**开辟了新的研究方向，时序匹配的保护比分类更复杂
- **DiT 作为轻量噪声生成器**的用法值得关注：不需要多步扩散，单步前向即可生成有效扰动
- **对比学习+数据隐私**的交叉值得进一步探索——对比信号可以有效放大"捷径"效应
- 该方法可能对其他时序匹配任务（如视频检索、Re-ID）也有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次研究VOT视频隐私保护，生成式TUE框架和TCL设计均为原创
- 实验充分度: ⭐⭐⭐⭐⭐ 跨模型/跨数据集/跨任务的全面迁移性验证
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述详细，流程图直观
- 价值: ⭐⭐⭐⭐⭐ 问题重要性高，方法实用性强，对AI安全领域有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining](../../ICLR2026/llm_safety/when_priors_backfire_on_the_vulnerability_of_unlearnable_examples_to_pretraining.md)
- [\[ICCV 2025\] Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)
- [\[ICLR 2026\] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](../../ICLR2026/llm_safety/perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)
- [\[CVPR 2025\] Protecting Your Video Content: Disrupting Automated Video-Based LLM Annotations](../../CVPR2025/llm_safety/protecting_your_video_content_disrupting_automated_video-based_llm_annotations.md)
- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](../../NeurIPS2025/llm_safety/evaluation_of_vision-llms_in_surveillance_video.md)

</div>

<!-- RELATED:END -->
