---
title: >-
  [论文解读] Few-Shot Personalized Scanpath Prediction
description: >-
  [CVPR 2025][人体理解][扫视路径预测] 提出少样本个性化扫视路径预测（FS-PSP）任务 和 Subject-Embedding Network（SE-Net），通过将主体嵌入学习与扫视路径预测解耦，仅需 1-10 张图像的注视数据即可适配新用户，在 OSIE、COCO-FreeView、COCO-Search18 三个数据集上 ScanMatch 指标超越第二名 5.9%-7.9%，且适配时间仅 3.6 秒、无需微调。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "扫视路径预测"
  - "少样本学习"
  - "个性化"
  - "主体嵌入"
  - "眼动追踪"
---

# Few-Shot Personalized Scanpath Prediction

**会议**: CVPR 2025  
**arXiv**: [2504.05499](https://arxiv.org/abs/2504.05499)  
**代码**: [https://github.com/cvlab-stonybrook/few-shot-scanpath](https://github.com/cvlab-stonybrook/few-shot-scanpath)  
**领域**: 视频理解  
**关键词**: 扫视路径预测, 少样本学习, 个性化, 主体嵌入, 眼动追踪

## 一句话总结

提出少样本个性化扫视路径预测（FS-PSP）任务 和 Subject-Embedding Network（SE-Net），通过将主体嵌入学习与扫视路径预测解耦，仅需 1-10 张图像的注视数据即可适配新用户，在 OSIE、COCO-FreeView、COCO-Search18 三个数据集上 ScanMatch 指标超越第二名 5.9%-7.9%，且适配时间仅 3.6 秒、无需微调。

## 研究背景与动机

**领域现状**：扫视路径预测（scanpath prediction）旨在预测人在观看图像时的注视点序列，包含位置和时间信息。个性化扫视路径预测（PSP）进一步要求为特定个体预测其独特的注意力模式，因个人文化背景、记忆和经验差异会影响注视行为。现有 PSP 方法如 ISP 和 EyeFormer 通过为每个训练用户分配一个可学习嵌入向量实现个性化。

**现有痛点**：现有 PSP 方法需要大量数据来训练每个用户的嵌入。ISP 在仅有 10 张支持样本时性能大幅下降；EyeFormer 需要至少 50 个扫视路径才能获得稳定的个性化嵌入。更根本的问题是，这些方法将主体嵌入作为扫视路径预测的"副产物"联合学习，新用户需要通过微调来重新学习嵌入，既耗时又容易过拟合。

**核心矛盾**：个性化需要足够多的数据来刻画个人注意力特征，但实际应用中不可能让每个新用户长时间在实验室录制眼动数据。需要在极少数据下（1-10 张图像）快速捕捉个人注视特征。

**本文目标** 设一个新用户提供了 1-10 张图像的扫视路径，如何在不微调模型的情况下即时为其预测个性化扫视路径？

**切入角度**：作者提出解耦策略——将"学习什么是个性化注意力特征"与"根据特征预测扫视路径"分开。先训练一个专门提取主体嵌入的网络 SE-Net，再训练一个以嵌入为条件的扫视路径预测器。新用户只需通过 SE-Net 前馈提取嵌入，即可用于预测。这类似原型网络的思路：从少量样本中提取原型表示。

**核心 idea**：用专门的主体嵌入网络解耦个性化特征提取和路径预测，使得新用户只需前馈一次即可获得可用的个性化嵌入，无需任何微调。

## 方法详解

### 整体框架

两阶段训练，推理时冻结两个模型。训练阶段：(1) 在 base 数据集上训练 SE-Net 提取已知用户的嵌入（分类损失 + 对比损失）；(2) 用 SE-Net 产生的嵌入训练 ISP-SENet（条件化的扫视路径预测器）。推理阶段：(1) 从新用户的 n-shot 支持集通过 SE-Net 提取嵌入并取平均；(2) ISP-SENet 以此嵌入为条件预测新图上的扫视路径。

### 关键设计

1. **SE-Net（Subject-Embedding Network）**:

    - 功能：从单个图像-扫视路径对中提取反映个人注意力特征的嵌入向量
    - 核心思路：特征提取分三层。(1) 图像+扫视路径语义特征：用 ResNet + Deformable Attention 编码图像获得 $F_I$，编码扫视路径获得 $F_S$。(2) Context-Scanpath Encoder (CSE)：将任务描述通过 RoBERTa 编码为任务嵌入 $t$，与图像特征通过 Self-Attention 融合为上下文 $C$，再与扫视路径特征（加上时长嵌入和位置嵌入）联合编码，丢弃 $C$ 部分得到更新的 $F_S'$（去除场景特异性偏见）。(3) User-Scanpath Decoder (USD)：初始化主体 token $e$，通过 Cross-Attention 从 $F_S'$ 中提取个人特征：$e = \text{ReLU}(\text{Linear}(e + \text{CrossAttn}(e, F_S')))$
    - 设计动机：直接学习嵌入（如 ISP 的查找表方式）无法泛化到新用户。SE-Net 以输入驱动的方式提取嵌入，能利用已知用户的先验经验；丢弃上下文 $C$ 可防止嵌入过度编码场景内容而非个人特征

2. **SE-Net 训练：分类 + 对比损失**:

    - 功能：确保嵌入能区分不同用户，同一用户的不同扫视路径嵌入相近
    - 核心思路：构建三元组 $(d, d_+, d_-)$，其中 $d_+$ 来自同一用户，$d_-$ 来自不同用户。训练损失为 $\mathcal{L}_{cls}(d) + \mathcal{L}_{cls}(d_+) + \mathcal{L}_{cls}(d_-) + \mathcal{L}_{contrast}$。分类损失让嵌入预测用户 ID；对比损失（三元组损失）让同一用户的嵌入比不同用户的更近：$\max(\|f(d)-f(d_+)\|^2 - \|f(d)-f(d_-)\|^2 + m, 0)$
    - 设计动机：分类损失提供强梯度信号让嵌入有区分性；对比损失保证嵌入空间的几何良好性（同类紧凑、异类分离），利于少样本时的原型聚合

3. **ISP-SENet（条件化扫视路径预测器）**:

    - 功能：以 SE-Net 产生的主体嵌入为条件，预测个性化扫视路径
    - 核心思路：基于 Gazeformer-ISP 架构，将原来的固定查找表嵌入替换为 SE-Net 的输出。训练时用 SE-Net 为 base 用户生成嵌入并冻结 SE-Net，只训练预测器。推理时两个网络都冻结
    - 设计动机：解耦设计让预测器专注于"给定个性化特征如何预测路径"，而非同时学习"什么是个性化特征"

### 损失函数 / 训练策略

SE-Net 训练：分类损失 + 三元组对比损失，25 epochs。ISP-SENet 训练：先有监督训练，再用 SCST（自临界序列训练）强化学习微调，各 10 epochs。$n$-shot 推理时取 $n$ 个支持样本的嵌入平均值（原型网络思想）。注视时长离散为 10 个 bin（替代连续值）以降低噪声。

## 实验关键数据

### 主实验

| 数据集 | n-shot | ISP-SENet SM↑ | 第二名 SM | 相对提升 |
|--------|--------|-------------|----------|---------|
| OSIE | 1 | 0.368 | 0.354 | +3.9% |
| OSIE | 10 | **0.375** | 0.354 | **+5.9%** |
| COCO-FreeView | 10 | **0.367** | 0.340 | **+7.9%** |
| COCO-Search18 | 10 | **0.482** | 0.449 | **+7.3%** |

适配时间对比：ISP-SENet 仅需 3.62 秒（前馈），Gazeformer-ISP 需 267 秒（微调）。

### 消融实验

| 配置 | SM (OSIE) | 说明 |
|------|-----------|------|
| ISP-SENet-Seen (训练集用户) | 0.390 | 上界 |
| ISP-SENet-Unseen (10-shot) | 0.375 | 接近上界 |
| 无 CSE 模块 | 下降明显 | 任务感知重要 |
| 无对比损失 | 嵌入区分度降低 | 几何结构必要 |

### 关键发现

- ISP-SENet 在 1-shot 时就达到接近 10-shot 的性能，说明 SE-Net 能从极少数据中提取有效的个性化特征
- Baseline 方法在微调时倾向过拟合支持集的图像内容而非学到注意力模式；ISP-SENet 通过丢弃上下文特征有效避免了这一问题
- Scanpath accuracy 指标显示 ISP-SENet 最能区分不同用户的预测（35.57 vs 31.99），说明个性化确实被捕获
- ISP-SENet-Unseen 性能接近 ISP-SENet-Seen（0.375 vs 0.390），证明少样本适配几乎不损失质量

## 亮点与洞察

- **解耦主体嵌入学习与路径预测是核心创新**：这使得新用户适配变成单次前馈操作（3.6秒），比微调方法快 70 倍。思路类似于 meta-learning 中"学会学习"的范式，但实现更简洁
- **丢弃上下文特征以去除场景偏见的设计很聪明**：在 CSE 中引入图像上下文帮助理解注视行为，但在最终嵌入中丢弃上下文部分，确保嵌入只编码"这个人怎么看"而非"这张图有什么"
- **注视时长离散化为 10 bin 是实用技巧**：微小时长差异（如 200ms vs 203ms）对个性化无意义，bin 化既降噪又减少参数量。这一做法可推广到其他涉及连续时间信号的任务

## 局限与展望

- 数据集的用户数量有限（10-15人），泛化到更大人群需要更大规模的研究
- SE-Net 的分类头在训练时只区分 base 用户，对"以前从未见过的注意力模式"的泛化能力取决于 base 用户的多样性
- 只在自由浏览和视觉搜索两种任务上验证，阅读、驾驶等其他注视场景未覆盖
- 对比损失中 margin $m$ 的选择需要根据任务类型调整（自由浏览用小 margin，搜索任务用大 margin）
- 原型聚合（取平均）可能不是最优策略，注意力加权聚合或基于图的聚合可能更好

## 相关工作与启发

- **vs ISP**：ISP 为每个用户学习固定嵌入向量（查找表），新用户必须微调。ISP-SENet 通过 SE-Net 动态生成嵌入，支持零微调适配
- **vs EyeFormer**：EyeFormer 用强化学习 + viewer encoder 做 PSP，但需 50+ 样本才稳定。ISP-SENet 在 1-shot 即可工作
- **vs 原型网络（Prototypical Networks）**：ISP-SENet 的"提取嵌入→取平均→条件预测"与原型网络异曲同工，但嵌入提取的输入是图像-扫视路径对而非简单图像，需要解耦场景信息和个人特征

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次提出 FS-PSP 任务，解耦设计合理有效
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、三种 n-shot 设置、多种 baseline 对比完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 对个性化注意力预测的实际应用（推荐系统、广告、辅助诊断）有直接推动价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images](fresa_feedforward_reconstruction_of_personalized_skinned_avatars_from_few_images.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [\[CVPR 2025\] PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)

</div>

<!-- RELATED:END -->
