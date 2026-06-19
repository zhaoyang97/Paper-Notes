---
title: >-
  [论文解读] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection
description: >-
  [NeurIPS 2025][人体理解][社交交互检测] 提出一种部位感知的自底向上群组推理框架，通过姿态引导的身体部位特征增强和基于相似度的个体关联来推断社交群组和细粒度交互，在 NVI 和 Café 数据集上达到新 SOTA。 领域现状： 社交交互理解涵盖群组活动识别、行人轨迹预测、群组活动检测等任务…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "社交交互检测"
  - "身体部位感知"
  - "自底向上推理"
  - "非语言交互"
  - "姿态引导"
---

# Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection

**会议**: NeurIPS 2025  
**arXiv**: [2511.03666](https://arxiv.org/abs/2511.03666)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 社交交互检测, 身体部位感知, 自底向上推理, 非语言交互, 姿态引导

## 一句话总结

提出一种部位感知的自底向上群组推理框架，通过姿态引导的身体部位特征增强和基于相似度的个体关联来推断社交群组和细粒度交互，在 NVI 和 Café 数据集上达到新 SOTA。

## 研究背景与动机

**领域现状**: 社交交互理解涵盖群组活动识别、行人轨迹预测、群组活动检测等任务。近期 NVI-DET 任务要求检测细粒度非语言交互（面部表情、手势、姿态、注视、触摸），以 ⟨individual, group, interaction⟩ 三元组形式输出。

**现有痛点**: 现有方法（如 NVI-DEHR）存在两个关键缺陷：(1) 直接检测社交群组而不显式建模人与人之间的关系，导致空间距离远的注视交互等场景下群组预测模糊；(2) 将每个人表示为整体（holistic）嵌入，忽略了身体部位信息，难以区分语义相似但不同的交互（如"相互注视"vs"注视跟随"，"挥手"vs"指向"）。

**核心矛盾**: 细粒度社交交互（如注视方向、手势类型）的区分高度依赖局部身体部位线索，但现有方法仅使用全局人物表征。群组的组成应该从个体行为和个体间关系中自然涌现，而非直接预测。

**本文目标**: 设计一个从细粒度身体部位线索出发、通过个体间关系推理自底向上推断群组和交互的框架。

**切入角度**: 引入姿态估计作为特权信息（privileged information）引导部位感知学习，仅在训练时使用姿态标注，推理时无需额外输入。

**核心idea**: 先检测个体→用部位特征增强个体嵌入→基于个体间相似度关联推断群组→分类细粒度交互。

## 方法详解

### 整体框架

基于 DETR 的检测框架，包含四个核心模块：特征提取器、个体解码器、个体嵌入增强器、群组解码器+相似度关联模块。

### 关键设计

1. **个体嵌入增强器 (Individual Embedding Enhancer)**: 将每个检测到的个体分解为 $P$ 个身体部位。→ 通过 $P$ 个可学习线性投影生成部位查询：$\mathbf{Q}_P = \mathbf{E}_I \cdot [\mathbf{W}_1, \dots, \mathbf{W}_P] \in \mathbb{R}^{N_I \times P \times D}$。→ 部位查询通过自注意力（跨部位）和交叉注意力（与图像特征图）进行精炼。→ 将部位嵌入与个体嵌入拼接后投影融合：$\mathbf{E}_A = [\mathbf{E}_I, \mathbf{E}_P^1, \dots, \mathbf{E}_P^P] \cdot \mathbf{W}_{\text{fuse}}$。→ 与直接使用外部姿态估计器不同，本方法在推理时无需额外输入。

2. **姿态引导伪监督 (Pose-Guided Pseudo-Supervision)**: 使用 ViTPose 提取关键点作为训练时的特权信息。→ 为每个关键点定义方形窗口（大小与个体框成比例：$s_i = \alpha \cdot \max(w_i, h_i)$），生成二值掩码 $M_i^p$。→ 用 MSE 损失约束部位查询的注意力图趋近对应掩码：$\mathcal{L}_{\text{part}} = \frac{1}{N_I P} \sum_{i,p} \|A_i^p - M_i^p\|_2^2$。→ 使用 13 个关键点（排除 4 个面部关键点避免空间重叠）。

3. **自底向上群组推理 (Bottom-Up Group Reasoning)**: 群组解码器使用可学习群组查询，同时注意图像特征和部位感知个体嵌入。→ 输出群组框坐标和多标签交互分类分数。→ 关键区别：不像先前方法直接用群组查询预测群组框，而是让群组查询聚合相关个体的信息来推断群组。

4. **基于相似度的关联 (Similarity-based Association)**: 计算群组嵌入和个体嵌入之间的相似度矩阵：$\mathbf{S} = \text{MLP}(\mathbf{E}_G) \cdot \text{MLP}(\mathbf{E}_I)^T$。→ 为每个群组选择相似度最高的个体作为代表。→ 使用 BCE 损失训练关联：$\mathcal{L}_{\text{assn}}$。→ 允许个体查询和群组查询数量不同（先前方法要求相同）。

### 损失函数 / 训练策略

总损失为五项加权和：
$$\mathcal{L} = \lambda_i \mathcal{L}_{\text{ind}} + \lambda_c \mathcal{L}_{\text{cls}} + \lambda_l \mathcal{L}_{\text{loc}} + \lambda_p \mathcal{L}_{\text{part}} + \lambda_a \mathcal{L}_{\text{assn}}$$

- $\mathcal{L}_{\text{ind}}$: 个体目标性，使用 Focal Loss
- $\mathcal{L}_{\text{cls}}$: 多标签交互分类，使用 Asymmetric Loss (ASL)
- $\mathcal{L}_{\text{loc}}$: 框定位，$\ell_1$ + GIoU
- $\mathcal{L}_{\text{part}}$: 部位注意力监督，MSE
- $\mathcal{L}_{\text{assn}}$: 群组-个体关联，BCE
- 使用 Hungarian 算法进行预测与真值的匹配

## 实验关键数据

### 主实验（NVI 数据集）

| 方法 | val mR@25 | val mR@50 | val mR@100 | val AR | test AR |
|------|:-:|:-:|:-:|:-:|:-:|
| m-QPIC | 56.89 | 69.52 | 78.36 | 68.26 | 70.32 |
| m-CDN | 55.57 | 71.06 | 78.81 | 68.48 | 71.52 |
| m-GEN-VLKT | 50.59 | 70.87 | 80.08 | 67.18 | 71.72 |
| NVI-DEHR | 54.85 | 73.42 | 85.33 | 71.20 | 74.67 |
| **Ours** | **59.43** | **76.62** | **87.43** | **74.49** | **78.52** |

### Café 数据集

| 方法 | Group mAP₁.₀ (view) | Group mAP₀.₅ (view) | Outlier mIoU (view) |
|------|:-:|:-:|:-:|
| Café-base | 14.36 | 37.52 | 63.70 |
| **Ours** | **18.23** | **46.88** | **67.62** |

### 消融实验

| 设置 | mR@25 | AR |
|------|:-:|:-:|
| 完整模型 | 59.43 | 74.49 |
| 移除增强器 | 55.20 | 72.17 |
| 移除相似度关联 | 55.95 | 72.75 |
| 两者都移除 | 56.29 | 70.86 |
| 移除 $\mathcal{L}_{\text{assn}}$ | 30.38 | **48.32** (大幅下降) |
| 移除 $\mathcal{L}_{\text{part}}$ | 54.32 | 73.58 |

### 与 MLLM 对比

| 方法 | mR@25 | AR |
|------|:-:|:-:|
| **Ours** | **63.59** | **78.52** |
| LLaVA (给GT群组框) | 21.09 | 37.14 |
| LLaVA-LoRA 微调 | 17.40 | 33.81 |

### 关键发现

- **部位增强器和相似度关联各贡献约 2 个 AR 点**，两者叠加效果大于单独使用。
- **关联损失 $\mathcal{L}_{\text{assn}}$ 至关重要**：移除后 AR 从 74.49 暴跌至 48.32。
- **13 个关键点最优**：排除面部的 4 个冗余关键点；使用全部 17 个反而略降。
- **姿态引导优于 CLIP 引导**：ViTPose 提供的空间精确性优于 VLM 的对比学习表征。
- **LLaVA 即使给定 GT 群组框也远不如本方法**：说明通用多模态模型在细粒度社交推理上仍有巨大差距。
- **在 Café 上无时序建模仍超越先前方法**：证明部位感知表示和自底向上推理的通用性。

## 亮点与洞察

- 将姿态估计作为"特权信息"仅用于训练是精妙的设计——获得细粒度监督信号但不增加推理成本。
- 自底向上的群组推理符合社交交互的本质：群组是从个体间关系中涌现的，而非被独立预测的实体。
- 注意力可视化清晰展示了群组解码器如何关注面部区域（面部表情交互）或手部区域（握手交互）。
- 相似度矩阵关联方案比 guided embedding 更灵活，允许个体和群组查询数量不同。

## 局限与展望

- 目前未利用时序信息，在视频场景中加入时序建模可能进一步提升。
- 部位数量 P=13 是固定的，可探索动态部位发现机制。
- 仅在 NVI 和 Café 两个数据集上验证，缺少更大规模或更多样场景的评估。
- 对遮挡严重的场景（部分身体部位不可见）的鲁棒性分析不够充分。
- 个体解码器和群组解码器的级联结构可能引入误差累积。

## 相关工作与启发

- NVI-DEHR 使用超图建模高阶关系但忽略了人与人的基本关系和部位信息，本文方法直接弥补了这两个缺陷。
- HOI-DET 中的 QPIC、CDN 等方法被适配为基准，但 NVI-DET 需要群组级推理，这是 HOI-DET 不涉及的。
- 特权信息学习（privileged information）的思想来自 Vapnik，在本文中优雅地应用于姿态监督。
- 人体姿态引导在 HOI-DET 中已有探索（如 Wu et al.、Lei et al.），但本文首次将其应用于社交交互检测且仅需训练时使用。

## 评分

- 新颖性: ⭐⭐⭐⭐ 部位感知+自底向上推理的组合在社交交互检测中是新颖的
- 实验充分度: ⭐⭐⭐⭐⭐ 消融全面，与 HOI/MLLM 基准对比充分，可视化分析丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义准确，图示直观
- 价值: ⭐⭐⭐⭐ 对细粒度社交理解任务有实质性贡献，设计理念有通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification](../../CVPR2025/human_understanding/hsemotion_team_at_abaw-10_competition_facial_expression_recognition_valence-arou.md)
- [\[NeurIPS 2025\] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness](some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)
- [\[CVPR 2026\] MoBind: Motion Binding for Fine-Grained IMU-Video Pose Alignment](../../CVPR2026/human_understanding/mobind_motion_binding_for_fine-grained_imu-video_pose_alignment.md)
- [\[CVPR 2026\] MOFA-VTON: More Fashion Possibilities with Fine-Grained Adaptations in Virtual Try-On](../../CVPR2026/human_understanding/mofa-vton_more_fashion_possibilities_with_fine-grained_adaptations_in_virtual_tr.md)
- [\[AAAI 2026\] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](../../AAAI2026/human_understanding/clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)

</div>

<!-- RELATED:END -->
