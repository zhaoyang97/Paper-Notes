---
title: >-
  [论文解读] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling
description: >-
  [AAAI 2026][医学图像][通用分割] 提出 ProPL 框架，通过共享视觉编码器 + 提示引导双解码器 + 不确定性驱动伪标签校准，首次实现通用半监督超声图像分割，在 5 个器官 8 个任务上以极少标注数据（1/16）超越全监督方法 5.18% mDice。 领域现状：超声图像分割是计算机辅助诊断的关键…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "通用分割"
  - "半监督学习"
  - "伪标签"
  - "提示引导"
  - "超声图像"
---

监督方法5.18% mDice | ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling | AAAI 2026 | arXiv 2511.15057"
tags:
  - AAAI 2026
  - semi-supervised learning
  - 半监督学习
  - ultrasound segmentation
  - 超声分割
  - 医学图像


# ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling

**会议**: AAAI 2026  
**arXiv**: [2511.15057](https://arxiv.org/abs/2511.15057)  
**代码**: [https://github.com/WUTCM-Lab/ProPL](https://github.com/WUTCM-Lab/ProPL)  
**领域**: 医学图像 / 超声分割  
**关键词**: 通用分割, 半监督学习, 伪标签, 提示引导, 超声图像

## 一句话总结

提出 ProPL 框架，通过共享视觉编码器 + 提示引导双解码器 + 不确定性驱动伪标签校准，首次实现通用半监督超声图像分割，在 5 个器官 8 个任务上以极少标注数据（1/16）超越全监督方法 5.18% mDice。

## 研究背景与动机

**领域现状**：超声图像分割是计算机辅助诊断的关键，但现有方法通常针对特定器官或任务设计，泛化性差。

**现有痛点**：
   - 全监督方法需要大量标注数据，超声图像标注尤其困难（散斑噪声、声影、组织伪影模糊边界）
   - 半监督方法虽减少数据需求，但仍局限于单任务
   - 通用分割框架（如 DoDNet、UniSeg）仅支持全监督，受限于标注数据

**核心问题**：如何构建一个能同时处理多器官多任务、且只需少量标注的通用超声分割框架？

**切入角度**：结合提示学习实现任务自适应 + 双解码器互学习生成可靠伪标签

## 方法详解

### 整体框架

输入超声图像 → 共享 ConvNeXt-Tiny 编码器 → 双解码器（标准解码器 $\mathcal{G}_{sd}$ + 提示解码器 $\mathcal{G}_{pd}$）→ 互相用伪标签监督对方。任务提示通过 BERT 编码后注入提示解码器。

### 关键设计

1. **Prompting-upon-Decoding (PuD)**:

    - 功能：将任务特定的文本提示注入解码过程
    - 核心思路：用 BERT 编码任务描述得到 $\bm{t}$，经 1D 卷积+线性映射对齐维度后，通过多头交叉注意力注入解码特征：$\bm{h}_k = \bm{z}_k' + \alpha \cdot \text{MHCA}(Q=\bm{z}_k', K=\bm{\tau}, V=\bm{\tau})$
    - 设计动机：不同于 one-hot 编码或可学习提示，文本提示语义更丰富且可扩展到新任务；$\alpha$ 可学习控制提示影响力度

2. **不确定性驱动伪标签校准 (UPLC)**:

    - 功能：基于预测不确定性过滤和校准伪标签
    - 核心思路：双解码器分别生成预测，预测不一致的区域不确定性高，仅使用高置信度区域的伪标签进行互学习
    - 设计动机：直接使用原始伪标签会引入噪声；不确定性估计利用双解码器的分歧作为信号

3. **通用超声数据集**:

    - 6,400 张图像，5 个器官（乳腺、胎儿、心脏、卵巢、甲状腺），8 个分割任务
    - 标注数据分区：1/16、1/8、1/4 三种设置

## 实验关键数据

### 主实验（1/16 标注数据）

| 方法类型 | 方法 | mDice% | mIoU% |
|---------|------|--------|-------|
| 单任务监督 | U-Net | 75.17 | 64.76 |
| 单任务半监督 | UniMatch | 79.38 | 69.66 |
| 通用监督 | DoDNet | 62.99 | 50.04 |
| 通用监督 | CLIP-UM | 63.70 | 51.27 |
| **通用半监督** | **ProPL** | **80.35** | **70.63** |

### 不同标注比例

| 标注比例 | ProPL mDice | vs 次优提升 |
|---------|-------------|-----------|
| 1/16 | 80.35% | +0.97% |
| 1/8 | 82.56% | +2.2% |
| 1/4 | 83.70% | +1.32% |

### 消融实验（1/16）

| 配置 | mDice | mIoU |
|------|-------|------|
| w/o 提示 (PuD) | 60.76 | 52.57 |
| w/o UPLC | 77.85 | 67.23 |
| Full ProPL | **80.35** | **70.63** |

### 关键发现
- 移除任务提示导致 mDice 下降 **19.59%**（80.35→60.76），说明提示对通用模型至关重要
- UPLC 贡献 2.5% mDice 提升，不确定性校准有效过滤噪声伪标签
- ProPL 仅 712MB 显存，在性能-效率 Pareto 前沿优于所有对比方法
- 通用监督方法（DoDNet、CLIP-UM）在超声数据上表现不佳（~63% mDice），说明通用模型需半监督辅助

## 亮点与洞察
- **首次定义"通用半监督超声分割"任务**：将多器官多任务的通用性和少标注的实际需求结合
- **文本提示 vs one-hot/可学习提示**：文本提示虽增加 18s/epoch 但语义更丰富，移除后模型崩溃
- **双解码器互学习**：一个解码器的高置信预测作为另一个的伪标签，通过分歧估计不确定性

## 局限与展望
- 数据集仅包含 2D 超声图像，未扩展到 3D 体积超声
- 提示模板需要人工设计，自动化提示生成可能进一步提升
- UPLC 的阈值依赖调参，自适应阈值策略值得探索
- 跨模态泛化未验证（如 CT/MRI 能否共用同一框架）

## 相关工作与启发
- **vs UniMatch (半监督)**：UniMatch 是单任务半监督 SOTA，ProPL 在通用设置下超越其 0.97%
- **vs DoDNet/UniSeg (通用监督)**：这些方法在超声数据上效果不佳，ProPL 通过半监督弥补标注不足
- **vs SAM-based**：SAM 方法需额外交互提示（点、涂鸦），ProPL 仅需任务文本描述

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义通用半监督超声分割，框架设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个任务、多标注比例、详细消融、效率分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，贡献明确
- 价值: ⭐⭐⭐⭐ 数据集+框架对临床超声分割有实际推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2025/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2026\] Semi-supervised Echocardiography Video Segmentation via Anchor Semantic Awareness and Continuous Pseudo-label Reforging](../../CVPR2026/medical_imaging/semi-supervised_echocardiography_video_segmentation_via_anchor_semantic_awarenes.md)
- [\[AAAI 2026\] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)

</div>

<!-- RELATED:END -->
