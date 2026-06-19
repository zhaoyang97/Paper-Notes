---
title: >-
  [论文解读] Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders
description: >-
  [ECCV 2024][自监督学习][掩码自编码器] 提出CropMAE——用同一图像的两个随机裁剪视图替代视频帧对来训练孪生掩码自编码器，在98.5%的极高掩码率下仅用2个可见patch即可学习物体边界感知表征，训练速度比SiamMAE提升最高23.8倍，同时在视频传播任务上达到竞争性能。 自监督预训练是视觉表征学习的核心…
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "掩码自编码器"
  - "孪生网络"
  - "视频分割"
  - "标签传播"
---

# Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders

**会议**: ECCV 2024  
**arXiv**: [2403.17823](https://arxiv.org/abs/2403.17823)  
**代码**: [https://github.com/alexandre-eymael/CropMAE](https://github.com/alexandre-eymael/CropMAE)  
**领域**: 自监督学习  
**关键词**: 自监督学习, 掩码自编码器, 孪生网络, 视频分割, 标签传播

## 一句话总结

提出CropMAE——用同一图像的两个随机裁剪视图替代视频帧对来训练孪生掩码自编码器，在98.5%的极高掩码率下仅用2个可见patch即可学习物体边界感知表征，训练速度比SiamMAE提升最高23.8倍，同时在视频传播任务上达到竞争性能。

## 研究背景与动机

自监督预训练是视觉表征学习的核心，其中掩码图像建模（MAE）通过重建被遮挡patch学习语义特征。SiamMAE进一步引入孪生架构，用视频中的两帧建立对应关系，在视频传播任务（目标分割、姿态传播等）上取得SOTA。

**然而SiamMAE存在两个关键瓶颈**：

**必须依赖视频数据集**：图像数据集的规模通常远大于视频数据集，且解码成本更低；视频训练受限于数据可用性和计算成本

**训练效率低**：需要2000 epochs才能收敛，因为需要从视频运动中隐式学习物体概念理解，预训练任务（从少量可见patch重建整帧）需要深度的语义知识

**核心洞察**：SiamMAE学到的物体边界理解能力真的来自视频中的显式运动吗？如果两个视图之间的**隐式图像变换**（裁剪、翻转等）才是真正的驱动力，那么完全可以用静态图像的不同裁剪替代视频帧对。

**本文核心idea**：用同一图像的随机裁剪（Global-to-Local策略）替代视频帧对，构造明确可求解的代理任务（局部视图始终包含在全局视图中），无需学习世界的概念知识即可完成重建，从而支持更高掩码率（98.5%）和更快收敛。

## 方法详解

### 整体框架

CropMAE的训练pipeline：输入一张图像 $I$ → 生成两个裁剪视图 $V_1$（全局，不遮挡）和 $V_2$（局部，极高掩码率）→ 孪生ViT编码器分别编码两个视图 → Transformer解码器通过交叉注意力从 $V_1$ 重建 $V_2$ → 最小化L2重建损失。预训练结束后丢弃解码器，编码器作为特征提取器用于下游任务。

### 关键设计

1. **裁剪策略（Global-to-Local Views）**: 探索了4种裁剪方式：Same Views（同一裁剪，性能最差 $\mathcal{J\&F}_m=36.6$）、Random Views（两个独立随机裁剪，60.0）、Local-to-Global（$V_1$ 从 $V_2$ 中裁剪，55.9）和Global-to-Local（$V_2$ 从 $V_1$ 中裁剪，**60.4**，最优）。Global-to-Local最优的关键在于：局部视图 $V_2$ 必然完整包含在全局视图 $V_1$ 中，因此重建任务**始终可解且无需先验概念知识**——模型只需(i)从少量可见patch定位局部视图在全局中的位置，(ii)确定重建所需的变换。

2. **极高掩码率（98.5%）**: 传统MAE使用75%掩码率，视频MAE用90%，SiamMAE用95%（9/196个可见patch）。CropMAE将掩码率推至98.5%，仅保留**2个可见patch**。这源于代理任务的本质差异：其他MAE需要通过"幻想"被遮挡内容来学习概念理解，而CropMAE的任务直接可解，因此需要更高掩码率来制造挑战。从95%到98.5%，可见patch数从9降至2（减少4.5倍），极大降低了注意力层的计算量。

3. **解码器架构**: 4层Transformer（$d_{model}=256$, $d_{ff}=2048$），交替使用自注意力（遮挡图像token间）和交叉注意力（遮挡token attend到可见图像token）。L2损失作用于归一化像素值。解码器刻意保持小于编码器（$256$-d vs $384$-d），避免解码器过强而编码器学不到好表征。

### 损失函数 / 训练策略

- **重建损失**：$\mathcal{L} = \| V_2 - R \|_2^2$，其中 $R$ 为重建输出，$V_2$ 经像素值归一化
- 优化器：AdamW，基础学习率 $1.5 \times 10^{-4}$
- 编码器：ViT-S/16（主要实验）或 ViT-B/16
- 无需Color Jitter和Gaussian Blur（实验证明有害），仅可选水平翻转
- 训练400 epochs即可（SiamMAE需2000 epochs）

## 实验关键数据

### 主实验

在三个视频传播下游任务上的比较（标签传播评估，无微调）：

| 方法 | 骨干 | 数据集 | Epochs | DAVIS $\mathcal{J\&F}_m$ | VIP mIoU | JHMDB PCK@0.1 |
|------|------|--------|--------|-------------------------|----------|---------------|
| SiamMAE (论文) | ViT-S/16 | K400 | 2000 | 62.0 | 37.3 | 47.0 |
| SiamMAE (复现) | ViT-S/16 | K400 | 400 | 57.9 | 33.2 | **46.1** |
| **CropMAE** | ViT-S/16 | K400 | 400 | 58.6 (+0.7) | 33.7 (+0.5) | 42.9 |
| **CropMAE** | ViT-S/16 | IN Sub | 400 | **60.4 (+2.5)** | 33.3 | 43.6 |
| **CropMAE** | ViT-B/16 | IN Sub | 400 | **60.9** | 32.8 | 44.3 |
| MAE | ViT-B/16 | IN | 1600 | 53.5 | 28.1 | 44.6 |
| VideoMAE | ViT-S/16 | K400 | 800 | 39.3 | 23.3 | 41.0 |

**关键发现**：在相同400 epochs预算下，CropMAE在DAVIS上比SiamMAE高2.5%（用ImageNet），且收敛更快（150 epochs即达58.0）。JHMDB表现略逊于SiamMAE，因该任务涉及人体姿态变形，视频真实运动更有帮助。

### 消融实验

| 配置 | DAVIS $\mathcal{J\&F}_m$ | 说明 |
|------|--------------------------|------|
| **裁剪策略** | | |
| Same Views | 36.6 | 无法学到传播能力 |
| Random Views | 60.0 | 有时可解 |
| Local-to-Global | 55.9 | 全局重建需要概念知识，较难 |
| **Global-to-Local** | **60.4** | 始终可解，最优 |
| **掩码率** | | |
| 75% (49 patches) | 45.3 | 任务太简单，编码器学不到有用特征 |
| 90% (19 patches) | 47.1 | 仍太简单 |
| 95% (9 patches) | 51.2 | SiamMAE的选择 |
| **98.5% (2 patches)** | **60.4** | 最优，极端但有效 |
| 99% (1 patch) | 58.6 | 稍过极端 |
| **解码器深度** | | |
| 2层 | 59.1 | 略浅 |
| **4层** | **60.4** | 最优 |
| 8层 | 57.0 | 解码器过大反而有害 |
| **数据增强** | | |
| + Color Jitter | 56.2 | 显著有害 |
| + Gaussian Blur | 59.6 | 略有害 |
| 无水平翻转 | 60.3 | 几乎无影响 |

### 训练速度

| 方法 | 数据集 | 帧数 | 掩码率 | GFLOPs | 速度提升 |
|------|--------|------|--------|--------|---------|
| SiamMAE | K400 | 2 | 95% | 5.8 | ×1.0 |
| CropMAE | K400 | 1 | 98.5% | 5.6 | **×1.29** |
| CropMAE | IN Subset | 1 | 98.5% | 5.6 | **×23.8** |

用ImageNet训练时速度提升23.8倍的原因：(1) 不需要视频解码，图像加载快得多；(2) 更高掩码率减少token数量，注意力计算二次复杂度下降显著。

### 关键发现

- 物体边界理解能力**不需要显式运动**：CropMAE在ImageNet上训练的注意力图清晰捕获物体边缘，与SiamMAE从视频学到的效果一致
- ImageNet优于K400：归因于ImageNet图像更多样化、更聚焦于物体中心，裁剪产生的代理任务质量更高
- CropMAE在150 epochs即超过SiamMAE 350 epochs的性能，验证了可解代理任务带来的快速收敛优势

## 亮点与洞察

- **核心反直觉发现**：学习物体边界和传播能力不需要视频中的运动信息，静态图像的随机裁剪就足够
- **极端主义的胜利**：98.5%的掩码率（仅2个可见patch）看似疯狂但确实最优，因为代理任务本身足够简单
- 方法的简洁性极强：无需对比学习中的负样本构造、无需momentum encoder、无需精心设计的数据增强
- 将视频预训练问题转化为图像预训练问题，大幅降低数据和计算门槛

## 局限与展望

- 在姿态传播（JHMDB）上逊于SiamMAE，因为随机裁剪无法模拟人体运动的复杂变形
- 模型和数据的可扩展性（更大ViT、更多数据）尚未充分探索
- 未在图像分类（ImageNet linear probing）等主流benchmark上评估，不清楚对非传播任务的效果
- 视频帧相比静态图像的独特贡献仍需深入研究

## 相关工作与启发

- 与SiamMAE的关系：CropMAE是SiamMAE的"简化版"，核心贡献在于证明视频帧对不是必需的
- 与MAE/VideoMAE的关系：通过更高掩码率和双视图裁剪，在传播任务上远超标准MAE
- 与对比学习（SimCLR/BYOL）的联系：随机裁剪也是对比学习的核心增强，但CropMAE不依赖数据增强的精心选择且不存在表征坍塌问题
- 启发：对于特定下游任务，代理任务的"可解性"可能比难度更重要

## 评分

- 新颖性: ⭐⭐⭐⭐ 用图像裁剪替代视频帧的思路简洁但反直觉，98.5%掩码率的发现有启发性
- 实验充分度: ⭐⭐⭐⭐ 三个下游任务+详尽消融+训练速度+注意力可视化，但缺少分类评估
- 写作质量: ⭐⭐⭐⭐ 论证逻辑清晰，"任务可解性"的分析透彻
- 价值: ⭐⭐⭐⭐ 大幅降低自监督预训练的数据和计算门槛，对资源受限场景有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[CVPR 2025\] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](../../CVPR2025/self_supervised/from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)
- [\[CVPR 2026\] In Pursuit of Pixel Supervision for Visual Pre-training](../../CVPR2026/self_supervised/in_pursuit_of_pixel_supervision_for_visual_pre-training.md)
- [\[ECCV 2024\] MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description](marineinst_a_foundation_model_for_marine_image_analysis_with_instance_visual_des.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)

</div>

<!-- RELATED:END -->
