---
title: >-
  [论文解读] LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation
description: >-
  [ECCV 2024][语义分割][半监督学习] 本文提出 LASS3D，在 MeanTeacher 半监督 3D 语义分割框架中引入大语言视觉模型（LVM）生成多层级文本描述来增强 3D 特征，并通过渐进式负学习策略有效利用低置信度伪标签点，在室内外数据集上取得显著提升。 领域现状：3D 点云语义分割需要大量精确的逐点标注…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "半监督学习"
  - "3D点云分割"
  - "大语言视觉模型"
  - "伪标签"
  - "负学习"
---

# LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 3D视觉 / 语义分割  
**关键词**: 半监督学习, 3D点云分割, 大语言视觉模型, 伪标签, 负学习

## 一句话总结

本文提出 LASS3D，在 MeanTeacher 半监督 3D 语义分割框架中引入大语言视觉模型（LVM）生成多层级文本描述来增强 3D 特征，并通过渐进式负学习策略有效利用低置信度伪标签点，在室内外数据集上取得显著提升。

## 研究背景与动机

**领域现状**：3D 点云语义分割需要大量精确的逐点标注，这在大规模数据集上极其耗时。半监督方法（如 MeanTeacher）通过教师-学生框架利用未标注数据的伪标签来缓解标注压力，已成为主流范式。

**现有痛点**：当前半监督 3D 分割方法存在两个核心问题。第一，大语言视觉模型（LVM）在 2D 半监督学习中已展现出强大能力，但在 3D 半监督语义分割中的应用几乎未被探索，3D 特征缺乏高级语义信息的引导。第二，现有方法对教师模型产生的伪标签通常采用硬阈值策略——高置信度的用作正监督，低置信度的直接丢弃，导致大量"不可靠"点中蕴含的信息被浪费。

**核心矛盾**：低置信度伪标签虽然不够可靠无法直接用于正监督，但它们仍然包含"不应该属于哪些类"的排除信息。简单丢弃等于放弃了这些有用的负面知识。同时，3D 点云与 2D 图像之间存在模态鸿沟，如何有效地将语言-视觉模型的语义知识传递到 3D 空间也是关键挑战。

**本文目标** （1）如何将 LVM 的语义信息注入到 3D 半监督分割中？（2）如何有效利用被现有方法丢弃的低置信度不可靠点？

**切入角度**：作者提出以 2D 图像作为桥梁连接文本与点云——利用现成的 LVM 对 2D 图像生成多层级文本描述，再通过自适应融合将文本语义嵌入 3D 特征空间。对于不可靠点，则采用负学习思想：虽不知道该点属于什么类，但可以让模型学习该点"不属于"置信度最高的错误类别。

**核心 idea**：用 LVM 文本描述增强 3D 特征实现语义引导，用渐进式负学习挖掘不可靠点的排除信息来提升半监督分割性能。

## 方法详解

### 整体框架

LASS3D 基于 MeanTeacher 框架构建。输入包含点云和对应的 2D 图像视角。在学生分支中，利用两个现成的 LVM 分别生成场景级和物体级文本描述，通过文本编码器获得多层级语义嵌入。学生网络提取 3D 点特征后，通过语义感知自适应融合模块将文本语义注入 3D 特征。教师分支通过 EMA 更新参数，对未标注数据生成伪标签，并对伪标签进行可靠/不可靠划分——可靠点用正常交叉熵损失训练学生网络，不可靠点用渐进式负学习策略进行补充训练。同时，通过知识蒸馏将学生分支中的文本增强语义传递给教师分支。

### 关键设计

1. **多层级文本描述与语义感知自适应融合模块（SAAF）**:

    - 功能：利用 LVM 生成多层级文本描述并自适应地融入 3D 特征
    - 核心思路：使用两个 LVM 分别生成场景级描述（如"室内办公室场景，有桌椅"）和物体级描述（如"一把木制椅子"），通过预训练文本编码器提取文本嵌入 $e_{scene}$ 和 $e_{obj}$。在融合时，采用类别感知的注意力机制：对于每个 3D 点特征 $f_i$，计算其与各类别文本嵌入的相似度作为注意力权重，然后加权融合文本特征到 3D 特征中。融合比例由可学习的门控参数自适应控制，避免文本信息淹没原始几何信息。
    - 设计动机：多层级描述从宏观场景语义和微观物体属性两个维度补充 3D 特征，弥补纯几何特征缺乏高级语义理解的不足。自适应门控确保不同场景下文本信息的融合程度可调。

2. **渐进式不可靠点利用策略（Progressive Unreliable Data Exploitation）**:

    - 功能：通过负学习逐步利用被传统方法丢弃的低置信度伪标签点
    - 核心思路：首先根据置信度阈值将教师模型预测分为可靠点和不可靠点。对于不可靠点，采用负学习（Negative Learning）——如果教师模型对某点预测的最高置信度类别可能不正确，那么至少可以告诉学生模型"这个点不是该类别"。具体通过互补标签实现：将概率最高的预测类作为负标签，让学生模型降低对该类的预测概率。随着训练推进，阈值逐步放宽，让更多原本不可靠的点参与训练，形成渐进式学习过程。
    - 设计动机：低置信度不意味着完全无用——一个点可能属于10个类中的任何一个，但最有可能的错误预测至少提供了排除信息。渐进策略让模型先从最不可靠的负信号中学习简单的排除，再逐步处理边界模糊的困难样本。

3. **知识蒸馏传递文本语义**:

    - 功能：将学生分支的文本增强特征传递给教师分支
    - 核心思路：教师分支不直接接入 LVM 模块（保持计算效率和 EMA 更新的稳定性），而是通过特征级蒸馏损失让教师分支间接获得文本增强的语义信息。具体计算学生和教师分支中间特征的 KL 散度或 MSE 损失，引导教师特征空间向文本增强的语义空间对齐。
    - 设计动机：如果教师分支也生成更好的语义特征，其产生的伪标签质量也会提升，从而形成良性循环。

### 损失函数 / 训练策略

总损失包含四部分：（1）有标签数据的交叉熵损失 $\mathcal{L}_{ce}$；（2）可靠伪标签的交叉熵损失 $\mathcal{L}_{pseudo}$；（3）不可靠点的负学习损失 $\mathcal{L}_{neg}$，通过互补标签降低错误类别预测概率；（4）知识蒸馏损失 $\mathcal{L}_{kd}$，对齐学生与教师的中间特征。渐进式阈值 $\tau$ 随训练轮次线性下降，逐步纳入更多不可靠点。

## 实验关键数据

### 主实验

| 数据集 | 标注比例 | 指标(mIoU) | LASS3D | 之前SOTA | 提升 |
|--------|---------|-----------|--------|---------|------|
| ScanNet v2 | 1% | mIoU | 55.8 | 51.2 | +4.6 |
| ScanNet v2 | 5% | mIoU | 65.3 | 62.1 | +3.2 |
| SemanticKITTI | 1% | mIoU | 48.7 | 44.9 | +3.8 |
| SemanticKITTI | 10% | mIoU | 58.2 | 55.6 | +2.6 |
| S3DIS | 5% | mIoU | 59.1 | 56.3 | +2.8 |

### 消融实验

| 配置 | mIoU | 说明 |
|------|------|------|
| Baseline (MeanTeacher) | 51.2 | 无文本增强和负学习 |
| + 多层级文本融合 | 53.9 | 加入 SAAF，提升 2.7% |
| + 负学习（固定阈值） | 54.6 | 利用不可靠点，提升 0.7% |
| + 渐进式负学习 | 55.3 | 渐进策略优于固定阈值 |
| + 知识蒸馏 (Full) | 55.8 | 完整模型 |

### 关键发现
- 多层级文本融合贡献最大（+2.7 mIoU），证明 LVM 语义信息对 3D 半监督分割的显著价值
- 渐进式负学习比固定阈值负学习多提 0.7，说明逐步扩展不可靠点范围的策略有效
- 在低标注比例（1%）下提升更显著，说明语义先验在数据匮乏时更关键
- 场景级和物体级文本描述缺一不可，单用场景级降 1.2 mIoU

## 亮点与洞察
- **以 2D 图像为桥梁连接文本与 3D 点云**是一个很实用的思路，避免了直接在 3D 上训练视觉语言模型的高昂成本。类似的桥梁思想可以迁移到 3D 目标检测、点云生成等任务中。
- **负学习挖掘不可靠数据**的思路非常巧妙——不能确认它是什么，但可以学到它不是什么。这种互补标签的思路在半监督/噪声标签的场景下具有普遍适用性。
- 渐进式阈值的设计简单有效，类似于课程学习的思想，先学容易的排除信号再处理困难的边界情况。

## 局限与展望
- 依赖 2D 图像视角生成文本描述，在纯点云（无对应图像）的场景下无法使用
- LVM 生成文本描述的质量直接影响效果，对长尾类别的描述可能不够准确
- 负学习需要合理的阈值调度策略，不同数据集可能需要不同的调度参数
- 未探索更先进的 3D 骨干网络（如 Point Transformer v3），与更强的 backbone 配合效果未知

## 相关工作与启发
- **vs LaserMix**: LaserMix 通过混合不同激光扫描区域实现数据增强，是纯数据驱动策略；LASS3D 从语义先验角度切入，两者互补，结合使用可能进一步提升
- **vs ST3D**: ST3D 也用自训练框架做 3D 半监督，但完全丢弃低置信度预测；LASS3D 的负学习策略是对自训练范式的有效补充
- **vs OpenScene**: OpenScene 也用 CLIP 做 3D-语言关联，但目标是开放词汇分割而非半监督学习；LASS3D 可以借鉴 OpenScene 的多尺度特征对齐策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 LVM 引入半监督 3D 分割有新意，负学习思路在该场景中应用巧妙
- 实验充分度: ⭐⭐⭐⭐ 涵盖室内外多个数据集、多种标注比例，消融全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述系统化
- 价值: ⭐⭐⭐⭐ 为 3D 半监督学习提供了语义增强和数据利用两个有效的新工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation](learning_from_the_web_language_drives_weakly-supervised_incremental_learning_for.md)
- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](../../AAAI2026/segmentation/s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[CVPR 2026\] From Softmax to Dirichlet: Evidential Learning for Semi-supervised Semantic Segmentation](../../CVPR2026/segmentation/from_softmax_to_dirichlet_evidential_learning_for_semi-supervised_semantic_segme.md)
- [\[ICCV 2025\] ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction](../../ICCV2025/segmentation/conformalsam_unlocking_the_potential_of_foundational_segmentation_models_in_semi.md)

</div>

<!-- RELATED:END -->
