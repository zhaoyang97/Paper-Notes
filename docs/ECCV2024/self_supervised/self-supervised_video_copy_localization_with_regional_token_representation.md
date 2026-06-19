---
title: >-
  [论文解读] Self-supervised Video Copy Localization with Regional Token Representation
description: >-
  [ECCV 2024][自监督学习][视频拷贝定位] 提出了一种自监督视频拷贝定位框架，通过在 Vision Transformer 中引入 Regional Token 捕获局部区域信息，并利用传递性（Transitivity Property）自动生成训练数据，在无需人工标注的情况下超越了有监督方法的性能。
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "视频拷贝定位"
  - "Regional Token"
  - "Transformer"
  - "传递性"
---

# Self-supervised Video Copy Localization with Regional Token Representation

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: 视频拷贝定位, 自监督学习, Regional Token, Vision Transformer, 传递性

## 一句话总结
提出了一种自监督视频拷贝定位框架，通过在 Vision Transformer 中引入 Regional Token 捕获局部区域信息，并利用传递性（Transitivity Property）自动生成训练数据，在无需人工标注的情况下超越了有监督方法的性能。

## 研究背景与动机

**领域现状**：视频拷贝定位（Video Copy Localization）旨在给定一对未裁剪视频，找出所有拷贝片段的起止时间戳。这在版权保护、内容审核、视频溯源等场景中有重要应用。当前主流方法通常提取帧级特征，构建帧到帧的相似度图（similarity map），然后训练检测器在相似度图中识别拷贝模式。

**现有痛点**：(1) 帧级特征通常是单一的全局表示，无法捕获局部信息，在"画中画"（picture-in-picture）等常见视频拷贝编辑场景中表现不佳——拷贝内容可能只占据画面的一小部分；(2) 检测器的训练需要大量人工标注数据（标注拷贝视频对及其时间戳），获取成本极高且耗时。

**核心矛盾**：视频拷贝检测需要同时具备全局感知（判断整帧是否为拷贝）和局部感知（在画中画等场景中定位拷贝区域），但现有方法只使用全局特征。同时，有监督学习对标注数据的依赖严重限制了方法的可扩展性。

**本文目标** (1) 如何在帧级特征中引入局部区域信息以应对复杂的视频拷贝编辑？(2) 如何消除对人工标注数据的依赖？

**切入角度**：作者提出两个关键洞察——首先，Vision Transformer 的 patch token 天然包含空间位置信息，可以学习关注特定局部区域；其次，视频拷贝的传递性（如果 A 是 B 的拷贝，B 是 C 的拷贝，那么 A 和 C 之间也存在拷贝关系）可以用来自动构造训练数据，无需人工标注。

**核心 idea**：用 Regional Token 扩展 ViT 以学习局部区域特征，用传递性自动生成拷贝视频对进行自监督训练，实现无标注的视频拷贝定位。

## 方法详解

### 整体框架
输入是一对待检测的视频，输出是所有拷贝片段的起止时间戳。框架分为三个阶段：(1) 特征提取——用带有 Regional Token 的 ViT 提取每帧的全局和局部特征；(2) 相似度图构建——计算两个视频所有帧对之间的特征相似度矩阵；(3) 拷贝检测——在相似度图上用自监督训练的检测器识别拷贝片段的对角线模式。

### 关键设计

1. **Regional Token 表示 (Regional Token Representation)**:

    - 功能：在 ViT 中引入额外的可学习 token，使其学习关注帧内的特定局部区域，增强对画中画等编辑场景的鲁棒性
    - 核心思路：在标准 ViT 的输入序列中，除了 CLS token 和 patch tokens，额外添加若干 Regional Tokens。这些 Regional Tokens 通过 self-attention 与所有 patch tokens 交互，在训练过程中自然学会关注不同的空间区域。采用不对称训练策略（asymmetric training）——教师网络使用完整图像生成特征，学生网络使用局部裁剪区域，Regional Token 被训练为在全局和局部视图之间保持一致。最终的帧级表示是 CLS token（全局）和 Regional Tokens（局部）的聚合
    - 设计动机：传统的 CLS token 聚合了所有 patch 的信息，是一个纯粹的全局表示。在画中画场景中，拷贝内容可能只占画面的 20%，全局特征的相似度会被大量无关区域稀释。Regional Tokens 学习聚焦于有意义的局部区域，在计算相似度时能有效匹配局部拷贝内容

2. **基于传递性的自监督数据生成 (Transitivity-based Self-supervised Data Generation)**:

    - 功能：利用视频拷贝的传递性自动生成带时间戳标注的拷贝视频对，完全消除对人工标注的需求
    - 核心思路：给定一个源视频 A，首先通过随机时序裁剪、速度变换等操作生成拷贝视频 B，并记录 A-B 之间的对应时间戳。然后对 B 再次进行不同的拷贝操作生成 C，记录 B-C 的对应时间戳。根据传递性，可以精确推导出 A-C 之间的拷贝时间戳关系。这样就自动生成了一个具有精确时间戳标注的训练三元组。通过对时序和空间维度施加多种数据增强（速度变换、画中画叠加、颜色抖动、模糊等），可以生成大量多样化的训练数据
    - 设计动机：现有的视频拷贝检测数据集（如 VCSL）虽然提供人工标注，但标注成本极高，数据规模有限。传递性是视频拷贝的固有数学性质，利用它可以无限制地生成训练数据。更重要的是，这些自动生成的数据覆盖了各种拷贝编辑类型的组合，训练出的检测器泛化性更强

3. **相似度图检测器 (Similarity Map Detector)**:

    - 功能：在帧到帧的相似度图中识别拷贝片段对应的对角线模式
    - 核心思路：首先计算两个视频所有帧对之间的余弦相似度，构建二维相似度图。拷贝片段在相似度图中表现为对角线方向的高相似度条纹。使用一个轻量级的 CNN 检测器在相似度图上滑动窗口检测这些对角线模式，输出拷贝片段的起止时间戳。检测器完全使用传递性策略生成的数据训练，无需任何人工标注
    - 设计动机：相似度图将视频拷贝定位问题转化为 2D 图像中的模式检测问题，可以利用成熟的目标检测技术。自监督生成的训练数据为检测器提供了足够多样化的模式样本

### 损失函数 / 训练策略
训练分两阶段：(1) Regional Token 的自监督预训练，使用 DINO 风格的不对称蒸馏损失——学生网络的 Regional Token 特征需要匹配教师网络的 CLS token 特征；(2) 检测器的训练，使用标准的二元交叉熵损失在传递性生成的相似度图上训练。

## 实验关键数据

### 主实验

**VCSL 数据集 (Video Copy Segment Localization)**:

| 方法 | 监督类型 | F1↑ | Precision↑ | Recall↑ |
|------|---------|-----|-----------|---------|
| TN+DTW | 有监督 | 较低 | 中等 | 较低 |
| ViT-CLS | 有监督特征 | 中等 | 中等 | 中等 |
| 本文方法 (无标注) | **自监督** | **最高** | **最高** | **最高** |

### 消融实验

| 配置 | F1↑ | 说明 |
|------|-----|------|
| Full model | 最优 | Regional Token + 传递性训练 |
| w/o Regional Token | 降低 3-5% | 仅用 CLS token，画中画场景表现差 |
| w/o Transitivity | 降低更多 | 使用简单的增强数据替代 |
| CLS-only + 有监督 | 低于本文方法 | 有监督但缺少局部特征 |

### 关键发现
- Regional Token 在画中画等局部拷贝场景中贡献最大——这类场景中全局特征几乎无法检测到拷贝关系，而 Regional Token 能准确匹配局部区域
- 传递性数据生成策略是自监督框架的关键：它不仅提供了大量训练数据，更重要的是覆盖了各种复杂的拷贝编辑组合，使检测器具有更强的泛化能力
- 方法在无需任何人工标注的情况下超越了使用人工标注数据的有监督方法，证明了数据多样性比标注质量更重要

## 亮点与洞察
- **传递性的利用极为巧妙**：这是视频拷贝问题的固有数学特性，但之前没有人将其用于自动生成带精确标注的训练数据。这个思路可以推广到其他具有传递性的匹配问题——如图像检索中的硬负样本挖掘、多跳推理等
- **Regional Token 的设计优雅而简洁**：不修改 ViT 的核心架构，仅添加几个可学习 token，通过 self-attention 的天然机制就能学到局部区域感知能力。这种设计可以直接迁移到其他需要局部感知的视觉检索任务
- 整体框架完全自监督，摆脱了对昂贵人工标注的依赖，具有很强的实际部署价值

## 局限与展望
- Regional Token 的数量是一个需要手动调节的超参数，太多会增加计算开销，太少可能无法覆盖复杂的局部编辑
- 传递性策略假设拷贝操作是确定性的，但实际的拷贝链可能涉及有损压缩等不可逆操作，累积误差可能影响标注精度
- 相似度图的尺度与视频长度成正比，对长视频的计算和存储开销较大
- 可以考虑将 Regional Token 扩展为可变数量，根据视频内容自适应地增减局部关注点

## 相关工作与启发
- **vs TransNet/ViSiL**: 这些方法使用标准的全局帧特征，在画中画场景中表现不佳。本文的 Regional Token 直接解决了局部感知的问题
- **vs VCSL benchmark**: 该基准测试的有监督方法依赖昂贵的人工标注，本文证明自监督方案可以超越它们
- **vs DINO/DINOv2**: 本文的 Regional Token 可以看作是对 DINO 自监督 ViT 的一种任务特定扩展——保留了 DINO 的自蒸馏框架，同时引入了面向视频拷贝定位的局部感知能力

## 评分
- 新颖性: ⭐⭐⭐⭐ Regional Token 和传递性数据生成都是新颖有效的设计
- 实验充分度: ⭐⭐⭐⭐ 在标准benchmark上超越有监督方法，消融实验验证了各组件贡献
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，两个关键挑战的解决方案对应明确
- 价值: ⭐⭐⭐⭐ 完全自监督的视频拷贝定位具有很高的实用价值，在版权保护领域可直接应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[CVPR 2026\] Progressive Mask Distillation for Self-supervised Video Representation](../../CVPR2026/self_supervised/progressive_mask_distillation_for_self-supervised_video_representation.md)
- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[ECCV 2024\] SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)
- [\[CVPR 2025\] AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing](../../CVPR2025/self_supervised/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)

</div>

<!-- RELATED:END -->
