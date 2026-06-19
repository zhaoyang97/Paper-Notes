---
title: >-
  [论文解读] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos
description: >-
  [CVPR 2025][视角选择] 本文提出 LangView，利用视角无关的文字叙述（narration）作为弱监督信号，通过比较各视角预测 caption 与真实叙述的匹配度来生成最佳视角伪标签，实现无需手动标注的多视角教学视频自动视角选择。 领域现状：多视角教学视频（如多机位拍摄的操作教程）需要选择每个时刻最具信息量的…
tags:
  - "CVPR 2025"
  - "视角选择"
  - "弱监督"
  - "多视角教学视频"
  - "语言引导"
  - "伪标签"
---

# Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos

**会议**: CVPR 2025  
**arXiv**: [2411.08753](https://arxiv.org/abs/2411.08753)  
**代码**: [https://vision.cs.utexas.edu/projects/which-view-shows-it-best](https://vision.cs.utexas.edu/projects/which-view-shows-it-best)  
**领域**: 其他  
**关键词**: 视角选择、弱监督、多视角教学视频、语言引导、伪标签

## 一句话总结
本文提出 LangView，利用视角无关的文字叙述（narration）作为弱监督信号，通过比较各视角预测 caption 与真实叙述的匹配度来生成最佳视角伪标签，实现无需手动标注的多视角教学视频自动视角选择。

## 研究背景与动机

**领域现状**：多视角教学视频（如多机位拍摄的操作教程）需要选择每个时刻最具信息量的视角展示给观众。这在自动电影摄制、教学视频编辑中有重要应用。

**现有痛点**：现有自动视角选择方法要么依赖手工规则（不够泛化），要么需要昂贵的"最佳视角"人工标注来训练（难以规模化）。缺乏标注限制了学习方法在此任务上的应用。

**核心矛盾**：需要大量最佳视角标注来训练选择模型，但标注本身就需要人类逐帧判断每个视角的信息量，既主观又昂贵。

**本文目标**：在完全没有最佳视角标注的前提下训练视角选择模型。

**切入角度**：多视角教学视频通常配有视角无关的文字叙述（如"用双手拆下后轮"）。核心假设：如果某个视角能更准确地预测这段叙述，说明它更好地展示了活动内容，因此是更好的视角。

**核心 idea**：用各视角的 caption 预测准确度作为视角质量的代理指标，自动生成最佳视角伪标签来训练视角选择器，无需任何人工视角标注。

## 方法详解

### 整体框架
两阶段：(1) **伪标签生成**：对每个视角用预训练 video captioner 生成 caption，比较与真实叙述的相似度，最匹配的视角被标为伪最佳视角；(2) **视角选择器训练**：用伪标签训练视角选择模型，并添加辅助任务（相机位姿预测）增强视角敏感性。推理时只需多视角视频输入，无需文字或位姿。

### 关键设计

1. **基于 Caption 准确度的伪标签生成**:

    - 功能：自动为多视角视频生成最佳视角标注
    - 核心思路：对 $N$ 个视角分别用预训练 captioner 预测叙述，计算每个预测与视角无关真实叙述的相似度（如 ROUGE/BERTScore），最高分的视角被标为正样本。
    - 设计动机：视角无关叙述描述了活动的完整信息（如所有涉及的物体和动作），能最好预测全部信息的视角自然是最具信息量的视角。

2. **辅助相机位姿预测任务**:

    - 功能：增强模型对视角差异的感知能力
    - 核心思路：训练选择器同时预测不同视角间的相对相机位姿（如第一人称 vs 第三人称的空间关系）。
    - 设计动机：纯粹的视角选择可能退化为只关注内容而忽略视角差异；位姿预测迫使模型理解视角的几何意义。

3. **训练-推理解耦**:

    - 功能：推理时不依赖文字和位姿
    - 核心思路：训练时文字用于生成伪标签（离线计算），位姿用于辅助任务。推理时选择器只接收多视角视频，直接输出每个时刻的最佳视角。
    - 设计动机：实际应用中最终用户只提供视频，不应要求额外输入。

### 损失函数 / 训练策略
交叉熵损失（视角选择分类）+ 辅助位姿预测损失。

## 实验关键数据

### 主实验
在 Ego-Exo4D 和 LEMMA 两个多视角教学视频数据集上：

| 方法 | 自动指标↑ | 人类评估↑ | 监督类型 |
|------|----------|----------|---------|
| 启发式方法 | 基线 | 基线 | 规则 |
| 全监督 SOTA | 较好 | 较好 | 人工标注 |
| LangView (本文) | 最优 | 最优 | 弱监督（仅叙述文字） |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 随机 caption 匹配 | 差 | 验证 caption 确实携带视角信息 |
| 去掉辅助位姿预测 | 下降 | 位姿感知对选择很重要 |
| 不同 captioner 模型 | 均有效 | 方法不过度依赖特定模型 |

### 关键发现
- 弱监督方法超越了全监督基线——说明语言信号比少量人工标注更丰富
- 辅助位姿预测显著提升性能，验证了视角几何理解的重要性
- 人类评估与自动指标一致，证明方法确实选出了人类偏好的视角

## 亮点与洞察
- **语言作为视觉监督信号**：将"文字描述能力"作为视角信息量的代理，是非常巧妙的弱监督思路，可迁移到其他需要衡量视觉信息量的任务。
- **无标注胜有标注**：弱监督方法超越全监督基线，说明大规模弱信号可以比少量强信号更有效。

## 局限与展望
- 依赖预训练 captioner 的质量，captioner 对特定活动的偏见会传播给选择器
- 当前假设最佳视角是单一的，但有些场景可能需要多视角融合
- 仅在教学视频验证，体育、会议等场景未探索

## 相关工作与启发
- **vs 全监督方法**: 需要昂贵人工标注，LangView 无需
- **vs 时序摘要方法**: 视频摘要在时间轴上选择关键帧，本文在视角轴上选择最佳相机
- 用语言作为视觉信号质量代理的思路有广泛应用前景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 caption 准确度作为视角质量代理的核心 idea 非常新颖
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+人类评估+详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述优秀，idea 表达清晰
- 价值: ⭐⭐⭐⭐ 对多视角视频编辑有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](../../ICCV2025/others/switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[CVPR 2025\] Three-View Focal Length Recovery From Homographies](three-view_focal_length_recovery_from_homographies.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](../../ICCV2025/others/intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](../../ECCV2024/others/mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)

</div>

<!-- RELATED:END -->
