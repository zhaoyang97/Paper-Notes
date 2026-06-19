---
title: >-
  [论文解读] VideoGEM: Training-Free Action Grounding in Videos
description: >-
  [CVPR 2025][视频理解][动作定位] VideoGEM 提出了首个基于预训练图像/视频语言模型的免训练空间动作定位方法，通过层权重加权和提示分解策略，在四个动作定位数据集上超越了现有需要训练的方法。 领域现状：视觉语言基础模型（如 CLIP）在零样本定位任务上展现了强大的能力，但主要集中在图像中的物体定位…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "动作定位"
  - "视觉语言模型"
  - "免训练"
  - "注意力机制"
  - "提示分解"
---

# VideoGEM: Training-Free Action Grounding in Videos

**会议**: CVPR 2025  
**arXiv**: [2503.20348](https://arxiv.org/abs/2503.20348)  
**代码**: [https://github.com/felixVogel02/VideoGEM](https://github.com/felixVogel02/VideoGEM)  
**领域**: 视频理解  
**关键词**: 动作定位, 视觉语言模型, 免训练, 注意力机制, 提示分解

## 一句话总结

VideoGEM 提出了首个基于预训练图像/视频语言模型的免训练空间动作定位方法，通过层权重加权和提示分解策略，在四个动作定位数据集上超越了现有需要训练的方法。

## 研究背景与动机

**领域现状**：视觉语言基础模型（如 CLIP）在零样本定位任务上展现了强大的能力，但主要集中在图像中的物体定位。将这些能力拓展到视频中的动作和事件定位面临巨大挑战，因为动作缺乏清晰的物理边界，通常由更高层次的语义概念描述。

**现有痛点**：当前的空间视频定位方法（如 CoMMA、WWW-CLIP）仍然需要专门的训练——要么用定位损失进行微调，要么在大规模视频文本对上训练。另一方面，虽然 GEM 等免训练方法在图像中的物体定位上表现良好，但动作定位需要模型捕捉超越物体边界的上下文信息。

**核心矛盾**：视觉语言模型存在强烈的物体偏置（object bias），当用动词-物体组合进行提示时，模型倾向于定位物体而非动作本身。此外，动作等高层语义概念通常在模型的较高层才涌现，而 GEM 对所有层赋予相同权重。

**本文目标**：设计一种免训练方法，在不改变任何预训练权重的前提下，让视觉语言模型能够在视频中进行空间动作定位。

**切入角度**：作者观察到高层语义概念（如动作）主要在 ViT 的较高层涌现，因此应给予这些层更高的权重；同时动作描述天然包含动词和物体两个独立组件，应该分别处理。

**核心 idea**：将 GEM 的 self-self attention 拓展到视频输入，并通过静态+动态层权重以及提示分解来捕捉高层语义动作概念。

## 方法详解

### 整体框架

VideoGEM 的输入是一段视频和对应的动作描述文本。整个流水线包含三个核心组件：(1) 将 GEM 的 self-self attention 扩展到视频帧处理，(2) 加权 GEM 层以优先考虑高层语义，(3) 将动作提示分解为动词、物体和动作三个子提示，分别计算热力图后加权合并得到最终定位。

### 关键设计

1. **视频 Self-Self Attention 扩展**:

    - 功能：将 GEM 机制从图像扩展到视频输入，支持跨帧的时空注意力
    - 核心思路：给定 $T$ 帧视频，每帧被划分为 $N$ 个 patch，得到 $T \times N$ 个 token。self-self attention 在所有帧的 token 上联合计算，自动聚合空间和时间信息。最终通过计算每个 patch token 与文本 embedding 的余弦相似度生成热力图
    - 设计动机：直接处理多帧允许视频 backbone（如 ViCLIP）自然捕捉时序上下文，而非逐帧独立处理

2. **静态+动态层权重**:

    - 功能：自适应地给不同 Transformer 层分配权重，优先考虑捕获高层语义的层
    - 核心思路：静态权重 $w_s^l$ 随层数单调递增，给高层以更高固定权重。动态权重 $w_d^l$ 通过评估移除某层后 CLS token 与文本对齐的变化来确定——移除后相似度下降最大的层最重要。两者通过 $w_c^l = w_s^l - 1/D + w_d^l$ 组合，确保权重总和不变
    - 设计动机：分析发现动作、动词等抽象概念在模型高层才涌现，均匀权重浪费了低层不相关的信息。动态权重进一步根据具体提示调整，因为不同概念可能在不同层上有不同程度的表征

3. **提示分解 (Prompt Decomposition)**:

    - 功能：将动作描述拆分为动词、物体和完整动作三个独立提示，分别定位后合并
    - 核心思路：提取动作描述中的动词和物体，分别生成格式化的提示文本（如 "A photo of a person [verb]ing"），各自计算热力图得到中心点预测 $c_{verb}$、$c_{obj}$、$c_{act}$，最终预测为加权平均 $c_{dec} = 0.2 \cdot c_{verb} + 0.2 \cdot c_{obj} + 0.6 \cdot c_{act}$
    - 设计动机：视觉语言模型存在物体偏置——直接用动词-物体组合提示时，模型倾向于只关注物体区域。分别处理可以让动词热力图聚焦于手等执行动作的部位，物体热力图聚焦于被操作的物体，两者互补修正最终定位

### 损失函数 / 训练策略

本方法完全免训练，不涉及损失函数或训练过程。所有操作都在推理时进行——利用预训练 backbone 的现有权重，通过 self-self attention 的平行路径和加权策略直接生成定位结果。

## 实验关键数据

### 主实验

| 方法 | 是否需要训练 | V-HICO | Daly | YC | gYT | 平均 |
|------|------------|--------|------|------|------|------|
| WWW-CLIP (CLIP*) | 是 | 62.34 | 71.35 | 58.35 | 56.98 | 62.26 |
| GEM (ViCLIP) | 否 | 65.08 | 73.75 | 53.62 | 51.28 | 60.93 |
| VideoGEM (CLIP) | 否 | 76.90 | 84.53 | 52.57 | 47.46 | 65.37 |
| VideoGEM (OpenCLIP) | 否 | 76.42 | 80.32 | 60.05 | 45.33 | 65.53 |
| VideoGEM (ViCLIP) | 否 | 75.75 | 78.25 | 55.10 | **57.21** | **66.58** |

### 消融实验

| 配置 (ViCLIP) | V-HICO | Daly | gYT | 平均 |
|--------------|--------|------|------|------|
| 无权重 | 74.79 | 76.84 | 56.39 | 65.60 |
| 仅动态权重 | 74.49 | 76.85 | 56.47 | 65.61 |
| 仅静态权重 | 76.18 | 78.38 | 56.75 | 66.58 |
| 静态+动态权重 | 75.75 | 78.25 | 57.21 | 66.58 |

### 关键发现

- VideoGEM 在所有 backbone 上平均精度超过最佳训练方法 3% 以上，且完全不需要训练
- 图像 backbone（CLIP/OpenCLIP）在物体导向的数据集（V-HICO、Daly）上表现更好，而视频 backbone（ViCLIP）在动作导向的 GroundingYouTube 上显著优于其他
- 层重要性分析显示：移除最后几层对精度影响最大，但完全去除低层也会降低性能——支持了"高层更重要但低层不可或缺"的设计理念
- 动态权重在 OpenCLIP 上效果更明显（GroundingYouTube 提升 3%+），因为 ViCLIP 的 CLS token 主要在最后一层成型

## 亮点与洞察

- **免训练超越有训练方法**是本文最大亮点。通过精巧地操纵预训练模型内部的注意力机制（而非改变权重），就能实现强大的动作定位，说明基础模型中已经编码了足够的空间语义信息
- **提示分解策略**可泛化到其他需要定位复杂语义概念的任务，如场景图定位、关系理解等——关键思路是将复合概念拆分为原子单元分别定位再组合
- **动态层权重机制**提供了一种通用方法来评估每一层对特定概念的贡献，可迁移到解释性分析和特征选择领域

## 局限与展望

- 在 YouCook-Interactions 上 VideoGEM 的提升有限，甚至低于训练方法，可能因为烹饪场景需要领域特定知识
- 提示分解依赖 NLP 工具提取动词和物体，对于复杂自然语言描述可能出错
- 层权重参数（$K$、$D$、静态权重值）需要手动调节，未来可以探索自适应机制
- 未讨论时序动作定位（temporal grounding），仅限空间定位

## 相关工作与启发

- **vs GEM**: GEM 只做图像物体定位，均匀权重。VideoGEM 扩展到视频并加入层加权和提示分解，在动作定位上大幅提升
- **vs WWW-CLIP**: WWW-CLIP 需要在 HT100M 上训练且微调 backbone，VideoGEM 免训练且效果更好——说明好的推理策略可以弥补训练数据的缺失
- **vs CoMMA**: CoMMA 用多层跨模态注意力需要专门训练，VideoGEM 只利用现有注意力层的 self-self attention 变体

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个免训练视频动作定位方法，但核心机制是 GEM 的增量扩展
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、三个 backbone、完整消融实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述合理
- 价值: ⭐⭐⭐⭐ 免训练方法的实用价值高，但需关注其在更多场景下的泛化性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)
- [\[CVPR 2025\] Number it: Temporal Grounding Videos like Flipping Manga](number_it_temporal_grounding_videos_like_flipping_manga.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[CVPR 2026\] Memory Matters: Boosting Training-Free Zero-Shot Temporal Action Localization with a Learnable Lookup Table](../../CVPR2026/video_understanding/memory_matters_boosting_training-free_zero-shot_temporal_action_localization_wit.md)
- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](../../ICCV2025/video_understanding/an_empirical_study_of_autoregressive_pre-training_from_videos.md)

</div>

<!-- RELATED:END -->
