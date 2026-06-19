---
title: >-
  [论文解读] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking
description: >-
  [ICCV 2025][视频理解][Open-Vocabulary MOT] 本文提出TRACT，一种利用轨迹级信息增强开放词汇多目标跟踪（OV-MOT）的方法，通过轨迹一致性强化（TCR）改善关联、通过轨迹特征聚合（TFA）和轨迹语义丰富（TSE）改善分类，在OV-TAO基准上显著提升了跟踪性能，尤其是分类准确率。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "Open-Vocabulary MOT"
  - "Trajectory Information"
  - "CLIP"
  - "Association"
  - "Classification"
---

# Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking

**会议**: ICCV 2025  
**arXiv**: [2503.08145](https://arxiv.org/abs/2503.08145)  
**代码**: 无（论文中提到code will be released）  
**领域**: 视频理解 / 多目标跟踪  
**关键词**: Open-Vocabulary MOT, Trajectory Information, CLIP, Association, Classification

## 一句话总结
本文提出TRACT，一种利用轨迹级信息增强开放词汇多目标跟踪（OV-MOT）的方法，通过轨迹一致性强化（TCR）改善关联、通过轨迹特征聚合（TFA）和轨迹语义丰富（TSE）改善分类，在OV-TAO基准上显著提升了跟踪性能，尤其是分类准确率。

## 研究背景与动机
开放词汇多目标跟踪（OV-MOT）旨在跟踪训练时未见过的任意类别目标。现有OV-MOT方法（如OVTrack、MASA）主要关注实例级的检测和关联，忽视了轨迹级信息的利用。然而，轨迹信息在视频跟踪中具有独特且关键的作用：

**关联层面**：开放词汇检测器的不稳定性导致某些帧中检测不准确或遗漏，轨迹信息可以帮助恢复中断的匹配关系，减少身份切换

**分类层面**：视频中频繁的模糊和遮挡导致错误分类，轨迹信息可以从多帧多角度综合判断目标类别

核心矛盾在于：现有方法仅在实例级操作，而轨迹作为视频特有的上下文信息被严重低估。本文从轨迹的双重价值出发，同时增强关联和分类两个环节。

## 方法详解

### 整体框架
TRACT采用两阶段跟踪器架构，基于"tracking-by-detection"范式。第一阶段使用可替换的开放词汇检测器生成检测结果 $\mathcal{R} = \{\mathbf{b}_i, \mathbf{c}_i, \mathbf{f}_i\}_{i=1}^N$。第二阶段分为两步：
- **轨迹增强关联**：使用TCR策略，利用特征库和类别库维持轨迹一致性
- **轨迹辅助分类**：使用TraCLIP模块，通过TFA和TSE策略从视觉和语言两个角度利用轨迹信息进行分类

### 关键设计

1. **轨迹一致性强化（TCR）**:

    - 功能：在关联阶段利用轨迹历史信息增强目标身份和类别的一致性
    - 核心思路：
        - **身份一致性**：为每条活跃轨迹维护轨迹记忆 $\mathbf{f}$ 和特征库 $\bar{\mathbf{f}} = \{f_{i-j}\}_{j=1}^{n_{\text{bank}}}$。轨迹记忆通过指数移动平均更新：$\mathbf{f}_i = \alpha \times f_i + (1-\alpha) \times \mathbf{f}_{i-1}$。相似度计算同时考虑轨迹记忆和特征库：$\mathtt{S}(\mathbf{t}, r) = \alpha \cdot \Psi(f_i, \mathbf{f}) + (1-\alpha) \cdot \frac{1}{n_{\text{bank}}} \sum_{j=1}^{n_{\text{bank}}} \Psi(f_i, f_{i-j})$
        - **类别一致性**：维护类别库 $\bar{\mathbf{c}} = \{c_{i-j}\}_{j=1}^{n_{\text{clip}}}$ 存储历史分类预测。根据检测置信度分层处理：高置信度直接采用当前分类，中等置信度用投票融合当前和历史预测，低置信度仅用历史投票
    - 设计动机：开放词汇检测不稳定，单帧检测容易出错，通过维护跨帧的特征和类别记忆可以平滑噪声

2. **轨迹特征聚合（TFA）**:

    - 功能：将轨迹中多帧的视觉特征聚合为统一的轨迹级特征
    - 核心思路：从轨迹中基于检测置信度采样 $n_{\text{clip}}$ 帧，使用CLIP视觉编码器逐帧提取2D特征 $\dot{\mathbf{f}} \in \mathbb{R}^{n \times d}$，通过自注意力和MLP增强：$\ddot{\mathbf{f}} = \dot{\mathbf{f}} + \mathtt{SA}(\mathtt{LN}(\dot{\mathbf{f}}))$，$\tilde{\mathbf{f}} = \ddot{\mathbf{f}} + \mathtt{MLP}(\mathtt{LN}(\ddot{\mathbf{f}}))$，最后全局平均池化得到轨迹特征 $\mathbf{f}^{traj}$
    - 设计动机：不同帧中目标可能处于不同遮挡、模糊状态，聚合多帧特征可以获得更完整的视觉表征

3. **轨迹语义丰富（TSE）**:

    - 功能：利用LLM将简单的类别名称扩展为包含视觉属性的丰富描述
    - 核心思路：使用ChatGPT为每个类别生成属性描述（如"Provide a brief description of the {category} focusing on two to three visual attributes"），拼接为 $\mathcal{A} = \mathtt{Concat}(\mathcal{V}, \Phi(\mathcal{V}))$。然后用CLIP文本编码器分别提取原始类别特征 $\mathcal{F}^{cate}$ 和属性增强特征 $\mathcal{F}^{attr}$
    - 设计动机：轨迹提供了多角度多光照下的目标信息，仅用类别名称无法充分利用这些丰富线索

### 三元分类选择
对每条轨迹，计算三种分类结果：基于原始类别名称的分类 $\mathbf{v}_{cate}$、基于属性描述的分类 $\mathbf{v}_{attr}$、以及基于检测器预测投票的分类 $\mathbf{v}_{det}$，选择相似度分数最高的作为最终类别。

### 训练策略
TCR模块无需训练。TraCLIP以CLIP ViT-L/14初始化，冻结视觉和语言编码器，仅训练自注意力和MLP模块。训练数据使用LVIS、YouTube-VIS和TAO训练集（仅已知类别），对LVIS图像通过数据增强生成伪轨迹。

## 实验关键数据

### 主实验（OV-TAO验证集，YOLO-World检测器）

| 方法 | Base TETA↑ | Base ClsA↑ | Novel TETA↑ | Novel ClsA↑ |
|------|-----------|-----------|------------|------------|
| MASA | 38.2 | 18.6 | 32.2 | 4.4 |
| **TRACT** | **39.4** | **22.6** | **33.7** | **5.3** |
| DeepSORT | 27.3 | 17.9 | 21.5 | 3.8 |
| OC-SORT | 31.2 | 16.9 | 24.4 | 3.7 |

### 消融实验

| 配置 | TETA | LocA | AssA | ClsA | 说明 |
|------|------|------|------|------|------|
| 基线(无任何策略) | 37.5 | 55.1 | 40.1 | 16.9 | MASA基线 |
| +TCR | 37.6 | 55.0 | 40.6 | 17.3 | 关联一致性提升 |
| +TCR+TFA | 38.5 | 54.9 | 40.5 | 19.9 | 轨迹特征显著提升分类 |
| +TCR+TFA+TSE | 38.6 | 54.9 | 40.6 | 20.3 | 属性描述进一步提升 |

### 关键发现
- ClsA指标提升最为显著（+3.4%），验证了轨迹信息对分类的核心价值
- TSE对Novel类别的分类提升更大（10.9→13.3），说明属性描述对未见类别特别有帮助
- 特征库长度 $n_{\text{bank}}=15$ 最优，且对速度影响很小（1.52→1.59 s/seq）
- 轨迹采样长度 $n_{\text{clip}}=5$ 已足够，更长反而无明显提升但显著降速

## 亮点与洞察
- 将轨迹信息的利用从"附带工具"提升为OV-MOT的核心设计要素
- TraCLIP作为即插即用的轨迹分类模块，可适配不同检测器和关联方法
- 三元分类选择机制巧妙地综合了检测器预测和轨迹级CLIP匹配两种信息源
- 对OV-MOT当前面临的数据和评估协议挑战有深入讨论（检测密度过高、标注不完整等）

## 局限与展望
- 轨迹信息未用于改善定位（LocA无变化），作者讨论了初步尝试但效果有限
- TSE依赖ChatGPT生成属性描述，增加了离线准备成本
- TraCLIP的训练数据仅包含已知类别，可能限制了对完全未见类别的泛化能力
- OV-TAO数据集本身标注不完整、检测密度过高，评估协议可能未完全反映真实场景

## 相关工作与启发
- **vs OVTrack**: OVTrack是OV-MOT的开创性工作但仅关注实例级别，TRACT通过轨迹级信息在ClsA上大幅超越
- **vs MASA**: MASA利用SAM进行实例匹配但忽略轨迹上下文，TRACT在其基础上通过轨迹策略获得一致提升
- **vs SLAack**: SLAack统一早期关联但未利用轨迹分类潜力，TRACT的TraCLIP提供了互补的分类视角

## 评分
- 新颖性: ⭐⭐⭐⭐ 轨迹级信息利用的思路清晰且有效，TraCLIP的即插即用设计实用
- 实验充分度: ⭐⭐⭐⭐ 多检测器、多数据集分组对比，消融全面覆盖各超参数
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义和方法阐述条理分明
- 价值: ⭐⭐⭐⭐ 为OV-MOT引入了有价值的轨迹视角，指出了数据和评估的根本问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[ECCV 2024\] SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking](../../ECCV2024/video_understanding/slack_semantic_location_and_appearance_aware_open-vocabulary_tracking.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)

</div>

<!-- RELATED:END -->
