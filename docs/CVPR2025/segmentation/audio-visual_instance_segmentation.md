---
title: "Audio-Visual Instance Segmentation"
authors: "Ruohao Guo, Xianghua Ying, Yaru Chen, Dantong Niu, Guangyao Li, Liao Qu, Yanyu Qi, Bowei Xing, Wenzhen Yue, Ji Shi, Qixiang Ye, Jian Yang"
affiliations: "Peking University, University of Surrey, UC Berkeley"
venue: "CVPR 2025"
date: 2023-10-27
tags: ["audio-visual", "instance segmentation", "multimodal", "sound source localization", "video segmentation"]
arxiv: "2310.18709"
code: "https://github.com/ruohaoguo/avis"
---

# Audio-Visual Instance Segmentation

## 研究背景与动机

人类在日常生活中自然地将听觉和视觉信息整合，以定位和识别环境中的声源。例如，听到狗叫声时，我们会自动将注意力聚焦到画面中的狗身上，并精确地感知其轮廓。这种**视听联合感知**能力在机器人导航、视频编辑、自动驾驶等领域具有重要应用价值。

现有的视听学习任务主要包括：

| 任务 | 输出 | 粒度 | 时序建模 |
|------|------|------|---------|
| Sound Source Localization | 热力图 | 区域级 | 单帧 |
| Audio-Visual Segmentation | 语义掩码 | 像素级 | 单帧/短序列 |
| Audio-Visual Source Separation | 分离音频 | 源级 | 多帧 |
| **AVIS (本文)** | **实例掩码+ID** | **实例级** | **全视频** |

现有方法的局限性：

**Sound Source Localization** 仅提供粗糙的区域级定位，无法生成精确的像素级分割

**Audio-Visual Segmentation** 进行语义级分割但不区分同类不同个体

**没有方法**在视频级别同时进行实例级分割和跨帧追踪

本文提出 **Audio-Visual Instance Segmentation (AVIS)** 这一新任务：给定视频和对应音频，对所有发声物体进行实例级分割并跨帧追踪。同时构建了 AVISeg 基准数据集和 AVISM 基线方法。

## 方法详解

### AVISeg 基准数据集

AVISeg 是首个用于音视频实例分割的大规模标注数据集：

| 属性 | 数值 |
|------|------|
| 视频数量 | 926 |
| 标注掩码 | 90,000+ |
| 物体类别 | 26 |
| 平均视频长度 | 8.3秒 |
| 分辨率 | 720p/1080p |
| 标注频率 | 每5帧标注 |
| 数据来源 | YouTube, ActivityNet, VGGSound |

#### 数据采集与标注流程

1. **视频筛选**：从大规模视频数据库中筛选包含明确声源的视频
2. **音频事件标注**：标注每段视频中的音频事件时间区间和类别
3. **实例分割标注**：对每个发声物体绘制实例级分割掩码
4. **跨帧关联**：为同一物体在不同帧中的掩码分配一致的实例ID
5. **质量控制**：多轮审核确保标注质量

### AVISM 基线方法

AVISM (Audio-Visual Instance Segmentation and tracking Model) 由两个核心模块组成：

#### 帧级声源定位器 (Frame-Level Sound Localizer, FLSL)

FLSL 负责在每一帧中定位发声物体：

- **音频特征提取**：使用预训练的 AudioSet 模型提取音频嵌入 $\mathbf{a}_t \in \mathbb{R}^{d}$
- **视觉特征提取**：使用 ResNet-50 / Swin-T 提取视觉特征图 $\mathbf{V}_t \in \mathbb{R}^{d \times H \times W}$
- **跨模态注意力**：

$$\mathbf{A}_{t} = \text{softmax}\left(\frac{\mathbf{a}_t \mathbf{V}_t^T}{\sqrt{d}}
\right) \mathbf{V}_t$$

- **实例预测**：基于注意力增强的特征，使用 Mask2Former 风格的解码器预测实例掩码和类别

#### 视频级声源追踪器 (Video-Level Sounding Tracker, VLST)

VLST 负责跨帧的实例关联和追踪：

- **实例嵌入**：为每个检测到的实例生成外观嵌入 $\mathbf{e}_i^t$
- **音频一致性约束**：确保同一实例在不同帧中与对应音频的相关性一致
- **匹配策略**：结合外观相似度和音频相关性的二部图匹配

$$\text{cost}(i, j) = \alpha \cdot \text{IoU}(m_i^t, m_j^{t+1}) + \beta \cdot \cos(\mathbf{e}_i^t, \mathbf{e}_j^{t+1}) + \gamma \cdot \text{audio\_sim}(i, j)$$

### 评估指标

AVIS 任务采用改编自 VIS (Video Instance Segmentation) 的评估指标，并加入音频相关的约束：

| 指标 | 含义 | 说明 |
|------|------|------|
| FSLA (Frame-level Sound Localization Accuracy) | 帧级声源定位准确率 | 衡量每帧中发声物体是否被正确分割 |
| HOTA (Higher Order Tracking Accuracy) | 高阶追踪准确率 | 综合衡量检测和关联质量 |
| mAP | 平均精度 | 标准实例分割指标 |
| IDsw | ID切换次数 | 衡量追踪一致性 |

## 实验结果

### 主实验结果

| 方法 | FSLA↑ | HOTA↑ | mAP↑ | IDsw↓ |
|------|-------|-------|------|-------|
| Mask2Former (仅视觉) | 31.25 | 48.92 | 27.8 | 145 |
| AVS + SORT | 35.67 | 52.34 | 31.2 | 128 |
| TAPIS | 38.91 | 56.18 | 34.5 | 107 |
| **AVISM (本文)** | **42.78** | **61.73** | **38.9** | **82** |

AVISM 在所有指标上均显著优于现有方法的组合方案，验证了专门为AVIS任务设计的架构的优越性。

### 消融实验

| 配置 | FSLA↑ | HOTA↑ |
|------|-------|-------|
| Full AVISM | 42.78 | 61.73 |
| w/o 音频输入 | 32.14 | 49.85 |
| w/o 跨模态注意力 | 37.92 | 55.41 |
| w/o 音频一致性约束 | 40.15 | 58.62 |
| w/o VLST (无追踪) | 42.78 | 52.37 |

音频信息是核心贡献因素（移除后 FSLA 下降 10.64%），VLST 追踪模块对 HOTA 贡献显著。

### 类别分析

| 类别 | FSLA↑ | 难度 |
|------|-------|------|
| 乐器 (piano, guitar) | 52.3 | 易 |
| 动物 (dog, cat, bird) | 43.7 | 中 |
| 车辆 (car, motorcycle) | 38.9 | 中 |
| 人类活动 (speaking, clapping) | 31.2 | 难 |

乐器类别最容易定位，因为声源位置固定且视觉特征显著；人类活动最难，因为声源（人）在画面中可能有多个且运动模式复杂。

## 总结与展望

本文提出了 AVIS 这一新的视听理解任务，构建了 AVISeg 基准（926视频，90K+掩码，26类别），并设计了 AVISM 基线方法（FLSL + VLST）。AVISM 达到 FSLA 42.78%、HOTA 61.73%，为后续研究提供了坚实的基础。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[CVPR 2025\] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](../../ICCV2025/segmentation/implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](../../ICCV2025/segmentation/towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)

</div>

<!-- RELATED:END -->
