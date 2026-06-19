---
title: >-
  [论文解读] SocialGesture: Delving into Multi-Person Gesture Understanding
description: >-
  [CVPR 2025][人体理解][多人手势识别] SocialGesture 是首个专注于多人社交场景下指示性手势（pointing/showing/giving/reaching）的大规模数据集，涵盖 9889 个视频片段和 42533 个手势实例，同时提出了时序定位、分类识别和 VQA 三类基准任务，系统揭示了当前模型在多人手势理解上的严重不足。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "多人手势识别"
  - "社交手势数据集"
  - "指示性手势"
  - "视频理解"
  - "视觉问答VQA"
---

# SocialGesture: Delving into Multi-Person Gesture Understanding

**会议**: CVPR 2025  
**arXiv**: [2504.02244](https://arxiv.org/abs/2504.02244)  
**代码**: [huggingface.co/datasets/IrohXu/SocialGesture](https://huggingface.co/datasets/IrohXu/SocialGesture)  
**领域**: 人体理解  
**关键词**: 多人手势识别, 社交手势数据集, 指示性手势, 视频理解, 视觉问答VQA

## 一句话总结

SocialGesture 是首个专注于多人社交场景下指示性手势（pointing/showing/giving/reaching）的大规模数据集，涵盖 9889 个视频片段和 42533 个手势实例，同时提出了时序定位、分类识别和 VQA 三类基准任务，系统揭示了当前模型在多人手势理解上的严重不足。

## 研究背景与动机

**领域现状**：手势识别是人体行为理解的重要分支。现有数据集（如 Jester、EgoGesture、HaGRID、LD-ConGR 等）主要关注单人场景下的设备控制手势或手语识别，或在受控环境中采集，缺乏真实社交互动中的自然手势。

**现有痛点**：（1）已有数据集几乎全部是单人场景，无法捕捉人与人之间通过手势进行的社交通信；（2）手势类别偏向HCI交互（如挥手、翘拇指等），忽略了社交通信中最核心的指示性手势（deictic gestures）；（3）缺乏手势发起者与目标的关系标注，无法研究手势的社交语义；（4）没有将手势与语言模态对齐，限制了 VLM 在手势理解上的发展。

**核心矛盾**：现实社交通信中手势和语言是共同起源于统一认知系统的，但当前研究将手势孤立在单人、受控、HCI导向的框架中，与真实社交场景严重脱节。

**本文目标**：（1）构建首个多人社交手势数据集；（2）提供多层次标注（手势类型、时空定位、人际关系、VQA）；（3）建立全面的基准实验，暴露现有模型的不足。

**切入角度**：作者从手势研究的认知科学理论出发——指示性手势（pointing、showing、giving、reaching）是人类建立共同注意力和促进社交互动的最基本手势类型——聚焦于这四类最重要的社交手势。

**核心 idea**：构建一个大规模、多人、自然场景的指示性手势数据集，配合全方位标注和多任务基准，推动多模态社交理解研究。

## 方法详解

### 整体框架

SocialGesture 的构建流程：（1）从 YouTube 和 Ego4D 采集包含多人互动的视频，涵盖社交游戏（44.51%）、综艺娱乐（22.31%）、Ego4D（21.91%）等多种场景；（2）视频预处理为 720p/30FPS 后降采样至 360p/5FPS 用于标注；（3）对四类指示性手势进行时间段标注、关键帧标注、发起者和目标的空间框标注以及自然语言描述标注；（4）基于标注设计三大类基准任务。

### 关键设计

1. **四类指示性手势定义与标注体系**:

    - 功能：提供清晰、可操作的手势分类标准，支撑高质量标注
    - 核心思路：基于 McNeill 的手势理论，将指示性手势细分为四类——**Pointing**（用手指引导他人注意力到特定目标）、**Showing**（展示物体给他人看）、**Giving**（有转移物体意图的动作）、**Reaching**（伸手获取物体的意图）。每类手势的关键区分在于"意图"而非"动作形态"，例如 Pointing 的核心是引导注意力而非手指伸展的具体方式。标注包括时间段、关键帧、发起者 bbox、目标 bbox（人或物）以及社交关系描述。
    - 设计动机：之前的手势数据集用"动作形态"定义类别（如五指张开、OK手势），但社交手势的核心是"意图"，因此需要基于意图的分类体系

2. **多层次基准任务设计**:

    - 功能：从不同难度和角度评估模型在多人手势理解上的能力
    - 核心思路：设计了三大类任务：（a）**时序定位**（Task 1）——在长视频中定位所有手势出现的时间段并分类，用 mAP@IoU 评估；（b）**手势识别**（Task 2-1 二分类 + Task 2-2 四分类）——在短视频片段中判断是否有手势以及手势类型；（c）**VQA**（Task 3-1/3-2/3-3）——全局感知（场景描述、人数统计）、手势理解（检测与分类）、手势定位（空间定位发起者和目标），用于评估 VLM。
    - 设计动机：仅有分类任务不足以全面评估社交手势理解能力。时序定位考验检测能力，分类考验识别能力，VQA 考验推理和多模态对齐能力。

3. **数据多样性与质量控制**:

    - 功能：确保数据集在场景、人群、手势类型上的覆盖度
    - 核心思路：视频选取标准为高清画质、2-10 人场景、2-30 分钟时长、场景多样（种族、性别、年龄）。数据来源包括 YouTube 多种频道类型（社交游戏、综艺、教育、产品评测、聚餐、烹饪）和 Ego4D。针对类别不平衡（pointing 远多于其他三类），在训练集进行重采样。
    - 设计动机：真实世界的手势自然不平衡且场景多样，需在数据采集和训练策略上同时处理

### 损失函数 / 训练策略

各基准任务使用标准的训练策略：时序定位用 ActionFormer；视频识别用各种预训练视频模型微调；VQA用各类 VLM 零样本或微调评估。统一 batch size 16，学习率 5e-4，标准数据增强。

## 实验关键数据

### 主实验

时序定位（ActionFormer + 不同特征提取器）：

| 特征提取器 | mAP@0.3 | mAP@0.5 | mAP@0.7 | Avg mAP |
|-----------|---------|---------|---------|---------|
| I3D | 24.85 | 9.31 | 0.96 | 10.73 |
| R(2+1)D | 14.38 | 7.23 | 1.77 | 7.29 |
| VideoMAEV2 | 27.23 | 13.33 | 2.76 | 14.73 |

手势 vs 非手势二分类：

| 模型 | 预训练 | 参数量 | Accuracy |
|------|--------|--------|----------|
| SlowFast-R50 | K400 | 35M | 80.82% |
| MViTv2-B | K400 | 51M | 83.29% |
| UniFormerV2-B/16 | CLIP | 115M | **84.43%** |

### 消融实验

四分类手势识别（全帧 vs 裁剪发起者区域）：

| 模型 | 全帧 Top1 | 裁剪区域 Top1 | 说明 |
|------|----------|-------------|------|
| TSN-R50 | 54.83% | 55.06% | CNN baseline |
| VideoSwin-L | **56.18%** | 54.94% | 全帧最佳 |
| UniFormerV2-B/16 | 53.37% | **64.72%** | 裁剪后显著提升 |

时序定位中滑动窗口步长的影响：

| Stride | Avg mAP |
|--------|---------|
| 16 | 5.94 |
| 8 | 10.73 |
| 4 | 19.19 |

### 关键发现

- 所有模型在四分类任务上表现极差（最高仅 56.18%/64.72%），说明社交手势识别对现有模型而言极具挑战
- 裁剪发起者区域后 UniFormerV2 提升了 11+ 个百分点，说明多人场景中背景干扰严重
- 时序定位的 avg mAP 仅 14.73（VideoMAEV2），远低于这些特征在 THUMOS/ActivityNet 上的表现——多人场景中手势细粒度且微妙
- 减小滑窗步长可显著提升定位精度（从 5.94 到 19.19），但整体结果仍不够理想
- 特征提取器都是在缺乏多人交互的数据上预训练的，导致特征与多人手势任务对齐不良

## 亮点与洞察

- **首个多人手势数据集**：填补了社交手势研究的关键空白。之前所有数据集都是单人场景，SocialGesture 首次引入人际关系维度，这对开发真正理解社交上下文的 AI 系统至关重要。
- **基于意图而非形态的分类**：四类指示性手势的定义基于交际意图（引导注意力 vs 展示 vs 转移 vs 获取），而非手指/手掌的具体形态，更符合人类社交通信的本质。这种定义方式可迁移到其他人类行为理解任务。
- **多层次任务设计巧妙**：从检测→识别→推理的递进式基准设计，不仅评估了传统视频模型也评估了 VLM，全面暴露了不同维度的短板。

## 局限与展望

- Pointing 手势占比过高导致类别严重不平衡，可能影响模型学习其他三类手势的能力
- 数据来源以 YouTube 为主，存在选择偏差（多为娱乐/游戏场景），缺少工作场所、课堂等更日常的社交场景
- 仅关注指示性手势，未覆盖节拍手势（beat）、图像手势（iconic）和隐喻手势（metaphoric）
- VQA 标注部分借助了 GPT-4o 生成，可能引入分布偏差
- 未来可以引入音频/语音模态，研究手势-语音的联合建模

## 相关工作与启发

- **vs HaGRID**：HaGRID 聚焦单人高分辨率手势的 18 类 HCI 手势，SocialGesture 聚焦多人社交场景的 4 类指示性手势，两者互补但场景假设完全不同
- **vs LD-ConGR**：LD-ConGR 关注远距离单人手势（鲁棒性），SocialGesture 关注近距离多人手势（社交语义），目标不同
- **vs Ego4D**：Ego4D 提供了第一人称视角的手-物交互，SocialGesture 复用了部分 Ego4D 数据但重新标注了社交手势关系，是从自我中心到社交中心的视角转换

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多人社交手势数据集，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 覆盖了定位、识别、VQA三类任务和大量baseline，消融也比较充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，手势分类的理论依据讲得很好
- 价值: ⭐⭐⭐⭐ 对多模态社交理解领域有重要推动作用，但实际方法创新有限（主要贡献是数据集）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)
- [\[ICML 2025\] LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](../../ICML2025/human_understanding/llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)
- [\[CVPR 2026\] DyaDiT: A Multi-Modal Diffusion Transformer for Socially Favorable Dyadic Gesture Generation](../../CVPR2026/human_understanding/dyadit_a_multi-modal_diffusion_transformer_for_socially_favorable_dyadic_gesture.md)
- [\[CVPR 2026\] MAMMA: Markerless Accurate Multi-person Motion Acquisition](../../CVPR2026/human_understanding/mamma_markerless_accurate_multi-person_motion_acquisition.md)
- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](../../AAAI2026/human_understanding/spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)

</div>

<!-- RELATED:END -->
