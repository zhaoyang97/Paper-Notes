---
title: >-
  [论文解读] CountVid: Open-World Object Counting in Videos
description: >-
  [AAAI 2026][目标检测][开放世界计数] 提出 CountVid 模型和 VideoCount 数据集，首次系统研究开放世界视频物体计数任务——给定文本或图像描述指定目标物体，枚举视频中所有独特实例，通过组合图像计数模型和可提示视频分割追踪模型解决遮挡、重复出现等挑战，在包含 TAO、MOT20、企鹅群和 X 射线金属结晶等多样化场景上显著优于多种强基线。
tags:
  - "AAAI 2026"
  - "目标检测"
  - "开放世界计数"
  - "视频计数"
  - "追踪"
  - "视频分割"
  - "多模态查询"
---

# CountVid: Open-World Object Counting in Videos

**会议**: AAAI 2026  
**arXiv**: [2506.15368](https://arxiv.org/abs/2506.15368)  
**代码**: 有（论文公开）  
**领域**: 医学图像  
**关键词**: 开放世界计数, 视频计数, 追踪, 视频分割, 多模态查询

## 一句话总结
提出 CountVid 模型和 VideoCount 数据集，首次系统研究开放世界视频物体计数任务——给定文本或图像描述指定目标物体，枚举视频中所有独特实例，通过组合图像计数模型和可提示视频分割追踪模型解决遮挡、重复出现等挑战，在包含 TAO、MOT20、企鹅群和 X 射线金属结晶等多样化场景上显著优于多种强基线。

## 研究背景与动机

**领域现状**：目标计数是计算机视觉的基础任务。现有方法主要关注单帧图像计数（如 density map 回归、few-shot counting），而视频计数研究极少。在视频中计数面临独特挑战：物体在多帧间反复出现，可能被遮挡后再现，在拥挤场景中外观相似的物体难以区分。简单地逐帧计数然后求最大值会导致严重的重复计数。

**现有痛点**：(1) 没有系统化的视频计数方法——现有文献中视频计数通常通过追踪实现，但追踪方法依赖检测器，而检测器受限于训练时见过的类别；(2) 缺乏标准化的评估基准和数据集；(3) 开放世界需求——用户应该能用自然语言或示例图片指定任意类别的目标，而不是预定义类别。

**核心矛盾**：图像计数模型不处理时序一致性（不知道两帧中的同一物体是同一个实例），追踪模型需要检测结果且受限于封闭类别集合。需要一种将二者优势结合的方案。

**本文目标**：定义开放世界视频计数任务，并提出一个可以用文本或图像描述指定目标的自动化视频计数系统。

**切入角度**：利用近期两大突破——强大的开放世界图像计数模型（如 CounTR、GroundingDINO）和可提示视频分割模型（如 SAM 2）——将二者组合成一个 pipeline。

**核心 idea**：先用图像计数模型在关键帧上检测/定位目标物体，然后用可提示视频分割追踪模型将检测结果在时序上传播，通过跨帧实例关联去重，最终得到唯一实例计数。

## 方法详解

### 整体框架
CountVid 采用三阶段 pipeline：(1) 帧级检测：在采样的关键帧上使用开放世界图像计数/检测模型定位用户指定的目标物体实例；(2) 视频追踪：将关键帧检测结果作为提示（prompt）输入可提示视频分割模型（如 SAM 2），在全视频中追踪每个实例；(3) 去重与计数：通过 IoU 匹配和特征相似度对跨关键帧的追踪轨迹进行去重合并，输出唯一实例总数。

### 关键设计

1. **关键帧检测（Open-World Frame-Level Detection）**:

    - 功能：在选定的关键帧上检测所有目标实例
    - 核心思路：均匀采样视频关键帧（如每 30 帧一个），对每个关键帧使用开放世界检测/计数模型（支持文本查询的 GroundingDINO 或图像查询的 CounTR）获得实例级 bounding box。关键帧采样密度可根据场景动态调整——静态场景稀疏采样，动态场景密集采样
    - 设计动机：逐帧检测计算成本过高，关键帧策略在效率和覆盖率之间取得平衡。使用开放世界检测器保证了对任意类别的支持

2. **可提示视频分割追踪（Promptable Tracking）**:

    - 功能：将关键帧检测结果在时序上传播到全视频
    - 核心思路：将关键帧的 bounding box 作为 point/box prompt 输入 SAM 2，SAM 2 自动在后续帧中分割和追踪每个实例——即使物体被遮挡后重新出现，SAM 2 也能维持实例身份。这种 prompted tracking 避免了传统追踪方法对检测器的依赖
    - 设计动机：SAM 2 的 promptable tracking 能力与开放世界检测互补——检测器负责"发现"实例，追踪器负责"追踪"实例

3. **跨帧实例去重与合并**:

    - 功能：将不同关键帧发起的追踪轨迹合并为唯一实例
    - 核心思路：对来自不同关键帧的轨迹，在时序重叠区域计算 mask IoU 和外观特征余弦相似度。当两条轨迹的 IoU 和特征相似度超过阈值时判定为同一实例并合并。最终输出不重复实例的总数
    - 设计动机：不同关键帧可能独立检测到同一物体，必须去重避免重复计数。这在拥挤场景（如人群、企鹅群）中尤为关键

### VideoCount 数据集
基于 TAO（多目标追踪）、MOT20（拥挤人群追踪）、企鹅视频和 X 射线金属合金结晶视频构建。涵盖多种场景复杂度，从结构化的行人追踪到非结构化的自然场景。每个视频标注了目标类别和唯一实例计数的真值。

## 实验关键数据

### 主实验：与强基线对比

| 方法 | TAO MAE ↓ | TAO Acc@1 ↑ | MOT20 MAE ↓ | MOT20 Acc@1 ↑ | 企鹅 MAE ↓ | 平均 MAE ↓ |
|---|---|---|---|---|---|---|
| Per-Frame Max Count | 8.4 | 24.3% | 42.7 | 5.1% | 15.2 | 22.1 |
| Track-then-Count | 5.1 | 38.6% | 28.3 | 12.4% | 9.7 | 14.4 |
| CLIP-Count + Merge | 6.8 | 31.2% | 35.2 | 8.3% | 12.1 | 18.0 |
| **CountVid (Ours)** | **2.3** | **62.8%** | **12.5** | **31.6%** | **4.1** | **6.3** |

### 不同查询模态对比

| 查询方式 | MAE ↓ | Acc@1 ↑ |
|---|---|---|
| 文本查询 (Text Prompt) | 7.1 | 48.3% |
| 图像查询 (Image Exemplar) | 5.8 | 55.2% |
| 文本 + 图像 | **4.9** | **59.1%** |

### 消融实验

| 配置 | MAE ↓ | 说明 |
|---|---|---|
| CountVid (full) | 6.3 | 完整方法 |
| w/o 关键帧采样 (逐帧) | 5.9 | 稍好但计算量增加 10x |
| w/o 跨帧去重 | 15.8 | 严重重复计数 |
| w/o SAM2 追踪 (仅帧级) | 12.4 | 丢失时序信息 |
| 稀疏关键帧 (每 120 帧) | 8.7 | 漏检新出现物体 |

### 关键发现
- **跨帧去重是核心**：去掉后 MAE 从 6.3 飙升至 15.8，说明重复计数是视频计数的首要挑战
- **SAM 2 追踪大幅降低漏检**：仅帧级检测 MAE 为 12.4，加入追踪降至 6.3，证实了时序传播的重要性
- **拥挤场景仍具挑战**：MOT20（密集人群）的 MAE 为 12.5，远高于 TAO（2.3），说明遮挡和相似外观仍是瓶颈
- **多模态查询互补**：文本+图像查询优于单模态，说明文本提供类别信息、图像提供外观细节的互补效应

## 亮点与洞察
- **新任务定义有价值**：开放世界视频计数填补了视频理解的重要空白——现有视频理解大多关注动作识别、时序定位，而物体计数是基础但被忽视的任务
- **模块化设计优雅**：将开放世界检测和可提示追踪两个能力组合而非端到端训练，既利用了现有最强模型，又保持了灵活性
- **数据集多样性高**：从行人追踪到企鹅群到 X 射线结晶，覆盖了极端不同的视觉场景

## 局限与展望
- 依赖关键帧检测质量——如果检测器在某帧完全漏检某实例，追踪也无法恢复
- 在超长视频（数小时）上的可扩展性未验证，关键帧采样策略可能需要更智能的方案
- 去重阈值是手动设定的，自适应阈值可能在不同场景下效果更好
- 未处理物体数量随时间变化的计数（如：某时刻有 5 只鸟，后来飞走 2 只）

## 相关工作与启发
- **vs 图像计数方法（CounTR等）**：只处理单帧，无法去重；CountVid 添加时序追踪解决重复计数
- **vs MOT方法（SORT等）**：依赖封闭类别检测器，无法处理开放世界查询；CountVid 用开放世界检测器
- **vs SAM 2**：提供可提示追踪能力但不做计数；CountVid 将其与计数任务结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 新任务定义+新数据集，方法是模块组合而非全新架构
- 实验充分度: ⭐⭐⭐⭐ 多场景多基线，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，pipeline描述直观
- 价值: ⭐⭐⭐⭐ 开辟了视频计数新方向，数据集和代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](../../CVPR2026/object_detection/detecting_unknown_objects_via_energy-based_separation.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](../../CVPR2025/object_detection/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](../../CVPR2026/object_detection/ew-detr_evolving_world_object_detection_via_incremental_low-rank_detection_trans.md)

</div>

<!-- RELATED:END -->
