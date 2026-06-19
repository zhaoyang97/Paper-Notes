---
title: >-
  [论文解读] Elysium: Exploring Object-level Perception in Videos via MLLM
description: >-
  [ECCV 2024][多模态VLM][多模态大语言模型] 提出 Elysium——一个端到端可训练的多模态大语言模型（MLLM），通过构建百万级视频目标感知数据集 ElysiumTrack-1M 和设计视觉 Token 压缩网络 T-Selector，将 MLLM 的目标级感知能力从静态图像扩展到视频领域，支持单目标跟踪（SOT）、引用单目标跟踪（RSOT）和视频引用表达生成（Video-REG）三大任务。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "多模态大语言模型"
  - "目标跟踪"
  - "视频目标感知"
  - "Token压缩"
  - "大规模数据集"
---

# Elysium: Exploring Object-level Perception in Videos via MLLM

**会议**: ECCV 2024  
**arXiv**: [2403.16558](https://arxiv.org/abs/2403.16558)  
**代码**: [有 (GitHub)](https://github.com/Hon-Wong/Elysium)  
**领域**: 视频理解  
**关键词**: 多模态大语言模型, 目标跟踪, 视频目标感知, Token压缩, 大规模数据集

## 一句话总结

提出 Elysium——一个端到端可训练的多模态大语言模型（MLLM），通过构建百万级视频目标感知数据集 ElysiumTrack-1M 和设计视觉 Token 压缩网络 T-Selector，将 MLLM 的目标级感知能力从静态图像扩展到视频领域，支持单目标跟踪（SOT）、引用单目标跟踪（RSOT）和视频引用表达生成（Video-REG）三大任务。

## 研究背景与动机

现有的 MLLM（如 Shikra、MiniGPT-v2 等）已在图像层面展示了出色的目标级感知能力（如图像 Grounding、目标检测），但在**视频领域的目标级任务**（如目标跟踪）上研究不足。作者将视频任务按粒度分为三类：

1. **视频级任务**（VideoQA、视频描述）：关注全局信息，可通过时间轴上的融合操作提取特征
2. **帧级任务**（视频 Grounding、密集视频描述）：需要逐帧区分和分析
3. **目标级任务**（SOT、MOT、VOS）：需要在每帧中定位目标并保持跨帧的时间一致性

将 MLLM 应用于视频目标级任务面临两大核心挑战：

- **数据匮乏**：现有跟踪数据集规模有限（如 LaSOT 仅 1.4K 轨迹），远不足以支撑 MLLM 的大规模预训练
- **计算瓶颈**：处理大量视频帧会在 LLM 的上下文窗口中产生巨大的视觉 Token 负担，严重制约可处理的帧数

已有的视频 MLLM 工作（如 Video-LLaMA、VideoChat）主要聚焦于视频级理解，要么通过时间轴融合压缩信息丢失帧级信息，要么依赖外部专家模型进行目标感知（如 PG-Video-LLaVA），缺乏端到端的统一方案。Elysium 的目标是用纯 MLLM 架构、不借助任何外部模型来处理视频中的目标级任务。

## 方法详解

### 整体框架

Elysium 采用经典的 MLLM 架构：**视觉编码器（CLIP-ViT-L）+ Token 压缩模块（T-Selector）+ 大语言模型（Vicuna）**。对于视频中的每一帧 $\mathbf{X}_v^i$，先通过视觉编码器提取特征 $\mathbf{F}_v^i \in \mathbb{R}^{N \times C}$，再通过 T-Selector 压缩为 $\mathbf{T}_v^i \in \mathbb{R}^{\alpha N \times D}$，其中 $\alpha \in (0,1]$ 为压缩比，$D$ 为 LLM 的隐藏维度。

### 关键设计

#### 1. **ElysiumTrack-1M 数据集构建**：解决大规模视频目标感知训练数据缺失问题

核心思路：从 WebVid-10M 视频数据集出发，通过自动化流水线生成百万级的"名词短语-轨迹"对。

构建流程分两步：
- **Step 1（生成名词短语-边界框对）**：用 spaCy 解析视频描述为名词短语，过滤虚拟词和复数词，用 Grounding DINO 在首帧/中间帧/末帧生成边界框，保留置信度 > 0.6 的结果
- **Step 2（扩展为轨迹对）**：用 MixFormer 从首帧边界框生成轨迹，保留置信度 > 0.8 的轨迹，用卡尔曼滤波过滤漂移轨迹，计算中间帧和末帧的 IoU（< 0.3 则丢弃）

最终生成 **127 万条**名词短语-轨迹对，是 TrackingNet（3.06 万轨迹）的 41 倍。整个过程在 24 张 A100 上仅需 6 天。

同时定义两个新任务：
- **RSOT（引用单目标跟踪）**：仅靠语言描述在视频中定位和跟踪目标，无需位置先验
- **Video-REG（视频引用表达生成）**：给定帧坐标，生成目标描述，要求时序感知能力

#### 2. **T-Selector Token 压缩网络**：在性能与计算效率间取得平衡

核心思路：基于"视频包含冗余信息"的假设，通过门控机制选择最重要的视觉 Token，而非对空间维度进行融合。

设计动机：实验发现空间维度的融合操作（如交叉注意力、拼接）会导致性能急剧下降。T-Selector 通过逐 Token 打分并保留 Top-K 的方式避免了空间融合带来的信息损失。

$$\mathbf{G}_v = \text{KeepTopK}(\text{Softmax}(\text{MLP}(\mathbf{F}_v)), k, \mathbf{F}_v)$$
$$\mathbf{T}_v = \text{MLP}(\mathbf{G}_v)$$

其中 $k = \alpha N$，MLP 门控层计算每个 Token 的重要性分数，Softmax 归一化后取 Top-K，最后通过另一个 MLP 将维度映射到 LLM 隐藏维度。默认设置 $\alpha N = 108$（原始 576 tokens 压缩至 108，压缩比约 5.3 倍）。

#### 3. **输入输出格式设计**：高效利用 Token 预算

- 为每帧的视觉 Token 添加时间戳，使模型能区分连续帧
- 坐标表示采用 [0, 100) 范围的整数格式，如 "[23,45,46,72]" 仅需 13 个 LLaMA token，相比 Shikra 的浮点格式（28 tokens）节省约一半
- 为不同任务设计专用 prompt 模板，增强模型对多种问题格式的鲁棒性

### 训练策略

采用两阶段渐进式训练：

**Stage 1：大规模图像数据预训练**
- 先用 LLaVA-558K 冻结 ViT 和 LLM，仅训练 T-Selector（lr=2e-3，8 GPUs）
- 然后解冻所有参数端到端训练，混合图像数据，30K steps（lr=5e-5，32 GPUs）

**Stage 2：高质量数据微调**
- 混合高质量图像数据 + 视频数据（VideoChat + ElysiumTrack-1M），22K steps
- 前 20K steps 每视频随机采样 2-8 帧，间隔 1-60 帧（模拟不同帧率和运动速度）
- 后 2K steps 扩展到 32 帧/视频，batch size=1

推理时：VideoQA 均匀采样 16 帧；SOT/RSOT 将长视频切分为 8 帧的 clip，相邻 clip 重叠 1 帧用于初始化跟踪。

## 实验关键数据

### 主实验

**图像 Grounding（RefCOCO 系列）**

| 模型 | Token数 | RefCOCO val | RefCOCO test-A | RefCOCO+ val | RefCOCOg val |
|------|---------|-------------|----------------|--------------|--------------|
| Shikra (7B) | 256 | 87.01 | 90.61 | 81.60 | 82.27 |
| MiniGPT-v2 (7B) | 256 | 88.69 | 91.65 | 79.97 | 84.44 |
| Ferret (7B) | 608 | 87.49 | 91.35 | 80.78 | 83.93 |
| **Elysium (7B)** | **108** | **89.07** | **92.12** | **82.86** | 82.92 |

Elysium 仅用 108 个视觉 Token 即超越使用 256-608 Token 的基线方法。

**零样本 SOT（LaSOT 数据集）**

| 模型 | 是否零样本 | AUC | P | P_Norm |
|------|-----------|-----|---|--------|
| SiamRPN++ | 否 | 49.6 | 56.9 | 49.1 |
| DiMP | 否 | 56.9 | 65.0 | 56.7 |
| SiamGAT | 否 | 53.9 | 63.3 | 53.0 |
| **Elysium** | **是** | **56.1** | 61.0 | 50.1 |

零样本下 AUC 达到 56.1，接近专用跟踪器（DiMP 56.9）。

### 消融实验

**T-Selector 压缩比消融（RefCOCO 系列平均精度）**

| 压缩模块 | Token数 $\alpha N$ | 平均精度 | 说明 |
|---------|---------|---------|------|
| 无压缩 | 256 | 76.81 | ViT@224 + MLP |
| 无压缩 | 576 | 81.45 | ViT@336 + MLP |
| 拼接 (Concat) | 144 | 54.65 | 空间融合导致严重退化 |
| 交叉注意力 (C.A.) | 144 | 49.23 | 更差的空间融合 |
| T-Selector | 1 | 43.70 | 极端压缩 |
| T-Selector | 36 | 74.48 | 较大性能损失 |
| T-Selector | 108 | 78.54 | **最佳性价比** |
| T-Selector | 256 | 80.09 | 接近无压缩性能 |

### 关键发现

1. **空间融合操作（拼接/交叉注意力）不适合视频目标感知**：相同 Token 数下，T-Selector 的门控选择策略显著优于融合方式（78.54 vs 54.65/49.23）
2. **Token 压缩存在临界点**：从 108 降到 72 时性能显著下降，108 是性能-效率的最佳平衡点
3. **零样本 SOT 可行**：通过大规模视频目标数据训练，MLLM 无需微调即可接近专用跟踪器
4. **MLLM 处理小目标的能力有限**：在 UAV123 等包含小目标的数据集上表现欠佳，受限于视觉编码器分辨率

## 亮点与洞察

- **数据驱动的工程洞察**：通过搭建自动化数据生产流水线（结合 Grounding DINO + MixFormer + 多种过滤策略），从通用视频数据中生成了百万级高质量目标跟踪注释，这种"用模型生成数据、再用数据训练模型"的范式值得借鉴
- **Token 选择优于 Token 融合**：对比拼接/交叉注意力等融合方式，按重要性选择 Token 的策略更好地保留了空间信息
- **坐标格式的 Token 效率**：整数坐标表示比浮点表示节省一半 Token，在长视频场景下收益显著

## 局限与展望

- 小目标跟踪效果不够理想，需要引入高分辨率视觉编码器
- 仅探索了 SOT 相关任务，未涉及 MOT、VOS、RVOS 等更复杂的目标级任务
- 数据集构建依赖 Grounding DINO 和 MixFormer 的质量，可能继承这些模型的偏差
- T-Selector 的 Token 选择是静态的，未考虑不同帧/任务可能需要不同的 Token 分配策略

## 相关工作与启发

- 与 PG-Video-LLaVA 等依赖外部专家模型的方案相比，Elysium 实现了端到端训练，架构更简洁
- RSOT 任务将语言引用与目标跟踪结合，为视觉-语言交互提供了新的研究范式
- ElysiumTrack-1M 的构建方法可推广到其他视频理解任务的数据集构建

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统性地用纯 MLLM 实现视频目标级感知，提出 RSOT 和 Video-REG 两个新任务
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖图像 Grounding、VideoQA、SOT、RSOT、Video-REG 多个任务，消融研究完整
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，问题定义明确，图表丰富
- **价值**: ⭐⭐⭐⭐ — 数据集和方法为视频 MLLM 的目标级感知研究奠定了基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](../../CVPR2025/multimodal_vlm/videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[ECCV 2024\] Zero-shot Object Counting with Good Exemplars (VA-Count)](zero-shot_object_counting_with_good_exemplars.md)
- [\[ECCV 2024\] MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)
- [\[CVPR 2026\] From Failure to Feedback: Group Revision Unlocks Hard Cases in Object-Level Grounding](../../CVPR2026/multimodal_vlm/from_failure_to_feedback_group_revision_unlocks_hard_cases_in_object-level_groun.md)
- [\[CVPR 2026\] Granulon: Awakening Pixel-Level Visual Encoders with Adaptive Multi-Granularity Semantics for MLLM](../../CVPR2026/multimodal_vlm/granulon_awakening_pixel-level_visual_encoders_with_adaptive_multi-granularity_s.md)

</div>

<!-- RELATED:END -->
