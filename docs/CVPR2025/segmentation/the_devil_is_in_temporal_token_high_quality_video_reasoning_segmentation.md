---
title: >-
  [论文解读] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation
description: >-
  [CVPR 2025][语义分割][视频推理分割] VRS-HQ 提出分层时间 token 编码（帧级 <SEG + 视频级 <TAK）和基于 token 驱动的关键帧选择策略，结合 SAM2 实现端到端的视频推理分割，在 ReVOS 上超越 VISA 达 9.1%。 视频推理分割（VRS）要求模型根据复杂的隐含意图文本生成…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "视频推理分割"
  - "时间token聚合"
  - "关键帧选择"
  - "SAM2"
  - "多模态大模型"
---

# The Devil is in Temporal Token: High Quality Video Reasoning Segmentation

**会议**: CVPR 2025  
**arXiv**: [2501.08549](https://arxiv.org/abs/2501.08549)  
**代码**: [VRS-HQ](https://github.com/sitong-gong/VRS-HQ)  
**领域**: 图像分割  
**关键词**: 视频推理分割, 时间token聚合, 关键帧选择, SAM2, 多模态大模型

## 一句话总结

VRS-HQ 提出分层时间 token 编码（帧级 `<SEG>` + 视频级 `<TAK>`）和基于 token 驱动的关键帧选择策略，结合 SAM2 实现端到端的视频推理分割，在 ReVOS 上超越 VISA 达 9.1%。

## 研究背景与动机

视频推理分割（VRS）要求模型根据复杂的隐含意图文本生成视频级分割掩码，现有方法存在三大瓶颈：
- **有限的时间上下文**：VISA、VideoLISA 等方法依赖单个 `<SEG>` token 表示关键帧或整个视频中的目标，无法充分捕获帧间变化和时空特征
- **不准确的关键帧检测**：VISA 使用外部模型 LLaMA-VID 检测关键帧，在复杂时间推理场景中可能产生不准确的关键帧，影响后续分割
- **解耦的分割与传播**：VISA 分别使用 SAM（关键帧分割）和 XMem（掩码传播），无法端到端训练和推理
- 单 token 的表示能力不足以同时编码帧内空间细节和帧间时间动态
- 需要一种统一的方案，在一个流程内完成时间推理、关键帧选择、分割和传播

## 方法详解

### 整体框架

VRS-HQ 由四个模块组成：(1) 基于 Chat-UniVi 的 MLLM 进行时间 token 编码（生成帧级 `<SEG>` 和视频级 `<TAK>` token）；(2) 时间动态聚合（TDA）将帧级特征融合到时间 token 中；(3) Token 驱动关键帧选择（TKS）利用 token 相似度和 SAM2 遮挡分数选择关键帧；(4) SAM2 执行关键帧分割和掩码传播。

### 关键设计1：分层时间 Token 编码

**功能**：分别捕获帧级空间信息和视频级时间语义，提供丰富的时空特征给分割模型。

**核心思路**：扩展 MLLM 词汇表引入两种特殊 token——帧级 `<SEG>` 和视频级 `<TAK>`。设计结构化对话模板："Please find {expression} in the Reference Video and segment it in each frame and the entire video respectively." MLLM 通过自回归学习生成包含多个 `<SEG>` token 和一个 `<TAK>` token 的响应。提取 MLLM 最后一层的嵌入 $\bar{h}_{seg} \in \mathbb{R}^{T' \times d'}$ 和 $\bar{h}_{tak} \in \mathbb{R}^{1 \times d'}$，通过 MLP 映射到 SAM2 的特征空间。

**设计动机**：单 token 不足以同时编码空间细节和时间连贯性。分层 token 让模型分别学习局部和全局信息，再通过融合结合两者优势。

### 关键设计2：时间动态聚合（TDA）

**功能**：基于余弦相似度的加权融合，将帧级空间特征整合到时间 token 中，同时维持目标的时间一致性。

**核心思路**：计算每个帧级 `<SEG>` token 与视频级 `<TAK>` token 之间的余弦相似度，归一化为权重 $\lambda_i$。融合公式为 $h'_{tak} = h_{tak} + \alpha \sum_{i=1}^{T'} \lambda_i h_{seg}[i]$，其中 $\alpha$ 为融合系数。高相似度的帧贡献更大权重，自然地将代表性帧的空间信息注入时间 token。训练时，相似度最高的帧被选为关键帧。

**设计动机**：简单平均融合会模糊目标位置信息，而基于相似度的加权融合能让时间 token 偏向最具代表性的帧，同时保留全局语义。

### 关键设计3：Token 驱动关键帧选择（TKS）

**功能**：推理时无需外部关键帧检测模型，利用 SAM2 的遮挡分数辅助精确选择关键帧。

**核心思路**：推理阶段，使用 CLIP 模型找到与文本表达最匹配的帧作为全局采样锚点。将每个采样帧视为候选关键帧，与融合后的 `<TAK>` 嵌入一起送入 SAM2 计算遮挡分数 $S_o = \mathcal{MD}(\mathcal{E}(\mathcal{X}_V^f), h'_{tak})$。最终结合 softmax 归一化的遮挡分数 $S'_o$ 和 token 相似度分数 $S_t$ 确定关键帧。

**设计动机**：外部关键帧检测模型的错误会传播到分割结果；SAM2 的遮挡分数天然反映目标在当前帧的可见度和置信度，是理想的关键帧选择指标。

### 损失函数

端到端训练损失：$L_{total} = \lambda_{txt} L_{txt} + \lambda_{mask} L_{mask}$，其中 $L_{mask} = \lambda_{bce} L_{bce} + \lambda_{dice} L_{dice}$。权重设置：$\lambda_{txt}=1, \lambda_{mask}=1, \lambda_{bce}=2, \lambda_{dice}=0.5$。

## 实验关键数据

### 主实验：ReVOS 视频推理分割基准

| 方法 | Backbone | Referring J&F | Reasoning J&F | Overall J&F |
|------|----------|---------------|---------------|-------------|
| **VRS-HQ (7B)** | Chat-UniVi-7B | **62.1** | **56.1** | **59.1** |
| VISA (13B) | Chat-UniVi-13B | 57.4 | 44.3 | 50.9 |
| VISA (7B) | Chat-UniVi-7B | 50.9 | 43.0 | 46.9 |
| TrackGPT (13B) | LLaVA-13B | 49.5 | 40.5 | 45.0 |
| LISA (13B) | LLaVA-13B | 46.6 | 36.7 | 41.6 |

### 消融实验：各组件贡献（ReVOS Overall J&F）

| 设置 | J&F |
|------|-----|
| VRS-HQ (完整) | **59.1** |
| 仅 `<SEG>` token（无 TDA） | 较低 |
| 无 TKS（外部关键帧选择） | 较低 |
| 无 SAM2 传播（使用 XMem） | 较低 |

### 关键发现

- VRS-HQ (7B) 在 ReVOS 上超越 VISA-13B **9.1%** 的 J&F，在 referring/reasoning/overall 三个子集上分别提高 5.9%/12.5%/9.1%
- 在三个标准 RVOS 数据集上也分别超越 7.3%/5.6%/6.5%
- TDA 的加权融合比简单平均融合效果显著更好
- SAM2 的遮挡分数是关键帧选择的有效信号，消除了对外部模型的依赖

## 亮点与洞察

- **分层 token 设计**："帧级+视频级"的分层编码思路可推广到其他视频理解任务
- **端到端推理**：利用 SAM2 统一分割和传播，消除了 VISA 的多模型级联问题
- **Token 驱动的关键帧选择**：利用模型自身的时间理解能力而非外部模型，更优雅也更鲁棒

## 局限与展望

- 依赖 SAM2 的遮挡分数质量，当 SAM2 对目标理解不准确时可能影响关键帧选择
- 候选关键帧数量受采样帧数限制，可能遗漏最佳关键帧
- 在超长视频场景中，MLLM 处理大量帧的能力可能受限
- 未来可探索多目标同时推理分割

## 相关工作与启发

- 与 VISA 的对比表明，单 token 是视频推理分割的核心瓶颈，分层 token 是有效解法
- SAM2 的即时分割+跨帧传播能力使其成为视频分割的理想后端
- 关键帧选择策略的改进对最终分割质量影响巨大，值得在其他视频任务中借鉴

## 评分

⭐⭐⭐⭐ — 清晰地识别了现有 VRS 方法的核心瓶颈（单 token + 外部关键帧检测），提出的分层 token 和 TKS 策略在大幅度提升性能的同时简化了推理流程。在 7B 模型规模下超越 13B 基线尤为亮眼。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Image Quality Assessment: From Human to Machine Preference](image_quality_assessment_from_human_to_machine_preference.md)
- [\[CVPR 2025\] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)

</div>

<!-- RELATED:END -->
