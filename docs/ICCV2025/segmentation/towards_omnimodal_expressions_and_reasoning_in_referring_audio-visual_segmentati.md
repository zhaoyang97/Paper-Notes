---
title: >-
  [论文解读] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation
description: >-
  [ICCV 2025][语义分割][音频-视觉分割] 提出 OmniAVS 数据集和 OISA 模型，将指代音频-视觉分割从简单声学属性感知拓展至全模态表达（文本/语音/声音/图像的任意组合）：和深度推理（理解声音内容+世界知识）：，在新基准及多个相关任务上取得 SOTA。 指代音频-视觉分割（RAVS）是一个新兴领域…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "音频-视觉分割"
  - "全模态指代"
  - "推理分割"
  - "多模态大语言模型"
  - "查询传播"
---

# Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation

**会议**: ICCV 2025  
**arXiv**: [2507.22886](https://arxiv.org/abs/2507.22886)  
**代码**: [OmniAVS](https://henghuiding.com/OmniAVS/)  
**领域**: 图像分割  
**关键词**: 音频-视觉分割, 全模态指代, 推理分割, 多模态大语言模型, 查询传播

## 一句话总结

提出 OmniAVS 数据集和 OISA 模型，将指代音频-视觉分割从简单声学属性感知拓展至**全模态表达（文本/语音/声音/图像的任意组合）**和**深度推理（理解声音内容+世界知识）**，在新基准及多个相关任务上取得 SOTA。

## 研究背景与动机

指代音频-视觉分割（RAVS）是一个新兴领域，旨在根据指代表达在视听场景中分割目标对象。现有数据集 Ref-AVS 存在三大局限：

**声音利用浅层化**：表达仅涉及声音的表面属性（如"谁发出最大声音"），不涉及声音内容的理解

**模态单一**：仅支持文本指代表达，缺乏语音、声音片段、图像等多模态输入

**缺乏推理需求**：表达不需要世界知识或复杂推理

以"谁最可能生病了？"为例，模型需要建立认知链：声音→咳嗽→生病，这超越了简单声学特征识别。同时，以 ChatGPT-4o 为代表的全模态 AI 强调了处理任意模态组合输入的重要性。

核心动机：构建一个真正理解声音内容、支持全模态指代、包含复杂推理的 RAVS 基准和基线模型。

## 方法详解

### 数据集 OmniAVS

**视频来源**：Creative Commons 网络视频 + TVQA 电视剧片段 + 自录制视频，从 10,871 个候选中精选 2,104 个视频。

**8 种表达类型**：
- I. 文本 | II. 语音 | III. 文本+声音 | IV. 语音+声音
- V. 文本+图像 | VI. 语音+图像 | VII. 文本+声音+图像 | VIII. 语音+声音+图像

**标注规则**：
- 表达必须关联视频中的声音，而非仅视觉线索
- 强调声音**内容**而非声音**行为**（如"警告的狗"而非"吠叫的狗"）
- 鼓励需要推理的表达，并提供推理解释
- 每个表达可指代 0 到多个目标

**数据规模**：2,104 视频、103k 帧、4,277 目标、206k 掩码、61,095 表达、34,841 推理解释。

### 模型 OISA（Omnimodal Instructed Segmentation Assistant）

**总体架构**：MLLM（音频编码器 + 视觉编码器 + LLM）+ 掩码头（ViT-Adapter + 像素解码器 + 掩码解码器）

- MLLM 基座：InternVL2-1B（InternViT-300M-448px + Qwen2-0.5B）
- 音频编码器：Whisper-large-v3 + 音频 MLP

### 关键设计一：音频-视觉交错（Audio-Visual Interleaving）

视频采样 $N$ 帧获取视觉 token $\{v_1, ..., v_N\}$，音频编码后分割为 $N$ 个片段 $\{a_1, ..., a_N\}$，按时间顺序交错排列：

$$\{v_1, a_1, v_2, a_2, ..., v_N, a_N\}$$

对比 VideoLLaMA 的顺序拼接 $\{v_1,...,v_N, a_1,...,a_N\}$ 或 video-SALMONN 的加权融合，交错策略**无需额外参数**即可实现时间对齐。在 TVQA 子集（包含大量对话、需精确音视对齐）上提升显著。

进一步在交错序列末尾追加完整音频 token $\mathbf{A}$，类似于 InternVL2 的缩略图策略，补充未截断的全局音频信息。

### 关键设计二：查询传播（Query Propagation）

MLLM 生成 `[SEG]` token 表示目标嵌入，传递给掩码解码器。

VideoLISA 的 OTSA（One-Token-Seg-All）策略用同一 `[SEG]` 独立分割每帧，但单个查询携带位置先验，难以适应目标运动（如从右到左），导致 ID 切换。

查询传播逐帧更新查询：

- 每帧分割后，将当前帧的输出查询传播到下一帧
- 查询在线细化，平滑捕获时间运动轨迹
- 有效建模上下文时序信息

$$\text{QP}: \quad q_{t+1} = \text{MaskDecoder}(q_t, F_t) \rightarrow q_{t+1}$$

### 训练流程

**阶段 1 — 音频-文本对齐**：使用 ASR 和 Audio Caption 数据集训练音频编码器 MLP，其余参数冻结。

**阶段 2 — 全模态指令分割微调**：在混合数据上训练（ADE20K、COCO-Stuff、RefCOCO 系列、MeViS、ReVOS、Ref-AVS、OmniAVS 等），使用 LoRA 微调 LLM，训练掩码头全部参数。损失包括交叉熵（文本）+ DICE + BCE（分割）。

## 实验

### OmniAVS 基准

| 方法 | 总体 $\mathcal{J\&F}$ | I(文本) | VII(文本+声+图) | VIII(语音+声+图) | METEOR |
|------|:---:|:---:|:---:|:---:|:---:|
| LMPM | 25.8 | 31.2 | - | - | - |
| MUTR | 32.3 | 35.4 | 41.6 | 40.5 | - |
| LISA-13B | 36.1 | 36.4 | 46.7 | 45.7 | 16.5 |
| **OISA-1B** | **41.1** | **40.1** | **52.6** | **53.0** | **21.7** |

OISA-1B 以仅 1B 参数超越 LISA-13B 5.0%，推理解释质量（METEOR +5.2）同步提升。多模态组合输入（VII/VIII）效果最好，证明多模态互补。

### Ref-AVS 基准

| 方法 | Seen $\mathcal{J}$ | Unseen $\mathcal{J}$ | Mix $\mathcal{J\&F}$ |
|------|:---:|:---:|:---:|
| EEMC | 34.2 | 49.5 | 41.9/58.1 |
| **OISA-1B** | **51.7** | **58.3** | **54.5/61.4** |

在 Seen 和 Unseen 分割上分别提升 +17.5 和 +8.8。

### 消融实验

**音视融合策略**：

| 融合方式 | TVQA子集 | 总体 |
|---------|:------:|:---:|
| Attention | 37.4 | 35.8 |
| 拼接 | 36.9 | 35.3 |
| **AVI + 拼接** | **42.0** | **40.5** |

**掩码头设计**：

| 查询类型 | 掩码头 | $\mathcal{J\&F}$ | FPS |
|---------|-------|:---:|:---:|
| OTSA | SAM | 38.1 | 4.3 |
| OTSA | M2F | 35.2 | 15.7 |
| **QP** | **SAM** | **41.2** | 4.1 |
| **QP** | **M2F** | **40.5** | **12.3** |

查询传播比 OTSA 在 M2F 头上提升 +5.3 $\mathcal{J\&F}$，同时保持 3x 速度优势。

### 关键发现

1. 音频-视觉交错是时间对齐的最优方案，在 TVQA（大量对话）子集上优势最明显
2. 模态越多性能越好（Split VII/VIII 最高），多模态确实提供互补信息
3. 查询传播大幅改善动态目标的跟踪质量，解决 OTSA 的 ID 切换问题
4. OmniAVS 比 Ref-AVS 难度高 17%（41.1 vs 58.0），验证了数据集的挑战性

## 亮点与洞察

- **数据集设计前瞻**：8 种模态组合 + 推理解释 + 多目标指代，为全模态 AI 提供了细粒度感知基准
- **从"听到"到"理解"**：推动 RAVS 从声学属性检测进化到声音内容推理
- **1B 模型超越 13B**：证明任务特化设计（AVI + QP）比纯参数量更重要

## 局限性

- 基座 LLM 仅 0.5B，在需要深度推理的场景（如 ReasonSeg）能力受限
- 复杂混叠声音的解耦仍是瓶颈（如多人同时说话 + 背景音）
- 语音表达通过 TTS 合成，与真实人类语音的分布有差距

## 相关工作

- 音视场景感知：AVSBench、Ref-AVS、Music-AVQA 等音视觉联合学习
- 推理分割：LISA、VideoLISA、VISA 等基于 MLLM 的推理分割方法
- 全模态模型：ChatGPT-4o、VideoLLaMA 等多模态理解系统

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — OmniAVS 数据集定义了全新的全模态推理分割范式
- **技术深度**: ⭐⭐⭐⭐ — AVI 和查询传播设计合理有效
- **实验**: ⭐⭐⭐⭐⭐ — 跨 OmniAVS/Ref-AVS/RefCOCO/MeViS/ReVOS 多任务全面验证
- **写作**: ⭐⭐⭐⭐ — 数据集动机和与 Ref-AVS 的对比论证清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](../../CVPR2025/segmentation/robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)

</div>

<!-- RELATED:END -->
