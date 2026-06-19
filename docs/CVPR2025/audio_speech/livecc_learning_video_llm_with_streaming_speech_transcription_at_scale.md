---
title: >-
  [论文解读] LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale
description: >-
  [CVPR 2025][音频/语音][流式视频LLM] 提出 LiveCC，通过将 ASR 转录词与视频帧沿时间轴密集交织训练视频 LLM，构建了 Live-CC-5M 预训练数据集，使 7B 模型在实时视频解说任务上超越 72B 模型（包括 Qwen2.5-VL-72B）。 领域现状：视频 LLM 通常在"观看完整视频后回…
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "流式视频LLM"
  - "ASR转录"
  - "实时解说"
  - "密集交织"
  - "YouTube CC"
---

# LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale

**会议**: CVPR 2025  
**arXiv**: [2504.16030](https://arxiv.org/abs/2504.16030)  
**代码**: [https://showlab.github.io/livecc](https://showlab.github.io/livecc)  
**领域**: 音频语音 / 视频理解  
**关键词**: 流式视频LLM, ASR转录, 实时解说, 密集交织, YouTube CC

## 一句话总结

提出 LiveCC，通过将 ASR 转录词与视频帧沿时间轴密集交织训练视频 LLM，构建了 Live-CC-5M 预训练数据集，使 7B 模型在实时视频解说任务上超越 72B 模型（包括 Qwen2.5-VL-72B）。

## 研究背景与动机

**领域现状**：视频 LLM 通常在"观看完整视频后回答问题"的离线模式下训练。但实际场景（如体育解说、视频直播）需要模型在视频流入过程中持续生成描述——流式理解。

**现有痛点**：现有视频 LLM 缺乏流式能力，因为训练数据是"视频-问题-答案"三元组，不包含时间密集的持续描述。YouTube 虽有海量带字幕的视频但字幕质量参差不齐。

**核心矛盾**：高质量流式训练数据稀缺——人工标注成本极高，而自动生成的字幕（CC）噪声大。

**切入角度**：将 YouTube CC 字幕视为 ASR 转录，与视频帧沿时间轴交织形成流式序列。用这种数据格式大规模预训练，再用高质量 WhisperX 转录数据进行 SFT。

**核心 idea**：ASR 词-视频帧时间交织 = 流式视频理解能力的大规模无监督学习。

## 方法详解

### 关键设计

1. **流式训练序列格式**:

    - 功能：让 LLM 学会在视频流入过程中持续生成描述
    - 核心思路：序列格式 `[Con]<F_{t:t+k}><W_{t:t+k}><F_{t+k:t+2k}><W_{t+k:t+2k}>...`，视频帧 2 FPS 采样，ASR 词按时间戳对齐穿插。LLM 在每个时间窗口预测下一段 ASR 词
    - 设计动机：与 caption 式（整段描述在最后）对比，streaming 式序列的解说质量从 14.0% 胜率跃升到 32.9%——密集交织让模型学到精确的时间对应关系

2. **两级数据构建（Live-CC-5M + Live-WhisperX-526K）**:

    - 功能：大规模低质预训练 + 小规模高质 SFT
    - 核心思路：Live-CC-5M 从 YouTube 收集 5M 视频片段的 CC 字幕（去除说话人视频），预训练流式能力。Live-WhisperX-526K 用 WhisperX 重新转录 526K 高质量片段，加上 GPT-4o 生成的问答提示，做 SFT
    - 设计动机：CC 字幕噪声大但规模大（5M），WhisperX 质量高但成本高（526K）——两级训练平衡了规模和质量

### 损失函数 / 训练策略

标准自回归语言模型损失，只在 ASR 词的 token 上计算损失（帧 token 不计算）。SFT 混合 Live-WhisperX-526K + LLaVA-Video-178K。推理延迟 <0.5s/帧@2FPS。

## 实验关键数据

### 主实验

| 任务 | LiveCC-7B | Qwen2-VL-7B | GPT-4o |
|------|-----------|-------------|--------|
| VideoMME (短视频) | **70.1%** | 69.4% | - |
| LiveSports-3K 胜率 | **41.5%** | 33.7% | 参考 |
| OVOBench | **超越72B模型** | - | - |

### 消融实验

| 配置 | 解说胜率 | 说明 |
|------|---------|------|
| Caption 式序列 | 14.0% | 非流式 |
| Streaming 式序列 | **32.9%** | +18.9% |
| 无 ASR 上下文 | 14.7% | 历史上下文关键 |
| + ASR 上下文 | **32.0%** | — |
| 1M 数据 | 29.1% | — |
| 5M 数据 | **32.9%** | 数据扩展有效 |

### 关键发现
- **流式序列格式远优于标题式**：胜率 32.9% vs 14.0%，说明时间密集交织是流式能力的关键
- **7B 超越 72B**：在实时解说中 LiveCC-7B 超越 Qwen2.5-VL-72B 和 LLaVA-Video-72B，说明数据范式比模型规模更重要
- **ASR 历史上下文重要**：之前说了什么直接影响接下来该说什么

## 亮点与洞察
- **范式创新**——将视频理解从"看完再答"转变为"边看边说"，数据格式的改变带来了能力的质变
- **YouTube CC 作为免费训练数据**——利用已有的海量字幕数据，无需额外标注
- **小模型打大模型**——正确的训练范式比堆参数更有效

## 局限与展望
- YouTube CC 质量低（需要大量预处理）
- 流式模式降低了指令遵循能力，需要基于 logits 的评估
- SFT 依赖 GPT-4o 生成提示，成本和偏置
- 只处理前向视觉输入

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 流式视频理解的训练范式创新
- 实验充分度: ⭐⭐⭐⭐ 多基准多消融，体育解说评估新颖
- 写作质量: ⭐⭐⭐⭐ 清晰完整
- 价值: ⭐⭐⭐⭐⭐ 为实时视频理解开辟了大规模训练路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[CVPR 2025\] Learning to Highlight Audio by Watching Movies](learning_to_highlight_audio_by_watching_movies.md)
- [\[CVPR 2026\] Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning](../../CVPR2026/audio_speech/pushing_the_frontier_of_audiovisual_perception_with_large-scale_multimodal_corre.md)
- [\[ACL 2026\] DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition](../../ACL2026/audio_speech/duivrs-2_an_llm-based_interactive_voice_response_system_for_large-scale_poi_attr.md)
- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)

</div>

<!-- RELATED:END -->
