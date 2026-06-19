---
title: >-
  [论文解读] SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods
description: >-
  [ACL 2025][AI安全][speech deepfake detection] 构建 SpeechFake 大规模语音深伪数据集，包含 300 万+深伪样本、3000+ 小时音频、40 种生成工具和 46 种语言，并通过基线实验系统分析了生成方法、语言多样性和说话人变化对检测性能的影响。 - 现有问题：现有语音深伪数…
tags:
  - "ACL 2025"
  - "AI安全"
  - "speech deepfake detection"
  - "dataset"
  - "multilingual"
  - "TTS"
  - "voice conversion"
  - "neural vocoder"
---

# SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods

**会议**: ACL 2025  
**arXiv**: [2507.21463](https://arxiv.org/abs/2507.21463)  
**代码**: [YMLLG/SpeechFake](https://github.com/YMLLG/SpeechFake)  
**领域**: AI Safety  
**关键词**: speech deepfake detection, dataset, multilingual, TTS, voice conversion, neural vocoder  

## 一句话总结

构建 SpeechFake 大规模语音深伪数据集，包含 300 万+深伪样本、3000+ 小时音频、40 种生成工具和 46 种语言，并通过基线实验系统分析了生成方法、语言多样性和说话人变化对检测性能的影响。

## 研究背景与动机

- **现有问题**：现有语音深伪数据集在规模和多样性方面存在明显不足——大多数公开数据集规模较小、生成技术陈旧或有限、且主要集中在英语或中文。
- **泛化瓶颈**：检测模型在遇到未见过的深伪技术时性能急剧下降，简单合并已有数据集会引入条件不匹配和训练复杂度问题。
- **前沿缺失**：近年涌现大量先进语音生成技术（如 CosyVoice、ChatTTS、GPT-SoVITS 等），但现有数据集未纳入这些最新方法。
- **本文方案**：构建 SpeechFake 数据集，分为双语数据集 (BD：英/中) 和多语言数据集 (MD：46 种语言)，使用 30 个开源工具和 10 个商业 API 生成深伪音频，全面覆盖 TTS、VC 和 NV 三类生成方法。

## 方法详解

### 整体框架

数据集构建流程包括：(1) **真实语音采集**：从 LibriTTS、VCTK、AISHELL1、AISHELL3 和 CommonVoice 获取真实语音；(2) **深伪生成**：按三种方法分类——TTS（文本到语音）、VC（语音克隆/转换）和 NV（神经声码器），使用 40 种不同工具生成；(3) **后处理**：VAD 过滤短于 0.5 秒的片段，选择性人工审核，统一为 16kHz 单声道 WAV 格式。

### 关键设计

- **双语 + 多语言分割**：BD 专注英/中双语（使用全部 40 种工具），MD 覆盖 46 种语言（使用 6 种多语言工具），训练集仅含英/中，测试集扩展到 46 种语言以评估跨语言泛化。
- **前沿方法覆盖**：纳入过去一年发布的最新语音生成技术（如 CosyVoice、ChatTTS、GPT-SoVITS 等），这些方法能生成极其逼真的合成语音。
- **丰富元数据**：提供生成方法、说话人 ID、语言、文本转写等标注，支持超越二分类的深度研究。

### 评估指标

- 使用 **等错率 (EER, Equal Error Rate)** 作为主要评价指标，与先前工作一致。

## 实验

### 主实验结果（EER%，越低越好）

| 训练数据 | 模型 | BD | BD-EN | BD-CN | ASV19 | WF | ITW | CDADD |
|----------|------|-----|-------|-------|-------|-----|-----|-------|
| ASV19 | AASIST | 39.36 | 41.05 | 39.07 | **1.88** | 21.17 | 45.27 | 49.53 |
| BD | AASIST | **3.48** | **3.98** | **2.68** | 23.62 | **4.30** | **7.53** | **22.52** |
| ASV19 | W2V+AASIST | 23.78 | 20.15 | 24.93 | **0.89** | 3.48 | 10.07 | 8.55 |
| BD | W2V+AASIST | **3.54** | **3.55** | **2.83** | 2.91 | **0.58** | **2.01** | **2.42** |

### 消融实验

| 分析维度 | 关键发现 |
|----------|----------|
| 跨生成器泛化 | TTS 训练数据泛化能力最佳（BD 整体 EER 14.26/AASIST），NV 最差（26.30）；不同生成方法间存在明显泛化鸿沟 |
| 跨语言泛化 | AASIST 在未见语言上 EER 显著上升（法语 22.54%、印地语 26.06%），W2V+AASIST 因多语言预训练在 50 epoch 后所有语言 EER <1% |
| 跨说话人影响 | 说话人变化对检测有影响，但训练数据的说话人多样性可有效缓解 |
| BD-EN vs BD-CN | 两子集在对方测试集上性能下降，使用完整 BD 训练效果最优 |

### 关键发现

1. 在 SpeechFake 上训练的模型对外部基准的泛化性远优于在 ASVspoof2019 上训练的模型（在 ITW 上 EER 从 45.27% 降至 7.53%）。
2. 生成方法是影响泛化的首要因素——在 TTS 数据上训练的模型对未见商用 TTS API 也表现良好（BD-UT EER 0.53%/AASIST）。
3. 语言因素在控制生成方法一致后仍对检测有影响，但多语言预训练特征提取器（如 Wav2Vec2.0 XLSR）可大幅缓解。
4. 数据集的规模和多样性是提升泛化能力的关键——简单增加同质数据不如增加生成方法和语言的多样性。

## 亮点

- 规模空前：300 万+深伪样本、3000+ 小时、40 种生成工具、46 种语言。
- 系统对比设计：分三类生成方法和双语/多语言两个维度独立分析各因素的影响。
- 纳入最新前沿生成技术，使基准具有前瞻性。
- 提供丰富元数据（方法类型、说话人、语言、转写），支持多角度研究。

## 局限性

- 多语言数据集训练集仅含英/中，其他语言仅出现在测试集中，可能低估跨语言微调的潜力。
- 质量过滤仅抽检约 1% 样本，可能遗漏部分低质量深伪。
- 40 种工具中部分因版权或技术原因生成数据量差异较大，可能导致分布不均。
- 未涵盖对抗性攻击（如 Malafide）和编解码器失真等场景。

## 相关工作

- **语音深伪数据集**：ASVspoof 系列（2015-2024）、WaveFake、In-the-Wild、MLAAD（23 种语言）、SpoofCeleb（268 万条）等。
- **语音生成技术**：从 CNN/RNN → Transformer → GAN/Flow/Diffusion → LLM-based TTS 的技术演进。
- **检测方法**：AASIST（异构图注意力网络）、W2V+AASIST（Wav2Vec2.0 + AASIST）等前端-后端检测架构。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 3 |
| 实用性 | 5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 总评 | 4.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Fairness Assessment of Dutch Hate Speech Detection](towards_fairness_assessment_of_dutch_hate_speech_detection.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](../../ICML2026/ai_safety/medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[CVPR 2026\] GenBreak: Red Teaming Text-to-Image Generation Using Large Language Models](../../CVPR2026/ai_safety/genbreak_red_teaming_text-to-image_generation_using_large_language_models.md)
- [\[CVPR 2026\] FVBench: Benchmarking Deepfake Video Detection Capability of Large Multimodal Models](../../CVPR2026/ai_safety/fvbench_benchmarking_deepfake_video_detection_capability_of_large_multimodal_mod.md)
- [\[CVPR 2026\] Unlearning without Forgetting: Securely Removing Targeted Concepts from Large-Scale Vision-Language Open-Vocabulary Detectors](../../CVPR2026/ai_safety/unlearning_without_forgetting_securely_removing_targeted_concepts_from_large-sca.md)

</div>

<!-- RELATED:END -->
