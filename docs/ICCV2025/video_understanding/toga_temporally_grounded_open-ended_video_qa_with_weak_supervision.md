---
title: >-
  [论文解读] TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision
description: >-
  [ICCV 2025][视频理解][Video QA] 提出TOGA——一种弱监督条件下的视觉语言模型，通过多尺度视觉语言连接器和一致性约束生成伪时序标签，在**无需任何时序标注**的情况下联合生成开放式答案与时间定位，在NExT-GQA、MSVD-QA和ActivityNet-QA上取得SOTA。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "Video QA"
  - "时序定位"
  - "弱监督"
  - "视觉语言模型"
  - "多尺度时序建模"
---

# TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision

**会议**: ICCV 2025  
**arXiv**: [2506.09445](https://arxiv.org/abs/2506.09445)  
**代码**: 未公开  
**领域**: 视频理解 / 视频问答 / 时序定位  
**关键词**: Video QA, 时序定位, 弱监督, 视觉语言模型, 多尺度时序建模

## 一句话总结

提出TOGA——一种弱监督条件下的视觉语言模型，通过多尺度视觉语言连接器和一致性约束生成伪时序标签，在**无需任何时序标注**的情况下联合生成开放式答案与时间定位，在NExT-GQA、MSVD-QA和ActivityNet-QA上取得SOTA。

## 研究背景与动机

视频问答（Video QA）要求模型不仅生成正确答案，还需在视频中定位支持答案的**时间段**，即grounded video QA。这一任务存在三大挑战：

**时间标注成本高昂**：获取精确的起止时间标注需要大量人力，现有方法如Grounded-VideoLLM依赖外部GPT-4生成标注或从ActivityNet-Captions借用标签，成本和噪声都不理想

**open-ended vs. multiple-choice**：之前的弱监督方法（如SeViLA、LLoVi）在推理时依赖**选项候选**来选择答案，限制了模型的开放生成能力；TOGA生成自由文本回答，难度更大

**答案与定位的独立预测问题**：现有方法分别预测答案和时间段（如后处理方式或独立模块），无法建模答案内容与时间窗口之间的依赖关系

TOGA的核心思路是：**联合生成答案和时序定位**（格式为 `Answer [start, end]`），并利用一致性约束在弱监督下生成高质量伪标签。

## 方法详解

### 整体框架

TOGA包含四个模块：

1. **视觉编码器**：冻结的CLIP-ViT-Large，对均匀采样的视频帧提取逐帧特征
2. **文本编码器**：冻结的LLM tokenizer + embedding层（Mistral-7B），生成token级文本特征
3. **多尺度视觉语言连接器 (MS-VLC)**：可训练，在两个时间分辨率上对齐视觉和文本特征
4. **文本解码器**：Mistral-7B Instruct，训练后联合生成答案和时序定位

### 多尺度视觉语言连接器 (MS-VLC)

MS-VLC是TOGA的核心创新之一。它在两个时间粒度上处理视频帧：

- **稀疏尺度**（4帧）：捕获低频时序特征，适合定位长时间段事件
- **密集尺度**（16帧）：捕获高频时序特征，适合定位短时间段事件

每个VLC模块由 RegNet + 3D卷积实现，两个尺度**共享参数**。多尺度处理策略借鉴了活动识别（SlowFast）和音频事件检测中的成功经验。

### 三阶段训练策略

TOGA采用渐进式多阶段训练，逐步获得定位能力：

**Stage 1 — 视觉-文本对齐**：仅训练MS-VLC模块。使用Video-ChatGPT的视频-文本对，包括视频描述、句子补全和问答任务。目标是让多尺度视频特征与文本特征对齐。

**Stage 2 — 指令微调（时序引用）**：训练MS-VLC + LLM解码器。核心是让模型理解带时间引用的prompt（如 `What is the activity in [10, 20]?`），并生成带时间定位的回答（如 `A boy is running [10, 20]`）。由于无真实标注，通过**裁剪视频时间段**生成伪标签：选定起止时间，将该段视为独立视频，用Stage 1的模型生成描述作为伪答案。

**Stage 3 — 一致性约束精化**：关键创新。通过一致性约束筛选高质量伪标签。具体而言，对于grounding问题 $Q_g$（如 `What is the boy doing?`）生成回答 `Stands up [5, 10]`，则构造对应的referring问题 $Q_r$（如 `What does the boy do in [5, 10]?`），期望回答一致（`Stands up`），并与ground truth答案对齐。这种**双向一致性**确保弱监督伪标签的可靠性。

### 损失函数

采用标准的**next token prediction**损失（与语言模型一致），在不同阶段使用不同的prompt格式：
- 仅答案：`answer`
- 仅定位：`[<<<start>>>, <<<end>>>]`
- 答案+定位：`answer [<<<start>>>, <<<end>>>]`

## 实验

### 数据集与指标

| 数据集 | 任务 | 特点 |
|--------|------|------|
| NExT-GQA | 弱监督Grounded QA | 长视频（平均40s），因果+时序问题 |
| ReXTime | 零样本Grounding | 跨时间段因果推理 |
| MSVD-QA | 开放式QA | 1970视频，50K+ QA对 |
| ActivityNet-QA | 开放式QA | 5800视频，58K QA对 |

### 主实验结果

**表1：NExT-GQA弱监督Grounded QA**

| 方法 | 开放式 | mIoU | IoU@0.5 | mIoP | IoP@0.5 | Acc@GQA |
|------|--------|------|---------|------|---------|---------|
| SeViLA | ✗ | 21.7 | 13.8 | 29.5 | 22.9 | 16.6 |
| LLoVi | ✗ | 20.0 | 15.3 | 37.3 | 36.9 | 24.3 |
| Grounded-VideoLLM | ✗ | 21.1 | 18.0 | 34.5 | 34.4 | 26.7 |
| **TOGA** | **✓** | **24.4** | **21.1** | **40.5** | **40.6** | **24.6** |

TOGA在**开放式**（更高难度）设置下仍超越所有closed-set方法。mIoU比最佳方法+3.3pp，IoP@0.5达到40.6%。

**表3：开放式Video QA**

| 方法 | MSVD-QA Acc | MSVD-QA Score | ActivityNet-QA Acc | ActivityNet-QA Score |
|------|------------|--------------|-------------------|---------------------|
| Video-LLaVA | 70.7 | 3.9 | 45.3 | 3.3 |
| Video-LLaMA2 | 70.9 | 3.8 | 50.2 | 3.3 |
| **TOGA** | **73.8** | **3.9** | **52.0** | **3.4** |

### 消融实验

**表4：多尺度VLC的重要性（NExT-GQA, IoU）**

| 模型 | All | Short | Medium | Long |
|------|-----|-------|--------|------|
| 仅稀疏 | 20.0 | 16.2 | 28.9 | 47.5 |
| 仅密集 | 22.1 | 18.3 | 32.2 | 32.1 |
| **多尺度(MS-VLC)** | **24.4** | **20.5** | **34.7** | **49.3** |

多尺度对短事件和长事件提升最明显——稀疏尺度擅长定位长时间段，密集尺度擅长短时间段，二者互补。

**一致性约束的作用**：移除Stage 3（仅用伪标签训练），mIoU从24.4骤降至12.1，证明一致性约束是弱监督定位成功的关键。

**表5：问题类型分析（Acc@GQA）**

| 因果-Why | 因果-How | 时序-Present | 时序-Past | 时序-Future |
|----------|---------|-------------|-----------|------------|
| 26.1 | 27.4 | 23.4 | 18.0 | 18.1 |

时序问题（尤其past/future）显著难于因果问题，需更强的长期推理能力。

## 亮点与洞察

1. **弱监督+开放式+联合生成**三重挑战的统一解法：不依赖外部模型或标注数据库，纯自举式训练
2. **一致性约束**是巧妙的自监督信号：grounding问题和referring问题的答案互相验证，过滤噪声伪标签
3. **联合生成优于分离预测**：模型可根据答案内容调整时间窗口，捕获答案与定位间的相关性
4. 推理效率高：平均0.6秒/样本（A100 GPU），适合实际应用

## 局限性

1. 开放式答案在标准评估指标下可能被低估（语义等价但文本不匹配的答案被判错）
2. 时序问题（past/future类型）准确率仍较低，长距离时序推理有提升空间
3. 伪标签质量受限于Stage 1模型的描述能力，对复杂场景（多人交互、快速动作变化）可能不够准确
4. 仅在40秒级别视频上验证，对更长视频（数分钟以上）的泛化能力未知

## 相关工作

- **Video QA**: FrozenBiLM, Video-ChatGPT, Video-LLaVA, Chat-UniVi
- **Grounded Video QA**: SeViLA, LLoVi, Grounded-VideoLLM, VideoStreaming
- **弱监督时序定位**: NExT-GQA, IGV

## 评分

- 创新性：⭐⭐⭐⭐ — 一致性约束的伪标签策略是弱监督grounded QA的新范式
- 实用性：⭐⭐⭐⭐ — 零标注需求降低了部署门槛
- 实验充分度：⭐⭐⭐⭐ — 多数据集+详细消融+问题类型分析
- 写作质量：⭐⭐⭐⭐ — 动机清晰，方法描述详尽

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[ICLR 2026\] Language-guided Open-world Video Anomaly Detection under Weak Supervision](../../ICLR2026/video_understanding/language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](../../CVPR2026/video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)

</div>

<!-- RELATED:END -->
