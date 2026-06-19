---
title: >-
  [论文解读] Factorized Learning for Temporally Grounded Video-Language Models
description: >-
  [ICCV 2025][视频理解][video-language model] 提出D2VLM框架，通过将视频理解分解为"先定位证据再基于证据生成回答"的范式，引入证据token捕捉事件级视觉语义，并设计分解式偏好优化(FPO)同时提升时序定位和文本回答能力。 视频语言模型在视频理解中展现出巨大潜力，但在精确的时序定位方面仍…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "video-language model"
  - "temporal grounding"
  - "preference optimization"
  - "evidence token"
  - "factorized learning"
---

# Factorized Learning for Temporally Grounded Video-Language Models

**会议**: ICCV 2025  
**arXiv**: [2512.24097](https://arxiv.org/abs/2512.24097)  
**代码**: [https://github.com/nusnlp/d2vlm](https://github.com/nusnlp/d2vlm)  
**领域**: 视频理解  
**关键词**: video-language model, temporal grounding, preference optimization, evidence token, factorized learning

## 一句话总结

提出D2VLM框架，通过将视频理解分解为"先定位证据再基于证据生成回答"的范式，引入证据token捕捉事件级视觉语义，并设计分解式偏好优化(FPO)同时提升时序定位和文本回答能力。

## 研究背景与动机

视频语言模型在视频理解中展现出巨大潜力，但在精确的时序定位方面仍面临挑战。作者观察到视频理解中两个核心任务存在逻辑层级关系：

**时序定位**是**文本回答**的基础——准确定位时序证据是生成可靠文本响应的前提

然而现有方法（如E.T.Chat、LITA、VTG-LLM）存在两个主要局限：

**目标耦合**：各种特殊token与文本token混合生成，缺乏清晰的逻辑结构，导致学习目标耦合

**忽视视觉语义**：现有特殊token（如时间戳token）主要关注时间戳的精确表示，缺乏对被定位事件的视觉语义的显式捕捉。而这些视觉语义本应作为后续文本回答生成的关键上下文

核心思路：从分解学习的角度，将视频理解显式拆分为"时序证据定位"和"基于证据的文本回答"两个任务，并设计证据token来桥接二者。

## 方法详解

### 整体框架

D2VLM将模型响应分解为两个阶段：(1) 纯时序定位阶段——定位和捕捉用于回答的视觉证据；(2) 交错文本-证据回答阶段——以证据引用方式生成包含时间信息和文本描述的回答。基于EVA-CLIP ViT-G/14视觉编码器、Q-Former特征压缩器和Phi-3-Mini-3.8B作为基座LLM。

### 关键设计

1. **证据token (Evidence Token, \<evi\>)**: 一种专用于时序定位的特殊token，不仅确定被定位事件的时间位置，还显式捕捉事件级视觉语义。当LLM生成\<evi\> token时，计算其与每帧LLM处理后视频token $\tilde{F}_V$的相似度，将高相似度帧的视觉语义聚合到\<evi\> token中（平均池化后相加）。公式：定位损失$L_{gnd}^{<evi>} = \frac{1}{T}\sum_{t=1}^{T}BCE(y^t, sim^t)$，一致性约束$L_{cons} = \frac{1}{K}\sum_{k=1}^{K}|F_{<evi>_k}^{S_1} - F_{<evi>_k}^{S_2}|$。设计动机：让\<evi\> token真正承载事件的视觉含义，在自回归范式下为后续文本生成提供实质性上下文。

2. **分解式偏好优化 (Factorized Preference Optimization, FPO)**: 将DPO扩展为同时处理时序定位和文本响应的分解优化。关键创新是显式建模定位概率：对每个\<evi\>_k token，其对时间区间$[s_k, e_k]$的定位概率为$$p_g([s_k,e_k]) = \prod_{t=1}^{T}\begin{cases}sim_k^t & \text{if } s_k \leq t \leq e_k \\ 1-sim_k^t & \text{otherwise}\end{cases}$$ FPO的log概率公式$\log\pi(R)$在标准token预测项基础上增加了显式的时序定位建模项。设计动机：标准偏好优化无法直接处理基于相似度的定位任务，FPO通过概率建模使定位能力也可进行偏好学习。

3. **分解式偏好数据合成 (Factorized Preference Data Synthesis)**: 通过对原始响应施加分解式扰动来生成负样本。扰动分为两类：时序定位扰动（时间偏移、随机增删事件、合并事件）和文本响应扰动（关键信息篡改、重复响应）。在子视频事件级别施加扰动，确保噪声来源可控。基于E.T. Instruct 164K数据集合成。设计动机：现有视频偏好数据缺乏时序定位注释，且通过输入退化方式生成的负样本质量不可控。

### 损失函数 / 训练策略

SFT阶段损失：$L = L_{sft} + L_{gnd} + L_{cons}$

- $L_{sft}$：标准token分类损失
- $L_{gnd}$：\<evi\> token与视频帧的定位BCE损失（两个阶段取平均）
- $L_{cons}$：两阶段\<evi\> token一致性L1损失

FPO阶段：在SFT模型基础上，使用合成偏好数据进行分解式偏好优化。训练在4×H100 GPU上1天内完成，使用LoRA微调。1 FPS帧采样，224×224分辨率。

## 实验关键数据

### 主实验

**E.T. Bench Grounding (5个子任务平均)**：

| 方法 | 参数量 | TVG F1 | EPM F1 | TAL F1 | EVS F1 | VHD F1 | Avg F1 |
|------|--------|--------|--------|--------|--------|--------|--------|
| TimeChat-7B | 7B | 26.2 | 3.9 | 10.1 | 29.1 | 40.5 | 22.0 |
| E.T.Chat-3.8B | 3.8B | 38.6 | 10.2 | 30.8 | 25.4 | 62.5 | 33.5 |
| Qwen2.5-VL-7B | 7B | 46.6 | 9.3 | 32.2 | 19.9 | 68.6 | 35.3 |
| **D2VLM-3.8B** | **3.8B** | **60.2** | **14.4** | **33.4** | **35.2** | **68.2** | **42.3** |

**Charades-STA 时序定位**：

| 方法 | R@1(IoU=0.5) | R@1(IoU=0.7) |
|------|-------------|-------------|
| TRACE-7B | 40.3 | 19.4 |
| VideoChat-T-7B | 48.7 | 24.0 |
| E.T.Chat-3.8B | 45.9 | 20.0 |
| **D2VLM-3.8B** | **50.3** | **26.0** |

**YouCook2 稠密视频描述**：

| 方法 | F1 | CIDEr | SODA_c |
|------|-----|-------|--------|
| TRACE-7B | 22.4 | 8.1 | 2.2 |
| **D2VLM-3.8B** | **26.4** | **10.6** | **3.2** |

### 消融实验

**生成目标设计**：

| 配置 | Grounding Avg F1 | Dense Cap Avg F1 | Dense Cap Avg Sim |
|------|-----------------|-----------------|-------------------|
| 基线 (耦合) | 21.2 | 14.3 | 11.3 |
| +分解目标 | 28.9 | 23.1 | 16.0 |
| +交错文本-evi生成 | 35.6 | 34.3 | 19.8 |
| +一致性约束 | 39.5 | 35.0 | 21.2 |
| +FPO | **42.3** | **37.5** | **21.8** |

**证据token设计**：

| 设计 | Grounding Avg F1 | Dense Cap F1 | Dense Cap Sim |
|------|-----------------|-------------|---------------|
| 无事件级建模 | 26.1 | 33.4 | 16.2 |
| 无视觉语义捕捉 | 37.1 | 27.5 | 17.7 |
| **完整设计** | **39.5** | **35.0** | **21.2** |

### 关键发现

- **分解目标 vs 耦合目标**：分解后定位提升7.7% F1，文本描述提升4.7% Sim，证明解耦的重要性
- **交错文本-evi生成是关键**：引入"证据引用"方式生成回答，定位+6.7%、文本+3.8%，强化了定位-回答的依赖关系
- **事件级 > 帧级**：事件级建模比帧级时间戳建模在定位上+11.0% F1
- **显式视觉语义捕捉对描述至关重要**：没有视觉语义捕捉，稠密描述F1下降7.5%、Sim下降3.5%
- 3.8B模型超越多数7-13B模型，证明设计优于规模

## 亮点与洞察

1. **正确识别了逻辑层级**：不是简单拼接定位和描述，而是建立"定位→回答"的因果链，契合teacher-forcing训练范式
2. **证据token的双重角色**：既作为生成token参与自回归解码，又作为查询token进行相似度定位和语义聚合，通过MLP投影解耦两种功能
3. **概率化定位建模**：将基于相似度的连续定位转化为可参与偏好优化的概率量，是FPO的关键技术贡献
4. **分解式数据合成的可控性**：噪声来源精确可知，无需人工过滤，保证偏好数据质量

## 局限与展望

1. 某些任务上绝对性能仍有限（如EPM F1仅14.4%，YouCook2 F1仅26.4%）
2. 分解式数据合成仅关注负样本生成，缺乏多样化的正样本增强
3. 当前仅探索了3.8B规模的模型，更大规模下的行为未知
4. 1 FPS采样可能丢失细粒度事件，限制了精确时序定位
5. 可探索将FPO扩展到多轮对话式视频问答场景

## 相关工作与启发

- **E.T.Chat/E.T. Bench**: 提出综合时序定位评估基准和Instruct数据集，是直接基线和数据来源
- **LITA**: 设计时间token用于精确时间戳表示，但缺乏事件级语义捕捉
- **DPO**: 标准偏好优化算法，本文将其扩展为支持时序定位的FPO
- **TRACE**: 结合特殊token和额外定位解码器，但仍使用耦合生成目标

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 分解学习视角新颖，FPO将偏好优化推广到时序定位领域，概率化建模巧妙
- **实验充分度**: ⭐⭐⭐⭐ 覆盖多种任务和数据集，逐组件消融清晰完整
- **写作质量**: ⭐⭐⭐⭐⭐ 逻辑层次分明，问题→解决→验证的叙事线索极为清晰
- **价值**: ⭐⭐⭐⭐⭐ 以更小的模型超越SOTA，FPO和证据token对视频LLM社区有广泛参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision](toga_temporally_grounded_open-ended_video_qa_with_weak_supervision.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](../../NeurIPS2025/video_understanding/sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](../../CVPR2025/video_understanding/efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)

</div>

<!-- RELATED:END -->
