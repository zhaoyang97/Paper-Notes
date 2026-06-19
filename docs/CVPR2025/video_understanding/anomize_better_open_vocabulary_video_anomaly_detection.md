---
title: "Anomize: Better Open Vocabulary Video Anomaly Detection"
description: "提出Anomize框架，通过LSTM时序编码+GPT-4分组引导文本编码+文本增强双流架构，实现开放词汇视频异常检测，XD-Violence AP达69.31%"
tags:
  - CVPR2025
  - LLM预训练
  - 开放词汇
  - 大语言模型
  - 时序建模
---

# Anomize: Better Open Vocabulary Video Anomaly Detection

**会议**: CVPR 2025  
**机构**: 武汉大学 / 复旦大学 / 北京大学  
**关键词**: 视频异常检测、开放词汇、LSTM、GPT-4、双流架构  

## 研究背景与动机

视频异常检测（Video Anomaly Detection, VAD）旨在从视频中识别偏离正常模式的事件。传统方法大多在封闭词汇设定下工作，即训练时已知所有可能的异常类别。然而在真实场景中，异常事件的种类是无限的——新型犯罪手法、罕见事故类型、突发公共事件等都无法预先枚举。

**开放词汇视频异常检测**是更实际的设定：模型不仅要检测已知类别的异常，还要能识别和分类从未见过的新异常类型。这个问题面临三大挑战：

**异常描述的模糊性**：同一类异常（如"暴力"）在不同上下文中表现差异极大，简单的类别名称无法捕捉这种多样性。

**时序建模不足**：CLIP等视觉-语言模型擅长单帧理解，但异常通常需要时序上下文（如"突然加速后撞车"）。

**已知与未知的泛化**：如何在已知异常上训练，同时保持对新异常的检测能力？

现有方法（如LAVAD、OVVAD）直接使用CLIP特征做异常检测，没有充分利用大语言模型的知识来丰富异常的文本描述，也缺乏有效的时序建模机制。

## 方法详解

### 整体框架

Anomize采用文本增强的双流架构，将视频异常检测分解为两个互补的流：动态流捕获时序演变模式，静态流匹配概念级语义。

### 组件1：LSTM时序编码器

传统方法直接在CLIP帧特征上做分类，丢失了时序信息。Anomize引入双向LSTM对帧序列进行编码：

$$h_t = 	ext{BiLSTM}([\overrightarrow{h_t}; \overleftarrow{h_t}]) = 	ext{BiLSTM}(f_{	ext{CLIP}}(I_t), h_{t-1})$$

LSTM的隐状态积累了历史信息，使模型能理解"正常行走→突然奔跑→撞击"这样的时序模式。

### 组件2：GPT-4分组引导文本编码

这是Anomize最创新的部分。传统方法使用固定的类别名称（如"fighting"）作为文本查询，但这过于简略。Anomize使用GPT-4通过三步生成丰富的异常描述：

**Step 1 - Group（分组）**：将异常类别按语义相似度分组
- 例如：{打架, 抢劫, 枪击} → "人际暴力"组

**Step 2 - Describe（描述）**：GPT-4为每组生成多角度的详细描述
- "人际暴力"→ 描述视觉特征、时序模式、环境线索等

**Step 3 - Encode（编码）**：将描述通过CLIP文本编码器转换为特征向量

这种分组-描述策略产生了更丰富、更具区分性的文本表征。

### 组件3：文本增强双流架构

**动态流**：LSTM编码的时序特征 + Text Augmenter模块
- Text Augmenter通过交叉注意力将文本描述信息注入视频特征
- 输出动态异常分数：$s_{	ext{dyn}} = 	ext{MLP}(	ext{CrossAttn}(h_t, T_{	ext{desc}}))$

**静态流**：概念库 + TopK匹配
- 预构建异常概念库（每个异常类别的多个描述特征）
- 对每帧特征，计算与概念库中所有特征的余弦相似度
- 取TopK最高相似度的平均作为静态异常分数

最终分数：$s = lpha \cdot s_{	ext{dyn}} + (1-lpha) \cdot s_{	ext{static}}$

### 两阶段训练

| 阶段 | 任务 | Epoch | 学习率 | 目的 |
|------|------|-------|--------|------|
| 阶段1 | 异常分类 | 16 | 1e-4 | 学习区分不同异常类别 |
| 阶段2 | 异常检测 | 64 | 5e-5 | 学习区分正常vs异常 |

阶段1使用分类损失训练动态流和Text Augmenter，阶段2使用MIL（Multiple Instance Learning）损失微调整个架构。

## 实验结果

### 主要结果

| 方法 | XD-Violence AP | XD-Violence Acc | UCF-Crime AUC |
|------|---------------|-----------------|---------------|
| CLIP baseline | 43.68% | 64.68% | 78.32% |
| LAVAD | 55.40% | 79.15% | 81.20% |
| OVVAD | 61.53% | 83.76% | 82.95% |
| **Anomize** | **69.31%** | **90.29%** | **84.49%** |
| 提升 vs OVVAD | +7.78 | +6.53 | +1.54 |

### 开放词汇能力

在新异常类别分类任务上（训练时未见过的异常类型）：
- Anomize: +56.53% 相比最佳基线
- 证明了GPT-4分组描述策略为新异常提供了良好的文本锚点

### 消融实验

| 配置 | XD-Violence AP |
|------|---------------|
| 仅动态流 | 62.15% |
| 仅静态流 | 58.43% |
| 双流（无GPT-4描述） | 64.87% |
| 双流（有GPT-4描述） | **69.31%** |

GPT-4描述带来了4.44%的AP提升，证明了丰富文本描述的重要性。

## 创新点总结

1. **分组引导的文本编码**：首次将LLM生成的结构化异常描述引入VAD
2. **动态+静态双流互补**：时序LSTM捕获动态模式，概念库匹配提供静态语义
3. **两阶段训练策略**：分类→检测的课程学习确保模型先学到好的类别表征

## 局限性

- 依赖GPT-4生成描述，增加了部署成本
- LSTM的长程依赖能力有限，超长视频可能效果下降
- 概念库需要人工或LLM定义，扩展到全新领域时需要额外适配

## 总结

Anomize提出了一个端到端的开放词汇视频异常检测框架，核心创新在于利用GPT-4的知识来丰富异常的文本表征，并通过LSTM时序编码和双流架构实现了对已知和未知异常的统一检测。在XD-Violence上的25.61%准确率提升尤为令人印象深刻。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](../../ICCV2025/video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[ICLR 2026\] Language-guided Open-world Video Anomaly Detection under Weak Supervision](../../ICLR2026/video_understanding/language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)
- [\[ECCV 2024\] SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking](../../ECCV2024/video_understanding/slack_semantic_location_and_appearance_aware_open-vocabulary_tracking.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](../../NeurIPS2025/video_understanding/monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)

</div>

<!-- RELATED:END -->
