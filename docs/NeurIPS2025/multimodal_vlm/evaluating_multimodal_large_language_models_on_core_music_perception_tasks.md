---
title: >-
  [论文解读] Evaluating Multimodal Large Language Models on Core Music Perception Tasks
description: >-
  [NeurIPS 2025][多模态VLM][多模态LLM] 本文通过三项核心音乐感知任务（切分节奏评分、移调检测、和弦辨识）系统性评估了多模态LLM在音频与MIDI两种输入下的表现，揭示了模型在符号推理上接近理想但在音频感知上存在显著缺陷的关键差距。 多模态基础模型（如 Qwen2.5-Omni、Gemini 2.5）宣称…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态LLM"
  - "音乐感知"
  - "音频理解"
  - "符号推理"
  - "LogicLM"
---

# Evaluating Multimodal Large Language Models on Core Music Perception Tasks

**会议**: NeurIPS 2025  
**arXiv**: [2510.22455](https://arxiv.org/abs/2510.22455)  
**代码**: 无（数据来源于 The MUSE Benchmark）  
**领域**: 多模态VLM  
**关键词**: 多模态LLM, 音乐感知, 音频理解, 符号推理, LogicLM  

## 一句话总结
本文通过三项核心音乐感知任务（切分节奏评分、移调检测、和弦辨识）系统性评估了多模态LLM在音频与MIDI两种输入下的表现，揭示了模型在符号推理上接近理想但在音频感知上存在显著缺陷的关键差距。

## 研究背景与动机
多模态基础模型（如 Qwen2.5-Omni、Gemini 2.5）宣称具备"音乐理解"能力，但现有评测（如 AIR-Bench、MMAR、MMAU 等）主要关注分类和描述任务，无法区分模型是真正理解音乐结构还是依赖表面频谱模式。音频-语言模型（SALMONN、Qwen-Audio、Audio Flamingo 2）在语音和声音识别上表现出色，但在音乐特有的关系属性上从未被系统测试。

本文的核心动机是：**将感知（perception）与推理（reasoning）分离**。模型在MIDI上表现好不代表它能"听"音频——因为MIDI剥离了微观时序、演奏力度、表情细微差别等真正使音乐有意义的特征。

## 方法详解

### 整体框架
作者设计了一个 3×3×2 的因子实验：
- **3种任务**：切分节奏评分（Syncopation Scoring）、移调检测（Transposition Detection）、和弦质量辨识（Chord Quality Identification）
- **3种推理策略**：Standalone（直接回答）、CoT（链式思维）、LogicLM（符号推理+确定性求解器）
- **2种输入模态**：Audio（原始音频波形）和 MIDI（符号化文本表示）
- 交叉 ZS（zero-shot）和 FS（few-shot），共 12 个条件/任务

### 关键设计

**三项核心任务的设计理由：**

1. **切分节奏评分**：20段8秒节奏片段（120BPM），hi-hat保持恒定八分音符，kick和snare在正拍/反拍变化。任务是数反拍kick/snare事件并映射到分类评分（0/2/4/6/8）。该任务**考察节奏期望违反的敏感度和度量位移感知**。

2. **移调检测**：20对乐段（均≈9秒），第二段要么是原旋律移调要么是不同旋律。任务是二分类（相同/不同旋律）。该任务**考察与绝对音高无关的旋律识别不变性**，这是人类跨调性识别旋律的核心感知能力。

3. **和弦质量辨识**：44段9秒片段（120BPM），每段包含块状和弦+上行琶音。四分类：大三和弦、小三和弦、属七和弦、减三和弦。该任务**考察音程模式识别而非绝对频率匹配**。

**LogicLM 适配音乐领域：**
- 改编自 Pan et al. 的 LogicLM 框架，让模型作为"感知公式化器"（Perceptual Formulators），生成可被机器检查的符号 schema
- 确定性求解器（solver.py）执行 schema，防止"不忠实推理"——即正确答案掩盖了错误的感知分析
- 当 schema 格式违规时，自动触发自修复循环让模型修正输出

**刺激材料构造：**
- 所有刺激均为真人音乐家录制的原创录音（来自 MUSE Benchmark）
- MIDI 版本通过 MIDI 键盘重新演奏并用 mido 导出为 .txt
- 音频与 MIDI 的 prompt 仅将"你将听到..."替换为"你将获得MIDI数据..."，保持 schema 不变

### 损失函数 / 训练策略
本文为评估性工作，不涉及模型训练。三个被评估的模型为：
- **Gemini 2.5 Pro** 和 **Gemini 2.5 Flash**：通过 google.genai SDK 调用
- **Qwen2.5-Omni 7B**：在 NYU HPC 上运行，使用相同 prompt、解码设置和评估流程
- 所有运行确定性（temperature=0），每次试验独立（无跨试验历史）

## 实验关键数据

### 主实验

| 模态 | Shot | 策略 | Flash(切分) | Pro(切分) | Qwen(切分) | Flash(移调) | Pro(移调) | Qwen(移调) | Flash(和弦) | Pro(和弦) | Qwen(和弦) |
|------|------|------|-------------|-----------|------------|-------------|-----------|------------|-------------|-----------|------------|
| Audio | ZS | Stand. | 30.0 | 25.0 | 20.0 | 55.6 | **94.7** | 75.0 | 31.8 | **47.7** | 31.8 |
| Audio | ZS | CoT | 35.0 | 25.0 | 20.0 | 76.9 | **95.0** | 65.0 | 31.8 | 43.2 | 31.8 |
| Audio | ZS | LogicLM | 20.0 | 20.0 | 20.0 | 65.0 | 80.0 | 50.0 | 11.4 | 18.2 | 6.8 |
| MIDI | ZS | Stand. | 84.2 | **95.0** | 25.0 | **100** | **100** | 85.0 | 50.0 | **97.7** | 22.7 |
| MIDI | ZS | CoT | 94.7 | **100** | 35.0 | 95.0 | **100** | 20.0 | **100** | **100** | 25.0 |
| MIDI | ZS | LogicLM | 90.0 | 80.0 | 20.0 | **100** | **100** | 10.0 | 93.2 | **100** | **100** |

**核心发现——模态差距巨大**：Gemini 模型在 MIDI 上远优于 Audio（p<0.001），切分和和弦任务差距最大（MIDI≈84-100% vs Audio≈6-65%）。

### 消融实验

| 因素 | 发现 |
|------|------|
| ZS vs FS | 无显著主效应（所有p>0.05）；FS仅在音频切分任务中帮助Pro（~25%→~65%） |
| 推理策略 | 切分：CoT在音频下略有增益，LogicLM仅在MIDI下有效；移调：Standalone/CoT最佳，LogicLM降低准确率；和弦：LogicLM-Audio因schema脆弱性崩溃 |
| 模型差异 | Gemini Pro总体最优；Qwen2.5-Omni全面落后，尤其在LogicLM下缺陷最大 |

### 关键发现
- **移调任务中的虚假成功**：Gemini Pro 经常保留正确序列长度但未能捕捉音程结构和轮廓方向（如Ground Truth的↓↑↑↓... vs模型输出的↑↓↓↑...）
- **LogicLM 暴露退化策略**：Standalone/CoT 可以掩盖感知错误，但 LogicLM 强制要求音乐一致性，揭示模型的真实感知缺陷
- **和弦辨识中的混淆**：音频模式下，相近和弦质量（如大三和弦 vs 属七和弦）容易混淆

## 亮点与洞察
1. **实验设计精巧**：Audio vs MIDI 的对比精确隔离了感知与推理能力，这是现有音乐AI评测从未做到的
2. **LogicLM适配音乐领域**：将神经-符号推理框架引入音乐认知评测，暴露了模型表面成功下的深层感知缺陷
3. **实践意义**：当前系统不应宣称"音乐理解"——MIDI上的天花板表现不等于音频原生能力

## 局限与展望
- 仅测试3个模型，未包含专门的音乐基础模型（如 MusicGen、Jukebox 等）
- 刺激材料规模较小（20-44段/任务），统计功效有限
- 未考虑更复杂的音乐任务（如多声部分析、曲式结构、情感表达）
- MIDI作为符号代理过度简化，可考虑 MusicXML 等更丰富格式
- 暂未量化音频前端的具体瓶颈（频谱分析能力 vs 时序追踪能力）

## 相关工作与启发
- 可与 AIR-Bench、MMAU 等通用音频评测互补，本文填补了"结构理解"维度的空白
- LogicLM 的适配思路可推广到其他需要"感知+推理"分离的多模态领域
- 对音乐AI应用（播放列表推荐、音乐教育）有直接指导意义——必须构建"audio-first"系统

## 评分
- 新颖性: ⭐⭐⭐⭐（首次系统性分离音乐感知与推理，LogicLM适配新颖）
- 实验充分度: ⭐⭐⭐（因子设计合理但数据规模偏小，模型覆盖不够广）
- 写作质量: ⭐⭐⭐⭐（结构清晰，论证有力）
- 价值: ⭐⭐⭐⭐（对音乐AI领域有重要警示价值，揭示感知瓶颈）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)
- [\[CVPR 2026\] DEVA: Fine-tuning Multimodal Large Language Models for Visual Perception Tasks](../../CVPR2026/multimodal_vlm/deva_fine-tuning_multimodal_large_language_models_for_visual_perception_tasks.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[ICCV 2025\] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/visnumbench_evaluating_number_sense_of_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
