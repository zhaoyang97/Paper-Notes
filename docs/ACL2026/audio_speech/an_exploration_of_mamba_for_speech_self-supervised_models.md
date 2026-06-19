---
title: >-
  [论文解读] An Exploration of Mamba for Speech Self-Supervised Models
description: >-
  [ACL 2026][音频/语音][Mamba] 首次全面探索Mamba架构作为语音自监督学习（SSL）基础模型的潜力，发现Mamba-based HuBERT在长上下文ASR、流式ASR和因果设置的probing任务中优于Transformer，同时保持线性时间复杂度。 领域现状：Transformer-based语音SS…
tags:
  - "ACL 2026"
  - "音频/语音"
  - "Mamba"
  - "语音自监督学习"
  - "HuBERT"
  - "状态空间模型"
  - "流式ASR"
---

# An Exploration of Mamba for Speech Self-Supervised Models

**会议**: ACL 2026  
**arXiv**: [2506.12606](https://arxiv.org/abs/2506.12606)  
**代码**: [GitHub](https://github.com/hckuo145/Mamba-based-HuBERT)  
**领域**: Speech / Self-Supervised Learning  
**关键词**: Mamba, 语音自监督学习, HuBERT, 状态空间模型, 流式ASR

## 一句话总结

首次全面探索Mamba架构作为语音自监督学习（SSL）基础模型的潜力，发现Mamba-based HuBERT在长上下文ASR、流式ASR和因果设置的probing任务中优于Transformer，同时保持线性时间复杂度。

## 研究背景与动机

**领域现状**：Transformer-based语音SSL模型（如HuBERT, wav2vec 2.0）取得了巨大成功，但其二次方复杂度在长序列处理时造成高计算成本和内存瓶颈。

**现有痛点**：(1) Mamba在语言建模中已展现出超越Transformer的能力，但在语音领域的应用仅限于单一任务的孤立研究；(2) 现有语音Mamba工作通常报告与Transformer相当甚至略差的性能，且常需要混合设计；(3) 缺乏统一的跨任务评估。

**核心矛盾**：Mamba的线性时间复杂度理论上非常适合语音的长序列特性，但其在语音SSL中的综合表现尚不明确。

**本文目标**：系统训练和评估Mamba-based HuBERT模型，全面探索其作为语音基础模型和特征提取器的潜力。

**切入角度**：用Mamba block替换HuBERT中的Transformer block，保持相同的训练流程（两轮迭代k-means伪标签训练），在ASR、SUPERB等多任务上评估。

**核心 idea**：Mamba天然的因果架构使其特别适合构建因果语音SSL模型，在流式ASR和长上下文场景中展现独特优势。

## 方法详解

### 整体框架

本文不发明新架构，而是做一项受控替换实验：把 HuBERT 中的 Transformer block 原样换成 Mamba block，CNN 特征编码器与位置编码器保持不变，训练流程也完全沿用 HuBERT 的两轮迭代（第一轮以 MFCC 为目标训练 250k 步，第二轮以第一轮第 6 层输出为伪标签训练 400k 步），在 LibriSpeech 960h 上预训练。这样唯一变量就是 backbone，输入语音 → Mamba 编码 → SSL 表示 → 下游 ASR/SUPERB 探针的全链路里，任何性能差异都可干净地归因到 Mamba 与 Transformer 的本质区别上。

### 关键设计

**1. 多种 Mamba 变体的系统对比：厘清因果性到底是 Mamba 的优势还是包袱**

Mamba 天生因果，这一性质在不同任务里方向相反——流式 ASR 只能看过去信息，因果是优势；而需要全局上下文的任务里，单向又可能是劣势。为把这条边界测清楚，本文同时评估因果设置（Mamba、Mamba+MLP）和双向设置（ExtBiMamba、InnBiMamba），并逐一与参数量相当的 Transformer 变体公平对照。这种成对设计让"因果 vs 双向"和"Mamba vs Transformer"两个维度可以解耦分析，而不是只给一个笼统的好坏结论。

**2. 长上下文和流式 ASR 评估：把 Mamba 线性复杂度的理论优势落到可测场景**

Mamba 相对 Transformer 最大的卖点是 $O(n)$ 而非 $O(n^2)$，但这个优势只有在长序列下才显形。为此设计两个针对性场景：长上下文 ASR 直接处理整段未切分语音，流式 ASR 则约束模型只能用过去信息逐帧解码；同时量化 MACs/秒与 RTF 随序列长度的变化曲线。结果正是在这里拉开差距——Transformer 在 80 秒以上即 OOM 无法运行，而 Mamba 计算量近乎恒定，可处理 5 分钟以上的语音。

**3. 表示质量分析：不止问"好不好"，还要拆开看"好在哪、为什么好"**

仅靠下游 WER 无法解释 Mamba 表示的内在特性，本文进一步做表示层面的剖析：用 phone purity 量化表示的语音学纯度，用 CCA（典型相关分析）刻画音素与说话人特征各自被编码的方式。借此发现 Mamba 的量化表示 phone purity 更高、对说话人信息编码更清晰，这对以 SSL units 为输入的 spoken language models 有直接价值，把单纯的性能数字升级成了可解释的表示学特性。

### 损失函数 / 训练策略

遵循 HuBERT 标准训练目标：masked prediction loss。使用 Adam 优化器，学习率先线性 warm-up（前 8%）再线性 decay。受计算资源限制，仅在单块 V100 上训练，batch size 取原始配置的 1/4。

## 实验关键数据

### 主实验

| 设置 | 模型 | 参数量 | WER | 关键发现 |
|------|------|--------|-----|--------|
| 流式ASR | Mamba HuBERT | 78M | 15.77% | 优于94M因果Transformer(16.66%) |
| 长上下文ASR | ExtBiMamba | - | 11.08% | Transformer因OOM无法运行 |
| 标准ASR | ExtBiMamba(Small) | - | 接近Transformer | 小规模有效 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 因果SUPERB | Mamba > Causal Transformer | 在音素和说话人任务上更优 |
| Phone Purity | Mamba更高 | 量化表示的语音质量更好 |
| CCA分析 | 说话人特征更distinct | Mamba对说话人信息编码更清晰 |
| ExtBiMamba Base | 低于Transformer | 大规模双向Mamba仍需改进 |

### 关键发现
- Mamba的因果性质是流式语音场景的天然优势——78M参数优于94M的因果Transformer
- 计算成本随序列长度几乎恒定，而Transformer在80秒以上OOM
- Mamba产生的量化表示phone purity更高，有利于以SSL units为输入的spoken language models
- 大规模双向Mamba（Base）仍全面低于Transformer，暗示可扩展性仍需改进

## 亮点与洞察
- 首次系统性地将Mamba作为语音基础模型进行全面评估，而非仅在单一任务上测试
- "因果性质是优势而非限制"的发现改变了对Mamba在语音中应用的认知
- 量化表示质量的发现对spoken language model领域有直接启示

## 局限与展望
- 双向Mamba的大规模训练效果不佳，可扩展性是关键挑战
- 仅在LibriSpeech上预训练和评估，多语言和噪声场景未测试
- 受限于单块V100，训练规模远小于原始HuBERT
- 未来可探索Mamba2等改进架构和更大规模的训练

## 相关工作与启发
- **vs 混合Mamba-Transformer**: 本文纯Mamba架构，更清晰地揭示Mamba的优劣势
- **vs SSAM**: SSAM关注通用音频而非语音，本文专注于语音SSL
- **vs Mamba流式ASR**: 之前的工作需要额外机制（lookahead等），本文展示纯Mamba即有优势

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次全面探索Mamba作为语音SSL基础模型
- 实验充分度: ⭐⭐⭐⭐⭐ ASR、SUPERB、表示分析、长上下文、流式等多维评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验细致
- 价值: ⭐⭐⭐⭐ 为语音领域的高效架构选择提供重要实证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[ACL 2026\] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care](exploration_of_perceptual_speech_features_for_clinical_decision-support_in_menta.md)
- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)
- [\[ACL 2026\] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)

</div>

<!-- RELATED:END -->
