---
title: >-
  [论文解读] USE: A Unified Model for Universal Sound Separation and Extraction
description: >-
  [AAAI2026 Oral][音频/语音][通用声音分离] 提出 USE 统一框架，通过 EDA 网络推断声源数量和声学线索实现声音分离 (SS)，多模态融合网络解释用户提供的文本/视频/标签线索实现目标声音提取 (TSE)，联合训练+跨任务对齐使两项任务互相增强，SS +1.4dB SDR，TSE 匹配准确率 86%。
tags:
  - "AAAI2026 Oral"
  - "音频/语音"
  - "通用声音分离"
  - "目标声音提取"
  - "多模态融合"
  - "EDA网络"
  - "跨任务对齐"
---

# USE: A Unified Model for Universal Sound Separation and Extraction

**会议**: AAAI2026 Oral  
**arXiv**: [2512.21215](https://arxiv.org/abs/2512.21215)  
**代码**: [https://hongyuwang414.github.io/USE-demo/](https://hongyuwang414.github.io/USE-demo/)  
**领域**: 语音 / 声音分离  
**关键词**: 通用声音分离, 目标声音提取, 多模态融合, EDA网络, 跨任务对齐

## 一句话总结
提出 USE 统一框架，通过 EDA 网络推断声源数量和声学线索实现声音分离 (SS)，多模态融合网络解释用户提供的文本/视频/标签线索实现目标声音提取 (TSE)，联合训练+跨任务对齐使两项任务互相增强，SS +1.4dB SDR，TSE 匹配准确率 86%。

## 研究背景与动机

**领域现状**：声音分离 (SS) 将混合音频分为独立源，目标声音提取 (TSE) 从混合中提取用户指定的目标。两者通常单独研究。

**现有痛点**：(1) SS 需要已知声源数量，未知数量时性能大降；(2) TSE 受限于单一模态线索（仅文本或仅视频），线索质量差时失败；(3) 两任务缺乏统一框架，无法利用声学分离知识辅助目标提取。

**核心矛盾**：SS 的吸引子（学到的源表示）和 TSE 的查询线索（用户提供的模态描述）语义上应该对应同一个目标，但两个任务独立训练无法建立这种桥接。

**本文目标** 统一 SS 和 TSE 到一个框架，共享语义空间，互相增强。

**切入角度**：用 EDA 网络的吸引子作为与用户线索语义对齐的桥梁。

**核心 idea**：EDA 吸引子和多模态线索在共享语义空间中对齐，统一分离和提取。

## 方法详解

### 整体框架
编码器-分离器-解码器骨干 + 两个辅助网络：(1) EDA 网络（用于 SS：推断声源数量+生成吸引子）；(2) 多模态线索网络（用于 TSE：融合文本/视频/标签）。两者通过跨任务对齐损失桥接。

### 关键设计

1. **EDA 网络（Encoder-Decoder Attractor）**:

    - 功能：自回归生成声源吸引子，同时推断声源数量
    - 核心思路：LSTM 编码器处理帧级嵌入→LSTM 解码器逐步生成吸引子 $\mathbf{a}_s$。每个吸引子有存在概率 $p_{\text{exi}} = \sigma(\mathbf{w}^\top \mathbf{a}_s + b)$，阈值 0.5 判断声源是否存在
    - 设计动机：解决未知声源数量问题——自回归生成直到存在概率低于阈值

2. **多模态线索网络**:

    - 功能：融合文本（DistilBERT）、视频（Swin Transformer）、声音标签（one-hot 嵌入）
    - 核心思路：各模态编码后沿时间维拼接，通过多头注意力（分离器特征做 Query）融合
    - 设计动机：多模态冗余——即使某个线索缺失/质量差，其他模态可补偿

3. **跨任务对齐损失**:

    - 功能：将 EDA 吸引子和用户线索映射到共享语义空间
    - 核心思路：$\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{InfoNCE}}$，用 PIT 确定的最优排列对齐吸引子和线索
    - 设计动机：使 SS 学到的声学表示与 TSE 的语义查询对齐，实现统一

### 损失函数 / 训练策略
两阶段训练：Stage 1 仅 SS+EDA（70 epochs, lr=1e-4）；Stage 2 联合 SS+TSE，以 30:70 比例随机选 EDA 或线索网络（30 epochs, lr=3e-5）。

## 实验关键数据

### 主实验

| 任务 / 模型 | 2Mix SI-SNRi↑ | 3Mix SI-SNRi↑ |
|------------|--------------|--------------|
| Libri2Mix: TDANet | 17.5 | - |
| **Libri2/3Mix: USE-B** | **17.8** | **15.0** |
| AudioSet SS: Sepformer | 7.4 | - |
| **AudioSet SS: USE-S (stage2)** | **8.8** | **7.2** |
| FUSS: TDCN++ | 11.2/11.6/7.4 | |
| **FUSS: USE-B** | **12.8/13.1/11.9** | |

### TSE 多模态对比

| 线索组合 | DCCRN SNRi | USE-B SNRi |
|---------|-----------|-----------|
| tag+text+video | 6.9 | **8.9** (+29%) |
| text only | 6.3 | **8.0** (+27%) |
| video only | 5.8 | **6.2** (+7%) |

### 关键发现
- 联合训练（Stage 2）在 SS 上比单独训练（Stage 1）进一步提升（AudioSet unseen 3Mix: 5.2→6.3 dB），说明 TSE 的语义知识反哺了 SS
- 吸引子-线索匹配准确率 86%（2Mix），验证了共享语义空间的有效性
- 未知声源数量时 USE-B* 性能几乎无损（17.7 vs 17.8），EDA 声源计数准确率 >80%
- 多模态线索冗余——tag+text 与 tag+text+video 效果接近（8.6 vs 8.9），视频贡献有限

## 亮点与洞察
- **吸引子-线索对齐是核心创新**——在两个看似不同的任务之间建立了语义桥梁。这个思路可迁移到任何"自动发现"与"用户指定"共存的场景
- **联合训练的双向增强**：SS 帮 TSE 学更好的分离，TSE 帮 SS 学语义感知——正向循环

## 局限与展望
- 3Mix 及以上场景下 EDA 声源计数准确率下降（65.3%），限制了复杂场景应用
- 视频线索贡献有限——可能需要更好的视频编码器或视频-音频时序对齐
- 仅在 AudioSet 类的通用声音上测试，音乐分离未验证

## 相关工作与启发
- **vs DCCRN**: 传统 TSE 方法，USE-B 在全部线索组合上超越 29%
- **vs TDANet**: 仅做 SS，USE-B 在 2Mix 上持平（17.8 vs 17.5）但额外支持 3Mix 和 TSE

## 评分
- 新颖性: ⭐⭐⭐⭐ SS+TSE 的统一框架+吸引子-线索语义对齐是有效新设计
- 实验充分度: ⭐⭐⭐⭐⭐ 多个数据集+SS/TSE/多模态+声源计数+匹配准确率
- 写作质量: ⭐⭐⭐⭐ 架构清晰，训练策略描述详细
- 价值: ⭐⭐⭐⭐ 实用的统一声音处理框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation](../../ICML2026/audio_speech/a_semantically_consistent_dataset_for_data-efficient_query-based_universal_sound.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](../../ACL2026/audio_speech/unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)
- [\[ICML 2026\] Attend to Anything: Foundation Model for Unified Human Attention Modeling](../../ICML2026/audio_speech/attend_to_anything_foundation_model_for_unified_human_attention_modeling.md)
- [\[ICLR 2026\] AlignSep: Temporally-Aligned Video-Queried Sound Separation with Flow Matching](../../ICLR2026/audio_speech/alignsep_temporally-aligned_video-queried_sound_separation_with_flow_matching.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)

</div>

<!-- RELATED:END -->
