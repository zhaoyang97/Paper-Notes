---
title: >-
  [论文解读] Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion
description: >-
  [ACL 2025][音频/语音][AI生成歌词检测] 提出 DE-detect，一个仅以音频为输入的多视角晚期融合管线，通过结合自动转录歌词的文本特征和语音模型提取的歌词相关音频特征，实现了对 AI 生成歌词的鲁棒检测，在域内外均优于单模态方法。 领域现状：AI 音乐生成工具（如 Suno、Udio）正在革新音乐产业…
tags:
  - "ACL 2025"
  - "音频/语音"
  - "AI生成歌词检测"
  - "多模态融合"
  - "语音嵌入"
  - "鲁棒检测"
  - "多视角融合"
---

# Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion

**会议**: ACL 2025  
**arXiv**: [2506.15981](https://arxiv.org/abs/2506.15981)  
**代码**: [https://github.com/deezer/robust-AI-lyrics-detection](https://github.com/deezer/robust-AI-lyrics-detection)  
**领域**: 语音  
**关键词**: AI生成歌词检测, 多模态融合, 语音嵌入, 鲁棒检测, 多视角融合

## 一句话总结

提出 DE-detect，一个仅以音频为输入的多视角晚期融合管线，通过结合自动转录歌词的文本特征和语音模型提取的歌词相关音频特征，实现了对 AI 生成歌词的鲁棒检测，在域内外均优于单模态方法。

## 研究背景与动机

**领域现状**：AI 音乐生成工具（如 Suno、Udio）正在革新音乐产业，但也给版权保护和内容审核带来巨大挑战。现有 AI 生成音乐（AIGM）检测方法主要分为基于音频和基于歌词两类。

**现有痛点**：(1) 基于音频的检测器虽然域内准确率高达 99%+，但对新生成器泛化极差，且对音高变化、噪声等音频攻击高度敏感；(2) 基于歌词的检测器需要干净、格式化的歌词文本，但在实际部署中只有音频可用，歌词元数据通常不可获取。

**核心矛盾**：歌词检测依赖不可得的干净文本，音频检测又对底层伪影过度敏感，两类方法都无法在真实场景中可靠工作。

**本文目标**：设计一个仅以音频为输入、既能利用歌词语义信息又能捕捉歌词相关音频线索的鲁棒 AIGM 检测系统。

**切入角度**：将音频同时视为两种模态——通过 ASR 转录获取歌词文本（what），通过语音模型捕捉韵律、语调等歌词相关的声学信息（how），然后进行晚期融合。

**核心 idea**：用多视角晚期融合将自动转录歌词（语义内容）和语音嵌入（声学线索）结合，实现鲁棒的仅音频 AI 歌词检测。

## 方法详解

### 整体框架

DE-detect 是一个模块化的晚期融合管线，整体流程为：

1. **输入**：仅音频信号
2. **文本分支**（上通道）：ASR 模型（Whisper large-v2）将音频转录为歌词 → 文本嵌入模型（LLM2Vec + Llama3 8B）生成歌词语义表示
3. **语音分支**（下通道）：语音模型（XEUS）直接从音频提取歌词相关的声学特征（韵律、语调、说话者特征等）
4. **晚期融合**：两个分支特征分别线性投影到 128 维 → 拼接 → MLP 分类器判断真/假

### 关键设计

#### 文本分支

- **功能**：将音频自动转录为歌词文本，再提取语义特征
- **核心思路**：使用 Whisper large-v2 做 ASR 转录，然后用 LLM2Vec（基于 Llama3 8B）提取整段歌词的上下文化语义嵌入
- **设计动机**：解决歌词不可用的问题，转录歌词虽有误差（WER 约 20-40%），但文本检测器对此具有鲁棒性。实验表明 Whisper large-v2 在检测任务上效果最优，说明更低的 WER 不一定对应更好的检测性能

#### 语音分支

- **功能**：从音频中提取歌词相关的声学信息，捕捉 AI 生成的韵律和发声特征
- **核心思路**：使用 XEUS 语音模型对音频做均值池化得到单一向量表示
- **设计动机**：转录只能捕获"说了什么"（what），而语音嵌入能捕获"怎么说的"（how），如韵律、语调、说话者特征等。XEUS 因为训练数据包含唱歌声音，表现最优（recall 92.2%）。实验还表明 XEUS 在区分真实/部分伪造音频时接近随机水平（50.5%），说明其特征不依赖音频伪影

#### 晚期融合设计

- **功能**：将文本和语音两个分支的特征融合进行最终分类
- **核心思路**：两个分支特征各自线性投影到 128 维，拼接后送入 MLP，用二元交叉熵损失训练
- **设计动机**：模块化晚期融合的优势在于：(1) 各组件可独立更新；(2) 保留各组件的优势（如多语言能力）；(3) 对组件变化具有鲁棒性。这在 AIGM 快速演变的环境下至关重要

### 损失函数/训练策略

- 使用**二元交叉熵损失**（Binary Cross-Entropy Loss）训练 MLP 分类器
- 训练数据：基于 Labrak et al. (2025) 的歌词数据集，包含 3,655 真实歌词和 3,535 AI 生成歌词（来自 3 个 LLM），覆盖 9 种语言和 6 种音乐流派
- AI 歌词的音频由 Suno v3.5 生成，真实歌词使用原始音频
- 最终数据集共 7,190 首歌曲，真假平衡

## 实验关键数据

### 主实验

| 模型 | Recall (en) | Recall (all) | AUROC (en) | AUROC (all) |
|------|-------------|--------------|------------|-------------|
| GT Lyrics (LLM2Vec) † | 91.3 | 94.3 | 99.0 | 97.3 |
| CNN (Spectrogram) ‡ | 97.5 | 97.4 | 99.9 | 99.8 |
| XEUS | 89.1 | 92.2 | 94.5 | 97.0 |
| Llama3 8B (LLM2Vec) | 90.6 | 90.7 | 97.6 | 94.8 |
| **DE-detect** | **93.9** | **94.9** | **98.2** | **98.5** |

DE-detect 在多语言宏平均 recall 达到 94.9%，AUROC 达到 98.5%，超过使用干净歌词的基线（94.3%），仅在域内略低于 CNN 频谱图方法。

### 消融实验

**域外鲁棒性评估（音频攻击 + Udio 泛化）**：

| 模型 | Stretch | Pitch | EQ | Noise | Reverb | Udio |
|------|---------|-------|----|-------|--------|------|
| CNN | 98.1 | 59.0 | 79.4 | 77.4 | 80.7 | 56.9 |
| XEUS | 92.5 | 92.3 | 92.3 | 92.4 | 92.4 | 85.9 |
| Llama3 8B | 90.0 | 89.7 | 89.6 | 89.3 | 89.6 | 85.9 |
| **DE-detect** | **94.1** | **93.9** | **94.0** | **93.9** | **94.1** | **87.9** |

CNN 在 Pitch 攻击下暴跌至 59.0%，Udio 泛化仅 56.9%；DE-detect 在所有攻击下保持 93.9-94.1%，Udio 泛化达 87.9%。

**部分伪造实验**（验证模型是否依赖音频伪影）：

| 模型 | Real vs. Partly-Fake | Fake vs. Partly-Fake |
|------|---------------------|---------------------|
| XEUS | 50.5（≈随机） | 92.0 |
| Llama3 8B | 64.9 | 90.0 |

XEUS 在区分真实/部分伪造时接近随机水平，证明其不依赖音频伪影而是关注歌词内容。

### 关键发现

1. **转录质量不是决定因素**：更低的 WER 不一定带来更好的检测性能，Whisper large-v2 虽非 WER 最低但检测效果最好
2. **语音嵌入不依赖音频伪影**：XEUS 在 real vs. partly-fake 实验中表现接近随机，说明其特征主要反映歌词内容而非生成器伪影
3. **多视角融合提供一致优势**：DE-detect 在所有域外场景中比单模态方法高 1.5-2% recall
4. **CNN 方法在域外严重退化**：尤其在 pitch 变化（59.0%）和 Udio 泛化（56.9%）上几乎不可用

## 亮点与洞察

1. **实用性极强**：整个管线仅需音频输入，完全不依赖歌词元数据，适合工业落地
2. **多视角融合的直觉清晰**：what（转录歌词语义）+ how（语音声学特征）的融合逻辑简洁优雅
3. **模块化设计面向未来**：各组件可独立升级，适应 AIGM 快速迭代的生态
4. **partly-fake 实验设计精巧**：通过控制变量巧妙验证了模型确实在检测歌词而非音频伪影

## 局限与展望

1. 训练数据主要基于 Suno v3.5 生成的音频，对其他生成器（如 Udio）存在偏差
2. 鲁棒性评估未覆盖多种攻击叠加的场景（如同时 pitch + noise）
3. 数据集规模有限（约 7K 首），未来需要更大规模、更多样化的数据集
4. 检测系统存在双用性风险——攻击者可能利用弱点绕过检测

## 相关工作与启发

- **Labrak et al. (2025)**：提出歌词检测数据集和文本基线，但依赖干净歌词
- **Afchar et al. (2024)**：CNN 频谱图方法在域内高效但泛化差，揭示了音频伪影检测的固有局限
- **XEUS (Chen et al., 2024b)**：强大的多语言语音模型，训练数据含唱歌声音，为本文语音分支提供了关键组件
- **启发**：在 AI 生成内容检测中，融合多种互补视角比单一模态更鲁棒，且模块化设计对快速演变的生成技术至关重要

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将专用语音嵌入用于音乐领域的 AI 歌词检测，多视角融合思路新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 域内/域外/攻击/部分伪造四维评估非常全面
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，图表设计直观
- **价值**: ⭐⭐⭐⭐ — 高度实用，对音乐产业 AI 内容治理有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] MusicDET: Zero-Shot AI-Generated Music Detection](../../ICML2026/audio_speech/musicdet_zero-shot_ai-generated_music_detection.md)
- [\[ACL 2025\] On the Robust Approximation of ASR Metrics](on_the_robust_approximation_of_asr_metrics.md)
- [\[ACL 2025\] T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback](t2a_feedback_audio_gen.md)
- [\[ICML 2026\] Probing Token Spaces under Generator Shift in AI-Generated Music Detection](../../ICML2026/audio_speech/probing_token_spaces_under_generator_shift_in_ai-generated_music_detection.md)
- [\[ACL 2025\] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)

</div>

<!-- RELATED:END -->
