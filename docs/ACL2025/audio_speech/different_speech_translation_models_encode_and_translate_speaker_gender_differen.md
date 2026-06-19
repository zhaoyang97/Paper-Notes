---
title: >-
  [论文解读] Different Speech Translation Models Encode and Translate Speaker Gender Differently
description: >-
  [ACL 2025][音频/语音][语音翻译] 通过注意力探针分析不同架构的语音翻译模型如何编码说话人性别信息，发现传统编码器-解码器模型能较好保留性别信息，而新型 speech+MT 架构的适配器会显著擦除性别信息，导致翻译中出现更严重的阳性默认偏差。 领域现状： 语音表示学习研究表明，模型内部表示能捕获语音学和说话人相关…
tags:
  - "ACL 2025"
  - "音频/语音"
  - "语音翻译"
  - "性别编码"
  - "可解释性"
  - "探针分析"
  - "翻译偏差"
---

# Different Speech Translation Models Encode and Translate Speaker Gender Differently

**会议**: ACL 2025  
**arXiv**: [2506.02172](https://arxiv.org/abs/2506.02172)  
**代码**: [https://github.com/hlt-mt/speech-translation-gender](https://github.com/hlt-mt/speech-translation-gender)  
**领域**: 语音  
**关键词**: 语音翻译, 性别编码, 可解释性, 探针分析, 翻译偏差

## 一句话总结

通过注意力探针分析不同架构的语音翻译模型如何编码说话人性别信息，发现传统编码器-解码器模型能较好保留性别信息，而新型 speech+MT 架构的适配器会显著擦除性别信息，导致翻译中出现更严重的阳性默认偏差。

## 研究背景与动机

**领域现状**: 语音表示学习研究表明，模型内部表示能捕获语音学和说话人相关特征（包括性别）。语音翻译（ST）领域正从传统编码器-解码器架构转向新型 speech+MT 架构——即通过适配器将预训练语音编码器接入机器翻译系统。

**现有痛点**: 当从"概念性别"语言（如英语）翻译到"语法性别"语言（如法语/意大利语/西班牙语）时，模型需要从上下文推断性别并正确施加语法曲变。例如 "I was born in..." → "Je suis né/née à..."。但现有系统普遍存在阳性默认倾向。

**核心矛盾**: ST 领域正在发生架构变革，但对于新旧架构如何编码和使用性别信息——以及这如何影响翻译中的性别偏差——知之甚少。

**本文目标**: (1) 不同 ST 架构是否编码了说话人性别信息？(2) 架构差异如何影响翻译中的性别赋值？(3) 性别编码能力与翻译准确性之间是否存在关联？

**切入角度**: 使用探针方法（probing）——一种成熟的可解释性技术——训练分类器从模型隐状态预测说话人性别，然后分析性别编码能力与翻译准确性的关系。

**核心 idea**: 性别编码能力与翻译性别准确性高度相关，speech+MT 架构的适配器会擦除性别信息导致阳性默认偏差加剧。

## 方法详解

### 整体框架

对三类 ST 模型进行探针分析：传统编码器-解码器（enc-dec）、SeamlessM4T（speech+MT）、ZeroSwot（speech+MT）。在三个翻译方向（En→Fr/It/Es）上分析性别编码与翻译准确性的关系。

### 关键设计

**模块: 注意力探针（Attention-based Probe）**

- **功能**: 从模型隐状态序列中提取性别信息进行二分类
- **核心思路**: 受 Q-Former 启发，使用单个可学习查询向量 $\mathbf{q} \in \mathbb{R}^d$ 与隐状态序列 $\mathbf{X} = \langle \mathbf{x}_1, ..., \mathbf{x}_L \rangle$ 做注意力运算。隐状态通过可学习权重矩阵 $\mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$ 投影为 Key 和 Value。输出 $\mathbf{o} \in \mathbb{R}^d$ 经线性层做分类。注意力权重 $\mathbf{a} \in \mathbb{R}^L$ 指示哪些位置对性别编码贡献最大
- **设计动机**: 
    - 避免 mean/max pooling 可能掩盖位置变化的问题
    - 避免固定位置探针不支持变长序列的限制
    - 保持架构简单（符合探针设计原则），同时比线性模型更具表达力
    - 模拟 ST 解码器通过交叉注意力机制访问编码器状态的方式

**探测位置**:
- enc-dec: 编码器输出
- speech+MT: 适配器前（pre-ad）和适配器后（post-ad）分别探测

### 损失函数/训练策略

- 探针训练使用从 MuST-C 训练集（en→es）采样的性别平衡训练集和验证集
- 两个测试集：test-generic（通用不平衡集）和 test-speaker（来自 MuST-SHE，专注说话人指代的性别平衡集）
- 评估指标：探针用 macro F1 + 单类 recall；翻译质量用 COMET；性别翻译用准确率 + 覆盖率

## 实验关键数据

### 主实验

性别探针 F1 分数（test-speaker 平均）:

| 模型 | 探测位置 | en→es | en→fr | en→it | 平均 |
|------|---------|-------|-------|-------|------|
| Seamless | post-ad | 51.72 | 54.51 | - | 53.95 |
| Seamless | pre-ad | 67.32 | 67.52 | - | 68.47 |
| ZeroSwot | post-ad | 61.80 | 61.62 | - | 61.36 |
| ZeroSwot | pre-ad | 90.25 | 90.02 | - | 89.61 |
| **enc-dec** | encoder | **93.14** | **94.59** | - | **94.64** |

翻译性别准确率（test-speaker）:

| 模型 | She Acc | He Acc | All Acc | COMET |
|------|---------|--------|---------|-------|
| Seamless | 14.33 | 90.25 | 53.35 | 80.36 |
| ZeroSwot | 50.69 | 74.90 | 62.80 | 83.94 |
| **enc-dec** | **78.53** | **92.25** | **85.57** | 74.77 |

### 消融实验

适配器对性别信息的影响（F1 下降幅度）:

| 模型 | pre-ad F1 | post-ad F1 | 下降 |
|------|-----------|------------|------|
| Seamless | 68.47 | 53.95 | ~21% |
| ZeroSwot | 89.61 | 61.36 | ~32% |

性别编码与翻译准确率的相关性：$R^2 = 0.99$，$p < 0.01$。

### 关键发现

1. **传统 enc-dec 模型**性别编码能力最强（F1 > 94），而 speech+MT 模型在适配器后大幅下降（Seamless post-ad 仅 53.95）
2. **适配器是关键瓶颈**: ZeroSwot 适配器前 pre-ad F1 = 89.61，适配器后 post-ad 骤降至 61.36，损失约 32%
3. **性别编码与翻译准确率高度相关**（$R^2=0.99$），编码越强翻译越准确
4. **阳性默认偏差**: Seamless 女性准确率最低仅 12.09%（en→es），男性却高达 90.55%，差距悬殊
5. 即使 enc-dec 模型也存在轻微阳性偏差：女性平均 78.53 vs 男性 92.25
6. ST 模型主要在序列早期位置编码性别信息

## 亮点与洞察

- **反直觉发现**: 传统上 NLP 认为移除性别信息可提升公平性，但本文表明在 ST 中保留性别编码反而产生更公平的翻译（女性翻译准确率更高）
- **注意力探针设计**巧妙——既像 Q-Former 一样灵活处理变长序列，又保持了探针应有的简洁性
- **适配器是性别信息的"瓶颈"**: 这一发现对 speech+MT 架构的设计有重要启示——映射到文本嵌入空间时丢失了说话人相关的声学信息
- 性别编码与翻译准确率 $R^2=0.99$ 的强相关性提供了因果推断的有力证据

## 局限与展望

1. **模型和语言覆盖有限**: 仅评估了三个模型和三个翻译方向，未涉及 LLM-based ST 系统
2. **二元性别框架**: 受限于数据可用性，仅分析 She/He 二元分类，未涵盖非二元性别
3. **探针的因果推断局限**: 探针性能与翻译性能的相关不意味因果，需要 amnesic probing 等进一步分析
4. 未探索如何修改适配器架构以更好保留性别信息
5. 伦理考量：利用声学特征推断性别可能导致误判（如跨性别者、儿童等群体）

## 相关工作与启发

- **MuST-SHE** 提供了评估 ST 性别翻译的专用语料，其自我声明的性别标签避免了误判风险
- **SeamlessM4T** 和 **ZeroSwot** 代表了 speech+MT 架构的两种训练策略（联合训练 vs 冻结 MT）
- **Q-Former**（BLIP-2）启发了注意力探针的设计
- 社会语音学研究为声学性别特征的存在提供了理论基础

## 评分

- **新颖性**: ⭐⭐⭐⭐ (首次系统性分析ST模型的性别编码, 发现适配器瓶颈)
- **实验充分度**: ⭐⭐⭐⭐ (三种架构×三个语言方向, R²=0.99的强相关)
- **写作质量**: ⭐⭐⭐⭐⭐ (论述严谨, 伦理讨论充分, 结构清晰)
- **价值**: ⭐⭐⭐⭐ (对ST架构设计和公平性均有实际指导意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)
- [\[ACL 2025\] Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation](eta-wavlm_efficient_speaker_identity_removal_in_self-supervised_speech_represent.md)
- [\[ACL 2025\] Improving Language and Modality Transfer in Translation by Character-level Modeling](improving_language_and_modality_transfer_in.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)

</div>

<!-- RELATED:END -->
