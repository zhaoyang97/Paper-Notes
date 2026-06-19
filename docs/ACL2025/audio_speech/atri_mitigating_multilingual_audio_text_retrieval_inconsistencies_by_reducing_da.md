---
title: >-
  [论文解读] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors
description: >-
  [ACL 2025][音频/语音][多语言音频文本检索] 从理论上分析多语言音频文本检索（ML-ATR）中跨语言不一致性的根本原因是训练数据分布误差，并提出 1-to-K 对比学习（KCL）和音频-英语共锚对比学习（CACL）两种策略来减少该误差，在召回率和一致性上达到 SOTA。 音频文本检索（ATR）旨在根据跨模态查询在…
tags:
  - "ACL 2025"
  - "音频/语音"
  - "多语言音频文本检索"
  - "跨语言一致性"
  - "对比学习"
  - "模态对齐"
  - "数据分布误差"
---

# ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors

**会议**: ACL 2025  
**arXiv**: [2502.14627](https://arxiv.org/abs/2502.14627)  
**代码**: [github.com/ATRI-ACL/ATRI-ACL](https://github.com/ATRI-ACL/ATRI-ACL)  
**领域**: 语音/音频  
**关键词**: 多语言音频文本检索, 跨语言一致性, 对比学习, 模态对齐, 数据分布误差

## 一句话总结

从理论上分析多语言音频文本检索（ML-ATR）中跨语言不一致性的根本原因是训练数据分布误差，并提出 1-to-K 对比学习（KCL）和音频-英语共锚对比学习（CACL）两种策略来减少该误差，在召回率和一致性上达到 SOTA。

## 研究背景与动机

音频文本检索（ATR）旨在根据跨模态查询在数据库中搜索匹配的音频片段或文本描述。虽然英语单语 ATR 性能不断提升，但多语言音频文本检索（ML-ATR）的研究仍然有限，面临两大核心挑战：

**多语言召回率不高**：现有 ML-ATR 方案在每个 epoch 中随机选择一种语言的文本与音频配对训练，导致模型无法充分学习音频与多语言文本之间的嵌入空间关系。

**跨语言检索结果不一致**：用不同语言查询同一音频时，检索结果排名差异很大。例如用英语描述和法语描述查询同一段声音，返回的排名可能截然不同。

现有 ML-CLAP 方案的训练方式本质上是在每个 epoch 随机采样一种语言做对比学习，这不仅降低了检索召回率，还导致检索一致性问题。作者首次从理论角度深入分析了这一问题的根源。

## 方法详解

### 整体框架

ATRI 方案的核心思想是：通过理论分析证明不一致性的根源是数据分布误差，然后设计两种策略来减少该误差。框架使用 CED-Base 作为音频编码器，SONAR 作为多语言文本编码器。

### 关键设计

1. **理论分析——权重误差上界推导**：作者首先从模态对齐方向误差的角度可视化了不一致性问题。理想情况下，音频嵌入应对齐到多语言文本嵌入的算术均值方向（绿色箭头），但随机采样导致音频只与单一语言文本对齐（红色箭头），两者之间的角度就是模态对齐方向误差。进一步推导出权重误差上界公式：

    $\|\mathbf{w}_{eT} - \mathbf{w}'_{eT}\| \leq a^T\|\mathbf{w}_{(e-1)T} - \mathbf{w}'_{(e-1)T}\| + \eta\sum_{(a,t)}\|p(a,t) - p'_e(a,t)\|\cdot(\text{梯度相关项})$

   展开后发现权重误差的根源完全来自各 epoch 的数据分布误差 $\sum\|p(a,t) - p'_i(a,t)\|$。

2. **1-to-K 对比学习（KCL）**：在每个 epoch 中同时使用所有 K 种语言的文本与音频做对比学习，理论上完全消除数据分布误差。损失函数包含 audio-to-text 和 text-to-audio 两个方向，对每种语言独立计算对比损失再求和。缺点是 GPU 显存开销随语言数 K 线性增长。

3. **音频-英语共锚对比学习（CACL）**：为解决 KCL 显存开销大的问题，提出 CACL 作为轻量替代。每条数据取三元组（音频，英语文本，随机其他语言文本），进行三组对比学习：

    - 音频-英语对齐 $\mathcal{L}^{ae}_{cacl}$
    - 音频-多语言对齐 $\mathcal{L}^{at}_{cacl}$
    - 英语-多语言对齐 $\mathcal{L}^{et}_{cacl}$ 
   
   CACL 的有效性可从两个视角理解：（a）英语-多语言对齐拉近了不同语言嵌入的距离，减小了模态对齐方向偏差；（b）每个 epoch 中训练了更多音频-文本对（且包含高质量英语文本），使实际数据分布更接近理论最优分布。关键优势是显存开销不随语言数增长。

### 损失函数 / 训练策略

- KCL 损失：$\mathcal{L}_{kcl} = \frac{1}{2NK}(\mathcal{L}^{a2t}_{kcl} + \mathcal{L}^{t2a}_{kcl})$
- CACL 损失：$\mathcal{L}_{cacl} = \frac{1}{6N}(\mathcal{L}^{ae}_{cacl} + \mathcal{L}^{at}_{cacl} + \mathcal{L}^{et}_{cacl})$
- 使用 ML-CLAP 预训练权重初始化，在翻译后的多语言 AudioCaps 和 Clotho 数据集上微调 10 个 epoch
- 批量大小 24，学习率 $5\times 10^{-6}$，温度参数 $\tau = 0.07$，Adam 优化器
- 单卡 A100 80GB 训练

## 实验关键数据

### 主实验

在 AudioCaps 数据集上的 T2A R@1 平均性能（8种语言）：

| 方案 | T2A R@1 (avg) | A2T R@1 (avg) | 相比 ML-CLAP 提升 |
|------|-------------|-------------|-----------------|
| ML-CLAP | 44.84 | 61.19 | - |
| CACL | 46.03 (+1.19) | 62.28 (+1.09) | 召回一致提升 |
| KCL | **46.81** (+1.97) | **62.91** (+1.72) | SOTA，R@1提升~2% |

英语单语 ATR 结果（AudioCaps T2A R@1）：

| 方案 | R@1 | R@5 | mAP10 |
|------|-----|-----|-------|
| ML-CLAP | 47.31 | 80.65 | 61.44 |
| CACL | 49.05 | 82.14 | 63.07 |
| KCL | **49.68** (+5%) | **82.44** | **63.34** |

### 一致性评估

| 方案 | AudioCaps MRV↓ | Clotho MRV↓ | 说明 |
|------|---------------|-------------|------|
| ML-CLAP | 较高 | 较高 | 跨语言一致性差 |
| CACL | 降低 | 降低 | 嵌入空间差距和距离缩小 |
| KCL | **最低** | **最低** | 一致性最佳 |

### 关键发现

- **理论与实验一致**：KCL 完全消除数据分布误差，性能最优；CACL 减少分布误差，性能次之。两者都优于随机采样的 ML-CLAP
- **KCL > CACL > ML-CLAP**：在绝大多数语言和指标上，KCL 持续领先
- **语言间性能差异**：日语和中文的指标较低，因为它们与其他语言的语法差异较大
- **CACL 的实用价值**：在性能接近 KCL 的情况下，显存和时间开销接近 ML-CLAP，是实际部署的更优选择
- **偶发异常**：极少数指标上 KCL 低于 CACL，归因于数据集噪声（Clotho 上更常见）

## 亮点与洞察

1. **理论驱动的方法设计**：从权重误差上界推导出发，发现数据分布误差是不一致性的根本原因，再针对性设计解决方案，逻辑链条严密
2. **理论与实验的精确对应**：KCL（消除误差）> CACL（减少误差）> ML-CLAP（随机），完美验证理论预测
3. **实用性考量**：提供两种方案供不同场景选择——性能优先用 KCL，资源优先用 CACL
4. **英语作为锚点的巧妙设计**：英语通常是翻译的源语言，质量最高，以此为共锚点是很自然且有效的选择

## 局限与展望

- 仅在翻译数据集上验证（AudioCaps 和 Clotho），缺少原生多语言数据的评估
- 翻译质量可能影响结果，特别是日语和中文等语法差异大的语言
- 8 种语言的覆盖范围有限，低资源语言的表现未知
- CACL 中选择英语作为锚点的假设可能不适用于所有场景
- 仅探索了 SONAR 文本编码器，对其他多语言编码器（如 mBERT）的效果未验证

## 相关工作与启发

- **与 ML-CLAP 的关系**：直接在 ML-CLAP 基础上改进，解决其随机语言采样的固有问题
- **CLIP 启发**：基于 CLIP 风格的对比学习框架，扩展到多语言多模态场景
- **对其他领域的启发**：数据分布误差导致模型权重偏离最优的分析思路，可推广到其他多语言多模态任务（如多语言视觉-语言预训练）
- **共锚点策略的普适性**：以高质量模态（如英语文本）为锚点对齐低质量模态的思路，可应用到多模态对齐的其他场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论分析角度新颖，权重误差上界的推导为方法设计提供了坚实基础
- 实验充分度: ⭐⭐⭐⭐ 8种语言、2个数据集、多种指标、一致性分析全面，但缺少原生多语言数据
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验分析详尽，图示直观
- 价值: ⭐⭐⭐⭐ 对多语言音频检索领域有实质性推进，理论发现具有更广泛的指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models](audio_token_consistency.md)
- [\[ACL 2025\] CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages](clamp_3_universal_music_information_retrieval_across_unaligned_modalities_and_un.md)
- [\[ACL 2025\] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)

</div>

<!-- RELATED:END -->
