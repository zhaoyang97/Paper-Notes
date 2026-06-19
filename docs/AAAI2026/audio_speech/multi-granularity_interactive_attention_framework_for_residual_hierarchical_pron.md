---
title: >-
  [论文解读] Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment
description: >-
  [AAAI2026][音频/语音][发音评估] 提出HIA框架，通过交互注意力模块（Interactive Attention Module）实现音素、词、句三粒度间的双向信息交互，结合残差层级结构缓解特征遗忘问题，在speechocean762数据集上所有粒度和方面指标均达到SOTA。 自动发音评估的重要性 计算机辅助发音…
tags:
  - "AAAI2026"
  - "音频/语音"
  - "发音评估"
  - "多粒度交互"
  - "注意力机制"
  - "残差层级结构"
  - "CAPT"
  - "语音评分"
---

# Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment

**会议**: AAAI2026  
**arXiv**: [2601.01745](https://arxiv.org/abs/2601.01745)  
**作者**: Hong Han, Hao-Chen Pei, Zhao-Zheng Nie, Xin Luo, Xin-Shun Xu  
**代码**: 未公开  
**领域**: 音频语音  
**关键词**: 发音评估, 多粒度交互, 注意力机制, 残差层级结构, CAPT, 语音评分  

## 一句话总结

提出HIA框架，通过交互注意力模块（Interactive Attention Module）实现音素、词、句三粒度间的双向信息交互，结合残差层级结构缓解特征遗忘问题，在speechocean762数据集上所有粒度和方面指标均达到SOTA。

## 背景与动机

### 自动发音评估的重要性

计算机辅助发音训练（CAPT）系统通过即时反馈帮助语言学习者改善发音，其核心是自动发音评估（APA）——对说话者发音质量进行多方面评分。早期APA方法集中在单一粒度：音素级别的发音准确度评估、词级别或句级别的各方面检测。这些单粒度方法虽在特定任务上表现良好，但未考虑语音信号天然的多粒度层级特性。

### 多粒度评估的必要性

语音信号具有固有的层级结构：音素组成词，词组成句子。低粒度的发音结果直接影响高粒度的评分——如果音素发音不准，词的整体得分必然受影响。单粒度建模无法揭示不同粒度间的隐式关联。因此，将多方面多粒度评估任务整合到统一模型中成为研究趋势。

### 现有多粒度方法的不足

现有方法仅考虑相邻粒度间的单向依赖（音素→词→句），缺乏粒度间的双向交互：（1）GOPT平行处理各粒度但缺乏粒度间交互；（2）HiPAMA使用层级结构但信息流单向；（3）Gradformer聚焦句级建模，忽略音素-词关联；（4）HierGAT的固定图结构限制了动态交互。特别是，同一个词在不同句子中可能有不同的重音模式，缺乏自上而下的交互建模是现有方法在词重音（word stress）上表现差的原因。此外，随着层级深度增加，初始编码特征可能被遗忘。

## 核心问题

如何在多方面多粒度发音评估中实现音素、词、句三粒度间的双向动态交互，同时缓解层级建模导致的特征遗忘问题？

## 方法详解

### 整体框架

HIA接收GOP特征和标准音素嵌入作为输入，经Transformer encoder编码后得到声学嵌入，再通过残差层级结构依次建模各粒度评分。核心组件是交互注意力模块和残差连接。

### 声学特征处理

使用Librispeech声学模型提取GOP特征（84维），包括Log Phone Posterior（LPP）和Log Posterior Ratio（LPR）：

$$\text{LPP}(p) \approx \frac{1}{t_e - t_s + 1} \sum_{t=t_s}^{t_e} \log P(p|o_t)$$

$$\text{LPR}(p_j|p_i) = \log P(p_j|\mathbf{o}; t_s, t_e) - \log P(p_i|\mathbf{o}; t_s, t_e)$$

总共42个纯音素，GOP特征为84维向量。将投影后的GOP特征、标准音素嵌入和可训练位置嵌入相加输入Transformer encoder。

### 交互注意力模块（Interactive Attention Module）

**核心创新**：首次在发音评估中实现三粒度间的双向交互。

1. **初始化粒度查询**：从声学嵌入投影出各粒度的query向量 $Q^l \in \mathbb{R}^{B \times D}$
2. **拼接自注意力**：将三粒度query拼接为 $Q = \{Q^{phn}, Q^{word}, Q^{utt}\} \in \mathbb{R}^{B \times 3 \times D}$，通过自注意力实现双向交互：$Q_{self} = \text{SelfAttn}(Q)$
3. **交叉注意力映射**：将自注意力头作query，声学嵌入$X$作key/value，映射到声学特征空间：$Q_{cross} = \text{CrossAttn}(Q_{self}, X)$
4. **投影输出**：经FFN后投影得到各粒度的交互注意力头 $H^{phn}$, $H^{word}$, $H^{utt}$

### 残差层级多粒度建模

**音素级**：声学嵌入$X$加交互注意力头$H^{phn}$，经1-D卷积和回归头输出音素准确度：

$$S^{phn} = \text{Conv}(X + H^{phn})$$

**词级**：结合声学嵌入、音素评分结果和词级注意力头，通过方面注意力机制建模词级多方面关联：

$$X^{word} = X + S^{phn} + H^{word}, \quad S^{word} = \text{AspectAttn}(X^{word})$$

**句级**：使用Transformer decoder捕获长程依赖，初始化可学习查询向量，将声学嵌入+词级评分+句级注意力头作key/value：

$$S^{utt} = \text{TransDecoder}(Q^{utt}, X + S^{word} + H^{utt})$$

**残差连接**：各粒度建模时都引入原始声学嵌入$X$，缓解层级加深导致的初始特征遗忘。

### 损失函数

各粒度各方面均使用MSE损失，总损失为所有粒度方面损失之和：

$$L_{\text{total}} = \sum_{i=1}^M \frac{1}{N} \sum_{j=1}^N L_{ij}$$

## 实验关键数据

数据集：speechocean762（5000句英语，250位非母语者，含儿童）。Adam优化器，lr=1e-3，5次不同种子取均值和标准差。

### 主实验（PCC↑，与SOTA对比）

| 模型 | 音素 PCC↑ | 词 Accuracy↑ | 词 Stress↑ | 词 Total↑ | 句 Fluency↑ | 句 Prosodic↑ | 句 Total↑ |
|------|----------|-------------|-----------|----------|-----------|------------|----------|
| GOPT | 0.612 | 0.533 | 0.291 | 0.549 | 0.753 | 0.760 | 0.742 |
| HiPAMA | 0.616 | 0.575 | 0.320 | 0.591 | 0.749 | 0.751 | 0.754 |
| Gradformer | 0.646 | 0.598 | 0.334 | 0.614 | 0.769 | 0.767 | 0.756 |
| **HIA** | **0.657** | **0.613** | **0.436** | **0.628** | **0.778** | **0.784** | **0.764** |
| 人类专家 | 0.555 | 0.589 | 0.212 | 0.602 | 0.665 | 0.651 | 0.675 |

HIA在词重音上PCC达0.436，较Gradformer提升30.5%（+0.102），是最显著的改进。HIA在除句级completeness外所有指标上超过人类专家评估者一致性。

### 交互注意力模块消融

| 配置 | 音素 PCC | 词 Stress | 词 Total | 句 Total |
|------|---------|----------|---------|---------|
| w/o所有交互头 | 0.626 | 0.335 | 0.605 | 0.748 |
| 仅词+句交互头 | 0.621 | 0.429 | 0.617 | 0.758 |
| 仅音素+句交互头 | 0.661 | 0.328 | 0.604 | 0.759 |
| 仅音素+词交互头 | 0.653 | 0.421 | 0.621 | 0.754 |
| **全部交互头（HIA）** | **0.657** | **0.436** | **0.628** | **0.764** |

### 残差层级结构消融

| 配置 | 音素 PCC | 词 Stress | 词 Total | 句 Total |
|------|---------|----------|---------|---------|
| 去除残差 | 0.647 | 0.382 | 0.603 | 0.748 |
| 去除层级 | 0.645 | 0.374 | 0.593 | 0.753 |
| **HIA** | **0.657** | **0.436** | **0.628** | **0.764** |

### 卷积层数消融

| 层数 | 音素 PCC | 词 Stress | 词 Total | 句 Total |
|------|---------|----------|---------|---------|
| 0层 | 0.638 | 0.415 | 0.601 | 0.754 |
| 1层 (HIA) | **0.657** | **0.436** | **0.628** | **0.764** |
| 2层 | 0.646 | 0.427 | 0.618 | 0.759 |
| 3层 | 0.645 | 0.421 | 0.617 | 0.755 |

## 亮点

- **首次双向粒度交互**：通过拼接三粒度query做自注意力+交叉注意力的简洁设计，实现音素↔词↔句的全双向信息流动，特别在词重音评估上带来30%+的提升
- **残差层级结构**：在层级逐粒度建模中引入原始声学嵌入的残差连接，有效缓解层级加深导致的特征遗忘
- **超越人类专家一致性**：HIA在几乎所有指标上超过5位专家评估者之间的一致性，展示了模型在发音评估中的实用价值
- **消融实验极为充分**：对交互注意力（逐粒度消融）、残差/层级结构、卷积层数、嵌入维度、注意力头数均做了详尽的消融分析

## 局限与展望

- **数据集单一**：仅在speechocean762上评估，该数据集规模较小（5000句），且completeness评分分布极度不均（4975/5000为满分），限制了部分指标的评估可靠性
- **输入特征依赖GOP**：使用传统的GOP特征作为输入，未利用自监督语音模型（如wav2vec 2.0、HuBERT）的表示能力，可能限制了性能上限
- **仅支持朗读场景**：框架设计针对read-aloud发音评估，不适用于开放式口语回答场景
- **模型规模有限**：嵌入维度48、单头注意力的小模型配置受限于数据集规模，当有更大数据集时需要重新探索最优配置

## 与相关工作的对比

- **GOPT**（2022）：Transformer多任务并行评估，但无粒度间交互，HIA在音素PCC上高7.4%，词Total高14.4%
- **HiPAMA**（2023）：引入层级结构建模粒度依赖，但信息流单向，HIA通过双向交互在词Stress上高36.3%（0.436 vs 0.320）
- **Gradformer**（2024）：卷积增强Transformer+粒度解耦，聚焦句级建模但忽略音素-词关联，HIA在所有指标上全面领先
- **HierGAT**（2024）：图神经网络层级建模，固定图结构限制动态交互，HIA的注意力机制动态交互更灵活
- **非GOP方法**（wav2vec2-based, LAS等）：使用自监督特征在句级Total上接近（0.725/0.766），但未提供多粒度评估能力

## 启发与关联

- 交互注意力模块的"拼接多粒度query→自注意力→交叉注意力"设计模式可迁移到其他多粒度任务：如文档摘要（词→句→段落）、视频理解（帧→片段→全片）
- 残差层级结构的特征遗忘缓解策略与DenseNet、U-Net等跨层连接思想一脉相承，但在序列建模中的应用值得进一步探索
- 词重音评估的大幅提升（+30%）验证了双向交互对捕获上下文依赖发音模式的重要性，启发在语音合成中也可利用句级信息指导词级韵律生成

## 评分

- 新颖性: ⭐⭐⭐⭐ — 双向粒度交互在发音评估中首次提出，交互注意力模块设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 消融涵盖几乎所有设计选择，数据相关性分析增加说服力
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题动机论述充分，图表直观
- 价值: ⭐⭐⭐⭐ — 在发音评估这一细分领域达到全面SOTA，实用价值高，但应用范围相对有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)
- [\[ICLR 2026\] MAPSS: Manifold-Based Assessment of Perceptual Source Separation](../../ICLR2026/audio_speech/mapss_manifold-based_assessment_of_perceptual_source_separation.md)
- [\[CVPR 2026\] AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding](../../CVPR2026/audio_speech/amuse_audio-visual_benchmark_and_alignment_framework_for_agentic_multi-speaker_u.md)
- [\[CVPR 2026\] Hierarchical Codec Diffusion for Video-to-Speech Generation](../../CVPR2026/audio_speech/hierarchical_codec_diffusion_for_video-to-speech_generation.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](../../ACL2026/audio_speech/full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)

</div>

<!-- RELATED:END -->
