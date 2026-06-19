---
title: >-
  [论文解读] ALGEN: Few-Shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment
description: >-
  [ACL 2025][文本嵌入反转攻击] 本文提出ALGEN，一种少样本文本嵌入反转攻击方法，通过将受害者的嵌入空间与攻击者的嵌入空间进行线性对齐，再利用训练好的嵌入到文本生成器重建原始文本，仅需1个泄露样本即可发起部分成功的攻击，1000个样本时Rouge-L达45.75。 领域现状：随着LLM和向量数据库（Vector…
tags:
  - "ACL 2025"
  - "文本嵌入反转攻击"
  - "少样本攻击"
  - "跨模型对齐"
  - "向量数据库安全"
  - "隐私泄露"
---

# ALGEN: Few-Shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment

**会议**: ACL 2025  
**arXiv**: [2502.11308](https://arxiv.org/abs/2502.11308)  
**代码**: 无  
**领域**: 其他  
**关键词**: 文本嵌入反转攻击、少样本攻击、跨模型对齐、向量数据库安全、隐私泄露

## 一句话总结
本文提出ALGEN，一种少样本文本嵌入反转攻击方法，通过将受害者的嵌入空间与攻击者的嵌入空间进行线性对齐，再利用训练好的嵌入到文本生成器重建原始文本，仅需1个泄露样本即可发起部分成功的攻击，1000个样本时Rouge-L达45.75。

## 研究背景与动机

**领域现状**：随着LLM和向量数据库（Vector DB）的普及，越来越多的私有文本数据被处理和存储为数值嵌入。Pinecone、Weaviate等向量数据库服务和RAG系统广泛使用嵌入来支持搜索和检索。

**现有痛点**：先前的嵌入反转攻击研究已证明可以从嵌入中重建原始文本，但这些方法需要大量的数据泄露来训练攻击模型——Li et al.需要100k样本，Morris et al.（Vec2Text）需要1-5百万样本，Huang et al.需要8k样本。这种大规模数据泄露假设在现实中难以满足。

**核心矛盾**：攻击者希望从少量泄露的嵌入中就能发起有效攻击，但现有方法的攻击模型都需要大量训练数据，且通常需要直接访问受害者编码器来获取嵌入进行训练。

**本文目标**：大幅降低嵌入反转攻击的数据需求，实现真正的少样本甚至单样本攻击。

**切入角度**：不直接在受害者嵌入上训练攻击模型，而是将受害者嵌入通过线性变换对齐到攻击者自有的嵌入空间，然后复用在攻击者空间训练好的通用解码器。

**核心 idea**：三步攻击流程——（1）训练本地嵌入到文本生成器；（2）用少量泄露的<文本,嵌入>对通过最小二乘法学习跨模型线性对齐；（3）将受害者嵌入对齐到攻击空间后解码。

## 方法详解

### 整体框架
ALGEN分为三个独立阶段：第一阶段在公开语料上训练攻击者本地的嵌入-到-文本生成模型（不涉及受害者）；第二阶段利用少量泄露数据学习受害者→攻击者的嵌入空间线性映射；第三阶段组合映射和解码器发起攻击。

### 关键设计

1. **本地嵌入到文本生成器（Embedding-to-Text Generator）**:

    - 功能：将攻击者自有编码器的嵌入解码回原始文本
    - 核心思路：选择FlanT5作为backbone，使用公开语料（MultiHPLT英文数据集150k条）微调FlanT5的decoder。输入为攻击者编码器 $enc_A$ 生成的句子嵌入 $\mathbf{e}_A$（通过mean pooling + L2归一化获得），训练目标为交叉熵损失。训练完成后该生成器可以将攻击空间中的嵌入解码为文本
    - 设计动机：这个生成器完全独立于受害者模型，可以提前训练。公开语料容易获取，不暴露攻击者的意图

2. **嵌入空间线性对齐（Embedding Space Alignment）**:

    - 功能：将受害者嵌入映射到攻击者嵌入空间
    - 核心思路：假设有一小组泄露的数据对 $(X, E_V)$，其中 $X$ 是文本，$E_V = enc_V(X)$ 是受害者嵌入。同时用攻击者编码器计算 $E_A = enc_A(X)$。通过最小二乘法求解 $E_V W \approx E_A$，闭式解为 $W = (E_V^T E_V)^{-1} E_V^T E_A$（Moore-Penrose伪逆）。这是一步完成的线性变换，不需要训练
    - 设计动机：不同嵌入空间之间存在近似线性关系，这在跨语言词向量对齐等研究中已被验证。线性映射只需极少的样本就能学到合理的变换——甚至1个样本就能提供部分有效的对齐

3. **反转攻击执行**:

    - 功能：组合对齐和解码完成文本重建
    - 核心思路：给定窃取到的受害者嵌入 $E_V$，通过 $\hat{X} = dec_A(E_V W)$ 重建原始文本。整个攻击流程非常高效，不需要迭代优化或GPU密集计算
    - 设计动机：线性对齐的计算成本极低，使得攻击可以快速大规模执行

### 损失函数 / 训练策略
生成器训练使用交叉熵损失，AdamW优化器（学习率1e-4，权重衰减1e-4），batch size 128。对齐阶段不需要训练，直接用最小二乘闭式解。

## 实验关键数据

### 主实验

| 方法 | 受害者模型 | Rouge-L | BLEU1 | COS |
|------|-----------|---------|-------|-----|
| Vec2Text Base | T5 | 17.38 | 21.47 | 0.4663 |
| Vec2Text Corrector | T5 | 15.81 | 18.35 | 0.4835 |
| **ALGEN** (1k样本) | T5 | **45.75** | **52.98** | **0.9464** |
| **ALGEN** (1k样本) | GTR | **38.27** | **42.59** | **0.8879** |
| **ALGEN** (1k样本) | OpenAI ada-2 | **41.45** | **46.70** | **0.9312** |
| **ALGEN** (1k样本) | OpenAI 3-large | **41.31** | **46.28** | **0.9066** |

### 消融实验

| 泄露样本数 | T5 Rouge-L | GTR Rouge-L | OpenAI Rouge-L | 说明 |
|-----------|-----------|-------------|---------------|------|
| 1 | ~10 | ~8 | ~10 | 单样本即可部分成功 |
| 10 | ~20 | ~15 | ~18 | 快速提升 |
| 100 | ~35 | ~28 | ~32 | 接近饱和 |
| 1000 | 45.75 | 38.27 | 41.45 | 性能平台 |
| 3000+ | ~47 | ~40 | ~43 | 提升趋缓 |

### 关键发现
- ALGEN仅用1k样本就大幅超越Vec2Text（需要百万级样本），Rouge-L从17.38提升到45.75
- 对闭源的OpenAI嵌入同样有效（Rouge-L 41+），证明黑盒攻击的可行性
- 跨域攻击（在MultiHPLT上训练、在mMarco上攻击）仍然有效，Rouge-L约20
- 跨语言攻击可行，英语训练的攻击模型可以反转法语/德语/西语嵌入
- 现有防御（WET水印、Shuffling、高斯噪声、差分隐私）均无法有效抵御ALGEN——只有添加极大噪声才能降低攻击性能，但同时会严重损害嵌入的下游任务效用

## 亮点与洞察
- 将嵌入空间对齐从NLP对齐任务迁移到安全攻击场景是一个优雅的思路。线性对齐的闭式解意味着只用一行矩阵乘法就完成了跨模型的"翻译"，计算成本几乎为零
- 仅1个泄露样本即可发起攻击，这大幅降低了攻击门槛，意味着向量数据库中即使极少量的数据泄露也可能导致大规模隐私风险
- 实验中命名实体（组织名、国家名等）也被成功恢复，说明攻击可以泄露真正敏感的信息

## 局限与展望
- 攻击性能的上限受限于本地生成器的解码能力（Rouge-L上限约54），改进生成器可能进一步提升
- 论文没有提出有效的防御方案，现有防御均被证明无效
- 线性对齐假设可能在嵌入空间差异很大时失效（如图像嵌入 vs 文本嵌入）
- 未考虑对话/长文本嵌入的反转，当前仅限于句子级别

## 相关工作与启发
- **vs Vec2Text (Morris et al.)**: Vec2Text需要百万级训练数据和迭代访问受害者编码器，ALGEN仅需1k样本和一次性线性对齐，效率高出数个数量级
- **vs Huang et al.**: 使用对抗训练进行嵌入对齐但需要8k样本且不可微，ALGEN的最小二乘对齐更简洁
- **vs Chen et al.**: 扩展了多语言反转攻击，ALGEN同样支持跨语言但所需数据量更少

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 少样本攻击范式是嵌入安全领域的重要进展，一步线性对齐优雅简洁
- 实验充分度: ⭐⭐⭐⭐⭐ 6种受害者模型、4种语言、3种数据集、4种防御的全面评估
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，推导完整
- 价值: ⭐⭐⭐⭐⭐ 揭示了嵌入服务的严重隐私风险，对向量数据库安全有重大警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Language Does Matter for Cross-Domain Few-Shot Visual Feature Enhancement](../../CVPR2026/others/language_does_matter_for_cross-domain_few-shot_visual_feature_enhancement.md)
- [\[ICML 2025\] Feedforward Few-shot Species Range Estimation](../../ICML2025/others/feedforward_few-shot_species_range_estimation.md)
- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](../../ICML2025/others/cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)

</div>

<!-- RELATED:END -->
