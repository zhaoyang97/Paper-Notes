---
title: >-
  [论文解读] Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text
description: >-
  [ACL 2025][掩码语言模型] TIACBM 提出了一种任务感知的反课程掩码微调策略：利用下游任务知识（如情感极性、词性标签）决定哪些 token 被掩码，并采用周期衰减的掩码率，在情感分析、文本分类和作者归属三个任务上均取得统计显著的性能提升。 掩码语言建模（MLM）是预训练语言模型的核心技术…
tags:
  - "ACL 2025"
  - "掩码语言模型"
  - "反课程学习"
  - "任务感知掩码"
  - "微调策略"
  - "情感分析"
---

# Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text

**会议**: ACL 2025  
**arXiv**: [2502.12953](https://arxiv.org/abs/2502.12953)  
**代码**: 有 ([https://github.com/JarcaAndrei/TIACBM](https://github.com/JarcaAndrei/TIACBM))  
**领域**: 其他  
**关键词**: 掩码语言模型, 反课程学习, 任务感知掩码, 微调策略, 情感分析

## 一句话总结

TIACBM 提出了一种任务感知的反课程掩码微调策略：利用下游任务知识（如情感极性、词性标签）决定哪些 token 被掩码，并采用周期衰减的掩码率，在情感分析、文本分类和作者归属三个任务上均取得统计显著的性能提升。

## 研究背景与动机

掩码语言建模（MLM）是预训练语言模型的核心技术，但存在两个被忽视的问题：

**随机选择被掩码的 token**：标准 MLM 中哪些 token 被掩码完全随机，未利用任务相关知识

**固定掩码比例**：通常在整个训练过程中保持 15% 的掩码率不变

已有研究（Ankner et al., 2024; Yang et al., 2023）发现衰减掩码率对文本预训练更有效——这本质上对应**反课程策略**（hard-to-easy），因为更高的掩码率使学习任务更困难。但这些工作仅关注预训练阶段，且未利用下游任务信息。

TIACBM 的核心创新在于：**在微调阶段引入 MLM 作为辅助目标**，并结合任务知识来选择性地掩码重要 token。

## 方法详解

### 整体框架

TIACBM 包含两个核心组件：
1. **周期衰减掩码率**：在训练中使用从高到低的掩码率向量 $\mathbf{r} = \{r_1 \geq \cdots \geq r_K\}$，每 K 步重置一次（周期性）
2. **任务感知 token 选择**：根据任务特定的 `task_relevance` 函数计算每个 token 的重要性，按概率抽样选择被掩码的 token

### 关键设计

1. **情感分析任务的掩码策略**

    - 核心假设：最具主观性的词是最重要的特征
    - 实现：使用 SentiWordNet 3.0 查找每个词的极性分数（正面 + 负面 = 主观性），主观性高的词更可能被掩码
    - 通过 Lesk 算法找到最可能的 synset 来确定正面/负面分数
    - 重要性分数：$s_i = s_{pos}^i + s_{neg}^i$

2. **文本分类（主题）任务的掩码策略**

    - 核心假设：实词（名词、动词、形容词、副词）比功能词更与主题相关
    - 实现：先对非实词赋予 0 重要性，对实词使用预训练模型的注意力权重作为重要性
    - 注意力重要性：对所有 attention block 和 head 取平均 $\mathbf{a} = \frac{1}{B \cdot H \cdot |\mathbf{x}|} \sum_h \sum_b \sum_j A_{b,j}^h$
    - 最终：$s_i = a_i$ 如果 $x_i$ 是实词，否则 $s_i = 0$

3. **作者归属任务的掩码策略**

    - 核心假设：功能词（介词、冠词、连词、符号、标点）反映写作风格
    - 实现：与文本分类相反，掩码功能词而非实词
    - 最终：$s_i = a_i$ 如果 $x_i$ 是功能词，否则 $s_i = 0$

4. **周期衰减掩码率**

    - 创建 K 个衰减掩码率的向量，每 K 个迭代周期重置
    - 动机：周期性重置防止模型在低掩码率阶段过拟合，同时保持 hard-to-easy 的反课程效果
    - 掩码的 token 数量 $N = \lfloor |\mathbf{x}| \cdot r_t \rfloor$

### 训练策略

微调同时优化分类损失和 MLM 重建损失，掩码策略影响 MLM 部分的重建目标。通过战略性地优先掩码判别性特征来防止特征共适应（feature co-adaptation），类似于 Dropout 的正则化效果。

## 实验关键数据

### 主实验——BERT/RoBERTa（Table 1 摘要）

| 策略 | Reuters(F1) | 20News(Acc) | SST2(Acc) | PAN19-P1(Acc) | PAN19-P5(Acc) |
|------|------------|-------------|-----------|---------------|---------------|
| 传统微调 | 90.61 | 84.63 | 93.38 | 58.24 | 66.10 |
| 固定掩码(15%) | 90.81 | 84.98 | 93.94 | 47.50 | 65.76 |
| Poesina (CL++) | 90.72 | 82.30 | 94.00 | 44.76 | 68.66 |
| Ankner (衰减) | 90.99 | 85.39 | 93.83 | 46.03 | 65.55 |
| 周期衰减 | 90.96 | 84.88 | 94.10 | 51.94 | 69.28 |
| **TIACBM** | **91.20** | **85.65** | **94.61** | **60.60** | **69.94** |

RoBERTa 上也有一致提升，所有结果均通过 Cochran's Q 检验（p < 0.001）。

### GPT-2 实验（Table 2）

| 策略 | SST2(Acc) | PAN19-P1(Acc) | PAN19-P5(Acc) |
|------|-----------|---------------|---------------|
| 传统微调 | 92.35 | 67.96 | 38.16 |
| **TIACBM** | **92.96** | **73.44** | **42.90** |

证明 TIACBM 不限于掩码语言模型，对 GPT-2 等自回归模型也有效。

### 关键发现

1. **TIACBM 在所有任务、所有模型上均为最优**，且改进具有统计显著性
2. **PAN19 作者归属任务上提升最显著**（BERT 上从 58.24% → 60.60%），因为功能词掩码直接针对作者风格特征
3. **周期衰减优于简单衰减**：对比 Ankner 的非周期衰减，周期版（无任务信息）在 PAN19 上已有提升
4. **任务信息是核心贡献**：对比周期衰减（无任务信息）和 TIACBM（有任务信息），后者在所有任务上均更优
5. **MLM 不仅有利于预训练，对微调也有帮助**——这是一个重要的经验性发现

## 亮点与洞察

1. **简洁有效**：方法实现简单，无需额外模型或数据，仅需任务特定的 token 重要性函数
2. **理论直觉清晰**：掩码判别性特征 → 防止特征共适应 → 类似有目标的数据增强/正则化
3. **适用性广**：在掩码LM（BERT/RoBERTa）和自回归LM（GPT-2）上均有效
4. **课程学习视角新颖**：首次明确将掩码率调度与反课程学习联系，并在微调阶段验证

## 局限与展望

- 没有通用的最优掩码率调度，K 和衰减形式需用户调优
- 任务特定的 token 重要性函数设计可能在某些任务上不直观（如回归任务、多标签任务）
- 仅在 3 类任务上实验，未涵盖 NER、QA、摘要等更多任务类型
- 未探讨与其他正则化技术（如 Dropout、R-Drop）的交互效果

## 相关工作与启发

- Ankner et al. (2024) 和 Yang et al. (2023) 发现衰减掩码率对预训练有利，本文将其扩展到微调并加入任务信息
- Poesina et al. (2024) 的 Cart-Stra-CL++ 使用数据制图进行易→难课程学习，但需要双倍训练时间
- Jarca et al. (2024) 在图像域发现易→难课程对掩码建模更好，但文本域结论相反（hard-to-easy 更优）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 任务感知掩码 + 周期反课程的组合是原创的贡献
- **实验充分度**: ⭐⭐⭐⭐ — 3 模型 × 4 数据集 × 5 基线的系统对比，统计检验充分
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，算法伪代码规范
- **价值**: ⭐⭐⭐ — 改进幅度有限（~1%），但方法简洁，易于集成到现有微调管道

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)

</div>

<!-- RELATED:END -->
