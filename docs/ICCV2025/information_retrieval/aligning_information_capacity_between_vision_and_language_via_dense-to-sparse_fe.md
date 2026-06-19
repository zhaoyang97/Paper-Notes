---
title: >-
  [论文解读] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation
description: >-
  [ICCV 2025][信息检索/RAG][图文匹配] 提出D2S-VSE框架，通过两阶段训练（稠密文本预训练+稠密到稀疏特征蒸馏微调）增强视觉语义嵌入的信息容量，解决图文匹配中图像与文本信息密度不对称的核心问题。 信息密度不对称问题 图文匹配任务的核心挑战在于视觉信号与文本信号之间存在固有的信息密度差异…
tags:
  - "ICCV 2025"
  - "信息检索/RAG"
  - "图文匹配"
  - "视觉语义嵌入"
  - "信息容量"
  - "稠密到稀疏蒸馏"
  - "跨模态检索"
---

# Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation

**会议**: ICCV 2025  
**代码**: [https://d2s-vse.github.io](https://d2s-vse.github.io)  
**领域**: 信息检索  
**关键词**: 图文匹配, 视觉语义嵌入, 信息容量, 稠密到稀疏蒸馏, 跨模态检索

## 一句话总结

提出D2S-VSE框架，通过两阶段训练（稠密文本预训练+稠密到稀疏特征蒸馏微调）增强视觉语义嵌入的信息容量，解决图文匹配中图像与文本信息密度不对称的核心问题。

## 研究背景与动机

### 信息密度不对称问题

图文匹配任务的核心挑战在于视觉信号与文本信号之间存在固有的信息密度差异。视觉信号是对真实世界现象的自然、客观记录，包含大量细粒度细节；而文本信号则是经过人类主观解读的语义表达，通常简短而稀疏。同一图像可能因不同视角被多种方式描述，这被称为"歧义问题"或"稀疏标注问题"。

### 现有方法的不足

现有方法（如PCME、DivE、AVSE）试图通过学习一组嵌入来匹配歧义样本，但这种做法降低了每个嵌入的**信息容量**（Information Capacity）。当文本嵌入的信息容量有限时，它们容易被具有局部相似语义的负样本干扰，导致检索性能下降。

### 核心洞察

**为什么信息容量至关重要？** 作者论证了从稠密图像描述中提取的文本嵌入天然具有更大的信息容量。如果能够在预训练阶段通过稠密文本对齐提升图像嵌入的信息容量，再在微调阶段将稠密文本嵌入的信息蒸馏到稀疏文本嵌入中，就能同时提升视觉和文本两侧的信息容量，实现更精准的图文匹配。

## 方法详解

### 整体框架

D2S-VSE是一个两阶段训练框架：

1. **预训练阶段**：利用LLaVA生成稠密文本描述，将图像与稠密文本对齐，增强视觉语义嵌入的信息容量
2. **微调阶段**：在图像-稀疏文本对上进行微调，同时通过稠密到稀疏蒸馏增强稀疏文本嵌入的信息容量

推理阶段只需要图像和数据集中的稀疏文本，不需要稠密文本，因此不引入额外计算开销。

### 阶段一：稠密文本预训练

**稠密文本生成**：使用LLaVA对数据集中的每张图像生成详细的文本描述（prompt为"Describe this image in detail"），超参数设置为Top-P=0.9, temperature=0.2, max-new-tokens=500。每张图像生成一条稠密描述，对应1/5的稀疏文本数量。

**图像-稠密文本对比学习**：使用三元组损失函数对齐图像和稠密文本：

$$\mathcal{L}_{pretrain} = \sum_{(I,T_d) \in D} [\alpha - S(I, T_d) + S(I, \hat{T}_d)]_+ + [\alpha - S(I, T_d) + S(\hat{I}, T_d)]_+$$

其中$\alpha$是边界参数，$\hat{I}$和$\hat{T}_d$分别是最难的负样本。

**为什么稠密文本能增强信息容量？** 稠密文本包含了图像中物体属性、关系、动作和上下文信息的全面描述。通过在相似信息密度的数据上训练，模型学习到具有更大信息容量的嵌入，能够更全面地理解视觉内容。

### 阶段二：稀疏描述微调

预训练虽然增强了嵌入的信息容量，但存在训练-测试数据的gap（训练用稠密文本，测试用稀疏文本）。单纯在稀疏文本上微调不能让稀疏文本嵌入包含更丰富的语义。

**核心创新：Transformer解码器**

受掩码信号建模（MAE）启发，将稀疏文本视为稠密文本的**掩码版本**。设计了一个Transformer解码器，通过特征蒸馏从稀疏文本嵌入中重建稠密文本嵌入：

$$\hat{t}_s = Decoder([w_s, m]) + t_s$$

其中$w_s$是稀疏文本的词嵌入，$m$是可学习的mask tokens（100个），$t_s$是稀疏文本嵌入。解码器输出重建tokens $\bar{m}$，取平均后与$t_s$相加得到最终嵌入$\hat{t}_s$。

**为什么用可学习mask tokens而非其他方法？** 这些mask tokens的作用是预测稀疏文本中缺失的上下文信息。通过与词嵌入拼接送入解码器，它们学会从稀疏文本的有限信息中推断出稠密文本所包含的更丰富语义，从而增强信息容量。

### 损失函数 / 训练策略

微调阶段联合优化两个目标：

**蒸馏损失**（负余弦相似度）：
$$\mathcal{L}_{distill}(t_d, \hat{t}_s) = 1 - \frac{t_d \cdot \hat{t}_s}{\|t_d\| \cdot \|\hat{t}_s\|}$$

**对齐损失**：
$$\mathcal{L}_{align} = \sum_{(I,T_s) \in D} [\alpha - S(I, T_s) + S(I, \hat{T}_s)]_+ + [\alpha - S(I, T_s) + S(\hat{I}, T_s)]_+$$

**总损失**：$\mathcal{L} = \mathcal{L}_{align} + \mathcal{L}_{distill}$

训练策略：预训练30 epochs（AdamW, lr=0.0005, batch=128），微调阶段冻结稠密文本编码器，仅微调图像和稀疏文本编码器。

## 实验关键数据

### 主实验

**Flickr30K（ViT-Base-224 + BERT-base）**：

| 方法 | Text R@1 | Text R@5 | Image R@1 | Image R@5 | rSum |
|------|----------|----------|-----------|-----------|------|
| VSE++ | 71.8 | 92.8 | 59.4 | 84.7 | 496.1 |
| LAPS | 74.0 | 93.4 | 62.5 | 87.3 | 507.3 |
| AVSE | 76.0 | 94.6 | 62.7 | 88.4 | 512.3 |
| **D2S-VSE** | **82.8** | **96.1** | **68.5** | **91.3** | **531.9** |

D2S-VSE在Text R@1上比AVSE提升6.8%，在Image R@1上提升5.8%，rSum提升19.6。

**Flickr30K（Swin-Base-384 + BERT-base）**：

| 方法 | Text R@1 | Image R@1 | rSum |
|------|----------|-----------|------|
| AVSE | 87.1 | 73.6 | 548.2 |
| **D2S-VSE** | **87.8** | **75.7** | **553.2** |

### 消融实验

**两阶段框架各组件效果（Flickr30K，ViT-Base-224）**：

| Pretrain | Align | Distill | Text R@1 | Image R@1 | rSum |
|----------|-------|---------|----------|-----------|------|
| ✓ | | | 56.2 | 41.7 | 425.2 |
| | ✓ | | 76.3 | 61.2 | 509.5 |
| ✓ | ✓ | | 79.2 | 66.8 | 525.7 |
| | ✓ | ✓ | 76.0 | 62.9 | 512.1 |
| ✓ | ✓ | ✓ | **82.8** | **68.5** | **531.9** |

关键发现：预训练阶段增强了图像嵌入的信息容量（Text R@1提升2.9%）；蒸馏进一步增强了稀疏文本嵌入的信息容量（Text R@1再提升3.6%）。

**蒸馏函数对比**：

| 蒸馏函数 | Text R@1 | Image R@1 | rSum |
|----------|----------|-----------|------|
| L1距离 | 81.9 | 68.0 | 528.9 |
| L2距离 | 82.1 | 67.9 | 530.2 |
| **负余弦相似度** | **82.8** | **68.5** | **531.9** |

**可学习Token位置策略**：

| 位置策略 | Text R@1 | Image R@1 | rSum |
|----------|----------|-----------|------|
| Prefix | 81.7 | 68.1 | 529.3 |
| Postfix | 82.0 | 67.8 | 529.5 |
| **Surround** | **82.8** | **68.5** | **531.9** |

Surround方式最优，因为稀疏文本在稠密文本中的位置未知，前后包围的方式能同时从前后上下文预测语义。

### 关键发现

1. **信息容量是核心**：实验清晰验证了预训练增强图像嵌入、蒸馏增强文本嵌入这两步的独立贡献
2. **解码器设计**：100个可学习token、4层深度、4个注意力头为最优配置
3. **与MAE的区别**：在MAE中过深的解码器会削弱编码器能力，但D2S-VSE的检索性能依赖解码器的预测能力，因此更深的解码器通常更好（4层最优）
4. **骨干泛化性**：方法在ViT、Swin Transformer、ResNet+GRU等多种骨干上均一致有效

## 亮点与洞察

1. **信息容量视角的独特性**：从信息容量角度重新审视图文匹配问题，提供了比"歧义问题"更本质的分析框架
2. **零推理开销**：推理时只需图像和稀疏文本，稠密文本和解码器均不参与，彻底无额外开销
3. **掩码信号建模的新应用**：将MAE的思想创新性地用于跨模态蒸馏，将稀疏文本视为稠密文本的掩码版本
4. **方法简洁有力**：核心思想简单清晰，实现高效，效果显著

## 局限与展望

1. 依赖LLaVA生成稠密文本，生成质量直接影响预训练效果
2. 当前一对一的图像-稠密文本对应关系可能不够灵活，多角度的稠密描述可能更有帮助
3. 未探索在更大规模预训练模型（如CLIP等）上的集成效果
4. 解码器在微调阶段引入了额外的训练开销，尽管推理无开销

## 相关工作与启发

- **AVSE**：提出不对称多视图嵌入匹配，是最强的对比方法
- **Long-CLIP / LoTLIP**：从长文本视角改进CLIP，但未充分利用稠密到稀疏的蒸馏
- **MAE**：掩码自编码器思想被本文创新性地用于跨模态蒸馏
- 启发：信息容量对齐的思路可推广到其他多模态任务中的信息不对称问题

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Revela: Dense Retriever Learning via Language Modeling](../../ICLR2026/information_retrieval/revela_dense_retriever_learning_via_language_modeling.md)
- [\[ACL 2025\] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](../../ACL2025/information_retrieval/drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](../../ACL2025/information_retrieval/atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[NeurIPS 2025\] Learning Task-Agnostic Representations through Multi-Teacher Distillation](../../NeurIPS2025/information_retrieval/learning_task-agnostic_representations_through_multi-teacher_distillation.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](../../ACL2026/information_retrieval/can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)

</div>

<!-- RELATED:END -->
