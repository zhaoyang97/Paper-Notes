---
title: >-
  [论文解读] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders
description: >-
  [NeurIPS 2025][可解释性][稀疏自编码器] 系统比较了Transformer前馈层（FF）的键值记忆特征与稀疏自编码器（SAE）学到的特征的可解释性，发现两者在现有评测指标上表现相当，FF-KV在某些方面甚至更优，质疑了SAE作为特征发现工具的必要性。 大语言模型（LLM）的可解释性研究近年来经历了从"自顶向下…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "稀疏自编码器"
  - "前馈层"
  - "键值记忆"
  - "特征发现"
---

# Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders

**会议**: NeurIPS 2025  
**arXiv**: [2510.22332](https://arxiv.org/abs/2510.22332)  
**代码**: [https://muyo8692.com/projects/ff-kv-sae](https://muyo8692.com/projects/ff-kv-sae)  
**领域**: 模型压缩 / 可解释性  
**关键词**: 稀疏自编码器, 前馈层, 键值记忆, 可解释性, 特征发现

## 一句话总结

系统比较了Transformer前馈层（FF）的键值记忆特征与稀疏自编码器（SAE）学到的特征的可解释性，发现两者在现有评测指标上表现相当，FF-KV在某些方面甚至更优，质疑了SAE作为特征发现工具的必要性。

## 研究背景与动机

大语言模型（LLM）的可解释性研究近年来经历了从"自顶向下"到"自底向上"特征发现的范式转变。在特征发现时代，两大趋势同时出现：(1) 训练外部代理模块（如稀疏自编码器SAE）来分解神经元激活，(2) 发展新的综合可解释性基准（如SAEBench）来评估特征质量。

然而，一个被忽视的关键问题是：**代理模块学到的特征是否真的比模型原始参数中已有的特征更好？** FF层本身可以被视为键值记忆（key-value memories）——Key矩阵 $\mathbf{W}_K$ 的每一行是一个"键"，Value矩阵 $\mathbf{W}_V$ 的每一行是对应的"值"（特征向量）。FF层天然地将激活分解为一组特征向量，这与SAE的操作在结构上完全一致（都是MLP架构）。

代理方法和FF-KV分析各有互补优势：SAE有处理超位（superposition）的理论动机，但也引入了额外的偏差——特定特征被重复发现、代理模块"幻觉"出不存在的特征、需要额外计算成本。而且FF的激活本身已经是自然稀疏的。**如果FF-KV和SAE分析得到可比的结果，根据奥卡姆剃刀原则，直接分析FF-KV更为可取。**

## 方法详解

### 整体框架

将FF层的Key-Value结构直接作为特征发现方法，用现代可解释性基准（SAEBench）和人工评估系统地比较FF-KV特征与SAE/Transcoder特征的可解释性。进一步分析代理模块发现的特征与原始FF模块特征的重叠程度（忠实度分析）。

### 关键设计

1. **FF-KV方法族**: 

    - **Vanilla FF-KV**: 直接将FF层的激活 $\phi(\mathbf{x}_{FF_{in}}\mathbf{W}_K + \mathbf{b}_K)$ 作为特征激活，$\mathbf{W}_V$ 的行作为特征向量
    - **TopK FF-KV**: 对FF激活施加Top-k稀疏化，仅保留最大的 $k$ 个激活值，与SAE的稀疏性对齐
    - **Normalized FF-KV**: 对 $\mathbf{W}_V$ 的每行进行L2归一化，将折扣的范数权重加到激活上，避免特征向量范数差异造成的偏差
    - **SwiGLU兼容**: 对现代LM采用的SwiGLU门控激活，上述方法均可自然适配

2. **SAEBench评估框架**: 使用8个互补指标全面评估：

    - **Feature Alive Rate**: 活跃特征比例
    - **Explained Variance**: 重建质量（FF-KV自动满分）
    - **Absorption Score**: 概念是否被过度分割（越低越好）
    - **Sparse Probing**: 特征的判别性和泛化能力
    - **Auto-Interpretation**: LLM能否用自然语言总结特征的激活模式
    - **SCR/TPP**: 虚假相关特征的解纠缠能力
    - **RAVEL**: 同一实体不同属性的可分性和可控性

3. **忠实度分析**: 使用Transcoder（TC）作为分析对象（TC是FF-KV的最近对应物），检查TC特征与原始FF特征的重叠度——即代理模块是否真正"翻译"了原始模块的工作方式，还是"幻觉"出了新特征。

### 损失函数 / 训练策略

FF-KV方法本身**不需要任何训练**（直接使用模型原有参数），这是其相对于SAE的重要优势。SAE使用重建损失+稀疏正则化训练，需要额外的计算资源。评估使用预训练的SAE（Gemma Scope、Llama Scope等）。

## 实验关键数据

### 主实验（SAEBench评测，Gemma-2-2B Layer 13）

| 方法 | Absorption↓ | Sparse Prob.↑ | AutoInterp↑ | RAVEL-ISO↑ | SCR(k=20)↑ |
|------|------------|--------------|-------------|------------|------------|
| SAE | 0.087 | **0.846** | **0.782** | **0.985** | **0.170** |
| Transcoder | 0.025 | 0.854 | 0.790 | 0.940 | 0.104 |
| FF-KV | **0.000** | 0.827 | 0.710 | 0.952 | 0.041 |
| TopK FF-KV | 0.000 | 0.768 | 0.772 | 0.943 | 0.045 |
| Random Transformer | 0.007 | 0.798 | 0.679 | - | 0.004 |

### 消融实验（人工评估，50个特征/方法）

| 方法 | 表面特征 | 概念特征 | 不可解释 | 来源判断准确率 |
|------|---------|---------|---------|-------------|
| FF-KV | 6 | 8 | **36** | **0.86** |
| TopK FF-KV | 9 | 9 | 32 | 0.28 |
| SAE | 6 | **9** | 35 | 0.13 |
| Transcoder | **16** | **11** | 23 | 0.18 |

### 关键发现

- **整体可比性**: SAE和FF-KV在SAEBench的8个指标上呈现相似范围的分数，绝对差异通常远小于种子/层间方差
- **FF-KV优势——Absorption**: FF-KV的Absorption分数近乎为0（远优于SAE的0.087），意味着FF-KV特征不会将简单概念过度分割——特征冗余度更低
- **SAE微弱优势——AutoInterp/SCR**: SAE在自动解释和虚假相关消除上略优，但差距不大
- **概念特征数量相当**: 人工评估中，FF-KV和SAE发现的概念级特征数量几乎相同（8 vs 9）
- **忠实度质疑**: 大多数Transcoder特征在原始FF模块中找不到对应物——代理模块可能"幻觉"出新特征而非翻译原始模块
- **Random Transformer基线**: 随机初始化的Transformer也能获得非trivial的可解释性分数，进一步质疑了代理方法的特征质量

## 亮点与洞察

- 对SAE主导的可解释性研究范式提出了重要质疑——一个不需要额外训练的基线方法就能达到相当的效果
- Absorption分数的比较尤其有说服力：SAE的特征分割倾向是其固有缺陷，而FF-KV天然避免了这个问题
- 忠实度分析揭示代理模块可能"幻觉"特征的问题，与已有批评（SAE能解释随机Transformer）相呼应
- 核心信息简洁有力：FF-KV应作为可解释性研究的强基线

## 局限与展望

- SAE理论上能处理超位（superposition），但当前评测指标可能无法充分捕捉这一优势
- SCR/TPP指标不稳定，结果应作为辅助参考
- 人工评估仅有一位标注者，可能引入个人偏差
- 仅在Gemma-2和Llama-3.1上评测，其他架构（如MoE模型）的行为有待验证
- 未讨论FF-KV方法在模型行为控制（steering）上的表现，这是SAE的重要应用场景

## 相关工作与启发

- **vs Geva et al. (2021)**: 最早提出FF-as-KV-memories的视角，本文用现代基准系统验证了这一观点
- **vs SAE (Cunningham et al., Bricken et al.)**: SAE在早期展示了令人鼓舞的可解释特征，但本文表明FF-KV能达到相当水平
- **vs Transcoder (Dunefsky et al.)**: Transcoder是FF-KV的最近代理对应物，本文发现其特征与原始FF重叠度低
- **vs 对SAE的批评 (Makelov et al., Huang et al.)**: 本文提供了额外证据支持对SAE一般性优势的质疑——FF-KV作为"零成本"基线已足够强

## 评分

- 新颖性: ⭐⭐⭐⭐ 将被忽视的FF-KV视角重新引入现代可解释性研究，切入角度新颖但方法本身并不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 自动评估（8指标）+人工评估+忠实度分析+多模型+多变体，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，实验组织有序，但LaTeX渲染问题影响了部分公式的可读性
- 价值: ⭐⭐⭐⭐⭐ 对可解释性社区极具警示意义——在追求更复杂的代理方法之前，先确认简单基线的表现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[NeurIPS 2025\] From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit](from_flat_to_hierarchical_extracting_sparse_representations_with_matching_pursui.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](../../ICML2026/interpretability/sparse_autoencoders_are_topic_models.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](../../ACL2026/interpretability/adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)

</div>

<!-- RELATED:END -->
