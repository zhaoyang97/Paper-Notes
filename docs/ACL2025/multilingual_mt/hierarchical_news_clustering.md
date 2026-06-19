---
title: >-
  [论文解读] Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings
description: >-
  [多语言/翻译][Matryoshka embeddings] 提出利用多语言Matryoshka嵌入实现层级化新闻聚类的方法：嵌入的不同维度子集对应不同粒度的语义相似性（主题→话题→事件），配合改进的层级凝聚聚类算法，在SemEval 2022 Task 8上达到SOTA（Pearson ρ=0.816）。
tags:
  - "多语言/翻译"
  - "Matryoshka embeddings"
  - "multilingual"
  - "hierarchical clustering"
  - "news similarity"
  - "agglomerative clustering"
---

# Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings

**会议/期刊**: ACL 2025  
**arXiv**: [2506.00277](https://arxiv.org/abs/2506.00277)  
**代码**: [GitHub](https://github.com/hanshanley/multilingual-matryoshka-news)  
**领域**: 多语言 / 新闻聚类  
**关键词**: Matryoshka embeddings, multilingual, hierarchical clustering, news similarity, agglomerative clustering  

## 一句话总结

提出利用多语言Matryoshka嵌入实现层级化新闻聚类的方法：嵌入的不同维度子集对应不同粒度的语义相似性（主题→话题→事件），配合改进的层级凝聚聚类算法，在SemEval 2022 Task 8上达到SOTA（Pearson ρ=0.816）。

## 研究背景与动机

- **问题定义**：全球化新闻生态中，需要在不同粒度（事件/话题/主题）上对多语言新闻进行聚类分析，以理解媒体覆盖模式和跨语言信息传播。
- **现有不足**：(1) 当前LLM-based方法多为单语言、不可扩展或无法区分不同粒度的相似性；(2) decoder-based LLM（如GPT-4）处理大规模文档成本过高；(3) encoder-based模型仅通过余弦相似度衡量"相似性"，定义模糊；(4) 聚类方法需要预知簇数量。
- **核心洞察**：Matryoshka表示学习的层级嵌套结构天然适合编码不同粒度的语义信息——低维捕获粗粒度主题，高维捕获细粒度事件。
- **本文方案**：训练多语言Matryoshka嵌入 + 设计层级凝聚聚类算法，无需预知簇数量即可自动发现不同粒度的新闻组。

## 方法详解

### 整体框架

两阶段方法：
1. **嵌入训练**：基于修改的AngIE损失训练多语言Matryoshka嵌入，在不同维度编码不同粒度的相似性
2. **层级聚类**：基于Reciprocal Agglomerative Clustering (RAC)的改进算法，逐层使用不同维度子集进行聚类

### 关键设计

- **层级化Matryoshka训练**：在不同维度施加不同相似性阈值——$d/4$维度区分"非常不相似" vs 其余；$d/2$维度区分"有些不相似" vs 其余；$d$维度区分所有四级相似性。这迫使嵌入在低维学习粗粒度概念，高维学习细粒度细节
- **修改的AngIE损失**：$\mathcal{L}_{mat} = \mathcal{L}_{\text{AngIE}_{diss}}(\mathbf{H}_{d/4}) + \mathcal{L}_{\text{AngIE}_{somewhat}}(\mathbf{H}_{d/2}) + \mathcal{L}_{\text{AngIE}_{same}}(\mathbf{H}_{d})$，结合余弦、对比和角度三个子目标
- **SimCSE增强**：训练时对每个样本使用不同dropout mask编码两次，产生隐式正样本对，强化单语言嵌入空间质量

### 聚类算法

三层层级聚类：
1. **第1层（主题）**：使用$d/4$维嵌入 + RNN合并，阈值$\lambda_1$
2. **第2层（话题）**：在第1层簇内部，使用$d/2$维嵌入 + RNN合并，阈值$\lambda_2$
3. **第3层（事件）**：在第2层簇内部，使用完整$d$维嵌入 + RNN合并，阈值$\lambda_3$

### 数据增强

- **风格增强**：GPT-4o对每篇文章生成3种不同风格的改写
- **实体敏感性**：使用Spacy+T5替换命名实体生成"有些相似"样本
- **语言扩展**：将原始10种语言扩展至54种，最终训练集扩展到410万文章对

## 实验

### 主实验结果（SemEval 2022 Task 8）

| 模型 | SE-22 (Pearson ρ) | SE-22 Extended |
|------|:-:|:-:|
| mE5-base (baseline) | 0.604 | 0.582 |
| fine-mE5-base (ours) | 0.817 | 0.812 |
| mat-mE5-base-192 (ours) | 0.799 | 0.808 |
| mat-mE5-base-384 (ours) | 0.792 | **0.816** |
| GateNLP-UShef (prev. SOTA) | 0.801 | – |

### 消融实验

| 消融项 | SE-22 ρ (192d) | SE-22 Ext ρ (192d) |
|--------|:-:|:-:|
| 完整模型 | 0.799 | 0.808 |
| 去除SimCSE dropout | 0.693 | 0.733 |
| 去除对比损失 | ≈0 | ≈0 |
| 仅用原始SE-22数据训练 | 0.828 | 0.706 |

### 聚类性能（Miranda数据集，BERTopic F1）

| 模型 | Precision | Recall | F1 |
|------|:-:|:-:|:-:|
| mE5-base | 0.8507 | 0.3715 | 0.5171 |
| mat-mE5-base-192 | 0.7895 | 0.8971 | **0.8399** |
| fine-mE5-base | 0.7791 | 0.5735 | 0.6607 |

### 关键发现

- Matryoshka嵌入在区分不同相似度级别上显著优于传统微调嵌入（AUROC在各级别均最高）
- SimCSE的dropout正样本对训练至关重要，移除后性能从0.799降至0.693
- 数据增强对多语言泛化不可或缺：仅用原始数据训练在扩展测试集上降至0.706
- 在BERTopic聚类中，Matryoshka-192维即可达到F1=0.84，大幅超越全维度fine-tuned模型（0.66）
- 多语言对齐：平均relational similarity与英语达0.753，最高为葡萄牙语(0.839)，最低为缅甸语(0.452)

## 亮点

- 将Matryoshka表示学习从"不同维度学习相同信息"重新定义为"不同维度学习不同粒度信息"，概念创新且直觉清晰
- 层级聚类算法自然匹配嵌入的层级结构，无需预知簇数量
- 支持54种语言的大规模多语言评估
- 训练数据增强策略（风格改写+实体替换+翻译）设计系统化

## 局限性

- 聚类阈值 $\lambda_1, \lambda_2, \lambda_3$ 需要在验证集上经验性调整，不同数据分布可能需要重新调参
- 仅支持512 token的上下文窗口，对长文新闻可能丢失信息
- 大量依赖GPT-4o进行数据增强和翻译，成本高且引入翻译偏差
- 低资源语言（缅甸语、卡纳达语）的嵌入对齐质量较差

## 相关工作

- **语义嵌入**：SimCSE (Gao et al., 2021)、E5 (Wang et al., 2022)、AngIE (Li & Li, 2024)
- **Matryoshka学习**：MRL (Kusupati et al., 2022) 的嵌套表示学习
- **新闻聚类**：BERTopic (Grootendorst, 2022)、Miranda et al. (2018) 的多语言新闻数据集
- **SemEval任务**：SemEval 2022 Task 8 (Chen et al., 2022) 的多语言新闻相似度评估
- **凝聚聚类**：RAC (Sumengen et al., 2021) 的互为最近邻聚合算法

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](less_but_better_efficient_multilingual_expansion.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)
- [\[ACL 2025\] Context Augmented Token-Level Post-Editing for Human Interpreting](context_augmented_token-level_post-editing_for_human_interpreting.md)
- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [\[ACL 2025\] Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](memorization_inheritance_seqkd.md)

</div>

<!-- RELATED:END -->
