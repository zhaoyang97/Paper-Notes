---
title: >-
  [论文解读] Align-Pro: Align Protein Representations Through Multi-Modal Learning
description: >-
  [ACL 2025][计算生物][蛋白质表示学习] Align-Pro通过多模态对比学习框架，将蛋白质的序列、结构和功能描述三种模态的表示对齐到统一的嵌入空间中，从而实现跨模态的蛋白质检索、分类和功能预测。 领域现状：蛋白质研究涉及多种数据模态：氨基酸序列（一维）、三维结构（通过AlphaFold等工具预测）、以及自然语言形…
tags:
  - "ACL 2025"
  - "计算生物"
  - "蛋白质表示学习"
  - "多模态对齐"
  - "序列-结构-功能"
  - "对比学习"
  - "预训练"
---

# Align-Pro: Align Protein Representations Through Multi-Modal Learning

**会议**: ACL 2025  
**领域**: 计算生物  
**关键词**: 蛋白质表示学习、多模态对齐、序列-结构-功能、对比学习、预训练

## 一句话总结
Align-Pro通过多模态对比学习框架，将蛋白质的序列、结构和功能描述三种模态的表示对齐到统一的嵌入空间中，从而实现跨模态的蛋白质检索、分类和功能预测。

## 研究背景与动机

**领域现状**：蛋白质研究涉及多种数据模态：氨基酸序列（一维）、三维结构（通过AlphaFold等工具预测）、以及自然语言形式的功能描述（如Gene Ontology注释）。目前各模态通常由独立的模型处理——ESM系列处理序列、GNN或Equivariant NN处理结构、BioBERT处理文本。

**现有痛点**：独立训练的模态编码器之间的表示空间不对齐，无法直接进行跨模态操作。例如，根据功能描述检索对应的蛋白质序列，或根据结构相似性推断功能，都需要额外的对齐步骤。现有的多模态蛋白质方法多局限于序列-结构的双模态对齐，忽略了功能文本这一重要模态。

**核心矛盾**：蛋白质的功能由其结构决定，结构又由序列编码，三者之间存在强关联，但现有方法难以在统一框架中捕捉这种三模态的对应关系。

**本文目标**：构建一个统一的三模态蛋白质表示学习框架，使得序列、结构和功能描述的嵌入在同一空间中对齐。

**切入角度**：借鉴CLIP等视觉-语言对比学习的成功经验，将两模态对齐推广到蛋白质领域的三模态场景。

**核心 idea**：通过三模态对比学习将序列、结构和功能描述的表示对齐到统一空间，实现"蛋白质版CLIP"。

## 方法详解

### 整体框架
Align-Pro包含三个编码器：序列编码器（基于ESM-2预训练模型）、结构编码器（基于GVP-GNN的等变图神经网络）、功能编码器（基于PubMedBERT的文本编码器）。三个编码器分别输出蛋白质的序列、结构和功能嵌入，通过三模态对比损失在大规模蛋白质数据库上训练，使得同一蛋白质的三种模态嵌入靠近，不同蛋白质的嵌入远离。

### 关键设计

1. **三模态对比学习目标（Tri-Modal Contrastive Loss）**:

    - 功能：对齐序列、结构和功能三种模态的表示空间
    - 核心思路：将传统的两模态InfoNCE损失扩展为三对两两对比损失的加权和：$\mathcal{L} = \alpha\mathcal{L}_{seq\text{-}struct} + \beta\mathcal{L}_{seq\text{-}func} + \gamma\mathcal{L}_{struct\text{-}func}$。每项对比损失使用温度缩放的余弦相似度。对同一蛋白质的跨模态对作为正样本，批内其他蛋白质作为负样本
    - 设计动机：三对两两损失比直接定义三元组损失更灵活，允许不同模态对之间有不同的对齐强度

2. **模态特定投影头（Modality-Specific Projection Heads）**:

    - 功能：将各编码器的输出映射到共享嵌入空间
    - 核心思路：每个编码器后接一个2层MLP投影头，将不同维度的编码器输出投影到相同维度的共享空间。为了保留模态内的判别信息，还添加了模态内对比损失作为辅助目标，确保投影后的表示不仅跨模态对齐，在模态内也保持区分度
    - 设计动机：直接在编码器空间对齐可能破坏预训练表示的质量，投影头提供了额外的适配空间

3. **功能描述增强策略（Function Description Augmentation）**:

    - 功能：解决蛋白质功能描述数据稀疏和质量不一的问题
    - 核心思路：利用LLM对Gene Ontology注释进行改写和扩展，生成多样化的功能描述。例如将简洁的GO注释"ATP binding"扩展为"This protein has ATP binding activity, meaning it can specifically interact with adenosine triphosphate molecules..."。这种增强既增加了训练数据量，又提供了功能文本的语义多样性
    - 设计动机：Gene Ontology的标注形式简短且格式化，作为自然语言训练数据太单调

### 损失函数 / 训练策略
总损失为三对对比损失加上三个模态内辅助损失的加权和。采用两阶段训练：第一阶段冻结编码器只训练投影头，第二阶段微调全部参数。使用UniProt数据库中约50万个有完整注释的蛋白质作为训练数据。

## 实验关键数据

### 主实验

| 任务 | 指标 | Align-Pro | ESM-2 | ProtST | OntoProtein | 提升 |
|------|------|-----------|-------|--------|-------------|------|
| GO功能预测 | Fmax | 0.694 | 0.651 | 0.672 | 0.658 | +2.2 |
| EC号预测 | Fmax | 0.882 | 0.856 | 0.871 | 0.862 | +1.1 |
| 跨模态检索 | R@10 | 78.3 | - | 72.1 | 69.5 | +6.2 |
| 折叠分类 | ACC(%) | 91.7 | 88.4 | 89.9 | 89.1 | +1.8 |

### 消融实验

| 配置 | GO预测(Fmax) | 跨模态检索(R@10) | 说明 |
|------|-------------|-----------------|------|
| Full model（三模态） | 0.694 | 78.3 | 完整框架 |
| 仅序列-结构 | 0.672 | 68.7 | 去掉功能模态 |
| 仅序列-功能 | 0.681 | 73.1 | 去掉结构模态 |
| 无功能增强 | 0.678 | 74.5 | 不用LLM扩展描述 |
| 无模态内损失 | 0.686 | 76.0 | 去掉辅助损失 |

### 关键发现
- 三模态对齐显著优于任何两模态组合，验证了序列-结构-功能三角关系的互补性
- 功能描述增强策略贡献明显（+1.6 Fmax），说明GP文本质量对跨模态学习很重要
- 在跨模态检索任务上优势最大（+6.2 R@10），说明统一嵌入空间对检索任务价值最高
- 结构模态对折叠分类帮助最大，功能模态对GO预测帮助最大，各模态各有所长

## 亮点与洞察
- 将CLIP式的对比学习成功推广到蛋白质三模态场景，提供了一个通用的蛋白质多模态嵌入空间。这种统一表示可以直接迁移到药物发现、蛋白质工程等下游应用
- LLM增强功能描述的策略巧妙地解决了生物数据库标注稀疏的问题，这种利用LLM弥补领域数据不足的思路值得在其他科学领域借鉴

## 局限与展望
- 训练数据局限于有完整三模态注释的蛋白质，这只占已知蛋白质的一小部分，半监督方法可能帮助利用只有部分注释的蛋白
- 对比学习的负样本选择策略会影响嵌入质量，当前简单的批内负采样可能不够高效
- 结构编码器依赖预测结构（AlphaFold），预测结构本身的误差会传播到嵌入空间
- 功能描述主要覆盖分子功能和生物过程，对蛋白质-蛋白质相互作用等更复杂的功能关系覆盖不足
- 未来可以将对齐框架扩展到更多模态，如蛋白质动力学模拟数据、实验功能数据等
- 跨物种的蛋白质功能预测是重要应用场景，多模态对齐可能帮助将研究充分的物种的知识迁移到研究较少的物种

## 相关工作与启发
- **vs ProtST**: ProtST只对齐序列和文本两个模态，本文增加了结构模态提供了更完整的信息
- **vs OntoProtein**: OntoProtein利用知识图谱增强蛋白质表示，但不进行显式的模态对齐
- **vs ESM-2**: ESM-2是强大的序列编码器但只处理单模态，本文通过多模态对齐进一步提升

## 评分
- 新颖性: ⭐⭐⭐⭐ 蛋白质三模态对齐框架有新意
- 实验充分度: ⭐⭐⭐⭐ 多个下游任务评估，消融完整
- 写作质量: ⭐⭐⭐⭐ 跨领域工作表述清晰
- 价值: ⭐⭐⭐⭐ 对计算生物学和NLP交叉领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PoinnCARE: Hyperbolic Multi-Modal Learning for Enzyme Classification](../../ICLR2026/computational_biology/poinncare_hyperbolic_multi-modal_learning_for_enzyme_classification.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/computational_biology/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](../../NeurIPS2025/computational_biology/learning_repetition-invariant_representations_for_polymer_informatics.md)
- [\[CVPR 2026\] Deciphering Genotype-Phenotype Mechanisms from High-Content Profiling via Knowledge-Guided Multi-modal Graph Learning](../../CVPR2026/computational_biology/deciphering_genotype-phenotype_mechanisms_from_high-content_profiling_via_knowle.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](../../NeurIPS2025/computational_biology/gflownets_for_learning_better_drug-drug_interaction_representations.md)

</div>

<!-- RELATED:END -->
