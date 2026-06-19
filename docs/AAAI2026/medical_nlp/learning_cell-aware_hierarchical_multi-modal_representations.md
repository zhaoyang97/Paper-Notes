---
title: >-
  [论文解读] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling
description: >-
  [AAAI 2026 Oral][医疗NLP][分子性质预测] 本文提出 CHMR 框架，通过结构感知传播解决生物模态缺失问题，引入树状向量量化(Tree-VQ)建模分子-细胞-基因间的层次依赖关系，在9个基准728个任务上分类提升3.6%、回归提升17.2%，实现鲁棒的细胞感知分子表征学习。 领域现状：分子性质预测（活性、…
tags:
  - "AAAI 2026 Oral"
  - "医疗NLP"
  - "分子性质预测"
  - "细胞感知表征"
  - "层次化向量量化"
  - "模态缺失"
  - "多模态融合"
---

# Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.21120](https://arxiv.org/abs/2511.21120)  
**代码**: [https://github.com/limengran98/CHMR](https://github.com/limengran98/CHMR)  
**领域**: 计算生物  
**关键词**: 分子性质预测, 细胞感知表征, 层次化向量量化, 模态缺失, 多模态融合

## 一句话总结

本文提出 CHMR 框架，通过结构感知传播解决生物模态缺失问题，引入树状向量量化(Tree-VQ)建模分子-细胞-基因间的层次依赖关系，在9个基准728个任务上分类提升3.6%、回归提升17.2%，实现鲁棒的细胞感知分子表征学习。

## 研究背景与动机

**领域现状**：分子性质预测（活性、毒性、副作用等）是药物发现的核心任务。近年来深度学习方法发展迅速，主要分两类：(1) 单模态方法——利用分子内部特征（原子属性、化学键、3D结构）进行预测；(2) 多模态方法——将分子结构与外部生物响应（细胞形态、基因表达）结合，以捕捉分子在细胞层面引发的信号级联效应。

**现有痛点**：(1) **模态不完整性普遍存在**——分子结构数据通常完整，但关联的细胞表型或基因表达数据由于实验限制和成本约束经常缺失（超过90%的分子缺少部分生物模态），且不同分子缺失模式不同（有的缺形态数据，有的缺转录组数据）；(2) **层次依赖建模不足**——分子扰动会引发跨生物层级的级联反应（化学结构→细胞过程→基因表达），但现有方法在扁平隐空间中做实例级对齐，无法捕捉多跳语义关系和跨层依赖。

**核心矛盾**：生物模态缺失导致分布偏移和模态不平衡，简单填充（零向量、均值）效果差；同时扁平对齐丢失了层级结构信息，限制了跨尺度生物机制的建模能力。

**本文目标** (1) 如何在模态高度缺失的情况下鲁棒地学习分子表征？(2) 如何显式建模分子、细胞、基因三个层级之间的层次依赖？

**切入角度**：作者观察到分子之间存在结构相似性，相似分子的生物响应也相似，可以通过图传播来增强缺失模态；同时生物响应天然具有从分子→细胞→基因的层次结构，可以用树状结构显式编码。

**核心 idea**：用结构感知图传播填补缺失生物模态，用树状向量量化编码跨尺度层次语义，实现面向缺失模态的鲁棒分子表征学习。

## 方法详解

### 整体框架

CHMR 包含四个核心模块：(1) **模态增强(MA)**——基于分子结构相似性图的迭代传播来填补缺失的生物模态；(2) **语义一致性对齐(SCA)**——在样本级和分布级对齐分子与细胞模态的表征；(3) **树状向量量化(Tree-VQ)**——用共享的二叉树捕捉跨模态的层次语义依赖；(4) **上下文传播重建(CPR)**——利用生物先验知识图上的随机游走进行跨模态上下文监督。四个模块联合优化。

### 关键设计

1. **模态增强与语义一致性对齐(MA + SCA)**:

    - 功能：为缺失的外部生物模态生成合理的伪特征，并确保增强后的特征与原始特征在语义上一致
    - 核心思路：**MA 部分**：构建分子间相似性矩阵 $\mathbf{W}$，每个分子保留 top-K 近邻，对缺失模态进行迭代传播 $\mathbf{x}_i^{c,(T)} = \sum_{j \in \mathcal{N}_K(v_i)} \mathbf{W}_{ij} \mathbf{x}_j^{c,(T-1)}$（观测到的模态保持不变）。**SCA 部分**：分两层对齐——样本级用 InfoNCE 对比损失 $\mathcal{L}_{IA}$ 将分子锚点特征与细胞特征对齐；分布级用 VICReg 损失 $\mathcal{L}_{DA}$ 确保增强后的特征不偏离原始分布
    - 设计动机：简单的零填充或随机填充会导致 5.3% 和 4.5% 的性能下降，而基于近邻传播利用了"结构相似→生物响应相似"的先验。VICReg 的方差和协方差正则化可以有效防止增强特征的分布偏移

2. **树状向量量化(Tree-VQ)**:

    - 功能：将多模态特征映射到一棵共享的二叉树上，用树的层级编码生物层次（浅层对应分子指纹，深层对应细胞表型和基因表达）
    - 核心思路：构建深度 $H$ 的二叉树 $\mathcal{T} = \bigcup_{h=1}^{H} \mathcal{E}^h$，每层 $2^h$ 个节点 embedding。对每个模态的投影特征 $\mathbf{p}^{\xi}$，从根节点开始逐层路由，每层通过余弦距离选择最近的子节点 $j^{*\xi,h} = \arg\min_j \tilde{\delta}_j^{\xi,h}$。路由受父节点约束（只能选父节点的两个子节点）。用对称 VQ 损失 $\mathcal{A}(\mathbf{p}^\xi, \mathbf{q}^{\xi,h}) = 1 - \cos(sg[\mathbf{q}^{\xi,h}], \mathbf{p}^\xi) + \eta(1 - \cos(\mathbf{q}^{\xi,h}, sg[\mathbf{p}^\xi]))$ 进行双向对齐
    - 设计动机：与扁平 VQ 相比，树结构天然对应生物层次（分子→细胞→基因），不同模态共享同一棵树使得异质特征被联合路由到统一的语义层次中。消融显示去掉 Tree-VQ 掉 3.9%，换成扁平 VQ 也掉 2.0%

3. **上下文传播重建(CPR)**:

    - 功能：利用生物先验知识图上的随机游走提供跨模态上下文监督信号
    - 核心思路：利用已知的分子-生物响应关联（如药物-靶点对、功能关联、共享调控通路）构建上下文图 $\mathcal{H}$，从每个节点出发随机游走长度 $L$，沿路径累积传播权重。通过解码器重建分子和生物模态特征，重建损失按随机游走权重加权 $\mathcal{L}_{CPR} = -\frac{1}{|\mathcal{V}|} \sum_{i} \sum_{l=0}^{L} \beta_{i,l} \mathcal{D}(\hat{\mathbf{x}}_{u_{i_l}}, \mathbf{x}_{u_{i_l}})$
    - 设计动机：在缺乏显式监督的情况下，生物先验知识图提供了额外的结构化监督信号，随机游走使得远程关联也能被利用

### 损失函数 / 训练策略

- 总体预训练损失: $\mathcal{L}_{total} = \mathcal{L}_{CPR} + \lambda_1 \mathcal{L}_{SCA} + \lambda_2 \mathcal{L}_{TreeVQ}$
- 最优超参数: $\lambda_1 = 10$, $\lambda_2 \in \{0.1, 1\}$, $\eta = 1$, 树深度 $h = 6$
- 下游评估时冻结预训练 backbone，仅训练轻量级预测头

## 实验关键数据

### 主实验

| 数据集 | 指标 | CHMR(Ours) | InfoAlign(SOTA) | 提升 |
|--------|------|-----------|-----------------|------|
| ChEMBL (41 tasks) | AUC% ↑ | **84.7±0.2** | 81.3±0.6 | +3.4 |
| ToxCast (617 tasks) | AUC% ↑ | **69.3±0.3** | 66.4±1.1 | +2.9 |
| Broad (32 tasks) | AUC% ↑ | **71.4±0.2** | 70.0±0.1 | +1.4 |
| Biogen (6 tasks) | MAE×100 ↓ | **40.9±0.3** | 49.4±0.2 | -17.2% |

Biogen 各子指标: HLM 33.7(vs 39.7), RLM 39.8(vs 48.4), ER 35.2(vs 39.2), 溶解度 34.9(vs 40.5), hPPB 53.1(vs 66.7), rPPB 48.5(vs 62.0)

### 消融实验

| 配置 | ChEMBL ↑ | Biogen ↓ | Δ(%) |
|------|----------|----------|------|
| Full Model | 84.7 | 40.9 | - |
| Zero imputation (w/o MA) | 81.6 | 44.8 | -5.3 |
| w/o SCA | 82.4 | 43.1 | -3.6 |
| w/o Tree-VQ | 82.3 | 43.4 | -3.9 |
| Flat VQ (替代 Tree-VQ) | 83.2 | 42.0 | -2.0 |
| w/o CPR | 82.6 | 43.0 | -3.5 |
| Mol-Only (仅分子模态) | 81.5 | 44.3 | -4.9 |
| InfoAlign (SOTA baseline) | 81.3 | 49.4 | -7.7 |

### 关键发现

- **Tree-VQ 是关键创新**：去掉后下降 3.9%，换成扁平 VQ 仍下降 2.0%，说明层次结构而非简单量化在起作用
- **四个模块协同增效**：每个模块单独去除都导致 2-5% 的性能下降，full model 一致性地优于所有消融变体
- **多模态融合必要性**：Mol-Only 掉 4.9%，加入任意一种生物模态都能部分恢复，完整三模态效果最佳
- **Biogen 回归任务提升最大**（MAE 降低 17.2%），说明细胞和基因信息对于 ADME 性质预测特别有价值
- 树深度 $h=6$ 最优，过浅表达力不足，过深容易过拟合

## 亮点与洞察

- **树状向量量化用于生物层次建模**是一个很精巧的设计：VQ 本身是离散化表示学习的成熟技术，但将其扩展为共享的多模态树结构，并将树的层级对应到生物层次（分子→细胞→基因），是一个新颖的形式化。t-SNE 可视化清楚显示 CHMR 同时实现了跨模态对齐和层次结构组织
- **结构感知传播**的增强策略巧妙利用了化学信息学的先验（结构相似的分子会引发相似的生物响应），比简单的 KNN 或均值填充更有生物学合理性
- 药物性质预测案例研究展示了框架的可解释性：不同模态提供互补的预测线索（1D指纹→代谢清除率，3D构象→蛋白结合亲和力，生物上下文→P-gp药物外排）

## 局限与展望

- 预训练数据涉及多个公开数据源的整合，数据质量和一致性可能受到影响
- Tree-VQ 的树深度和结构是预定义的固定二叉树，缺乏自适应的动态树生长机制
- 模态增强依赖分子间结构相似性矩阵的质量，对于结构新颖的分子（无相似邻居）可能效果有限
- 缺少蛋白质靶点信息的整合，限制了在靶点相关性质预测中的应用

## 相关工作与启发

- **vs InfoAlign**: InfoAlign 也整合了分子、细胞形态和转录组，但采用扁平对齐且不处理层次依赖，CHMR 在所有基准上一致超越，Biogen 上领先 17.2%
- **vs CLOOME**: CLOOME 使用 InfoLOOB 做结构-表型对对比学习，但仅处理两个模态且无缺失模态机制，限制了泛化性
- **vs GraphMVP**: GraphMVP 实现了 2D-3D 语义迁移但局限于分子内部模态，不涉及外部生物响应

## 评分

- 新颖性: ⭐⭐⭐⭐ Tree-VQ 的层次化多模态量化是真正的技术创新，但整体框架（对齐+重建+量化）的组合较重
- 实验充分度: ⭐⭐⭐⭐⭐ 728个任务、9个数据集、20+基线、详尽消融和超参分析，极为充分
- 写作质量: ⭐⭐⭐⭐ 方法部分数学化程度高，符号系统清晰，但整体篇幅偏长
- 价值: ⭐⭐⭐⭐ 对药物发现中的分子表征学习有明确的实践意义，特别是在处理模态缺失场景下

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](../../ACL2026/medical_nlp/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[AAAI 2026\] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)
- [\[AAAI 2026\] LUCID: Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems](lucid_learning-enabled_uncertainty-aware_certification_of_stochastic_dynamical_s.md)
- [\[ACL 2026\] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](../../ACL2026/medical_nlp/learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](../../ACL2026/medical_nlp/promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)

</div>

<!-- RELATED:END -->
