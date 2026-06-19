---
title: >-
  [论文解读] Unsupervised Feature Selection Through Group Discovery
description: >-
  [AAAI 2026][可解释性][无监督特征选择] 提出 GroupFS，首个端到端可微分的无监督特征选择框架，能同时发现潜在特征分组并选择信息量最大的组，无需预定义分组或标签监督。 1. 领域现状： 高维数据中，特征选择（FS）是降噪、提升泛化和可解释性的关键手段。无监督 FS 在无标签场景下尤为重要，但也更具挑战性…
tags:
  - "AAAI 2026"
  - "可解释性"
  - "无监督特征选择"
  - "特征分组"
  - "图拉普拉斯"
  - "Gumbel-Softmax"
  - "稀疏正则化"
---

# Unsupervised Feature Selection Through Group Discovery

**会议**: AAAI 2026  
**arXiv**: [2511.09166](https://arxiv.org/abs/2511.09166)  
**代码**: 无  
**领域**: 机器学习 / 特征选择 (Feature Selection)  
**关键词**: 无监督特征选择, 特征分组, 图拉普拉斯, Gumbel-Softmax, 稀疏正则化

## 一句话总结

提出 GroupFS，首个端到端可微分的无监督特征选择框架，能同时发现潜在特征分组并选择信息量最大的组，无需预定义分组或标签监督。

## 研究背景与动机

1. **领域现状**: 高维数据中，特征选择（FS）是降噪、提升泛化和可解释性的关键手段。无监督 FS 在无标签场景下尤为重要，但也更具挑战性，因为缺少标签指导选择过程。
2. **现有痛点**: 大多数无监督 FS 方法逐特征独立评估，忽略了特征之间的关系。现实中特征常"协同作用"：相邻像素、功能连接的脑区、相关的金融指标等。某些方法尝试捕获组结构，但依赖预定义分组或标签监督。
3. **核心矛盾**: 特征分组和特征选择是耦合问题——不知道组结构就无法做组级选择，而没有选择信号又难以发现有意义的组。
4. **本文目标**: 在完全无监督的条件下，同时发现特征的潜在分组结构，并选择最有信息量的特征组。
5. **切入角度**: 构建样本图和特征图两个图结构，利用 Gumbel-Softmax 做可微分分组，STG（随机门）做组级选择，拉普拉斯平滑约束保证组的语义一致性。
6. **核心 idea**: 通过在样本图和特征图上施加拉普拉斯平滑约束，并用组稀疏正则化，端到端地发现和选择特征组。

## 方法详解

### 整体框架

给定数据矩阵 $X \in \mathbb{R}^{N \times d}$，GroupFS 通过 Gumbel-Softmax 学习特征到 $C$ 个组的软分配矩阵 $M \in \mathbb{R}^{d \times C}$，同时用 STG 门控为每组学习重要性权重 $z_j$。总损失 $\mathcal{L} = \mathcal{L}_s + \lambda_1 \mathcal{L}_f + \lambda_2 \mathcal{L}_{reg}$ 由三部分组成：样本平滑、特征平滑、组稀疏。

### 关键设计

1. **样本级平滑损失 $\mathcal{L}_s$**:
    - **功能**: 选择在数据流形上变化平滑的特征，保留样本间的内在几何结构
    - **核心思路**: 首先通过 Gumbel-Softmax 得到分配矩阵 $M$，结合 STG 门控得到特征权重 $\hat{z}_i = \sum_{j=1}^C M_{ij} \cdot z_j$，对输入做掩码 $\widetilde{X} = X_B \odot \hat{Z}$。在掩码后的数据上构建随机游走矩阵 $P_{\widetilde{X}}$，最大化 $\mathcal{L}_s = -\frac{1}{Bd} \text{tr}(\widetilde{X}^\top P_{\widetilde{X}}^t \widetilde{X})$
    - **设计动机**: 特征值在样本流形上的平滑性是评价特征质量的自然准则（来自 Laplacian Score 的思想），此处将其扩展到组级选择

2. **特征级平滑损失 $\mathcal{L}_f$**:
    - **功能**: 确保相似特征获得相似的组分配，使学到的分组尊重特征空间的内在结构
    - **核心思路**: 将分配矩阵通过可训练投影 $F = MQ \in \mathbb{R}^{d \times C}$ 嵌入，在特征图的归一化拉普拉斯 $L_{\text{feat}}$ 上施加平滑约束 $\text{tr}(F^\top L_{\text{feat}} F)$，加正交正则化 $\|F^\top F - I\|_F^2$ 防止退化：$\mathcal{L}_f = \frac{1}{dC}[\text{tr}(F^\top L_{\text{feat}} F) + \beta \|F^\top F - I\|_F^2]$
    - **设计动机**: 如果两个特征在特征图上高度相似（高权重边连接），它们应该被分到同一组；正交约束确保不同组的嵌入多样且非冗余

3. **组稀疏正则化 $\mathcal{L}_{reg}$**:
    - **功能**: 鼓励只保留少量信息量大的特征组，产生紧凑可解释的模型
    - **核心思路**: $\mathcal{L}_{reg} = \frac{1}{C} \sum_{j=1}^C \mathbb{P}(z_j > 0) \cdot \frac{1}{d} \sum_{i=1}^d M_{ij}$，同时惩罚组的激活概率和组的大小
    - **设计动机**: 激活概率高且包含特征多的组惩罚大，驱动模型保持少量小型活跃组

### 损失函数 / 训练策略

- 总损失: $\mathcal{L} = \mathcal{L}_s + \lambda_1 \mathcal{L}_f + \lambda_2 \mathcal{L}_{reg}$
- Gumbel-Softmax 温度从 10 退火到 $10^{-2}$（线性衰减），使分配逐渐趋向 one-hot
- 谱聚类初始化分配矩阵的 logits（$p_{\text{main}} = 0.7$）
- F 的列每步归零均值、归一化 $\ell_2$ 范数，防止收敛到平凡解
- STG 门初始化 $\mu_j = 0.5$，噪声 $\sigma = 0.5$
- 自适应核 (self-tuning kernel, K=7 近邻)，扩散时间 $t=2$

## 实验关键数据

### 主实验

| 数据集 | GroupFS | 最佳Baseline | 提升 | 特征数 |
|--------|---------|-------------|------|--------|
| Lung500 | **93.0±6.8** | 91.3±6.7 (CAE) | +1.7% | 234 |
| HeartDisease | **83.1±0.5** | 82.6±0.4 (MCFS) | +0.5% | 10 |
| AR10P | **32.5±4.1** | 24.9±2.9 (MGAGR) | +7.6% | 362 |
| PIE10P | **38.4±2.5** | 35.0±3.6 (DUFS) | +3.4% | 49 |
| NMNIST 3-8 | **83.3±0.1** | 77.3±1.0 (LS) | +6.0% | 51 |
| ALLAML | **70.6±1.4** | 70.6±1.4 (LS) | 持平 | 274 |

固定预算场景下 9 个数据集中 6 个排名第一或并列第一，平均超越次优方法 +3.84%。

### 消融实验

| 配置 | RGsim | TPR | FDR | 说明 |
|------|-------|-----|-----|------|
| C ≤ 2+(d-10) | ~1.0 | ~1.0 | ~0 | 完美恢复信息组和噪声分离 |
| C = 2 | 低 | 高 | 高 | 组数太少，噪声混入 |
| C > 2+(d-10) | 下降 | 高 | 低 | 信息组被过度拆分 |

### 关键发现

- 自适应预算场景下也在 6/9 数据集上取得最佳准确率
- NMNIST 3-8 上发现 7 个空间连贯的像素组，最重要的组对应区分数字 3 和 8 的关键区域
- Student Performance 数据集：最重要组包含饮酒相关特征，第二组包含动机相关（缺勤、过去失败等），语义高度连贯
- 相关强度 $\rho$ 越高，损失越低（组内一致性越强越好）
- 对中等水平的加性噪声具有鲁棒性

## 亮点与洞察

- 首次将特征分组发现与组级选择统一到一个端到端可微框架中，无需人工预定义分组
- 双图设计（样本图 + 特征图）的组合非常自然，各自捕获不同维度的结构信息
- 可解释性实验令人信服：在 NMNIST 上发现的像素组空间连贯且与判别区域对齐
- Gumbel-Softmax + STG 的组合巧妙地将离散分组和门控选择同时嵌入到可微优化中

## 局限与展望

- 依赖欧几里得距离构建图，对弯曲的非欧流形可能表示不准确
- 学习单一的全局组重要性，忽略了条件相关或时间依赖的场景
- 组数 $C$ 的选择依赖启发式方法，敏感性较高
- 计算复杂度 $O(N^2 d)$（全批次）和 $O(tN^3)$（多步扩散），大规模数据可能有瓶颈
- 未与深度特征选择方法（如基于 VAE 或 contrastive learning 的方法）对比

## 相关工作与启发

- **vs Laplacian Score (LS)**: LS 逐特征独立打分，GroupFS 利用组结构协同选择，在多数数据集上显著优于 LS
- **vs DUFS**: DUFS 也用随机门控和样本图，但逐特征操作不考虑组结构；GroupFS 将门控提升到组级别
- **vs CompFS**: CompFS 依赖标签监督来学习组，GroupFS 完全无监督；且 CompFS 发现的组语义连贯性较差
- **vs MGAGR**: MGAGR 使用预定义分组，运行时间极长（大数据集 >10000 CPU 小时），GroupFS 更灵活高效

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次端到端无监督联合分组+选择，框架设计优雅
- 实验充分度: ⭐⭐⭐⭐ 9 个数据集+合成数据+可解释性实验，两种评估场景，但缺少运行时间对比
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，实验设置详尽，附录完备
- 价值: ⭐⭐⭐⭐ 对高维无标签数据的特征选择有实际价值，可解释性是一大优势

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](../../NeurIPS2025/interpretability/tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)
- [\[CVPR 2026\] H-Sets: Hessian-Guided Discovery of Set-Level Feature Interactions in Image Classifiers](../../CVPR2026/interpretability/h-sets_hessian-guided_discovery_of_set-level_feature_interactions_in_image_class.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[ICML 2025\] Concept-Based Unsupervised Domain Adaptation](../../ICML2025/interpretability/concept-based_unsupervised_domain_adaptation.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)

</div>

<!-- RELATED:END -->
