---
title: >-
  [论文解读] A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random
description: >-
  [NeurIPS 2025][计算生物][模型聚类] 在高斯混合模型的聚类框架中，统一解决变量选择（区分信号变量、冗余变量和噪声变量）与MNAR缺失数据建模，通过两阶段策略（LASSO惩罚排序加BIC角色分配）和谱距离自适应惩罚权重实现高维场景下的高效推理，并证明了可辨识性和渐近选择一致性。 领域现状：基于模型的聚类（mod…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "模型聚类"
  - "变量选择"
  - "MNAR缺失"
  - "高斯混合模型"
  - "LASSO惩罚"
  - "BIC一致性"
---

# A Unified Framework for Variable Selection in Model-Based Clustering with Missing Not at Random

**会议**: NeurIPS 2025  
**arXiv**: [2505.19093](https://arxiv.org/abs/2505.19093)  
**代码**: 无  
**领域**: 计算生物  
**关键词**: 模型聚类, 变量选择, MNAR缺失, 高斯混合模型, LASSO惩罚, BIC一致性

## 一句话总结
在高斯混合模型的聚类框架中，统一解决变量选择（区分信号变量、冗余变量和噪声变量）与MNAR缺失数据建模，通过两阶段策略（LASSO惩罚排序加BIC角色分配）和谱距离自适应惩罚权重实现高维场景下的高效推理，并证明了可辨识性和渐近选择一致性。

## 研究背景与动机

**领域现状**：基于模型的聚类（model-based clustering）通过假设数据来自有限混合模型来进行概率聚类，高斯混合模型（GMM）是经典实例。在高维数据中，许多变量可能与聚类结构无关或冗余，需要变量选择来提升聚类性能和可解释性。Maugis等人提出的SRUW模型将变量分为四种角色：聚类相关（S）、可由S线性解释的冗余（U）、与S有线性关系的回归集（R）、独立噪声（W）。

**现有痛点**：(1) 现有变量选择方法（如Clustvarsel、Selvar）都假设数据完整或仅处理MAR缺失，但在转录组学等实际场景中，缺失机制往往是MNAR（缺失与未观测值相关），直接忽略导致聚类和变量选择双重失误。(2) 之前的工作要么只处理变量选择不处理缺失，要么只处理缺失不做变量选择——没有统一框架。(3) Celeux等人的两阶段排序方法虽然高效，但缺乏理论保证。

**核心矛盾**：MNAR缺失破坏了标准EM的ignorability假设，而变量选择又依赖准确的参数估计——两个问题相互纠缠，必须联合解决。

**本文目标** 在MNAR缺失数据存在的情况下，同时完成变量角色识别（S/R/U/W）和聚类参数估计，并给出理论保证。

**切入角度**：利用MNARz机制（缺失概率仅依赖于潜在类别成员身份而非观测值本身），可以将MNAR问题重新转化为在augmented数据（原始数据+缺失指示矩阵）上的MAR问题，从而保留EM算法的可行性。

**核心 idea**：将SRUW变量选择框架与MNARz缺失机制联合建模，通过谱距离自适应惩罚的两阶段策略实现高效推理，并证明渐近选择一致性。

## 方法详解

### 整体框架
输入是含缺失值的数据矩阵 $\mathbf{Y} \in \mathbb{R}^{N \times D}$ 和缺失掩码 $\mathbf{C}$。流程分两个阶段：Stage A（排序）在快速填补后的数据上用LASSO惩罚GMM对变量排序；Stage B（角色分配）在原始不完整数据上沿排名顺序逐步用BIC判断每个变量的角色（S/U/W），同时使用MNARz机制的EM算法估计参数。输出是变量分区 $(\hat{\mathbb{S}}, \hat{\mathbb{R}}, \hat{\mathbb{U}}, \hat{\mathbb{W}})$、聚类数 $\hat{K}$ 和参数估计。

### 关键设计

1. **SRUW-MNARz联合模型**:

    - 功能：统一建模变量角色和缺失机制
    - 核心思路：完整数据密度分解为三部分的乘积——聚类分量 $f_{\text{clust}}(\mathbf{y}^{\mathbb{S}}|\alpha_k)$、回归分量 $f_{\text{reg}}(\mathbf{y}^{\mathbb{U}}|\mathbf{y}^{\mathbb{R}})$ 和独立分量 $f_{\text{indep}}(\mathbf{y}^{\mathbb{W}})$，再乘以MNARz缺失分量 $f_{\text{MNARz}}(\mathbf{c}_n|z_{nk}=1;\psi_k)$。MNARz假设缺失概率仅依赖类别标签 $k$，因此缺失项可以从积分中提出来，保持ignorability
    - 设计动机：MNARz机制的关键优势是将MNAR转化为augmented data上的MAR——在$(Y, C)$上做推断等价于MAR。这避免了一般MNAR的不可识别性问题，同时允许缺失模式本身携带聚类信息

2. **谱距离自适应惩罚权重**:

    - 功能：替代传统的逆偏相关系数来构建惩罚矩阵 $\mathbf{P}_k$
    - 核心思路：从初始精度矩阵 $\hat{\Psi}_k^{(0)}$ 构建无向图 $\mathcal{G}^{(k)}$，计算对称归一化Laplacian $\mathbf{L}_{sym}^{(k)}$，目标是将 $\Psi_k$ 收缩至对角矩阵（空图），谱距离 $D_{\text{LS}} = \|\text{spec}(\mathbf{L}_{sym}^k)\|_2$，自适应权重 $\mathbf{P}_{k,ij} = (D_{\text{LS}} + \epsilon)^{-1}$
    - 设计动机：谱距离提供了图结构复杂度的全局度量，比逐元素的偏相关系数更稳定，尤其在高维小样本且有缺失的情况下

3. **两阶段选择策略**:

    - 功能：高效完成变量排序和角色分配
    - 核心思路：Stage A对快速填补后的数据沿 $(\lambda, \rho)$ 正则化路径拟合惩罚GMM，根据每个变量的均值被压为零的频率计算排序分数 $\mathcal{O}_K(d)$。Stage B沿排名列表单次遍历，每加入一个新变量就在原始不完整数据上拟合无惩罚的SRUW-MNARz并用BIC判断角色
    - 设计动机：Stage A用快速填补加惩罚模型实现低成本排序（避免组合爆炸），Stage B在原始数据上做精确推断（避免填补误差传播）。两阶段解耦使得计算复杂度从指数降为线性

### 损失函数 / 训练策略
Stage A使用惩罚对数似然（含ℓ1均值惩罚和adaptive glasso精度矩阵惩罚），沿数据驱动的几何路径搜索 $(\lambda, \rho)$。Stage B使用标准EM算法最大化SRUW-MNARz的BIC准则。初始化采用K-means++/层次聚类+软E-step估计缺失率。

## 实验关键数据

### 主实验（合成数据，MNAR 50%缺失率）

| 模型 | 平均ARI | 标准差 | vs SelvarMNARz | p值 |
|------|---------|--------|---------------|-----|
| **SelvarMNARz** | **0.511** | 0.052 | — | — |
| Clustvarsel | 0.363 | 0.088 | 显著差于 | <0.001 |
| Selvar | 0.348 | 0.108 | 显著差于 | <0.001 |
| VarSelLCM | 0.344 | 0.101 | 显著差于 | <0.001 |

### 消融实验

| 缺失率 | 缺失类型 | SelvarMNARz ARI | 基线最佳ARI | 说明 |
|--------|---------|-----------------|-------------|------|
| 5% | MAR | ~0.85 | ~0.82 | 低缺失率差异小 |
| 20% | MAR | ~0.75 | ~0.60 | 基线开始退化 |
| 50% | MNAR | 0.511 | 0.363 | 基线严重退化，本文稳健 |
| 50% | MAR | ~0.55 | ~0.40 | MNAR更具挑战但本文仍最优 |

### 关键发现
- **SelvarMNARz在所有缺失率和缺失机制下都最优**，且随缺失率增加性能下降最温和——50% MNAR下ARI仍有0.51，而基线方法降到0.34-0.36
- **变量选择准确性**：SelvarMNARz在所有配置下正确恢复聚类变量集和聚类数 $K$，而填补后聚类的流水线方法在缺失率大于20%时开始误判变量角色
- **转录组学实验**：在1267基因的拟南芥数据上，SelvarMNARz识别出18个聚类，将P6/P7从之前分析的"信号变量"重分类为"冗余变量"（可被P1-P4解释），揭示了正确处理缺失机制如何改变生物学结论

## 亮点与洞察
- **统一框架的必要性论证**令人信服：不是简单地把变量选择和缺失处理"拼起来"，而是理论上证明了两者必须联合解决——分开处理会导致选择不一致
- **MNARz到MAR的转化**是关键理论贡献：通过将缺失指示矩阵纳入augmented data，将MNAR的不可处理性转化为标准MAR推断。这个trick对任何涉及潜变量+MNAR的问题都有参考价值
- **选择一致性定理（Theorem 3）**的证明策略（RSC条件到排序一致到BIC一致）为惩罚似然方法在混合模型中的理论分析提供了新工具

## 局限与展望
- 目前仅限于连续数据的高斯混合模型，无法处理分类或混合类型变量——作者承认扩展到离散数据（如latent class模型）是自然的下一步
- MNARz假设（缺失仅依赖类别）虽然比一般MNAR容易处理，但在某些实际场景中可能过强——例如基因表达值极低时更可能缺失，这属于一般MNAR
- 两阶段策略中Stage A的快速填补可能在高缺失率下引入偏差，虽然Stage B在原始数据上修正，但排序的初始偏差可能导致次优结果
- 计算复杂度分析不够详细，在真正的大规模高维场景（D > 10000）下的可扩展性不清楚

## 相关工作与启发
- **vs Maugis et al. (SRUW)**: 提出了SRUW变量分区框架但只处理MAR/完整数据，本文将其扩展到MNARz并补充了理论保证
- **vs Celeux et al. (Selvar)**: 提出了两阶段排序策略但无理论保证，本文证明了该策略在MNARz下的选择一致性
- **vs Sportisse et al. (MNARz)**: 提出了MNARz机制的可辨识性分析，本文在其基础上与SRUW变量选择联合建模

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一框架的构建和理论保证有扎实贡献，但各组成部分（SRUW、MNARz、两阶段策略）都已有前人工作
- 实验充分度: ⭐⭐⭐⭐ 合成数据系统性强（多缺失率多机制加统计检验），真实转录组数据有生物学解读
- 写作质量: ⭐⭐⭐ 数学符号体系较重，理论定理的非正式版本有助理解但技术细节需查补充材料
- 价值: ⭐⭐⭐⭐ 对组学等高维缺失数据聚类场景有直接实用价值，理论保证增强了方法的可信度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Autoencoding Random Forests](autoencoding_random_forests.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICML 2025\] Multivariate Conformal Selection](../../ICML2025/computational_biology/multivariate_conformal_selection.md)

</div>

<!-- RELATED:END -->
