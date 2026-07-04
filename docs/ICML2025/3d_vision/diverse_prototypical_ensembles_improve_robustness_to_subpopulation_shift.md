---
title: >-
  [论文解读] Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift
description: >-
  [ICML2025][3D视觉][subpopulation shift] 提出 Diversified Prototypical Ensemble (DPE)，用多个多样化的原型分类器替换标准线性分类头，通过显式（inter-prototype similarity loss）和隐式（bootstrap 采样）两种多样化策略，在不需要子群标注的情况下自适应发现子群决策边界，显著提升 worst-group accuracy。
tags:
  - "ICML2025"
  - "3D视觉"
  - "subpopulation shift"
  - "prototype classifier"
  - "ensemble diversity"
  - "worst-group accuracy"
  - "distribution robustness"
---

# Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift

**会议**: ICML2025  
**arXiv**: [2505.23027](https://arxiv.org/abs/2505.23027)  
**代码**: [minhto2802/dpe4subpop](https://github.com/minhto2802/dpe4subpop)  
**领域**: others (鲁棒性 / 分布偏移)  
**关键词**: subpopulation shift, prototype classifier, ensemble diversity, worst-group accuracy, distribution robustness

## 一句话总结

提出 Diversified Prototypical Ensemble (DPE)，用多个多样化的原型分类器替换标准线性分类头，通过显式（inter-prototype similarity loss）和隐式（bootstrap 采样）两种多样化策略，在不需要子群标注的情况下自适应发现子群决策边界，显著提升 worst-group accuracy。

## 研究背景与动机

### 问题定义

Subpopulation shift 指训练集和测试集在子群分布上存在差异，是分布偏移的常见形式。Yang et al. (2023) 将其分为四类：

**虚假相关 (Spurious Correlations)**：非因果特征误导预测（如背景水域→水鸟）

**属性不平衡 (Attribute Imbalance)**：某些属性值出现频率远高于其他

**类别不平衡 (Class Imbalance)**：部分标签严重欠表示

**属性泛化 (Attribute Generalization)**：测试时出现训练中未见的属性值

### 现有方法的不足

- ERM 训练的分类器倾向于学习多数子群的特征，在少数子群上表现差
- gDRO、JTT 等方法依赖**子群标注**，而真实数据中往往无法获得
- 显式识别少数群体的方法增加了复杂度，且难以泛化到未见子群
- 简单重采样/重加权（如 CRT、DFR）虽有效但仍受限于已知子群结构

### 核心动机

单一分类器只能学到一条决策边界，容易被多数子群主导。如果使用**集成**并显式鼓励成员之间的**多样性**，不同成员可以捕捉到不同子群的决策边界——即使没有子群标签也能做到。原型分类器（prototypical classifier）天然保持特征空间的几何结构，适合在有限数据下发现子群。

## 方法详解

### 整体流程：两阶段训练

**阶段一**：用标准 ERM 在全量训练集上训练特征提取器 $f: \mathbb{R}^n \to \mathbb{R}^d$，然后冻结 $f$。
沿用 Kirichenko et al. (2022) 和 Izmailov et al. (2022) 的发现：即使存在虚假相关，ERM 学到的特征表示仍然包含核心判别信息。

**阶段二**：用验证集的类别均衡子集训练 DPE 分类头（替换原线性层），**不需要子群标注**。

### 原型分类器

给定 $K$ 个类别，每个类别定义一个可学习原型 $p^{(i)} \in \mathbb{R}^d$。分类概率基于特征到各类原型的距离：

$$P(y|X) = \frac{\exp(-D(f(X), p^{(y)}))}{\sum_{i=1}^{K} \exp(-D(f(X), p^{(i)}))}$$

其中距离函数 $D$ 为缩放欧式距离（对归一化向量）：

$$D(x, y) = |d_s| \cdot \left\| \frac{x}{\|x\|} - \frac{y}{\|y\|} \right\|_2$$

$d_s$ 为可学习缩放因子。损失函数引入温度 $\tau$：

$$\mathcal{L}(X, y) = -\log \frac{\exp(-D(f_\theta(X), p^{(y)}) / \tau)}{\sum_{i=1}^{K} \exp(-D(f_\theta(X), p^{(i)}) / \tau)}$$

### 原型集成

每个类别使用 $N$ 个原型，得到集合 $\{p_j^{(i)}\}_{i=1,...,K,\; j=1,...,N}$。最终预测为 $N$ 个成员的平均概率：

$$\hat{y} = \arg\max_{k \in \{1,...,K\}} \frac{1}{N} \sum_{j=1}^{N} P_j^{(k)}(y|X)$$

### 多样化策略

#### 1. 显式多样化：Inter-Prototype Similarity (IPS) Loss

对第 $n$ 个集成成员，IPS 损失惩罚同类原型之间的相似性：

$$\mathcal{L}_{\text{IPS}} = \sum_{k=1}^{K} \sum_{i=1}^{n} \sum_{j=1}^{n} \mathbb{1}_{\{i \neq j\}} \frac{|\langle p_i^{(k)}, p_j^{(k)} \rangle|}{n \cdot d}$$

- 按 $n$（当前成员数）和 $d$（嵌入维度）缩放
- 训练第 $n$ 个成员时，冻结前 $n-1$ 个成员的原型，只优化当前成员的 $\{p_n^{(k)}\}_{k=1,...,K}$
- 总损失 = $\mathcal{L}(X, y) + \mathcal{L}_{\text{IPS}}$

#### 2. 隐式多样化：Bootstrap Aggregation

每个集成成员在验证集的不同类别均衡子集上训练，随机子集的差异隐式鼓励不同成员学到不同的决策边界。

#### 训练方式

集成成员**顺序训练**（非并行），每个新成员通过 IPS loss 与已冻结的前序成员保持多样性。

## 实验关键数据

### 数据集

9 个真实数据集，覆盖视觉和 NLP，涵盖四种 subpopulation shift 类型：

| 数据集 | 领域 | 偏移类型 |
|--------|------|----------|
| Waterbirds | 视觉 | 虚假相关 |
| CelebA | 视觉 | 虚假相关 |
| MetaShift | 视觉 | 虚假相关 |
| ImageNetBG | 视觉 | 虚假相关 |
| NICO++ | 视觉 | 属性泛化 |
| Living17 | 视觉 | 属性泛化 |
| CheXpert | 医学影像 | 属性不平衡 |
| CivilComments | NLP | 属性不平衡 |
| MultiNLI | NLP | 虚假相关 |

### 主要结果：Worst-Group Accuracy (WGA)

**无子群标注条件下（ERM backbone）：**

| 方法 | Waterbirds | CelebA | CivilComments | MultiNLI | MetaShift | ImageNetBG | NICO++ | Living17 |
|------|-----------|--------|---------------|----------|-----------|------------|--------|----------|
| ERM | 69.1 | 57.6 | 63.2 | 66.4 | 82.1 | 76.8 | 35.0 | 48.0 |
| CRT | 76.3 | 69.6 | 67.8 | 65.4 | 83.1 | 78.2 | 33.3 | - |
| DFR | 89.0 | 73.7 | 64.4 | 63.8 | 81.4 | 74.4 | 38.0 | - |
| **ERM+DPE** | **91.0** | **81.9** | **69.9** | **69.3** | 84.1 | **87.9** | **50.0** | **54.0** |

关键观察：
- DPE 在 9 个数据集中的 **8 个**取得最优 WGA
- 在 Waterbirds 上比 DFR 提升 2.0%，在 CelebA 上提升 8.2%
- 在 ImageNetBG 上提升巨大：87.9% vs. CRT 的 78.2%（+9.7%）
- NICO++（属性泛化）：50.0% vs. DFR 的 38.0%（+12.0%），提升最为显著
- 在 Living17 上达到 54.0%，而 CRT/DFR 甚至无法报告结果

**更强 ERM* backbone + 其他方法对比：**

| 方法 | Waterbirds | CelebA | CivilComments | MultiNLI |
|------|-----------|--------|---------------|----------|
| ERM* | 77.9 | 66.5 | 69.4 | 66.5 |
| RWY | 86.1 | 82.9 | 67.5 | 68.0 |
| AFR | 90.4 | 82.0 | 68.7 | - |

DPE 使用标准 ERM backbone 就已超越或持平这些使用更强 backbone 的方法。

## 亮点与洞察

1. **不需要子群标注**：绝大多数竞争方法需要显式子群标签或已知子群数量，DPE 完全自动发现子群结构
2. **即插即用**：只需替换最后的线性分类层，冻结特征提取器，训练代价极低
3. **可视化验证**：在 Waterbirds 数据集上，不同原型确实捕获了不同语义子群（如"陆地上的鸟"vs."水中的鸟"），验证了多样化策略的有效性
4. **IPS loss 设计简洁**：通过内积绝对值的归一化求和，既优雅又有效地推动原型分散
5. **适用范围广**：视觉 + NLP，虚假相关 + 属性不平衡 + 属性泛化，全面覆盖

## 局限与展望

1. **集成成员数 N 的选择**：论文未深入讨论如何选择最优 N，仅通过消融实验部分展示；在无验证子群标注时 N 的调优存在困难
2. **CheXpert 缺失**：在 Table 1 的 ERM+DPE 行中 CheXpert 列为"-"，未解释原因，考虑到医学影像是重要应用场景，这是遗憾
3. **顺序训练**：集成成员必须顺序训练，无法并行，当 N 较大时训练时间线性增长
4. **仅替换线性层**：冻结特征提取器意味着如果 ERM 特征本身质量差，DPE 的上限受限
5. **距离函数选择**：仅使用缩放欧式距离，未探索 Mahalanobis 距离等更灵活的度量
6. **理论分析不足**：缺乏关于为什么多样化原型集成一定能覆盖所有子群的理论保证

## 相关工作与启发

- **Kirichenko et al. (2022)** & **Izmailov et al. (2022)**：证明 ERM 特征已足够好，DPE 建立在此基础上冻结特征做分类头重训
- **Snell et al. (2017)**：原型网络的开创性工作，DPE 将其从 few-shot 扩展到子群鲁棒性场景
- **DivDis (Lee et al., 2022)**：通过消歧促进集成多样性，DPE 借鉴了显式多样化的思路
- **D-BAT (Pagliardini et al., 2023)**：源分布一致 + OOD 分歧的集成学习，启发了 DPE 的多样化方向
- **SubpopBench (Yang et al., 2023)**：统一评测框架，DPE 在其上全面验证

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 原型分类器 + 显式/隐式多样化组合应用于子群鲁棒性是新颖的，但各组件单独来看并非全新
- **实验充分度**: ⭐⭐⭐⭐ — 9个数据集、4种偏移类型、多个 baseline，实验覆盖面广且结果强劲，但 CheXpert 缺失扣分
- **写作质量**: ⭐⭐⭐⭐ — 行文清晰，动机图（Fig 1-2）直观，方法描述严谨
- **价值**: ⭐⭐⭐⭐ — 方法简单实用、即插即用、无需子群标注，对实际部署有较高价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 3D Test-time Adaptation via Graph Spectral Driven Point Shift](../../ICCV2025/3d_vision/3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)
- [\[CVPR 2026\] Artiverse: A Diverse and Physically Grounded Dataset for Articulated Objects](../../CVPR2026/3d_vision/artiverse_a_diverse_and_physically_grounded_dataset_for_articulated_objects.md)
- [\[ICLR 2026\] Weight Space Representation Learning on Diverse NeRF Architectures](../../ICLR2026/3d_vision/weight_space_representation_learning_on_diverse_nerf_architectures.md)
- [\[CVPR 2026\] Multimodal Semantic Bias Mitigation for Diverse Text-To-3D Generation](../../CVPR2026/3d_vision/multimodal_semantic_bias_mitigation_for_diverse_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
