---
title: >-
  [论文解读] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier
description: >-
  [AAAI 2026 (Oral)][可解释性][特征归因] 提出首个基于数据分布的特征归因方法 DFAX，通过比较目标实例在目标类与非目标类条件概率之差来量化特征重要性，首次给出特征归因的形式化定义，在10个数据集上显著优于SHAP/LIME等基线且速度快数个数量级。 特征归因（Feature Attribution）是可…
tags:
  - "AAAI 2026 (Oral)"
  - "可解释性"
  - "特征归因"
  - "可解释AI"
  - "核密度估计"
  - "模型无关"
  - "分布方法"
---

# Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier

**会议**: AAAI 2026 (Oral)  
**arXiv**: [2511.09332](https://arxiv.org/abs/2511.09332)  
**代码**: 无  
**领域**: 可解释AI / 特征归因  
**关键词**: 特征归因, 可解释AI, 核密度估计, 模型无关, 分布方法

## 一句话总结

提出首个基于数据分布的特征归因方法 DFAX，通过比较目标实例在目标类与非目标类条件概率之差来量化特征重要性，首次给出特征归因的形式化定义，在10个数据集上显著优于SHAP/LIME等基线且速度快数个数量级。

## 研究背景与动机

特征归因（Feature Attribution）是可解释AI的核心技术之一，旨在为黑盒模型的每个输入特征计算贡献分数，帮助用户理解模型决策。现有方法主要分两大家族：

**局部近似方法**（如LIME、DLIME、MAPLE）：在目标实例周围拟合简单的代理模型（如线性回归），从中提取特征重要性。但这类方法只利用了数据集的一个局部子集，无法充分利用全局信息。

**扰动方法**（如SHAP、PFI）：通过扰动/遮挡特征并观察模型输出变化来衡量特征重要性。但很多实现（如SHAP的Shapley采样值）会创建合成实例，混合来自目标实例和背景数据集的特征值，产生分布外（OOD）数据。

**核心问题**：尽管研究了多年，特征归因问题一直缺乏**形式化的问题定义**。没有明确的标准，方法之间的比较和评估都缺乏理论基础。

**本文动机**：
- 给出特征归因的形式化定义（Definition 1），明确要求解释必须由底层数据分布 $\mathcal{P}$ 支撑，不能使用OOD实例
- 在此定义下分析现有方法的合规性，发现很多流行方法（如LIME、SHAP的常用实现）不满足这一要求
- 提出一种从分布角度直接进行特征归因的方法，克服现有方法的局限

## 方法详解

### 整体框架

DFAX（Distributional Feature Attribution eXplanations）采用全新的分布视角：直接利用数据分布来计算特征归因分数，而不是拟合局部代理模型或进行扰动。其核心思想是：对于每个特征，比较目标实例在「目标类」和「非目标类」条件下的概率密度差异。

### 关键设计

#### 1. **形式化问题定义（Definition 1）**

给定分类器 $f$、目标实例 $\mathbf{x}^*$ 和数据集 $\mathbf{X}$（均为分布 $\mathcal{P}$ 的i.i.d.样本），特征归因的任务是为每个特征 $s \in \mathcal{A}$ 计算分数 $I(\mathbf{x}^*, s | \mathbf{X})$，满足：
- 分数量化特征值 $\mathbf{x}_s^*$ 对分类器产生预测 $y^*$ 的影响程度
- 解释模型 $I(\cdot|\mathbf{X})$ 必须直接基于**未修改的**数据集 $\mathbf{X}$ 构建
- 任何改变底层分布的 $\mathbf{X}$ 修改都会使解释无效

这个定义的关键意义在于：禁止使用合成/OOD数据来生成解释。LIME通过随机扰动生成合成邻域、SHAP的采样实现通过混合特征值创建合成实例，都违反了这一准则。

#### 2. **DFAX方法（Definition 2）**

对于目标实例 $\mathbf{x}^*$（预测类别 $y^*$）和特征 $s$，DFAX的归因分数定义为：

$$I(\mathbf{x}^*, s | \mathbf{X}) = K^s(\mathbf{x}^* | \mathbf{X}_{\{y^*\}}) - K^s(\mathbf{x}^* | \mathbf{X} \setminus \mathbf{X}_{\{y^*\}})$$

其中：
- $K^s$ 是在特征 $s$ 定义的一维子空间上的核密度估计（KDE）
- $\mathbf{X}_{\{y^*\}}$ 是数据集中被分类器预测为目标类的子集
- 第一项衡量目标实例在**目标类数据**中的概率密度
- 第二项衡量在**所有其他类数据**中的概率密度
- 差值越大，说明该特征值越能区分目标类与其他类

**设计动机**：如果某个特征值在目标类分布中概率高、在其他类分布中概率低，说明这个特征值是区分目标类的关键特征，应该获得高归因分数。

#### 3. **KDE实现与加速**

DFAX支持两种KDE实现：
- **DFAX_G**：使用高斯核密度估计（GKDE），超参数为带宽 $\gamma = \frac{1}{2\sigma^2}$
- **DFAX_S**：使用SiNNE（简化的iNNE），超参数为子采样大小 $\psi$ 和集成数量 $t$

**关键加速技巧**：如果核函数可以通过有限维特征映射近似（如Nyström方法），可以预计算数据集的核均值映射 $\hat{\Phi}(\mathbf{X})$，之后每个目标实例的密度估计只需 $\mathcal{O}(1)$ 时间。

### 损失函数 / 评估策略

DFAX不涉及训练，是一种惰性学习方法，类似KNN。评估使用标准的 deletion score（逐步删除重要特征后分类概率的AUC，越低越好）和 insertion score（逐步加入重要特征后分类概率的AUC，越高越好）。

**DFAX的独特优势**：
- **完全解耦于分类器**：构建解释后不再需要查询分类器，适合分类器不可用或查询代价高的场景
- **全局信息利用**：使用整个数据集 $\mathbf{X}$ 的分布信息，而非局部子集
- **可解释数据本身**：用真实标签替代预测，可以解释数据固有的类结构

## 实验关键数据

### 主实验

在10个真实数据集上（覆盖表格/文本/图像，规模从520到70000样本），使用多种分类器（RF、LR、SVM、MLP、ResNet等）进行评估。

| 方法 | Deletion ↓ 均分 | Deletion 排名 | Insertion ↑ 均分 | Insertion 排名 |
|------|----------------|--------------|-----------------|---------------|
| **DFAX_G** | **0.3244** | **1.5** | **0.7708** | **1.2** |
| **DFAX_S** | 0.3344 | 1.6 | 0.7470 | 2.0 |
| DLIME | 0.4595 | 4.1 | 0.6612 | 3.8 |
| LINEX | 0.5287 | 4.8 | 0.6326 | 4.1 |
| SLISE | 0.5457 | 5.5 | 0.6068 | 5.1 |
| SHAP | 0.5246 | 5.4 | 0.5463 | 7.3 |
| MAPLE | 0.5671 | 6.1 | 0.5800 | 6.1 |
| Random | 0.5709 | 7.0 | 0.5838 | 6.4 |

DFAX在10个数据集中的9个上排名第一或第二，大幅领先所有基线。SHAP和MAPLE的insertion分数甚至低于随机基线。

### 消融实验

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| DFAX_G（GKDE） | Deletion 0.3244, Insertion 0.7708 | 整体最佳 |
| DFAX_S（SiNNE） | Deletion 0.3344, Insertion 0.7470 | 略逊于GKDE |
| HER2st基因归因 | 准确率 95.64% vs DLIME 79.51% | 仅用一半基因即保持高预测精度 |
| RottenTomatoes情感词 | 成功识别compelling/bad等关键词 | DLIME选择了不相关的词如real/humor |

### 关键发现

1. **运行效率**：DFAX在所有数据集上运行最快，通常比其他方法快1-2个数量级。在MNIST上，DFAX约0.01秒，而SHAP需约100秒，SLISE需约1000秒。
2. **定义合规性分析**：满足Definition 1的方法（DLIME、SLISE、MAPLE）普遍优于不满足的方法（LIME、SHAP采样实现），验证了形式化定义的实际意义。
3. **空间转录组学应用**：在HER2st数据上识别关键癌症基因时，DFAX只用一半基因就保持95.64%的预测准确率，而DLIME仅79.51%，展示了在真实科学发现场景中的潜力。

## 亮点与洞察

- **首次形式化定义特征归因问题**，为整个领域提供了理论基础和评估标准，揭示了LIME/SHAP等流行方法的根本缺陷
- **分布视角的新范式**：跳出了"拟合代理模型"和"扰动观察"两大传统思路，直接从条件概率密度的角度理解特征重要性
- **极致简洁与高效**：方法定义优雅（一个减法），计算高效（预计算后O(1)），无需超参数调优即可获得良好性能
- **完全解耦于分类器**：一旦有预测结果，不再需要查询分类器，适用于隐私敏感/昂贵查询场景

## 局限与展望

- 目前仅处理单个特征的归因，未扩展到**特征组归因**（作者列为近期工作）
- KDE在高维空间中可能效率降低（维度灾难），但本文通过逐维计算规避了这个问题
- 没有讨论**公理性质**（如Shapley值满足的效率性、对称性等），作者将其列为未来工作
- 需要有标签的数据集 $\mathbf{X}$（或分类器预测）来划分类别子集

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次形式化定义+全新分布视角，开创性工作
- **实验充分度**: ⭐⭐⭐⭐⭐ — 10个数据集、6种基线、定量+定性+效率全面评估
- **写作质量**: ⭐⭐⭐⭐⭐ — 论文结构清晰，定义-分析-方法-实验逻辑链完整
- **价值**: ⭐⭐⭐⭐ — 对XAI领域有重要理论和实践价值，Oral论文

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] The Perceived Fragility of Explanations in Audio Models: Manipulation of Attribution with Unchanged Predictions](../../ICML2026/interpretability/the_perceived_fragility_of_explanations_in_audio_models_manipulation_of_attribut.md)
- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](../../ICML2026/interpretability/manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[CVPR 2026\] Back to the Feature: Explaining Video Classifiers with Video Counterfactual Explanations](../../CVPR2026/interpretability/back_to_the_feature_explaining_video_classifiers_with_video_counterfactual_expla.md)
- [\[ACL 2025\] Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](../../ACL2025/interpretability/normalized_aopc_faithfulness_metrics.md)
- [\[ICCV 2025\] VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow](../../ICCV2025/interpretability/vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)

</div>

<!-- RELATED:END -->
