---
title: >-
  [论文解读] FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning
description: >-
  [ECCV 2024][自监督学习][OOD检测] 提出FlowCon，一种基于密度估计的OOD检测方法，创新性地将正规化流（normalizing flow）与监督对比学习结合——在流模型的潜在空间中使用基于Bhattacharyya系数的对比损失学习类别条件高斯分布，无需外部OOD数据或重训分类器即可实现高效的OOD检测。
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "OOD检测"
  - "正规化流"
  - "对比学习"
  - "密度估计"
  - "Bhattacharyya系数"
---

# FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning

**会议**: ECCV 2024  
**arXiv**: [2407.03489](https://arxiv.org/abs/2407.03489)  
**代码**: [https://github.com/saandeepa93/FlowCon_OOD](https://github.com/saandeepa93/FlowCon_OOD)  
**领域**: 自监督学习  
**关键词**: OOD检测, 正规化流, 对比学习, 密度估计, Bhattacharyya系数

## 一句话总结

提出FlowCon，一种基于密度估计的OOD检测方法，创新性地将正规化流（normalizing flow）与监督对比学习结合——在流模型的潜在空间中使用基于Bhattacharyya系数的对比损失学习类别条件高斯分布，无需外部OOD数据或重训分类器即可实现高效的OOD检测。

## 研究背景与动机

深度学习模型在闭集（closed-world）假设下训练，假定测试时输入分布与训练分布一致。然而在实际部署中，模型不可避免地会遇到**分布外（OOD）样本**，包括语义偏移（far-OOD，新类别出现）和协变量偏移（near-OOD，输入空间变化但标签空间不变）。模型对OOD样本可能给出任意错误的高置信度预测，在医疗诊断、自动驾驶等安全关键领域后果严重。

**现有方法的痛点**：

**Post-hoc方法**（MSP、ODIN、Energy、ReAct等）直接操作预训练分类器的softmax分数，简单有效但在**近OOD场景下性能显著下降**。

**Outlier-based方法**（Heatmap等）需要外部OOD数据集用于训练，但OOD数据的空间极其庞大，假设特定的OOD分布可能引入偏差。

**密度方法**（Mahalanobis、ResFlow等）虽然原理上更可靠（显式建模ID数据分布），但存在严重的实用性问题：
   - **ResFlow**需要为**每个类别、每个网络层**训练一个独立的流模型（如CIFAR-100 + ResNet18 = 400个流模型），训练代价随数据集和模型复杂度爆炸式增长。
   - **Zhang et al.**提出联合训练分类器和流模型，但需要**重训原始分类器**，不适合实际部署。
   - 传统流模型学习单一高斯分布，**忽略类别信息**，可能对OOD样本赋予高似然值。

**核心矛盾**：如何在**不使用外部OOD数据、不重训分类器**的前提下，用**单一模型**高效学习带类别信息的密度估计，在各种OOD场景（far/near/mixed）下都有鲁棒表现？

**本文核心idea**：在预训练分类器的倒数第二层特征上训练一个流模型，同时用两个损失函数——流损失 $\mathcal{L}_{flow}$（最大化对数似然）和新提出的对比损失 $\mathcal{L}_{con}$（用Bhattacharyya系数作为相似度函数进行监督对比学习）——使流模型学习**类别条件的多模态高斯分布**，而非单一高斯。

## 方法详解

### 整体框架

FlowCon的训练pipeline：给定输入图像 $x$，预训练（冻结）的分类器提取倒数第二层深度特征 $z_{emb}$，流模型将 $z_{emb}$ 变换为潜在嵌入 $z_{flow}$ 及其对应分布参数 $\mathcal{N}(\mu, \sigma)$。训练时同时优化流损失和对比损失。推理时，计算测试样本在所有类别分布上的似然，取最大值作为OOD分数。

### 关键设计

1. **基于似然的对比相似度函数（Flow-based Contrastive Similarity）**：传统对比学习使用特征向量的点积/余弦相似度。FlowCon创新性地利用流模型的似然值定义新的相似度函数：

$$S_{flow}(z_i, z_j, \mathcal{N}_i) = \exp\left(\left(p_Z(z_i|\mathcal{N}_i) \cdot p_Z(z_j|\mathcal{N}_i)\right)^{\tau_1}\right)$$

其中 $p_Z(z_i|\mathcal{N}_i) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left[-\frac{1}{2}\left(\frac{z_i - \mu_i}{\sigma_i}\right)^2\right]$。

当 $\tau_1 = 0.5$ 时，exp内的乘积即为**Bhattacharyya系数**的广义形式，这是一种专门度量两个分布间相似性的经典统计量。

**设计动机**：将高维向量点积降维为标量似然值的乘积，既简化了计算，又在概率分布层面（而非特征空间）进行对比，使学习目标与OOD检测的密度估计本质更加一致。

2. **结合流损失的对比学习目标（FlowCon Loss）**：将新的相似度函数代入监督对比损失框架：

$$\mathcal{L}_{con} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{S_{flow}(z_i, z_p, \mathcal{N}_i) / \tau_2}{\sum_{a \in A(i)} S_{flow}(z_i, z_a, \mathcal{N}_i) / \tau_2}$$

与传统SCL不同，FlowCon的锚点不仅包括潜在向量 $z_i$，还包括**分布 $\mathcal{N}_i$**。总损失为：$\mathcal{L} = \mathcal{L}_{con} + \lambda \mathcal{L}_{flow}$，其中 $\lambda = 0.07$。

**设计动机**：$\mathcal{L}_{con}$ 在分布空间中进行对比，拉近同类分布、推远异类分布；$\mathcal{L}_{flow}$ 确保每个数据点的潜在嵌入属于其对应的类别分布。两个损失协同工作，使潜在空间形成**类别分明的多模态高斯分布**（见Fig.1 toy实验的直观展示）。

3. **OOD检测推理（OOD Detection Score）**：训练结束后，为每个类别 $c$ 计算经验分布参数：

$$\mu_c = \frac{1}{|\mathcal{X}_c|}\sum_{i \in \mathcal{X}_c} \mu_i, \quad \sigma_c = \frac{1}{|\mathcal{X}_c|}\sum_{i \in \mathcal{X}_c} \sigma_i$$

测试样本的OOD分数为：$S(x_{test}) = \max_{i \in \{1,...,k\}} p_Z(z_{test}|\mathcal{N}_{y=i})$——即在所有类别分布中的最大似然值。ID样本应获得高似然，OOD样本应获得低似然。

**设计动机**：将 $n$ 个训练样本的分布简化为 $k$ 个类别分布，大幅降低推理时的计算量，同时保留了类别区分能力。

### 损失函数 / 训练策略

- 流模型采用**RealNVP**架构，8个coupling blocks，单流层
- 在ResNet18的512维和WideResNet的128维倒数第二层特征上训练
- Adam优化器，学习率 $1\times10^{-5}$，权重衰减 $1\times10^{-5}$
- 训练700个epoch，batch size 64，图像尺寸 $32 \times 32$
- 超参数：$\lambda = 0.07$，$\tau_1 = 1.5$，$\tau_2 = 0.1$

## 实验关键数据

### 主实验：Far-OOD检测性能

ID数据为CIFAR-10/CIFAR-100，OOD为6个外部数据集的平均结果：

| ID数据集(模型) | 方法 | AUROC↑ | AUPR-S↑ | AUPR-E↑ | FPR-95↓ |
|---------------|------|--------|---------|---------|---------|
| CIFAR-10 (ResNet18) | MSP | 90.72 | 97.89 | 63.48 | 55.21 |
| CIFAR-10 (ResNet18) | Energy | 91.72 | 97.90 | 72.12 | 37.97 |
| CIFAR-10 (ResNet18) | ResFlow‡ | 95.60 | 99.35 | 82.82 | 13.22 |
| CIFAR-10 (ResNet18) | Heatmap† | 96.47 | 99.17 | 83.73 | 15.37 |
| **CIFAR-10 (ResNet18)** | **FlowCon** | **97.19** | **99.43** | **85.65** | 16.26 |
| CIFAR-100 (ResNet18) | MSP | 79.29 | 95.04 | 40.34 | 76.58 |
| CIFAR-100 (ResNet18) | Heatmap† | 86.74 | 96.49 | 58.78 | 52.73 |
| **CIFAR-100 (ResNet18)** | **FlowCon** | **88.22** | **96.85** | **67.89** | **41.85** |

### Near-OOD / Mixed-OOD 检测

| 场景 | 方法 | AUROC↑ | FPR-95↓ | 说明 |
|------|------|--------|---------|------|
| C10→C100 Mixed (ResNet) | Energy | 85.60 | 55.20 | post-hoc基线 |
| C10→C100 Mixed (ResNet) | ResFlow | 76.40 | 67.20 | 流模型，性能差 |
| **C10→C100 Mixed (ResNet)** | **FlowCon** | **93.97** | **35.95** | **全指标最优** |
| C100→C10 Near (ResNet) | Energy | 77.06 | 81.15 | 近OOD困难 |
| C100→C10 Near (ResNet) | ResFlow | 58.29 | 79.00 | 流模型崩塌 |
| **C100→C10 Near (ResNet)** | **FlowCon** | **82.80** | **67.60** | **全指标最优** |

在最具挑战性的Mixed-OOD场景中，FlowCon的AUROC高达93.97%，较Energy提升8.37%，较ResFlow提升17.57%。

### 消融实验：λ值对性能的影响

在CIFAR-100 (WideResNet) Far-OOD场景下：

| λ值 | AUROC↑ | AUPR-S↑ | AUPR-E↑ | FPR-95↓ | 说明 |
|-----|--------|---------|---------|---------|------|
| 0.05 | 75.62 | 92.70 | 41.84 | 72.58 | 流损失权重太低 |
| **0.07** | **83.62** | **96.60** | **53.34** | **60.28** | **最佳平衡** |
| 0.30 | 75.75 | 92.76 | 48.61 | 63.67 | 流损失过大 |
| 0.50 | 78.60 | 93.96 | 49.07 | 65.92 | 性能下降 |
| 1.00 | 78.57 | 93.24 | 45.94 | 67.85 | 对比损失被压制 |

### 分类保持性验证

| 数据集 | 模型 | 原始分类器 | FlowCon | 差异 |
|--------|------|-----------|---------|------|
| CIFAR-10 | ResNet18 | 94.3% | 94.2% | -0.1% |
| CIFAR-10 | WideResNet | 93.3% | 93.8% | +0.5% |
| CIFAR-100 | ResNet18 | 75.8% | 74.9% | -0.9% |
| CIFAR-100 | WideResNet | 70.9% | 71.1% | +0.2% |

FlowCon学到的类别分布可直接用于分类（Bayes决策），精度与原始分类器几乎一致。

### 关键发现

- FlowCon在ResNet18上的所有OOD场景中均取得最佳或接近最佳性能，且对CIFAR-100（100类）同样鲁棒
- 相比ResFlow需要400个模型（100类×4层），FlowCon只需训练**1个模型**即可在倒数第二层特征上完成OOD检测
- 似然直方图分析显示：FlowCon中OOD样本的最高似然值**从未超过**ID样本的最高似然值，解决了流模型对OOD赋予高似然的经典问题
- UMAP可视化表明FlowCon学到了良好的类别聚类结构，near-OOD样本与语义相似的ID类别重叠——这与性能下降的趋势一致

## 亮点与洞察

- **巧妙的概率融合**：用Bhattacharyya系数替代余弦相似度作为对比学习的相似度函数，实现了概率分布空间中的对比学习，而非传统的特征空间对比
- **单模型替代多模型**：相比ResFlow的每类每层一个模型的暴力方案，FlowCon用一个模型在一个特征层上就实现了更好的性能
- **类别保持性**：对比损失不仅帮助OOD检测，还保持了原始分类器的分类能力——一个分支同时解决OOD检测和ID分类
- **不需要OOD数据**：整个训练过程完全基于ID数据，不需要任何OOD数据的假设

## 局限与展望

- **WideResNet上性能相对较弱**：128维特征对于耦合层流模型来说维度偏低，RealNVP/Glow在高维数据上表现更好。未来可探索适合低维特征的流架构
- **维度限制**：正规化流要求输入输出维度相同，限制了模型在不同分类器上的灵活性
- **训练时间较长**：700个epoch的训练仍然不够高效
- **近OOD场景仍有提升空间**：like its UMAP visualization shows, near-OOD的类别重叠是所有方法的共同挑战

## 相关工作与启发

- **Normalizing Flows (Dinh et al., RealNVP)**：可逆生成模型，提供精确的对数似然计算
- **ResFlow (Zisselman et al.)**：类别级残差流做OOD检测，但训练成本高
- **SupCon (Khosla et al.)**：监督对比学习的经典框架，FlowCon在其基础上替换了相似度函数
- **Kirichenko et al.**：揭示了流模型对OOD赋予高似然的问题，FlowCon通过类别条件分布解决了这一问题
- 启发：对比学习中的相似度函数可以根据任务特性定制化设计，不必局限于余弦相似度或点积

## 评分

- 新颖性: ⭐⭐⭐⭐ 将Bhattacharyya系数作为流模型对比学习的相似度函数，概念新颖且数学优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三种OOD场景、两种分类器、四个指标、似然直方图、UMAP可视化、分类保持验证、λ消融，极其完整
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，toy实验直觉展示精彩，公式推导严谨
- 价值: ⭐⭐⭐⭐ 为密度方法的OOD检测提供了高效可行的新方案，单模型设计有工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Adaptive Multi-head Contrastive Learning](adaptive_multihead_contrastive_learning.md)
- [\[ECCV 2024\] Rethinking Unsupervised Outlier Detection via Multiple Thresholding](rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)
- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)

</div>

<!-- RELATED:END -->
