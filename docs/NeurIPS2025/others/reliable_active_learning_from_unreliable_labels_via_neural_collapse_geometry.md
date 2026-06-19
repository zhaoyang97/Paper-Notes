---
title: >-
  [论文解读] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry
description: >-
  [NeurIPS 2025 (Workshop)][主动学习] 提出 NCAL-R，利用深度网络训练后期涌现的 Neural Collapse 几何结构，设计类均值对齐扰动（CMAP）和特征波动（FF）两个评分指标来选择样本，使主动学习在标签噪声和分布偏移下更加可靠，在 ImageNet-100 和 CIFAR-100 上一致优于传统 AL 基线。
tags:
  - "NeurIPS 2025 (Workshop)"
  - "主动学习"
  - "Neural Collapse"
  - "特征几何"
  - "噪声标签鲁棒"
  - "OOD泛化"
---

# Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry

**会议**: NeurIPS 2025 (Workshop)  
**arXiv**: [2510.09740](https://arxiv.org/abs/2510.09740)  
**代码**: [https://github.com/Vision-IIITD/NCAL](https://github.com/Vision-IIITD/NCAL)  
**领域**: 主动学习 / 可靠机器学习  
**关键词**: 主动学习, Neural Collapse, 特征几何, 噪声标签鲁棒, OOD泛化

## 一句话总结

提出 NCAL-R，利用深度网络训练后期涌现的 Neural Collapse 几何结构，设计类均值对齐扰动（CMAP）和特征波动（FF）两个评分指标来选择样本，使主动学习在标签噪声和分布偏移下更加可靠，在 ImageNet-100 和 CIFAR-100 上一致优于传统 AL 基线。

## 研究背景与动机

**领域现状**：主动学习（Active Learning）通过优先选择最有信息量的样本来减少标注成本，主流策略包括基于不确定性（uncertainty）、基于多样性（diversity）和基于代表性（representativeness）的方法。

**现有痛点**：传统 AL 方法在理想条件下有效，但在现实场景中面临三重挑战：(1) 标签噪声——标注者会犯错，AL 启发式方法（尤其是不确定性方法）倾向反复选择错标样本，放大错误；(2) 分布偏移——训练数据与测试数据分布不一致时，传统方法的选择策略失效；(3) 跨数据集/架构迁移性差——很多方法需要针对特定任务调参。

**核心矛盾**：不确定性高的样本可能确实信息量大，但也可能只是被错标了或属于 OOD 数据——传统方法无法区分"有价值的不确定性"和"有害的不确定性"。

**本文目标** 如何在标签不可靠和分布可能偏移的条件下，选择既能强化类间分离度又能暴露真正模糊区域的样本？

**切入角度**：训练后期（terminal phase）深度网络的特征会出现 Neural Collapse (NC) 现象——类内特征坍缩到均值、类均值排列成等角紧框架。这种结构化的几何信息提供了超越传统启发式的选择信号：能扰动类间几何结构的样本（CMAP 高）是有价值的，而特征在训练过程中剧烈波动的样本（FF 高）暗示真正的歧义性。

**核心 idea**：用 Neural Collapse 的几何结构来识别"对特征空间有结构性影响"的样本，替代传统的不确定性/多样性启发式。

## 方法详解

### 整体框架

NCAL-R 在每个 AL 轮次中，先在当前标注集上训练模型至 NC 阶段，然后对未标注池中每个样本计算 CMAP 和 FF 两个分数，标准化后取平均作为综合评分，选取 top-k 样本进行标注。无需辅助网络、伪标签或任务特定调参，可应用于任何能提取特征嵌入的骨干网络。

### 关键设计

1. **类均值对齐扰动 (CMAP)**:

    - 功能：量化候选样本对类间几何结构的扰动程度
    - 核心思路：定义类均值对齐 CMA 为所有类均值对之间的余弦相似度平均值。对于候选样本 $x$（模型预测为类 $c$），假设将其加入标注集后更新类均值 $\tilde{\mu}_t^c$，计算 CMA 的变化量 $\mathrm{CMAP}(x) = \mathrm{CMA}(\mathcal{L}_t \cup x) - \mathrm{CMA}(\mathcal{L}_t)$。经推导简化为一个点积：$(\bar{\tilde{\mu}}_t^c - \bar{\mu}_t^c)^\top (M_t - \bar{\mu}_t^c)$，计算高效。高 CMAP 意味着该样本会显著改变类均值间的相互关系，标注它有助于减小类均值间的相关性（降低泛化误差上界）
    - 设计动机：根据 Jin et al. (2020) 的理论，泛化误差上界与权重相关性有关，而 NC 条件下类均值与分类器权重对齐，因此最小化 CMA 等价于最小化泛化误差的一个代理

2. **特征波动 (FF)**:

    - 功能：捕捉样本在训练过程中表示的不稳定性
    - 核心思路：给定训练终期的多个检查点 $\{\theta_t\}_{t=T_i}^{T_f}$，统计样本 $x$ 在连续检查点间预测标签翻转的次数 $\mathrm{FF}(x) = \sum_{t=T_i+1}^{T_f} \mathbf{1}[\hat{y}_t(x) \neq \hat{y}_{t-1}(x)]$。高 FF 表示即使在大部分特征已稳定的 NC 阶段，该样本的预测仍在反复跳变——这标识了真正的决策边界样本
    - 设计动机：传统不确定性指标（如熵）是某一时刻的快照，而 FF 是跨时间的稳定性度量，更能区分"模型暂时不确定"（FF 低）和"本质上处于类边界"（FF 高）

3. **联合采集策略**:

    - 功能：综合结构性影响和预测不稳定性两个维度
    - 核心思路：分别对 CMAP 和 FF 按均值和标准差标准化后取平均 $\text{Score}(x) = (\text{CMAP}(x) + \text{FF}(x))/2$，选择分数最高的 k 个样本。这保证选出的样本既对特征几何有结构性影响、又处于真正的歧义区域
    - 设计动机：CMAP 侧重类间结构优化，FF 侧重歧义发现，两者互补

## 实验关键数据

### 主实验（CIFAR-10, OOD 检测 AUROC, ImageNet-100 训练）

| 方法 | 10% | 15% | 20% | 25% | 30% | 35% |
|------|-----|-----|-----|-----|-----|-----|
| Random | 77.18 | 80.57 | 84.13 | 85.45 | 86.89 | 87.82 |
| CoreSet | 81.56 | 83.73 | 85.66 | 87.10 | 88.29 | 88.95 |
| CDAL | 81.78 | 84.28 | 85.90 | 86.34 | 87.98 | 88.92 |
| **NCAL** | **82.49** | **85.55** | **87.89** | **89.15** | **90.53** | **91.53** |

### OOD 泛化（30% 标签预算, ImageNet-100 训练后线性探测）

| 方法 | ImgNet-R | CIFAR100 | Flowers | NINCO | CUB | Avg |
|------|----------|----------|---------|-------|-----|-----|
| Random | 18.06 | 41.64 | 58.69 | 64.23 | 37.84 | 46.95 |
| CDAL | 17.56 | 41.98 | 58.13 | 65.87 | 38.53 | 47.21 |
| **NCAL** | **19.27** | **43.78** | **60.87** | **67.66** | **40.01** | **48.98** |
| 100% data | 20.01 | 45.31 | 61.77 | 69.90 | 42.29 | 50.87 |

### GCD（广义类别发现, 60-40 已知-新类划分）

| 方法 | All Classes | Old Classes | New Classes |
|------|------------|-------------|-------------|
| Random | 33.20 | 50.34 | 20.35 |
| CoreSet | 32.23 | 49.98 | 18.92 |
| **NCAL** | **35.07** | **51.95** | **23.05** |

### 关键发现

- NCAL 在所有标签预算下（10%-35%）都一致优于基线，且优势在低预算时更明显
- OOD 泛化提升约 2% 平均值，说明 NC 引导的特征空间确实更具迁移性
- 新类发现准确率提升 +2.1 点（vs 最佳基线），表明 NCAL 的特征空间能自然适应新类别
- 类间距离分析显示 NCAL 的平均类间距为 15.944（vs Random 15.114），更好的类间分离
- 长尾分布下 NCAL 提升约 3%（45.15% vs 42.30%），表明几何引导对不平衡数据也有效

## 亮点与洞察

- **将 Neural Collapse 从一个"解释性理论"转化为"实用工具"**：NC 过去主要用于理解训练动态，本文首次将其系统地应用于 AL 的样本选择，开辟了 NC 的实用化方向。这个思路可以迁移到课程学习、数据选择等类似场景
- **CMAP 的推导巧妙**：通过 NC 条件下类均值 ≈ 分类器权重的关系，将泛化误差上界转化为特征空间的几何量度，再进一步简化为一个高效的点积计算。理论优雅且计算实用
- **无需额外组件的轻量方案**：不需要辅助网络、伪标签或特定架构，只需要模型的特征嵌入和训练检查点

## 局限与展望

- Workshop 论文，实验规模偏小（ResNet-18 骨干，最大数据集 ImageNet-100），在大规模模型和数据集上的表现未知
- FF 需要存储多个训练检查点，存储和计算开销随模型规模和检查点数量增长
- NC 理论要求训练至接近零误差，但实际中模型可能不会完全达到 NC 状态，此时 CMAP 的理论保证可能弱化
- 没有与更新的 AL 方法（如 BADGE、BAIT）对比

## 相关工作与启发

- **vs CoreSet**: CoreSet 追求特征空间的覆盖多样性，NCAL 追求类间几何的结构性优化。后者更有理论基础且在 GCD 任务上优势更大
- **vs CDAL**: CDAL（Contextual Diversity）考虑上下文多样性，但仍是静态快照式的选择。NCAL 的 FF 引入了时间维度的不稳定性信号
- **vs ActiveOOD (SISOMe)**: ActiveOOD 依赖 OOD 过滤启发式，在 closed-set AL 中表现反而差；NCAL 的统一框架在 OOD 和 closed-set 中都有效

## 评分

- 新颖性: ⭐⭐⭐⭐ Neural Collapse 指导 AL 是新颖的切入点，理论推导优雅
- 实验充分度: ⭐⭐⭐ Workshop 论文篇幅限制，实验规模偏小，缺少大模型验证
- 写作质量: ⭐⭐⭐⭐ 简洁清晰，公式推导紧凑
- 价值: ⭐⭐⭐⭐ 开辟了 NC 在 AL 中的应用方向，CMAP+FF 的设计方法论有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](../../CVPR2025/others/instance-wise_supervision-level_optimization_in_active_learning.md)
- [\[NeurIPS 2025\] On a Geometry of Interbrain Networks](on_a_geometry_of_interbrain_networks.md)
- [\[ICML 2025\] AutoAL: Automated Active Learning with Differentiable Query Strategy Search](../../ICML2025/others/autoal_automated_active_learning_with_differentiable_query_strategy_search.md)

</div>

<!-- RELATED:END -->
