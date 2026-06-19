---
title: >-
  [论文解读] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach
description: >-
  [CVPR 2025][主动学习] 提出EAOA框架，通过基于自由能的认知不确定性（EU）和偶然不确定性（AU）度量，结合自适应粗到细的查询策略，在开放集主动学习场景中有效选择既属于已知类又具有高信息量的样本。 主动学习（AL）通过迭代选择最有信息量的样本进行标注来降低标注成本，但在开放集场景下面临严峻挑战：未标注数据池中包…
tags:
  - "CVPR 2025"
  - "主动学习"
  - "开放集标注"
  - "认知不确定性"
  - "偶然不确定性"
  - "能量模型"
---

# Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach

**会议**: CVPR 2025  
**arXiv**: [2502.19691](https://arxiv.org/abs/2502.19691)  
**代码**: [GitHub](https://github.com/)  
**领域**: 其他  
**关键词**: 主动学习, 开放集标注, 认知不确定性, 偶然不确定性, 能量模型

## 一句话总结

提出EAOA框架，通过基于自由能的认知不确定性（EU）和偶然不确定性（AU）度量，结合自适应粗到细的查询策略，在开放集主动学习场景中有效选择既属于已知类又具有高信息量的样本。

## 研究背景与动机

主动学习（AL）通过迭代选择最有信息量的样本进行标注来降低标注成本，但在开放集场景下面临严峻挑战：未标注数据池中包含未知类样本。

现有方法存在两个根本性问题：

1. **仅关注低认知不确定性（EU）**：如LfOSA、EOAL优先选择可能属于已知类的样本，虽然查询精度高，但选出的样本信息量有限，模型性能不佳
2. **仅关注高偶然不确定性（AU）**：如BUAL优先选择预测高度不确定的样本，但对未知类样本的AU度量本身就没有意义（因为AU仅在闭集情境下有效）

作者通过实验验证：分别单独使用EU或AU都表现不佳，但有效结合两者可以显著提升性能。核心思路是：先用EU筛选出大概率属于已知类的候选集（确保闭集属性），再在该候选集中用AU选择最具信息量的样本。

这一洞察简洁而深刻：AU的语义前提是样本属于已知类分布，只有先通过EU过滤，AU才有意义。

## 方法详解

### 整体框架

EAOA维护两个网络：一个 $(C+1)$ 类检测器（用于评估EU）和一个 $C$ 类目标分类器（用于评估AU），采用粗到细的两阶段查询策略。

### 关键设计

**设计一：基于能量的认知不确定性（Energy-based EU）**

- **功能**：从学习和数据驱动两个视角可靠评估样本是否属于已知类
- **核心思路**：将EU定义为已知类自由能得分与未知类自由能得分之差：$EU(x) = E_{kno}(x) - E_{unk}(x)$。学习视角直接用检测器预测；数据驱动视角通过K近邻箭头投票构造概率分布，两者通过GMM转化为概率后逐元素相乘融合
- **设计动机**：单一的自由能不足以衡量EU，因为没有利用已标注未知类样本信息。通过 $(C+1)$ 类检测器和双视角融合，在数据稀缺的AL场景中获得更可靠的EU估计

$$EU(x) = -\log\sum_{c=1}^{C} e^{f_c(x)} + \log(1 + e^{f_{C+1}(x)})$$

**设计二：基于能量的偶然不确定性（Energy-based AU）**

- **功能**：评估样本在已知类间的混淆程度（边界样本信息量更大）
- **核心思路**：将AU定义为全类自由能得分与次要类（去掉最可能类）自由能得分之差，反映样本距离决策边界的远近
- **设计动机**：传统的熵度量在开放集中无效，因为未知类样本也会获得高熵。通过自由能差值，AU仅在闭集候选中有效使用

$$AU(x) = -\log\sum_{c=1}^{C} e^{f_c(x)} + \log\left[\sum_{c=1}^{C} e^{f_c(x)} - e^{f_{y_{max}}(x)}\right]$$

**设计三：目标驱动的自适应采样策略**

- **功能**：自动调节候选集大小以平衡EU过滤和AU选择的效果
- **核心思路**：每轮先选 $kb$ 个低EU样本组成候选集，再从中选 $b$ 个高AU样本作为查询集。$k$ 值根据实际查询精度 $rP$ 与目标精度 $tP$ 的比较自适应调整
- **设计动机**：$k$ 太小则AU无法发挥作用，$k$ 太大则闭集假设被破坏。自适应策略避免了手动调参，增强了跨数据集的泛化性

### 损失函数

检测器训练同时使用交叉熵损失和Margin-based能量损失：

$$\mathcal{L}_{detector} = \mathcal{L}_{ce} + \lambda_e \mathcal{L}_{energy}$$

能量损失对已知类样本最大化 $E_{kno}$，对未知类样本最小化 $E_{kno}$，通过margin参数 $m_{kno}$ 和 $m_{unk}$ 控制分离程度。

## 实验关键数据

### 主实验：CIFAR-100（mismatch ratio=40%）

| 方法 | 最终测试精度 | 平均查询精度 |
|------|-------------|-------------|
| Random | ~68% | ~60% |
| CCAL | ~70% | ~72% |
| LfOSA | ~72% | ~82% |
| EOAL | ~73% | ~85% |
| BUAL | ~72% | ~70% |
| **EAOA (Ours)** | **~76%** | **~83%** |

### 跨数据集一致性

| 数据集 | Mismatch Ratio | EAOA优势 |
|--------|---------------|---------|
| CIFAR-10 | 20%/30%/40% | 所有轮次最优 |
| CIFAR-100 | 20%/30%/40% | 所有轮次最优 |
| Tiny-ImageNet | 10%/15%/20% | 所有轮次最优 |

### 消融实验

| 组件 | 测试精度 | 平均查询精度 |
|------|---------|-------------|
| 仅Free Energy (EU) | ~73% | ~84% |
| + 数据驱动EU | ~74% | ~85% |
| + Energy Loss | ~74.5% | ~86% |
| + AU (完整EAOA) | **~76%** | **~83%** |

### 关键发现

1. 单独低EU选择（Certainty）与Random性能相近，证明了高AU在闭集假设下的必要性
2. 融合EU和AU后性能显著提升，验证了"先保证闭集再看信息量"的核心假设
3. EAOA在所有AOSA方法中训练时间最短，同时保持最高精度
4. 目标精度 $tP$ 在0.4-0.8范围内性能波动很小，参数鲁棒性强

## 亮点与洞察

1. **从不确定性分解角度重新理解开放集AL**：清晰指出EU和AU在开放集中的不同角色，为何两者需要级联而非简单混合
2. **能量框架的统一性**：EU和AU都基于自由能理论推导，理论基础一致且优雅
3. **双视角EU估计**：学习视角+数据驱动视角的融合，特别适合AL中数据稀缺的场景

## 局限与展望

1. 实验仅在图像分类任务上验证，未扩展到检测或分割等更复杂任务
2. 假设未知类数据可以统一归为一个类别，当未知类内部差异极大时可能效果不佳
3. GMM拟合是启发式选择，是否有更好的概率转换方式值得探索
4. 可以考虑将框架扩展到更实际的数据流场景（streaming setting）

## 相关工作与启发

- **LfOSA**：首次引入 $(C+1)$ 类检测器框架，本文在此基础上改进EU度量
- **EOAL**：提出在检测器中用熵区分已知/未知类，但忽略了AU的作用
- **能量模型（EBM）**：自由能作为OOD检测指标的思想来自Liu et al. (NeurIPS 2020)
- 启发：在其他涉及数据选择的场景（如课程学习、数据清洗），EU/AU的分解思路可能同样有价值

## 评分

⭐⭐⭐⭐ — 理论动机清晰、方法推导严谨、实验全面。将EU和AU的分析统一到自由能框架中，既有理论深度又实用高效。不足是实验任务偏简单（仅分类），缺乏目标检测等复杂任务验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](../../ICML2025/others/rethinking_aleatoric_and_epistemic_uncertainty.md)
- [\[ECCV 2024\] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](../../ECCV2024/others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)
- [\[CVPR 2025\] Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](instance-wise_supervision-level_optimization_in_active_learning.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
