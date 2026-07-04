---
title: >-
  [论文解读] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation
description: >-
  [ECCV2024][active learning] 提出 BUAL 框架，通过 Random Label Negative Learning 将未知类样本推向高置信区域、已知类样本推向低置信区域，结合双向不确定性采样策略，在开放集场景下有效选出高信息量的已知类样本。 - 主动学习的核心目标：从未标注数据池中迭代选出最有信…
tags:
  - "ECCV2024"
  - "active learning"
  - "Open-Set Annotation"
  - "Negative Learning"
  - "uncertainty estimation"
---

# Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation

**会议**: ECCV2024  
**arXiv**: [2402.15198](https://arxiv.org/abs/2402.15198)  
**代码**: [GitHub](https://github.com/zongcc/BUAL)  
**领域**: 其他  
**关键词**: active learning, Open-Set Annotation, Negative Learning, uncertainty estimation

## 一句话总结

提出 BUAL 框架，通过 Random Label Negative Learning 将未知类样本推向高置信区域、已知类样本推向低置信区域，结合双向不确定性采样策略，在开放集场景下有效选出高信息量的已知类样本。

## 背景与动机

- **主动学习的核心目标**：从未标注数据池中迭代选出最有信息量的样本送给标注者，以最小标注成本训练高效模型
- **封闭集假设的局限**：传统 AL 方法假设未标注池中的类别与目标任务完全一致，但实际场景中往往混入大量未知类（open-set）样本
- **已有方法的两难困境**：
    - 传统不确定性方法（LC / Margin / Entropy）倾向于选低置信度样本，但未知类样本同样具有低置信度，容易误选
    - 开放集标注方法（CCAL、LfOSA）优先选最可能属于已知类的样本，但这些往往是模型已经掌握的"简单样本"，对训练帮助有限
    - 两类方法对 openness ratio 敏感：OSA 方法在低开放度时不如随机采样，传统方法在高开放度时失效

## 核心问题

**如何在开放集场景中，同时满足"高信息量"和"属于已知类"两个目标进行样本选择？**

关键洞察：如果能将未知类样本推向高置信度区域，则现有基于不确定性的 AL 方法可以直接用于开放集场景——低置信度区域中剩下的就是信息量高的已知类样本。

## 方法详解

### 1. Random Label Negative Learning (RLNL)

**核心思想**：利用 Negative Learning（互补标签学习）微调模型，实现已知/未知类样本在置信度空间上的分离。

**训练流程**：

- **第一阶段（正向训练）**：用标注的已知类数据 $D_l^{kno}$ 以交叉熵正常训练 $K$ 类分类器 $f_p(\cdot)$（positive classifier）
- **第二阶段（负向微调）**：替换最后一层分类头，用 Negative Learning 损失微调得到 $f_n(\cdot)$（negative classifier）

**Negative Learning 损失函数**：

$$\ell_{NL}(f, \bar{y}) = -\sum_{k=1}^{K} \bar{y}_k \log(1 - p_k)$$

**随机标签分配策略**：

- 对已标注的已知类样本：从 $\mathcal{Y} \setminus y^l$（排除真实标签）中均匀采样互补标签
- 对未标注样本：从 $\mathcal{Y}$（全部类别）中均匀随机采样标签
- 每个训练 iteration 重新采样一次标签

**为什么 RLNL 有效？**

- 未标注的**已知类样本**有 $1/K$ 概率被分到正确标签，此时会受到较大惩罚被推向低置信区域；同时它们在特征空间中与已标注数据有重叠，受到先验知识的隐式约束
- 未标注的**未知类样本**永远不会被分到正确标签（因为其真实类别不在 $\mathcal{Y}$ 中），在批量梯度更新下会振荡到远离决策边界的高置信区域
- t-SNE 可视化实验验证：RLNL 后未知类样本特征明显远离决策边界，已知类样本保持在标注数据附近

### 2. Bidirectional Uncertainty (BU) 采样策略

由于负向分类器 $f_n(\cdot)$ 训练不稳定，预测会在 epoch 间振荡，因此：

- 每隔 $m$ 个 epoch 采集一次 $f_n$ 的预测，共采集 $t$ 次后取平均 $\mathcal{P}^-$
- 同时用 $f_p$ 获取正向预测 $p^+$

**双向不确定性采样公式**：

$$x^* = \arg\max_x \; p_{K+1}^{aux}(x) \cdot unc_n + r \cdot [1 - p_{K+1}^{aux}(x)] \cdot unc_p$$

其中：

- $unc_p$：正向分类器的不确定性（对已知类样本更准确）
- $unc_n$：负向分类器的不确定性（对区分未知类更有效）
- $p_{K+1}^{aux}$：**局部平衡因子**，来自辅助 $K+1$ 类分类器，值越大说明样本越可能属于未知类
- $r$：**全局平衡因子**，即上一轮查询中已知类样本的比例，反映当前数据池的开放程度

**自适应退化**：当没有未知类样本时，$r=1, p_{K+1}^{aux}=0$，公式退化为标准不确定性采样。

### 3. 三种具体实例化

- **B-LC (Bidirectional Least Confident)**：基于最大预测概率的互补
- **B-Margin**：基于前两大预测概率之差
- **B-Entropy**：基于预测熵

## 实验关键数据

**数据集**：CIFAR-10、CIFAR-100、Tiny-ImageNet，openness ratio 设置为 0.2/0.4/0.6/0.8

**主要结论**（最终轮平均准确率）：

| 方法 | CIFAR-10 (0.6) | CIFAR-100 (0.6) | Tiny-ImageNet (0.6) |
|---|---|---|---|
| Random | 87.2 | 58.7 | 50.9 |
| Margin | 89.0 | 58.8 | 50.8 |
| LfOSA | 87.0 | 62.4 | 52.4 |
| CCAL | 88.0 | 64.7 | 50.3 |
| **B-Margin** | **92.6** | **68.3** | **55.7** |

- BUAL 在所有 openness ratio 下均最优，且对开放度变化**不敏感**
- LfOSA 虽然已知类识别率最高，但查询的样本与已标注数据高度重叠（t-SNE 验证），是模型已掌握的简单样本
- 消融实验：仅用 $unc_p$ 得 87.5，仅用 $unc_n$ 得 89.4，双向结合得 90.8，加上两个平衡因子得 92.5

## 亮点

1. **思路巧妙**：不直接识别未知类，而是通过 RLNL 将未知类"推走"，使得传统不确定性方法自然适用于开放集场景
2. **理论直觉清晰**：利用已知类样本有概率被分到正确标签而受惩罚、未知类永远不会被分到正确标签这一不对称性
3. **通用性强**：BUAL 是一个框架，可以将任意基于不确定性的 AL 方法扩展到开放集场景
4. **自适应机制**：全局（$r$）和局部（$p_{K+1}^{aux}$）平衡因子使方法在不同开放度下均稳定

## 局限与展望

1. **计算开销**：需要训练三个分类器（$f_p, f_n, f_{aux}$），RLNL 阶段还需要多次采集预测取平均
2. **子集采样**：$D_{sub}$ 的随机采样可能引入偏差，尤其在类别极度不平衡时
3. **仅验证图像分类**：未涉及 NLP、检测、分割等任务的开放集主动学习
4. **Negative Learning 的收敛性**：论文承认 $f_n$ 的预测不稳定需要多次平均，未给出收敛保证
5. **openness ratio 已知的假设**：虽然通过 $r$ 自适应估计，但初始轮的估计可能不准确

## 与相关工作的对比

| 方法 | 策略类型 | 核心思路 | 对开放度的鲁棒性 |
|---|---|---|---|
| LC / Margin / Entropy | 不确定性 | 选低置信样本 | 高开放度时失效 |
| Coreset / BADGE | 多样性/混合 | 选分布代表性样本 | 未知类特征差异大导致误选 |
| CCAL | 对比学习 | 选语义上像已知类的样本 | 低开放度时不如随机 |
| LfOSA | MAV 建模 | 选最大激活值高的样本 | 选到的是简单样本 |
| DIAS | 开集识别 | 先识别未知类再过滤 | 标注数据少时识别能力差 |
| **BUAL** | 双向不确定性 | 推开未知类 + 双向采样 | **各开放度下均稳定** |

## 启发与关联

- Negative Learning 在噪声标签学习中已有应用，本文将其创造性地迁移到开放集主动学习中，这种**跨任务方法迁移**的思路值得借鉴
- 双平衡因子的设计（全局 + 局部）是一种通用的自适应策略，可推广到其他需要平衡多目标的采样问题
- 与 out-of-distribution detection 领域有潜在联系：RLNL 实质上是一种无监督的 OOD 信号增强方法

## 评分
- 新颖性: ⭐⭐⭐⭐ — RLNL 思路新颖，将 negative learning 用于推开未知类是创新性贡献
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集 × 四种开放度，消融完整，可视化分析到位
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图示直观，整体逻辑流畅
- 价值: ⭐⭐⭐⭐ — 提供了一个实用的框架将封闭集 AL 方法扩展到开放集场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](../../CVPR2025/others/rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ECCV 2024\] FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)
- [\[CVPR 2025\] Open Set Label Shift with Test Time Out-of-Distribution Reference](../../CVPR2025/others/open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [\[ECCV 2024\] DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation](dc-solver_improving_predictor-corrector_diffusion_sampler_via_dynamic_compensati.md)

</div>

<!-- RELATED:END -->
