---
title: >-
  [论文解读] Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments
description: >-
  [ACL 2025 (Industry Track)][自动作文评分] 本文提出了一种基于核加权序数分类交叉熵（KWOCCE）的置信度建模方法，通过利用 CEFR 等级的序数结构和分数分箱策略，实现最高 47% 评分在 100% CEFR 一致性下释放，99% 在 ≥95% 一致性下释放，显著优于无置信度过滤时的约 92%。
tags:
  - "ACL 2025 (Industry Track)"
  - "自动作文评分"
  - "置信度建模"
  - "序数分类"
  - "KWOCCE 损失"
  - "CEFR"
---

# Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments

**会议**: ACL 2025 (Industry Track)  
**arXiv**: [2505.23315](https://arxiv.org/abs/2505.23315)  
**代码**: 无（使用专有数据集）  
**领域**: 其他  
**关键词**: 自动作文评分, 置信度建模, 序数分类, KWOCCE 损失, CEFR

## 一句话总结

本文提出了一种基于核加权序数分类交叉熵（KWOCCE）的置信度建模方法，通过利用 CEFR 等级的序数结构和分数分箱策略，实现最高 47% 评分在 100% CEFR 一致性下释放，99% 在 ≥95% 一致性下释放，显著优于无置信度过滤时的约 92%。

## 研究背景与动机

自动作文评分（AES）系统在大规模考试中越来越普遍，但在高风险场景下，确保评分可靠性至关重要。核心问题在于：

**何时该信任自动评分？** 现有 AES 系统对所有预测统一释放，但不同分数区间的可靠性差异很大

**CEFR 等级的序数性被忽略**：CEFR 从 A1 到 C2 是有序的，预测 B1 误为 B2 和误为 C2 的严重程度完全不同，但标准分类损失无法区分

**置信度建模不足**：大多数方法简单地依赖 softmax 概率或预测区间，缺乏对序数结构的建模

**分数边界敏感性**：在等级边界附近的误差对考试结果影响最大（如从 B1 变为 B2 可能改变求职/留学资格）

本文的动机就是将序数分类思想引入 AES 置信度建模，开发一个混合评分系统（HMS），在自动评分与人工审阅之间取得更优的平衡。

## 方法详解

### 整体框架

提出的混合评分系统（HMS）由两部分组成：
- **自动评分器（AM）**：基于 Transformer 编码器的回归模型，输出分数和 LLM embedding
- **置信度模型**：下游分类器，接收 AM 的 embedding、预测分数和 CEFR 切分点，输出 0-1 置信度分数

低置信度的评分被转交人工审阅，高置信度的直接释放。

### 关键设计

1. **二分类基线（Binary Classification）**：

    - 最简单的框架：判断 AM 分数是否正确对应 CEFR 等级
    - 使用交叉熵（CE）损失，最终概率作为置信度
    - 问题：缺乏对不确定性的精细估计

2. **CEFR 级别 N-ary 分类**：

    - 模型输出所有 CEFR 等级的概率分布
    - 使用分类交叉熵（CCE）损失
    - 置信度 = AM 预测 CEFR 对应的概率值
    - 优势：能捕捉竞争 CEFR 概率的情况

3. **分数级别分箱 N-ary 分类（核心创新）**：

    - 将连续分数离散化为独立类别，再根据 CEFR 切分点对同一等级内的分数概率求和
    - 更细粒度地建模 AM 在不同分数区间的可靠性差异
    - 这一步带来了巨大的性能提升（F1 从 0.733 → 0.954）

4. **KWOCCE 损失函数族（核心贡献）**：

    - 在标准 CCE 基础上引入基于序数距离的核函数惩罚
    - 四种核函数：
        - **线性核**：$K_{linear}(x,N) = \max(0, 1 - |x|/N)$
        - **对数核**：$K_{log}(x,N;\alpha) = \max(0, 1 - \alpha\log(1+|x|)/\log(N))$
        - **指数核**：基于 sigmoid 形式，小误差几乎不惩罚，大误差指数级惩罚
        - **高斯核**：$K_{gaussian}(x;\alpha) = \exp(-(x/\alpha)^2)$，钟形惩罚
    - 核心公式：$\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{N} w_i \log \hat{y}_{ic_i}$

### 损失函数 / 训练策略

- 基线 Keras OCC 损失：$\text{loss} = (w + 1) \cdot \text{CE}$，权重 $w = |argmax \mathbf{y} - argmax \hat{\mathbf{y}}| / (K-1)$
- KWOCCE 通过不同核函数替换线性权重，实现非线性距离感知惩罚
- 使用均值缩减（mean reduction）确保梯度稳定
- 训练集 23 万响应，验证集 5.8 万，评估集 644 响应（322 名考生）

## 实验关键数据

### 主实验：架构比较（表格）

| 分类器类型 | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 二分类 | 0.578 | 0.579 | 0.997 | 0.733 |
| CEFR N-ary | 0.642 | 0.693 | 0.869 | 0.772 |
| 分数分箱 N-ary | 0.913 | 0.913 | 1.000 | **0.954** |

> 粒度越细，分类性能越好。分数分箱 N-ary 带来了质的飞跃。

### 消融实验：不同损失函数在评分释放上的表现（表格）

| 损失函数 | 100% CEFR 一致 (% 释放) | 95% CEFR 一致 (% 释放) |
|---------|----------------------|---------------------|
| CCE 基线 | 29.80% | 91.83% |
| Keras OCC | 36.31% | 91.97% |
| KWOCCE Linear | **47.35%** | 98.16% |
| KWOCCE Log (α=3) | 19.86% | **98.89%** |
| KWOCCE Exp (α=1,β=3) | 41.01% | 99.12% |
| KWOCCE Gaussian (α=0.5) | 35.73% | 98.75% |

> KWOCCE Linear 在 100% 一致性下释放比例最高（47.35%），KWOCCE Exp 在 95% 一致性下接近 99%。

### 关键发现

1. **粒度越细效果越好**：从二分类到 CEFR N-ary 再到分数分箱，性能单调提升
2. **序数结构关键**：所有 KWOCCE 变体在 95% CEFR 一致性下显著优于标准 CCE
3. **不同核函数适用不同场景**：Linear 适合追求高覆盖率，Exp 适合追求高一致性
4. **实际意义重大**：从无置信度过滤的 92% 一致性提升到有条件释放下的 99%+
5. **与原始性能对比**：AM 本身 RMSE 为 1.095，所有带置信度模型在释放子集上 RMSE 均降低

## 亮点与洞察

- **工业实用价值极高**：直接解决了高风险考试中 AES 评分可信度的核心痛点
- **KWOCCE 设计优雅**：用统一的核函数框架涵盖了线性、对数、指数、高斯多种惩罚方案，可灵活适配不同场景
- **"置信度即可释放性"**：将置信度建模转化为直接关联考试公平性的指标（CEFR 一致率 × 释放比例），而非抽象的校准度
- **粒度洞察**：证明了更细粒度的分类目标能带来更好的置信度估计，对不确定性建模有启发意义

## 局限与展望

1. **单一考试数据**：仅在一个特定考试上验证，泛化性未知
2. **评估集较小**：644 响应在统计功效上可能不足
3. **专有数据**：数据和 CEFR 切分标准不可公开，难以复现
4. **未做人类评估**：没有将置信度决策与人工审阅者对比
5. **核函数超参调优**：不同核的 α、β 需要调优，最优配置可能因考试而异
6. **仅测试 Transformer 编码器 AM**：未验证在 LLM-based AM 上的效果

## 相关工作与启发

- **Castagnos et al. (2022)** 的对数序数损失是 KWOCCE 的重要启发，但仅限于单一对数形式
- **Polat et al. (2025)** 的类距离加权交叉熵用于医学严重度分类，思路类似但未引入核函数泛化
- 本文启发：序数感知的损失函数设计在任何有等级结构的评估/评分任务中都有应用价值，如情感强度、疾病严重度、产品质量分级等

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 3.5 | KWOCCE 是已有方法的自然扩展，核函数框架有创新 |
| 实验充分度 | 3.5 | 消融充分但数据集单一，评估集偏小 |
| 写作质量 | 4 | Industry Track 风格，清晰实用 |
| 价值 | 4 | 工业应用价值高，直接解决实际部署问题 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](dress_dataset_rubric_based_essay_scoring_efl_writing.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ECCV 2024\] Enhancing Optimization Robustness in 1-bit Neural Networks through Stochastic Sign Descent](../../ECCV2024/others/enhancing_optimization_robustness_in_1-bit_neural_networks_through_stochastic_si.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](../../CVPR2025/others/improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)

</div>

<!-- RELATED:END -->
