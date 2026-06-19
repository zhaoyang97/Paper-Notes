---
title: >-
  [论文解读] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification
description: >-
  [AAAI 2026][时间序列][反事实解释] 提出 CONFETTI，一种面向多变量时间序列（MTS）分类的多目标反事实解释方法，通过结合类激活图（CAM）引导的子序列提取与 NSGA-III 多目标优化，在预测置信度、稀疏性和接近度三个目标间实现最优平衡，在 7 个 UEA 数据集上全面超越现有方法。
tags:
  - "AAAI 2026"
  - "时间序列"
  - "反事实解释"
  - "多变量时间序列"
  - "多目标优化"
  - "可解释AI"
  - "NSGA-III"
---

# Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification

**会议**: AAAI 2026  
**arXiv**: [2511.13237](https://arxiv.org/abs/2511.13237)  
**代码**: [https://github.com/serval-uni-lu/confetti](https://github.com/serval-uni-lu/confetti)  
**领域**: Time Series / Explainable AI  
**关键词**: 反事实解释, 多变量时间序列, 多目标优化, 可解释AI, NSGA-III

## 一句话总结
提出 CONFETTI，一种面向多变量时间序列（MTS）分类的多目标反事实解释方法，通过结合类激活图（CAM）引导的子序列提取与 NSGA-III 多目标优化，在预测置信度、稀疏性和接近度三个目标间实现最优平衡，在 7 个 UEA 数据集上全面超越现有方法。

## 研究背景与动机

深度学习模型（CNN、RNN、Transformer 等）在多变量时间序列分类任务上取得了优异的性能，但其"黑箱"特性严重阻碍了决策者对预测结果的理解和信任。现有 XAI 方法虽能提供部分洞察，但难以揭示完整的决策空间。反事实解释（Counterfactual Explanations, CE）通过展示"对输入做哪些最小改变可以改变预测结果"来弥补这一缺陷，但现有 MTS 反事实方法存在**核心矛盾**：

- **CoMTE / AB-CE**：聚焦于最大化预测置信度，但可能需要对原始时间序列进行大幅修改
- **SETS / LASTS**：聚焦于接近度（proximity），但可能产生分布外的实例
- **TSEvo**：虽然是多目标方法，但采用无先验的种群搜索，在高维或长时间序列上计算代价高昂且效率低

**核心 idea**：CONFETTI 引入 CAM 权重作为先验知识来指导搜索过程，通过四步流水线——找到最近异类邻居（NUN）→ 提取最重要子序列 → 初始替换生成种子 CE → NSGA-III 多目标优化——在置信度、稀疏性和接近度三个目标之间同时优化，并且通过设计保证了有效性和合理性。

## 方法详解

### 整体框架
CONFETTI 由四个阶段组成：
1. **NUN 检索**：找到与待解释实例预测类别不同的最近邻实例
2. **子序列提取**：利用 CAM 权重定位最具影响力的子序列
3. **朴素阶段（Naive Stage）**：用 NUN 对应子序列替换原始序列，生成初始 CE
4. **优化阶段**：基于 NSGA-III 多目标优化，平衡三个目标

### 关键设计

1. **NUN 检索（Nearest Unlike Neighbor）**:

    - 功能：在参考集 R 中找到与查询实例 X_i 预测类别不同的最近邻
    - 核心思路：先按类别过滤，再用 k-NN 搜索，保留分类器置信度高于阈值 θ 的候选
    - 设计动机：使用真实数据分布中的实例作为反事实目标，天然保证了生成 CE 的合理性（plausibility）

2. **CAM 引导的子序列提取**:

    - 功能：利用 NUN 的类激活图（CAM）权重，通过滑动窗口找到长度为 ℓ 的最大累积权重子序列
    - 核心思路：对 CAM 权重跨通道取平均，然后进行线性扫描找到最重要的连续时间段
    - 设计动机：将修改限定在模型最关注的区域，避免无差别地替换整个序列，从而提升稀疏性

3. **NSGA-III 多目标优化**:

    - 功能：在初始 CE 的基础上，通过进化算法进一步优化三个目标
    - 核心思路：二分搜索缩小时间窗口，用 Das-Dennis 参考点生成策略、双点交叉和位翻转变异来演化种群
    - 设计动机：既保证 CE 的有效性约束 P(f(C_j)=c) ≥ θ，又在稀疏性和接近度之间找到帕累托前沿

### 损失函数 / 训练策略

CONFETTI 的优化是一个三目标问题（非深度学习损失）：
- **m1（最大化）**：预测置信度之和，衡量反事实被目标类接受的程度
- **m2（最小化）**：归一化 Hamming 距离，衡量修改元素的比例（稀疏性）
- **m3（最小化）**：L1/L2/DTW 距离，衡量修改幅度（接近度）

约束条件：每个 CE 的目标类置信度不低于阈值 θ。用户可通过权重参数 α ∈ [0,1] 在置信度和稀疏性之间调节偏好。

## 实验关键数据

### 主实验

使用 7 个 UEA 数据集，2 个模型架构（FCN 和 ResNet），对比 CoMTE、SETS、TSEvo 三个基线。

**置信度对比**（θ=0.95 设定，均值跨模型平均）：

| 数据集 | CoMTE | SETS | CONFETTI (θ=0.95) |
|--------|-------|------|-------------------|
| AWR | 0.953 | 0.940 | **0.978** |
| BasicMotions | 0.917 | 0.487 | **0.965** |
| ERing | 0.701 | 0.766 | **0.981** |
| NATOPS | 0.755 | * | **0.976** |
| 平均 | 0.86 | - | **0.98** |

**稀疏性对比**（α=0.0 设定）：

| 数据集 | CoMTE | TSEvo | CONFETTI (α=0.0) |
|--------|-------|-------|-------------------|
| AWR | 0.731 | 0.002 | **0.926** |
| BasicMotions | 0.486 | 0.003 | **0.822** |
| Epilepsy | 0.461 | 0.011 | **0.822** |
| 平均 | 0.56 | 0.01 | **0.88** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| FCN 全指标 (α=0.5, θ=0.95) | COV=100%, VAL=1.00, CONF=0.97, SPA=0.81 | 完整方法最优权衡 |
| FCN (α=0.0, θ=0.51) | SPA=0.88, CONF=0.59 | 最大化稀疏性 |
| FCN (α=0.5, θ=0.51) | SPA=0.85, CONF=0.69 | 平衡设定 |
| 无 CAM 权重 | 跳过子序列提取和朴素阶段 | 性能下降，但仍可作为模型无关方法运行 |

### 关键发现

- CONFETTI 是唯一在所有数据集和模型上达到 100% 覆盖率和 100% 有效率的方法
- 在稀疏性上比 CoMTE 平均高出 32 个百分点，比 TSEvo 高出 87 个百分点
- θ 参数允许用户在高置信（θ=0.95）和高稀疏（θ=0.51）场景之间灵活切换
- 所有方法的 yNN 得分均为 0.99，说明生成的 CE 都具有良好的合理性

## 亮点与洞察

- **CAM 先验的引入**是本文最关键的贡献：将原本无先验的搜索空间压缩到模型最关注的子序列上，大幅提升了效率和稀疏性
- **理论保证**：Theorem 1 证明了优化阶段产生的所有 CE 的 Hamming 距离不超过初始 CE，即优化过程只会改善或维持稀疏性
- **二分搜索策略**在时间窗口长度上的应用非常巧妙，避免了穷举搜索
- α 和 θ 两个参数为不同应用场景提供了灵活性

## 局限与展望

- 依赖 CAM 提取，仅适用于包含全局平均池化的模型架构（如 FCN、ResNet），不支持任意模型
- 无 CAM 时退化为模型无关模式，但总体框架设计优雅地处理了这一情况
- 仅在 UEA 数据集上验证（最长 207 时间步），对超长时间序列的可扩展性有待验证
- 反事实解释的人类可理解性评估缺失，未进行用户研究

## 相关工作与启发

- CoMTE 是首个 MTS 反事实方法，但逐通道替换导致稀疏性差
- TSEvo 基于 NSGA-II 做无先验搜索，计算代价高且解质量不稳定
- 本文的 CAM 引导思路可以推广到其他模态（图像、文本）的反事实解释中
- NSGA-III 在模块化多目标优化中的应用值得借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](../../ICLR2026/time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)
- [\[AAAI 2026\] Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)

</div>

<!-- RELATED:END -->
