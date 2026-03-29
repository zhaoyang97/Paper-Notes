# Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric

**会议**: ACL 2025
**arXiv**: [2502.17184](https://arxiv.org/abs/2502.17184)
**代码**: https://github.com/UmeanNever/NovelSum
**领域**: LLM Alignment / 数据工程
**关键词**: Data Diversity, Instruction Tuning, NovelSum, Data Selection, Metric Design

## 一句话总结
系统分析 11 种现有多样性度量方法的局限性，提出 NovelSum——一种同时考虑样本间差异和信息密度的数据多样性指标，与指令微调性能达到 0.97 相关性。

## 研究背景与动机
1. **领域现状**：数据多样性对指令微调至关重要，各种 diversity-aware 数据选择方法不断涌现。
2. **现有痛点**：精确定义和度量数据多样性的基本问题未被充分探索，导致数据工程成为黑盒过程。
3. **核心矛盾**：现有多样性度量各有偏好，但无法同时捕捉样本差异性和信息空间密度。
4. **本文要解决什么？** 提供一个可靠的多样性度量指标，强关联于微调模型性能。
5. **切入角度**：大规模实验验证 11 种指标与模型性能的相关性，找出失败原因并设计新指标。
6. **核心idea一句话**：数据集多样性 = 每个样本的"新颖性"之和，新颖性由邻近加权的密度感知距离定义。

## 方法详解

### 整体框架
(1) 用多种数据选择策略构建 53 个 IT 数据集 → (2) 用 11 种指标度量多样性 → (3) 微调并评测 → (4) 分析相关性 → (5) 提出 NovelSum。

### 关键设计
1. **邻近加权求和 (Proximity-Weighted Sum)**:
   - 做什么：计算样本的独特性得分
   - 核心思路：对每个样本，按距离排序的邻居赋予递减权重 $w(x_i, x_j) = 1/\pi_i(j)$，近邻影响大于远点
   - 设计动机：DistSum 被远点主导，KNN 只看最近邻，邻近加权取平衡

2. **密度感知距离 (Density-Aware Distance)**:
   - 做什么：在语义距离基础上引入局部密度因子
   - 核心思路：$\Delta(x_i, x_j) = \sigma(x_j)^\beta \cdot d(x_i, x_j)$，高密度区域（如数学/代码）的距离被放大
   - 设计动机：语义相近的数学样本可能包含大量不同信息，纯语义距离低估了高密度区的多样性

3. **NovelSelect 数据选择策略**:
   - 做什么：基于 NovelSum 的贪心数据选择
   - 核心思路：迭代选择使 NovelSum 增量最大的样本
   - 设计动机：将指标直接转化为可操作的选择策略

### 损失函数 / 训练策略
全量数据集 = WizardLM + ShareGPT + UltraChat，固定选取 10,000 样本。使用标准 SFT 训练 LLaMA-3-8B 和 Qwen-2.5-7B。

## 实验关键数据

### 主实验（LLaMA-3-8B，相关性对比）

| 指标 | Pearson r | Spearman r | 平均 r |
|------|----------|-----------|-------|
| NovelSum | **0.98** | **0.95** | **0.97** |
| Vendi Score | 0.61 | 0.64 | 0.63 |
| DistSum_cosine | 0.74 | 0.69 | 0.72 |
| Facility Location | 0.52 | 0.48 | 0.50 |
| KNN Distance | 0.71 | 0.66 | 0.69 |
| Partition Entropy | 0.55 | 0.51 | 0.53 |

### 消融实验

| 组件 | 效果 |
|------|------|
| 去掉 Proximity Weight | 类似 DistSum，被远点主导，r 下降 ~0.25 |
| 去掉 Density-Aware | 高密度区域多样性被低估，r 下降 ~0.15 |
| α 超参数 | α=1 表现最优 |
| β 超参数 | β=0.5 最佳 |

### 关键发现
- 词汇多样性指标（TTR, vocd-D）与 IT 性能几乎无关
- 基于距离的指标忽略信息密度，高估离群数据集
- 基于分布的指标忽略样本独特性，低估高距离数据集
- NovelSum 在两个不同骨干模型上均保持强相关性

## 亮点与洞察
- "论文新颖性"的类比非常直觉——样本的新颖性取决于它与同领域相关工作的差异
- 仿真实验巧妙地可视化了不同指标的行为差异
- NovelSelect 证明了指标到策略的直接转化可行

## 局限性 / 可改进方向
- 仅在通用 IT 场景验证，领域特化（如医学/代码）场景未测
- 嵌入模型的选择可能影响结果
- 计算复杂度随数据量增长，大规模数据需要近似算法

## 相关工作与启发
- **vs QDIT (Bukharin et al. 2024)**: QDIT 优化 Facility Location，NovelSum 证明 FL 忽略样本独特性
- **vs Repr Filter (Liu et al. 2023)**: Repr Filter 基于 KNN 阈值，NovelSum 的密度感知方法更精确


## 补充细节
- 数据源：WizardLM + ShareGPT + UltraChat，固定选 10000 样本
- 嵌入模型：BERT 用于提取样本语义表示
- 评测基准：MT-bench 和 AlpacaEval，聚合为 Z-score
- NovelSum 超参数 alpha 和 beta 分别控制邻近权重衰减和密度影响
- NovelSelect 贪心算法：迭代选择使 NovelSum 增量最大的样本
- 在 Qwen-2.5-7B 上也验证了 NovelSum 的高相关性
- 代码开源且包括所有 53 个数据集的构建脚本
- K-Center-Greedy 和 Repr Filter 数据集在距离指标上高但实际效果不一定好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统分析多样性指标与 IT 性能的相关性，NovelSum 方法论新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 53 个数据集 x 2 个模型 x 11 种指标，极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，发现-设计-验证的叙事线完整
- 价值: ⭐⭐⭐⭐⭐ 为指令微调数据工程提供了可量化的多样性指导
