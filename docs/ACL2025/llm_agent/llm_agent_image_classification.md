# Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models

**会议**: ACL 2025  
**arXiv**: [2506.01334](https://arxiv.org/abs/2506.01334)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: Concept Bottleneck Models, LLM Agent, Interpretable Classification, Dynamic Concept Bank, CLIP  

## 一句话总结

提出 Conditional Concept Bottleneck Models (CoCoBMs) 和 LLM-driven Concept Agent 框架，通过类别条件化的概念评分机制和基于环境反馈的动态概念库优化，在 6 个数据集上提升分类准确率 6% 的同时将可解释性提升约 30%。

## 研究背景与动机

Concept Bottleneck Models (CBMs) 将图像分类分解为通过可解释的人类可读概念进行决策的过程，是可解释深度学习的代表性方法。近年来基于 CLIP 的 CBMs 利用 LLM 生成候选概念，消除了手动构建概念库和标注的需要。然而，三个关键问题仍未解决：

1. **概念数量不确定**：最优概念数量是什么？LaBo 为 CUB 数据集使用了 10,000 个概念，而 LM4CV 仅使用 32 个概念却取得了类似性能。过多导致冗余，过少导致覆盖不足——两个数量级的差异说明当前方法缺乏系统的概念数量优化机制。
2. **共享评分机制的局限**：传统 CBMs 对所有类别使用相同的概念评分（shared scoring），但同一个概念对不同类别的贡献可能截然不同（如"红色羽毛"对"红衣主教鸟"比对"鸽子"的辨识价值高得多）。
3. **概念编辑的局限**：现有方法仅在测试时允许人类手动编辑概念分数来纠正错误，无法规模化且未利用 LLM 的事实知识自动纠正激活错误的概念。

## 方法详解

### 整体框架

框架包含两个核心创新：

1. **CoCoBMs**：使用类别条件化的概念评分和权重机制
2. **Concept Agent**：一个具有记忆、规划和动作模块的 LLM Agent，通过环境反馈动态优化概念库

### 关键设计

#### 1. Conditional Concept Bottleneck Models (CoCoBMs)

**问题分析**：传统 CBMs 的概念评分为 $\vec{s_c} = P(\vec{s_c} | x_i, \mathcal{C})$，所有类别共享同一组概念分数。CoCoBMs 引入类别条件化评分：

$$\vec{s_c^j} = \|_{k=1}^{M} P(s_{c_k}^j | x_i, y_j, c_k)$$

每个概念 $c_k$ 的分数依赖于假想标签 $y_j$，形成 $R^{N \times M}$ 的概念矩阵而非共享的概念向量。

**Condition Learning**：采用 prompt learning 策略，将可学习的条件提示附加到文本输入中：

$$p_k^j = [t_1][t_2] \ldots [t_q][y_j][c_k]$$

其中 $[y_j]$ 和 $[c_k]$ 分别是类别名和概念名的 token，$t_i$ 是与 CLIP word embedding 同维度的可学习向量，所有标签和概念共享这些 token 以防止信息泄露。

**Editable Matrix**：引入可编辑矩阵 $E$ 来约束与事实知识冲突的概念激活：

$$E_{jk} = \begin{cases} 1, & \text{if } c_k \notin y_j \\ 0, & \text{if } c_k \in y_j \end{cases}$$

当 $E_{jk}=1$（概念与类别不相容）时，强制 $s_{c_k}^j = \min(s_{c_k}^j, 0)$，将概念分数截断为非正值。

#### 2. Concept Agent

Agent 包含三个模块：

**Memory Module**：
- 维护生成概念列表 $M_g$、删除概念列表 $M_d$、事实验证的概念-标签对 $M_f$
- 存储每次迭代后更新的概念库

**Action Module**：
- **Concept Generation**：用 LLM 对每个类别生成候选概念，prompt 模板为"What are the helpful visual features to distinguish [CLS] from other [S-CLS]?"
- **Concept Selection**：采用 learning-to-search 方法从候选池中选择固定数量的概念
- **Fact Verification**：用 LLM 通过多选题判断每个概念-标签对的相关性（critical/occasionally/unrelated）
- **Instance Selection**：通过 K-Means 聚类选择代表性样本作为 few-shot 环境
- **Environment Perception**：用 CoCoBMs 作为工具与环境交互，获取验证集反馈

**Planning Module**：
- 分析概念的 Score Activation Pattern：对每个概念计算在验证集上的标准化贡献分数
- 计算二值激活模式 $P_{act}^c = [a_1^c, \ldots, a_N^c]$，其中 $a_j^c = 1$ 当 $\bar{s}_c^j > t_a$
- **冗余概念检测**：(1) 不贡献于任何标签的概念；(2) 与其他概念具有相同激活模式且不如对方有效的概念
- **不足概念检测**：(1) 无概念被激活的标签；(2) 与其他标签共享完全相同概念集的标签
- 将不足信息反馈给 Action Module 指导新概念生成

### 损失函数 / 训练策略

CoCoBMs 使用加权二分类交叉熵损失：

$$-\frac{1}{N} \sum_{j=1}^{N} \left[W_p y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\right]$$

其中正类权重 $W_p = N$ 用于补偿每个样本内的标签不平衡。

训练策略：Agent 迭代地精炼概念库直到所有标签都可被可靠识别且冗余消除后，在完整数据集上训练 CoCoBMs 获得最终性能。

## 实验关键数据

### 主实验

**6 个数据集上的分类准确率对比（CLIP ViT-B/32 backbone）**：

本方法确定的概念数量约等于标签数量，与 LaBo-n、LM4CV-n 公平对比：

- 对比 LaBo-n：平均提升 **6.15%**
- 对比 LaBo-3n（3 倍概念数）：仍高 **0.51%**
- 对比 LM4CV-n：提升 **5.97%**
- 对比 LM4CV-2n：提升 **3.21%**
- 对比 LF-CBM（如 CIFAR-10 使用 16 倍概念数）：高 **1.36%**
- 与 black-box 模型差距：对比 linear probing 仅差 3.45%，对比 prompt learning 仅差 2.44%

**可解释性评估**：
- 总体可解释性得分：**77.46%**
- 真实性（Truthfulness）：**81.59%**
- 可区分性（Distinguishability）：**73.34%**
- 较 LM4CV-2n 提升约 **30%**

### 关键发现

1. **动态 vs 静态 grounding**（Table 1）：
   - CIFAR-100：准确率 72.67% → 74.63%；可解释性 67.90% → **79.30%**
   - Flower：准确率 87.45% → 89.51%；可解释性 70.59% → **82.35%**
   - 动态 grounding 平均可解释性提升 **10.76%**

2. **Editable Matrix 的作用**（Table 2）：
   - 移除 E 后 CIFAR-100 可解释性从 79.30% 骤降至 **39.60%**
   - Flower 从 82.35% 降至 **35.59%**
   - E 轻微牺牲准确率（~0.1-2%）但大幅提升可解释性，说明事实约束对可解释性至关重要

3. **Few-shot 环境中的鲁棒性**：随样本数增加准确率提升，但可解释性保持稳定，证明模型在有限数据下仍能保持良好的可解释性。

4. **Case Study（CIFAR-10）**：Agent 经 4 次迭代后最终概念库仅含 9 个概念。过程中发现了由数据集偏差或 CLIP 预训练偏差导致的无效概念，以及因相同激活模式被合并的冗余概念。

## 亮点与洞察

1. **类别条件化评分**：打破了传统 CBMs 共享概念分数的假设，通过 prompt learning 优雅地实现了类别特定的概念评估，且原始 CBM 只是其特例（沿标签维度折叠即可恢复）。
2. **Agent 驱动的动态概念库**：将概念库构建从静态的一次性过程转变为基于反馈的迭代优化过程，这是概念瓶颈模型领域的首次尝试。
3. **可解释性的定量评估**：提出了基于 LLM 的 truthfulness 和 distinguishability 评估指标，填补了 CBM 领域缺乏可解释性定量评估的空白。
4. **随机词也能分类**：实验揭示了一个重要发现——512 个随机词组成的概念库也能取得不错的分类准确率，强调了可解释性评估的必要性，不能仅看准确率。

## 局限性

1. **Fact Verification 可扩展性**：所有可能的概念-类别对都需要验证，类别和概念数增大时计算成本急剧增长。
2. **LLM 内部知识的不确定性**：概念生成依赖 LLM 的知识库，其内在偏差和随机性可能影响概念质量。
3. **CLIP 依赖**：视觉感知完全依赖 CLIP 的对齐能力，CLIP 的预训练偏差可能导致某些视觉特征无法被有效检测。
4. **仅评估分类任务**：未探索在目标检测、分割等更复杂视觉任务上的扩展。

## 相关工作与启发

- **CBMs 演进**：从 Koh et al. (2020) 的原始 CBM 到 Label-free CBMs (Oikarinen et al., 2023)、LaBo (Yang et al., 2023)、LM4CV (Yan et al., 2023a)，概念库构建从手动标注到 LLM 生成不断进步。
- **LLM Agent 架构**：遵循经典的 Memory-Planning-Action 框架 (Yao et al., 2023)，将 Agent 范式引入视觉理解任务。
- **Prompt Learning**：借鉴 CoOp (Zhou et al., 2022) 的可学习提示策略，用于条件化评分。
- **启发**：将 LLM Agent 的规划能力与领域特定模型工具化结合，是一个值得推广的模式——Agent 不直接解决视觉任务，而是优化解决视觉任务的工具。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4.5 |
| 技术深度 | 4.5 |
| 实验充分性 | 4.5 |
| 实用价值 | 3.5 |
| 写作质量 | 4 |
| 总体评分 | 4.2 |
