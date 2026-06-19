---
title: >-
  [论文解读] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation
description: >-
  [NeurIPS 2025][社会计算][图增量学习] 提出 GraphKeeper 框架应对**图域增量学习（Graph Domain-IL）**中的灾难性遗忘，通过域特异性 LoRA 参数隔离 + 领域内/间解耦 + 基于岭回归的无偏差知识保存三组件，比次优方法提升 6.5%-16.6%，且可无缝集成图基础模型。
tags:
  - "NeurIPS 2025"
  - "社会计算"
  - "图增量学习"
  - "域增量学习"
  - "LoRA"
  - "知识解耦"
  - "灾难性遗忘"
---

# GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation

**会议**: NeurIPS 2025  
**arXiv**: [2511.00097](https://arxiv.org/abs/2511.00097)  
**代码**: [GitHub](https://github.com/RingBDStack/GraphKeeper)  
**领域**: 社会计算  
**关键词**: 图增量学习, 域增量学习, LoRA, 知识解耦, 灾难性遗忘

## 一句话总结

提出 GraphKeeper 框架应对**图域增量学习（Graph Domain-IL）**中的灾难性遗忘，通过域特异性 LoRA 参数隔离 + 领域内/间解耦 + 基于岭回归的无偏差知识保存三组件，比次优方法提升 6.5%-16.6%，且可无缝集成图基础模型。

## 研究背景与动机

图增量学习（GIL）要求模型在新图数据到来时持续更新。现有方法聚焦于**任务增量（Task-IL）**和**类增量（Class-IL）**设定，均在单个图域内操作。但随着图基础模型（GFM）的兴起，模型需要整合来自**多个不同域**的图数据，即**域增量学习（Domain-IL）**设定。

**Domain-IL 的独特挑战**：

**嵌入偏移（Embedding Shifts）**：学习新域需要较大的参数变化，导致旧域图的嵌入发生偏移

**决策边界偏离（Decision Boundary Deviations）**：端到端训练中分类器随嵌入模型一起更新，旧域决策边界被破坏

实验验证：代表性 GIL 方法 SSM 在 Class-IL 中表现良好，但在 Domain-IL 中严重失效。跨域的结构和语义差异远大于同域内的类别差异，使现有方法无力应对。

## 方法详解

### 整体框架

GraphKeeper 三大模块针对灾难性遗忘的两个成因分别设计：

1. **多域图解耦（Multi-domain Graph Disentanglement）**：防止嵌入偏移和混淆
2. **无偏差知识保存（Deviation-Free Knowledge Preservation）**：维持稳定决策边界
3. **域感知分布判别（Domain-aware Distribution Discrimination）**：测试时匹配未知域的图

### 关键设计

**模块 1：多域图解耦**

*多域特征对齐*：不同域的图特征维度不同，先通过截断 SVD 统一投影到 $\bar{d}$ 维空间：
$$\tilde{F}_i = \text{Proj}(F_i), \quad \tilde{F}_i \in \mathbb{R}^{|G_i| \times \bar{d}}$$

*域特异性 LoRA*：在预训练 GNN 上为每个图域配备独立的 LoRA 模块：
$$h^l = \xi^l(h^{l-1}, W_i^l) + \phi_i^l(h^{l-1}, W_{i,\text{down}}^l W_{i,\text{up}}^l)$$
学习新域时冻结旧域的 LoRA 参数，从结构上保证旧域嵌入不偏移。

*域内解耦*：基于对比学习增强同域内不同类的判别性：
$$\mathcal{L}_{\text{intra}} = -\sum_{j=1}^{|G_i|} \log \frac{\sum_{o \in S_j^{\text{pos}}} \exp(\text{sim}(x_j, x_o^{\text{aug}}))}{\sum_{o' \in S_j^{\text{pos}} \cup S_j^{\text{neg}}} \exp(\text{sim}(x_j, x_{o'}^{\text{aug}}))}$$
其中 $S^{\text{pos}}$ 为同类节点，$S^{\text{neg}}$ 为异类节点，$x^{\text{aug}}$ 来自增强视图。

*域间解耦*：将当前域节点推离旧域的嵌入原型（通过聚类获得）：
$$\mathcal{L}_{\text{inter}} = \frac{1}{|G_i|} \sum_{j=1}^{|G_i|} \sum_{k=1}^{|P|} \frac{1}{\|x_j - P_k\|_2^2 + \epsilon}$$
最小化此目标使不同域在嵌入空间中充分分离。

**模块 2：无偏差知识保存**

关键思想：**将分类器从嵌入模型中分离**，用岭回归闭式解替代梯度更新，避免反向传播导致的决策边界偏离。

第 $i$ 个增量域的最优权重：
$$W_i = (X_{(1:i)}^T X_{(1:i)} + \lambda I)^{-1} X_{(1:i)}^T Y_{(1:i)}$$

但历史数据不可访问，因此使用**递归更新**：
$$W_i = [W_{i-1} - M_i X_i^T X_i W_{i-1} \| M_i X_i^T Y_i]$$
$$M_i = M_{i-1} - M_{i-1} X_i^T (I + X_i M_{i-1} X_i^T)^{-1} X_i M_{i-1}$$

这保证了与全数据闭式解**完全等价**的精确更新，无需存储任何历史图数据。

**模块 3：域感知分布判别**

测试图的域未知时，需要匹配到正确的域特异性 LoRA 模块。步骤：
1. 用随机初始化且冻结的 GNN 将特征映射到高维空间（将相似域的原型分开）
2. 通过最近原型匹配确定测试图所属域：
$$c_{\text{test}} = \arg\max_k \exp(-\|D_{\text{test}} - D_k\|_2^2)$$

### 损失函数 / 训练策略

总体优化目标：
$$\mathcal{L} = \gamma_1 \mathcal{L}_{\text{intra}} + \gamma_2 \mathcal{L}_{\text{inter}}$$

注意：决策模块（岭回归）不通过 $\mathcal{L}$ 反向传播更新，而是在嵌入学习完成后通过闭式解直接计算。这种解耦设计是防止决策边界偏离的关键。

## 实验关键数据

### 主实验

**Domain-IL 场景下6组域序列的平均结果**：

| 方法 | Group 1 AA↑ | Group 3 AA↑ | Group 5 AA↑ |
|------|------------|------------|------------|
| Fine-Tune | 23.9 | 20.9 | 19.7 |
| Joint（上界） | 66.6 | 78.0 | 74.5 |
| EWC | 23.3 | 20.8 | 20.6 |
| ER-GNN | 23.3 | 28.7 | 24.8 |
| DeLoMe | 49.3 | 70.2 | 63.2 |
| PDGNNs | 52.4 | 65.5 | 64.3 |
| TPP | 52.6 | 57.1 | 56.7 |
| **GraphKeeper** | **69.2** | **80.6** | **75.5** |

GraphKeeper 比次优方法提升 6.5%-16.6%，且**超越 Joint 基线**（后者可访问所有历史数据）。

**与图基础模型集成**（few-shot Domain-IL）：

| 方法 | Group 1 AA↑ | Group 3 AA↑ |
|------|------------|------------|
| GCOPE（原始） | 20.6 | 13.2 |
| GCOPE + GraphKeeper | **显著提升** | **显著提升** |
| MDGPT（原始） | 低 AA / 高 AF | 低 AA / 高 AF |
| MDGPT + GraphKeeper | **高 AA / 低 AF** | **高 AA / 低 AF** |

### 消融实验

各模块的消融（从论文分析推断）：
- 移除域间解耦→域嵌入混淆加剧，性能显著下降
- 移除域特异性 LoRA→嵌入偏移不可控，退化严重
- 岭回归替换为梯度更新分类器→决策边界偏离，遗忘加剧
- 移除高维随机映射→域原型混淆，测试时域匹配错误率上升

### 关键发现

1. **现有 GIL 方法在 Domain-IL 下全面失效**：EWC、GEM、LWF 等方法与 Fine-Tune 差距极小
2. **GraphKeeper 超越 Joint 上界**：说明单个 GNN 难以有效融合多域知识，参数隔离是必要的
3. **与 GFM 无缝集成**：为预训练的图基础模型赋予持续更新能力，同时保持其 few-shot 优势
4. **遗忘几乎为零**：AF 指标接近 0，远优于所有基线（含使用记忆回放的方法）
5. DeLoMe/PDGNNs 的相对较好表现依赖于 SGC/APPNP 骨架（牺牲可塑性换稳定性），换成 GCN 性能显著下降

## 亮点与洞察

- **问题定义新颖**：首次系统研究图域增量学习（Domain-IL），区别于传统的 Task-IL/Class-IL
- **解耦思想透彻**：从嵌入偏移和决策边界偏离两个正交维度分析遗忘原因，设计针对性方案
- **岭回归闭式解的巧妙应用**：递归更新公式保证精确性，无需存储历史数据，O(1) 空间复杂度
- **域判别的高维随机映射**：简单但有效，利用随机投影分离域原型
- **与 GFM 的结合前景**：为图基础模型的增量更新提供了可行路径

## 局限与展望

1. 每个新域需要一个 LoRA 模块，域数量增长时存储和推理开销线性增加
2. 域原型通过聚类获得，聚类质量影响域间解耦效果
3. 域感知判别依赖原型距离，当域分布有大量重叠时可能失效
4. 特征对齐使用截断 SVD，信息损失不可避免
5. 仅在节点分类任务上验证，未涉及图级别任务（图分类、链接预测）

## 相关工作与启发

- **图增量学习**：SSM、ER-GNN 等代表方法，本文揭示了它们在 Domain-IL 下的失效
- **多域图预训练**：GCOPE、MDGPT 等 GFM，GraphKeeper 为其赋予增量学习能力
- **LoRA 用于图学习**：域特异性 LoRA 的设计借鉴了 NLP 领域的持续学习方法
- **启发**：递归岭回归的知识保存机制可能适用于其他需要闭式更新的增量学习场景（如推荐系统的增量更新）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次定义并系统解决图域增量学习问题
- **技术深度**: ⭐⭐⭐⭐ — 三模块设计各有理论支撑，递归更新公式推导严谨
- **实验充分度**: ⭐⭐⭐⭐ — 15个数据集、多组域序列、多基线对比、GFM 集成实验
- **实用性**: ⭐⭐⭐⭐ — 可直接集成到现有 GFM，代码已开源
- **总体**: ⭐⭐⭐⭐

## 背景与动机

## 核心问题

## 方法详解

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[CVPR 2025\] Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[AAAI 2026\] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](../../AAAI2026/social_computing/from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](../../ACL2026/social_computing/bayesian_social_deduction_with_graph-informed_language_models.md)

</div>

<!-- RELATED:END -->
