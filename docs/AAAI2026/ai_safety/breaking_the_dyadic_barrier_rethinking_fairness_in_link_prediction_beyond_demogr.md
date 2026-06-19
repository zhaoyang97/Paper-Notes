---
title: >-
  [论文解读] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity
description: >-
  [AAAI 2026][AI安全][链接预测] 本文揭示了链接预测中二元公平性（dyadic fairness）和 Demographic Parity（ΔDP）的三大根本缺陷——GNN 表达力不足、子群偏差被掩盖、对排序不敏感——并提出基于 NDKL 的排序感知公平度量和后处理算法 MORAL，在六个数据集上实现了 SOTA 的公平性-效用权衡。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "链接预测"
  - "公平性"
  - "Demographic Parity"
  - "排序公平"
  - "图神经网络"
  - "NDKL"
---

# Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity

**会议**: AAAI 2026  
**arXiv**: [2511.06568](https://arxiv.org/abs/2511.06568)  
**代码**: [joaopedromattos/MORAL](https://github.com/joaopedromattos/MORAL)  
**领域**: AI Safety / Algorithmic Fairness  
**关键词**: 链接预测, 公平性, Demographic Parity, 排序公平, 图神经网络, NDKL  

## 一句话总结

本文揭示了链接预测中二元公平性（dyadic fairness）和 Demographic Parity（ΔDP）的三大根本缺陷——GNN 表达力不足、子群偏差被掩盖、对排序不敏感——并提出基于 NDKL 的排序感知公平度量和后处理算法 MORAL，在六个数据集上实现了 SOTA 的公平性-效用权衡。

## 背景与动机

链接预测是图机器学习的核心任务之一，广泛应用于社交推荐、知识图谱补全等场景。然而，有偏的链接预测可能加剧社会不平等——例如在 LinkedIn 等职业社交网络中，偏向男性的连接推荐会导致女性/少数族裔获得的工作机会减少 33%（mcdonald2009networks），长期积累后形成"过滤气泡"和"玻璃天花板"效应。

现有公平链接预测方法普遍采用**二元公平性框架**（dyadic fairness），将节点对分为组内（intra：$E_{s\text{-}s} \cup E_{s'\text{-}s'}$）和组间（inter：$E_{s'\text{-}s}$），以 Demographic Parity（ΔDP）衡量两者的预测概率差异。然而，这种粗粒度的分组方式会掩盖子群内部的系统性偏差，且 ΔDP 作为基于分类的度量对排序顺序完全不敏感，无法捕捉曝光偏差。因此，亟需更具表达力的公平性评估框架和相应的去偏算法。

## 核心问题

1. **GNN 表达力不足影响公平性**：标准 GNN 受限于 1-WL 测试的表达力上界，对称邻域中不同敏感群体的节点会产生相似的嵌入表示，无法区分 $E_{s'\text{-}s'}$ 和 $E_{s'\text{-}s}$ 等不同类型的节点对，导致公平节点嵌入无法转化为公平的链接预测。
2. **子群偏差被 dyadic 聚合所掩盖**：将 $E_{s\text{-}s}$ 和 $E_{s'\text{-}s'}$ 合并为"组内"后，模型可能系统性地过度预测一种子群（如男-男远多于女-女），但 ΔDP 因聚合平均而无法检测到这种"公平性舞弊"（fairness gerrymandering）。
3. **ΔDP 对排序不敏感**：ΔDP 是置换不变的，一个将组内对全部排在顶部的偏差排序与一个均匀交错的公平排序可以获得相同的 ΔDP 值——曝光偏差（exposure bias）完全被忽略。

## 方法详解

### 理论贡献：两大公平性性质

**性质 1（非二元分布保持公平性）**：公平性度量应将所有敏感属性组合视为独立子群，目标是使预测的边类型分布 $\hat{\boldsymbol{\pi}}$ 与原始图的真实分布 $\boldsymbol{\pi} = (\pi_{s\text{-}s}, \pi_{s'\text{-}s}, \pi_{s'\text{-}s'})$ 尽可能一致，通过最小化 $\text{dist}(\hat{\boldsymbol{\pi}}, \boldsymbol{\pi})$ 实现。

**性质 2（排序感知性）**：公平性度量应对每种节点对类型在排序中的比例和位置敏感。具体地，应最小化 $\sum_{k=1}^{|\mathcal{C}|} \text{dist}(\hat{\boldsymbol{\pi}}_k, \boldsymbol{\pi}) \cdot \delta_k$，其中 $\hat{\boldsymbol{\pi}}_k$ 为 top-$k$ 中的子群分布，$\delta_k$ 为单调递减的曝光衰减权重。

### 公平排序度量：NDKL

论文采用 Normalized Cumulative KL-Divergence（NDKL）作为公平性度量：

$$\text{NDKL} = \frac{1}{Z} \sum_{k=1}^{|\mathcal{C}|} \frac{1}{\log_2(k+1)} D_{\text{KL}}(\hat{\boldsymbol{\pi}}_k \| \boldsymbol{\pi})$$

其中 $Z = \sum_{i=1}^{|\mathcal{C}|} \frac{1}{\log_2(i+1)}$ 为归一化项。NDKL 同时满足性质 1（基于三子群的 KL 散度）和性质 2（以 $1/\log_2(k+1)$ 作为位置权重，top 位置偏差惩罚更重）。

**定理 1**：在全局满足 demographic parity 的约束下，NDKL 的上下界为 $0 \leq \text{NDKL} \leq \max_i \log(1/\pi_i)$。

### MORAL 算法

MORAL（Multi-Output Ranking Aggregation for Link fairness）是一个轻量级后处理框架，包含两个阶段：

**阶段 1：解耦训练**。针对三种边类型分别训练独立的链接预测模型 $f_{s\text{-}s}$、$f_{s\text{-}s'}$、$f_{s'\text{-}s'}$，每个模型只在对应类型的边上训练。这种解耦策略消除了单一模型在不同群体间性能不均衡的问题。

**阶段 2：贪心排序聚合**。在推理时，维护一个边类型的累计曝光分布估计。对于排名列表的每个位置，从三个模型的候选边中选择加入后使累计 KL 散度最小的那条边。具体来说：

- 初始化曝光计数 $\boldsymbol{c} \leftarrow (0,0,0)$
- 对于每个排名位置 $t = 1, \ldots, n$：
    - 对每个候选群组 $j$，取其最高分候选边，临时更新计数后计算 $D_{\text{KL}}(\mathbf{q}' \| \boldsymbol{\pi})$
    - 选择使 KL 最小的群组和候选边，加入排名列表
- 优化的是公平优先目标：$\min_R \mathcal{L}_A(R) \text{ s.t. } \mathcal{L}_B(R) \leq \min_{R'} \mathcal{L}_B(R')$

**计算效率**：每个解耦模型每个 epoch 只处理 $\frac{|E_{\text{train}}|}{|S| \cdot \binom{|S|}{2} \cdot b}$ 个梯度，聚合阶段为贪心 $O(n \cdot G)$，整体开销很小。

## 实验关键数据

### 数据集

在 6 个真实世界数据集上评估，覆盖社交网络、信用评估、体育等不同场景：

| 数据集 | 节点数 | 边数 | 敏感属性 | $E_{s'\text{-}s}$% | $E_{s\text{-}s}$% | $E_{s'\text{-}s'}$% | 拓扑类型 |
|--------|--------|------|----------|------------|-----------|-------------|----------|
| Facebook | 1045 | 18726 | 性别 | 42% | 44% | 14% | 外围型 |
| German | 1000 | 15220 | 年龄 | 20% | 61% | 19% | 外围型 |
| NBA | 403 | 7435 | 国籍 | 27% | 63% | 10% | 外围型 |
| Pokec-n | 66569 | 361934 | 性别 | 5% | 66% | 29% | 社区型 |
| Pokec-z | 67796 | 432572 | 性别 | 5% | 58% | 37% | 社区型 |
| Credit | 30000 | 96165 | 年龄 | 12% | 86% | 2% | 外围型 |

### 主要结果（NDKL↓ / prec@1000↑）

| 方法 | Facebook | Credit | German | NBA | Pokec-n | Pokec-z |
|------|----------|--------|--------|-----|---------|---------|
| UGE | 0.05 / 0.97 | 0.80 / 1.00 | 0.08 / 0.69 | 0.07 / 0.58 | 0.06 / 0.90 | 0.06 / 0.91 |
| FairWalk | 0.06 / 0.96 | 0.06 / 1.00 | 0.11 / 0.94 | 0.06 / 0.55 | 0.07 / 1.00 | 0.07 / 1.00 |
| FairEGM | 0.09 / 0.97 | 0.11 / 1.00 | 0.05 / 0.62 | 0.07 / 0.60 | OOM | OOM |
| **MORAL** | **0.04** / 0.95 | **0.01** / 1.00 | **0.03** / 0.96 | **0.02** / 0.80 | **0.03** / 0.98 | **0.04** / 0.98 |

**关键发现**：

1. **ΔDP 无法检测子群偏差**：在 top-100 预测中，多数基线方法严重偏向某一子群类型（如 $E_{s\text{-}s}$ 远多于 $E_{s'\text{-}s'}$），但 ΔDP 值仍然很低——验证了 dyadic 聚合的致命缺陷。
2. **ΔDP 无法区分排序质量**：固定子群比例后，最差排序和最优排序的 NDKL 差距极大，但 ΔDP 完全相同—— 证实了排序感知性的必要性。
3. **MORAL 全面领先**：在所有 6 个数据集上 NDKL 均为最低（0.01–0.04），同时 prec@1000 保持在 0.80–1.00 的高水平，且不存在 OOM 问题。

## 亮点

1. **问题洞察深刻**：从 GNN 表达力、子群聚合陷阱、排序不敏感三个互补角度系统性地批判了现有 dyadic fairness 框架，每个论点都有清晰的理论分析和直观示例。
2. **NDKL 度量设计精妙**：通过对数位置权重自然实现了"top 位置偏差惩罚更重"的直觉，同时保持了与信息论的优雅连接（KL 散度）。
3. **方法极简且有效**：MORAL 是纯后处理方法，可搭配任意链接预测模型，无需修改训练过程，贪心聚合的计算开销极低。
4. **解耦训练策略巧妙**：针对不同边类型分别训练模型，既避免了单一模型的群体间性能不均衡，又天然适配后续的分组排序聚合。

## 局限与展望

1. **仅考虑二值敏感属性**：虽然论文声称框架可推广到多值/多属性，但所有实验均基于二值属性（$|S|=2$，三种边类型），多值属性下子群数量 $\binom{|S|}{2}$ 快速增长时的效果未验证。
2. **贪心策略非最优**：贪心排序聚合是对公平优先目标的近似解，不保证全局最优；极端不平衡分布下（如 Credit 数据集 $E_{s'\text{-}s'}$ 仅 2%）可能存在振荡。
3. **目标分布假设较强**：MORAL 以原始图的边类型分布 $\boldsymbol{\pi}$ 作为"公平目标"，隐含假设原图的子群比例是理想的——但如果原图本身就存在历史偏差，则保持分布可能固化不公平。
4. **缺少动态/在线场景评估**：真实推荐系统中链接预测是持续进行的，累积反馈可能改变图结构和公平约束，论文未讨论在线部署的适配。
5. **仅使用 GCN 编码器**：所有实验基于 GCN + 点积解码器，未验证更强的链接预测模型（如 SEAL、NBFNet）是否改变结论。

## 与相关工作的对比

| 维度 | 现有方法 | MORAL |
|------|---------|-------|
| 公平性框架 | 二元（intra vs. inter） | 非二元（三子群独立） |
| 公平性度量 | ΔDP（分类指标，排序不敏感） | NDKL（排序感知，位置加权） |
| 去偏策略 | 预处理/训练中/节点嵌入级别 | 后处理 + 解耦训练 |
| 可扩展性 | FairAdj/FairLP/EDITS/FairEGM 在大图上 OOM | 所有数据集均可运行 |
| 模型无关性 | 大部分方法与特定架构绑定 | 可搭配任意链接预测模型 |
| 效用-公平权衡 | 提高公平性往往大幅牺牲准确率 | NDKL 最优的同时 prec@1000 保持高水平 |

与信息检索领域的公平排序方法（DetConstSort、DELTR）相比，MORAL 专门针对链接预测的组合性质设计了解耦训练策略，而非直接套用通用排序公平方案。

## 启发与关联

1. **"度量即偏见"**：本文是一个教科书级案例——仅更换评估度量就能揭示隐藏偏差。启示是在任何公平性研究中，应先严格审视度量本身的性质（排序敏感性、子群覆盖性）再讨论算法。
2. **解耦训练的普适性**：将模型按子群解耦的思想可推广到其他图任务（如节点分类中的子群专用分类器），甚至推荐系统中的多目标排序。
3. **与推荐系统公平性的联系**：NDKL 最初来自信息检索（Geyik et al., 2019），本文将其引入图学习，提示两个领域的方法可以更深入互鉴。
4. **对 GNN 表达力研究的启示**：公平性需求为超越 1-WL 的 GNN 架构（如高阶 GNN、子图 GNN）提供了新的应用动机。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 对 dyadic fairness 的三维度系统批判和 NDKL 在链接预测中的引入均有很好的原创性
- 实验充分度: ⭐⭐⭐⭐ — 6 个数据集、10 个基线方法、多维度验证；缺少更强 GNN 和多值属性实验
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑清晰，理论和实例穿插得当，Figure 1-2 的直观示例非常有效
- 价值: ⭐⭐⭐⭐ — 对公平链接预测评估范式的根本性反思，MORAL 简洁实用，有很好的实践指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](../../ICML2026/ai_safety/beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](../../ICML2025/ai_safety/breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)

</div>

<!-- RELATED:END -->
