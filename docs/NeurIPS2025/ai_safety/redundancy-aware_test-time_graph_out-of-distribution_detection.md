---
title: >-
  [论文解读] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection
description: >-
  [NeurIPS 2025][AI安全][图OOD检测] 提出 RedOUT 框架，通过最小化结构熵构建编码树来消除图结构中的冗余信息，结合冗余感知图信息瓶颈(ReGIB)原理，在测试时无需修改预训练模型参数即可有效区分ID和OOD图样本，在10个数据集对上平均AUC达87.46%。 图结构数据的OOD检测面临独特挑战：由于…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "图OOD检测"
  - "结构熵"
  - "信息瓶颈"
  - "测试时检测"
  - "编码树"
---

# Redundancy-Aware Test-Time Graph Out-of-Distribution Detection

**会议**: NeurIPS 2025  
**arXiv**: [2510.14562](https://arxiv.org/abs/2510.14562)  
**代码**: [有](https://github.com/name-is-what/RedOUT)  
**领域**: 其他  
**关键词**: 图OOD检测, 结构熵, 信息瓶颈, 测试时检测, 编码树

## 一句话总结

提出 RedOUT 框架，通过最小化结构熵构建编码树来消除图结构中的冗余信息，结合冗余感知图信息瓶颈(ReGIB)原理，在测试时无需修改预训练模型参数即可有效区分ID和OOD图样本，在10个数据集对上平均AUC达87.46%。

## 研究背景与动机

图结构数据的OOD检测面临独特挑战：由于图的非欧几何特性和复杂拓扑结构，模型在遇到分布外数据时容易做出高置信度的错误预测。现有方法主要分为两类：

**端到端方法**：从零开始训练OOD专用GNN（如GOOD-D），利用图对比学习提取判别性表示

**后处理方法**：采用预训练GNN，在推理时微调OOD检测器（如GOODAT通过可学习图掩码在测试样本上优化）

然而，这些数据驱动范式存在一个核心问题：**结构冗余导致语义偏移**。测试图中既包含区分ID/OOD的关键结构（如图1(b)中虚线椭圆标注的distinctive components），也包含大量相似的无关结构元素。这些冗余结构会干扰模型对关键模式的捕获。GOODAT虽然尝试通过图掩码解决这一问题，但其可学习的图增强可能改变子结构的语义信息或造成信息丢失。

作者的关键观察：在AIDS/DHFR数据集对上，对图表示进行结构熵最小化后，OOD得分的方差减小，ID和OOD样本的分布重叠显著降低（图1(c)）。这直观地表明，去除冗余后保留的图的区分性部分能更有效地区分分布。

## 方法详解

### 整体框架

RedOUT 是一个无监督的测试时图OOD检测框架，核心思路是：
1. **ReGIB原理**：将图信息分解为本质信息和无关冗余
2. **结构熵最小化**：构建编码树作为本质视图的实例化
3. **层次表示学习**：基于编码树学习去冗余的图表示
4. **OOD检测**：利用校准的OOD得分进行检测，不修改预训练模型参数

### 关键设计

#### 1. **冗余感知图信息瓶颈 (ReGIB)**

标准GIB目标为 $\min -I(f(G);Y) + \beta I(G;f(G))$。作者引入本质视图 $G^*$ 并用伪标签 $\tilde{Y}$ 替代未知标签，扩展为ReGIB：

$$\min \text{ReGIB} \triangleq -I(f(G^*);f(G)) + I(f(G^*);f(G)|\tilde{Y}) + \beta I(G^*;f(G^*))$$

三项的直觉含义：
- **第一项** $I(f(G^*);f(G))$：预测项，确保本质信息被保留
- **第二项** $I(f(G^*);f(G)|\tilde{Y})$：压缩项，鼓励丢弃在给定伪标签下的冗余信息
- **第三项** $I(G^*;f(G^*))$：压缩项，对应最小化本质视图的结构复杂度

设计动机：预训练模型仅在ID数据上训练，对OOD数据的预测更加随机，难以识别区分性结构。ReGIB通过 $G^*$ 的区分性模式为模型提供校准信号。

#### 2. **结构熵最小化与编码树构建**

通过Proposition 4证明 $\min I(G^*;f(G^*)) \leq \min I(G^*;G) = \min \mathcal{H}(G^*)$，因此最小化结构熵即可获得去冗余的本质视图。

结构熵定义为：$\mathcal{H}^T(G) = -\sum_{v_\tau \in T} \frac{g_{v_\tau}}{vol(\mathcal{V})} \log \frac{vol(v_\tau)}{vol(v_\tau^+)}$

编码树构建分两步：(1) 构建全高度二叉编码树；(2) 压缩至固定高度 $k$。使用 MERGE 和 DROP 两个算子高效完成，时间复杂度为 $O(h(|\mathcal{E}|\log|\mathcal{V}|+|\mathcal{V}|))$，近似线性。

#### 3. **层次表示学习**

编码树编码器设计为自底向上的消息传递：$\mathbf{x}_v^{(l)} = \text{MLP}^{(l)}(\sum_{u \in \mathcal{C}(v)} \mathbf{x}_u^{(l-1)})$

特征从叶节点向根节点聚合，最终通过readout函数获得树嵌入 $\mathbf{Z}_T$。

### 损失函数 / 训练策略

最终优化目标为：$\mathcal{L} = \mathcal{L}_{Cl} + \lambda \mathcal{L}_{CRI}$

- $\mathcal{L}_{Cl}(G^*, G)$：对比学习损失，最大化本质视图与原始图表示的互信息
- $\mathcal{L}_{CRI}(G, G^*)$：条件冗余消除损失，最小化给定伪标签下的冗余互信息

OOD检测时，直接以整体损失 $\mathcal{L}$ 作为OOD检测得分。结构熵最小化为预处理步骤（无需额外学习），预训练模型参数完全冻结。

## 实验关键数据

### 主实验

在10个TUDataset和OGB数据集对上的OOD检测结果（AUC%）：

| ID/OOD数据集对 | RedOUT | GOOD-D(次优) | GOODAT | 提升 |
|---|---|---|---|---|
| BZR/COX2 | **95.06** | 94.99 | 82.16 | +0.07 |
| PTC-MR/MUTAG | **94.45** | 81.21 | 81.84 | +12.61 |
| AIDS/DHFR | **99.98** | 99.07 | 96.43 | +0.91 |
| ClinTox/LIPO | **86.56** | 69.18 | 62.46 | **+17.38** |
| Esol/MUV | **95.00** | 91.52 | 85.91 | +3.48 |
| 平均AUC | **87.46** | 80.73 | 76.89 | **+6.73** |
| 平均排名 | **1.3** | 2.4 | 3.9 | - |

### 消融实验

异常检测任务结果（AUC%）：

| 数据集 | RedOUT | GOOD-D | GLocalKD | 说明 |
|---|---|---|---|---|
| ENZYMES | **69.14** | 65.64 | 57.18 | 蛋白质图 |
| DHFR | **54.45** | 53.08 | 52.11 | 分子图 |
| BZR | **66.86** | 61.24 | 55.32 | 分子图 |
| IMDB-B | 53.12 | 54.89 | **56.42** | 社交网络 |

### 关键发现

1. **分子图上优势巨大**：RedOUT在分子图数据集上平均提升约10%，因为分子图的语义信息直接体现在结构组成中（如功能团）
2. **社交网络是弱项**：在IMDB-B/IMDB-M上未取得最佳结果，因为这些数据来自相同来源，仅标签不同，难以仅凭结构区分
3. **高效可扩展**：在Erdős-Rényi图上的实验表明，运行时间和内存使用随节点数近似线性增长

## 亮点与洞察

1. **首次将结构熵引入测试时OOD检测**：将信息论工具（结构熵）与图信息瓶颈有机结合，为图OOD检测提供了新范式
2. **理论保证完善**：给出了ReGIB目标的上下界，证明了最优编码器能保留更多与真实标签相关的信息（Theorem 3.4）
3. **无需修改预训练模型**：编码树构建是预处理步骤，完全不改变预训练模型参数，实用性强
4. **冗余消除的信息论视角**：从互信息分解的角度出发，将"去冗余"这一直觉形式化

## 局限与展望

1. 在社交网络等结构特征不明显的图数据上效果有限
2. 编码树高度 $k$ 需要预先设定，对不同图可能需要不同的最优高度
3. 依赖伪标签 $\tilde{Y}$ 的质量，若预训练模型在ID数据上表现不佳可能影响效果
4. 可考虑将方法扩展到节点级别或子图级别的OOD检测

## 相关工作与启发

- 与GOODAT相比，RedOUT避免了可学习增强可能引入语义偏移的问题，而是从信息论角度直接消除冗余
- 结构熵最小化作为图结构的层次抽象工具，在图学习的其他任务（如图分类、社区检测）中也有潜在应用
- ReGIB原理可推广到其他需要区分本质信息和冗余信息的图学习任务

## 评分

- 新颖性: ⭐⭐⭐⭐ （结构熵+GIB的组合新颖，但各组件本身并非全新）
- 实验充分度: ⭐⭐⭐⭐⭐ （18个baseline，10个数据集对，异常检测，时间分析）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，但公式较多读起来有一定门槛）
- 价值: ⭐⭐⭐⭐ （为图OOD检测提供了新范式，实用性好）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/ai_safety/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[NeurIPS 2025\] SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/ai_safety/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](../../ICCV2025/ai_safety/fret_feature_redundancy_elimination_for_test_time_adaptation.md)

</div>

<!-- RELATED:END -->
