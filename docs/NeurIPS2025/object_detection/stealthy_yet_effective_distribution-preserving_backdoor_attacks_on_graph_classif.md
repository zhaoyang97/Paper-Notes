---
title: >-
  [论文解读] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification
description: >-
  [NeurIPS 2025][目标检测][backdoor attack] 提出 DPSBA，一种面向图分类的 clean-label 后门攻击框架，通过对抗训练生成分布内（in-distribution）触发子图，同时抑制结构异常和语义异常，在保持高攻击成功率的同时显著提升隐蔽性。 GNN 在图分类任务中表现强大…
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "backdoor attack"
  - "图分类"
  - "分布保持"
  - "对抗训练"
  - "图神经网络"
---

# Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification

**会议**: NeurIPS 2025  
**arXiv**: [2509.26032](https://arxiv.org/abs/2509.26032)  
**代码**: [有](https://github.com/TheCoderOfs/DPSBA)  
**领域**: AI Safety / 图学习安全  
**关键词**: backdoor attack, 图分类, 分布保持, 对抗训练, GNN安全

## 一句话总结

提出 DPSBA，一种面向图分类的 clean-label 后门攻击框架，通过对抗训练生成分布内（in-distribution）触发子图，同时抑制结构异常和语义异常，在保持高攻击成功率的同时显著提升隐蔽性。

## 研究背景与动机

GNN 在图分类任务中表现强大，但容易受到后门攻击的威胁。现有图级后门攻击方法面临两大核心问题：

**结构偏差（Structural Deviation）**：现有方法（如 ER-B、GTA、Motif）通过注入稀有/非自然子图作为触发器，这些子图偏离了干净图的结构分布。实验表明，这些方法生成的后门图在异常检测模型（如 SIGNET）下的异常评分分布与干净图存在明显偏移。例如在 AIDS 数据集上，Motif 触发器的 AUC 高达 99.71%，几乎被完美检测。

**语义偏差（Semantic Deviation）**：传统方法通过标签翻转（label flipping）将后门图的标签改为攻击目标类，导致图的真实语义与标注不一致。虽然 clean-label 设置可缓解此问题，但会导致攻击成功率（ASR）显著下降。

**核心挑战**：如何设计一种图级后门攻击，既保持干净样本的分布特性、避免标签操纵，又能同时兼顾攻击有效性和隐蔽性？

## 方法详解

### 整体框架

DPSBA 包含两个阶段：
- **毒化样本构造**：选择难样本 → 确定注入位置 → 生成触发子图
- **触发器优化**：通过对抗训练联合优化攻击效果和分布隐蔽性

### 关键设计

1. **难样本选择（Hard Sample Selection）**

   **功能**：从目标类中挑选模型最不确定的样本进行毒化。

   **核心思路**：使用代理模型计算每个图对目标标签的置信度 $\text{cfd}(G) = \text{softmax}(f_\theta(G))_{y_t}$，选择置信度最低的 $p\%$ 样本。

   **设计动机**：低置信度样本位于决策边界附近，更容易被微小扰动改变预测结果，因此是 clean-label 攻击的理想候选。这避免了标签翻转带来的语义不一致。

2. **触发器位置选择（Trigger Location Selection）**

   **功能**：确定在图中哪些节点注入触发子图。

   **核心思路**：两阶段筛选 —— 先选度中心性最高的 $2M$ 个候选节点，再通过代理模型消融评估每个节点的影响力 $S(v) = |f_\theta(G + \Delta_v) - f_\theta(G)|$，选择影响力最大的 $M$ 个节点。

3. **触发器生成与注入（Topology-Feature Generator）**

   **功能**：使用可学习网络生成触发子图的拓扑结构和节点特征。

   **核心思路**：
    - **拓扑生成器**：MLP 将注入区域的邻接矩阵映射为可学习的软结构，通过二值化机制生成二进制邻接矩阵 $A_{binary} = \mathbb{I}(A > 0.5)$
    - **特征生成器**：MLP 根据注入位置的原始特征生成触发节点特征 $X' = \sigma(W_2 X + b_2)$，确保生成特征与数据分布对齐

4. **对抗异常最小化（Adversarial Anomaly Minimization）**

   **功能**：通过对抗训练使触发图在统计上难以被异常检测模型区分。

   **核心思路**：引入两个判别器 —— 拓扑判别器 $D_{\theta_t}$（GCN 实现）和特征判别器 $D_{\theta_f}$（MLP 实现），生成器与判别器进行极小极大博弈：

    $\min_{\omega_t} \max_{\theta_t} \mathcal{L}_d^{(t)} = \sum_{G \sim \mathcal{G}_c} \log D_{\theta_t}(G) + \sum_{G \sim \mathcal{G}_b} \log(1 - D_{\theta_t}(G_{g_t}(\omega_t)))$

### 损失函数 / 训练策略

联合优化攻击损失和对抗异常损失：

- **攻击损失**：$\mathcal{L}_{atk} = -\log f_{\theta^*}(G_{g_t})_{y_t}$
- **拓扑优化**：$\min_{\omega_t} \sum \mathcal{L}_{atk} + \alpha \mathcal{L}_d^{(t)}$
- **特征优化**：$\min_{\omega_f} \sum \mathcal{L}_{atk} + \beta \mathcal{L}_d^{(f)}$

采用分阶段对抗训练：先优化拓扑生成器和判别器，再优化特征生成器和判别器。代理模型在每个阶段后微调以保持梯度信号的可靠性。超参数 $\alpha$、$\beta$ 平衡隐蔽性与攻击效果。

## 实验关键数据

### 主实验

在 4 个 TUDataset 数据集上，使用 3 种图分类器（GCN、GIN、SAGPool），与 5 种基线攻击方法对比。

| 数据集 | 模型 | 方法 | ASR(%) | CAD(%) | AUC(%) |
|--------|------|------|--------|--------|--------|
| AIDS | GCN | GTA | 93.21 | 5.14 | 99.34 |
| AIDS | GCN | **DPSBA** | **94.76** | **2.38** | **72.65** |
| AIDS | GIN | Motif | 97.75 | 2.28 | 99.71 |
| AIDS | GIN | **DPSBA** | **95.87** | **1.94** | **73.66** |
| FRANKENSTEIN | GCN | GTA | 99.35 | 2.74 | 100.00 |
| FRANKENSTEIN | GCN | **DPSBA** | **98.37** | **1.01** | **68.96** |
| ENZYMES | GCN | GTA | 95.33 | 3.00 | 71.20 |
| ENZYMES | GCN | **DPSBA** | **96.67** | **-0.67** | **66.11** |

### 消融实验

| 配置 | ASR | AUC | 说明 |
|------|-----|-----|------|
| DPSBA 完整 | 94.76% | 72.65% | 最佳平衡 |
| 去除拓扑判别器 | ~95% | ~85% | 结构异常增加 |
| 去除特征判别器 | ~94% | ~80% | 特征偏移增加 |
| 使用 dirty-label | ASR↑ | AUC↑ | 更易被检测 |
| 非对抗训练 | ASR↑ | AUC↑↑ | 隐蔽性大幅下降 |

### 关键发现

- DPSBA 在所有数据集上将 AUC（异常检测可检出率）降低了 **20-30 个百分点**，同时保持了甚至提升了 ASR
- 在 AIDS 数据集上，Motif 的 AUC 为 99.71%（几乎完美被检测），而 DPSBA 仅 73.66%
- Clean-label 设置下 DPSBA 的 CAD（干净准确率下降）普遍低于其他方法，说明对正常样本的影响更小

## 亮点与洞察

- **首次系统分析**了图分类后门攻击中结构偏差和语义偏差的问题，并提出了理论下界（将总变差距离与最优 AUC 关联）
- **干净标签 + 对抗训练**的组合设计非常优雅：既不修改标签（避免语义偏差），又通过判别器约束触发器的分布（抑制结构偏差）
- 攻击者的威胁模型设计合理：黑盒知识 + 有限毒化能力，更贴近真实场景

## 局限与展望

- 触发器大小固定为 4 个节点，对于大规模图可能不够灵活
- 仅考虑了 SIGNET 一种异常检测方法作为评估标准
- clean-label 设置在部分数据集上（如 Motif-S）的 ASR 仍有提升空间
- 未考虑自适应防御（adaptive defense），即防御者知道攻击方法后的对抗

## 相关工作与启发

- 与节点级分布保持攻击 [Zhang et al.] 的关键区别：图分类需要修改全局消息传递以影响整图表示，分布保持难度显著更高
- 对抗训练的思路可应用于其他安全领域，如对抗鲁棒性训练中保持样本分布
- 该工作也提醒：现有图分类后门防御依赖异常检测的范式可能不够可靠

## 评分

- 新颖性: ⭐⭐⭐⭐ — clean-label + 分布保持的组合在图后门攻击中首次提出
- 实验充分度: ⭐⭐⭐⭐⭐ — 4数据集×3分类器×5基线，消融全面
- 写作质量: ⭐⭐⭐⭐ — 问题分析清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 对GNN安全研究有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](../../CVPR2025/object_detection/test-time_backdoor_detection_for_object_detection_models.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](../../CVPR2025/object_detection/distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [\[AAAI 2026\] An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice](../../AAAI2026/object_detection/an_overall_real-time_mechanism_for_classification_and_quality_evaluation_of_rice.md)
- [\[CVPR 2026\] Distribution-Aligned Multimodal Fusion for Robust Object Detection](../../CVPR2026/object_detection/distribution-aligned_multimodal_fusion_for_robust_object_detection.md)

</div>

<!-- RELATED:END -->
