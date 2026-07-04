---
title: >-
  [论文解读] HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport
description: >-
  [ICML2025][图学习][异质图神经网络] 提出 HGOT，首次将最优传输理论引入异质图自监督学习，用 branch view（元路径视图）与 central view（聚合视图）之间的 Fused Gromov-Wasserstein 传输计划替代传统对比学习中的数据增强与正负样本选取，在节点分类上平均提升超过 6%。
tags:
  - "ICML2025"
  - "图学习"
  - "异质图神经网络"
  - "自监督学习"
  - "最优传输"
  - "元路径"
  - "Gromov-Wasserstein"
---

# HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport

**会议**: ICML2025  
**arXiv**: [2506.02619](https://arxiv.org/abs/2506.02619)  
**作者**: Yanbei Liu, Chongxu Wang, Zhitao Xiao, Lei Geng, Yanwei Pang, Xiao Wang
**代码**: 待确认  
**领域**: 图学习  
**关键词**: 异质图神经网络, 自监督学习, 最优传输, 元路径, Gromov-Wasserstein

## 一句话总结

提出 HGOT，首次将最优传输理论引入异质图自监督学习，用 branch view（元路径视图）与 central view（聚合视图）之间的 Fused Gromov-Wasserstein 传输计划替代传统对比学习中的数据增强与正负样本选取，在节点分类上平均提升超过 6%。

## 研究背景与动机

现有异质图神经网络（HGNN）大多依赖有监督/半监督学习，但异质图的标注代价高昂。自监督学习（SSL）为替代方案，其中对比学习是主流策略。然而对比学习面临两大挑战：

**图增强策略的设计困难**（C1）：异质图数据结构离散，轻微扰动可能改变图性质，导致原图与增强图的标签不一致。同时"最大化/最小化相似度"概念模糊，缺乏明确度量。

**异质语义信息的完整聚合**（C2）：异质图包含多种节点类型、多种关系和不同连接度，简单聚合会扭曲拓扑结构、丢失上下文语义。

HGOT 的核心动机是：**完全不使用图增强、不需要正负样本对**，而是通过最优传输发现图空间与表示空间之间的匹配关系。

## 方法详解

HGOT 框架由三个核心模块组成：节点特征变换、聚合视图生成、多视图最优传输对齐。

### 1. 节点特征变换（Node Feature Transformation）

异质图中不同类型节点的特征维度不同，需投影到统一空间。对类型为 $\varphi_i$ 的第 $i$ 个节点：

$$\mathbf{h}_i = \mathbf{W}_{\varphi_i} \cdot \mathbf{x}_i + \mathbf{b}_{\varphi_i}$$

其中 $\mathbf{W}_{\varphi_i}$ 为类型特定的线性变换矩阵，投影后所有节点特征维度相同，为后续 OT 计算做准备。

### 2. 聚合视图生成（Aggregated View Generation）

将原始异质图按元路径分解为多个同质子图（branch view），再聚合为 central view。

**节点聚合**：对每条元路径 $p$，使用多头注意力机制生成节点 $v_i$ 在该元路径下的表示：

$$\tilde{\mathbf{h}}_i^p = \text{CONCAT}_{k=1}^{K} \;\Psi\!\left(\sum_{v_j \in \mathcal{N}_p(v_i)} \alpha_{ij}^p \mathbf{W}^p \mathbf{h}_i\right)$$

注意力系数 $\alpha_{ij}^p$ 通过 softmax 归一化计算。

**元路径级聚合**：用另一层注意力网络加权融合所有元路径表示：

$$\mathbf{h}_i^{\text{agg}} = \sum_{p=1}^{|\mathcal{P}|} \beta^p \cdot \tilde{\mathbf{h}}_i^p$$

权重 $\beta^p$ 可解释为第 $p$ 条元路径的贡献，由 type-level attention vector $\mathbf{q}$ 通过 tanh 变换后 softmax 得到。

**边聚合**：对所有元路径的邻接矩阵做逻辑 OR 运算：

$$\mathbf{A}_{\text{agg}} = \mathbf{A}_1 \lor \mathbf{A}_2 \lor \cdots \lor \mathbf{A}_{|\mathcal{P}|}$$

确保只要任一元路径视图中两节点相连，聚合视图中就保留该边，且不引入重复边。

### 3. 多视图最优传输对齐（Multi-view OT Alignment）

核心创新：用最优传输取代对比学习。计算每个 branch view 与 central view 之间的传输计划，对齐图空间与表示空间。

**节点分布上的 Wasserstein 距离**：

$$\mathcal{D}_n(\mathbf{H}_p, \mathbf{H}_{\text{agg}}) = \min_{\pi \in \Pi(\mu, \nu)} \langle \mathcal{F}(\mathbf{H}_p, \mathbf{H}_{\text{agg}}), \pi \rangle$$

代价矩阵 $\mathcal{F}_{ij} = \mathcal{C}_X(\mathbf{H}_p^i, \mathbf{H}_{\text{agg}}^j)$ 使用余弦距离。

**边结构上的 Gromov-Wasserstein 距离**：

$$\mathcal{D}_e(\mathbf{A}_p, \mathbf{A}_{\text{agg}}) = \min_{\pi \in \Pi(\mu, \nu)} \langle \mathcal{E}(\mathbf{A}_p, \mathbf{A}_{\text{agg}}) \otimes \pi, \pi \rangle$$

代价函数 $\mathcal{C}_A(\mathbf{A}_p^{ik}, \mathbf{A}_{\text{agg}}^{jl}) = |\mathbf{A}_p^{ik} - \mathbf{A}_{\text{agg}}^{jl}|$，直接衡量边结构差异。

**融合 Gromov-Wasserstein 距离**：同时考虑节点特征和边结构：

$$\mathcal{D}_g(\mathcal{G}_p, \mathcal{G}_{\text{agg}}) = \min_{\pi} \;\sigma \sum_{ij} \mathcal{C}_X \cdot \pi_{ij}^{\mathcal{G}} + (1-\sigma) \sum_{ijkl} \mathcal{C}_A \cdot \pi_{ij}^{\mathcal{G}} \pi_{kl}^{\mathcal{G}}$$

超参数 $\sigma \in [0,1]$ 平衡节点特征与边结构的贡献。最终目标函数驱动编码器学习与图空间匹配关系一致的节点表示。

### 训练流程

- 对每对 (branch view, central view)，用 Sinkhorn 算法求解最优传输计划 $\pi^{\mathcal{G}}$
- 同时对表示空间求传输计划 $\pi^{\mathcal{Z}}$
- 对齐二者使得表示一致性与图拓扑一致性相吻合
- 整体为端到端无监督训练，无需标签

## 实验关键数据

**数据集**：ACM、DBLP、Freebase、AMiner（四个常用异质图基准）

**下游任务**：节点分类、节点聚类

**主要结果**：
- 节点分类任务上，HGOT 相比 SOTA 方法平均准确率提升超过 **6%**
- 在所有四个数据集上均取得最佳或竞争性结果
- 对比基线包括 HAN、HeCo、HGCML、HGMAE 等主流异质图自监督方法

**消融实验**：
- 去掉 OT 对齐模块后性能显著下降，验证了 OT 机制的有效性
- 仅使用 Wasserstein 距离（不含边信息）或仅使用 GW 距离（不含节点特征）均不如融合版本
- 聚合视图的 OR 运算优于简单相加或取交集

## 亮点与洞察

1. **首次将最优传输引入异质图自监督学习**：绕过了对比学习中图增强设计和正负样本选取的核心瓶颈，思路清晰且具有理论美感。
2. **Branch view + Central view 设计**：通过元路径分解获取局部语义（branch），再聚合为全局语义（central），用 OT 对齐两者关系，逻辑自洽。
3. **融合 Gromov-Wasserstein**：同时利用节点特征和边结构信息计算传输距离，比单独使用任一信息更完整。
4. **无需数据增强**：避免了离散图结构下增强带来的标签一致性问题。

## 局限与展望

1. **计算复杂度**：Gromov-Wasserstein 距离涉及四阶张量，即使使用 Sinkhorn 近似，在大规模图上仍可能成为瓶颈。论文未给出时间/内存开销的详细对比。
2. **元路径依赖**：方法仍然依赖人工定义的元路径，对于元路径不易定义的异质图场景（如 schema 复杂或动态变化的图），适用性存疑。
3. **评估任务有限**：仅评估了节点分类和聚类，未涉及链接预测、图分类等异质图上同样重要的任务。
4. **数据集规模较小**：四个基准数据集均为中小规模，大规模异质图上的表现未知。
5. **$\sigma$ 超参敏感性**：融合 GW 距离中的 $\sigma$ 需要调优，论文对超参搜索的讨论不够充分。

## 相关工作与启发

- **HeCo**（Wang et al. 2021）：通过网络 schema 和元路径对齐局部/全局信息的对比学习，是 HGOT 的直接对标方法。
- **HGCML**（Wang et al. 2023）：多视图对比学习 + 新正样本采样策略。
- **HGMAE**（Tian et al. 2023）：将 masked autoencoder 应用于异质图。
- **Gromov-Wasserstein**（Mémoli 2011; Vayer et al. 2020）：图之间的最优传输距离框架，HGOT 将其推广到异质图场景。

**启发**：OT 作为一种度量两个分布距离的通用工具，可进一步探索其在动态异质图、知识图谱补全等场景中的应用。此外，将 OT 与生成式自监督学习（如 MAE）结合可能获得互补优势。

## 可复现性要点

- 需实现 Sinkhorn 算法或调用 POT 库求解 OT
- 异质图预处理需按元路径分解为同质子图，构建各元路径邻接矩阵
- 关键超参：$\sigma$（节点/边权重平衡）、K（注意力头数）、$d$（投影维度）、$d_m$（type-level attention 维度）
- 代码是否开源待确认，无代码链接时需根据论文从头实现

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将 OT 用于异质图 SSL，视角新颖
- 实验充分度: ⭐⭐⭐ — 4 个数据集 2 个任务，消融完整但规模/任务覆盖有限
- 写作质量: ⭐⭐⭐⭐ — 公式体系清晰，行文逻辑通顺
- 价值: ⭐⭐⭐⭐ — 为异质图自监督学习提供了对比学习之外的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[AAAI 2026\] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs](../../AAAI2026/graph_learning/gcl-ot_graph_contrastive_learning_with_optimal_transport_for_heterophilic_text-a.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[ECCV 2024\] SENC: Handling Self-collision in Neural Cloth Simulation](../../ECCV2024/graph_learning/senc_handling_self-collision_in_neural_cloth_simulation.md)

</div>

<!-- RELATED:END -->
