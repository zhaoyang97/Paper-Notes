---
title: >-
  [论文解读] On Stealing Graph Neural Network Models
description: >-
  [AAAI2026][图学习][图神经网络] 证明了在严格查询限制下（如仅100次查询），攻击者可通过"本地获取encoder（随机初始化/SSL训练）+ K-means策略性查询选择"两阶段方法高效窃取GNN模型，在Physics数据集上仅用100次查询即达91%准确率，而现有SOTA需约5000次查询加额外embedding访问才能达到类似水平。
tags:
  - "AAAI2026"
  - "图学习"
  - "图神经网络"
  - "模型安全"
  - "自监督学习"
  - "查询选择"
  - "黑盒攻击"
  - "Inductive/Transductive"
---

# On Stealing Graph Neural Network Models

**会议**: AAAI2026  
**arXiv**: [2511.07170](https://arxiv.org/abs/2511.07170)  
**作者**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pręgowska, Tomasz P. Michalak  
**代码**: [m-podhajski/OnStealingGNNs](https://github.com/m-podhajski/OnStealingGNNs)  
**领域**: 图学习  
**关键词**: GNN模型窃取, 模型安全, 自监督学习, 查询选择, 黑盒攻击, Inductive/Transductive  

## 一句话总结

证明了在严格查询限制下（如仅100次查询），攻击者可通过"本地获取encoder（随机初始化/SSL训练）+ K-means策略性查询选择"两阶段方法高效窃取GNN模型，在Physics数据集上仅用100次查询即达91%准确率，而现有SOTA需约5000次查询加额外embedding访问才能达到类似水平。

## 背景与动机

### GNN模型面临的安全威胁

图神经网络在节点分类、链接预测、图分类和推荐系统中广泛应用，但与所有神经网络一样面临安全威胁。模型窃取（Model Stealing）攻击中，攻击者通过查询victim模型的API获取输入-输出对，训练一个功能相近的surrogate模型。典型防御手段是限制查询次数，但现有GNN窃取研究普遍假设无限查询访问，忽略了实际部署中严格的查询限制。

### 现有方法的假设过于宽松

已有GNN窃取方法（Shen et al., Podhajski et al.）依赖victim模型返回embedding等中间表示来窃取encoder，并假设无限查询次数。Data-free方法虽不需要数据但仍需大量查询（如100次查询×250节点=25000个查询节点）。这些假设在实际场景中不成立——真实API通常只返回类别标签且有严格的查询配额。

### 自监督学习的关键观察

来自SSL研究的重要发现：在inductive设定下，随机初始化的GCN encoder配合训练好的MLP head即可取得接近完全训练模型的性能。例如DGI报告Reddit上随机encoder达93.3%而SSL训练仅94.0%，BGRL在Physics上差距仅2个百分点。这意味着攻击者可能根本不需要查询victim来获取encoder——在inductive设定下用随机初始化，在transductive设定下用SSL本地训练即可。

## 核心问题

在攻击者只能获得类别标签（非embedding）且查询次数严格受限的hardest黑盒场景下，如何高效窃取GNN模型？

## 方法详解

### 整体框架（三阶段）

1. **本地获取Encoder**：不与victim交互，在本地获取feature extractor
2. **策略性查询选择**：利用encoder生成的嵌入空间选择最有信息量的查询节点
3. **训练MLP Head**：用查询得到的类别标签训练MLP，与encoder组合得到surrogate模型

### 威胁模型

- **黑盒设定**：不知道victim的参数、架构和训练数据 $\mathbf{G}_V$
- **查询限制**：最多 $q_n$ 次查询，每次仅返回类别标签
- **数据假设**：攻击者拥有与victim训练数据同分布的无标签图 $\mathbf{G}_D$

### 阶段1: Encoder获取

**Inductive设定**：直接使用随机初始化的GCN作为encoder，完全不与victim交互。T-SNE可视化表明随机encoder在inductive设定下已能产生结构化的embedding，各类节点呈明显分簇。

**Transductive设定**：使用自监督学习（LaGraph）在攻击者的全部数据 $\mathbf{G}_D$ 上本地训练encoder。Transductive图通常较小（如Cora 2708节点），SSL训练成本低。SSL在该设定下提升显著：Cora上从69.3%提升至82.3%（+13.0%）。

### 阶段2: 查询选择

利用encoder生成的嵌入 $\mathbf{H} = f(\mathbf{X}_D, \mathbf{A}_D) \in \mathbb{R}^{n \times b}$，使用K-means将节点划分为 $q_n$ 个聚类，从每个聚类中选择最接近质心的节点作为查询节点 $\{v_1', \ldots, v_{q_n}'\}$。这确保查询覆盖整个输入空间，最大化每次查询的信息增益，类似主动学习中的diversity sampling。

### 阶段3: MLP训练

对选定节点查询victim获取标签 $\{y_1, \ldots, y_{q_n}\}$，用Cross Entropy损失训练MLP组件 $g$：
$$\hat{y} = f_s(\mathbf{X}, \mathbf{A}) = g(f(\mathbf{X}, \mathbf{A}))$$
其中 $f$ 为encoder，$g$ 为MLP head，$f_s$ 为最终的surrogate模型。

## 实验关键数据

### Inductive设定 (Target: SAGE, Surrogate: GCN, $q_n=100$)

| 方法 | Reddit Acc | CS Acc | Physics Acc | Photo Acc | WikiCS Acc |
|------|-----------|--------|------------|-----------|------------|
| Target（victim）| 94.8 | 93.9 | 96.0 | 93.0 | 72.5 |
| E2E | 47.0±4.5 | 73.6±3.9 | 89.9±1.1 | 81.2±0.8 | 61.6±1.3 |
| Shen et al.* | 77.2±5.1 | 77.7±0.8 | 90.6±0.5 | 84.4±0.8 | 64.9±1.0 |
| Podhajski et al.* | 79.9±4.1 | 78.0±0.5 | 89.9±0.2 | 84.0±1.0 | 64.0±1.1 |
| datafree | 13.6±4.1 | 24.8±2.8 | 55.5±5.0 | 24.9±2.8 | 38.6±2.1 |
| **R-init+Select (Ours)** | **82.5±1.2** | **78.4±2.1** | **91.2±0.4** | **86.8±1.0** | **65.5±1.8** |

*标记方法需要额外访问victim embedding（更弱的威胁模型）

### Transductive设定 (Target: GCN, Surrogate: GCN, $q_n=10$)

| 方法 | Cora Acc | Cora Fid | Citeseer Acc | Citeseer Fid | Pubmed Acc | Pubmed Fid |
|------|---------|---------|-------------|-------------|-----------|----------|
| Target | 83.3 | — | 72.1 | — | 80.0 | — |
| E2E | 47.5±3.7 | 45.7±1.0 | 37.2±6.1 | 41.1±7.5 | 61.0±4.9 | 67.5±5.0 |
| datafree | 18.1±2.7 | 21.1±3.9 | 22.1±3.3 | 23.1±3.8 | 33.2±2.9 | 33.4±3.0 |
| SSL+Random | 56.1±2.7 | 56.8±3.0 | 51.3±5.1 | 57.6±5.5 | 66.1±7.3 | 72.7±9.0 |
| **SSL+Select (Ours)** | **69.9±1.2** | **72.5±1.3** | **66.3±1.9** | **72.4±2.3** | **67.0±6.0** | **80.1±4.7** |

### 随机初始化 vs SSL训练 Encoder对比

| 设定 | 数据集 | Random Acc | SSL-Trained Acc | 增益 |
|------|-------|-----------|----------------|-----|
| Inductive | Reddit | 93.3 | 94.0 | +0.7 |
| Inductive | Physics | 93.7 | 95.7 | +2.0 |
| Transductive | Cora | 69.3 | 82.3 | +13.0 |
| Transductive | Citeseer | 61.9 | 71.8 | +9.9 |

### 防御下的鲁棒性

在10%预测翻转防御下，本方法仍在所有设定中保持最高性能，说明防御效果有限。

## 亮点

- **首次研究严格查询限制下的GNN窃取**：将问题分解为"encoder获取"和"head窃取"两个独立阶段，揭示了此前被忽视的严重安全威胁
- **随机初始化encoder的有效性**：在inductive设定下，随机GCN encoder已能产生高质量特征表示，完全无需与victim交互即可获取模型主体部分
- **资源效率极高**：仅需100次查询和单个商用CPU（AMD EPYC 7742）即可完成攻击，对比Shen et al.需约5000次查询+GPU+victim embedding，查询效率提升约15倍
- **K-means查询选择的有效性**：对比随机选择和其他主动学习策略（farthest-first、coreset herding等），K-means在所有数据集上一致最优
- **同时覆盖inductive和transductive两种设定**，是首个在两种范式下都有效的GNN窃取方法

## 局限与展望

- **数据分布假设**：要求攻击者拥有与victim训练数据同分布的无标签图数据 $\mathbf{G}_D$，在实践中这一假设可能不完全成立
- **仅限节点级任务**：当前仅针对节点分类，未涉及图分类或链接预测等其他GNN任务
- **Surrogate架构选择**：虽然方法不依赖架构匹配，但实验中使用GCN作为surrogate对所有victim架构可能非最优
- **防御讨论有限**：仅评估了预测翻转防御，未考虑更复杂的防御机制如水印嵌入或差分隐私

## 与相关工作的对比

- **vs Shen et al. / Podhajski et al.**: 这两个方法需要victim返回embedding（更弱的威胁模型），且假设无限查询，本方法仅需类别标签和100次查询即超越它们
- **vs datafree**: 虽不需要数据但需大量查询（25000节点），在100次查询下性能极差（Physics仅55.5%），本方法达91.2%
- **vs wu2021model**: 限于transductive设定且假设无限查询，本方法在更强约束下取得更好结果

## 启发与关联

- 随机初始化GNN encoder的有效性与Weisfeiler-Lehman图同构测试理论相关——GNN的结构感知能力部分来自架构本身而非学习到的权重
- K-means查询选择与主动学习中的diversity sampling在理念上一致，可借鉴更多主动学习策略
- 本文揭示的安全威胁对GNN as a Service (GNNaaS) 的部署有直接实际影响

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在严格查询限制下研究GNN窃取，将encoder获取与head窃取解耦的洞察有价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 8个数据集、两种设定、多种victim/surrogate架构、防御评估、McNemar检验
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法逐步展开，Table 1的方法对比一目了然
- 价值: ⭐⭐⭐⭐ — 切实揭示了GNN的安全漏洞，对模型部署的安全防护有警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[ICLR 2026\] AdS-GNN - a Conformally Equivariant Graph Neural Network](../../ICLR2026/graph_learning/ads-gnn_-_a_conformally_equivariant_graph_neural_network.md)
- [\[ICML 2025\] HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport](../../ICML2025/graph_learning/hgot_self-supervised_heterogeneous_graph_neural_network_with_optimal_transport.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[ICML 2026\] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](../../ICML2026/graph_learning/beyond_model_base_retrieval_weaving_knowledge_to_master_fine-grained_neural_netw.md)

</div>

<!-- RELATED:END -->
