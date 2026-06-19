---
title: >-
  [论文解读] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs
description: >-
  [AAAI 2026][图学习][图对比学习] 提出 GCL-OT 框架，首次将最优传输（OT）引入异质性文本属性图的图对比学习中，通过 RealSoftMax 相似度估计、滤波提示机制和 OT 引导的潜在同质性挖掘三个模块，分别应对部分异质性、完全异质性和潜在同质性三种多粒度异质性挑战。 文本属性图（TAG）：将文本实体作…
tags:
  - "AAAI 2026"
  - "图学习"
  - "图对比学习"
  - "最优传输"
  - "异质性图"
  - "文本属性图"
  - "多粒度异质性"
  - "RealSoftMax"
  - "Sinkhorn"
---

# GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs

**会议**: AAAI 2026  
**arXiv**: [2511.16778](https://arxiv.org/abs/2511.16778)  
**代码**: [github.com/users-01/GCL-OT](https://github.com/users-01/GCL-OT)  
**领域**: 图学习 / 图对比学习  
**关键词**: 图对比学习, 最优传输, 异质性图, 文本属性图, 多粒度异质性, RealSoftMax, Sinkhorn  

## 一句话总结

提出 GCL-OT 框架，首次将最优传输（OT）引入异质性文本属性图的图对比学习中，通过 RealSoftMax 相似度估计、滤波提示机制和 OT 引导的潜在同质性挖掘三个模块，分别应对部分异质性、完全异质性和潜在同质性三种多粒度异质性挑战。

## 研究背景与动机

**文本属性图（TAG）** 将文本实体作为节点、关系作为边，广泛应用于学术引用、电商推荐等场景。现有方法通常结合 GNN 与语言模型（LM），通过对比学习对齐结构与文本表示。

然而，**异质性 TAG** 在现实中普遍存在（如约会网络中异类相吸），现有方法面临三大挑战：

**部分异质性**（Partial Heterophily）：节点文本与邻居仅部分语义对齐，现有方法难以捕捉细粒度语义不匹配

**完全异质性**（Complete Heterophily）：节点文本与邻居完全无关（如随机共购），SOTA 模型会被此类噪声误导

**潜在同质性**（Latent Homophily）：语义相似但未连接的潜在邻居，因边缺失或隐式链接而被忽略

传统 InfoNCE 仅支持 1:1 硬对齐，DGI 的 N:1 方式同样不足。**最优传输天然支持 N:N 软对齐**，可分配部分质量实现灵活匹配。

## 方法详解

### 整体框架

GCL-OT 由四个阶段组成：
1. **多视图特征编码**：LLM 增强文本 → PLM 编码文本（token 级 + 句子级）；GNN 编码结构
2. **层次化 OT 对齐**：RealSoftMax 处理部分异质性 + Filter-Prompt 处理完全异质性
3. **潜在同质性挖掘**：OT 分配矩阵作为辅助监督
4. **联合优化**：对比损失 + 节点分类损失

### 关键设计一：RealSoftMax 相似度估计（部分异质性）

对节点 vᵢ 的第 k 个邻居嵌入和节点 vⱼ 的第 w 个词嵌入，定义双向相似度：

$$s_{ij} = \frac{1}{2}\left(\mathbb{E}_k[\text{RSM}_\beta(\{h_{ik}^{\mathcal{N}} \cdot h_{jw}^{\varpi}\}_{w=1}^W)] + \mathbb{E}_w[\text{RSM}_\beta(\{h_{iw}^{\varpi} \cdot h_{jk}^{\mathcal{N}}\}_{k=1}^K)]\right)$$

其中 RSM_β 在 β→0 时退化为 max，β→∞ 时退化为 mean，实现平滑插值。第一项找到每个邻居最相关的词，第二项反之，双向强调信息性交互并抑制背景噪声。

### 关键设计二：Filter-Prompt 全局过滤（完全异质性）

将结构-文本全局相似度矩阵 S 扩展为 (N+1)×(N+1) 矩阵：

$$\bar{S} = \begin{bmatrix} S & z \\ z^\top & z_{N+1} \end{bmatrix}$$

其中 z 为可学习的提示向量。若某嵌入的最大相似度低于对应 z 值，则在 OT 对齐中将其与提示向量对齐而非其他节点，从而自适应排除无关噪声。

OT 问题通过低秩 Sinkhorn（LRSinkhorn）高效求解，复杂度从 O(N²) 降至 O(Nr)。

### 关键设计三：OT 引导的潜在同质性挖掘

利用全局 OT 分配矩阵 Q̂* 作为软监督信号，构建对比目标：

$$P = I + \hat{Q}^*$$

P 归一化后作为软标签，ℒ_LHM 损失鼓励语义相似但未连接的节点嵌入靠近，避免将潜在正样本误判为负样本。

### 损失函数

总损失由三部分组成：

$$\mathcal{L} = \mathcal{L}_{NC} + \lambda \mathcal{L}_{GCL\text{-}OT}$$

- ℒ_NC：节点分类的交叉熵损失
- ℒ_GCL-OT = ℒ_MHA + ℒ_LHM：多层对齐损失 + 潜在同质性挖掘损失

理论证明 ℒ_MHA 和 ℒ_LHM 均提供比标准 InfoNCE 更紧的互信息下界。

## 实验

### 主实验：节点分类准确率（%）

| 方法 | Cora | PubMed | Products | Wisconsin | Cornell | Texas |
|------|------|--------|----------|-----------|---------|-------|
| GCN | 89.11 | 85.33 | 75.64 | 46.98 | 44.36 | 54.21 |
| TAPE-RevGAT | 92.80 | 96.04 | 79.76 | 87.77 | 88.46 | 85.90 |
| ENGINE-LLAMA | 91.48 | 95.24 | 80.05 | 85.50 | 77.36 | 75.68 |
| **GCL-OT-GCN** | **93.54** | 96.08 | 81.50 | 88.68 | 88.64 | **89.47** |
| **GCL-OT-SAGE** | **93.73** | **96.62** | 81.73 | **89.26** | 88.21 | **90.01** |

GCL-OT 在 9 个数据集上全面超越基线，尤其在异质性数据集（Wisconsin/Cornell/Texas）上提升显著。

### 消融实验

| 变体 | 效果 |
|------|------|
| 去除 ℒ_GCL-OT | 性能显著下降，对比学习对结构-文本对齐至关重要 |
| 去除 ℒ_MHA | 异质性数据集（Texas、Cornell）显著下降 |
| 去除 ℒ_LHM | 弱结构信号数据集（Amazon）性能下降 |
| 完整模型 | 所有数据集最佳，三个组件互补 |

### 关键发现

1. **格式异质性是核心挑战**：同质 InfoNCE 替换 GCL-OT 后，在混合标签邻域和强语义异质性场景下性能显著下降
2. **鲁棒性优势**：边扰动（删除 500 条边）下 GCL-OT+GCN 相比 vanilla GCN 相对提升 24.74%；文本扰动下稳定保持 ~85% 准确率
3. **无监督设定有效**：无标签对比预训练 + 线性探针在 Wisconsin 达到 72.83%，超越 PolyGCL、HeterGCL 等专用方法
4. **训练效率良好**：GCL-OT+GCN 在精度最高的同时训练时间与 TAPE 相当

## 亮点

- **首次将最优传输引入异质性 TAG 的图对比学习**，概念创新且方法完整
- **多粒度异质性的精细分析**（部分/完全/潜在）——每种模式都有针对性模块
- RealSoftMax 的温度参数 β 优雅地在 max 和 mean 之间插值，简洁有效
- Filter-Prompt 机制巧妙利用可学习提示向量"吸收"完全异质性噪声
- 理论保证：证明了比 InfoNCE 更紧的互信息界和更低的 Bayes 错误上界

## 局限性

- 对 LLM 文本增强（GPT-3.5）有依赖，增加部署成本和隐私风险
- LRSinkhorn 的低秩近似在极端异质性下的精度损失未深入分析
- 仅评估 DistilBERT 等轻量 PLM，未测试更强的 LLM 编码器对性能上限的影响
- 子图采样可能损失全局信息，大规模图上的可扩展性验证有限（最大 ~170K 节点）
- 超参数（λ, β, Sinkhorn 迭代次数）需调优，虽论文显示对 β 敏感度较低

## 相关工作

- **TAG 学习**：TAPE (He et al. 2024) 用 LLM 解释增强 GNN；ConGraT (Brannon et al. 2024) 对比预训练文本+图；SimTeG 展示好的文本嵌入即可大幅提升 GNN
- **异质图 GCL**：PolyGCL 用谱多项式滤波器对比不同同质性；HeterGCL 用结构+语义模块利用标签不一致信号
- **OT + GCL**：THESAURUS 用 Gromov-Wasserstein 对齐节点嵌入与语义原型；FOSSIL 用融合 GW 距离做子图级对比

## 评分

⭐⭐⭐⭐ (4/5)

问题定义清晰（多粒度异质性），方法设计精巧（三个模块各有针对性），实验全面（9 个数据集、有监督+无监督、消融、鲁棒性）。理论分析为方法提供了互信息和 Bayes 错误的形式保证。主要扣分在于对 LLM 增强的依赖和大规模可扩展性验证不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[ICML 2025\] HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport](../../ICML2025/graph_learning/hgot_self-supervised_heterogeneous_graph_neural_network_with_optimal_transport.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](../../NeurIPS2025/graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[ICML 2026\] ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs](../../ICML2026/graph_learning/eralign_energy-based_representation_alignment_of_gnns_and_llms_on_text-attribute.md)

</div>

<!-- RELATED:END -->
