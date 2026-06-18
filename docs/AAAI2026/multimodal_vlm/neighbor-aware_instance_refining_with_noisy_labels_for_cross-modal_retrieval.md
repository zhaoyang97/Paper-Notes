---
title: >-
  [论文解读] Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval
description: >-
  [AAAI 2026][信息检索/RAG][跨模态检索] 提出 NIRNL 框架，通过跨模态边距保持（CMP）增强样本区分度，并利用邻域感知实例精炼（NIR）将训练数据三分为纯净/困难/噪声子集，分别定制不同优化策略，统一了鲁棒学习、标签校准和实例选择三种范式，在高噪声率下实现了 SOTA 跨模态检索性能。
tags:
  - "AAAI 2026"
  - "信息检索/RAG"
  - "跨模态检索"
  - "噪声标签"
  - "邻域感知"
  - "实例精炼"
  - "鲁棒学习"
---

# Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval

**会议**: AAAI 2026  
**arXiv**: [2512.24064](https://arxiv.org/abs/2512.24064)  
**代码**: [GitHub](https://github.com/perquisite/NIRNL)  
**领域**: 信息检索  
**关键词**: 跨模态检索, 噪声标签, 邻域感知, 实例精炼, 鲁棒学习

## 一句话总结

提出 NIRNL 框架，通过跨模态边距保持（CMP）增强样本区分度，并利用邻域感知实例精炼（NIR）将训练数据三分为纯净/困难/噪声子集，分别定制不同优化策略，统一了鲁棒学习、标签校准和实例选择三种范式，在高噪声率下实现了 SOTA 跨模态检索性能。

## 研究背景与动机

跨模态检索（Cross-Modal Retrieval, CMR）旨在从不同模态（如图像和文本）中检索语义相关的样本。现有 CMR 方法大多依赖于精确标注的数据来学习多模态共享语义空间的表示，但实际中收集大规模、高质量标注数据既昂贵又耗时，多模态数据的标注不可避免地包含噪声。噪声标签会严重损害学习模型，削弱检索性能。

现有的鲁棒 CMR 方法大致分为三类，但各有局限：

**鲁棒学习**（如 RONO）：设计鲁棒损失函数来容忍噪声影响。但它依赖噪声分布的先验假设，只能"容忍"噪声而无法消除噪声对性能上限的限制。

**标签校准**（如 UOT-RCL）：直接纠正噪声标签。但当类别边界模糊或噪声分布与真实分布严重重叠时，可能引入新噪声或放大已有错误。

**实例选择**（如 RSHNL, NRCH）：过滤噪声样本后用干净数据训练。但对预设阈值敏感，易过滤干净样本或遗漏噪声样本，同时浪费大量训练数据。

**核心挑战**：在复杂噪声场景下，如何动态协调**模型性能上限**、**校准可靠性**和**数据利用率**三者之间的平衡？NIRNL 正是为统一解决这三个维度的问题而设计的。

## 方法详解

### 整体框架

NIRNL 由两个核心模块并行工作：

1. **CMP（Cross-modal Margin Preserving）**：在嵌入空间层面约束正负样本对的相对距离，增强表示的判别性
2. **NIR（Neighbor-aware Instance Refining）**：利用跨模态邻域共识生成软标签，将数据集精细划分为纯净、困难和噪声三个子集，并为每个子集设计定制化的优化策略

### 关键设计

1. **跨模态边距保持（CMP）**：CMP 通过 triplet-style 的 hinge loss 约束正负样本对的相对距离，使同类样本更紧凑、异类样本更分散：

$$\mathcal{L}_{CMP} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq i}^{N} |\Gamma(f_i^{\mathcal{V}}, f_j^{\mathcal{T}}) - \Gamma(f_i^{\mathcal{V}}, f_i^{\mathcal{T}}) + \mathcal{M}|_+$$

加上对称的文本→图像方向。其中 $\mathcal{M}$ 是预定义的边距，$|\cdot|_+$ 为 hinge 函数。CMP 对所有样本施加约束，作为全局的结构正则化器。

2. **邻域感知实例精炼（NIR）**：NIR 的核心思路是通过 KNN 邻域共识来评估标签可靠性。具体分为几步：

    - **软标签生成**：对每个样本 $i$，在视觉和文本模态分别找到 $K$ 个最近邻，统计邻居的类别分布作为软标签：
    $\hat{p}(c|\mathcal{V}_i) = \frac{1}{K} \sum_{k=1, \mathcal{V}_k \in \mathcal{N}_i^{\mathcal{V}}}^{K} \mathbb{I}[y_k^c = 1]$

    - **三分数据集**：根据软标签与 ground-truth 标签的一致性将样本分为三类：
        - **纯净子集** $\mathcal{D}_P$：两个模态的软标签都与 ground-truth 一致（标签高度可靠）
        - **困难子集** $\mathcal{D}_H$：仅一个模态一致（标签可靠性不确定）
        - **噪声子集** $\mathcal{D}_N$：两个模态的软标签都与 ground-truth 不一致（标签大概率错误）

    - **Wasserstein 重心提取**：通过 EM 算法计算每个类别在共享空间中的语义重心 $\bar{u}_c$，用于后续各子集的损失计算

3. **三子集差异化优化策略**：

    - **纯净子集**：直接使用交叉熵损失 $\mathcal{L}_P$ 进行优化，充分利用可靠的监督信号
    - **困难子集**：使用加权交叉熵损失 $\mathcal{L}_H$，权重 $\ell_i = 1 - (1-s(\mathcal{V}_i))(1-s(\mathcal{T}_i))$ 反映样本属于正确重心的概率，对可能被污染的标注施加更小的权重
    - **噪声子集**：首先融合两个模态的软标签进行标签校正 $\hat{y}_i = \arg\max_c \hat{p}_i^c$，然后使用鲁棒的 MAE 损失 $\mathcal{L}_N$ 来缓解标签校正可能引入的偏差

### 损失函数 / 训练策略

总体训练目标：

$$\mathcal{L} = (\mathcal{L}_P + \mathcal{L}_H + \mathcal{L}_N) + \alpha \mathcal{L}_{CMP}$$

其中前三项分别只作用于对应子集的样本，$\mathcal{L}_{CMP}$ 作用于所有样本。$\alpha$ 是平衡系数。三个子集的划分在训练过程中动态更新——随着模型表示质量的提升，邻域结构变得更加准确，数据划分也更精确，形成良性循环。

## 实验关键数据

### 主实验

在三个基准数据集（Wikipedia、XMedia、INRIA-Websearch）上，在 0.2/0.4/0.6/0.8 四种噪声率下评估。

**Wikipedia 数据集（MAP%）**：

| 噪声率 | 指标 | NIRNL | RSHNL (AAAI'25) | RONO (CVPR'23) | 提升 |
|--------|------|-------|-----------------|----------------|------|
| 0.2 | I2T / T2I | 51.6 / 46.6 | 49.1 / 45.4 | 50.5 / 47.1 | +2.5 / +1.2 |
| 0.4 | I2T / T2I | 51.7 / 46.5 | 44.3 / 41.6 | 48.8 / 45.8 | +7.4 / +4.9 |
| 0.6 | I2T / T2I | 49.2 / 46.1 | 38.3 / 36.4 | 45.3 / 41.8 | +10.9 / +9.7 |
| 0.8 | I2T / T2I | 41.7 / 39.4 | 27.8 / 26.8 | 41.6 / 38.2 | +13.9 / +12.6 |

**XMedia 数据集（均值 MAP%）**：

| 方法 | 噪声率=0.2 均值 | 噪声率=0.8 均值 | 总均值 |
|------|---------------|---------------|--------|
| NIRNL | 92.3 | 91.3 | 91.8 |
| RSHNL | 91.2 | 85.6 | 88.6 |
| RONO | 91.2 | 87.5 | 89.5 |

**INRIA-Websearch 数据集（均值 MAP%）**：

| 方法 | 噪声率=0.2 均值 | 噪声率=0.8 均值 | 总均值 |
|------|---------------|---------------|--------|
| NIRNL | 53.1 | 50.4 | 52.0 |
| RSHNL | 53.1 | 42.9 | 49.5 |
| NRCH | 43.0 | 41.3 | 42.2 |

NIRNL 在所有数据集、所有噪声率上均达到最佳结果，尤其在高噪声率（0.6、0.8）下优势更加明显。

### 消融实验

在 0.6 噪声率下的消融分析：

| 变体 | Wikipedia 均值 | XMedia 均值 | Websearch 均值 | 说明 |
|------|--------------|-------------|---------------|------|
| NIRNL-1 | 24.4 | 40.8 | 8.3 | 移除 CMP |
| NIRNL-2 | 44.8 | 88.6 | 46.7 | 丢弃噪声子集 |
| NIRNL-3 | 47.1 | 90.3 | 51.1 | 困难子集不加权 |
| NIRNL-4 | 40.5 | 90.8 | 50.4 | 移除重心对齐 |
| **NIRNL** | **47.7** | **91.8** | **52.1** | 完整框架 |

### 关键发现

- **CMP 对性能影响最大**：移除 CMP 后 Wikipedia 上性能暴跌至 24.4（降 49%），说明结构化嵌入空间是噪声鲁棒性的基础
- **噪声子集的信息不可忽视**：丢弃噪声子集（NIRNL-2）导致性能下降，说明通过标签校正可以从"坏标签"中挽回有用信息
- **困难子集的加权策略有效**：不加权（NIRNL-3）导致模型对噪声标签过于敏感
- **NIRNL 在训练后期能正确识别大部分干净样本**：随训练推进，纯净子集中真正干净样本的比例持续提升
- **RSHNL 在 Wikipedia 上出现过拟合**：因其未能捕获全局邻域分布结构

## 亮点与洞察

1. **三范式统一**：将鲁棒学习、标签校准和实例选择三种策略有机统一在一个框架中，是本文最大的创新点。通过"先分类再定制"的思路，既避免了鲁棒学习无法消除噪声的问题，又避免了标签校准可能引入新噪声的风险，还避免了实例选择浪费数据的缺陷
2. **跨模态邻域共识**：利用来自两个不同模态的邻域信息进行交叉验证，比单模态的噪声检测更加可靠。当且仅当两个模态都"投票"通过，样本才被认为是纯净的
3. **Wasserstein 重心的巧妙运用**：使用最优传输理论中的 Wasserstein 重心来提取类别语义中心，比简单的均值中心更鲁棒
4. **差异化损失设计合理**：纯净用 CE、困难用加权 CE、噪声用 MAE——MAE 对噪声标签更鲁棒的理论保证被巧妙利用
5. **实验设置全面**：四种噪声率 × 三个数据集 × 双向检索 × 10 种基线方法的对比非常充分

## 局限与展望

1. **骨干网络固定**：实验中保持特征提取器（VGG-19、AlexNet）冻结，未探索端到端微调的效果——使用更强的预训练模型（如 CLIP）可能带来进一步提升
2. **仅验证对称噪声**：实验只使用了对称标签噪声，未涉及更现实的不对称噪声或实例依赖噪声
3. **三分策略的阈值设定**：虽然使用了邻域共识而非硬阈值，但 KNN 中 $K$ 的选择仍需调参，论文未详细讨论 $K$ 的敏感性
4. **计算开销**：每个 epoch 需要计算全局 KNN 和 Wasserstein 重心，对于大规模数据集可能带来额外开销
5. **仅限图文检索**：虽然框架具有通用性，但仅在图文双模态上验证，未拓展到视频、音频等更多模态

## 相关工作与启发

- **RSHNL (AAAI'25)**：使用自步学习策略的实例选择方法，是本文最直接的对比基线，但忽略了全局邻域结构且浪费了噪声数据
- **RONO (CVPR'23)**：采用判别中心学习的鲁棒方法，只压制而不纠正噪声
- **GNN4CMR (TPAMI'23)**：图神经网络方法，在高噪声下性能急剧下降
- **启发**：邻域共识 + 三分策略的思路可推广到其他受噪声影响的多模态学习任务（如视觉问答、图文生成中的噪声配对问题）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 三范式统一的框架设计新颖，邻域共识三分法有独到之处
- **技术深度**: ⭐⭐⭐⭐ — Wasserstein 重心、差异化损失的理论基础扎实
- **实验充分性**: ⭐⭐⭐⭐⭐ — 三数据集、四噪声率、十种基线、完整消融和鲁棒性分析
- **实用价值**: ⭐⭐⭐⭐ — 有代码开源，噪声标签场景的实际需求明确
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，图表丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels](../../ICML2026/information_retrieval/care_class-adaptive_expert_consensus_for_reliable_learning_with_long-tailed_nois.md)
- [\[CVPR 2025\] NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](../../CVPR2025/information_retrieval/neighborretr_balancing_hub_centrality_in_cross-modal_retrieval.md)
- [\[CVPR 2026\] POGA: Paraphrased and Oppositional Graph Alignment for Fine-Grained Cross-Modal Retrieval](../../CVPR2026/information_retrieval/poga_paraphrased_and_oppositional_graph_alignment_for_fine-grained_cross-modal_r.md)
- [\[ACL 2025\] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](../../ACL2025/information_retrieval/maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)
- [\[CVPR 2026\] Mask to Align, Weight to Disambiguate: Reliable Unsupervised Cross-Modal Hashing with Masked-Weight Contrast](../../CVPR2026/information_retrieval/mask_to_align_weight_to_disambiguate_reliable_unsupervised_cross-modal_hashing_w.md)

</div>

<!-- RELATED:END -->
