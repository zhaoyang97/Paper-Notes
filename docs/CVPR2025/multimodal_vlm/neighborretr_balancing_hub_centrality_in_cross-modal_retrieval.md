---
title: >-
  [论文解读] NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval
description: >-
  [CVPR 2025][信息检索/RAG][跨模态检索] 提出 NeighborRetr，通过三重机制解决跨模态检索中的 Hubness 问题（少数样本垄断近邻）：中心性加权损失（降低 hub 样本的训练权重）、邻域调整损失（区分好/坏 hub）和均匀正则化（确保每个样本被公平检索），在 MSR-VTT 文本→视频 R@1 达 49.5%（+0.9% SOTA）。
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "跨模态检索"
  - "Hub问题"
  - "中心性加权"
  - "邻域调整"
  - "均匀正则化"
---

# NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval

**会议**: CVPR 2025  
**arXiv**: [2503.10526](https://arxiv.org/abs/2503.10526)  
**代码**: [https://github.com/zzezze/NeighborRetr](https://github.com/zzezze/NeighborRetr)  
**领域**: 信息检索  
**关键词**: 跨模态检索, Hub问题, 中心性加权, 邻域调整, 均匀正则化

## 一句话总结

提出 NeighborRetr，通过三重机制解决跨模态检索中的 Hubness 问题（少数样本垄断近邻）：中心性加权损失（降低 hub 样本的训练权重）、邻域调整损失（区分好/坏 hub）和均匀正则化（确保每个样本被公平检索），在 MSR-VTT 文本→视频 R@1 达 49.5%（+0.9% SOTA）。

## 研究背景与动机

### 领域现状

**领域现状**：跨模态检索（如文本→视频、文本→图像）将不同模态的数据映射到共享嵌入空间。然而高维嵌入空间存在 Hubness 问题——少数样本成为大量查询的近邻（hub），而多数样本几乎不被检索到（anti-hub）。

**现有痛点**：（1）Hub 样本中混杂"好 hub"（语义确实相关）和"坏 hub"（仅因空间位置特殊），不能简单抑制所有 hub；（2）anti-hub 的存在导致大量相关样本永远不会被检索到；（3）现有对比学习对 hub/anti-hub 差异视而不见。

**核心矛盾**：单纯抑制 hubness 会误杀好 hub（真正相关的高频样本），不抑制则坏 hub 垄断检索结果。

**切入角度**：用记忆库在线估计每个样本的中心性（被检索频率），然后通过不同机制分别处理好hub/坏hub/anti-hub。

**核心 idea**：中心性加权 + 好/坏 hub 区分 + anti-hub 均匀正则化 = 解决跨模态 hubness。

### 解决思路

**本文目标**：### 关键设计

1. **中心性加权损失**：$w(x_i) = \exp(C(x_i)/\kappa)$，高中心性（hub）样本在对比损失中权重降低，减少其对学习的主导效应

2. **邻域调整损失**：用"去中心化相似度"区分好/坏 hub——好 hub 的去中心化相似度高（在减去中心性后仍相关），坏 hub 低

3. **均匀正则化**：$\mathcal{L}_{Opt}$ 强制检索。


## 方法详解

### 关键设计

1. **中心性加权损失**：$w(x_i) = \exp(C(x_i)/\kappa)$，高中心性（hub）样本在对比损失中权重降低，减少其对学习的主导效应

2. **邻域调整损失**：用"去中心化相似度"区分好/坏 hub——好 hub 的去中心化相似度高（在减去中心性后仍相关），坏 hub 低

3. **均匀正则化**：$\mathcal{L}_{Opt}$ 强制检索概率分布趋向均匀，确保 anti-hub 也有被检索的机会

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_{Wti} + \mathcal{L}_{Nbi} + \mathcal{L}_{Opt}$ + 细粒度 WTI 模块。记忆库用于在线估计中心性。

## 实验关键数据

| 基准 | R@1 | Rsum | 说明 |
|------|-----|------|------|
| MSR-VTT T→V | **49.5%** | **207.7** | +0.9 vs HBI |
| MSR-VTT V→T | **48.7%** | **207.5** | +1.9 |
| MSVD T→V | SOTA | — | 多基准一致最优 |

### 消融实验
- 坏 hub 减少、好 hub 增强、anti-hub 最小化——三者协同
- 分离模态内/跨模态加权提升稳定性
- 均匀正则化对低频样本的提升最大

### 关键发现
- Hub 问题在跨模态检索中是系统性的——不解决会有 3-5% 的性能天花板
- 好/坏 hub 的区分是关键创新——比简单抑制所有 hub 好 1-2%

## 亮点与洞察
- **首次系统解决跨模态 hubness**——从理论（中心性度量）到实践（三重损失）
- **好/坏 hub 区分**——不是所有高频样本都是坏的，语义相关的 hub 是有价值的

## 局限与展望
- 超参数 κ 需逐数据集调
- 记忆库大小影响效率
- 假设单正样本查询

## 评分
- 新颖性: ⭐⭐⭐⭐ 跨模态 hubness 的系统解决方案新颖
- 实验充分度: ⭐⭐⭐⭐ 7 个基准
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为跨模态检索提供了被忽视问题的解法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](../../ACL2025/information_retrieval/maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)
- [\[AAAI 2026\] Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval](../../AAAI2026/information_retrieval/neighbor-aware_instance_refining_with_noisy_labels_for_cross-modal_retrieval.md)
- [\[ACL 2025\] CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](../../ACL2025/information_retrieval/cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)
- [\[CVPR 2026\] POGA: Paraphrased and Oppositional Graph Alignment for Fine-Grained Cross-Modal Retrieval](../../CVPR2026/information_retrieval/poga_paraphrased_and_oppositional_graph_alignment_for_fine-grained_cross-modal_r.md)
- [\[CVPR 2026\] Mask to Align, Weight to Disambiguate: Reliable Unsupervised Cross-Modal Hashing with Masked-Weight Contrast](../../CVPR2026/information_retrieval/mask_to_align_weight_to_disambiguate_reliable_unsupervised_cross-modal_hashing_w.md)

</div>

<!-- RELATED:END -->
