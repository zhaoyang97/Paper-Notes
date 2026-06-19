---
title: >-
  [论文解读] SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding
description: >-
  [ICML2025][AI安全][联邦学习] 提出 SecEmb，一种利用嵌入更新稀疏性的无损安全联邦推荐协议，通过函数秘密共享（FSS）在保护用户评分物品索引和梯度隐私的同时，将上传/下载通信开销降低最高 90 倍、用户端计算时间降低最高 70 倍。 联邦推荐系统（FedRec）通过分布式训练保护用户数据…
tags:
  - "ICML2025"
  - "AI安全"
  - "联邦学习"
  - "推荐系统"
  - "安全聚合"
  - "稀疏嵌入"
  - "函数秘密共享"
  - "隐私保护"
---

# SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding

**会议**: ICML2025  
**arXiv**: [2505.12453](https://arxiv.org/abs/2505.12453)  
**代码**: [NusIoraPrivacy/SecEmb](https://github.com/NusIoraPrivacy/SecEmb)  
**领域**: AI安全 / 联邦推荐  
**关键词**: 联邦学习, 推荐系统, 安全聚合, 稀疏嵌入, 函数秘密共享, 隐私保护

## 一句话总结

提出 SecEmb，一种利用嵌入更新稀疏性的无损安全联邦推荐协议，通过函数秘密共享（FSS）在保护用户评分物品索引和梯度隐私的同时，将上传/下载通信开销降低最高 90 倍、用户端计算时间降低最高 70 倍。

## 研究背景与动机

联邦推荐系统（FedRec）通过分布式训练保护用户数据，但面临以下核心矛盾：

- **通信瓶颈**：传统 FedRec 需要上传/下载完整物品嵌入矩阵 $Q \in \mathbb{R}^{m \times d}$，通信量随物品数 $m$ 线性增长，对带宽受限的边缘设备造成巨大压力
- **隐私泄露风险**：现有稀疏感知的联邦协议（如 FMSS、SparseSecAgg）虽能减少通信量，但不可避免地暴露非零更新的坐标（即用户评分了哪些物品），在推荐场景中这些坐标直接等价于用户行为隐私
- **关键稀疏机会**：实际中用户只与极小部分物品交互（如 ML10M 中 90% 用户评分物品 ≤ 300，而物品总量 27,278），嵌入梯度矩阵高度稀疏

核心问题：能否同时实现 (1) 通信/计算量仅依赖于用户交互物品数 $m'$（而非总物品数 $m$），(2) 服务器对个体用户评分索引和梯度值一无所知（仅获得聚合结果）？

## 方法详解

SecEmb 由两个协作模块组成，均基于 **函数秘密共享（FSS）** 构建：

### 模块一：隐私保护嵌入检索

用户仅需下载自己交互物品的嵌入，而非整个嵌入矩阵，同时对服务器隐藏目标物品索引。

**核心思路**：将每个评分物品编码为点函数，利用 FSS 实现两服务器私有检索。

1. 用户 $u$ 将每个评分物品 $i$ 编码为点函数 $f_{u,i}: \mathcal{I} \to \mathbb{G}$，当 $x = i$ 时输出 1，否则输出 0
2. 用 FSS 生成密钥对 $(reK_{u,i}^0, reK_{u,i}^1)$，分别发送给两个服务器
3. 每个服务器 $s$ 计算秘密份额：

$$\mathbf{v}_{u,i}^s = \sum_j \text{FSS.Eval}(reK_{u,i}^s, j) \cdot Q_j$$

4. 用户收到两个份额后重建目标嵌入：$Q_{\text{idx}(i)} = \mathbf{v}_{u,i}^0 + \mathbf{v}_{u,i}^1$

### 模块二：安全梯度聚合

利用稀疏梯度矩阵的行稀疏结构，将非零梯度行编码为点函数进行安全聚合。

**行级编码优化**：将每个非零行整体编码为一个点函数（而非逐元素编码），从 $m'd$ 个密钥减少到 $m'$ 个：

$$f_{u,i}(x) = \begin{cases} \mathbf{g}_{Q_i^u} \in \mathbb{R}^d & \text{if } x = \text{idx}(i) \\ \mathbf{0} \in \mathbb{R}^d & \text{otherwise} \end{cases}$$

服务器聚合：$\mathbf{v}_{Q_j}^s = \sum_u \sum_{i \in [m']} \text{FSS.Eval}(agK_{u,i}^s, j)$

### 路径共享优化

关键观察：检索阶段和聚合阶段的 FSS 密钥针对相同的物品索引，因此密钥的前半部分（确定非零位置的种子与修正词）可完全复用。聚合阶段用户只需生成并传输最终修正词 $CW^{n+1}$，节省 $n$ 次 AES 操作。

### 复杂度分析

| 指标 | SecEmb | 传统安全 FedRec |
|------|--------|----------------|
| 下载通信 | $O(m'bd + |\theta|b)$ | $O(mbd + |\theta|b)$ |
| 上传通信 | $O(m'(\lambda \log m + bd) + |\theta|b)$ | $O(mbd + |\theta|b)$ |
| 用户计算 | $O(m' \log m \cdot \text{AES} + |\theta|)$ | $O(md + |\theta|)$ |

其中 $m' \ll m$ 是用户交互物品数，$\lambda$ 是安全参数。通信量从 $O(m)$ 降至 $O(m' \log m)$。

## 实验关键数据

### 实验设置
- **数据集**：ML100K、ML1M、ML10M、ML25M、Yelp（物品数从 1,682 到 93,386）
- **模型**：MF、NCF、FM、DeepFM
- **对比方法**：SecFedRec（两服务器 ASS）、SVD、CoLR、8bit 量化、三值量化

### 通信开销降低（上传，每用户每轮，MB）

| 数据集 | 物品数 | SecFedRec | SecEmb | 降低倍数 |
|--------|--------|-----------|--------|---------|
| ML100K | 1,682 | 0.86 | 0.17 | **5.0x** |
| ML1M | 3,883 | 1.99 | 0.27 | **7.4x** |
| ML10M | 10,681 | 5.47 | 0.28 | **19.2x** |
| ML25M | 62,423 | 31.96 | 0.51 | **62.1x** |
| Yelp | 93,386 | 47.81 | 0.52 | **91.2x** |

（以 MF 模型为例，FM 有类似趋势）

### 无损 vs 有损压缩（MF, RMSE↓越低越好）

| 方法 | ML25M RMSE | ML25M 压缩比 | Yelp RMSE | Yelp 压缩比 |
|------|-----------|-------------|----------|------------|
| 8bit量化 | 0.870 | 4.0x | 1.050 | 4.0x |
| 三值量化 | 0.874 | 8.0x | 1.050 | 8.0x |
| SVD | 0.873 | 16.2x | 1.050 | 16.2x |
| CoLR | 0.873 | 16.3x | **1.049** | 16.3x |
| **SecEmb** | **0.864** | **62.1x** | 1.050 | **91.2x** |

SecEmb 在大规模数据集上通信压缩比远超有损方法，且精度无损甚至略优（因无信息损失）。

### 计算时间

用户端密钥生成时间在 Yelp 数据集上实现约 **70x** 加速（MF/FM），随物品数增大优势更加显著。

## 亮点与洞察

1. **无损压缩的优雅设计**：通过 FSS 将稀疏更新压缩为紧凑密钥，无需量化或降维，零精度损失
2. **行级编码 + 路径共享**：两个关键优化将用户上传密钥数从 $m'd$ 降至 $m'$，并复用检索阶段的 AES 计算路径
3. **隐私保障完整**：在半诚实两服务器模型下，服务器无法获知用户评分了哪些物品（索引隐私）及具体梯度值，仅获得聚合结果；形式化安全证明基于 FSS 安全性
4. **可组合 DP**：协议可与差分隐私无缝集成，每个服务器独立对聚合份额加噪即可实现 $(\epsilon, \delta)$-DP
5. **实际收益随规模增大而放大**：物品数越大，稀疏优势越明显（Yelp 91x vs ML100K 5x）

## 局限与展望

1. **两服务器非共谋假设**：安全性依赖两个服务器不串通，这在实际部署中可能难以保证；若放松为单服务器或恶意对手模型，协议需大幅修改
2. **服务器端计算开销增加**：FSS 评估需要服务器对所有物品执行 AES 操作，计算量为 $O(nm'm \cdot \text{AES})$，当用户和物品数都很大时可能成为瓶颈
3. **仅针对嵌入层优化**：对于 DeepFM 等含大量稠密参数的模型，嵌入层占比不高时优势减弱（ML100K 上 DeepFM 仅 1.05x）
4. **统一 $m'$ 的效率损失**：为隐藏真实交互数，所有用户需统一到相同的 $m'$，对轻度用户存在不必要的填充开销
5. **实验未涉及恶意用户场景**：未考虑拜占庭用户提交恶意梯度的鲁棒性问题

## 相关工作与启发

- **FedMF / FedNCF**：早期安全联邦推荐，使用同态加密或通用 SecAgg，通信量随物品数线性增长
- **LightSecAgg / FastSecAgg**：优化 SecAgg 的计算复杂度，但未利用梯度稀疏性
- **FMSS**：稀疏感知联邦推荐框架，但暴露非零坐标（即评分物品索引）
- **FSS (Boyle et al., 2015/2016)**：函数秘密共享的理论基础，本文将其首次应用于联邦推荐的稀疏嵌入场景
- **启发**：FSS 的点函数编码思路可推广到其他具有结构化稀疏的联邦场景（如 NLP 中的词嵌入、图神经网络中的节点嵌入）

## 评分

- 新颖性: ⭐⭐⭐⭐ — FSS 在联邦推荐中的应用具有原创性，行级编码和路径共享优化设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 5 个数据集 × 4 个模型，通信/计算/精度全面评估，对比充分
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，从初始构造到优化版本的递进结构易于理解
- 价值: ⭐⭐⭐⭐ — 解决了安全联邦推荐中效率与隐私不可兼得的核心矛盾，实际意义显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [\[NeurIPS 2025\] Environment Inference for Learning Generalizable Dynamical System](../../NeurIPS2025/ai_safety/environment_inference_for_learning_generalizable_dynamical_system.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](../../CVPR2026/ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[ICML 2025\] Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)

</div>

<!-- RELATED:END -->
