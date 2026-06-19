---
title: >-
  [论文解读] FicGCN: Unveiling the Homomorphic Encryption Efficiency from Irregular Graph Convolutional Networks
description: >-
  [ICML 2025][AI安全][同态加密] 提出FicGCN框架，通过延迟感知的打包策略、稀疏密文内聚合（SpIntra-CA）和基于区域的节点重排三项创新，解决GCN不规则稀疏性与同态加密SIMD计算模式之间的根本矛盾，在Corafull等大规模图上实现最高4.10×的端到端加速。 领域现状： GCN在医疗、金融等隐私…
tags:
  - "ICML 2025"
  - "AI安全"
  - "同态加密"
  - "图卷积网络"
  - "CKKS"
  - "稀疏聚合"
  - "隐私保护推理"
---

# FicGCN: Unveiling the Homomorphic Encryption Efficiency from Irregular Graph Convolutional Networks

**会议**: ICML 2025  
**arXiv**: [2506.10399](https://arxiv.org/abs/2506.10399)  
**代码**: 无  
**领域**: 隐私计算 / 同态加密推理  
**关键词**: 同态加密, 图卷积网络, CKKS, 稀疏聚合, 隐私保护推理

## 一句话总结

提出FicGCN框架，通过延迟感知的打包策略、稀疏密文内聚合（SpIntra-CA）和基于区域的节点重排三项创新，解决GCN不规则稀疏性与同态加密SIMD计算模式之间的根本矛盾，在Corafull等大规模图上实现最高4.10×的端到端加速。

## 研究背景与动机

**领域现状**: GCN在医疗、金融等隐私敏感场景中广泛应用，CKKS同态加密方案允许在密文上直接计算以保护隐私，但引入超过三个数量级的计算开销。

**现有痛点**: (1) 同态加密下的GCN核心瓶颈是矩阵乘法 $\hat{A}XW$ 中的旋转（Rotation）和乘法操作；(2) GCN邻接矩阵 $A$ 天然稀疏，但这种不规则稀疏与CKKS的SIMD并行模式严重冲突——稀疏乘法需要按节点进行不同的聚合，而SIMD要求整个密文向量统一操作；(3) 现有方案CryptoGCN利用了稀疏性但密文槽利用率低，Penguin打包效率高但未利用稀疏性。

**核心矛盾**: GCN聚合阶段的按节点不规则计算模式 vs CKKS密文的SIMD整体操作要求——如何在完全打包的密文上高效执行不规则稀疏聚合？

**本文目标**: 在CKKS同态加密下，同时利用GCN稀疏性和高密文槽利用率来最小化推理延迟。

**切入角度**: 将密文内部旋转（intra-ciphertext rotation）类比于内部求和技巧，用比特分解实现 $O(\log N)$ 次旋转完成所有节点的邻居聚合。

**核心 idea**: 密文内部用对数次旋转实现稀疏聚合，配合面向聚合效率优化的节点排列，突破稀疏性与SIMD的冲突。

## 方法详解

### 整体框架

FicGCN工作流程包含三个核心模块：(1) **延迟感知打包**：根据数据维度和HE操作延迟比，优化每个密文打包的列数 $t$，在聚合和组合两阶段间取得全局最优平衡；(2) **SpIntra-CA**：密文内部通过比特分解旋转实现稀疏邻居聚合，将 $O(N)$ 旋转压缩至 $O(\log^2 N)$；(3) **NOO节点重排**：基于BFS的区域划分+交错排列+贪心冲突最小化，进一步降低旋转开销。推理阶段自动选择每层最优聚合模式（SpIntra-CA或Inter-CA）。

### 关键设计

1. **稀疏密文内聚合（SpIntra-CA）**:

    - 功能：在密文内部通过旋转操作高效构建"邻居密文"实现所有节点的并行聚合
    - 核心思路：将每个节点到其邻居位置的旋转距离做比特分解，按位从低到高依次执行 $2^0, 2^1, \ldots, 2^{\lfloor\log M\rfloor}$ 步旋转，每步用Mask明文选择需要/不需要旋转的节点。迭代公式：$\{ct\} \leftarrow \{ct\} \otimes Mask_1 \oplus Rot(\{ct\}, 2^{m-1}) \otimes Mask_2$。理想情况下仅需 $\log(M)$ 次旋转
    - 设计动机：朴素方法逐节点提取需 $O(N)$ 次旋转且每次仅对一个节点有效；借鉴密文内部求和的比特分解技术，让每次旋转同时服务于多个节点，效率从 $O(N)$ 降至 $O(\log^2 N)$

2. **节点顺序优化（NOO）**:

    - 功能：优化密文中节点的排列顺序以最小化SpIntra-CA的旋转距离和冲突次数
    - 核心思路：三步流程——(1) BFS区域划分（阈值TH限制区域大小）；(2) 区域间交错排列使每个区域成为密文的子环，保持旋转一致性；(3) 区域内贪心排列：逐节点选择与已固定节点冲突最少的位置
    - 设计动机：节点在密文中的排列直接决定旋转距离和冲突频率——邻居节点越近旋转越短，区域内冲突越少额外密文越少

3. **延迟感知打包与聚合模式调度**:

    - 功能：逐层自动选择最优密文打包参数和聚合模式
    - 核心思路：当 $M > N \cdot F'$ 时优化目标 $\mathcal{J}(t;F,n) = 2\lceil F \cdot n/t \rceil + 20\lceil\log(t)\rceil$；逐层比较Inter-CA（$2\lceil Fn/t \rceil$ Rot）和SpIntra-CA（$10cn\log^2(N)$ Rot），取延迟更低者
    - 设计动机：Rotation操作延迟是乘法/加法的20倍以上，打包方式直接影响Rot次数——全局优化比固定策略高效

### 损失函数 / 训练策略

FicGCN聚焦推理阶段加速，不改变模型训练流程。GCN模型使用Adam优化器在明文上训练（batch=64, lr=0.01, 200 epoch），特征维度32→16。CKKS参数设置128-bit安全性（$\Delta=2^{30}, M=2^{12}$）。所有实验在单线程Intel i7-9750H上测试延迟。

## 实验关键数据

### 主实验

端到端推理延迟对比（加速倍数以Gazelle为基准）：

| 数据集 | 节点数 | Gazelle | Penguin | CryptoGCN | FicGCN+NOO | 加速 |
|--------|:------:|--------:|--------:|----------:|-----------:|:----:|
| Cora | 2,708 | 1535.3s | 128.8s | 131.1s | **64.1s** | 21.6× |
| Citeseer | 3,327 | 2897.3s | 142.9s | 150.4s | **80.0s** | 36.2× |
| Corafull | 19,793 | / | 35565s | 31735s | **7733s** | >120× |
| NTU | 25 | - | - | 1731.1s | **1373.8s** | 1.26× |

### 消融实验

各组件对Cora数据集的贡献：

| 组件 | Rot数 | 延迟(s) | 说明 |
|------|:-----:|:------:|------|
| Inter-CA基线 | 0 | 70.08 | 无稀疏利用 |
| SpIntra-CA (w/o AOO) | 7.04K | 47.65 | 稀疏聚合有效 |
| + AOO | 5.74K | 40.39 | 旋转减少18.5% |
| + CPOO | - | ≈34 | 进一步剪枝14-46% Rot |
| + NOO | 1.93K | **69.28** (全pipeline) | Rot再降66% |

### 关键发现

1. **大规模图优势更显著**: Corafull（19793节点）上FicGCN比CryptoGCN快4.1×，比Penguin快4.6×
2. **密文槽利用率始终100%**: 对比CryptoGCN的66-90%，FicGCN完全消除浪费
3. **NOO对比普通重排（Rabbit）优势明显**: Rot数从2.85K降至1.93K，针对HE环形结构专门优化
4. **模型精度无损**: Cora上加密推理精度0.792 vs 明文backbone 0.815，差距可接受

## 亮点与洞察

- **直击根本矛盾**: 不规则稀疏性vs SIMD是HE+GCN的核心难题，三个模块分别解决打包、聚合、排列层面的冲突
- **比特分解旋转的insight**: 将密文内部求和技巧推广到GCN稀疏聚合——一次旋转服务多节点
- **可扩展性**: 大规模图（2万节点）上优势尤为明显，因为SpIntra-CA的 $O(\log^2 N)$ 复杂度对大图更友好
- **端到端优化**: 不仅优化单一模块，而是全局调度打包×聚合×排列

## 局限与展望

- 仅在GAE/STGCN两种架构上验证，未测试更复杂的GNN变体（如GAT、GIN）
- 单线程CPU测试，未探索GPU/多线程并行潜力
- NOO的贪心搜索对超大规模图（百万节点）可能计算开销较大
- 未与MPC（多方安全计算）方案进行公平对比
- 非线性层（$x^2$近似）对模型精度的影响未深入分析

## 相关工作与启发

- **vs CryptoGCN**: 利用稀疏但密文利用率仅66-83%——FicGCN同时达到100%利用率和稀疏加速
- **vs Penguin**: 高效打包但无稀疏利用——FicGCN通过SpIntra-CA补充稀疏维度
- **vs Gazelle**: 通用HE矩阵乘法——FicGCN是GCN专用优化，在Corafull上快120×+
- **启发**: CKKS密文的环形结构可以通过精心设计的节点排列来"适配"图的拓扑结构——密码学特性与计算图拓扑的深度融合

## 评分

- 新颖性: ⭐⭐⭐⭐ SpIntra-CA的比特分解旋转思路新颖，NOO针对CKKS环结构的设计有创意
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、四种SOTA对比、完整消融实验
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，图示辅助理解好
- 价值: ⭐⭐⭐⭐ 对隐私保护GCN推理有实际加速意义，大规模图场景尤其有用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](../../NeurIPS2025/ai_safety/influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[ECCV 2024\] Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](../../ECCV2024/ai_safety/unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
