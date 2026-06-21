---
title: >-
  [论文解读] Multi-Scale Diffusion-Guided Graph Learning with Power-Smoothing Random Walk Contrast for Multi-View Clustering
description: >-
  [ICLR 2026][图学习][Multi-View Clustering] 提出 MANGO 框架，用「熵引导的多尺度图扩散」动态融合不同步长的相似度矩阵以兼顾局部与全局结构，再用「随机游走 + β 幂平滑」纠正对比学习中的假负样本，并通过共享结构嵌入的视图一致性模块缓和一致性与特异性的矛盾，在 12 个数据集上刷新多视图聚类 SOTA。
tags:
  - "ICLR 2026"
  - "图学习"
  - "Multi-View Clustering"
  - "扩散模型"
  - "对比学习"
  - "Random Walk"
  - "False Negatives"
---

# Multi-Scale Diffusion-Guided Graph Learning with Power-Smoothing Random Walk Contrast for Multi-View Clustering

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ynT6rqo4Lp](https://openreview.net/forum?id=ynT6rqo4Lp)  
**代码**: 待确认  
**领域**: 图学习 / 多视图聚类 / 对比学习  
**关键词**: Multi-View Clustering, Graph Diffusion, Contrastive Learning, Random Walk, False Negatives  

## 一句话总结
提出 MANGO 框架，用「熵引导的多尺度图扩散」动态融合不同步长的相似度矩阵以兼顾局部与全局结构，再用「随机游走 + β 幂平滑」纠正对比学习中的假负样本，并通过共享结构嵌入的视图一致性模块缓和一致性与特异性的矛盾，在 12 个数据集上刷新多视图聚类 SOTA。

## 研究背景与动机
- **领域现状**：基于图的深度多视图聚类（GDMVC）通过显式建模样本拓扑关系，已成为整合多源异构信息、发现潜在簇结构的主流路线，常配合对比学习来精炼图结构。
- **现有痛点**：作者归纳出三个长期未解的技术难题——(1) 依赖**静态图结构**，只用局部邻域算相似度，无法建模跨视图的全局语义关联，导致信息损失与失真；(2) 对比学习中的**假负样本污染**，把语义相近的同类样本误当负样本，错误信号经梯度反传不断累积形成"噪声优化"正反馈，持续侵蚀相似度质量；(3) **一致性与特异性的两难**，过度对齐损害模态独特性、对齐不足又破坏跨视图语义对应，最终模糊簇边界。
- **核心矛盾**：固定扩散步长无法同时刻画"直接相连样本的局部细节"和"远距样本的全局语义"，而无标签场景下又难以保证负样本真的不相关，二者叠加让对比信号既不够全局又含噪。
- **本文目标**：在统一框架内同时解决静态图、假负样本、一致性-特异性三大难题，得到更鲁棒、更具语义表达力的图结构与聚类划分。
- **核心idea**：**多尺度扩散**用熵作为质量度量动态加权融合多步扩散矩阵；**随机游走纠正 + 幂平滑**重塑对比目标分布以过滤假负样本；**结构感知视图一致性**靠共享结构嵌入对齐语义、同时保留视图特异判别特征。

## 方法详解

### 整体框架
MANGO 由四个模块串联：自表达模块先用逐视图 MLP 编码并通过稀疏自重构得到保留局部几何的嵌入；幂平滑随机游走对比模块在嵌入上做去假负的对比学习；视图一致性模块用共享映射对齐跨视图语义；最后熵引导的多尺度扩散把精炼后的亲和矩阵融合成下游谱聚类用的图。四个模块的损失联合优化。

```mermaid
flowchart LR
    X["多视图输入 X^v"] --> ENC["逐视图 MLP 编码<br/>自表达稀疏重构 Lrec"]
    ENC --> CL["幂平滑随机游走对比<br/>Lcontra(intra+inter)"]
    CL --> VC["视图一致性<br/>共享结构嵌入 Lconsist"]
    VC --> DIFF["熵引导多尺度扩散<br/>融合 Afusion → Afinal"]
    DIFF --> SC["谱聚类"]
```

### 关键设计

**1. 自表达模块：稀疏自重构打底，给下游一个干净的嵌入** 每个视图先经编码器 $Z^v = f^v(X^v)$，再以稀疏系数矩阵做自重构 $\hat{X}^v = C^v Z^v$，其中 $C^v$ 由各视图内对样本对余弦相似度用自适应阈值 $b$ 过滤后加权融合得到。训练目标是重构损失 $L_{rec} = \frac{1}{2}\sum_v \|X^v - \hat{X}^v\|_F^2$，并辅以 L1/L2 混合正则 $L_{reg} = \sum_v \lambda\|C^v\|_1 + \frac{1-\lambda}{2}\|C^v\|_F^2$ 防过拟合。这一步让嵌入同时保留全局语义与局部几何，为后续对比、一致性、扩散提供鲁棒输入。

**2. 随机游走纠正 + 幂平滑对比：把假负样本从对比目标里"洗"掉** 传统 InfoNCE 默认所有非锚点样本同等重要、且都是真负样本，但跨视图异质性会让同类样本看起来不相似，制造大量假负样本。MANGO 先用样本嵌入的欧氏距离建亲和矩阵 $A_{ij} = \exp(-\sigma\|z_i - z_j\|^2)$ 并行归一化为转移矩阵 $M_{ij} = A_{ij}/\sum_k A_{ik}$，再算 $t$ 步转移 $M^t$ 捕捉高阶流形结构，用插值参数 $\eta$ 构造目标分布 $T = \eta I + (1-\eta)M^t$，其中 $T_{ij}$ 直接作为对比损失里的负样本权重——语义近邻的样本权重被压低，从而不再当强负样本。在此基础上对负样本项施加 $\beta$ 幂运算做非线性平滑，得到视图内对比损失
$$L_{intra} = \frac{1}{m}\sum_p\left(-\frac{1}{n}\sum_i \log\frac{\exp(s(z_i^p,z_i^p)/\tau)}{\exp(s(z_i^p,z_i^p)/\tau)+\sum_{j\neq i}T_{ij}\exp(s(z_i^p,z_j^p)/\tau)^\beta}\right)$$
视图间损失 $L_{inter}$ 结构同理但负样本用均匀权重 $W_{ij}$，二者以 $\mu$ 平衡得 $L_{contra} = L_{intra} + \mu L_{inter}$。$\beta$ 幂专门压制极端值负样本的整体影响，减少噪声传播。

**3. 结构感知视图一致性：用共享结构嵌入对齐语义又不抹平特异性** 不同视图噪声分布与语义侧重不同，同类样本嵌入差异大，直接送入融合会引入冲突噪声。该模块把视图一致性建模为最大化视图嵌入间的互信息 $I(Z^p;Z^q)$，并通过其下界学一个映射 $f_{p\to q}$ 使 $\hat{Z}^p = f_{p\to q}(Z^p)\approx Z^q$，得到成对一致性损失 $L_{p\to q} = \frac{1 - d(\hat{Z}^p, Z^q)}{\tau}$（$d$ 为余弦距离），对所有视图对求平均即 $L_{consist}$。共享结构嵌入保证局部结构一致，同时保留各视图异构特征，从而协调全局语义对齐与局部模态独特性、缓解簇边界模糊。

**4. 熵引导多尺度扩散：让模型自动决定每个尺度该信多少** 大多数图方法用静态图、对特征质量高度敏感。MANGO 对归一化亲和矩阵 $A_{norm}$ 做多步扩散 $\{\tilde{A}_0,\tilde{A}_1,\dots,\tilde{A}_t\}$（每步保留每行 top-K 后重归一化），并用熵衡量各尺度扩散矩阵的质量：对每行非零元算 $H(\tilde{A}_i^t) = -\sum_{j:\tilde{A}_{ij}^t>0}\tilde{A}_{ij}^t\log\tilde{A}_{ij}^t$，熵越低说明连接分布越集中、语义结构越清晰。于是用平均熵的倒数作为该尺度权重融合：
$$A_{fusion} = \sum_{t=0}^{T}\frac{1}{\bar{H}(\tilde{A}^t)}\tilde{A}^t$$
最后做对称归一化与对角增强 $A_{final}[i,j] = \frac{1}{2}(A_{fusion}[i,j]+A_{fusion}[j,i])\cdot k$ 供谱聚类。整体损失为 $L = L_{reg} + \alpha L_{rec} + \beta L_{contra} + \gamma L_{consist}$。

## 实验关键数据

### 主实验表格
12 个数据集（人脸/文本/场景/物体/数字，规模 165–60000），对比 8 个 SOTA（MFLVC、MSESC、CVCL、LSGMC、MVD、DIVIDE、SCM、CANDY），指标 ACC/NMI/ARI（10 次平均）。

| 数据集 | 指标 | 次优方法 | MANGO |
|---|---|---|---|
| Yale | ACC | 0.711 (LSGMC) | **0.729** |
| ORL | ACC | 0.882 (MVD) | **0.926** |
| BBC-Sport | ACC | 0.936 (LSGMC) | **0.959** |
| Scene-15 | ARI | 0.314 (LSGMC) | **0.388** |
| ALOI-100 | ACC | 0.753 (DIVIDE) | **0.887** (+13.4%) |
| STL10 | ACC | 0.937 (SCM) | **0.960** |
| HandWritten | ACC | 0.976 (LSGMC) | **0.978** |

MANGO 在绝大多数数据集/指标上取得最优；ALOI-100 上 ACC 比次优 DIVIDE 高约 13.4%，提升最显著。

### 消融实验表格
在 MSRC-v1 / Reuters 上逐模块消融（√ 表示启用），random=假负样本纠正，diffusion=自适应扩散：

| 配置 | Lcontra | Lconsist | random | diffusion | MSRC-v1 ACC | Reuters ACC |
|---|---|---|---|---|---|---|
| (a) 仅 Lrec | | | | | 0.770 | 0.502 |
| (c) | √ | √ | | | 0.800 | 0.535 |
| (d) | √ | √ | √ | | 0.893 | 0.507 |
| (g) | √ | √ | √* | √ | 0.863 | 0.551 |
| (h) 完整 | √ | √ | √ | √ | **0.950** | **0.587** |

多尺度 vs 固定步长（MANGO-w 为固定 3 步扩散）：Reuters 上 ACC 从 0.499→0.583，差距明显；小数据集（如 ORL）几乎持平，说明多尺度对复杂大规模结构收益更大。

### 关键发现
- 完整模型显著优于各消融变体，模块间协同生效；随机游走纠正（d vs c）和扩散模块均带来可观提升。
- 对比 CANDY/DIVIDE/SCM 等对比学习算法，MANGO 多数场景更优，验证幂平滑随机游走对比能提升表征质量。
- 浅层方法（LSGMC、MVD）小数据集表现好但大数据集易 OOM；MANGO 在浅/深方法上均稳定领先，兼具性能与鲁棒性。
- 对 $\alpha,\beta,\gamma$ 大范围网格搜索均表现稳定，超参鲁棒；t-SNE 可视化显示簇更清晰紧凑。

## 亮点与洞察
- **用熵的倒数做尺度权重**是个朴素但有效的洞见：把"连接分布是否集中"量化为可微的尺度质量信号，让模型自动决定每个扩散步长的贡献，绕开了固定步长的瓶颈。
- **把假负样本问题转成转移矩阵上的权重重标定**：随机游走得到的 $T_{ij}$ 既是语义近邻度量也是负样本权重，比硬性删除负样本更平滑；再叠加 $\beta$ 幂专门压极端值，两层去噪互补。
- 三个创新精确对应引言提的三个痛点，框架结构清晰、动机-方法-实验闭环完整。

## 局限与展望
- 损失里 $\alpha,\beta,\gamma$ 跨 3 个量级（1e3–1e6）网格搜索，加上 $\sigma,\lambda,\eta,t,\tau,\mu$ 等多个超参，实际调参成本不低；论文虽称鲁棒但量级选择仍需先验。
- 多步扩散 + $t$ 步随机游走涉及反复矩阵乘，论文未给计算复杂度与运行时间分析，大规模（如 MNIST-3V 6 万样本）下的扩展性仅靠 top-K 截断缓解。
- 假负样本纠正依赖嵌入质量构造的转移矩阵，若早期嵌入很差可能"将错就错"，论文未讨论冷启动鲁棒性。
- 代码与 OpenReview 之外的开源情况待确认，复现性有待验证。

## 相关工作与启发
- 深度多视图聚类按处理视图关系分为联合方法、对齐方法和其他方法；MANGO 实际同时融合了联合（自表达+对比）与对齐（视图一致性）两条路线。
- 与 CVCL（跨视图对比）、Trosten 等指出对比对齐中负样本偏置的工作一脉相承，但 MANGO 用随机游走显式建模高阶流形来纠偏，而非变分互信息。
- 图扩散思路可启发其他需要"局部细节 + 全局语义"折中的图任务（如半监督节点分类、图异常检测），熵引导的多尺度融合是一个可移植的轻量组件。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 三个组件各自非全新（图扩散、随机游走纠负、互信息对齐），但"熵引导多尺度融合 + 幂平滑随机游走对比"的组合与针对三大痛点的系统设计有较好整合度。
- **实验充分度**: ⭐⭐⭐⭐ 12 个数据集、8 个 SOTA、三指标、消融 + 多尺度专项 + 超参敏感 + t-SNE，覆盖全面；缺复杂度/运行时分析略减分。
- **写作质量**: ⭐⭐⭐⭐ 痛点-方法-实验对应清晰，公式完整；部分句式偏模板化、图复杂度说明不足。
- **价值**: ⭐⭐⭐⭐ 在多视图聚类上稳定刷新 SOTA，熵引导扩散与去假负的对比组件具一定可迁移性，对图聚类社区有实用参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Constant Degree Matrix-Driven Incomplete Multi-View Clustering via Connectivity-Structure and Embedding Tensor Learning](constant_degree_matrix-driven_incomplete_multi-view_clustering_via_connectivity-.md)
- [\[ICLR 2026\] Confident Block Diagonal Structure-Aware Invariable Graph Completion for Incomplete Multi-view Clustering](confident_block_diagonal_structure-aware_invariable_graph_completion_for_incompl.md)
- [\[ICLR 2026\] Dual-Branch Representations with Dynamic Gated Fusion and Triple-Granularity Alignment for Deep Multi-View Clustering](dual-branch_representations_with_dynamic_gated_fusion_and_triple-granularity_ali.md)
- [\[ICLR 2026\] Learning with Dual-level Noisy Correspondence for Multi-modal Entity Alignment](learning_with_dual-level_noisy_correspondence_for_multi-modal_entity_alignment.md)
- [\[ICLR 2026\] FLOCK: A Knowledge Graph Foundation Model via Learning on Random Walks](flock_a_knowledge_graph_foundation_model_via_learning_on_random_walks.md)

</div>

<!-- RELATED:END -->
