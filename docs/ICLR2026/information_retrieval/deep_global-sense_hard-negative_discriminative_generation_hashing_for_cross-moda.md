---
title: >-
  [论文解读] Deep Global-sense Hard-negative Discriminative Generation Hashing for Cross-modal Retrieval
description: >-
  [ICLR 2026][信息检索/RAG][跨模态] DGHDGH 首次把"困难负样本生成"引入跨模态哈希，用一张跨模态结构图做双向迭代消息传播来感知**全局样本相关性**，再据此做**通道级、难度自适应**的锚-负样本插值，合成既贴近锚点又不越界到其它类别的硬负样本，从而把汉明共空间训得更有判别力。
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "跨模态"
  - "Hard Negative Generation"
  - "Graph Message Propagation"
  - "Hamming Co-space"
  - "Channel-wise Interpolation"
---

# Deep Global-sense Hard-negative Discriminative Generation Hashing for Cross-modal Retrieval

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=GAQEsnnQtG](https://openreview.net/forum?id=GAQEsnnQtG)  
**代码**: [https://github.com/QinLab-WFU/DGHDGH](https://github.com/QinLab-WFU/DGHDGH)  
**领域**: 跨模态检索 / 跨模态哈希 / 困难负样本生成  
**关键词**: Cross-modal Hashing, Hard Negative Generation, Graph Message Propagation, Hamming Co-space, Channel-wise Interpolation  

## 一句话总结
DGHDGH 首次把"困难负样本生成"引入跨模态哈希，用一张跨模态结构图做双向迭代消息传播来感知**全局样本相关性**，再据此做**通道级、难度自适应**的锚-负样本插值，合成既贴近锚点又不越界到其它类别的硬负样本，从而把汉明共空间训得更有判别力。

## 研究背景与动机
- **领域现状**：深度跨模态哈希检索（DCHR）把图像/文本投到共享的紧凑二进制汉明空间，让语义相似的异构样本拿到相近编码，从而把跨模态检索变成高效的哈希查找。要提升判别力，关键是在训练中提供"信息量更大"的信号，而困难负样本（hard negative）能提供更强的对抗梯度、逼模型学到更细的边界。
- **现有痛点**：困难负样本"挖掘"（mining）受限于 mini-batch 里天然硬样本太稀缺；于是出现了困难负样本"生成"（HNG），通常用已有负样本做线性插值来合成更难的样本。但现有 HNG **几乎只看局部锚-负相关性**，忽视嵌入空间的全局几何结构。
- **核心矛盾**：只按单个锚-负对插值，生成的样本可能**误闯入第三方类别的分布**——例如给紫色图像锚选蓝色文本负样本插值，合成点却落进了红色类别区域。这种"语义越界"在判别能力本就紧张的跨模态共空间里尤其致命，反而削弱了判别性。
- **本文目标**：在生成阶段显式建模**类间全局关系**，让合成的负样本既有合适难度、又尊重语义流形，不破坏共空间的整体分布。
- **核心 idea**：**【全局感知 + 通道级难度自适应】** 先用图网络在整个 batch 上学全局相关性（RGP），再用这些相关性指导每个通道独立的、随训练逐步变难的插值（DGS），且**不需要额外的生成器网络**，即插即用、轻量高效。

## 方法详解

### 整体框架
DGHDGH 由三段串起来：双 Transformer 提哈希码 → RGP 用图消息传播学全局相关性 → DGS 据此做通道级自适应插值合成硬负样本，最后用真实+合成负样本的三元组损失做判别哈希学习，整体在"生成优化"与"哈希学习"两个目标间交替训练。

```mermaid
flowchart LR
    A[图像/文本<br/>双Transformer+哈希层] -->|hash codes| B[RGP 全局相关传播]
    B -->|节点/边交替更新<br/>学到全局相关性 edge| C[DGS 判别式全局合成]
    C -->|通道级难度自适应插值<br/>合成硬负样本| D[哈希学习<br/>真实+合成三元组损失]
    B -.语义保持分类层.-> D
```

### 关键设计

**1. RGP 全局相关传播：用一张图把"局部插值"升级为"全局感知"。** 这是解决"语义越界"的根。把整个 batch 的哈希码放进结构图 $G=(V,E)$，节点 $V_i^{k=0}=\tilde h_i$ 存样本嵌入，边 $E_{ij}^{k=0}=\tilde h_i\odot\tilde h_j$ 编码两两相关性，并**并行维护图像图、文本图、跨模态图三张图且共享参数**，让跨模态语义鸿沟在联合更新中被拉近。消息传播用一个**双 Transformer 做节点与边的异步交替更新**：先传节点消息、再传边消息，保证节点信息持续注入后续的边更新。节点 Transformer 用带正样本掩码的 MMSA——每个节点当 anchor，只和它的负样本交互（把正样本、尤其跨模态里的同一样本 mask 掉），避免高注意力权重淹没负样本之间的细微差异，并把邻接边信息融回节点：$V_i'=\mathrm{LN}\big(\mathrm{MMSA}(V^k)_i+\sum_j E_{ij}^k+V_i^k\big)$。边 Transformer 则用交叉注意力把节点信息汇入边，$E_{ij}'=\mathrm{LN}\big(\mathrm{CA}(E_{ij}^k,V_i^{k+1},V_j^{k+1})+E_{ij}^k\big)$，让每条边从全局视角感知其关键点相关性、自适应调节合成难度。经 $n_2$ 轮迭代后，边就编码了足够的全局相关性。

**2. DGS 通道级难度自适应合成：让每个维度按全局相关性独立决定"插多深"。** 与传统插值"所有通道共用一个系数"不同，DGS 把每个锚-负对的最终边 $E_{an}^{n_2}$ 过一个全连接层+Sigmoid 得到**通道级插值向量** $\lambda_{an}=\mathrm{Sigmoid}(\mathrm{FC}(E_{an}^{n_2}))$，为每个通道提供自适应的融合权重。插值本身随训练逐步变难：

$$\tilde h_{an}'=\begin{cases}(1-\eta)\tilde h_a+\eta\tilde h_n,&\text{if }d_{ap}<d_{an}\\ \tilde h_n,&\text{otherwise}\end{cases},\quad \eta=\big(d_{ap}+\lambda_{an}\tau(d_{an}-d_{ap})\big)/d_{an}$$

其中**自步缩放因子** $\tau=e^{-1/l_{avg}}$ 由上一个 epoch 的平均损失 $l_{avg}$ 决定：模型越拟合、$l_{avg}$ 越小，$\tau$ 越收紧插值区间上界，合成的负样本就越难。这样难度跟着收敛进度走，避免训练早期就给过难样本。

**3. 生成优化的三重约束：让合成样本"更难但不变味、且多样"。** DGS 合成的样本要满足三点，对应三个损失：**语义保持** $L_{sp}=\mathrm{CE}(\mathrm{CL}(\tilde h_{an}'),l_n)$——额外接一个只在真实样本上训练、对合成样本不回传梯度的分类层来约束合成样本不偏离原负类语义；**插值相似** $L_{is}=1-\cos(\tilde h_{an}',\tilde h_a)$——直接逼合成样本贴近锚点（更难）；**系数多样** $L_{cd}=1-\sigma(\lambda_{a-})$——用同一锚点下所有插值系数的标准差鼓励不同对之间差异化，防止合成塌缩。三者加权得 $L_{go}=\gamma_{is}L_{is}+\gamma_{sp}L_{sp}+\gamma_{cd}L_{cd}$。

**4. 交替式判别哈希学习：真实与合成负样本一起喂三元组。** 哈希学习用标准三元组损失，先只用真实样本 $L_{real}$（覆盖 I→I/I→T/T→I/T→T 四种组合），再引入 DGS 合成的硬负样本得 $L_{syn}$，合成总损失 $L_{hl}=L_{real}+\gamma_{syn}L_{syn}$，其中 $\gamma_{syn}=1-e^{1/L_{go}}$ 随图网络收敛逐步加大硬负样本占比。再配合两个跨模态共享参数的语义保持分类层（$L_{sp1}$ 约束哈希码、$L_{sp2}$ 约束图传播后的节点）维持语义一致性。整个训练在 $L_{go}$ 与 $L_{hl}$ 之间交替，让"造样本"和"学哈希码"协同推进。

## 实验关键数据

### 主实验表格
三个基准（MIRFLICKR-25K / NUS-WIDE / MS COCO），所有方法统一用 CLIP ViT-B/32 backbone，报告 mAP@all(%)，下表节选 I→T 任务：

| 方法 | 来源 | MIRFLICKR 64bit | NUS-WIDE 128bit | MS COCO 128bit |
|------|------|----------------|-----------------|----------------|
| DNpH | TMM'24 | 85.88 | 71.58 | 68.74 |
| DHaPH | TKDE'24 | 85.31 | 71.55 | 75.43 |
| DECH | AAAI'25 | 83.83 | 72.41 | 68.49 |
| DDBH | TCSVT'25 | 86.10 | 72.29 | 78.24 |
| **DGHDGH** | OURS | **87.13** | **73.76** | **79.19** |

I→T 与 T→I 两个方向、四个码长上基本全面取得最优或次优；MS COCO 128bit 上 I→T 比最强 baseline DDBH 高约 0.95 个点。

### 消融实验表格
组件消融（MIRFLICKR-25K，三损失项交叉消融，I→T / T→I 平均 mAP）：

| $L_{is}$ | $L_{sp}$ | $L_{cd}$ | Avg. I→T | Avg. T→I |
|:---:|:---:|:---:|:---:|:---:|
| | | | 81.64 | 80.04 |
| ✓ | | | 83.20 | 81.13 |
| | ✓ | | 83.97 | 82.07 |
| ✓ | ✓ | | 85.61 | 83.68 |
| | ✓ | ✓ | 85.81 | 83.90 |
| ✓ | ✓ | ✓（全模型） | 更高 | 更高 |

另有模块级消融：w/o RGP（直接用初始边做插值源）、w/o DGS（去掉生成阶段）、w/o EMF（去掉 RGP 里的边消息融合）、w/o HAP（去掉 DGS 里的难度自适应参数），各项均带来下降，验证每个部件都有贡献。

### 关键发现
- 用 **Fisher ratio** 与 P@H≤2 度量判别力，DGHDGH 在多数设定下取得最高类间可分性，印证"全局感知 HNG 让共空间更有判别力"。
- 三个生成损失里 $L_{sp}$（语义保持）单独贡献最大，说明"合成样本别变味"是 HNG 能否奏效的关键；三者叠加进一步提升。
- 作为即插即用模块，可增强已有跨模态哈希方法，且无需额外生成器网络。

## 亮点与洞察
- **把"全局几何"显式引入负样本生成**：用三图并行+节点/边异步交替传播，专门修正局部插值"误闯第三方类别"的老问题，思路清晰且对症。
- **难度自适应做到了通道级 + 时间级双重**：$\lambda_{an}$ 让不同通道插值深浅不同，$\tau$ 让难度随收敛进度自步增长，比单系数插值更细腻。
- **无额外生成器**：靠图传播得到的边直接产出插值向量，相比 GAN 类 HNG 更轻、更稳、更易迁移。

## 局限与展望
- 维护图像/文本/跨模态三张图并跑双 Transformer 消息传播，batch 内是 $O(B^2)$ 边规模，大 batch 或大数据下显存/计算开销值得关注（论文用 batch=128）。
- 评测集中在三个经典中小规模检索基准，更大规模、长尾或开放域跨模态检索上的表现还需验证。
- 自步因子 $\tau=e^{-1/l_{avg}}$、$\gamma_{syn}=1-e^{1/L_{go}}$ 等耦合式调度较多，超参与训练稳定性的鲁棒性有待更系统的分析。

## 相关工作与启发
- **信息学习两大家族**：挖掘类（如 Distance-Weighted Sampling）vs 增强类（GAN、插值如 DAS、记忆如 XBM）。本文属增强类里的生成派，但补上了"全局几何"这一维。
- **HNG 前作**：HDML 等基于局部邻域合成、难与全局几何对齐，本文正是针对这一缺口。
- **启发**：把"图消息传播感知全局结构"嫁接到"样本合成"上，是一个可推广的范式——度量学习/对比学习里凡是靠插值造硬样本的场景，都可借鉴用全局相关性约束合成不越界。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首次把困难负样本生成引入跨模态哈希，"全局图感知 + 通道级难度自适应 + 无生成器"的组合有清晰的问题动机和针对性。
- **实验充分度**: ⭐⭐⭐⭐ 三基准、四码长、I2T/T2I 双向、多类信息学习方法对比，配 Fisher ratio/P@H 判别力度量与细粒度组件/损失消融，较扎实；规模偏经典基准、缺超大规模验证。
- **写作质量**: ⭐⭐⭐⭐ 动机用图示讲清"语义越界"，方法公式完整、模块职责分明，可读性好。
- **价值**: ⭐⭐⭐⭐ 即插即用、无额外生成器，对跨模态哈希与更广义的硬负样本生成都有借鉴意义，已开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Hierarchical Encoding Tree with Modality Mixup for Cross-modal Hashing](hierarchical_encoding_tree_with_modality_mixup_for_cross-modal_hashing.md)
- [\[ICLR 2026\] Improving Semantic Proximity in Information Retrieval through Cross-Lingual Alignment](improving_semantic_proximity_in_information_retrieval_through_cross-lingual_alig.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](../../ACL2025/information_retrieval/hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)
- [\[ICLR 2026\] Demystifying Deep Search: A Holistic Evaluation with Hint-free Multi-Hop Questions and Factorised Metrics](demystifying_deep_search_a_holistic_evaluation_with_hint-free_multi-hop_question.md)

</div>

<!-- RELATED:END -->
