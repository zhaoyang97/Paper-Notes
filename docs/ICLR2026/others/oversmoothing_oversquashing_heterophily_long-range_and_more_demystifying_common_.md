---
title: >-
  [论文解读] Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning
description: >-
  本文系统梳理了图机器学习领域围绕 oversmoothing、oversquashing、同质/异质性和长程依赖的九个常见误区，通过简洁反例逐一反驳，将"oversquashing"拆分为计算瓶颈：和拓扑瓶颈：两个独立概念，厘清了领域中广泛存在的概念混淆。 领域现状： 图上消息传递（message-passing）深度学习…
tags:

---

# Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning

- **会议**: ICLR2026
- **arXiv**: [2505.15547](https://arxiv.org/abs/2505.15547)
- **代码**: [GitHub](https://github.com/AdrianArnaiz/demystifyingGraphML)
- **领域**: 其他
- **关键词**: oversmoothing, oversquashing, heterophily, long-range dependencies, message-passing, GNN

## 一句话总结

本文系统梳理了图机器学习领域围绕 oversmoothing、oversquashing、同质/异质性和长程依赖的九个常见误区，通过简洁反例逐一反驳，将"oversquashing"拆分为**计算瓶颈**和**拓扑瓶颈**两个独立概念，厘清了领域中广泛存在的概念混淆。

## 研究背景与动机

**领域现状**: 图上消息传递（message-passing）深度学习经历快速发展，研究者关注其固有局限——oversmoothing（OSM，节点表征趋同）、oversquashing（OSQ，信息压缩损失）、同质/异质性对分类的影响、以及长程信息传播等议题。

**痛点**: 由于领域进展速度过快，大量"常被接受的信念"（common beliefs）在未经充分验证的情况下被广泛传播和引用，造成研究者概念混淆、问题定义模糊，严重阻碍了有针对性的进一步研究。

**核心矛盾**: 文献中将 oversmoothing 视为性能下降的根本原因、将 oversquashing 与拓扑瓶颈直接等同、将异质性等同于"困难"等论断，并非普遍成立，却几乎成为公认结论。

**目标**: 明确指出这些信念的局限性，通过简明反例予以反驳，让研究者能够区分并精准定义待解决的问题。

**切入角度**: 不批驳具体工作，而是将文献中散布的含混论断归纳为九条 common beliefs，逐条用数学定义+反例的方式"去神秘化"。

**核心 idea**: "Oversquashing"应当被拆解为**计算瓶颈**（computational bottleneck，源于计算树的指数膨胀）和**拓扑瓶颈**（topological bottleneck，源于图连通性）两个独立问题，它们可独立存在也可互不相关。

## 方法详解

### 整体框架

本文不提出新模型，而是做一件"概念去神秘化"（demystifying）的工作：把图机器学习里散落在大量论文中、被当成公认结论却没被严格验证的含混论断，归纳成**九条 common belief**，再给每一条配一个**简洁但形式上充分的反例**逐一反驳。九条信念按三大主题分组——oversmoothing（OSM，过平滑）、同质/异质性（homophily/heterophily）、oversquashing（OSQ，过压缩）；每一条的论证都走同一套路：先给数学定义，再摆一个小图反例，最后落一句 take-home message。论文最核心的主张是把被混用的"oversquashing"拆成**计算瓶颈**和**拓扑瓶颈**两个独立概念。下表是九条信念的全景：

| 主题 | Beliefs |
|------|---------|
| OSM | 1. OSM 导致性能下降；2. OSM 是所有 DGN 的固有属性 |
| 同质/异质 | 3. 同质好、异质差；4. 长程传播在异质图上评估；5. 不同类别意味着不同特征 |
| OSQ | 6. OSQ=拓扑瓶颈；7. OSQ=计算瓶颈；8. OSQ 对长程任务有害；9. 拓扑瓶颈与长程问题关联 |

### 关键设计

三个设计点对应表中三大主题，顺序与论文章节一致（OSM → 同质/异质 → OSQ），最后一条 OSQ 拆分是全篇的核心贡献。

**1. Oversmoothing 既不普遍存在、也不是性能下降的根因**

针对"OSM 导致性能下降、且是所有深度图网络的固有属性"这条信念，本文先指出连度量 OSM 的指标本身都不自洽。文献常用 Dirichlet Energy（DE）和 Rayleigh Quotient（RQ）两个量来刻画节点表征是否趋同：

$$\mathrm{DE}(\mathbf{H}^\ell) = \mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell), \quad \mathrm{RQ} = \frac{\mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell)}{\|\mathbf{H}^\ell\|_F^2}$$

关键在于：OSM 是否发生、朝哪个方向走，都强烈依赖架构与超参的选择。GIN 的 DE 在标准设置下不降反**爆炸**；只要把权重矩阵乘 2（即 $AX(2W)$）就能逆转 DE 的趋势；而 DE 和 RQ 在同一个模型上常给出互相矛盾的结论。既然换聚合、缩放权重、换指标都会改变结论，OSM 自然谈不上普遍，也没有唯一定义。更进一步，即便 DE 真的下降，也不等于性能就该垮——不同类别的节点会先各自坍缩到各自的点（"有益平滑"阶段），**类别可分性**反而可能保持甚至提升，真正拖垮准确率的更多是梯度消失和过拟合。

**2. 把同质/异质性与任务难度、长程依赖解耦开**

这一组信念把"异质 = 难"、"长程传播任务只出现在异质图上"当成了常识，本文用两个反例同时拆掉。其一，在一个完全异质的二部图（bipartite graph）里，仅凭节点入度的差异，1 层 sum-based DGN 就能做到完美分类——异质到极致反而平凡，"高异质"和"困难"之间没有必然联系。其二，在一个高度同质的图上，如果任务是判断某节点到特定节点的距离是否大于 5，就必须依赖长程传播，可这张图明明是同质的。两个反例合起来说明：**长程任务与异质性彼此正交**，由任务诱导出的类别标签不能反推图属性，因此二者不该被绑定讨论。

**3. 把 Oversquashing 拆成计算瓶颈与拓扑瓶颈两个独立问题**

这是全篇的核心贡献。文献里 OSQ 几乎被默认等同于"图的拓扑瓶颈"，但本文给出计算瓶颈的严格定义后，证明二者可以彼此独立地出现。所谓**计算瓶颈**（computational bottleneck），指的是节点 $v$ 在 $K$ 层消息传递后计算树展开的多重集大小 $|\mathcal{M}_v^K|$，它源于架构本身：

$$\mathcal{M}_v^K := \mathcal{M}_v^{K-1} \uplus \left\{\biguplus_{u \in \mathcal{M}_v^{K-1}} \mathcal{N}_u\right\}$$

它随层数指数膨胀，与图是否存在连通性意义上的**拓扑瓶颈**（topological bottleneck，由曲率/谱隙刻画、内在于图结构）无关。两个反例把这种独立性摆得很清楚：网格图（grid graph）没有任何拓扑瓶颈，却因计算树爆炸而很快有严重的计算瓶颈；反过来，存在拓扑瓶颈的图在层数很少时计算瓶颈反而很轻。区分这两件事有直接的实践意义——现有的图重连（graph rewiring）通过加边缩短节点距离来缓解拓扑瓶颈，却会在层数不变时**恶化**计算瓶颈；而消息过滤（message filtering，让网络学会该交换多少消息、过滤掉哪些）不改图结构就能同时减小计算瓶颈和敏感度，恰好走相反的路子。

### 损失函数

本文为分析型工作，不提出新模型，反例多为人工构造的小图；实验仅复现标准 DGN 训练（交叉熵损失做节点分类），用来对照说明 OSM 指标的不一致。

## 实验关键数据

### 主实验：OSM 指标在不同架构下的表现（Cora, 50 seeds）

| 架构 | W: DE 趋势 | 2W: DE 趋势 | W: RQ 趋势 | 2W: RQ 趋势 |
|------|-----------|-------------|-----------|-------------|
| GCN | 坍缩→0 | 爆炸↑ | 平稳 | 平稳 |
| GAT | 坍缩→0 | 爆炸↑ | 线性衰减 | 稳定 |
| SAGE | 坍缩→0 | 爆炸↑ | 线性衰减 | 部分稳定 |
| GIN | 爆炸↑ | 爆炸↑↑ | 线性衰减 | 部分稳定 |

> 结论：同一模型在两个指标下可表现出完全矛盾的趋势。OSM 的观测高度依赖度量选择和超参。

### 消融实验：有/无偏置下性能退化与 DE 对比

| 配置 | 深度增加时 DE | 深度增加时准确率 |
|------|------------|--------------|
| GCN (有 bias) | 坍缩 | 下降 |
| GCN (无 bias) | **不坍缩** | 同样下降 |

> 说明性能下降不能归因于 OSM——去除 bias 消除了 DE 坍缩但准确率仍下降，真正原因是梯度消失/过拟合。

### 关键发现

1. **OSM 非普遍现象**: 换聚合函数（GIN）、微调权重缩放（2W）或换度量指标（DE vs RQ）都会改变结论
2. **计算瓶颈 ≠ 拓扑瓶颈**: 网格图无拓扑瓶颈但对角节点间 effective resistance 线性增长，仍有严重计算瓶颈；graph rewiring 改善拓扑但可能恶化计算瓶颈
3. **异质性 ≠ 困难性**: 完全异质图存在 1 层 DGN 完美分类的情况；高同质图也可能需要长程传播

## 亮点与洞察

- 首次将"oversquashing"**显式拆分**为计算瓶颈与拓扑瓶颈两个独立概念，具有高度启发性
- 反例简洁且形式化，如二部图完美分类、网格图无拓扑瓶颈但计算瓶颈严重，易于记忆和传播
- 九条 belief 的梳理方式像"debug 清单"，可作为图 ML 研究者的参考手册
- 呼吁社区在论述中更精确地使用术语、避免过度泛化

## 局限性

- 主要是概念澄清和反例驱动，未提出新的解决方案或模型
- 大部分反例基于人工构造的小图，实际大规模图上的情况更复杂
- 面向图 ML 社区的"元研究"，对其他领域的直接实用价值有限
- 部分 belief 的"反驳"本质是"不总是成立"，而非"从不成立"——边界条件仍需深入探讨

## 相关工作与启发

- **Oversmoothing**: Cai & Wang (2020) 提出 Dirichlet Energy 度量；Roth & Liebig (2024) 分析 Rayleigh Quotient；Zhang et al. (2025) 指出未训练网络的 OSM 不反映训练后行为
- **Oversquashing**: Topping et al. (2022) 连接到曲率和 Jacobian 灵敏度；Errica et al. (2025) 提出 message filtering 减少计算瓶颈
- **Heterophily**: Ma et al. (2022) 提出新指标关联异质性与可区分度；Platonov et al. (2023) 构建更严格的异质性基准
- **启发**: 未来工作应明确区分度量对象（DE vs RQ vs 可分性）、问题类型（计算 vs 拓扑）、以及任务与图属性的关系

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — 将分散的困惑系统化为九条 belief 并给出形式化反驳，conceptual contribution 很强
- **实用性**: ⭐⭐⭐ — 概念澄清价值大但无新方法
- **写作**: ⭐⭐⭐⭐⭐ — 逻辑清晰，反例精炼，表格总结一目了然
- **影响力**: ⭐⭐⭐⭐ — 对图 ML 社区的术语使用和问题定义具有长期影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] OSIRIS: Bridging Analog Circuit Design and Machine Learning with Scalable Dataset Generation](osiris_bridging_analog_circuit_design_and_machine_learning_with_scalable_dataset.md)
- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[CVPR 2026\] Learning Long-term Motion Embeddings for Efficient Kinematics Generation](../../CVPR2026/others/learning_long-term_motion_embeddings_for_efficient_kinematics_generation.md)
- [\[CVPR 2026\] FedSDR: Federated Graph Learning with Structural Noise Detection and Reconstruction](../../CVPR2026/others/fedsdr_federated_graph_learning_with_structural_noise_detection_and_reconstructi.md)

</div>

<!-- RELATED:END -->
