---
title: >-
  [论文解读] Graph Representational Learning: When Does More Expressivity Hurt Generalization?
description: >-
  [ICLR 2026][图学习][图神经网络] 本文提出一族由图不变量参数化的伪度量 ζ-TMD，推导出依赖"训练-测试图结构相似度 + 模型复杂度 + 训练集规模"的数据相关泛化界，从理论上解释了"更高表达力的 GNN 何时反而泛化更差"——只有当增加的表达力与任务的结构-标签相关性对齐时才有益，否则只会徒增复杂度并恶化泛化。
tags:
  - "ICLR 2026"
  - "图学习"
  - "图神经网络"
  - "泛化界"
  - "Tree Mover's Distance"
  - "伪度量"
  - "PAC-Bayes"
  - "结构-标签相关性"
---

# Graph Representational Learning: When Does More Expressivity Hurt Generalization?

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=C6vpifaZvU](https://openreview.net/forum?id=C6vpifaZvU)  
**代码**: 已开源（GitHub，链接见原文）  
**领域**: 图表示学习 / GNN 泛化理论  
**关键词**: GNN 表达力, 泛化界, Tree Mover's Distance, 伪度量, PAC-Bayes, 结构-标签相关性  

## 一句话总结
本文提出一族由图不变量参数化的伪度量 ζ-TMD，推导出依赖"训练-测试图结构相似度 + 模型复杂度 + 训练集规模"的数据相关泛化界，从理论上解释了"更高表达力的 GNN 何时反而泛化更差"——只有当增加的表达力与任务的结构-标签相关性对齐时才有益，否则只会徒增复杂度并恶化泛化。

## 研究背景与动机
**领域现状**：GNN 研究的一条主线是不断提升模型的"表达力"，即区分非同构图的能力，通常用 Weisfeiler-Lehman（WL）层级来衡量。从标准消息传递网络（MPNN，受限于 1-WL）到 k-GNN、子图 GNN、高阶 folklore GNN，表达力被一路推高。

**现有痛点**：表达力与实际预测性能之间的关系始终不清楚。一方面，更强的模型在很多任务上确实更好，哪怕它们并没有在该 benchmark 上多区分出几个非同构图；另一方面，Bechler-Speicher 等人发现，完全丢弃图结构（比 1-WL 还弱）的模型在某些任务上反而能击败精心设计的 GNN。经典复杂度度量（VC 维、Rademacher 复杂度）在过参数化区间无法解释这种现象——网络能记住任意标签，但只有当数据与标签存在真实相关时才泛化。

**核心矛盾**：表达力本身既不绝对有害也不绝对有益。作者在合成 cycle-counting 任务上观察到：中等表达力的 F4-MPNN（用环计数增强的 MPNN）泛化最好，而最强的 LF-GNN 反而过拟合；在 BZR/MUTAGENICITY/NCI109 上逐步加标签噪声，训练损失照样收敛到零，但一旦图-标签相关性被破坏，测试误差陡升。

**本文目标**：回答两个关键问题——(1) 何时增加表达力有益、何时有害？(2) 若更强的 GNN 表现更好，是否真因为它在图可分性意义上的表达力提升？

**核心 idea（任务对齐的伪度量 + 数据相关泛化界）**：把"泛化取决于结构-标签相关性"这一经验直觉形式化。引入一族 **ζ-TMD 伪度量**，每个由一个图不变量（度分布、k-WL 颜色、motif 计数等）参数化以刻画特定层级的表达力；假设标签与某个固定 ζ-TMD 强相关，则泛化界自然分解为"模型复杂度项"与"训练-测试结构相似度项"，从而精确刻画表达力与泛化的权衡。

## 方法详解

### 整体框架
框架分三步：先把"图不变量/颜色细化算法"统一为可被 1-WL 强模拟（strongly simulatable）的 CRA，并基于此把 Tree Mover's Distance（TMD）推广为 **ζ-TMD** 伪度量；再证明对应的 ζ-MPNN 对该伪度量 Lipschitz 连续；最后用 PAC-Bayes 机制把 Lipschitz 性质转化为数据相关的泛化界，并据此分析"表达力超过任务所需"何时会抬高界。

```mermaid
flowchart TD
    A[图不变量 / CRA ζ<br/>度分布·k-WL·motif计数] --> B[强模拟变换 Rζ<br/>用1-WL模拟ζ]
    B --> C[ζ-TMD 伪度量<br/>TMD on Rζ(G),Rζ(H)]
    C --> D[ζ-MPNN 对 ζ-TMD<br/>Lipschitz连续 Thm3.4]
    D --> E[PAC-Bayes 泛化界 Thm4.1<br/>复杂度项 + 结构相似度项 ξζ]
    E --> F[何时表达力有害? Thm5.1<br/>F'⊊F⊊F̃ 三档对比]
```

### 关键设计

**1. ζ-TMD：把表达力嵌进可比的伪度量。** 标准 TMD 是两图深度为 $t$ 的有根树分布间的 Wasserstein 距离 $\mathrm{TMD}_t(G,H)=\mathrm{Wasserstein}(\mathcal{T}^t(G),\mathcal{T}^t(H))$，只刻画 1-WL 这一档表达力。作者观察到许多更强的架构（k-GNN、子图 GNN、motif 增强）都可通过"输入变换 $R_\zeta$ + 标准消息传递"来强模拟（Jogl et al. 2024）：对图 $G$ 跑 $t$ 步 CRA $\zeta$，等价于对变换图 $R_\zeta(G)$ 跑 $t$ 步 1-WL。于是直接定义 $\zeta\text{-TMD}_t(G,H):=\mathrm{TMD}_t(R_\zeta(G),R_\zeta(H))$。这个构造的巧妙之处在于：它是合法伪度量（Prop 3.2），且若 $G,H$ 能被 $\zeta$ 在 $T$ 步内区分则 $\zeta\text{-TMD}_{T+1}>0$（Prop 3.3），从而把"离散的可分/不可分"细化为"连续的结构相似度"，让不同表达力档位有了统一的、可度量的尺度。

**2. ζ-MPNN 的 Lipschitz 连续性。** 这是连接表达力与泛化的桥梁。Theorem 3.4 证明：若 ζ-MPNN 有 $T$ 层、消息/更新函数 Lipschitz 常数被 $L_{g^{(t)}},L_{f^{(t)}}$ 控制、接全局 sum pooling 和 Lipschitz 分类器，则 $\|h(G)-h(H)\|\le L\cdot \zeta\text{-TMD}_{T+1}(G,H)$，其中 $L=L_c 2^T\prod_{t=1}^T L_{f^{(t)}}L_{g^{(t)}}$。直观含义：结构上相似（ζ-TMD 小）的图必被映射到相近的表示。以 F-MPNN（用 motif/环计数增强节点特征）为例，Corollary 3.5 给出 $\|h(G)-h(H)\|\le L\cdot F\text{-TMD}_{T+1}(G,H)$，并强调同样的推导可平移到 k-GNN 等架构。

**3. 数据相关泛化界与结构-标签对齐假设。** 关键建模假设是"标签与某伪度量强相关"：每个类别 $k$ 存在关于伪度量的 Lipschitz 函数 $\eta_k$ 使 $\Pr(y_G=k\mid G)=\eta_k(G)$，记作 $y\sim pm$。定义训练-测试结构距离 $\xi_{pm}=pm(G_{te},G_{tr})=\max_{G\in G_{te}}\min_{H\in G_{tr}}pm(G,H)$。在 $y\sim\zeta\text{-TMD}_{T+1}$ 下，Theorem 4.1 用 PAC-Bayes（Neyshabur 2018 + Ma et al. 2021 的相关设定）给出泛化界，简化为
$$L^0_{te}(\tilde h)\le \hat L^\gamma_{tr}(\tilde h)+O\Big(\underbrace{\tfrac{\mathrm{MC}(\mathcal C\circ\mathcal E_\zeta)}{N_{tr}^{2\alpha}\gamma^{1/D}\delta}}_{\text{复杂度项}}+\underbrace{C\,\xi_\zeta}_{\text{结构相似度项}}\Big).$$
复杂度项 $\mathrm{MC}$ 聚合权重谱范数、隐藏维度 $b$、最大度 $d$；结构相似度项 $\xi_\zeta$ 衡量训练-测试图在 ζ-TMD 下的对齐度。这个分解是全文的核心洞察：泛化好 = 模型把"标签所相关的那种结构相似"映射成表示相似，同时把容量压住。

**4. 何时表达力有害的判别条件。** 设标签与 $F\text{-TMD}_{T+1}$ 强相关，取三档嵌套表达力 $\mathcal F'\subsetneq\mathcal F\subsetneq\tilde{\mathcal F}$。Theorem 5.1 表明：从 $\mathcal F'$ 提升到恰好捕获任务相关结构的 $\mathcal F$，界由同一个 $\xi_F$ 控制、泛化基本不变；但继续提升到捕获了任务无关额外结构的 $\tilde{\mathcal F}$ 时，结构差异变大（$\xi_{\tilde F}\ge\xi_F$，Lemma G.1），泛化界显著抬升。结论清晰：**最优 GNN 的表达力应恰好等于捕获任务相关结构-标签相关性所需的水平——不多不少**，多余的表达力只会放大训练-测试结构差异并增加计算开销。

## 实验关键数据

### 主实验表格（Task 1：Erdős–Rényi 等随机图，按"3-环+4-环计数"中位数二分类）

| 模型 | 表达力 | 测试准确率 |
|------|--------|-----------|
| MPNN | 1-WL | 0.8490 ± 0.0045 |
| F3-MPNN | +3环 | 0.8657 ± 0.0085 |
| **F4-MPNN** | +4环（任务对齐） | **0.9793 ± 0.0068** |
| F7-MPNN | +长环 | 0.9660 ± 0.0065 |
| Sub-G（子图 GNN） | 更强 | 0.8623 ± 0.0058 |
| L-G（Local 2-GNN） | 更强 | 0.8543 ± 0.0063 |
| LF-G（Local Folklore 2-GNN） | 最强 | 0.8450 ± 0.0135 |

所有模型训练准确率 >0.99，但与任务（4 环计数）对齐的 F4-MPNN 测试最高；表达力更强的 LF-G 反而最低，直接验证 Theorem 5.1。

### 消融实验表格（Task 2：TUDataset 真实图，准确率 vs. 到训练集的 TMD 距离）

| 数据集 | 现象 | 对应理论 |
|--------|------|---------|
| BZR | TMD 越大测试准确率单调下降（1/3/5 层 GIN 均如此） | Theorem 4.1 的 $\xi_\zeta$ 项 |
| MUTAGENICITY | 同上，且理论界曲线与经验泛化误差几乎贴合 | 界 ≈ 实测 gap |
| NCI109 | 同上 | 同上 |

加标签噪声实验：训练损失仍收敛到 0，但图-标签相关性被破坏后测试误差陡升，说明记忆能力 ≠ 泛化能力。

### 关键发现
- **任务对齐 > 绝对表达力**：F4-MPNN（恰好捕获 4 环）击败所有更强模型。
- **结构距离决定泛化**：测试图离训练集越远（TMD 大），准确率越低，跨数据集一致。
- **界很紧**：相比标准 PAC-Bayes 界（量级约 $10^{16}$），本文界在 $10^4$ 量级，紧了十几个数量级，且与实测 gap 曲线吻合。

## 亮点与洞察
- **把"离散表达力"翻译成"连续可比的伪度量"**：ζ-TMD 让"1-WL/k-WL/motif"等不同档位的表达力第一次能在同一尺度上量化训练-测试相似度，是整套理论能落地的关键枢纽。
- **泛化界的两项分解极具指导性**：复杂度项 vs. 结构相似度项，干净地解释了"表达力的双刃剑"——增表达力既可能降低 $\xi_\zeta$（有益）也可能抬高它（有害），取决于是否与任务对齐。
- **给出可操作的设计准则**：最优表达力 = 恰好捕获任务结构-标签相关性，呼应了"用 TMD 多样性做数据增强能提升泛化"（Southern et al. 2025）的经验发现。
- **理论与经验闭环**：合成任务（可控相关性）+ 真实数据（TMD-准确率曲线）+ 标签噪声（记忆 vs 泛化）三类实验互相印证。

## 局限与展望
- **界不紧、难做模型间直接比较**：不同模型对应不同 TMD，界仅是定性指导而非精确估计，无法直接据其排序模型优劣。
- **相关伪度量未知且昂贵**：实际任务中"标签到底与哪个 ζ-TMD 强相关"通常未知，且 TMD（Wasserstein）计算代价高，限制了直接应用，如何自动识别合适度量是开放问题。
- **理论假设较强**：依赖"标签与某固定伪度量强相关"的 Lipschitz 建模，现实数据未必严格满足。
- **展望**：作者建议探索能提升训练-测试结构相似度的图增强技术（合成图），以系统性地收紧界、提升泛化。

## 相关工作与启发
- **表达力**：1-WL 上界（Xu 2019）、k-GNN（Morris 2020）、子图 GNN（Frasca 2022）、结构/位置编码（Barceló 2021），以及 Jogl et al. 2024 的"输入变换 + 标准消息传递"统一视角（本文强模拟的基础）。
- **泛化**：VC 维（Scarselli 2018）、Rademacher（Garg 2020）、PAC-Bayes（Liao 2021）、graphon 界（Levie 2024; Maskey 2022/2025）；最接近的是 Ma et al. 2021（建模标签-特征对齐，但只支持节点级、不支持可训练 GNN）和 Vasileiou et al. 2025（用 Tree/Forest 距离的鲁棒性框架，但不建模结构-标签相关、仅限标准 MPNN）。
- **启发**：本文把"伪度量 + 结构-标签相关"作为分析 GNN 泛化的核心抓手，提示后续工作（含表达力设计、数据增强、度量学习）都应回到"任务对齐"这个第一性原则上来，而非盲目追求 WL 层级。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 用 ζ-TMD 把表达力与泛化在统一伪度量框架下打通，"何时表达力有害"的判别条件（Thm 5.1）是该问题罕见的清晰理论回答。
- **实验充分度**: ⭐⭐⭐⭐ 合成 + 6 个真实数据集 + 标签噪声 + 界 vs. 实测 gap 对比，10 折交叉验证，覆盖完整；但真实任务规模偏小、缺大图验证。
- **写作质量**: ⭐⭐⭐⭐ 动机—理论—实验逻辑闭环，Figure 1 三联图直击核心矛盾；理论符号较密，对非理论读者门槛偏高。
- **价值**: ⭐⭐⭐⭐⭐ 回答了 GNN 领域长期开放问题（Morris 2024 提出），给出"恰好够用的表达力"这一可操作设计准则，对架构选择与数据增强均有指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Trade-off Between Expressivity and Privacy in Graph Representation Learning](on_the_trade-off_between_expressivity_and_privacy_in_graph_representation_learni.md)
- [\[ICLR 2026\] Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion](glance_for_context_learning_when_to_leverage_llms_for_node-aware_gnn-llm_fusion.md)
- [\[ICLR 2026\] Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization](adaptive_mixture_of_disentangled_experts_for_dynamic_graph_out-of-distribution_g.md)
- [\[ICLR 2026\] GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization.md)
- [\[ICLR 2026\] A Graph Meta-Network for Learning on Kolmogorov–Arnold Networks](a_graph_meta-network_for_learning_on_kolmogorovarnold_networks.md)

</div>

<!-- RELATED:END -->
