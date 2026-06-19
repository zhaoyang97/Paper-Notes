---
title: >-
  [论文解读] Deep Neural Sheaf Diffusion
description: >-
  [ICML2026][图学习][神经 sheaf 扩散] 本文指出 Neural Sheaf Diffusion (NSD) 在深层会因 sheaf Laplacian 的"分歧信号"随扩散收敛而消失，从而失去理论上保证的抗坍缩能力；DNSD 用 **sheaf 邻接算子**替代 Laplacian，并配合 LayerNorm、奇函数激活与逐 stalk 门控，使 sheaf 架构第一次能稳定堆叠到 16 层，在合成长程任务上比 GNN/NSD 基线最多提升 30 pp，在真实异质图基准上也一致领先。
tags:
  - "ICML2026"
  - "图学习"
  - "神经 sheaf 扩散"
  - "图神经网络"
  - "sheaf 邻接算子"
  - "过平滑"
  - "图基础模型"
---

# Deep Neural Sheaf Diffusion

**会议**: ICML2026  
**arXiv**: [2605.19021](https://arxiv.org/abs/2605.19021)  
**代码**: https://github.com/remibourgerie/deep-neural-sheaf-diffusion  
**领域**: 图学习  
**关键词**: 神经 sheaf 扩散, 深度 GNN, sheaf 邻接算子, 过平滑, 图基础模型

## 一句话总结
本文指出 Neural Sheaf Diffusion (NSD) 在深层会因 sheaf Laplacian 的"分歧信号"随扩散收敛而消失，从而失去理论上保证的抗坍缩能力；DNSD 用 **sheaf 邻接算子**替代 Laplacian，并配合 LayerNorm、奇函数激活与逐 stalk 门控，使 sheaf 架构第一次能稳定堆叠到 16 层，在合成长程任务上比 GNN/NSD 基线最多提升 30 pp，在真实异质图基准上也一致领先。

## 研究背景与动机

**领域现状**：标准 GNN（GCN / GAT 等）通过逐层的"邻居加权平均"传递信息，理论上层数越深感受野越大；但实践中深层 GNN 普遍训练失败，文献用 *oversmoothing*（节点表征趋同）与 *oversquashing*（远距离信号被压缩）来概括这一困境。其根源在于：消息传递本质上是"凸组合"，反复迭代必然抹平差异。

**现有痛点**：Bodnar 等 (2022) 提出的 Neural Sheaf Diffusion (NSD) 用 **cellular sheaf** 把每条边赋予一个可学习的线性映射 $\mathcal{F}_{v\trianglelefteq e}$，由此构造 sheaf Laplacian $\Delta_\mathcal{F}$ 替代普通图 Laplacian；其理论上证明：在合适的 restriction maps 下，sheaf 扩散的稳态可以分离几乎任意标签配置，从而**不会**因深度而坍缩。然而本文实证发现：这一保证**在实践中失效**——NSD 同样会在层数变深时性能崩塌。

**核心矛盾**：sheaf Laplacian 的本质是"测量相邻 stalk 之间的分歧"，即 $\Delta_\mathcal{F}\mathbf{X}$ 衡量"还没对齐多少"。扩散过程的目标恰恰就是消除分歧，于是 $\Delta_\mathcal{F}\mathbf{X}$ 会随层数增长而**单调趋于 0**，深层网络只能在越来越微弱的残差上做更新，损失对深层参数也变得几乎不敏感；再加上 ReLU 的非对称截断、跨层尺度漂移、噪声成分被均匀传播，理论与实践之间被撕开一道大口子。

**本文目标**：把 NSD 的"理论可深"真正落实成"实践可深"，让 sheaf 架构成为图基础模型 (graph foundation model) 的可堆叠骨干。

**切入角度**：既然问题出在"用分歧量驱动更新"，那就把 update operator 换成"用依赖量驱动更新"——即用 sheaf **adjacency** 算子 $A_\mathcal{F}$ 替代 Laplacian $\Delta_\mathcal{F}$。Bodnar 等的原始推导里其实就出现过这个 sheaf 卷积算子（他们的 Eq. 4），但最终架构里却只把非线性套在 Laplacian 项上、丢掉了 identity 分量，因而把"信号会随深度消失"的副作用引了进来。DNSD 正是把这条被丢弃的路径补回去，再叠上深度训练的标准组件（LayerNorm + 奇激活 + gating），形成一套连贯的"深度可用 sheaf 网络"配方。

**核心 idea**：把 sheaf 扩散从 "subtract a vanishing disagreement" 改为 "aggregate matrix-valued dependency"，并用 LayerNorm/odd activation/per-stalk gating 稳住深层动力学。

## 方法详解

### 整体框架
DNSD 想解决的是"NSD 理论上抗坍缩、实践中却一深就崩"这个落差，做法是把驱动更新的算子从"度量分歧"换成"度量依赖"，再补齐深度网络的标准稳定件。每层为每个节点维护一组 $d\times f$ 的 stalk 表征 $\mathbf{X}_v^{(l)}$，先由两端节点表征学出该层的 restriction maps $\mathcal{F}^{(l)}_{v\trianglelefteq e}$，据此用 sheaf **adjacency** 算子 $A_\mathcal{F}^{(l)}$（而非 Laplacian）聚合邻居，再经奇激活、逐 stalk 门控筛选，最后残差相加并做 LayerNorm。完整的层更新写作

$$\mathbf{X}^{(l+1)} = \mathrm{LN}\!\big((1+\epsilon^{(l)})\mathbf{X}^{(l)} - (\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot \sigma_{\mathrm{odd}}(A_\mathcal{F}^{(l)}\mathbf{X}^{(l)} W_1^{(l)}) W_2^{(l)}\big)$$

输出再投回任务空间做节点分类。和 NSD 原始更新逐处对照：NSD 用 $\Delta_\mathcal{F}$、用 ReLU、无 LayerNorm、无门控；DNSD 把这四处全部换掉，其中 adjacency 替换被消融验证为最关键的因子。

### 关键设计

**1. Sheaf 邻接算子替代 Laplacian：堵住深层信号消失的根因**

NSD 失效的根子在于它的聚合算子 $\Delta_\mathcal{F}=D_\mathcal{F}^{-1/2} L_\mathcal{F} D_\mathcal{F}^{-1/2}$ 度量的是"邻居之间还有多少没对齐的分量"，而扩散迭代恰恰在把这个分歧推向 0，于是深层里 $\sigma(\Delta_\mathcal{F}\mathbf{X} W_1)W_2$ 等于反复把"接近零的小信号"喂给非线性，深层参数几乎收不到梯度。DNSD 把它换成 sheaf adjacency $A_\mathcal{F}$，其块矩阵元为 $(A_\mathcal{F})_{uv}=\mathcal{F}_{u\trianglelefteq e}^\top \mathcal{F}_{v\trianglelefteq e}$，更新项随之变成 $\sigma(A_\mathcal{F}\mathbf{X} W_1)W_2$——用矩阵值边函数聚合邻居的**整体**表征而不是它们的差。这个"依赖信号"不会随扩散收敛而消失，无论初始化阶段还是堆到 16 层都保持信息含量，因此直接修复了"理论保证 ↔ 实践崩塌"的断裂。作者还从 graph attention 的角度给了一个统一解释：GAT 同样是 adjacency-based，但用标量 softmax 注意力，而 DNSD 相当于把 GAT 的 scalar attention scores 换成 matrix-valued 边映射，并把归一化从 attention scores 挪到了节点表征上。

**2. LayerNorm + 奇激活：稳住换 adjacency 后冒出的两个新问题**

换 adjacency 让信号不再消失，但深层暴露出两个新麻烦，需要一组稳定件压住。其一是跨层表征的尺度漂移：连续残差叠非线性后各层 magnitude 不一致，优化随之失稳；DNSD 用 row-wise LayerNorm 把每个 stalk $\tilde{\mathbf{X}}_u^{(l)}\in\mathbb{R}^{d\times f}$ 沿特征维 $f$ 标准化（$\mu_u,\sigma_u\in\mathbb{R}^d$），再用可学习仿射 $\gamma^{(l)},\beta^{(l)}\in\mathbb{R}^f$ 重新拉伸，把前向与反向同时稳住。其二是 ReLU 的非对称截断：在"残差 − message"的减法结构里它只能往一个方向调整，叠多层会让特征几何持续漂移；DNSD 改用有界奇函数 $\sigma_\mathrm{odd}=\tanh$，既保住正负对称，又靠有界性控制更新幅度。这两条思路直接借自深度 Transformer/ResNet，但必须按 stalk 维度组织——若把所有节点拉成一个长向量做 BatchNorm，sheaf 的结构信息就会被打散。

**3. 逐节点逐 stalk 门控：限制噪声沿深度累积**

即便有了 adjacency + LN，反复加权聚合仍会让某些噪声分量（类似 attention sink）随深度累积。DNSD 为每个节点 $u$、每个 stalk 维 $s$ 学一个标量门 $[(\mathbf{G}^{(l)})_u]_s\in[0,1]$，以 $(\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot(\cdot)$ 对"聚合并经非线性后的更新项"做逐通道筛选。这个门由当前 stalk 表征 $\mathbf{X}_{u,s}^{(l)}$ 和"聚合但未过激活"的中间量 $\bar{\mathbf{X}}_{u,s}^{(l)}$ 拼接后过 $\mathrm{sigmoid}(w_g^{(l)}[\cdot;\cdot]+b_g^{(l)})$ 得到，其中 $w_g^{(l)}\in\mathbb{R}^{1\times 2f}$ 在全 stalk 间共享、参数极少。它让模型能有选择地把某些维度"少更新一点"甚至"完全过门"，从而约束噪声沿深度堆积、保护表征质量。

### 损失函数 / 训练策略
任务沿用 NSD 的节点分类设置（合成 G0–G10 与 6 个真实异质图基准），交叉熵损失；restriction maps 取 diagonal 或 full 两种参数化（orthogonal 在深层难训练，作者放到 future work）；层数扫 $\{2,4,8,12,16\}$，公平地复现 NSD 在相同超参预算下的结果。

## 实验关键数据

### 主实验
合成长程任务 G0–G10（3 类社区识别，逐步把 10% 同质边重连成跨社区边）：

| 数据集 (level) | 指标 | DNSD-diag (adj+odd+gate) | NSD-diag | 提升 |
|---|---|---|---|---|
| G4 (L12) | acc % | **86.1 ± 1.8** | 51.2 ± 2.1 | +34.9 pp |
| G5 (L12) | acc % | **81.5 ± 5.5** | 51.2 ± 0.7 | +30.3 pp |
| G6 (L16) | acc % | **75.6 ± 4.7** | 49.1 ± 1.7 | +26.5 pp |
| G7 (L12) | acc % | **63.4 ± 4.4** | 49.1 ± 1.2 | +14.3 pp |
| G10 (L16) | acc % | **96.2 ± 1.3** | 85.5 ± 4.7 | +10.7 pp |

DNSD-full (adj+odd+gate) 在 G10 达到 **97.5 ± 0.8**（NSD-full 仅 84.0 ± 4.0），且最佳层数普遍出现在 L12–L16，而 NSD/MPNN/GAT 的最佳点几乎全部停留在 L2–L4——直接验证了"DNSD 能用深度，NSD 不能"。

### 消融实验
合成数据集上的"逐项加回"消融（diag, 最佳层深）：

| 配置 (diag) | G4 acc | G6 acc | 说明 |
|---|---|---|---|
| 仅原 NSD (无 adj/odd/gate) | 51.2 | 49.1 | 基线，深层失效 |
| + adj | 53.5 → 60+ | 60.4 | **单加 adj** 已经把 G5–G6 推到 60+，是主因 |
| + adj + odd | **86.1**(G4) | 75.6 | 加 odd 激活补足深层稳定性 |
| + adj + gate (无 odd) | 83.5 | 74.4 | gate 单独次要，但有协同 |
| + adj + odd + gate (full) | **75.0**(G5,L16) | 64.6(L16) | full map 配齐三件套，G10 拿 97.5% |

真实异质图基准（Roman Empire / Amazon Ratings / Minesweeper / Tolokers / Questions / Penn94，diag, L≤8）：DNSD 在每个数据集上都跑出"第一/二/三名"中的前列；尽管层数被限制到 ≤8（受限于算力），趋势仍与合成实验一致。

### 关键发现
- **adjacency 替换是主因子**：消融把 adj 单独加上去就能把深层精度从 ~50% 拉到 60–80%，odd activation 和 gating 是"锦上添花的稳定剂"。
- **DNSD 的最佳层数集中在 L12–L16**，而 GNN/NSD 基线的最佳层数停在 L2–L4——这是论文最直接的"可深用"证据。
- **理论保证 ≠ 工程可用**：NSD 的反坍缩定理是真的，但因为信号衰减、ReLU 截断、尺度漂移、噪声累积四个"低层细节"，整套架构在深层依然崩塌；这提示我们看任何"深度可扩展"的图模型理论时都要追问其在工程实现下的有效性。
- **DNSD ≈ matrix-valued GAT with representation normalization**：Table 1 把 GAT / NSD / DNSD 沿"更新算子（依赖 vs 差分）、边变换（标量 vs 矩阵值）、归一化（attention scores vs 表征）、深层行为（平均化 vs 信号消失 vs 缓解衰减）"四轴对齐，是一个非常有启发的统一视角。

## 亮点与洞察
- **"被丢掉的项才是关键"**：DNSD 几乎没有发明新算子——sheaf 卷积 $A_\mathcal{F}$ 早就出现在 Bodnar 等原文的离散化推导里，只是被原始 NSD 架构丢掉了。把"被丢的 identity 分量"补回去就能让深层 sheaf 网络可用，这一类"回到前一篇论文的中间步骤里寻找答案"的研究路径，对未来分析任何"理论强但实践差"的工作都极具借鉴价值。
- **统一 GAT/NSD/DNSD 的对比表**：用"依赖 vs 差分 + 标量 vs 矩阵 + 归一化位置"三轴对照，给"未来什么样的图层适合做基础模型骨干"提供了清晰坐标。
- **可迁移的设计模式**：把 LayerNorm + odd activation + gating 当作"深度可堆叠图模型的标准三件套"——这套配方在任何"层间反复加权聚合"的架构（不只 sheaf，普通 GNN、超图、simplicial complex 上的算子）里都值得一试。

## 局限与展望
- 作者承认 **orthogonal restriction maps** 在深层难以稳定训练，被显式留到 future work；这其实是 sheaf 框架最具表达力的一类参数化，未来如能驯服可能进一步提升。
- 真实数据实验受算力约束只跑到 **L≤8**，DNSD 在更深层（L12–L16）的真实图收益没被完全验证。
- 合成 G 数据集的"deep receptive field requirement"是按"k-NN + 重连"构造的，比较人造；现实大图（如社交网络）的长程依赖结构可能更复杂，DNSD 是否还能保持优势仍需更多验证。
- 计算开销：matrix-valued 边映射比 GAT 的 scalar attention 要贵，论文未深入分析 wall-clock，这是落地图基础模型时绕不开的工程问题。
- 自然延伸：把 DNSD 的"adjacency + 三件套"理念迁回到 **simplicial / cellular complex** 上的更高阶扩散，或与 multi-hop sheaf (Bamberger 等) 结合，可能得到既深又宽的图骨干。

## 相关工作与启发
- **vs NSD (Bodnar et al., 2022)**：DNSD 与 NSD 共享 cellular sheaf 这一数学骨架，但把更新算子从 Laplacian 改成 adjacency，并叠加 LN/odd/gating。NSD 在 L2–L4 即停摆，DNSD 可稳定到 L16，二者本质差异在于"更新到底来自分歧还是依赖"。
- **vs GAT (Veličković et al., 2017; Brody et al., 2021)**：两者都是 adjacency-based，但 GAT 用 scalar 注意力 + softmax 归一化导致 convex aggregation，深层会平均化；DNSD 用 matrix-valued 边映射 + 表征级 LN，允许非凸聚合，因此在深层不会坍缩。可以把 DNSD 看作"GAT 的 matrix-valued、归一化错位版"。
- **vs 多跳 / 注意力型 sheaf 扩张 (Barbero 2022a/b, Zaghen 2024, Bamberger 2024)**：这些工作或预计算映射、或加注意力、或加多跳，但都没正面解决"深层信号消失"问题；DNSD 是第一个把"深度可堆叠"作为头号目标并系统给出方案的 sheaf 架构。
- **vs Transformer 深度训练技巧 (ResNet / LN / 残差)**：DNSD 的 LayerNorm 与残差结构思想直接来自深度网络通用经验，但作者强调"必须按 stalk 维度而非节点维度做归一化"——把通用技巧落到 sheaf 框架时仍需结构感知。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新构造一个数学对象，而是发现原文里"被遗弃的项"才是关键，并配齐深度训练标准件，洞察价值很高。
- 实验充分度: ⭐⭐⭐⭐ 合成数据上系统扫了 5 个层深 × 11 个扰动等级 × 多种配置；真实数据覆盖 6 个异质图基准；唯一遗憾是真实图层数受限到 L≤8。
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑链非常清晰，从"理论保证 → 实践失效 → 四个机理 → 四点修复 → GAT 视角统一解释"层层推进，Table 1 的三轴对比堪称范例。
- 价值: ⭐⭐⭐⭐ 直接给"图基础模型可深堆叠骨干"提供了一个有据可循的候选，并把"adj vs Laplacian"这一选择推到了所有 sheaf 后续工作必须正视的位置。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](../../ICLR2026/graph_learning/cooperative_sheaf_neural_networks.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] What Makes a Desired Graph for Relational Deep Learning?](what_makes_a_desired_graph_for_relational_deep_learning.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)

</div>

<!-- RELATED:END -->
