---
title: >-
  [论文解读] Learning Flexible Forward Trajectories for Masked Molecular Diffusion
description: >-
  [ICLR 2026][计算生物][掩码扩散模型] 本文发现把掩码扩散模型（MDM）直接搬到分子图生成上会因为不同分子在前向加噪中坍缩到同一中间态（state-clashing）而严重退化，于是提出 MELD——用一个可学习的噪声调度网络给每个原子/化学键分配各自的掩码速率，让前向轨迹相互错开，从而在 QM9/ZINC250K 上做到 100% 化学有效性且分布对齐 SOTA。
tags:
  - "ICLR 2026"
  - "计算生物"
  - "掩码扩散模型"
  - "分子图生成"
  - "可学习噪声调度"
  - "state-clashing"
  - "逐元素扩散"
---

# Learning Flexible Forward Trajectories for Masked Molecular Diffusion

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=raVuVPbnQL](https://openreview.net/forum?id=raVuVPbnQL)  
**代码**: [https://holymollyhao.github.io/MELD](https://holymollyhao.github.io/MELD)  
**领域**: 计算生物 / 分子生成 / 离散扩散  
**关键词**: 掩码扩散模型, 分子图生成, 可学习噪声调度, state-clashing, 逐元素扩散  

## 一句话总结
本文发现把掩码扩散模型（MDM）直接搬到分子图生成上会因为不同分子在前向加噪中坍缩到同一中间态（state-clashing）而严重退化，于是提出 MELD——用一个可学习的噪声调度网络给每个原子/化学键分配各自的掩码速率，让前向轨迹相互错开，从而在 QM9/ZINC250K 上做到 100% 化学有效性且分布对齐 SOTA。

## 研究背景与动机
**领域现状**：掩码扩散模型在文本、图像等离散数据上表现亮眼（D3PM、MaskGIT、MDLM、MD4），它把前向过程定义为"逐元素掩码"，反向过程则并行地填补被 mask 的元素，兼具自回归式的质量和扩散式的并行采样效率。分子生成此前主要靠 score-based（GDSS、GruM）或 substitution-based 离散扩散（DiGress），MDM 在分子图上几乎是空白地带。

**现有痛点**：作者把标准 MDM 直接套到分子图上，结果发现性能不升反降——能生成有效分子，但分布对齐极差（ZINC250K 上 FCD 比最佳基线高 91%、scaffold 相似度低 99.8%）。这不是简单的调参问题，而是结构性缺陷。

**核心矛盾**：问题根源在于 **state-clashing（状态坍缩）**。分子图词表小、对称性强，用一个对所有原子和键都相同的"元素无关"掩码速率加噪时，两个语义完全不同的分子很容易被掩码成同一个中间态。比如把 o-苯二胺和 m-苯二胺的氮碳键都 mask 掉，两者都坍缩成同一个对称苯环骨架。此时后验 $p(g\mid g_t)$ 是高度多模态的（多个原始分子对应同一个 $g_t$），但 MDM 的去噪器是**单模态**的——它独立地预测每个节点和边 $p_\theta(g\mid g_t)=\prod_i p_\theta(x^i\mid g_t)\prod_{i<j}p_\theta(e^{ij}\mid g_t)$，只能输出一个"平均图"。叠加 KL 散度的 mode-covering 特性，模型只能把概率质量摊成高熵分布，生成的分子自然偏离真实分布甚至违反化学规则。

**本文目标**：让 MDM 在保持并行解码优势的同时，避免不同分子在前向过程中撞到同一个中间态。

**核心 idea**：**[让前向过程也变得可学习]** 与其用固定的、元素无关的噪声调度，不如给每个图元素（原子、键）学一条专属的腐蚀轨迹，使不同分子的加噪路径尽量错开，从源头减少 state-clashing。

## 方法详解

### 整体框架
MELD（Masked Element-wise Learnable Diffusion）在标准 MDM 之外多加了一个**噪声调度网络**，把前向过程从"固定调度"变成"可学习、逐元素"的调度，并与去噪器联合训练。训练时同时优化前向（噪声调度网络 $\phi$）和反向（去噪器 $\theta$），让每个原子/键拥有不同的掩码速率 $\gamma^i_{t,\phi}$，从而把原本会撞在一起的前向轨迹拉开。

```mermaid
flowchart LR
    A[分子图 g0<br/>节点x/边e] --> B[可学习元素嵌入 H]
    B --> C[噪声调度网络 fφ<br/>逐元素 power-law]
    C --> D["逐元素掩码速率 γ^i_t,φ"]
    D --> E[前向加噪 q_φ-gt-g0]
    E --> F[STGS 可微采样<br/>保持梯度流]
    F --> G[去噪器 pθ<br/>DiT backbone]
    G --> H[重构 g0]
    H -. 联合优化 θ,φ .-> C
```

### 关键设计

**1. 可学习的逐元素嵌入：给每个原子/键一个可区分的"身份证"**  要让调度网络给不同元素分配不同速率，首先得有信息区分这些元素。作者指出图位置编码在芳香环这类对称结构里会失效（无法区分等价节点），而把含噪图本身喂进调度网络又会破坏前向边际 $q(g_t\mid g_0)$ 的闭式可解性。于是他们直接学一个嵌入矩阵 $H\in\mathbb{R}^{D\times N}$，第 $i$ 列 $h_i$ 作为第 $i$ 个节点的嵌入，边 $\{i,j\}$ 的嵌入取 $h_{ij}=h_i+h_j$。训练中随机置换 $H$ 的列，使节点数/边数相同但拓扑不同的图状态也能被区分开。

**2. 逐元素 power-law 噪声调度：把"何时被 mask"交给网络决定**  噪声调度沿用 power-law 形式，但指数由元素嵌入决定。对节点 $i$，掩码概率为

$$\gamma^i_{t,\phi}=1-(1-\epsilon)\cdot t^{w^i_\phi},\qquad w^i_\phi=\sigma_{sf}\big(f_\phi(h_i)\big)$$

其中 $\sigma_{sf}$ 是 softplus、$f_\phi$ 是线性层、$\epsilon=10^{-4}$ 保证数值稳定。每个原子和键因此有了各自的腐蚀速率——比如延迟氮原子被 mask 的时机，就能避免 o-苯二胺过早坍缩成对称苯环。训练损失沿用 MDM 的加权交叉熵积分（公式 2），但 $\gamma$ 及其导数 $\dot\gamma$ 现在依赖 $\phi$，调度网络因此被纳入梯度。

**3. STGS 保持离散采样的梯度流：让前向也能反传**  分子离散扩散每一步要从类别分布里**采样**一个 one-hot 图，这个 argmax 操作会切断指向调度参数 $\phi$ 的梯度。作者用 Straight-Through Gumbel-Softmax 解决：先用 Gumbel-Softmax 得到软分布 $p_{soft,k}=\frac{\exp((z_k+g_k)/\eta)}{\sum_l\exp((z_l+g_l)/\eta)}$（$g_k$ 为 Gumbel 噪声），再取 argmax 得硬 one-hot $p_{hard}$，最后令 $p=p_{hard}-\text{sg}(p_{soft})+p_{soft}$。前向用离散 $p_{hard}$ 构图，反向把它当连续 $p_{soft}$，梯度 $\frac{\partial p}{\partial z}=\frac{\partial p_{soft}}{\partial z}$ 得以贯通整个前向过程实现端到端训练。

**4. 靠置换边际化保证分布的排列不变性**  逐元素调度 + 可学习嵌入天然依赖节点顺序，而图生成只要求学到的分布是排列不变的（不必架构层面严格等变）。MELD 不约束架构，而是对所有排列做边际化：$p(g)=\sum_\pi p(g,\pi)$，训练时随机置换 $H$ 的列恰好对应最大化该边际对数似然的 ELBO，$\log p(g)\ge \mathbb{E}_\pi[\log p(g\mid\pi)]+\text{const}$。这种随机对称化是自回归图生成里成熟的范式。

## 实验关键数据

### 主实验表格

QM9 / ZINC250K 无条件生成（生成 1 万分子，↑越大越好/↓越小越好）：

| Method | QM9 Valid.↑ | QM9 FCD↓ | QM9 Scaf.↑ | ZINC Valid.↑ | ZINC FCD↓ | ZINC Scaf.↑ |
|---|---|---|---|---|---|---|
| GruM (最强 baseline) | 99.69 | 0.11 | 0.945 | 98.65 | 2.26 | 0.530 |
| DiGress | 98.19 | 0.10 | 0.936 | 94.99 | 3.48 | 0.416 |
| MDM w/ power-law | 100.00 | 3.62 | 0.628 | 100.00 | 26.09 | 0.001 |
| **MELD (Ours)** | **100.00** | **0.09** | **0.947** | **100.00** | **1.51** | **0.559** |

标准 MDM 虽 100% 有效却分布严重失配（ZINC Scaf. 仅 0.001），MELD 把 FCD 从 26.09 压到 1.51、Scaf. 拉到 0.559，且 100% 有效。

Polymer 属性条件生成（11 项约束 + 合成分），平均 MAE：

| Method | Valid.↑ | FCD↓ | MAE↓ |
|---|---|---|---|
| GraphDiT (前 SOTA) | 82.45 | 6.64 | 0.921 |
| MDM w/ power-law | 17.31 | 26.56 | 1.620 |
| **MELD (Ours)** | **99.10** | **5.93** | **0.798** |

相比 GraphDiT，MELD 平均 MAE 降 13.4%；相比标准 MDM，有效性提升约 5 倍、属性对齐平均改善 50%。

### 消融实验表格

ZINC250K 上不同噪声调度策略对比：

| Schedule | Method | FCD↓ | NSPDK↓ | Scaf.↑ |
|---|---|---|---|---|
| Fixed | Power-law | 26.09 | 0.0683 | 0.001 |
| Fixed | DiffusionBERT | 1.95 | 0.0009 | 0.491 |
| Learnable | GenMD4 (类别级) | 3.19 | 0.0017 | 0.429 |
| Learnable | TabDiff (列共享) | 2.15 | 0.0009 | 0.486 |
| Learnable | MELD (仅节点) | 1.63 | 0.0009 | 0.536 |
| Learnable | MELD (仅边) | 1.73 | 0.0009 | 0.525 |
| Learnable | **MELD (节点+边)** | **1.51** | **0.0006** | **0.559** |

类别级（GenMD4）或列共享（TabDiff）调度都不如真正的逐元素调度——前者延迟所有碳原子仍可能坍缩成对称苯环，只有 per-element 的细粒度控制才能彻底解决 state-clashing。

### 关键发现
- **几乎零开销**：MELD 只多加一个嵌入矩阵 $H$，约 +0.01M 参数；在 10–200 原子规模下 FLOPs、显存、单步耗时与标准 MDM 基本持平（200 原子时 0.165s vs 0.132s）。
- **更快重构**：可学习调度让 MELD 在反向过程中比固定 power-law 更早还原分子片段（图 3，$t=T/4$ 时已显著恢复）。
- **可扩展到大分子**：在 Guacamol 大规模数据集上以 300 epoch（DiGress 需 1000 epoch）即超越所有扩散基线，100% 有效。

## 亮点与洞察
- **问题诊断比方法更值钱**：本文真正的贡献是揭示并形式化了 state-clashing——把"MDM 在分子上失效"归因到"对称结构 + 小词表 + 元素无关掩码"的组合，并用预测熵可视化（图 2）直观佐证。这种"先讲清为什么会坏"的叙事比单纯堆方法更有说服力。
- **让前向过程可学习**：多数扩散工作只学反向去噪、把前向当固定先验，MELD 把前向噪声调度也参数化并联合训练，是一个干净且通用的视角转换。
- **极小代价撬动大收益**：仅 +0.01M 参数就把分布对齐从崩溃拉到 SOTA，性价比极高，工程上几乎可无痛集成进现有 DiT-based MDM。

## 局限与展望
- **优势依赖领域特性**：作者自己指出，文本/蛋白质序列词表大、对称少，state-clashing 本就少见，MELD 的相对收益会减弱——方法的"杀手锏"高度绑定分子图这类高对称小词表场景。
- **置换边际化是近似**：靠随机置换 $H$ 来近似排列不变性，是 ELBO 下界而非严格保证，对大图或高度对称分子的覆盖充分性存疑。
- **属性对齐-多样性权衡**：条件生成里 MELD 多样性（85.91）略低于部分基线，存在 alignment 与 diversity 的固有取舍，未深入探讨如何调控。
- **可学习嵌入的可解释性**：学到的逐元素速率到底对应什么化学直觉（哪些键该晚 mask）缺乏系统分析，可作为后续可解释性方向。

## 相关工作与启发
- **掩码扩散谱系**：D3PM 引入 absorbing mask、MaskGIT 做并行图像解码、MD4/MDLM 把扩散目标简化为加权交叉熵——MELD 站在这条线上，把"可学习调度"从文本/表格（GenMD4、TabDiff、DiffusionBERT）迁移到分子图并做成 per-element。
- **分子扩散两条路线**：score-based（GDSS、GruM 用 SDE 连续松弛）和 substitution-based 离散扩散（DiGress 的 Markov 转移）；MELD 开辟了 mask-based 这第三条，且保留并行解码。
- **启发**：state-clashing 的视角可推广到任何"高对称 + 小词表"的离散生成场景（如某些结构化代码/序列）；"把前向噪声调度参数化并联合训练"也可能反哺连续扩散——不同样本何时加多少噪声本就不该一刀切。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — state-clashing 的发现与形式化新颖且解释力强，逐元素可学习调度是干净的视角转换；调度可学习本身在文本/表格已有先例，故非满分。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖无条件/条件生成、5 个数据集、含消融与计算开销分析、大分子可扩展性；多样性权衡和 ELBO 近似的边界探讨略浅。
- **写作质量**: ⭐⭐⭐⭐⭐ — "先讲清病因再开方"的叙事极清晰，预测熵可视化与反向重构对比图直观有力。
- **价值**: ⭐⭐⭐⭐ — 几乎零开销即把 MDM 在分子生成上从崩溃拉到 SOTA，工程可落地性强；但收益高度依赖分子图的领域特性，跨域普适性有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DCFold: Efficient Protein Structure Generation with Single Forward Pass](dcfold_efficient_protein_structure_generation_with_single_forward_pass.md)
- [\[ICLR 2026\] Graph Diffusion Transformers are In-Context Molecular Designers](graph_diffusion_transformers_are_in-context_molecular_designers.md)
- [\[ICLR 2026\] Enhancing Diffusion-Based Sampling with Molecular Collective Variables](enhancing_diffusion-based_sampling_with_molecular_collective_variables.md)
- [\[ICLR 2026\] SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion](sigmadock_untwisting_molecular_docking_with_fragment-based_se3_diffusion.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](controllable_sequence_editing_for_biological_and_clinical_trajectories.md)

</div>

<!-- RELATED:END -->
