---
title: >-
  [论文解读] On the Expressive Power of GNNs for Boolean Satisfiability
description: >-
  [ICLR 2026][图学习][图神经网络表达力] 从 Weisfeiler-Leman (WL) 测试角度严格证明了完整的 WL 层级无法区分可满足与不可满足的 3-SAT 实例，揭示了 GNN 用于 SAT 求解的理论表达力极限，同时识别出平面 SAT 和随机 SAT 等 GNN 可成功区分的正面实例族。
tags:
  - "ICLR 2026"
  - "图学习"
  - "图神经网络表达力"
  - "布尔可满足性"
  - "Weisfeiler-Leman 测试"
  - "SAT 求解"
  - "理论分析"
---

# On the Expressive Power of GNNs for Boolean Satisfiability

**会议**: ICLR 2026  
**arXiv**: [2602.08745](https://arxiv.org/abs/2602.08745)  
**代码**: [GitHub](https://github.com/sakupeltonen/sat-expressivity)  
**领域**: 图学习  
**关键词**: 图神经网络表达力, 布尔可满足性, Weisfeiler-Leman 测试, SAT 求解, 理论分析

## 一句话总结

从 Weisfeiler-Leman (WL) 测试角度严格证明了完整的 WL 层级无法区分可满足与不可满足的 3-SAT 实例，揭示了 GNN 用于 SAT 求解的理论表达力极限，同时识别出平面 SAT 和随机 SAT 等 GNN 可成功区分的正面实例族。

## 研究背景与动机

布尔可满足性问题（SAT）是经典 NP 完全问题。近年来 GNN 成为学习式 SAT 求解的主流架构（如 NeuroSAT、QuerySAT），因为 CNF 公式天然表示为文字-子句二部图（LCG）。然而 GNN 表达力受 WL 测试约束——WL 等价的图输入必然得到相同输出。

**核心问题**：GNN 是否有足够的表达力来推理可满足性？

现有工作缺乏从表达力角度对 GNN 在 SAT 任务上的系统分析。尽管有人可能直觉认为"SAT 是 NP-hard，所以 GNN 必然不够强"，但表达力与计算复杂度是正交概念——例如平面 SAT 也是 NP 完全的，但 4-WL 可以识别所有平面图。

## 方法详解

### 整体框架

本文是一篇纯理论工作，不训练任何模型，而是回答一个问题：把 SAT 公式喂给 GNN，它到底有没有足够的表达力推理出可满足性？由于 GNN 的判别能力被 Weisfeiler-Leman（WL）测试卡死——两个 WL 等价的图输入必然得到相同输出——这个问题等价于在问「WL 测试能不能区分可满足与不可满足的公式」。

整篇论证沿着一条链展开。首先得给 SAT 公式选一个**不丢信息的图表示**，否则讨论"WL 能不能区分两个公式"根本没有意义；本文用补了否定连接的 LCN 表示打底。接着是全文的负面核心：借经典的 Cai-Fürer-Immerman 构造，造出一对"一个可满足、一个不可满足却任意阶 WL 都分不开"的公式，并进一步说明即便顺序求解器边定变量边判，也绕不过这道天花板。然后把负面结论从单个构造推广到一整族——3-正则 SAT 对 WL 彻底失效，却仍是 NP 完全。最后从反面划出 WL 仍然有效的区域（平面 SAT、随机 SAT），并在 G4SAT 随机实例与 SAT 竞赛工业实例上量化"区分一个实例究竟需要几轮 WL"。

### 关键设计

**1. LCN 图表示：把否定连接补回来，否则图本身就丢信息**

学习式求解器普遍把 CNF 公式画成 LCG——文字与子句之间的二部图。本文指出这个表示其实是有漏洞的：一旦去掉节点标签来保持 GNN 需要的排列不变性，仅凭 LCG 就无法分辨一个文字和它的否定，信息从源头就丢了。补救办法是显式加入每个变量与其否定之间的边（并用与文字-子句边不同的颜色标记），得到 LCN（Literal-Clause Graph with Negation connections）。加上这层边后表示变得充分——无标签的 LCN 在同构意义下能唯一确定原 SAT 公式（把按文字-文字边配成对的节点任意标成 $x_i, \neg x_i$、再从文字-子句边读出子句即可还原公式），后续所有关于"WL 能否区分两个公式"的讨论才站得住脚。相比之下变量-子句图（VCG）会让同一公式出现不同构的表示、文字关联图等表示则有损，都不适合做表达力分析。

**2. 主负面结论：存在 WL 永远分不开的"一可满足一不可满足"实例对，且顺序求解也救不回**

这是全文的核心。要证明 WL 表达力不够，就得造出两个公式：一个可满足、一个不可满足，但它们的 LCN 在任意阶 WL 下都等价。构造借用了 Cai-Fürer-Immerman (1992) 的经典手法：给定图 $G$，先写公式 $f_G$ 编码"$G$ 是否存在每个节点出度为偶数的定向"，而这样的定向存在当且仅当边数 $m$ 为偶数——答案只取决于边数奇偶性这一**全局量**；再造一个"扭曲"版本 $\tilde{f}_G$，把其中一条边改成双向，于是 $f_G$ 与 $\tilde{f}_G$ 恰好一个可满足、一个不可满足。关键在于这点扭曲是局部不可见的，二者的 LCN 在 $n$-WL 下仍然不可区分（定理 5.3）：

$$\exists\, f, \tilde{f} \text{ (3-SAT)}: |f| = O(n) \text{ 变量}, \ f \text{ 可满足}, \ \tilde{f} \text{ 不可满足}, \ \text{LCN}(f) \equiv_{n\text{-WL}} \text{LCN}(\tilde{f})$$

这与 Tseitin 公式（resolution 的经典难实例）同出一脉——全局的不一致性无法靠局部消息传递察觉，正是 WL 失效的根源；本文还指出 Cai-Fürer-Immerman 构造与 Tseitin 公式之间的这层联系此前未被注意到。

这个负面结论不止停留在"一次性判定"上。很多 GNN-SAT 求解器（如 QuerySAT）不是一次出答案，而是逐步设置变量、边解边判，一个自然的指望是：固定掉一部分变量后，残余公式或许就能被 WL 区分了。引理 5.4 否掉了这个指望——只要两个公式原本在 $k$-WL 下不可区分，即便已经设置了部分变量，它们的残余公式依然不可区分；推论 5.5 把结论推到极致：哪怕设置了 $\Theta(n)$ 个变量，WL-powerful 的 GNN 仍分不开可满足与不可满足的残余公式。也就是说，靠"试探—回退—重启"的顺序策略同样无法绕过表达力天花板，理论上的 $k$-WL 不可区分被实打实地搬进了现实的计算流程。

**3. 3-正则 SAT：一个 WL 彻底失效却仍是 NP 完全的家族**

定义 $k$-正则 SAT：每个文字恰好出现在 $k$ 个子句中、每个子句恰含 $k$ 个文字。这种高度对称的结构对 WL 是毁灭性的——定理 5.1 证明 3-正则 SAT 仍是 NP 完全的，而观察 5.2 指出 WL 测试无法区分任何两个变量数相同的 3-正则 SAT 公式（所有节点的局部邻域长得一模一样，消息传递得不到任何区分信号）。相比第 2 点里精心扭曲出来的单个实例对，这一整族"难且 WL 看不见"的公式更有说服力：它说明 WL 失效不是构造出来的特例，而会成片地出现在结构规整的实例上。

**4. 正面结果：哪些 SAT 家族 WL 反而能分得开**

负面结果之外，本文也划出 GNN 仍然有戏的区域，正面回击"NP-hard 就一定 WL 不可区分"的直觉。平面 SAT（定理 6.1）：因为平面图能被 4-WL 完全识别，所有平面 SAT 实例都可被 4-WL 区分——而平面 SAT 本身仍是 NP 完全的，这恰好说明**表达力与计算复杂度是正交的两件事**。随机 SAT（引理 6.3）：对用 Wu 和 Ramanujan (2021) 方法从随机文字关联图生成的 CNF 公式，本文证明它们以高概率能被 WL 识别，这也解释了后面实验里随机实例几乎对 WL 毫无难度的现象——而学习式求解器恰恰常在随机实例上训练，于是在工业实例上才会暴露表达力不足。

## 实验关键数据

本文不训练模型，而是直接测量 WL 表达力是否够用。做法是对一个公式运行 $r$ 轮 WL，把落进同一 WL 等价类的文字加上等值约束、得到残余公式 $f_r$，再检查 $f_r$ 是否仍可满足；最小的使 $f_r$ 可满足的轮数记为 $r_{\text{crit}}$，$r_{\text{converge}}$ 则是 WL 颜色不再细化的收敛轮数。$r_{\text{crit}}$ 越小、说明越少的消息传递轮数就足以判定该实例；若公式根本无法靠这种方式被区分，则标为 **unsat**（WL 表达力不足以求解）。下面所有表格里的数字都是这两个量。

> ⚠️ 度量定义以原文为准

### 主实验：随机实例（G4SAT 基准）

| 家族 | 难度 | $r_{\text{crit}}$ | $r_{\text{converge}}$ | 变量数 | 数量 |
|------|------|---------|-------------|--------|------|
| 3-SAT | easy | 2.97±0.18 | 3.68±0.47 | 26±9 | 1000 |
| 3-SAT | medium | 3.00±0.04 | 3.92±0.28 | 119±47 | 1000 |
| 3-SAT | hard++ | 3.08±0.28 | 4.00±0.00 | 5001±62 | 25 |
| k-clique | easy | 4.12±0.73 | 6.26±0.83 | 33±13 | 960 |

随机实例通常只需 3-4 轮 WL 即可区分文字，约 40% 的公式在收敛时所有文字已获得唯一标识。

### SAT 竞赛实例

| 家族 | $r_{\text{crit}}$ | $r_{\text{converge}}$ | 变量数 | 数量 |
|------|---------|-------------|--------|------|
| argumentation | 2.94±0.44 | 4.31±0.87 | 1266±625 | 16 |
| circuit-multiplier | **unsat** | 7.18±0.40 | 1075±50 | 11 |
| cryptography | 15.74±14.67 | 17.63±14.34 | 41510±29705 | 19 |
| heule-nol | **unsat** | 8.60±1.26 | 1419±0 | 10 |
| maxsat-optimum | 26.64±3.64 | 29.27±2.49 | 22157±5623 | 11 |

448 个竞赛实例中仅 234 个可用 WL 求解；69 个家族中 38 个包含 WL 无法区分的实例。

### 消融实验

**3-SAT 的 WL 收敛模式分析**：
- 第 1 轮：每个文字看到自己的度数
- 第 2 轮：无法细分（因为所有邻居子句度数相同 = 3）
- 第 3 轮：文字观察到共享子句中其他文字的度数，实现有效区分
- 第 4 轮：WL 通常收敛

### 关键发现

1. **随机 vs 工业实例差异巨大**：随机实例对 WL 几乎无难度，但工业实例经常超出 WL 表达力
2. **正则结构是致命的**：heule-nol 家族编码网格着色问题，正则结构使 WL 完全失效
3. **与 resolution 复杂度的联系**：本文构造的不可区分实例与 Tseitin 公式（resolution 难实例）结构相似，局部推理无法检测全局不一致性
4. k-clique 和 k-vercov 家族有少量实例（2.0% 和 0.3%）WL 无法求解，源于底层图的对称性

## 亮点与洞察

- **里程碑式的理论结果**：首次证明完整 WL 层级对 SAT 的表达力极限，澄清了"NP-hard = WL 不可区分"的常见误解（定理 6.1 反例）
- **Tseitin-CFI 联系**：发现 Cai-Fürer-Immerman 构造与 Tseitin 公式的深层联系，为 WL 测试和证明复杂度之间建立了新桥梁
- **实践指导清晰**：GNN 适合处理随机 SAT，但对工业 SAT 需要超越 WL 的架构（如对称性破缺、高阶 GNN）

## 局限与展望

1. 理论结果基于最坏情况构造，实际中稀疏/结构化实例可能有更好行为
2. 实验仅测试 WL 等级的表达力是否充分，未考虑泛化性从数据学习的能力
3. 未探索具体的超越 WL 的架构方案（如 k-GNN、随机特征增强）在 SAT 上的效果
4. 工业实例的实验受限于 10MB 大小限制，更大规模实例的行为未知

## 相关工作与启发

- **NeuroSAT / QuerySAT / SATformer**：端到端学习式 SAT 求解器，均受本文理论限制
- **Cai-Fürer-Immerman (1992)**：本文将其经典不可区分图构造推广到 SAT 领域
- **GNN 表达力文献**（GIN, k-GNN, 高阶 WL）：本文为这些工作在 SAT 应用上提供了具体的正/负面结果
- 对未来学习式 SAT 求解器设计有重要启示：需要针对工业实例的结构特性设计专门的表达力增强策略

## 评分

- 新颖性：★★★★★ — 首个系统性的 GNN-SAT 表达力理论分析
- 技术深度：★★★★★ — 证明严格、构造巧妙，与代数/复杂性理论紧密联系
- 实验充分度：★★★★☆ — 随机 + 竞赛实例覆盖全面，验证了理论预测
- 写作质量：★★★★☆ — 结构清晰，理论与实践交织良好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On The Expressive Power of GNN Derivatives](on_the_expressive_power_of_gnn_derivatives.md)
- [\[ICLR 2026\] GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback](gnn-as-judge_unleashing_the_power_of_llms_for_graph_learning_with_gnn_feedback.md)
- [\[ICLR 2026\] Multi-Scale Diffusion-Guided Graph Learning with Power-Smoothing Random Walk Contrast for Multi-View Clustering](multi-scale_diffusion-guided_graph_learning_with_power-smoothing_random_walk_con.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [\[ICLR 2026\] Canonical Tree Cover Neural Networks for Expressive and Invariant Graph Learning](canonical_tree_cover_neural_networks_for_expressive_and_invariant_graph_learning.md)

</div>

<!-- RELATED:END -->
