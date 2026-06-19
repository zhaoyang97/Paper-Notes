---
title: >-
  [论文解读] Finding Diverse Solutions Parameterized by Cliquewidth
description: >-
  [AAAI 2026][参数化复杂性] 将"寻找多样化解"的参数化框架从treewidth扩展到更强的cliquewidth图参数，证明任何基于cliquewidth分解的单调动态规划都可以以极小额外开销转换为求解多样化版本的算法，并提出了一族新的Venn多样性度量函数。 问题背景 在组合优化中，通常只求解一个最优解…
tags:
  - "AAAI 2026"
  - "参数化复杂性"
  - "多样性解"
  - "cliquewidth"
  - "动态规划"
  - "MSO1逻辑"
  - "Venn多样性度量"
---

# Finding Diverse Solutions Parameterized by Cliquewidth

**会议**: AAAI 2026  
**arXiv**: [2405.20931](https://arxiv.org/abs/2405.20931)  
**作者**: Karolina Drabik, Tomáš Masařík (University of Warsaw)  
**代码**: 未公开（纯理论工作）  
**领域**: 其他  
**关键词**: 参数化复杂性, 多样性解, cliquewidth, 动态规划, MSO1逻辑, Venn多样性度量  

## 一句话总结

将"寻找多样化解"的参数化框架从treewidth扩展到更强的cliquewidth图参数，证明任何基于cliquewidth分解的单调动态规划都可以以极小额外开销转换为求解多样化版本的算法，并提出了一族新的Venn多样性度量函数。

## 研究背景与动机

### 问题背景
在组合优化中，通常只求解一个最优解。然而在实际应用中，数学建模往往是对真实问题的抽象，可能遗漏了难以形式化的约束。因此，与其寻找单一最优解，不如寻找一组**尽可能不同的解**（diverse solutions），让决策者从中选择最适合实际需求的方案。这一思想在约束编程、混合整数优化、进化计算、社会选择等领域已有广泛应用。

### 已有工作的不足
- Baste等人(2022)证明了在**treewidth**参数化下，可以将求解单个问题的DP转换为求解多样化版本，且额外开销很小
- 但treewidth只能描述稀疏图类，对于稠密图类（如团图、完全二部图等）treewidth很大
- **Cliquewidth**是比treewidth更强的图参数，能够额外描述某些稠密图类，但此前缺乏在cliquewidth上求解多样化问题的系统性框架
- 已有研究仅关注 DivSum 和 DivMin 两种多样性度量，而这两种度量各自存在退化情况

### 核心动机
填补cliquewidth参数化下多样化问题的理论空白，同时提出更丰富的多样性度量函数族，为实际应用提供更灵活的多样性衡量工具。

## 方法详解

### 问题建模
- **顶点问题**：给定图 $G$，求满足某种性质的顶点子集 $S \subseteq V(G)$（如顶点覆盖、独立集、支配集等）
- **多样化问题**：给定 $r$ 个顶点问题 $\mathcal{P}_1, \ldots, \mathcal{P}_r$、多样性度量 Div 和阈值 $d$，求 $r$ 个解 $S_1, \ldots, S_r$ 使得 $\mathrm{Div}(S_1, \ldots, S_r) \geq d$
- **Hamming距离**：$\mathrm{HamDist}(S, S') = |(S \setminus S') \cup (S' \setminus S)|$
- **两种经典度量**：DivSum（所有对的Hamming距离之和）和 DivMin（所有对中最小的Hamming距离）

### 新贡献1：Venn多样性度量族
本文提出了一族新的多样性度量——**Venn f-多样性**。核心思想是：对每个顶点 $v$，根据其在 $r$ 个解中的"成员向量" $m(S_1,\ldots,S_r,v) \in \{0,1\}^r$，通过函数 $f:\{0,1\}^r \to \mathbb{N}$ 计算其"影响力"，然后对所有顶点求和：

$$\mathrm{Div}_f(S_1, \ldots, S_r) = \sum_{v \in V} f(m(S_1, \ldots, S_r, v))$$

DivSum 是Venn度量的特例，其中 $f(m) = |\{i: m[i]=1\}| \cdot |\{j: m[j]=0\}|$。文中还提出了 Div* 度量来惩罚 DivSum 中多次复制两个不相交集合的退化行为：

$$\mathrm{Div}^*(S_1,\ldots,S_r) = \sum_{v \in V}(r^2 - |\{i \in [r]: v \in S_i\}|^2)$$

### 新贡献2：单调动态规划框架
**单调性定义**：一个DP核心 $\mathfrak{C}$ 是单调的，如果存在顶点成员函数 $\rho$，使得每个解 $S$ 可以完全由witness $\alpha$ 在叶节点上的值通过 $\rho$ 确定。直觉上，每个部分解是其子树部分解的不相交并集。

**主定理（Theorem 1.2）**：给定 $r$ 个顶点问题的单调DP，每个在分解节点上最慢运行时间为 $t_\mathcal{D}$，则对任意Venn多样性函数，多样化版本可在 $\mathcal{O}(|V(\mathcal{D})| \cdot t_\mathcal{D}^r)$ 时间内求解。运行时间**不依赖于多样性目标值** $d$。

**关键Lemma 3.8**：Venn多样性函数满足可加性——对不相交集合 $S$ 和 $P$ 上的部分解：

$$\mathrm{Div}(S_1 \cup P_1, \ldots, S_r \cup P_r) = \mathrm{Div}(S_1, \ldots, S_r) + \mathrm{Div}(P_1, \ldots, P_r) - |V(G)| \cdot f(0^r)$$

这使得DP的合并操作可以正确地组合多样性值。

### 新贡献3：MSO1可表达问题的线性FPT算法
**推论1.5**：任何MSO1可表达的顶点问题的多样化版本，都可以在以cliquewidth、解的数量和公式中量词数为参数的线性FPT时间内求解。

证明基于对Courcelle-Makowsky-Rotics定理的重新构造。使用**归约求值树**（reduced evaluation trees）和**偏求值树乘积**（tree product）操作，设计了新的单调DP。关键步骤：
1. 对每个cliquewidth分解节点构造偏求值树
2. 通过同构类归约保持树的大小有界（大小为关于量词数和cliquewidth的塔函数）
3. 证明归约后的树乘积与未归约版本同构（Lemma 5.12, Combining Lemma）
4. 证明构造的DP核心满足单调性

### DivMin的处理
对于 DivMin 度量（Theorem 1.3），需要存储所有 $\binom{r}{2}$ 对之间的Hamming距离向量（而非单个标量），因此运行时间增加到 $\mathcal{O}(|V(\mathcal{D})| \cdot t_\mathcal{D}^r \cdot d^{r^2})$，因为距离向量不可比较，需要保留所有可能的距离组合。

### 示例：多样化顶点覆盖
论文以 k-Vertex Cover 完整展示了框架应用：
1. 设计cliquewidth上的标准DP，运行时间 $\mathcal{O}(|V(\mathcal{D})| \cdot (k+1)^{2\omega} \cdot \omega)$
2. 证明该DP是单调的（Lemma 4.3）
3. 应用主定理得到多样化版本，运行时间 $\mathcal{O}(|V(\mathcal{D})| \cdot (k+1)^{2r\omega} \cdot r\omega)$
4. 展示了一个非单调DP的例子（k-Min Vertex Cover），说明非单调DP可以轻易改造为单调的

## 理论结果汇总

### 表1：主要定理对比

| 定理 | 图参数 | 多样性度量 | 运行时间 | 来源 |
|------|--------|-----------|---------|------|
| Theorem 1.4 | treewidth | DivSum | $\mathcal{O}(\lvert V(\mathcal{T})\rvert \cdot t_\mathcal{T}^r)$ | Baste et al. 2022（改进） |
| **Theorem 1.2** | **cliquewidth** | **Venn f-多样性** | $\mathcal{O}(\lvert V(\mathcal{D})\rvert \cdot t_\mathcal{D}^r)$ | **本文** |
| **Theorem 1.3** | **cliquewidth** | **DivMin** | $\mathcal{O}(\lvert V(\mathcal{D})\rvert \cdot t_\mathcal{D}^r \cdot d^{r^2})$ | **本文** |
| **Corollary 1.5** | **cliquewidth** | **Venn / DivMin** | **线性FPT** | **本文** |

核心发现：cliquewidth上的额外开销（从单个解到 $r$ 个多样化解）与treewidth上完全一致——运行时间的 $r$ 次方，这表明更强的图参数并不会导致多样化的额外代价。

### 表2：多样化问题的逻辑-图参数复杂性景观

| 逻辑 | 图参数 | 多样性度量 | 复杂度类 | 来源 |
|------|--------|-----------|---------|------|
| MSO2 | treewidth | DivSum | 线性FPT | Baste et al. 2022 |
| FO | nowhere dense | DivSum/DivMin | FPT（非线性） | Hanaka et al. 2021 |
| A&C DN | cliquewidth | DivSum/DivMin | 三次FPT | Bergougnoux et al. 2023 |
| A&C DN | mimwidth | DivSum/DivMin | XP | Bergougnoux et al. 2023 |
| **MSO1** | **cliquewidth** | **Venn f-多样性** | **线性FPT** | **本文** |

各结果在逻辑表达力和图类范围上不可比——更强的逻辑覆盖更多问题，更弱的图参数覆盖更大的图类。本文填补了MSO1+cliquewidth这一自然组合的空白。注意MSO2在cliquewidth上的模型检验可能不在XP内（除非E=NE），因此MSO1是cliquewidth上能期望的最强逻辑。

## 亮点

- **自然而重要的理论拓展**：从treewidth到cliquewidth的推广方向自然，但技术上需要克服新的挑战（cliquewidth分解的节点类型不同于tree decomposition），且cliquewidth能覆盖稠密图类，极大拓展了适用范围
- **Venn多样性度量族**：提出了整个度量函数族而非单一度量，允许用户根据实际需求和已获解的退化情况动态选择度量函数，具有独立的理论价值。例如 Div* 度量可惩罚 DivSum 退化地复制少量不同解的行为
- **几乎最优的额外开销**：多样化版本的运行时间仅是原始DP运行时间的 $r$ 次方，与treewidth上的结果一致，且对Venn度量不依赖目标值 $d$
- **单调性概念的提出与验证**：形式化了"单调DP"的概念，声称文献中绝大多数cliquewidth上的DP都是单调的或可以轻易改造为单调的，并通过非单调DP的反例（Section 4.4）加以说明
- **完整的MSO1构造性证明**：通过归约求值树重新证明了Courcelle定理的cliquewidth版本，并证明得到的DP满足单调性，这是一个技术上non-trivial的贡献

## 局限与展望

- **仅限顶点问题**：当前框架仅适用于顶点子集问题，未扩展到边问题（如哈密顿回路），论文提出这是一个开放问题——cliquewidth参数化下哈密顿回路的多样化版本是否有XP算法？
- **DivMin的额外开销**：DivMin度量需要 $d^{r^2}$ 的额外因子，是否可以消除是另一个开放问题
- **纯理论工作**：没有任何实验验证，缺乏对不同Venn多样性度量在实际问题上的实证比较，论文作者也建议未来进行系统性的实验研究
- **Venn度量不包含DivMin**：DivMin不是Venn度量的特例，因此主定理需要单独处理，且运行时间更差
- **非单调DP的通用转化**：虽然声称大多数DP是单调的，但论文未给出将非单调DP系统性转化为单调DP的通用方法
- **参数函数的塔型增长**：MSO1结果中，关于cliquewidth和量词数的参数函数是多层指数塔，虽然是FPT但实际可行性有限

## 与相关工作的对比

- **Baste et al. (AI 2022)**：在treewidth上建立了多样化问题框架，支持MSO2逻辑和DivSum度量。本文将其推广到cliquewidth，逻辑从MSO2降为MSO1（这是理论上必要的），但新增了Venn度量族的支持
- **Hanaka et al. (2021)**：在nowhere dense图类上对FO可定义问题给出FPT多样化算法，图类更广但逻辑更弱，运行时间远差于线性，且严重依赖模型检验机器 [Grohe-Kreutzer-Siebertz]
- **Bergougnoux, Dreier, Jaffke (2023)**：在cliquewidth上对A&C DN逻辑给出三次FPT算法；在mimwidth上给出XP算法。A&C DN仅是MSO1的存在性片段，与本文不可比
- **多样化问题的专题研究**：多样化命中集 [BJM+19]、多样化最小s-t割 [dLS23]、拟阵中的多样性 [FGP+24]、多样化匹配对 [FGJ+24]、多样化SAT [MMR24]等，均针对特定问题而非一般性框架
- **cliquewidth上的DP文献**：反馈顶点集 [BSTV13]、三角检测 [CDP19]、哈密顿回路 [BKK19]、Steiner树 [BK24]等，本文框架可直接应用于这些单调DP的多样化版本

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从treewidth到cliquewidth的推广方向自然，但Venn度量族的提出和MSO1的构造性证明具有技术新意
- 实验充分度: ⭐⭐ — 纯理论工作，无实验验证，缺乏实证分析
- 写作质量: ⭐⭐⭐⭐⭐ — 条理清晰，定义严谨，示例丰富（顶点覆盖的完整四节展示、退化行为的Venn图和路径图示例），开放问题讨论充分
- 价值: ⭐⭐⭐⭐ — 填补了cliquewidth+MSO1的自然理论空白，Venn度量族的提出为多样性研究提供了新视角

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)
- [\[CVPR 2026\] Bias In, Bias Out? Finding Unbiased Subnetworks in Vanilla Models](../../CVPR2026/others/bias_in_bias_out_finding_unbiased_subnetworks_in_vanilla_models.md)
- [\[ICML 2026\] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks](../../ICML2026/others/how_the_optimizer_shapes_learned_solutions_in_equivariant_neural_networks.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](../../CVPR2025/others/practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)

</div>

<!-- RELATED:END -->
