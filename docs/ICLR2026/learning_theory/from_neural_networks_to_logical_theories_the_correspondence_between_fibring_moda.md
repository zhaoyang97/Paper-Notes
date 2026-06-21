---
title: >-
  [论文解读] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks
description: >-
  [ICLR 2026][学习理论][fibring] 本文首次在 fibring 神经网络（把一个网络的预激活喂给一个 fibring 函数去生成另一个网络的权重与输入，再把子网络输出注回母网络）与 fibring 模态逻辑之间建立**精确对应**，并据此把 GNN、GAT、Transformer encoder 统一刻画为 fibred 模态逻辑公式片段，给出三者的**非均匀逻辑表达力**结果。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "神经符号 AI（逻辑表达力）"
  - "fibring"
  - "模态逻辑"
  - "Kripke 模型"
  - "图神经网络"
  - "GAT"
  - "Transformer"
  - "非均匀表达力"
  - "神经符号 AI"
---

# From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=P1iAEhonhY](https://openreview.net/forum?id=P1iAEhonhY)  
**代码**: 待确认  
**领域**: 学习理论 / 神经符号 AI（逻辑表达力）  
**关键词**: fibring、模态逻辑、Kripke 模型、GNN、GAT、Transformer encoder、非均匀表达力、神经符号 AI  

## 一句话总结
本文首次在 fibring 神经网络（把一个网络的预激活喂给一个 fibring 函数去生成另一个网络的权重与输入，再把子网络输出注回母网络）与 fibring 模态逻辑之间建立**精确对应**，并据此把 GNN、GAT、Transformer encoder 统一刻画为 fibred 模态逻辑公式片段，给出三者的**非均匀逻辑表达力**结果。

## 研究背景与动机
**领域现状**：神经符号 AI 的一条主线是把现代神经架构与逻辑推理对应起来，以求更可解释、可验证。已有工作分别刻画了 GNN（被 Weisfeiler-Leman 测试上界、Presburger 逻辑给出均匀表达力、有界 GNN 对应一阶逻辑片段）和 Transformer encoder（UHAT 对应 AC⁰ 片段、AHAT 对应 LTL(C,+) 等带计数模态的时序逻辑）的逻辑表达力。同时 Bronstein 等人指出 Transformer encoder 本质上是作用在完全图上、带位置编码与注意力的 GNN。

**现有痛点**：尽管 GNN 与 Transformer 在表达力刻画上共享"计数模态"等相似元素，但**至今没有统一理论同时覆盖两类架构**。另一方面，2004 年提出的 fibring 神经网络（让一个网络的参数成为另一个网络的函数）虽然受 fibring 逻辑启发，但**它与 fibring 模态逻辑之间的精确对应一直未被严格建立**——这条 20 年前埋下的线索始终是非形式化的类比。

**核心矛盾**：fibring 逻辑天然支持"把可数族模态逻辑组合进单一 fibred 语言并共享语义"，这恰好是统一多架构所需的组合机制；但缺少把神经计算映射到 Kripke 语义的形式化桥梁，使得这一潜力无法兑现。

**本文目标**：补上这道缺口——形式化"与 fibred 神经网络相容的 fibred 模型"，证明 fibred 神经网络的输出与某个 fibred 逻辑公式的真值**逐输入一致**，再用它统一推导 GNN/GAT/Transformer 的非均匀表达力。**核心 idea**：把神经网络看作"逐层叠加的底层模态逻辑的组合"，用 fibring 作为统一镜头。

## 方法详解

### 整体框架
论文先把 2004 年的 fibring 神经网络重新定义为一棵以神经架构为节点的有向树（fibring architecture），每条边指定母网络的某一层与一组"fibred 神经元"位置；再在逻辑侧构造一个 fibred 模态语言，为树中每个 (节点, 层) 配一个模态算子 □_{v,ℓ}，并定义"相容 fibred 模型"这一子类作为合法的 fibred 逻辑 L_F。两侧通过"相容性条件"对齐：网络的输入-输出行为对应 Kripke 世界间的可达关系，母网络向子网络的委派对应世界间的 fibring 跳转。最后把 GAT（及作为其特例的 GNN、作为完全图 GAT 的 Transformer encoder）的计算改写成 fibred 神经网络，从而读出它们对应的 fibred 逻辑公式族。

```mermaid
graph TD
    A[母网络 N 计算到第 ℓ 层<br/>得到预激活 x_i] --> B[fibring 函数 f̃<br/>生成子网络实例 N_i + 输入 y_i]
    B --> C[子网络<br/>fibred 计算 ⟨N_i,F_i,f̃_i⟩y_i]
    C --> D[输出注回母网络<br/>替换位置集 S_i 的分量]
    D --> E[母网络续算至输出层 N^ℓk↑]
    F[逻辑侧: 每个 v,ℓ 配模态算子 □_v,ℓ] -.相容性 C0-C2.-> A
    F --> G[fibred 公式 ψφ,F = ∧_i □_vi,ℓi ψφ,F_i]
    E -.真值一致 Thm 4.5.-> G
```

### 关键设计

**1. fibred 神经网络的重定义：把"网络生成网络"组织成有向树**。原始 fibring 仅处理两个网络，本文推广到任意数量与组合。fibring architecture 是一棵节点标注架构 $A_v$、边标注 (层号 $\ell$, 位置集 $S$) 的有向树；fibred 网络 $\tilde N=\langle N,F,\tilde f\rangle$ 在每条边上配一个 fibring 函数 $\tilde f_{(v,v')}$，它把维度 $\le d_\ell$ 的向量映成"一个子架构 $A_{v'}$ 的网络实例 + 一个合法输入"。计算自根递归进行：算到带子边的层就暂停，调用 fibring 函数把当前向量的一部分送进子网络，子网络的输出再按位置集 $S_i$ 拼回母网络。形式上每个 stage 产出四元组 $(x_i,N_i,y_i,h_i)$，$h_i$ 是把 $x_i$ 在 $S_i$ 处替换成子 fibred 网络在 $y_i$ 上的结果，最终输出 $\tilde N(x)=N^{\ell_k\uparrow}(h_k)$。若根网络输出标量，则按 $>0$ 与否解释为 true/false 分类器。这一重定义让 fibring 能直接对接现代多层架构的逐层委派结构。

**2. 相容 fibred 模型：用 admissibility 把可达关系钉死在网络行为上**。逻辑侧的关键是 Definition 4.1 的 admissible 映射 $\pi$——它给每个 Kripke 世界配一个网络输入向量，要求 $\pi$ 单射且满足 $w$ 与 $w'$ 在模型 $m$ 中可达 $\iff N(\pi(w))=N(\pi(w'))$。也就是说，**世界间是否相连，完全由网络在对应输入上的输出是否相等决定**，可达关系不再是任意指定，而是网络语义的镜像。在此基础上 Definition 4.2 给出相容性条件 (C0)–(C2)：(C0) 根模型把每个世界映到 $\{0,1\}^n$ 中 bit 即命题真值的向量；(C1) 每个 $\pi_{v,\ell}$ 对子网络 $N_v^{\ell\uparrow}$ admissible；(C2) fibring 跳转 $f_{v,\ell}$、$f_{v_i,1}$ 与网络计算产出的 $(x_i,N_i,y_i,h_i)$ 逐一对齐，且各模型间命题真值集合保持一致。

**3. 相容模型类是合法 fibred 逻辑：非空 + 同构封闭**。要让"相容模型"成为可用的逻辑 L_F，必须证明 $\mathrm{Comp}_F(v,\ell)$（所有相容 fibred 模型在 $(v,\ell)$ 分量上的投影）满足合法 fibred 逻辑的两个条件（Proposition 4.3）。**非空性**通过显式构造：对每个 $\tilde N$ 与每个 $x\in\{0,1\}^n$，按"为每个收集到的向量造一个世界、当 $N_v^{\ell\uparrow}$ 输出相等时连边、命题真值沿根模型递归定义"的办法，直接由网络计算造出一个相容模型。**同构封闭**：若 $m\in\mathrm{Comp}_F(v,\ell)$ 来自某相容 $M$、$\pi:m\cong m'$，则把 $M$ 的该分量换成 $m'$、把 $\pi_{v,\ell}$ 换成 $\pi_{v,\ell}\circ\pi^{-1}$，可达关系、真值、fibring 跳转都沿 $\pi$ 平移而不改变可观测行为，故 $m'$ 仍相容。两条性质合起来说明 L_F 是"非空且在 Kripke 同构下封闭"的合理模型类。

**4. 精确对应定理：网络分类与公式真值逐输入一致**。Definition 4.4 把命题公式 $\varphi$ 沿 fibring 树递归提升为 fibred 公式 $\psi(\varphi,F)=\bigwedge_i \square_{v_i,l_i}\psi(\varphi,F_i)$（$l_i$ 为离开 $v_i$ 的最大层标号）。Theorem 4.5 则证明：对根架构的任意网络实例 $N$，存在命题公式 $\varphi$（取 $N$ 的特征公式 $\varphi:=\bigvee_{h:N(h)>0}(\bigwedge_{h_k=1}p_k\wedge\bigwedge_{h_k=0}\neg p_k)$），使得**对任意输入 $x$、任意匹配 $F$ 的 fibring 函数、任意相容 fibred 模型 $M$**，都有
$$M,(\pi_{u,1})^{-1}(x)\models\psi(\varphi,F)\iff \langle N,F,\tilde f\rangle\text{ 把 }x\text{ 分类为 True}.$$
证明对 $F$ 的深度归纳：叶子情形由 (C0) 保证 $\varphi$ 在根模型上与 $N(h)>0$ 一致；归纳步利用 (C1)–(C2) 与 □ 语义，使评估 $\psi(\varphi,F)$ 恰好等价于沿 fibring 跳转逐子树评估，两侧遵循同一递归结构，故真值与最终根输出 $>0$ 一致。

**5. 应用：GAT/GNN/Transformer 的非均匀刻画**。在 truncated ReLU 激活、局部求和聚合、布尔输入、硬注意力等假设下，论文以 GAT 为主线（GNN 是无注意力的 GAT，Transformer encoder 是作用在完全图、带位置编码的 GAT）。Theorem 5.1 证明：对每个三元组 $\tau=\langle G,\mathcal G,u\rangle$（GAT、图、节点），存在 fibring 架构 $F_\tau$、根网络实例 $N^\tau$ 与一族随节点特征 $x$ 变化的 fibring 函数 $\tilde f_x^\tau$，使 fibred 网络 $\langle N^\tau,F_\tau,\tilde f_x^\tau\rangle$ 在 $x_u$ 上的计算等于 $\mathcal G(G,x,u)$——关键在于 $F_\tau$ 的树结构正是节点 $u$ 的**展开树（unraveling tree）**，与 GAT 沿展开树深度的递归计算同构，**对固定 $(G,u)$ 架构不变、只有 fibring 函数随 $x$ 变**。Theorem 5.2 再调用 4.5 的对应，得到 fibred 逻辑 L_{F_τ} 中一个**不依赖节点特征**的公式 $\tilde\varphi_\tau$，使 $M,(\pi_{u_\tau,1})^{-1}(x_u)\models\tilde\varphi_\tau\iff\mathcal G(G,x,u)=\text{true}$，即一族公式刻画该网络实例——这正是非均匀表达力。Transformer 只需把根的 $\pi_{u,1}$ 换成映到 $\{pos(t,s),1+pos(t,s)\}^n$ 的 bijection 以纳入位置编码。

## 理论结果与核心论断
本文是纯理论工作，无实验表格，核心贡献是以下定理链。

### 主要定理

| 结果 | 内容 | 意义 |
|------|------|------|
| Prop 4.3 | $\mathrm{Comp}_F(v,\ell)$ 非空且在 Kripke 同构下封闭 | L_F 是合法 fibred 逻辑 |
| Thm 4.5 | fibred 网络的 True 分类 ⟺ fibred 公式 $\psi(\varphi,F)$ 在对应世界为真（对所有输入/fibring 函数/相容模型成立） | 神经计算与逻辑语义**精确对应** |
| Thm 5.1 | 任意 GAT/GNN/Transformer 的逐输入计算可由（架构固定、fibring 函数随输入变的）fibred 网络复现 | 架构 → fibred 网络的非均匀描述 |
| Thm 5.2 | 上述网络实例可由 L_{F_τ} 中一族（不依赖节点特征的）公式刻画 | 三类架构的**非均匀逻辑表达力** |

### 关键发现
- **展开树是对应的几何核心**：fibring 架构的树结构 = 目标节点的展开树，使 GAT 的递归聚合与 fibred 网络的递归委派天然对齐。
- **公式只依赖根网络与架构、不依赖输入特征**：因此同一 $(G,u)$ 下不同 $x$ 共享同一公式骨架，只是 fibring 函数（语义跳转）不同——这是"非均匀"的精确含义。
- **统一了此前各自独立的刻画**：GNN、GAT、Transformer encoder 第一次在同一 fibred 逻辑框架下被刻画，而非各用 WL 测试 / Presburger / AC⁰ / LTL 等不同工具。

## 亮点与洞察
- **补上 20 年悬而未决的形式化缺口**：2004 年 fibring 神经网络与 fibring 逻辑的类比终于被证成精确对应，admissibility + 相容性条件 (C0)–(C2) 是把"网络行为"翻译成"Kripke 可达关系"的关键技术。
- **一个框架装下三类架构**：把 Transformer encoder 视作完全图上带位置编码的 GAT、GNN 视作无注意力 GAT，使统一刻画只需一条主线 + 两处退化。
- **指向均匀表达力与可解释性的研究纲领**：作者论证收集 fibred 网络在各分量上的向量集合与 Benedikt 等人 GNN 的 "ℓ-spectrum" 密切相关，其有限性正是均匀表达力（Presburger 公式）证明的关键，从而把"非均匀 → 均匀"的统一变成可操作的路线；同时"逆向工程网络学到的 fibred 逻辑理论"与当前模块化可解释性研究（circuit tracing 等）精神相通。

## 局限与展望
- **只到非均匀、未达均匀表达力**：核心结果给出的是逐输入的公式族，而非刻画整个架构实例的单一公式；如何把公式族坍缩成单一（如 Presburger）公式，对 GAT 与 Transformer encoder 仍是开放问题（Transformer 目前只有下界）。
- **假设较强**：技术结果限定 truncated ReLU、局部求和聚合、有理系数、布尔输入、硬注意力，软注意力 / 一般激活 / 连续特征尚未覆盖。
- **纯理论、无实证**：fibring 镜头在可解释性与形式验证上的"应用潜力"目前还是 speculation，尚无从真实网络抽取可读逻辑规则的实验验证。
- **展望**：用 ℓ-spectrum 的有限性把 fibred 公式族统一为均匀公式、把 fibring 用于跨架构的形式验证（借助模态/fibred 逻辑可满足性复杂度结果）、以及"逆向工程典型输入下的 fibred 公式"以抽取可解释规则。

## 相关工作与启发
- **fibring 逻辑与神经网络**：Gabbay (1999) 的 fibring 逻辑、Garcez & Gabbay (2004) 的 fibring 神经网络是本文的两个源头；本文把后者推广并与前者精确对应。
- **GNN 表达力**：Barceló 等 (2020) 的 WL 上界、Grohe (2023) 的布尔电路/描述复杂度非均匀结果、Nunn/Benedikt (2024) 的 Presburger 均匀表达力、Cuenca Grau (2025) 的有界 GNN 对应一阶逻辑片段——本文与之互补并试图统一。
- **Transformer 表达力**：Hao (2022) 的 UHAT→AC⁰、Barcelo (2024) 的 AHAT→LTL(C,+) 下界；本文把 Transformer encoder 纳入同一 fibred 框架。
- **可解释性**：与 circuit tracing（Ameisen 2025）等把网络计算拆成模块化推理步骤的工作精神相通，启发"逆向工程网络学到的逻辑理论"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次建立 fibring 神经网络与 fibring 模态逻辑的精确对应，并以单一框架统一 GNN/GAT/Transformer 的逻辑刻画，填补 20 年理论缺口。
- 实验充分度: ⭐⭐⭐ 纯理论工作，无实证实验；定理链完整自洽，但 fibring 在可解释性/验证上的应用价值仍停留在论证层面。
- 写作质量: ⭐⭐⭐⭐ 形式化严谨、动机与对应关系叙述清晰，附图与归纳证明到位；但相容性条件 (C0)–(C2) 与 admissibility 的记号较重，对非逻辑背景读者门槛高。
- 价值: ⭐⭐⭐⭐ 为神经符号 AI 提供统一的逻辑表达力镜头，并指出通往均匀表达力与可解释性的明确研究纲领，对理论社区有较强奠基意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Proper Velocity Neural Networks](proper_velocity_neural_networks.md)
- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Random Spiking Neural Networks are Stable and Spectrally Simple](random_spiking_neural_networks_are_stable_and_spectrally_simple.md)
- [\[ICLR 2026\] Feature Compression is the Root Cause of Adversarial Fragility in Neural Networks](feature_compression_is_the_root_cause_of_adversarial_fragility_in_neural_network.md)
- [\[ICLR 2026\] Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks](tractability_via_low_dimensionality_the_parameterized_complexity_of_training_qua.md)

</div>

<!-- RELATED:END -->
