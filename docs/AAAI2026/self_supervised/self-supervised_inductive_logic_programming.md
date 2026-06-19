---
title: >-
  [论文解读] Self-Supervised Inductive Logic Programming
description: >-
  [AAAI 2026][自监督学习][inductive logic programming] 提出自监督归纳逻辑编程（SS-ILP）新设定及 Poker 系统，仅从少量正标签样本和无标签样本出发，自动生成正负样本，配合最大化通用的二阶确定性范式（SONF）背景理论，在无负样本情况下学习含递归和谓词发明的逻辑程序。
tags:
  - "AAAI 2026"
  - "自监督学习"
  - "inductive logic programming"
  - "meta-interpretive learning"
  - "grammar learning"
  - "predicate invention"
---

# Self-Supervised Inductive Logic Programming

**会议**: AAAI 2026  
**arXiv**: [2507.16405](https://arxiv.org/abs/2507.16405)  
**代码**: [https://github.com/stassa/aaai_26_experiments/](https://github.com/stassa/aaai_26_experiments/)  
**领域**: 人工智能 / 逻辑编程  
**关键词**: inductive logic programming, self-supervised learning, meta-interpretive learning, grammar learning, predicate invention

## 一句话总结

提出自监督归纳逻辑编程（SS-ILP）新设定及 Poker 系统，仅从少量正标签样本和无标签样本出发，自动生成正负样本，配合最大化通用的二阶确定性范式（SONF）背景理论，在无负样本情况下学习含递归和谓词发明的逻辑程序。

## 研究背景与动机

归纳逻辑编程（ILP）是从正负样本和背景理论中学习逻辑程序的范式。元解释学习（MIL）能从少量样本学习出含递归和发明谓词的程序，泛化性好。然而，传统 ILP 面临两个核心负担：

1. **背景理论需要针对每个学习任务手工定制**：需要领域专家根据目标程序精心设计，限制了实际应用
2. **负样本需要手动选择**：缺乏负样本会导致过度泛化

核心矛盾：ILP 的学习能力依赖于精心设计的背景理论和负样本，但这恰恰是制约其广泛应用的瓶颈。

本文的核心 idea 是**矛盾检测**：如果两个原子 $e_1, e_2$ 都被同一假设 $H$ 接受（$H \models \{e_1, e_2\}$），那么假设 $e_1$ 是正样本而 $e_2$ 是负样本就构成矛盾。通过迭代检测这种矛盾，Poker 可以自动将无标签样本分类为正或负样本，避免过度泛化。

## 方法详解

### 整体框架

Poker 的工作流程分三步：(1) **Generalise**——从正标签样本 $E^+$ 构建初始假设集合 $T$；(2) **Generate**——用 $T$ 作为生成器生成新样本，加入无标签样本 $E^?$ 并全部假设为负样本；(3) **Label**——逐一检验假设为负的样本，通过矛盾检测重新标记为正或确认为负，迭代特化 $T$ 直到一致。

### 关键设计

**1. 自监督矛盾检测算法**

算法核心逻辑（Algorithm 1）：

- **Generalise**：构建满足约束的初始假设集 $T = \{H_{|H| \leq l} : \mathcal{M} \cup B \cup V \models H \wedge B \cup H \models e^+ \in E^+\}$
- **Generate**：合并所有假设 $T' = \bigcup T$，用 $T'$ 生成 $k$ 个新样本，全部加入假设负样本集 $E^- = E^? \cup \{e_i : B \cup T' \models e_i\}$
- **Label**：对每个假设为负的 $e \in E^-$，移除接受它的假设得到 $T' = T \setminus \{H : B \cup H \models e\}$。如果 $B \cup T' \models E^+$（即移除后不影响正样本），确认 $e$ 为负样本（$T \leftarrow T'$）；否则存在矛盾——$e$ 实际是正样本，移回 $E^+$

**定理 1**：Poker 返回正确假设的概率随无标签样本数量单调递增。

**2. 二阶确定性范式（SONF）**

SONF 是一组带约束的元规则，足够通用以学习目标谓词 $\Theta$ 所有可能解释 $I$ 下的所有正确程序。具体地，本文提出两种 SONF：

**Chomsky-Greibach 范式（C-GNF）**——用于上下文无关语言 DCG 学习：
- Identity 元规则：$P(x,y) \leftarrow Q(x,y)$，约束 $\text{target}(P) \wedge (\text{background}(Q) \vee \text{empty}(Q))$
- Chain 元规则：$P(x,y) \leftarrow Q(x,z), R(z,y)$，约束 $P \neq Q \wedge (\text{target}(P) \vee \text{invented}(P)) \wedge \neg\text{target}(Q)$
- Tri-Chain 元规则：$P(x,y) \leftarrow Q(x,z), R(z,u), S(u,y)$（用于回文等）

**Lindenmayer 范式（LNF）**——用于 L-System 文法学习。

SONF 的核心优势：不再需要为每个学习任务定制元规则，一个 SONF 适用于整个问题类别。例如 C-GNF 可学习任意二元字母表上下文无关文法。

**3. 约束元规则系统**

扩展了元规则定义，加入丰富的约束：
- $\text{target}(P)$：$P$ 必须实例化为目标谓词符号
- $\text{invented}(P)$：$P$ 必须实例化为发明谓词符号
- $\text{background}(P)$：$P$ 必须实例化为背景理论符号
- $\text{invented}(P,Q) \rightarrow P \neq Q$：防止发明谓词间的冗余递归
- 复合约束：$C_M^1 \wedge C_M^2$、$C_M^1 \vee C_M^2$、$\neg C_M$

这些约束主要控制递归、消除冗余，提升搜索效率。

### 损失函数 / 训练策略

ILP 不使用梯度优化。Poker 基于 SLD-Refutation 进行假设构建和特化，使用 SWI-Prolog 的表格执行（SLG-Resolution）控制递归并提升性能。学习过程是符号推理，不涉及数值优化。

## 实验关键数据

### 主实验

**实验 1：L-System 文法学习**（Dragon Curve、Hilbert Curve、Koch Curve、Sierpinski Triangle）

| 系统 | k（自动生成样本数） | 生成准确率趋势 | 假设大小趋势 |
|------|---------------------|---------------|-------------|
| Poker | 0 | 较低 | 较大（过度泛化） |
| Poker | 增大 | 单调提升 | 单调减小 |
| Louise | 0（始终） | 随正样本增加而下降 | 随正样本增加而增大 |

Poker 的生成准确率随自动生成样本数增加而提升，假设复杂度同时下降；Louise 在无负样本时持续过度泛化。

**实验 2：二元上下文无关语言学习**（6 种 CFL：偶校验、$1^n0^n$、$n_a=n_b$、$a^nb^m$、回文、匹配括号）

| 设置 | TPR | TNR | 说明 |
|------|-----|-----|------|
| Poker, k=0 | ~1.0 | ~0.0 | 无自动生成样本，完全过度泛化 |
| Poker, k 逐步增大 | ~1.0 | 逐步→1.0 | TPR 和 TNR 同时最大化 |
| Louise（所有 k） | ~1.0 | ~0.0 | 持续过度泛化 |

在所有 6 种 CFL 上，Poker 随 $k$ 增大 TPR 和 TNR 均提升至最高。实验在 100 组随机采样上重复，报告标准误差。

### 消融实验

**自动生成样本数 $k$ 的影响**（以 $1^n0^n$ 语言为例）：

| $k$ | TPR | TNR | 行为 |
|-----|-----|-----|------|
| 0 | ~1.0 | ~0.0 | 完全过度泛化 |
| 低（如 10） | ~1.0 | 部分提升 | 开始特化 |
| 高（如 100+） | ~1.0 | ~1.0 | 正确学习 |

验证了定理 1 的预测：准确率随 $k$ 单调递增。

### 关键发现

- 自动生成负样本是避免过度泛化的关键，且准确率随样本数单调改善
- SONF 成功替代了手工定制的元规则集，一个 C-GNF 适用于所有二元 CFL
- Louise（SOTA MIL 系统）在无负样本情况下持续过度泛化，验证了本文方法的必要性
- 背景理论越通用，负样本越重要——这是本文的核心 trade-off

## 亮点与洞察

- **范式创新**：定义了 SS-ILP 新设定，首次在 ILP 中实现自监督学习（自动生成并标记正负样本）
- **理论保证**：证明了假设正确性概率随无标签样本量单调递增
- **SONF 概念**：用通用二阶范式替代任务特定元规则，是 MIL 实践中的重要简化
- **矛盾检测机制简洁优雅**：利用移除-检验的方式区分正负样本，无需预设任务知识

## 局限与展望

- SONF 的推导过程仍需手工完成（从 Chomsky/Greibach 范式推导），尚未自动化
- 实验限于文法学习（CFL 和 L-System），未展示在更广泛 ILP 任务上的效果
- 可扩展性存疑：假设集 $T$ 可能随问题规模指数增长
- 与 Popper 等现代 ILP 系统的对比仅为初步实验，Popper 作者确认其不支持无负样本学习
- 无标签样本的通用性（generality）对性能有影响，但本文仅实验了自动生成样本

## 相关工作与启发

- **vs Louise (Vanilla-Louise)**: Louise 是 SOTA MIL 系统，使用 Top Program Construction，但无法处理无负样本场景——持续过度泛化
- **vs Popper**: Popper 不需要二阶背景理论但仍需 mode declarations 和负样本，且仅适用于特定递归数据结构
- **vs DeepLog**: DeepLog 从单样本学习但受限于 dyadic logic（arity 2），Poker 的 SONF 不受 arity 限制
- **vs MetaAbd**: 神经符号混合系统，仍需手工定义问题特定的 abducible predicates

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 定义了全新的 SS-ILP 设定，SONF 概念有理论突破性
- 实验充分度: ⭐⭐⭐ 实验覆盖两类文法学习任务，但缺少更广泛的 ILP 基准
- 写作质量: ⭐⭐⭐⭐ 形式化定义严谨，示例直观，文法学习场景选择恰当
- 价值: ⭐⭐⭐⭐ 有望推动 ILP 在无负样本场景下的实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](../../ICML2026/self_supervised/understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[ICML 2026\] Can Local Learning Match Self-Supervised Backpropagation?](../../ICML2026/self_supervised/can_local_learning_match_self-supervised_backpropagation.md)
- [\[CVPR 2026\] Progressive Mask Distillation for Self-supervised Video Representation](../../CVPR2026/self_supervised/progressive_mask_distillation_for_self-supervised_video_representation.md)

</div>

<!-- RELATED:END -->
