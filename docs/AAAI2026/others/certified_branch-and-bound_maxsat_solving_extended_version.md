---
title: >-
  [论文解读] Certified Branch-and-Bound MaxSAT Solving (Extended Version)
description: >-
  [AAAI 2026][MaxSAT] 为 Branch-and-Bound MaxSAT 求解器实现了基于 VeriPB 证明系统的认证，覆盖了 look-ahead 边界方法和多值决策图（MDD）编码两大核心技术，在 MaxCDCL 求解器上的实验表明证明日志的中位开销仅 19%，填补了 MaxSAT 求解范式认证的最后空白。
tags:
  - "AAAI 2026"
  - "MaxSAT"
  - "证明日志"
  - "分支定界"
  - "VeriPB"
  - "多值决策图"
---

# Certified Branch-and-Bound MaxSAT Solving (Extended Version)

**会议**: AAAI 2026  
**arXiv**: [2511.10273](https://arxiv.org/abs/2511.10273)  
**代码**: [Zenodo](https://zenodo.org/) (Vandesande et al., 2025)  
**领域**: 其他  
**关键词**: MaxSAT, 证明日志, 分支定界, VeriPB, 多值决策图

## 一句话总结

为 Branch-and-Bound MaxSAT 求解器实现了基于 VeriPB 证明系统的认证，覆盖了 look-ahead 边界方法和多值决策图（MDD）编码两大核心技术，在 MaxCDCL 求解器上的实验表明证明日志的中位开销仅 19%，填补了 MaxSAT 求解范式认证的最后空白。

## 研究背景与动机

### 问题定义

组合优化求解器在安全关键领域（航天器控制、器官移植匹配、交通基础设施验证等）广泛应用。但求解器可能因源代码 bug 产生**不可行解或错误声称最优性**。需要一种系统性方法保证求解器输出的正确性。

### 为何需要证明日志（Proof Logging）

**形式化验证**求解器本身代价过高，且影响性能

**认证算法**（Certifying Algorithms）：求解器不仅输出答案，还输出正确性证明，由独立的证明检查器验证
3. 在 SAT 领域，证明日志已成为竞赛的强制要求

### MaxSAT 的认证现状

MaxSAT（SAT 的优化变体）的主要求解范式：

**解改进搜索**（Solution-Improving）：已有认证方法 ✅

**核引导搜索**（Core-Guided）：已有认证方法 ✅

**隐式打击集**（Implicit Hitting Set）：同期论文处理 ✅

**分支定界**（Branch-and-Bound）：**本文填补的空白** ❌→✅

### MaxCDCL 的挑战

MaxCDCL（当前 SOTA 的分支定界求解器，赢得 2023/2024 MaxSAT 评测冠军）结合了 CDCL 搜索和复杂的边界函数。认证的主要难点在于：

**Look-ahead 方法**产生条件不可满足核（local cores），需要证明这些核的正确性

**MDD 编码**将伪布尔约束编码为 CNF，由于 BDD/MDD 的规约性，一个变量可能表示多个等价约束——需要在证明中展示它们确实等价

## 方法详解

### 整体框架

在 MaxCDCL 求解器中，使用 VeriPB 证明系统（基于 0-1 整数线性不等式的证明系统）记录每步推理。VeriPB 维护一个**证明配置** $\mathcal{F}$，使用 cutting planes 证明系统进行推导：文字公理、线性组合、除法、饱和，以及反向单位传播（RUP）。

### 关键设计

#### 1. **认证 Look-Ahead 边界方法**

**加权局部核（Weighted Local Core）**：三元组 $q = \langle w, R, K \rangle$，其中 $R \subseteq \alpha$（当前赋值的子集），$K$ 包含目标文字的否定，满足 $F \wedge R \wedge K \models \bot$。

**$\mathcal{O}$-相容核集**：核集 $\mathcal{Q}$ 相容当每个目标文字被核使用的总权重不超过其在目标函数中的权重。这确保了以下两种推导：

- **软冲突**（Proposition 3.1）：若 $w_\mathcal{O}(\mathcal{Q}) \geq v^*$（核权重总和 ≥ 当前最优），当前节点不可能改进，可回溯
- **硬化**（Proposition 3.2）：若某目标文字的剩余权重 + 核总权重 ≥ $v^*$，可将该文字传播为非代价发生

**证明方法**：

- **Proposition 4**：每个核 $q$ 对应的子句 $C_q = \bigvee_{\ell \in R \cup K} \bar{\ell}$ 可通过 RUP 证明（因为核的发现本身就来自传播到矛盾）
- **Theorem 5**：从核子句和解改进约束出发，通过至多 $3|\mathcal{O}| + 2|\mathcal{Q}| + 1$ 步 cutting planes 推导出 $C_\mathcal{Q}$（原因子句）

推导过程示例：对目标 $\mathcal{O} = 3x_1 + 5x_2 + 5x_3 + 6x_4$，核 $q_1 = \langle 3, \{y_1\}, \{\bar{x_1}, \bar{x_2}\}\rangle$，$q_2 = \langle 5, \{y_2\}, \{\bar{x_3}, \bar{x_4}\}\rangle$：
1. RUP 推导核子句
2. 按权重乘核子句并相加
3. 从解改进约束推导目标文字的上界
4. 两者相加、化简、除法得到最终子句

#### 2. **认证 BDD/MDD 编码**

BDD 用于将解改进约束 $\mathcal{O} \leq v^* - 1$ 编码为 CNF。核心挑战：BDD 的规约性意味着一个节点 $\eta = \text{bdd}(k, l, u)$ 同时表示 $\sum_{i \geq k} v_i b_i \leq l$（严格）和 $\sum_{i \geq k} v_i b_i \leq u$（宽松）两个等价约束。

**定义约束**：对每个节点 $\eta$，需要证明：
$$v_\eta \Rightarrow \sum_{i \geq k} v_i b_i \leq l \quad \text{且} \quad v_\eta \Leftarrow \sum_{i \geq k} v_i b_i \leq u$$

**Proposition 10**（关键引理）：给定子节点的定义约束，可通过以下步骤推导父节点的定义约束：
1. 引入两个具化变量 $v_\eta$ 和 $v_\eta'$
2. 对变量 $b_k$ 进行分情况讨论
3. 使用矛盾证明法（利用子节点定义约束）
4. 短 cutting planes 推导统一两个变量

**Proposition 11**：当 BDD 跳过变量（reducedness 导致的多约束表示）时，通过 $6(k-k')+3$ 步推导统一定义约束。

**MDD 推广**：MDD 在 at-most-one 约束下对变量集合分支（$|I|+1$ 个子节点而非 2 个）。Proposition 10 的证明只需将 2 个分情况扩展为 $|I|+1$ 个。

#### 3. **VeriPB 证明系统的关键规则**

- **Cutting Planes**：文字公理、线性组合、除法、饱和
- **RUP**（Reverse Unit Propagation）：若 $\mathcal{F} \wedge \neg C$ 传播到矛盾，则可添加 $C$
- **Redundance-based Strengthening**：用于具化（reification）和矛盾证明
- **Objective Improvement**：找到新解时添加解改进约束

### 损失函数 / 训练策略

不涉及机器学习训练。算法策略为标准 MaxCDCL 的 CDCL + look-ahead 循环。

## 实验关键数据

### 主实验

| 指标 | 数值 | 说明 |
|------|------|------|
| 基准实例数 | 701（不含证明日志可解） | MaxSAT Evaluation 2024 |
| 有证明日志无法解的实例 | 6 | 极少数实例因开销超时 |
| 证明日志开销中位数 | **19%** | 对多数实例开销可控 |
| 90% 分位开销 | 4.61× | 10% 实例开销较大 |
| 证明检查成功率 | 485/695 (69.8%) | 202 个超时，8 个超内存 |
| 证明检查时间中位数 | 42.94× 求解时间 | 检查时间是瓶颈 |

### 消融实验

| 配置 | 影响 | 说明 |
|------|------|------|
| MDD 定义约束的节点数 | 线性于目标文字数 | 大目标文字数实例开销显著 |
| BDD vs MDD | MDD 提供更紧凑编码 | 但认证代价也更高 |
| RUP vs explicit derivation | RUP 更简洁但检查更慢 | 带提示的 RUP 可能改善检查 |

### 关键发现

1. **证明日志开销对大多数实例是可接受的**（中位 19%），但对 10% 的实例（通常是目标文字数多的）开销可达 4.6 倍以上
2. **证明检查是当前瓶颈**：中位需要求解时间的 43 倍。限制了实际部署
3. MDD 节点的定义约束推导（线性于目标文字数）是高开销实例的主要原因
4. 本文的工作使 MaxSAT 的**所有主要求解范式**（解改进、核引导、隐式打击集、分支定界）都获得了 VeriPB 认证

## 亮点与洞察

1. **填补最后空白**：分支定界是最后一个未被认证的 MaxSAT 求解范式，本文的工作完成了 MaxSAT 认证的全景图
2. **MDD 等价性证明的巧妙处理**：通过构建定义约束对并沿 BDD 线性传递，将 NP-hard 的等价性检查问题在特定结构下高效解决
3. **软件工程价值**：证明日志不仅保证正确性，还是强大的测试和调试方法论
4. **Theorem 5 的推导步数界**：$O(|\mathcal{O}| + |\mathcal{Q}|)$，证明了认证的多项式开销

## 局限与展望

1. **证明检查太慢**（43× 求解时间）：需要更快的证明检查器（如 PBOxide）或证明剪裁器
2. 对大目标文字数实例的证明日志开销过大——可探索更紧凑的定义约束表示
3. 未实现对 literal unlocking 等最新技术的认证
4. 实验限于 MaxSAT Evaluation 2024 实例，未测试更大规模工业实例
5. 内存使用未详细报告

## 相关工作与启发

- **VeriPB 证明系统**（Bogaerts et al., 2023）提供了统一的伪布尔约束证明框架
- **Vandesande et al. (2022)** 建立了 MaxSAT 的 VeriPB 认证方法论基础
- **Berg et al. (2023, 2024)** 分别认证了核引导搜索和解改进搜索中的 wlog 推理
- **Ihalainen et al. (2026)**（同期论文）认证了隐式打击集搜索
- WMaxCDCL (Coll et al., 2025) 是当前 SOTA 的分支定界求解器

## 评分

- 新颖性: ⭐⭐⭐⭐ （MDD 编码的认证是全新贡献，look-ahead 认证也是首次）
- 实验充分度: ⭐⭐⭐⭐ （在完整 MaxSAT 评测数据集上验证，量化了开销）
- 写作质量: ⭐⭐⭐⭐⭐ （形式化严谨，算法伪代码清晰，示例helpful）
- 价值: ⭐⭐⭐⭐⭐ （完成了 MaxSAT 认证的最后一块拼图，对安全关键应用意义重大）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Certified but Fooled! Breaking Certified Defences with Ghost Certificates](certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)
- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [\[AAAI 2026\] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)
- [\[ICLR 2026\] FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives](../../ICLR2026/others/fastlsq_solving_pdes_in_one_shot_via_fourier_features_with_exact_analytical_deri.md)
- [\[CVPR 2026\] The Missing GAP: From Solving Square Jigsaw Puzzles to Handling Real World Archaeological Fragments](../../CVPR2026/others/the_missing_gap_from_solving_square_jigsaw_puzzles_to_handling_real_world_archae.md)

</div>

<!-- RELATED:END -->
