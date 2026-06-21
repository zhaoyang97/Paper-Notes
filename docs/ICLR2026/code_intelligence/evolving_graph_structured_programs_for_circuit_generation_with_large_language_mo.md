---
title: >-
  [论文解读] Evolving Graph Structured Programs for Circuit Generation with Large Language Models
description: >-
  [ICLR 2026][代码智能][逻辑综合] CircuitEvo 把电路图编码成「图结构程序」这一 LLM 友好的文本形式，再用 LLM + 进化式提示策略迭代生成更紧凑的电路，并配一个有理论保证的「结构感知功能补全」模块兜底正确性，是首个能在保证 100% 功能正确的同时持续压缩电路规模的 LLM 逻辑综合方法。
tags:
  - "ICLR 2026"
  - "代码智能"
  - "逻辑综合"
  - "电路生成"
  - "LLM 算法设计"
  - "进化式程序生成"
  - "EDA"
---

# Evolving Graph Structured Programs for Circuit Generation with Large Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DUtS9K9HH6](https://openreview.net/forum?id=DUtS9K9HH6)  
**代码**: [MIRALab-USTC/CircuitEvo](https://github.com/MIRALab-USTC/CircuitEvo)  
**领域**: 代码智能 / LLM 程序合成  
**关键词**: 逻辑综合, 电路生成, LLM 算法设计, 进化式程序生成, EDA  

## 一句话总结
CircuitEvo 把电路图编码成「图结构程序」这一 LLM 友好的文本形式，再用 LLM + 进化式提示策略迭代生成更紧凑的电路，并配一个有理论保证的「结构感知功能补全」模块兜底正确性，是首个能在保证 100% 功能正确的同时持续压缩电路规模的 LLM 逻辑综合方法。

## 研究背景与动机
**领域现状**：逻辑综合（Logic Synthesis, LS）是芯片设计 EDA 流程的第一步，目标是把真值表（行为级描述）转成门级电路，在功能完全正确的前提下最小化电路规模（节点数），因为初始电路质量直接决定后续优化和最终芯片的功耗/性能/面积（PPA）。这是一个 NP-hard 组合优化问题，工业与学术工具普遍依赖人工启发式（SOP、BDD），近年也涌现出大量学习方法。

**现有痛点**：学习类方法可分两路——一路把电路表示成序列/树状布尔表达式做符号回归（Boolformer、DSR、SPL），另一路把电路表示成与非图（AIG）用深度学习生成（DNAS）。但这些非文本表示对 LLM 不友好，且它们普遍**在"结构紧凑"和"功能正确"之间难以两全**：要么牺牲精度换小尺寸，要么为了正确性堆冗余结构。LLM-based 的 ICSR 精度甚至只有 83.5%，远达不到电路必须 100% 正确的硬约束。

**核心矛盾**：LLM 有强大的生成与算法设计能力，但它对图功能（布尔逻辑）的精确理解能力很差，随机生成的电路无法保证逐位正确；而电路任务对正确性零容忍——一位错就是废品。如何"既借 LLM 探索压缩空间，又不被它的逻辑幻觉拖垮"是关键。

**本文目标**：在保证生成电路 100% 满足真值表的前提下，持续迭代地压缩电路规模。

**核心 idea**：**【表示】把电路图重写成图结构程序**让 LLM 能读能写；**【生成】用进化式提示驱动 LLM** 探索紧凑解；**【兜底】用一个有定理保证的功能补全模块**把 LLM 生成的"近似对"电路强行修正成"完全对"，从而把 LLM 的创造力和电路的硬正确性约束解耦开。

## 方法详解

### 整体框架
CircuitEvo 维护一个全部功能正确的电路程序种群，每轮迭代做两件事：先用 LLM 进化生成器从种群采样父代、按四种进化策略生成一批候选程序；再用结构感知功能优化器把这些候选补全成功能正确的紧凑电路，按规模（fitness）择优回填种群。如此循环 $I$ 轮后输出最紧凑的程序。

```mermaid
flowchart LR
    A[真值表 T] --> B[初始化种群<br/>Shannon 分解]
    B --> C[进化程序生成器<br/>LLM + 4 种进化提示]
    C --> D[局部程序搜索<br/>取关键子结构 Pk]
    D --> E[功能补全<br/>Remark1 定理 + ABC]
    E --> F[Fitness=电路规模<br/>择优回填种群]
    F -->|迭代 I 轮| C
    F --> G[输出最紧凑电路]
```

### 关键设计
**1. 图结构电路程序表示：把电路翻译成 LLM 能读写的文本** 现有工作用表达式树或 AIG 建模电路，这些非文本结构 LLM 难以理解。CircuitEvo 设计的程序由三段组成：IO 定义（声明输入输出变量）、结构描述（自底向上逐行写出每个节点及其输入依赖，如 `node1 = X1 * X2`，显式编码节点连通性）、功能定义（用主输出节点的布尔函数表达要满足的真值表）。这种写法把电路的拓扑层级信息编码进文本，相比序列/树表示更利于 LLM 对图的"阅读理解"和生成，是整个框架的地基。

**2. 进化式程序生成：用领域提示策略引导 LLM 探索** 单纯 zero-shot 让 LLM 生成电路质量很差，因此把 LLM 嵌进进化框架。初始化阶段借鉴 Shannon 分解定理，按某个变量 $X_i$ 把真值表拆成 $T(\cdots)=X_i\cdot T_1 + X_i' \cdot T_2$ 两个子表，用传统启发式各自合成子程序再拼回一个正确程序，换不同分解变量就得到多样且正确的初始种群。生成阶段的提示由三部分构成：问题规格（程序格式 + 紧凑目标）、few-shot 示例（按 $prob(P_i)=\frac{1}{\text{rank}(P_i)+1+N}$ 概率从种群采样，平衡质量与多样性）、以及**四种进化策略**——探索类 E1/E2 对父代对做交叉式重组拓宽结构搜索空间，精化类 R1/R2 改单个程序的布尔算子和变量配置。此外还注入领域约束：观察到多输出电路的输出常有简单逻辑关系 $Y_m = f_m(Y_1,\dots,Y_{m-1})$（如 $Y_2 = \neg Y_1$），把这种输出间依赖预先写进提示能指数级缩小搜索空间。每轮四种策略各跑 $N$ 次产出 $4N$ 个候选。

**3. 结构感知功能补全：用定理把"近似对"修成"完全对"** 这是保证 100% 正确的核心。基于 Remark 1（电路功能补全定理）：对真值表的每个目标布尔函数 $F_i$ 和 LLM 生成电路的功能 $F_g^i$，必存在辅助函数 $F_{ga}^i, F_{gb}^i$ 使得 $F_i = (F_g^i + F_{ga}^i) * F_{gb}^i$，其中 $+,*$ 是逻辑或/与。于是把辅助函数转回真值表 $T_a, T_b$，用传统启发式（ABC）合成出 $P_a, P_b$ 两个子结构，再通过结构哈希消除公共子图、按公式做功能融合，拼回原程序即补全为完全正确。

**4. 局部程序搜索：先剪枝再补全，避免冗余** 直接补全可能引入冗余逻辑损害紧凑性。观察到"初始程序精度越高、规模越小，补全后越紧凑"，于是对含 $n_p$ 个节点的程序生成 $n_p$ 个候选子程序（第 $i$ 个取前 $i$ 个节点构成局部子图），用贪心选精度最高的子程序 $P_k$ 作为补全起点。这一步在功能优化前先剔除冗余组件，让最终电路更小。

## 实验关键数据
评测在 Arithmetic / Random / LogicNets / Espresso 四个基准的 16 个电路上进行，最大规模达 16 输入 / 69 输出（搜索空间 $2^{69\times2^{16}}$），骨干 LLM 用 Deepseek-V3、Qwen2.5-7B-Instruct、GPT-3.5-turbo，后端 LS 工具用 ABC。

### 主实验表格（精度 & 规模）

| 指标 | Boolformer | SPL | DSR | ICSR | DNAS | CircuitEvo |
|---|---|---|---|---|---|---|
| 平均精度 Acc(%)↑ | 91.1 | 91.5 | 89.5 | 83.5 | 99.9 | **100.0** |
| 平均规模 Init Node↓ | 660.9 | 703.6 | 634.8 | 736.8 | 532.9 (DNAS) | **470.1** |

- CircuitEvo 所有电路均达 **100% 精度**（相比 LLM 基线 ICSR 精度提升 16.5%）；规模相比 SOTA 平均**降低 6.74%**，单个电路（如 LogicNets2、Random3）压缩最高达 20%+。
- 技术映射后（mcnc.genlib）的 area 和 delay 在四个基准上全面优于所有基线，说明紧凑性能转化为真实硬件效率。
- 加优化算子（resyn2）后，优化后规模相比 SOTA 进一步平均降低 **11.09%**，证明它提供了更好的初始解。

### 消融实验表格

| 配置 | Arithmetic3 Acc/Node | Espresso4 Acc/Node | LogicNets1 Acc/Node |
|---|---|---|---|
| CircuitEvo (完整) | 100% / **1206** | 100% / **848** | 100% / **139** |
| w/o Evolution | 100% / 1229 | 100% / 940 | 100% / 148 |
| w/o Local | 100% / 1178* | 100% / 911 | 100% / 140 |
| w/o Completion | **67.8%** / 27 | **86.4%** / 9 | **70.1%** / 2 |

### 关键发现
- **去掉功能补全（w/o Completion）精度崩到 60~86%**——直接验证了 LLM 单独无法保证电路正确，补全模块是 100% 正确率的根本来源。
- **去掉进化/局部搜索/程序表示精度仍 100% 但规模变大**——这些模块各自贡献紧凑性，进化生成和局部搜索是压缩的主力。
- 即使用较弱的 GPT-3.5-turbo 也能达到与 Deepseek-V3 相当的结果，说明框架本身（而非超强 LLM）是性能来源。

## 亮点与洞察
- **把 LLM 的"创造力"和电路的"零容错"彻底解耦**：生成器只管探索压缩、补全器用定理保证正确，这个分工是整个方法成立的关键，也回避了"逼 LLM 学会精确布尔逻辑"这条死路。
- **图结构程序表示有迁移价值**：用文本程序编码图的拓扑+功能信息，给"LLM 处理图任务"提供了一个轻量可读的接口范式，不止电路，可能适用于更广的图结构生成。
- **功能补全定理是真正的硬核贡献**：$F_i=(F_g^i+F_{ga}^i)*F_{gb}^i$ 给"如何把近似电路补成精确电路"提供了可计算的构造性证明，把工程 trick 上升为理论保证。
- 是 FunSearch / EoH 这类"LLM + 进化程序生成"范式在 EDA 硬约束场景的成功落地，展示了该范式在"必须 100% 正确"领域的可行路径。

## 局限与展望
- 评测电路规模仍偏小（最大 16 输入），而真实芯片可达数十亿晶体管，能否 scale 到工业级电路存疑；搜索空间随输入数指数增长是固有瓶颈。
- 重度依赖外部 LS 工具 ABC 做初始化、子结构合成与补全，方法本质是"LLM 探索 + 传统启发式兜底"的混合体，并非纯 LLM 端到端。
- 个别电路（LogicNets1/3）规模出现负提升，说明对某些结构进化探索未必优于传统启发式。
- 每轮 $4N$ 次 LLM 调用 + 多轮迭代，生成成本与时间开销较高（论文用 GPT-3.5 而非 GPT-4o 正是为省成本）。

## 相关工作与启发
- **机器学习 for LS**：早期 IWLS 用符号回归学布尔函数（Boolformer、DSR、SPL），近期转向 AIG + 深度学习生成（DNAS）。CircuitEvo 另辟蹊径用"程序表示 + LLM 进化"。
- **LLM for 算法设计**：FunSearch、EoH 等把算法写成程序、用 LLM 迭代优化。本文把"电路 = 功能受约束的图算法"，将该范式迁移到 EDA，并补上了"硬正确性保证"这块短板。
- **启发**：在任何"LLM 输出必须满足严格约束（正确性/安全性/物理可行性）"的任务里，"LLM 自由探索 + 可证明的后处理补全/投影"是一条比"硬训 LLM 学约束"更务实的路线。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个把 LLM 进化框架用于逻辑综合并保证 100% 正确，图结构程序表示 + 功能补全定理组合有原创性。
- 实验充分度: ⭐⭐⭐⭐ 四基准 16 电路、三种 LLM、七个基线、pre/post-mapping 双指标 + 完整消融，证据扎实；规模偏小是主要短板。
- 写作质量: ⭐⭐⭐⭐ 动机—表示—生成—补全脉络清晰，定理与算法伪代码完整，图示直观。
- 价值: ⭐⭐⭐⭐ 在 EDA 这一高价值硬约束场景给出可落地范式，6.74% 规模压缩对芯片 PPA 有实际意义，对"LLM+约束"任务有方法论启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CrossPL: Systematic Evaluation of Large Language Models for Cross Programming Language Interoperating Code Generation](crosspl_systematic_evaluation_of_large_language_models_for_cross_programming_lan.md)
- [\[ICLR 2026\] From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph](from_large_to_small_transferring_cuda_optimization_expertise_via_reasoning_graph.md)
- [\[ACL 2025\] Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](../../ACL2025/code_intelligence/tree_of_evolution_code_gen.md)
- [\[ICLR 2026\] LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models](learnat_learning_nl2sql_with_ast-guided_task_decomposition_for_large_language_mo.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](../../ACL2025/code_intelligence/galla_graph_aligned_large_language_models.md)

</div>

<!-- RELATED:END -->
