---
title: >-
  [论文解读] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm
description: >-
  [ACL 2025][LLM 其他][自动提示优化] 系统综述 80+ 种基于启发式搜索算法的自动 Prompt 优化方法，提出五维分类体系（Where/What/What criteria/Which operators/Which algorithms）将碎片化研究统一到一个完整的分析框架下。 领域现状：手动 Promp…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "自动提示优化"
  - "启发式搜索"
  - "指令优化"
  - "进化算法"
  - "分类体系"
---

# A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm

**会议**: ACL 2025  
**arXiv**: [2502.18746](https://arxiv.org/abs/2502.18746)  
**代码**: [GitHub](https://github.com/jxzhangjhu/Awesome-LLM-Prompt-Optimization)  
**领域**: LLM/NLP  
**关键词**: 自动提示优化, 启发式搜索, 指令优化, 进化算法, 分类体系

## 一句话总结
系统综述 80+ 种基于启发式搜索算法的自动 Prompt 优化方法，提出五维分类体系（Where/What/What criteria/Which operators/Which algorithms）将碎片化研究统一到一个完整的分析框架下。

## 研究背景与动机
**领域现状**：手动 Prompt 工程（如 CoT、"step by step"、"take a deep breath"）完全依赖人类直觉和反复试错，无法系统性地发现最优 prompt。自动 Prompt 优化（APO）将 prompt 设计视为搜索/优化问题，通过算法迭代精炼 prompt 以最大化下游任务性能。

**现有痛点**：APO 领域自 2023 年 APE 和 OPRO 以来爆发增长，但研究极度碎片化——有的用进化算法（EvoPrompt）、有的用贝叶斯优化（InstructZero）、有的用 LLM 自身作为优化器（OPRO）、有的用梯度信号（GCG）——缺乏统一分类框架来理解这些方法的异同。

**核心矛盾**：如何将方法论差异巨大的各种 APO 方法纳入一个连贯的分类体系？现有综述要么范围过广（覆盖所有 prompt 技术）、要么过窄（仅关注某一类方法），无法提供清晰的全景图。

**本文目标**：提供一个全面而有组织的综述，用正交的五维分类将碎片化研究统一起来，使研究者能快速定位任何一篇新工作在坐标系中的位置。

**切入角度**：聚焦两个范围限制——(a) 指令式（instruction-focused）而非示例选择/排序；(b) 启发式搜索而非 RL 或模型集成方法，通过精确的范围界定避免泛泛而谈。

**核心 idea**：五维分类体系（Where × What × What criteria × Which operators × Which algorithms）是理解自动 Prompt 优化的完整正交坐标系，任何 APO 方法都可以在这五个维度上精确定位。

## 方法详解

### 整体框架
本文不是提出新方法，而是构建分析框架。核心贡献是五维分类体系（taxonomy），将 APO 方法分解为五个正交维度，每个维度内部再细分子类。整体逻辑：首先确定在哪个空间优化（Where），然后确定优化什么（What），用什么标准评估（What criteria），如何生成候选 prompt（Which operators），以及用什么搜索策略导航解空间（Which algorithms）。

### 关键设计

1. **Where — 优化空间**：

    - **软提示空间（Soft Prompt）**：在连续嵌入空间中优化，可用梯度信号。子类包括：梯度→嵌入（Prefix Tuning, P-Tuning 直接优化连续向量）、梯度→目标 token（GCG 用梯度定位最优替换位置然后在离散 token 中搜索）、梯度→词表映射（AutoPrompt 用梯度搜索词表中最优 token）、零阶优化（ZOPO 用 NTK 近似的高斯过程估计梯度，不需要反向传播）
    - **离散提示空间（Discrete Prompt）**：直接优化自然语言文本字符串，天然不可微分。需要黑盒优化方法。如 ProTeGi 用 LLM 反馈生成"伪梯度"（文本形式的改进建议），EvoPrompt 用遗传算法对 prompt 进行变异和交叉
    - **软→离散投影**：部分方法在软空间优化后投影回离散空间（如 GCG），但投影过程存在信息损失，这是一个开放问题

2. **What — 优化目标**：

    - **仅指令优化**（最常见范式）：直接精炼指令文本，如 OPRO 迭代改写系统 prompt
    - **指令 + 示例联合优化**：三种子范式——
        - 示例→指令（MoP：先聚类示例再为每个簇生成专门指令）
        - 指令→示例（MIPRO：先优化指令再生成匹配的 few-shot 示例）
        - 并行优化（EASE：用 bandit 算法同时搜索最佳指令-示例组合）
    - **指令 + 可选示例**（PhaseEvo：根据任务特性动态决定是否添加示例）
    - **关键洞察**：指令和示例的联合优化显著优于单独优化，但搜索空间会指数级膨胀

3. **What criteria — 优化标准**：

    - **任务性能**：准确率、F1 等传统指标
    - **鲁棒性**：对 prompt 扰动、对抗输入的稳定性
    - **效率**：API 调用次数、收敛速度、计算成本
    - **可解释性**：优化后 prompt 是否可理解（软 prompt 通常不可解释）
    - **安全性**：优化不应导致有害输出（GCG 的对抗 prompt 优化技术可被反向用于红队测试）
    - **多目标优化**：同时优化性能 + 安全 + 成本，越来越受关注

4. **Which operators — 候选生成算子**：

    - **零父本算子**：从零生成候选，不依赖已有 prompt（Lamarckian 式从任务描述直接生成、模型驱动的随机初始化）
    - **单父本算子**：基于一个已有 prompt 生成变体——语义改写（局部/全局）、LLM 反馈驱动（让 LLM 分析错误案例并提供改写建议）、人类反馈驱动、梯度反馈驱动、增删替换操作
    - **多父本算子**：组合多个已有 prompt——交叉（取两个 prompt 的互补部分）、差异（学习两个 prompt 间的差异并应用到第三个上）、EDA 式估计分布采样

5. **Which algorithms — 搜索算法**：

    - **Bandit 算法**：将 prompt 视为臂，用 UCB/Thompson Sampling 平衡探索-利用
    - **束搜索（Beam Search）**：维护 top-k 候选集，每步扩展最优候选（ProTeGi）
    - **蒙特卡洛树搜索（MCTS）**：将 prompt 构建为树节点，用 UCT 导航（PromptAgent）
    - **进化算法族**：GA/差分进化/CMA-ES/模拟退火，将 prompt 视为"个体"进行种群进化
    - **迭代精炼**：最简单但有效，LLM 反复改写自己的 prompt（OPRO 范式）

## 实验关键数据

### 代表方法对比

| 方法 | 优化空间 | 优化目标 | 搜索算法 | 代表性贡献 |
|------|---------|---------|---------|-----------|
| APE (Zhou et al. 2023) | 离散 | 仅指令 | 迭代精炼 | 首个 LLM 生成+筛选指令的方法 |
| OPRO (Yang et al. 2024) | 离散 | 仅指令 | 迭代精炼 | LLM 自身作为优化器的范式奠基 |
| ProTeGi (Pryzant et al.) | 离散 | 仅指令 | 束搜索 | "伪梯度"概念——文本形式的改进信号 |
| EvoPrompt (Guo et al.) | 离散 | 仅指令 | GA/DE | 首次将进化算法引入 prompt 优化 |
| PromptBreeder (Fernando) | 离散 | 仅指令 | 进化算法 | 自进化——变异策略本身也进化 |
| MIPRO (Opsahl-Ong et al.) | 离散 | 指令+示例 | 贝叶斯 | 指令→示例的联合优化范式 |
| InstructZero (Chen et al.) | 软→离散 | 仅指令 | 贝叶斯优化 | 软空间优化 + 离散投影 |
| GCG (Zou et al. 2023) | 软→离散 | 仅指令 | 贪心坐标梯度 | 梯度定位 + 离散搜索，用于对抗 prompt |
| PromptAgent (Wang et al.) | 离散 | 仅指令 | MCTS | 蒙特卡洛树搜索导航 prompt 空间 |
| PhaseEvo (Cui et al.) | 离散 | 指令+可选示例 | 进化算法 | 动态决定是否加 few-shot |

### 工具框架对比

| 工具 | 核心功能 | 特色 |
|------|---------|------|
| DSPy | 声明式 prompt 编程 | 将 prompt 视为可编译/优化的程序模块 |
| TextGrad | 文本梯度优化 | 用 LLM 反馈模拟梯度下降 |
| PromptFoo | A/B 测试框架 | 系统化评估 prompt 变体 |
| PromptBench | 鲁棒性测试 | 评估 prompt 对扰动的稳定性 |

### 关键发现
1. **离散空间方法主导**：大多数最新方法选择直接在自然语言空间优化，因为 (a) 对闭源 API 友好 (b) 结果可解释 (c) 不需要模型权重
2. **OPRO 范式简单有效**：让 LLM 自身作为优化器的迭代精炼方法，虽然算法最简单，但在很多场景下效果不输复杂搜索算法
3. **进化算法在高维空间有优势**：当 prompt 空间复杂（长指令、多约束）时，进化算法的种群多样性有助于避免局部最优
4. **联合优化 > 单独优化**：指令 + 示例联合优化平均比单独优化指令提升 5-15%，但计算成本也显著增加

## 亮点与洞察
- **五维分类体系清晰全面**：将碎片化的研究统一到一个框架下，任何一篇新论文都可以在五个维度上精确定位——这对快速扫描新工作非常实用
- **LLM 自身作为优化器（OPRO 范式）** 打破了传统优化-评估分离的思路——优化器和被优化对象是同一个模型，这种自指性（self-referential）架构可能是最有前景的方向
- **优化标准应超越精度**：鲁棒性和安全性同等重要。GCG 最初用于对抗攻击（发现对齐绕过 prompt），但其技术可反向用于红队测试和防御——这揭示了 APO 的双刃剑本质
- **软→离散投影是关键瓶颈**：软空间优化可用梯度但结果不可解释、不可迁移；离散空间可解释但不可微分。如何高效桥接两个空间是未解决的核心问题

## 局限与展望
- **无统一实验对比**：作为综述最大的遗憾——仅分类但未在统一基准上比较各种方法，读者仍无法回答"哪种方法最好"这个关键问题
- **多模态 prompt 优化未覆盖**：主要关注文本 prompt，视觉 prompt（如 SAM 的 point/box prompt）、音频 prompt 的优化方法未涉及
- **动态 N-shot 问题被低估**：大多数方法假设固定数量的 few-shot 示例，但最优示例数量应随任务和输入动态变化
- **成本分析缺失**：不同 APO 方法的 API 调用次数和成本差异巨大（从几十次到几万次），但综述未系统比较
- **快速发展的领域**：2024 年底到 2025 年初的最新方法（如 AgentOptimizer、RL-based 方法）可能未覆盖

## 相关工作与启发
- **vs 其他 Prompt 综述**：Sahoo et al. (2024) 覆盖更广（所有 prompt 技术）但深度不足；本文范围更精确（启发式搜索 + 指令）所以每类方法分析更深入
- **搜索算法选择启发**：Prompt 优化本质是黑盒优化/搜索问题——搜索算法的选择至关重要。简单任务用迭代精炼即可，复杂任务（长指令、多约束）考虑进化算法或 MCTS
- **与 Agent 工具优化的联系**：APO 的框架可以直接迁移到 Agent 的工具选择和参数优化——搜索空间从"prompt 文本"变为"工具调用序列"
- **DSPy 视角**：将 prompt 视为可编译的程序模块（而非固定字符串）可能是比逐字优化更有前途的范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 综述贡献在分类体系，方法为回顾性；五维分类有原创性但单个维度的划分并非全新
- 实验充分度: ⭐⭐⭐ 无实验，纯综述——缺少统一基准上的定量对比是最大遗憾
- 写作质量: ⭐⭐⭐⭐⭐ 分类清晰、层次分明，Fig.1 的树状分类图直观实用，覆盖 80+ 方法但不觉得杂乱
- 价值: ⭐⭐⭐⭐⭐ 对 Prompt 工程从业者有实用参考价值，五维坐标系是快速定位新工作的有效工具，GitHub awesome list 本身也有社区价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)
- [\[ACL 2025\] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](erm_prompt_optimization_memory.md)

</div>

<!-- RELATED:END -->
