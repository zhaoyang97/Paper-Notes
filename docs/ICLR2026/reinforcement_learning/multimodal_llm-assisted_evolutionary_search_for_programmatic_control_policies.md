---
title: >-
  [论文解读] Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies
description: >-
  [ICLR 2026][强化学习][programmatic policy] MLES 把多模态大模型当作"会看回放的策略程序员"，配合演化搜索直接生成可读的程序化控制策略——用执行画面（行为证据）诊断失败模式并定向修改代码，在 Lunar Lander 和 Car Racing 上做到了和 PPO 相当的性能，同时全程透明可追溯。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "programmatic policy"
  - "多模态"
  - "evolutionary search"
  - "interpretable RL"
  - "behavioral evidence"
---

# Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=OHFNJoNtjW](https://openreview.net/forum?id=OHFNJoNtjW)  
**代码**: [https://github.com/QingL2000/MLES](https://github.com/QingL2000/MLES)  
**领域**: 强化学习 / 可解释控制策略 / LLM 辅助演化搜索  
**关键词**: programmatic policy, multimodal LLM, evolutionary search, interpretable RL, behavioral evidence  

## 一句话总结
MLES 把多模态大模型当作"会看回放的策略程序员"，配合演化搜索直接生成可读的程序化控制策略——用执行画面（行为证据）诊断失败模式并定向修改代码，在 Lunar Lander 和 Car Racing 上做到了和 PPO 相当的性能，同时全程透明可追溯。

## 研究背景与动机
**领域现状**：深度强化学习（DRL）在控制任务上表现强劲，但策略以神经网络的形式存在，是难以理解、验证和调试的黑箱。在自动驾驶、医疗等安全攸关场景，这种不透明性直接阻碍了落地。与此同时，"LLM 辅助演化搜索"（LES）范式兴起，把 LLM 当作变异/交叉算子、配合演化计算迭代优化候选设计，已在代码、算法、启发式发现上取得成功；在 RL 领域，Eureka 等工作证明 LES 能用来塑造奖励函数。

**现有痛点**：(1) 黑箱策略无法被人类理解和验证，知识也难以复用迁移；(2) 现有 LES 在 RL 里只敢做奖励函数塑造这种"辅助组件"设计，没人直接用它合成端到端的控制策略；(3) 传统 LES 只靠标量分数（episode reward）反馈，LLM 只知道"策略好不好"，不知道"为什么失败"，于是退化成盲目的试错。

**核心矛盾**：高性能与透明性难以兼得——DRL 给性能但不透明，符号方法（遗传编程+DSL）透明但依赖手工设计的领域专用语言、扩展性差。

**本文目标**：提出一个能**直接合成**程序化控制策略的框架，性能对标强 DRL baseline，同时策略本身和发现过程都对人类透明可追溯。

**核心 idea**：**【关键洞察】** 让多模态 LLM 不只看分数，而是像人类专家一样"看回放"——把策略执行的视觉轨迹（行为证据 Behavioral Evidence）喂给 MLLM，让它诊断失败模式（如急转弯失控、奖励黑客），再针对性地改写策略代码。这把 LLM 驱动的自动发现从"随机试错"变成"有依据的诊断式精修"。

## 方法详解

### 整体框架
MLES（Multimodal LLM-assisted Evolutionary Search）把控制策略发现建模为：在 MLLM 隐式定义的开放式语义策略空间中，由演化搜索循环驱动的搜索过程。每个候选策略是一个"个体"，由四元组刻画——**代码**（可执行 Python 程序，定义决策逻辑）、**思想**（自然语言描述设计意图）、**量化指标**（环境执行得到的奖励等，作为选择与收敛的 ground-truth 目标）、**行为证据 BE**（执行画面/视频，作为生成时的诊断信号）。框架闭环运转：从策略池选父代 → 取其行为证据 → 构造多模态少样本提示 → MLLM 生成子代策略 → 评估并生成新的行为证据 → 更新策略池。论文用 EoH（Evolution of Heuristics）作为演化骨架做了一个具体实例（MLES-EoH），刻意最小化对搜索算子的结构改动，以干净地隔离"视觉反馈行为分析"这一核心贡献的增益。

```mermaid
flowchart LR
    A[任务描述+初始策略] --> B[策略池]
    B -->|指数秩选择| C[选取父代个体]
    C --> D[取父代行为证据 BE]
    D --> E[Prompt Sampler<br/>构造多模态少样本提示]
    E --> F[MLLM 集成<br/>生成子代策略代码]
    F --> G[Evaluator<br/>环境执行→量化指标]
    G --> H[Behavior Summarizer<br/>执行轨迹→行为证据]
    H --> B
    G -.量化指标排序保留 top-N.-> B
```

### 关键设计

**1. 行为证据（Behavioral Evidence, BE）：从"多好"到"为什么"的诊断信号。** 这是 MLES 区别于普通 LES 的命门所在。普通 LES 只把标量指标 $F(\pi)$（episode reward）反馈给 LLM，LLM 只知道策略排第几，却看不到它在哪里栽了跟头。MLES 引入 Behavior Summarizer 模块，把 Evaluator 收集的原始执行轨迹转成"富含原理"的行为证据——可以是图像、视频、甚至文本状态序列（论文用了 frame stacking 这种帧堆叠方式把一段执行压成一张多帧图）。通过分析 BE，MLLM 能直接观察到诸如"急转弯时因速度过快冲出赛道""利用 bug 刷分的奖励黑客"等失败模式。论文把这设计成明确的分工：量化指标负责选择与收敛，BE 负责指导生成。这模仿了人类专家精修策略的方式——不只看分数，更要看行为模式来定位问题根源。

**2. 多模态修改算子（M1_M / M2_M）：把诊断落到代码改写上。** MLES 在 EoH 标准探索算子之外，新增了两个真正用上 BE 的修改算子。M1_M 让 MLLM 联合分析策略代码和它的行为证据，识别行为缺陷后改写控制逻辑（比如发现"转向过激导致来回摆动"，就降低转向系数、调整转弯阈值）；M2_M 让 MLLM 找出策略里的关键参数，根据观测到的证据做定向调参。配合保留的两个探索算子——E1（检视父代代码与思想，合成策略逻辑根本不同的新策略，探索未知区域）和 E2（提取多个父代的共性模式做泛化，类似交叉操作）——形成"创意引擎（E）+ 精修引擎（M）"的协同。消融显示去掉 M1_M/M2_M 在 Car Racing 上掉 12%~14%，是降幅最大的，证明基于执行证据的定向修改才是效率的核心来源。

**3. 多模态少样本提示构造 + 指数秩选择：把 MLLM 从"码农"升级成"推理智能体"。** 每一步 Prompt Sampler 根据当前算子动态拼装四部分上下文：(1) 静态任务描述、(2) 父代信息（代码+思想）、(3) 相关行为证据、(4) 算子专属指令。这个复合提示让 MLLM 不只是写代码，而是观察—诊断—改进的推理智能体。父代选择用演化计算里标准的指数秩选择，选择概率 $p_i \propto e^{-c \cdot r_i}$（$r_i$ 是按性能排的秩）——因为 MLES 每次"变异"都要花一次昂贵的 MLLM 推理（不像传统演化算法变异很便宜），所以必须维持高选择压力来保证样本效率，同时给低秩个体保留非零概率以维持多样性。策略池管理则对子代去冗余、按量化指标保留 top-N（N=16）个体进入下一代。

## 实验关键数据

### 主实验表格
两个 Gym 控制任务，每个方法跑 5 次独立实验，报告均值/SEM/最佳。Lunar Lander 分数 >1.00 表示全部成功着陆（越高越省油），Car Racing 满分 100 表示完美跑完所有赛道。

| 方法 | LunarLander 训练均值 | LunarLander 测试均值 | CarRacing 训练均值 | CarRacing 测试均值 |
|------|------|------|------|------|
| DQN | 1.017 | 0.508 | 79.78 | 71.72 |
| PPO | 1.032 | 0.846 | **99.21** | 94.55 |
| Initial policy | 0.629 | 0.653 | 17.77 | 17.62 |
| EoH | 1.053 | 0.776 | 89.81 | 79.29 |
| **MLES** | **1.090** (SEM±0.005) | 0.819 | 98.07 | **96.36** |

MLES 在 Lunar Lander 训练上最高、Car Racing 上次优（紧追 PPO），且测试集上 Car Racing 反超 PPO（96.36 vs 94.55）。相比同框架但无 BE 的 EoH，MLES 在两任务上都显著领先且 SEM 更小（更稳定），在更难的 Car Racing 上差距尤为明显。

### 消融实验表格
三次独立运行，$\Delta$Perf 为相对完整 MLES 的性能降幅。

| 变体 | LunarLander | $\Delta$Perf | CarRacing | $\Delta$Perf |
|------|------|------|------|------|
| MLES（完整） | 1.090 | 0.00% | 98.70 | 0.00% |
| w/o E1 | 1.093 | +0.17% | 87.53 | -11.32% |
| w/o E2 | 1.082 | -0.68% | 89.04 | -9.79% |
| w/o M1_M | 0.997 | -8.54% | 86.71 | -12.15% |
| w/o M2_M | 1.020 | -6.41% | 84.77 | -14.12% |

多模态修改算子（M1_M/M2_M）移除后降幅最大，证明基于执行证据的定向修改是核心；探索算子（E1/E2）在简单的 Lunar Lander 上影响极小甚至略有提升（激进探索反而引入无益方差），但在复杂的 Car Racing 上变得至关重要。

### 关键发现
- **搜索效率**：MLES 在 Lunar Lander 上约 5000 次环境重置即收敛，远快于 PPO/DQN；Car Racing 上收敛速度与 PPO 相当但方差显著更低。
- **冷启动能力**：完全从零开始（无初始策略）+ 仅 2000 次 LLM 查询，MLES 仍能自主发现高性能策略；换用更强的 GPT-5-mini 后无种子也能解决任务。
- **基座模型红利**：GPT-5-mini 在 Car Racing 上仅用约 17% 的样本预算就达到 GPT-4o-mini 的满分，说明 MLES 会随基座模型进步而自然增益。
- **可读性验证**：20 名 CS 背景研究生评审生成的策略代码，普遍反映能读懂、且代码内嵌的详细注释很有帮助。

## 亮点与洞察
- **把"看回放"工程化**：行为证据（BE）这一设计精准击中普通 LES "只有分数没有诊断"的痛点，让自动发现从盲目试错变成诊断式精修，是真正有画面感的创新点。
- **量化指标与 BE 的明确分工**：指标管选择/收敛、证据管生成/修改，职责清晰，避免了把所有反馈混在一起的混乱。
- **直接合成 vs 辅助塑造**：跳出 LES 只敢做奖励函数塑造的舒适区，第一次端到端直接合成程序化策略，对标 DRL。
- **透明性是双层的**：不仅产物（策略代码）人类可读，发现过程（演化谱系图，每一步为何生成、如何精修）也完全可追溯，图 5 的 Car Racing 演化谱系把"contour 检测→ROI 聚焦→自适应刹车"的逐步改进展示得很清楚。

## 局限与展望
- **泛化仍是挑战**：所有方法（含 MLES）测试性能相对训练都有一致下降，论文只在附录用策略集成做了缓解尝试。
- **任务规模有限**：仅在 Lunar Lander 和 Car Racing 两个经典 Gym 任务上验证，更高维、更长程或真实机器人控制任务尚未涉及。
- **依赖 MLLM 推理成本**：每次变异都需一次 MLLM 推理，虽然指数秩选择提升了样本效率，但相对传统演化算法的"廉价变异"，单步成本高、LLM 查询预算（2000 次）是硬约束。
- **行为证据的设计依赖任务**：BE 的形式（帧堆叠、ROI 等）目前仍需针对任务构造 Behavior Summarizer，自动化程度有提升空间。
- **展望**：框架对底层搜索算法无关（本文只实例化为 EoH），可换更强骨架；随基座模型变强会自然受益，有望成为发展透明可验证控制策略的新范式。

## 相关工作与启发
- **可读策略的两条路线**：post-hoc 方法把黑箱策略蒸馏成决策树/数学公式/DSL 程序（如 VIPER、Bastani 等），但本质是事后解释代理模型；direct 方法（遗传编程+DSL）直接搜索可读策略，但依赖手工 DSL。MLES 属于 direct 路线，但用 MLLM 的开放语义空间替代了僵硬的手工 DSL，绕开了 DSL 设计的扩展性瓶颈。
- **LLM 辅助演化搜索**：FunSearch、EoH、ReEvo 等把 LLM 当演化算子做代码/启发式发现；Eureka 等把 LES 用于 RL 奖励函数塑造。MLES 的差异在于把 LES 从"设计辅助组件"推进到"直接合成策略"，并补上了多模态行为反馈这一环。
- **启发**：行为证据这一思路对任何"LLM 生成 + 执行反馈"的闭环（代码生成、agent 自我改进、机器人技能合成）都有借鉴价值——别只给标量奖励，把执行轨迹/可视化喂回去，让模型能诊断而非猜测。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把多模态行为证据引入 LES 闭环、并第一次直接端到端合成程序化控制策略，切入点新颖且抓住了普通 LES 的真痛点。
- **实验充分度**: ⭐⭐⭐ — baseline（DQN/PPO/EoH）、消融、冷启动、基座模型、人工可读性评审都覆盖了，但仅两个经典 Gym 任务、规模偏小，泛化下降问题未充分解决。
- **写作质量**: ⭐⭐⭐⭐ — 框架图、演化谱系图、分工论述清晰，"创意引擎+精修引擎"的比喻到位，可读性好。
- **价值**: ⭐⭐⭐⭐ — 为透明可验证控制策略提供了有说服力的新范式，且会随基座模型进步而增益，对安全攸关 RL 落地有现实意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [\[ICLR 2026\] GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies](goldenstart_q-guided_priors_and_entropy_control_for_distilling_flow_policies.md)
- [\[ICLR 2026\] Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM](refining_hybrid_genetic_search_for_cvrp_via_reinforcement_learning-finetuned_llm.md)
- [\[ICLR 2026\] MARS-Sep: Multimodal-Aligned Reinforced Sound Separation](mars-sep_multimodal-aligned_reinforced_sound_separation.md)
- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)

</div>

<!-- RELATED:END -->
