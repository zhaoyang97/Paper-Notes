---
title: >-
  [论文解读] Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts
description: >-
  [ICML2025][机器人][Neuro-Symbolic] 提出 SPG（Sketch-Plan-Generalize）——一种神经符号智能体框架，将归纳式概念学习分解为三阶段流水线：概念签名推断（Sketch）、基于 MCTS 的 grounded 动作序列搜索（Plan）、以及 LLM 驱动的程序归纳泛化（Generalize），在从少量演示中学习可组合、可泛化的空间抽象概念方面显著优于纯 LLM 和纯神经方法。
tags:
  - "ICML2025"
  - "机器人"
  - "Neuro-Symbolic"
  - "Program Synthesis"
  - "MCTS"
  - "Inductive Generalization"
  - "Concept Learning"
  - "Embodied AI"
---

# Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts

**会议**: ICML2025  
**arXiv**: [2404.07774](https://arxiv.org/abs/2404.07774)  
**领域**: 神经符号学习 / 具身概念学习  
**关键词**: Neuro-Symbolic, Program Synthesis, MCTS, Inductive Generalization, Concept Learning, Embodied AI

## 一句话总结

提出 SPG（Sketch-Plan-Generalize）——一种神经符号智能体框架，将归纳式概念学习分解为三阶段流水线：概念签名推断（Sketch）、基于 MCTS 的 grounded 动作序列搜索（Plan）、以及 LLM 驱动的程序归纳泛化（Generalize），在从少量演示中学习可组合、可泛化的空间抽象概念方面显著优于纯 LLM 和纯神经方法。

## 研究背景与动机

有效的人机协作要求机器人能从极少量的人类演示中快速学习个性化的概念表征。理想的概念表征需要满足四项关键性质：

**基于演示的 grounding**：直接从人类演示中学习，反映人类意图，而非仅依赖先验世界知识

**归纳泛化**：从几个特定高度的塔推断出构建任意高度塔的通用程序

**层次化组合**：复杂概念可以表示为简单概念的组合（如楼梯 = 递增高度的塔序列）

**约束可修改性**：学到的概念表征能灵活适应新指令中的附加约束（如颜色限制）

**现有方法的不足**：

- **纯 LLM 方法**（Singh et al., 2022; Liang et al., 2023）：依赖先验世界知识生成计划，无法从演示中学习语言上未知的新概念，归纳泛化能力差
- **纯神经方法**（Liu et al., 2023）：可从演示学习但缺乏符号归纳建模，泛化能力局限于训练分布，且缺少模块化复用能力
- **神经符号方法**（Grand et al., 2023; Ellis et al., 2021）：在程序空间中搜索可泛化程序，但枚举式搜索在复杂程序空间（如 Python 程序）中计算不可行，且难以用人类演示有效引导搜索

**核心洞察**：将归纳式概念学习分解为三个互补阶段——粗粒度签名推断、基于奖励的 grounded 规划搜索、以及从多个具体计划中提炼通用程序——可以绕过大规模程序空间的搜索难题。

## 方法详解

### 整体框架

SPG 将概念学习任务分解为三个阶段的流水线：

$$\text{Demonstrations} \xrightarrow{\text{Sketch}} \text{Signature} \xrightarrow{\text{Plan}} \text{Grounded Plans} \xrightarrow{\text{Generalize}} \text{Inductive Program}$$

整体框架的设计哲学是"先具体再抽象"：先通过搜索获取多个具体实例的 grounded 计划，再利用 LLM 的代码生成能力将其提炼为归纳式通用程序。

### 阶段一：Sketch（概念签名推断）

给定带有语言标注的新概念演示，利用 LLM 推断该概念的**粗粒度函数签名**（function signature）：

- **输入**：语言描述 + 视觉演示
- **输出**：函数名、参数列表、参数类型
- **作用**：界定搜索空间的边界，将无限的程序空间约束为有限的签名族

例如，对于"tower"概念，Sketch 阶段可能输出签名 `def build_tower(height: int, color: str) -> ActionSequence`，明确了需要参数化的维度。

该阶段的关键在于，LLM 不需要写出完整实现——仅需"草拟"概念的接口规约，大幅降低了对 LLM 代码生成准确性的要求。

### 阶段二：Plan（MCTS 引导的 grounded 规划）

以函数签名为框架，使用蒙特卡洛树搜索（MCTS）在 grounded 动作序列空间中搜索：

- **搜索空间**：给定可用的原子动作（如放置方块、移动、旋转），组合构成动作序列
- **奖励信号**：衡量构建结果与人类演示目标结构的匹配程度
- **搜索策略**：MCTS 利用 UCB（Upper Confidence Bound）平衡探索与利用

$$\text{UCB}(s, a) = \bar{Q}(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

其中 $\bar{Q}(s, a)$ 为动作 $a$ 在状态 $s$ 下的平均回报，$N(s)$ 为状态访问次数，$c$ 为探索系数。

对每个演示实例，Plan 阶段输出一个具体的 grounded 动作序列。多个不同参数配置的演示产生多个 grounded 计划，为后续泛化提供"归纳基础"。

**关键设计**：奖励函数基于神经表征计算结构相似度，结合视觉编码器将 3D 结构编码为向量，度量与目标演示的距离。这使得系统能够在没有精确符号匹配的情况下，通过连续空间的相似性信号引导离散动作搜索。

### 阶段三：Generalize（程序归纳泛化）

利用 LLM 的代码生成能力，将来自多个演示的 grounded 计划**蒸馏**为归纳式通用程序：

- **输入**：多个 grounded 计划 + 函数签名 + 演示参数
- **处理**：LLM 对比不同参数下的具体计划，识别不变模式和参数化维度
- **输出**：一个可参数化的归纳程序（如 `for i in range(height): place_block(x, y+i)`）

该程序具备：
- **归纳泛化**：可处理训练中未见的参数值（如从高度 3、5 的塔泛化到高度 100）
- **模块复用**：已学概念可作为子程序被更复杂概念调用
- **持续学习**：新概念不断加入概念库，支持层次化组合

### 训练策略与持续概念学习

SPG 采用**无训练的组合式策略**，不依赖端到端梯度优化：

1. **概念检测**：智能体在执行指令时，检测是否遇到未知概念（概念库中无对应程序）
2. **主动请求演示**：识别未知概念后，主动请求人类提供 2-3 个不同参数配置的演示
3. **三阶段学习**：依次执行 Sketch → Plan → Generalize
4. **概念库更新**：将新学习的程序存入概念库，后续可被直接调用或组合

这种设计使得系统具备**持续概念学习**能力——概念库不断扩充，且新概念可以建立在已有概念之上。例如，先学习"tower"和"row"，再通过组合学习"staircase"（递增高度的 tower 序列）和"pyramid"（递减长度的 row 堆叠）。

### 指令跟随与约束推理

学到的程序表征支持在新指令中灵活施加约束：

- **离散约束**：如"构建一个楼梯，且不能有绿色方块紧挨蓝色方块"
- **实现方式**：LLM 将约束编译为程序中的条件判断，插入到已学程序的适当位置
- **优势**：程序化表征天然支持条件分支和约束注入，而纯神经表征难以精确实施硬约束

## 实验关键数据

### 概念学习泛化准确率

| 方法 | 简单概念 (Tower, Row) | 复杂概念 (Staircase, Pyramid) | 组合概念 (Boundary, L-shape) | 平均泛化率 |
|------|----------------------|------------------------------|------------------------------|-----------|
| LLM-only (GPT-4) | 较高 | 低 | 极低 | ~40% |
| 纯神经 (Struct-Net) | 中等 | 低 | 低 | ~35% |
| DreamCoder (枚举搜索) | 中等 | 中等 | 低 | ~45% |
| **SPG (本文)** | **高** | **高** | **中高** | **~75%** |

SPG 在归纳泛化性能上大幅领先所有基线方法，尤其在需要层次化组合的复杂概念上优势最为显著。

### 各阶段消融实验

| 配置 | Sketch | Plan | Generalize | 泛化准确率 | 变化 |
|------|--------|------|------------|-----------|------|
| 完整 SPG | ✓ | ✓ | ✓ | ~75% | — |
| 无 Sketch（直接搜索） | ✗ | ✓ | ✓ | ~55% | -20% |
| 无 MCTS（LLM 直接生成计划） | ✓ | ✗ | ✓ | ~50% | -25% |
| 无 Generalize（使用 grounded 计划） | ✓ | ✓ | ✗ | ~30% | -45% |
| LLM 端到端代码生成 | ✗ | ✗ | ✗ | ~40% | -35% |

消融实验清晰表明：Generalize 阶段贡献最大（体现归纳泛化的核心价值），Plan 阶段次之（MCTS 搜索提供高质量的 grounded 基础），Sketch 阶段也不可或缺（约束搜索空间，提升搜索效率）。

### 指令跟随与约束满足

| 任务类型 | SPG | LLM-only | 纯神经 |
|----------|-----|----------|--------|
| 无约束构建 | 高 | 中 | 中 |
| 颜色约束 | 高 | 中低 | 低 |
| 空间关系约束 | 高 | 低 | 极低 |
| 组合约束 | 中高 | 极低 | 极低 |

程序化表征在约束满足任务上展现出显著优势，因为约束可以被直接编译为程序逻辑。

## 亮点与洞察

1. **"先具体再抽象"的巧妙分解**：避免直接在庞大的程序空间中搜索，而是先用 MCTS 在较小的动作空间中获取具体计划，再由 LLM 归纳抽象——这种分治策略显著降低了问题复杂度
2. **神经+符号互补的典范**：神经组件提供连续的匹配奖励信号引导搜索，符号组件提供可组合、可泛化的程序结构——两者缺一不可
3. **持续概念学习**：概念库设计支持增量式知识积累和层次化复用，贴近人类学习新概念的认知过程
4. **LLM 作为"归纳引擎"**：不要求 LLM 一步到位生成完整程序，而是利用其模式识别能力从多个具体示例中抽取共性——降低了对 LLM 代码生成准确性的依赖
5. **约束可组合性**：程序表征天然支持约束注入，解决了纯神经方法难以处理硬约束的痛点

## 局限性

1. **演示需求**：每个新概念需要 2-3 个不同参数配置的演示，在实际人机交互场景中可能增加用户负担
2. **MCTS 搜索开销**：对于具有长动作序列的复杂结构，MCTS 搜索时间可能显著增长，限制实时应用
3. **概念空间局限**：当前主要验证在 3D 方块构建任务上，向更通用的空间概念（如柔性物体、连续形状）迁移尚待验证
4. **LLM 归纳的鲁棒性**：Generalize 阶段依赖 LLM 的代码生成能力，对于高度非线性或不规则的概念模式，LLM 可能难以准确归纳
5. **错误传播风险**：三阶段流水线中，Sketch 阶段的签名错误可能会级联影响后续所有阶段
6. **未考虑物理约束**：当前框架主要处理几何/空间关系，未纳入物理稳定性等现实世界约束

## 相关工作

- **LLM 作为规划器**（SayCan, Code-as-Policies）：利用 LLM 先验知识生成动作计划，但缺乏从演示中学习新概念的能力
- **DreamCoder**（Ellis et al., 2021）：通过枚举搜索+压缩在 DSL 中合成程序，但在复杂程序空间中计算不可行
- **LILO**（Grand et al., 2023）：结合 LLM 引导的程序搜索和库学习，是 SPG 最接近的对比方法
- **Struct-Net**（Liu et al., 2023）：纯神经的结构预测方法，从演示学习但泛化受限
- **程序归纳合成**（Gulwani et al., 2017）：从输入-输出示例中合成程序的经典范式，SPG 将其扩展到空间概念域

SPG 的独特贡献在于将 LLM 代码生成、MCTS 搜索和神经视觉表征有机结合，创建了一个既能从演示学习又能归纳泛化的统一框架。

## 评分

- 新颖性: ⭐⭐⭐⭐ （三阶段分解范式新颖，巧妙结合 LLM 和 MCTS）
- 实验充分度: ⭐⭐⭐⭐ （多种概念类型的泛化实验 + 详细消融 + 指令跟随评估）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，问题定义明确，图示直观）
- 价值: ⭐⭐⭐⭐ （为少样本概念学习提供了可行的神经符号方案，持续学习设计具有实际意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](../../NeurIPS2025/robotics/towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[ICML 2025\] Efficient Robotic Policy Learning via Latent Space Backward Planning](efficient_robotic_policy_learning_via_latent_space_backward_planning.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](../../NeurIPS2025/robotics/learning_spatial-aware_manipulation_ordering.md)
- [\[AAAI 2026\] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation](../../AAAI2026/robotics/spatialactor_exploring_disentangled_spatial_representations_for_robust_robotic_m.md)

</div>

<!-- RELATED:END -->
