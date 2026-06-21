---
title: >-
  [论文解读] MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning
description: >-
  [ICLR 2026][机器人][场景图] MomaGraph 把空间关系、功能关系和部件级交互节点统一进一张任务导向：的场景图，并用强化学习训练一个 7B VLM 先"画图"再"规划"，在自建基准上以 71.6% 准确率超过最强基线 11.4 个点。 领域现状：家用移动操作机器人既要导航又要操作，需要一种紧凑、语义丰富的场…
tags:
  - "ICLR 2026"
  - "机器人"
  - "场景图"
  - "具身任务规划"
  - "移动操作"
  - "空间-功能关系"
  - "强化学习"
  - "Graph-then-Plan"
---

# MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3eTr9dGwJv](https://openreview.net/forum?id=3eTr9dGwJv)  
**项目主页**: [https://HybridRobotics.github.io/MomaGraph/](https://HybridRobotics.github.io/MomaGraph/)  
**代码**: 待确认  
**领域**: 机器人 / 具身智能、场景图、视觉语言模型  
**关键词**: 场景图、具身任务规划、移动操作、空间-功能关系、强化学习、Graph-then-Plan  

## 一句话总结
MomaGraph 把空间关系、功能关系和部件级交互节点统一进一张**任务导向**的场景图，并用强化学习训练一个 7B VLM 先"画图"再"规划"，在自建基准上以 71.6% 准确率超过最强基线 11.4 个点。

## 研究背景与动机
**领域现状**：家用移动操作机器人既要导航又要操作，需要一种紧凑、语义丰富的场景表示来回答"物体在哪、怎么用、哪些部件能动"。场景图（scene graph）天然适合这个任务，已在导航、操作、空间智能等下游任务中展现潜力。

**现有痛点**：作者指出已有场景图有三处硬伤。其一，边只编码**单一类型**关系——要么纯空间（几何布局），要么纯功能（遥控器控制电视、旋钮调参数）；只看空间忽略了可操作性，只看功能又丢了空间约束，导致表示不完整、不可执行。其二，多数方法只建模**静态快照**，无法适应物体位置或状态变化的动态环境。其三，缺乏**任务相关性**，不区分哪些信息对当前任务真正有用，规划效率低下。

**核心矛盾**：认知科学表明人类在新环境里的感知是**动态且任务导向**的——像在 iPad 上看地图，先粗看定位再放大看细节，而现有场景图把所有信息一视同仁、且彼此割裂。如何让一张图同时兼顾"空间布局 + 功能可操作性 + 部件粒度 + 任务对齐 + 状态更新"，是核心矛盾。

**本文目标**：构造一种统一空间-功能关系、含部件级交互节点、紧凑、可随交互动态更新、且高度对齐任务指令的场景表示，并据此完成具身任务规划。

**核心idea**：
- **统一场景图表示**：首次把空间关系与功能关系合并到同一张图，并引入部件级（handle/knob/button）交互节点，产出更细粒度、紧凑、任务相关的结构。
- **Graph-then-Plan 范式**：让单个 VLM 先生成任务导向场景图作为中间结构表示，再基于该图做高层规划，提升推理可靠性与可解释性。
- **RL 训练 + 图对齐奖励**：用强化学习（DAPO）配合专门设计的图对齐奖励，教 VLM 主动"探索-推理"出准确的任务导向图，而非死记模板。

## 方法详解

### 整体框架
给定一组多视角图像 $\{I_i\}_{i=1}^n$ 和自然语言指令 $T$，MomaGraph 要构造一张指令条件下的任务导向场景图 $G_T=(N_T, E^T_s, E^T_f)$：$N_T$ 是与任务 $T$ 相关的物体节点（必要时含部件级交互节点），$E^T_s$ 编码空间关系，$E^T_f$ 编码功能关系，两类边都是从"触发物体"指向"被影响物体"的有向边。整条管线分三步：先用 RL 训练好的 7B VLM（MomaGraph-R1）从多视角观测预测这张图，再以图为中间表示做 Graph-then-Plan 规划，最后在执行过程中根据观测到的状态变化动态更新图。

```mermaid
flowchart LR
    A[多视角图像 + 指令 T] --> B[MomaGraph-R1<br/>7B VLM]
    B --> C[任务导向场景图 G_T<br/>节点+空间边+功能边]
    C --> D[Graph-then-Plan<br/>高层任务规划]
    D --> E[执行动作 a_t]
    E --> F[观测新状态 s_t+1]
    F --> G[状态感知动态更新 U·<br/>剪枝错误假设/确认对应]
    G --> C
```

### 关键设计

**1. 统一空间-功能任务导向场景图：把"在哪"和"怎么用"放进同一张图。** MomaGraph 的核心是节点不再只是粗粒度物体，而是按任务 $T$ 选出最小够用的相关物体，并在需要交互特定部件时纳入 handle/knob/button 等部件级节点（如"开冰箱"既含 fridge 又含其 handle）。边同时承载两套语义：功能关系定义为"一个物体改变另一个物体状态的能力"，划分为 `[OPEN OR CLOSE]`、`[ADJUST]`、`[CONTROL]`、`[ACTIVATE]`、`[POWER BY]`、`[PAIR WITH]` 共 6 类（其中 `[PAIR WITH]` 不改内部状态而改空间构型，专为装配任务设计）；空间关系含方向型（左/右/前/后/高/低）与距离型（近/远/接触）共 9 类。指令本身刻意**不点名**所有相关物体（如"装满浴缸"需模型自己推断出浴缸、水龙头、按钮），迫使模型学会把自然语言落地到正确的物体与关系集合上，而非依赖物体名被显式给出。

**2. 强化学习 + 图对齐奖励：用结果反馈教 VLM "画对图"。** 开源 VLM 直接生成准确任务图能力有限，作者用 DAPO 算法在 MomaGraph-Scenes 上训练 Qwen2.5-VL-7B-Instruct，关键是设计了三段式图对齐奖励 $R(G^{pred}_T, G^{gt}_T)$。动作类型项确保预测对动作 $R_{action}=\mathbb{I}[a^{pred}=a^{gt}]$；边的空间-功能整合项以语义相似度对齐预测边与真值边 $R_{edges}=\frac{1}{|E^T_{gt}|}\sum_{e_j\in E^T_{gt}}\max_{e_i\in E^T_{pred}}S_{edge}(e_i,e_j)$，其中 $S_{edge}$ 同时看空间和功能标签的相似度；节点完整性项用 IoU 衡量任务相关节点集合的重合度 $R_{nodes}=\frac{|N^{pred}_T\cap N^{gt}_T|}{|N^{pred}_T\cup N^{gt}_T|}$。最终奖励再叠加 JSON 格式校验 $R_{format}$ 与长度惩罚 $R_{length}$：$R=w_a\cdot(R_{action}+R_{edges}+R_{nodes})+w_f\cdot R_{format}+w_l\cdot R_{length}$。这套奖励把"图必须同时抓住空间布局、功能关系并紧扣任务需求"这一核心洞察直接编码进训练信号，RL 的探索-迭代特性让模型发现构图策略而非复刻记忆模式。

**3. 状态感知动态更新：让一对多的歧义边随交互收敛成唯一对应。** 真实环境里同类物体常并存且功能对应初始不确定——一个灶台有多个旋钮，但只有一个控制当前任务所需的火头，靠外观无法判定。作者不关心交互策略本身，而专注于把观测到的状态变化吸收进图来消歧。时刻 $t$ 的图 $G^{(t)}_T=(N^{(t)}_T, E^{T,(t)}_s, E^{T,(t)}_f)$ 中功能边可能含一对多的假设映射；当智能体执行动作 $a_t$、观测到新状态 $s_{t+1}$ 后，更新函数 $G^{(t+1)}_T=U(G^{(t)}_T, a_t, s_{t+1})$ 会剔除不一致假设、强化已确认对应。例如转动某个旋钮点燃了火头而其他无效，那条旋钮→火头的 `[control]` 边被确立，其余旋钮的边被剪枝，从而把模糊的一对多假设演化成紧凑、状态感知、唯一可靠的对应关系。

## 实验关键数据

### 主实验表格（MomaGraph-Bench 各 Tier 准确率 %，w/ Graph 设置）

| 类型 | 模型 | T1 | T2 | T3 | T4 | Overall |
|------|------|----|----|----|----|---------|
| 闭源 | Claude-4.5-Sonnet | 83.7 | 70.3 | 72.3 | 69.5 | 73.9 |
| 闭源 | GPT-5 | 79.8 | 68.2 | 75.0 | 63.6 | 71.6 |
| 闭源 | Gemini-2.5-Pro | 79.0 | 69.5 | 72.7 | 65.2 | 71.6 |
| 开源 | DeepSeek-VL2 (4.5B) | 56.9 | 53.6 | 61.3 | 45.4 | 54.3 |
| 开源 | **MomaGraph-R1 (7B)** | **76.4** | **71.9** | **70.1** | **68.1** | **71.6** |

MomaGraph-R1 以 7B 规模在开源模型中达到 SOTA，Overall 71.6%（比最好基线 +11.4），追平甚至接近 GPT-5、Gemini-2.5-Pro 等闭源大模型。全表所有模型在 w/ Graph 设置下都优于 w/o Graph，验证 Graph-then-Plan 的普适收益。

### 消融实验表格（统一 vs 单一关系，Overall %）

| 模型 | 空间-only | 功能-only | 统一(Unified) |
|------|-----------|-----------|----------------|
| MomaGraph-R1 | 59.9 | 64.9 | **71.6** |
| LLaVA-Onevision | 54.0 | 57.0 | **66.0** |

在固定图拓扑、只限制边类型的公平对比下，统一空间-功能表示在两个不同基座上都显著优于任一单一关系变体（MomaGraph-R1 上统一比功能-only +6.7、比空间-only +11.7），证明"统一表示"而非"某一特定架构"是性能来源。

### 关键发现
- **Graph-then-Plan 普遍有效**：即便强如 GPT-5 直接规划也会漏前置步骤（如忘了"先插电再开机""先过滤再煮水"），先生成结构化图再规划能稳定产出与真值逻辑一致的完整动作序列。
- **单一关系图不够用**：只空间或只功能都不足以支撑具身规划，必须统一建模。
- **RL > 模仿**：DAPO + 图对齐奖励让 7B 开源模型获得跨环境、跨任务配置的鲁棒泛化，并迁移到真实机器人实验。
- 数据集 MomaGraph-Scenes 含约 1,050 个任务子图、6,278 张多视角图、覆盖 350+ 家庭场景与 93 条指令；基准 MomaGraph-Bench 含 294 个室内场景、352 个任务图、1,446 张图，按 6 种推理能力 × 4 个难度 Tier 组织，全部来自未见环境以测真泛化。

## 亮点与洞察
- **把割裂的两套图统一**是真正的概念性贡献：空间 + 功能 + 部件级节点三合一，让场景图第一次"既知道东西在哪，又知道东西怎么用，还知道哪个零件能动"，直接对齐了移动操作"导航 + 操作"的双重需求。
- **Graph-then-Plan 的解耦思想**漂亮：把"理解场景"和"生成动作"拆成两步，用显式中间结构提升可靠性与可解释性，且让单个 VLM 同时承担构图与规划，避免了 SayPlan 等方法假设已有可靠 3D 图的不现实前提。
- **状态感知动态更新**抓住了现实痛点——多个同类物体的功能歧义只能靠交互反馈消解，把"试错-剪枝"形式化进图更新函数，思路简洁且贴近机器人闭环。
- **数据 + 基准 + 模型一条龙**：补齐了该方向长期缺失的数据与评测基础设施，6 能力 × 4 Tier 的基准设计对后续工作有参考价值。

## 局限与展望
- **不涉及底层交互策略**：状态更新依赖"动作已执行 + 状态被正确观测"，论文明确不管 manipulation policy 本身，真实部署中观测噪声、动作失败如何反馈进图更新仍是开口。
- **数据规模偏小**：约 1,050 个子图、93 条指令，且部分来自 AI2-THOR 仿真与再标注数据，指令多样性与真实场景覆盖仍有限。
- **基准为多选 VQA 形式**：用 multiple-choice 评估规划与场景理解便于打分，但与开放式真实规划之间仍有 gap，刷分能力未必等价于闭环执行能力。
- **状态更新函数 $U(\cdot)$ 的实现细节**（如何判定一致/不一致、置信度如何累积）在正文较简略，鲁棒性与可扩展性待进一步验证。

## 相关工作与启发
- **场景图方向**：ConceptGraphs 等只建空间布局、开放词表表示几何关系；另一类功能图（如部分 affordance 工作）只抓控制关系。MomaGraph 的贡献正是统一二者并加部件节点 + 状态变化建模。
- **VLM 零样本具身规划**：VLM 直接当规划器对视觉噪声敏感、语义落地浅，且缺结构化物体-关系表示。SayPlan 假设已有可靠 3D 场景图（不现实），其他 graph-then-plan 方法把构图与规划当作分离模块；MomaGraph 让单个 VLM 联合完成两件事。
- **启发**：对任何"先建结构化中间表示再决策"的任务（不限机器人），"用 RL + 结构对齐奖励训模型自己产出中间表示"是一条值得借鉴的路径；而"任务导向地裁剪表示，只保留与当前目标相关的最小信息"也呼应了认知科学的注意力机制，可推广到检索、规划、agent 记忆等场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 统一空间-功能 + 部件节点 + 状态感知更新的场景图表示是清晰的概念性创新，Graph-then-Plan 让单 VLM 联合构图与规划也有新意；不过场景图、graph-then-plan、RL 训练 VLM 各自都有先例，更多是巧妙整合。
- **实验充分度**: ⭐⭐⭐⭐ — 自建数据 + 基准 + 闭源/开源多基线对比 + 单一/统一关系消融 + 真机迁移，证据链完整；扣分在数据规模偏小、基准为多选 VQA 形式、状态更新模块的独立消融较薄。
- **写作质量**: ⭐⭐⭐⭐ — 动机用认知科学类比讲得清楚，三处痛点-三处设计对应工整，公式与定义规范；部分实现细节（更新函数 $U$）偏简。
- **价值**: ⭐⭐⭐⭐ — 补齐了统一场景图的数据/基准/模型基础设施，7B 开源追平闭源大模型且能上真机，对具身规划社区实用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Planning with an Embodied Learnable Memory](planning_with_an_embodied_learnable_memory.md)
- [\[CVPR 2026\] RoboAgent: Chaining Basic Capabilities for Embodied Task Planning](../../CVPR2026/robotics/roboagent_chaining_basic_capabilities_for_embodied_task_planning.md)
- [\[ICML 2026\] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](../../ICML2026/robotics/embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)
- [\[CVPR 2026\] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](../../CVPR2026/robotics/recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)
- [\[ICLR 2026\] EVLP: Learning Unified Embodied Vision-Language Planner with Reinforced Supervised Fine-Tuning](evlp_learning_unified_embodied_vision-language_planner_with_reinforced_supervise.md)

</div>

<!-- RELATED:END -->
