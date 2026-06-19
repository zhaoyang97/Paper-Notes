---
title: >-
  [论文解读] Language Model as Planner and Formalizer under Constraints
description: >-
  [ACL 2026][LLM推理][约束规划] 本文提出 CoPE 基准，通过向经典规划环境注入形式化分类的自然语言约束，揭示出仅一句约束即可将当前最强 LLM 的规划性能减半，暴露了 LLM 规划鲁棒性的严重不足。 领域现状：LLM 在规划领域有两种主流范式——LLM-as-Planner 直接端到端生成动作序列…
tags:
  - "ACL 2026"
  - "LLM推理"
  - "约束规划"
  - "LLM-as-Planner"
  - "LLM-as-Formalizer"
  - "基准测试"
  - "PDDL"
---

# Language Model as Planner and Formalizer under Constraints

**会议**: ACL 2026  
**arXiv**: [2510.05486](https://arxiv.org/abs/2510.05486)  
**代码**: [GitHub](https://github.com/CassieHuang22/LLM-as-Formalizer-constraints)  
**领域**: LLM评测  
**关键词**: 约束规划, LLM-as-Planner, LLM-as-Formalizer, 基准测试, PDDL

## 一句话总结

本文提出 CoPE 基准，通过向经典规划环境注入形式化分类的自然语言约束，揭示出仅一句约束即可将当前最强 LLM 的规划性能减半，暴露了 LLM 规划鲁棒性的严重不足。

## 研究背景与动机

**领域现状**：LLM 在规划领域有两种主流范式——LLM-as-Planner 直接端到端生成动作序列，LLM-as-Formalizer 将自然语言描述转为 PDDL 等形式语言再用求解器推导方案。两种方法在标准规划基准上均展现了不俗的能力。

**现有痛点**：然而，现有基准（如 BlocksWorld、Gripper 等）大多诞生数十年，环境描述简单、同质化严重，且高度可能被 LLM 训练数据覆盖。这种简单性可能导致对 LLM 规划能力的**过度高估**，在下游安全敏感场景中构成隐患。

**核心矛盾**：真实世界的规划指令通常包含用户或资源施加的**个性化需求与约束**，而标准基准完全缺少这些元素。已有的增强方法仅加入噪声或词汇扰动，未改变语义本身。

**本文目标**：构建一个语义层面增强的约束规划基准，系统评估 LLM 在约束条件下的规划和形式化能力。**切入角度**：将约束按语言学和实用主义方法形式化为四类（Initial、Goal、Action、State），确保分类的完备性。**核心 idea**：简单的一句话约束即可大幅降低 LLM 性能，且这种性能下降在问题复杂度增加和词汇混淆时进一步加剧。

## 方法详解

### 整体框架

CoPE (Constrained Planning Environments) 在 BlocksWorld 和 CoinCollector 两个域上，为每个问题手动标注自然语言约束及其四种形式语言的 ground-truth 编码。评估流程：给定域描述 $D_d$、问题描述 $D_p$、PDDL 头部 $\mathcal{DF}'$ 和约束 $\mathcal{C}$，LLM 生成计划（Planner）或形式化代码（Formalizer），最终用 VAL 验证器验证计划正确性。

### 关键设计

**1. 四类约束的形式化定义：先把"约束"这件事讲清楚，分类才能完备**

要系统评估 LLM 在约束下的表现，前提是约束本身有一套不重不漏的分类，否则结论无从归因。CoPE 按约束作用于规划问题的哪个部位，把自然语言约束严格分成四类：Initial 改写初始状态、Goal 改写目标状态、Action 限制合法的动作序列、State 限制合法的状态轨迹。分类不是拍脑袋切的，而是基于原始动作/状态空间（primitive）与被约束后空间（modified）之间的集合关系来定义，并证明 State 这一子类在形式上涵盖了所有可能的约束，从而保证了分类的完备性。有了这套完备分类，后续"哪类约束最难、哪种形式语言最擅长哪类"的分析才有干净的坐标系——因为 PDDL、PDDL3、LTL、SMT 对不同类别约束的表达力本就参差不齐。

**2. 多形式语言对比评估：让四种形式语言在同一批约束上同台竞技**

Formalizer 路线的成败很大程度取决于"用哪种形式语言去编码约束"，但此前没人在同一基准上把它们摆在一起比。CoPE 把每条约束分别编码成 PDDL 1.2、PDDL3、LTL 和 SMT（Z3）四种语言的 ground-truth，并配三条技术路线观察 LLM 的生成行为：Generation 直接一次性生成带约束的代码、Editing 先生成无约束版本再补上约束、Revision 则允许基于求解器报错最多迭代修正 3 次。不同约束天然偏好不同语言——PDDL3 的语法本就为状态约束而设、SMT 擅长把状态谓词建模成可满足性问题——这套横向对比因此能直接告诉后来者：面对某类约束该选哪条工具链，而不必各自从头试错。

**3. 鲁棒性拓展实验：把约束和"复杂度 / 词汇污染"叠加，看脆弱性会不会被放大**

只在标准小规模问题上测约束还不够，真正的隐患在于约束是否会放大 LLM 本就存在的脆弱性。CoPE 为此设计了两个加压版本：BlocksWorld-XL 把方块数撑到 50，测实体空间膨胀后约束的杀伤是否加剧；MysteryBlocksWorld 则把所有类型、谓词、动作名替换成无意义占位符，斩断模型对训练数据中熟悉词汇的依赖，测约束与"数据污染解除"叠加后的真实能力。这两组实验把"约束影响"从单一难度切片拓展成一条压力曲线，正是它们暴露出最尖锐的结论——Formalizer 原有的复杂度鲁棒性和词汇鲁棒性，在约束面前几乎完全消失。

### 损失函数 / 训练策略

本文为评估型工作，不涉及模型训练。核心评估指标为 **plan correctness**——预测计划在 ground-truth PDDL 环境中能否成功从初始状态转移到目标状态。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 无约束 | 有约束 | 下降幅度 |
|--------|------|--------|--------|----------|
| BlocksWorld | LLM-as-Planner (Gemini-3-Flash) | ~85% | ~55% | ~30% |
| BlocksWorld | LLM-as-PDDL-Formalizer (Gemini) | ~70% | ~40% | ~30% |
| CoinCollector | LLM-as-Planner (Gemini) | ~90% | ~60% | ~30% |
| BlocksWorld | PDDL3 Formalizer | 低于 PDDL | 更低 | 语法/编译错误多 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Generation | 基线 | 直接生成约束代码 |
| Editing | 部分提升 | 先生成无约束版本再编辑 |
| Revision | 进一步提升 | 基于求解器错误迭代修正 |
| BlocksWorld-XL (50 blocks) | 性能骤降 | 复杂度放大后约束影响更严重 |
| MysteryBlocksWorld | Formalizer 鲁棒性消失 | 约束 + 词汇混淆双重打击 |

### 关键发现
- 一句话约束一致性地将性能减半，所有 LLM、方法、语言组合均受影响
- LLM-as-Planner 在无约束时整体优于 Formalizer，但 Formalizer 对问题复杂度更鲁棒
- PDDL3 虽然有约束语法支持，但因训练数据稀缺表现反而不如标准 PDDL
- 约束引入后，Formalizer 原有的复杂度鲁棒性和词汇扰动鲁棒性**完全消失**

## 亮点与洞察
- 约束分类的形式化定义非常严谨，证明了完备性，可作为后续工作的理论基础
- 实验设计覆盖 4 个 LLM × 4 种形式语言 × 3 种技术 × 4 类约束 × 4 个数据集，分析维度丰富
- 揭示了一个重要结论：**简单的语义修改比词汇噪声更能有效挑战 LLM**，为基准设计提供新思路
- CoPE 的设计理念——通过语义增强而非数据扰动来对抗数据污染——值得在其他 NLP 评估任务中借鉴

## 局限与展望
- 约束类型仅考虑单约束，未讨论约束的合取、否定和歧义，真实场景的约束更为多样
- BlocksWorld 和 CoinCollector 域仍较简单，与真实世界规划场景（如机器人操作、资源调度）有较大差距
- 评估指标（plan correctness）可能存在 false positive——计划碰巧正确但代码未真正编码约束，不过验证显示比例可忽略
- 未来方向：支持更复杂的约束组合、扩展到更多域、开发约束感知的规划工具链
- 自主 Agent 在下游任务中的安全风险值得关注，形式化表示可提供人类审计和形式验证的透明性

## 相关工作与启发
- **vs 标准 IPC 基准**: CoPE 通过语义修改而非仅加噪声来挑战 LLM，更能暴露真实能力
- **vs LLM+P (Liu et al., 2023)**: 同为 Formalizer 路线但未考虑约束，CoPE 揭示其局限
- **vs Mystery BlocksWorld**: CoPE 表明约束比词汇混淆更能削弱 Formalizer 的鲁棒性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化的约束规划 LLM 评估基准，形式化分类严谨
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多模型 × 多语言 × 多技术 × 多域，分析极为详尽
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，结构合理，图表丰富
- 价值: ⭐⭐⭐⭐ 为 LLM 规划研究敲响警钟，指明了从简单基准到现实约束的重要研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICML 2026\] On the Generalization Gap in Self-Evolving Language Model Reasoning](../../ICML2026/llm_reasoning/on_the_generalization_gap_in_self-evolving_language_model_reasoning.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](../../ICLR2026/llm_reasoning/estimating_the_empowerment_of_language_model_agents.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)

</div>

<!-- RELATED:END -->
