---
title: >-
  [论文解读] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards
description: >-
  [NeurIPS 2025][AI for Social Impact] 本文主张 AI for Social Impact (AISI) 领域的学术生态需要双轨改革：拓宽"影响力"的定义以认可非部署/非方法创新的贡献，同时对已部署系统采用因果推断级别的严格评估标准。 问题背景 AI/ML for Social Impact…
tags:
  - "NeurIPS 2025"
  - "AI for Social Impact"
  - "evaluation standards"
  - "deployment"
  - "field experiments"
  - "research ecosystem"
  - "causal inference"
---

# Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards

**会议**: NeurIPS 2025  
**arXiv**: [2510.18238](https://arxiv.org/abs/2510.18238)  
**作者**: Bryan Wilder (Carnegie Mellon University), Angela Zhou (University of Southern California)
**代码**: 无（Position Paper）  
**领域**: 其他  
**关键词**: AI for Social Impact, evaluation standards, deployment, field experiments, research ecosystem, causal inference

## 一句话总结

本文主张 AI for Social Impact (AISI) 领域的学术生态需要双轨改革：拓宽"影响力"的定义以认可非部署/非方法创新的贡献，同时对已部署系统采用因果推断级别的严格评估标准。

## 研究背景与动机

### 问题背景

AI/ML for Social Impact (AISI) 领域近十年发展迅速，已有 AAAI Special Track on AI for Social Impact、IJCAI Multi-Year Track On AI And Social Good 等专门轨道，以及多所大学的专题课程和暑期项目。然而，当前学术生态存在两个结构性缺陷：

**缺陷一：对"理想项目"的单一模板**

现有审稿标准将"方法创新 + 实际部署"视为 AISI 论文的黄金标准。AAAI AISI Track 的审稿标准明确优先考虑已部署或接近部署的项目（"Scope and promise for social impact"）和方法新颖性（"Novelty of approach"）。IJCAI 评估标准包括"contribution to state-of-the-art AI"和"collaboration with stakeholders/partners"，以"potential deployment opportunities"为关键期望。

这种单一模板导致三个问题：
1. 研究者被迫将所有项目套入"新方法→部署"框架，即使合作伙伴的需求并非如此
2. 纯应用贡献（帮助组织正确使用现有工具）难获学术认可
3. 有实践潜力但未部署的方法研究被低估

**缺陷二：部署评估缺乏严格性**

即使在部署框架内，当前评估实践远未达到经济学、医学等领域的标准。部署被视为"终点线"，但简单的 before-after 比较可能被时间趋势、人群变化等混淆因素所干扰。ML 社区普遍缺乏因果推断方法论训练。

### 核心立场

研究者和审稿者必须同时做到：(1) 采纳更广泛的社会影响概念——不局限于部署；(2) 对已部署系统的评估采用更严格的标准。

## 方法详解

### 改革路径一：承认非方法贡献 (Non-method Contributions)

当合作伙伴（非营利组织、政府机构）缺乏 ML 专业知识时，研究者帮助其正确使用已有工具即可产生重大影响。这类工作可回答对 ML 领域有广泛科学意义的问题：

- **ML 如何嵌入组织决策？** 建模选择如何改变工具的使用方式？
- **ML 在特定领域的真实价值？** 改进预测是否真的改善了结果？如何改善？
- **哪些 formulation 不适用？** 负面结果同样有科学价值——其他研究者可从前人的 formulation 探索中学习
- **复杂度控制在什么水平？** 更简单的 formulation 是否已足够？

作者同时指出边界：如果 ML 模型本身对应用领域有科学贡献（如临床预测模型），此类论文更适合在应用领域期刊发表。ML 会议应欢迎那些通过应用项目得出关于 ML 使用和影响本身的科学结论的论文。

### 改革路径二：承认非部署贡献 (Non-deployment Contributions)

方法研究即使未部署也可通过以下途径产生社会影响：

**改变从业者认知**：最有影响力的方法工作是改变应用研究者和数据科学家"做事方式"的工作。要理解合作伙伴数据分析师的多样化学科背景（社会科学、健康科学、政治科学等），使方法的核心思想能翻译到这些领域。

**为维护而设计 (Design for Maintenance)**：Mattern (2018) 指出维护被系统性地低估。合作伙伴通常没有 ML PhD 团队维护复杂管线。DellaVigna et al. (2024) 发现，利用现有基础设施的项目更容易从 pilot 走向正式部署——工具越简单越容易被采用。医学中的 nomogram（逻辑回归系数计算器，可在病床前手动计算）就是典型案例。

**单变量基准 (Single-variable Benchmarks)**：Perdomo et al. (2023)、Stoddard et al. (2024)、Salganik et al. (2020) 表明单变量预测在许多社会预测任务中性能可与复杂 ML 方法相媲美。报告此类基准有助于：(a) 衡量复杂 ML 的边际改进；(b) 帮助不同资源水平的组织选择方案；(c) 支持发现的可迁移性——不同组织的数据列和 schema 可能不同，但单变量基准可直接复现比较。

### 改革路径三：提高部署评估标准

作者根据部署类型提出差异化的评估框架：

**Pilot Test（概念验证）**：目标是评估可行性和可接受性，而非效果。作者应明确标识为 pilot，主要结论限于可行性发现，不应据此声称方法有效。

**Randomized Controlled Trial (RCT)**：应采纳以下最佳实践——
- **预注册 (Preregistration)**：试验开始前公开注册协议和分析策略，防止 p-hacking。ML 会议应要求作者声明是否预注册
- **统计功效分析 (Power Analysis)**：确定样本量能检测到的最小效应量，避免浪费合作伙伴资源
- **结果有效性 (Outcome Validity)**：论证结果指标与真实福利的关联，而非仅依赖平台默认记录的指标
- **异质性分析的规范策略**：预注册子组分析或算法程序，控制多重比较导致的假阳性
- **随机化单元选择**：个人水平 vs. 群组水平（服务中心/医院/学校），取决于算法的作用层级
- **资源分配困境**：算法常作为另一干预的分配机制，理想 RCT 应在群组水平随机化，但实施伙伴可能只有一个"群组"

**Non-randomized Deployment（非随机部署）**：应理解为事件研究 (event study)——
- 核心思想：构建反事实 (counterfactual)——如果没有部署系统，会发生什么？
- **Interrupted Time Series**：利用部署前时间趋势外推反事实；论文用 Figure 1 示意——如果部署前成果已在改善，简单 before-after 比较会高估甚至掩盖负面因果效应
- 更高级设计：Differences-in-Differences（需对照组 + parallel trends 假设）、Synthetic Control（需多个可比对照单元）
- 最低要求：报告部署前结果趋势和前后人群构成变化

## 实验关键数据

本文为 position paper，不含传统实验。以下用表格梳理核心论证结构。

**表1：三类贡献维度与当前评审覆盖情况**

| 贡献类型 | 具体形式 | 当前评审认可度 | 作者建议 |
|---------|---------|--------------|---------|
| 非方法贡献 | 帮助伙伴正确使用已有 ML 工具 | 低：ML 会议不接受无新方法的论文 | 认可研究 ML 使用/影响本身的科学结论 |
| 非方法贡献 | 研究 ML 如何进入组织决策流程 | 低：主要在 HCI/跨学科场合 | ML 会议应欢迎深入方法细节的使用研究 |
| 非部署贡献 | 改变从业者对估计策略的认知 | 中：有一定认可但缺细粒度标准 | 增加"可维护性"和"可迁移性"评审维度 |
| 非部署贡献 | 简单工具/单变量基准 | 低：简单方法难展示"方法新颖性" | 正常化简单基准，不应视为贬低创新 |
| 部署贡献 | 已部署系统 + 效果评估 | 高：最受认可的贡献类型 | 要求更高评估严格性（RCT/event study） |

**表2：三种部署评估方法的比较**

| 评估方法 | 适用场景 | 核心要求 | 主要威胁/局限 |
|---------|---------|---------|-------------|
| Pilot Test | 初始小规模测试 | 明确标识为 pilot；结论限于可行性 | 不适用于声称效果 |
| RCT | 有条件进行随机化 | 预注册、功效分析、结果有效性、异质性策略 | 随机化单元选择、资源分配困境 |
| Non-randomized (Event Study) | 无法随机化 | 构建反事实、报告 pre-trends、控制人群变化 | 无对照组时依赖 time series 外推假设 |

## 亮点与洞察

1. **生态系统视角**：本文不讨论某个方法或论文的好坏，而是系统性分析 AISI 领域的激励结构和协调失败 (coordination failure)——这在 ML 社区极为少见的"元研究"

2. **"Design for Maintenance" 理念**：简单工具更容易被采用和维护，但学术界的逆向激励使简单方法难以发表。Nomogram（病床前手动计算逻辑回归系数）案例说明实际采用的技术远比文献讨论的简单

3. **因果推断视角引入 ML 评估**：明确指出 ML 部署评估本质上是因果推断任务，需要 counterfactual reasoning，而非传统 accuracy/loss 指标。Figure 1 的 interrupted time series 示意图直观有力

4. **Portfolio 思维**：让研究者建立跨贡献类型的 portfolio（有时做方法、有时做应用、有时做部署），降低单个项目的激励扭曲，提高整体生态可持续性

5. **单变量基准的深层含义**：许多社会预测问题的 Bayes error 很高（低信噪比），复杂 ML 边际增益有限。这与推荐系统的 popularity baseline 现象类似

6. **对合作伙伴的道德责任**：要求每个项目都包含方法创新会使研究者将合作引向"容易出论文"而非"伙伴最需要"的方向——这是对投入有限资源的伙伴组织的不公

## 局限性

1. **缺乏定量证据**：论证主要依赖逻辑推理和案例分析，未进行系统性 meta-analysis 或对审稿决定的统计分析

2. **实施门槛高**：提出的评估标准（RCT 预注册、power analysis、event study）对多数 ML 研究者门槛较高，跨学科因果推断训练并非标配

3. **审稿改革可执行性存疑**：ML 会议审稿者以方法导向研究者为主，如何保证他们能评估 event study 或 RCT 质量是实际挑战

4. **偏向北美学术体系**：讨论的激励结构（tenure、promotion、collaboration 模式）更贴近北美环境

5. **引用案例的自洽性**：论文引用的 AISI 项目案例（如希腊 COVID-19 RL 系统）本身是否符合其提出的严格评估标准，未做检验

6. **与 ML+X 场合的关系模糊**：ML+Health、ML+Science 等专门场合在快速发展，AISI 如何与之互补而非竞争讨论不足

## 相关工作与启发

### 相关工作

- **AISI 领域奠基文献**：Tomavsev et al. (2020)、Tambe et al. (2022)、Rolnick et al. (2024) 强调 AISI 需要与合作伙伴深度协作、面对 last-mile 挑战
- **AISI 代表性项目**：Bastani et al. (2021) 希腊 COVID-19 旅客检测 RL 系统；Shi et al. (2021) 食物救援志愿者调度系统
- **因果推断方法论**：Cunningham (2021)、Angrist & Pischke (2009) 提供 RCT 和 event study 标准参考；Freyaldenhoven et al. (2021) 讨论 panel event-study 设计
- **ML 可复现性**：Beam et al. (2020)、Pineau et al. (2021) 关注 ML 计算评估的可复现性，本文将此延伸到部署评估
- **Validity 研究**：Coston et al. (2023)、Jacobs & Wallach (2021) 将社会科学 validity 概念引入 ML 决策系统评估
- **预测难度基准**：Perdomo et al. (2023)、Salganik et al. (2020) 证明简单基准在社会预测中的竞争力

### 启发

- **对 ML 会议的直接建议**：(1) 要求 field experiment 声明是否预注册；(2) 展示 AISI 特有贡献类型示例；(3) 在 review guidelines 加入可维护性、单变量基准等细粒度维度
- **对工程实践的启示**：在 ML 系统部署设计阶段就规划评估方案（RCT 或 event study），而非部署后回顾分析
- **与 AI Safety/Alignment 的共鸣**：两者都在追问"如何知道 AI 系统真的有正面效果？"，但本文从实用主义因果推断角度切入
- **KISS 原则在学术界的矛盾**：工程界推崇 KISS，但学术激励使简单方法难以发表，需要制度层面系统性解决

## 评分

- 新颖性: ⭐⭐⭐⭐ (作为 position paper 提出了清晰的结构性批评和三步改革方案)
- 实验充分度: ⭐⭐ (无实验，论证基于案例和逻辑推理)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰、论证严谨、建议具体可操作)
- 推荐度: ⭐⭐⭐⭐ (对 AISI 生态有实际指导意义，event study 讨论对所有 ML 部署都有参考价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](../../ACL2025/others/sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[ICML 2026\] Comprehensive AI Governance Requires Addressing Non-Model Gains](../../ICML2026/others/comprehensive_ai_governance_requires_addressing_non-model_gains.md)
- [\[ICML 2025\] Position: AI Evaluation Should Learn from How We Test Humans](../../ICML2025/others/position_ai_evaluation_should_learn_from_how_we_test_humans.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)

</div>

<!-- RELATED:END -->
