---
title: >-
  [论文解读] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data
description: >-
  [ACL 2026][多智能体][自动特征工程] 提出 MALMAS，一个记忆增强的 LLM 多智能体系统用于表格数据自动特征生成，通过六个专职 Agent 分工探索不同特征空间维度 + 三级记忆机制（过程/反馈/概念）实现跨轮迭代优化，在 16 个分类和 7 个回归数据集上超越现有基线。 领域现状：自动特征生成是 Auto…
tags:
  - "ACL 2026"
  - "多智能体"
  - "自动特征工程"
  - "多智能体系统"
  - "记忆增强"
  - "表格数据"
  - "AutoML"
---

# Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data

**会议**: ACL 2026  
**arXiv**: [2604.20261](https://arxiv.org/abs/2604.20261)  
**代码**: [GitHub](https://github.com/fxdong24/MALMAS)  
**领域**: LLM/NLP  
**关键词**: 自动特征工程, 多智能体系统, 记忆增强, 表格数据, AutoML

## 一句话总结
提出 MALMAS，一个记忆增强的 LLM 多智能体系统用于表格数据自动特征生成，通过六个专职 Agent 分工探索不同特征空间维度 + 三级记忆机制（过程/反馈/概念）实现跨轮迭代优化，在 16 个分类和 7 个回归数据集上超越现有基线。

## 研究背景与动机

**领域现状**：自动特征生成是 AutoML 的关键环节，目标是从原始表格数据中自动构造高质量特征。传统方法（如 DFS、OpenFE）依赖预定义算子库进行组合搜索，而近期 LLM 方法（如 CAAFE）引入语义信息来指导特征变换，但仍有局限。

**现有痛点**：(1) 传统方法受限于固定算子集，无法利用任务语义，搜索空间狭窄；(2) LLM 方法虽然引入了语义信号，但依赖单一生成策略、思维模式固化，导致特征空间探索仍然受限；(3) 更关键的是，现有 LLM 方法缺乏来自下游学习目标的反馈机制——生成过程与模型性能脱钩，只能做低效的试错探索。

**核心矛盾**：特征空间的高维度和多样性与单一 Agent 的有限探索能力之间的矛盾，以及"生成→评估→优化"闭环的缺失。

**本文目标**：设计一个多 Agent 协作 + 记忆驱动的自动特征生成框架，能够 (1) 通过角色分工广泛探索特征空间，(2) 通过多级记忆实现跨轮的经验积累和策略调整。

**切入角度**：从特征工程实践中的"黄金特征"分类出发，沿三个正交维度（变换复杂度、数据范围、数据类型依赖性）设计专职 Agent，并引入过程记忆（做了什么）、反馈记忆（效果如何）、概念记忆（为什么有效）三级经验系统。

**核心 idea**：将特征生成分解为多个专职 Agent 的并行探索 + Router Agent 动态调度 + 三级记忆驱动的迭代优化。

## 方法详解

### 整体框架
每轮迭代：Router Agent 从 Agent 池中选择本轮激活的子集 → 每个活跃 Agent 根据元数据+记忆构建 prompt，与 LLM 多轮交互生成特征 → 评估生成特征在下游模型上的验证性能 → 更新三级记忆 → Summary Agent 汇总全局概念记忆 → 选取 TopN 特征加入数据集 → 进入下一轮。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["表格数据 + 任务元数据"] --> B["Router Agent 动态调度<br/>读元数据与记忆选本轮激活子集"]
    subgraph AG["六专职 Agent + Router 并行架构"]
        direction TB
        C["一元 / 交叉 / 时序 / 聚合 / 局部变换 / 局部模式<br/>各管一类特征变换，覆盖三个正交维度"]
    end
    B --> AG
    AG --> D["下游模型评估<br/>XGBoost 验证 AUC / NRMSE"]
    subgraph MEM["三级记忆机制"]
        direction TB
        E["过程记忆 ProcMem<br/>做了什么"] --> F["反馈记忆 FeedMem<br/>效果如何·信用分配"] --> G["概念记忆 ConMem<br/>为什么有效"]
    end
    D --> MEM
    MEM --> H["全局概念记忆与跨 Agent 知识传递<br/>Summary Agent 汇总 GlobalMem"]
    H -->|"TopN 特征加入数据集，进入下一轮"| B
    H --> I["输出增强特征集"]
```

### 关键设计

**1. 六专职 Agent + Router 的并行架构：让不同 Agent 各管一类特征变换，避免单一思路把特征做得千篇一律**

单个 LLM 反复生成特征容易同质化（feature homogenization），思路固化在少数几种变换上。MALMAS 把特征构造拆给六个专职 Agent，分别负责一元变换（Unary）、交叉组合（Cross-Compositional）、时序特征（Temporal）、聚合构造（Aggregation-Construct）、局部变换（Local-Transform）和局部模式（Local-Pattern），这组角色恰好覆盖变换复杂度、数据范围、数据类型三个正交维度，让它们去探索互补的特征区域而非互相重叠。每一轮并不全员上阵：Router Agent 先读任务元数据和累积记忆，动态挑出本轮该激活的 Agent 子集，把算力花在当前数据集最可能见效的方向上。

**2. 三级记忆机制（Procedural + Feedback + Conceptual）：给无状态的 LLM 生成装上"做了什么→效果如何→为什么有效"的经验链**

没有记忆的 LLM 每轮都从零生成，既会重复探索同样的变换，也学不到哪些操作真正有用。MALMAS 用三级记忆把每轮反馈沉淀下来：过程记忆（ProcMem）记录每次变换的完整 trace（基列、变换类型、特征名、描述、轮次），让后续轮次绕开已试过的路；反馈记忆（FeedMem）把每个生成特征与它在下游模型上的验证指标绑定，做显式的信用分配，知道哪条变换带来了增益；概念记忆（ConMem）则由 LLM 从前两者中蒸馏出可复用的启发式规则。三层逐级抽象，对应短期避错、中期导向、长期策略适应，让迭代不再是盲目试错。

**3. 全局概念记忆与跨 Agent 知识传递：把一个 Agent 学到的有效模式广播给全队，减少重叠探索**

局部记忆只服务单个 Agent，A 学到的经验 B 用不上，仍会各自造出冗余特征。每轮结束后，Summary Agent 汇总所有活跃 Agent 的概念记忆和反馈记忆，凝练成一份全局概念记忆 GlobalMem；下一轮 Router 的调度决策和每个 Agent 的 prompt 构建都会参考它。这样某个 Agent 发现的好模式能传播到全队，Router 也能据此更聪明地分派任务，把有效探索的成果在 Agent 之间复用起来。

### 一个完整示例：在 Titanic 上跑一轮迭代

以 Titanic 数据集为例。某一轮开始时，Router 读到任务元数据（含 Age、Fare、Pclass、Sex 等列）和已有记忆，判断时序特征意义不大，于是只激活一元变换、交叉组合和聚合构造三个 Agent。一元 Agent 对 Fare 做对数变换，交叉 Agent 把 Pclass 与 Sex 组合出新类别，聚合 Agent 按舱位算平均票价。三者生成的特征接入下游 XGBoost 评估，反馈记忆记录下"log(Fare) 让 AUC 微升、Pclass×Sex 增益最大"；Summary Agent 把"舱位与性别的交叉对生存预测有效"写进全局概念记忆。下一轮 Router 据此继续侧重交叉组合方向，而过程记忆确保不会再重复生成 log(Fare)。几轮迭代下来，Titanic 的 AUC 收敛到 0.872，高于次优方法的 0.849。

### 损失函数 / 训练策略
目标是最大化验证集上下游模型的性能指标（分类用 AUC，回归用 NRMSE）。使用 XGBoost 作为下游模型，每轮通过 TopN-Features 筛选保留最优特征。

## 实验关键数据

### 主实验（分类 AUC，16 数据集平均排名）

| 方法 | 类型 | Mean Rank |
|------|------|-----------|
| DFS | 传统 | 3.69 |
| OpenFE | 传统 | 3.12 |
| CAAFE | LLM | 3.57 |
| OCTree | LLM | 4.81 |
| LLMFE | LLM | 3.75 |
| **MALMAS** | **多Agent+记忆** | **1.12** |

### 消融实验（关键组件贡献）

| 配置 | 说明 |
|------|------|
| 单 Agent (无 Router) | 特征多样性下降，同质化严重 |
| 无记忆 | 每轮独立生成，大量重复探索 |
| 无全局记忆 | Agent 间无知识传递，冗余特征增多 |
| 无反馈记忆 | 无法从历史中学习哪些变换有效 |
| **MALMAS (完整)** | 最优表现，Mean Rank 1.12 |

### 关键发现
- **MALMAS 在 16 个分类数据集中平均排名 1.12**，远超第二名 OpenFE (3.12)
- **在难数据集上优势更明显**：如 Titanic (0.872 vs 次优 0.849)、Credit_G (0.775 vs 次优 0.758)
- **记忆机制是关键**：概念记忆将"为什么某变换有效"抽象为可复用规则，指导后续探索
- **Router Agent 的动态调度**避免了对所有数据集千篇一律地激活所有 Agent

## 亮点与洞察
- **三级记忆的层次设计**很有启发性：从操作 trace 到信用分配到策略抽象，对应了认知心理学中的程序性记忆→工作记忆→元认知。可以迁移到任何需要迭代优化的多 Agent 系统
- **Router Agent 的动态调度**解决了"所有 Agent 都跑一遍"的计算浪费，实现了 task-dependent 的资源分配
- **从"黄金特征"分类出发设计 Agent 角色**是一个好实践——将领域知识编码到 Agent 分工中

## 局限与展望
- Agent 角色划分是手工设计的，能否自动发现最优分工？
- 记忆管理没有遗忘机制，长轮次下可能导致上下文膨胀
- 下游模型固定为 XGBoost，在深度学习模型上的效果未验证
- 未与 AutoML 全流程方法（如 Auto-sklearn）做端到端对比
- 可以探索 Agent 间的对抗/辩论机制来提升特征质量

## 相关工作与启发
- **vs CAAFE**: CAAFE 用单个 LLM 生成特征，受限于单一生成策略。MALMAS 通过多 Agent 分工+记忆反馈大幅扩展了探索空间
- **vs OpenFE**: OpenFE 用树模型的算子搜索，高效但受限于预定义算子。MALMAS 利用 LLM 的语义理解生成更多样的变换
- **vs Generative Agents**: 后者在社会模拟中使用多 Agent+记忆，MALMAS 将类似范式引入特征工程，是该思路的新应用方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 多 Agent + 三级记忆用于特征生成是新的组合，但各组件单独看并不新
- 实验充分度: ⭐⭐⭐⭐ 23 个数据集覆盖广，但消融细节不够详细
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，但符号略显冗余
- 价值: ⭐⭐⭐⭐ 对 AutoML 社区有实际贡献，三级记忆的设计思路有广泛迁移价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)
- [\[ACL 2025\] DocAgent: A Multi-Agent System for Automated Code Documentation Generation](../../ACL2025/multi_agent/docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)
- [\[ACL 2026\] A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation](a_multi-agent_framework_for_feature-constrained_difficulty_control_in_reading_co.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](../../AAAI2026/multi_agent/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)

</div>

<!-- RELATED:END -->
