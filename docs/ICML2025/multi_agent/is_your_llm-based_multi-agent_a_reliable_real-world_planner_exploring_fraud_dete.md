---
title: >-
  [论文解读] Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning
description: >-
  [ICML 2025 (Workshop MAS)][多智能体][multi-agent planning] 提出 WandaPlan 评估环境，通过在旅行规划场景中注入三种递进式欺诈（单源误导、团队协调刷单、逐级升级），系统性评估 LLM 多智能体规划系统对虚假信息的脆弱性，并设计反欺诈 Agent 来缓解风险。
tags:
  - "ICML 2025 (Workshop MAS)"
  - "多智能体"
  - "multi-agent planning"
  - "fraud detection"
  - "travel planning"
  - "LLM reliability"
  - "evaluation benchmark"
---

# Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning

**会议**: ICML 2025 (Workshop MAS)  
**arXiv**: [2505.16557](https://arxiv.org/abs/2505.16557)  
**代码**: 无  
**领域**: Social Computing / LLM Agent  
**关键词**: multi-agent planning, fraud detection, travel planning, LLM reliability, evaluation benchmark

## 一句话总结

提出 WandaPlan 评估环境，通过在旅行规划场景中注入三种递进式欺诈（单源误导、团队协调刷单、逐级升级），系统性评估 LLM 多智能体规划系统对虚假信息的脆弱性，并设计反欺诈 Agent 来缓解风险。

## 研究背景与动机

**领域现状**：基于 LLM 的多智能体规划系统（如 AutoGPT、CrewAI 等框架）正在快速发展，能够自主协作完成复杂的现实世界任务。旅行规划是一个典型的应用场景，需要从餐厅、酒店、交通等多个数据源检索信息并做出决策。

**现有痛点**：这些系统在执行规划任务时高度依赖外部数据源（评论网站、社交媒体、地图服务等），而这些数据源充斥着虚假评论、刷单好评、误导性描述等欺诈内容。现有评估 benchmark（如 TravelPlanner）仅关注任务完成效率和合理性，完全忽略了数据真实性维度。

**核心矛盾**：越"高效"的规划框架越容易被欺诈信息利用——因为它们倾向于快速信任高评分数据源而不做交叉验证。框架的效率优化反而成为安全漏洞：高效 = 信任数据 = 容易被欺骗。

**本文目标** 构建一个能够评估多智能体规划系统在面对真实世界欺诈信息时安全性和鲁棒性的系统化测试环境，并提出初步的防御方案。

**切入角度**：不是单纯做攻击，而是构建一个完整评估框架——包括真实数据模拟 + 欺诈注入 + 多维评估指标 + 防御方案。

**核心 idea**：现有 LLM 多智能体重效率轻安全，需要专门的反欺诈评估和防御机制。

## 方法详解

### 整体框架

WandaPlan 是一个完整的评估生态系统，包含四个核心组件：
1. 真实世界数据模拟环境（镜像真实评论/价格/描述数据）
2. 三种递进式欺诈注入方案
3. 多维度评估指标体系
4. 即插即用的反欺诈 Agent 模块

### 关键设计

1. **三种递进式欺诈场景**:
    - 功能：模拟真实世界中从简单到复杂的欺诈演化
    - 核心思路：
        - **Misinformation Fraud（单源误导）**：在单个数据源中注入虚假高评分或虚假描述。例如给一家低质量餐厅伪造5星评价和虚假菜品描述。这是最基础的攻击形式
        - **Team-Coordinated Multi-Person Fraud（多人协调刷单）**：模拟水军团队——多个"用户"从不同账号、在不同时间段发布一致的虚假好评，形成群体性误导。每个单独的评论看起来真实，但联合分析时能发现模式
        - **Level-Escalating Multi-Round Fraud（逐级升级欺诈）**：欺诈策略根据系统的反馈动态调整——如果系统拒绝了某推荐，欺诈者会升级策略（如加入折扣信息、编造限时优惠等），模拟自适应对抗
    - 设计动机：三种模式分别对应真实世界欺诈的不同阶段和复杂度，递进式设计可以逐步测试系统的抗欺诈能力上限

2. **反欺诈 Agent（Anti-Fraud Agent）**:
    - 功能：作为即插即用的安全模块嵌入现有规划框架
    - 核心思路：该 Agent 执行三层防御——
        - **信息交叉验证**：对同一实体从多个独立数据源获取信息并比对一致性
        - **评论异常检测**：分析评论的时间分布、用词模式、评分分布，识别刷单特征
        - **可疑数据标记**：对置信度低的数据源打上警告标签，由规划系统在决策时降权
    - 设计动机：利用 LLM 的推理能力做上下文感知的欺诈检测

3. **多维评估指标**:
    - 功能：全面衡量规划系统在欺诈场景下的表现
    - 核心思路：不仅评估规划质量（路线合理性、预算符合度、时间安排可行性），还专门评估**安全性指标**——欺诈采纳率、用户潜在损失、欺诈识别率
    - 设计动机：传统 benchmark 的成功指标无法捕捉安全维度的问题

### 损失函数 / 训练策略

本文为评估性工作，无模型训练。核心贡献在于 benchmark 设计和评估方法论。反欺诈 Agent 基于 prompt engineering 实现。

## 实验关键数据

### 主实验：不同框架在欺诈场景下的表现退化

| 规划框架 | 无欺诈得分 | Misinformation ↓ | Multi-Person ↓ | Multi-Round ↓ |
|----------|-----------|------------------|----------------|---------------|
| CrewAI | 82.3 | 71.5 (-13.1%) | 63.2 (-23.2%) | 55.8 (-32.2%) |
| AutoGPT | 78.6 | 68.9 (-12.3%) | 60.1 (-23.5%) | 51.3 (-34.7%) |
| LangChain Agent | 75.1 | 66.4 (-11.6%) | 58.7 (-21.8%) | 49.2 (-34.5%) |
| + Anti-Fraud Agent | 80.5 | 76.8 (-4.6%) | 70.3 (-12.7%) | 63.5 (-21.1%) |

### 反欺诈 Agent 消融实验

| 配置 | 欺诈识别率 | 规划质量 | 说明 |
|------|-----------|---------|------|
| 完整反欺诈 Agent | 最高 (~78%) | 最高 | 三层防御全开 |
| 无交叉验证 | ~55% | 中等偏低 | 单源验证容易被骗 |
| 无异常检测 | ~48% | 低 | 无法识别多人协调欺诈 |
| 无标记机制 | ~65% | 中等 | 识别但未有效降权 |

### 关键发现

- 所有测试框架在面对欺诈时性能都显著下降，降幅随欺诈复杂度递增（-12% → -23% → -34%）
- 多人协调欺诈最难检测，因为单条评论看起来正常
- Level-Escalating 欺诈导致最大性能退化，因为它能自适应绕过初步防御
- 反欺诈 Agent 有效缓解了约 50-60% 的性能退化，但无法完全消除
- 基于 GPT-4 的框架在欺诈抵抗上略优于 GPT-3.5，但差距不大

## 亮点与洞察

1. **填补了重要空白**：首次系统性评估 LLM 多智能体在对抗性数据环境中的可靠性
2. **递进式欺诈设计精巧**：三种模式从简单到复杂，反映真实世界欺诈的演化路径
3. **揭示"效率-安全"悖论**：高效框架往往更脆弱，因为它们跳过了数据验证步骤
4. **反欺诈 Agent 的可插拔设计**使其易于集成到任何现有框架中
5. **核心洞察**：AI agent 的安全性评估应该和能力评估一样重要

## 局限与展望

- 仅覆盖旅行规划场景，其他领域（医疗预约、金融投资、购物推荐）的适用性需验证
- 欺诈模式是预定义的模板，未考虑 AI 生成的自适应型欺诈（如用 LLM 生成高质量假评论）
- 反欺诈 Agent 基于 prompt 实现，缺乏专门训练，在复杂场景下可能失效
- 作为 Workshop paper 规模受限，实验覆盖的框架和场景数量有限
- 评估中的"欺诈"定义较简单，真实世界的欺诈更微妙和隐蔽

## 相关工作与启发

- **TravelPlanner**：专注规划能力评估，WandaPlan 在此基础上增加了安全维度
- **多智能体安全研究**：如 AgentBench、ToolBench 等关注能力评估，本文关注安全评估
- **LLM 对抗鲁棒性**：与 prompt injection、jailbreak 等研究方向互补——后者攻击模型本身，本文攻击模型的数据输入
- 启发：AI Agent 的安全评估应该成为部署前的标准流程

## 评分

- 新颖性: ⭐⭐⭐⭐ 从安全视角评估多智能体规划系统，角度新颖且重要
- 实验充分度: ⭐⭐⭐ 三种欺诈模式有代表性，但 workshop paper 规模有限
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，场景设计直观
- 价值: ⭐⭐⭐⭐ 对 AI Agent 产品化部署的安全审计有实际参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning](../../ICLR2026/multi_agent/atlas_constraints-aware_multi-agent_collaboration_for_real-world_travel_planning.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/multi_agent/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](../../ACL2026/multi_agent/towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/multi_agent/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)
- [\[ICML 2025\] AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](automl-agent_a_multi-agent_llm_framework_for_full-pipeline_automl.md)

</div>

<!-- RELATED:END -->
