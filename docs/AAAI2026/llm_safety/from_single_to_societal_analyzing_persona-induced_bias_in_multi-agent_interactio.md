---
title: >-
  [论文解读] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions
description: >-
  [AAAI 2026][LLM安全][Multi-Agent System] 本文首次系统研究了 LLM 多智能体交互中的人格诱导偏见，通过在协作问题解决和说服任务中的受控实验，揭示了三个关键发现：(1) 不同人格在可信度和坚持度上存在显著偏差（优势群体如男性和白人被视为更不可信）；(2) 智能体表现出显著的内群体偏好；(3) 这些偏见在多轮、多智能体场景中持续存在且有放大趋势。
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "Multi-Agent System"
  - "Persona Bias"
  - "LLM 偏见"
  - "内群体偏好"
  - "社会身份理论"
---

# From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions

**会议**: AAAI 2026  
**arXiv**: [2511.11789](https://arxiv.org/abs/2511.11789)  
**代码**: [https://github.com/Jiayi-LizzZ/Persona-Induced-Bias-in-MAS](https://github.com/Jiayi-LizzZ/Persona-Induced-Bias-in-MAS)  
**领域**: AI Safety  
**关键词**: Multi-Agent System, Persona Bias, LLM 偏见, 内群体偏好, 社会身份理论

## 一句话总结
本文首次系统研究了 LLM 多智能体交互中的人格诱导偏见，通过在协作问题解决和说服任务中的受控实验，揭示了三个关键发现：(1) 不同人格在可信度和坚持度上存在显著偏差（优势群体如男性和白人被视为更不可信）；(2) 智能体表现出显著的内群体偏好；(3) 这些偏见在多轮、多智能体场景中持续存在且有放大趋势。

## 研究背景与动机
基于 LLM 的多智能体系统正被广泛用于模拟人类交互和解决协作任务。一个常见做法是为每个智能体分配不同的人格（persona，如人口统计信息、性格特征等），以鼓励行为多样性。

**现有痛点**：已有研究表明，为单个 LLM 分配不同人格会显著影响其问题解决表现（如"黑人不擅长数学"的刻板印象）。然而，人格诱导的偏见是否也存在于多智能体交互中？这个关键问题几乎未被探索。

**核心矛盾**：人格丰富了智能体行为，但也可能引入偏见。现有研究要么聚焦单智能体的偏见，要么关注默认智能体间的话语偏见，没有研究人格如何在交互中塑造社会行为（如信任和坚持）。

**切入角度**：设计渐进式受控实验——从二元交互中的单人格效应，到人格对交互，再到多智能体多轮场景——系统揭示人格诱导偏见的深度和广度。

**核心问题**：人格分配是否以及如何影响多智能体交互中的社会行为特征（可信度和坚持度）？

## 方法详解

### 整体框架
三阶段渐进式分析：(1) 隔离单个人格在双方交互中的影响 → (2) 分析人格对的交互动力学 → (3) 验证在多智能体多轮复杂场景中的泛化性。

### 关键设计

1. **单人格效应分析（§4）**:

    - 功能：隔离单个人格对可信度（trustworthiness）和坚持度（insistence）的影响
    - 核心思路：设计受控的双智能体单轮交互，一个分配人格 A_p，另一个为默认智能体 A_d
        - 可信度 T(p)：A_d 服从 A_p 的概率（A_p 作为说服者时）
        - 坚持度 I(p)：A_p 不服从 A_d 的概率（A_p 作为被说服者时）
    - 设计动机：隔离单一变量（人格），消除交叉交互的混淆因素
    - 量化指标：Max-min difference Δ_{max-min}（极差）和 Average absolute difference Δ_{avg}（与无人格基线的平均偏差）

2. **人格对交互分析（§5）**:

    - 功能：两个均有人格的智能体交互，观察服从率 C(p1→p2)
    - 核心思路：分析个体偏差如何在对交互中累积或抵消
    - 设计动机：现实中所有智能体通常都有人格，需要理解对效应
    - 可视化：热力图展示所有人格对的服从率，横轴按可信度递减，纵轴按坚持度递增

3. **复杂场景泛化验证（§6）**:

    - 功能：扩展到多智能体（2→6个）和多轮交互（1→5轮）
    - 核心思路：在 CPS 任务中度量 Win Rate（某人格初始答案成为共识的概率），在说服任务中度量 Persuasion Effectiveness（说服成功率）
    - 设计动机：验证二元交互中观察到的偏见是否在更复杂社会动态中持续存在

### 实验控制措施
- **预生成初始回复**：确保人格是影响结果的唯一变量
- **平衡冲突设计**（CPS）：偶数智能体分两组，一组正确答案+人格p1，一组错误答案+人格p2
- **反向对称**：聚合正反初始分配的结果，消除方向偏差
- **排除人格相关声明**（说服任务）：用 GPT-4o 移除涉及性别/种族的声明
- **温度设为 0**：确保结果稳定性

### 任务设置
- **协作问题解决 (CPS)**：GPQA 数据集的研究生级别多选题（筛选后 455 题）
- **说服 (Persuasion)**：PMIYC 框架的主观声明（筛选后 854 条）
- **人格集**：P_{gender} = {woman, man, trans woman, trans man, non-binary}; P_{race} = {White, Black, Asian, Hispanic}
- **模型**：GPT-4o, Gemini-1.5-Pro, DeepSeek-V3

## 实验关键数据

### 主实验：单人格效应（可信度和坚持度变异）

| 任务 | 模型 | 性别 Δ_{max-min} | 种族 Δ_{max-min} |
|------|------|------------------|------------------|
| CPS | GPT-4o | 1.30% / 2.10% | 4.95% / 1.85% |
| CPS | Gemini-1.5-Pro | 10.80% / 4.85% | 8.45% / 6.40% |
| 说服 | GPT-4o | 5.40% / 5.70% | 12.30% / 6.85% |
| 说服 | Gemini-1.5-Pro | 4.80% / 9.60% | 12.90% / 5.65% |

*（每格为"可信度变异 / 坚持度变异"）*

### 消融/深入分析

| 现象 | 数据 | 说明 |
|------|------|------|
| 优势群体更不受信任 | White 可信度 66.4%（GPT-4o 说服任务），比其他种族至少低 9.9% | 与社会学中精英面临不信任增长的发现一致 |
| 优势群体更易服从 | men 平均服从率 60.7%（vs 所有性别平均 56.3%）| 与"资源缓冲"理论一致 |
| 内群体偏好 | 全部配置平均：同人格服从率明显高于异人格 | CPS中Deepseek: 全部56.9% vs 同人格62.6%（种族） |
| 多轮放大效应 | Black→White 说服率比 White→Black 高近 10%（5轮后） | Deepseek-V3 差距从 20%→24% |
| 多人放大效应 | Gemini: PE(Black→White)与PE(White→Black)差距从10%增至27% | 随说服者人数增加 |

### 关键发现
- **偏见普遍存在**：平均可信度变异 5.3%，坚持度变异 4.3%，仅改变人格标签就能一致改变行为
- **模型差异**：Gemini-1.5-Pro 偏见最严重（CPS 性别可信度极差 10.8%），但所有模型在说服任务中差异均统计显著（α=0.01）
- **反直觉发现**：男性和白人被视为更不可信——这挑战了"男性更固执"的刻板印象，但与社会科学中关于优势群体面临不信任和倾向信任他人的发现一致
- **人格准确率差异很小**：GPQA 上不同人格的准确率极差仅 2.2%（Gemini 性别），说明偏见来源于交互而非能力差异
- **woman 人格的答案更容易成为共识**：5轮后，woman 答案被采纳的概率比 man 平均高 8%

## 亮点与洞察
- **渐进式实验设计**：从隔离分析到组合效应到泛化验证，方法论上非常严谨
- **与社会心理学理论的对应**：内群体偏好对应社会身份理论、优势群体的低信任对应精英不信任理论、高服从率对应"资源缓冲"理论
- **控制变量做得最彻底**：预生成初始回复、平衡冲突设计、反向对称——确保人格是唯一变量
- **揭示了交互放大效应**：个体偏见在交互中累积和放大，多轮多人场景使差距进一步扩大

## 局限与展望
- 仅测试了性别和种族两个维度，未涵盖年龄、职业、教育水平等人格
- 人格通过简单系统提示分配（"You are [persona]"），更复杂的人格设定可能产生不同效果
- 温度设为 0 确保了稳定性但也限制了对随机性的分析
- 仅聚焦可信度和坚持度两个社会特征，其他如领导力、创造力等未探索
- 偏见缓解策略仅作未来工作提及，未提供具体方案
- 成本约束限制了多智能体实验的规模

## 相关工作与启发
- **Gupta et al. (2024)**：揭示单智能体中的人格推理偏见，本文将其扩展到交互场景
- **Borah & Mihalcea (2024)**：发现 LLM 输出中的偏见可通过智能体交互被强化，但聚焦默认智能体
- **Ashery et al. (2025)**：即使个体无偏，集体偏见仍可能涌现——与本文发现互补
- **社会身份理论 (Hogg 2016)**：完美解释了内群体偏好现象
- **启发**：任何使用人格分配的多智能体系统都应进行偏见审计，并考虑去偏策略（如匿名化交互或对抗性平衡设计）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（首次系统研究多智能体交互中的人格偏见，揭示了新的行为现象）
- 实验充分度: ⭐⭐⭐⭐⭐（3模型×2任务×5性别×4种族，渐进式实验设计极其严谨）
- 写作质量: ⭐⭐⭐⭐⭐（结构清晰，研究路线图明确，社会学理论连接恰当）
- 价值: ⭐⭐⭐⭐⭐（对多智能体系统的公平性和可靠性有重要警示意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A2ASecBench: A Protocol-Aware Security Benchmark for Agent-to-Agent Multi-Agent Systems](../../ICLR2026/llm_safety/a2asecbench_a_protocol-aware_security_benchmark_for_agent-to-agent_multi-agent_s.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](../../ACL2026/llm_safety/privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](../../ACL2026/llm_safety/memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ACL 2026\] Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage](../../ACL2026/llm_safety/lying_with_truths_open-channel_multi-agent_collusion_for_belief_manipulation_via.md)

</div>

<!-- RELATED:END -->
