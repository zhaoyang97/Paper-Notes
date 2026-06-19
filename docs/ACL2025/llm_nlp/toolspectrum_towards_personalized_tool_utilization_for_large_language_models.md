---
title: >-
  [论文解读] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models
description: >-
  [ACL 2025][LLM 其他][个性化工具使用] 提出ToolSpectrum基准，首次定义并评估LLM的个性化工具使用能力——根据用户画像和环境因素选择最合适的工具，实验表明个性化显著提升用户体验，但现有LLM在联合推理用户和环境因素时能力有限。 领域现状：将外部工具集成到LLM中已成为增强其能力的主流方案…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "个性化工具使用"
  - "大语言模型"
  - "用户画像"
  - "环境因素"
  - "基准测试"
---

# ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2505.13176](https://arxiv.org/abs/2505.13176)  
**代码**: [https://github.com/BUAA-IRIP-LLM/ToolSpectrum](https://github.com/BUAA-IRIP-LLM/ToolSpectrum)  
**领域**: 信号通信  
**关键词**: 个性化工具使用, 大语言模型, 用户画像, 环境因素, 基准测试

## 一句话总结

提出ToolSpectrum基准，首次定义并评估LLM的个性化工具使用能力——根据用户画像和环境因素选择最合适的工具，实验表明个性化显著提升用户体验，但现有LLM在联合推理用户和环境因素时能力有限。

## 研究背景与动机

**领域现状**：将外部工具集成到LLM中已成为增强其能力的主流方案，在旅行规划、在线购物、知识获取等领域取得了显著进展。现有工具学习基准（如ToolBench、API-Bank、AppBench等）主要评估LLM的工具选择和执行能力。

**现有痛点**：现有方法只关注功能性工具选择——即选择能完成用户指令的工具，忽视了具有重叠功能的工具之间的个性化选择。现实中，多个工具可能都能完成同一任务（如Amazon和Temu都能购物），但根据用户预算偏好、年龄限制、天气条件等因素，最优选择可能截然不同。

**核心矛盾**：现有基准将工具选择视为纯功能匹配问题，但真正的用户满意度需要LLM理解"谁在什么情况下需要什么"——这需要同时推理用户画像(Profile)和环境因素(Environment)，现有方法无法捕捉这种上下文敏感的个性化需求。

**本文目标** (1) 定义个性化工具使用任务，(2) 构建覆盖用户画像和环境因素的评测基准，(3) 评估现有LLM的个性化工具使用能力。

**切入角度**：作者将个性化工具使用形式化为$t = \text{Model}(I, \mathcal{P}, \mathcal{E}, \mathcal{T})$，其中$\mathcal{P}$为用户画像、$\mathcal{E}$为环境因素、$\mathcal{T}$为工具集，输出包含APP选择、API调用、必需参数和个性化可选参数。

**核心 idea**：首次将个性化推荐思想引入工具学习领域，构建了同时考虑用户画像和环境因素的工具使用基准。

## 方法详解

### 整体框架

ToolSpectrum的构建分为四个阶段：(1) 工具集收集——从9个常用应用领域收集具有重叠功能的APP和API；(2) 用户画像和环境因素定义与收集——定义人口统计、性格、偏好三类用户属性和自然环境、数字环境、应用政策三类环境因素；(3) 工具调用结果收集——模拟用户指令和个性化调用结果；(4) 质量评估——多维度评分和人工验证。最终数据集包含Profile(450条)、Environment(220条)、Profile&Environment(330条)三种类型。

### 关键设计

1. **用户画像定义(Profile)**:

    - 功能：建模影响工具选择的用户个体因素
    - 核心思路：分为三大类——Demographics（性别、年龄、体重、身高、职业、教育、收入等key-value对）、Personality（兴趣爱好的自然语言描述）、Preference（历史应用使用偏好）。例如身高体重影响购衣尺码选择，收入影响价格敏感度，运动爱好影响健康APP选择
    - 设计动机：借鉴个性化推荐系统中用户建模的经验，但将其扩展到工具使用场景

2. **环境因素定义(Environment)**:

    - 功能：建模影响工具选择的外部上下文
    - 核心思路：分为Natural Environment（天气、日期、时间、地点等key-value对）、Digital Environment（网络状况、设备配置等）、App Domain Policy（应用特定的政策规则，如未成年人消费限制）。例如暴风雨天气应推荐火车票而非飞机票，弱网环境应降低图片质量
    - 设计动机：以往个性化研究大多只关注用户画像，忽略了环境约束对工具选择的重要影响

3. **工具调用结果的标准化输出**:

    - 功能：统一评估个性化工具使用的各个维度
    - 核心思路：输出结构化为字典$\{APP \mapsto a, API \mapsto s, RP \mapsto r, OP \mapsto o\}$，包含APP选择、API选择、必需参数提取和个性化可选参数填充四个层次。如果用户指令在考虑画像和环境后违反应用政策，应返回None
    - 设计动机：将评估分解为多个粒度，可以精准定位LLM在个性化推理中的薄弱环节

### 损失函数 / 训练策略

本文是评测基准工作，不涉及训练。数据构建使用GPT-4o生成初始数据，经过自动评分（三个维度，1-10分，低于8分的丢弃，移除21.8%数据）和人工抽样验证（50/domain，平均得分8.7）。

## 实验关键数据

### 主实验

| 模型 | Profile APP | Profile RP | Profile PP | Env APP | Env OP | Both APP | Both OP |
|------|------------|------------|------------|---------|--------|----------|---------|
| Qwen2.5-7B | 0.73 | 0.59 | 0.27 | 0.66 | 0.03 | 0.22 | 0.06 |
| Qwen2.5-32B | 0.74 | 0.67 | 0.47 | 0.77 | 0.14 | 0.24 | 0.15 |
| GPT-4o | **最优** | **最优** | **最优** | **最优** | **最优** | **最优** | **最优** |
| Qwen2.5-3B | 0.16 | 0.12 | 0.12 | 0.55 | 0.00 | 0.12 | 0.04 |

### 消融实验

| 条件 | 关键观察 | 说明 |
|------|---------|------|
| Profile Only | APP/API准确率相对较高 | 模型能基本识别用户偏好 |
| Environment Only | OP(可选参数)普遍极低 | 环境信息利用能力弱 |
| Profile & Environment | 全面下降 | 联合推理最具挑战 |
| 小模型(3B) | 各指标极低 | 个性化推理需要足够的模型容量 |
| 大模型(32B+) | PP/OP提升明显 | 模型规模对个性化参数生成至关重要 |

### 关键发现

- **个性化显著提升效果**：整合个性化因素后，工具使用的有效性明显提高，证实了个性化在工具学习中的重要性
- **联合推理是最大挑战**：即使是SOTA模型，在同时考虑Profile和Environment时性能也大幅下降，模型倾向于优先考虑其中一个维度而忽略另一个
- **可选参数(OP)是瓶颈**：所有模型在个性化/环境相关的可选参数生成上表现极差（大多≤0.15），说明模型还不能有效将上下文信息转化为具体的参数设置
- **模型规模效应显著**：3B模型几乎无法完成个性化工具使用，7-32B有明显提升，但即使最大的开源模型也远未解决问题

## 亮点与洞察

- **问题定义的贡献大于方法**：首次将个性化引入工具学习，定义了一个此前被忽略但实际极为重要的问题空间，这比任何具体方法创新都更有长期价值
- **评估粒度设计巧妙**：将评估分解为APP→API→RP→OP四个层次，可以精确诊断模型在个性化推理链条中的断点
- **应用政策违规检测**：引入"返回None"的机制来测试模型是否能识别出违法政策的操作（如未成年人高额消费），这在实际部署中至关重要

## 局限与展望

- **数据规模有限**：总共仅1000条数据，每个领域约100条，可能不够充分评估模型能力
- **场景覆盖不完整**：9个领域虽然常见，但缺少如教育、生产力工具等重要应用领域
- **GPT-4o构建偏差**：数据和标注均由GPT-4o生成，可能引入该模型的偏见；人工验证仅抽样50条/领域
- **缺少改进方法**：仅诊断了问题，没有提出提升LLM个性化工具使用能力的方法，如RAG增强、prompt工程或微调策略
- **多模型合作场景未考虑**：实际场景中可能需要多步工具调用和工具间交互，当前基准仅评估单步调用

## 相关工作与启发

- **vs ToolBench / API-Bank**：这些基准评估通用工具使用能力，本文聚焦于功能重叠工具间的个性化选择，是工具学习的新维度
- **vs τ-Bench**：τ-Bench考虑了Environment但不包含Profile，且不是专门评估个性化，本文同时覆盖两个维度
- **vs LaMP / PersonaChat**：这些个性化基准关注对话/文本生成，不涉及工具使用，本文将个性化概念拓展到工具调用场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义个性化工具使用问题，方向新颖且实际需求明确
- 实验充分度: ⭐⭐⭐⭐ 评测了大量开源和闭源模型，分析维度丰富
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，形式化完整，图表直观
- 价值: ⭐⭐⭐⭐ 为LLM Agent的个性化发展指明方向，基准本身有较高复用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)
- [\[ACL 2025\] LLMs + Persona-Plug = Personalized LLMs](llms_persona-plug_personalized_llms.md)

</div>

<!-- RELATED:END -->
