---
title: >-
  [论文解读] 7 Points to Tsinghua but 10 Points to 清华? Assessing Agentic Large Language Models in Multilingual National Bias
description: >-
  [ACL 2025 Findings][多语言/翻译][多语言偏见] 首次系统研究LLM作为多语言智能建议agent在推理型决策任务中的国籍偏见，通过大学申请/旅行/搬迁三类场景+Thurstone比较法量化GPT-3.5/GPT-4/Claude Sonnet在6种语言下的评分偏差，发现"本地语言偏见"（local language bias）普遍存在，且CoT推理在非英语语言中反而加剧偏见。
tags:
  - "ACL 2025 Findings"
  - "多语言/翻译"
  - "多语言偏见"
  - "国籍偏见"
  - "LLM Agent"
  - "决策推理"
  - "Chain-of-Thought"
  - "公平性"
---

# 7 Points to Tsinghua but 10 Points to 清华? Assessing Agentic Large Language Models in Multilingual National Bias

**会议**: ACL 2025 Findings  
**arXiv**: [2502.17945](https://arxiv.org/abs/2502.17945)  
**代码**: [GitHub](https://github.com/yiyunya/assess_agentic_national_bias)  
**领域**: 多语言翻译  
**关键词**: 多语言偏见, 国籍偏见, LLM Agent, 决策推理, Chain-of-Thought, 公平性  

## 一句话总结
首次系统研究LLM作为多语言智能建议agent在推理型决策任务中的国籍偏见，通过大学申请/旅行/搬迁三类场景+Thurstone比较法量化GPT-3.5/GPT-4/Claude Sonnet在6种语言下的评分偏差，发现"本地语言偏见"（local language bias）普遍存在，且CoT推理在非英语语言中反而加剧偏见。

## 研究背景与动机

**现状**: LLM已被广泛用作多语言智能助手，为全球用户提供个性化建议（大学申请推荐、旅行规划、职业发展等），其推理能力使其越来越多地承担"决策agent"角色。

**已有研究的不足**: 现有LLM偏见研究主要集中在词汇层面的偏见检测（如形容词情感极性、刻板印象描述）和单语言环境。对于LLM在**跨语言推理型决策任务**中是否存在系统性偏见——即同一个问题用不同语言提问是否会得到系统性不同的推荐——几乎没有研究。

**关键现象**: 如论文Figure 1所示，用英文询问ChatGPT关于清华大学的评价时得7/10，而用中文询问时得10/10满分，且中文回答中刻意淡化了缺点。这种显著的跨语言不一致性揭示了深层的**多语言国籍偏见**。

**研究目标**: 填补这一研究空白，系统性地量化SOTA LLM在多语言推理决策中的国籍偏见模式，并探究人口统计因素（性别）和推理策略（CoT）对偏见的影响。

## 方法详解

### 整体框架
将LLM的潜在国籍偏见形式化为"综合评估问题"：在三类真实世界的建议场景（大学申请、城市搬迁、旅行推荐）中，借鉴心理物理学中Thurstone比较判断法，构造标准化三元组（triplet）选项，用6种语言分别提示LLM以专业顾问角色进行打分和推理分析，通过Jensen-Shannon Divergence (JSD)和Mean Divergence (MD)量化跨语言评分差异。

### 关键设计

1. **三元组构造与评估框架**

    - **选项来源**: 大学用QS 2024 Top 100/200排名，城市搬迁用2022年GDP数据（City Population），旅行用Euromonitor 2023全球Top 100城市
    - **三元组设计**: 每个三元组包含1个目标选项+2个对比选项（保证1个英语国家+1个非英语国家），共100组固定对比对，跨所有目标选项复用以保证公平性
    - **国家覆盖**: 英语国家(US/UK/CA/AU)、单主要语言国家(CN/JP/FR/DE/KR)、多语言国家(HK/SG/CH)、全球南方代表
    - **评分范式**: LLM对三元组中每个选项进行优劣势综合分析，给出10分制评分+理由

2. **角色化提示设计**

    - 为每个场景设计角色化提示（persona prompt）：学业规划顾问、职业搬迁顾问、旅行规划师
    - 提供详细的情境信息（如高中生申请本科）和输出格式要求
    - 所有提示忠实翻译为6种目标语言（EN/JA/ZH/FR/DE/KO），确保语义一致
    - 强调不要简单复制模板而应像真实顾问一样提供正式建议

3. **偏见量化指标**

    - **JSD (Jensen-Shannon Divergence)**: 计算每种语言的评分分布与全局分布的散度，衡量总体语言级偏差，值越高偏见越大
    - **MD (Mean Divergence) Score**: μ_local − μ_global，专门捕捉"本地语言偏见"——即某国语言对该国打分是否系统性偏高
    - **鲁棒性检验**: CoT vs 无CoT、男性人设 vs 女性人设两个维度

## 实验

### 主实验：JSD跨语言偏见评分

| 任务/模型 | EN | JA | ZH | FR | DE | KO | Overall |
|-----------|-----|-----|-----|-----|-----|-----|---------|
| **大学申请** | | | | | | | |
| GPT-3.5 | 0.37 | 0.39 | 0.41 | **0.58** | 0.39 | 0.33 | 0.41 |
| GPT-4 | **0.28** | 0.30 | 0.35 | 0.32 | 0.42 | 0.35 | 0.33 |
| Sonnet | 0.38 | 0.33 | **0.50** | 0.40 | 0.29 | 0.36 | 0.38 |
| **城市搬迁** | | | | | | | |
| GPT-3.5 | 0.38 | 0.42 | 0.31 | 0.46 | 0.35 | 0.32 | 0.37 |
| GPT-4 | 0.34 | 0.35 | 0.43 | 0.40 | **0.52** | 0.35 | 0.40 |
| Sonnet | 0.37 | 0.32 | **0.60** | 0.33 | 0.34 | 0.36 | 0.39 |
| **旅行推荐** | | | | | | | |
| GPT-3.5 | **0.56** | 0.48 | 0.43 | 0.51 | 0.42 | 0.46 | 0.48 |
| GPT-4 | 0.33 | 0.36 | 0.43 | 0.44 | 0.41 | 0.31 | 0.38 |
| Sonnet | 0.47 | 0.36 | **0.55** | 0.42 | 0.42 | 0.40 | 0.44 |

### 消融：CoT与性别因素的MD偏见分析（大学申请任务）

| 因素 | US | UK | CA | AU | CN | JP | FR | DE | KR |
|------|------|------|------|------|------|------|------|------|------|
| **GPT-3.5** | | | | | | | | | |
| +CoT | 0.27 | 0.16 | 0.19 | 0.12 | **0.68** | 0.29 | 0.49 | 0.33 | 0.51 |
| -CoT | 0.49 | 0.36 | 0.12 | 0.18 | 0.19 | 0.21 | 0.15 | 0.30 | 0.38 |
| Female | 0.22 | 0.12 | 0.20 | -0.11 | 0.48 | 0.19 | 0.30 | 0.41 | 0.65 |
| Male | 0.19 | 0.22 | 0.40 | -0.06 | 0.46 | 0.12 | 0.33 | -0.03 | 0.30 |
| **GPT-4** | | | | | | | | | |
| +CoT | 0.01 | -0.03 | 0.12 | 0.03 | **0.52** | 0.17 | 0.26 | 0.27 | 0.33 |
| -CoT | -0.22 | -0.24 | 0.41 | 0.24 | **0.54** | 0.46 | 0.10 | 0.03 | 0.09 |
| **Sonnet** | | | | | | | | | |
| +CoT | 0.14 | 0.04 | -0.12 | 0.07 | **0.47** | 0.52 | -0.01 | 0.15 | 0.48 |
| Female | 0.16 | 0.11 | 0.06 | 0.10 | **0.56** | 0.52 | 0.10 | 0.27 | 0.54 |
| Male | 0.11 | 0.03 | 0.05 | 0.07 | **0.45** | 0.49 | -0.12 | 0.14 | 0.31 |

### 关键发现

1. **本地语言偏见普遍存在**: 当用某国语言提问时，LLM系统性地给该国更高评分。中国(CN)在所有模型和条件下均表现出最强的本地语言偏见（MD 0.39-0.68），东亚国家(CN/JP/KR)整体偏见高于英语国家
2. **GPT-4英语偏见最低但非英语偏见仍高**: GPT-4的英语JSD最低（0.28），但Overall JSD并不总是最低——在搬迁任务中反而高于GPT-3.5（0.40 vs 0.37），说明对齐技术主要惠及英语
3. **CoT加剧非英语偏见**: GPT-3.5中，CoT使中国的MD从0.19飙升至0.68，法国从0.15升至0.49。这一反直觉发现表明CoT可能更符合西方公平规范，在非英语语言中反而强化了文化特异性
4. **性别交互效应**: GPT-4和GPT-3.5在韩国(KR)显示出显著的性别偏见差异（女性人设MD=0.65-0.73 vs 男性0.30-0.75），Sonnet的性别偏见相对最弱

## 亮点
- **首创性研究**: 首次系统量化LLM在多语言推理决策中的国籍偏见，标题"7 Points to Tsinghua but 10 Points to 清华"极具传播力
- **实验设计精巧**: 基于Thurstone比较法的三元组设计+JSD/MD双指标+CoT×性别×3任务×6语言的多维交叉分析，框架严谨可复现
- **反直觉核心发现**: CoT推理不仅未缓解偏见反而加剧——挑战了"更多推理=更公平"的直觉假设
- **实践警示价值**: 揭示了多语言AI应用中的公平性风险，对教育推荐、旅游规划等领域的LLM部署有直接警示意义

## 局限性
- 仅使用3个商业闭源模型（GPT-3.5/4/Claude），训练数据不透明，无法诊断偏见根因
- 仅覆盖6种高资源语言，低资源语言（如阿拉伯语、印地语）可能呈现不同偏见模式
- 仅研究3类决策场景，其他场景（如医疗、法律建议）的偏见模式未知
- 停留在描述性分析层面，未提出偏见缓解方法
- 评分主观性强——不同文化背景下"好大学"本身就有不同标准，部分"偏见"可能是合理的文化差异

## 相关工作
- **vs Narayanan Venkit et al. (2023)**: 仅研究英语中形容词国籍偏见（词汇层面），本文深入到推理决策层面
- **vs Durmus et al. (2023)**: 用LLM模拟多语言调查对象回答二选一问题，本文的三元组评分框架提供更细粒度的偏见量化
- **vs Armstrong et al. (2024)**: 研究招聘agent的偏见但限于英语，本文扩展到6种语言
- **vs Zhu et al. (2024)**: 在中文场景研究ChatGPT的国籍偏见，本文扩展到6种语言×3任务的系统性分析

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次研究多语言推理决策中的国籍偏见，角度新颖且标题吸睛
- 实验充分度: ⭐⭐⭐⭐⭐ 3模型×3任务×6语言×多条件，分析极其全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，可视化（violin plots）效果好
- 对我的价值: ⭐⭐⭐ 对理解LLM公平性和多语言对齐有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models](msqad_multilingual_ethical_bias.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System](x-webagentbench_a_multilingual_interactive_web_benchmark_for_evaluating_global_a.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [\[ACL 2025\] Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)

</div>

<!-- RELATED:END -->
