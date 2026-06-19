---
title: >-
  [论文解读] SoMe: A Realistic Benchmark for LLM-based Social Media Agents
description: >-
  [AAAI 2026][LLM Agent][social media agent] 提出首个面向社交媒体智能体的综合性评测基准 SoMe，包含 8 项任务、900 万+真实帖子和 17,869 条标注查询，评估 13 个主流 LLM 的社交媒体代理能力，揭示现有模型在复杂社交任务上仍有较大差距。 LLM 驱动的智能体在社交…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "social media agent"
  - "LLM benchmark"
  - "tool-use"
  - "agent evaluation"
  - "social media analysis"
---

# SoMe: A Realistic Benchmark for LLM-based Social Media Agents

**会议**: AAAI 2026  
**arXiv**: [2512.14720](https://arxiv.org/abs/2512.14720)  
**代码**: [https://github.com/LivXue/SoMe](https://github.com/LivXue/SoMe)  
**领域**: LLM Agent  
**关键词**: social media agent, LLM benchmark, tool-use, agent evaluation, social media analysis

## 一句话总结

提出首个面向社交媒体智能体的综合性评测基准 SoMe，包含 8 项任务、900 万+真实帖子和 17,869 条标注查询，评估 13 个主流 LLM 的社交媒体代理能力，揭示现有模型在复杂社交任务上仍有较大差距。

## 研究背景与动机

LLM 驱动的智能体在社交媒体上的应用日益广泛，包括事件分析、内容推荐、用户行为模拟等。然而，现有评估工作存在显著不足：

1. **单一任务局限**：已有基准（如 BotSim、TrendSim）仅关注单一任务（如用户模拟），无法全面评估智能体能力
2. **数据不足**：现有评估数据规模有限，且缺乏真实的 ground truth（如 TrendSim 仅靠 LLM 评估合理性）
3. **核心矛盾**：社交媒体环境具有高噪声、时间动态性和多样性，对智能体的多轮数据处理和长上下文推理提出极高要求

SoMe 的核心 idea 是构建一个包含真实数据、多任务、工具交互的综合平台，让 LLM 智能体在贴近真实的社交媒体环境中接受全方位评估。

## 方法详解

### 整体框架

SoMe 由三部分组成：(1) 8 项社交媒体任务定义，(2) 基于 MCP 协议的 8 个代理工具平台，(3) 来自 32 个社交平台的 900 万+真实数据。智能体接收任务查询后，通过调用工具获取和分析数据，进行多步推理后输出答案，最终由 LLM 评分器评估结果。

### 关键设计

**设计一：三类八任务的层次化评估体系**

任务被分为三大类：
- **帖子中心任务**：实时事件检测（RED）、流式事件摘要（SES）、虚假信息检测（MID）——需要多轮数据处理和外部知识
- **用户中心任务**：用户行为预测（UBP）、用户情绪分析（UEA）、用户评论模拟（UCS）——需要理解用户偏好和行为模式
- **综合任务**：内容推荐（MCR）、社交媒体问答（SMQ）——需要同时分析大量帖子和用户

这种设计覆盖了从数据分析、用户理解到知识推理的全面能力维度。

**设计二：基于 MCP 的工具交互平台**

提供 8 个工具用于数据获取、管理和分析：

| 工具名称 | 功能描述 |
|---------|---------|
| DataFolder | 从指定文件夹输出数据 |
| SearchPost | 按位置和时间搜索帖子 |
| SearchTopic | 按主题搜索帖子 |
| SearchUser | 搜索特定用户及其帖子 |
| RetrievePost | 在数据文件夹中检索相关帖子 |
| RetrieveKnowledge | 从知识库检索相关报告 |
| PostClustering | 对帖子进行聚类 |
| PostSummarization | 对帖子聚类进行摘要 |

所有工具遵循 MCP 协议，确保与主流 LLM 的兼容性。

**设计三：半自动化标注流水线**

- 对 UBP/UCS/MCR：基于模板自动生成查询和 ground truth，无需人工介入
- 对 RED/SES/UEA/SMQ：采用多轮人-LLM 交互式标注（基于 Qwen3-32B），10 名专业标注员验证
- 对 MID：合并 LIAR-RAW 和 RAWFC 开源数据集

### 数据统计

| 任务 | 查询数 | 数据量 | 数据类型 |
|------|--------|--------|---------|
| RED | 568 | 476,611 | 帖子 |
| SES | 154 | 7,898,959 | 帖子 |
| MID | 1,451 | 27,137 | 帖子&知识 |
| UBP | 3,000 | 840,200 | 帖子&用户 |
| UEA | 2,696 | 840,200 | 帖子&用户 |
| UCS | 4,000 | 840,200 | 帖子&用户 |
| MCR | 4,000 | 840,200 | 帖子&用户 |
| SMQ | 2,000 | 8,651,759 | 帖子&用户 |
| **总计** | **17,869** | **9,242,907** | 全部 |

## 实验关键数据

### 主实验

评估 13 个主流 LLM 在 8 项任务上的表现（得分 0-100）：

| 模型 | 规模 | RED | SES | MID | UBP | UEA | UCS | MCR | SMQ | 平均 |
|------|------|-----|-----|-----|-----|-----|-----|-----|-----|------|
| Gemini-2.5-Flash | N/A | 54.92 | **44.87** | 45.62 | 57.50 | 41.94 | 56.00 | 62.75 | 71.01 | **54.33** |
| GPT-4o | N/A | 47.59 | 36.17 | 50.24 | 55.17 | 31.53 | 52.48 | 61.63 | 64.21 | 49.88 |
| Qwen3-32B | 32B | 44.25 | 41.04 | 47.42 | **67.03** | 33.53 | 54.98 | **63.28** | **80.27** | 53.98 |
| Qwen3-8B | 8B | 40.38 | 36.69 | 45.21 | 61.73 | 33.03 | 53.33 | 60.55 | 76.18 | 50.89 |
| DeepSeek-R1-Qwen3-8B | 8B | 17.71 | 28.83 | 26.46 | 43.53 | 21.18 | 31.10 | 34.33 | 51.84 | 31.87 |
| Llama-3.1-8B | 8B | 3.37 | 20.78 | 40.45 | 34.23 | 37.11 | 33.98 | 47.40 | 31.05 | 31.65 |

### 任务完成率（TCR）分析

| 模型 | 规模 | 平均 TCR |
|------|------|----------|
| DeepSeek-V3 | 671B | 98.60% |
| Qwen3-32B | 32B | 98.45% |
| Gemini-2.5-Flash | N/A | 96.61% |
| Llama-3.1-8B | 8B | 67.20% |
| DeepSeek-R1-Qwen3-8B | 8B | 69.73% |

### 关键发现

- **所有模型表现不理想**：大多数任务得分低于 70，RED/SES/MID 等开放性任务低于 50
- **推理能力≠代理能力**：DeepSeek-R1-Qwen3-8B 虽推理强，但代理任务全面落后 Qwen3-8B（8 项任务分别下降 21%-56%）
- **模型规模效应**：Qwen3-32B > Qwen3-14B (+2.2%) > Qwen3-8B (+3.8%)
- **工具幻觉严重**：DeepSeek-R1 和 Devstral 的工具调用幻觉率分别高达 29% 和 28%
- **工具响应幻觉普遍**：即使 1T 规模的 Kimi-K2 也有 7% 的工具响应幻觉率

## 亮点与洞察

- 首个覆盖 8 项任务、900 万+真实数据的社交媒体智能体综合基准
- 揭示了"推理能力强不等于代理能力强"的重要发现，对未来 LLM 训练有指导意义
- 工具幻觉分析（响应幻觉 vs 调用格式幻觉）为智能体可靠性研究提供了新切入点
- 基于 MCP 协议的工具平台设计确保了广泛兼容性

## 局限与展望

- 评估依赖 LLM 打分器（RED/SES/SMQ），可能引入评分偏差
- 当前仅覆盖英文和中文内容，缺乏多语言评测
- 任务难度分布不均，部分任务（如 SMQ）相对简单，区分度有限
- 未评估多智能体协作场景

## 相关工作与启发

- **vs BotSim**: BotSim 仅评估用户行为模拟，SoMe 覆盖 8 项任务且数据量远超（900万 vs 数万）
- **vs TrendSim**: TrendSim 缺乏 ground truth 仅用 LLM 评估合理性；SoMe 有 10 名标注员验证的真实标注
- **vs OSWorld/WebArena**: 这些基准评估计算机/网页操作，但社交媒体环境更加嘈杂和开放

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个全面社交媒体智能体基准，任务设计和数据规模均达到新高度
- 实验充分度: ⭐⭐⭐⭐ 13 个模型、8 项任务的全面评测，含工具幻觉等深入分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析深入，图表丰富
- 价值: ⭐⭐⭐⭐ 对社交媒体智能体研究有重要推动作用，揭示了推理vs代理能力的关键差异

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[CVPR 2026\] Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection](../../CVPR2026/llm_agent/agent4faceforgery_multi-agent_llm_framework_for_realistic_face_forgery_detection.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](../../ICLR2026/llm_agent/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](../../ICML2026/llm_agent/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ICLR 2026\] A Benchmark for Deep Information Synthesis (DeepSynth)](../../ICLR2026/llm_agent/a_benchmark_for_deep_information_synthesis.md)

</div>

<!-- RELATED:END -->
