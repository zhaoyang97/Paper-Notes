---
title: >-
  [论文解读] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States
description: >-
  [ACL2025][LLM评测][心智理论] 提出 DynToM 基准，通过 1,100 个社会情境中 5,500 个时序关联场景和 78,100 道题目，评估 LLM 追踪人类心理状态时序演化的能力，揭示模型平均落后人类 44.7%。 1. 心智理论是社会交互基础：Theory of Mind (ToM) 是理解他人信念、…
tags:
  - "ACL2025"
  - "LLM评测"
  - "心智理论"
  - "动态推理"
  - "心理状态追踪"
  - "社会认知"
  - "LLM评估"
---

# Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States

**会议**: ACL2025  
**arXiv**: [2505.17663](https://arxiv.org/abs/2505.17663)  
**代码**: [GitHub & HuggingFace](https://github.com/)  
**领域**: LLM评测  
**关键词**: 心智理论, 动态推理, 心理状态追踪, 社会认知, LLM评估

## 一句话总结
提出 DynToM 基准，通过 1,100 个社会情境中 5,500 个时序关联场景和 78,100 道题目，评估 LLM 追踪人类心理状态时序演化的能力，揭示模型平均落后人类 44.7%。

## 背景与动机
1. **心智理论是社会交互基础**：Theory of Mind (ToM) 是理解他人信念、情感、意图的核心能力，随着 LLM 越来越多地参与人机交互（如心理支持对话），评估其 ToM 能力变得至关重要。
2. **现有评估停留在静态快照**：SocialIQA、BigToM、TOMBENCH 等基准主要评估孤立场景中的静态心理状态，忽视了真实社交中心理状态**随时间持续演化**的关键特征。
3. **缺乏时间维度的 ToM 评估**：没有基准系统地捕捉心理状态在多个连续场景之间的变化——例如用户的信念如何因一系列对话事件而逐步转变。
4. **心理状态间存在复杂因果关系**：心理学研究表明信念影响情感，信念和情感共同影响意图，三者共同驱动行为——需要一个框架来建模这种互依关系的时序演化。
5. **组合推理是 LLM 的短板**：追踪多场景心理状态变化需要组合推理（compositional reasoning），已有研究表明 Transformer 在组合复杂度增长时性能显著下降。
6. **实际应用中动态理解不可或缺**：心理咨询 AI、情感陪伴系统、社交模拟等应用都需要模型理解用户心理状态如何随交互推进而变化，静态评估无法衡量此能力。

## 方法详解

### 四步构建框架

**Step 1: 社会情境构建**
- 收集 261 个社会地点（跨 13 个类别，如工作场所、教育场所等）
- 从美国人口普查数据池中采样角色属性：姓名、性别、职业、教育、种族、人格特质（7 个维度）
- 每个情境包含 1 个地点 + 2 个角色 + 角色关系
- GPT-4-Turbo 基于 4 个示例生成角色关系，4 名标注员验证，保留率 92%

**Step 2: 心理状态轨迹设计**
- 追踪 4 种心理状态：信念 (belief)、情感 (emotion)、意图 (intention)、行为 (action)
- 基于心理学理论 (D'Andrade, 1995) 建模三种因果关系：信念→情感；信念+情感→意图；信念+情感+意图→行为
- 每个情境设计 5 个时序场景的心理状态演化轨迹，包含触发状态转变的具体线索 (cues)
- 4 名标注员在连贯性、合理性、真实性三维度上以 5 分制评分，低于 4.0 的淘汰，保留率 85.4%

**Step 3: 场景生成**
- 基于 Step 2 的轨迹，GPT-4-Turbo 为每个场景生成背景描述和自然对话
- 对话形式自然展现角色心理状态变化
- 4 名标注员评估一致性、连贯性、真实性，低于 4.0 的重新生成，保留率 88.7%

### 四类问题设计

| 问题类型 | 评估目标 | 难度 |
|---------|---------|------|
| **Understanding** | 识别单个场景中特定心理状态 | 基础 |
| **Transformation-1** | 检测相邻场景间状态是否发生变化 | 中等 |
| **Transformation-2** | 理解状态变化背后的因果机制 | 较难 |
| **Transformation-3** | 追踪全部场景的状态演化序列 | 最难 |

- 选项设计利用轨迹信息：正确答案来自目标状态，干扰项来自同场景的其他状态或其他场景的同类状态
- 最终经标注员在清晰度和可答性维度验证后，共收集 **78,100** 道多选题

### 数据集规模
- 1,100 个社会情境，2,200 个角色，261 个地点
- 5,500 个场景，平均场景长度 457.9 词
- 78,100 道题：Understanding 28.2%, Transformation-1 22.5%, Transformation-2 43.7%, Transformation-3 5.6%

## 实验关键数据

### 表1：LLM 在 DynToM 上的表现（准确率 %）

| 模型 | Understanding 均值 | Transformation 均值 | 总均值 |
|------|-------------------|-------------------|--------|
| **人类基线** | 82.3 | 76.6 | **77.7** |
| GPT-4o | 88.8 | 49.5 | **64.0** |
| Llama-3.1-70B | 83.6 | 42.5 | **57.1** |
| Qwen2-72B | 81.7 | 32.3 | **48.5** |
| GPT-4-Turbo | 72.5 | 34.5 | **47.6** |
| Llama-3.1-8B | 30.2 | 17.5 | **22.3** |
| DeepSeek-V2 | 4.5 | 7.6 | **7.2** |

**关键发现**：LLM 平均表现仅 33.0%，落后人类 44.7%；Understanding 到 Transformation 的性能断崖式下降（平均 48.2%→24.7%），说明模型能识别静态状态但难以追踪动态变化。

### 表2：GPT-4o 错误类型分析

| 错误类型 | 比例 | 含义 |
|---------|------|------|
| Full error | 50-58% | 状态识别和变化推理均失败 |
| Local error | 13-18% | 状态识别正确但变化推理失败 |
| Restoration error | 8-16% | 变化推理正确但状态识别失败（表面模式匹配） |
| Fully correct | 13-17% | 均正确 |

**关键发现**：Full error 占主导，信念状态错误率最高（58%）；Restoration error 的存在暗示模型依赖浅层模式匹配而非真正理解。

### "Lost in the Middle" 现象
- 模型在中间场景（span 2-3, 3-4）表现最差，呈 U 型曲线
- 7 个场景序列中 span 3-4 准确率仅 26%
- 截断后面场景后中间准确率提升 21 个百分点（26%→47%）

## 亮点
1. **首个动态 ToM 基准**：从静态快照评估跨越到时序演化评估，填补了心理状态动态追踪评估的空白
2. **系统化四步构建框架**：社会情境→心理轨迹→场景生成→问题设计，每步都有严格的人工验证
3. **78,100 道题的大规模**：远超现有 ToM 基准（TOMBENCH 2,860 题、BigToM 600 题），统计功效更强
4. **渐进式问题设计精妙**：Understanding → Transformation-1/2/3 逐步升级，能精确定位模型在推理链中的断裂点
5. **发现 "Lost in the Middle" 在 ToM 中的表现**：首次在社会认知任务中验证了该现象，为长上下文 ToM 推理研究提供重要证据

## 局限与展望
1. **模型覆盖有限**：仅评估 10 个模型，缺少 Claude 家族和最新开源模型（如 Qwen2.5、DeepSeek-V3）
2. **提示方法单一**：仅使用 vanilla 和 CoT，未尝试 Think-Twice、Self-Consistency 等可能更适合 ToM 的方法
3. **心理状态类型可扩展**：仅覆盖信念/情感/意图/行为，未包含知识 (knowledge)、欲望 (desire) 等认知维度
4. **场景数量固定为 5**：虽然论文提到可调整，但未探索 5 以外长度对性能的系统性影响（仅有初步的 6/7 场景实验）
5. **仅限文本模态**：真实社交中心理状态还通过表情、语调等多模态信号传递，当前基准未涵盖
6. **依赖 GPT-4-Turbo 生成数据**：可能引入生成模型自身的偏差，部分场景的自然度受限

## 与相关工作的对比

### vs. TOMBENCH (Chen et al., 2024)
TOMBENCH 包含 2,860 道题评估基本 ToM 能力，但**仅测试孤立场景的静态心理状态**。DynToM 通过 5 个时序关联场景形成连续故事线，评估心理状态的**动态演化和因果链条**，更贴近真实社交交互。TOMBENCH 上多数 LLM 已接近完美表现，而 DynToM 上最佳模型仅达 64%。

### vs. BigToM (Gandhi et al., 2023)
BigToM 仅 600 道题，包含社会地点和角色关系但**不追踪心理状态的时间变化**。DynToM 的规模（78,100 题）和动态评估维度远超 BigToM，且通过 Transformation 问题揭示了 LLM 在状态追踪方面的关键缺陷。

### vs. OpenToM (Xu et al., 2024)
OpenToM 包含 2,384 题，有社会地点和可互相依赖的心理状态，但**缺少动态心理状态和渐进式问题设计**。DynToM 是唯一同时具备详细角色画像、角色关系、互依心理状态、动态演化和大规模题库的基准。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统评估 LLM 的动态心智理论能力，框架设计和问题分层均属创新
- 实验充分度: ⭐⭐⭐⭐ 10 个模型 + 人类基线 + 深入错误分析 + Lost-in-the-Middle 验证
- 写作质量: ⭐⭐⭐⭐ 结构完整清晰，心理学理论驱动的设计有说服力
- 价值: ⭐⭐⭐⭐⭐ 揭示了 LLM 在动态社会认知方面的根本性不足，对人机交互领域有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](../../ICML2025/llm_evaluation/position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ACL 2025\] Navigating Rifts in Human-LLM Grounding: Study and Benchmark](navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)
- [\[ACL 2025\] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](../../ECCV2024/llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)

</div>

<!-- RELATED:END -->
