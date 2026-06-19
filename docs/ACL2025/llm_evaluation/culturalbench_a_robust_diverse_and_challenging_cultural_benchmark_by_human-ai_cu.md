---
title: >-
  [论文解读] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming
description: >-
  [ACL 2025][LLM评测][文化知识] 通过 Human-AI CulturalTeaming（人机协作红队测试）流水线构建 CulturalBench，包含 1,696 个人类撰写并经五人独立验证的文化知识问题，覆盖 45 个全球地区和 17 个主题。CulturalBench-Hard（True/False格式）对最强模型（OpenAI o1）也仅 61.5%，远低于人类的 92.4%，揭示了模型在多答案问题上的模式寻求倾向和跨区域文化知识的不均衡表现。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "文化知识"
  - "人机协作红队测试"
  - "多区域覆盖"
  - "模式寻求偏差"
  - "True/False评估"
---

# CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming

**会议**: ACL 2025  
**arXiv**: [2410.02677](https://arxiv.org/abs/2410.02677)  
**代码**: [HuggingFace](https://hf.co/spaces/kellycyy/CulturalBench)  
**领域**: 文化知识评估 / LLM 基准  
**关键词**: 文化知识, 人机协作红队测试, 多区域覆盖, 模式寻求偏差, True/False评估

## 一句话总结

通过 Human-AI CulturalTeaming（人机协作红队测试）流水线构建 CulturalBench，包含 1,696 个人类撰写并经五人独立验证的文化知识问题，覆盖 45 个全球地区和 17 个主题。CulturalBench-Hard（True/False格式）对最强模型（OpenAI o1）也仅 61.5%，远低于人类的 92.4%，揭示了模型在多答案问题上的模式寻求倾向和跨区域文化知识的不均衡表现。

## 研究背景与动机

LLM 的文化代表性不均衡是一个长期问题，但构建高质量的文化知识基准面临多重挑战：

**现有基准鲁棒性不足**：
   - 质量验证不充分：大多数基准仅在数据收集中间步进行质检，而非对最终数据集全量验证
   - 过度依赖网络数据源：Wikipedia 等来源可能已被模型在预训练中见过
   - LLM 生成的基准存在偏差传播风险

**话题覆盖面窄**：
   - 多数基准采用预定义主题（如食物、约会），难以捕捉不同地区独有的文化元素
   - 仅覆盖 1-12 个主题，缺乏多样性

**评估格式的局限**：
   - 多选题格式可通过启发式方法（如选项与国家名称的嵌入相似度）获得远超随机的准确率（40.4% vs 25% 随机），无需理解问题内容
   - 模型可能在猜测而非真正展现文化理解

CulturalBench 旨在构建一个鲁棒、多样、有挑战性的基准来解决这些问题。

## 方法详解

### 整体框架

CulturalTeaming 数据收集流水线分三步：
1. 红队数据收集（人机协作）
2. 人类质量检查（五人独立验证）
3. 多数投票筛选

### 关键设计

#### 1. Human-AI 红队数据收集

- **功能**：引导人类标注者迭代地提出能挑战模型的文化问题
- **核心思路**：
    - **问题构建**：标注者基于个人文化经验头脑风暴文化相关场景（如"新加坡人用纸巾占座"），AI 助手将场景转为结构化四选一问题
    - **问题验证与修改**：标注者在交互平台上用构建的问题挑战 AI 验证器，平台提供修改策略和示例（如"反转问题"）使问题更有挑战性
    - **内部筛选**：研究人员从 3,600+ 问题中过滤掉与特定地区无关的问题，保留 3,000+
- **设计动机**：借鉴 AI 安全红队测试思路，通过人机对抗收集具有挑战性的数据
- **发现式主题方法**：不预设主题集，鼓励标注者从个人经验出发自由探索

#### 2. 五人独立人类质量检查

- **功能**：每个问题由 5 名独立标注者验证
- **核心思路**：
    - 通过 Prolific 平台招募，要求标注者的国籍和 18 岁前主要居住地匹配问题所涉区域
    - 采用多标签选择设置：标注者可选择多个正确答案
    - 额外提供"无正确选项"和"无相关知识"选项，避免猜测
- **设计动机**：文化知识的正确性难以验证，需要专家级人类验证全量最终数据
- 多数投票阈值：≥4/5 标注者一致

#### 3. 双格式基准构建

**CulturalBench-Easy（多选题）**：
- 1,696 道四选一题目
- 单模式题（一个正确答案）：直接使用
- 多模式题（多个正确答案）：重构为复合选项（如"A. (i) 和 (iv)"）加"选择所有适用项"指令

**CulturalBench-Hard（True/False）**：
- 1,696 × 4 = 6,784 道二元判断题
- 每道原始题的四个选项各变为一道 True/False 题
- 必须四个判断全部正确才算正确回答一道题
- 随机基线：0.5⁴ = 6.25%

### 主题发现

通过 GPT-4o 分类，识别出 17 个主题，分属三大类：
- **日常生活**：食物、工作场所等
- **社交礼仪**：问候、社交规范等
- **广泛社会**：庆祝活动、宗教等

不同地区标注者关注的主题不同：意大利人偏重食物（38.9%），以色列人聚焦宗教（23.8%）。

## 实验关键数据

### 主实验：29 个 LLM 在 CulturalBench-Hard 上的表现

| 模型 | CulturalBench-Easy | CulturalBench-Hard |
|------|-------------------|-------------------|
| **人类** | **92.4%** | **92.4%** |
| 随机 | 25.0% | 6.25% |
| OpenAI o1 | 89.6% | 61.5% |
| GPT-4o | - | 60.4% |
| Claude 3.5 Sonnet | - | ~56% |
| Llama-3.1-70B | - | 54.6% |
| Llama-3.1-8B | - | 36.0% |
| GPT-3.5 Turbo | - | 34.5% |
| Cohere Aya-8b | - | 28.7% |

最佳模型与人类在 Hard 版本差距 30.9 个百分点。

### 消融实验：问题类型分析

| 问题类型 | 模型平均准确率 | 最佳模型(o1) | 人类 |
|---------|-------------|------------|------|
| 单模式（1个正确答案，N=1554） | 49.6% | ~65% | ~95% |
| 多模式（多个正确答案，N=142） | 20.9% | ~20% | ~89% |
| **差距** | **28.7%** | **45.5%** | **6.1%** |

模型在多答案问题上表现断崖式下降，而人类仅略有下降。

### 区域表现差异

| 区域 | 模型平均准确率 |
|------|-------------|
| 北美 | 57.9% |
| 北欧 | 51.8% |
| 南亚 | 51.5% |
| 南美 | 41.5% |
| 东欧 | 41.5% |
| 中东/西亚 | 37.8% |

### 启发式基线分析

| 方法 | CulturalBench-Easy 准确率 |
|------|-------------------------|
| 随机猜测 | 25.0% |
| 选项与国名嵌入相似度 | 40.4% |
| 最佳模型 | 89.6% |

不需要问题，仅凭选项与国家名称的相似度就能达到 40.4%，说明 Easy 版本的多选格式存在捷径。

### 关键发现

1. **CulturalBench-Hard 极具挑战性**：最佳模型仅 61.5%，远低于人类 92.4%
2. **多选格式存在捷径**：嵌入相似度启发式可达 40.4%，说明 Easy 版本可能高估模型文化知识
3. **模型的模式寻求倾向**：在多答案问题上表现极差（-28.7%），倾向于过拟合到单一最可能答案
4. **模型规模正相关**：同一家族内，更大模型表现更好
5. **区域表现不均衡**：北美、北欧、南亚表现好于南美、东欧、中东
6. **本地供应商无文化优势**：Qwen/DeepSeek 在东亚、Mistral 在西欧均不如 GPT-4o
7. **性能天花板**：模型家族跨版本改进越来越小，可能接近性能瓶颈

## 亮点与洞察

- **Human-AI CulturalTeaming 流水线**：将 AI 安全红队测试思路创造性地应用于文化知识基准构建
- **五人全量验证**：100% 最终问题由五名独立标注者验证，质量保障远超同类工作
- **发现式主题方法**：不预设话题，让标注者自由探索，捕获了 17 个多样化主题
- **Hard 版本设计精妙**：True/False 格式有效消除了多选题的启发式捷径
- **多答案问题揭示模式寻求偏差**：揭示了 LLM 在处理文化多样性时的本质弱点

## 局限与展望

1. **仅英语**：未评估模型在当地语言上的文化知识表现，可能遗漏"理解语言但不理解文化"的情况
2. **验证者样本小**：部分欠代表地区（如孟加拉国）Prolific 上的活跃标注者不足 30 人，仅能招募 5 人
3. **国家/地区粒度过粗**：同一国家内部的文化多样性（如英国的威尔士 vs 英格兰）未能充分捕获
4. **标注者代表性问题**：受 Prolific 平台限制，某些文化视角可能过度或不足代表
5. **未测试多模态**：仅文本格式，未包含视觉文化知识

## 相关工作与启发

- 与 FORK、BERTAQA、CVQA、NormAd、Blend 等文化基准形成系统对比
- CulturalBench 在三个维度全面领先：验证覆盖率（100%）、主题多样性（17个）、挑战性（最佳模型 61.5%）
- 人机协作红队测试范式可推广到其他主观性强的评估基准构建
- True/False 评估格式对其他多选基准的评估也有借鉴意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ (CulturalTeaming 流水线新颖，Hard版本设计巧妙)
- **实验充分度**: ⭐⭐⭐⭐⭐ (29个模型，区域分析，问题类型分析，启发式基线分析，时间版本分析)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰，分析全面，相关工作对比充分)
- **价值**: ⭐⭐⭐⭐⭐ (高质量开源基准，揭示LLM文化知识的系统性弱点，方法论可复用)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](chatbench_from_static_benchmarks_to_human-ai_evaluation.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](../../ICML2026/llm_evaluation/from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ACL 2025\] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)

</div>

<!-- RELATED:END -->
