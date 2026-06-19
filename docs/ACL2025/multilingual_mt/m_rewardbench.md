---
title: >-
  [论文解读] M-RewardBench: Evaluating Reward Models in Multilingual Settings
description: >-
  [ACL 2025][多语言/翻译][reward model] 构建首个多语言奖励模型评估基准 M-RewardBench（23种 typologically 多样语言、2.87K 偏好实例，覆盖 Chat/Safety/Reasoning/Translation 四类能力），系统评估多种 RM 后发现英语与非英语性能存在显著差距，且 RM 偏好可在语言间发生实质性漂移。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "reward model"
  - "Multilingual"
  - "RLHF"
  - "Preference Benchmark"
  - "偏好漂移"
---

# M-RewardBench: Evaluating Reward Models in Multilingual Settings

**会议**: ACL 2025  
**arXiv**: [2410.15522](https://arxiv.org/abs/2410.15522)  
**代码**: [https://github.com/for-ai/m-rewardbench](https://github.com/for-ai/m-rewardbench)  
**领域**: LLM对齐 / 多语言 / 奖励模型评估  
**关键词**: reward model, Multilingual, RLHF, Preference Benchmark, 偏好漂移

## 一句话总结

构建首个多语言奖励模型评估基准 M-RewardBench（23种 typologically 多样语言、2.87K 偏好实例，覆盖 Chat/Safety/Reasoning/Translation 四类能力），系统评估多种 RM 后发现英语与非英语性能存在显著差距，且 RM 偏好可在语言间发生实质性漂移。

## 研究背景与动机

**领域现状**：奖励模型（RM）是当前 LLM 对齐（RLHF/DPO）的核心组件，通过学习人类偏好来引导语言模型生成高质量输出。RewardBench 等基准已成为评估 RM 性能的标准工具。

**现有痛点**：RM 的训练和评估几乎完全在英语环境下进行。全球大多数用户使用非英语语言与 LLM 交互，但我们对 RM 在这些语言上能否正确判断人类偏好几乎毫无了解。这意味着经过 RLHF 对齐的模型在非英语场景下的对齐质量可能存在系统性缺失。

**核心矛盾**：英语上表现优秀的 RM 在其他语言上是否同样可靠？如果不是，这对多语言 LLM 部署意味着什么？

**本文目标** 通过构建多语言 RM 评估基准并系统评估，量化 RM 的跨语言能力差距和影响因素。

**切入角度**：基于英语 RewardBench 通过高质量翻译构建 23 种语言的偏好数据，保持评估维度一致以实现公平跨语言对比。

**核心 idea**：首个多语言 RM 基准 + 系统评估揭示了 RM 跨语言偏好漂移和性能差距。

## 方法详解

### 整体框架

从英语 RewardBench 出发，精选偏好实例，高质量翻译到 23 种语言，然后在该基准上系统评估 classifier-based、generative 和 implicit 三类 RM。

### 关键设计

1. **多语言基准构建（M-RewardBench Dataset）**:

    - 功能：构建覆盖 23 种 typologically 多样语言的 RM 评估数据集
    - 核心思路：从 RewardBench 精选偏好实例（chosen/rejected 对），覆盖四种能力维度——Chat（对话质量）、Safety（安全性）、Reasoning（推理能力）、Translation（翻译能力，新增维度）。翻译流程包含严格的质量控制：先机器翻译后人工校验，确保语义等价
    - 设计动机：直接采用英语基准无法反映多语言能力；从零构建每种语言的原生偏好数据成本过高且难以保持评估一致性。翻译方案在成本和可比性间取得平衡
    - 覆盖语言：23 种，包含中文、日语、韩语、阿拉伯语、印地语、法语、德语、西班牙语、俄语、土耳其语等，覆盖多种语系和文字系统，总计 2.87K 偏好实例

2. **系统化评估框架**:

    - 功能：在 M-RewardBench 上全面评估多种 RM 架构的跨语言表现
    - 核心思路：评估三类 RM——(1) Classifier-based RM（如 UltraRM），通过回归头输出标量分数；(2) Generative RM / LLM-as-a-judge（如 GPT-4），直接生成偏好判断；(3) Implicit RM（如 DPO 训练的模型），通过似然差异隐式表达偏好。每种 RM 在 23 种语言上逐语言评估
    - 设计动机：不同 RM 架构的多语言泛化模式可能不同——classifier-based 可能受语言表示影响更大，而 generative RM 可能受提示语言影响

3. **多维度分析**:

    - 功能：深入分析影响 RM 跨语言表现的因素
    - 核心思路：(1) 英语 vs 非英语整体差距分析，(2) 跨语言偏好漂移分析——同一偏好实例在不同语言下 RM 的 chosen/rejected 判断是否发生反转，(3) 翻译质量与 RM 性能的相关性分析，(4) 语言资源水平（高/中/低资源）与 RM 性能的关系
    - 设计动机：仅知道"有差距"不够，需要理解差距的来源才能指导改进

## 实验关键数据

### 主实验（跨语言 RM 性能）

| 维度 | 英语 | 非英语平均 | 差距 |
|------|------|-----------|------|
| Overall Accuracy | 最高 | 显著低于英语 | 明显 |
| Chat | 高 | 下降明显 | 中等 |
| Safety | 高 | 下降明显 | 中等 |
| Reasoning | 高 | 下降最大 | 大 |

### 消融实验（影响因素分析）

| 因素 | 影响方向 | 说明 |
|------|---------|------|
| 翻译质量 | 正相关 | 翻译质量越高，RM 在该语言上表现越好 |
| 语言资源量 | 正相关 | 高资源语言（法/德/西）优于低资源语言（斯瓦希里/乌尔都） |
| RM 架构 | 因语言而异 | 没有单一架构在所有语言上最优 |

### 关键发现

- **偏好漂移现象**：同一偏好实例翻译到不同语言后，RM 的 chosen/rejected 判断可能发生反转——说明 RM 的偏好判断具有语言依赖性，非语言无关
- **翻译质量→RM 性能的因果链**：翻译质量越高的语言版本 RM 表现越好，为提升多语言 RM 指向了明确方向
- **高资源语言优势明显**：中/日/韩/法/德等高资源语言 RM 表现接近英语，而低资源语言差距大幅扩大
- **不同能力维度差距不均**：Reasoning 维度的跨语言差距最大，可能反映推理任务对语言细微语义的更高敏感度

## 亮点与洞察

- 首个针对奖励模型的多语言评估基准，填补了 RLHF 多语言部署的关键评估空白——在此之前，多语言场景下 RM 是否可靠是一个被忽视的盲区
- 偏好漂移现象的发现具有重要理论意义——它表明人类偏好的 RM 衡量不是语言无关的，同一偏好在不同语言表达下可能得到相反判断
- 23 种语言 × 多种 RM 架构 × 四个能力维度的全面评估，覆盖度和系统性在该方向是首创

## 局限与展望

- 基于翻译构建可能引入 translationese 偏差——翻译文本的语言特征与原生文本不同，可能影响 RM 判断
- 每种语言约 125 条实例，规模偏小，部分语言的统计显著性可能不足
- 未涉及 code-switching 或混合语言场景，而这在多语言实际使用中很常见
- 主要分析翻译质量和资源量两个因素，文化差异、语言结构差异等更深层因素未探讨
- Translation 维度是新增的，与原始 RewardBench 不完全可比

## 相关工作与启发

- **vs RewardBench**: 仅英语单语言评估；M-RewardBench 扩展到 23 种语言实现跨语言对比
- **vs MEGA/XTREME 等多语言评估**: 评估 LLM 本身的多语言能力；M-RewardBench 专注奖励模型这一特殊组件
- **vs 多语言 RLHF 工作**: 之前少有工作评估 RM 在多语言下的表现，本文是开创性工作

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多语言RM评估基准，方向重要且开创性
- 实验充分度: ⭐⭐⭐⭐ 23种语言×多种RM架构，但每语言数据量偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现有条理，动机阐述充分
- 价值: ⭐⭐⭐⭐ 对多语言LLM对齐部署有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](multilingual_llm_english_accent.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2025\] X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System](x-webagentbench_a_multilingual_interactive_web_benchmark_for_evaluating_global_a.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](../../ACL2026/multilingual_mt/evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)

</div>

<!-- RELATED:END -->
