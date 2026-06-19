---
title: >-
  [论文解读] Context Informs Pragmatic Interpretation in Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][语用推理] 通过迭代参考游戏（iterated reference games）系统评估 VLM 的语用推理能力，发现模型在无上下文时表现远逊于人类，但在获得相关对话历史后能快速学习达到约 80% 准确率，揭示了 VLM 对上下文信息的强烈依赖性。 多轮对话是人类交流的核心特征—…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "语用推理"
  - "参考游戏"
  - "上下文敏感性"
  - "VLM认知评估"
  - "抽象视觉推理"
---

# Context Informs Pragmatic Interpretation in Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.03908](https://arxiv.org/abs/2511.03908)  
**代码**: [GitHub](https://github.com/benpry/vlm-tg-context/tree/coginterp)  
**领域**: 多模态VLM  
**关键词**: 语用推理, 参考游戏, 上下文敏感性, VLM认知评估, 抽象视觉推理

## 一句话总结

通过迭代参考游戏（iterated reference games）系统评估 VLM 的语用推理能力，发现模型在无上下文时表现远逊于人类，但在获得相关对话历史后能快速学习达到约 80% 准确率，揭示了 VLM 对上下文信息的强烈依赖性。

## 研究背景与动机

多轮对话是人类交流的核心特征——共享的对话历史支撑着语义约定的形成。**迭代参考游戏**是研究这类交流的经典范式：描述者需要用语言描述一个目标图形，让匹配者从多个选项中正确选择。随着游戏轮次推进，双方会逐渐形成简短的约定化表达。

这项能力对 AI 对话系统至关重要，需要两个行为特征：

**语用解释能力**: 根据语境理解话语的实际含义

**上下文敏感性**: 利用先前交互信息指导当前理解

然而，抽象图形（tangram）的参考游戏对 AI 系统仍然极具挑战，特别是在少样本设定下。本研究是首次系统比较人类与最新开放权重 VLM 在迭代参考游戏中的语用推理表现。

## 方法详解

### 整体框架

**数据来源**: 使用 Boyce et al. 的迭代参考游戏数据集，每局游戏中玩家看到 12 个 tangram 图形排列的网格，描述者描述高亮目标，匹配者选择。每局 2-6 名玩家，进行 6 轮共 72 次试验。

**评估设置**: 选取 10 局游戏，评估 4 个开放权重 VLM：

- Qwen 2.5 VL 32B
- Gemma 3 27B
- Llama 3.2 11B
- Kimi VL A3B

模型接收系统提示、12 个标签图形的拼接图像、先前试验的对话历史（作为 chat history），输出 A-L 字母的 log 概率。以正确目标的归一化概率作为准确率。

### 关键设计

**8 种控制条件**（系统操控上下文的数量、顺序和相关性）：

| 条件 | 上下文来源 | 同局游戏 | 试验顺序 | 目标图形可见 |
|------|-----------|---------|---------|------------|
| Yoked | 同局 | ✓ | 原始顺序 | ✓ |
| Shuffled | 同局 | ✓ | 随机打乱 | ✓ |
| Backward | 同局 | ✓ | 逆序 | ✓ |
| Ablated | 同局 | ✓ | 原始顺序 | ✗ |
| Other-within | 不同局(单局) | ✓ | 原始顺序 | ✓ |
| Other-across | 不同局(多局) | ✗ | 原始顺序 | ✓ |
| Random | 不同局(多局) | ✗ | 随机打乱 | ✓ |
| No context | 无 | ✗ | 无 | ✗ |

### 人类对照

- **原始玩家**: 参与原始游戏的互动玩家
- **朴素匹配者**: 仅阅读对话文本但未参与原始游戏的人类被试
    - Yoked 条件 ($N=99$)、Shuffled ($N=97$)、Backward ($N=89$)、Random ($N=107$)

## 实验关键数据

### 主实验

**各条件下模型 vs 人类准确率**（稳态估计）：

| 条件 | 人类(原始) | 人类(朴素) | Gemma 3 | Qwen 2.5 | Llama 3.2 | Kimi VL |
|------|-----------|-----------|---------|----------|-----------|---------|
| No context | ~0.75 | — | ~0.15 | ~0.15 | ~0.12 | ~0.12 |
| Yoked (后期) | ~0.95 | ~0.75 | ~0.80 | ~0.40 | ~0.75 | ~0.75 |
| Other-within | — | — | ~0.40 | ~0.35 | ~0.30 | ~0.35 |

**模型-人类相关性**（逐试验层面）：

| 条件 | 人类 split-half | Qwen 2.5 | Gemma 3 | Llama 3.2 | Kimi VL |
|------|----------------|----------|---------|-----------|---------|
| Yoked | .42 [.32, .50] | .10 | .20 | .25 | .27 |
| Backward | .48 [.40, .56] | .23 | .31 | .40 | .35 |
| Random | — | .61 | .58 | .55 | .57 |

### 消融实验

**上下文相关性分析**（核心系统性操控）：

| 对比 | 发现 |
|------|------|
| Yoked vs Shuffled | 打乱顺序降低性能，类似人类表现 |
| Yoked vs Backward | 模型在逆序条件下表现优于打乱，与人类相反 |
| Yoked vs Other-within | 不同游戏的上下文大幅降低性能（0.3-0.5） |
| Yoked vs Ablated | 移除目标图形的先前描述导致性能大幅下降 |

### 关键发现

1. **无上下文时模型远逊于人类**: 人类首次见到描述时准确率约 0.75，模型仅略高于随机（0.083）
2. **模型可快速利用相关上下文**: 在 Yoked 条件下，多数模型在后期达到约 0.80 准确率
3. **上下文必须来自同一游戏**: 不同游戏的上下文几乎无用（0.3-0.5），说明约定是游戏特异的
4. **模型与人类的错误模式不同**: 逐试验相关性很弱（r = .10-.40），远低于人类间一致性

## 亮点与洞察

- **精巧的实验设计**: 8 种控制条件系统分离了上下文数量、顺序和相关性的影响
- **认知科学视角**: 将语用推理置于人类交流研究的经典范式中，而非仅做模型对比
- **揭示模型本质差异**: 人类具有"直觉"理解能力（零样本即可达 0.75），VLM 严重依赖显式上下文
- **Backward 条件的有趣发现**: 模型从约定化表达反推早期表达比人类更擅长，暗示不同的推理机制
- **方法论贡献**: 为评估 VLM 的上下文敏感性和语用推理提供了可复现的框架

## 局限性

- 仅测试抽象图形（tangram），向自然图像的泛化性未验证
- 模型获得了比人类更丰富的反馈（模型知道正确答案 vs 人类仅知对错）
- 仅评估理解（comprehension），生成合适描述是更难的任务
- 测试模型数量有限（4 个），且仅为开放权重模型
- Workshop paper 篇幅有限，部分分析不够深入

## 相关工作与启发

- **Clark (1996)**: 共同基础（common ground）理论为本研究提供了认知基础
- **Hawkins et al. (2021)**: 从伙伴到群体的层级约定形成模型
- **Gul et al. (2024) CoGen**: 尝试训练模型参与参考游戏的生成端，仍表现不佳
- 启发：VLM 的"理解"可能更多是模式匹配而非真正的语用推理；需要区分 in-context learning 和真正的语用适应能力

## 评分

- ⭐ 新颖性: 4/5 — 首次系统评估 VLM 在迭代参考游戏中的语用推理，实验设计精巧
- ⭐ 实验充分度: 3/5 — 控制条件丰富，但模型数量和刺激类型受限
- ⭐ 写作质量: 4/5 — 逻辑清晰，图表直观，认知科学背景阐述充分
- ⭐ 价值: 3/5 — Workshop paper 深度适中，但为 VLM 语用能力评估开辟了有意义的方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[NeurIPS 2025\] ExGra-Med: Extended Context Graph Alignment for Medical Vision-Language Models](exgra-med_extended_context_graph_alignment_for_medical_vision-language_models.md)
- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](../../ACL2026/multimodal_vlm/cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)

</div>

<!-- RELATED:END -->
