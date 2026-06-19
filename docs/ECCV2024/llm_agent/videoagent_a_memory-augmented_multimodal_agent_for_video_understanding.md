---
title: >-
  [论文解读] VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding
description: >-
  [ECCV 2024][LLM Agent][视频理解Agent] 提出 VideoAgent，一个记忆增强的多模态 Agent，通过构建结构化记忆（temporal memory 存储事件描述 + object memory 存储物体跟踪状态）并利用 4 个工具与记忆交互，零样本完成长视频问答任务，在 NExT-QA 上平均 +6.6%、EgoSchema 上 +26.0%，接近 Gemini 1.5 Pro 的性能。
tags:
  - "ECCV 2024"
  - "LLM Agent"
  - "视频理解Agent"
  - "结构化记忆"
  - "多模态工具使用"
  - "长视频理解"
---

# VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding

**会议**: ECCV 2024  
**arXiv**: [2403.11481](https://arxiv.org/abs/2403.11481)  
**代码**: [https://github.com/YueFan1014/VideoAgent](https://github.com/YueFan1014/VideoAgent)  
**领域**: LLM Agent / 视频理解  
**关键词**: 视频理解Agent, 结构化记忆, 多模态工具使用, 长视频理解, LLM Agent

## 一句话总结

提出 VideoAgent，一个记忆增强的多模态 Agent，通过构建结构化记忆（temporal memory 存储事件描述 + object memory 存储物体跟踪状态）并利用 4 个工具与记忆交互，零样本完成长视频问答任务，在 NExT-QA 上平均 +6.6%、EgoSchema 上 +26.0%，接近 Gemini 1.5 Pro 的性能。

## 研究背景与动机

**领域现状**：视频理解尤其是长视频问答是极具挑战性的任务。现有方法主要分为两类：(a) 端到端视频 LLM（如 Video-ChatGPT、VideoChat），将视频帧直接送入多模态模型；(b) 基于 Agent 的方法，利用 LLM 的推理和工具调用能力来理解视频。

**现有痛点**：
   - **端到端方法**的上下文窗口有限，难以处理长视频（几分钟到几十分钟），通常只能均匀采样少量帧，丢失大量时序信息
   - **已有 Agent 方法**缺乏对视频的结构化表示，每次查询都需要重新处理视频内容，效率低且容易忽略全局时序关系
   - 视频中的长程时序关系（如因果推理、事件顺序）和物体跨帧追踪是核心难点，但现有方法都没有很好地同时处理这两方面

**核心矛盾**：长视频包含海量信息，LLM 的上下文窗口无法直接容纳。如何在保持信息完整性的同时，让 LLM 能高效地按需获取视频中的关键信息？

**本文目标**：
   - 如何为长视频构建一个紧凑但信息完整的结构化表示？
   - 如何让 LLM Agent 按需检索和利用这些信息来回答复杂问题？

**切入角度**：借鉴人类理解视频的方式——先形成整体印象（事件流），同时记住关键物体的出现和状态变化（物体记忆），回答问题时按需回溯查找。将这种认知模式形式化为"结构化记忆 + 工具调用"的 Agent 框架。

**核心 idea**：用结构化记忆（事件描述 + 物体状态数据库）预处理视频，再让 LLM 用 4 个专门工具与记忆交互来回答问题。

## 方法详解

### 整体框架

VideoAgent 采用两阶段流程：

**阶段一：记忆构建（Memory Construction）**
- 输入：原始视频
- 处理步骤：将视频切分为 2 秒片段 → 分别构建 temporal memory 和 object memory
- 输出：结构化记忆存储

**阶段二：推理回答（Inference）**
- 输入：用户问题 + 结构化记忆
- 处理步骤：LLM 根据问题选择工具 → 多步交互 → 综合信息生成答案
- 输出：文本答案

这种"先理解后问答"的设计让记忆构建只需执行一次，之后可以反复回答不同问题，大幅提高效率。

### 关键设计

1. **Temporal Memory（时序记忆）**：

    - 功能：存储视频的时序事件流，包括每个片段的描述、文本特征和视觉特征
    - 核心思路：将视频按 2 秒切分为片段序列 $\{s_1, s_2, ..., s_n\}$。对每个片段，使用视频描述模型（video captioning model）生成事件描述文本 $c_i$。同时计算每个片段的文本特征和视觉特征（用于后续的相似度检索）。所有信息存入 temporal memory：$M_{temp} = \{(c_i, f_i^{text}, f_i^{visual})\}_{i=1}^{n}$
    - 设计动机：事件描述以文本形式捕获"发生了什么"，便于 LLM 直接阅读理解；视觉/文本特征用于基于查询的片段定位。2 秒的粒度在信息密度和处理开销之间取得平衡

2. **Object Memory（物体记忆）**：

    - 功能：跟踪和存储视频中所有物体的类别、视觉特征、出现时段等信息
    - 核心思路：使用目标检测器在各帧中检测物体，然后通过一种新颖的重识别（re-identification）方法进行跨片段物体追踪。关键创新在于 re-ID 方法——利用 CLIP 视觉特征计算不同帧中物体的相似度，维护一个全局物体 ID 表。物体信息存入 SQL 数据库中，每个物体包含字段：类别（category）、CLIP 特征（features）、出现的片段列表（appearing segments）
    - 表示形式：$M_{obj} = \text{SQL\_DB}\{(\text{id}, \text{category}, \text{clip\_feat}, \text{segments})\}$
    - 设计动机：很多视频问答涉及特定人物或物体（如"穿红衣服的人做了什么？"、"那个杯子最后放在哪里？"），需要物体级别的跟踪信息。SQL 数据库的形式让 LLM 可以通过结构化查询精确获取信息

3. **工具集（Tool Set）**：
   VideoAgent 设计了 4 个与记忆交互的工具，LLM 通过零样本工具使用能力来调用它们：

   **(a) Caption Retrieval（事件描述检索）**：
    - 输入：起始片段 $s_i$ 和结束片段 $s_j$
    - 功能：从 temporal memory 中提取 $[s_i, s_j]$ 之间的所有事件描述（最多 15 条）
    - 用途：了解某段时间内发生了什么事

   **(b) Segment Localization（片段定位）**：
    - 输入：文本查询 $q$
    - 功能：根据查询特征与 temporal memory 中存储的片段特征的相似度，定位最相关的视频片段
    - 用途：找到"关键动作发生的时刻"

   **(c) Visual Question Answering（视觉问答）**：
    - 输入：问题 + 目标视频片段
    - 功能：使用 video LLM 对指定的短视频片段进行描述和问答
    - 用途：对特定片段进行深入视觉分析

   **(d) Object Memory Querying（物体记忆查询）**：
    - 输入：关于特定物体/人物的问题
    - 功能：查询 SQL 数据库，检索物体的类别、出现时间、特征等信息
    - 用途：回答物体/人物相关的问题（如"谁出现在第二个场景？"）

4. **多步推理过程**：

    - 功能：LLM 作为中央控制器，根据问题进行多步推理，每步调用一个工具
    - 核心思路：每一步包含三个部分：
        - **Chain of Thought（思考）**：LLM 分析当前掌握的信息和还缺少什么
        - **Action（行动）**：选择调用哪个工具及参数
        - **Observation（观察）**：工具返回的结果
    - 循环直到 LLM 认为信息足够，输出最终答案
    - 设计动机：多步推理让 Agent 能处理复杂问题——先定位相关片段，再读取描述，必要时深入分析特定帧，最后综合推理得出答案。这比一次性处理整个视频灵活得多

### 损失函数 / 训练策略

VideoAgent 是一个**零样本框架**，无需针对特定任务训练：
- 各组件使用现成的预训练模型：视频描述模型用于生成 captions，CLIP 用于计算特征和物体重识别，video LLM 用于视觉问答
- LLM 的工具使用通过 prompt engineering 实现（in-context learning），不需要微调
- 唯一关键的设计决策是 prompt 的构建和工具描述的撰写

## 实验关键数据

### 主实验

在长视频问答基准上的对比（NExT-QA 和 EgoSchema）：

| 方法 | 类型 | NExT-QA Acc (%) | EgoSchema Acc (%) |
|------|------|----------------|-------------------|
| InternVideo | 端到端 | 60.0 | 32.1 |
| Video-ChatGPT | 端到端 | 54.4 | 36.0 |
| SeViLA | 端到端 | 73.4 | 25.7 |
| LLoVi (GPT-3.5) | Agent | 67.7 | 50.3 |
| Gemini 1.5 Pro | 闭源端到端 | — | 63.2 |
| **VideoAgent** | **Agent** | **~71.3** | **~60.2** |
| vs 最强开源基线提升 | — | **+6.6 (avg)** | **+26.0** |

关键发现：
- VideoAgent 在 EgoSchema（长达数分钟的自中心视频）上取得了巨大提升 +26.0%，说明结构化记忆对长视频理解帮助极大
- 接近闭源 Gemini 1.5 Pro 的性能（EgoSchema 60.2 vs 63.2），但完全基于开源模型
- NExT-QA 上的提升相对温和（+6.6%），因为 NExT-QA 视频较短，结构化记忆的优势没有充分体现

### 消融实验

记忆组件和工具的消融分析：

| 配置 | NExT-QA | EgoSchema | 说明 |
|------|---------|-----------|------|
| Full VideoAgent | **最高** | **最高** | 全部组件 |
| 去掉 Object Memory | 下降 | 下降 | 无法回答物体相关问题 |
| 去掉 Segment Localization | 下降 | 显著下降 | 长视频中定位能力关键 |
| 去掉 Caption Retrieval | 显著下降 | 显著下降 | 事件描述是核心信息源 |
| 仅用 VQA 工具 | 最低 | 最低 | 退化为逐帧问答 |

### 关键发现

- **Caption Retrieval 是最关键的工具**：去掉后性能下降最多，说明预先生成的事件描述是 Agent 理解视频的主要信息来源
- **Object Memory 对人物/物体相关问题至关重要**：在 EgoSchema（大量涉及手部操作和物体交互）中作用尤其明显
- **Segment Localization 在长视频中不可或缺**：帮助 Agent 快速定位到相关时间段，避免在整个视频中大海捞针
- **2 秒片段粒度**：是实验中发现的最佳平衡点——更短会增加 captions 数量导致冗余，更长会丢失精细事件信息
- **多步推理比单步显著更好**：Agent 平均需要 2-4 步工具调用才能回答一个问题，说明问题确实需要多步信息聚合

## 亮点与洞察

- **结构化记忆是视频 Agent 的关键差异化设计**：不同于直接将视频帧送入 LLM，VideoAgent 预先将视频"翻译"为结构化文本表示（事件描述 + 物体数据库），让 LLM 可以用自然语言和 SQL 查询高效获取信息。这种"先理解-后检索"的范式比"直接看视频"更符合长视频场景
- **Re-identification 方法简洁有效**：利用 CLIP 特征做跨帧物体匹配，避免了训练专门的 tracker。在 SQL 数据库中维护物体 ID 表，使得物体记忆的查询和更新非常规范化
- **零样本框架的现实意义**：不需要任何训练数据，直接组合现成模型，部署成本低。这种 plug-and-play 的设计让框架能够随着各组件模型的升级而自然变强

## 局限与展望

- **记忆构建质量依赖基础模型**：事件描述的准确性取决于 video captioning 模型的能力，物体检测/跟踪依赖检测器性能。基础模型的错误会在记忆中积累，无法修正
- **固定 2 秒粒度不够灵活**：不同视频内容密度差异很大（如格斗场景 vs 静态对话），固定粒度无法适应。可以考虑自适应分段策略
- **缺少空间关系建模**：object memory 只记录物体的类别和出现时段，没有空间位置信息（如"左边的人"、"桌子上面的杯子"），限制了空间推理能力
- **工具调用效率问题**：多步推理的每一步都需要调用 LLM，延迟较高。对于实时或近实时应用不太友好
- **仅限问答任务**：当前框架针对视频 QA 设计，未覆盖视频生成、编辑、摘要等其他视频理解任务

## 相关工作与启发

- **vs LLoVi**：LLoVi 也用 LLM 处理视频描述，但缺少物体级记忆和交互式工具调用，本质上是"读 captions 后做 QA"。VideoAgent 的结构化记忆和多工具交互使其能处理更复杂的推理
- **vs Video-ChatGPT / VideoChat**：端到端视频 LLM 直接处理视频帧，受限于上下文窗口。VideoAgent 通过记忆中转突破了这个限制，但也引入了记忆构建的额外开销
- **vs Gemini 1.5 Pro**：Gemini 有超长上下文窗口可直接处理整个视频，但它是闭源模型。VideoAgent 作为开源方案已接近其性能，展示了 Agent 范式的潜力
- **启发**：记忆增强 Agent 的思路可以推广到其他长序列理解任务（长文档 QA、音频理解等）。核心思想是"先结构化存储再按需检索"，本质上是用外部存储扩展 LLM 的有效上下文

## 评分

- 新颖性: ⭐⭐⭐⭐ 结构化记忆（temporal + object）的双记忆设计直觉且有效，re-ID 方法也有创新
- 实验充分度: ⭐⭐⭐⭐ 在 NExT-QA 和 EgoSchema 两个主要基准上验证，有消融和案例分析
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，案例演示直观，易于理解
- 价值: ⭐⭐⭐⭐⭐ Agent+记忆的范式对长视频理解影响深远，EgoSchema +26% 的提升非常显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding](../../CVPR2026/llm_agent/symphony_a_cognitively-inspired_multi-agent_system_for_long-video_understanding.md)
- [\[ECCV 2024\] Agent3D-Zero: An Agent for Zero-shot 3D Understanding](agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](../../CVPR2026/llm_agent/think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](../../CVPR2026/llm_agent/haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[CVPR 2026\] SciEducator: Scientific Video Understanding and Educating via Deming-Cycle Multi-Agent System](../../CVPR2026/llm_agent/scieducator_scientific_video_understanding_and_educating_via_deming-cycle_multi-.md)

</div>

<!-- RELATED:END -->
