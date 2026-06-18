---
title: >-
  [论文解读] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video
description: >-
  [多模态VLM] 提出 RTV-Bench 基准，包含 552 个视频和 4608 个 QA 对，通过**多时间戳问答**（同一问题在不同时间点答案不同）、**层级问题结构**和**多维评估**三大设计，系统评测 MLLM 在实时视频流中的持续分析能力，揭示了在线模型优于离线模型、单纯增大模型或增加帧数收益有限等关键发现。
tags:
  - "多模态VLM"
---

# RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video

## 一句话总结

提出 RTV-Bench 基准，包含 552 个视频和 4608 个 QA 对，通过**多时间戳问答**（同一问题在不同时间点答案不同）、**层级问题结构**和**多维评估**三大设计，系统评测 MLLM 在实时视频流中的持续分析能力，揭示了在线模型优于离线模型、单纯增大模型或增加帧数收益有限等关键发现。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在视觉感知、理解和推理方面取得快速进展，Video-LLM 的研究从短视频片段扩展到长视频内容，越来越多工作整合视频、音频、字幕等多模态信号。

**现有痛点**：现有视频 benchmark（如 Video-MME、MVBench）主要面向离线评估，使用静态 QA 对，无法测试模型在连续动态视频流中的实时响应能力。VStream、StreamingBench、OVOBench 虽有改进，但对实时响应性的评估仍不充分——忽视了模型对序列到达的视觉输入中转换和瞬时细节的捕捉能力。

**核心矛盾**：实时视频场景要求模型随视觉场景演变持续维护连贯理解并实时更新状态，但现有 benchmark 的"单问题-单答案"静态评估范式无法有效测试这种持续分析能力。

**本文目标**：构建一个细粒度的实时视频分析基准，系统评测 MLLM 在动态视频流中的持续感知、理解和推理能力。

**切入角度**：从三个维度切入——(1) 多时间戳 QA 让同一概念性问题在不同时间点有不同正确答案；(2) 层级问题结构从基础到高级逐步递进；(3) 多维评估覆盖 8 个维度提供细粒度诊断。

**核心 idea**：通过"答案随时间变化的同一问题"这一设计，直接测试模型对动态状态转换的敏感性和持续追踪能力。

## 方法详解

### 整体框架

RTV-Bench 是一个面向实时视频分析的细粒度 benchmark，包含 552 个多样视频（总时长 167.2 小时，平均 18.2 分钟/视频）和 4608 个人工标注的 QA 对。视频覆盖三大领域（智能驾驶、体育赛事、第一人称视角）和 16 个子类别。评估体系包含 Accuracy（准确率）和 Score（条件化得分，需基础问题全对才计算高级问题得分）两个指标。对在线模型直接在查询时间戳处提问，对离线模型则截取到查询时间点为止的视频片段进行模拟。

### 关键设计

1. **多时间戳问答机制（Multi-Timestamp QA）**
    - **功能**：在同一视频中，同一概念性问题在不同时间戳处有不同正确答案
    - **核心思路**：不同于 OVO-Bench 在不同时间戳引入不同问题，RTV-Bench 重新审视同一概念查询（如"A 手里拿着什么？"），正确答案随场景展开而变化。人工标注每个答案选项对应的最早有效时间戳
    - **设计动机**：更严格地测试模型的持续分析能力——要求模型主动追踪时间变化并持续更新理解，而非仅定位相关信息

2. **层级问题结构（Hierarchical Question Structure）**
    - **功能**：每组问题包含约 3 个递进难度的选择题，前两题为基础感知题，第三题为高难度综合推理题
    - **核心思路**：高阶问题逻辑依赖于对基础感知和理解的掌握，配合 Score 指标（仅在基础题全对时才计算高级题得分），确保评估反映真实的层级推理能力
    - **设计动机**：防止模型通过认知捷径在复杂问题上获得虚假高分，确保高级推理建立在扎实的基础理解之上

3. **多维评估体系（Multidimensional Evaluation）**
    - **功能**：覆盖 8 个维度的细粒度诊断——时间感知(TP)、场景感知(SP)、视觉感知(VP)、未来预测(FP)、现象理解(PU)、意图分析(IA)、全局理解(GU)、时空推理(SR)
    - **核心思路**：将能力维度分为感知、理解、推理三大类，每类下设 2-3 个子维度，超越聚合分数提供模型能力的精细画像
    - **设计动机**：为研究者提供模型在动态场景中各方面能力和局限性的信息化视图，指导针对性改进

### 损失函数 / 训练策略

本文为 benchmark 论文，不涉及模型训练。评估指标设计上：

- **Accuracy**：直接计算正确答案比例
- **Score（条件化得分）**：$\text{Score} = \frac{\sum_{i=1}^{N} B_i \cdot N_{q2,i}^{\text{correct}}}{\sum_{i=1}^{N} N_{q2,i}^{\text{total}}}$，其中 $B_i$ 为基础题全对指示器。该指标确保只有在基础问题全部正确时，高级问题的得分才被计入，反映模型的可靠性和层级推理一致性

## 实验关键数据

### 主实验

| 模型 | 规模 | 感知 Acc/Score | 理解 Acc/Score | 推理 Acc/Score | FQA Acc | MTQA Acc | 总体 Acc/Score |
|------|------|---------------|---------------|---------------|---------|----------|---------------|
| GPT-4o | - | 51.61/21.90 | 49.31/20.76 | 48.71/23.95 | 56.53 | 44.73 | **50.02/22.10** |
| IXC2.5-OL | 7B | 47.21/15.87 | 48.22/15.23 | 46.18/14.45 | 59.05 | 38.21 | 47.33/15.40 |
| VITA-1.5 | 7B | 45.66/12.80 | 44.12/11.83 | 43.37/10.15 | 55.06 | 36.32 | 44.51/11.80 |
| VideoChat-Online | 4B | 46.86/12.30 | 46.34/12.80 | 43.53/11.00 | 55.16 | 38.21 | 45.83/12.10 |
| Qwen2.5-VL | 7B | 42.30/7.70 | 39.85/7.00 | 38.16/6.90 | 44.07 | 37.46 | 40.41/7.13 |
| VideoLLaMA2 | 7B | 40.62/8.67 | 39.85/7.77 | 37.49/6.75 | 45.77 | 34.95 | 39.55/7.90 |
| LLaVA-Video | 7B | 35.83/5.03 | 33.81/3.77 | 35.15/5.75 | 36.28 | 34.17 | 34.90/4.80 |

### 消融实验

**帧数与模型规模的影响（Qwen2.5-VL）**

| 模型规模 | 8帧 | 16帧 | 32帧 | 64帧 | 趋势 |
|---------|------|------|------|------|------|
| 3B | ~37% | ~37% | ~38% | ~37% | 非单调波动 |
| 7B | ~39% | ~40% | ~40% | ~40% | 微弱增长 |
| 32B | ~39% | ~39% | ~40% | ~40% | 轻微提升 |
| 72B | ~40% | ~40% | ~40% | ~40.78% | 最优但收益递减 |

### 关键发现

1. **在线模型显著优于离线模型**：IXC2.5-OL（47.33%）大幅超越最优离线模型 Qwen2.5-VL（40.41%），即使最弱在线模型 VITA-1.5 也超过离线代表 VideoLLaMA2
2. **增加帧数收益有限甚至有害**：增加采样帧数不能一致提升性能，部分情况下反而导致性能下降（如 IXC2.5-OL 帧数增多后性能明显下降），说明过多时间输入可能导致注意力稀释
3. **模型规模正相关但收益递减**：从 3B 到 72B 准确率提升约 2-3 个百分点，大模型更能稳定受益于帧数增加，但绝对收益有限
4. **MTQA 任务是核心瓶颈**：所有模型在多时间戳 QA 上的准确率（33%-44%）远低于基础 QA（35%-59%），说明持续状态追踪仍是根本性难题
5. **Score 与 Accuracy 差距巨大**：所有模型的 Score 远低于 Accuracy，表明模型在基础问题上频繁犯错，高级推理的可靠性堪忧

## 亮点与洞察

- **"同一问题、答案随时间变化"的设计非常精妙**：比"不同时间问不同问题"更能测试持续追踪能力，是对实时理解评估范式的质性提升
- **条件化 Score 指标有启发性**：通过要求基础题全对才计算高级题得分，有效识别了"猜对复杂题但基础理解有缺陷"的模型，这种层级一致性评估值得推广
- **反直觉发现具有重要指导意义**：增加帧数不一定有效这一发现挑战了"信息越多越好"的朴素假设，指向了时间选择性建模和自适应帧利用的研究方向
- **在线 vs 离线的系统性对比**为架构选择提供了清晰指引：专用流式处理架构的优势来自持续状态更新，而非简单的离线预处理

## 局限与展望

1. **仅限视觉模态**：未纳入音频等重要模态，实际实时场景中多模态信号互为补充
2. **评估规模有限**：552 个视频、4608 QA 对的规模相对适中，场景多样性有待扩展
3. **离线模型评估公平性存疑**：为离线模型截取视频片段的做法改变了任务设定，可能低估了离线模型的真实能力
4. **缺少对模型内部机制的深入分析**：发现了"帧数无用"等反直觉现象但未深入分析原因，仅停留在观察层面
5. **评测模型范围可扩展**：可加入更多最新的流式视频模型（如 VideoLLM-Online 系列）以及更大规模闭源模型

## 相关工作与启发

- **StreamingBench / OVOBench**：前序流式视频 benchmark，但使用不同问题在不同时间戳评测，RTV-Bench 的同一问题设计更严格
- **VStream**：首次尝试实时视频评测，但侧重于延长视频时长而非持续分析
- **IXC2.5-OmniLive**：通过模块并行和长期记忆实现流式处理，在 RTV-Bench 上表现最优的开源模型
- **VITA-1.5**：分阶段融合训练的在线模型，验证了流式架构设计的必要性
- **启发**：未来实时视频理解需要的不是更大的模型或更多的帧，而是原则性的时间建模（temporal-selective attention）和选择性信息聚合策略

## 评分

- **创新性** ⭐⭐⭐⭐ — 多时间戳 QA（同一问题答案随时间变化）和条件化 Score 指标设计新颖，填补了实时视频持续分析评测的空白
- **实用性** ⭐⭐⭐⭐ — 对当前 MLLM 实时视频能力给出了系统性诊断，8 维评估和层级指标设计可被后续 benchmark 借鉴
- **可靠性** ⭐⭐⭐⭐ — 552 视频 + 4608 QA 规模适中，人工标注质量高，多轮审核流程严格，但离线模型评估方式存在公平性争议
- **综合** ⭐⭐⭐⭐ — 高质量 benchmark 论文，核心设计理念正确且有启发性，揭示了重要的反直觉发现，为实时视频 MLLM 发展提供了有价值的方向指引

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](../../ICCV2025/multimodal_vlm/orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ACL 2025\] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](../../ACL2025/multimodal_vlm/madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)

</div>

<!-- RELATED:END -->
