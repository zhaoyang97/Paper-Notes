---
title: >-
  [论文解读] Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously
description: >-
  [CVPR 2025][视频理解][流式视频理解] 提出 Video Streaming Thinking (VST) 范式，在视频播放过程中交替执行"看"和"想"——模型边接收视频帧边生成中间推理链，将 CoT 计算摊销到预查询阶段，从而在保持实时响应（0.56s QA延迟）的同时实现 StreamingBench 79.5% 的 SOTA。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "流式视频理解"
  - "链式思维"
  - "强化学习"
  - "知识图谱"
  - "在线推理"
---

# Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously

**会议**: CVPR 2025  
**arXiv**: [2603.12262](https://arxiv.org/abs/2603.12262)  
**代码**: [https://github.com/1ranGuan/VST](https://github.com/1ranGuan/VST)  
**领域**: 视频理解  
**关键词**: 流式视频理解, 链式思维, 强化学习, 知识图谱, 在线推理

## 一句话总结

提出 Video Streaming Thinking (VST) 范式，在视频播放过程中交替执行"看"和"想"——模型边接收视频帧边生成中间推理链，将 CoT 计算摊销到预查询阶段，从而在保持实时响应（0.56s QA延迟）的同时实现 StreamingBench 79.5% 的 SOTA。

## 研究背景与动机

**领域现状**：在线视频理解（streaming video understanding）要求 VideoLLM 实时处理连续视频输入并即时响应。现有方法主要通过压缩视觉 token 或 KV cache 检索来管理上下文窗口，实现高效的流式感知。

**现有痛点**：两大路线各有缺陷——(a) 流式感知方法（StreamForest、TimeChatOnline）专注视觉 token 管理，LLM 几乎不参与推理分析，缺乏深度理解能力；(b) 离线 CoT 推理方法（Video-R1）在查询后执行 step-by-step 推理，QA 延迟高达 8.8s，无法满足实时需求。

**核心矛盾**：显式推理能力与实时响应性之间存在本质冲突——推理需要时间，但实时系统要求低延迟。

**本文目标** 如何在不牺牲实时性的前提下赋予流式 VideoLLM 强大的推理能力？

**切入角度**：受人脑神经耦合机制启发——大脑的逻辑流与外部信息流同步进行。不是在查询后才推理，而是在视频播放期间就不断进行中间推理，把推理成本"摊销"到查询前。

**核心 idea**：将 CoT 推理从"查询后被动生成"转变为"视频播放中主动生成"，实现推理与感知的时间并行。

## 方法详解

### 整体框架

VST 将流式视频理解建模为多轮对话任务。视频流被分割为若干 clip，每个 clip 到达时，模型基于当前 clip 和历史记忆生成一段"streaming thought"（中间推理文本），写入一个双记忆系统——短期视觉缓冲（当前 clip 的原生视觉 token）+ 长期文本记忆（历史 thought 的 FIFO 队列）。当用户查询到达时，模型直接基于已积累的推理记忆和当前视觉上下文生成最终回答，QA 延迟极低。

联合概率分解为：$p(\mathbf{y}|\mathbf{q}, \mathcal{V}) = p(\mathbf{y}|\mathbf{q}, \mathbf{c}^K, \mathbf{m}^K) \prod_{k=1}^{K-1} p(\mathbf{z}^k|\mathbf{c}^k, \mathbf{m}^{k-1})$

前面的乘积项是"流式思考"过程（在查询到达前完成），最后一项是"直接回答"（查询到达后瞬间完成）。

### 关键设计

1. **VST-SFT（监督微调阶段）**：

    - 功能：将离线 VideoLLM 适配为流式推理模式
    - 核心思路：将视频序列显式组织为 $(memory, (clip_1, thought_1), ..., (clip_{K-1}, thought_{K-1}), clip_K, query, answer)$ 的多轮格式，用 streaming video attention mask 限制视觉 token 只能看到最近 $L$ 个，文本 token 在因果约束下全局可见
    - 设计动机：流式推理要求严格遵守时间因果性——第 $k$ 步只能看到第 $k$ 步及之前的信息，不能"偷看未来"。attention mask 强制执行这个约束
    - 长视频处理：将序列切分为多个 segment，跨 segment 传递 memory 状态

2. **VST-RL（强化学习阶段）**：

    - 功能：从 off-policy 模仿转向 on-policy 自主探索，提升中间推理质量
    - 核心思路：采用 GRPO 策略，模型在流式环境中进行 agentic loop 生成轨迹 $\mathcal{T}$，只对最终答案计算 verifiable reward，但将 advantage 回传给整个轨迹的所有 token（包括中间 thought）
    - 设计动机：中间推理没有 ground truth，但好的中间推理应该带来正确的最终答案。通过 RL 让模型自主学习"什么样的 streaming thought 对下游回答最有帮助"
    - 关键发现：VST-SFT 主要提升 backward memory（+9.2%），VST-RL 主要提升 forward prediction（+12.7%），两者互补

3. **知识图谱数据合成 Pipeline**：

    - 功能：自动生成 100K 高质量流式推理训练数据
    - 核心思路：(a) 用 PySceneDetect 分割视频场景 → (b) 用 Gemini 3.0 Flash 提取实体和关系构建知识图谱 → (c) DFS 采样多跳证据链 → (d) 基于证据链生成流式 QA 对和中间 CoT
    - 设计动机：现有 CoT 数据都针对离线场景（可以看到全部视频），不适合流式设置。知识图谱保证数据的时间因果性和多跳推理质量
    - 质量控制：world-knowledge check、format alignment、logical consistency、repetition check、thought validation

### 训练策略

- 基座模型：Qwen2.5-VL-7B，视频 2fps 采样
- VST-SFT：冻结视觉编码器，LR 5e-6，1 epoch，每视频最多 384 帧
- VST-RL：DAPO 算法，rollout batch 256，group size 8，LR 5e-7
- 训练数据：100K VST + 50K LLaVA-Vid QA（SFT）；11K 多选/计数题（RL）

## 实验关键数据

### 主实验

| 模型 | StreamingBench | OVO-Bench | VideoMME (Long) | LongVideoBench | VideoHolmes | QA 延迟 |
|------|---------------|-----------|-----------------|----------------|-------------|---------|
| GPT-4o | 73.3% | 59.5% | 65.3% | 66.7% | 42.0% | — |
| Qwen2.5-VL-7B | 73.7% | 55.0% | — | 54.7% | 32.9% | 0.54s |
| Video-R1 w/CoT | — | — | — | — | 36.5% | 8.80s |
| StreamForest-7B | 77.3% | 55.6% | — | — | — | — |
| **VST-7B** | **79.5%** | **59.3%** | **55.3%** | **58.0%** | **41.9%** | **0.56s** |

VST-7B 在在线基准上超越所有开源模型（包括 StreamForest +2.2%），且超越 GPT-4o +6.2%。同时在离线基准上也有竞争力，特别是 VideoHolmes +5.4%（vs Video-R1），QA 延迟只有 Video-R1 的 1/15.7。

### 消融实验

| 配置 | OVO-Bench Overall | VideoMME Overall |
|------|------------------|------------------|
| Qwen2.5-VL-7B baseline | 50.5% | 62.9% |
| + LLaVA-Vid 50K SFT | 52.3% | 61.8% |
| + VST-SFT only | 57.4% | 63.0% |
| + VST-RL only | 56.8% | 62.8% |
| + VST-SFT & VST-RL | **59.3%** | **64.9%** |

### 关键发现

- **VST-SFT 和 VST-RL 功能互补**：SFT 主要增强 backward tracing（+9.2%），RL 主要增强 forward prediction（+12.7%），组合使用效果最佳
- **模型规模可扩展**：3B→7B→32B 均有一致提升，StreamingBench 绝对提升 +7.7/+7.8/+9.2%
- **推理次数影响**：Backward task 随 thinking steps 增加持续提升（1→16步），Real-Time 和 Forward 在 ≥4 步后饱和，过多记忆反而引入冗余

## 亮点与洞察

- **"将推理成本摊销到播放时间"是一个非常优雅的 insight**：视频播放本身有等待时间（帧间间隔），利用这个空隙做推理，实现"零额外延迟"的 test-time scaling。这个思路可以迁移到任何有自然等待时间的场景（语音对话、实时传感器数据处理等）。
- **RL 只奖励最终答案、但 advantage 传播到中间 thought**：不需要给中间推理步骤标注 ground truth，只要最终答案对了就算好。这是一种非常实用的"弱监督"方式训练中间过程。
- **知识图谱驱动的数据合成**保证了数据的多跳推理质量和时间因果性，比简单让 LLM 生成 CoT 数据可靠得多。

## 局限与展望

- **Token 消耗**：streaming thought 是文本生成，额外消耗 LLM token。作者也提到可以探索 latent reasoning（在隐空间推理而非生成文本）来降低 token 开销
- **纯文本记忆**：长期记忆全是文本，丢失了视觉细节。未来可与视觉 KV-cache 管理方法结合
- **固定 thinking 频率**：每个 clip 都生成一段 thought，但有些 clip（如静态场景）可能不需要推理。自适应决定何时 think 可以进一步提升效率
- **数据合成依赖 Gemini**：pipeline 的质量上限受限于 Gemini 3.0 Flash 的能力

## 相关工作与启发

- **vs Video-R1**：Video-R1 在查询后做 CoT 推理，延迟 8.8s 但准确率只有 36.5%（VideoHolmes）。VST 预查询推理 + 延迟 0.56s + 准确率 41.9%，完胜。说明"何时推理"比"推理多少"更重要。
- **vs StreamForest/TimeChatOnline**：这些方法只做流式感知（视觉 token 管理），不做显式推理。VST 在此基础上加入思维流，StreamingBench +2.2%。
- **vs LongVILA-R1**：同样用 RL 增强视频推理，但 LongVILA-R1 是离线设置。VST 是首个将 RL 应用于流式视频理解的工作。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "thinking while watching"范式创新且自然，灵感来源于神经科学
- 实验充分度: ⭐⭐⭐⭐⭐ 5个基准、多个规模（3B/7B/32B）、详细消融、延迟分析、case study
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式规范，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 对流式视频理解领域有重要影响，提出了新的研究范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[CVPR 2026\] Time Blindness: Why Video-Language Models Can't See What Humans Can?](../../CVPR2026/video_understanding/time_blindness_why_video-language_models_cant_see_what_humans_can.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)

</div>

<!-- RELATED:END -->
