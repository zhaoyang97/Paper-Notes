---
title: >-
  [论文解读] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents
description: >-
  [ACL 2025][多模态VLM][reward modeling] 本文提出Agent-RewardBench，首个评估多模态LLM作为agent奖励模型能力的基准，覆盖感知/规划/安全三个维度和7个真实场景，包含1,136条高质量step-level样本，实验揭示即使最强模型GPT-4o也仅达61.4%准确率，且强模型在安全维度反而表现更差。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "reward modeling"
  - "多模态"
  - "benchmark"
  - "safety"
  - "planning"
---

# Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents

**会议**: ACL 2025  
**arXiv**: [2506.21252](https://arxiv.org/abs/2506.21252)  
**代码**: 有 (GitHub)  
**领域**: 多模态VLM  
**关键词**: reward modeling, multimodal agent, benchmark, safety, planning

## 一句话总结
本文提出Agent-RewardBench，首个评估多模态LLM作为agent奖励模型能力的基准，覆盖感知/规划/安全三个维度和7个真实场景，包含1,136条高质量step-level样本，实验揭示即使最强模型GPT-4o也仅达61.4%准确率，且强模型在安全维度反而表现更差。

## 研究背景与动机
1. **领域现状**：多模态agent在网页导航、具身智能等任务中展现潜力，目前主流增强方法是模仿学习（SFT），使用专家标注轨迹进行微调。近期研究开始探索用奖励模型（RM）提供反馈来提升agent能力。
2. **现有痛点**：虽然奖励模型在指导agent训练和搜索中至关重要，但目前没有基准来评估MLLM作为agent奖励模型的能力。现有奖励基准（如RewardBench）聚焦于对话/数学/检索场景，不覆盖agent特有的感知、规划和安全能力。
3. **核心矛盾**：agent任务需要多维度的奖励反馈（感知是否准确、规划是否合理、行为是否安全），但我们不知道哪些MLLM适合做agent的奖励模型，也不知道它们在各维度上的能力差异。
4. **本文目标** (1) 构建覆盖多维度、多场景的agent奖励基准；(2) 支持step-level奖励评估而非仅评估最终结果；(3) 通过难度控制和人工验证确保数据质量。
5. **切入角度**：从agent任务的三个核心能力维度（感知、规划、安全）出发，收集7个真实场景数据，通过10个模型采样响应+小模型难度过滤+人工验证的三阶段流程构建benchmark。
6. **核心 idea**：首次系统评估MLLM在agent任务中的奖励建模能力，揭示"强模型≠好奖励模型"的反直觉发现。

## 方法详解

### 整体框架
输入是来自7个真实agent场景的任务prompt，对每个中间步骤从10个不同MLLM采样多个响应，构建正负样本对 $(r^+, r^-)$，经过小模型难度过滤和人工验证后得到1,136条高质量评估样本。评估时让目标MLLM判断哪个响应更好。

### 关键设计

1. **三维度七场景的数据来源设计**:

    - 功能：全面覆盖agent任务所需的核心能力
    - 核心思路：感知维度从SeeClick（web/mobile/desktop视觉定位）和MFE-ETP（具身空间感知）选取数据；规划维度从Mind2Web（网页多步规划）、PCA（Minecraft/自动驾驶/虚拟家庭）和TravelPlanner（旅行规划）选取；安全维度从弹窗攻击场景和MSSBench（具身安全）选取。共初始约1,682条样本
    - 设计动机：agent奖励模型需要同时具备视觉理解、序列决策和安全对齐能力，仅评估单一维度不足以反映真实应用需求

2. **Step-level奖励评估**:

    - 功能：在任务的每个中间步骤进行细粒度奖励评估
    - 核心思路：对agent任务的每个步骤分别采样响应，构建该步骤的正负样本对，让被评估模型判断哪个步骤响应更好。这比仅评估最终结果提供了更细致的反馈
    - 设计动机：agent规划具有明确的步骤划分，step-level评估能发现模型在规划过程中的具体薄弱环节

3. **三阶段数据构建流程（采样→难度控制→人工验证）**:

    - 功能：确保评估数据质量高且难度适中
    - 核心思路：首先从5个闭源+5个开源模型采样响应，每个query采样10对正负样本；然后用3个小模型（Pixtral-12B、LLaVA-OneVision-7B、InternVL2-8B）进行双向难度过滤，去掉过难和过易的样本；最后3名AI方向研究生进行人工验证，剔除标注错误的样本。原始1,443条经过滤后保留1,136条
    - 设计动机：过易的数据区分度低，过难的数据可能有标注噪声——难度控制确保基准在有效区分不同模型能力的同时保持高质量

## 实验关键数据

### 主实验

| 模型 | 感知Avg | 规划Avg | 安全Avg | **总体Avg** |
|------|--------|--------|--------|-----------|
| gemini-1.5-pro | 73.4 | 69.6 | 37.7 | **61.6** |
| gpt-4o | 65.9 | 73.2 | 39.2 | **61.4** |
| claude-3.5-sonnet | 73.3 | 71.2 | 22.4 | **57.9** |
| Qwen2-VL-72B | 69.1 | 60.1 | 34.3 | **55.3** |
| gemini-1.5-flash | 66.1 | 64.7 | 47.8 | **60.2** |
| Qwen2-VL-7B | 57.5 | 51.8 | 38.7 | 49.7 |
| Llama-3.2-11B | 53.5 | 50.6 | 38.0 | 47.8 |

### 安全维度细分

| 模型 | Web安全 | 具身安全 | 安全Avg |
|------|--------|---------|--------|
| gemini-1.5-flash | 26.0 | **69.5** | **47.8** |
| gpt-4o | 17.5 | 61.0 | 39.2 |
| claude-3.5-sonnet | 15.0 | 29.9 | 22.4 |
| gpt-4o-mini | **35.0** | 56.7 | 45.9 |

### 关键发现
- 即使最强闭源模型（gemini-1.5-pro），在Agent-RewardBench上也仅达61.6%准确率，说明agent奖励建模仍是一个巨大挑战
- **强模型不等于强安全奖励模型**：GPT-4o总体排名靠前，但安全维度仅39.2%；Claude-3.5-Sonnet安全更低至22.4%。反而gpt-4o-mini在Web安全上更好（35.0% vs GPT-4o的17.5%）
- 开源模型（如Llama-3.2-11B）在感知（53.5%）和规划（50.6%）上接近随机水平，说明专门的agent奖励训练必不可少
- 规划维度上，GPT-4o（73.2%）显著优于其他模型，但在具身场景中的规划能力（68.2%）弱于旅行规划（76.2%），说明视觉+物理推理对规划的额外挑战

## 亮点与洞察
- "强模型≠好安全奖励模型"是本文最重要的反直觉发现——这暗示安全对齐需要专门的训练策略，不能简单依赖模型通用能力的提升
- Step-level评估的设计思路非常实用——可以直接迁移到LLM推理过程的reward modeling中（如process reward model的评估）
- 三阶段数据构建流程（多模型采样→小模型过滤→人工验证）提供了构建高质量benchmark的通用范式

## 局限与展望
- 仅评估了奖励建模的判别能力（选择更好的响应），未评估生成式奖励（给出分数或文本反馈）
- 安全维度样本量较少（Web安全仅100条、具身安全82条），可能影响统计稳定性
- 未验证Agent-RewardBench得分与实际agent性能提升之间的相关性（即奖励模型好是否真的能带来更好的agent）
- 数据来源偏向英语场景，跨语言agent的奖励评估未覆盖
- 未探索组合多个弱奖励模型来获得更好的奖励信号（ensemble策略）
- benchmark的更新机制未讨论——随着模型进步，数据集可能需要定期更新以保持区分度

## 相关工作与启发
- **vs RewardBench (Zhou et al., 2024)**: RewardBench评估chat/math/retrieval场景，Agent-RewardBench专注agent场景且增加了感知、安全维度和step-level评估
- **vs Mind2Web**: Mind2Web是agent规划的评估基准，本文将其数据转化为奖励建模评估，视角从"agent能力"转向"奖励模型能力"
- **vs Differential Prompting for agents**: 本文揭示了模仿学习的瓶颈，为奖励模型引导的agent训练提供了评估基础设施
- Agent-RewardBench的安全维度发现可指导未来agent安全对齐研究的重点方向

## 评分
- 总体评价: 开创性工作，为agent领域从RL角度提供了关键评估设施，安全维度发现尤其重要
- 新颖性: ⭐⭐⭐⭐⭐ 首个专注agent奖励建模的基准，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 覆盖8个模型的全面评估，但安全维度样本量偏少
- 写作质量: ⭐⭐⭐⭐ 结构清晰，motivation阐述充分
- 价值: ⭐⭐⭐⭐⭐ 对agent领域从SFT向RL过渡提供了关键评估工具

<!-- 覆盖: 3维度(perception/planning/safety), 7场景, 1136条样本, 8个模型评测 -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)

</div>

<!-- RELATED:END -->
