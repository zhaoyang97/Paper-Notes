---
title: >-
  [论文解读] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning
description: >-
  [NeurIPS 2025 Spotlight][强化学习][搜索强度缩放] 提出 DeepDiver，一个 RL 驱动的搜索推理框架，在真实开放网络环境中训练 LLM 的信息寻求能力，催生"搜索强度缩放"（SIS）涌现行为——7B 模型在知识密集任务上可媲美 671B 的 DeepSeek-R1。
tags:
  - "NeurIPS 2025 Spotlight"
  - "强化学习"
  - "搜索强度缩放"
  - "信息检索"
  - "大语言模型"
  - "开放网络问答"
---

# DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.24332](https://arxiv.org/abs/2505.24332)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 搜索强度缩放, 强化学习, 信息检索, 大语言模型, 开放网络问答

## 一句话总结

提出 DeepDiver，一个 RL 驱动的搜索推理框架，在真实开放网络环境中训练 LLM 的信息寻求能力，催生"搜索强度缩放"（SIS）涌现行为——7B 模型在知识密集任务上可媲美 671B 的 DeepSeek-R1。

## 研究背景与动机

信息寻求（information seeking）是一项核心认知能力，涉及迭代证据收集、反思推理和冲突信息解决。现有 LLM 在这方面面临以下问题：

**Prompting 方法固定流程限制**：ReAct、IRCoT 等方法使用预定义规则，无法适应动态复杂问题

**SFT 方法过拟合训练语料**：SELF-RAG 等方法内化了特定语料的推理模式，泛化性差

**RL 方法仅在"干净"环境评估**：R1-Searcher、DeepResearcher 等仅在 HotpotQA/Wikipedia 等结构化数据上训练评估，真实网络充满噪声和冲突信息

**四种信息寻求行为未被完整覆盖**：
   - 证据收集与补充（传统 QA 数据集主要覆盖此类）
   - **冲突解决**（处理矛盾信息）
   - **验证与去噪**（交叉检查事实）
   - **反思与纠正**（重新评估推理路径）

后三种行为在真实网络搜索中至关重要，但 Wiki-based 环境无法激发。

## 方法详解

### 整体框架

DeepDiver 采用 **Cold-start SFT → RL 训练** 的两阶段流程，在真实搜索引擎环境中训练 LLM 的迭代检索-推理能力。

### 关键设计

#### 1. WebPuzzle 数据集

一个 24K 训练 + 275 测试的开放网络 QA benchmark，覆盖 Wiki 和开放域查询：

| 数据类型 | 生成方式 | 特点 |
|---|---|---|
| Cross-Page QA | 从网页提取事实生成"反转"问题 | 需跨页推理 |
| Open Riddle | 选择实体属性进行模糊化/泛化 | 高度挑战性 |
| Wiki Riddle | 同上但基于 Wiki 来源 | 有结构化知识 |

难度标注：用 DeepSeek-R1 测试 4 次，按正确次数分 easy/medium/hard，确保 RL 训练奖励信号稳定。测试集由 5 位专家手工标注。

#### 2. Cold-start SFT 初始化

使用多样数据蒸馏 DeepSeek-R1 的响应：
- 2000 条 WebPuzzle 样本（跨难度）
- 300 条真实用户查询
- 2200 条通用推理问题
- 1000 条用户查询 + 检索文档

#### 3. GRPO + 迭代 RAG

在每轮迭代中，模型交替执行推理和搜索直到产生答案。关键设计：
- **Loss mask**：仅对模型生成的 token 计算 GRPO 损失，检索文本不参与梯度更新
- **额外搜索奖励**：当无搜索 rollout 均失败但有搜索 rollout 成功时，给搜索成功的 rollout 额外 +1.0 奖励
- **松严奖励过渡**：前 80 步用宽松评分（10 分制，≥6 得 1.0），后续切换严格评分（3 轮评估，≥2/3 正面）

#### 4. 搜索强度缩放（SIS）

SIS 是 DeepDiver 的核心涌现能力——模型**自适应地增加搜索频率和深度**以应对更复杂的问题。论文通过训练过程分析证明 SIS 是涌现行为而非奖励工程的产物：

- 额外搜索奖励的触发频率从 step 0-9 的 4.5% 骤降到 step 70-80 的 0.1%
- 搜索轮次的增长发生在 step 80-120，此时额外奖励已基本不活跃
- 模型主动利用外部工具弥补内部知识不足，无需直接激励

### 损失函数 / 训练策略

- GRPO 优势估计：$A_i = r_i - \text{mean}(r)$（组内相对奖励）
- 检索文本 mask：仅模型生成部分贡献梯度
- 训练数据：从 24K WebPuzzle 中精选 7K（2K SFT + 5K RL），按难度均衡混合
- 骨干模型：Qwen2.5-7B-Instruct 和 Pangu-7B-Reasoner

## 实验关键数据

### 主实验：与基线对比

| 方法 | WebPuzzle | C-SimpleQA-500 | FRAMES-zh | BamBoogle-zh |
|---|---|---|---|---|
| Qwen2.5-7B (无搜索) | 7.4 | 28.4 | 14.1 | 19.7 |
| Qwen2.5-7B (迭代RAG) | 17.0 (2.24轮) | 65.3 | 30.9 | 40.8 |
| Cold-Start-SFT | 27.9 (1.85轮) | 75.5 | 35.1 | 48.4 |
| R1-Distill | 29.8 (1.75轮) | 78.7 | 40.1 | 52.6 |
| **DeepDiver-Qwen7B** | **37.6 (2.51轮)** | **81.9** | **44.5** | **63.4** |
| DeepSeek-R1 (迭代RAG) | 37.1 (1.48轮) | 84.8 | 65.8 | 79.3 |

7B 的 DeepDiver 在 WebPuzzle 开放域任务上**超越 671B 的 DeepSeek-R1**（37.6 vs 37.1）。

### 与 Wiki-based 方法对比（英语评测）

| 方法 | WebPuzzle-en | BamBoogle | FRAMES | HotpotQA |
|---|---|---|---|---|
| R1-Searcher | 13.7 (1.9轮) | 46.7 | 25.3 | 57.9 |
| DeepResearcher | 15.0 (7.5轮) | 53.9 | 33.6 | 56.6 |
| **DeepDiver-Qwen** | **26.1 (14.7轮)** | **56.8** | 32.0 | **58.4** |

尽管仅用中文训练，DeepDiver 在英语开放域任务上大幅超越 Wiki-based 方法。

### 信息寻求能力隔离测试

去除仅靠内部知识就能回答的问题后：
- DeepDiver 在所有领域**超越 DeepSeek-R1**，WebPuzzle 领先 5.1 分
- 7B 模型在全集上的劣势主要源于内部知识量不足，而非信息寻求能力

### 消融实验：奖励函数设计

| 策略 | WebPuzzle 变化 | FRAMES-zh 变化 |
|---|---|---|
| 持续宽松奖励 | 几乎无提升 | 下降 7 分 |
| 宽松→严格过渡 | **+9 分**（29.1→37.6） | 持续上升 |

### 关键发现

1. **搜索强度与性能正相关**：搜索轮次增加伴随训练奖励上升，SIS 能力使模型动态调整搜索深度
2. **开放网络训练增强泛化**：WebPuzzle 训练的模型在 Wiki-based 测试上也表现优异
3. **SIS 是涌现行为**：不是奖励工程的产物，额外搜索奖励仅是短暂的早期脚手架
4. **从封闭到开放的泛化**：在 ProxyQA 长文写作任务上，DeepDiver 超越 R1 蒸馏模型 9.47 分

## 亮点与洞察

1. **问题定义精准**：将信息寻求行为分为四类（证据收集、冲突解决、验证去噪、反思纠正），并论证了 Wiki-based 环境的局限性
2. **SIS 涌现的严谨验证**：通过追踪额外奖励触发频率的衰减，令人信服地证明 SIS 是涌现而非工程产物
3. **松严奖励过渡的实用洞察**：早期用宽松奖励稳定训练，后期用严格奖励突破瓶颈——这一策略对 RL 训练具有普遍参考价值
4. **跨语言泛化**：仅用中文训练却能在英语测试中表现优异，说明信息寻求能力具有语言无关性

## 局限与展望

1. **7B 模型内部知识有限**：全集性能受限于参数量导致的知识量不足，更大模型可能获得更好效果
2. **搜索引擎依赖**：性能受限于搜索引擎质量和可用性
3. **计算成本高**：DeepDiver 的搜索轮次（平均 2.5+ 轮，每轮最多 15 次查询）远高于基线，推理成本显著增加
4. **评测依赖 LLM 评审**：虽然采用了严格/宽松双模式，但 LLM-as-judge 本身存在偏差

## 相关工作与启发

- **与 DeepSeek-R1 的技术关联**：同样使用 GRPO，但扩展到迭代 RAG 场景，证明 GRPO 在工具使用训练中也有效
- **与 R1-Searcher/DeepResearcher 的关键差异**：使用真实开放网络而非 Wiki 环境训练，培养出更强的信息寻求能力
- **SIS 的启示**：类似于 test-time compute scaling 的思想，但作用于搜索维度——难题需要更多搜索，简单题则快速回答

## 评分

- 新颖性: ⭐⭐⭐⭐ (SIS 概念新颖，WebPuzzle 填补数据集空白)
- 实验充分度: ⭐⭐⭐⭐⭐ (隔离测试、消融、跨语言、跨领域泛化分析极其详尽)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，分析逻辑严密)
- 价值: ⭐⭐⭐⭐⭐ (对 LLM+搜索的 RL 训练有重要指导意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[NeurIPS 2025\] ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[ICLR 2026\] Adaptive Scaling of Policy Constraints for Offline Reinforcement Learning](../../ICLR2026/reinforcement_learning/adaptive_scaling_of_policy_constraints_for_offline_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
