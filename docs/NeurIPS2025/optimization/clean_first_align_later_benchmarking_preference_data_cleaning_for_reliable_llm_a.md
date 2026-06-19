---
title: >-
  [论文解读] Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment
description: >-
  [NeurIPS 2025 (D&B Track)][优化/理论][偏好数据清洗] 本文提出 PrefCleanBench，首个系统评估 13 种偏好数据清洗方法在 LLM 对齐中效果的综合基准，覆盖多种数据集、模型架构和优化算法，揭示了数据预处理在负责任 AI 开发中被忽视但至关重要的角色。 领域现状：人类反馈在 LLM…
tags:
  - "NeurIPS 2025 (D&B Track)"
  - "优化/理论"
  - "偏好数据清洗"
  - "LLM对齐"
  - "RLHF"
  - "DPO"
  - "benchmark"
---

# Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment

**会议**: NeurIPS 2025 (D&B Track)  
**arXiv**: [2509.23564](https://arxiv.org/abs/2509.23564)  
**代码**: [https://github.com/deeplearning-wisc/PrefCleanBench](https://github.com/deeplearning-wisc/PrefCleanBench)  
**领域**: 优化  
**关键词**: 偏好数据清洗, LLM对齐, RLHF, DPO, benchmark

## 一句话总结

本文提出 PrefCleanBench，首个系统评估 13 种偏好数据清洗方法在 LLM 对齐中效果的综合基准，覆盖多种数据集、模型架构和优化算法，揭示了数据预处理在负责任 AI 开发中被忽视但至关重要的角色。

## 研究背景与动机

**领域现状**：人类反馈在 LLM 对齐中扮演核心角色——无论是 RLHF 中训练奖励模型、还是 DPO 等直接优化算法，都依赖高质量的偏好数据（即 chosen/rejected 配对数据）。当前主流对齐方法包括 DPO、CPO、KTO、IPO、SLiC、rDPO、ORPO、AOT 等多种优化算法。

**现有痛点**：
   - 人类反馈固有地存在噪声和不一致性——标注者之间意见分歧、标注质量参差不齐
   - 有噪声的偏好数据会降低奖励模型质量，阻碍对齐效果
   - 虽然已有多种自动化数据清洗方法被提出，但它们分散在不同论文中，缺乏统一评估框架
   - 不同清洗方法的效果在不同场景下表现各异，没有系统性的对比研究

**核心矛盾**：数据清洗方法众多但缺乏统一评估标准——每个方法都声称有效，但在什么条件下有效、对什么模型和算法有效，仍不清楚

**本文解决什么**：
   - 建立统一的偏好数据清洗评估协议
   - 系统对比 13 种清洗方法的效果
   - 发现决定清洗成功与否的关键因素
   - 评估清洗方法的泛化性（跨数据集、跨模型、跨算法）

**切入角度**：从 benchmark 视角出发，不提出新的清洗方法，而是对现有方法进行公平、全面的比较

**核心idea**：建立首个标准化的 LLM 对齐偏好数据清洗基准 PrefCleanBench，通过控制变量实验揭示数据质量在对齐中的关键作用。

## 方法详解

### 整体框架

PrefCleanBench 的评估 pipeline 分为四个阶段：
1. **数据集准备**：下载和处理目标数据集（AnthropicHH、UltraFeedback、PKU-SafeRLHF、HelpSteer2）
2. **数据清洗**：对每个数据集应用 13 种清洗方法，生成多个清洗版本
3. **训练与生成**：在清洗/未清洗数据上训练对齐模型，生成回复
4. **评估**：使用 win-tie rate 和 avg. gold reward 评估对齐效果

### 关键设计

**13 种清洗方法分类**：PrefCleanBench 涵盖 5 大类清洗策略，细分为 13 种具体方法：

1. **LLM Judge**：使用大语言模型（如 GPT-4）作为裁判，评估偏好对的质量

    - `llm_judge_r`：移除低质量对（reject 策略）
    - `llm_judge_f`：翻转标签（flip 策略）

2. **Reward Gap (RwGap)**：利用奖励模型计算 chosen 和 rejected 之间的奖励差距

    - `rw_gap_r_{0.1-0.4}`：按不同阈值移除低差距对
    - `rw_gap_f_{0.1-0.4}`：翻转差距为负的对

3. **Voting**：多个模型或标注者投票决定偏好

    - `vote_all_r/f`：全票一致才保留/翻转
    - `vote_maj_r/f`：多数票决定

4. **InsTag**：基于指令标签的质量过滤

    - `ins_tag_cmp`：复杂度过滤
    - `ins_tag_div`：多样性过滤

5. **IFD (Instruction Following Difficulty)**：基于指令遵循难度的过滤

    - `ifd_r_{0.1-0.4}`：按阈值移除
    - `ifd_gap_r/f_{0.1-0.4}`：基于 IFD 差距的移除/翻转

**评估维度**：

| 维度 | 选项 |
|------|------|
| 数据集 | AnthropicHH, UltraFeedback, PKU-SafeRLHF, HelpSteer2 |
| 基础模型 | Llama-3-8B, Qwen2.5-7B, Phi-2, Mistral-7B-v0.3 |
| 优化算法 | DPO, CPO, AOT, KTO, IPO, SLiC, rDPO, ORPO |

**评估指标**：
- **Win-Tie Rate**：清洗后模型 vs 未清洗模型的正面对比胜率
- **Avg. Gold Reward**：使用独立的黄金标准奖励模型评估生成回复质量

### 损失函数 / 训练策略

本文不提出新的训练方法，而是使用标准的对齐训练流程。每种优化算法使用其原始论文的训练配置。核心创新在于系统地控制变量：固定模型架构和优化算法，只改变数据清洗策略，从而公平比较清洗效果。

## 实验关键数据

### 主实验

不同清洗方法在 DPO + Llama-3-8B 上的效果（win-tie rate 示例）：

| 清洗方法 | AnthropicHH | UltraFeedback | PKU-SafeRLHF | HelpSteer2 | 平均 |
|---------|-------------|---------------|-------------|------------|------|
| No Clean (baseline) | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| LLM Judge (r) | 较高 | 中等 | 较高 | 中等 | >50% |
| RwGap (r, 0.2) | 中等 | 较高 | 中等 | 中等 | ~50% |
| Voting (maj, r) | 中等 | 中等 | 较高 | 低 | ~50% |
| IFD Gap (r, 0.2) | 低 | 较高 | 中等 | 较高 | ~50% |

跨优化算法一致性：

| 优化算法 | 清洗是否普遍有益 | 最佳清洗方法 | 最差清洗方法 |
|---------|----------------|------------|------------|
| DPO | 部分情况 | 视数据集而定 | 过度清洗 |
| CPO | 部分情况 | LLM Judge 系列 | - |
| KTO | 较一致 | RwGap 系列 | - |
| IPO | 不一致 | - | - |

### 消融实验

**清洗比例的影响**（以 RwGap 为例）：

| 清洗阈值 | 数据保留率 | DPO效果 | CPO效果 | KTO效果 |
|---------|-----------|---------|---------|---------|
| 0.1 | ~90% | 小幅提升 | 持平 | 小幅提升 |
| 0.2 | ~80% | 适中提升 | 小幅提升 | 适中提升 |
| 0.3 | ~70% | 效果不一 | 效果不一 | 下降 |
| 0.4 | ~60% | 下降 | 下降 | 下降 |

### 关键发现

1. **没有万能的清洗方法**：没有单一清洗方法在所有数据集、模型和算法组合上都表现最好
2. **过度清洗有害**：移除过多数据（>30%）通常导致对齐效果下降，数据量的减少可能比数据质量的提升带来更大的负面影响
3. **清洗效果与优化算法强相关**：某些清洗方法在 DPO 上有效但在 KTO 上无效，反之亦然
4. **LLM Judge 类方法相对稳健**：基于 LLM 裁判的清洗方法在多数场景下表现较好，但成本较高
5. **数据集特性影响显著**：高噪声数据集（如 AnthropicHH）从清洗中受益更多，而高质量数据集（如 HelpSteer2）清洗带来的增益有限
6. **泛化性挑战**：在一个数据集上最优的清洗策略在另一个数据集上可能次优甚至有害

## 亮点与洞察

- **首个系统性基准**：填补了偏好数据清洗领域缺乏统一评估的空白
- **实验设计严谨**：通过控制变量策略（固定模型/算法/数据两个维度，只变一个）实现公平比较
- **模块化代码发布**：开源了所有 13 种方法的模块化实现，降低了后续研究的门槛
- **实践指导价值**：为从业者选择合适的清洗方法提供了实证依据
- **强调数据预处理在负责任 AI 中的角色**：这一视角在当前以算法创新为主导的研究中较为稀缺

## 局限与展望

- 仅评估了开源模型（最大 8B 参数），更大模型（70B+）的表现可能不同
- 评估指标（win-tie rate 和 gold reward）可能不能完全捕捉对齐质量的所有维度（如安全性、事实性）
- 未涉及多轮对话场景的偏好数据清洗
- 清洗方法的计算成本对比不够充分（LLM Judge 的 API 成本 vs 轻量级方法）
- 未探索多种清洗方法的组合使用

## 相关工作与启发

- **RLHF / DPO 系列**：偏好对齐的主流方法，本文评估的下游应用
- **数据中心 AI（Data-Centric AI）**：强调数据质量而非模型创新的研究范式
- **课程学习 / 数据选择**：从训练数据中选择最有价值子集的相关工作
- **奖励建模**：奖励模型质量直接受偏好数据质量影响，是数据清洗的直接受益者

## 评分

- **新颖性**: ⭐⭐⭐ — 不提出新方法，但首次系统评估是重要贡献
- **技术深度**: ⭐⭐⭐ — 方法论简单但实验设计严谨
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4数据集 × 4模型 × 8算法 × 13方法的组合覆盖非常全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，发现有实践指导价值
- **实用性**: ⭐⭐⭐⭐⭐ — 对从事 LLM 对齐的研究者和工程师直接有用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[ICML 2025\] Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift](../../ICML2025/optimization/right_now_wrong_then_non-stationary_direct_preference_optimization_under_prefere.md)
- [\[NeurIPS 2025\] Towards Reliable and Holistic Visual In-Context Learning Prompt Selection](towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)

</div>

<!-- RELATED:END -->
