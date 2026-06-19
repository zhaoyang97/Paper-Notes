---
title: >-
  [论文解读] Sparse Rewards Can Self-Train Dialogue Agents
description: >-
  [ACL 2025 (Findings)][对话系统][自我训练] 提出 JOSH（Juxtaposed Outcomes for Simulation Harvesting）自对齐算法，让 LLM 对话 Agent 通过稀疏奖励的模拟环境自主提升性能，无需外部人类反馈，并构建了 ToolWOZ 稀疏奖励工具调用模拟环境加以验证。
tags:
  - "ACL 2025 (Findings)"
  - "对话系统"
  - "自我训练"
  - "稀疏奖励"
  - "对话Agent"
  - "工具调用"
  - "模拟环境"
---

# Sparse Rewards Can Self-Train Dialogue Agents

**会议**: ACL 2025 (Findings)  
**arXiv**: [2409.04617](https://arxiv.org/abs/2409.04617)  
**代码**: [GitHub](https://github.com/asappresearch/josh-llm-simulation-training)  
**领域**: 其他  
**关键词**: 自我训练, 稀疏奖励, 对话Agent, 工具调用, 模拟环境  

## 一句话总结

提出 JOSH（Juxtaposed Outcomes for Simulation Harvesting）自对齐算法，让 LLM 对话 Agent 通过稀疏奖励的模拟环境自主提升性能，无需外部人类反馈，并构建了 ToolWOZ 稀疏奖励工具调用模拟环境加以验证。

## 研究背景与动机

**领域现状**：LLM Agent 在多轮对话任务中的进步主要依赖两个支柱——监督微调（SFT）和高质量的人类反馈（RLHF/DPO）。这些方法通过人工标注的对话示例或偏好数据来训练模型。

**现有痛点**：随着基础 LLM 能力不断增强，获取有意义的人类反馈变得越来越困难和昂贵。在某些专业领域（如复杂工具调用、API 编排），基础 LLM 可能已经接近甚至超过普通人类标注员的水平，使得传统的人类反馈驱动方法变得不切实际。此外，人工标注对话数据（尤其是多轮带工具调用的对话）成本极高且难以保证一致性。

**核心矛盾**：模型需要大量高质量的训练信号来提升，但人类反馈的获取成本和质量正在成为瓶颈。能否让模型自己产生训练数据来自我提升？

**本文目标**：设计一种无需外部人类反馈的自我改进范式，让 LLM Agent 通过与模拟环境交互自主提升其多轮对话和工具调用能力。

**切入角度**：观察到很多对话任务可以定义明确的成功/失败条件（如：是否完成了预订、是否正确查询了信息），这种稀疏的二元奖励信号虽然简单，但已足够指导模型学习正确行为。关键是如何高效利用这种稀疏信号。

**核心 idea**：利用束搜索式的模拟交互生成多个对话轨迹，通过稀疏奖励筛选出成功轨迹作为正样本进行自训练（self-training），从而在不依赖人类反馈的情况下持续改进模型。

## 方法详解

### 整体框架

JOSH 的工作流程为：(1) LLM Agent 在模拟环境中与 User Simulator 进行多轮对话，同时尝试多个并行轨迹（beam search）；(2) 环境根据预设的目标条件给出稀疏奖励（成功/失败）；(3) 收集成功轨迹作为训练数据；(4) 用这些自生成的高质量数据对 LLM 进行微调。整个过程无需人类参与。

### 关键设计

1. **JOSH 搜索算法（Juxtaposed Outcomes for Simulation Harvesting）**:

    - 功能：从模拟对话中高效提取成功行为轨迹
    - 核心思路：在每一步对话交互中，JOSH 不只生成一个 agent 回复，而是并行生成多个候选回复（类似 beam search），每个候选形成一个独立的对话分支。对话结束后，环境给出稀疏的二元奖励（完成目标=1，失败=0）。JOSH 保留所有成功的对话分支作为正样本。通过增大 beam size 可以增加找到成功轨迹的概率，即使模型初始能力较弱也能收集到足够的正样本
    - 设计动机：传统 self-play 方法在 LLM 场景中效率低下，因为失败案例占多数。JOSH 通过并行搜索大幅提高了找到成功案例的效率

2. **ToolWOZ 模拟环境**:

    - 功能：提供一个标准化的稀疏奖励工具调用对话模拟器
    - 核心思路：基于经典的 MultiWOZ 对话数据集构建，将其改造为一个可交互的工具调用模拟环境。环境包含用户模拟器（给出对话目标和回复）、工具/API 接口（如酒店预订、餐厅查询）和奖励函数（检查是否正确完成了所有子任务）。奖励策略是稀疏的——只在整个对话结束时给出一个总体的成功/失败信号，中间步骤不提供反馈
    - 设计动机：现有的对话训练基准多依赖稠密的逐步反馈，不符合现实场景。稀疏奖励更接近真实部署（用户只在最终表达满意/不满），从这种信号中学习更有实际价值

3. **偏好标注与 LoRA 微调策略**:

    - 功能：将搜索到的成功和失败轨迹转化为模型更新信号
    - 核心思路：JOSH 收集的对话轨迹自然形成了偏好对（preference pairs）——同一个对话起点出发的成功分支和失败分支。使用这些偏好对通过 DPO/KTO 等偏好优化算法微调模型。同时采用 LoRA 进行参数高效微调，避免全参数更新导致的一般能力退化
    - 设计动机：直接用 SFT 只利用正样本，而偏好优化能同时从正负样本中学习，信息利用更充分

### 损失函数 / 训练策略

采用 KTO（Kahneman-Tversky Optimization）损失进行偏好学习，结合 LoRA 的参数高效微调。训练数据完全由 JOSH 自动生成，不需要人工标注。多轮迭代：生成 → 筛选 → 训练 → 再生成，形成自我改进的循环。

## 实验关键数据

### 主实验

| 模型/方法 | ToolWOZ 成功率 | τ-bench 成功率 | MT-Bench 分数 | 说明 |
|-----------|---------------|---------------|--------------|------|
| GPT-4o-mini (baseline) | 基线 | 基线 | 基线 | 未经过自训练 |
| GPT-4o-mini + JOSH | 显著提升 | 显著提升 | 保持 | Frontier 模型也获益 |
| LLaMA-3-8B (baseline) | 较低 | 较低 | 基线 | 小模型初始能力弱 |
| LLaMA-3-8B + JOSH | 大幅提升 | 大幅提升 | 保持 | 提升幅度最明显 |
| LLaMA-3-8B + SFT only | 中等提升 | 中等提升 | 略降 | 只用正样本效果有限 |

### 消融实验

| 配置 | ToolWOZ 成功率 | 说明 |
|------|---------------|------|
| JOSH + KTO | 最优 | 完整方案 |
| JOSH + SFT (仅正样本) | 次优 | 不利用负样本 |
| 无 JOSH (直接生成) | 差 | 低 beam size 无法找到足够正样本 |
| Beam size = 2 | 低 | 搜索空间不足 |
| Beam size = 8 | 最优 | 足够多的并行探索 |
| 无 LoRA (全参数) | 下降 | 过拟合 + 一般能力退化 |

### 关键发现

- JOSH 在小模型（LLaMA-3-8B）上的提升幅度远大于大模型（GPT-4o-mini），因为小模型有更大的改进空间
- Beam size 对性能影响显著，size=8 是效果和效率的平衡点
- JOSH 训练后模型的 MT-Bench 分数保持不变，说明工具调用能力的提升不以牺牲通用对话能力为代价
- τ-bench（另一个工具调用基准）上的跨数据集泛化验证了方法的通用性

## 亮点与洞察

- **无需人类反馈的自我改进**：JOSH 打破了对人类标注的依赖，通过模拟环境中的自我博弈实现持续进步。这在人类反馈成本日益增高的背景下尤其有价值
- **稀疏奖励的高效利用**：只用成功/失败的二元信号就能引导学习，避免了设计复杂奖励函数的需求。这种"环境当老师"的思路可以迁移到其他任务型对话场景
- **跨基准泛化**：在 ToolWOZ 上训练的策略能直接迁移到 τ-bench，说明 JOSH 学到的不是环境特定的 trick，而是通用的工具调用能力

## 局限与展望

- ToolWOZ 基于 MultiWOZ 构建，领域局限于酒店、餐厅等预定场景，更开放域的对话任务需要新的环境
- 稀疏奖励假设对话目标可以明确定义并自动评估，这在开放式闲聊等场景中不适用
- JOSH 的 beam search 过程计算成本较高，尤其在使用大模型时数倍于标准推理
- 当前只验证了单轮 JOSH 迭代的效果，多轮自我改进是否存在上限或退化尚不清楚
- 未来可探索将 JOSH 扩展到更复杂的 Agent 场景（如代码生成、浏览器操作）

## 相关工作与启发

- **vs Self-Play (AlphaGo 式)**: AlphaGo 的自我博弈需要完美的环境模型，JOSH 放宽了这个要求，仅需稀疏奖励的模拟器
- **vs RLHF/DPO**: 传统偏好学习依赖人类标注偏好对，JOSH 通过模拟环境自动生成偏好对，完全去人工
- **vs ReST/STaR**: 这些自训练方法主要针对推理任务（如数学），JOSH 是首个系统性地将其推广到多轮工具调用对话的工作
- 该工作为 Agent 自我训练提供了一套可复用的框架（JOSH 库已开源），可以作为后续 Agent 训练研究的 baseline

## 评分

- 新颖性: ⭐⭐⭐⭐ 自训练+稀疏奖励用于对话Agent是新颖组合，但各组件并非全新
- 实验充分度: ⭐⭐⭐⭐ 多模型、多基准验证，消融详细，但缺少多轮迭代分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，ToolWOZ 构建过程详尽
- 价值: ⭐⭐⭐⭐ 为无人类反馈的Agent改进提供了实用方案，开源完善

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/dialogue/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](../../ACL2026/dialogue/dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[ICML 2025\] Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents](../../ICML2025/dialogue/position_uncertainty_quantification_needs_reassessment_for_large-language_model_.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ACL 2025\] Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)

</div>

<!-- RELATED:END -->
