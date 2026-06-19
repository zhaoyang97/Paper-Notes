---
title: >-
  [论文解读] SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning
description: >-
  [ACL 2025][Issue解决] 提出 SoRFT（Subtask-oriented Reinforced Fine-Tuning），将 GitHub Issue 解决任务分解为文件定位、函数定位、行定位和代码编辑四个子任务，通过拒绝采样SFT + 基于规则的PPO强化学习两阶段训练，显著提升开源LLM在 SWE-Bench 上的 Issue 解决能力。
tags:
  - "ACL 2025"
  - "Issue解决"
  - "强化微调"
  - "子任务分解"
  - "代码生成"
  - "SWE-Bench"
---

# SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning

**会议**: ACL 2025  
**arXiv**: [2502.20127](https://arxiv.org/abs/2502.20127)  
**代码**: 无  
**领域**: 其他  
**关键词**: Issue解决, 强化微调, 子任务分解, 代码生成, SWE-Bench

## 一句话总结

提出 SoRFT（Subtask-oriented Reinforced Fine-Tuning），将 GitHub Issue 解决任务分解为文件定位、函数定位、行定位和代码编辑四个子任务，通过拒绝采样SFT + 基于规则的PPO强化学习两阶段训练，显著提升开源LLM在 SWE-Bench 上的 Issue 解决能力。

## 研究背景与动机

当前主流的 Issue 解决框架（如 Agentless、OpenHands）主要依赖商业模型（如 Claude-3.5-Sonnet、GPT-4o），存在高成本和隐私泄露问题。已有的开源模型训练方法仅依赖监督微调（SFT），容易导致泛化能力差、模型更容易产生幻觉和事实错误。

近期 DeepSeek-R1 等工作表明，基于规则的强化学习可以有效提升模型在数学等复杂任务上的表现。而开源社区中存在大量已解决的 Issue 及其 Pull Request，天然包含 ground-truth patch，这为构建基于规则的奖励信号提供了理想条件。核心问题是：能否利用这些 (issue, patch) 对进行基于规则的强化学习，提升开源 LLM 的 Issue 解决能力？

## 方法详解

### 整体框架

SoRFT 包含三个核心部分：
1. Issue 解决任务的子任务分解
2. 拒绝采样监督微调（Rejection-sampled SFT）
3. 基于规则的强化学习（Rule-based RL）

整体流程为：先将 Issue 解决分解为四个子任务并构建训练数据，使用 teacher LLM 采样 CoT 数据后过滤负样本进行 SFT，随后使用 PPO + 规则奖励进行强化学习。

### 关键设计

1. **子任务分解（Subtask Decomposition）**: 将 Issue 解决分解为四个阶梯式子任务：文件定位（根据 Issue 和仓库结构定位修改文件）、函数定位（根据文件骨架定位修改函数）、行定位（精确定位需修改的代码行）、代码编辑生成（生成 Search/Replace 格式的代码修改）。每个子任务的 ground-truth 均从对应 PR 中提取，使得训练信号构建更加清晰。

2. **拒绝采样 SFT**: 使用 teacher LLM（Claude-3.5-Sonnet）对每个子任务采样 CoT 数据，然后基于 ground-truth 过滤负样本——对定位任务过滤与 ground-truth 无交集的样本，对代码编辑过滤与 ground-truth 修改行无交集的样本。最终整合所有子任务的 CoT 数据进行 SFT，帮助模型掌握任务格式和推理方法。

3. **基于规则的强化学习**: 设计 Fβ score（β=3，侧重 recall）作为奖励函数，替代传统奖励模型。对定位任务，从模型输出中提取定位结果，与 ground-truth 计算 Fβ 分数；对代码编辑，提取修改代码与 ground-truth 比较。同时加入格式惩罚：如输出为空或包含输入中不存在的目标，奖励直接设为 0，有效防止 reward hacking。

### 损失函数 / 训练策略

- SFT 阶段：使用 FastChat/DeepSpeed 全参数微调，全局 batch size 128，训练 2 epochs，最大学习率 1e-5，cosine 衰减 + 3% warmup
- RL 阶段：使用 OpenRLHF 框架实现 PPO，温度 1.0 采样，结合 vLLM 加速推理
- 训练数据：从 660 个高质量 Python 开源仓库中筛选，60k SFT 样本 + 30k RL 样本
- 严格排除 SWE-Bench 测试集中的仓库，防止数据污染

## 实验关键数据

### 主实验

| 模型 | 框架 | SWE-Bench Verified | SWE-Bench Lite |
|------|------|-------------------|----------------|
| Claude-3.5-Sonnet | Agentless | 50.8% | 40.7% |
| GPT-4o | SWE-SynInfer | 31.8% | 20.7% |
| SWE-Gym-Qwen-7B | OpenHands | 10.6% | 10.0% |
| SWE-Gym-Qwen-14B | OpenHands | 16.4% | 12.7% |
| Lingma-SWE-GPT-7B | SWE-SynInfer | 18.2% | 12.0% |
| **SoRFT-Qwen-7B** | Agentless | **21.4%** | **14.0%** |
| SWE-Fixer-Qwen-72B | SWE-Fixer | 30.2% | 23.3% |
| Lingma-SWE-GPT-72B | SWE-SynInfer | 30.2% | 22.0% |
| **SoRFT-Qwen-32B** | Agentless | **30.8%** | **24.0%** |

### 消融实验

| 配置 | Verified %Resolved | %Applied | 说明 |
|------|-------------------|----------|------|
| Qwen-7B base | 7.6% | 55.6% | 基线 |
| + SFT | 18.0% | 85.2% | SFT 提升显著 |
| + SFT + RL (SoRFT) | 21.4% | 95.6% | RL 进一步提升 |
| Qwen-32B base | 25.6% | 84.4% | 大模型基线 |
| + SFT | 28.8% | 90.6% | SFT 提升 |
| + SFT + RL (SoRFT) | 30.8% | 95.8% | 完整 SoRFT |

### 关键发现

- **SoRFT-Qwen-7B 超越了 SWE-Gym-Qwen-32B**（21.4% vs 20.6%），展示了小模型通过精细训练可超越大模型
- **SoRFT-Qwen-32B 超越 Lingma-SWE-GPT-72B**（30.8% vs 30.2%），参数量仅为后者一半
- 奖励规则的鲁棒性至关重要：使用简单的 hit score 会导致 reward hacking（模型倾向生成更少推理、更多答案），而 Fβ score 有效抑制了这一问题
- PPO 训练中观察到与 DeepSeek-R1 一致的现象：思考长度先减少后增加
- SoRFT 在通用代码任务上也有提升：LiveCodeBench 34.18→34.64，RepoQA 85.0→90.0

## 亮点与洞察

- 子任务分解思路清晰，将端到端困难问题转化为可监督的阶梯式任务
- 巧妙利用开源社区的 (issue, PR) 数据对作为 ground-truth，避免了昂贵的人工标注
- Pipeline 框架比 Agent 框架更适合构建训练信号：Agent 的中间步骤难以评估，而 Pipeline 各阶段可独立评分
- Fβ score 的设计选择（β=3 侧重 recall）基于合理直觉：定位任务中漏掉目标比多选更严重

## 局限与展望

- 仅在 Python 仓库上做实验，缺乏多语言验证（但框架本身语言无关）
- 基于规则的奖励存在 false negative 问题：一个 Issue 可能有多个正确解决方案，仅与单一 ground-truth 比较可能错误惩罚正确解
- 未来可引入单元测试执行结果作为更客观的代码质量评估信号

## 相关工作与启发

- 与 DeepSeek-R1 的思路一致：用基于规则的 RL 替代传统奖励模型
- Agentless 框架的阶段式设计为子任务分解提供了自然的拆分边界
- 启发：软件工程领域拥有大量自然的 ground-truth 信号（PR、测试用例），是 RL 训练的天然试验场

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将 reinforced fine-tuning 应用于 Issue 解决，子任务奖励设计有创意
- 实验: ⭐⭐⭐⭐ — 消融充分，与多个框架和基线对比公平
- 写作: ⭐⭐⭐⭐ — 流程清晰，图示直观
- 实用性: ⭐⭐⭐⭐⭐ — 直接降低 Issue 解决的成本，开源模型替代商业模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)
- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)

</div>

<!-- RELATED:END -->
