---
title: >-
  [论文解读] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning
description: >-
  [NeurIPS 2025][LLM安全][机器遗忘] 本文提出 PULSE 评估协议，从预训练知识遗忘和多次顺序遗忘的可持续性两个实际维度出发，揭示了现有遗忘方法在 LMM 上的严重不足——遗忘预训练知识会导致 90% 以上通用能力丧失，连续遗忘 5 次后模型泛化能力几乎完全崩溃。 随着大语言模型（LLM）和大规模多模态模…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "机器遗忘"
  - "大规模多模态模型"
  - "预训练知识"
  - "可持续性"
  - "评估基准"
---

# PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning

**会议**: NeurIPS 2025  
**arXiv**: [2507.01271](https://arxiv.org/abs/2507.01271)  
**代码**: 无  
**领域**: AI安全 / 机器遗忘评估  
**关键词**: 机器遗忘, 大规模多模态模型, 预训练知识, 可持续性, 评估基准

## 一句话总结

本文提出 PULSE 评估协议，从预训练知识遗忘和多次顺序遗忘的可持续性两个实际维度出发，揭示了现有遗忘方法在 LMM 上的严重不足——遗忘预训练知识会导致 90% 以上通用能力丧失，连续遗忘 5 次后模型泛化能力几乎完全崩溃。

## 研究背景与动机

随着大语言模型（LLM）和大规模多模态模型（LMM）的普及，训练数据可能包含个人隐私信息和受版权保护的内容，遗忘（unlearning）技术因此受到关注。目前已有多种遗忘方法被提出（如 GA、NPO、SIU 等），LLM 领域也有 TOFU 和 MUSE 等评估基准。

然而，**LMM 遗忘领域缺乏实际的评估框架**。唯一的 LMM 遗忘基准 MLLMU-Bench 存在两个关键缺陷：

**只考虑微调知识遗忘**：仅测试遗忘最近一次微调中学到的知识，无法评估对预训练阶段获得的知识的遗忘效果。而在实际场景中，需要遗忘的信息很可能在预训练阶段就已被学会。

**只考虑单次遗忘操作**：现实中遗忘请求是持续到来的（如不同用户陆续提出数据删除要求），需要对同一模型进行多次遗忘操作。

PULSE 协议正是为了填补这两个关键评估空白而设计的。

## 方法详解

### 整体框架

PULSE 在传统"先微调、再遗忘"的评估管线之上，增加了两个新的评估维度：

- **预训练知识遗忘**：评估遗忘方法能否有效遗忘在预训练阶段获得的知识
- **可持续性评估**：将遗忘目标分成多个子集，依次执行遗忘操作，评估模型在多次遗忘后的性能变化

### 关键设计

1. **问题形式化**：

    - 将数据分为遗忘目标 $\mathcal{D}_{\text{unlearn}}$ 和保留数据 $\mathcal{D}_{\text{retain}}$
    - 评估两方面：有效性（$\mathcal{D}_{\text{unlearn}}$ 上准确率下降）和泛化性（$\mathcal{D}_{\text{retain}}$ 和 MMBench 上准确率保持）
    - 关键设定：**不论是否提供图像输入，模型对遗忘目标都不应泄露任何信息**——既评估多模态任务也评估纯文本任务

2. **预训练知识遗忘设计**：

    - 不同于传统方法从微调数据中选择遗忘目标，而是从预训练阶段模型已"知道"的知识中选择
    - 从 MLLMU-Bench 数据集中 153 位真实名人中，筛选出 LLaVA-v1.5-13B 准确率最高的 45 人
    - 20 人作为 $\mathcal{D}_{\text{unlearn}}$，25 人作为 $\mathcal{D}_{\text{retain}}$
    - 每个人关联 10 个问答对（5 个多模态 + 5 个纯文本）

3. **可持续性评估设计**：

    - 将 $\mathcal{D}_{\text{unlearn}}$（50 人）分为 5 个子集，每个子集 10 人
    - 对模型依次执行 5 次遗忘操作
    - 在每次操作后追踪有效性和泛化性指标的变化

### 损失函数 / 训练策略

评估了三种遗忘方法：
- **GA（梯度上升）**：在 $\mathcal{D}_{\text{unlearn}}$ 上沿梯度反方向更新参数
- **GA+KLR**：在 GA 基础上加 KL 散度正则化，保持更新后模型与原始模型接近
- **NPO**：偏好优化方法，将遗忘数据视为负例

使用 LLaVA-v1.5-13B 作为基础模型，微调和遗忘均使用 LoRA。

## 实验关键数据

### 主实验（微调知识 vs. 预训练知识遗忘）

| 知识类型 | 方法 | $\mathcal{D}_{\text{unlearn}}$ 遗忘率 | $\mathcal{D}_{\text{retain}}$ 保持率 | MMBench 保持率 |
|---------|------|--------------------------------------|--------------------------------------|---------------|
| 微调知识 | GA | 高（有效遗忘） | ~70% | ~90% |
| 微调知识 | GA+KLR | 中等 | ~75% | ~92% |
| 微调知识 | NPO | 高 | ~72% | ~91% |
| 预训练知识 | GA | 高（有效遗忘） | 显著下降 | **<10%**（灾难性） |
| 预训练知识 | GA+KLR | 中等 | 下降 | **<10%** |
| 预训练知识 | NPO | 高 | 下降 | **<10%** |

### 消融实验（模态差异 & 参数更新目标）

| 参数更新目标 | 遗忘方法 | $\mathcal{D}_{\text{unlearn}}$ Multi↓ | $\mathcal{D}_{\text{unlearn}}$ Text↓ | MMBench↑ |
|------------|---------|--------------------------------------|--------------------------------------|----------|
| 遗忘前 | - | 78.0 | 76.8 | 75.1 |
| Proj+LLM | GA | 9.6 | 35.2 | 71.1 |
| LLM only | GA | 24.8 | 33.2 | 48.8 |

### 关键发现

- **预训练知识极难遗忘**：虽然 $\mathcal{D}_{\text{unlearn}}$ 上的准确率确实下降了，但 MMBench 分数暴跌超过 90%。这意味着遗忘预训练知识的代价是几乎完全丧失模型的通用多模态能力
- **可持续性完全不足**：经过 5 次顺序遗忘后，所有方法的泛化指标（$\mathcal{D}_{\text{retain}}$ 和 MMBench）几乎归零，表明当前方法完全无法应对现实中的连续遗忘场景
- **模态间遗忘不均衡**：更新 Proj+LLM 时，多模态任务准确率从 78.0% 降到 9.6%，但纯文本任务仅降到 35.2%——说明现有方法可能只是"破坏了图像与知识的对齐"，而非真正遗忘了知识本身
- **参数选择的矛盾**：仅更新 LLM 时 MMBench 大幅下降（48.8%），但同时更新 Proj 和 LLM 时 MMBench 仅小幅下降（71.1%）。可能原因是允许更新投影矩阵让模型可以通过"断开模态间连接"来"偷懒遗忘"

## 亮点与洞察

- **填补关键评估空白**：首次为 LMM 遗忘提出覆盖预训练知识和可持续性的系统评估协议
- **实验发现极具警示性**：预训练知识遗忘导致 90%+ 能力丧失这一发现，直接质疑了当前遗忘方法在实际部署中的可行性
- **多模态 vs. 纯文本的分析**：揭示了 LMM 遗忘中一个被忽视的关键问题——遗忘多模态任务≠遗忘纯文本任务中的相同知识
- **评估设计的实际性**：基于模型实际行为选择遗忘目标（而非依赖预训练数据访问），更贴近实际部署场景

## 局限与展望

- 仅评估了 LLaVA-v1.5-13B 一个模型，其他 LMM（如 LLaVA-NeXT、InternVL 等）的表现可能不同
- 仅测试了三种较基础的遗忘方法（GA、GA+KLR、NPO），未覆盖更先进的方法
- 排除了 SIU 等仅针对多模态任务的方法，虽然理由充分但限制了方法比较的全面性
- 数据集规模较小（20-50 人），可能不足以得出高置信度的统计结论
- 可持续性实验中固定了"每次遗忘 10 人、共 5 次"的设定，更多变体（如不同批次大小、更多次数）值得探索
- 未探讨可能的解决方案（如弹性权重合并、模型蒸馏等），仅定位问题

## 相关工作与启发

- MUSE 为 LLM 遗忘提出了包括可持续性在内的多维评估，PULSE 将这一思路扩展到多模态领域
- TOFU 在 LLM 上使用虚构人物数据评估遗忘，MLLMU-Bench 是首个 LMM 遗忘基准
- Yao et al. (2024) 在 LLM 上探索了预训练知识遗忘，但需要访问预训练数据
- 本文的核心启示：**当前的遗忘技术在 LMM 上还远未达到实用标准**，需要根本性的方法突破

## 评分

- 新颖性: ⭐⭐⭐⭐ （评估框架的新视角有价值，但方法层面无贡献）
- 实验充分度: ⭐⭐⭐ （模型和方法覆盖偏窄）
- 写作质量: ⭐⭐⭐⭐ （问题动机清晰，实验设计合理）
- 价值: ⭐⭐⭐⭐ （对 LMM 遗忘社区具有重要的基准设定意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](../../ACL2025/llm_safety/mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](../../ACL2025/llm_safety/manu_modality_aware_unlearning.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)

</div>

<!-- RELATED:END -->
