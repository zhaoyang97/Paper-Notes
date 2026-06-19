---
title: >-
  [论文解读] Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks
description: >-
  [ACL 2025][LLM推理][自纠正] 提出 Self-Correction Learning (SCL)，通过将 VLM 自身产生的自纠正数据（成功和失败的纠正样本）分类为偏好/非偏好对，利用 DPO 进行偏好微调，从根本上提升模型直接生成正确答案的能力，而非仅仅依赖推理时的迭代修正。 视觉语言模型 (VLM) 虽然…
tags:
  - "ACL 2025"
  - "LLM推理"
  - "自纠正"
  - "VLM"
  - "DPO"
  - "偏好优化"
  - "多模态推理"
---

# Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks

**会议**: ACL 2025  
**arXiv**: [2410.04055](https://arxiv.org/abs/2410.04055)  
**代码**: [有](https://github.com/ivy3h/SCL)  
**领域**: Multimodal / VLM  
**关键词**: 自纠正, VLM, DPO, 偏好优化, 多模态推理

## 一句话总结

提出 Self-Correction Learning (SCL)，通过将 VLM 自身产生的自纠正数据（成功和失败的纠正样本）分类为偏好/非偏好对，利用 DPO 进行偏好微调，从根本上提升模型直接生成正确答案的能力，而非仅仅依赖推理时的迭代修正。

## 研究背景与动机

视觉语言模型 (VLM) 虽然在多模态任务中表现出色，但不可避免地会生成有缺陷的回答。自纠正 (Self-Correction) 是一种让模型识别并修正自身输出错误的方法，被视为提升模型质量的有效途径。

然而，现有自纠正研究存在三个核心问题：

**研究对象局限**：以往自纠正工作主要集中在 LLM 上，VLM 的自纠正能力——尤其是涉及视觉和语言信息的多模态自纠正——几乎未被探索

**推理时自纠正不可靠**：推理阶段的自纠正效果高度依赖提示词设计，且模型的内在推理能力并未改变，同样的错误会反复出现

**目标错位**：现有方法关注的是"更好的修正"（即通过迭代修改来纠正初始回答），而非"更好的初始生成"（即直接生成正确答案）

本文核心论点：自纠正的终极目标不应该是修补初始错误，而应是从根本上提升模型直接生成正确回答的能力。推理时的自纠正只是权宜之计，模型参数没有更新意味着底层推理能力未变。

## 方法详解

### 整体框架

SCL 框架包含三个阶段：推理阶段 → 数据集构建 → 偏好微调。

1. 推理阶段：使用视觉自纠正提示在 VLM 上进行内在自纠正，研究 VLM 能否在无外部反馈情况下自我修正
2. 数据集构建：基于自纠正过程构建 SelfCorSet 偏好数据集
3. 偏好微调：使用 DPO 在 SelfCorSet 上微调 VLM

### 关键设计

1. **视觉自纠正提示设计**：设计了三种针对视觉信息的自纠正提示，引导 VLM 从不同角度审视图像信息

    - VP-1（综合细节提示）：引导模型检查是否遗漏图像细节
    - VP-2（上下文理解提示）：引导模型验证对场景整体语境的理解
    - VP-3（场景分析提示）：引导模型确认对图像的全面理解

   设计动机：VLM 整合了视觉和语言信息，自纠正需要同时审视多模态内容，通用提示可能不够细致。

2. **SelfCorSet 偏好数据集构建**：基于初始回答 (IR) 和修正回答 (RR) 的正确性组合，将自纠正样本分为四类

   | 类型 | IR → RR | 含义 |
   |------|---------|------|
   | Type 1 | 正确→正确 | 保持正确 |
   | Type 2 | 错误→正确 | 成功纠正 |
   | Type 3 | 正确→错误 | 有害纠正 |
   | Type 4 | 错误→错误 | 未能纠正 |

   核心思路：Type 2 的 RR 和 Type 3 的 IR 作为偏好样本（正确回答），Type 2 的 IR 和 Type 3 的 RR 作为非偏好样本（错误回答）。这样就构成了 DPO 需要的偏好-非偏好对。

3. **模型特异性数据集**：每个 VLM 都有自己独立的 SelfCorSet，因为不同 VLM 有不同的自纠正行为模式。这体现了"self"的核心理念——利用模型自身的纠正模式来提升自身。

### 损失函数 / 训练策略

使用标准 DPO 损失函数进行偏好微调：

$$\mathcal{L}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}[\log\sigma(\beta(\log\frac{\pi_\theta(R_c|Q,I)}{\pi_{\text{ref}}(R_c|Q,I)} - \log\frac{\pi_\theta(R_r|Q,I)}{\pi_{\text{ref}}(R_r|Q,I)}))]$$

训练细节：
- 使用 LoRA（rank=8）进行参数高效微调
- 7/8B 模型在单卡 4090 24GB 上训练约 1.5 GPU 小时
- SelfCorSet 数据量：LLaVA-1.5-7B (4,797), MiniCPM (1,853), InternLM (2,361)

## 实验关键数据

### 主实验（推理时自纠正效果）

| 模型 | 基准 (SP) | +CP | +VP-1 | +VP-2 | +VP-3 |
|------|-----------|-----|-------|-------|-------|
| MiniCPM (RealWorldQA) | **61.70** | 38.56 | 48.50 | 47.32 | 43.00 |
| MiniCPM (MMBench) | **79.00** | 68.60 | 76.40 | 61.00 | 54.00 |
| LLaVA-1.5-7B (MMBench) | **68.40** | 54.00 | 57.20 | 45.40 | 54.00 |

**结论**：VLM 在推理阶段的内在自纠正大多导致性能下降，说明推理时自纠正不可靠。

### SCL 训练效果对比

| 模型 | 方法 | RealWorldQA | MMStar | MMBench | SEEDBench | Rank |
|------|------|------------|--------|---------|-----------|------|
| LLaVA-1.5-7B | 基线 | 50.90 | 32.97 | 70.24 | 66.64 | 4.50 |
| +POVID | 51.50 | 33.68 | 71.44 | 65.52 | 3.25 |
| +CSR | 51.03 | 32.59 | 70.44 | 65.12 | 4.25 |
| +SIMA | 49.41 | 32.40 | 71.04 | 64.68 | 4.38 |
| **+SCL** | **52.71** | **36.11** | 71.00 | **67.84** | **1.63** |

### 消融实验

| 训练数据比例 p | SEEDBench | AI2D |
|---------------|-----------|------|
| p=0.2 | 66.88 | 52.90 |
| p=0.4 | 67.80 | 53.60 |
| p=0.6 | 67.84 | 54.20 |
| p=1.0 | **67.84** | **54.81** |

### 关键发现

1. VLM 推理时自纠正大多导致性能下降，Type 3（有害纠正）的比例高于 Type 2（成功纠正）
2. 多轮自纠正效果持续退化：MiniCPM 在 RealWorldQA 上从 61.70% → 48.50% → 39.22% → 42.61%
3. SCL 在所有四个 VLM 上均取得最优 Rank，证明自纠正数据可以有效用于偏好微调
4. 即使数据量较小（p=0.4），微调效果依然显著

## 亮点与洞察

1. **视角独到**：将自纠正重新定义为"提升模型内在能力"而非"迭代修补"，这一视角转换是本文最大贡献
2. **自我数据的巧妙利用**：即使自纠正在推理时失败，它产生的成功/失败样本对仍可作为宝贵的训练信号
3. **类人学习行为**：VLM 类似于人类，能从"做对的事"和"做错的事"中同时学习
4. **实用性强**：训练成本极低（单卡 1.5 小时），数据集完全自生成，无需外部标注

## 局限与展望

1. 仅在 MCQ（多项选择题）上评估，未涉及开放式视觉问答、视频理解等更复杂任务
2. SelfCorSet 中部分"成功纠正"可能是模型偶然猜对、推理过程仍有瑕疵（可靠性问题）
3. 数据集为模型特异性设计，不同 VLM 间不能直接迁移
4. 未探索更大规模数据增强的潜力
5. 多轮自纠正产生更丰富的错误模式，可能提供更多训练信号

## 相关工作与启发

- 与 SIMA 对比：SIMA 用单轮温度采样 + 自我评判构建偏好数据，SCL 用自纠正过程中的初始-修正回答对构建
- 与 SCoRe 对比：SCoRe 是多轮在线 RL 方法，SCL 是离线 DPO，更简单高效
- 与 CSR 对比：CSR 引入迭代学习和奖励范式，SCL 无需额外奖励模型

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将自纠正从推理时方法转化为训练信号的视角新颖，利用失败纠正作为负样本的设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ 多个 VLM、8 个基准测试、多个基线对比、消融实验完整
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，RQ 驱动的研究设计使论文结构紧凑
- **价值**: ⭐⭐⭐⭐ 方法简单有效、成本低，为 VLM 自我改进提供了实用范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Many LLMs Are More Utilitarian Than One](../../NeurIPS2025/llm_reasoning/many_llms_are_more_utilitarian_than_one.md)
- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](improve_vlm_cot_reasoning.md)
- [\[ACL 2025\] CoT-ICL Lab: A Synthetic Framework for Studying Chain-of-Thought Learning from In-Context Demonstrations](cot-icl_lab_a_synthetic_framework_for_studying_chain-of-thought_learning_from_in.md)
- [\[ACL 2025\] Ranked Voting based Self-Consistency of Large Language Models](ranked_voting_based_self-consistency_of_large_language_models.md)

</div>

<!-- RELATED:END -->
