---
title: >-
  [论文解读] Synergistic Weak-Strong Collaboration by Aligning Preferences
description: >-
  [ACL 2025][LLM对齐][弱强模型协作] 本文提出 CoWest 框架，通过让专业化的弱模型（如 LLaMA3-8B）生成初始草稿，再由通用强模型（如 GPT-4）精炼，并利用协作反馈通过 DPO 微调弱模型以对齐强模型偏好，在反事实推理、医学和伦理三个领域显著超越单模型和已有协作方法。 当前 LLM 在通用推理上…
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "弱强模型协作"
  - "偏好对齐"
  - "DPO"
  - "知识互补"
  - "LLM 协作"
---

# Synergistic Weak-Strong Collaboration by Aligning Preferences

**会议**: ACL 2025  
**arXiv**: [2504.15188](https://arxiv.org/abs/2504.15188)  
**代码**: 公开可用（论文声明）  
**领域**: 其他  
**关键词**: 弱强模型协作, 偏好对齐, DPO, 知识互补, LLM 协作

## 一句话总结

本文提出 CoWest 框架，通过让专业化的弱模型（如 LLaMA3-8B）生成初始草稿，再由通用强模型（如 GPT-4）精炼，并利用协作反馈通过 DPO 微调弱模型以对齐强模型偏好，在反事实推理、医学和伦理三个领域显著超越单模型和已有协作方法。

## 研究背景与动机

当前 LLM 在通用推理上表现出色，但在需要专有或领域特定知识的专业化任务上仍有不足。对大模型进行领域微调面临两大障碍：（1）许多流行 LLM（如 GPT-4、Gemini）是黑盒模型，参数不可访问；（2）即使可微调，计算成本巨大且存在隐私风险——微调需要将敏感数据暴露给模型。

已有弱强模型协作方法存在的问题：（1）预定义的交互机制缺乏灵活性，如弱模型仅提供固定形式的知识片段；（2）仅使用单模型的反馈来微调另一个模型，忽略了**协作过程本身产生的反馈信息**。协作反馈能帮助弱模型理解强模型的偏好，增强双方的互利合作。

本文的核心思想是：不仅让弱强模型协作推理，还要从协作过程中提取偏好信号来优化弱模型，使其输出更符合强模型的需求。

## 方法详解

### 整体框架

CoWest 包含两个阶段：
- **训练阶段**：（1）用任务数据 SFT 微调弱模型获得领域能力；（2）构建协作偏好数据；（3）用 DPO 对齐弱模型。
- **推理阶段**：弱模型对查询生成初始输出（含推理链），强模型接收弱模型输出和原始查询进行精炼，产出最终答案。

### 关键设计

1. **弱模型 SFT 微调**：在任务特定训练集 $\mathcal{D}_{\text{SFT}} = \{(x, \hat{y})\}$ 上，通过标准负对数似然损失微调弱模型 $\pi_w$，使其获得领域专业能力。使用 LoRA 进行高效微调。

2. **协作偏好反馈构建**：核心创新在于偏好数据的来源——不是人类标注，而是协作过程本身。具体步骤：

    - **强模型单独推理**：直接用 CoT 提示让强模型回答，得到输出 $z \sim \pi_s(z|x)$。
    - **弱强协作推理**：弱模型先生成解释和初始结果 $y \sim \pi_w(y|x)$，再传给强模型精炼得到 $y^* \sim \pi_s(y^*|y)$。
    - **偏好评估**：外部评估器 $E$ 对两种输出打分，评估维度为推理逻辑连贯性和与真实答案的一致性。评估差值 $\Delta = E(\pi_s \circ y, x) - E(z, x)$：若 $\Delta > 0$，弱模型的贡献有益，其输出为正样本 $y_+$；否则为负样本 $y_-$。

3. **DPO 偏好优化**：用构建的偏好三元组 $\mathcal{D}_{\text{PT}} = \{(x, y_+, y_-)\}$ 通过 DPO 微调弱模型：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\alpha\log\frac{\pi_w^*(y_+|x)}{\pi_w(y_+|x)} - \alpha\log\frac{\pi_w^*(y_-|x)}{\pi_w(y_-|x)}\right)\right]$$

优化目标是让弱模型生成的输出在经强模型精炼后能获得更高评分。

4. **协作推理**：最终推理时，查询先经对齐后的弱模型 $\pi_w^*$ 处理，再由强模型精炼：$y^* = \pi_s \circ (x, \pi_w^* \circ x)$。

### 损失函数 / 训练策略

训练分两步：第一步 SFT（标准交叉熵损失），第二步 DPO（偏好优化损失）。评估器使用与强模型相同的 LLM（如 GPT-4），确保偏好信号反映强模型的真实偏好。

关键理论洞察：在简化假设下（强模型对任何查询的评分恒定），优化后的弱模型 $\pi_w^*$ 会对所有导致协作评分不高于强模型单独表现的输出赋零概率。即弱模型学会了只产生能正面贡献于协作的输出。

## 实验关键数据

### 主实验

| 方法类别 | 方法 | Counterfactuals (EM/F1) | Medicine (Acc/F1) | Ethics (Acc/F1) |
|---------|------|------------------------|------------------|----------------|
| 弱模型 | Llama-3-8B (SFT) | 69.71/72.69 | 73.08/58.26 | 64.29/62.40 |
| 强模型 | GPT-4 (CoT) | 57.42/65.60 | 71.80/57.69 | 39.00/39.58 |
| RAG | FLARE | 62.07/70.59 | 72.40/58.89 | 55.27/54.97 |
| 协作 | SuperICL | 68.85/74.82 | 73.64/58.33 | 66.18/63.86 |
| 协作 | RLWF | 70.52/75.04 | 72.01/57.65 | 64.85/62.10 |
| **协作** | **CoWest** | **75.85/77.34** | **75.10/60.13** | **68.33/65.61** |

CoWest 相对最优单模型的提升：Counterfactuals +6.14 EM，Medicine +2.02 Acc，Ethics +4.04 Acc。

### 消融实验

交互策略消融（EM/Acc 报告）：

| 交互格式 | 无对齐 | 有对齐 | 说明 |
|---------|-------|-------|------|
| Direct Answer | 较低 | 中等 | 直接回答信息量有限 |
| Domain Knowledge | 中等 | 较高 | 提供领域背景知识 |
| Chain of Thought | 较高 | **最高** | CoT 详细推理路径最有效 |

不同强模型的影响：

| 强模型 | Counterfactuals | Ethics |
|-------|----------------|--------|
| GPT-4 | **75.9%** | 38.2% |
| Llama-3-70B | 72.1% | 62.3% |
| GPT-3.5-Turbo | 70.8% | **68.3%** |
| Llama-2-70B | 68.5% | 55.7% |

不同弱模型的影响：

| 弱模型 | 参数量 | 总体表现 |
|-------|-------|---------|
| Llama-3-8B | 8B | 最优 |
| Llama-2-7B | 7B | 较优 |
| Phi-3-mini | 3B | 一般 |
| TinyLlama | 1B | 较弱 |

### 关键发现

1. **协作显著优于单模型**：CoWest 在所有三个数据集上都显著超越弱模型（尽管已 SFT）和强模型单独表现。GPT-4 在反事实推理上单独仅 57.42%，协作后达 75.85%。

2. **偏好对齐是关键**：在所有交互格式下，有对齐的版本都显著优于无对齐版本，证明协作反馈的有效性。

3. **CoT 格式最有效**：Chain of Thought 格式在所有数据集上表现最好，其详细推理路径帮助强模型更好地理解和精炼弱模型的输出。

4. **强模型的通用能力而非绝对优势才是关键**：GPT-4 在反事实推理上最强但在伦理上不如 GPT-3.5-Turbo。仅仅"比弱模型强"不够，强模型需要有足够的纠错能力。

5. **弱模型的基础能力很重要**：较大的弱模型（8B/7B）显著优于小模型（3B/1B），弱模型需要足够的基础能力才能产生有价值的初始输出。

6. **训练数据量存在最优点**：反事实推理（仅 2K 样本）在 1K 偏好数据时就达到峰值，继续增加反而因重复采样降低质量；而更大数据集（Medical/Ethics）持续到 2K 仍在提升。

## 亮点与洞察

- 协作偏好的构建思路巧妙：对比"强模型单独做"和"弱强协作做"的效果差异来判断弱模型贡献的正负，无需人类标注。
- 理论分析提供了有意义的保证：优化后的弱模型不会产生"帮倒忙"的输出。
- 框架对黑盒强模型友好，不需要获取强模型参数，只需要其 API 调用，具有很强的实践意义。
- 实验发现强模型效果的领域依赖性颇有启发：没有万能的强模型，不同领域需要匹配不同的强模型。

## 局限与展望

- **单轮反馈**：仅进行了一轮偏好对齐迭代，未探索多轮迭代对齐是否能带来持续提升。
- **模型家族受限**：实验仅使用了 Llama 和 GPT 系列，其他模型架构（如 Mistral、Qwen）的效果未验证。
- **评估器偏差**：使用与强模型相同的 LLM 作为评估器可能引入偏差，评估器偏好强模型风格的输出。
- **推理延迟**：需要两次串行的 LLM 调用（弱模型→强模型），推理延迟翻倍。
- **偏好数据构建成本**：需要对每个训练样本分别调用强模型两次（单独推理和协作推理），API 成本较高。

## 相关工作与启发

- **SuperICL**（Xu et al., 2024）：用小模型输出提示大模型，但交互方式固定。CoWest 通过偏好对齐使交互动态化。
- **Weak-to-Strong Generalization**（Burns et al., 2024）：强模型从弱模型的监督中学习，但需要强模型参数可访问。CoWest 保持强模型为黑盒。
- **DPO**（Rafailov et al., 2023）：本文将 DPO 应用于弱模型对齐，偏好信号来源从人类/AI 评估扩展到协作过程本身。
- **RAG**：检索增强生成提供静态上下文，而 CoWest 的弱模型输出是经过推理的动态知识，适应性更强。

## 评分

- 新颖性: ⭐⭐⭐⭐ 协作偏好反馈的思路新颖，将 DPO 从传统的人类/AI 偏好扩展到协作过程偏好
- 实验充分度: ⭐⭐⭐⭐ 三个领域、多种基线、详细消融（交互策略/模型选择/数据量），但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 方法动机和流程描述清晰，理论分析简洁有力
- 价值: ⭐⭐⭐⭐ 对黑盒 LLM 时代的实际应用有指导意义，框架通用且易于实现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Aligning to What? Limits to RLHF Based Alignment](aligning_to_what_limits_to_rlhf_based_alignment.md)
- [\[ACL 2026\] Student Guides Teacher: Weak-to-Strong Inference via Spectral Orthogonal Exploration](../../ACL2026/llm_alignment/student_guides_teacher_weak-to-strong_inference_via_spectral_orthogonal_explorat.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](../../AAAI2026/llm_alignment/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)
- [\[ACL 2025\] Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences](teaching_an_old_llm_secure_coding.md)
- [\[ICLR 2026\] Aligning Deep Implicit Preferences by Learning to Reason Defensively](../../ICLR2026/llm_alignment/aligning_deep_implicit_preferences_by_learning_to_reason_defensively.md)

</div>

<!-- RELATED:END -->
