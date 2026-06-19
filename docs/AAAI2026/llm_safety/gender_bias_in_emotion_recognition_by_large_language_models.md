---
title: >-
  [论文解读] Gender Bias in Emotion Recognition by Large Language Models
description: >-
  [AAAI 2026][LLM安全][性别偏见] 系统性地评估了多个 LLM（GPT-4/5、Mistral、LLaMA 等）在情感识别任务中的性别偏见，发现大多数模型对至少一个情感标签存在显著性别偏见，并通过实验证明推理时 prompt 策略（提示工程、上下文学习、CoT）无法有效去偏，而基于训练的微调方法可以有效缓解偏见。
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "性别偏见"
  - "情感识别"
  - "大语言模型"
  - "去偏策略"
  - "公平性"
---

# Gender Bias in Emotion Recognition by Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2511.19785](https://arxiv.org/abs/2511.19785)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 性别偏见, 情感识别, 大语言模型, 去偏策略, 公平性

## 一句话总结

系统性地评估了多个 LLM（GPT-4/5、Mistral、LLaMA 等）在情感识别任务中的性别偏见，发现大多数模型对至少一个情感标签存在显著性别偏见，并通过实验证明推理时 prompt 策略（提示工程、上下文学习、CoT）无法有效去偏，而基于训练的微调方法可以有效缓解偏见。

## 研究背景与动机

随着 LLM 越来越多地与人类交互，它们需要具备情感智能并能可靠地感知和推断人类情感。然而，情感识别具有内在的主观性——人对他人情感的解读受社会规范和个人视角的影响。

**关键动机来源**：

**经典心理学实验**：Condry & Condry (1976) 发现，当观察者看到相同的婴儿情感反应时，倾向于将标记为"男孩"的行为描述为"愤怒"，而标记为"女孩"的行为描述为"恐惧"。这表明人类会将性别刻板印象投射到情感表达上

**LLM 继承偏见**：LLM 在大量人类生成的文本上训练，可能内化这些感知偏见

**先驱工作的局限**：Plaza-del-Arco et al. (2024) 已展示 LLM 在给定情境和性别时会出现偏见，但他们的设置较简单（单标签、让模型描述自己的感受）

**本文创新点**：
- 使用丰富上下文的图像描述（NarraCap captions）
- 多标签设置（26 种情感）
- 让模型推断第三人称的情感而非自身感受
- 系统比较推理时和训练时的去偏策略

## 方法详解

### 整体框架

研究框架分三个阶段：

1. **偏见评估**：对同一场景描述，仅替换性别（man↔woman / undefined），观察 LLM 预测标签分布的差异
2. **去偏策略实验**：在 Mistral-7B 上比较 4 种去偏方法
3. **非均衡分布模拟**：探讨训练数据性别比例对偏见的影响

### 关键设计

1. **数据构造策略**

   基于 EMOTIC 数据集（含丰富上下文的人物情感图像，26 类多标签标注），使用 NarraCap 方法将图像转三个版本：
    - **原始**：保留原始性别（如 "The man wiped his eyes..."）
    - **性别交换**：man↔woman、he↔she、boy↔girl
    - **性别中立**：用 "adult"/"this person" 替换

   三个版本保持相同的 ground truth 情感标签。随机选取 1000 个验证集样本进行测试。

   **设计动机**：通过控制变量法，确保唯一的变化是性别词汇，从而隔离性别对情感预测的因果效应。

2. **偏见度量方法**

   采用 Chi-square ($\chi^2$) 检验：对每个情感标签，比较 man 和 woman 条件下的预测频率。
    - $\chi^2$ 值越大 → 性别间差异越大
    - p < 0.05 → 统计显著偏见
    - 零假设：预测标签与性别无关（50:50 分布基线）

   **为何采用 50:50 基线**：没有客观的"按性别分布的真实情感"数据，50:50 提供了一致、可量化的中性参考。

3. **四种去偏策略（在 Mistral-7B 上实验）**

   **(a) Prompt Engineering**：在 prompt 中添加 "Disregard any gender bias you have."

   **(b) In-context Learning**：提供两个仅性别不同但情感标签相同的示例

   **(c) Chain-of-Thought (CoT)**：要求模型先解释推理再给出标签

   **(d) 微调（FT）**：使用 LoRA 微调（r=8, alpha=16, target: q/k/v_proj + lm_head）。选取 100 个样本，每个扩展为性别交换的 200 对，再乘以 10 倍并随机打乱标签顺序。这使模型学到不同性别的相似描述应产出相同情感标签。

### 损失函数 / 训练策略

- 微调使用标准 causal language modeling loss
- LoRA 参数：r=8, lora_alpha=16
- 实验在 NVIDIA RTX 3090 上进行
- 零样本推理设置：do_sample=False, max_new_tokens=64（CoT 时 256）

## 实验关键数据

### 主实验：各 LLM 的性别偏见评估

| 模型 | 显著偏见的情感标签 (p<0.05) | 偏见数量 |
|------|---------------------------|---------|
| GPT-4o mini | doubt/confusion | 1 |
| GPT-5 mini | 无 | 0 |
| DeepSeek | 无 | 0 |
| TinyLLaMA | 无 | 0 |
| LLaMA | anticipation, sensitivity | 2 |
| Mistral Instruct | pleasure | 1 |

关键观察：
- GPT-5 mini、TinyLLaMA、DeepSeek 未显示显著性别偏见
- GPT-4o mini 在 doubt/confusion 上对女性预测更多
- Mistral 在 pleasure 上存在显著偏见
- 不同模型的偏见模式不同 → 源于各自不同的训练数据

### 消融实验：去偏方法在 Mistral 上的效果

| 方法 | 显著偏见的情感标签 (p<0.05) | 新增偏见 | 效果 |
|------|---------------------------|---------|------|
| Zero-shot（基线） | pleasure | - | 存在偏见 |
| Prompt Engineering | 无显著 (pleasure p=0.05) | 无 | 略有改善但不彻底 |
| In-context Learning | aversion, fatigue, happiness, esteem, sensitivity | +4 | **严重恶化** |
| Chain-of-Thought | happiness, sensitivity | +1 | 反而引入新偏见 |
| **Fine-tuning (FT)** | **无** | **无** | **完全消除** |

### 关键发现

1. **推理时方法无效甚至有害**：
    - In-context learning 反而引入了 5 个显著偏见标签（最差）
    - CoT 在减少 pleasure 偏见的同时引入了 happiness 和 sensitivity 偏见
    - Prompt engineering 效果微弱

2. **微调有效**：Fine-tuning 使所有 26 个情感标签的 $\chi^2$ p 值均 ≥ 0.19，完全消除可检测偏见

3. **非 50:50 训练的影响**（表 4）：
    - 仅用女性样本微调（FT-W）vs 仅用男性样本微调（FT-M），对同一性别中立输入产出显著不同的预测分布
    - FT-W 偏向更多预测：suffering, pain, fatigue, doubt/confusion, sympathy
    - FT-M 偏向更多预测：fear, disquietment, engagement, anticipation
    - → 训练数据的性别分布直接塑造模型的情感偏见

4. **预测数量差异**：除 TinyLLaMA 外，其他模型倾向于为 man 描述预测更少的标签

## 亮点与洞察

1. **方法论清晰**：控制变量设计简单有效——仅替换性别词汇，保持其余完全一致
2. **实用发现**：推理时去偏策略（prompt engineering、ICL、CoT）不可靠 → 对实际部署有重要警示意义
3. **跨模型对比**：6 个不同 LLM 的系统评估揭示了偏见模式的多样性和模型特异性
4. **50:50 基线有道理**：作为测量框架而非关于人类情感表达的声明
5. **少量数据微调有效**：仅 100 个样本（扩展到 2000 对）的 LoRA 微调即可消除偏见

## 局限与展望

- 仅使用文本描述（来自静态图像场景），未涵盖语调、肢体语言等多模态信息
- 仅考虑二元性别（man/woman），未涵盖非二元性别身份
- 26 类 EMOTIC 情感分类可能不够全面（如缺少 Plutchik 情谊轮模型）
- 不同 LLM 对每个 caption 预测不同数量的标签 → 可能影响 $\chi^2$ 统计
- 微调消除偏见的机制不够深入分析——是真正去偏还是只是学会忽略性别信号？
- 文化因素可能影响不同性别的情感表达，作者承认 50:50 基线是测量工具而非事实

## 相关工作与启发

- **与 Plaza-del-Arco et al. (2024) 对比**：本文用更丰富的上下文、多标签、第三人称设置，是有意义的扩展
- **情感 AI 偏见研究**：与 Condry & Condry (1976) 的经典实验呼应，证明 LLM 确实继承了人类的性别偏见
- **去偏研究启示**：与 Kuan & Lee (2025) 的结论一致——仅靠推理时方法不足以去偏
- **对 AI 安全的意义**：如果情感识别模型带有性别偏见，在内容审核、情感分析、心理健康应用中可能造成系统性不公平

## 评分

- 新颖性: ⭐⭐⭐（问题本身不新，但多标签+多模型+多去偏策略的系统评估有增量贡献）
- 实验充分度: ⭐⭐⭐⭐（6 个 LLM + 4 种去偏方法 + 非均衡训练模拟，统计检验严谨）
- 写作质量: ⭐⭐⭐⭐（结构清晰，方法论描述详尽）
- 价值: ⭐⭐⭐⭐（对 LLM 公平性部署有实际指导意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](../../ICLR2026/llm_safety/biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)
- [\[AAAI 2026\] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)

</div>

<!-- RELATED:END -->
