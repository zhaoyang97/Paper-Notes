---
title: >-
  [论文解读] Configurable Preference Tuning with Rubric-Guided Synthetic Data
description: >-
  [ICML 2025][可解释性][可配置偏好调优] 提出Configurable Preference Tuning (CPT)框架，通过基于细粒度rubric生成的合成偏好数据训练LLM，使模型能在推理时仅通过修改system prompt就动态调整行为风格，无需重新训练，在多个基座模型上准确率从0.52-0.68提升至0.76-0.83。
tags:
  - "ICML 2025"
  - "可解释性"
  - "可配置偏好调优"
  - "DPO"
  - "合成数据"
  - "评分标准引导"
  - "系统提示控制"
---

# Configurable Preference Tuning with Rubric-Guided Synthetic Data

**会议**: ICML 2025  
**arXiv**: [2506.11702](https://arxiv.org/abs/2506.11702)  
**代码**: [https://github.com/vicgalle/configurable-preference-tuning](https://github.com/vicgalle/configurable-preference-tuning)  
**领域**: 可解释性  
**关键词**: 可配置偏好调优, DPO, 合成数据, 评分标准引导, 系统提示控制

## 一句话总结
提出Configurable Preference Tuning (CPT)框架，通过基于细粒度rubric生成的合成偏好数据训练LLM，使模型能在推理时仅通过修改system prompt就动态调整行为风格，无需重新训练，在多个基座模型上准确率从0.52-0.68提升至0.76-0.83。

## 研究背景与动机
现有LLM对齐方法（如RLHF/DPO）在训练时"固化"了一组单一、静态的偏好，形成"一刀切"的行为模式。但人类偏好本质上是多维的、动态的、依赖上下文的——不同用户、不同场景、不同文化背景下对LLM输出的期望完全不同。要改变已对齐模型的行为（如调整写作风格、安全等级或角色设定），通常需要昂贵的重新微调。

现有个性化RLHF方法主要有两类：(1) 学习一组隐式特征然后用权重组合（如RFM），但可解释性差；(2) 通过用户ID/历史交互推断隐式表示来条件化奖励模型（如P-RLHF），同样缺乏显式可控性。核心矛盾在于：如何在不重新训练的前提下，让模型能够根据人类可读的指令精确调整输出行为？

本文的切入角度是：利用结构化的评分标准（rubric）来定义期望属性，通过rubric引导生成合成偏好数据，使学生模型学会根据不同system prompt切换行为模式。核心idea：**将rubric-score组合编码为system prompt，让同一对响应在不同system prompt下互换chosen/rejected角色，从而教会模型"配置化"地响应**。

## 方法详解

### 整体框架
CPT的核心pipeline分四步：(1) 定义细粒度rubric → (2) 用教师模型按不同分数目标生成响应 → (3) 将rubric+分数总结为简洁的system prompt → (4) 构造对称偏好对并用DPO训练学生模型。整个过程不需要新的人工标注，完全依赖合成数据。

### 关键设计
1. **Rubric定义与分数条件生成**:

    - 功能：定义一组rubric $\{\mathcal{R}_i\}$，每个rubric详细描述响应的某个属性维度（如"非传统性"、"华丽巴洛克风格"、"神秘象征主义"等），并按5个等级给出详细评分标准
    - 核心思路：对于每个rubric $\mathcal{R}$ 和用户prompt $x$，用增强提示 $\phi(x, \mathcal{R}, \text{score})$ 指导教师模型生成符合特定分数水平的响应 $y \sim p(y|\phi(x, \mathcal{R}, \text{score}))$
    - 设计动机：通过显式的评分标准让生成过程可控、可解释，而非依赖隐式的偏好学习

2. **System Prompt合成**:

    - 功能：将每个rubric+分数组合总结为2-3句话的简洁system prompt
    - 核心思路：$s = \text{summarize}(\mathcal{R}, \text{score})$，由教师模型自动完成。例如rubric要求"非传统风格"的高分，对应system prompt为"Generate a text that is fragmented, illogical, and filled with unexpected connections..."
    - 设计动机：system prompt是推理时的控制接口，必须简洁且人类可读

3. **对称偏好对构造**:

    - 功能：从同一对响应 $(y_1, y_2)$ 构造两个互补的DPO训练样本
    - 核心思路：选定rubric $\mathcal{R}$，生成两个不同分数目标的响应和对应system prompt。关键创新是**同一对响应在不同system prompt下角色互换**：
        - 样本1: $(s_1, x, y_1, y_2)$ — 在$s_1$下$y_1$优于$y_2$
        - 样本2: $(s_2, x, y_2, y_1)$ — 在$s_2$下$y_2$优于$y_1$
    - 设计动机：这种对称构造迫使学生模型真正学会根据system prompt $s$ 来切换偏好，而非简单记忆哪个响应"更好"

### 损失函数 / 训练策略
使用标准DPO损失，但额外条件化在system prompt $s$ 上：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(s,x,y_w,y_l)\sim\mathcal{D}} \left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|s,x)}{\pi_{\text{ref}}(y_w|s,x)} - \beta\log\frac{\pi_\theta(y_l|s,x)}{\pi_{\text{ref}}(y_l|s,x)}\right)\right]$$

训练采用LoRA进行参数高效微调，仅训练1个epoch。合成数据集规模为900个样本（来自4个rubric × 3个分数目标的组合）。教师模型使用DeepSeek-R1和o3-mini，评估judge使用Claude 3.5 Sonnet。

## 实验关键数据

### 主实验

| 模型 | 配置 | Accuracy | Kendall's τ | Spearman's ρ |
|------|------|----------|-------------|--------------|
| Rocinante-12B | baseline | 0.55 | 0.62 | 0.76 |
| Rocinante-12B | CPT | 0.76 | 0.76 | 0.88 |
| Qwen3-4B | baseline | 0.63 | 0.78 | 0.90 |
| Qwen3-4B | CPT | 0.77 | 0.82 | 0.93 |
| Mistral-Nemo-12B | baseline | 0.60 | 0.62 | 0.74 |
| Mistral-Nemo-12B | CPT | 0.83 | 0.81 | 0.93 |
| Mistral-Small-24B | baseline | 0.52 | 0.73 | 0.85 |
| Mistral-Small-24B | CPT | 0.78 | 0.80 | 0.92 |
| Phi-4-14B | baseline | 0.68 | 0.79 | 0.92 |
| Phi-4-14B | CPT | 0.77 | 0.82 | 0.93 |

### 教师模型生成质量验证

| 分数目标 | 模型 | Judge评分(/100) |
|---------|------|----------------|
| 无rubric | DS-R1 | 80.1 |
| 无rubric | o3-mini | 71.0 |
| low score | DS-R1 | 14.1 |
| low score | o3-mini | 23.1 |
| extremely high | DS-R1 | 96.3 |
| extremely high | o3-mini | 97.9 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| CPT + BoN采样 | 更高分数、更少采样次数 | CPT为BoN提供更好的初始分布 |
| baseline + BoN采样 | 需要更多采样次数达到相同质量 | 基座模型的分布不如CPT集中 |

### 关键发现
- CPT在所有5个基座模型上都一致地大幅提升了分数准确率和排序相关性
- 最大提升出现在Mistral-Nemo-12B上（Acc: 0.60→0.83, ρ: 0.74→0.93）
- 教师模型确实能生成符合rubric不同分数水平的文本（极高分目标可达96-97分）
- CPT与Best-of-N采样正交互补：CPT模型在任意N下都优于baseline

## 亮点与洞察
- **对称偏好对设计**非常巧妙：同一对响应通过交换system prompt就能生成两个训练样本，既增加数据效率，又强制模型学会依赖system prompt做判断
- **rubric→system prompt→偏好数据**的三级流水线让整个过程完全自动化，不需要人工标注
- 仅用900个合成样本+1 epoch LoRA就能显著改变模型行为，数据效率极高
- 框架具有良好的可扩展性：可以定义任意新rubric来控制新维度

## 局限与展望
- 当前仅在开放式写作任务上验证，未涉及结构化输出、代码生成等更复杂场景
- rubric的设计仍需人工，如何自动生成/优化rubric是开放问题
- 生成质量依赖教师模型能力，教师模型的偏见可能传播到合成数据
- 当前仅支持单维度控制，多维度属性的组合控制尚未探索
- 评估依赖LLM judge，可能存在评估偏差
- 合成数据规模较小（900样本），大规模场景下效果有待验证

## 相关工作与启发
- **vs RFM (Barreto et al.)**: RFM通过端到端学习隐式奖励特征再用权重组合，CPT直接用显式rubric定义"特征"，更透明可控
- **vs P-RLHF (Li et al.)**: P-RLHF用用户嵌入条件化模型，CPT用自然语言system prompt，可解释性和通用性更强
- **vs 标准DPO**: 标准DPO固化单一偏好，CPT扩展DPO使其支持条件化偏好

## 评分
- 新颖性: ⭐⭐⭐⭐ 对称偏好对和rubric引导的合成数据方案新颖，但核心仍是DPO+system prompt条件化
- 实验充分度: ⭐⭐⭐ 5个模型验证一致性好，但场景单一（仅写作风格），缺少安全性/角色等维度
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，示例丰富，动机论述充分
- 价值: ⭐⭐⭐⭐ 提供了一种实用的推理时行为控制方案，对个性化LLM部署有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](../../ACL2026/interpretability/evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[NeurIPS 2025\] Rectifying Shortcut Behaviors in Preference-based Reward Learning](../../NeurIPS2025/interpretability/rectifying_shortcut_behaviors_in_preference-based_reward_learning.md)
- [\[CVPR 2025\] Geometry-Guided Camera Motion Understanding in VideoLLMs](../../CVPR2025/interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)
- [\[NeurIPS 2025\] Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis](../../NeurIPS2025/interpretability/toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)
- [\[ACL 2026\] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](../../ACL2026/interpretability/curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)

</div>

<!-- RELATED:END -->
