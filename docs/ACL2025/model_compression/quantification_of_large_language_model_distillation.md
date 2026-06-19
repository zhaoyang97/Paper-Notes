---
title: >-
  [论文解读] Quantification of Large Language Model Distillation
description: >-
  [ACL 2025][模型压缩][知识蒸馏量化] 本文提出了两种互补的LLM蒸馏量化方法——身份一致性评估（ICE）和响应相似性评估（RSE），通过越狱攻击挖掘模型身份信息泄露和多粒度响应相似性来衡量模型的蒸馏程度，发现大多数知名LLM（除Claude、Doubao和Gemini外）都表现出较高的蒸馏程度。
tags:
  - "ACL 2025"
  - "模型压缩"
  - "知识蒸馏量化"
  - "身份一致性"
  - "响应相似性"
  - "LLM同质化"
  - "越狱攻击"
---

# Quantification of Large Language Model Distillation

**会议**: ACL 2025  
**arXiv**: [2501.12619](https://arxiv.org/abs/2501.12619)  
**代码**: [https://github.com/Aegis1863/LLMs-Distillation-Quantification](https://github.com/Aegis1863/LLMs-Distillation-Quantification)  
**领域**: 模型压缩  
**关键词**: 知识蒸馏量化, 身份一致性, 响应相似性, LLM同质化, 越狱攻击

## 一句话总结
本文提出了两种互补的LLM蒸馏量化方法——身份一致性评估（ICE）和响应相似性评估（RSE），通过越狱攻击挖掘模型身份信息泄露和多粒度响应相似性来衡量模型的蒸馏程度，发现大多数知名LLM（除Claude、Doubao和Gemini外）都表现出较高的蒸馏程度。

## 研究背景与动机
模型蒸馏已成为构建LLM的基础技术，通过从强模型到弱模型的知识转移可以显著降低成本。然而，蒸馏也带来了模型同质化的风险——不同团队开发的模型变得越来越相似，降低了多样性，削弱了应对复杂或新颖任务的能力。

当前面临的核心矛盾是：蒸馏过程不透明，缺乏标准基准数据，而且蒸馏知识可能以抽象形式嵌入表示中难以直接解读。更关键的是，学术界广泛使用蒸馏数据但缺乏对其问题的批判性审视。本文的切入点是：**从身份认知矛盾和响应相似性两个角度，系统地量化LLM的蒸馏程度**，为LLM开发的透明性和独立性提供工具。

## 方法详解

### 整体框架
提出两个互补的评估指标：ICE检测模型是否在蒸馏过程中意外继承了教师模型的身份信息，RSE测量目标模型与参考模型响应的相似程度。两者结合提供全面的蒸馏量化评估。

### 关键设计
1. **身份一致性评估（ICE）**:

    - 核心思路：如果模型A是从模型B蒸馏而来，那么A可能意外学习了B的身份信息（如名称、开发者等）
    - 利用GPTFuzz开源越狱框架，迭代生成对抗性prompt来绕过模型的自我意识约束
    - 定义事实集F，包含各源模型的身份描述（如"我是Claude，由Anthropic开发"）
    - 三级评估指标：
        - **Loose Score**: 任何身份矛盾即视为成功攻击
        - **Strict Score**: 仅当模型错误地将自己识别为其他已知实体时才计入
        - **Hard Score**: 最严格，要求prompt不含身份关键词，但回复中包含身份关键词（过滤上下文诱导）
    - 攻击prompt覆盖5个领域：团队归属、合作关系、行业参与、技术专长、地理信息

2. **响应相似性评估（RSE）**:

    - 核心思路：蒸馏模型的响应风格、逻辑结构和内容细节会与教师模型相似
    - 使用GPT-4o-0806作为参考模型（因为GPT系列是最常被蒸馏的来源）
    - 三个评估数据集：ArenaHard（通用推理）、Numina（数学推理）、ShareGPT（指令跟随）
    - 采用LLM-as-a-judge方法，将相似性评分为1-5分五个等级
    - 从风格、逻辑、内容三个维度评估
    - 与传统的n-gram相似度和BERTScore对比，RSE能捕捉逻辑层面的信息

3. **验证与对比分析**:

    - 对Qwen2.5-7B-Instruct进行SFT验证RSE有效性：随SFT epoch增加，RSE评分持续上升
    - Base模型 vs Instruct模型的对比
    - 推理模型（DeepSeek-R1等）的评估

### 损失函数 / 训练策略
本文是评估方法，不涉及训练。ICE使用GPTFuzz的MCTS算法迭代优化攻击prompt，50个种子prompt，每次选取子集进行优化。RSE使用LLM裁判打分。

## 实验关键数据

### 主实验 - ICE结果

| 模型 | Loose Score | Strict Score | Hard Score | 蒸馏程度 |
|------|------------|-------------|------------|---------|
| Claude3.5-Sonnet | 极低 | 极低 | 极低 | 低 |
| Doubao-Pro-32k | 极低 | 极低 | 极低 | 低 |
| Gemini-2.0-Flash | 低 | 低 | 低 | 低 |
| GLM4-Plus | 高 | 高 | 中 | 高 |
| Qwen-Max-0919 | 高 | 高 | 较高 | 高 |
| DeepSeek-V3 | 高 | 0.25 | 0.07 | 高 |

### 主实验 - RSE结果（以GPT4o-0806为参考）

| 模型 | RSE评分 | 2-gram | BERTScore | 蒸馏程度 |
|------|---------|--------|-----------|---------|
| Llama3.1-70B-Instruct | 3.628 | 0.213 | 0.828 | 低 |
| Doubao-Pro-32k | 3.720 | 0.216 | 0.823 | 低 |
| Claude3.5-Sonnet | 3.740 | 0.189 | 0.823 | 低 |
| DeepSeek-V3 | 4.102 | 0.220 | 0.837 | 高 |
| Qwen-Max-0919 | 4.174 | 0.252 | 0.838 | 高 |
| GPT4o-0513 | 4.240 | 0.269 | 0.841 | 高 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Base vs Instruct (Qwen系列) | Base Strict Score更高 | Base模型更容易泄露蒸馏痕迹 |
| DeepSeek-V3 vs R1 | 无显著差异 | R1基于V3训练，身份微调不多 |
| RSE验证(SFT 1-3 epoch) | Score持续增长(3.554→4.222 on ArenaHard) | 证明RSE能有效检测蒸馏 |
| Qwen-Max引用Claude | Qwen-Max含Claude引用 | 暗示Qwen-Max可能蒸馏了Claude |

### 关键发现
- Claude、Doubao、Gemini在ICE和RSE上都表现出较低的蒸馏程度，说明这些模型更可能是独立开发的
- 大多数开源和闭源LLM（包括GLM4-Plus、Qwen-Max、DeepSeek-V3等）都表现出较高的蒸馏程度
- Base模型的蒸馏程度普遍高于对齐后的Instruct模型
- Qwen-Max的回复中经常出现Claude3.5-Sonnet的引用，而Qwen 2.5系列主要引用GPT
- ICE显示LLM在"团队"、"行业"和"技术"类别上更容易被越狱

## 亮点与洞察
- 首次系统性地提出LLM蒸馏量化框架，填补了一个重要的研究空白
- ICE方法非常巧妙：利用越狱攻击来挖掘蒸馏留下的"身份指纹"，思路新颖
- RSE通过多维度评估（风格/逻辑/内容）比简单的文本相似度更有信息量
- 实验覆盖了主流闭源和开源模型，结论具有行业参考价值
- Qwen-Max引用Claude的发现很有趣，暗示了复杂的蒸馏链条

## 局限与展望
- ICE依赖越狱攻击的成功率，如果模型安全对齐做得好可能会漏检蒸馏
- RSE以GPT4o为参考模型，但如果目标模型是从其他模型（如Claude）蒸馏的，可能被低估
- Loose Score的正样本准确率仅0.78-0.90，假阳性问题需要关注
- 无法区分"直接蒸馏"和"间接蒸馏"（如使用GPT生成的数据训练）
- 研究idea：可以结合模型内部表示（如attention pattern分析）来提供更细粒度的蒸馏链条追溯
- 缺少对蒸馏程度与实际性能关系的深入分析

## 相关工作与启发
- 与数据污染检测方法（如LM Contamination Index）有联系，但本文关注的是模型间的知识转移而非训练集泄露
- 越狱攻击（GPTFuzz）被重新定义为检测工具而非攻击工具，视角转换很有启发
- 对LLM开发透明性的呼吁有重要的社会意义，特别是在学术界大量使用蒸馏数据的背景下

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次提出LLM蒸馏量化问题并给出可操作框架，但ICE基于已有的GPTFuzz
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种模型、多种指标、有人工验证，但缺少ground truth蒸馏关系的验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验组织合理，但部分符号和公式可以更简洁
- 价值: ⭐⭐⭐⭐ 对LLM开发透明性有重要推动作用，但量化结果的可靠性仍需更多验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](drpruning_robust_pruning.md)
- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)
- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)

</div>

<!-- RELATED:END -->
