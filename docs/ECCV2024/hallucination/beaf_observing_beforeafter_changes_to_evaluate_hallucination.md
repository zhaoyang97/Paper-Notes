---
title: >-
  [论文解读] BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models
description: >-
  [ECCV 2024][幻觉检测][视觉语言模型] BEAF提出"前-后对比"的幻觉评估范式：通过图像编辑移除物体后观察VLM回答的变化，引入TU/IG/SB/ID四个变化感知指标，揭示了传统文本轴评估无法发现的幻觉行为。 1. 领域现状：VLM（如LLaVA、InstructBLIP等）展现出强大的多模态推理能力…
tags:
  - "ECCV 2024"
  - "幻觉检测"
  - "视觉语言模型"
  - "幻觉评估"
  - "场景操纵"
  - "图像编辑"
  - "Benchmark"
---

# BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2407.13442](https://arxiv.org/abs/2407.13442)  
**代码**: [https://beafbench.github.io/](https://beafbench.github.io/)  
**领域**: 幻觉检测  
**关键词**: 视觉语言模型, 幻觉评估, 场景操纵, 图像编辑, Benchmark

## 一句话总结

BEAF提出"前-后对比"的幻觉评估范式：通过图像编辑移除物体后观察VLM回答的变化，引入TU/IG/SB/ID四个变化感知指标，揭示了传统文本轴评估无法发现的幻觉行为。

## 研究背景与动机

1. **领域现状**：VLM（如LLaVA、InstructBLIP等）展现出强大的多模态推理能力，但容易产生幻觉——输出不反映输入图像的真实内容。POPE是主流幻觉评估benchmark，采用判别式问答格式。

2. **现有痛点**：(1) 现有benchmark（POPE、CIEM、AMBER）只操纵文本轴（构造不同问题），不操纵视觉轴，无法判断VLM是否真正"看到"了物体还是仅靠语言偏见回答；(2) 某些物体总是共现（如桌子和椅子），仅靠问答评估无法区分VLM是真理解还是利用共现偏置。

3. **核心矛盾**：VLM是多模态输入（图像+文本），但现有评估只考虑文本轴变化。如果某物体被移除后VLM仍说"Yes"，说明它根本没看图像，但传统accuracy指标无法捕捉这一信息。

4. **本文要解决什么？** 设计同时考虑视觉和文本双轴变化的评估框架，通过观察VLM对场景变化的感知能力来更精细地分析幻觉。

5. **切入角度**：核心假设——如果从图像中移除苹果后问"有苹果吗？"，真正理解场景的模型应从"Yes"变为"No"。通过跟踪答案变化可区分"真理解"与各种幻觉模式。

6. **核心idea一句话**：通过图像编辑操纵视觉场景，观察VLM回答的前后变化，引入变化感知指标来细粒度评估幻觉。

## 方法详解

### 整体框架

BEAF benchmark包含两部分：(1) 数据集构建——从MS-COCO选取500张原始图像，通过SAM+LaMa移除物体生成1727张操纵图像，配合26K个图像-问题对；(2) 评估指标——设计4个变化感知指标（TU、IG、SB、ID），从不同维度分析VLM的幻觉行为。

### 关键设计

1. **三阶段图像操纵Pipeline**:
    - 做什么：从原始图像中精确移除物体生成高质量操纵图像
    - 核心思路：Stage 1用SAM提取掩码+LaMa修复模型自动移除目标物体；Stage 2人工过滤低质量结果（阴影残留、修复失败等）；Stage 3人工精细修复——消除幽灵阴影、残影、碎片物体等线索
    - 设计动机：如果操纵图像中存在物体移除的痕迹（如阴影），VLM可能通过这些线索猜测物体曾经存在，污染评估。三阶段设计确保操纵图像接近自然图像

2. **四个变化感知评估指标**:
    - **True Understanding (TU)**：衡量模型是否真正理解场景——移除前后都答对。$TU = \frac{|Filter(True, True, True)|}{|Filter(*, *, True)|} \times 100$
    - **IGnorance (IG)**：衡量模型缺乏认知——移除前后都答错。高IG说明模型对该物体一无所知
    - **StuBbornness (SB)**：衡量模型固执不变——移除后仍给同样答案。分为SBp（固执说Yes）和SBn（固执说No），$SB = 100 - TU - IG$
    - **InDecision (ID)**：衡量模型对非相关物体的回答变化——不应变但变了，说明回答是随机的
    - 设计动机：传统accuracy无法区分"答对但不理解"和"真正理解"。例如模型移除前后都说"Yes"，传统评估认为移除前答对，但BEAF通过SBp指标揭示这是固执而非理解

3. **双轴分析框架**:
    - 做什么：结合视觉轴（场景变化）和文本轴（问题变化）进行全面分析
    - 核心思路：对每个(原始图像, 操纵图像, 问题)三元组，记录模型在两个图像上的回答，结合问题是否与被移除物体相关（标志R），计算四个指标
    - 设计动机：单独的文本轴评估可能高估模型能力——某些"正确"回答实际上是基于共现偏置而非视觉理解

### 损失函数 / 训练策略

纯评估工作，无训练过程。

## 实验关键数据

### 主实验

| 模型 | 参数量 | TU↑ | IG↓ | SB↓ | ID↓ | F1↑ |
|------|--------|-----|-----|-----|-----|-----|
| LLaVA | 13B | 56.4 | 8.3 | 35.3 | 11.2 | 67.1 |
| InstructBLIP | 13B | 42.1 | 3.5 | 54.4 | 8.7 | 56.8 |
| Shikra | 7B | 58.2 | 7.1 | 34.7 | 10.5 | 68.9 |
| mPLUG-Owl | 7B | 45.3 | 12.4 | 42.3 | 14.1 | 56.0 |

### 消融实验（传统评估 vs BEAF评估对比）

| 模型 | POPE Accuracy | BEAF TU | 差异分析 |
|------|--------------|---------|----------|
| InstructBLIP | ~85% | 42.1% | POPE高估了理解能力 |
| LLaVA | ~83% | 56.4% | 差距更小说明LLaVA更依赖视觉 |

### 关键发现

- **InstructBLIP的高SBp暴露严重问题**：该模型倾向于不管场景如何变化都回答"Yes"，传统accuracy无法发现这一偏好
- **Shikra的位置感知训练有助于减少幻觉**：其TU最高，可能因为位置感知策略帮助了物体存在性判断
- **传统评估中的"正确答案"可能是幻觉**：BEAF揭示了不少原本被认为"non-hallucination"的回答实际是SB（固执重复同一答案）
- **物体共现关系影响幻觉模式**：移除一个物体后，与之常共现的其他物体的ID值显著升高

## 亮点与洞察

- **评估范式创新**：从"静态问答"转向"动态变化感知"，这是VLM评估的重要范式转变。通过操纵视觉输入观察行为变化，类似心理学中的对照实验设计
- **四指标体系精确刻画幻觉类型**：TU/IG/SB/ID分别对应"真理解/无知/固执/犹豫"，提供了比accuracy更细粒度的诊断工具
- **揭示了VLM的"鸵鸟策略"**：许多VLM在面对不确定时不是随机猜测，而是固执地重复某个偏好答案（通常是Yes），这在SBp指标中清晰体现

## 局限性 / 可改进方向

- 数据集规模较小（500张原始图像），可能无法覆盖所有场景类型
- 图像编辑质量虽经人工检查，但仍可能存在不完美的修复痕迹
- 目前只评估判别式问答（Yes/No），未覆盖生成式描述的幻觉
- 只测了4个VLM，需要扩展到GPT-4V等更新模型
- 物体移除可能改变场景的整体语义（如移除人后场景语义大变），影响非相关物体的ID度量

## 相关工作与启发

- **vs POPE**: POPE只操纵文本轴（构造正/负样本问题），BEAF同时操纵视觉轴，提供更全面的评估
- **vs AMBER**: AMBER增加了生成式评估但仍限于文本轴操纵，BEAF的视觉操纵维度是独特贡献
- 这种"前后对比"的评估思路可迁移到视频理解幻觉评估（前后帧物体变化）和3D场景理解评估

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 视觉轴操纵+变化感知指标，评估范式有突破性
- 实验充分度: ⭐⭐⭐⭐ 指标设计严谨，多模型对比分析深入
- 写作质量: ⭐⭐⭐⭐ 概念解释清晰，示例直观
- 价值: ⭐⭐⭐⭐⭐ 对VLM幻觉评估领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](../../ACL2026/hallucination/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ICML 2025\] Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models](../../ICML2025/hallucination/look_twice_before_you_answer_memory-space_visual_retracing_for_hallucination_mit.md)
- [\[CVPR 2026\] SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models](../../CVPR2026/hallucination/svhalluc_benchmarking_speech-vision_hallucination_in_audio-visual_large_language.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](../../ACL2026/hallucination/mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](../../ACL2026/hallucination/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)

</div>

<!-- RELATED:END -->
