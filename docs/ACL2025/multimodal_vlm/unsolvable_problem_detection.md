---
title: >-
  [论文解读] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models
description: >-
  [ACL 2025][多模态VLM][unsolvable problem detection] 提出 Unsolvable Problem Detection (UPD) 任务，通过三类不可解问题（缺失答案、不兼容选项、图文不匹配）系统评估大型多模态模型在面对无法回答的 MCQA 问题时是否能正确拒绝作答，揭示了现有 benchmark 无法衡量的可信度维度。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "unsolvable problem detection"
  - "trustworthiness"
  - "多模态"
  - "answer refusal"
  - "MCQA"
---

# Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models

**会议**: ACL 2025  
**arXiv**: [2403.20331](https://arxiv.org/abs/2403.20331)  
**代码**: [https://github.com/AtsuMiyai/UPD](https://github.com/AtsuMiyai/UPD)  
**领域**: 多模态VLM  
**关键词**: unsolvable problem detection, trustworthiness, multimodal evaluation, answer refusal, MCQA

## 一句话总结

提出 Unsolvable Problem Detection (UPD) 任务，通过三类不可解问题（缺失答案、不兼容选项、图文不匹配）系统评估大型多模态模型在面对无法回答的 MCQA 问题时是否能正确拒绝作答，揭示了现有 benchmark 无法衡量的可信度维度。

## 研究背景与动机

**领域现状**：多选题问答（MCQA）是评估 LMM 理解能力的主流方式，MMBench、MMMU 等 benchmark 被广泛使用。当前 LMM 在这些 benchmark 上表现优异，许多模型准确率已超过 80%。

**现有痛点**：高准确率并不意味着模型真正理解了答案。模型可能只是在所有选项中选择"最不离谱"的那个，而非真正知道正确答案。当正确答案不存在于选项中时，模型仍然会"强行"选一个答案，暴露了其缺乏真正理解的问题。

**核心矛盾**：现有评测只考虑"答案存在且可解"的理想场景，忽略了现实中可能出现的不可解情况。LLM 领域已有工作研究拒答能力，但多模态场景的不可解类型更多样（图文不匹配等），缺乏系统评估。

**本文目标** (1) 定义多模态场景下的不可解问题分类体系；(2) 构建严格的评测 benchmark；(3) 系统评估现有 LMM 的拒答能力并分析瓶颈。

**切入角度**：将多模态 MCQA 中的不可解问题归纳为三类：答案缺失（AAD）、选项集不兼容（IASD）、图文不匹配（IVQD），覆盖了图像、问题、选项三要素之间所有可能的不一致。

**核心 idea**：通过构造三类不可解问题来验证模型是否"真正理解"答案，而非仅仅做选项排除。

## 方法详解

### 整体框架

基于 MMBench 构建 MM-UPD Bench，包含三个子 benchmark：MM-AAD（820题）、MM-IASD（919题）、MM-IVQD（356题），共 2095 题。输入为带图像的多选题，模型需判断问题是否可解，可解则正常回答，不可解则拒绝作答。评测使用 Dual Accuracy 作为核心指标——要求模型在标准问题上答对且在对应不可解问题上正确拒答才算成功。

### 关键设计

1. **三类不可解问题定义**:

    - 功能：系统覆盖多模态 MCQA 中所有不可解情况
    - 核心思路：AAD 去掉正确选项考察模型是否能识别答案缺失；IASD 用完全无关的选项集替换，考察模型是否能识别选项与问题不匹配；IVQD 打乱图文配对，考察模型是否能识别图文不相关
    - 设计动机：三类问题分别测试不同层次的理解——AAD 测细粒度判断，IASD 测基本语义匹配，IVQD 测视觉-文本对齐

2. **三阶段 Benchmark 构建流程**:

    - 功能：确保评测数据高质量、无歧义
    - 核心思路：(1) 用 text-only GPT-4 + CircularEval 过滤掉不需要图像就能回答的问题；(2) 通过删除正确选项（AAD）、打乱选项集（IASD）、打乱图文对（IVQD）生成不可解问题；(3) 人工审核去除歧义样本
    - 设计动机：如果问题本身不依赖图像就能回答，则无法有效评测多模态理解。人工审核确保不可解问题确实不可解

3. **Dual Accuracy 评测指标**:

    - 功能：综合衡量模型在可解与不可解问题上的表现
    - 核心思路：只有当模型在标准问题上答对 **且** 在对应不可解问题上正确拒答时才计正确。$\text{Dual Acc} = \mathbb{1}[\text{Standard correct}] \times \mathbb{1}[\text{UPD correct}]$
    - 设计动机：单独看 Standard Accuracy 或 UPD Accuracy 都不够，前者忽略拒答能力，后者可能把什么都拒绝的模型分数打高

### 损失函数 / 训练策略

本文是评测工作，不涉及训练损失。评测设置包含三种 prompt 策略：Base（无提示）、Option（增加"以上皆非"选项）、Instruction（显式指示可以拒答）。额外探索了 CoT、Self-reflection 和微调三种改进策略。

## 实验关键数据

### 主实验

| 模型 | AAD-Base Dual | IASD-Base Dual | IVQD-Base Dual | AAD-Inst Dual | IASD-Inst Dual |
|------|-------------|---------------|---------------|-------------|---------------|
| LLaVA-OV-7B | 4.5 | 5.5 | 2.5 | 25.9 | 27.1 |
| InternVL2-8B | 28.5 | 30.1 | 28.4 | 34.0 | 56.5 |
| InternVL2-40B | 43.5 | 45.0 | 42.7 | 67.9 | 75.7 |
| Qwen2.5-VL-7B | 32.2 | 46.1 | 71.1 | 58.5 | 70.4 |
| GPT-4o | 45.6 | 56.1 | 65.2 | 59.3 | 68.0 |

（Dual Accuracy %。这些模型在原始 MMBench 上准确率均 >80%，但 UPD Base 可低至 <5%）

### 消融实验（CoT / Self-Reflection 效果）

| 模型 | 方法 | AAD Dual | IASD Dual | IVQD Dual |
|------|------|---------|----------|----------|
| LLaVA-OV-7B | Base | 4.5 | 5.5 | 2.5 |
| LLaVA-OV-7B | CoT | 37.9 | 36.7 | 14.9 |
| LLaVA-OV-7B | Self-Reflection | 27.6 | 35.4 | 31.7 |
| GPT-4o | Base | 45.6 | 56.1 | 65.2 |
| GPT-4o | CoT | 47.7 | 48.4 | 57.2 |
| GPT-4o | Self-Reflection | 55.2 | 57.9 | 57.9 |

### 关键发现

- MMBench 准确率与 UPD 表现几乎无相关性（UPD Accuracy 与 Original Standard 的相关系数最低仅 6.5），说明原有 benchmark 完全没有衡量这个维度
- 开源模型与闭源模型差距巨大：多数开源模型 Base 设置下 UPD 准确率 <10%，而 GPT-4o 在 IVQD-Base 达到 90.2%。差距来自闭源模型经过拒绝训练
- CoT 和 Self-Reflection 对语言端存在瓶颈的模型（如 LLaVA-OV）有效，对已有较强拒答能力的模型效果有限
- 瓶颈分析：即使直接告诉模型正确答案，LLaVA-OV 和 Qwen2VL 仍无法正确选择"None of the above"，说明问题在 LLM 的拒答能力而非视觉理解
- Qwen2.5-VL-7B 是开源 7B 级别中 UPD 表现最均衡的模型

## 亮点与洞察

- **瓶颈定位方法新颖实用**：通过"直接告诉模型答案+观察能否拒答"来区分瓶颈在视觉端还是语言端，简洁有效。这个方法可以迁移到任何需要诊断多模态模型组件瓶颈的场景
- **三类 UPD 的区分度设计巧妙**：AAD 与 IASD 的对比直接揭示模型是"选项太细分不出来"还是"根本不具备拒答意识"——如果 IASD（完全无关选项）都答错，说明模型压根不会说"不知道"
- **Dual Accuracy 指标可推广**：这种"正反双向验证"的思路可迁移到幻觉检测（在忠实和幻觉问题上都判断正确）、安全评估（正常使用和攻击场景同时考核）等

## 局限与展望

- 本文侧重评测设计，未提出针对 UPD 的新方法，fine-tuning 实验只是初步探索
- Benchmark 基于 MMBench 构建，问题难度偏低，未来需要更难的基础题目（MMMU 级别）
- 只考虑了单图单轮 MCQA，多图推理、开放式问答、多轮对话中的"拒绝回答"未涉及
- Fine-tuning 改善 UPD 可能以牺牲通用任务性能为代价，如何平衡两者是开放问题

## 相关工作与启发

- **vs Wang et al. (2025) LLM 拒答研究**: 他们只研究了 AAD 且针对 LLM，本文扩展到多模态并增加 IASD 和 IVQD 两类新问题，提供更细粒度诊断
- **vs SQuAD 2.0**: SQuAD 2.0 在阅读理解中引入不可答问题，本文将类似思想扩展到多模态 MCQA 并设计了更丰富的不可解类型
- **vs 幻觉检测**: UPD 关注"模型是否知道自己不能答"，幻觉关注"模型答了但答错了"，两者互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 三类不可解问题的分类和 Dual Accuracy 指标设计新颖，但核心思想延续自 LLM 拒答研究
- 实验充分度: ⭐⭐⭐⭐⭐ 评测了 20+ 模型，包含瓶颈分析、CoT/Self-Reflection、Fine-tuning 等多角度实验
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，图表设计直观，发现归纳层层递进
- 价值: ⭐⭐⭐⭐ 揭示了现有 benchmark 忽略的可信度维度，对 LMM 安全对齐有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ACL 2025\] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)
- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)
- [\[ACL 2025\] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification](sciver_evaluating_foundation_models_for_multimodal_scientific_claim_verification.md)

</div>

<!-- RELATED:END -->
