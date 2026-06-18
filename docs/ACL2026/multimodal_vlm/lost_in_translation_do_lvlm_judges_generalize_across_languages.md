---
title: >-
  [论文解读] Lost in Translation: Do LVLM Judges Generalize Across Languages?
description: >-
  [ACL 2026 Findings][多语言/翻译][多语言评估] 本文提出 MM-JudgeBench，首个大规模多语言多模态评判模型基准（25 种语言、60K+ 偏好实例），评估 22 个 LVLM 发现当前 LVLM 评判器存在显著的跨语言性能差异——模型大小和架构不能预测多语言鲁棒性，即使最先进的评判器也表现不一致，突显了多语言多模态评估基准的必要性。
tags:
  - "ACL 2026 Findings"
  - "多语言/翻译"
  - "多语言评估"
  - "LVLM评判"
  - "奖励模型"
  - "跨语言泛化"
  - "视觉语言基准"
---

# Lost in Translation: Do LVLM Judges Generalize Across Languages?

**会议**: ACL 2026 Findings  
**arXiv**: [2604.19405](https://arxiv.org/abs/2604.19405)  
**代码**: [https://github.com/tahmedge/mm-judgebench](https://github.com/tahmedge/mm-judgebench)  
**领域**: 多语言 / 模型评估  
**关键词**: 多语言评估, LVLM评判, 奖励模型, 跨语言泛化, 视觉语言基准

## 一句话总结

本文提出 MM-JudgeBench，首个大规模多语言多模态评判模型基准（25 种语言、60K+ 偏好实例），评估 22 个 LVLM 发现当前 LVLM 评判器存在显著的跨语言性能差异——模型大小和架构不能预测多语言鲁棒性，即使最先进的评判器也表现不一致，突显了多语言多模态评估基准的必要性。

## 研究背景与动机

**领域现状**：自动评估器（奖励模型/LLM-as-Judge）在 LVLM 开发中扮演核心角色，从训练对齐到模型选择和基准测试。然而，现有评估几乎完全基于英语。

**现有痛点**：(1) VL-RewardBench 和 Multimodal RewardBench 仅覆盖英语；(2) 多语言扩展（如 M-RewardBench）仅限文本模态；(3) 没有现有基准能统一研究跨语言和跨模态的奖励模型行为。

**核心矛盾**：LVLM 评判器被期望在多语言多模态设定中使用，但其可靠性仅在英语上验证。同一模型在英语上表现优秀但在法语上可能选择错误答案。

**本文目标**：(1) 构建首个多语言多模态评判基准；(2) 大规模评估 22 个 LVLM 的跨语言评判一致性；(3) 揭示当前奖励建模的多语言局限。

**切入角度**：使用高质量翻译模型（Gemini-3-Pro）将 VL-RewardBench 和 OpenCQA 翻译到 24 种语言（加英语共 25 种），严格质量过滤后构建控制实验。

**核心 idea**：通过固定视觉输入仅变化语言来隔离跨语言评估效应，揭示 LVLM 评判器在语言维度上的脆弱性。

## 方法详解

### 整体框架

整体上是"先选翻译器、再造数据、最后多维评估"的流程，并顺带产出一份多语言训练集：(1) 翻译模型选择——对比 Gemini 系列的翻译质量（LaBSE 和 CometKiwi 指标），选择 Gemini-3-Pro；(2) 数据集构建——翻译 VL-RewardBench（视觉语言偏好判断）和 OpenCQA（图表问答判断）到 24 种语言，经质量过滤得到 60K+ 实例，构成评估基准 MM-JudgeBench；(3) 多维度评估——对 22 个 LVLM 做配对准确率、位置偏差、长度偏差分析；(4) 多语言训练集——把 MM-RewardBench 同样翻译到 24 种语言得到 100K+ 实例的 M-MM-RewardBench，供开源模型做领域适应微调。下面三个关键设计依次对应数据集构建、评估协议与训练集。

### 关键设计

**1. MM-JudgeBench 数据集构建：固定视觉、只变语言，把"跨语言脆弱性"隔离出来**

现有评判基准要么只有英语（VL-RewardBench、Multimodal RewardBench），要么扩展了多语言却丢掉了视觉模态（M-RewardBench），没人能同时盯住语言和模态两个维度。本文用两个互补子集补上这个空缺：M-VL-RewardBench 测通用视觉语言偏好，M-OpenCQA 测图表中心的视觉文本推理；每个提示把查询和两个候选答案一起翻译到目标语言，而图像保持不变。这样一来，同一道题在 25 种类型学差异巨大的语言（从阿拉伯语到越南语）下唯一变动的就是文字，评判器选错就只能归因于语言而非内容。

为了让构建在成本上可控，作者用一条提示一次性翻译全部 24 种语言，相比逐语言调用直接省下 24 倍 API 开销；质量上则用 LaBSE 和 CometKiwi 双指标卡 0.75 阈值，低于阈值的样本经人工回译复核后重翻或删除，最终留下 60K+ 高质量实例。

**2. 多维度评估协议：不止看对不对，还要看模型为什么对**

只看配对准确率（正确识别偏好响应的比例）会把系统性偏差藏起来——一个评判器可能"碰巧"选对，却始终偏爱排在前面或更长的答案，这种倾向在真实部署里会放大成稳定的误差。因此协议额外测两类偏差：位置偏差通过把每对答案正序、反序各呈现一次、比较两次准确率的差值来量化；长度偏差则看模型是否系统性地偏向更长但错误的答案。三个指标合在一起，才能区分"真懂"和"靠捷径蒙对"。

**3. 多语言训练集 M-MM-RewardBench：给开源模型一条多语言适配的出路**

实验发现开源评判器在非英语上掉得最狠，光做诊断不给解药并不够。作者顺手把 MM-RewardBench 也翻译到 24 种语言，得到 100K+ 偏好实例的训练集，并刻意与评估数据不重叠，专门用于对开源模型做领域适应微调。它的价值在实验里得到验证——多语言微调能显著拉回非英语上的评判性能。

### 损失函数 / 训练策略

评估为零样本提示，要求 LVLM 选择更好的答案并提供理由。领域适应微调使用标准 SFT 在 M-MM-RewardBench 上进行。评估指标为配对准确率。

## 实验关键数据

### 主实验

**22 个 LVLM 在 MM-JudgeBench 上的平均准确率和方差**

| 模型 | 平均准确率 | 方差 | 说明 |
|------|----------|------|------|
| GPT-5 | 81.3% | 0.2 | 最稳定 |
| Gemini-2.5-Flash | ~78% | 低 | 接近 GPT-5 |
| Qwen3-VL-32B | ~77% | 低 | 开源最佳 |
| Gemma-3-27B | ~74% | 中 | 部分语言下降明显 |
| InternVL-3.5-8B | ~70% | 高 | 跨语言变异大 |
| LLaVA-Critic-7B | ~55% | 高 | 专用评判模型但仅英语训练 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 英语评估 | 最高 | 所有模型在英语上最强 |
| 低资源语言（哈萨克语等） | 下降最大 | 训练数据覆盖不足 |
| 效率优化变体 | 多语言崩溃 | 如 Gemini-Flash-Lite 英语强但多语言差 |
| + 推理增强 | 提升 | 要求提供理由改善评判 |
| + 多语言微调 | 显著提升 | 领域适应有效 |

### 关键发现

- 模型大小不能预测多语言鲁棒性——小模型 Qwen3-VL 在多语言上比许多更大模型更一致
- 效率优化变体（如 Flash-Lite）在英语上接近全尺寸版本，但多语言上严重退化
- LLaVA-Critic（专门训练的评判模型）因仅在英语上训练，多语言表现极差
- 位置偏差和长度偏差在非英语语言中更严重
- 领域适应微调和推理增强评判都能改善多语言性能

## 亮点与洞察

- 揭示了 LVLM 评判器的多语言"盲区"——整体平均分掩盖了语言间的巨大差异
- 效率优化变体的多语言崩溃是重要的实用警告——降低成本可能以牺牲公平性为代价
- 训练集 M-MM-RewardBench 的发布为社区改善多语言评判提供了直接支持

## 局限与展望

- 翻译可能引入系统性偏差（所有翻译来自同一模型）
- 25 种语言仍未覆盖世界上多数语言
- 未分析翻译质量如何影响评估结果
- 未来需要原生多语言（非翻译）的评估数据

## 相关工作与启发

- **vs VL-RewardBench**: 仅英语；MM-JudgeBench 扩展到 25 种语言
- **vs M-RewardBench**: 仅文本模态；MM-JudgeBench 增加视觉模态
- **vs Multimodal RewardBench**: 英语多模态；MM-JudgeBench 同时多语言和多模态

## 评分

- 新颖性: ⭐⭐⭐⭐ 填补了多语言多模态评判评估的空白
- 实验充分度: ⭐⭐⭐⭐⭐ 22 个模型、25 种语言、60K+ 实例
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现的实践含义阐述充分
- 价值: ⭐⭐⭐⭐⭐ 基准和训练集的发布对社区有持续价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)
- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](../../NeurIPS2025/multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](../../ACL2025/multilingual_mt/accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2026\] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages](from_traditional_taggers_to_llms_a_comparative_study_of_pos_tagging_for_medieval.md)
- [\[ACL 2025\] Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](../../ACL2025/multilingual_mt/lost_in_multilinguality_dissecting_cross-lingual_factual_inconsistency_in_transf.md)

</div>

<!-- RELATED:END -->
