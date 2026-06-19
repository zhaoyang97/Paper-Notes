---
title: >-
  [论文解读] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models
description: >-
  [NeurIPS 2025][多模态VLM][主动安全] 提出PaSBench基准评估多模态语言模型的主动风险感知能力——要求模型在无用户提问的情况下主动观察环境并发出安全预警。评测36个模型发现最强模型（Gemini-2.5-pro）仅达71%准确率且45%的风险无法稳定检测，核心瓶颈是不稳定的主动推理能力而非知识缺失。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "主动安全"
  - "风险检测"
  - "LLM评测"
  - "benchmark"
  - "前瞻性推理"
---

# Towards Evaluating Proactive Risk Awareness of Multimodal Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.17455](https://arxiv.org/abs/2505.17455)  
**代码**: [HuggingFace](https://huggingface.co/datasets/Youliang/PaSBench)  
**领域**: 多模态VLM  
**关键词**: 主动安全, 风险检测, LLM评测, benchmark, 前瞻性推理

## 一句话总结

提出PaSBench基准评估多模态语言模型的主动风险感知能力——要求模型在无用户提问的情况下主动观察环境并发出安全预警。评测36个模型发现最强模型（Gemini-2.5-pro）仅达71%准确率且45%的风险无法稳定检测，核心瓶颈是不稳定的主动推理能力而非知识缺失。

## 研究背景与动机

现实生活中人们常常因为安全知识或意识不足而面临风险。理想的AI安全系统应该是**主动的（proactive）**——不等用户主动提问，而是自动观察环境和行为、检测潜在危险并及时预警。

现有AI安全研究存在根本性的范式局限：

**被动范式（Reactive）**：现有安全benchmark（SafeText、HealthBench、LabSafetyBench等）都假设用户已经意识到风险并主动提问，本质上是问答模式

**关注AI自身安全**：大量工作研究LLM是否会产生有害输出（毒性、偏见），而非LLM是否能保护人类安全

**缺乏主动能力评估**：没有benchmark专门评估模型的主动风险检测能力

本文定义了一个全新的任务——**主动风险检测**：给定一系列观察（文本日志或图像序列），模型需要在无用户询问的情况下，判断当事人是否正处于或即将面临不安全状况，并主动发出警告。这比传统安全评测远更贴近真实场景。

## 方法详解

### 整体框架

PaSBench的构建遵循知识收集→样本生成→质量控制的流程，最终产出两类评测：文本日志集（288条）和图像集（128条），覆盖5大安全领域。

### 关键设计

1. **知识收集与筛选**

   从中文安全科普书籍和政府官方网站收集日常生活安全知识，遵循5项严格原则：
    - **用户特异性**：聚焦个人行为导致的风险（如采食野生蘑菇），排除群体/社会层面风险
    - **风险确定性**：风险与伤害间有明确因果关联
    - **知识相关性**：只保留当前有效的安全知识
    - **后果严重性**：风险必须导致显著伤害
    - **可验证性**：不确定的知识需通过Google搜索5分钟内验证

   三名标注员交叉核查，从495条中筛选出288条知识点。

2. **图像观察生成**

   对每个知识点，GPT-4o生成1-4张文本到图像提示草稿，人工修改后用GPT-4o-image逐步生成图像序列（后一张参考前一张以保持一致性）。每张图像经人工检查：一致性、真实性、语义表达准确性。不合格可重试最多10次。最终收集128个图像样本。

3. **文本日志观察生成**

   随机生成人物画像（姓名、性别、住所），结合安全知识用DeepSeek-R1生成职业和爱好（需与风险类型匹配），然后生成完整日志。格式为：[时间]...[地点]...[环境观察]...[行为观察]。关键约束：观察必须在安全事件发生**之前**结束，使模型的提醒有实际预防价值。

4. **评测协议**

   对每个样本运行模型N=16次，报告三个指标：
    - **准确率（Average-of-N）**：正确识别并解释风险的比例
    - **潜力（Best-of-N）**：至少一次正确的比例
    - **鲁棒性（Worst-of-N）**：所有16次都正确的比例

   正确需同时满足：(1) 识别——警告用户停止危险行为；(2) 解释——给出合理的风险原因。用GPT-4.1作为评判模型（人工验证准确率94.5%）。

### 损失函数 / 训练策略

PaSBench是纯评测benchmark，不涉及训练。

## 实验关键数据

### 主实验：模型风险检测率

| 模型 | 图像准确率 | 图像鲁棒性 | 文本准确率 | 文本鲁棒性 |
|------|-----------|-----------|-----------|-----------|
| Gemini-2.5-pro | 71% | 55% | 64% | ~45% |
| Gemini-2.0-pro | ~65% | ~45% | **最佳** | ~50% |
| GPT-4.1 | ~60% | ~35% | ~58% | ~38% |
| Claude-3.5-sonnet | ~55% | ~30% | ~55% | ~35% |
| o1 | - | - | ~50% | ~25% |
| Qwen2.5-VL-7B | 23% | <5% | - | - |
| GPT-4.1-nano | ~25% | <5% | 20% | <5% |

### 知识 vs 检测能力对比

| 评测类型 | Gemini-2.5-pro | GPT-4.1-nano |
|---------|---------------|--------------|
| 多选题知识测试 | 87%~94.5% | >80% |
| 被动模式（给知识问是否违反） | 93% (552/596 失败样本中) | 75% (1047/1393) |
| 主动模式（无提示直接检测） | ~71% | ~25% |

### 消融实验：主动 vs 被动能力

| 模型 | 主动失败样本数 | 被动模式下成功检测比 | 说明 |
|------|-------------|---------------------|------|
| Gemini-2.5-pro（图像） | 596 | 552/596 (93%) | 绝大多数失败是主动能力不足 |
| GPT-4.1-nano（图像） | 1393 | 1047/1393 (75%) | 同上 |
| Gemini-2.5-pro（文本） | 1646 | 1217/1646 (74%) | 知识有，应用不稳定 |

### 关键发现

- **模型拥有安全知识但无法稳定主动应用**：多选题准确率>80%但主动检测率仅20-71%
- **推理模型不一定更好**：非推理模型Gemini-2.0-pro在文本集上是最佳，o1等推理模型反而表现不佳
- **模型规模matters**：几乎所有对比中大模型一致优于小模型（pro > flash, sonnet > haiku）
- **潜力高但鲁棒性低**：GPT-4.1-nano通过128次采样可覆盖91.4%的风险，但单次仅30%
- 图像和文本检测率高度相关（Pearson 0.897），说明瓶颈是模态无关的主动分析能力
- 观察序列长度在当前范围内不显著影响性能

## 亮点与洞察

- **定义了全新的评测维度**："主动安全"是一个被严重忽视但极其重要的AI能力
- 诊断分析非常深入：通过知识测试→被动检测→主动检测的层层剥离，精准定位了瓶颈
- Best-of-N和Worst-of-N的三维评测比单一准确率更有洞察力
- 明确指出**当前瓶颈不是知识不足，而是主动推理的不稳定性**，为改进指明方向

## 局限与展望

- 样本量较小（416个），每个图像样本仅2-3张子图，文本4-8段观察，可能不足以测试长序列理解
- 未考虑风险严重程度分级和误报问题，实际部署时需平衡预警频率和用户体验
- 数据主要来自中文安全科普，可能存在文化偏差
- 文章提出的改进方向（GRPO强化学习鼓励主动提醒、"提议-验证"流水线）值得深入探索
- 未覆盖连续数据流场景，实际部署需解决何时截断流做评估的问题

## 相关工作与启发

- SafeText/HealthBench等被动安全评测是本文的对标对象
- ProAgent等主动LLM工作关注的是对话中主动提问，本文推广到安全监控场景
- GRPO（DeepSeek-R1的训练方法）被建议用于训练主动能力
- 启发：主动型AI是下一代安全助手的核心能力，可结合可穿戴设备、家居传感器构建实时保护系统

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 定义了全新的"主动安全"评测任务，填补重要空白
- **实验充分度**: ⭐⭐⭐⭐ 36个模型的大规模评测和深入诊断分析；数据量较小是遗憾
- **写作质量**: ⭐⭐⭐⭐⭐ 动机阐述令人信服，实验分析层层递进，洞察力强
- **价值**: ⭐⭐⭐⭐⭐ 对AI安全领域具有方向性意义，数据集已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](../../ACL2025/multimodal_vlm/cant_see_the_forest_for_the.md)

</div>

<!-- RELATED:END -->
