---
title: >-
  [论文解读] Acknowledging Focus Ambiguity in Visual Questions
description: >-
  [ICCV 2025][多模态VLM][视觉问答] 首次定义并系统研究视觉问答中的**焦点歧义**（focus ambiguity）问题——当问题中的语言描述可能指向图像中多个合理区域时，现有 VQA 系统完全忽略了这种歧义。作者构建了 VQ-FocusAmbiguity 数据集（5,500 样本 + 12,880 实例分割），并证明现代模型在识别和定位焦点歧义方面表现很差。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉问答"
  - "焦点歧义"
  - "数据集"
  - "视觉定位"
  - "多模态基准"
---

# Acknowledging Focus Ambiguity in Visual Questions

**会议**: ICCV 2025  
**代码**: 无 (数据集公开于 [https://vizwiz.org/](https://vizwiz.org/))  
**领域**: 多模态VLM  
**关键词**: 视觉问答, 焦点歧义, 数据集, 视觉定位, 多模态基准

## 一句话总结

首次定义并系统研究视觉问答中的**焦点歧义**（focus ambiguity）问题——当问题中的语言描述可能指向图像中多个合理区域时，现有 VQA 系统完全忽略了这种歧义。作者构建了 VQ-FocusAmbiguity 数据集（5,500 样本 + 12,880 实例分割），并证明现代模型在识别和定位焦点歧义方面表现很差。

## 研究背景与动机

VQA（视觉问答）研究已经考虑了多种歧义来源——如答案的主观性、粒度差异等——但**没有一项工作**考虑过问题本身中语言描述指向位置的歧义性。

**为什么这很重要？** 举一个具体例子：一位盲人用户拍照后问"What is the cleaning product?"，如果图片中同时出现了洗洁精和窗户清洁剂，VQA 系统应该能意识到这个问题存在焦点歧义，而不是武断地只回答一个。错误的回答可能导致严重后果——比如盲人误用窗户清洁剂来洗碗。

**核心区分：问题定位 vs 答案定位**。作者强调了一个关键洞察：**问题的焦点区域和答案的视觉证据可以不同**。例如，问"What is above the mirror?"时，问题焦点是镜子（mirror），而答案证据是镜子上方的物体。在 330 个 AnswerTherapy 样本中，79% 的歧义问题和 36% 的非歧义问题存在问题定位与答案定位的不一致。这说明将问题焦点作为独立研究对象是必要的。

## 方法详解

### 整体框架

本文的核心贡献是**数据集构建 + 任务定义 + 基准测试**，而非提出新模型。工作包含三个层面：

1. **VQ-FocusAmbiguity 数据集**：5,500 个视觉问题，每个配有完整的焦点区域分割标注
2. **两个新任务**：（a）识别问题是否存在焦点歧义，（b）定位问题的所有可能焦点区域
3. **现代模型基准测试**：评估 4 个基础模型在任务 (a) 上的表现，3 个方法/管线在任务 (b) 上的表现

### 关键设计

**数据集构建——四个多样化来源**：

数据来源于四个不同的数据集，覆盖了多样化的图像内容和问题类型：

| 数据源 | 图像特征 | 问题来源 | 非歧义比例 |
|--------|---------|---------|-----------|
| PACO (COCO 2017) | 复杂场景，多物体 | 合成 + 标注者 | 50% (2,272) |
| MSRA-B | 单前景物体 | 合成（"What is this?"变体） | 100% (626) |
| AnswerTherapy-VQAv2 | COCO场景 | 人工创建 | 47% (82) |
| AnswerTherapy-VizWiz | 视障人士拍摄 | 视障用户语音 | 53% (83) |

**为什么需要四个来源？** 这是为了确保数据在以下维度上的多样性：（1）单物体 vs 复杂场景，（2）明眼人 vs 视障人士拍摄，（3）不同位置和大小的目标物体，（4）打字问题 vs 口语转录问题。

**PACO 标注流程**的精心设计：
- 标注界面展示图像及所有可用分割
- 先提供 AI 生成的候选问题供选择（标注者可选择：从头写、直接采用、修改后采用）
- 然后标注者选择问题可能指向的所有分割区域
- 对于歧义问题，标注者自主创作的比例最高（55%），说明 AI 较难生成好的歧义问题

**歧义原因分析**：通过对 265 个歧义样本的人工编码，发现两个主要原因：
- **相同类别多实例**（61.5%）：如"What is next to the mirror?" 而图中有多面镜子
- **不同类别多实例**（31%）：如"What is this?" 指向不确定的物体，多见于视障用户的模糊提问

**数据集划分**——支持 zero/few-shot 学习：训练集和验证集各 70 个样本（从每个来源随机采样 10 个歧义 + 10 个非歧义），测试集 5,360 个样本。这反映了当前 SOTA 通常来自基础模型的 zero/few-shot 设置这一趋势。

### 损失函数 / 训练策略

本文**不训练新模型**，而是在两个任务上对现有模型进行系统评测：

**任务一：焦点歧义识别**（二分类）
- 模型：GPT-4o、InternVL2-76B、Qwen2.5-VL-7B、Molmo-7B
- 5 种提示策略：ZS、ZS-CoT、ZS-ECoT（结构化推理引导）、FS、FS-ECoT
- 评估指标：Accuracy、Weighted F1、Positive Rate、Undecided Rate

**任务二：焦点区域定位**（实例分割）
- 模型方案：GLaMM（直接分割）、GPT-4o+GLaMM（描述后分割）、Molmo+SAM（点定位后分割）
- 评估指标：mAP、Union IoU、Max IoU

## 实验关键数据

### 主实验

焦点歧义识别结果（所有百分比）：

| 模型 | 最佳 Prompt | Accuracy | F1 | Positive Rate |
|------|------------|----------|-----|--------------|
| GPT-4o (>200B) | ZS-CoT | 69.6 | 69.8 | 53.3 |
| InternVL2 (76B) | ZS-CoT | 56.7 | 54.8 | 27.9 |
| Qwen2.5-VL (7B) | ZS-ECoT | 65.5 | 65.3 | 59.0 |
| Molmo (7B) | ZS-CoT | 56.9 | 57.1 | 48.1 |

焦点区域定位结果：

| 方法 | 最佳 Prompt | mAP | Union IoU | Max IoU |
|------|------------|-----|-----------|---------|
| GLaMM | ZS | 13.01 | 41.90 | 43.69 |
| GPT-4o+GLaMM | FS | 14.24 | 40.97 | 47.83 |
| Molmo+SAM | ZS-CoT | 24.3 | 36.2 | 45.4 |

### 消融实验

问题特征与歧义性的关系分析：

| 特征 | 歧义问题 | 非歧义问题 | 说明 |
|------|---------|-----------|------|
| 平均词数 | 较少 | 较多 | 更多单词提供更多消歧上下文 |
| 含复数名词比例 | 4.7% | 23.8% | 复数形式天然减少歧义 |
| 分割区域数量中位数 | 3 | 1（定义） | 歧义问题平均 4 个焦点区域 |
| 问题-答案定位匹配 | 21% 匹配 | 64% 匹配 | 歧义问题更需要独立的问题定位 |

CoT 提示策略的提升效果：
- Molmo-7B：ZS-CoT 比 ZS 提升 **18.4 pp**
- Qwen2.5-VL：ZS-ECoT 比 ZS 提升 **8.3 pp**
- 7B 模型通过 CoT 可达到甚至超越 76B InternVL2

### 关键发现

1. **所有模型表现都很差**：最佳准确率仅 69.6%（GPT-4o），说明焦点歧义是一个未被解决的难题
2. **训练数据很重要**：Qwen2.5-VL 和 Molmo 在 PixMo 数据集上训练了区域级计数和指向任务，这与歧义识别天然相关（计数单区域 = 无歧义，多区域 = 有歧义）
3. **InternVL2 虽然 76B 但表现最差**：缺乏区域级训练数据，且一致偏向预测"无歧义"（Positive Rate 最低仅 27.9%）
4. **定位任务中 Molmo+SAM 的 mAP 最高**：因为 Molmo 倾向指向多个区域，但 Union IoU 反而低，因为 SAM 的点提示分割不够完整
5. **部件分割极为困难**：PACO 数据中模型定位物体部件的能力远差于完整物体

## 亮点与洞察

1. **问题本身作为歧义源的首次系统研究**：将 VQA 歧义研究从"答案歧义"扩展到"问题焦点歧义"，这是一个重要的概念区分
2. **实用价值高**：直接关联到视障用户的实际使用场景，歧义感知的 VQA 系统可以交互式引导用户消歧
3. **数据集设计精巧**：四个来源覆盖不同场景，nearly balanced 的歧义/非歧义分布，支持 zero/few-shot 评估范式
4. **对 CoT 的深入分析**：不仅测了 CoT 有效性，还发现训练数据和提示策略的互补重要性

## 局限与展望

1. **数据规模有限**：5,500 样本可能不足以训练领域特定模型，尤其是训练集仅 70 个样本
2. **仅限英文问题**：跨语言的焦点歧义可能有不同特性
3. **未提出解决方案模型**：仅做基准测试，未设计专门的歧义感知 VQA 模型
4. **两步管线的级联误差**：GPT-4o+GLaMM 和 Molmo+SAM 的两阶段方法存在误差传播问题
5. **视频和多模态扩展**：焦点歧义在视频 QA、多模态对话中同样存在，值得进一步探索

## 相关工作与启发

- **VQA Therapy (ICCV 2023)**：关注答案定位，本文在其基础上增加了问题定位的新维度
- **Molmo/PixMo**：证明区域级训练对歧义识别有帮助，未来可针对性训练
- **GLaMM**：当前最好的语言引导分割模型，但仍只生成单个 mask，无法处理多焦点场景
- **SAM**：点提示的分割能力强但不完整，需要更好的提示生成策略
- 启发：构建以"用户意图理解"而非"单一答案预测"为目标的 VQA 系统

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VQ-FocusAmbiguity: Acknowledging Focus Ambiguity in Visual Questions](vq_focusambiguity_acknowledging_focus_ambiguity_visual_questions.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](../../ACL2025/multimodal_vlm/teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[ICCV 2025\] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining](iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)

</div>

<!-- RELATED:END -->
