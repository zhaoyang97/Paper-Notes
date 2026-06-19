---
title: >-
  [论文解读] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration
description: >-
  [ACL 2025][多模态VLM][文化理解] 本文提出了一种半自动化的文化 VLM 基准构建框架，通过人-VLM 协作生成多选 VQA 样本，并以此构建了聚焦韩国文化的 K-Viscuit 数据集（657 题），揭示了开源与闭源 VLM 在文化理解上的显著差距。 当前 VLM 主要在西方为主的数据集（COCO、VQAv2…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "文化理解"
  - "VLM基准测试"
  - "韩国文化"
  - "人机协作标注"
  - "多选视觉问答"
---

# Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration

**会议**: ACL 2025  
**arXiv**: [2406.16469](https://arxiv.org/abs/2406.16469)  
**代码**: [有](https://huggingface.co/datasets/ddehun/k-viscuit)（数据集）  
**领域**: 多模态VLM  
**关键词**: 文化理解, VLM基准测试, 韩国文化, 人机协作标注, 多选视觉问答

## 一句话总结

本文提出了一种半自动化的文化 VLM 基准构建框架，通过人-VLM 协作生成多选 VQA 样本，并以此构建了聚焦韩国文化的 K-Viscuit 数据集（657 题），揭示了开源与闭源 VLM 在文化理解上的显著差距。

## 研究背景与动机

当前 VLM 主要在西方为主的数据集（COCO、VQAv2 等）上训练，导致在非西方文化场景中表现不佳。构建文化感知的 VLM 基准面临以下挑战：

**人工标注成本高**：手动为每种文化创建 VQA 样本耗时且资源密集

**认知固着（Cognitive Fixation）**：人类标注者倾向于生成有限类型的问题，限制了数据多样性

**跨文化扩展困难**：现有文化基准（MaRVL、GD-VCR、CVQA）的构建方法难以高效迁移到新文化

核心动机：能否利用 VLM 的生成能力辅助人类标注者，**既提高效率又增加问题多样性**，同时通过人工验证保证文化准确性？

## 方法详解

### 整体框架

K-Viscuit 的构建分为四个阶段：

1. **概念选择（Concept Categorization）**：参考 Intercontinental Dictionary Series（IDS），定义 10 个核心概念类别：食物、饮料、游戏、庆典、宗教、工具、服装、文化遗产、建筑、农业
2. **图像选择（Image Selection）**：韩国母语标注者从 Wikimedia Commons 收集 CC 许可图像，每个具体物品在同一类别中最多出现两次
3. **问题生成（Question Generation）**：结合人工示范和 VLM（GPT-4-Turbo）自动生成
4. **人工验证（Human Verification）**：韩国母语者审核生成质量和文化相关性

### 关键设计

#### 1. 两类问题设计

| 类型 | 描述 | 数量 | 平均词长 |
|------|------|------|---------|
| **Type 1 - 视觉识别** | 评估基本视觉信息（如物品识别） | 237 | 10.1 |
| **Type 2 - 文化知识应用** | 需要更深的文化推理或多步推断 | 420 | 15.5 |

每张图像创建 1 个 Type 1 问题 + 1~4 个 Type 2 问题。这种分类的关键优势：
- Type 1 测试模型对文化特定视觉元素的识别能力
- Type 2 评估超越简单识别的文化理解深度

#### 2. AI 辅助标注流程

VLM（GPT-4-Turbo）接收以下输入生成问答对：
- 目标图像
- 人工标注的示范样本（每个概念类别至少 3 个）
- 详细的标注指南
- 图像特定的背景知识描述

**关键约束**：指南强调四个选项之间必须保持**高度相似性**，避免模型通过排除法答题。人工示范中也遵循此原则。

#### 3. 严格的人工验证

验证不仅检查事实正确性，更关注：
- 问题是否真正反映预期的文化细微差异
- Type 2 问题是否确实需要文化知识（而非仅靠视觉识别）
- 选项的干扰项是否足够具有迷惑性

许多 VLM 生成的事实准确但文化深度不足的样本被淘汰，确保数据集的文化共鸣。

#### 4. 英语文本但测试文化理解

所有文本用英语编写，有意将多文化理解与多语言能力分离。对于缺乏英语对应词的韩语概念，采用标准罗马化转写。

### 损失函数 / 训练策略

本文为基准构建工作，不涉及模型训练。评估使用标准多选 VQA 范式：
- 输入 = 图像 + 问题 + 四个选项（按字母顺序排列） + 输出格式指令
- 评估指标：准确率（Accuracy）

## 实验关键数据

### 主实验

**不同 VLM 在 K-Viscuit 上的表现（表2摘要）**：

| 模型 | 整体准确率 | 食物 | 游戏 | 庆典 | 服装 | 建筑 |
|------|----------|------|------|------|------|------|
| InstructBLIP-7B | 50.84 | 40.85 | 38.46 | 53.19 | 62.16 | 60.55 |
| LLaVA-1.6-13B | 57.08 | 45.07 | 36.54 | 68.09 | 70.27 | 69.72 |
| Llama-3.2-11B | 68.04 | 61.27 | 50.00 | 72.34 | 75.68 | 69.72 |
| Claude-3-opus | 70.02 | 62.68 | 59.62 | 72.34 | 78.38 | 67.89 |
| GPT-4-Turbo | 80.82 | 73.94 | 78.85 | 85.11 | 86.49 | 79.82 |
| GPT-4o | **89.50** | **88.73** | **86.54** | **95.74** | **91.89** | **91.74** |

**按题型分析（表3）**：

| 模型 | Type 1（视觉识别） | Type 2（文化知识） | 整体 |
|------|-------------------|-------------------|------|
| InstructBLIP-7B | 45.57 | 53.81 | 50.84 |
| Llama-3.2-11B | 69.20 | 67.38 | 68.04 |
| GPT-4o | 92.41 | 87.86 | 89.50 |

**有趣发现**：多数模型在 Type 2 上反而高于 Type 1，暗示视觉识别带有文化上下文的物品本身就具有挑战性。

### 消融实验

**人工评估（图5）**：
- 韩国人平均准确率：80.2（标准差 2.69）
- 非韩国人平均准确率：47.0（标准差 5.95）
- GPT-4-Turbo 与韩国人水平相当，验证了 VLM 辅助标注的有效性

**韩语输入测试（表4）**：
- 仅韩语输入通常不提升性能
- Gemini-1.5-Pro 在英语+韩语双语输入下有提升（81.58 → 83.41）

**视觉依赖性分析（图7）**：
- 将真实图像替换为高斯噪声图像后，所有模型准确率大幅下降
- Llama-3.2-11B 下降最多，Molmo-7B-D 下降最少
- 确认 K-Viscuit 确实需要视觉理解

**检索增强生成（表7，Food 类别）**：

| 模型 | 无检索 | 检索增强 | Oracle文档 |
|------|--------|---------|-----------|
| LLaVA-1.6-7B | 43.66 | 68.31 | 78.87 |
| GPT-4-Turbo | 73.94 | 78.17 | 88.73 |
| GPT-4o | 88.73 | 83.10 | 92.25 |

外部知识检索可显著提升开源模型，但闭源强模型有时反而被低质量检索结果干扰。

### 关键发现

1. **闭源 vs 开源差距巨大**：GPT-4o（89.5%）比最好的开源模型 Llama-3.2-11B（68.0%）高出 21.5 个百分点
2. **「游戏」类别最难**：所有模型在此类别上表现最差（最高 86.54%，开源最高仅 50%）
3. **视觉识别≠容易**：Type 1 问题对开源模型反而更难，因为识别文化特定物品需要在训练中见过足够的文化多样性样本
4. **生成式设置更难**：LLaVA-1.6-13B 从多选 45.07% 降到生成式 36.25%

## 亮点与洞察

1. **半自动化框架的实用性**：人-VLM 协作标注大幅降低了成本，同时 VLM 的建议增加了问题多样性，解决了人类认知固着问题
2. **选项设计精巧**：高度相似的干扰项（2628 个选项中 2129 个唯一）有效防止了模型靠排除法得分
3. **多维度分析全面**：从人工评估、语言影响、视觉依赖性、检索增强、生成式评估等多个角度深入分析
4. **框架可迁移**：虽然聚焦韩国文化，但框架设计可直接应用于其他文化

## 局限与展望

1. **图像选择仍需人工**：无法完全自动化数据集生成
2. **选项顺序敏感**：VLM 对多选项顺序敏感，虽然随机打乱缓解但不完全解决
3. **文化覆盖有限**：657 个样本仅覆盖韩国文化的子集
4. **仅评估英语能力**：将跨文化和跨语言完全分离是理想化假设
5. **可探索微调方向**：文章仅测试了检索增强，未探索在文化数据上微调开源模型的效果

## 相关工作与启发

- **MaRVL (Liu et al., 2021)**：多语言视觉推理数据集，5 种语言文化
- **CVQA (Romero et al., 2024)**：综合多语言 VQA 基准，本文在韩国子集上对比
- **CLIcK (Kim et al., 2024)**：针对韩语 LLM 的文化知识基准（纯文本）
- **IDS (Key and Comrie, 2015)**：洲际词典系列，提供跨文化概念选择框架
- 启发：人-AI 协作标注范式（先 AI 生成，后人工筛选精炼）可推广到其他需要专家知识的标注任务

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 3.5 |
| 实用性 | 4 |
| 实验完整度 | 4.5 |
| 写作清晰度 | 4 |
| 总评 | 4 |

框架设计合理，分析全面深入，特别是检索增强和生成式评估的拓展分析很有价值。作为 benchmark 论文，数据量偏小（657 题），但问题质量高、实验充分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors](multimm_cultural_metaphor.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](../../ACL2026/multimodal_vlm/vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[NeurIPS 2025\] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness](../../NeurIPS2025/multimodal_vlm/capability_a_comprehensive_visual_caption_benchmark_for_eval.md)
- [\[ACL 2025\] Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users](evaluating_multimodal_language_models_as_visual_assistants_for_visually_impaired.md)

</div>

<!-- RELATED:END -->
