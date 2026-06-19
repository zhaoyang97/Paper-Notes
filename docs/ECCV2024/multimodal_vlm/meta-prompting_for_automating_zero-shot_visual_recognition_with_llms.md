---
title: >-
  [论文解读] Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs
description: >-
  [ECCV2024][多模态VLM][zero-shot recognition] 提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段 meta-prompting 策略自动化生成多样化的类别特定 VLM prompt，无需人工设计 LLM 查询即可显著提升 CLIP 等模型的 zero-shot 识别性能。
tags:
  - "ECCV2024"
  - "多模态VLM"
  - "zero-shot recognition"
  - "提示学习"
  - "VLM"
  - "LLM"
---

# Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs

**会议**: ECCV2024  
**arXiv**: [2403.11755](https://arxiv.org/abs/2403.11755)  
**代码**: [jmiemirza/Meta-Prompting](https://github.com/jmiemirza/Meta-Prompting)  
**领域**: LLM/NLP  
**关键词**: zero-shot recognition, prompt ensembling, meta-prompting, VLM, LLM

## 一句话总结

提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段 meta-prompting 策略自动化生成多样化的类别特定 VLM prompt，无需人工设计 LLM 查询即可显著提升 CLIP 等模型的 zero-shot 识别性能。

## 背景与动机

CLIP 等双编码器 VLM 在 zero-shot 分类中表现出色，其核心是将图像与文本映射到共享嵌入空间并计算余弦相似度。已有研究表明，使用多个类别特定的 prompt 进行集成（prompt ensembling）可以显著提升分类精度。

现有方法（如 CUPL、DCLIP、Waffle）虽然利用 LLM 生成类别描述来扩展 prompt 多样性，但仍存在关键问题：**需要人工为每个下游数据集手工设计 LLM 查询模板**。这不仅耗费人力，还可能引入设计者的主观偏差，限制生成 prompt 的多样性和覆盖范围。

作者提出核心问题：手工设计 LLM 查询是否会限制最终 VLM prompt 的质量？答案是肯定的——最小化人工干预后，zero-shot 识别精度显著提升。

## 核心问题

如何在**完全自动化**的前提下，从 LLM 中提取丰富的视觉世界知识，生成多样化、任务特定、类别特定的 VLM prompt，从而提升 zero-shot 视觉识别性能？

## 方法详解

### 整体框架

MPVR 采用**两阶段 meta-prompting** 策略，逐步从 LLM 中提取视觉知识：

**第一阶段：Meta-Prompting 生成任务特定的 LLM 查询模板**

Meta prompt 由三部分组成：

1. **System Prompt**：通用指令，描述任务和期望输出格式，在所有实验中保持不变
2. **In-context Example**：一个示例下游任务及其对应的 LLM 查询模板（所有实验中固定使用 DTD 数据集作为示例，当目标数据集为 DTD 时切换为 EuroSAT）
3. **Downstream Task Specification**：目标任务的简短描述和元数据，是唯一随任务变化的部分，可从公开 API 或数据集网页获取

将 meta prompt 输入 LLM（GPT 或 Mixtral），生成 $N=30$ 个多样化的任务特定 LLM 查询模板，这些模板包含 `<class name>` 占位符，融合了任务特定的视觉风格知识但仍是类别无关的。

**第二阶段：生成类别特定的 VLM Prompt**

将第一阶段生成的查询模板中的 `<class name>` 替换为具体类别名称，再次查询 LLM，为每个查询模板生成 10 个类别特定的 VLM prompt（每个 prompt 限制 50 tokens）。最终得到大量多样化的类别描述。

### 零样本分类

对于类别 $c$，将其所有 VLM prompt 通过文本编码器 $\psi$ 得到嵌入，取均值作为类别表示 $\psi_c$。测试图像通过视觉编码器 $\phi$ 得到嵌入，计算余弦相似度进行分类：

$$l_{\hat{c}}(x) = \frac{e^{\cos(\psi_{\hat{c}}, \phi(x))/\tau}}{\sum_{c \in C} e^{\cos(\psi_c, \phi(x))/\tau}}$$

### 关键设计思想

两阶段方法的核心优势在于**多样性的级联放大**：第一阶段生成多样化的查询视角（如不同视觉风格、拍摄角度、场景描述），第二阶段在每个视角下生成具体的类别描述，最终得到覆盖面极广的 prompt 语料。

## 实验关键数据

### 主实验（ViT-B/32 CLIP，20 个数据集）

| 方法 | 平均提升（vs CLIP S-TEMP） | 最大提升 |
|------|---------------------------|---------|
| MPVR (GPT) | +5.0% 平均 | +19.8%（EuroSAT） |
| MPVR (Mixtral) | +4.5% 平均 | +18.2%（EuroSAT） |

- 在 20 个数据集中的 18 个上超越所有 baseline
- 对比 CUPL：Flowers-102 上分别提升 5.1%（GPT）和 6.3%（Mixtral）
- 对比 DCLIP：UCF-101 上提升 5.3%（GPT）和 3.3%（Mixtral）

### 跨骨干网络泛化（表 2，20 个数据集平均）

| 骨干 | CLIP S-TEMP | MPVR (GPT) | 提升 |
|------|-------------|------------|------|
| OpenAI ViT-B/16 | 61.9% | 66.7% | +4.8% |
| OpenAI ViT-L/14 | 69.2% | 73.4% | +4.2% |
| MetaCLIP ViT-L/14 | 71.0% | 74.3% | +3.3% |

### 消融实验

- **Meta prompt 各组件缺失**（EuroSAT，ViT-B/16）：缺少数据集名称降至 46.7%，缺少元数据降至 42.0%，完整 MPVR 达 55.6%
- **单阶段 vs 两阶段**：两阶段 MPVR（55.6%）优于单阶段（51.2%），也优于仅用模板（47.2%）
- **文本源集成**：GPT + Mixtral 嵌入均值集成效果最佳（67.0%，ViT-B/16）
- **对比 MMLM**：CLIP ViT-B/32（57.2%）大幅优于 LLaVA-1.6-7B（30.0%），证明双编码器在判别式识别任务上的优势

### 数据规模

利用 MPVR 从 GPT 和 Mixtral 生成了约 250 万条唯一类别描述，构成首个大规模 LLM 视觉知识语料库。

## 亮点

1. **完全自动化**：人类仅需提供任务的简短描述和类别列表，无需手工设计任何 LLM 查询
2. **两阶段策略巧妙**：先生成任务感知的查询模板，再生成类别描述，多样性级联放大
3. **开源模型也有效**：首次证明开源 LLM（Mixtral）生成的描述也能有效增强 VLM 的 zero-shot 能力，性能接近 GPT
4. **广泛泛化**：在 20 个跨域数据集、多种 VLM 骨干上均取得一致提升
5. **开源大规模语料**：释放 250 万条类别描述，可直接复用

## 局限与展望

1. **依赖文本质量**：LLM 生成的描述可能包含不准确的视觉信息，目前无质量过滤机制
2. **计算开销**：两阶段查询 LLM 且每个类别需大量 prompt，推理时文本编码开销较大
3. **Stanford Cars 困难**：在该数据集上 prompt ensembling 整体效果有限，暗示某些细粒度任务仍需视觉线索
4. **In-context Example 固定**：始终使用同一个示例数据集，未探索动态选择最佳示例的策略
5. **仅限分类任务**：方法局限于 zero-shot 分类，未扩展到检测、分割等更广泛的视觉任务

## 与相关工作的对比

| 方法 | 需要手工设计 LLM 查询 | 支持开源 LLM | 20 数据集平均（ViT-B/32） |
|------|---------------------|-------------|------------------------|
| CLIP (DS-TEMP) | 是（模板） | - | 59.7% |
| CUPL | 是（每个数据集） | 否 | 约 60% |
| DCLIP | 是（属性查询） | 否 | 约 59% |
| Waffle+Con+GPT | 是（概念+随机） | 否 | 约 61% |
| **MPVR (GPT)** | **否** | 否 | **65.0%** |
| **MPVR (Mixtral)** | **否** | **是** | **63.8%** |

核心差异在于 MPVR 将人工设计从"为每个数据集写 LLM 查询"简化为"提供数据集描述"，同时通过两阶段策略获得更丰富多样的 prompt。

## 启发与关联

- **Meta-prompting 范式**的价值不局限于视觉识别，可推广到任何需要 LLM 生成结构化输出的场景
- 结果表明 CLIP 文本编码器对语义丰富的描述响应更好，暗示 VLM 的文本理解能力被简单模板低估
- 两阶段"先粗后细"的知识提取思路可以借鉴到其他 LLM 辅助的任务中
- 开源 LLM 与闭源 LLM 效果差距不大，对实际部署有重要意义

## 评分

- 新颖性: ⭐⭐⭐⭐ — 两阶段 meta-prompting 自动化框架思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 20 个数据集、多种 VLM/LLM 骨干、大量消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ — 实用性强，可直接复用语料库和框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](../../CVPR2026/multimodal_vlm/beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](../../ICCV2025/multimodal_vlm/synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[ECCV 2024\] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](../../CVPR2025/multimodal_vlm/visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)

</div>

<!-- RELATED:END -->
