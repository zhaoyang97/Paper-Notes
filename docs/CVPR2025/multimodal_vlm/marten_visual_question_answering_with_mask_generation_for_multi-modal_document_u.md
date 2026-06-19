---
title: >-
  [论文解读] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding
description: >-
  [CVPR 2025][多模态VLM][文档理解] 提出VQAMask预训练范式，在VQA文本解析基础上引入辅助的Mask生成任务（推理时丢弃），通过显式的空间对齐监督增强视觉编码器对文档图像中文字区域的感知能力，建立Marten模型在多项文档理解任务上达到8B级MLLM的SOTA。 领域现状： 多模态大语言模型（MLLM）…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "文档理解"
  - "VQAMask"
  - "空间感知对齐"
  - "Mask生成"
  - "OCR"
  - "视觉文本识别"
---

# MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding

**会议**: CVPR 2025  
**arXiv**: [2503.14140](https://arxiv.org/abs/2503.14140)  
**代码**: [GitHub](https://github.com/PriNing/Marten)  
**领域**: 多模态VLM  
**关键词**: 文档理解、VQAMask、空间感知对齐、Mask生成、OCR-free、视觉文本识别

## 一句话总结

提出VQAMask预训练范式，在VQA文本解析基础上引入辅助的Mask生成任务（推理时丢弃），通过显式的空间对齐监督增强视觉编码器对文档图像中文字区域的感知能力，建立Marten模型在多项文档理解任务上达到8B级MLLM的SOTA。

## 研究背景与动机

**领域现状**：
多模态大语言模型（MLLM）已广泛应用于文档理解任务，包括文档VQA、表格VQA、图表VQA等。现有方法分为OCR依赖和OCR-free两类，后者通过端到端方式直接从图像预测答案，是主流发展方向。

**现有痛点**：
1. 现有OCR-free MLLM的预训练任务（全文识别、文本定位等）主要在语义层面做隐式对齐
2. 缺乏对视觉文本空间位置的显式监督，模型可能过度依赖LLM的语言能力而非真正"看"图像
3. 没有空间感知监督会导致模型幻觉——输出依赖LLM的语义推测而非视觉编码器的实际观察
4. 高分辨率文档图像中文字密集、细小，仅靠语义对齐不足以准确定位和识别

**核心矛盾**：
现有VQA式预训练仅隐式地让模型学习图文对应关系，缺乏显式的空间位置监督，导致视觉编码器学到的特征不够空间敏感。

**本文目标**
设计一种兼顾语义对齐和空间对齐的预训练方法，让视觉编码器真正学会"在哪里看"。

**切入角度**：
在LLM中间层引入额外的Mask生成模块，利用视觉token和语言token的交叉注意力生成文字区域的掩码，从而显式指导空间对齐。推理时丢弃此模块，零额外开销。

**核心 idea**：
在预训练时增加一个辅助Mask生成任务，强迫视觉特征与文字空间位置对齐，推理时丢弃Mask模块，不增加任何推理成本。

## 方法详解

### 整体框架

Marten的训练分两个阶段：
1. **VQAMask对齐训练**：同时优化VQA文本解析和Mask生成两个任务，视觉编码器和MLP可训练，LLM冻结
2. **视觉语言生成训练**：丢弃Mask模块，解冻LLM，在多种文档VQA数据上进行SFT

模型架构：视觉基础模型（VFM）→ 模态连接器（MLP）→ LLM → 语言输出 + Mask生成模块（MGM，仅训练时）

### 关键设计1：VQAMask预训练范式

**功能**：同时实现语义级和空间级的视觉-语言对齐。

**核心思路**：两个子任务联合优化：
- **VQA文本解析**：6类OCR相关QA任务（全文识别、坐标识别、文本定位、公式/表格/图表转换），通过LLM输出层的自回归损失$\mathcal{L}_{vqa}$优化
- **Mask生成**：在LLM中间层提取视觉token（query）与语言token（key）的交叉注意力图，经转置卷积恢复到原图分辨率，用Dice Loss + Cross-Entropy Loss监督：
$$\mathcal{L}_{mask} = l_{\text{DICE}}(\tilde{\mathbf{M}}, \mathbf{M}) + l_{\text{CE}}(\tilde{\mathbf{M}}, \mathbf{M})$$

**设计动机**：Mask生成作为显式的空间监督信号，强迫视觉token在LLM中间层就与对应文字区域建立精确的空间对应关系。推理时丢弃MGM，零额外开销。

### 关键设计2：Mask生成模块（MGM）

**功能**：从LLM隐藏状态中提取空间注意力信息并生成文字区域掩码。

**核心思路**：
1. 取LLM第k-1层的隐藏状态$(\mathbf{V}^k, \mathbf{Q}^k, \mathbf{A}^k)$
2. 将问题和答案token拼接为$\mathbf{H}^k = [\mathbf{Q}^k, \mathbf{A}^k]$
3. 用4层Transformer做视觉token（query）→语言token（key/value）的交叉注意力
4. 将1D视觉token重组为2D空间，经转置卷积$\phi$恢复到原图分辨率
5. 输出预测Mask $\tilde{\mathbf{M}} = \phi(\mathbf{V}_{attn})$

**Pixel Shuffle改进**：采用局部窗口（4×4）的pixel shuffle而非全局操作，保持空间结构不被破坏。

**设计动机**：通过交叉注意力让视觉token根据语言token的语义指导高亮对应的文字区域，实现精确的空间对齐。

### 关键设计3：无标注Mask获取Pipeline

**功能**：大规模自动生成文字区域Mask标签，无需人工标注。

**核心思路**：三阶段pipeline：
1. **检测**：用PaddleOCR检测图像中所有文字区域，裁剪文字实例图像
2. **聚类**：对每个裁剪图像用K-means将像素分为两类，中心距离小的为前景（文字），边缘像素值校验
3. **拼接**：将所有裁剪Mask按原始坐标拼回完整图像

基于此pipeline构建**MTMask6M**数据集：600万图像-Mask对，覆盖文档（336万）、表格（60万）、图表（47.5万）、公式（20万）、场景文本（39.6万）等。

**设计动机**：文档场景中文字与背景的边界通常清晰，简单的聚类二值化即可获得高质量Mask，避免了昂贵的人工标注。

## 实验关键数据

### 主实验：与8B级MLLM对比

| 方法 | DocVQA | InfoVQA | DeepForm | KLC | WTQ | TabFact | FUNSD | SROIE |
|------|--------|---------|----------|-----|-----|---------|-------|-------|
| DocOwl-1.5 | 82.2 | 50.7 | 68.8 | 38.7 | 40.6 | 80.2 | - | - |
| Mini-Monkey | 87.4 | 60.1 | - | - | - | - | 42.9 | 70.3 |
| MM1.5 | 88.1 | 59.5 | - | - | 46.0 | 75.9 | - | - |
| **Marten** | **88.5** | **60.5** | **75.0** | **40.5** | **46.2** | **84.2** | **44.4** | **80.4** |

- 在DocVQA/InfoVQA上较之前SOTA提升0.4%
- 在DeepForm上提升6.2%，SROIE上提升10.1%
- 在KIE（关键信息抽取）任务上优势尤为明显

### 消融实验

| 设置 | DocVQA | InfoVQA | TextVQA |
|------|--------|---------|---------|
| 仅VQA | 87.1 | 58.3 | 78.6 |
| VQA + Mask (VQAMask) | **88.5** | **60.5** | **79.8** |

- VQAMask在所有任务上均带来一致提升
- Mask生成任务在不同视觉编码器（SigLIP、InternViT）和LLM（Qwen2、InternLM2）上均有效

### 关键发现

- Pixel shuffle的局部窗口策略优于全局策略
- 选择LLM中间层（而非最后层）做Mask生成效果最佳
- MTMask6M的数据多样性对性能至关重要

## 亮点与洞察

1. **辅助任务思想优雅**：Mask生成作为训练时的辅助任务，推理时完全丢弃，实现了"免费"的性能提升
2. **空间对齐 vs 语义对齐**：揭示了现有方法过度依赖LLM语义能力的问题，通过显式空间监督回归视觉感知本质
3. **自动Mask生成Pipeline**：利用文档场景文字-背景对比明显的特性，简单高效地获取标注
4. **通用性强**：VQAMask对视觉编码器和LLM的选择不敏感，可作为通用的预训练增强策略
5. **600万大规模数据集**：MTMask6M涵盖5大类文档场景，是该领域最大的Mask标注数据集

## 局限性

1. Mask获取pipeline依赖OCR工具（PaddleOCR），在极端场景可能失败
2. K-means聚类假设文字与背景色差明显，对低对比度文档可能不适用
3. 仅在8B级模型上验证，更大/更小模型的效果未知
4. Mask监督仅关注文字区域，对非文字视觉元素（图形、图标等）无直接帮助
5. 两阶段训练流程增加了训练复杂度

## 相关工作与启发

- **KOSMOS-2.5** [Peng et al.]：提出视觉文本定位任务，但仅输出bounding box而非像素级Mask
- **DocOwl 1.5** [Hu et al.]：多任务结构化文档解析，提供了DocStruct4M数据集
- **UReader** [Ye et al.]：Read Full Text任务提升文档理解
- **启发**：辅助任务在预训练中的作用值得深入探索，"训练时用、推理时丢"的策略可推广到其他领域。空间对齐是文档MLLM的关键短板。

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：VQAMask预训练范式设计新颖，"训练时增强、推理时免费"的理念实用性强。600万数据集和自动标注pipeline贡献显著。在多项文档理解任务上实现了一致的SOTA。不足之处在于方法核心（辅助Mask任务）的创新深度有限，且实验分析可更充分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/multimodal_vlm/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](docopilot_improving_multimodal_models_for_document-level_understanding.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](../../ACL2025/multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)

</div>

<!-- RELATED:END -->
