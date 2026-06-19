---
title: >-
  [论文解读] Relation-Rich Visual Document Generator for Visual Information Extraction
description: >-
  [CVPR 2025][多模态VLM][文档理解] 提出 RIDGE，一个关系丰富的视觉文档生成器，通过 LLM 生成层次化结构文本内容 + 自监督学习生成内容驱动的布局，合成带有实体类别和链接标注的文档图像，显著提升 VIE 模型在多个基准上的性能。 视觉信息提取（VIE）是文档理解的核心任务，需要从文档图像中识别实体类别…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "文档理解"
  - "合成数据生成"
  - "视觉信息提取"
  - "布局生成"
  - "层次结构学习"
---

# Relation-Rich Visual Document Generator for Visual Information Extraction

**会议**: CVPR 2025  
**arXiv**: [2504.10659](https://arxiv.org/abs/2504.10659)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 文档理解, 合成数据生成, 视觉信息提取, 布局生成, 层次结构学习

## 一句话总结

提出 RIDGE，一个关系丰富的视觉文档生成器，通过 LLM 生成层次化结构文本内容 + 自监督学习生成内容驱动的布局，合成带有实体类别和链接标注的文档图像，显著提升 VIE 模型在多个基准上的性能。

## 研究背景与动机

视觉信息提取（VIE）是文档理解的核心任务，需要从文档图像中识别实体类别（如 header/key/value）和实体链接关系。然而该任务面临严重的数据稀缺问题：

1. **标注成本极高**：开放集数据需要标注 $O(n^2)$ 的实体链接关系，人工成本巨大。FUNSD 仅有 199 张文档图像，XFUND 每种语言也仅 199 张
2. **隐私限制**：表单类文档常含个人信息，难以公开获取
3. **现有合成方法局限**：DocSynth 依赖手工设计布局且分辨率太低，SynthDoG 的布局基于规则随机放置网格导致不自然，DocILE 依赖 100 个人工标注模板，多样性有限
4. **布局生成方法的内容断裂**：现有布局生成方法只关注拓扑结构（bounding box + 类别），忽略文本内容与布局的关联，无法支持 VIE 训练

## 方法详解

### 整体框架

RIDGE 采用两阶段生成流水线：**阶段一**使用 LLM 生成带层次结构注释的文档文本内容（HST 格式），**阶段二**通过内容驱动的布局生成模型（CLGM）将文本实体放置到合理的 bounding box 位置，最终渲染为文档图像。此外还引入层次结构学习（HSL）训练范式来增强文档理解模型的能力。

### 关键设计

1. **层次化结构文本（HST）生成**:
    - 功能：利用 LLM 生成带有丰富实体类别和链接关系注释的文档内容
    - 核心思路：设计一种结构化文本格式 HST，整个文档用 `<content>` 标签包裹，段落用 `<h1>/<h2>` 等层级标签组织，键值对用冒号 `:` 表示直接关联，不同数量的连字符 `-` 表示层级嵌套关系。这种格式可以自动解析出实体文本、类别（header/key/value/other）和链接关系
    - 设计动机：传统合成方法要么不生成内容，要么内容无关系标注。HST 让 LLM 在生成内容的同时隐式编码了关系信息，无需人工标注即可得到完整的实体标注。通过 few-shot exemplar 引导 LLM 理解格式

2. **内容驱动布局生成模型（CLGM）**:
    - 功能：根据文本内容自动生成多样且合理的文档布局
    - 核心思路：将文档布局序列化为 JSON 格式（相比 HTML/CSS 更紧凑），使用掩码策略将 bounding box 坐标替换为 `<FILL_i>` 特殊标记，训练 LLM 预测这些坐标。自监督训练仅需 OCR 结果（文本 + bbox），无需类别/链接标注。训练时随机打乱实体顺序，迫使模型从文本内容推断布局关系；推理时保持 HST 的阅读顺序。布局生成公式为 $S(M) = f_\theta(S(D_{\backslash M}))$
    - 设计动机：传统布局生成用类别驱动（category-driven），但表单文档的类别标注很少。CLGM 用内容驱动（content-driven）+自监督，只需要容易获取的 OCR 结果就能训练。随机打乱训练顺序让模型必须理解文本语义才能推断合理位置，从而自动学到内容-布局关系

3. **层次结构学习（HSL）训练范式**:
    - 功能：利用生成的 HST 增强下游模型对文档层次关系的理解能力
    - 核心思路：设计三个训练任务——(a) HSP：解析整个文档为 HST 格式；(b) HSP with Localization：给定区域 bbox 解析局部 HST；(c) VIE with HSP：先输出相关段落的 HST，再回答信息提取问题（类似 Chain-of-Thought）
    - 设计动机：现有方法（如 DocOwl-1.5 用换行符模拟布局、LayoutLLM 用 box+text 格式）只学了空间位置，未捕捉层次结构关系。HSL 通过让模型"先理解结构再回答"来增强结构理解，同时 CoT 风格提升了可解释性

### 损失函数 / 训练策略

- CLGM 使用标准自回归交叉熵损失：$\mathcal{L} = -\sum_{k=1}^{K} \log P(S(M)^k | S(M)^{<k}, S(D_{\backslash M}), \theta)$
- 骨干网络为 LLaMA-3.1-8B，使用 LoRA 微调，最大序列长度 8000
- 训练数据：约 100K 文档图像的 OCR 标注（来自 RVL-CDIP 的 form/specification/resume/memo 类 + FUNSD/XFUND/HUST-CELL）
- 合成文档包含约 3K 英文 + 3K 中文文档，产生 444K 指令样本

## 实验关键数据

### 主实验

| 数据集 | 指标 | Qwen2-VL-7B | +RIDGE | 提升 |
|--------|------|-------------|--------|------|
| FUNSD (开放集) | F1% | 59.89 | 66.48 | +6.59 |
| XFUND-ZH (开放集) | F1% | 62.08 | 69.84 | +7.76 |
| CORD (封闭集) | F1% | 82.71 | 84.47 | +1.76 |
| CORD– | ANLS% | 80.40 | 85.53 | +5.13 |
| EPHOIE | Acc% | 76.91 | 77.89 | +0.98 |
| POIE | ANLS% | 96.01 | 96.71 | +0.70 |

### 消融实验

| 配置 | FUNSD F1% | XFUND-ZH F1% | CORD F1% | 说明 |
|------|-----------|-------------|----------|------|
| Qwen2-VL-7B 基线 | 59.89 | 62.08 | 82.71 | 无额外训练 |
| +VIE 数据 | 64.87 | 67.32 | 83.77 | 仅用 RIDGE 生成的 VIE 数据 |
| +VIE + HSL | 66.48 | 69.84 | 84.47 | 加上层次结构学习 |

领域特定生成消融：

| 配置 | SROIE– ANLS% | EPHOIE Acc% | EPHOIE ANLS% |
|------|-------------|------------|-------------|
| +RIDGE | 97.74 | 77.89 | 87.79 |
| +RIDGE+RIDGE-DS | 98.05 | 80.91 | 89.82 |

### 关键发现

- 开放集 VIE 提升显著大于封闭集（FUNSD +6.59% vs SROIE +0.24%），因为 RIDGE 主要模拟开放集场景
- RIDGE 仅用合成数据即可超越 DocOwl-1.5 等专门为 VDU 训练的模型
- 在 LayoutLMv3 的零样本场景下，RIDGE 预训练实现 62.77%（FUNSD）和 69.75%（XFUND-ZH）的 SER 性能，证明语义类别标注的可靠性
- 领域特定文档生成（receipt/exam cover page）可在封闭集上带来额外~4% 的提升

## 亮点与洞察

- **两阶段解耦设计巧妙**：内容生成和布局生成分离，且布局生成只需 OCR 结果而非人工标注，极大降低了数据获取门槛
- **自监督训练的随机打乱策略**：训练时随机排列实体顺序，迫使模型从文本内容本身推断布局，这是让模型自动学习内容-布局关系的关键
- **可解释性副产品**：VIE with HSP 训练让模型在回答前先输出层次结构，既是训练策略也是可解释性机制
- HST 格式设计简洁而有效，巧妙利用了现有 HTML 标签和缩进表示层次

## 局限与展望

- 目前主要在表单类文档上训练，对差异极大的文档类型（如学术论文、发票等）需额外训练
- 合成数据规模（6K 文档）相对较小，扩大规模可能带来更大提升
- 未探索端到端的内容-布局联合生成方案
- 封闭集 VIE 的提升有限，领域特定生成需要额外提示工程

## 相关工作与启发

- 延续了用 LLM 做布局生成的路线（LayoutNUWA），但创新地加入了文本内容驱动
- 与 SynthDoG、DocILE 等合成方法相比，RIDGE 是首个能自动生成带关系标注的合成文档
- Chain-of-Thought 风格的 VIE with HSP 可推广到其他需要结构化推理的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 内容驱动+自监督的布局生成是新颖的组合
- 实验充分度: ⭐⭐⭐⭐ 覆盖7个VIE基准、开放集/封闭集、多种下游模型、消融完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 解决了VIE领域数据稀缺的实际痛点，合成数据方案可直接使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](../../AAAI2026/multimodal_vlm/exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[ICML 2026\] Text-Conditional JEPA for Learning Semantically Rich Visual Representations](../../ICML2026/multimodal_vlm/text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)

</div>

<!-- RELATED:END -->
