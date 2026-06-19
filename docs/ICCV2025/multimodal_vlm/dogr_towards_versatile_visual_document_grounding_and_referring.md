---
title: >-
  [论文解读] DOGR: Towards Versatile Visual Document Grounding and Referring
description: >-
  [ICCV 2025][多模态VLM][文档理解] 提出文档定位与指代数据引擎 DOGR-Engine，构建首个全面评估文档定位/指代能力的基准 DOGR-Bench（7类任务×3种文档），并开发首个兼具精准文本定位和交互式grounding/referring能力的文档理解MLLM——DOGR。 随着多模态大模型（MLLM…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "文档理解"
  - "视觉定位"
  - "多模态大模型"
  - "数据引擎"
  - "OCR"
---

# DOGR: Towards Versatile Visual Document Grounding and Referring

**会议**: ICCV 2025  
**arXiv**: [2411.17125](https://arxiv.org/abs/2411.17125)  
**代码**: [https://github.com/zyinan99/DOGR](https://github.com/zyinan99/DOGR)  
**领域**: Multimodal VLM / Document Understanding  
**关键词**: 文档理解, 视觉定位, 多模态大模型, 数据引擎, OCR

## 一句话总结

提出文档定位与指代数据引擎 DOGR-Engine，构建首个全面评估文档定位/指代能力的基准 DOGR-Bench（7类任务×3种文档），并开发首个兼具精准文本定位和交互式grounding/referring能力的文档理解MLLM——DOGR。

## 研究背景与动机

随着多模态大模型（MLLM）的进展，grounding和referring能力对细粒度文档理解至关重要，但该领域存在三大空白：

**数据匮乏**：现有文档数据质量差——OCR工具的文本识别和边界框标注不准确，且在复杂布局中难以提取语义连贯的文本块；现有指令微调数据仅覆盖基础referring任务（区域OCR、摘要、翻译），缺乏grounding任务

**评测缺失**：没有全面评估文档 grounding 和 referring 能力的基准，任务定义不清晰

**模型能力不足**：现有MLLM（GPT-4o、Gemini等）在基础文本定位和区域识别上表现很差，更无法在对话和推理中整合 grounding 和 referring 能力

这些问题导致 MLLM 在细粒度文档理解方面的潜力远未被挖掘。

## 方法详解

### 整体框架

系统由三部分组成：(1) DOGR-Engine 数据引擎，(2) DOGR-Bench 评测基准，(3) DOGR 模型。

DOGR 模型采用通用MLLM架构：InternViT-300M-448px 视觉编码器 + 投影层 + Qwen2-7B-Instruct LLM。支持动态分块处理高分辨率图像，使用 pixel shuffle 提高效率，坐标离散化为 0-999 的整数。

### 关键设计

1. **DOGR-Engine 数据引擎**：生成两类高质量数据。

    - **多粒度解析数据（2.1M）**：覆盖word、phrase、line、paragraph和full-page五个粒度，跨poster/chart/PDF三种文档类型
        - **Poster**：利用 Crello 数据集的元信息，通过 Re-rendering Strategy（修改单个文本块属性后重渲染，像素差分获取精确bbox）获得精准标注
        - **Chart**：从 ChartQA 提取信息用 Matplotlib 重绘，同样用 Re-rendering 获取bbox；1/3数据移除文本、1/3随机遮挡半数文本以防止模型过度依赖文字
        - **PDF**：结合 MinerU（有阅读顺序但不完整）和 PyMuPDF（完整但无序）的 Merge Strategy，消除重复、修复截断块、按列优先排序

    - **指令微调数据（700K）**：覆盖grounding、referring、grounding-and-referring和plain QA四类
        - Poster/Chart：将full-page解析数据输入GPT-4o生成带grounding标注的QA
        - PDF：设计 Post-annotating Strategy——先用文档图像（而非全文文本）输入GPT-4o生成QA，再用 `<ocr></ocr>` 标记原文文本，用 PyMuPDF 回溯bbox并插入 `<bbox></bbox>`
        - 规则过滤器移除格式错误和不准确标注的样本

2. **DOGR-Bench**：首个文档grounding/referring综合评测基准，3.6K样本。

    - 按输入/输出格式定义7类任务：
        - **Grounding**: $G_a$（短答案+bbox）、$G_r$（推理+grounding）、$G_o$（开放式+grounding）
        - **Referring**: $R_t$（给定bbox作答）
        - **Grounding+Referring**: $GR_a$、$GR_r$、$GR_o$
    - 评测指标：grounding用 $F1_{all}$（IoU>0.5且文本匹配），文本用 Accuracy/BLEU

3. **三阶段训练策略**：

    - **Pre-aligning**：仅训练投影层，lr=1e-3，使用LLaVA-558K
    - **Pre-training**：全模型可训练，使用DocStruct4M + 2.1M多粒度解析数据
    - **Fine-tuning**：全模型训练，使用700K指令微调数据+筛选的其他数据集，总计~2M样本

### 损失函数 / 训练策略

标准的自回归语言建模损失（next-token prediction）。三阶段中学习率逐步降低（视觉编码器2e-6，其他1e-5），序列长度从4096扩展到6144。

## 实验关键数据

### 主实验 (DOGR-Bench)

| 模型 | $G_a$ Acc | $G_a$ F1 | $G_r$ Acc | $G_r$ F1 | $R_t$ Acc | $GR_a$ Acc | $GR_a$ F1 |
|------|----------|---------|----------|---------|----------|-----------|----------|
| GPT-4o | 79.0 | 8.8 | 47.0 | 3.8 | 39.8 | 50.0 | 0.5 |
| Gemini 1.5 Pro | 77.7 | 9.4 | 62.0 | 5.8 | 37.2 | 46.0 | 5.7 |
| Gemini 2.5 Flash | 80.2 | 38.8 | 59.3 | 25.4 | 40.7 | 55.0 | 22.8 |
| Qwen2.5-VL-7B | 63.8 | 19.5 | 35.0 | 9.8 | 43.0 | 40.0 | 1.5 |
| **DOGR** | **83.2** | **73.0** | **67.7** | **52.5** | **60.3** | **82.8** | **66.9** |

DOGR 在 grounding 性能上大幅领先：$F1_{all}$ 在 $G_a$ 上达到73.0，而最强闭源模型 Gemini 2.5 Flash 仅38.8；在referring任务 $R_t$ 上也达60.3%准确率。

文本定位能力（DocLocal4K）：

| 模型 | 定位 ALL IoU@0.5 | 识别 ALL BLEU-4 |
|------|-----------------|----------------|
| GPT-4o | 5.27 | 4.38 |
| Qwen2.5-VL-7B | 21.51 | 28.98 |
| **DOGR** | **86.64** | **77.88** |

### 消融实验

| 设置 | DocVQA | InfoVQA | DeepForm | ChartQA | VisualMRC |
|------|--------|---------|----------|---------|-----------|
| Baseline (DocOwl数据) | 87.57 | 64.58 | 66.13 | 80.88 | 265.9 |
| + 多粒度解析数据 (MG) | - | - | - | - | - |
| + 指令微调数据 (IT) | 89.24 | 67.45 | 68.89 | 81.92 | 287.25 |

多粒度解析数据在所有粒度上提升定位和识别性能；指令微调数据在InfoVQA上提升2.87%，在VisualMRC上提升21.35 CIDEr。

### 关键发现

- 现有MLLM（包括GPT-4o和Gemini）在文档grounding上严重不足：F1几乎为0-10%
- 文档定位是grounding/referring的基础能力，现有模型基本不具备
- DOGR在传统文档理解任务上也保持竞争力（DocVQA 91.7, ChartQA 83.6）
- PDF文档的grounding和referring最具挑战性（布局复杂、文本密集）

## 亮点与洞察

- **数据引擎设计巧妙**：Re-rendering Strategy 通过像素差分获取精确bbox，无需OCR；Merge Strategy 结合两个PDF解析工具取长补短；Post-annotating 策略用图像替代全文输入GPT-4o大幅降低成本
- **任务定义系统化**：用输入/输出格式的组合矩阵定义7类任务，覆盖grounding×referring及其交叉
- **揭示关键问题**：首次系统性地展示了MLLM在文档grounding上的能力缺陷，数据量化了差距
- **PDF的 Merge Strategy** 在保证完整性的同时恢复阅读顺序，具有实际工程价值

## 局限与展望

- 模型依赖于高质量的预渲染和重渲染数据，对野外真实文档图像的泛化可能受限
- PDF处理中 PyMuPDF 无法定位的文本只能回退为纯文本，存在标注覆盖不完整的问题
- 仅处理了poster/chart/PDF三种文档类型，未覆盖手写文档、扫描件等
- 坐标离散化为0-999可能损失精度，尤其在高分辨率大尺寸文档中
- 可以探索更细粒度的segmentation-level定位

## 相关工作与启发

- 与 Kosmos-2.5/mPLUG-1.5/Fox 等现有文档grounding工作相比，DOGR 首次将 grounding 和 referring 整合到对话和推理流程中
- InternVL系列的动态分块策略被DOGR沿用，证明了其在文档场景的有效性
- 数据引擎思路可推广到其他需要精确空间标注的多模态任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个文档grounding+referring全栈方案（数据引擎+基准+模型），任务定义系统化
- **实验充分度**: ⭐⭐⭐⭐ 多维度评测（grounding/referring/传统VQA），覆盖闭源和开源模型
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据引擎的流程图详尽
- **价值**: ⭐⭐⭐⭐⭐ 填补文档grounding评测空白，数据引擎和基准对社区有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](../../CVPR2025/multimodal_vlm/mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)
- [\[ICCV 2025\] Visual Intention Grounding for Egocentric Assistants](visual_intention_grounding_for_egocentric_assistants.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[ICCV 2025\] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](../../CVPR2025/multimodal_vlm/videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)

</div>

<!-- RELATED:END -->
