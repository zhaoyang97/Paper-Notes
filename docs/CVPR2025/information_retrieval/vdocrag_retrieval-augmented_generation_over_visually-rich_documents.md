---
title: >-
  [论文解读] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents
description: >-
  [CVPR 2025][信息检索/RAG][文档RAG] 构建首个直接以文档图像（而非解析文本）为输入的 RAG 框架，用 LVLM 作为双编码器检索器 + 两种自监督预训练任务（对比+生成）实现文档图像检索，在 ChartQA 上比文本 RAG 高 24 个点。 领域现状：Document QA 的 RAG 系统通常先用…
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "文档RAG"
  - "视觉检索"
  - "LVLM检索器"
  - "自监督预训练"
  - "OpenDocVQA"
---

# VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents

**会议**: CVPR 2025  
**arXiv**: [2504.09795](https://arxiv.org/abs/2504.09795)  
**代码**: [https://vdocrag.github.io](https://vdocrag.github.io)  
**领域**: 信息检索  
**关键词**: 文档RAG、视觉检索、LVLM检索器、自监督预训练、OpenDocVQA

## 一句话总结
构建首个直接以文档图像（而非解析文本）为输入的 RAG 框架，用 LVLM 作为双编码器检索器 + 两种自监督预训练任务（对比+生成）实现文档图像检索，在 ChartQA 上比文本 RAG 高 24 个点。

## 研究背景与动机

**领域现状**：Document QA 的 RAG 系统通常先用 OCR 解析文档为纯文本，再用文本检索器检索+生成答案。但文档中大量非文本信息（图表、表格布局、图像）在文本解析中丢失。

**现有痛点**：(1) 文本解析丢失图表的视觉结构（如柱状图的高度对比、表格的行列关系）。(2) 现有视觉检索器（如 VisRAG-Ret）基于 CLIP 类模型，对文档特有的布局和文字理解不够。(3) 缺少需要跨多个文档页面推理的标准基准。

**核心矛盾**：文档理解需要视觉和文本的联合理解，但 RAG 的检索步骤通常只用文本，丢失了关键视觉信息。

**本文目标** 设计端到端的视觉文档 RAG 框架——从检索到生成都直接操作文档图像。

**切入角度**：用 LVLM（InternVL2）作为文档图像编码器和查询编码器，通过两种自监督预训练任务（RCR 对比学习 + RCG 生成式）教会 LVLM 将文档图像压缩为可检索的 embedding。

**核心 idea**：用 LVLM 做文档图像的 dual-encoder 检索器，通过 OCR 文本的对比学习和自定义注意力掩码的生成式预训练实现自监督，构建端到端视觉文档 RAG。

## 方法详解

### 整体框架
查询 → VDocRetriever（LVLM 编码器，<EOS> token 作为 embedding）→ 检索最相关的文档图像 → VDocGenerator（同一 LVLM）接收查询+检索到的文档图像 → 生成答案。

### 关键设计

1. **VDocRetriever（LVLM 双编码器）**:

    - 功能：将文档图像和查询独立编码为可比较的 embedding
    - 核心思路：使用 InternVL2-4B 分别编码文档图像和查询文本，取 <EOS> token 的隐状态作为 embedding。支持动态高分辨率图像编码（每页多个 patch）
    - 设计动机：LVLM 比 CLIP 更懂文档——它在预训练时见过大量文本密集图像和布局结构

2. **RCR 预训练（表征压缩via检索）**:

    - 功能：用对比学习教 LVLM 将文档图像压缩为可检索的 embedding
    - 核心思路：OCR 提取文档中的文本 → 文本-图像对比学习（InfoNCE 损失）。OCR 文本作为文档图像的"自然正样本"
    - 设计动机：文档图像中的文字与图像本身形成天然的跨模态配对，无需人工标注

3. **RCG 预训练（表征压缩via生成）**:

    - 功能：通过自定义注意力掩码让 <EOS> token 学习更好的全局表征
    - 核心思路：将 OCR 文本作为解码目标，但强制生成时只能注意 <EOS> token（不能直接看图像 patch token）。这迫使 <EOS> token 必须压缩所有文档信息
    - 设计动机：消融显示 RCG 比 RCR 影响更大（去掉 RCG 检索掉 5.6 个点 vs 去掉 RCR 掉 1.4 个点），说明生成式压缩比对比学习更能提升检索质量

### 损失函数 / 训练策略
RCR：InfoNCE 对比损失。RCG：next-token 生成损失 + 自定义注意力掩码。OpenDocVQA 数据集：43K QA 跨 206K 图像，7+ 文档类型，含多跳推理问题。

## 实验关键数据

### 主实验

| 方法 | ChartQA | SlideVQA | InfoVQA | DUDE |
|------|---------|----------|---------|------|
| 文本 RAG | 28.0 | 28.6 | 40.5 | 40.1 |
| **VDocRAG** | **52.0** | **44.2** | **56.2** | **48.5** |
| 提升 | **+24.0** | **+15.6** | **+15.7** | **+8.4** |

### 消融实验

| 预训练 | SlideVQA 检索 | InfoVQA 检索 |
|--------|-------------|-------------|
| 完整 | 77.3 | 72.9 |
| 无 RCR | 75.9 (-1.4) | 71.1 (-1.8) |
| 无 RCG | 71.7 (-5.6) | 68.8 (-4.1) |
| 无两者 | 71.0 (-6.3) | 66.8 (-6.1) |

### 关键发现
- **视觉 RAG >> 文本 RAG**：ChartQA 上 +24 个绝对点，因为图表的视觉结构在文本解析中完全丢失
- **RCG 比 RCR 更重要**：生成式预训练强制 <EOS> 压缩全局信息，对检索质量影响更大
- **OpenDocVQA 填补了空白**：首个需要跨多页视觉推理的开放域文档 QA 数据集

## 亮点与洞察
- **"直接看图不读文"的文档理解范式**极具潜力——避免了 OCR 错误和格式丢失
- **RCG 的注意力掩码设计**巧妙——通过限制生成时的信息来源倒逼 <EOS> 学习更好的表征

## 局限与展望
- LVLM 编码器较重（4.2B），大规模检索场景需要离线编码 + 索引
- 仅支持静态文档，对动态网页/交互式文档未验证
- OpenDocVQA 语言偏向英文，多语言文档支持未探索

## 相关工作与启发
- **vs VisRAG**：VisRAG 用 SigLIP 做视觉检索。VDocRetriever 用 LVLM + 自监督预训练，在多个基准上接近或超越
- **vs 文本 RAG（E5-Mistral 等）**：文本 RAG 在文字密集文档上有优势，但在图表/表格文档上大幅落后于 VDocRAG

## 评分
- 新颖性: ⭐⭐⭐⭐ LVLM 做文档检索器 + RCG 预训练是新颖贡献
- 实验充分度: ⭐⭐⭐⭐ 检索+生成双评估，预训练消融清晰
- 写作质量: ⭐⭐⭐⭐ 框架描述清楚，OpenDocVQA 数据集贡献有价值
- 价值: ⭐⭐⭐⭐⭐ 对文档理解 RAG 有直接工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ACL 2025\] Re-identification of De-identified Documents with Autoregressive Infilling](../../ACL2025/information_retrieval/reidentification_deidentified.md)
- [\[CVPR 2025\] Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)
- [\[CVPR 2025\] RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings](range_retrieval_augmented_neural_fields_for_multi-resolution_geo-embeddings.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](../../ACL2025/information_retrieval/gear_generation_augmented_retrieval.md)

</div>

<!-- RELATED:END -->
