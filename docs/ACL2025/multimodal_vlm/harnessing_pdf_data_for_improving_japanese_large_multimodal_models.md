---
title: >-
  [论文解读] Harnessing PDF Data for Improving Japanese Large Multimodal Models
description: >-
  [ACL 2025][多模态VLM][日语大模型] 提出一套全自动 PDF 数据提取管道，从日语 PDF 中提取图文对并生成指令数据，通过持续微调 LLaVA1.5 框架显著提升日语多模态模型性能，在 Heron-Bench 上实现 2.1%~13.8% 的提升。 大型多模态模型（LMM）在英语上表现出色…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "日语大模型"
  - "PDF数据"
  - "多模态训练"
  - "数据管道"
  - "持续微调"
---

# Harnessing PDF Data for Improving Japanese Large Multimodal Models

**会议**: ACL 2025  
**arXiv**: [2502.14778](https://arxiv.org/abs/2502.14778)  
**代码**: [https://github.com/ku21fan/PDF-JLMM](https://github.com/ku21fan/PDF-JLMM)  
**领域**: Multimodal / VLM  
**关键词**: 日语大模型, PDF数据, 多模态训练, 数据管道, 持续微调

## 一句话总结

提出一套全自动 PDF 数据提取管道，从日语 PDF 中提取图文对并生成指令数据，通过持续微调 LLaVA1.5 框架显著提升日语多模态模型性能，在 Heron-Bench 上实现 2.1%~13.8% 的提升。

## 研究背景与动机

大型多模态模型（LMM）在英语上表现出色，但在日语等非英语语言上受限于高质量训练数据的匮乏。当前日语 LMM 面临的核心困境：

**数据来源单一**：多数开源日语 LMM 依赖翻译的英语数据集（如 LLaVA 的日语翻译版），导致模型主要学习西方文化内容，缺乏日本特有文化知识（如樱花、日式建筑等）

**PDF 数据未被利用**：与网络爬取的图文数据不同，PDF 包含大量来自书籍、报告、宣传册等的高价值但未被开发的信息。据作者所知，尚无研究利用 PDF 数据增强日语 LMM

**人工标注成本高**：大规模 PDF 的手动图文标注不现实

关键研究问题：**PDF 数据能否有效增强日语 LMM？如何自动化地从 PDF 中提取有用的训练数据？**

## 方法详解

### 整体框架

三阶段训练流程：Stage 1 预训练（558K 日语图文对）→ Stage 2 指令微调（620K 日语指令数据）→ Stage 3 持续微调 CFT（362K PDF衍生指令数据）。核心创新在 Stage 3 的数据构建管道。

### 关键设计

1. **PDF 收集与筛选**：从日本国立国会图书馆网络归档项目获取超 5,138 万个 PDF。筛选策略——仅选 5 页以下的 PDF（页数多的更像书籍，图片少）；仅提取首页（图片通常出现在首页）；用 PyMuPDF 检测含图片的 PDF。最终筛选出 20 万个 PDF 页面。**设计动机**：手动观察数百个 PDF 后发现的经验规律，用简单规则高效过滤大量无图 PDF。

2. **布局分析 + OCR 提取**：不直接从 PDF 读取数据（PyMuPDF 可能提取不可见图像或错误拆分图像），而是先将 PDF 转为 JPEG 图像，再用 Surya 工具进行布局分析和 OCR。Surya 基于预训练深度学习模型，支持 90+ 语言。但性能并不完美——偶尔将日语字符误识别为印地语；过滤掉宽/高小于 50 像素的"图像"。

3. **图文配对**：使用 Japanese-Cloob（日本广泛使用的 CLIP 类模型，月活 30 万用户）计算图像嵌入和 OCR 文本嵌入的余弦相似度，选最相似文本作为配对文本。**关键发现**：由于 OCR 质量不完美（断行、复杂汉字误识别），直接使用提取的图文对训练效果差。

4. **PDF-style text 生成**：为探索"如果图文对提取更准确会怎样"，用 GPT-4o-mini 为每张图像生成"PDF 风格文本"——不是直接描述图像，而是模拟 PDF 中围绕图像的间接说明文本。实验证明这种方式比原始提取的图文对有效得多。

5. **指令数据生成**：直接将图像输入 GPT-4o-mini 生成日语指令数据（对话格式），配对文本作为可选上下文。**核心发现**：当配对文本质量差时，仅用图像生成指令数据反而效果更好。因此最终方案中，所有 362K 指令数据都是仅基于图像生成的。

6. **NSFW 和 PII 过滤**：使用 GPT-4o-mini 检测并过滤不安全内容和个人身份信息。

### 训练策略

- 基于 LLaVA1.5 框架，将视觉编码器从 CLIP 替换为 SigLIP
- 使用 LoRA 进行参数高效微调
- 主模型 PDF-JLMM 使用 Swallow（Llama3-8B 的日语微调版）作为基座 LLM
- 训练耗时（4×A100）：Stage 1 约 11h，Stage 2 约 42h，Stage 3 约 19h

## 实验关键数据

### 主实验（与现有日语 LMM 比较）

| 模型 | JA-LLaVA-Bench(COCO) | JA-LLaVA-Bench(Wild) | Heron-Bench |
|------|---------------------|---------------------|-------------|
| GPT-4V | 90.1 | 94.1 | 79.7 |
| Qwen-VL 7B | 80.4 | 54.0 | 49.7 |
| Heron BLIP v1 7B | 89.5 | 45.1 | 45.4 |
| EvoVLM-JP-v1 7B | 69.2 | 56.4 | 45.1 |
| **PDF-JLMM 8B** | **88.2** | **65.8** | **65.8** |
| LLaVA1.5-Llama3 8B | 86.9 | 56.9 | 61.6 |
| LLaVA1.5-Phi3-medium 14B | 86.8 | 74.1 | 57.4 |

### PDF 数据量的影响（Heron-Bench）

| LLM | Stages 1&2 | +50K PDF | +100K PDF | +150K PDF | +200K PDF |
|-----|-----------|---------|----------|----------|----------|
| Swallow 8B | 54.7 | 65.7 | **65.8** | 63.8 | 64.6 |
| Llama3 8B | 54.8 | 58.7 | 61.0 | **61.8** | 61.6 |
| Phi3-mini 3.8B | 43.3 | 51.9 | 53.0 | **57.1** | 54.3 |
| Phi3-medium 14B | 54.2 | **58.8** | 56.3 | 57.4 | 58.1 |

### 原始图文对 vs 指令数据

| 训练数据 | L-COCO | L-Wild | Heron |
|---------|--------|--------|-------|
| Stages 1&2（基线） | 84.0 | 59.8 | 54.7 |
| Top 1 图文对 | 77.0 | 37.4 | 40.0 |
| PDF-style text | 81.5 | 56.5 | 65.5 |
| **指令数据（仅图像）** | **87.3** | **61.6** | **65.7** |

### PDF 数据 vs 翻译英语数据

| Stage 2 数据来源 | L-COCO | L-Wild | Heron |
|----------|--------|--------|-------|
| LLaVA-v1.5-Instruct-620K-JA | 84.0 | 59.8 | 54.7 |
| **Instruct-from-200K PDF (362K)** | **88.1** | **72.7** | **70.0** |

### 关键发现

- **PDF 衍生数据对所有模型规模（3.8B~14B）和所有 LLM 基座（日语/非日语）都有效**
- **Heron-Bench 上最大提升 13.8%**（Phi3-mini），最小提升 2.1%（Phi3-medium）
- **直接使用原始图文对反而导致性能下降**（Heron 从 54.7 降至 40.0），说明 OCR 噪声对训练是有害的
- **仅用图像生成的指令数据优于使用图像+配对文本**，因为噪声文本会干扰指令质量
- **PDF 数据优于翻译英语数据**：即使样本数更少（362K vs 620K），Heron-Bench 高出 15.3%，证明文化扎根内容的价值
- **翻译其他英语数据集（Vision-Flan、Image-Textualization）甚至导致日语性能下降**
- 日语词汇量影响显著：Phi3 仅 837 个日语词汇，14B 模型性能反不如 8B 的 Llama3
- 数据量非线性增长效果递减：超过 100K~150K PDF 后性能趋于饱和

## 亮点与洞察

- **首次证明 PDF 作为多模态训练资源的价值**：PDF 包含大量文化特定内容，这是翻译数据集无法替代的
- **"仅用图像生成指令"比"用图像+噪声文本"更好**：这是一个反直觉但重要的发现——低质量的配对文本不如没有
- **文化知识的重要性**：训练后模型能正确识别"樱花"而非仅说"白色花朵"，这是翻译数据无法提供的
- **自动化管道虽不完美但有效**：即使 OCR 和配对存在误差，只要后续指令生成策略得当，仍能产出高质量训练数据
- **PDF-style text 实验巧妙**：通过 GPT 生成理想化的配对文本作为"上限实验"，证明了如果提取技术更好，图文对方式也大有潜力

## 局限与展望

- 依赖 GPT-4o-mini 生成指令数据和 PDF-style text，引入了 API 成本和潜在的质量偏差
- 仅在 LLaVA1.5 框架上实验，未验证对更先进架构（如 Qwen-VL2、InternVL）的泛化性
- Surya 的 OCR 和布局分析在通用 PDF 上性能受限，制约了图文对直接使用的可能性
- 仅在日语上实验，虽然声称可推广到其他语言但未验证
- 评估仅使用了 Heron-Bench 系列基准，未在 JDocQA、JMMMU 等新基准上测试

## 相关工作与启发

- 与现有 PDF 数据提取工作（侧重从论文中提取图表-标题对）不同，本文将范围扩展到一般 PDF，并探索了非标题文本的配对
- LLaVA 的指令数据生成策略在此得到了有效的跨语言适配
- VILA-jp 使用交错数据提升日语 LMM，而本文使用 PDF 数据提供了互补的数据来源
- 对低资源语言 LMM 训练的启示：与其翻译英语数据，不如挖掘该语言的本土内容源

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次利用 PDF 数据增强日语 LMM，全自动管道设计有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 消融极为详尽：模型规模、LLM选择、数据量、图文对vs指令、翻译vs原生数据均有对比
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，每个实验都围绕一个明确的问题展开
- **价值**: ⭐⭐⭐⭐ — 为低资源语言 LMM 训练提供了可复制的方法论，PDF 数据源的思路有广泛启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](../../ICCV2025/multimodal_vlm/compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](../../CVPR2025/multimodal_vlm/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)

</div>

<!-- RELATED:END -->
