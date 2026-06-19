---
title: >-
  [论文解读] Multimodal OCR: Parse Anything from Documents
description: >-
  [CVPR 2025][多模态VLM][文档解析] 提出 Multimodal OCR (MOCR) 范式，将文档中的文本和图形（图表、图标、UI 等）统一解析为结构化文本表示（包括 SVG 代码），3B 模型在 olmOCR-Bench 上达到 83.9 SOTA，图形解析超越 Gemini 3 Pro。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "文档解析"
  - "OCR"
  - "SVG"
  - "图形重构"
  - "多模态预训练"
---

# Multimodal OCR: Parse Anything from Documents

**会议**: CVPR 2025  
**arXiv**: [2603.13032](https://arxiv.org/abs/2603.13032)  
**代码**: [https://github.com/rednote-hilab/dots.mocr](https://github.com/rednote-hilab/dots.mocr)  
**领域**: 多模态VLM  
**关键词**: 文档解析, OCR, SVG, 图形重构, 多模态预训练

## 一句话总结

提出 Multimodal OCR (MOCR) 范式，将文档中的文本和图形（图表、图标、UI 等）统一解析为结构化文本表示（包括 SVG 代码），3B 模型在 olmOCR-Bench 上达到 83.9 SOTA，图形解析超越 Gemini 3 Pro。

## 研究背景与动机

**领域现状**：现有文档解析主要是 text-centric——识别和组织文本内容，遇到图表、图标等图形区域就裁剪为像素图，丢弃其中的结构和语义信息。

**现有痛点**：文档中大量信息通过图形传达（图表、流程图、化学结构式等），但传统 OCR 管道将这些视为"黑箱"。这使得文档解析本质上是"有损"的，限制了从文档中提取的结构化监督信号量。

**核心矛盾**：图形不像文本有标准的文字表示，需要找到一种统一的可执行、可编辑、可验证的表示形式来捕获图形的结构信息。

**本文目标** 设计一个统一架构同时解析文档中的文本和图形，把图形也变成可复用的结构化输出。

**切入角度**：现代 VLM 已具备从图像生成可执行表示的能力（如 SVG），可以将文档图形解析为可渲染代码而非像素裁剪。

**核心 idea**：文档解析不应止步于文字识别，图形也是一等公民——用 SVG 代码统一表示，实现"parse anything"。

## 方法详解

### 整体框架

dots.mocr 是一个 3B 参数的端到端模型，包含：1.2B 从零训练的高分辨率视觉编码器（支持 ~11M 像素输入）+ 轻量多模态连接器 + Qwen2.5-1.5B 解码器。输入文档图像，输出有序结构化序列 $\mathbf{S} = [(\mathcal{B}_k, c_k, p_k)]$，其中 $\mathcal{B}_k$ 是空间区域，$c_k$ 是元素类型，$p_k$ 是内容载荷（文本→纯文本/表格标记/LaTeX；图形→SVG 代码）。

### 关键设计

1. **统一解析格式**：

    - 功能：文本和图形用同一序列格式表示，按阅读顺序生成
    - 核心思路：文本区域的 payload 是文字转录（plain text / table markup / LaTeX），图形区域的 payload 是 SVG 代码
    - 设计动机：统一格式使端到端训练成为可能，模型可以利用文本和图形之间的语义关系

2. **四阶段训练 Recipe**：

    - 功能：从通用视觉理解逐步过渡到 MOCR 专用能力
    - Stage 1: 通用视觉训练，建立稳定的 vision-language 接口
    - Stage 2: 广泛预训练（通用视觉 + 纯文本文档解析），建立文本 OCR 基础
    - Stage 3: MOCR 专用训练，减少通用数据比例，增加图形解析（image-to-SVG）
    - Stage 4: Instruction tuning，高质量监督数据精炼
    - 分辨率逐阶段提升，匹配渐增的任务难度

3. **多源数据引擎**：

    - PDF 文档：用 dots.ocr 作为 auto-labeling 引擎，分层采样（语言/领域/复杂度）
    - 网页渲染：HTML/DOM 提供对齐的结构化信号，天然包含 SVG 图标和图表
    - 原生 SVG 资源：使用 svgo 标准化 → 去重 → 域级平衡 → 复杂度采样
    - 通用视觉数据：保持广泛能力
    - SVG 数据处理关键步骤：canonicalization、viewBox 归一化、复杂度缩减

4. **OCR Arena 自动评估**：

    - 功能：用 LLM-as-Judge（Gemini 3 Flash）做成对比较，Elo 评分
    - 核心思路：传统 WER/NED 等指标对表面形式差异过于敏感，Elo 评分更能反映真实质量
    - 反位置偏差：每对比较做两次（交换顺序），结果不一致则算平局

### 损失函数 / 训练策略

统一的自回归目标：预测结构化解析序列，条件是输入图像 + 任务指令。通过 mixture reweighting 和 curriculum scheduling 控制优化稳定性。

## 实验关键数据

### 主实验（文档解析）

| 模型 | olmOCR-Bench | OmniDocBench1.5 (Elo) | XDocParse (Elo) | Avg Elo |
|------|-------------|---------------------|----------------|---------|
| Gemini 3 Pro | — | 1128.0 | 1323.7 | **1210.7** |
| **dots.mocr** | **83.9** | 1059.0 | 1210.7 | **1124.7** |
| dots.ocr | 79.1 | 1027.2 | 1190.3 | 1086.2 |
| HunyuanOCR | — | 1003.9 | 951.1 | 984.2 |

### 图形解析

| 方法 | UniSVG Score | ChartMimic | Design2Code | ChemDraw |
|------|-------------|-----------|-------------|----------|
| Gemini 3 Pro | 0.735 | 0.788 | 0.760 | 0.839 |
| OCRVerse | 0.763 | 0.799 | — | 0.881 |
| **dots.mocr-svg** | **0.902** | **0.905** | **0.834** | **0.901** |

### 关键发现

- **开源 SOTA**：dots.mocr 在 Elo 排名中仅次于 Gemini 3 Pro，在 olmOCR-Bench 上刷新 SOTA (83.9)
- **图形解析大幅超越闭源模型**：UniSVG 上超 Gemini 3 Pro +0.167，ChartMimic 超 +0.117。3B 模型在图形重构上超越大型闭源模型
- **从零训练视觉编码器比直接用预训练编码器更好**：针对文档解析任务定制特征表示
- **统一训练的互利**：文本解析和图形解析共享的表示学习相互增强

## 亮点与洞察

- **"parse anything"的范式转变非常有远见**：将以前丢弃的图形转化为可复用的结构化监督信号——这不只是更好的 OCR，而是为多模态预训练开辟新数据源。每个 PDF 里的图表都能变成 image→code 训练样本。
- **SVG 作为图形的"文字"表示**：SVG 可渲染、可编辑、层次化，是图形信息的最佳结构化载体。这个选择比 Python/HTML 更通用。
- **视觉编码器从零训练**：对于文档解析这种精细任务，文档的特征分布与 ImageNet 差异太大，从零训练反而更优。

## 局限与展望

- **不支持单 pass 同时输出文本和 SVG**：当前需要分别运行页面文本解析和区域级 SVG 解码
- **复杂真实照片无法用 SVG 表示**：方法仅适用于"可程序化描述"的图形
- **SVG 非唯一性**：同一图像可由多种 SVG 代码生成，增加训练难度

## 相关工作与启发

- **vs GOT-OCR / DeepSeek-OCR**：这些是端到端文本 OCR，不处理图形。MOCR 将覆盖范围扩展到图形，更全面。
- **vs UniSVG / StarVector**：这些专注于 image→SVG 的单一任务。MOCR 在统一框架中同时做文本 OCR 和图形解析。
- **启发**：这种"把所有东西都解析为可执行代码"的思路,与 CodePercept 有相通之处——代码/SVG 作为比自然语言更精确的表示形式。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将图形提升为一等解析目标的范式新颖，但技术组件本身较标准
- 实验充分度: ⭐⭐⭐⭐⭐ 文档解析+图形解析双重评估，Elo系统，olmOCR-Bench细分，6个SVG基准
- 写作质量: ⭐⭐⭐⭐ 系统性强，但论文较长
- 价值: ⭐⭐⭐⭐⭐ 开源3B模型 + 数据引擎 + 新评估范式，对文档AI社区贡献巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[CVPR 2026\] Describe Anything Anywhere At Any Moment](../../CVPR2026/multimodal_vlm/describe_anything_anywhere_at_any_moment.md)
- [\[AAAI 2026\] OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive](../../AAAI2026/multimodal_vlm/oida-qa_a_multimodal_benchmark_for_analyzing_the_opioid_industry_documents_archi.md)
- [\[CVPR 2026\] MCHDoc: A Comprehensive Benchmark for Reading Multi-Carrier Chinese Historical Documents](../../CVPR2026/multimodal_vlm/mchdoc_a_comprehensive_benchmark_for_reading_multi-carrier_chinese_historical_do.md)
- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](../../ICML2026/multimodal_vlm/very_efficient_listwise_multimodal_reranking_for_long_documents.md)

</div>

<!-- RELATED:END -->
