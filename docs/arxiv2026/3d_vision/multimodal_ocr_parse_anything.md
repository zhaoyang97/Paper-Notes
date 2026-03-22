# Multimodal OCR: Parse Anything from Documents

**会议**: arXiv 2026  
**arXiv**: [2603.13032](https://arxiv.org/abs/2603.13032)  
**作者**: Handong Zheng, Yumeng Li, Kaile Zhang, Liang Xin, Guangwei Zhao et al.
**代码**: [https://github.com/rednote-hilab/dots](https://github.com/rednote-hilab/dots)  
**领域**: 3D视觉  
**关键词**: multimodal, ocr, parse, anything, documents  

## 一句话总结
我们提出了多模态 OCR (MOCR)，这是一种文档解析范例，可将文本和图形联合解析为统一的文本表示。

## 背景与动机
We present Multimodal OCR (MOCR), a document parsing paradigm that jointly parses text and graphics into unified textual representations.. Unlike conventional OCR systems that focus on text recognition and leave graphical regions as cropped pixels, our method, termed dots.mocr, treats visual elements such as charts, diagrams, tables, and icons as first-class parsing targets, enabling systems to parse documents while preserving semantic relationships across elements.

## 核心问题
我们提出了多模态 OCR (MOCR)，这是一种将文本和图形联合解析为统一文本表示的文档解析范式。与专注于文本识别并将图形区域保留为裁剪像素的传统 OCR 系统不同，我们的方法称为dots.mocr，将图表、图表、表格和图标等视觉元素视为一流的解析目标，使系统能够解析文档，同时保留元素之间的语义关系。

## 方法详解

### 整体框架
- We present Multimodal OCR (MOCR), a document parsing paradigm that jointly parses text and graphics into unified textual representations.
- Unlike conventional OCR systems that focus on text recognition and leave graphical regions as cropped pixels, our method, termed dots.mocr, treats visual elements such as charts, diagrams, tables, and icons as first-class parsing targets, enabling systems to parse documents while preserving semantic relationships across elements.

### 关键设计
1. **关键组件1**: Unlike conventional OCR systems that focus on text recognition and leave graphical regions as cropped pixels, our method, termed dots.mocr, treats visual elements such as charts, diagrams, tables, and icons as first-class parsing targets, enabling systems to parse documents while preserving semantic relationships across elements.
2. **关键组件2**: It offers several advantages: (1) it reconstructs both text and graphics as structured outputs, enabling more faithful document reconstruction; (2) it supports end-to-end training over heterogeneous document elements, allowing models to exploit semantic relations between textual and visual components; and (3) it converts previously discarded graphics into reusable code-level supervision, unlocking multimodal supervision embedded in existing documents.
3. **关键组件3**: On document parsing benchmarks, it ranks second only to Gemini 3 Pro on our OCR Arena Elo leaderboard, surpasses existing open-source document parsing systems, and sets a new state of the art of 83.9 on olmOCR Bench.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 在文档解析基准测试中，它在我们的 OCR Arena Elo 排行榜上排名第二，仅次于 Gemini 3 Pro，超越了现有的开源文档解析系统，并在 olmOCR Bench 上创下了 83.9 的新水平。
- 在结构化图形解析方面，dots.mocr 在图像到 SVG 基准测试中实现了比 Gemini 3 Pro 更高的重建质量，在图表、UI 布局、科学图表和化学图表上展示了强大的性能。

## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
