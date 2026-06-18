---
title: >-
  [论文解读] LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating
description: >-
  [ACL 2025][多模态VLM][长文档理解] 提出 LongDocURL 基准，覆盖理解/数值推理/跨元素定位三大任务类别共 20 个子任务，包含 2325 个高质量 QA 对、覆盖 33000+ 页文档，系统评估 26 种模型配置暴露了当前 LVLM 在长文档理解上的关键性能差距。 领域现状：LVLM（如 GPT-4…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "长文档理解"
  - "多模态基准"
  - "LVLM"
  - "跨元素定位"
  - "文档问答"
---

# LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating

**会议**: ACL 2025  
**arXiv**: [2412.18424](https://arxiv.org/abs/2412.18424)  
**代码**: [dengc2023/LongDocURL](https://github.com/dengc2023/LongDocURL)  
**领域**: 多模态VLM / 文档理解  
**关键词**: 长文档理解, 多模态基准, LVLM, 跨元素定位, 文档问答  

## 一句话总结

提出 LongDocURL 基准，覆盖理解/数值推理/跨元素定位三大任务类别共 20 个子任务，包含 2325 个高质量 QA 对、覆盖 33000+ 页文档，系统评估 26 种模型配置暴露了当前 LVLM 在长文档理解上的关键性能差距。

## 研究背景与动机

**领域现状**：LVLM（如 GPT-4o、Qwen2-VL、InternVL2）在文档理解上取得显著进展，能处理复杂文档元素、更长上下文和多样化任务。但评估基准的发展严重滞后于模型能力——单页基准（DocVQA）模型已轻松超过 95% 准确率，多页基准（MP-DocVQA, DUDE）仅覆盖不超过 20 页文档。

**现有痛点**：(1) 现有长文档基准 MMLongBench-Doc 平均仅 47.5 页、约 1k 有效样本，且仅 33.0% 问题涉及跨页信息，无法反映真实长文档场景的复杂度；(2) 所有现有基准都聚焦于理解和简单问答，完全忽略了跨元素定位任务——评估模型分析不同元素类型间关系的能力（如段落→标题映射、图表→表格关联）；(3) 元素覆盖不全，多数基准未能同时覆盖段落、表格、图表、标题等所有文档元素类型。

**核心矛盾**：模型能力快速进步（支持 128K+ 上下文），但评估基准仍停留在短文档、单一任务阶段——基准的天花板效应使得无法区分模型在真正复杂长文档场景中的优劣，阻碍了该领域的进一步发展。

**本文目标** (1) 构建真正的长文档多模态基准——平均 85.6 页、覆盖 8 类文档类型、33000+ 页总量；(2) 首创跨元素定位任务类别——评估模型分析不同元素类型间关系的能力；(3) 提供 20 个细粒度子任务支持深入分析模型在不同维度的能力差异。

**切入角度**：定义三大一级任务类别：理解（直接提取信息）、推理（数值计数/计算/比较/总结）、定位（跨元素类型关系分析），按任务类别 × 答案证据页数 × 证据元素类型细分为 20 个子任务。通过半自动流水线（GPT-4o 生成 + 自动验证 + 人工复核）高效构建高质量数据。

**核心 idea**：构建覆盖理解-推理-定位三维度、平均 85.6 页的长文档多模态基准，首创跨元素定位任务并系统暴露 LVLM 的性能差距。

## 方法详解

### 整体框架

LongDocURL 的构建包含四模块半自动流水线：(1) **Extract & Filter**——从 CommonCrawl 爬取 200K PDF，按页数（50-150 页）和语言（英文）筛选，用 GPT-4o 分类文档类型，最终保留 396 篇文档覆盖 8 种类型（研究报告、用户手册、书籍、论文等），平均 85.6 页/43622.6 tokens；(2) **QA Generation**——使用 PyMuPDF 和 Docmind 解析 PDF 提取"text-type-bbox"三元组作为文档符号表示，设计多步迭代提示查询 GPT-4o 生成 QA 对及对应证据来源；(3) **Automated Verification**——按任务相关性、格式正确性和忠实性三标准自动验证 QA 对质量；(4) **Human Verification**——人工复核负样本回收（部分可修正为有效样本）、视觉文档一致性检查（用原始 PDF 而非解析文本）和交叉验证。最终产出 2325 个高质量 QA 对。

### 关键设计

1. **三大任务类别与 20 子任务**:
    - 功能：提供多维度细粒度的 LVLM 评估框架
    - 核心思路：理解（53.5%）——直接从文档提取信息（如关键词识别、表格解析）；推理（16.6%）——数值计数、计算、比较和总结；定位（29.9%）——分析不同元素类型间的关系（如 Para-Title Locating: 给定段落摘要找到对应标题，Cross-Table Locating: 跨表格信息关联）
    - 设计动机：跨元素定位是全新的任务类别——现有基准完全忽略了模型理解文档结构中元素间关系的能力，但这是真实文档使用中的核心需求

2. **半自动质量控制流水线**:
    - 功能：在保证数据质量的前提下高效构建大规模基准
    - 核心思路：GPT-4o 生成 → 规则化自动验证（任务相关性+格式+忠实性）→ 人工复核回收负样本与交叉检查
    - 设计动机：Cross-Title Locating 任务约 75.2% 初始样本为负样本，而 Cross-Table Locating 仅 19.6%——自动验证有效筛选了低质量样本，人工复核进一步回收可修正样本

3. **Cut-off 输入范式**:
    - 功能：公平评估不同上下文长度能力的模型
    - 核心思路：对无法处理全部页面的 LVLM 截取证据周围 30 页连续页面作为输入
    - 设计动机：现实中大多数开源模型无法处理 150 页完整输入，cut-off 确保评估可行性

### 评估协议

三阶段评估：(1) 模型自由生成回答（温度=0.0）；(2) GPT-4o 提取精简答案；(3) 按 5 种答案格式（Integer/Float/String/List/None）计算标准化分数。

## 实验关键数据

### 主实验（26 种配置总分）

| 模型 | 类型 | 参数量 | 总分 |
|------|------|--------|------|
| **GPT-4o** | 闭源 | - | **64.5** |
| Claude-3.5-Sonnet | 闭源 | - | 41.8 |
| InternVL2-Pro | 开源 | - | **30.6** |
| Qwen2.5-Instruct | 开源 | 32B | 26.6 |
| Qwen2-VL | 开源 | 7B | 25.0 |
| LLaVA-OneVision-Chat | 开源 | 7B | 24.6 |

### LongDocURL vs 现有基准对比

| 基准 | 平均页数 | QA数量 | 多页问题% | 跨元素问题% | 含定位任务 |
|------|---------|--------|----------|------------|----------|
| DocVQA | 1.0 | - | 0% | - | ✗ |
| MP-DocVQA | 8.3 | - | 0% | - | ✗ |
| MMLongBench-Doc | 47.5 | 1,082 | 33.0% | 22.6% | ✗ |
| **LongDocURL** | **85.6** | **2,325** | **52.9%** | **37.1%** | **✓** |

### 数据统计

| 统计项 | 数值 |
|--------|------|
| 文档总数 | 396 篇、8 种类型 |
| 平均页数 / token数 | 85.6 页 / 43,622.6 tokens |
| QA 对总数 | 2,325 |
| 理解/推理/定位比例 | 53.5% / 16.6% / 29.9% |
| 多页问题占比 | 52.9% |
| 跨元素问题占比 | 37.1% |

### 关键发现

- **巨大的闭源-开源差距**：GPT-4o 得分 64.5 分，最佳开源模型仅 30.6 分，差距超 **33.9 分**
- **跨元素定位是最难的任务**：即便 GPT-4o 在定位子任务上也表现不佳，说明当前 LVLM 缺乏跨元素关系推理能力
- **多页问题比单页更难**：52.9% 的多页问题是驱动整体性能下降的主因
- **OCR+LLM 文本输入方式在部分场景可与图像输入竞争**：说明 OCR 管道方法在长文档场景仍有价值

## 亮点与洞察

- 首创跨元素定位（Locating）任务类别，填补了文档理解基准的空白
- 文档规模远超现有基准（平均 85.6 页 vs MMLongBench-Doc 的 47.5 页）
- 半自动构建流水线在质量与效率间取得了良好平衡
- 20 个细粒度子任务支持对模型能力的多维度深入分析

## 局限与展望

- 仅覆盖英文文档，缺少多语言评估
- 截取 30 页输入是对模型能力的妥协，无法评估真正全文理解能力
- 文档来源偏向学术/商业类型，日常文档覆盖不足
- 依赖 GPT-4o 做答案提取，可能引入评估偏差
- 文档解析质量（PyMuPDF/Docmind）会影响文本输入方式的公平性

## 相关工作与启发

- **单页基准**: DocVQA, ChartQA — 已达天花板
- **短多页基准**: MP-DocVQA, DUDE — 不超过 20 页
- **长文档基准**: MMLongBench-Doc (47.5 页), M-Longdoc (210.8 页) — 缺少定位任务
- **启发**：跨元素定位的思路可扩展到网页理解（跨 DOM 元素关系）和多模态 RAG 评估

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs](../../NeurIPS2025/multimodal_vlm/towards_comprehensive_scene_understanding_integrating_first_and_third-person_vie.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[ACL 2025\] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](mmmu_pro_robust_benchmark.md)
- [\[ACL 2025\] EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](effivlm_bench_vlm_acceleration.md)

</div>

<!-- RELATED:END -->
