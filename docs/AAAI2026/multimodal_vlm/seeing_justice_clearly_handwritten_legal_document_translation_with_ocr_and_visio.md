---
title: >-
  [论文解读] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models
description: >-
  [AAAI 2026][多模态VLM][手写文档识别] 本文系统性对比了传统 OCR+机器翻译（OCR-MT）流水线与视觉大语言模型（vLLM）在手写马拉地语法律文档翻译为英语任务上的表现，发现两类方法均未达到法律级部署要求，OCR-MT 受级联错误影响严重，vLLM 存在严重的幻觉问题，但 vLLM 展现出统一端到端处理的发展潜力。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "手写文档识别"
  - "OCR"
  - "视觉语言模型"
  - "法律文档翻译"
  - "低资源语言"
---

# Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models

**会议**: AAAI 2026  
**arXiv**: [2512.18004](https://arxiv.org/abs/2512.18004)  
**代码**: [github](https://github.com/anviksha-lab-iitk/SJC)  
**领域**: 多模态VLM  
**关键词**: 手写文档识别, OCR, 视觉语言模型, 法律文档翻译, 低资源语言

## 一句话总结

本文系统性对比了传统 OCR+机器翻译（OCR-MT）流水线与视觉大语言模型（vLLM）在手写马拉地语法律文档翻译为英语任务上的表现，发现两类方法均未达到法律级部署要求，OCR-MT 受级联错误影响严重，vLLM 存在严重的幻觉问题，但 vLLM 展现出统一端到端处理的发展潜力。

## 研究背景与动机

### 领域现状
印度司法系统是全球最复杂的法律体系之一，基层法院和警察局仍大量依赖手写文档记录，包括首次信息报告（FIR）、案件日记、证人陈述和法庭诉讼记录等。这些文档对刑事和民事诉讼至关重要，但其手写、非结构化的特性使得归档、检索和分析极为困难。

### 现有痛点

**手写文本识别困难**：手写风格差异大，书写质量参差不齐，传统 OCR 系统（如 Tesseract、EasyOCR、PaddleOCR）在手写法律文档上表现不佳

**低资源语言挑战**：马拉地语等印度语言缺少大规模数字化语料库，OCR 和翻译模型均面临数据稀缺问题

**级联错误传播**：OCR-MT 流水线中，OCR 阶段的识别错误会直接影响下游翻译质量，导致法律语义丢失

**法律术语的特殊性**：法律文档包含专业术语、官方印章、签名和结构化表格，增加了识别和翻译的难度

### 核心矛盾
法律文档的数字化迫切需要准确、可扩展的翻译系统，但现有技术路线（无论是模块化的 OCR-MT 还是端到端的 vLLM）在处理手写低资源语言法律文档时都存在根本性的局限。

### 本文切入角度
构建一个统一的评测框架，系统对比两种范式（OCR-MT vs. vLLM）在真实法律文档场景中的表现，为后续研究提供可行的基线和方向性指导。

## 方法详解

### 整体框架
本文并非提出全新方法，而是构建了一个**系统性对比实验框架**，评测两大类方法在手写马拉地语法律文档翻译任务上的表现。

### 关键设计

1. **OCR-MT 流水线（6种组合）**:

    - **OCR 工具**: Tesseract、EasyOCR、PaddleOCR 三种 OCR 引擎
    - **翻译模型**: IndicTrans2（支持22种印度语言的 Transformer 编码器-解码器模型）和 Sarvam-1（针对10种印度语言优化的2B参数模型）
    - **工作流程**: 扫描文档图像 → OCR 文本提取 → 机器翻译 → 英语输出
    - **设计动机**: 模块化架构便于定位性能瓶颈——是 OCR 阶段还是翻译阶段导致的质量下降

2. **vLLM 端到端翻译（3个模型）**:

    - **模型选择**: Chitrarth（印度语言视觉-语言桥接模型）、Maya-8B（多语言指令微调模型）、Ovis2（34B int4量化版和16B版）
    - **核心思路**: 直接将手写文档图像输入 vLLM，通过零样本提示要求模型输出英语翻译
    - **设计动机**: 绕过 OCR 中间步骤，避免级联错误，利用 vLLM 的多模态推理能力

3. **评测协议设计**:

    - **OCR 评测**: 使用字符错误率（CER）和词错误率（WER）衡量马拉地语文本提取保真度
    - **翻译评测**: 采用人工评估，从**流畅性**（语法正确性）、**充分性**（原意保留度）、**正确性**（与金标准对齐度）三个维度打分
    - **数据集**: 约60份真实法律来源的扫描PDF马拉地语文档，由母语者翻译、法律语言专家审校

## 实验关键数据

### 主实验

| 方法 | 代表模型 | 手写文本表现 | 翻译质量 | 主要问题 |
|------|---------|-------------|---------|---------|
| OCR-MT | EasyOCR + IndicTrans2 | 印刷体尚可，手写体差 | 受OCR错误影响严重 | 级联错误传播，法律语义丢失 |
| OCR-MT | PaddleOCR + Sarvam-1 | 最差 | 混合语言输出 | 手写支持最弱 |
| OCR-MT | Tesseract + IndicTrans2 | 中等 | 不完整翻译 | 缺少手写适配 |
| vLLM | Chitrarth | 无法识别 | 完全幻觉 | 生成虚构会议内容 |
| vLLM | Maya-8B | 部分识别 | 不相关输出 | 将法律文档误判为学习指南 |
| vLLM | Ovis2-34B (int4) | 部分识别 | 部分正确但编造内容 | 能识别结构但语义错误 |
| vLLM | Ovis2-16B | 相对最好 | 部分翻译 | 不完整且部分不连贯 |

### 消融实验（OCR 模型对比）

| OCR 模型 | 印刷体性能 | 手写体性能 | 综合评价 |
|---------|-----------|-----------|---------|
| EasyOCR | 较好 | 中等（仍有困难） | 三者中最优 |
| PaddleOCR | 中等 | 差 | 数字和日期识别有误 |
| Tesseract | 中等 | 较差 | 低资源语言支持有限 |

### 关键发现

1. **OCR 阶段是 OCR-MT 流水线的主要瓶颈**: EasyOCR 在三种 OCR 工具中表现最好，但仍无法有效处理不一致的手写风格
2. **错误传播严重**: OCR 将"Gaav"（意为"村庄"）音译为"Gaon"而非翻译为"Village"，导致下游翻译完全失败
3. **vLLM 的幻觉问题**: Chitrarth 生成了关于虚构会议的描述，包含不存在的人名、日期和地点；Maya-8B 将法律文档输出为学习指南
4. **vLLM 的结构识别优势**: Ovis2 系列能部分识别文档结构（如账号、姓名、地点），但内容准确性不足
5. **法律文档的高风险性**: 在法律领域，vLLM 的幻觉问题构成严重风险——生成貌似合理但完全虚构的文本

## 亮点与洞察

1. **问题定义清晰**: 从印度司法系统的真实需求出发，选择了具有实际应用价值的任务场景
2. **全面的对比框架**: 覆盖了 OCR-MT 和 vLLM 两大范式共9种组合，评测维度丰富
3. **揭示了 vLLM 在高风险领域的根本问题**: 幻觉不仅是性能问题，更是安全性和可信性问题
4. **数据集贡献**: 构建了高质量的手写马拉地语法律文档数据集，经母语者翻译和法律专家审校
5. **未来方向指引**: 提出混合 OCR-vLLM 流水线、领域特定微调、提示工程等具体研究方向

## 局限与展望

1. **数据集规模小**: 仅约60份文档，不足以支撑大规模定量评估
2. **缺乏自动评测指标**: 翻译质量主要依赖人工评估，可重复性有限
3. **未进行微调实验**: 所有 vLLM 均在零样本设定下评测，未探索微调的潜力
4. **单一语言对**: 仅覆盖马拉地语→英语，未扩展到其他印度语言
5. **未考虑混合方案**: 文中虽提到可将 OCR 结构线索与 vLLM 语境翻译结合，但未实际实验
6. **边缘部署分析缺失**: 声称关注低资源环境部署，但未进行计算效率或模型压缩实验

## 相关工作与启发

- **VISTA-OCR / olmOCR**: 引入生成式、布局感知的 OCR 流水线，可能更适合法律文档的复杂排版
- **Nirnayak**: 印度法律领域 OCR 应用的先驱工作，但受限于 OCR 错误传播
- **TransDocAnalyser**: 专门针对 FIR 文档的框架，结合 FastRCNN+ViT 编码器和 BERT 解码器
- **PLATTER**: 支持10种印度语言的端到端手写 OCR 框架，可作为本文 OCR 模块的升级替代
- **启发**: 混合 OCR+vLLM 方案（用 OCR 做结构检测，用 vLLM 做上下文翻译）可能是最有前景的方向

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency](../../ACL2025/multimodal_vlm/improving_mllms_document_image_machine_translation_via_synchronously_self-review.md)
- [\[ACL 2026\] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction](../../ACL2026/multimodal_vlm/texocr_advancing_document_ocr_models_for_compilable_page-to-latex_reconstruction.md)
- [\[CVPR 2025\] Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly](../../CVPR2025/multimodal_vlm/unveiling_the_ignorance_of_mllms_seeing_clearly_answering_incorrectly.md)
- [\[ACL 2026\] When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models](../../ACL2026/multimodal_vlm/when_seeing_overrides_knowing_disentangling_knowledge_conflicts_in_vision-langua.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)

</div>

<!-- RELATED:END -->
