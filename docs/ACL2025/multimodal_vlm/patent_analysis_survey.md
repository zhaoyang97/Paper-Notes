---
title: >-
  [论文解读] A Survey on Patent Analysis: From NLP to Multimodal AI
description: >-
  [ACL 2025][多模态VLM][patent analysis] 系统综述了 NLP 和多模态 AI 在专利分析四大核心任务（分类、检索、质量分析、生成）中的应用，提出基于专利生命周期的分类体系，揭示了从 Word2Vec+LSTM 到 BERT/GPT 再到多模态模型的方法演进趋势及重要研究空白。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "patent analysis"
  - "NLP"
  - "多模态"
  - "PLM"
  - "LLM"
---

# A Survey on Patent Analysis: From NLP to Multimodal AI

**会议**: ACL 2025  
**arXiv**: [2404.08668](https://arxiv.org/abs/2404.08668)  
**代码**: [GitHub](https://github.com/AI4Patents-survey)  
**领域**: 多模态VLM  
**关键词**: patent analysis, NLP, multimodal AI, PLM, LLM

## 一句话总结

系统综述了 NLP 和多模态 AI 在专利分析四大核心任务（分类、检索、质量分析、生成）中的应用，提出基于专利生命周期的分类体系，揭示了从 Word2Vec+LSTM 到 BERT/GPT 再到多模态模型的方法演进趋势及重要研究空白。

## 研究背景与动机

**领域现状**：全球专利数据量呈指数级增长，USPTO 和 EPO 每年受理数十万件专利申请。专利审查涉及分类、检索、质量分析和撰写等多个环节，传统上高度依赖人工审查员的专业经验和大量时间投入。近年来，预训练语言模型（PLM）和大语言模型（LLM）在自然语言处理领域的突破性进展，为专利分析自动化带来了前所未有的机遇。

**核心痛点**：现有的专利 AI 综述（Gomez & Moens 2014、Krestel et al. 2021、Ali et al. 2024）存在三个关键缺陷：一是未覆盖 PLM/LLM 的最新应用进展；二是缺乏按任务维度和方法特性进行的系统分类；三是忽视了多模态学习（专利文本+图像）在检索和分类中的潜力。专利文本独特的法律语言结构（如权利要求的嵌套式表述）和专利图像的非自然特性（黑白线条图、标注数字）使得通用 NLP 方法不能直接迁移。

**本文切入角度**：提出一种基于专利生命周期任务的新分类体系（taxonomy），按四大核心任务（分类、检索、质量分析、生成）和三类方法（传统 NN、集成模型、PLM/LLM）双维度组织文献，为研究者构建面向特定任务的方法提供路线图。同时维护公开 GitHub 仓库，持续更新分类论文列表。

## 方法详解

### 整体框架

综述围绕专利生命周期中四大核心任务构建分层组织：**专利分类**（IPC/CPC 层级多标签分类）→ **专利检索**（文本和图像的先有技术检索）→ **专利质量分析**（引用数、专利族规模等指标预测）→ **专利生成**（摘要、权利要求等自动撰写）。每个任务下按方法演进阶段（传统 ML → 传统 NN → PLM → LLM/多模态模型）组织相关工作。

### 关键设计

1. **专利分类方法的三阶段演进**:

    - 功能：自动将专利分配到 IPC/CPC 层级分类体系中的多个标签
    - 核心思路：从早期 Word2Vec+LSTM/GRU（Grawe et al. 2017、Risch & Krestel 2018）到集成多种嵌入和深度模型的组合方法（Kamateri et al. 2022 使用 Bi-LSTM+Bi-GRU+多种分区技术），再到 BERT/SciBERT/XLNet 微调（Roudsari et al. 2022 达到 precision 0.82），performance 逐步提升。最新 Sentence-BERT+KNN 方法（Bekamiri et al. 2024）在 recall 和 F1 上表现最优。此外，Ghauri et al. (2023) 首次将 CLIP+MLP 用于专利图像分类（流程图、电路图、技术图纸等）
    - 设计动机：专利文本中包含大量技术术语和复杂结构，领域自适应预训练（如 SciBERT）能更好捕获专利领域语义

2. **专利检索的多模态融合趋势**:

    - 功能：根据查询（文本或图像）检索相关专利文档和图像，支持新颖性评估和侵权分析
    - 核心思路：文本检索从 SVM+词嵌入（Setchi et al. 2021）演进到 BERT（Kang et al. 2020）和 Sentence-BERT+TransE 知识图谱嵌入（Siddharth et al. 2022）。图像检索从 CNN/ResNet50（Kucer et al. 2022）发展到自监督深度度量学习（Higuchi et al. 2023 使用 InfoNCE+ArcFace）。最前沿的 Lo et al. (2024) 将 BLIP-2 和 GPT-4V 融合用于专利文本+图像联合检索，采用分布感知对比损失解决长尾类别问题
    - 设计动机：专利检索天然需要跨模态理解——设计专利以图像为主而实用专利以文本为主，多模态融合能全面覆盖

3. **专利生成中 LLM 的快速渗透**:

    - 功能：自动撰写专利摘要、独立权利要求、从属权利要求和说明书
    - 核心思路：从 GPT-2 微调生成权利要求（Lee & Hsiang 2020a）到 Patentformer 利用 T5/GPT-J 从权利要求+图纸生成说明书（Wang et al. 2024a），再到基于 RLHF 的 PatentGPT-J（Lee 2024）和多 Agent 框架（Wang et al. 2024b 使用 Qwen2/LLaMA3/GPT-4o）。一个重要发现是通用 LLM（如 Llama-3、GPT-4）在权利要求生成上优于领域特化模型（Jiang et al. 2024）
    - 设计动机：专利撰写需要精确的法律语言和技术描述，LLM 的强大文本生成能力可大幅减少专利律师的时间成本

## 实验关键数据

### 主实验

**专利分类性能对比**（USPTO 数据集）：

| 方法 | 嵌入 | 模型 | Precision | 分类级别 |
|------|------|------|-----------|---------|
| Risch & Krestel (2018) | FastText | GRU | 0.53 | 全文 |
| Lee & Hsiang (2020b) | — | BERT-base | 0.74 (acc) | Subclass |
| Roudsari et al. (2022) | Word2Vec/FastText | XLNet | **0.82** | Title/Abstract |
| Bekamiri et al. (2024) | SBERT | KNN | 最优 recall/F1 | Claim/Title/Abstract |

**专利检索方法对比**：

| 方法 | 数据类型 | 模型 | 训练方式 | 数据集 |
|------|---------|------|---------|--------|
| Setchi et al. (2021) | 文本 | SVM/NB/RF | 有监督 | — |
| Pustu-Iren et al. (2021) | 文本+图像 | RoBERTa+CLIP | 预训练 | EPO |
| Kucer et al. (2022) | 图像 | ResNet50 | 微调 | DeepPatent |
| Lo et al. (2024) | 文本+图像 | BLIP-2+GPT-4V | 预训练+有监督 | DeepPatent2 |

### 消融实验

| 分析维度 | 发现 | 影响 |
|---------|------|------|
| 文本 vs 图像检索 | 多模态 Transformer 模型 > 单模态 | 最高 mAP |
| 分类级别（Section→Subgroup） | 级别越细，准确率下降越明显 | Subclass 最高仅 0.74 |
| 专利文档组件 | Claim > Abstract > Full text | 信息密度影响 |
| 通用 vs 领域 LLM | 通用 LLM ≥ 领域特化模型 | 泛化性更强 |

### 关键发现

- PLM 的引入将专利分类 precision 从 0.53 大幅提升至 0.82，SciBERT 等领域自适应预训练模型对技术语言理解更优
- 多模态检索是明确趋势——专利图像（黑白线条图）与自然图像差异巨大，需要专门的视觉编码器
- 通用 LLM（GPT-4、Llama-3）在专利生成上竟优于领域特化模型（PatentGPT-J），反映了大规模预训练的泛化优势
- 专利质量分析缺乏统一的"金标准"评估指标——前向引用是唯一与实际价值直接关联的指标
- LLM 生成的专利文本面临幻觉风险和法律合规性挑战，RLHF 和 RAG 是有前景的改进方向

## 亮点与洞察

- 首次提出基于专利生命周期任务的系统分类体系，填补了现有综述缺乏任务导向组织的空白
- 系统梳理了从传统 NN → PLM → LLM → 多模态模型的清晰演进脉络，为后续研究提供路线图
- 指出四个重要未来方向：多模态专利基础模型、基于 RAG 的幻觉缓解、专利知识图谱构建、跨司法管辖区检索
- 维护公开 GitHub 仓库持续更新，实用价值高
- 揭示了专利域与通用 NLP 之间的显著方法差距——当前专利分类中使用的模型远落后于最先进的 LLM

## 局限与展望

- 综述侧重学术方法，对工业界（如 USPTO、EPO）实际部署的 AI 系统覆盖不足
- 各方法的性能缺乏统一基准对比——数据集子集、分类层级、评价指标差异使横向比较困难
- 对专利文本的特殊语言结构（如权利要求的嵌套法律语言）对模型设计的影响讨论不够深入
- 未讨论数据标注成本、模型可解释性等实际部署中的关键挑战
- 多模态方法的讨论相对单薄，缺乏统一的多模态基准测试

## 相关工作与启发

- **vs Gomez & Moens (2014)**: 早期 NLP+专利综述，本文覆盖了 PLM/LLM 时代的全新进展
- **vs Krestel et al. (2021)**: 聚焦信息提取，本文扩展到分类、检索、质量分析、生成四大任务
- **vs Ali et al. (2024)**: 综述 AI 方法但未覆盖最新 LLM 趋势和多模态融合方法

## 评分

- 新颖性: ⭐⭐⭐ 分类体系有创新但综述类工作天然受限
- 实验充分度: ⭐⭐⭐ 文献覆盖全面但缺乏统一实验验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图表丰富、组织逻辑性强
- 价值: ⭐⭐⭐⭐ 为专利 AI 领域提供了有价值的全景地图和未来方向指引
---
title: >-
  [论文解读] A Survey on Patent Analysis: From NLP to Multimodal AI
description: >-
  [多模态] 全面综述了 NLP 和多模态 AI 在专利分析中的应用，提出基于专利生命周期任务的新分类体系，涵盖专利分类、检索、质量分析和生成四大任务，揭示了现有方法从传统 NN 到 PLM/LLM 的演进趋势及未来方向。
tags:
  - 多模态
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](../../AAAI2026/multimodal_vlm/towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[ACL 2025\] MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation](mira_empowering_one-touch_ai_services_on_smartphones_with_mllm-based_instruction.md)
- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](../../ICCV2025/multimodal_vlm/grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[CVPR 2026\] PhyCritic: Multimodal Critic Models for Physical AI](../../CVPR2026/multimodal_vlm/phycritic_multimodal_critic_models_for_physical_ai.md)

</div>

<!-- RELATED:END -->
