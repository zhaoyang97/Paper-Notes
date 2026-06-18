---
title: >-
  [论文解读] DocoPilot: Improving Multimodal Models for Document-Level Understanding
description: >-
  [CVPR 2025][信息检索/RAG][文档理解] 本文构建了 Doc-750K——一个包含 758K 问答对和 3.1M 图像的高质量文档级多模态数据集，并基于此训练原生文档理解模型 Docopilot，在 MM-NIAH 上超越 InternVL2-8B 达 19.9 个百分点，无需 RAG 即可高效处理多页文档。
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "文档理解"
  - "长上下文"
  - "多模态数据集"
  - "原生文档模型"
  - "多页推理"
---

# DocoPilot: Improving Multimodal Models for Document-Level Understanding

**会议**: CVPR 2025  
**arXiv**: [2507.14675](https://arxiv.org/abs/2507.14675)  
**代码**: [https://github.com/OpenGVLab/Docopilot](https://github.com/OpenGVLab/Docopilot)  
**领域**: 信息检索  
**关键词**: 文档理解、长上下文、多模态数据集、原生文档模型、多页推理

## 一句话总结

本文构建了 Doc-750K——一个包含 758K 问答对和 3.1M 图像的高质量文档级多模态数据集，并基于此训练原生文档理解模型 Docopilot，在 MM-NIAH 上超越 InternVL2-8B 达 19.9 个百分点，无需 RAG 即可高效处理多页文档。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在图像级任务（如 OCR、VQA、图像描述）上取得了显著进展，但在文档级理解——即跨多页提取、整合关键信息——方面仍然表现不佳。现有开源 MLLM 主要在图像级数据上训练，缺乏长上下文处理能力。

**现有痛点**：检索增强生成（RAG）是当前主流的长文档处理方案，但存在三个核心问题：(1) **检索碎片化**——检索到的信息缺乏文档整体结构；(2) **多阶段误差累积**——错误检索会传播到后续回答；(3) **额外时间开销**——检索步骤增加了响应延迟，限制了实时交互能力。

**核心矛盾**：高质量的文档级多模态数据集极度稀缺（标注成本高、构建流程缺失），而没有好的训练数据，就无法训练出原生的长上下文文档理解模型，只能依赖有缺陷的 RAG。

**本文目标** (1) 如何高效构建大规模高质量的文档级多模态训练数据？ (2) 如何在不依赖 RAG 的情况下训练原生的文档级 MLLM？

**切入角度**：作者利用学术论文的结构化特性（论文有标题、摘要、实验节等明确结构），设计了一套自动化数据构建流水线，从 Sci-Hub、arXiv、OpenReview 等来源提取真实的问答对，避免了人工标注的高成本。

**核心 idea**：通过构建大规模高质量文档级数据集 Doc-750K，结合工程优化（多模态数据打包 + Ring Attention + Liger Kernel）训练原生文档 MLLM，在文档理解任务上同时超越 RAG 方法的准确性和效率。

## 方法详解

### 整体框架

整个系统包含两大部分：(1) 数据引擎——从原始文档到训练数据的自动化流水线；(2) 模型训练——基于 ViT-MLP-LLM 架构，通过工程优化实现长上下文文档的高效训练和推理。模型输入为文档内容（交错文本-图像或多图格式）+ 问题，输出为答案。

### 关键设计

1. **数据引擎与 Doc-750K 数据集**:

    - 功能：自动化构建大规模文档级问答训练数据
    - 核心思路：数据引擎分三步工作。**第一步**，从 Sci-Hub、arXiv、OpenReview 收集原始文档（PDF/HTML）。**第二步**，文档内容提取——将每篇文档处理为两种格式：交错文本-图像格式（使用 MinerU 工具提取，如 `<text>\n<image>\n<text>`）和多图格式（每页渲染为一张图像）。**第三步**，问答对构建——对 OpenReview 论文直接提取真实的审稿问答；对结构化论文设计 5 种代理任务（摘要撰写、标题生成、表格/图片描述、实验节撰写、翻译）；对其他文档使用 GPT-4o 生成 QA 对（仅占 4.8%）。最终数据集包含 758K 问题、3.1M 图像、251K 对话，其中 31.6% 为真实问答对
    - 设计动机：利用学术论文天然的层级结构，无需人工标注就能构建高质量、多样化的文档级 QA 数据。真实 QA（如审稿问答）确保了数据质量，代理任务（如根据正文写摘要）天然要求模型理解全文跨页信息

2. **训练效率优化三件套**:

    - 功能：解决长文档训练中的 GPU 显存瓶颈和训练效率问题
    - 核心思路：**(a) 多模态数据打包**——用优先队列将多个短样本拼接为长序列，设置图像数阈值 $T_{img}$ 和 token 数阈值 $T_{tok}$，最大化 GPU 利用率，避免 padding 浪费。**(b) Ring Attention**——将长序列分块分配到多个 GPU 上，通过重叠通信与注意力计算来突破单设备显存限制。**(c) Liger Kernel**——通过内核融合、原地操作、输入分块等技术进一步降低显存消耗并提升训练吞吐量
    - 设计动机：文档级输入的 token 数量远超常规图像级输入（平均 11,245 文本 token + 6,178 图像 token），不做优化则无法在现有硬件上训练

3. **SFT 数据配方**:

    - 功能：防止模型在文档领域过拟合，保持通用能力
    - 核心思路：将 Doc-750K 与其他开源数据集混合，覆盖四类场景：多页文档 QA（核心，含 MP-DocVQA、DUDE 等）、多图通用 QA（MMDU-45K）、单页文档 QA（DocVQA、ChartQA 等）、纯文本 QA（LongAlpaca、LongCite 等）
    - 设计动机：仅用 Doc-750K 训练会导致模型过度特化于学术论文场景，混合多源数据可提升模型在不同文档类型上的鲁棒性

### 损失函数 / 训练策略

标准的 next-token prediction 和对话式 SFT 训练。使用 ViT-MLP-LLM 架构（基于 InternVL），Visual Transformer 编码图像，两层 MLP 做投射对齐，预训练 LLM 生成答案。

## 实验关键数据

### 主实验

| 模型 | MM-NIAH Overall | MP-Doc ANSL↑ | MMLong-Doc Acc↑ | DocGenome SP Acc↑ |
|------|----------------|-------------|-----------------|-------------------|
| InternVL2-8B | 41.9 | 79.5 | 18.6 | 60.3 |
| InternVL2-26B | 48.4 | - | - | - |
| Docopilot-2B | **49.2** | 76.2 | 21.8 | 45.1 |
| Docopilot-8B | **61.8** | 84.5 | 31.4 | 66.2 |
| GPT-4o | - | - | 42.8 | 71.8 |

Docopilot-8B 在 MM-NIAH 上超越 InternVL2-8B +19.9 点，超越 InternVL2-26B 且推理延迟仅为后者的 31%。Docopilot-2B 以 <10% 的参数量就达到了与 InternVL2-26B 可比的性能。

### 消融实验

| 配置 | MM-NIAH Overall | 说明 |
|------|----------------|------|
| InternVL2-8B baseline | 41.9 | 无文档级训练 |
| + Doc-750K only | ~55 | 文档数据显著提升 |
| + SFT 混合数据 | 61.8 | 混合训练进一步提升 |
| InternVL2-8B + RAG | 51.0 | RAG 提升有限且增加延迟 |

### 关键发现

- **原生长上下文训练远优于 RAG**：Docopilot 不仅准确率更高，推理延迟也大幅降低（无需检索步骤），尤其在多轮交互中优势明显
- **数据质量 > 数据量**：Doc-750K 中 31.6% 的真实 QA 对是性能提升的关键，纯合成数据效果有限
- **小模型也能做好文档理解**：2B 模型通过高质量文档数据训练就能达到 26B 模型的水平，说明训练数据对文档理解能力的重要性大于模型规模
- **多模态数据打包显著提升训练效率**，使得在消费级 GPU（RTX 4090 x2 级别）上就能完成文档级 MLLM 的训练

## 亮点与洞察

- **数据引擎设计巧妙**：利用学术论文的天然结构（标题→摘要→实验→图表说明）设计代理任务，让模型在回答这些任务时自然学会跨页信息整合。这种"结构即监督"的思路可迁移到法律文档、医疗报告等其他结构化文档领域
- **工程优化三件套的组合**使得长文档训练从不可行变为可行，且这套方案完全可复用于其他需要长上下文训练的多模态任务
- **OpenReview 审稿数据的利用**：直接将真实的审稿问答作为训练数据，既保证了问题的深度和多样性，又是零标注成本的高质量数据来源

## 局限与展望

- **数据偏向学术论文**：Doc-750K 主要来自 Sci-Hub/arXiv/OpenReview，对商业文档（合同、财报）、法律文件、多语言文档的覆盖不足
- **基准测试局限**：MM-NIAH 等基准主要测试"大海捞针"式的信息检索能力，对复杂推理（如跨页因果推理、多步计算）的评估尚不充分
- **GPT-4o 生成的 QA 可能存在幻觉**：虽然仅占 4.8%，但这部分数据的质量没有人工验证
- **模型架构无创新**：直接沿用 ViT-MLP-LLM 架构，主要贡献在数据和训练策略层面

## 相关工作与启发

- **vs M3DocRAG**：M3DocRAG 是检索增强方案，在 MP-Doc 上 ANSL=84.4 略高于 Docopilot 的 76.2（2B 版本），但 RAG 方法增加了推理延迟，且在需要全局理解的 MM-NIAH 上表现更差
- **vs Docmatix**：Docmatix 有 950 万 QA 对但主要是图像级别，Doc-750K 虽然规模更小但专注于文档级（平均 token 数 >11K），且包含真实 QA 对
- **vs mPLUG-DocOwl2**：DocOwl2 在 MM-NIAH 上仅 6.6 分，说明纯架构创新在缺乏文档级训练数据时效果有限，凸显了 Doc-750K 的价值

## 评分

- 新颖性: ⭐⭐⭐ 方法本身无架构创新，核心贡献在数据集和工程优化
- 实验充分度: ⭐⭐⭐⭐ 在多个文档级基准上全面评测，并与 RAG 方法做了公平对比
- 写作质量: ⭐⭐⭐⭐ 数据引擎描述清晰，流水线可复现
- 价值: ⭐⭐⭐⭐⭐ 首个大规模高质量文档级多模态数据集，填补了行业空白，对社区价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/information_retrieval/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[ACL 2025\] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](../../ACL2025/information_retrieval/towards_storage-efficient_visual_document_retrieval_an_empirical_study_on_reduci.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](../../ICLR2026/information_retrieval/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)
- [\[CVPR 2025\] GENIUS: A Generative Framework for Universal Multimodal Search](genius_a_generative_framework_for_universal_multimodal_search.md)

</div>

<!-- RELATED:END -->
