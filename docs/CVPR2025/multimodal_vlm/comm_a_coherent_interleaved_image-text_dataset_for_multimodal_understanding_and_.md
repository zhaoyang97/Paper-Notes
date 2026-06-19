---
title: >-
  [论文解读] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation
description: >-
  [CVPR 2025][多模态VLM][交错图文数据集] 针对现有交错图文数据集（MMC4/OBELICS）叙事连贯性差、实体风格不一致的核心问题，构建 CoMM 数据集（227K 文档、2.28M 图片），通过定向采集指令型内容 + 三维质量过滤策略确保文本连贯、图像一致、图文对齐，并提出 4 个交错生成评测任务。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "交错图文数据集"
  - "多模态连贯性"
  - "Few-shot学习"
  - "图文生成"
  - "偏好数据集"
---

# CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation

**会议**: CVPR 2025  
**arXiv**: [2406.10462](https://arxiv.org/abs/2406.10462)  
**代码**: [GitHub](https://github.com/HKUST-LongGroup/CoMM)  
**领域**: 多模态VLM / 数据集  
**关键词**: 交错图文数据集, 多模态连贯性, Few-shot学习, 图文生成, 偏好数据集

## 一句话总结

针对现有交错图文数据集（MMC4/OBELICS）叙事连贯性差、实体风格不一致的核心问题，构建 CoMM 数据集（227K 文档、2.28M 图片），通过定向采集指令型内容 + 三维质量过滤策略确保文本连贯、图像一致、图文对齐，并提出 4 个交错生成评测任务。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLMs）在跨模态生成方面进展迅速，但生成连贯的交错图文序列仍然困难。Emu2、DreamLLM 等模型在交错生成中存在叙事不连贯和实体风格不一致的问题。

**现有痛点**：根源在于训练数据质量差。现有两大交错图文数据集 MMC4 和 OBELICS 存在严重问题：(1) **图片稀缺**——MMC4 中位数仅 2 张/文档，OBELICS 仅 1 张/文档；(2) **图文关联弱**——网页爬取的图片可能是广告或装饰图，与正文无关；(3) **风格不一致**——同一文档中图片来自不同来源，视觉风格割裂。

**核心矛盾**：叙事连贯性和实体一致性是交错图文内容的内在要求，但大规模自动爬取的数据天然缺乏这些特性。手动标注成本极高，需要自动化但有效的质量控制方案。

**本文切入角度**：不从随机网页爬取，而是从指令型（WikiHow 等教程）和视觉叙事网站定向采集——这类内容因统一意图（如"做菜"）和结构化呈现（Step 1/2/3），天然具有文本连贯性和图像一致性。在此基础上再用多维过滤策略进一步提升质量。

## 方法详解

### 整体框架

CoMM 数据集构建分三阶段：(1) 从指令型和视觉叙事网站定向采集原始交错图文数据；(2) 应用包含文本序列过滤、图像序列过滤、图文对齐过滤的三维质量过滤策略；(3) 通过步骤顺序打乱构造负样本，生成偏好学习数据集。最终规模为 227K 文档、2.28M 图片，平均每文档 10.1 张图片（vs MMC4 的 3.6 张和 OBELICS 的 2.9 张）。

### 关键设计

1. **三维质量过滤策略（Multi-Perspective Filter Strategy）**：
    - **文本序列过滤**：使用 LLM（Llama3）评估文档中文本步骤之间的发展性和连贯性，按评估分数删除逻辑不连贯的文本序列
    - **图像序列过滤**：设计基于 CLIP 视觉编码器的过滤指标 $\mathcal{F}(\{x_i\}) = \frac{1}{N-1}\sum_{i=2}^{N}\text{Sim}(x_i, x_{i-1}) - \frac{2}{(N-1)(N-2)}\sum_{i=2}^{N}\sum_{j=1}^{i-1}\text{Sim}(x_i, x_j)$，第一项衡量相邻图片的视觉连贯性（越高越好），第二项衡量全局图片的多样性（越低越好），两项平衡确保序列既连贯又有发展
    - **图文对齐过滤**：先用 CLIP 计算图文相似度分数删除 <0.1 的低匹配对，再用 GPT-4o 或 Llama3 结合上下文评估图文对齐（因为交错内容中文本描述的是操作进展而非图片 caption，单纯 CLIP 不够）

2. **偏好数据集构造**：通过四种方式生成负样本——打乱文本顺序（图片不动）、打乱图片顺序（文本不动）、同时独立打乱文本和图片、打乱步骤顺序（步骤内图文保持对齐但步骤间顺序随机化）。这些负样本与原始正样本配对形成偏好数据集，可用于 RLHF/DPO 等后训练阶段。

3. **四个交错生成评测任务**：
    - Task 1（图到文序列生成）：给定图片序列 $\{x_i\}_{i=1}^N$，生成对应文本序列
    - Task 2（文到图序列生成）：给定文本序列 $\{t_i\}_{i=1}^N$，生成对应图片序列
    - Task 3（交错内容续写）：给定前 k 步的图文交错内容，生成后续步骤
    - Task 4（基于问题的交错生成）：仅给定问题提示，生成完整交错图文内容

### 损失函数 / 训练策略

数据集可用于 SFT（监督微调）阶段的交错图文建模训练，也可通过偏好数据集用于 RLHF 后训练。评估指标方面，文本用 METEOR/ROUGE，图片用 FID/IS/SSIM/PSNR，交错内容通过 GPT-4o 从图像连贯性、图片质量、文档完整性、图文关联性四个维度打分。还提出 IRS（Illustration Relevance Score）指标评估图文关联性。

## 实验关键数据

### 主实验：数据集质量对比

| 指标 | MMC4 | OBELICS | **CoMM** |
|------|------|---------|----------|
| 发展性 (DLP, GPT-4o) | 4.75 | 5.97 | **7.64** |
| 完整性 (CPL, GPT-4o) | 5.12 | 5.88 | **7.07** |
| 图文对齐 (ITA, GPT-4o) | 4.66 | 3.81 | **8.91** |
| 图像序列质量 (ImgS, $\mathcal{F}$) | 0.21 | 1.00 | **4.27** |
| 中位图片数/文档 | 2 | 1 | **4** |
| 平均图片数/文档 | 3.6 | 2.9 | **10.1** |

### 消融实验：Few-shot 下游任务（CIDEr / Accuracy，选取代表性结果）

| 数据集训练来源 | COCO 0-shot | COCO 32-shot | TextVQA 32-shot | VQAv2 32-shot |
|---------------|-------------|-------------|----------------|---------------|
| Baseline | 79.5 | 99.7 | 30.6 | 56.9 |
| + MMC4 | 97.1 | 107.0 | 27.2 | 49.6 |
| + OBELICS | 83.5 | 102.2 | 29.2 | 53.2 |
| **+ CoMM** | **100.3** | **112.9** | **36.3** | **57.5** |

### 关键发现

- CoMM 在所有 4 个质量维度上均大幅领先 MMC4 和 OBELICS，尤其图文对齐 (ITA) 从 4.66/3.81 提升到 8.91
- CoMM 训练的模型在 7 个下游任务上的 few-shot 表现一致优于 MMC4 和 OBELICS，尤其在长上下文（16/32-shot）中优势更明显
- 相比 MMC4/OBELICS 的图片稀缺问题（中位数 1-2 张/文档），CoMM 平均 10.1 张/文档，且文档长度与图片数量正相关
- 定向采集策略（指令型网站）从源头保障了内容质量，三维过滤策略进一步去除噪声

## 亮点与洞察

- **"数据源选择比过滤更重要"的策略**：不从通用网页爬取再过滤，而是直接从天然高质量的指令型/视觉叙事网站采集——这一决策使后续过滤的起点就很高
- **图像序列过滤指标设计巧妙**：连贯性项 + 多样性项的差值设计，既避免了完全重复的图片序列（高分但无信息量），又保证了视觉风格一致
- **偏好数据集构造零成本**：通过步骤顺序打乱自动生成负样本，无需额外标注
- **提出 4 个评测任务和评估框架**：填补了交错生成能力评估的空白

## 局限与展望

- 数据来源以英文指令型网站为主，领域覆盖偏向生活技能/DIY，学术/新闻等知识密集型交错内容覆盖不足
- NSFW 过滤阈值为 0.1（较严格），可能过度清除一些医学/教育相关内容
- 未评估偏好数据集在 RLHF/DPO 后训练中的实际效果（仅构造未训练）
- 图像质量评估依赖 GPT-4o 的主观打分，可能引入评估偏差
- 可探索：视频指令内容（如 YouTube 教程）作为更丰富的交错多模态数据源

## 相关工作与启发

- **vs MMC4 (Zhu et al.)**：MMC4 基于 C4 文本语料用 CLIP 插图，图片稀缺且关联弱；CoMM 从源头定向采集，图片丰富且自然对齐
- **vs OBELICS (Laurençon et al.)**：OBELICS 保留网页原始结构但质量参差不齐；CoMM 聚焦连贯性场景 + 主动过滤
- **vs OpenLEAF (An et al.)**：OpenLEAF 用 GPT-4 + SDXL 在推理时生成交错内容（training-free），但依赖生成模型对文本描述的理解，不如在高质量数据上直接训练
- **启发**：数据质量 > 数据规模——227K 高质量文档 + 10.1 图/文档的 CoMM 优于数百万文档的 MMC4/OBELICS

## 评分

⭐⭐⭐⭐ (4/5)

- 新颖性 ⭐⭐⭐⭐：数据源选择策略和三维过滤设计有洞察，但核心思路（更好数据→更好模型）并不令人意外
- 实验充分度 ⭐⭐⭐⭐：数据集质量对比 + 7 个下游 few-shot 任务 + 新评测基准，覆盖面广
- 写作质量 ⭐⭐⭐⭐：结构清晰，质量指标定义明确，数据分析可视化丰富
- 实用价值 ⭐⭐⭐⭐⭐：数据集开源可直接使用，4 个评测任务填补空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[ICLR 2026\] A High Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation](../../ICLR2026/multimodal_vlm/a_high_quality_dataset_and_reliable_evaluation_for_interleaved_image-text_genera.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)

</div>

<!-- RELATED:END -->
