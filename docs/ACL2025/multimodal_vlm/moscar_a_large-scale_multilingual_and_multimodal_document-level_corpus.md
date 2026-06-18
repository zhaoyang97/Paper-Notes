---
title: >-
  [论文解读] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus
description: >-
  [ACL 2025][多语言/翻译][多语言] 提出 mOSCAR——首个大规模多语言多模态文档级语料库（163种语言、303M文档、200B tokens、1.15B图片），从 Common Crawl 中提取交错的图文文档，并证明在此数据上训练的多语言 mLLM 能获得显著的 few-shot 学习提升。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "多语言"
  - "多模态"
  - "交错图文数据"
  - "Web爬取"
  - "few-shot学习"
---

# mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus

**会议**: ACL 2025  
**arXiv**: [2406.08707](https://arxiv.org/abs/2406.08707)  
**代码**: [oscar-project](https://oscar-project.github.io/documentation/versions/mOSCAR/)  
**领域**: 多语言翻译  
**关键词**: 多语言、多模态、交错图文数据、Web爬取、few-shot学习  

## 一句话总结

提出 mOSCAR——首个大规模多语言多模态文档级语料库（163种语言、303M文档、200B tokens、1.15B图片），从 Common Crawl 中提取交错的图文文档，并证明在此数据上训练的多语言 mLLM 能获得显著的 few-shot 学习提升。

## 研究背景与动机

- **领域现状**: 多模态大语言模型（mLLMs）已取得巨大进展，Flamingo 等模型证明在交错图文序列上训练可涌现出 in-context learning 能力，但其使用的 M3W 数据集是私有且仅有英文版本
- **现有痛点**: 目前公开的交错图文数据集（如 mmc4、OBELICS）仅支持英语；现有多语言图文数据集（如 WIT）规模有限或仅包含 caption 对，无法支撑多语言 mLLM 的预训练
- **核心矛盾**: 全球有 7000+ 种语言，但多语言多模态预训练资源极度匮乏；通过机器翻译生成的多语言数据存在翻译质量差和文化偏差问题
- **本文目标**: 构建一个开源、大规模、覆盖多语言的交错图文文档数据集，用于训练具有 few-shot 学习能力的多语言 mLLM
- **切入角度**: 从 Common Crawl 原始网页中直接提取多语言的交错图文文档，通过系统的过滤、去重和安全过滤管道保障数据质量
- **核心 idea**: 首次系统性地从 Common Crawl 中提取并发布覆盖 163 种语言的大规模交错图文文档语料库

## 方法详解

### 整体框架

mOSCAR 的构建流程包括六个阶段：数据采集 → 语言识别 → 纯文本过滤 → 纯图像过滤 → 数据去污染 → 图文联合过滤。整个管道从三个 2023 年 Common Crawl dump 的 WARC 文件中提取数据。

### 关键设计

#### 1. 数据采集与 DOM 解析
- **功能**: 从 WARC 文件中提取 HTML DOM 树中的文本节点和图像节点
- **核心思路**: 使用 FastWARC 库处理 WARC 文件，通过深度优先遍历 DOM 树提取 `<p>`, `<h*>`, `<title>` 等文本节点和 `<img>` 图像节点
- **设计动机**: HTML 文档天然保持了文本与图像的交错关系，直接从网页结构中提取可保留原始的多模态文档组织形式

#### 2. 语言识别
- **功能**: 为每个文档确定主要语言
- **核心思路**: 使用 open-LID 检测器对每个文本节点进行语言识别，取概率最高的 3 种语言，然后按字符数加权投票确定文档语言
- **设计动机**: 按字符数加权是因为短文本节点（如 "Subscribe"、"Newsletter"）容易误导语言识别

#### 3. 多层安全过滤
- **功能**: 多层过滤确保数据安全和质量
- **核心思路**:
    - **NSFW 文本过滤**: 使用英文正则匹配成人内容，移除匹配文档（约 0.5%）
    - **毒性过滤**: 使用 FLORES 的多语言有毒词列表，文档包含 ≥2 个有毒词时移除
    - **PII 去除**: 用正则替换邮箱、电话、信用卡号、IP 地址等
    - **NSFW 图像过滤**: 组合 nsfw-detector（MobileNet）+ NudeNet 双模型检测
    - **CSAM 检测**: 使用 Thorn 的专有分类器，阈值 0.4 以偏向召回
- **设计动机**: 显式偏向召回率（recall），宁可误删安全内容也要最小化不安全内容的风险

#### 4. 图文联合过滤
- **功能**: 确保文档中的图像和文本彼此相关
- **核心思路**: 使用多语言 NLLB-SigLIP 计算文档内所有图文对的余弦相似度，通过模拟检索任务（随机采样 63 个负样本），如果图/文在 top-8 中则保留
- **设计动机**: 避免使用固定的相似度阈值（不同语言的最优阈值各异），改用相对排名来判断相关性

#### 5. 去重策略
- **功能**: 多粒度去重以提高训练效率
- **核心思路**: 
    - 文档内文本节点精确去重 + Levenshtein ratio（阈值 0.95）近似去重
    - 跨文档使用 MinHashLSH 近似去重（平均移除 19%）
    - 图像使用 URL 匹配 + perceptual hash（pHash）去重
    - 图像跨语言不去重，以促进跨语言迁移
- **设计动机**: 跨语言保留相同图像，因为不同语言文档中出现的相同图像可以促进跨语言知识迁移

### 损失函数

本文是数据集工作，不涉及新损失函数设计。模型训练使用标准的 Flamingo 架构（OpenFlamingo），采用自回归语言建模损失。

## 实验关键数据

### 主实验：多语言 OpenFlamingo 在 8 个基准上的表现

| 设置 | xFlickR&CO | XM3600 | xGQA | MaXM | MaRVL | XVNLI | Multi30K | CoMMuTE |
|------|-----------|--------|------|------|-------|-------|---------|---------|
| mOSCAR+cap (16-shot) | **39.46** | **23.67** | **35.23** | **27.47** | 49.84 | 34.85 | **23.85** | 62.78 |
| cap only (16-shot) | 19.87 | 12.07 | 13.37 | 4.89 | 49.79 | 32.70 | 0.74 | 60.25 |

- 额外使用 mOSCAR 训练的模型在 16-shot 设置下相比仅用 caption 训练的模型，在 xFlickR&CO 上提升 **+19.59**，在 xGQA 上提升 **+21.86**

### 消融实验：mOSCAR vs WIT（公平对比 35M 文档）

| 设置 | xFlickR&CO | XM3600 | xGQA | MaXM |
|------|-----------|--------|------|------|
| mOSCAR+cap (8-shot) | **36.77** | **22.15** | **33.90** | **24.41** |
| WIT+cap (8-shot) | 8.91 | 3.63 | 27.06 | 16.81 |

- 在等量数据对比下，mOSCAR 远优于 WIT，尤其在 captioning 任务上差距巨大

### 关键发现

1. **Few-shot 增益显著**: 从 0-shot 到 16-shot，mOSCAR 模型在 VQA 任务上平均提升 +6.71 分，caption 任务上提升 +19.39 CideR；仅用 caption 训练的模型分别仅提升 +2.82 和 +9.08
2. **多语言无损**: 在全部 43 种语言上训练的多语言模型在英文表现上并未比仅在英文子集上训练的模型差（Table 7）
3. **跨语言图像多样性**: 多语言采样的图像比仅英文的图像更多样（Vendi score 54.78 vs 52.36），表明不同语言文档中的图像分布存在文化差异

## 亮点与洞察

- **系统工程贡献突出**: 完整且严格的数据处理管道覆盖了安全（NSFW/CSAM/PII/有毒内容）、质量（文本节点过滤/去重）和相关性（图文联合过滤）三个核心维度
- **跨语言不去重图像**的设计巧妙——相同图像配合不同语言文本可促进多语言视觉对齐
- **图文联合过滤**使用检索式相对排名代替绝对阈值，天然适应不同语言的分布差异
- 是 OBELICS、mmc4 等英文交错数据集的多语言扩展，填补了重要空白

## 局限与展望

- 未进行系统的偏见分析（网页数据可能反映互联网上的偏见），需要额外的对齐训练来缓解
- 低资源语言虽然被覆盖，但数据量仍远少于高资源语言
- 仅使用 Gemma-2B 作为骨干模型进行验证，更大模型上的表现有待探索
- 过滤管道的文化适应性有限（如毒性词列表可能不适用于所有文化背景）
- 数据集规模虽大但图像需要用户自行下载，存在链接失效风险

## 相关工作与启发

- **Flamingo → OpenFlamingo**: 交错图文训练范式已被验证为 mLLM in-context learning 的关键
- **OBELICS / mmc4**: 英文交错数据集的成功证明了从网页提取文档的可行性
- **LAION-5B 的教训**: 大规模 web 数据的安全问题（如 CSAM）需要系统性解决
- **NLLB-SigLIP**: 多语言 CLIP 变体为图文联合过滤提供了跨语言支持
- **启发**: 在构建多语言多模态数据集时，"质量-多样性权衡"是核心设计决策

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐: 首个大规模多语言交错图文数据集，填补重要空白
- **实用性** ⭐⭐⭐⭐⭐: 开源发布、CC BY 4.0 许可，直接推动多语言 mLLM 研究
- **实验** ⭐⭐⭐⭐: 多维度评估（质量/多样性/安全性/模型训练），但仅验证了小规模模型
- **写作** ⭐⭐⭐⭐: 管道描述详尽，但论文较长且实验部分集中在附录

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)
- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)
- [\[ACL 2025\] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)

</div>

<!-- RELATED:END -->
