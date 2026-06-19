---
title: >-
  [论文解读] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)
description: >-
  [ACL 2025][多语言/翻译][多语言术语] 构建首个大规模多语言 AI 术语数据集 GIST（约 5K 术语、5 种语言），采用 LLM 抽取 + 人工众包翻译 + LLM 选择的混合框架，并通过 prompting 后翻译优化方法在 BLEU/COMET 等指标上一致提升机器翻译中 AI 术语的翻译质量。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "多语言术语"
  - "AI术语翻译"
  - "众包翻译"
  - "后翻译优化"
  - "LLM-Human混合框架"
---

# Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)

**会议**: ACL 2025  
**arXiv**: [2412.18367](https://arxiv.org/abs/2412.18367)  
**代码**: [GitHub](https://github.com/jiarui-liu/MultilingualAITerminology)  
**机构**: Carnegie Mellon University, University of Michigan, University of Toronto, Max Planck Institute
**领域**: NLP / 机器翻译 / 术语翻译  
**关键词**: 多语言术语, AI术语翻译, 众包翻译, 后翻译优化, LLM-Human混合框架

## 一句话总结

构建首个大规模多语言 AI 术语数据集 GIST（约 5K 术语、5 种语言），采用 LLM 抽取 + 人工众包翻译 + LLM 选择的混合框架，并通过 prompting 后翻译优化方法在 BLEU/COMET 等指标上一致提升机器翻译中 AI 术语的翻译质量。

## 研究背景与动机

机器翻译领域已取得显著进展，但 AI 专业术语翻译仍是顽疾。"Coreference Resolution""Explain-Away Effect" 等术语在 Google Translate 等通用系统中频繁被错误翻译，导致非英语研究者理解困难甚至产生误解。

- **现有资源严重不足**：ACL 60-60 计划仅覆盖约 250 个术语，远不能满足需求
- **LLM 翻译不一致**：Claude 3 Sonnet、GPT-3.5-Turbo 和 Google Translate 三者的三模型一致率仅约 15%（中文较高为 42.71%），双模型一致率约 40%
- **人工构建难以扩展**：完全依赖领域专家构建多语言术语库成本高且耗时
- **影响面广**：非英语人群占全球 AI 社区的重要比例，Hugging Face 模型卡、数据卡等平台文档中的术语翻译错误会导致模型和数据集的误用

## 方法详解

### 整体框架

GIST 构建流程为：**术语抽取 → 人工众包翻译 → LLM 最佳候选选择 → 后翻译集成**。从 18 个顶级会议（AAAI/IJCAI/CVPR/ECCV/ICCV/ICLR/ICML/NeurIPS/KDD/ACL/EMNLP/NAACL/EACL/LREC/COLING/CoNLL/SIGIR/WWW）的 879 篇获奖论文中抽取术语，翻译为阿拉伯语、中文、法语、日语和俄语。

### 关键设计

1. **LLM + 多轮质量保证的术语抽取**：使用 LLaMA-3-70B-Instruct 从获奖论文中抽取术语，术语定义标准严格——必须是名词/名词短语、AI 专有、在非 AI 领域无意义或有不同含义。处理粒度为最多 64 词的句子块。后续经过多轮过滤：去除仅出现 1 篇的术语 → 去除缩写和特殊字符开头 → GPT-4o 进一步筛选 → 3 位领域专家人工审核。同时整合 Wikipedia AI 词汇表、各国政府 AI 术语词典等外部资源。

2. **人工众包 + LLM 验证的混合翻译**：先用 Claude 3 Sonnet/GPT-3.5-Turbo/Google Translate 测试自动翻译可行性，发现三模型一致率极低（大多数语言仅 ~15%），因此采用 Amazon Mechanical Turk 众包——每个术语收集 10 个人工翻译 + 1 个 Google Translate 翻译，再由 GPT-4o 从 11 个候选中选择最佳翻译。众包流程含严格资质测试和日常质量监控。

3. **无需重训练的术语集成方法**：探索三种后翻译集成策略——(a) **Prompting 优化**：用 GPT-4o-mini 对初始翻译进行修正，将术语词典作为上下文提供；(b) **词对齐+替换**：用多语言 BERT 做词对齐，找到源术语在译文中的对应位置并替换为 GIST 中的翻译；(c) **约束解码**：包括约束束搜索和 token 级 logits 调整。

## 实验关键数据

### GIST 数据集统计

| 统计项 | 阿拉伯语 | 中文 | 法语 | 日语 | 俄语 |
|--------|---------|------|------|------|------|
| 术语数量 | 4,844 | 6,426 | 6,527 | 4,770 | 5,167 |
| 英文唯一词数 | 2,470 | 3,244 | 3,470 | 2,424 | 2,615 |
| 目标语言唯一词数 | 3,161 | 2,838 | 4,036 | 2,050 | 4,210 |
| 英文词/术语 | 2.02±0.59 | 2.05±0.68 | 2.07±0.67 | 2.02±0.58 | 2.01±0.59 |
| 目标语言字符/术语 | 15.22±5.66 | 4.66±1.96 | 21.27±8.49 | 6.89±3.16 | 20.20±7.83 |

### Prompting 后翻译优化效果（60-60 评估集，BLEU 提升）

| 模型 | 阿拉伯语 | 中文 | 法语 | 日语 | 俄语 |
|------|---------|------|------|------|------|
| gpt-4o-mini | 23.58 → +1.07 | 32.64 → +1.60 | 40.80 → +3.08 | 21.46 → +0.64 | 17.25 → +1.07 |
| aya-expanse | 20.11 → +1.23 | 27.31 → +1.33 | 33.05 → +2.46 | 14.59 → +0.61 | 16.59 → +1.59 |
| nllb | 22.38 → +1.37 | 17.29 → +1.92 | 34.93 → +2.86 | 6.19 → +2.42 | 17.30 → +1.54 |
| seamless | 23.13 → +1.16 | 26.26 → +0.97 | 40.04 → +2.08 | 14.56 → +0.74 | 17.18 → +1.71 |
| aya-23-8B | 19.98 → +0.54 | 26.08 → +0.47 | 33.85 → +2.28 | 15.06 → +0.87 | 15.77 → +1.05 |

### GPT-4o 选择 vs. 多数投票（Task 1 人工评估）

| 评判结果 | 阿拉伯语 | 中文 | 法语 | 日语 | 俄语 |
|---------|---------|------|------|------|------|
| 两种翻译都好 | 45.76% | 50.59% | 48.67% | 56.99% | 54.43% |
| GPT-4o 选择更好 | 28.54% | 28.76% | 30.44% | 24.37% | 30.26% |
| 多数投票更好 | 20.37% | 17.62% | 18.89% | 15.44% | 13.04% |
| 两种都差 | 4.46% | 2.70% | 1.89% | 2.62% | 1.22% |

### GIST vs. 60-60 术语翻译质量（Task 2 人工评估）

| 评判结果 | 阿拉伯语 | 中文 | 法语 | 日语 | 俄语 |
|---------|---------|------|------|------|------|
| 两种翻译都好 | 46.42% | 37.17% | 39.48% | 57.28% | 39.09% |
| GIST 更好 | 29.38% | 43.02% | 43.64% | 31.46% | 45.00% |
| 60-60 更好 | 17.65% | 16.04% | 13.77% | 6.99% | 8.64% |
| 两种都差 | 5.68% | 3.21% | 2.60% | 4.08% | 5.68% |

### 关键发现

1. **Prompting 方法一致有效**：在几乎所有语言、模型和评估指标（BLEU/COMET/ChrF/ChrF++/TER）上均提升翻译质量，统计检验 p-value = 0.00
2. **词对齐方法效果因语言而异**：对中文、日语有效（形态变化小，直接替换不破坏语法），但对阿拉伯语、法语、俄语有时反而降低质量（这些语言需要性、数、格等形态一致性）
3. **GPT-4o 选择优于多数投票**：在所有 5 种语言中，GPT-4o 选择的翻译候选均显著优于多数投票选择
4. **GIST 翻译质量优于 60-60**：在所有 5 种语言的人工对比评估中，GIST 的翻译一致且显著优于 ACL 60-60 评估集
5. **数据集覆盖充分**：稀释曲线分析表明 60% 的论文子集即可覆盖 80% 以上术语（t-statistic = 64.78, p = 0），域分布涵盖统计学(13.31%)、数学(12.24%)、CS(11.74%)、NLP(11.50%)、DS(9.98%)、CV(6.57%) 等

## 论文亮点与不足

### 亮点

- 首个大规模（5K）多语言 AI 术语数据集，规模远超 60-60（~250 个术语）
- LLM 抽取 + 人工翻译 + LLM 验证的混合框架兼顾效率与质量
- 无需重训练的后翻译集成方案实用性强
- 在 ACL Anthology 网站上提供翻译演示系统（acl6060.org），可实时对比原始翻译与优化后翻译
- 实验覆盖 5 个模型 × 5 种语言 × 5 个指标 × 2 个评估集，实验设计严谨

### 不足

- 假设术语英文-目标语言为一对一映射，忽略了同一术语可能有多个同等有效翻译的情况
- 仅覆盖 5 种语言，远未覆盖全球语言多样性
- AI 领域边界模糊，术语覆盖不可能穷尽
- 数据集更新依赖人工判断新术语的相关性和影响力，LLM 受知识截止日期限制难以完全自动化
- 约束解码和 logits 调整方法运行速度慢约 100 倍，且生成质量差（重复术语或完全忽略术语）

## 相关工作与对比

- **ACL 60-60 计划**：仅约 250 个术语的评估集，GIST 在规模上扩展约 20 倍且翻译质量经人工评估验证更优
- **全人工构建方法**：成本高难以扩展；GIST 通过 LLM+众包混合降低成本
- **全自动方法**：准确性不足；三模型一致率仅 ~15%（中文 42.71%）
- **训练式术语集成**：需要数据增广+微调或修改模型架构，不适应新术语；GIST 的后翻译方法无需重训练
- **约束解码方法**：虽然灵活但速度慢且牺牲翻译准确率；prompting 方法在效果和效率上均更优
4. **中日两语言特殊**：中文和日语的每术语目标字符数很少（4.66 和 6.89），反映了汉字的信息密度
5. **约束解码方法不稳定**：虽然保证术语出现但有时损害整体翻译流畅性

## 亮点与洞察

- **混合人机协作流程**具有很好的可扩展性：LLM 负责大规模抽取和筛选，人类专家保证质量
- 从获奖论文中抽取术语的策略巧妙——兼顾代表性和质量控制
- 无需重训练的术语集成方法对实际应用有很大价值，特别是 prompting 方法的简洁性
- 开发的 ACL Anthology 网站演示展示了实际应用场景
- 覆盖 2000-2023 年的时间跨度确保了术语的时间完整性

## 局限与展望

- 仅覆盖 5 种目标语言，可扩展到更多语言（如韩语、印地语）
- 术语仅来自获奖论文，可能遗漏重要但非获奖论文中的术语
- 术语抽取和翻译的时效性维护是长期挑战
- 约束解码方法需要根据模型架构定制，通用性有限
- 未评估术语翻译对下游 NLP 任务的实际影响

## 相关工作与启发

- 与 ACL 60-60 计划相比，GIST 在规模上提升了一个数量级
- 与 Feng et al. (2024) 的 LLM 术语翻译研究互补——GIST 提供了大规模验证
- 术语集成方法中 prompting 方法的优势呼应了 LLM few-shot learning 的范式
- 启发：领域特定的多语言资源建设应采用 LLM+人工的混合流程，而非纯人工或纯自动

## 评分

- **新颖性**: ⭐⭐⭐ — 首个大规模AI术语的多语言数据集，贡献主要在资源层面
- **实验充分度**: ⭐⭐⭐⭐ — 多语言覆盖、自动+人工评估、三种集成方法对比
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集构建流程描述详细
- **价值**: ⭐⭐⭐⭐ — 对促进AI知识的全球化和多语言可及性有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)
- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)
- [\[ACL 2025\] LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World](lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)
- [\[ACL 2025\] An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)](an_expanded_massive_multilingual_dataset_for_high-performance_language_technolog.md)

</div>

<!-- RELATED:END -->
