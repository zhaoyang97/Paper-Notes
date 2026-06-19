---
title: >-
  [论文解读] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages
description: >-
  [多语言/翻译] 构建了首个面向中国少数民族语言（藏语、维吾尔语、哈萨克语、蒙古语）的标准化LLM评估基准MiLiC-Eval，包含9类任务2.4万实例，揭示了当前LLM在非主流书写系统上的严重不足。 - 领域现状： LLM在英语和中文等高资源语言上表现出色，但对数千种低资源语言的支持严重不足，尤其是使用非主流书写系统的语言…
tags:
  - "多语言/翻译"
---

# MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages

## 论文信息

- **会议**: ACL 2025
- **arXiv**: [2503.01150](https://arxiv.org/abs/2503.01150)
- **代码**: [https://github.com/luciusssss/MiLiC-Eval](https://github.com/luciusssss/MiLiC-Eval)
- **领域**: 多语言 / 低资源语言评估
- **关键词**: 低资源语言, 少数民族语言, 多语言LLM, 评测基准, 书写系统

## 一句话总结

构建了首个面向中国少数民族语言（藏语、维吾尔语、哈萨克语、蒙古语）的标准化LLM评估基准MiLiC-Eval，包含9类任务2.4万实例，揭示了当前LLM在非主流书写系统上的严重不足。

## 研究背景与动机

- **领域现状**: LLM在英语和中文等高资源语言上表现出色，但对数千种低资源语言的支持严重不足，尤其是使用非主流书写系统的语言。
- **关键差距**: 中国的藏语、维吾尔语、哈萨克语和蒙古语虽有数千万使用者，但在NLP研究中被严重边缘化，且缺乏标准化评测基准。
- **已有基准局限**: 现有多语言基准（XTREME、MEGA等）对低资源书写系统覆盖不足、任务类型单一、缺乏跨任务可比性，部分使用机器翻译数据导致评估失真。
- **核心挑战**: 这些语言使用非拉丁文字（传统蒙文、藏文等），对分词和语言建模构成额外挑战。

## 方法详解

### 整体框架

MiLiC-Eval涵盖4种少数民族语言（藏语bo、维吾尔语ug、哈萨克语kk、蒙古语mn）的9类任务，共2.4万实例。基准设计遵循三个原则：关注非主流书写系统、跨语言/跨任务平行性、细粒度技能评估。

### 关键设计

1. **非主流书写系统聚焦**: 首次benchmark阿拉伯字母哈萨克语和传统蒙古文（而非主流西里尔文），直击LLM最薄弱环节
2. **跨语言与跨任务平行性**: 6项任务在6种语言（含中英）间提供平行数据，相同文本用于多个任务，避免单一任务格式带来的偏见评估
3. **层次化技能评估体系**: 将9类任务映射到语言能力（词汇→语法→语用）和问题解决能力（话题建模→上下文理解→生成→推理）两个维度

### 九类评估任务

| 任务 | 每语言实例数 | 评估技能 |
|------|-----------|---------|
| 词汇理解 | 1,000 | 词汇知识 |
| 话题分类(句子) | 492 | 话题建模 |
| 话题分类(篇章) | 600 | 话题建模 |
| 阅读理解 | 250 | 上下文理解 |
| 回复选择 | 507 | 语用推理 |
| 标题生成 | 1,000 | 文本生成 |
| 机器翻译(文章) | 1,012 | 翻译能力 |
| 机器翻译(对话) | 773 | 翻译能力 |
| 数学推理 | 250 | 符号推理 |

## 实验

### 主实验：各模型语言平均得分

| 模型 | 藏语(bo) | 维吾尔语(ug) | 哈萨克语(kk) | 蒙古语(mn) | 平均 |
|------|---------|------------|------------|-----------|------|
| Qwen-2.5-7B | 29.4 | 48.0 | 37.0 | 24.9 | 34.8 |
| Qwen-3-8B | 34.5 | 56.5 | 46.7 | 28.7 | 41.6 |
| Gemma-3-12B | **53.3** | **63.7** | **57.5** | 25.1 | **49.9** |
| EMMA-500-7B | 25.3 | 42.5 | 27.4 | 17.8 | 28.2 |
| GPT-4.1 | 57.0 | 72.0 | 65.9 | 27.2 | 55.5 |
| Gemini-2.0-Flash | **72.9** | **75.0** | **70.9** | **66.8** | **71.4** |

### 消融分析：机器翻译数据 vs 人工翻译数据评估对比

| 语言 | 阅读理解(下降%) | 回复选择(下降%) | 数学推理(下降%) |
|------|-------------|-------------|-------------|
| 藏语 | 40.0 (-21%) | 36.9 (-15%) | 11.9 (-52%) |
| 维吾尔语 | 41.8 (-19%) | 42.2 (-17%) | 31.3 (-29%) |
| 哈萨克语 | 40.3 (-19%) | 32.3 (-16%) | 19.3 (-22%) |

### 关键发现

1. **蒙古语是最大短板**: 即使最佳开源模型Gemma-3在蒙古语上也仅略好于随机，传统蒙古文的tokenization效率极低（GPT-4.1需432 tokens/句，西里尔蒙古文仅需54）
2. **专门多语言适配模型反而不占优**: EMMA-500和BayLing-2虽专门训练了少数民族语言数据，但表现不如原生多语言LLM（Gemma-3、Qwen-3）
3. **技能不均衡**: LLM具备基本词汇理解和话题建模能力，但在生成和翻译等需要语法知识的任务上严重不足
4. **书写系统切换问题严重**: GPT-4o-mini在蒙古语标题生成中36%的情况下切换到西里尔文，哈萨克语高达95%
5. **机器翻译评估数据导致严重失真**: 使用NLLB翻译数据评估时性能最高下降52%（数学推理）

## 亮点

- 首个系统性覆盖中国少数民族语言非主流书写系统的LLM基准
- 跨任务平行设计避免了单一任务格式的评估偏见
- 人工翻译数据确保评估的真实性和可靠性
- 揭示了tokenization效率与模型性能之间的强相关

## 局限性

- 仅覆盖4种少数民族语言，未涉及壮语、苗语等其他语言
- 部分任务数据量较小（如数学推理仅250条），可能影响统计显著性
- 基准主要评估ICL能力，未涉及微调后的模型表现
- 技能分类框架的完备性有待验证

## 相关工作

- **多语言基准**: XTREME（Hu et al., 2020）、MEGA（Ahuja et al., 2023）、Belebele（Bandarkar et al., 2024）
- **低资源语言NLP**: EMMA-500（Ji et al., 2024）、LLaMAX-3（Lu et al., 2024）
- **中国少数民族语言**: MC2语料库（Zhang et al., 2024b）、WCM（Yang et al., 2022）
- **翻译评估**: FLORES+（NLLB Team et al., 2024）、SIB-200（Adelani et al., 2024）

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总评 | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](../../ICLR2026/multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[ACL 2025\] Did Translation Models Get More Robust Without Anyone Even Noticing?](translation_robustness.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)

</div>

<!-- RELATED:END -->
