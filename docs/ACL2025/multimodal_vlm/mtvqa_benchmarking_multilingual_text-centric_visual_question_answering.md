---
title: >-
  [论文解读] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering
description: >-
  [ACL 2025][多语言/翻译][多语言VQA] 构建了 MTVQA——首个覆盖 9 种语言的多语言文本中心视觉问答基准，通过人类专家标注解决翻译方法的"视觉-文本不对齐"问题，评估显示最佳 MLLM（InternVL-2.5，32.2%）与人类表现（79.7%）差距巨大，揭示了多语言文本理解的严峻挑战。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "多语言VQA"
  - "文本中心视觉问答"
  - "MLLM评估"
  - "低资源语言"
  - "基准数据集"
---

# MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering

**会议**: ACL 2025  
**arXiv**: [2405.11985](https://arxiv.org/abs/2405.11985)  
**代码**: [有](https://github.com/bytedance/MTVQA)  
**领域**: 多语言翻译  
**关键词**: 多语言VQA, 文本中心视觉问答, MLLM评估, 低资源语言, 基准数据集

## 一句话总结

构建了 MTVQA——首个覆盖 9 种语言的多语言文本中心视觉问答基准，通过人类专家标注解决翻译方法的"视觉-文本不对齐"问题，评估显示最佳 MLLM（InternVL-2.5，32.2%）与人类表现（79.7%）差距巨大，揭示了多语言文本理解的严峻挑战。

## 研究背景与动机

文本中心视觉问答（TEC-VQA）是评估 AI 在文本丰富场景中理解能力的重要代理任务，但目前存在两个核心痛点：

**语言覆盖偏差**：现有 TEC-VQA 基准几乎只关注英文和中文等高资源语言，低资源语言（阿拉伯语、泰语、越南语等）被严重忽视

**翻译方案的根本缺陷**：之前的多语言 VQA 工作（xGQA、MaXM）通过翻译引擎扩展问答对，但在 TEC-VQA 场景下产生严重的"视觉-文本不对齐"问题——翻译只处理了问答文本，忽略了图像中的视觉文本

举例来说，一张俄语菜单的图片，翻译引擎可能错误翻译菜名，导致问答对与图像中实际显示的文字不一致。这种不对齐在一般 VQA 中不是问题（因为问答不依赖图中文字），但在 TEC-VQA 中是致命的。

## 方法详解

### 整体框架

MTVQA 的构建遵循三个阶段：（1）多源文本丰富图像采集；（2）两轮人类专家标注（maker-checker 范式）；（3）大规模 MLLM 基准测试。

### 关键设计

1. **图像采集与清洗**：

    - 来源三方面：公开数据集（ICDAR MLT19 等，30%）、网络爬取（Common Crawl，20%）、实地拍摄（50%）
    - 实地拍摄在各语言所在国家/地区完成，确保图像的原生性和高质量
    - 多语言 OCR 引擎筛选含文本的图像，算法过滤敏感/不良内容
    - 覆盖 20+ 细粒度场景：菜单、地图、账单、PPT、论文等
    - 最终 2,116 张测试图 + 6,678 张训练图

2. **两轮人类专家标注**：

    - **标注员资质**：母语使用 10 年以上 + 大学学历以上
    - **第一轮（生成）**：每张图由 3 个标注员生成 5 个问答对，前 3 个要求直接阅读图中文字回答，后 2 个要求对图中文字进行推理
    - **第二轮（校验）**：另一组 2 个标注员独立审核每个问答对的相关性、准确性、简洁性和伦理性
    - 10% 抽样检查质量，不合格返回重标
    - 这种设计成本高（约 90,000 美元，历时 5 个月），但确保了标注质量

3. **9 种语言覆盖**：

    - 阿拉伯语(AR)、韩语(KO)、日语(JA)、泰语(TH)、越南语(VI)、俄语(RU)、法语(FR)、德语(DE)、意大利语(IT)
    - 涵盖了多种文字系统（阿拉伯文、韩文、日文、泰文、拉丁文、西里尔文）
    - 最终数据：28,607 个问答对，8,794 张图

### 评估设计

- 采用 Accuracy 而非 ANLS 作为主指标，因为 ANLS 无法准确反映图像中文本内容的正确性
- 统一 prompt 格式限制输出长度，使答案简洁可评
- 测试人类基线：每种语言 10 位母语者

## 实验关键数据

### 主实验——MLLM 多语言 TEC-VQA 表现（Accuracy %）

| 模型 | AR | DE | FR | IT | JA | KO | RU | TH | VI | Avg |
|------|------|------|------|------|------|------|------|------|------|------|
| **人类** | 76.9 | 80.2 | 84.1 | 78.0 | 79.1 | 81.7 | 76.3 | 78.4 | 82.8 | **79.7** |
| InternVL2.5-78B | 15.9 | 39.0 | 45.6 | 42.9 | 21.1 | 33.9 | 12.2 | 23.8 | 41.5 | **32.2** |
| Qwen2-VL-72B | 20.7 | 36.5 | 44.1 | 42.8 | 21.6 | 37.4 | 15.6 | 17.7 | 41.6 | 30.9 |
| GPT-4o | 20.2 | 34.2 | 41.2 | 32.7 | 20.0 | 33.9 | 11.5 | 22.5 | 34.2 | 27.8 |
| Claude3 Opus | 15.1 | 33.4 | 40.6 | 34.4 | 19.4 | 27.2 | 13.0 | 19.5 | 29.1 | 25.7 |
| TextSquare | 3.7 | 27.0 | 30.8 | 26.7 | 3.2 | 7.2 | 6.7 | 5.2 | 12.4 | 13.6 |

### 关键对比分析

| 维度 | 发现 |
|------|------|
| 人类 vs 最佳 MLLM | 79.7% vs 32.2%，差距 47.5%，巨大提升空间 |
| 拉丁字母语言 vs 非拉丁 | DE/FR/IT 普遍高于 AR/JA/TH/RU，因训练数据偏差 |
| 文本专用 vs 通用 MLLM | TextMonkey(9.9%) < MiniCPM(17.3%) < InternVL(32.2%)，文本专用模型因只关注英中而落后 |
| OCR+GPT-4 vs GPT-4V | 21.6% vs 22.0%，各有优劣 |
| OCR+GPT-4V | 28.3%，最佳组合 |
| 指令微调提升 | Xcomposer-4KHD: 11.2% → 19.7%（+8.5%） |

### 错误分析统计

| 错误类型 | 比例 |
|---------|------|
| OCR 识别失败 | 39% |
| 推理不足 | 34% |
| 语言偏差 | 15% |
| 幻觉 | 12% |

### 关键发现

1. 所有模型在非拉丁字母语言上显著更差，阿拉伯语和俄语尤甚
2. 开源 Qwen2-VL 和 InternVL 系列已能超越 GPT-4V/GPT-4o
3. 文本专用 MLLM 因过度关注英中而在多语言场景下被通用模型反超
4. Few-shot 提升有限且会饱和（zero-shot 22.0% → 5-shot 24.8%）
5. 英文提问 vs 原语言提问效果几乎无差异，瓶颈在视觉文本感知而非语言理解

## 亮点与洞察

1. **问题定义精准**：清楚地指出了翻译方案在 TEC-VQA 上的根本缺陷（视觉-文本不对齐），并用高质量人工标注来解决
2. **覆盖面广**：9 种语言、20+ 场景类型、文档+自然场景双覆盖，是目前最全面的多语言 TEC-VQA 基准
3. **揭示性强的差距**：32.2% vs 79.7% 的巨大差距清晰表明 MLLM 在多语言文本理解上还远未及格
4. **OCR 失败占 39%**：提示了提升文本感知（而非语言理解）才是关键

## 局限与展望

1. **语言覆盖仍有限**：9 种语言仍遗漏了许多低资源语言（印地语、斯瓦希里语等）
2. **答案形式单一**：只要求简短答案，缺少需要长答案的推理题
3. **评价指标局限**：Accuracy 要求精确匹配，对阿拉伯语等形态丰富语言可能过于严苛
4. **成本壁垒高**：约 9 万美元的标注成本使得快速扩展语言覆盖困难
5. **训练集较小**：6,678 张训练图的规模限制了指令微调效果

## 相关工作与启发

- **通用多语言 VQA**：xGQA (7语言)、MaXM (7语言) 采用翻译方案，只适合不依赖图中文字的场景
- **TEC-VQA 基准**：TextVQA、DocVQA、OCRBench 都以英文为主
- MTVQA 的核心启发：为低资源语言构建 TEC-VQA 基准必须做原生标注，翻译方案行不通

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个原生标注的多语言 TEC-VQA 基准
- **实验充分度**: ⭐⭐⭐⭐⭐ — 评估了 20+ 模型，含闭源和开源，有人类基线和详细错误分析
- **写作质量**: ⭐⭐⭐⭐ — 数据集构建过程描述详尽，实验分析深入
- **价值**: ⭐⭐⭐⭐⭐ — 填补了多语言 TEC-VQA 的重要空白，揭示了 MLLM 的巨大改进空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AskQE: Question Answering as Automatic Evaluation for Machine Translation](askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [\[ACL 2026\] What Factors Affect LLMs and RLLMs in Financial Question Answering?](../../ACL2026/multilingual_mt/what_factors_affect_llms_and_rllms_in_financial_question_answering.md)
- [\[ACL 2025\] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages](milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages.md)
- [\[ACL 2025\] Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning](multilingual_speech_data_quality.md)
- [\[ACL 2025\] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)

</div>

<!-- RELATED:END -->
