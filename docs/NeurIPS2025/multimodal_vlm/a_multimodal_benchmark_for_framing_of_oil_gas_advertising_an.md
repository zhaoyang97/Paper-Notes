---
title: >-
  [论文解读] A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection
description: >-
  [NeurIPS 2025 (Datasets and Benchmarks Track)][多模态VLM][洗绿检测] 构建了首个面向石油天然气行业视频广告的多模态框架分析基准数据集（706 个视频、13 种框架类型、50+ 实体、20 个国家），系统评估了 6 款 VLM 在检测 greenwashing 相关 framing 中的能力，发现 GPT-4.1 零样本在环境类标签上达 79% F1 但绿色创新仅 46%，揭示了隐式框架分析和文化背景理解仍是 VLM 的核心挑战。
tags:
  - "NeurIPS 2025 (Datasets and Benchmarks Track)"
  - "多模态VLM"
  - "洗绿检测"
  - "框架分析"
  - "视频广告"
  - "视觉语言模型"
  - "石油天然气行业"
---

# A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection

**会议**: NeurIPS 2025 (Datasets and Benchmarks Track)  
**arXiv**: [2510.21679](https://arxiv.org/abs/2510.21679)  
**代码**: [GitHub](https://github.com/climate-nlp/multimodal-oil-gas-benchmark) / [HuggingFace](https://huggingface.co/datasets/climate-nlp/multimodal-oil-gas-benchmark)  
**领域**: 多模态VLM / 社会计算  
**关键词**: 洗绿检测, 框架分析, 视频广告, 视觉语言模型, 石油天然气行业

## 一句话总结

构建了首个面向石油天然气行业视频广告的多模态框架分析基准数据集（706 个视频、13 种框架类型、50+ 实体、20 个国家），系统评估了 6 款 VLM 在检测 greenwashing 相关 framing 中的能力，发现 GPT-4.1 零样本在环境类标签上达 79% F1 但绿色创新仅 46%，揭示了隐式框架分析和文化背景理解仍是 VLM 的核心挑战。

## 研究背景与动机

**领域现状**：石油天然气（O&G）公司通过精心设计的公关活动塑造品牌形象，常被指控通过展示气候友好的形象进行"洗绿"（greenwashing）。框架分析是理解企业战略传播的重要工具——"框架就是选择性地呈现现实的某些方面"。

**现有痛点**：已有的 greenwashing 检测基准仅关注文本模态（如广告文案的 framing 分析），完全忽略了视觉信息。然而定性研究表明视频广告中的图像策略是洗绿的重要手段——展示风力发电机、微笑工人等视觉符号传达隐式的"绿色"印象。约 30% 的广告视频甚至没有语音，纯靠视觉传达信息。

**核心矛盾**：视频中的框架信息往往是隐式的（implicit framing）——不直接用文字说"我们环保"，而是通过太阳能电池板画面暗示。VLM 需要同时理解视觉符号、文化语境和企业策略才能准确分类，但目前缺乏评估这种能力的基准。

**本文目标** (1) 构建首个多模态（视频+文本+转录）的 O&G 广告框架分析基准；(2) 覆盖 Facebook 和 YouTube 两个平台的不同广告策略；(3) 系统评估当前 VLM 在此任务上的能力和瓶颈。

**切入角度**：从两个互补平台（Facebook 的政治广告 vs YouTube 的企业品牌宣传）收集数据，分别定义细粒度框架类型，并设计 entity-aware 的 1-shot 提示策略来提升 VLM 表现。

**核心 idea**：构建首个多模态 O&G 广告 framing 基准，定量暴露 VLM 在视频 greenwashing 检测任务中隐式框架和文化理解方面的系统性不足。

## 方法详解

### 整体框架

任务定义为多标签分类：输入为一段 O&G 实体发布的社交媒体视频，输出为一组框架类型标签。数据集分为 Facebook 子集（7 种细粒度气候阻挠框架类型，320 个视频）和 YouTube 子集（6 种印象型框架类型，386 个视频）。评估基于零样本和 1-shot VLM 推理。

### 关键设计

1. **双平台双框架体系**:

    - 功能：全面覆盖 O&G 行业不同平台上的差异化广告策略
    - 核心思路：**Facebook 子集**继承 Holder et al. 的气候阻挠框架（7 类标签：CA=社区经济、CB=创造就业、GA=减排转型、GC=清洁天然气、PA=实用能源选择、PB=原材料用途、SA=国内能源独立），标签来自先前文本标注。**YouTube 子集**全新定义 6 类印象型标签（Community&Life、Work、Environment、Green Innovation、Economy&Business、Patriotism），由人类专家从视频内容直接标注
    - 设计动机：Facebook 广告短小、政治化、文字主导；YouTube 视频长、品牌化、视觉隐含——两个平台的框架策略本质不同，单一体系无法覆盖

2. **Entity-aware 1-shot 提示构建**:

    - 功能：为每个测试视频选择最相关的示例进行 in-context learning
    - 核心思路：(1) **Entity Restriction (ER)**：将候选示例限制为同一实体的训练样本；(2) **Embedding-based Search (ES)**：用 CLIP 分别编码视频帧和转录文本，加权求和为视频表示 $\bm{e} = 0.5\bm{e}_{\rm Frame} + 0.5\bm{e}_{\rm Transcript}$，用余弦相似度检索最相似的训练样本作为 1-shot 示例
    - 设计动机：同一企业的广告策略具有一致性（如 BP 总是强调转型，Exxon 强调就业），entity-aware 选择比随机选择更有信息量

3. **帧-转录对齐的动态采样**:

    - 功能：从长视频中选择最具代表性的帧与对应文本
    - 核心思路：根据 Whisper-1 提取的转录段落时间戳，取每段起止时间的中点对应帧，最多采样 $N_{\rm Frame}$ 帧（GPT-4.1/Qwen2.5-VL 为 10，InternVL2/DeepSeek 为 3）。每帧与对应转录段配对输入 VLM
    - 设计动机：视频长度差异大（Facebook 平均 18s，YouTube 平均 76s），需要动态采样；帧与文本的对齐确保 VLM 获得匹配的多模态上下文

### 损失函数 / 训练策略

不涉及模型训练——所有 VLM 均以零样本或 few-shot 推理方式测试。标注使用 Fleiss' Kappa 衡量一致性，YouTube 子集达到 0.61。

## 实验关键数据

### 主实验

| 模型 | 参数量 | YouTube (All) | Facebook (All) | 最佳标签F1 | 最差标签F1 |
|------|--------|--------------|----------------|-----------|-----------|
| GPT-4.1 (0-shot) | — | 71.0 | 61.1 | 84.9 (Comm.) | 46.1 (Green Innov.) |
| GPT-4.1 (1-shot) | — | 69.3 | 72.6 | 80.6 (Comm.) | 41.6 (Green Innov.) |
| Qwen2.5-VL (0-shot) | 32B | 60.7 | 49.0 | 73.5 (Env.) | 42.8 (Patrio.) |
| Qwen2.5-VL (1-shot) | 32B | 66.2 | 70.5 | 77.4 (Env.) | 45.8 (Green Innov.) |
| GPT-4o-mini (0-shot) | — | 60.5 | 54.2 | 72.8 (Env.) | 39.2 (Green Innov.) |
| DeepSeek-VL2 (1-shot) | 4.5B | 49.7 | 62.3 | 68.6 (Comm.) | 21.4 (Green Innov.) |

### 消融实验

| 配置 | YouTube | Facebook | 说明 |
|------|---------|----------|------|
| 完整 (T+ES+ER) | 66.2 | 70.5 | Qwen2.5-VL 32B |
| 去掉转录 (T=✗) | 61.2 | 60.6 | 转录对Facebook影响更大 |
| 去掉嵌入搜索 (ES=✗) | 64.0 | 59.1 | 随机选择示例效果差 |
| 去掉实体限制 (ER=✗) | 65.6 | 68.1 | 实体信息对YouTube帮助有限 |

### 关键发现

- **Green Innovation 一致是最难标签**：所有模型在此类别上表现最差（最佳仅 46.1% F1），因为"绿色创新"的视觉表达极为隐含——实验室场景可能是研发也可能是日常运营
- **GPT-4.1 零样本优于 1-shot**（YouTube），但其他模型受益于 1-shot——说明最强模型已内化了足够的领域知识，额外示例反而引入噪声
- **转录信息在 Facebook 上至关重要**：去掉转录后 F1 从 70.5 降到 60.6（-14%），因为 Facebook 广告的视频常仅是背景，文字才是核心信息载体
- **小模型也能有效**：4.5B 的 DeepSeek-VL2 在 Facebook 1-shot 上达到 62.3%，接近 GPT-4o-mini，说明任务特定提示设计比纯粹扩大模型更重要
- 约 30% Facebook 视频无转录，纯视觉理解对这些样本至关重要

## 亮点与洞察

- **首个多模态 greenwashing 基准**填补了该领域从文本到视频的关键空白，数据集设计考虑了跨平台（Facebook vs YouTube）、跨实体（50+ 公司/倡论团体）、跨文化（20 国）的多样性
- **Entity-aware 提示策略**简单但有效——利用同一企业广告策略的一致性提升 few-shot 效果，这种思路可迁移到任何实体级分类任务（如品牌情感分析、政治广告检测）
- **"印象型标注"的设计哲学**——YouTube 子集的标签刻意捕捉"观众的主观印象"而非客观事实，这正契合 greenwashing 的本质：它操纵的是印象而非事实

## 局限与展望

- **YouTube 标注覆盖有限**：仅 386 个视频，训练/测试各半，对低资源标签（如 Patriotism、Green Innovation）统计可靠性不足
- **Facebook 标签为"远程标注"**：原始标注基于广告文本而非视频本身，视频与文本标签之间可能存在不对齐
- **仅英语**：虽然覆盖 20 国，但主要分析英语内容，未涉及非英语市场的本土化策略
- **帧采样的信息损失**：将完整视频压缩为 3-10 帧必然丢失大量时序和叙事信息，特别是 YouTube 长视频
- **未提供 fine-tuning 基线**：所有实验均为零样本/few-shot，未测试在训练集上微调的效果上界
- **"greenwashing" 的判定标准**：数据集检测的是框架（framing），从框架到 greenwashing 的推断需要额外的企业行为数据

## 相关工作与启发

- **vs Rowlands et al. (2024)**: 他们定义了 Facebook 文本广告的 framing 分类任务，本文将其扩展到多模态视频，并新增 YouTube 平台和全新标签体系
- **vs Holder et al. (2022)**: 他们定性分析了 O&G Facebook 广告的框架策略，本文将定性发现转化为定量可评估的 NLP/VLM 基准
- **vs 通用视频理解基准 (ActivityNet, Kinetics)**: 本基准关注的是"策略意图"而非"物理动作"——检测"这个视频想传达什么印象"比"视频中发生了什么"困难得多
- 论文展示了 VLM 在社会科学应用中的巨大潜力和短板，为 AI for Social Good 的文献增添了重要基准

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多模态 greenwashing 检测基准，双平台+印象型标注设计有新意
- 实验充分度: ⭐⭐⭐⭐ 6 款 VLM + 消融 + 错误分析，但缺少 fine-tuning 上界
- 写作质量: ⭐⭐⭐⭐ 数据集构建过程详尽透明
- 价值: ⭐⭐⭐⭐ 填补重要空白，开源数据和代码，对 AI + 气候政策研究有直接推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A3: Towards Advertising Aesthetic Assessment](../../CVPR2026/multimodal_vlm/a3_towards_advertising_aesthetic_assessment.md)
- [\[CVPR 2026\] MMSD3.0: A Multi-Image Benchmark for Real-World Multimodal Sarcasm Detection](../../CVPR2026/multimodal_vlm/mmsd30_a_multi-image_benchmark_for_real-world_multimodal_sarcasm_detection.md)
- [\[CVPR 2026\] VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models](../../CVPR2026/multimodal_vlm/vismem_latent_vision_memory_unlocks_potential_of_vision-language_models.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](../../ACL2025/multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [\[ICML 2025\] LEMoN: Label Error Detection using Multimodal Neighbors](../../ICML2025/multimodal_vlm/lemon_label_error_detection_using_multimodal_neighbors.md)

</div>

<!-- RELATED:END -->
