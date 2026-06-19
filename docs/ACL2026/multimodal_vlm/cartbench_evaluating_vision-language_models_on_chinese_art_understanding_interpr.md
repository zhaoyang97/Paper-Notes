---
title: >-
  [论文解读] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity
description: >-
  [ACL 2026][多模态VLM][中国艺术] 本文构建了 CArtBench——一个基于故宫博物院藏品的多任务基准，评估 VLM 在中国艺术理解中的四种能力（证据问答、结构化鉴赏、可辩护重解读、真伪辨别），发现即使最强模型在证据关联和风格-年代推理上也存在显著性能下降，而真伪辨别接近随机水平。 领域现状：VLM 越来越多…
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "中国艺术"
  - "博物馆基准"
  - "视觉语言模型"
  - "鉴赏能力"
  - "真伪辨别"
---

# CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity

**会议**: ACL 2026  
**arXiv**: [2604.11632](https://arxiv.org/abs/2604.11632)  
**代码**: [https://github.com/Big-Sid/CARTBENCH-Chinese-Artwork-Benchmark](https://github.com/Big-Sid/CARTBENCH-Chinese-Artwork-Benchmark)  
**领域**: 多模态VLM/文化理解  
**关键词**: 中国艺术, 博物馆基准, 视觉语言模型, 鉴赏能力, 真伪辨别

## 一句话总结

本文构建了 CArtBench——一个基于故宫博物院藏品的多任务基准，评估 VLM 在中国艺术理解中的四种能力（证据问答、结构化鉴赏、可辩护重解读、真伪辨别），发现即使最强模型在证据关联和风格-年代推理上也存在显著性能下降，而真伪辨别接近随机水平。

## 研究背景与动机

**领域现状**：VLM 越来越多地被用作通用多模态助手，但其评估主要由网络图像和西方中心概念主导。中文和文化聚焦的基准虽有扩展，但主要集中在短文本识别和问答上。

**现有痛点**：(1) 现有基准缺乏面向专家的解释能力评估——需要文化锚定和明确视觉证据支持的深度理解；(2) 中国艺术的许多视觉惯例是时代敏感的，策展级理解需要将可观察线索与历史背景联系；(3) 真伪判断是文化遗产的核心工作流程，但当前 VLM 在此方面的能力从未被评测。

**核心矛盾**：VLM 可能在短文本问答上表现良好，但其高准确率可能掩盖在证据关联、结构化鉴赏和鉴真等深层能力上的严重不足。

**本文目标**：构建一个统一的基准来全面评估 VLM 在中国艺术理解中的策展级能力。

**切入角度**：将故宫博物院藏品的 Wikidata 实体与权威图录页面对齐，构建跨多朝代、五大艺术类别的博物馆基准。

**核心 idea**：从短文本 QA 扩展到证据锚定问答、结构化鉴赏、可辩护解读和真伪辨别四个递进任务层次，揭示 VLM 在文化理解中的系统性失败模式。

## 方法详解

### 整体框架

CArtBench 通过三阶段管道构建：(1) 从 Wikidata 检索故宫博物院的图像藏品；(2) 将藏品与官方图录描述对齐；(3) 专家指导下的筛选和分类。基于构建的数据，实例化四个互补任务。

### 关键设计

**1. CuratorQA（策展级问答）：用难度分层 + 题型分类，把"高准确率"拆开看清模型到底卡在哪**

通用 VLM 在中文艺术上的总分往往不低，但单看一个总准确率根本看不出它是真懂还是只会认表面图案。CuratorQA 因此把 14,421 个问题（覆盖 1,589 件艺术品）切成两个轴：难度上分 P1（只需视觉证据即可作答）和 P2（必须结合艺术知识才能推理），题型上分主题识别、场景分类、构图格式、技法风格、图像学检测、风格-年代推理共 6 类。

这样的二维切分让每个失败都能被精确定位——是缺视觉证据关联（P1 题失分），还是缺文化背景推理（P2、风格-年代题失分）。问答对由 GPT-5.2 生成、再经专家审核，抽查 1000 条错误率仅 0.47%，保证了规模化生成下的标注可信度。

> ⚠️ 原文标注的生成模型名为 GPT-5.2，以原文为准。

**2. CatalogCaption（结构化鉴赏）：用四段式长文本，逼出 QA 测不到的综合能力**

证据问答只检验"点状"识别，但策展级理解更体现在能否把视觉观察、技法、历史与审美串成一段连贯的鉴赏文字。CatalogCaption 因此挑出 86 件艺术品，要求模型生成包含基本信息、技法分析、历史背景、美学评价四段的结构化鉴赏文本，再与故宫权威图录描述对比打分。

长文本生成比选择题难得多——它要求模型同时调动视觉理解和文化知识、并组织成专家可接受的表达，因此能暴露出"短 QA 高分却写不出像样鉴赏"这类被总准确率掩盖的短板。

**3. ReInterpret（可辩护重解读）：用经典作品当锚点，测模型能否"出新而不离谱"**

证据问答和鉴赏检验的还是"复述既有认知"，而 ReInterpret 要测更难的一步——模型能否在尊重画面与文化背景的前提下给出超越常规的新解读。它选取 25 件常用于艺术教学和鉴赏训练的中国经典作品作为锚点：这些作品已有大量成熟讨论和约定俗成的解读方式，正好用来测模型能否跳出标准叙事、又不脱离图像证据。

评测借鉴托兰斯创造性思维测验（TTCT）设计了两阶段问卷：第一阶段是合理性闸门（plausibility gate），过滤掉严重误读、违背艺术史共识或事实编造的输出；第二阶段对通过闸门的输出在解读新颖性、整合连贯性、证据推理、阐述表现力、创造性洞察五个维度上做 1–5 分人工评分。实验发现真正的瓶颈不在表达质量而在"可辩护性"——模型主要靠更稳定地通过第一阶段闸门来拉高得分，而非在第二阶段的解读质量上拉开差距。

**4. ConnoisseurPairs（真伪辨别）：用视觉极相似的真伪对，做一次诊断性压力测试**

真伪判断是文化遗产工作的核心环节，也是最考验"超越表面识别"的深层推理，但此前从未被纳入 VLM 评测。ConnoisseurPairs 构造了 10 对视觉上高度相似的真品-仿品对，要求模型基于整体一致性和细微线索判断哪件为真。

这一任务规模虽小，定位却是诊断性压力测试：它直接探测模型能否像鉴赏家那样从笔触、构图、材质等微弱信号中推断真伪。实验中所有模型在此都接近随机水平，恰好暴露了当前 VLM 在深层视觉推理上的盲区。

### 损失函数 / 训练策略

不涉及模型训练。评估使用统一协议结合自动指标、格式合规检查和专家评分。

## 实验关键数据

### 主实验

**CuratorQA 总体准确率（9 种 VLM）**

| 模型 | 总体准确率 | QA6(风格-年代推理) |
|------|----------|------------------|
| Qwen3-VL-235B | 0.84 | 0.56 |
| Qwen3-VL-30B | 0.80 | 0.42 |
| Qwen2.5-VL-72B | 0.81 | 0.53 |
| Qwen2.5-VL-32B | 0.80 | 0.53 |

### 消融实验

- 高整体准确率掩盖了在证据关联（QA5）和风格-年代推理（QA6）上的显著性能下降
- 长文本鉴赏（CatalogCaption）远未达到专家参考水平
- 真伪辨别（ConnoisseurPairs）在所有模型上接近随机水平，凸显鉴赏家级推理的极端困难

### 关键发现

- VLM 在短文本识别上的高分可能掩盖在证据关联和文化推理上的严重不足
- 风格-年代推理是最困难的子任务，最强模型也仅达到 56%
- 真伪辨别接近随机表现，说明当前 VLM 缺乏鉴赏家级别的视觉推理能力
- 不同艺术类别间存在显著性能差异

## 亮点与洞察

- 首个博物馆级中国艺术 VLM 基准，跨越识别、鉴赏、解读和鉴真四个层次
- 与故宫博物院权威图录对齐确保了数据的权威性
- 真伪辨别任务设计独特，直指 VLM 深层推理的盲区
- 评估协议设计严谨，结合自动指标和专家评分

## 局限与展望

- ReInterpret 和 ConnoisseurPairs 规模较小（25/10），属于诊断性评估
- 数据主要来自故宫博物院，可能存在收藏偏差
- 真伪辨别任务的专家标注成本极高，难以大规模扩展
- 未来可扩展到更多博物馆和更多艺术传统

## 相关工作与启发

- 与 CVLUE、CulturalVQA 等文化感知基准互补，但深入到专家级评估层次
- 与 ArtEmis（情感）、MuseumQA（事实）形成任务互补
- 为文化遗产领域的 AI 应用提供了更严格的评测标准

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个涵盖真伪辨别的博物馆级中国艺术 VLM 基准
- 实验充分度: ⭐⭐⭐⭐ 9 种 VLM、四种任务的全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务设计动机充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](../../ACL2025/multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](cross-modal_taxonomic_generalization_in_vision-_language_models.md)

</div>

<!-- RELATED:END -->
