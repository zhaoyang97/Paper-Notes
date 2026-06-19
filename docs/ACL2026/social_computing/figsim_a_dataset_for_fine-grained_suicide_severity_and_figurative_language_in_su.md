---
title: >-
  [论文解读] FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes
description: >-
  [ACL2026 Findings][社会计算][suicide memes] FigSIM 构建了首个面向自杀相关 meme 的细粒度多模态数据集，标注 figurative phenomenon、自杀严重度和自杀相关内容，并用 16 类模型验证了当前模型在隐喻、讽刺和高严重度风险识别上仍会系统性低估。
tags:
  - "ACL2026 Findings"
  - "社会计算"
  - "suicide memes"
  - "细粒度风险标注"
  - "具象语言"
  - "多模态审核"
  - "心理健康安全"
---

# FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes

**会议**: ACL2026 Findings  
**arXiv**: [2606.02523](https://arxiv.org/abs/2606.02523)  
**代码**: https://github.com/LiuliuChen/FigSIM  
**领域**: 社交计算 / 安全与心理健康 NLP  
**关键词**: suicide memes, 细粒度风险标注, 具象语言, 多模态审核, 心理健康安全  

## 一句话总结
FigSIM 构建了首个面向自杀相关 meme 的细粒度多模态数据集，标注 figurative phenomenon、自杀严重度和自杀相关内容，并用 16 类模型验证了当前模型在隐喻、讽刺和高严重度风险识别上仍会系统性低估。

## 研究背景与动机
**领域现状**：社交媒体上的心理健康研究长期关注文本帖子中的抑郁、焦虑和自杀风险，也有一些工作开始处理图像和 meme。平台内容审核通常把风险内容看成二分类或粗粒度安全类别，主流模型包括文本分类器、视觉模型、多模态模型和大模型审核接口。

**现有痛点**：自杀相关 meme 很特殊：它们既可能是求助、共情或 coping，也可能包含会让人不适或具有潜在伤害的内容。meme 的幽默、反讽、隐喻和模板化视觉元素会遮蔽真实意图，使人工和自动系统都难以判断严重度。现有数据集几乎没有针对 suicide memes 的细粒度标注，也缺乏专门评测模型能否理解“间接表达的风险”。

**核心矛盾**：内容审核需要保护用户，尤其是年轻用户免受潜在有害内容影响；但自杀相关表达也可能是求助和同伴支持，粗暴删除或误判又可能伤害表达者。要做更细的 moderation，必须先知道内容表达的严重度、是否含 figurative language、是否包含潜在有害或保护性因素。

**本文目标**：作者希望建立一个可以支持研究和审核策略设计的基准数据集，包含 1049 条自杀相关 meme，并围绕三个任务评测模型：figurative language detection、suicide severity detection、suicide-related content detection。

**切入角度**：论文把心理健康安全问题和 meme understanding 结合起来，不只标一个“有害/无害”，而是设计了 5 个标注维度：figurative phenomenon、自杀严重度、自杀相关内容、modality、context。严重度标注参考 C-SSRS，内容标注参考 #chatsafe 和 Mindframe 等在线自杀传播指南。

**核心 idea**：要让模型真正服务于自杀相关 meme 审核，必须同时建模细粒度严重度、隐喻/讽刺等间接表达，以及视觉文本互补带来的上下文依赖。

## 方法详解
FigSIM 是数据集与 benchmark 论文。它的方法核心不是提出新模型，而是从数据收集、标注体系、专家校准、模型基线和错误分析几个环节建立一个可复用评测框架。

### 整体框架
数据来自 r/SuicideMeme，这是一个专门分享自杀相关 meme 的 subreddit。作者用 Pushshift Reddit API 收集 2018 年 4 月到 2022 年 12 月的 submissions，抽取图片 URL 并下载图片。过滤规则包括：OCR 可读且文本为英语、去除近重复 meme、保留纯视觉 meme。过滤后得到 1967 张图片，由于标注资源限制，随机抽样 1050 张进入标注，最终 1 张因质量控制被移除，得到 1049 条数据。

标注方案分五类：figurative phenomenon 是多标签，包括 metaphor、pun/double meaning、irony/sarcasm 和 none；suicide severity 是有序单标签，包括 wish to be dead、suicide ideation、suicide planning、suicide attempts、suicide death 和 none；suicide-related content 是多标签，包括 stylized method/location depiction、naturalistic depiction、protective factors、harmful factors、third-person description 和 none；modality 判断文本、图像或互补；context 判断是否需要外部背景知识。

模型评测覆盖文本、图像、多模态和专门 mental-health meme 模型。监督模型用 60:20:20 stratified split，训练集 633，验证集 208，测试集 208。单标签任务用 CrossEntropyLoss，多标签任务用 BCEWithLogitsLoss，并针对不平衡使用 class-reweighted losses 和阈值调节。MLLM 用 zero-shot / few-shot prompting。

### 关键设计

**1. 临床量表与在线传播指南结合的标注体系：把 meme 风险从"是否自杀相关"细化为可操作的多维标签**

把自杀相关内容压缩成二分类会丢掉最关键的信息——同样涉及自杀的 meme，是求助还是诱导、严重度是轻是重，需要的审核动作完全不同。FigSIM 因此让严重度标签借鉴临床量表 C-SSRS（wish to be dead、suicide ideation、suicide planning、suicide attempts，并额外加入 suicide death），让自杀相关内容标签借鉴在线自杀传播指南 #chatsafe 和 Mindframe，区分 stylized method/location depiction、naturalistic depiction、protective factors、harmful factors 和 third-person description。这套细粒度标签把"内容的影响取决于表达方式、严重度和上下文"这一判断显式编码进数据，让后续研究能定位哪类内容更需要分级审核或干预。

**2. 心理健康专家参与的多轮标注流程：用校准和裁决把高主观任务的标注稳住**

figurative expression 和 suicide severity 都高度主观，直接交给无校准众包会得到噪声很大的标签，对自杀这类敏感主题也不符合伦理。本文因此设计了多轮流程：先做两轮 pilot annotation（每轮 50 张），由注册心理学家和研究者参与；再用一个 calibration batch 检查标注者之间的一致性；进入大规模标注后由两名标注者独立标注、第三名标注者对分歧做 adjudication。专家的介入不是为了"标得快"，而是为了把主观判断约束在清晰的 guideline 之内，提升概念清晰度和伦理稳健性。

**3. 多类型模型基线与偏差分析：用错误结构而非平均分来评估安全可靠性**

只报平均 F1 会掩盖审核场景里真正致命的风险——高严重度内容被低估，远比低严重度被误报更危险。本文因此横向评测文本（BERT/MentalBERT/RoBERTa）、图像（ResNet/ViT/DINOv2）、多模态（CLIP/BLIP-2）、多种 MLLM，以及 Yadav et al.、M3H 等任务邻近模型，随后不止看总分，而是拆解 per-label F1、严重度的 over/underprediction、context-required 错误和 modality 相关错误。正是这层偏差分析，才让"模型在隐喻、讽刺下系统性低估高严重度"这一安全靶点暴露出来。

### 损失函数 / 训练策略
监督基线中，suicide severity 和 modality/context 等单标签分类使用 CrossEntropyLoss；figurative phenomenon 和 suicide-related content 这类多标签任务使用 BCEWithLogitsLoss。为了处理类别不平衡，作者使用 class-reweighted losses，并为多标签任务调节 decision thresholds。数据按 60:20:20 划分为 633/208/208，保持各标注维度分布。MLLM 不做微调，使用基于标签定义和 guideline 的 zero-shot / few-shot prompt；few-shot 每个任务随机选 6 个覆盖所有标签的例子。

## 实验关键数据

### 主实验

| 任务 | 最佳模型 | Macro-F1 | Weighted-F1 | 观察 |
|--------|------|------|----------|------|
| Figurative detection | Claude-sonnet-4-5 zero-shot | 70.21±0.82 | 80.69±0.42 | irony/sarcasm 最好识别，pun 和 metaphor 更难 |
| Suicide severity detection | Gemini-3-pro few-shot | 71.60±2.43 | 71.73±2.52 | few-shot 对细粒度严重度帮助更明显 |
| Suicide-related content detection | Gemini-3-pro zero-shot | 58.51±4.06 | 58.16±2.53 | 三个任务中最难，harmful factors 与 naturalistic depiction F1 较低 |
| CLIP | 三任务 Macro-F1 | 62.06 / 49.97 / 33.70 | 80.21 / 52.10 / 43.75 | 对 figurative 有一定能力，但安全语义不足 |
| M3H | Suicide severity Macro-F1 | 61.47±1.74 | 62.58±1.95 | mental-health meme 迁移有效，但比最佳 MLLM 低约 10 点 |

### 数据集与标注质量

| 维度 | 指标 | 数值 | 说明 |
|------|---------|------|------|
| 数据规模 | 最终样本数 | 1049 | 从 1050 标注样本中移除 1 个质控失败样本 |
| 划分 | Train / Val / Test | 633 / 208 / 208 | stratified split |
| IAA: Suicide Severity | Cohen's κ | 0.65 | Substantial |
| IAA: Modality | Cohen's κ | 0.88 | Almost perfect |
| IAA: Context | Cohen's κ | 0.58 | Moderate |
| IAA: Figurative Phenomenon | 平均 Cohen's κ | 0.56 | Moderate，反映隐喻/讽刺主观性 |
| IAA: Suicide-related Content | 平均 Cohen's κ | 0.63 | Substantial |

### 错误分析与审核发现

| 分析项 | 数字 | 结论 |
|------|------|------|
| Azure 对 None 的 suicide/self-harm flag rate | 0.304 | 审核模型会对低严重度或无严重度内容误报 |
| Azure 对 Suicide attempts 的 suicide/self-harm flag rate | 0.661 | 总体随严重度升高而更敏感 |
| OpenAI 对 Suicide attempts 的 suicide/self-harm flag rate | 0.575 | 同样随严重度上升，但敏感性低于 Azure |
| Figurative irony/sarcasm 下严重度低估率 | absent 0.429, present 0.604 | 有讽刺时更容易低估严重度 |
| Figurative metaphor 下严重度低估率 | absent 0.551, present 0.833 | 隐喻样本少，但低估趋势最强 |
| Context-required 错误 | severity 5/26, figurative 16/26, content 12/26 | 需要外部背景时模型明显更难 |

### 关键发现
- 文本模型在 suicide severity 和 suicide-related content 上通常强于 image-only 模型，说明 meme 中的 OCR 文本承载了大量风险信息。
- MLLM 在三个任务上整体最强，但开源 MLLM 和部分安全过滤机制可能导致拒答或性能不稳。
- 少样本提示并不总是有效：figurative detection 中 few-shot 可能因 prompt sensitivity 和主观性导致下降，但 suicide severity 更受益于示例。
- 最关键的安全问题是高严重度 underprediction，尤其当严重度通过隐喻、讽刺或视觉双关表达时。

## 亮点与洞察
- **数据集问题定义得很准**：论文没有把 suicide meme 简化成“自杀/非自杀”，而是把严重度、具象语言和内容类型分开，这更贴近内容审核中的实际决策。
- **标注过程对敏感主题足够谨慎**：心理学家、青年自杀预防研究者、pilot rounds、adjudication 和伦理审批共同减少了粗糙标注风险。
- **错误分析比平均分更有价值**：模型低估高严重度内容是平台安全中的核心风险，论文把这个偏差和 figurative language 关联起来，给后续模型设计提供了明确靶点。
- **moderation 分析揭示现有 API 的盲区**：审核模型整体会随严重度提高而更容易 flag，但同一严重度下 figurative memes 往往分数更低，说明隐晦表达可能绕开安全系统。

## 局限与展望
- 数据来自单一 subreddit，社区文化、幽默规范和表达方式不一定能泛化到 TikTok、X、Instagram、论坛或非英语社区。
- 标注只基于 meme 内容本身，假设除非明确第三人称，否则内容可能反映间接自我表达；这不是临床判断，也不能推断创作者真实心理状态。
- “fine-grained” 是相对既有二分类而言，标签仍然不能覆盖所有自杀相关表达的细微差别。
- figurative phenomenon 和 context 的 κ 只有 moderate，说明任务本身主观性强。后续可加入解释标注、更多专家 adjudication 和跨文化标注者比较。
- 数据发布只提供 annotations 和 retrieval script，这有利于隐私和平台政策合规，但也可能带来复现实验时图片消失、URL 失效等问题。

## 相关工作与启发
- **vs Hateful Memes / toxic meme benchmarks**: 这些数据集关注仇恨或攻击性内容，通常不处理自杀风险的临床严重度；FigSIM 的标签体系更适合心理健康安全。
- **vs 文本自杀风险检测**: 文本模型无法处理 meme 的视觉模板、反讽和图文互补；FigSIM 证明这类内容需要 multimodal 和 context-aware 理解。
- **vs mental-health meme 模型 M3H**: M3H 在 suicide severity 上有迁移能力，但仍与最佳模型有约 10 点差距，说明抑郁/焦虑 meme 的知识不能直接覆盖 suicide-specific 风险。
- **启发**：内容审核模型需要显式训练“间接表达到严重度”的映射，而不只是识别是否出现自杀相关词。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个细粒度 suicide meme 数据集，任务定义和标签体系都很有领域针对性。
- 实验充分度: ⭐⭐⭐⭐☆ 基线覆盖广、错误分析扎实；但数据源单一，跨平台泛化仍待验证。
- 写作质量: ⭐⭐⭐⭐☆ 结构清楚，伦理和局限写得克制；部分模型名称来自未来系统，读者需要注意时间语境。
- 价值: ⭐⭐⭐⭐⭐ 对社交平台安全、心理健康 NLP 和多模态审核都很有基准价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)
- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[ACL 2026\] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)
- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)
- [\[CVPR 2025\] Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness](../../CVPR2025/social_computing/project-probe-aggregate_efficient_fine-tuning_for_group_robustness.md)

</div>

<!-- RELATED:END -->
