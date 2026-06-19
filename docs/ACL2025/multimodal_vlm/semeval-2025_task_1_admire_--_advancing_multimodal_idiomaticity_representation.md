---
title: >-
  [论文解读] SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation
description: >-
  [ACL 2025 (SemEval Workshop)][多模态VLM][习语理解] 设计了 SemEval-2025 AdMIRe 共享任务——通过图像排序和图像序列补全两个子任务，在多模态（文本+图像）和多语言（英语+巴西葡萄牙语）场景下评估模型对习语表达的理解能力，最佳系统通过混合专家和多查询平滑策略达到了接近人类水平的表现。
tags:
  - "ACL 2025 (SemEval Workshop)"
  - "多模态VLM"
  - "习语理解"
  - "多模态"
  - "视觉-语言模型"
  - "共享任务"
  - "混合专家"
---

# SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation

**会议**: ACL 2025 (SemEval Workshop)  
**arXiv**: [2503.15358](https://arxiv.org/abs/2503.15358)  
**代码**: 有 (数据集: doi.org/10.15131/shef.data.28436600.v1, CC-BY-4.0)  
**领域**: Multimodal/VLM  
**关键词**: 习语理解, 多模态, 视觉-语言模型, 共享任务, 混合专家

## 一句话总结

设计了 SemEval-2025 AdMIRe 共享任务——通过图像排序和图像序列补全两个子任务，在多模态（文本+图像）和多语言（英语+巴西葡萄牙语）场景下评估模型对习语表达的理解能力，最佳系统通过混合专家和多查询平滑策略达到了接近人类水平的表现。

## 研究背景与动机

习语（Idiomatic Expressions）是自然语言中的一大难题——其含义无法从组成词的字面意义直接推导，例如 "eager beaver" 不是指"热情的海狸"，而是描述一个积极热心的人。尽管大语言模型在通用基准上表现出色，但在处理隐喻性语言时仍然不稳定。

已有的习语理解数据集（如 NCTTI、FLUTE、MAGPIE）多集中在纯文本领域，但研究表明这些任务可能并不真正要求模型拥有良好的习语语义表示。视觉模态的引入可以更严格地测试模型是否真正"理解"了习语的含义——模型需要区分习语的字面含义（literal）和比喻含义（figurative）对应的不同图像场景。

此外，习语的错误处理会在实际应用中造成严重后果，例如机器翻译中以色列总理因习语翻译错误而"称赞"Eurovision 获胜者为"真正的牛"。

## 方法详解

### 整体框架

AdMIRe 构建了两个子任务：
- **子任务 A（静态图像选择）**: 给定包含潜在习语名词复合词的上下文句子和 5 张图像，按照图像与句中习语/字面含义的匹配程度排序
- **子任务 B（图像序列补全）**: 给定表示习语字面或比喻含义的前 2 张序列图像和 4 张候选图，选择正确的第 3 张补全图像，同时判断表达的是习语还是字面含义

### 关键设计

1. **目标复合词选取**: 从 NCTTI、FLUTE、MAGPIE 等现有数据集筛选名词复合词，要求具备双义性（duality）——既有合理可想象的字面含义，也有比喻含义。如 "silver bullet" 既可以是银色子弹，也可以是万能解决方案。排除了纯组合性表达（如 olive oil）和难以视觉化的习语（如 kangaroo court）。

2. **五级图像设计（子任务 A）**: 每个表达生成 5 张图像，分别对应：强比喻、弱比喻、弱字面、强字面、干扰项。标注员为每种场景撰写视觉描述，然后用 Midjourney 生成统一风格的卡通图像，配合统一的 style reference 确保视觉一致性。

3. **三帧叙事序列（子任务 B）**: 类似三格漫画的形式，标注员描述 3 个视觉场景，加上 2 个替代结尾，测试模型对习语随时间展开语义的理解。

4. **多语言支持**: 涵盖英语（100 个表达）和巴西葡萄牙语（55 个表达）。葡语中特意去掉了连字符（区分字面与习语的形式标记）以避免给模型提供捷径。

5. **文本替代模态**: 用 LLaVA 生成图像描述文本，使纯文本模型也能参赛，降低参与门槛。

### 评估指标

- 子任务 A: Top-1 准确率 + DCG（折损累积增益），排名权重 [3,1,0,0,0]
- 子任务 B: 图像补全准确率 + 句子类型（习语/字面）识别准确率

## 实验关键数据

### 子任务 A 排行榜（英语，文本+图像）

| 排名 | 团队 | 测试集 Top-1 Acc | 测试集 DCG | 扩展集 Top-1 Acc | 扩展集 DCG |
|------|------|-----------------|-----------|-----------------|-----------|
| 1 | PALI-NLP | **0.93** | 3.52 | **0.83** | 3.43 |
| 2 | dutir914 | 0.93 | 3.46 | 0.79 | 3.28 |
| 3 | AlexUNLP-NB | 0.93 | 3.45 | 0.72 | 3.22 |
| 4 | AIMA | 0.87 | 3.44 | 0.48 | 2.90 |
| 5 | daalft | 0.87 | 3.43 | 0.81 | 3.35 |

### 人类评估 vs 系统表现（扩展评估集）

| 评估方 | Top-1 Acc | DCG |
|--------|-----------|------|
| 标注员平均 | 0.71 | 3.22 |
| 最佳个体 | 0.86 | 3.41 |
| 专家池（平均排名聚合） | 0.83 | 3.39 |
| PALI-NLP（最佳系统） | **0.83** | **3.43** |

### 关键发现

1. **最佳系统达到人类水平**: PALI-NLP 在扩展评估集上的 top-1 准确率（0.83）与专家池方法（0.83）持平，DCG（3.43）甚至超过专家池（3.39）。

2. **混合专家是关键策略**: 4 支团队采用混合专家方法，通过多种模型/提示变体平滑不同模型在习语理解上的不一致性——没有单一模型能完整掌握习语现象。

3. **LLM 的习语偏向性**: 3 支使用生成式 LLM 的团队发现模型倾向于将所有表达都判定为习语用法，PALI-NLP 通过"先让 LLM 生成字面用法示例"的策略将分类准确率从 91.4% 提升至 98.6%。

4. **AlexUNLP 的同义词替换策略**: 将检测为习语的复合词替换为组合性同义词（dirty money → illegal money），绕过 VLM 偏好字面解读的倾向。

5. **扩展评估集更稳健**: 测试集上表现好的模型在扩展集上可能大幅下滑，暗示存在过拟合风险。

6. **葡语表现出乎意料**: 英语和葡语任务之间的差距比预期小，表明多语言 LLM 对非英语图形化语言的理解在改善。

## 亮点与洞察

- **任务设计的精巧性**: 五级图像（从强比喻到干扰项）的粒度使评估不仅检测"是否正确"，还能衡量对语义距离的细粒度理解。
- **视觉模态的独特价值**: 图像要求模型真正理解习语的语义，而不仅仅做表面文本匹配——这比纯文本基准更能暴露模型的理解缺陷。
- **成本敏感的设计**: 提供文本描述作为图像替代，使资源有限的团队也能参加。
- **人类评估基线的设置**: 任务对人类也非简单（平均标注员 Top-1 仅 71%），表明这确实是一个有挑战性的研究问题。

## 局限与展望

- 子任务 B 数据量偏小（英语仅 30 个表达），吸引的参赛团队也较少
- 使用 Midjourney 生成图像可能引入生成模型自身的偏差
- 五级排序的"期望排序"（如习语条件下的排序）有一定主观性——字面图像和干扰图像的相对顺序是否确定？
- Google Translate 生成的葡语翻译质量可能影响多语言评估的公平性
- 图像描述由 LLaVA 自动生成，可能丢失图像中的关键视觉线索

## 相关工作与启发

- 多模态习语处理是 NLU 领域一个重要但被低估的研究方向
- "先生成字面用法再分类"的 debiasing 策略对其他存在偏向性的生成任务也有参考价值
- 图像序列（时序模态）的引入为习语理解提供了新维度，可延伸到视频模态

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 实验充分度 | 4 |
| 写作质量 | 4.5 |
| 价值 | 4 |

作为共享任务概述论文，在数据集设计、评估指标设计和对参赛系统的分析方面都做得相当细致。对习语现象的多模态建模具有重要的研究意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](../../NeurIPS2025/multimodal_vlm/t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)
- [\[ACL 2025\] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](chartcoder_chart_to_code.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](../../CVPR2025/multimodal_vlm/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[ACL 2025\] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models](coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)

</div>

<!-- RELATED:END -->
