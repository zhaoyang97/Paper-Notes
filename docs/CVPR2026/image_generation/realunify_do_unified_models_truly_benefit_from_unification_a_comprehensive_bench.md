---
title: >-
  [论文解读] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark
description: >-
  [CVPR 2026][图像生成][统一模型] 本文提出 RealUnify，首个专门评估统一模型中理解与生成能力双向协同效果的基准，通过1000个人工标注实例和直接/分步双重评估协议，揭示了当前统一模型虽然具备理解和生成能力，但在端到端场景中仍无法实现真正的能力协同。 1. 领域现状：多模态统一模型（如 BAGEL、Jan…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "统一模型"
  - "能力协同"
  - "理解与生成"
  - "基准测试"
  - "双向评估"
---

# RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark

**会议**: CVPR 2026  
**arXiv**: [2509.24897](https://arxiv.org/abs/2509.24897)  
**代码**: [https://github.com/FrankYang-17/RealUnify](https://github.com/FrankYang-17/RealUnify)  
**领域**: 图像生成  
**关键词**: 统一模型, 能力协同, 理解与生成, 基准测试, 双向评估

## 一句话总结
本文提出 RealUnify，首个专门评估统一模型中理解与生成能力双向协同效果的基准，通过1000个人工标注实例和直接/分步双重评估协议，揭示了当前统一模型虽然具备理解和生成能力，但在端到端场景中仍无法实现真正的能力协同。

## 研究背景与动机
1. **领域现状**：多模态统一模型（如 BAGEL、Janus-Pro 等）将视觉理解（VQA）和视觉生成（T2I）集成到单一架构中，已成为通向通用AI的重要方向。
2. **现有痛点**：现有评估框架（如 MME-Unify、UniEval）主要将理解和生成分开评估，或仅简单组合两类任务，无法判断统一模型是否真正实现了"1+1>2"的协同效果。
3. **核心矛盾**：统一模型的最大价值在于理解和生成的双向增益——用理解指导生成、用生成辅助理解。但目前缺乏严格的基准来验证这种双向协同是否真实存在。
4. **本文目标** 设计一套能精确衡量统一模型能力协同程度的基准，回答"统一是否真的带来了比单独能力更强的表现"。
5. **切入角度**：将任务设计为必须依赖理解-生成协同才能完成的形式，并通过分步评估协议诊断瓶颈来源。
6. **核心 idea**：通过精心设计的双向协同任务和直接/分步双重评估协议，首次系统性地检验统一模型是否实现了理解与生成的真正协同。

## 方法详解

### 整体框架
RealUnify 包含1000个人工标注实例，覆盖10个类别32个子任务。核心设计围绕两条轴线展开：**理解增强生成（UEG）**——需要推理（常识、逻辑等）来指导图像生成；**生成增强理解（GEU）**——需要心理模拟或重建来解决推理任务。评估采用直接评估和分步评估两种协议。

### 关键设计

**1. 理解增强生成（UEG）：逼模型先「想明白」再画**

普通 T2I 基准（如美学评分、文本对齐）只看图画得好不好看、和 prompt 搭不搭，却测不出模型有没有把知识和推理用进生成里。UEG 这条轴专门设计需要"先理解后生成"的任务，并切成 6 类来覆盖不同的理解类型：世界知识（生成需要客观事实的图像）、常识推理（生成符合日常现象的图像）、数学推理（先算对再生成正确结果）、逻辑推理（满足逻辑约束的生成）、科学推理（套用物理/化学/生物原理）、代码到图像（解析代码逻辑后生成对应画面）。共同点是——光会画图不够，必须先调动理解能力把答案"想对"，画错就说明协同断在了"理解→生成"这一步。

**2. 生成增强理解（GEU）：逼模型「用图像思考」来答题**

反过来这条轴测的是另一个方向：模型能不能借助生成能力来帮自己理解。它包含 4 类任务——心理重建（把打乱的图像块在脑内拼回去再回答）、心理追踪（追踪颜色线段经多步变换后的最终状态）、注意力聚焦（用生成手段高亮关键区域来辅助识别）、认知导航（迷宫/地图寻路，需要生成中间可视化才好推理）。这些题如果只靠语言链式推理往往很难，理想的统一模型应该能"画一步、看一步"地把中间状态显式生成出来。GEU 就是要看模型是真的在用生成辅助理解，还是只是绕过生成、硬用语言推理蒙答案。

**3. 双重评估协议（Direct + Stepwise）：把「没能力」和「有能力但不会整合」分开**

只看端到端结果有个致命模糊：模型做不好，到底是基础能力（理解或生成）本身就弱，还是两种能力都在、只是没法自发拼到一起？为此每道题都跑两套协议。直接评估（Direct）要求一气呵成端到端完成；分步评估（Stepwise）把任务拆成独立的理解段和生成段喂给模型——UEG 先让它理解、再让它生成，GEU 先让它生成、再让它理解。对比两者就能定位瓶颈：如果 Stepwise 明显高于 Direct，说明能力都在、卡在整合；如果 Stepwise 也上不去，才是基础能力的锅。这套设计是 RealUnify 区别于以往基准的核心——它不止给分，还给诊断。

**4. 投票式生成评估（Polling Evaluation）：用问答验证替代美学打分**

UEG 生成的图正确与否，靠 FID/CLIP 这类美学/相似度指标判不出来——"图里的算式结果对不对""物理现象画没画对"是内容层面的事。RealUnify 为每个生成目标预先列一组验证问题，再让 Gemini 2.5 Pro 作为评判模型对生成图像逐题打分投票，以内容是否命中目标来判对错。相比传统指标，这种基于问答的自动验证更贴合知识密集型生成的评估需求。

### 损失函数 / 训练策略
本文是基准测试工作，不涉及模型训练。数据构建方面：UEG 任务由10位人类专家手动设计，经3位评审交叉验证；GEU 任务部分自动生成后由专家标注。Gemini 2.5 Pro 作为评判模型的可靠性通过与人类专家评分的一致性验证。

## 实验关键数据

### 主实验

| 模型 | UEG Direct | UEG Step | GEU Direct | GEU Step | 总分 |
|------|-----------|----------|-----------|----------|------|
| Nano Banana (闭源) | 63.0 | - | 31.8 | - | 50.5 |
| BAGEL (开源最佳) | 32.7 | 47.7 | 39.3 | 35.8 | 35.3/42.9 |
| UniPic2 | 37.5 | 40.5 | 24.0 | 23.8 | 32.1/33.8 |
| OneCAT | 37.5 | 39.0 | 31.3 | 29.3 | 35.0/35.1 |
| Oracle (Gemini+GPT-Image) | - | 72.7 | - | 31.8 | - |

### 消融实验

| 评估方式 | BAGEL UEG | BAGEL GEU | 说明 |
|---------|----------|----------|------|
| Direct | 32.7 | 39.3 | 端到端，无法自发整合 |
| Stepwise | 47.7 | 35.8 | UEG显著提升，GEU反而下降 |
| Oracle (GT中间结果) | 更高 | 更高 | 说明基础能力存在但整合能力不足 |

### 关键发现
- **UEG 分步评估大幅提升**：BAGEL 从 32.7% 提升到 47.7%，说明模型内部有知识但无法自发整合到生成中。
- **GEU 分步评估反而下降**：分解后性能降低，说明模型在直接评估中依赖理解捷径，而非真正利用生成能力。
- **开源 vs 闭源差距巨大**：UEG 上开源最佳 37.5% vs 闭源 63.0%；但 GEU 上开源模型（BAGEL 39.3%）反而超过闭源（31.8%）。
- **Oracle上界远未达到**：组合专家模型在 UEG 达 72.7%，当前最佳统一模型仅 47.7%（分步），差距巨大。

## 亮点与洞察
- **分步评估揭示了"有但不会用"的现象**：这是最核心的发现——模型有理解能力，也有生成能力，但无法在端到端场景中自发整合。这种诊断性评估设计可以迁移到其他需要多能力协同的AI系统评估中。
- **GEU 任务的"理解捷径"发现**：模型在需要"先生成再理解"的任务上，实际上绕过了生成直接用理解回答，分步强制生成后反而表现更差。这揭示了当前模型对生成能力的利用严重不足。
- **投票评估机制**：用问题列表 + LLM 评判来验证生成图像正确性，比传统 FID/CLIP 更适合知识密集型生成评估。

## 局限与展望
- 评估依赖 Gemini 2.5 Pro 作为裁判模型，存在评估偏见风险（虽然与人类评分有一定一致性）。
- 仅包含1000个实例，某些子任务样本量可能不足以支撑统计显著性。
- 缺乏对提升协同能力的训练方法探索——只诊断了问题但没提出解决方案。
- 未来可探索特定的训练策略（如交替训练、协同奖励）来促进真正的能力融合。

## 相关工作与启发
- **vs MME-Unify**: 后者同时评估理解和生成，但不测试两者的协同；RealUnify 专门设计需要协同才能完成的任务。
- **vs T2I-CoReBench/WISE**: 这些基准初步探索了理解对生成的帮助，但不系统、不双向，且缺乏分步诊断。
- **vs 专家模型组合**：Oracle 实验表明，简单组合最佳专家模型（Gemini + GPT-Image）就能达到 72.7%，远超任何统一模型，暗示统一架构本身并非核心——训练策略和归纳偏置才是关键。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统性评估统一模型能力协同的基准，分步评估协议设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 12个统一模型+6个专家基线，双重评估协议，评判可靠性验证
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，结论有说服力
- 价值: ⭐⭐⭐⭐ 为统一模型研究指明了"真正需要优化什么"的方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[CVPR 2026\] Do Less, Achieve More: Do We Need Every-Step Optimization for RL Fine-tuning of Diffusion Models?](do_less_achieve_more_do_we_need_every-step_optimization_for_rl_fine-tuning_of_di.md)
- [\[CVPR 2026\] UniGen-1.5: Enhancing Image Generation and Editing through Reward Unification in RL](unigen-15_enhancing_image_generation_and_editing_through_reward_unification_in_r.md)
- [\[CVPR 2026\] UnicEdit-10M: A Dataset and Benchmark Breaking the Scale-Quality Barrier via Unified Verification for Reasoning-Enriched Edits](unicedit-10m_a_dataset_and_benchmark_breaking_the_scale-quality_barrier_via_unif.md)

</div>

<!-- RELATED:END -->
