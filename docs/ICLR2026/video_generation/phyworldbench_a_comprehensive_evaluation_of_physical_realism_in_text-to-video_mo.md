---
title: >-
  [论文解读] $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models
description: >-
  [ICLR2026][视频生成][文本到视频生成] PhyWorldBench 构建了一个覆盖 50 类物理子现象、1,050 个提示词和 12 个主流文本到视频模型的大规模 benchmark，用人工评测与 context-aware MLLM 评估器系统揭示了当前视频生成模型在真实物理、复杂交互和反物理指令遵循上的明显短板。
tags:
  - "ICLR2026"
  - "视频生成"
  - "文本到视频生成"
  - "物理真实性"
  - "Benchmark"
  - "多模态评测"
  - "反物理场景"
---

# $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=rlZeILv3fm](https://openreview.net/forum?id=rlZeILv3fm)  
**代码**: 待确认  
**领域**: 视频生成 / 物理真实性评测  
**关键词**: 文本到视频生成, 物理真实性, Benchmark, 多模态评测, 反物理场景

## 一句话总结
PhyWorldBench 构建了一个覆盖 50 类物理子现象、1,050 个提示词和 12 个主流文本到视频模型的大规模 benchmark，用人工评测与 context-aware MLLM 评估器系统揭示了当前视频生成模型在真实物理、复杂交互和反物理指令遵循上的明显短板。

## 研究背景与动机
**领域现状**：文本到视频生成模型近两年在画面质量、主体一致性和镜头语言上进步很快，Sora、Kling、Pika、Gen-3 以及 Hunyuan、Wanx、Open-Sora 等模型都能生成视觉上很有吸引力的视频。问题是，视觉真实不等于物理真实：一个苹果可以看起来很清晰、光影很漂亮，但下落轨迹、碰撞、速度变化、破碎过程仍然可能完全不符合现实世界。

**现有痛点**：已有 T2V benchmark 往往更关注画质、文本对齐、时序一致性或组合性，物理相关 benchmark 也常只覆盖少量物理类别，或者集中在动作/动态常识的一小块区域。这样会带来两个盲点：第一，模型可能在简单运动上表现不错，但在流体、刚体、能量守恒、尺度效应、人和动物运动等复杂物理上崩掉；第二，模型是否真的理解物理规律，还是只是在训练数据中复现常见画面模式，很难被区分出来。

**核心矛盾**：视频生成模型的训练目标更偏向像素分布、语义对齐和视觉连续性，而物理规律是一组跨时间、跨对象、跨尺度的约束。模型可以通过平滑运动、电影化构图或合理化 prompt 来“看起来像视频”，却不一定能处理突然破碎、多力交互、反重力、能量转化这类需要物理因果的场景。

**本文目标**：作者希望构建一个足够宽、足够细、可复现的物理真实性评测基准。它不仅要覆盖基础物理和复合物理，还要加入故意违反现实物理的 Anti-Physics 场景；不仅要有 prompt，还要有明确的 Yes/No 标准；不仅要靠大规模人工评测，也要探索能否用现有 MLLM 以低成本近似人工评测。

**切入角度**：PhyWorldBench 的切入点是把“物理真实性”拆成可检查的标准：视频里有没有正确对象和事件，物理现象有没有按现实规律发生。这个拆法比单纯打一个主观质量分更直接，也能把语义失败和物理失败分开。

**核心 idea**：用结构化物理类别、三种 prompt 变体、人工标注标准和 context-aware MLLM 评估器，建立一个专门拷问文本到视频模型物理真实性的系统性 benchmark。

## 方法详解

### 整体框架
PhyWorldBench 本质上不是一个新的视频生成模型，而是一套“数据集 + 评测协议 + 自动评估器 + 模型诊断”的 benchmark。输入端是围绕物理现象设计的文本 prompt；输出端是各个 T2V 模型生成的视频；评测端先检查语义是否对齐，再检查视频是否满足该 prompt 对应的关键物理标准。

整个流程可以概括为四步：先由物理文献和专家意见定义物理类别，再为每个子类构造多种 prompt；然后为每个 prompt 写出 Basic Standards 和 Key Standards；最后对 12 个模型生成的 12,600 个视频做人工评测，并验证一个零样本 MLLM 评估方案 CAP 来降低后续评测成本。

### 关键设计
**1. 分层物理 taxonomy：把物理真实性从抽象概念拆成 50 个可测子类**

这篇论文最重要的工程工作是把“物理是否真实”拆成一个系统化目录。PhyWorldBench 先定义 10 个主物理类别，每个类别再分成 5 个子类别，总计 50 个物理子类。主类别覆盖 Object Motion and Kinematics、Interaction Dynamics、Energy Conservation、Fluid and Particle Dynamics、Rigid Body Dynamics、Lighting and Shadows、Deformations and Elasticity、Scale and Proportions、Human and Animal Motion，以及 Anti-Physics。

这种设计的价值在于，它避免了 benchmark 只测“球滚动”“物体下落”一类低维场景。比如刚体动力学里会涉及旋转、力矩、平衡、质心、冲击和形变；流体和粒子动力学里会涉及水、烟雾、浮力、黏性和粒子行为；尺度和比例则检查同样的力学关系在不同尺度下是否仍合理。模型如果只是记住常见视频片段，很容易在这些跨类别、跨物体、跨尺度的场景中暴露弱点。

**2. 三种 prompt 变体：区分模型缺少物理知识还是缺少提示信息**

每个物理子类别下有 7 个场景，每个场景再写 3 种 prompt：Event Prompt、Physics-enhanced Prompt 和 Detailed Narrative Prompt。Event Prompt 是最短事件描述，例如“火箭升空”；Physics-enhanced Prompt 会补入自然物理后果，例如“垂直向上、沿直线进入空中”；Detailed Narrative Prompt 则加入更丰富的环境、光影和叙事细节。

这个设计很关键，因为它把“模型是否理解物理”和“prompt 是否说得够清楚”拆开了。若 Physics-enhanced Prompt 明显提升结果，说明模型有一定 prompt-following 能力，但需要用户把物理现象显式写出来；若 Detailed Narrative Prompt 没有提升物理正确性，说明增加电影化细节主要改善表面观感，而不一定改善深层物理。论文后面的实验正好支持这个判断：显式加入物理现象通常更有用，单纯把 prompt 写得更长、更生动并不可靠。

**3. Basic Standards 与 Key Standards：把评测从主观观感变成可投票的 Yes/No 标准**

PhyWorldBench 不直接让评测者给一个笼统分数，而是为每个 prompt 写两个层次的标准。Basic Standards 检查对象和事件是否出现，比如视频里是否有“两名足球运动员”、是否真的发生“互相冲撞”。Key Standards 检查关键物理现象是否出现，比如接触点是否自然、碰撞后身体动量和方向变化是否平滑可信。

论文沿用并整理出两个核心指标：Semantic Adherence（SA）和 Physical Commonsense（PC）。可以把它理解成：$SA=1$ 表示对象和事件与文本对上，$PC=1$ 表示关键物理现象也符合现实规律；最终成功率常用 Both 表示，也就是同时满足 $SA=1$ 和 $PC=1$ 的比例。这个二元评测虽然粗粒度，但好处是边界清楚、可多数投票、便于扩展到大量视频。

**4. Anti-Physics 类别：专门测试模型是在理解物理还是在合理化训练分布**

Anti-Physics 是这篇 benchmark 中比较有洞察力的一块。作者故意设计“反重力”“能量凭空产生”“物体穿模”“时间反转”“无限复制/分裂”等违反现实规律的 prompt。表面上看这像是在鼓励不真实，但它的测试目的很明确：如果 prompt 要求“酒杯里的酒呈现反重力”，模型却生成静止酒杯或普通液体，那么模型不是在遵循指令，而是在把输入合理化成训练数据里更常见的现实场景。

这类测试能揭示传统物理 benchmark 不容易看到的问题。一个真正可控的视频生成模型应该既能在现实场景里遵守物理，也能在创意场景里按用户要求生成逻辑一致的“反物理”效果。当前模型在 Anti-Physics 上普遍大幅掉分，说明它们既没有稳定的物理模型，也没有足够强的反事实控制能力。

**5. CAP 自动评估器：告诉 MLLM “这是 AI 生成视频”，减少它替视频找借口**

大规模人工评测很贵，所以论文提出 Context-Aware Prompt（CAP）作为零样本自动评估器。作者观察到一个有趣现象：MLLM 在看真实世界视频时会倾向于合理化画面，即使画面有物理问题，也可能默认“现实中发生了就一定有原因”。当 prompt 明确告诉 MLLM 这些帧来自 AI 生成视频、可能存在质量问题时，它对物理错误更敏感。

CAP 的做法不是直接暗示“视频是错的”，而是分两步走：第一步让 MLLM 描述视频中的对象、事件和观察到的现象；第二步再根据给定标准回答 Yes/No。论文报告 CAP 在 SA 上达到 80.3 ROC-AUC，在 PC 上达到 75.1 ROC-AUC，比普通 GPT-o1 的 75.4 / 61.6 明显更好，尤其 Physical Commonsense 提升 13.5 个点。这说明一个非常简单的上下文提示和 CoT 式评估流程，就能显著改善 MLLM 对生成视频物理真实性的判断。

## 实验关键数据

### 主实验
论文评测了 12 个主流文本到视频模型，其中 5 个是闭源/商业模型，7 个是开源模型。每个模型对 1,050 个 prompt 生成视频，总计 12,600 个视频，并通过 Amazon Mechanical Turk 进行人工评测，每个视频由 3 名标注者投票。

| 模型 | 类型 | Overall SA | Overall PC | Overall Both | 主要结论 |
|------|------|------------|------------|--------------|----------|
| Pika 2.0 | 闭源 | 0.521 | 0.314 | 0.262 | 人工评测总体最好，但 Anti-Physics Both 只有 0.011 |
| Sora-Turbo | 闭源 | 0.384 | 0.261 | 0.208 | 物理正确性较强，人类 leaderboard 闭源第 2 |
| Kling-1.6 | 闭源 | 0.357 | 0.241 | 0.188 | 画面和运动较强，但电影化风格可能掩盖物理错误 |
| Wanx-2.1 | 开源 | 0.339 | 0.235 | 0.189 | 开源模型中 Both 最好，接近部分闭源模型 |
| Hunyuan 720p | 开源 | 0.344 | 0.250 | 0.185 | 开源模型中 PC 最好，物理常识得分较稳 |
| LTX-Video | 开源 | 0.194 | 0.085 | 0.062 | 整体最弱，常见低保真和形变问题 |

这个结果最醒目的不是谁赢了，而是绝对分数都偏低。即便 Pika 2.0 是总体最强，Both 也只有 0.262；也就是说，在这个 benchmark 上，能同时做到文本语义正确和物理现象正确的视频比例只有约四分之一。Anti-Physics 更困难，许多模型的 Both 接近 0，说明它们在反事实物理控制上基本还不可靠。

| Benchmark | Physics Categories | Prompts | 结论 |
|-----------|--------------------|---------|------|
| VideoPhy | 5 | 688 | 有物理常识评测，但覆盖较窄 |
| PhyGenBench | 27 | 160 | 覆盖类别较多，但 prompt 数较少 |
| Physics-IQ | 5 | 396 | 关注基础物理理解 |
| T2VPhysBench | 3 | 84 | 更小规模的物理一致性测试 |
| PhyWorldBench | 50 | 1050 | 覆盖最广、prompt 最多，难度也更高 |

与已有 benchmark 对比，PhyWorldBench 的特点是覆盖面更宽，而且每个场景都有三种 prompt 变体和明确评测标准。它更像一个压力测试，而不是只验证模型能否生成少数常见物理片段。

### 消融实验
CAP 自动评估器的消融表明，context 和 CoT 两个组件都重要，尤其对 PC 指标影响很大。

| 自动评估方法 | SA ROC-AUC | PC ROC-AUC | 说明 |
|--------------|------------|------------|------|
| Qwen-VL-2.0 | 72.4 | 59.8 | 开源 MLLM 直接评测，物理判断较弱 |
| Gemini-2.0-Flash | 74.6 | 60.9 | 直接评测仍难处理细粒度物理 |
| GPT-4o | 72.1 | 60.1 | 没有 CAP 时 PC 仍不理想 |
| GPT-o1 | 75.4 | 61.6 | 基线最强，但物理 commonsense 仍偏低 |
| CAP w/o CoT | 76.3 | 73.6 | 加入“AI 生成视频”上下文后 PC 大幅提升 |
| CAP w/o Context | 77.3 | 65.6 | 有推理步骤但不提示生成视频，提升有限 |
| CAP | 80.3 | 75.1 | context + 两步推理效果最好 |

Prompt 类型实验也给出一个很实用的结论：把物理现象明确写进 prompt，通常比单纯写长、写美更有帮助。

| 模型 | Event Prompt | Physics-enhanced Prompt | Detailed Prompt | 观察 |
|------|--------------|-------------------------|-----------------|------|
| CogVideoX-1.5 | 0.123 | 0.177 | 0.168 | 物理增强明显提高成功率 |
| Hunyuan 720p | 0.159 | 0.198 | 0.155 | 详细叙事反而不如物理增强 |
| Open-Sora 2.0 | 0.167 | 0.177 | 0.173 | 提升较小，但方向一致 |
| Wanx-2.1 | 0.175 | 0.202 | 0.190 | 物理增强最好 |
| Step-video-T2V | 0.158 | 0.182 | 0.179 | 详细 prompt 接近物理增强，但仍略低 |

论文还专门分析了两个失败模式。对于突发视觉变化，所有模型在 Abrupt 场景中只有 53/240 通过，成功率 22.1%，而 Non-abrupt 场景为 101/240、42.1%；Pika 在 Abrupt 上是 30.0%，Non-abrupt 上是 75.0%。对于复杂场景，所有模型 Complex 场景只有 28/240 通过，成功率 11.7%，Less-complex 为 44/240、18.3%；Pika 也从 40.0% 掉到 15.0%。这说明当前模型倾向于生成平滑、容易的版本，而不是忠实模拟破碎、碰撞、多力交互这类困难物理过程。

### 关键发现
- 当前 T2V 模型的视觉质量和物理正确性不是一回事。Pika、Sora、Kling 等模型可以生成很漂亮的视频，但在 Both 指标上仍远低于可用于严肃仿真的水平。
- 闭源模型整体领先，但开源模型并没有完全落后。Wanx-2.1 和 Hunyuan 720p 在部分指标上接近甚至超过一些闭源模型，说明物理真实性不是闭源模型独占优势。
- Anti-Physics 是最难的类别之一。模型常常把反物理 prompt 合理化为现实场景，而不是生成用户要求的反事实现象。
- Prompt 工程能缓解但不能解决问题。Physics-enhanced Prompt 能提高成功率，但 Detailed Prompt 不稳定，说明深层物理能力缺口不能靠叙事细节补齐。
- CAP 的价值在于低成本筛查。它还不能替代人工评测，但已经能在零样本设定下显著提高 MLLM 对物理错误的敏感性。

## 亮点与洞察
- **Anti-Physics 设计很有辨识度**：它把“遵守现实物理”和“遵循用户指令”这两个目标拆开。这个思路可以迁移到其他生成任务，例如图像生成里的反常透视、机器人仿真里的反事实动作、世界模型里的不可能事件。
- **Basic / Key Standards 的拆分很实用**：很多视频失败不是物理失败，而是对象或事件本身没生成对。先测 SA 再测 PC，可以避免把语义对齐错误误判成物理理解错误。
- **CAP 提醒我们评估 prompt 也有分布偏置**：MLLM 默认把视频当现实世界观察，会倾向于找理由解释异常。告诉它“这是 AI 生成视频”并不是作弊，而是把评估上下文校准到正确分布。
- **实验揭示了模型的“平滑化逃避”倾向**：破碎、碰撞、突然状态变化和多力交互是当前模型明显不擅长的区域。未来视频生成若要服务机器人、自动驾驶、科学模拟，必须显式处理这些非平滑动力学。
- **对用户 prompt 有直接启发**：如果目标是物理真实，prompt 里应该明确写出关键物理后果，例如加速度、碰撞后的反弹、液体流动方向、阴影变化，而不是只写氛围和镜头语言。

## 局限与展望
- **Yes/No 标准仍然较粗**：一个视频可能部分符合物理规律，但二元标签只能记成通过或不通过。未来可以引入分级评分，区分“轻微不自然”和“物理完全错误”。
- **人工标注仍有主观性**：虽然每个视频由 3 名标注者投票，但复杂物理现象的可见性、是否足够真实，仍可能存在分歧。
- **CAP 仍会受视觉美学偏置影响**：论文自己也指出，Kling 的电影化风格可能让 CAP 打出偏高分，而人类评测会更严格惩罚这类物理不自然。
- **benchmark 覆盖宽但不等于可穷尽物理世界**：50 个子类已经很丰富，但真实世界还包含更细的材料属性、接触力学、复杂流体、软体动力学和长时程因果。
- **对生成模型只做黑盒评测**：闭源模型的内部 prompt 重写、采样策略和训练数据不可见，因此 benchmark 能指出失败现象，但不能直接定位模型内部原因。
- **未来方向**：可以把 PhyWorldBench 与物理引擎、轨迹估计、3D 场景重建结合起来，让评测从“看起来是否符合”走向“可测量的动力学一致性”。

## 相关工作与启发
- **vs VBench / EvalCrafter**: 这些 benchmark 更全面地覆盖视频质量、时序一致性、主体保持和通用生成能力；PhyWorldBench 更聚焦物理真实性，把物理现象拆成可检查的类别与标准。
- **vs VideoPhy / PhyGenBench**: 这些工作已经开始关注视频生成中的物理常识，但 PhyWorldBench 的物理类别和 prompt 数量更大，并额外加入 Anti-Physics 来测试反事实控制。
- **vs Physics-IQ / T2VPhysBench**: 它们同样关注物理一致性，但规模和覆盖面更小；PhyWorldBench 更适合作为大模型 leaderboard 和系统诊断工具。
- **对视频生成模型研发的启发**: 只提升分辨率、时长和镜头美学不够，模型需要更强的动力学表征、对象持久性、接触建模和事件级状态变化能力。
- **对自动评测的启发**: 评测生成视频时，MLLM prompt 应该显式说明评测对象是 AI 生成内容，并让模型先观察再判断，避免它用现实世界先验替错误视频圆场。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ Benchmark 方向已有相关工作，但 Anti-Physics、三种 prompt 变体和 CAP 评估器组合得很有辨识度。
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个模型、1,050 个 prompt、12,600 个视频和人工/自动双评测，规模与分析都比较扎实。
- 写作质量: ⭐⭐⭐⭐☆ 主线清楚，图表信息量大；少数地方模型数量表述和表格标题略有不一致，但不影响核心结论。
- 价值: ⭐⭐⭐⭐⭐ 对视频生成、世界模型、物理一致性评估和 prompt 设计都有直接参考价值，是一个很适合作为后续模型诊断工具的 benchmark。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] NarrLV: Towards a Comprehensive Narrative-Centric Evaluation for Long Video Generation](narrlv_towards_a_comprehensive_narrative-centric_evaluation_for_long_video_gener.md)
- [\[ICLR 2026\] VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation](videophy-2_a_challenging_action-centric_physical_commonsense_evaluation_in_video.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[CVPR 2026\] VideoRealBench: A Chain-of-Thought Realism Evaluation Benchmark for Generated Human-Centric Videos](../../CVPR2026/video_generation/videorealbench_a_chain-of-thought_realism_evaluation_benchmark_for_generated_hum.md)
- [\[ICML 2026\] V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation](../../ICML2026/video_generation/v2v-bench_a_comprehensive_benchmark_for_video-to-video_generation_evaluation.md)

</div>

<!-- RELATED:END -->
