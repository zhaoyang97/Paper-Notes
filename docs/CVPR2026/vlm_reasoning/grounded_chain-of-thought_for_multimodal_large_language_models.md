---
title: >-
  [论文解读] Grounded Chain-of-Thought for Multimodal Large Language Models
description: >-
  [CVPR 2026][VLM Reasoning][多模态大模型] 提出"接地链式思考（GCoT）"新任务和 MM-GCoT 基准：让多模态大模型在回答前逐步说出推理并给出每一步的坐标依据，再用"答案-接地一致性"指标量化视觉幻觉，结果发现 12 个先进 MLLM 普遍"答对但看错"，且幻觉与模型规模无关。
tags:
  - "CVPR 2026"
  - "VLM Reasoning"
  - "多模态大模型"
  - "视觉幻觉"
  - "接地推理"
  - "链式思考"
  - "评测基准"
---

# Grounded Chain-of-Thought for Multimodal Large Language Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Grounded_Chain-of-Thought_for_Multimodal_Large_Language_Models_CVPR_2026_paper.html)  
**代码**: https://github.com/DoubtedSteam/MM-GCoT  
**领域**: 多模态VLM / 视觉幻觉 / 评测基准  
**关键词**: 多模态大模型, 视觉幻觉, 接地推理, 链式思考, 评测基准

## 一句话总结
提出"接地链式思考（GCoT）"新任务和 MM-GCoT 基准：让多模态大模型在回答前逐步说出推理并给出每一步的坐标依据，再用"答案-接地一致性"指标量化视觉幻觉，结果发现 12 个先进 MLLM 普遍"答对但看错"，且幻觉与模型规模无关。

## 研究背景与动机

**领域现状**：多模态大模型（MLLM）在各类视觉-语言基准上逼近人类水平，但在视觉-空间推理上仍弱，且存在"视觉幻觉"——生成的描述/推理并不真正基于图像内容。

**现有痛点**：有一类更隐蔽的幻觉是模型**答案对、但依据错**：它靠数据分布偏置（语言偏置）蒙对答案，注意力却落在与问题无关的区域。现有幻觉基准只能判断"答案是否正确"，无法揭示模型在多模态推理中到底有没有用对视觉证据；而现有的接地问答（Grounded QA）虽然给了接地框，却缺少标注的中间推理步骤，看不到"每一步看哪里"。

**核心矛盾**：评测体系只看终点（答案对不对），就会把"靠偏置蒙对"的不可靠模型误判为可靠，给具身 AI 等物理世界应用埋下风险。要量化幻觉，必须把"答案"和"它声称的视觉依据"对齐起来检查。

**本文目标**：(1) 设计一个能同时考查视觉感知、空间接地、并诊断幻觉的任务；(2) 构造配套基准让现有 MLLM 可被客观评测；(3) 用实验揭示当前 MLLM 在视觉推理一致性上的真实状况。

**切入角度**：借鉴 LLM 的链式思考（CoT），但把每一步推理都"接地"——要求模型把问题拆成若干步、每步给出相关实体的边界框坐标作为直观依据，最后才给答案。

**核心 idea**：用"逐步接地的推理链 + 答案-接地一致性指标"把视觉幻觉从"看不见"变成"可量化"。

## 方法详解

### 整体框架
本文不是训练一个新模型，而是定义任务 + 造基准 + 设评测体系三件事。给定一张图和一个问题，GCoT 要求 MLLM 先分解任务、逐步推理并给出每个任务相关元素的空间坐标，最后基于这些步骤给出答案和最终答案区域的坐标。配套的 MM-GCoT 基准用四阶段管线从 Visual Genome 造出 1200 个含多步接地标注的样本（分 Attribute / Judgement / Object 三类）；评测体系给出答案准确率、接地准确率、答案-接地一致性三个指标，并设计三种 prompt 设置去"逼出"模型的接地能力。

### 关键设计

**1. GCoT 任务形式化：把"答案"和"逐步视觉依据"绑在一起**

痛点是传统 VQA 只给一个映射 $F:I,T\to A$，无从知道答案是否基于所见。GCoT 把它改写成一个多步决策过程：

$$P(A|I,T)=\prod_{t=1}^{T}P(R_t,G_t|I,T,G_{<t},R_{<t})$$

其中 $I,T$ 是输入图像与文本，$A$ 是最终答案，$R_t$、$G_t$ 分别是第 $t$ 步的文本思考和对应空间坐标。模型必须在每一步显式给出"我现在看的是哪个实体、它在图里哪个框"，从而把隐藏的推理过程暴露出来。这一形式与只关注知识推理的视觉 CoT（如 LLaVA-CoT）不同——GCoT 的思考是关于视觉-空间感知的、且每步都有坐标可核验；它也比 Grounded QA 多了"中间推理步骤"。作者还指出该形式天然适配 GRPO 等 RL 方案，可在无 GCoT 标注的数据上训练。

**2. MM-GCoT 基准的四阶段构造：保证推理复杂度与接地精度**

为了既有多步推理又有精确坐标，作者设计了四阶段管线。第一步，用 IoU 匹配把 Visual Genome 的区域描述与物体标注对齐（一个物体只有一个区域且 IoU 达阈值时配成对）；第二步，以匹配好的物体为节点、按空间与语义关系建"空间关系图"，从图上迭代采样关系路径生成多步推理链；第三步，用结构化模板汇集边界框坐标、物体属性与上下文关系；第四步，用 LLM 把模板翻译成流畅自然语言问题，最后人工校验。最终得到 1200 个样本，分三类：Attribute（问目标物体的类型/位置）、Judgement（判断描述对错）、Object（识别指定位置的物体类别）。这套"关系图采样"保证了推理链既真实可达又有明确接地真值。

**3. 答案-接地一致性指标：把视觉幻觉变成一个可计算的数**

痛点是只看答案准确率（A-Acc，文本匹配）和接地准确率（G-Acc，用 Acc@0.5，即 IoU>0.5）都无法刻画"答对但看错"。作者提出一致性指标：

$$\text{Con.}=\frac{|S_{ca,cb}|}{|S_{ca,cb}|+|S_{ca,wb}|+|S_{wa,cb}|}$$

其中 $S_{ca,cb}$ 是"答案对且框对"、$S_{ca,wb}$ 是"答案对但框错"、$S_{wa,cb}$ 是"答案错但框对"。直观说：在所有"答案对或框对"的样本里，真正"既答对又看对"的占比有多高。一致性低就意味着模型大量地"答对了却没看对地方"或"看对了却答错"，即视觉幻觉严重。这个指标把幻觉从定性描述变成可横向比较的标量。

**4. 三种 prompt 设置：从不同角度逼出并诊断接地行为**

多数 MLLM 虽会做视觉接地，却很难同时输出"答案 + 接地"。作者设计三种 prompt 来激发并对比其接地问答能力：**Answer-First**（先答后给框）、**Grounding-First**（先给答案区域框再答）、**Grounding-CoT**（多轮对话里逐步给每一步实体的框、最后给答案）。三者难度递增：grounding-first 最易（只需定位最终答案），grounding-CoT 最难（要多步推理 + 全程逐实体接地）。同一模型在三种设置下三个指标的落差，本身就是诊断其推理可靠性的信号——例如一致性在 grounding-CoT 下若大幅下降，说明模型做不到"步步看对"。

## 实验关键数据

### 主实验
在 MM-GCoT 上评测 12 个代表性 MLLM（LLaVA、LLaVA-OneVision、Qwen2.5-VL、InternVL2.5）。下表为 Answer-First 设置下的三项均值指标（A-Acc=答案准确率，G-Acc=接地准确率 Acc@0.5，Consist.=答案-接地一致性，均为越高越好，单位 %）。

| 模型 | A-Acc | G-Acc | Consist. |
|------|-------|-------|----------|
| LLaVA-OneVision-72B | 74.7 | 16.4 | 15.3 |
| InternVL2.5-78B | 64.0 | 42.9 | 36.6 |
| Qwen2.5-VL-72B | 73.2 | 39.7 | 38.8 |
| **Qwen2.5-VL-7B** | 71.1 | 63.5 | **56.3** |

最显眼的反差：LLaVA-OneVision-72B 答案准确率高达 74.7%，一致性却只有 15.3%——答对一大半，但绝大多数没看对地方。即便最大的 InternVL2.5-78B，一致性也只有 36.6%。一致性最高的反而是 7B 量级的 Qwen2.5-VL-7B（56.3%）。

### 消融实验
跨规模与跨 prompt 设置的对照（一致性 Consist.，%）：

| 对照 | 配置 A | 配置 B | 结论 |
|------|--------|--------|------|
| 模型规模（Qwen2.5-VL） | 7B：56.3 | 72B：38.8 | 7B 反超 72B 达 17.5 |
| 模型规模（grounding-first） | 7B | 72B | 7B 比 72B 高 40.6 |
| prompt 设置（InternVL2.5-38B） | answer-first：39.2 | grounding-CoT：17.8 | 切到 CoT 掉 21.4 |
| 推理步内一致性（Qwen2.5-VL-7B, Judgement） | Step1：92.8 | Step2：7.2 | 直接给答案框的比例骤降 |

### 关键发现
- **答对 ≠ 看对**：大多数 MLLM 一致性极低，说明普遍存在视觉幻觉——答案常靠语言偏置而非真正的视觉证据。
- **幻觉与规模无关**：Qwen2.5-VL 从 3B 到 72B 答案准确率只涨约 4%，但 7B 的一致性反而比 72B 高 17.5%（grounding-first 下高 40.6%）；超大模型在 MM-GCoT 上反而更差，疑似数据过拟合。
- **接地能力与推理能力不挂钩**：InternVL2.5-38B 比 8B 答案准确率高 5.5%，接地反而在 Object 上降 0.3%；强接地不帮答题、强推理不保接地。
- **多步推理不可靠**：grounding accuracy 在各推理步间无明显趋势，模型并不依赖前面已给出的视觉证据；Grounding-CoT 一致性普遍低于 Answer-First，越要它步步接地越露馅。

## 亮点与洞察
- **把幻觉量化的钥匙是"一致性"而非"准确率"**：只看答案对错会高估可靠性，一致性指标第一次让"答对但看错"可被测量和排名，这个评测视角可迁移到任何需要核验依据的多模态任务。
- **用关系图采样造多步接地链很聪明**：从 Visual Genome 的物体关系图采样路径，天然得到"可达 + 有坐标真值"的多步推理链，避免了纯人工标注的高成本与随意性。
- **"规模不解决幻觉"是有力的反直觉结论**：小模型一致性反超大模型，提示一致性是结构性/训练目标问题，单纯堆参数无效，对具身 AI 选型有直接警示。
- **三种 prompt 设置是诊断探针**：同模型在三设置下的指标落差本身就编码了它"答题 vs 接地"的脱节程度，是一种无需训练的诊断手段。

## 局限与展望
- **只评测、未给解法**：本文聚焦诊断，提出任务和基准却没给出系统性降低幻觉的训练方法（仅指出可用 GRPO 等 RL 在该形式上训练），改进留给后续。
- **接地真值来自 Visual Genome**：基准依赖 VG 的区域/物体标注与 IoU 匹配，标注噪声和单区域约束可能限制覆盖的关系复杂度。
- **规模偏小**：1200 个样本、三类任务，相对真实世界视觉-空间推理的多样性仍有限。
- **prompt 敏感**：结果对三种 prompt 设置敏感，模型表现可能部分受指令格式影响而非纯能力差异。

## 相关工作与启发
- **vs 现有幻觉基准（POPE 等）**：它们只判断答案/描述是否被视觉支持，无法看"推理中怎么用视觉证据"；MM-GCoT 通过逐步接地把证据使用过程暴露出来。
- **vs Grounded QA**：Grounded QA 也给接地框验证可信度，但缺中间推理步骤；GCoT 在整条推理链上识别并接地视觉证据。
- **vs 视觉 CoT（LLaVA-CoT / ScienceQA）**：它们偏视觉知识推理、不核验每步是否有正确视觉证据；GCoT 专攻视觉-空间感知并用坐标核验每一步。
- **vs VoCoT**：同样关注接地推理步骤，但本文在接地数据构造方式上不同，且专门面向"缓解视觉幻觉"并提出配套一致性指标。

## 评分
- 新颖性: ⭐⭐⭐⭐ GCoT 任务 + 一致性指标把"答对但看错"首次量化，视角新；但属诊断性工作、未给解法
- 实验充分度: ⭐⭐⭐⭐ 12 个 MLLM × 3 prompt 设置 × 3 子集，横向充分，含逐步与跨规模分析
- 写作质量: ⭐⭐⭐⭐ 任务定义与发现清晰，表格信息密集
- 价值: ⭐⭐⭐⭐ 为多模态可信度评测立了新标尺，对具身 AI 选型与后续抗幻觉研究有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fuel Gauge: Estimating Chain-of-Thought Length Ahead of Time in Large Multimodal Models](fuel_gauge_estimating_chain-of-thought_length_ahead_of_time_in_large_multimodal_.md)
- [\[CVPR 2026\] UniT: Unified Multimodal Chain-of-Thought Test-time Scaling](unit_unified_multimodal_chain-of-thought_test-time_scaling.md)
- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[CVPR 2026\] Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models](deeper_thought_weaker_aim_understanding_and_mitigating_perceptual_impairment_dur.md)
- [\[CVPR 2026\] Chain-of-Thought Guided Multi-Modal Object Re-Identification](chain-of-thought_guided_multi-modal_object_re-identification.md)

</div>

<!-- RELATED:END -->
