---
title: >-
  [论文解读] Can MLLMs Reason Beyond Language? VisReason: A Comprehensive Benchmark for Vision-Centric Reasoning
description: >-
  [ACL2026 Findings][多模态VLM][视觉中心推理] VisReason 构建了一个包含 1,505 道日常视觉推理题的多模态 benchmark，专门测试模型是否能直接基于视觉证据推理，结果显示最强模型平均准确率也只有 47.5%，显著低于人类 71.4%，且 CoT 与更大推理预算只能带来有限提升。
tags:
  - "ACL2026 Findings"
  - "多模态VLM"
  - "视觉中心推理"
  - "多模态评测"
  - "视觉 grounding"
  - "CoT"
  - "能力诊断"
---

# Can MLLMs Reason Beyond Language? VisReason: A Comprehensive Benchmark for Vision-Centric Reasoning

**会议**: ACL2026 Findings  
**arXiv**: [2605.25364](https://arxiv.org/abs/2605.25364)  
**代码**: https://github.com/CASIA-IVA-Lab/VisReason  
**领域**: 多模态VLM / 视觉推理 / Benchmark  
**关键词**: 视觉中心推理、多模态评测、视觉 grounding、CoT、能力诊断

## 一句话总结
VisReason 构建了一个包含 1,505 道日常视觉推理题的多模态 benchmark，专门测试模型是否能直接基于视觉证据推理，结果显示最强模型平均准确率也只有 47.5%，显著低于人类 71.4%，且 CoT 与更大推理预算只能带来有限提升。

## 研究背景与动机
**领域现状**：现有 MLLM 在 MMMU、MathVista、科学问答、图表推理等 benchmark 上表现越来越强，很多结果被解读为模型具备复杂视觉推理能力。

**现有痛点**：不少视觉推理题实际可以被“图像转语言”后解决。论文用 Qwen3-VL-32B 做诊断发现，在 MMMU、MMMU-Pro、MathVista 等数据集上，把图像替换为 Qwen-VL-Max 生成的 caption 后性能下降不到 6.35%，说明这些任务很大一部分推理可在语言空间完成。

**核心矛盾**：人类视觉推理经常依赖直接观察形状、布局、局部差异、空间关系和隐含规则；而 MLLM 可能只是把图像压缩成文本描述，再做语言推理。现有 benchmark 难以区分“会视觉推理”和“会读 caption 后推理”。

**本文目标**：构建一个视觉证据不可被语言充分替代的 benchmark，用统一协议评估 proprietary 和 open-source MLLM，并分析模型容量、CoT、推理预算和错误类型对视觉中心推理的影响。

**切入角度**：VisReason 聚焦日常视觉场景，而不是 STEM 知识题。题目覆盖定位、找不同、图案计数、3D 空间、棋盘、数独、地理线索、cue insight、归纳和演绎推理，要求模型在感知与推理之间来回校验。

**核心 idea**：用“视觉证据是否不可替代”作为 benchmark 设计原则，检验 MLLM 是否真正 reason beyond language。

## 方法详解

### 整体框架
VisReason 想戳破的现象是：很多被当作"视觉推理"的题，其实把图像转成 caption 后照样能答对，模型只是在做语言推理而非真正看图。围绕"视觉证据不可被语言替代"这一设计原则，论文先定义视觉中心推理的类别体系，再人工收集、改写、验证出一批多格式题目，最后用大规模模型评测加能力拆解来解释模型为什么失败。具体地，输入是图像和问题，输出按题型分为选择题、填空、开放答案和 bounding box；评测时选择题/填空用规则抽取答案后精确匹配，开放题交给 GPT-5-mini 严格判分，bounding box 用 IoU 匹配并按题目归一化，最终报告各类别准确率和平均准确率。

### 关键设计

**1. 视觉中心数据筛选：只留下"图像证据无法被文字替代"的题**

如果一道题能被 caption 完整替代，那答对它根本说明不了模型的视觉推理能力，诊断也就无从谈起。为此作者从在线视觉推理题和中文公务员考试题库收集候选，由研究生 annotator 标准化成 VQA 格式，再系统过滤掉四类"假视觉"样本：只看图就能答的、只看文本就能答的、依赖外部知识的、以及答案不唯一的。经过这道闸门，留下的每一题都必须真正回到图像本身去找证据——这也是后文"换 caption 后 VisReason 暴跌 48.12% 而旧基准只跌不到 6.35%"这一诊断能成立的前提。

**2. 三层推理类别体系：把视觉推理拆成感知、结构、概念三层，对准不同瓶颈**

视觉推理是个笼统的词，不拆开就看不出模型卡在哪一环。VisReason 把它分成三层共 10 个类别、36 个子类：感知层是 Localized Reasoning、Spot the Difference、Pattern Counting，主要考视觉 grounding；结构层是 3D-Spatial、Board、Sudoku、Geolocation，考的是从图里抽取结构化状态并执行规则；概念层是 Cue Insight、Inductive Reasoning、Deductive Reasoning，更偏抽象推理。这样分层之后，"定位/找不同失分"就能归到 grounding，"数独/棋盘失分"能归到结构状态抽取，每一类失败都能对应到一个具体的能力短板，而不是混成一个笼统的低分。

**3. 模型诊断协议：把规模、推理预算、CoT 和错误类型放在同一框架下解剖**

平均分只能告诉你"难"，说不清"为什么难"，也回答不了"scaling 或 CoT 到底能不能补上视觉 grounding"。协议因此在统一的 zero-shot CoT 风格提示下评估一大批模型——GPT-4o、o4-mini、GPT-5 系列、Gemini 系列、InternVL3、Qwen2.5-VL、Qwen3-VL 的 thinking/non-thinking 版、MiMo-VL、Keye-VL 等——并在评测之上叠加四类分析：caption 替换诊断（验证视觉证据是否真不可替代）、类别相关性、错误分类（error taxonomy）和原子能力（atomic capability）分析。正是这套拆解才得出"感知错误 + 推理错误占了约 90%、CoT 对视觉证据主导任务几乎无帮助"这类结论，而不是停在一个孤零零的准确率上。

### 损失函数或训练策略
本文不训练模型，主要是 benchmark 构造与零样本评测。题目构建阶段使用多轮人工校验保证图像、问题、答案一致；评测阶段所有模型使用四类题型对应的 prompt，并尽量按同一 answer-format 要求输出。对于 bounding-box，预测框与真值框用 Hungarian matching 匹配，IoU 大于 0.5 记为正确并按题目归一化。

## 实验关键数据

### 主实验
| 数据集 / 模型组 | 指标 | 最强模型 | 人类 | 结论 |
|----------------|------|----------|------|------|
| VisReason overall | Avg accuracy | Gemini-3-Pro 47.5 | 71.4 | 当前 MLLM 与人类仍有 23.9 点差距 |
| Proprietary vs open-source | Avg accuracy | Gemini-3-Pro 47.5 vs Qwen3-VL-235B-A22B-Thinking 26.7 | 71.4 | 专有模型明显领先开源模型 |
| Localized Reasoning | Avg model accuracy | 全模型平均 6.9 | 81.7 | 精确视觉定位是主要瓶颈之一 |
| Spot-the-Difference | Avg model accuracy | 全模型平均 1.3 | 77.4 | 对应/差异检测几乎失效 |
| CoT prompting on GPT-4o | Avg improvement | +1.1 | - | 文本推理链对视觉 grounding 帮助有限 |

| VisReason 统计 | 数量 / 比例 |
|----------------|-----------|
| 总题数 | 1,505 |
| Multiple-choice | 615 (40.9%) |
| Fill-in-the-blank | 441 (29.3%) |
| Open-ended | 309 (20.5%) |
| Bounding-box | 140 (9.3%) |
| 推理类别 / 子类 | 10 / 36 |
| 问题长度 min / avg / max | 5 / 45.9 / 282 tokens |

### 消融实验
| 分析项 | 关键指标 | 说明 |
|--------|----------|------|
| 图像换 caption | 现有 benchmark 下降 < 6.35%，VisReason 下降 48.12% | VisReason 更依赖直接视觉证据 |
| 模型容量 scaling | Qwen2.5/Qwen3 系列随规模增大普遍上升 | capacity 有帮助，但远未到人类水平 |
| 推理预算 scaling | GPT-5-mini 从低到中预算有增益，高预算趋于饱和 | test-time compute 有边际收益递减 |
| 错误类型 | 感知错误 + 推理错误约占 90% | 主要不是格式问题，而是视觉解析、grounding、计数、规则选择和幻觉推断 |

### 关键发现
- VisReason 能清楚地区分模型能力：弱模型接近随机，最强 proprietary 模型也远低于人类，说明任务不是简单饱和型 benchmark。
- CoT 对高层 cue insight、归纳和演绎可能有帮助，但对找不同、3D 空间、计数等视觉证据主导任务帮助很小，甚至可能退化。
- 模型在 bounding-box 题上的低分不只是定位框格式问题，而是需要边推理边维护空间假设，这比单纯检测更难。

## 亮点与洞察
- 论文最有价值的地方是重新定义“视觉推理评测”的标准：不是题目里有图就算视觉推理，而是必须证明视觉证据不能被语言描述轻易替代。
- 类别设计覆盖日常视觉推理，比只看数学图表或学科知识更能暴露 MLLM 的基础视觉认知短板。
- 结果对“多给 CoT 就会推理”提出了很强的反例：当瓶颈是视觉解析和证据定位时，语言侧思考预算并不能自动补足感知缺陷。

## 局限与展望
- VisReason 只覆盖静态图像，没有涉及视频、交互式感知、具身任务或连续动作决策；未来可扩展到动态视觉推理。
- 数据来自在线题库和考试题，虽然人工过滤过，但仍可能带有题源风格偏差；模型训练集污染和 benchmark overfitting 也是长期风险。
- 开放题使用 GPT-5-mini 判分，虽然严格但仍引入 judge model 的主观性；后续可加入更多人工复核或多 judge 一致性分析。
- 当前主要是评测而非训练。未来可以把 VisReason 的 atomic capabilities 用作训练课程，针对 correspondence/difference、structured state extraction 等瓶颈设计专项数据。

## 相关工作与启发
- **vs MMMU / MathVista**: 这些 benchmark 评估视觉学科知识和数学推理，但部分题目可通过 caption 保留大部分信息；VisReason 则显式追求 caption 不可替代。
- **vs Sudoku-Bench / REBUS / VGRP-Bench**: 这些任务聚焦特定 puzzle 或规则推理，VisReason 覆盖更广的日常视觉场景和多种答案形式。
- **vs CoT-based MLLM reasoning**: CoT 强化语言侧推理链，VisReason 的结果提醒我们，多模态推理需要把视觉 parsing、grounding 和符号推理联动训练，而不是只延长文本输出。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 用 caption 替代诊断来界定视觉中心推理，benchmark 设计问题意识很强。
- 实验充分度: ⭐⭐⭐⭐⭐ 模型覆盖、类别分析、CoT/scale/error 分析都很全面。
- 写作质量: ⭐⭐⭐⭐☆ 结构清楚，数据和结论直接，但大表较长，需要读者抓重点。
- 价值: ⭐⭐⭐⭐⭐ 对多模态评测、VLM 训练数据设计和视觉 reasoning 研究都有高参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](../../CVPR2026/multimodal_vlm/think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[CVPR 2026\] Beyond Single Images: A Comprehensive Benchmark for Album-Level Vision-Language Understanding](../../CVPR2026/multimodal_vlm/beyond_single_images_a_comprehensive_benchmark_for_album-level_vision-language_u.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ACL 2026\] Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models](almieyar-oryx-bloombench_a_bilingual_multimodal_benchmark_for_cognitively_inform.md)
- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
