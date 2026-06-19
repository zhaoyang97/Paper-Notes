---
title: >-
  [论文解读] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations
description: >-
  [ICML 2026][LLM推理][空间推理] FloorplanQA 用 2,000 个 JSON/XML 格式的 2D 室内布局 + 16,000 道几何题（距离/可见性/路径/放置等）系统性诊断了 15 个前沿 LLM 的"纯符号空间推理"能力，发现它们能算简单距离却普遍栽在并集、规划和约束满足上，且 Python 工具增强能修复算术错误但救不了算法层面的失败。
tags:
  - "ICML 2026"
  - "LLM推理"
  - "空间推理"
  - "结构化表示"
  - "JSON 布局"
  - "几何推断"
  - "LLM 诊断基准"
---

# FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations

**会议**: ICML 2026  
**arXiv**: [2507.07644](https://arxiv.org/abs/2507.07644)  
**代码**: https://OldDelorean.github.io/FloorplanQA/ (项目页)  
**领域**: LLM 评测 / 空间推理 / 室内布局  
**关键词**: 空间推理、结构化表示、JSON 布局、几何推断、LLM 诊断基准  

## 一句话总结
FloorplanQA 用 2,000 个 JSON/XML 格式的 2D 室内布局 + 16,000 道几何题（距离/可见性/路径/放置等）系统性诊断了 15 个前沿 LLM 的"纯符号空间推理"能力，发现它们能算简单距离却普遍栽在并集、规划和约束满足上，且 Python 工具增强能修复算术错误但救不了算法层面的失败。

## 研究背景与动机

**领域现状**：LLM 在结构化推理上展示出惊人能力，建筑设计、辅助规划、具身交互等场景已经开始让模型直接吃 JSON 格式的布局（坐标、尺寸、朝向），而不是依赖图像。这意味着模型必须在符号几何上做推理，不是在像素上做感知。

**现有痛点**：现有空间推理基准要么停留在定性关系（"在左边/上方"），要么把布局塞回图像让 VLM 评估，要么藏在 ALFRED/R2R 这类具身任务里隐式考察导航。没有一个干净的基准能单独探针"给一段 JSON 布局，模型能不能算清楚 1.7 米的过道、判断 2×3 米的桌子能否塞进去、找出从灶台到门的避障路径"。

**核心矛盾**：真实部署里模型常被用作几何求解器或代码生成器的"前端"，但这些工具链能跑通的前提是模型本身有起码的空间直觉——若连无工具下的几何不变量都守不住，工具增强只会放大错误。LayoutGPT、FirePlace 等工作都在评估生成结果的真实感，没人系统测过模型在符号输入下的几何一致性。

**本文目标**：构造一个**纯符号、可自动判分、覆盖度量/拓扑/动态三大类**的诊断基准，定量回答"今天的 LLM 在结构化布局上能做什么、栽在哪、为什么"。

**切入角度**：把建筑师真实用的抽象（房间多边形 + 物体包围盒 + 开口）直接喂给模型，所有题目的 ground truth 都用确定性几何算法算出来（不依赖另一个模型评分），从而排除评估循环引用。

**核心 idea**：用 "**结构化 JSON 布局 + 模板化几何问题 + 规则判分**" 三件套，把空间推理从开放任务变成可逐项打分的诊断协议，并提出"度量-拓扑-动态"三层能力梯度。

## 方法详解

### 整体框架
FloorplanQA 想干一件事：给定一段纯符号的 2D 室内布局（JSON/XML，只有坐标、尺寸、语义标签，没有像素），逐项测出 LLM 到底能算清楚哪些几何性质、栽在哪。整套基准由三块拼成——2,000 个布局（1,800 个用 Gemini 2.5 Pro 自生成、200 个从 HSSD 真实场景抽取）、每个布局配 8 道共 16,000 道几何题、一套零样本评测协议（每题硬性要求 `Final answer` 行，正则解析后用对应度量自动打分）。所有布局共享一个右手 2D 坐标系，物体表示为带语义标签的多边形：合成数据是轴对齐矩形 4 顶点，HSSD 是任意多边形。

布局生成走两阶段流水线：先用 prompt 约束房间几何（形状、邻接、走廊净空、对称分区），再按风格化模板（如卧室必含床和储物）填充家具；约三分之一候选被规则验证器过滤掉（沙发挡门、冰箱穿桌等不合理摆放）。HSSD 来源的 3D 场景投影到 2D 后用 Douglas-Peucker（$\epsilon=0.01$m）简化多边形。

### 关键设计

**1. 三层任务分类法（Metric / Topology / Dynamic）：把排行榜变成能力剖面图**

传统单一难度的 QA 给一个均分，看不出模型究竟"在哪一档掉链子"。FloorplanQA 把 8 种题型按推理强度分成三层：Metric（成对距离、视角）只要算坐标；Topology（自由面积、最大可容矩形、放置可行性、可见性）要做集合运算和约束满足；Dynamic（重定位、最短路径）要在改变布局的过程中推理。这三层恰好对应"算术 / 约束推理 / 多步规划"三种典型失败模式，于是排行榜不再是一个数字，而是一张能力剖面。为了让判分确定，每题还配一个固定输出格式码——N（标量，相对误差 $\leq 2\%$，自由面积放宽到 $5\%$）、B（布尔，真值相等）、L（列表，集合相等）、S（序列，要求避障且 Fréchet 距离 $\leq 0.6$m）；模板里的对象名一律替换成布局里真实存在的家具实例（如 `fridge_1`、`table_3`），保证每道题的引用唯一无歧义。

**2. 去除评估循环的确定性判分：生成与判分彻底解耦**

让 Gemini 生成布局、又让 Gemini 来评测，会引入循环引用式的 leakage。FloorplanQA 从协议层面切断这条路：布局一次性预生成并冻结，所有 ground truth 由 shapely 等几何库精确算出（多边形并集面积、A* 路径、shoelace 公式算重心），与任何 LLM 输出无关。评测时模型被强制输出简短结构化推理 + `Final answer:` 行，正则抽取后直接比对参考答案，无答案或格式错误统一记 0。协议还记录 token 截断率（API `stop_reason = TOKEN LIMIT`），同时报"原始准确率"和"仅完成响应准确率"——前者防止推理模型靠超长输出刷分，后者作为性能上界。判分全程用代码而非模型，因此既客观又可复现。

**3. 多维消融探针（输入格式 / 语义扰动 / 工具增强 / VLM 渲染）：把失败拆成四个独立维度**

单一准确率会掩盖"模型为什么错"，所以本文设计四组消融，分别回答模型到底理解的是 JSON 语法还是几何语义。（a）JSON ↔ XML 等价改写，准确率波动在 $\pm 3$ pp 内，说明模型抓的是布局语义而非序列化语法；（b）物体标签置换（把"沙发"标签换给"椅子"但几何不变），Repositioning 任务从 $60.5\%$ 暴跌到 $40.0\%$（gpt-oss-120B），暴露移动类任务其实在吃语言关联而非几何属性；（c）GPT-4.1 接 Python interpreter，算术题提升 $+30$～$+43$ pp，但在 Max Box（$-4.5$）、Shortest Path（$-12.5$）反而退步；（d）三种渲染（标注框、图标、AI 生成照片）叠加 JSON 都在 JSON-only 基线 $\pm 4$ pp 内，纯图像则降到 $19\%$-$40\%$。四组探针把"算术错 / 算法错 / 视觉无用 / 语义偏置"四个独立维度分离开，让后续改进方向（接几何求解器 vs. 训几何约束目标）有据可依。

### 训练策略
本文是评测论文，不训练任何模型。所有 15 个 LLM 在 `temperature=0`、零样本、统一 prompt schema 下评估；大模型给 12,288 token，中小模型给 8,192 token，GPT-5 因强制开 reasoning 单独配 4,096 token。

## 实验关键数据

### 主实验
共评估 15 个模型（7 个 reasoning：GPT-5、gpt-oss-120B、DeepSeek-R1-0528、GPT-5-mini、Gemini Flash 2.5、gpt-oss-20B、Qwen3-30B-Thinking；8 个通用：Claude Sonnet 4、GPT-4.1、Kimi-K2、Qwen3-Coder-480B、Qwen3-235B、GPT-4.1-mini、Qwen3-30B、Devstral-Small），跑完整 16,000 题。整体结果按题型呈现三层梯度：

| 任务类别 | 代表题型 | 合成布局平均准确率 | HSSD 准确率 | 主要失败模式 |
|----------|----------|--------------------|-------------|-------------|
| Metric | Pair Distance / View Angle | 75-95\% | 35-60\% | HSSD 多边形重心算错（shoelace 公式失误） |
| Topology (约束) | Placement / Visibility / Repositioning | 60-85\% | 40-70\% | 多约束联合验证失败 |
| Topology (优化) + Dynamic | Free Space / Max Box / Shortest Path | 5-45\% | < 30\% | 把"面积和"当"面积并"、不会做避障搜索 |

### 消融实验

工具增强（GPT-4.1 + Python interpreter）在 HSSD 上的对比：

| 任务 | Raw | + Tools | $\Delta$ | 解读 |
|------|-----|---------|----------|------|
| Pair Distance | 56.0 | 99.0 | $+43.0$ | 算术瓶颈被工具解掉 |
| View Angle | 55.0 | 96.0 | $+41.0$ | 同上 |
| Visibility | 46.5 | 86.5 | $+40.0$ | 显式几何检查胜出 |
| Repositioning | 47.0 | 83.5 | $+36.5$ | 工具能跑碰撞检测 |
| Placement | 64.5 | 95.0 | $+30.5$ | 受益于精确判定 |
| Free Space | 16.0 | 44.0 | $+28.0$ | 并集仍是难点 |
| Max Box | 7.5 | 3.0 | $-4.5$ | 工具产生错误算法 |
| Shortest Path | 25.0 | 12.5 | $-12.5$ | 边界接触误判为碰撞 |

### 关键发现
- 厨房准确率最高——每个厨房布局平均仅 0.52 对物体重叠，而卧室/客厅是 1.52-1.82，HSSD 高达 4.39，重叠多则并集运算和净空判定显著变难。
- Reasoning 模型在 Free Space、Max Box 上比通用模型高 $+10$～$+40$ pp，但 Shortest Path 收益微薄——说明额外推理预算能救几何并集，救不了搜索类规划。
- Repositioning 在物体标签互换后大幅掉点（如 gpt-oss-120B 从 60.5% 降到 40.0%），证明模型依赖"沙发=大件家具"这类语言常识而非真实坐标。
- 工具增强反而让 Max Box / Shortest Path 退步——失败案例显示模型生成的 Python 代码里"边界接触被当成碰撞"或"矩形朝向枚举不全"，瓶颈是算法层面的空间推理，不是数值精度。

## 亮点与洞察
- **诊断协议而非排行榜**：把题目按推理类型而非难度分组，让基准能告诉你"模型在哪个能力维度短"，而不是单一一个均分数字。这种"能力剖面"做法可以迁移到代码、数学、逻辑等任意符号推理评测。
- **工具增强的"反直觉退化"**：直觉上 Python 解释器应该全面提升，结果在规划任务反而掉点。这反过来证明 LLM 的失败并非算术问题——给它笔和纸它会算错；给它计算器它会算对几何问题，但仍然写不出正确的搜索算法。这一观察对"agent + tool"范式是一个重要警告。
- **JSON vs. XML 几乎不影响、图像几乎不帮忙**：两个负面结果非常干净，说明 LLM 是从布局语义层面理解空间，序列化语法和视觉渲染都是表层包装；这意味着未来研究应该聚焦"几何归纳偏置"而非"换个 prompt 格式"。

## 局限与展望
- 布局只到 2D 室内场景且家具基本轴对齐，未覆盖 3D 高度差、多层建筑、室外场景或动态环境。
- 评测仅做零样本无 few-shot——可能低估了 in-context 几何示例的辅助作用。
- 1,800 个合成布局由 Gemini 2.5 Pro 生成，虽然评测时所有模型独立判分但布局分布带有该模型的几何偏好。
- 作者展望：（近期）接 A*/shapely 等专用求解器，（中期）训练时加入几何一致性目标和违例样本，（长期）多步交互评测——让 agent 自验证并修正方案。

## 相关工作与启发
- **vs CLEVR / SpatialSense**: 它们考视觉中的定性空间关系（"左/右/上/下"），FloorplanQA 考符号坐标下的定量几何（距离、面积、路径），任务类型互补，难度更纯粹。
- **vs ScanQA / 3DSRBench**: 它们针对 3D 场景的 VLM 问答，强调点云/网格理解；本文专注 2D 符号输入下纯 LLM 的几何推理。
- **vs LayoutGPT / Holodeck / FirePlace**: 那些工作是用 LLM 生成布局并评估输出真实感，本文是探针——给定布局，测模型能否推断布局的几何性质，方向正好相反。
- **vs Yamada et al. 2023 (Evaluating Spatial Understanding)**: 他们用自然语言描述定性关系评测，本文用结构化 JSON + 坐标评测精确几何，把模糊的"空间理解"细化为可量化的"度量/拓扑/动态"三层能力。

## 评分
- 新颖性: ⭐⭐⭐⭐ 题型分类法和"诊断协议"思路明确，但单看资源仍属于"benchmark 论文"的常规结构。
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个模型 × 8 题型 × 4 类布局 + 4 组消融（输入格式 / 语义扰动 / 工具增强 / VLM 渲染），全面到几乎可以一手覆盖未来一年的对比需求。
- 写作质量: ⭐⭐⭐⭐ 三层任务分类讲得很清晰，工具增强反向退化和重叠率与精度的相关分析尤其有洞察力。
- 价值: ⭐⭐⭐⭐⭐ 给"LLM + 空间推理"方向提供了第一个能精确分项判分的诊断基准，对 agent / 建筑设计 / 具身 AI 都有直接参考价值。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICML 2025\] Adversarial Manipulation of Reasoning Models using Internal Representations](../../ICML2025/llm_reasoning/adversarial_manipulation_of_reasoning_models_using_internal_representations.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](evaluating_relational_reasoning_in_llms_with_rel.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)

</div>

<!-- RELATED:END -->
