---
title: >-
  [论文解读] From Charts to Code: A Hierarchical Benchmark for Multimodal Models
description: >-
  [ACL 2026][代码智能][图表生成] 本文提出 Chart2Code，一个包含 2,186 个任务、覆盖 22 种图表类型的层次化基准，分为图表复现（Level 1）、图表编辑（Level 2）和长表格转图表（Level 3）三个递进难度级别，评测 29 个 SOTA 多模态模型，发现即使最强的 GPT-5.2 在编辑任务上的图表质量评分仅 33.41，揭示了当前模型在实际图表代码生成中的显著不足。
tags:
  - "ACL 2026"
  - "代码智能"
  - "图表生成"
  - "代码生成"
  - "多模态基准"
  - "层次化评估"
  - "视觉忠实度"
---

# From Charts to Code: A Hierarchical Benchmark for Multimodal Models

**会议**: ACL 2026  
**arXiv**: [2510.17932](https://arxiv.org/abs/2510.17932)  
**代码**: [GitHub](https://github.com/CSU-JPG/Chart2Code)  
**领域**: 代码智能 / 多模态理解  
**关键词**: 图表生成, 代码生成, 多模态基准, 层次化评估, 视觉忠实度

## 一句话总结

本文提出 Chart2Code，一个包含 2,186 个任务、覆盖 22 种图表类型的层次化基准，分为图表复现（Level 1）、图表编辑（Level 2）和长表格转图表（Level 3）三个递进难度级别，评测 29 个 SOTA 多模态模型，发现即使最强的 GPT-5.2 在编辑任务上的图表质量评分仅 33.41，揭示了当前模型在实际图表代码生成中的显著不足。

## 研究背景与动机

**领域现状**：图表是科学论文和商业报告中最重要的可视化工具。随着大型多模态模型（LMM）的快速发展，AI 系统不仅能理解图表，还能生成可执行的绑图代码（chart-to-code），有望大幅提升生产力。

**现有痛点**：(1) 现有基准（如 ChartMimic）已趋于性能饱和——GPT-4o 在 ChartMimic 上已达 82.2%，无法区分当前和未来模型的能力；(2) 缺乏覆盖真实使用场景的系统性基准——用户不仅需要复现图表，更常需要编辑图表（换类型、加元素）和从原始长表格生成图表，但这些场景未被充分测试；(3) 现有评估主要关注代码正确性，忽略了渲染出的图表的视觉忠实度。

**核心矛盾**：现有基准报告的高分与模型在实际使用中的表现之间存在巨大差距——模型在简单复现上得分很高，但在更常见的编辑和数据转图表场景中表现大幅下降，现有基准无法暴露这些问题。

**本文目标**：(1) 构建一个从用户视角出发的层次化图表代码生成基准；(2) 设计多层次评估协议（代码级 + 图表级）；(3) 全面评测 29 个 SOTA 多模态模型。

**切入角度**：从真实用户工作流出发设计三个递进难度级别——简单复现→复杂编辑→长表格转图表——逐步增加对模型理解、推理和代码生成能力的要求。

**核心 idea**：将图表代码生成从单一的复现任务扩展为覆盖完整用户工作流的层次化基准，同时引入代码级和图表级双层评估来全面衡量生成质量。

## 方法详解

### 整体框架

Chart2Code 把图表代码生成形式化为 $C = f(R, I, D)$：给定参考图表 $R$、用户自然语言指令 $I$ 和可选数据源 $D$（支持文本、表格截图、Excel 三种模态），多模态模型 $f$ 要生成可执行的 Python 绘图代码 $C$。基准沿真实用户工作流铺成「复现→编辑→长表格转图」三级递进难度，并对产物同时做代码级（规则化 Base 评分 + LLM 评分）和图表级（LMM 视觉评分 + 人类评估）的双层打分，以暴露"代码对但图不对"的隐藏差距。

### 关键设计

**1. 三级层次化任务设计：把用户工作流拆成递增难度**

现有基准只测最简单的复现，而用户真实需求是"看图画图→改图→看数据画图"。Chart2Code 据此分三级：Level 1 图表复现含纯视觉复现 DR（只给图）和带数据的风格迁移 CRD/CFD（给图+数据），共 863 个任务；Level 2 图表编辑要求换图表类型、加趋势线、算相关系数、按类别拆分等复杂修改，共 1,010 个任务；Level 3 长表格转图表要从平均 2,647 行的原始长表中提取、计算再绘图，共 313 个任务。难度逐级叠加理解、推理与长上下文检索，正好卡在当前模型的能力边界上。

**2. 多层次评估协议：代码对不代表图对**

相同数据可以用不同代码路径渲染出视觉差异很大的图，所以仅看代码相似度会高估质量。评估因此分三路：代码级 Base 评分解析 Matplotlib 等库的 Figure 对象，从颜色、网格、布局、图例、视觉元素、数据、文本、类型共 8 个维度做规则化打分（比 ChartMimic 的 4 维度更全、更快、更准）；代码级 LLM 评分用 GPT-5-mini 从代码层面评视觉忠实度；图表级 LMM 评分再用 GPT-5-mini 对比 GT 图与生成图的多维相似性，并辅以人类评估校验 LMM 打分的一致性。

**3. 丰富的数据多样性：防止单一能力刷分**

基准覆盖 22 种图表类型（雷达图、热力图、散点图、箱线图、树图、误差条、饼图、小提琴图等），并在每一级强化区分度：Level 1 强调图表唯一性（719 个独特图表），Level 2 给每个图至少配一条编辑指令（1,010 个独特编辑，平均指令长达 267 词、为各级最长，反映编辑任务的复杂度），Level 3 含 71 个 Excel 文件（最长 30,427 行）。多类型、多任务的覆盖保证基准不会被某种单一能力的模型刷穿。

### 损失函数 / 训练策略

本文为基准测试工作，不涉及模型训练，所有模型用其公开权重或 API 推理：开源模型在 NVIDIA V100 GPU 上运行，非 thinking 模型推理长度设为 4,096 tokens，图像一律以原始分辨率输入。

## 实验关键数据

### 主实验

**Level 1 图表复现（部分模型）**

| 模型 | DR Exec% | DR Base | DR LMM | CRD Base | CFD Base |
|------|----------|---------|--------|----------|----------|
| GPT-5.2 | 97.08 | 79.91 | 43.73 | 66.31 | 73.02 |
| Gemini-3-Pro | 97.50 | 78.65 | 45.42 | 69.23 | 70.78 |
| Claude-Sonnet-4 | 96.52 | 65.60 | 32.36 | 61.46 | 65.27 |
| Qwen3-VL-32B | - | - | - | - | - |
| InternVL-3-38B | 85.26 | 53.57 | 16.68 | 58.17 | 60.17 |

**Level 2 图表编辑（8 维度代码级评分）**

| 模型 | Exec% | Color | Data | Text | Type | Base | LMM |
|------|-------|-------|------|------|------|------|-----|
| GPT-5.2 | 96.04 | 58.44 | 64.66 | 83.77 | 94.52 | 70.93 | 33.03 |
| Gemini-3-Pro | 97.23 | 52.32 | 62.75 | 77.16 | 93.86 | 70.78 | 33.41 |
| Claude-Sonnet-4 | 90.20 | 47.17 | 54.88 | 80.52 | 93.29 | 63.65 | 25.40 |

### 消融实验

**代码级 vs 图表级评分差距**

| 模型 | Level 2 代码级 Base | Level 2 图表级 LMM | 差距 |
|------|-------------------|-------------------|------|
| GPT-5.2 | 70.93 | 33.03 | -37.90 |
| Gemini-3-Pro | 70.78 | 33.41 | -37.37 |
| Claude-Sonnet-4 | 63.65 | 25.40 | -38.25 |

**开源模型 thinking vs non-thinking（Level 2，MiMo-VL-7B）**

| 配置 | Base | LMM |
|------|------|-----|
| MiMo-VL-7B-SFT (non-thinking) | 63.99 | 22.61 |
| MiMo-VL-7B-RL (non-thinking) | 64.41 | 21.43 |
| MiMo-VL-7B-SFT (thinking) | 65.49 | 16.95 |
| MiMo-VL-7B-RL (thinking) | 70.45 | 30.04 |

### 关键发现

- 从 Level 1 到 Level 2/3，所有模型性能显著下降——即使 GPT-5.2 在编辑任务上的图表质量评分（LMM）仅 33.03，说明编辑和数据转图表远未解决
- 代码级评分与图表级评分存在巨大差距（约 37-38 分），说明代码层面的正确性不能代表视觉层面的忠实度
- 闭源模型（GPT-5.2、Gemini-3-Pro）在所有级别上大幅领先开源模型，开源模型中 Qwen3-VL-32B 和 InternVL-3.5-38B 表现最佳
- Thinking 模式对部分模型有帮助（MiMo-VL-7B-RL thinking LMM 30.04 vs non-thinking 21.43），但不是所有模型都受益
- Level 3 长表格任务极具挑战性，需要长上下文理解、信息检索、数学计算和代码生成的综合能力

## 亮点与洞察

- 层次化设计精准对应了用户工作流的三个阶段——从"看图画图"到"改图"到"看数据画图"，难度递增揭示了模型真正的能力边界
- 代码级和图表级评估的巨大差距是重要发现——提示社区不能仅靠代码相似度来评估图表生成质量
- 8 维度规则化评分可迁移到其他图表/代码生成任务中使用

## 局限与展望

- 评估依赖 GPT-5-mini 作为 judge，存在 LMM 评估偏差
- Level 3 的标注和 GT 代码构建极为耗时，限制了数据规模（仅 313 个任务）
- 仅评估了 Python/Matplotlib 生态的代码生成，未覆盖 R、D3.js 等其他绑图工具
- 未测试模型的迭代修正能力——实际工作中用户会多轮交互修正图表

## 相关工作与启发

- **vs ChartMimic**: ChartMimic 仅测试图表复现（Level 1），且已被 GPT-4o 达到 82.2% 基本饱和；Chart2Code 扩展到编辑和长表格场景，且当前 SOTA 仅 33 分
- **vs ChartEdit**: ChartEdit 仅覆盖简单局部编辑（233 个样本），Chart2Code 的 Level 2 包含 1,010 个复杂编辑任务（加趋势线、算相关系数等）
- **vs Plot2Code**: Plot2Code 仅 132 个样本且无规则化评估，Chart2Code 规模和评估体系都显著更全面

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个层次化图表代码生成基准，任务设计贴合真实用户需求
- 实验充分度: ⭐⭐⭐⭐⭐ 29 个模型的全面评测，多层次评估协议，人类评估验证
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，数据统计丰富，但表格较多需反复查阅
- 价值: ⭐⭐⭐⭐⭐ 揭示了图表代码生成的真正难点，为社区指明了改进方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](../../CVPR2026/code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[NeurIPS 2025\] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](../../NeurIPS2025/code_intelligence/table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[AAAI 2026\] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](../../AAAI2026/code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)

</div>

<!-- RELATED:END -->
