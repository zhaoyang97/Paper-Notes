---
title: >-
  [论文解读] CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning
description: >-
  [NeurIPS 2025][代码智能][LLM 鲁棒性] 提出 CodeCrash 压力测试框架，通过功能等价的结构扰动和误导性自然语言注入（注释/print/暗示），系统评估 17 个 LLM 的代码推理鲁棒性，揭示模型平均性能下降 23.2%，CoT 仅能挽回至 13.8%，并首次发现大推理模型（LRM）中的 "Reasoning Collapse" 现象。
tags:
  - "NeurIPS 2025"
  - "代码智能"
  - "LLM 鲁棒性"
  - "代码推理"
  - "自然语言扰动"
  - "推理崩溃"
  - "benchmark"
---

# CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2504.14119](https://arxiv.org/abs/2504.14119)  
**代码**: [Website](https://cuhk-arise.github.io/CodeCrash/)  
**领域**: 医学图像  
**关键词**: LLM 鲁棒性, 代码推理, 自然语言扰动, 推理崩溃, benchmark

## 一句话总结
提出 CodeCrash 压力测试框架，通过功能等价的结构扰动和误导性自然语言注入（注释/print/暗示），系统评估 17 个 LLM 的代码推理鲁棒性，揭示模型平均性能下降 23.2%，CoT 仅能挽回至 13.8%，并首次发现大推理模型（LRM）中的 "Reasoning Collapse" 现象。

## 研究背景与动机

**领域现状**：LLM 在代码生成、补全、修复等任务上表现卓越，已被集成到 GitHub Copilot 等 IDE 工具中。然而真实代码库中充满歧义标识符、无用代码和不一致注释等噪声。

**现有痛点**：传统鲁棒性研究主要关注代码结构变换（重命名变量、修改控制流、插入不可达代码），这些仅测试模式匹配能力；NL 方向的扰动研究集中在 NL-to-Code 任务（评估提示敏感性），无人系统评估 LLM 是否能在代码推理中优先考虑可执行语义而非自然语言线索。

**核心矛盾**：LLM 在代码理解中将注释和自然语言线索视为高优先级证据，但这些在程序执行中语义无关——模型无法区分功能性代码和非功能性上下文。

**本文目标**：设计一个覆盖结构层和 NL 层的代码推理鲁棒性 benchmark，剖析 LLM 在面对误导信息时的失败模式。

**切入角度**：将扰动分为上下文级（明显错误的 NL 线索）和推理级（貌似合理但错误的暗示），创造性地利用 AST 解析保证扰动后代码功能等价。

**核心 idea**：LLM 过度依赖 NL 线索走捷径推理 → CoT 能缓解但不能根除 → LRM 的内部推理虽更鲁棒但在合理暗示下触发病态自省和推理崩溃。

## 方法详解

### 整体框架
输入代码经 AST 解析标准化为"vanilla"基线（VAN），然后施加指定扰动，再经 AST 重新生成确保语法正确和功能等价。扰动后的代码送入 LLM，记录其预测输出并与实际执行结果对比，计算 Pass@1 准确率。数据来源：CruxEval（1,279 题中的代码推理题）+ LiveCodeBench（真实算法题）。

### 关键设计

1. **聚合结构扰动（PSC-ALL）**:

    - 功能：同时应用标识符重命名（REN）、条件表达式重格式化（RTF）和垃圾代码注入（GBC，含死循环、死块、全局变量）
    - 核心思路：从 CCTest 和 dead-loop poisoning 方法中聚合多种 PSC 变换，构造更复杂但功能等价的程序
    - 设计动机：单独的结构扰动效果一致较弱，聚合后代表"传统方法的最强组合"，作为对照基线

2. **上下文级误导扰动（MCC & MPS）**:

    - 功能：在不改变代码逻辑的前提下注入明显错误的 NL 线索
    - MCC（Misleading Code Comments）：在 8 种关键 AST 节点处注入错误注释（如"这个分支永不执行"）
    - MPS（Misleading Print Statements）：将相同误导信息嵌入可执行的 print 语句
    - 设计动机：两种注入格式（注释 vs print）用于验证 LLM 的脆弱性不限于特定格式
    - 利用 GPT-4o 生成误导消息，人工过滤确保通用、明显错误且与代码逻辑矛盾

3. **推理级误导扰动（MHC）**:

    - 功能：注入关于程序输出的高层级错误暗示（Misleading Hint Comments）
    - 核心思路：用 GPT-4o 生成保持类型和结构但值错误的"合理答案"，作为注释注入到函数定义或 return 语句处
    - 设计动机：不同于上下文级扰动的浅层矛盾，MHC 瞄准推理过程——测试模型能否批判性地评估冲突信息而非合理化暗示走捷径
    - 实验发现：在 LCB（更复杂算法题）上 MHC 导致平均 33.0% 性能下降，远超 Crux 的 12.4%——任务越复杂，模型越愿意采纳暗示

4. **评估设置**:

    - 17 个 LLM（GPT-4o/mini、Claude 3.5、Gemini、DeepSeek-V3、LLaMA-3.x、Qwen-2.5 系列）
    - 3 个 LRM（o3-mini、DeepSeek-R1、QwQ-32B）
    - 两种推理模式：直接推理 + CoT
    - 温度 0.2，top-p 0.95

### 失败模式分析

**Distractibility**：模型在 CoT 推理中途被注释/print 吸引，放弃先前正确的推理路径。例如 Qwen2.5-72B 在正确判断条件后，读到"此分支永不执行"的注释，立刻推翻自己的结论。

**Rationalization**：模型接受 MHC 暗示后反向构造解释来合理化错误答案。例如模型对第四个字符推理正确（'w'），但在下一步突然改变结论以对齐暗示的错误输出。

**Reasoning Collapse**：QwQ-32B 特有现象——在 MHC 扰动下触发不可控的递归自我验证，产生 32K+ token 的推理过程，最后 12K token 全是"Hmm"。这是认知失调的表现：模型试图同时合理化暗示和忠于推理轨迹，两者冲突导致崩溃。

## 实验关键数据

### 主实验（直接推理 - 输出预测相对下降）

| 模型 | VAN (Crux/LCB) | PSC-ALL | MCC | MPS | MHC | 平均 |
|------|----------------|---------|-----|-----|-----|------|
| GPT-4o | 71.3/64.5 | -15.0%/-28.4% | -14.0%/-29.1% | -17.2%/-24.3% | -6.4%/-26.6% | -18.4% |
| Claude 3.5 Sonnet | 71.5/73.8 | -14.8%/-34.1% | -8.1%/-9.6% | -8.6%/-10.7% | -14.4%/-43.4% | -16.3% |
| DeepSeek-V3 | 67.9/67.8 | -12.9%/-35.6% | -16.6%/-41.4% | -10.1%/-34.2% | -10.7%/-29.7% | -21.1% |
| Qwen2.5-7B | 43.3/41.4 | -37.9%/-30.9% | -58.0%/-38.3% | -45.2%/-19.8% | -26.9%/-55.6% | -39.8% |
| **全部 17 模型平均** | - | **-24.6%** | **-24.3%** | **-23.8%** | **-20.1%** | **-23.2%** |

### CoT 推理对比

| 模型 | 直接推理平均下降 | CoT 平均下降 | 挽回幅度 |
|------|----------------|-------------|---------|
| GPT-4o | -18.4% | -4.2% | +14.2 pp |
| Claude 3.5 Sonnet | -16.3% | -6.7% | +9.6 pp |
| DeepSeek-V3 | -21.1% | -12.4% | +8.7 pp |
| LLaMA-3.1-8B | -23.0% | -19.8% | +3.2 pp |
| **全部平均** | **-23.2%** | **-13.8%** | **+9.4 pp** |

### LRM 鲁棒性与 token 消耗

| 模型 | VAN Pass@1 | PSC-ALL | MCC | MPS | MHC | MHC 最大 token |
|------|-----------|---------|-----|-----|-----|---------------|
| o3-mini-high | 98.1/100 | +0.1%/-0.2% | -3.6%/-5.6% | +0.9%/-0.6% | -13.4%/-28.4% | 20,000 |
| DeepSeek-R1 | 95.4/99.8 | -1.3%/-1.3% | -3.5%/-2.7% | -0.4%/-0.4% | -2.4%/-0.6% | 16,079 |
| QwQ-32B | 93.2/99.0 | -0.9%/-0.2% | -3.4%/-4.6% | -1.2%/-1.9% | -0.8%/-1.1% | **32,764** |

LRM 在 PSC-ALL 和 MCC/MPS 下几乎不受影响，但 MHC 下 token 消耗增加 2-3 倍，QwQ-32B 出现 4 次 Reasoning Collapse。

### 关键发现
- NL 嵌入扰动（MCC/MPS）效果与结构扰动（PSC-ALL）相当（~24% 下降），说明 LLM 同等程度地受注释和代码结构干扰
- MHC 在复杂任务（LCB）上的影响（-33.0%）远超简单任务（Crux，-12.4%）——任务越难，模型越倾向走捷径
- 模型规模和版本升级系统性地改善鲁棒性（LLaMA 8B → 70B → 405B），但 Gemini 系列是例外
- MPS vs MCC 14/17 模型表现出一致的扰动方法偏好，说明格式偏好是固有的

## 亮点与洞察
- **从代码视角揭示 LLM 的"合理化"行为**：之前关于 LLM rationalization 的发现主要来自 Anthropic 在多选题场景的研究，CodeCrash 将其扩展到代码推理领域——模型不仅接受暗示，还会构造虚假的执行路径来"证明"暗示的正确性
- **Reasoning Collapse 是全新的失败模式**：不同于简单的幻觉或重复生成，它是推理模型试图同时满足两个矛盾目标（忠实推理 vs 合理化暗示）时产生的灾难性认知失调，表现为推理 token 的二次增长后崩溃
- **输入预测 vs 输出预测的不对称性**：MCC/MPS 在输入预测任务上反而可能提升性能（误导线索提供了可利用的答案信息），暴露了输入预测作为代码理解评估手段的局限性

## 局限与展望
- 误导消息由 GPT-4o 生成，可能引入特定的语义偏差
- 仅评估了输入/输出预测两类任务，未覆盖代码生成、修复等下游任务
- LRM 评估仅 3 个模型（o3-mini、R1、QwQ-32B），采样次数为 N=1 受资源限制
- 未探索防防御策略（如训练时加入 NL 干扰作为数据增强）
- Reasoning Collapse 目前仅在 QwQ-32B 上观察到 4 例，统计功效有限

## 相关工作与启发
- **vs CCTest**：CCTest 引入了 PSC 变换但仅限于结构层面；CodeCrash 系统整合了 NL 层扰动，发现 NL 扰动与结构扰动表面效应相当但失败模式截然不同
- **vs Anthropic Sycophancy 研究**：Anthropic 发现 LLM 在多选题中倾向合理化偏置暗示；CodeCrash 表明同样的 rationalization 在代码推理中更为危险，因为代码有确定性的执行语义
- **vs Prompt Injection 研究**：CodeCrash 中的 MHC 可被视为一种代码域的 prompt injection——通过合理化的元信息操控模型行为

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究 NL 嵌入扰动对代码推理的影响，发现 Reasoning Collapse 这一全新现象
- 实验充分度: ⭐⭐⭐⭐⭐ 17 个 LLM + 3 个 LRM，4 种扰动类型 × 2 种推理模式 × 2 个数据集，规模宏大
- 写作质量: ⭐⭐⭐⭐ 结构清晰，失败案例分析详细，但论文较长
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 在代码辅助场景的可靠性评估和改进方向有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](../../ACL2026/code_intelligence/storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[NeurIPS 2025\] A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)
- [\[NeurIPS 2025\] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)

</div>

<!-- RELATED:END -->
