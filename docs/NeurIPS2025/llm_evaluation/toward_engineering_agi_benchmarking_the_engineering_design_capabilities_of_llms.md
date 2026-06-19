---
title: >-
  [论文解读] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs
description: >-
  [NeurIPS 2025][LLM评测][工程设计] 提出 EngDesign——首个跨 9 个工程领域（操作系统、计算机架构、控制系统、机械、结构、数字硬件、模拟电路、机器人、信号处理）的 LLM 工程设计能力基准，用仿真驱动的评估管线替代传统的问答匹配，揭示即使最强推理模型 o3 也仅达 34% 通过率。
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "工程设计"
  - "LLM基准测试"
  - "仿真评估"
  - "多领域工程"
  - "AGI"
---

# Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2509.16204](https://arxiv.org/abs/2509.16204)  
**代码**: [https://agi4engineering.github.io/Eng-Design/](https://agi4engineering.github.io/Eng-Design/)  
**领域**: 机器人  
**关键词**: 工程设计, LLM基准测试, 仿真评估, 多领域工程, AGI

## 一句话总结

提出 EngDesign——首个跨 9 个工程领域（操作系统、计算机架构、控制系统、机械、结构、数字硬件、模拟电路、机器人、信号处理）的 LLM 工程设计能力基准，用仿真驱动的评估管线替代传统的问答匹配，揭示即使最强推理模型 o3 也仅达 34% 通过率。

## 研究背景与动机

**领域现状**：LLM 在传统问答基准（MMLU、HumanEval、GPQA 等）上已表现出色，在教科书级别的工程问题回答上也有不少探索。但现有评测都聚焦于知识回忆和标准答案匹配，未涉及真正的工程设计能力——后者需要综合领域知识、约束推理、权衡折衷，并通过专业仿真验证功能可行性。

**现有痛点**：(1) 现有工程类基准局限于单一领域的事实问答，无法反映真实设计工作的复杂性；(2) 评估方式依赖精确匹配或 LLM-as-judge，但工程设计问题通常没有唯一正确答案（如无穷多个有效控制器设计）；(3) 缺乏跨多个工程学科的统一评测平台。

**核心矛盾**：现实中的工程设计是开放式的——给定性能指标和约束条件，可能存在多种有效方案。但现有基准用封闭式的问答评估这种开放式能力，导致评估结论不可靠。需要一种能验证"设计是否可行"而非"答案是否正确"的评估方式。

**本文目标** 构建一个覆盖多工程领域的基准，通过仿真驱动的评估管线客观衡量 LLM 的实际工程设计能力。

**切入角度**：不让 LLM 做选择题，而是让它"像工程师一样设计"——输出控制器参数、电路设计、GPU 架构代码等，然后用 SPICE 仿真、MATLAB 控制系统工具箱、有限元分析等专业工具自动验证设计是否满足性能要求。

**核心 idea**：用仿真驱动的评估管线替代静态答案匹配，在 9 个工程领域的 101 个开放式设计任务上评测 LLM 的真正工程设计能力。

## 方法详解

### 整体框架

EngDesign 包含 101 个设计任务，横跨 9 个工程领域，共 473 个评分项。每个任务由四部分组成：(1) 任务描述（作为 LLM prompt，平均 779 tokens）；(2) 评分规则（多项评分，满分 100，支持部分得分）；(3) 评估管线（自动化仿真脚本）；(4) 参考设计（验证可行性的标准解）。LLM 输出经过结构化解析后送入任务特定的仿真评估管线，得到通过/失败判定、0-100 分数和详细日志。

### 关键设计

1. **仿真驱动评估管线**:

    - 功能：通过专业仿真工具客观验证 LLM 生成的工程设计是否满足功能要求
    - 核心思路：每个任务配备特定的评估脚本——控制系统用 MATLAB 仿真闭环响应（检查上升时间、超调量、相位裕度等）；模拟电路用 SPICE 仿真验证增益、带宽等指标；结构设计用有限元分析；数字硬件进行功能仿真等。评估输出三样东西：通过/失败二值判断、0-100 细粒度得分、仿真日志。67 个任务完全开源（EngDesign-Open），34 个需要商业仿真工具（MATLAB、Cadence 等）
    - 设计动机：传统基准用字符串匹配或 LLM 评判，无法可靠评估开放式设计的功能正确性。仿真评估是工程领域的金标准，把评估从"文本看起来对不对"升级为"设计实际能不能工作"

2. **结构化 LLM 输出机制**:

    - 功能：确保不同 LLM 的输出格式统一，可被评估管线自动解析
    - 核心思路：使用 Python instructor 库（基于 Pydantic），定义 schema 模板指定期望字段（如设计参数、代码片段等）。LLM 输出被约束为两部分：reasoning 字段（推理过程）和 ConfigFile 类（设计结果），后者自动解析后触发仿真
    - 设计动机：评估管线需要程序化地提取设计参数，自由格式的 LLM 输出无法可靠解析

3. **部分给分机制和多阶段质控**:

    - 功能：细粒度量化设计质量，识别模型在哪些方面部分成功
    - 核心思路：每个任务分解为多个评分项（平均约 4.7 项/任务），例如控制器设计可能包含"稳定性 20 分 + 上升时间 20 分 + 超调量 20 分 + 稳态误差 20 分 + 鲁棒性 20 分"。即使整体不过关，模型在某些子指标上的成功也能被记录。基准构建经过初始任务设计→LLM 过滤→第一轮评审→领域专家评审→最终集成五个阶段
    - 设计动机：二值评分掩盖了模型在部分能力上的进步。部分给分更细致地揭示能力差距，为改进提供方向

### 损失函数 / 训练策略

不涉及训练。评估协议为每个任务独立运行 LLM 3 次，报告通过率和平均得分。还设计了迭代式评估——将前一轮的设计输出和评估反馈作为新 prompt 给 LLM，模拟工程师的迭代改进过程。

## 实验关键数据

### 主实验

| 模型 | 类型 | Overall Pass% | OS | Ctrl | DHD | Robo | SigP | Stru |
|------|------|---------------|-----|------|-----|------|------|------|
| GPT-4o | Chat | 15.7 | 4.2 | 18.5 | 10.3 | 26.7 | 17.7 | 25.6 |
| Claude-3.7-Sonnet | Chat | 22.6 | 0.0 | 16.7 | 33.3 | 33.3 | 21.6 | 30.8 |
| o1 | Reasoning | 29.2 | 37.5 | 24.1 | 41.0 | 50.0 | 25.5 | 23.1 |
| **o3** | Reasoning | **34.4** | 25.0 | 35.2 | 20.5 | **63.3** | **41.2** | 30.8 |
| o4-mini-high | Reasoning | 34.0 | 37.5 | 27.8 | **47.2** | 46.7 | 35.3 | 35.9 |
| DeepSeek-R1 | Reasoning | 25.5 | 5.3 | 36.4 | 38.5 | 26.7 | 20.5 | 41.7 |
| Gemini-2.5-Pro | Reasoning | 29.5 | 9.5 | 33.3 | 43.6 | 56.7 | 12.8 | **50.0** |

### 消融实验（迭代设计）

| 模型 | 1轮 Pass% | 5轮 Pass% | 10轮 Pass% |
|------|-----------|-----------|------------|
| GPT-4o | ~14 | ~25 | ~30 |
| o1 | ~26 | ~40 | ~48 |
| o3 | ~30 | ~48 | ~58 |
| o4-mini | ~28 | ~42 | ~50 |

### 关键发现

- **模拟 IC 设计是所有模型的绝对盲区**：所有 12 个模型在 Analog IC Design 上的通过率均为 0%，即使 10 轮迭代也无法攻克。这反映了模拟电路设计需要极其精细的物理直觉和参数调优经验，目前 LLM 完全不具备
- **推理模型显著优于对话模型**：o3 (34.4%) vs GPT-4o (15.7%)，差距超过 2 倍。推理模型的鲁棒性也更强——o1 的推理鲁棒性达 0.62，而 Gemini-2.0-Flash 仅 0.20
- **迭代设计大幅提升性能**：o3 从单轮 ~30% 提升到 10 轮 ~58%，说明 LLM 能从仿真反馈中学习并改进设计，模拟了工程师的迭代工作流
- **主要失败模式**：领域知识不足（DKE）和约束违反（CVE）合占 55-67% 的失败，过度依赖先验知识（PKO）和幻觉（HAL）占 25-30%，计算错误（CE）仅占 <9%
- 不同模型在不同领域的优势差异很大：Claude 在数字硬件强于 o3，但在信号处理上明显弱于 o3

## 亮点与洞察

- **开创性的评估范式转变**：从"答案对不对"到"设计能不能用"，这可能是 LLM 评估在工程领域最重要的方法论贡献。仿真驱动的评估完全客观可复现，消除了 LLM-as-judge 的主观性
- **部分给分的精细粒度**：传统 benchmark 的二值评分过于粗糙，EngDesign 的多维评分能精确定位模型在哪个具体工程环节失败——这对指导模型改进极有价值
- **迭代设计协议**：模拟真实工程师的工作流程，o3 在 10 轮迭代后达到 ~58%，这提示反馈驱动的 agent 范式可能是 LLM 在工程任务中的正确使用方式

## 局限与展望

- 34 个任务需要 MATLAB/Cadence 等商业软件，限制了完全开放复现
- 101 个任务的规模相对有限，某些领域（如模拟 IC 设计仅 5 个任务）样本太少
- 任务分布不均匀反映了贡献者的研究方向偏好而非工程领域的真实权重分布
- 23 个任务包含图像输入，纯文本模型（如 DeepSeek-R1/v3）在这些任务上天然受限
- 未评估多模型协同（如 agent 框架）的效果，仅测试单模型单次或迭代生成

## 相关工作与启发

- **vs MMLU/GPQA 等 QA 基准**：这些基准测试知识回忆，EngDesign 测试知识应用和设计综合能力。两者衡量的是 LLM 能力图谱中完全不同的维度
- **vs HumanEval**：HumanEval 测试代码生成的功能正确性，EngDesign 将类似的程序化验证思路扩展到工程设计领域，但设计问题比编程更开放
- **vs ControlAgent/AnalogCoder 等领域特定工作**：这些工作在特定领域用 LLM 辅助设计，EngDesign 提供了跨领域的统一评测框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个跨多领域的仿真驱动工程设计基准，评估范式具有开创性
- 实验充分度: ⭐⭐⭐⭐ 12 个模型 × 101 个任务 × 3 轮 + 迭代实验，但任务规模还可以更大
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，结果分析透彻，错误分类学很有见地
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 在工程领域的能力边界提供了严肃的评估，揭示了当前模型与真正"AI工程师"之间的巨大差距

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction](../../ICML2026/llm_evaluation/buildarena_a_physics-aligned_interactive_benchmark_of_llms_for_engineering_const.md)
- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](../../ICLR2026/llm_evaluation/human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](../../ACL2026/llm_evaluation/engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](../../ACL2026/llm_evaluation/bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)

</div>

<!-- RELATED:END -->
