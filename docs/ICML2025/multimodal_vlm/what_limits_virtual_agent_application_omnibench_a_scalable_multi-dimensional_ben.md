---
title: >-
  [论文解读] What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities
description: >-
  [ICML 2025 (Oral)][多模态VLM][virtual agent] 本文提出 OmniBench——一个基于图结构的可扩展虚拟 Agent 基准，通过自动化流水线合成可控复杂度的任务，配合 OmniEval 多维评估框架，在 20 个应用场景中生成 36K 个任务，系统揭示了虚拟 Agent 在不同能力维度上的短板。
tags:
  - "ICML 2025 (Oral)"
  - "多模态VLM"
  - "virtual agent"
  - "benchmark"
  - "graph-based tasks"
  - "multi-dimensional evaluation"
  - "automated pipeline"
---

# What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities

**会议**: ICML 2025 (Oral)  
**arXiv**: [2506.08933](https://arxiv.org/abs/2506.08933)  
**代码**: [https://omni-bench.github.io/](https://omni-bench.github.io/)  
**领域**: 多模态VLM  
**关键词**: virtual agent, benchmark, graph-based tasks, multi-dimensional evaluation, automated pipeline

## 一句话总结
本文提出 OmniBench——一个基于图结构的可扩展虚拟 Agent 基准，通过自动化流水线合成可控复杂度的任务，配合 OmniEval 多维评估框架，在 20 个应用场景中生成 36K 个任务，系统揭示了虚拟 Agent 在不同能力维度上的短板。

## 研究背景与动机

**领域现状**: 基于多模态大语言模型（MLLM）的虚拟 Agent 近年来取得了显著进展，能够在手机、电脑等平台上自主完成界面操作任务。评估这些 Agent 的 benchmark（如 AITW、OSWorld）越来越多。

**现有痛点**: 现有基准面临三大限制：
   - **任务复杂度不可控**: 手工标注的任务复杂度参差不齐，难以系统分析复杂度对性能的影响
   - **规模和多样性有限**: 人工标注成本高，导致场景覆盖不足（通常只有几百个任务）
   - **缺乏多维评估**: 只关注最终任务成功率，无法诊断 Agent 在具体能力维度（如空间推理、长期规划）上的短板

**核心矛盾**: 如何在保证任务质量的同时实现大规模、多场景、多维度的 Agent 评估？人工标注质量高但不可扩展，自动生成可扩展但质量难控。

**本文目标**: 构建一个既可扩展又能多维评估虚拟 Agent 能力的基准。

**切入角度**: 利用图（graph）结构表示任务——每个节点为原子操作（subtask），边为依赖关系。通过子任务组合自动合成可控复杂度的复合任务。

**核心 idea**: 将 Agent 任务建模为有向无环图（DAG），通过图的拓扑结构自动控制任务复杂度，通过图的节点类型自然引入多种能力维度的评估。

## 方法详解

### 整体框架
输入：20 个预定义场景的原子操作库
输出：36K 个图结构化任务 + 多维评估结果

Pipeline:
1. **原子操作定义**: 为每个场景定义基本操作（如"点击按钮"、"输入文本"、"滑动屏幕"）
2. **图合成**: 自动组合原子操作为图结构任务，通过图的深度、宽度、分支因子控制复杂度
3. **环境实例化**: 将抽象图任务实例化为具体的应用场景
4. **多维评估**: 用 OmniEval 从多个维度评估 Agent

### 关键设计

1. **图结构任务合成（Graph-Based Task Synthesis）**:

    - 功能：自动生成复杂度可控的任务
    - 核心思路：任务 = DAG $G = (V, E)$，其中 $V$ 为原子操作集合，$E$ 为依赖关系。图的拓扑特征决定了任务复杂度：
        - **深度**: 最长操作序列，反映长期规划需求
        - **宽度**: 最大并行操作数，反映多目标管理能力
        - **分支**: 条件分支，反映决策推理能力
    - 关键公式：任务复杂度 $C(G) = f(\text{depth}, \text{width}, \text{branching})$
    - 设计动机：图结构提供了一个灵活且可解释的任务表示，允许精确控制复杂度的每个维度

2. **跨平台自动化流水线（Cross-Platform Automated Pipeline）**:

    - 功能：将抽象图任务自动转化为不同平台（Android、Web、Desktop）上的具体任务
    - 核心思路：每个原子操作对应平台特定的 UI 操作模板。通过模板实例化 + 环境快照，生成可执行的任务
    - 设计动机：跨平台使 benchmark 不局限于特定应用生态

3. **OmniEval 多维评估框架（Multi-Dimensional Evaluation Framework）**:

    - 功能：从 10 个能力维度评估 Agent
    - 评估维度包括：
        - **基础操作**: 点击/输入/滑动的准确率
        - **空间感知**: UI 元素定位和布局理解
        - **逻辑推理**: 条件判断和分支选择
        - **长期规划**: 多步任务的路径规划
        - **错误恢复**: 操作失败后的纠正能力
        - **多模态理解**: 图标/图像的语义理解
        - 等共 10 个维度
    - 核心思路：每个子任务标注了其对应的能力维度，通过子任务级评估聚合为维度级评分
    - 设计动机：仅靠端到端成功率无法找到 Agent 的具体短板

### 损失函数 / 训练策略
在 OmniBench 的图结构化数据上微调 Agent（用于训练实验）。使用标准的行为克隆损失。

## 实验关键数据

### 主实验

| 模型 | 整体成功率 | 子任务准确率 | 逻辑推理 | 长期规划 | 错误恢复 |
|------|-----------|-----------|---------|---------|---------|
| GPT-4o | **61.2%** | **73.8%** | **68.5%** | 51.3% | 42.1% |
| Claude 3.5 | 58.7% | 71.2% | 65.3% | **53.8%** | 39.6% |
| Qwen2-VL-72B | 47.3% | 62.4% | 54.1% | 41.2% | 35.8% |
| CogAgent | 42.8% | 58.1% | 48.7% | 36.5% | **43.2%** |
| 人类 | 95.2% | 97.8% | 98.1% | 93.4% | 91.7% |

### 消融实验

| 配置 | 训练数据效率 | 说明 |
|------|-----------|------|
| 图结构化数据训练 | **+15.3%** 成功率 | 相比平面数据显著提升 |
| 人工标注数据训练 | +12.1% 成功率 | 质量高但数量少 |
| 平面合成数据训练 | +8.7% 成功率 | 缺乏结构信息 |
| 无训练 (zero-shot) | 基线 | — |

| 任务复杂度 | GPT-4o 成功率 | Qwen2-VL 成功率 | 说明 |
|-----------|-------------|----------------|------|
| 深度 1-2 (简单) | 82.5% | 68.3% | 简单任务差距小 |
| 深度 3-5 (中等) | 61.2% | 45.7% | 复杂度增加，差距加大 |
| 深度 6-10 (困难) | 38.1% | 22.4% | 长期规划是最大瓶颈 |

### 关键发现
- 合成数据的人类接受率达 91%，质量接近人工标注
- 图结构化训练数据比平面数据更高效地提升 Agent 性能（+15.3% vs +8.7%）
- **长期规划和错误恢复**是所有模型的最大短板——即使 GPT-4o 也只有约 50% 和 42%
- 任务复杂度（图深度）增加时性能急剧下降，说明当前 Agent 的规划能力严重不足
- 开源模型和闭源模型在基础操作上差距不大，但在推理和规划上差距显著

## 亮点与洞察
- **Oral 论文，基准设计出色**: 图结构化任务合成是优雅的解决方案
- **规模空前**: 36K 个任务覆盖 20 个场景，远超现有基准
- **多维评估填补空白**: 首次系统地从 10 个能力维度诊断 Agent
- **训练价值**: 图结构数据不仅用于评估，还能高效训练 Agent

## 局限与展望
- 自动合成的任务可能缺乏真实用户任务的"自然性"
- 10 个能力维度的划分是否完备有待讨论
- 目前聚焦于 UI 操作任务，API 调用型 Agent 未覆盖
- 评估结果可能受具体 prompt 模板影响

## 相关工作与启发
- OSWorld (Xie et al., 2024): 基于真实操作系统的 Agent 基准
- AndroidWorld (Rawles et al., 2024): Android 平台 Agent 基准
- 本文的图结构合成思路可推广到其他需要可控复杂度的 Agent 评估场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 图结构化任务合成和多维评估框架是重要创新
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模评估，多模型比较，训练验证
- 写作质量: ⭐⭐⭐⭐ 系统清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 作为 Oral 论文，对 Agent 评估领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Context is Key: A Benchmark for Forecasting with Essential Textual Information](context_is_key_a_benchmark_for_forecasting_with_essential_textual_information.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](../../CVPR2026/multimodal_vlm/hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](../../ACL2025/multimodal_vlm/agent_rewardbench.md)
- [\[AAAI 2026\] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](../../AAAI2026/multimodal_vlm/unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)
- [\[ICML 2026\] CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict](../../ICML2026/multimodal_vlm/cyberjurors_a_multi-agent_simulation_task_for_e-commerce_disputes_verdict.md)

</div>

<!-- RELATED:END -->
