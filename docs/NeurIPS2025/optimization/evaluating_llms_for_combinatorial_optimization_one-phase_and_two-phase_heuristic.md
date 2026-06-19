---
title: >-
  [论文解读] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing
description: >-
  [NeurIPS 2025][优化/理论][LLM评估] 本文提出一个结合 LLM 与进化算法的系统性评估框架，用于评估 LLM 在 2D 装箱问题上生成和优化启发式算法的能力，GPT-4o 在 2 轮迭代内即达到最优解，将平均箱数从 16 降至 15，空间利用率从 0.76-0.78 提升至 0.83。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "LLM评估"
  - "组合优化"
  - "2D装箱问题"
  - "进化算法"
  - "启发式生成"
---

# Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing

**会议**: NeurIPS 2025  
**arXiv**: [2509.22255](https://arxiv.org/abs/2509.22255)  
**代码**: [附录提供](https://arxiv.org/abs/2509.22255)  
**领域**: 优化 / LLM 应用  
**关键词**: LLM评估, 组合优化, 2D装箱问题, 进化算法, 启发式生成

## 一句话总结
本文提出一个结合 LLM 与进化算法的系统性评估框架，用于评估 LLM 在 2D 装箱问题上生成和优化启发式算法的能力，GPT-4o 在 2 轮迭代内即达到最优解，将平均箱数从 16 降至 15，空间利用率从 0.76-0.78 提升至 0.83。

## 研究背景与动机

**领域现状**：LLM 的评估正从传统 NLP 任务扩展到数学推理和算法设计等专业领域。2D 装箱问题（2D-BPP）是经典 NP-hard 组合优化问题，要求将矩形物品放入最少数量的固定尺寸箱中。

**现有痛点**：(a) 传统启发式如 Finite First-Fit (FFF) 和 Hybrid First-Fit (HFF) 虽然高效但解质量有限；(b) FunSearch 等工作已展示 LLM 在进化循环中生成高性能启发式的潜力，但缺乏**系统性评估框架**来量化 LLM 在此类任务中的能力。

**核心矛盾**：如何系统地评估 LLM 是否真正理解复杂算法约束、能否持续改进解质量、以及与传统方法的定量差距。

**本文目标** 建立评估 LLM 在组合优化中生效的结构化方法论，包括评估指标、迭代流程和基线对比。

**切入角度**：受 FunSearch 启发，引入"岛屿模型"进化循环——LLM 生成 → 正确性验证 → 评分 → 聚类（岛屿） → 反馈优化。

**核心 idea**：通过迭代进化框架让 LLM 在 2D-BPP 上自动生成、评估和改进启发式，并用多维指标系统评估其优化能力。

## 方法详解

### 整体框架
输入为结构化 prompt（包含问题约束、函数原型、I/O 格式），LLM（GPT-4o）生成 Python 启发式脚本 → 正确性验证（无重叠、边界内、完整装入）→ 多维评分（箱数、空间利用率、运行时间）→ 岛屿聚类 → 选取 Top-3 岛屿各一脚本反馈到下一轮 prompt，循环 6 轮。

### 关键设计

1. **数学建模**:

    - 功能：将 2D-BPP 形式化为整数规划
    - 核心思路：$n$ 个物品尺寸 $(w_i, h_i)$，箱尺寸 $(W, H)$，决策变量 $z_{ij}=1$ 表示物品 $i$ 在箱 $j$ 中，$u_j = 1$ 表示箱 $j$ 被使用。目标 $\min \sum_{j=1}^n u_j$，约束包括：每个物品恰好在一个箱中 $\sum_j z_{ij} = 1$、坐标约束 $0 \leq x_{ij} \leq (W-w_i)z_{ij}$、非重叠约束。
    - 总利用率：$\rho_{\text{total}} = \frac{\sum_{i} w_i h_i}{(\sum_j u_j) WH}$

2. **六步迭代进化流程**:

    - **Step 1 结构化 Prompt**：精确描述问题约束、函数签名、I/O 格式。提供模板函数确保语法一致
    - **Step 2 代码生成与验证**：每轮生成 20 个脚本，严格验证约束满足性（无重叠、边界内、不重复分配），不合格者淘汰
    - **Step 3 评分**：主指标为箱数，次指标为空间利用率，再次指标为执行时间
    - **Step 4 岛屿聚类**：将高分脚本按逻辑相似性分组为"岛屿"，保持多样性防止过早收敛
    - **Step 5 反馈优化**：从 Top-3 岛屿各随机选一脚本作为 "best-shot" 示例嵌入下一轮 prompt，引导 LLM 从成功策略中学习
    - **Step 6 终止条件**：6 轮后停止（基于资源限制和边际收益递减）

3. **基线方法**:

    - **FFF (Finite First-Fit)**：按高度排序，逐个放入第一个可用位置，$O(n^2)$ 复杂度。单阶段贪心策略
    - **HFF (Hybrid First-Fit)**：两阶段方法——先用 FFDH 做条带装箱生成水平层，再用 FFD 将条带装入箱中。$O(n\log n)$ 复杂度

### 实验配置
- 数据集：20 轮迭代，每轮 50 个随机正方形（边长 10-50 单位），箱 200×100
- 模型：GPT-4o，BPE tokenization
- 硬件：Intel Core i5-8250U，8GB RAM
- 所有方法在完全相同的数据集上评估

## 实验关键数据

### 主实验

| 方法 | 平均箱数↓ | 执行时间 (s) | 空间利用率↑ |
|------|----------|-------------|-----------|
| FFF | 16.05 | 0.002446 | 0.76 |
| HFF | 16.00 | 0.024438 | 0.78 |
| **LLM (GPT-4o)** | **15.00** | 0.0103 | **0.83** |

- 箱数减少 6.25%，空间利用率提升 6.4%（vs HFF）
- 执行时间介于 FFF 和 HFF 之间，竞争力强

### 空间利用率分布分析

| 方法 | 前几箱利用率 | 后几箱利用率 | 利用率波动 |
|------|-----------|-----------|----------|
| FFF | 87.50% | 68.00% | 大（后期下降显著）|
| HFF | 86.83% | 63.54% | 大（后期下降显著）|
| LLM | ~83% | ~83% | **小（全程均匀）** |

### 关键发现
- LLM 在**仅 2 轮迭代**内即达到最优解，收敛速度远快于 6 轮上限，说明 LLM 具备高效的约束满足学习能力
- LLM 生成的启发式在各箱之间保持更均匀的利用率（83%），传统方法后期箱利用率大幅下降
- LLM 展示了超越显式编程的"优化直觉"——生成的装箱策略包含未被明确指定的复杂模式
- 6 轮进化后解质量趋于稳定，边际改进减小

## 亮点与洞察
- **进化反馈机制的有效性**：岛屿模型保持了策略多样性，避免 LLM 收敛到单一策略。"best-shot" 反馈使 LLM 的每一轮输出质量递增，这种 LLM + 进化框架可迁移到其他组合优化问题
- **LLM 的约束理解能力**：GPT-4o 能成功内化几何和逻辑约束（非重叠、边界内），验证代码的 100% 通过率表明 LLM 对复杂约束的理解达到可靠水平
- **低资源下的快速收敛**：仅 2 轮迭代即达到最优，暗示 LLM 的算法设计能力可能远超此任务的复杂度需求

## 局限与展望
- **规模有限**：仅在 50 个正方形、200×100 箱的中等规模上测试，工业级问题（数千物品、异形件）的表现未知
- **LLM 不确定性**：非确定性输出影响可重复性，需多次运行取统计量
- **API 成本约束**：限制了迭代轮数和候选脚本数，更大搜索空间可能发现更优解
- **仅正方形物品**：实际 2D-BPP 涉及矩形甚至不规则形状，是否支持旋转等约束未探索
- **单模型评估**：仅用 GPT-4o，缺乏不同 LLM（Claude、Gemini 等）的对比

## 相关工作与启发
- **vs FunSearch [Romera-Paredes et al., 2024]**: FunSearch 使用 LLM + evaluator 发现新启发式（如 online bin packing），但未系统评估 LLM 能力。本文建立了更完整的评估方法论
- **vs 传统元启发式（SA、GA）[Ferreira, 2017]**: 传统方法需手工设计搜索算子。LLM 直接生成完整算法代码，降低了人工参与
- **vs 深度强化学习方法 [Lee, 2025]**: DRL 需要大量训练和问题特定的状态/动作设计。LLM 方法零训练，即开即用
- **与 prompt engineering 的关系**：本文的成功高度依赖 prompt 设计质量，系统化的 prompt 优化研究是重要未来方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 方法框架来自 FunSearch 的迁移，创新主要在评估方法论
- 实验充分度: ⭐⭐⭐⭐ 评估指标全面但问题规模较小，缺乏更广泛的对比
- 写作质量: ⭐⭐⭐⭐⭐ 框架描述清晰，方法论表述完整
- 价值: ⭐⭐⭐⭐ 为 LLM 在组合优化领域的评估提供了初步方法论参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD](from_information_to_generative_exponent_learning_rate_induces_phase_transitions_.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[NeurIPS 2025\] Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner](purifying_shampoo_investigating_shampoos_heuristics_by_decomposing_its_precondit.md)
- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](oracle-efficient_combinatorial_semi-bandits.md)

</div>

<!-- RELATED:END -->
