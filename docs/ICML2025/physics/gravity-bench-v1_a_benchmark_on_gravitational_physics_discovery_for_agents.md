---
title: >-
  [论文解读] Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents
description: >-
  [ICML 2025][物理/科学计算][引力物理] 提出 Gravity-Bench-v1：，一个基于引力动力学模拟的环境交互式：基准测试，评估 AI Agent 在受限观测预算下进行科学发现（包括 OOD 物理场景）的能力，发现当前模型在观测规划和预算利用方面存在显著不足。 现代科学起源于对行星运动的反复观测与推理…
tags:
  - "ICML 2025"
  - "物理/科学计算"
  - "引力物理"
  - "科学发现"
  - "AI Agent"
  - "benchmark"
  - "部分可观测环境"
  - "观测规划"
  - "双星系统"
---

# Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents

**会议**: ICML 2025

**arXiv**: [2501.18411](https://arxiv.org/abs/2501.18411)

**作者**: Nolan Koblischke, Hyunseok Jang, Kristen Menou, Mohamad Ali-Dib

**领域**: 物理 / AI for Science / 基准测试

**关键词**: 引力物理, 科学发现, AI Agent, benchmark, 部分可观测环境, 观测规划, 双星系统

## 一句话总结

提出 **Gravity-Bench-v1**，一个基于引力动力学模拟的**环境交互式**基准测试，评估 AI Agent 在受限观测预算下进行科学发现（包括 OOD 物理场景）的能力，发现当前模型在观测规划和预算利用方面存在显著不足。

## 研究背景与动机

现代科学起源于对行星运动的反复观测与推理。现有 AI 评估基准主要集中在知识评测（如 GPQA、MMLU）或通用问题求解（如 ARC、HellaSwag），缺乏对 AI 在**真实科学发现过程**中能力的评估——包括规划观测、在不确定性下推理、发现新现象等。

Gravity-Bench-v1 的设计动机：

- **模拟完整科学过程**：Agent 不仅需要分析数据，还需**主动规划观测**、在有限预算内收集数据
- **包含 OOD 场景**：修改引力定律（$F_G \propto r^{-(2+\alpha)}$）和引入阻力，测试**科学泛化能力**
- **开放式解题空间**：不限定解题方法，允许 Agent 发现优于人类基线的策略
- **提供博士级参考解**：以人类专家方案作为上界，校准 AI 表现

## 方法详解

### 整体框架

Gravity-Bench-v1 由三部分组成：

1. **模拟环境**：基于 Rebound（引力 N 体模拟器）的双星系统模拟
2. **观测协议**：full-obs（完整数据）和 budget-obs-100（最多 100 次观测）
3. **任务设计**：16 个双星模拟 × 50 个任务 = **206 个任务-模拟对**

### 关键设计

**环境设计**：所有轨道在笛卡尔 $(x, y)$ 平面上，模拟使用 WHFast（能量守恒积分器）或 IAS15（自适应 15 阶积分器，用于修改引力场景）。时间步长为轨道周期的 $1/5000$。

**观测工具**：Agent 通过 observe 工具收集数据，每次调用最多获取 10 个数据点，总预算 $N_{\text{obs}} = 100$。Agent 可选择观测时刻，鼓励策略性规划。

**对称性破坏策略**：
- 质心偏离原点
- 引入本动运动（proper motion）
- 模拟真实天文观测的"混乱性"

**OOD 场景**（6 个）：
- 3 个阻力场景：$\ddot{x}_i = -v_i / \tau$，需推断阻力时标 $\tau$
- 3 个修改引力场景：$F_G \propto r^{-(2+\alpha)}$，需推断偏差 $\alpha$

### 任务类型

任务涵盖：恒星质量推断、轨道周期、离心率、半长轴、能量守恒验证、Kepler 第三定律验证、最大速度、Roche 瓣半径、角动量、修改引力指数等。

### 评估方法

答案正确标准：相对误差低于**任务特定阈值**（5%–70%），阈值基于 PhD 级方案在 100 次均匀采样下的性能退化程度设定。

### Baseline Agent

采用 ReAct 风格的 Agent，配备 observe 工具和 Python 解释器（含 numpy、scipy、pandas），支持多步推理和代码执行。

## 实验关键数据

### 主实验结果

| 模型 | 准确率 (budget-obs-100) | 准确率 (full-obs) | 总成本 ($) | 平均使用观测数 |
|:---|:---|:---|:---|:---|
| o1-2024-12-17 | — | 64.0%† | $100.07 | — |
| Claude 3.5 Sonnet | 21.5% ± 2.5% | 39.5% ± 3.2% | $15.88 | 24.3 |
| Claude 3.5 Haiku | 16.1% ± 2.3% | 34.1% ± 3.1% | $3.33 | 12.6 |
| GPT-4o | 15.5% ± 2.1% | 36.1% ± 3.2% | $9.60 | 12.2 |
| GPT-4o-mini | 8.3% ± 1.5% | 26.7% ± 2.8% | $0.60 | 13.4 |
| **PhD 级方案** | **82.5%** | **100.0%** | — | 100.0 |

### 关键发现

| 发现 | 细节 |
|:---|:---|
| **观测预算严重未利用** | GPT-4o 平均仅用 12/100 次观测，Claude 用 24/100 次 |
| **OOD 任务极具挑战** | 仅 o1 能一致性解决修改引力任务（2/6），Claude 3.5 Sonnet 解决 1/6 |
| **质量假设是主要失败模式** | GPT-4o 在 33% 的错误解中假设质量=1，正确解中仅 5% |
| **Agent 倾向"速成"** | 找到看似合理的答案就停止，不进一步验证 |
| **规划能力差异大** | Claude 3.5 Sonnet 偶尔通过精细规划达到 <1% 误差，但不稳定 |

### 规划案例分析

在最大速度估计任务中（40 次观测）：
- **成功案例**：Claude 先粗采样定位高速区域，再迭代细化时间分辨率，最终误差 2%
- **失败案例**：同一模型，未记录峰值速度时刻，将分辨率提升误判为速度增长，最终误差 45%

## 亮点与洞察

- **环境交互式评估范式**：相比静态 QA 基准，更接近真实科学发现过程
- **OOD 设计精妙**：修改引力指数 $\alpha$ 的任务几乎不可能通过记忆解决，是真正的泛化测试
- **揭示了 Agent 的"科学幻觉"**：模型倾向于假设对称性和简化条件，而非从数据推导
- **开放式解空间**：理论上 Agent 可以发现比人类更优的观测策略

## 局限性

1. **仅覆盖二体引力**：物理复杂度有限，未涉及三体、流体等更复杂系统
2. **2D 轨道**：所有轨道在平面上，未考虑投影效应
3. **Agent 框架较单一**：仅测试 ReAct 风格 Agent，未探索其他架构（如 tree-of-thought）
4. **o1 测试不完整**：由于 API 内容策略限制，17/206 问题被拒绝，且仅单次运行
5. **成本较高**：o1 complete evaluation 花费 $100+，限制了大规模实验

## 相关工作与启发

- **SWE-bench, RE-bench, BrowserGym**：环境交互式 Agent 基准，但面向软件/网页领域
- **DiscoveryBench, DiscoveryWorld**：数据驱动发现基准，但较为静态
- **The AI Scientist (Lu et al., 2024)**：自动化科研工作流，但不聚焦于物理发现
- **ScienceAgentBench**：科学研究 Agent 评估，但面向代码层面

**启发**：该工作展示了**部分可观测环境 + 预算约束**的评估范式对于衡量 Agent 科学能力的价值，模型在"何时停止观测"和"如何验证答案"方面的不足值得关注。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 环境式物理发现基准是新的评估范式
- **技术深度**: ⭐⭐⭐⭐ — Rebound 模拟严谨，任务设计考虑周全
- **实用性**: ⭐⭐⭐⭐ — 对评估和改进 AI 科学发现能力有重要参考价值
- **写作质量**: ⭐⭐⭐⭐⭐ — 案例分析清晰，失败模式讨论深入
- **综合评分**: 8/10 — 填补了物理发现 Agent 评估的重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Causal Discovery of Latent Variables in Galactic Archaeology](causal_discovery_of_latent_variables_in_galactic_archaeology.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/physics/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [\[NeurIPS 2025\] POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](../../NeurIPS2025/physics/polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)
- [\[ICML 2025\] Causal-PIK: Causality-based Physical Reasoning with a Physics-Informed Kernel](causal-pik_causality-based_physical_reasoning_with_a_physics-informed_kernel.md)

</div>

<!-- RELATED:END -->
