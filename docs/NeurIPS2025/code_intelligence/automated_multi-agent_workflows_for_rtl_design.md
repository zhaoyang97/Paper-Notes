---
title: >-
  [论文解读] Automated Multi-Agent Workflows for RTL Design
description: >-
  [NeurIPS 2025 (ML for Systems Workshop)][代码智能][多智能体工作流] VeriMaAS 是一个多智能体框架，通过将 HDL 形式化验证反馈（Yosys + OpenSTA）集成到工作流自动生成过程中，自适应地为 RTL 代码生成任务选择推理算子（I/O → CoT → ReAct → SelfRefine → Debate），以仅数百个训练样本实现比微调基线高 5-7% 的 pass@k 性能。
tags:
  - "NeurIPS 2025 (ML for Systems Workshop)"
  - "代码智能"
  - "多智能体工作流"
  - "RTL代码生成"
  - "形式化验证"
  - "Verilog"
  - "自动化工作流编排"
---

# Automated Multi-Agent Workflows for RTL Design

**会议**: NeurIPS 2025 (ML for Systems Workshop)  
**arXiv**: [2509.20182](https://arxiv.org/abs/2509.20182)  
**代码**: 有（[GitHub](https://github.com/dstamoulis/maas/tree/verimaas/verithoughts)）  
**领域**: LLM Agent / EDA / Hardware Design  
**关键词**: 多智能体工作流, RTL代码生成, 形式化验证, Verilog, 自动化工作流编排

## 一句话总结

VeriMaAS 是一个多智能体框架，通过将 HDL 形式化验证反馈（Yosys + OpenSTA）集成到工作流自动生成过程中，自适应地为 RTL 代码生成任务选择推理算子（I/O → CoT → ReAct → SelfRefine → Debate），以仅数百个训练样本实现比微调基线高 5-7% 的 pass@k 性能。

## 研究背景与动机

**RTL 代码生成的挑战**：随着 LLM 在代码生成领域取得突破，RTL (Register-Transfer Level) 硬件设计代码生成成为新的前沿方向。然而，相比通用编程任务，HDL 和 EDA 资源在互联网上相对稀缺，带来了独特挑战：

**微调成本高**：现有方法 [RTLCoder, VeriThoughts] 依赖昂贵的任务特定微调，需要大量 GPU 预算和数万个训练样本

**推理成本高**：大型推理模型（如 o4）虽无需微调，但将计算负担转移到了推理阶段

**工作流设计人工化**：现有的多智能体工作流方法主要面向 QA 和数学任务，对专业领域（如 RTL 设计）存在适用性鸿沟

**核心洞察**：HDL 领域有一个独特优势 — 形式化验证和综合工具（Yosys, OpenSTA）可以提供精确的设计质量反馈。本文的关键想法是将这些 EDA 工具的反馈直接集成到工作流生成过程中，动态指导算子选择。

## 方法详解

### 整体框架

VeriMaAS 的流程如下：

1. 给定 RTL 设计任务，系统根据输入查询和任务难度自适应采样一组推理算子
2. 每个阶段产生的 Verilog 候选设计通过 Yosys（综合验证）和 OpenSTA（时序/功耗分析）进行评估
3. 综合日志和错误信息反馈给控制器，动态调整后续的算子选择策略

### 关键设计

**解空间定义**：

定义算子集合 O = {Zero-shot I/O, CoT, ReAct, SelfRefine, Debate}。大多数现有提示方案都可视为该解空间中的单一算子序列。例如：
- 始终使用 CoT → O = {O_CoT}
- Self-Refine → O = {O_CoT, O_SelfRefine}

目标是为每个任务找到最优算子组合 O，在 K=20 个候选样本上最大化 pass@k。

**级联控制器**：

控制器 C 是 VeriMaAS 的核心，采用**级联策略**按复杂度递增选择算子：

```
I/O → CoT → ReAct → SelfRefine → Debate
```

在每个阶段 c，控制器计算置信度分数 s_c：
- 运行 K=20 个 Verilog 候选设计通过 Yosys 和 OpenSTA
- s_c = 未通过验证/综合/时序/功耗分析的设计百分比
- 若 s_c 超过阶段阈值 τ_c，进入下一阶段使用更复杂的算子
- 否则返回当前候选解

**形式化验证集成**：

这是本文与通用多智能体工作流方法的根本区别：
- 利用 Yosys 进行综合和面积估算
- 利用 OpenSTA 进行时序和静态功耗分析
- 使用 Skywater 130nm PDK 进行综合
- 失败率作为任务复杂度的代理指标，直接驱动算子升级决策

### 损失函数 / 训练策略

**多目标优化**：

```
max_T E_{(q,a)~D} [U(T;q,a,O) - λ·C(T;q,a,O)]
```

其中：
- U(·) = pass@k 得分（效用）
- C(·) = 每查询平均 token 数（成本）
- λ = 1e-3

**阈值学习**：

从 VeriThoughts 训练集随机采样 500 个数据点，基于 K=20 候选设计的综合失败统计，计算第 20/40/60/80 百分位数作为五个算子的阶段阈值 T = {τ_1, ..., τ_C}。

核心优势：该"调参"过程仅需数百个数据点，比全量微调所需的数万样本减少了**一个数量级**。

## 实验关键数据

### 主实验

**表1：VeriMaAS vs. 各类基线模型（pass@k 比较）**

| 模型 | 方法 | VeriThoughts pass@1 | VeriThoughts pass@10 | VerilogEval pass@1 | VerilogEval pass@10 |
|------|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | Instruct | 80.64 | 90.87 | 50.26 | 61.02 |
| GPT-4o-mini + VeriMaAS | Agent | **83.09** (+2.45) | **92.85** (+1.98) | **52.05** (+1.79) | **64.02** (+3.00) |
| o4-mini | Reasoning | 93.85 | 97.88 | 75.67 | 85.13 |
| o4-mini + VeriMaAS | Agent | **94.09** (+0.24) | **98.17** (+0.29) | **76.15** (+0.48) | 84.50 (-0.63) |
| Qwen2.5-7B | Instruct | 44.90 | 82.33 | 22.92 | 51.47 |
| RTLCoder-7B | Fine-tuned | – | – | 34.60 | 45.50 |
| Qwen2.5-7B + VeriMaAS | Agent | **56.62** (+11.72) | **86.29** (+3.96) | **29.10** (+6.18) | **56.45** (+4.98) |
| Qwen2.5-14B | Instruct | 67.89 | 94.13 | 33.78 | 62.04 |
| VeriThoughts-14B | Fine-tuned | 78.50 | 92.10 | 43.70 | 55.14 |
| Qwen2.5-14B + VeriMaAS | Agent | **74.24** (+6.35) | **95.78** (+1.65) | **41.47** (+7.69) | **62.48** (+0.44) |
| Qwen3-8B | Reasoning | 84.11 | 98.82 | 58.21 | 74.64 |
| Qwen3-8B + VeriMaAS | Agent | **88.13** (+4.02) | **99.05** (+0.23) | **59.87** (+1.66) | 74.18 (-0.46) |
| Qwen3-14B | Reasoning | 89.35 | 98.64 | 65.87 | 75.62 |
| Qwen3-14B + VeriMaAS | Agent | **92.16** (+2.81) | **98.75** (+0.11) | **66.96** (+1.09) | **75.71** (+0.09) |

关键观察：
- 开源 LLM 上提升最显著：Qwen2.5-7B pass@1 提升 +11.72%，超越 RTLCoder-7B 微调基线
- 闭源模型提升较小但一致（o4-mini pass@1 +0.24%），表明多智能体编排即使在高基线下仍有价值
- VerilogEval pass@10 在部分模型上出现微小下降，可能与算子切换引入的多样性变化有关

**表2：VeriMaAS vs. 单一智能体提示策略（含 token 成本）**

| 模型 | 提示方式 | VT pass@1 | VT pass@10 | Tokens (k) | VE pass@1 | VE pass@10 | Tokens (k) |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| o4-mini | + CoT | 94.11 (+0.26) | 97.86 | 1.10 (1.09×) | 76.06 (+0.39) | 84.35 | 1.60 (1.06×) |
| o4-mini | + ReAct | 91.96 (-1.89) | 98.04 | 1.70 (1.68×) | 74.33 (-1.34) | 84.10 | 2.14 (1.42×) |
| o4-mini | + SelfRefine | 94.31 (+0.46) | 98.57 | 2.24 (2.22×) | 75.71 (+0.04) | 84.05 | 3.23 (2.14×) |
| o4-mini | + VeriMaAS | 94.09 (+0.24) | 98.17 | **1.21 (1.20×)** | **76.15** (+0.48) | 84.50 | **1.71 (1.13×)** |
| GPT-4o-mini | + CoT | 82.25 (+1.61) | 92.05 | 0.71 (1.42×) | 51.25 (+0.99) | 62.07 | 0.77 (1.33×) |
| GPT-4o-mini | + VeriMaAS | **83.09** (+2.45) | **92.85** | 1.26 (2.52×) | 52.05 (+1.79) | **64.02** | **0.85 (1.47×)** |

VeriMaAS 在 token 成本上接近轻量级 CoT，远低于 SelfRefine（约 2× 开销），但性能更优。

### 消融实验

**表3：PPA 感知优化的后综合指标变化**

| 模型 | VT Pass@10 | ΔArea% | ΔPower% | ΔDelay% | VE Pass@10 | ΔArea% | ΔPower% | ΔDelay% |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o-mini | 92.46 (-0.39) | -9.18↓ | +1.6↑ | -10.32↓ | 62.93 (-1.09) | -18.83↓ | -3.26↓ | -19.47↓ |
| o4-mini | 98.06 (-0.11) | -14.86↓ | 0.00 | -15.87↓ | 84.18 (-0.32) | -12.22↓ | +1.70↑ | -3.52↓ |
| Qwen2.5-7B | 86.33 (+0.04) | -13.44↓ | -8.67↓ | -13.91↓ | 56.45 (0.00) | **-28.79↓** | +4.07↑ | -24.58↓ |
| Qwen2.5-14B | 95.72 (-0.06) | -16.8↓ | -14.57↓ | -21.39↓ | 62.33 (-0.15) | -16.17↓ | +5.22↑ | -15.53↓ |
| Qwen3-8B | 99.04 (-0.01) | -22.81↓ | -3.68↓ | -20.14↓ | 74.06 (-0.12) | -9.98↓ | -6.04↓ | -9.03↓ |
| Qwen3-14B | 98.75 (0.00) | -9.99↓ | +2.12↑ | -9.94↓ | 75.64 (-0.07) | -11.66↓ | -7.85↓ | -11.39↓ |

关键发现：
- 面积和延迟普遍显著降低（最高 -28.79% 面积、-24.58% 延迟）
- 功耗存在权衡：部分模型功耗略有上升（如 VerilogEval 上 Qwen2.5-14B +5.22%）
- pass@10 几乎无损（最大下降仅 -1.09%），说明 PPA 优化不牺牲功能正确性
- 这证明了控制器**可灵活重新优化**以针对不同设计目标，而微调方法则将目标固化在模型权重中

### 关键发现

1. **开源模型获益最大**：VeriMaAS 在 Qwen2.5-7B/14B 上的提升（+6-12% pass@1）远超其在 o4-mini 上的提升（+0.24%），说明工作流自动化能有效弥补模型能力不足
2. **成本效率优势**：仅需约 500 个训练数据点进行阈值校准，相比 VeriThoughts 等微调方法所需的数万样本，训练成本降低一个数量级
3. **级联策略的有效性**：不同复杂度的任务自动匹配不同级别的推理算子，简单任务用 I/O，复杂任务逐步升级到 Debate
4. **PPA 灵活优化**：作为概念验证，通过简单修改目标函数中的成本项即可实现面积/延迟优化，展示了框架的可扩展性

## 亮点与洞察

- **形式化验证作为自然的任务难度信号**：这是本文最巧妙的设计。在通用 QA 领域，很难获得客观的"答案质量"信号；但在 RTL 设计中，Yosys 编译失败率直接反映任务复杂度，为控制器提供了精确的反馈
- **Training-free 的工作流自动化**：与需要梯度更新的微调方法不同，VeriMaAS 通过统计阈值校准实现工作流优化，极大降低了领域适配成本
- **从通用→专业的 bridge**：本文展示了如何将通用的多智能体工作流方法（MaAS, AFlow）适配到专业硬件设计领域，关键是找到领域特有的反馈信号

## 局限与展望

1. 控制器目前采用简单的级联策略和百分位阈值，未来可探索树搜索或 RL 策略以实现更细粒度的工作流决策
2. 当前仅使用开源 Yosys + OpenSTA，扩展到商业 EDA 工具和工业 PDK 可能释放更大的 PPA 优化潜力
3. PPA 优化的 benchmark 子集(-PPA-Tiny) 由 o4 伪预言选择，引入了评估偏差
4. VerilogEval pass@10 在部分配置下出现轻微下降，算子切换策略可能导致候选多样性降低
5. 仅评估了五个固定算子，未探索算子组合或自定义算子的可能性

## 相关工作与启发

- **MaAS** [Zhang et al., 2025]：本文的级联控制器直接建立在 MaAS 的多智能体即服务框架之上
- **AFlow** [Zhang et al., 2025]：另一种自动工作流生成方法，专注于通用 QA
- **VeriThoughts** [Yubeaton et al., 2025]：RTL 基准和微调方法，本文的主要对比基线
- **RTLCoder** [Liu et al., 2024]：轻量级微调 RTL 方案，VeriMaAS 在同等模型大小下超越其性能

本文启发了一个更广泛的方向：在任何有形式化验证工具的专业领域（如定理证明、电路设计、编译器优化），都可以将验证反馈集成到多智能体工作流中以实现自动化的推理策略选择。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将形式化验证反馈引入多智能体工作流自动化是有意义的创新
- **技术深度**: ⭐⭐⭐ — 方法相对简洁（级联控制器 + 百分位阈值），但设计合理
- **实验质量**: ⭐⭐⭐⭐ — 覆盖 6 个模型 × 2 个 benchmark，含成本分析和 PPA 消融
- **实用性**: ⭐⭐⭐⭐⭐ — 低训练成本、即插即用、适用于多种 LLM，落地价值高
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，但 workshop paper 篇幅限制了部分细节

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CARD: Towards Conditional Design of Multi-agent Topological Structures](../../ICLR2026/code_intelligence/card_towards_conditional_design_of_multi-agent_topological_structures.md)
- [\[ICML 2026\] UniRTL: 统一代码与图实现鲁棒 RTL 表示学习](../../ICML2026/code_intelligence/unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning.md)
- [\[NeurIPS 2025\] Learning From Design Procedure To Generate CAD Programs for Data Augmentation](learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)
- [\[NeurIPS 2025\] A Self-Improving Coding Agent](a_selfimproving_coding_agent.md)
- [\[ACL 2025\] CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](../../ACL2025/code_intelligence/compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)

</div>

<!-- RELATED:END -->
