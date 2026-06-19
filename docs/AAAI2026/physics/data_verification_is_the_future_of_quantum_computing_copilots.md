---
title: >-
  [论文解读] Data Verification is the Future of Quantum Computing Copilots
description: >-
  [AAAI 2026][物理/科学计算][量子电路优化] 这是一篇 position paper，提出量子计算 AI 助手（Copilot）必须将数据验证从事后过滤提升为架构级基础——通过三个立场论证：(1) 验证数据是最低要求，(2) 先验约束优于后验过滤，(3) 受物理定律约束的科学领域需要验证感知架构。实验表明无验证数据的 LLM 在电路优化上最高仅达 79% 准确率。
tags:
  - "AAAI 2026"
  - "物理/科学计算"
  - "量子电路优化"
  - "LLM幻觉"
  - "形式验证"
  - "数据质量"
  - "验证优先架构"
---

# Data Verification is the Future of Quantum Computing Copilots

**会议**: AAAI 2026  
**arXiv**: [2602.04072](https://arxiv.org/abs/2602.04072)  
**代码**: 无  
**领域**: 量子计算 / AI4Science  
**关键词**: 量子电路优化, LLM幻觉, 形式验证, 数据质量, 验证优先架构

## 一句话总结
这是一篇 position paper，提出量子计算 AI 助手（Copilot）必须将数据验证从事后过滤提升为架构级基础——通过三个立场论证：(1) 验证数据是最低要求，(2) 先验约束优于后验过滤，(3) 受物理定律约束的科学领域需要验证感知架构。实验表明无验证数据的 LLM 在电路优化上最高仅达 79% 准确率。

## 研究背景与动机

**领域现状**：量子程序生成和编译需要跨多个抽象层——从高层电路描述到底层门排列，要求零容错。LLM 已开始用于量子辅助（IBM Qiskit Code Assistant、AlphaTensor-Quantum 实现 37-47% T门减少、AlphaEvolve 等），但面临根本性限制。

**现有痛点**：(1) LLM 做的是统计模式匹配，不是形式推理，多步逻辑操作系统性失败；(2) 幻觉在数学上是不可避免的，不能通过规模扩展解决；(3) 量子领域训练数据极度稀缺——最全面的 QDataSet 花了 3 个月 HPC 集群时间仅生成 1-2 量子比特系统的 14TB 数据。

**核心矛盾**：量子电路的正确性是二元的（对/错），不容近似。有效设计占所有可能设计的比例随量子比特数指数级下降（$O((\delta/d)^n)$），后验过滤在计算上不可行。

**本文目标** 为量子计算 Copilot（及更广泛的 AI4Science）提出验证优先的架构范式。

**切入角度**：以量子加法器（Cuccaro Adder）为具体案例，通过exhaustive search + Lean/Z3形式验证，量化有效设计与总设计空间的比例，并评估 34 个 LLM 的表现。

**核心 idea**：对于正确性为二元约束的科学领域，验证必须嵌入到 AI 系统的训练数据和生成循环中，而非作为事后过滤。

## 方法详解

### 整体框架
这不是一个方法论文，而是 position paper。核心论证通过三个立场 (P1-P3) 展开，每个立场配有实验或分析支撑。

### 关键设计

1. **P1: 验证数据是最低要求**:

    - 论点：未验证数据上的训练等同于隐式数据投毒——与自然语言任务不同，量子电路约束不允许噪声被大数据平均掉，单个无效门序列就违反酉性
    - 实验支撑：评估 34 个公开 LLM（270M-120B 参数），每个回答 70,000 道多选题（2-8 量子比特各 10,000 题）。验证感知模型（gemma3:12b, gpt-oss:120b）达到 60-79% 准确率，通用模型在 25% 随机基线附近（21-29%）
    - 校准分析：GPT-OSS 模型对正确答案给出 60-80% 置信度，Gemma3 仅 20-35%，表明验证感知训练不仅提升准确率，还改善不确定性量化

2. **P2: 先验约束优于后验过滤**:

    - 论点：随设计组件增长，有效设计占比指数级下降。d 种门选择、$O(n)$ 个位置 → 总空间 $O(d^n)$，约束后有效设计 $O(\delta^n)$，有效比例 $O((\delta/d)^n)$
    - 实验支撑：对 MAJ/UMA 模块做exhaustive search+Lean/Z3 验证，发现 540 个有效 MAJ 设计和 529 个有效 UMA 设计，组合得 285,660 个有效加法器。而总设计空间超 148 万亿，有效比例约 $10^{-9}$
    - 设计含义：验证应嵌入到 token 生成循环中——每个电路片段在提交前需满足 SMT 约束或符号规约

3. **P3: 验证感知架构适用于所有硬约束科学领域**:

    - 论点：量子电路的"泄漏抽象"（leaky abstraction）和"不可分解性"（non-decomposability）是跨学科共性——局部最优不保证全局最优
    - Cuccaro Adder 案例：看似次优的 UMA 设计（更多门）在展开到整体电路后反而因并行化而更优，这种跨抽象层的非分解性使贪心方法失效
    - 外推：药物发现（优化单一属性损害其他属性）、组合优化（贪心算法可能得到最差解）、工程设计（多学科耦合）都有相同特性

### 损失函数 / 训练策略
评估使用加权评分函数：$\text{Score} = 0.5 \times \text{Toffoli}_n + 0.25 \times \text{Depth}_n + 0.25 \times \text{TotalGates}_n$，四个候选电路都满足加法器规约但优化程度不同，正确答案为最低分电路。

## 实验关键数据

### 主实验
34 个 LLM 在量子电路优化选择题上的表现（每模型 70,000 题）。

| 模型类别 | 代表模型 | 准确率 | 置信度（正确答案） |
|--------|------|------|----------|
| 验证感知模型 | gpt-oss:120b | **~79%** | 60-80% |
| 验证感知模型 | gemma3:12b | ~60% | 20-35% |
| 通用大模型 | 多数模型 | 21-29% | 接近随机 |
| 格式失败模型 | mixtral:8x7b, deepseek-r1 | <25% | 输出格式错误 |

### 设计空间分析

| 指标 | 数值 |
|------|------|
| 有效 MAJ 设计 | 540 |
| 有效 UMA 设计 | 529 |
| 有效加法器总数 | 285,660+ |
| 总设计空间 | >148 万亿 |
| 有效比例 | ~$10^{-9}$ |
| 形式验证工具 | Lean + Z3 |

### 关键发现
- **规模扩展不能解决问题**：参数量从 270M 到 120B，不具备验证数据的模型仍停留在随机基线附近
- **有效设计空间极小**：$10^{-9}$ 的有效比例使后验过滤在计算上不可行
- **输出格式失败是常见问题**：多个模型（包括 deepseek-r1 变体）因输出冗长解释而非单字母答案，低于随机基线
- 验证感知训练同时改善准确率和校准度——是模型真正"理解"而非记忆的指标

## 亮点与洞察
- **量化了"有效设计的指数级稀缺"**：$10^{-9}$ 的比例是一个有力的论据，它不仅适用于量子计算，也适用于任何有硬约束的组合设计问题。这个概念可以迁移到芯片设计、蛋白质折叠等领域的 AI 系统设计
- **"隐式数据投毒"的类比很有启发性**：在正确性为二元的领域，未验证的训练数据不是噪声而是毒药——这一观察对所有 AI4Science 应用都有警示意义
- **泄漏抽象的形式化**：非分解性 $f(g_1, \ldots, g_m) \neq \sum f_k(g_k)$ 是跨学科共性问题的简洁抽象

## 局限与展望
- 作为 position paper，实验仅限于量子加法器这一特定案例，其他量子算法（QFT、Grover等）的验证难度可能不同
- 仅评估了多选题形式，未测试 LLM 的开放式电路生成能力
- 提出的"将验证嵌入生成循环"的架构尚无具体实现，主要是未来方向
- 验证数据的生成本身成本极高（QDataSet 花了 3 个月 HPC），如何规模化验证数据生产是核心瓶颈
- 未讨论 LLM + 形式验证器的混合架构的具体工程挑战（如延迟、计算成本）

## 相关工作与启发
- **vs AlphaTensor-Quantum**: AlphaTensor 将正确性嵌入搜索目标，实现了 37-47% T门减少，佐证了 P2（先验约束优于后验过滤）
- **vs AlphaEvolve**: 通过机器可检查规约过滤变体，是验证优先范式的成功案例
- **vs QCircuitNet**: 将任务与可执行验证 oracle 配对，体现了 P1（验证数据是最低要求）

## 评分
- 新颖性: ⭐⭐⭐⭐ 将形式验证从软件工程提升到 AI 架构级别，在量子计算语境下论证有说服力
- 实验充分度: ⭐⭐⭐ 34 个模型的评估有一定规模，但仅限加法器案例，缺乏验证优先架构的实际实现
- 写作质量: ⭐⭐⭐⭐⭐ Position paper 写作清晰，三个立场层层递进，案例选择恰当
- 价值: ⭐⭐⭐⭐ 对 AI4Science 社区有重要的方向性指导意义，验证优先范式适用范围广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AQER: A Scalable and Efficient Data Loader for Digital Quantum Computers](../../ICLR2026/physics/aqer_a_scalable_and_efficient_data_loader_for_digital_quantum_computers.md)
- [\[AAAI 2026\] Fast 3D Surrogate Modeling for Data Center Thermal Management](fast_3d_surrogate_modeling_for_data_center_thermal_management.md)
- [\[ICML 2025\] OmniArch: Building Foundation Model For Scientific Computing](../../ICML2025/physics/omniarch_building_foundation_model_for_scientific_computing.md)
- [\[CVPR 2025\] ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks](../../CVPR2025/physics/atp_adaptive_threshold_pruning_for_efficient_data_encoding_in_quantum_neural_net.md)
- [\[AAAI 2026\] Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)

</div>

<!-- RELATED:END -->
