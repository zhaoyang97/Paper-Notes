---
title: >-
  [论文解读] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation
description: >-
  [AAAI 2026][优化/理论][神经组合优化] 提出 EvoReal 框架，利用 LLM 驱动的进化搜索生成结构上接近真实世界的 VRP 合成实例，再通过两阶段渐进微调策略将预训练神经求解器适配到真实基准，在 TSPLib (1.05% gap) 和 CVRPLib (2.71% gap) 上大幅超越已有神经求解器。
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "神经组合优化"
  - "车辆路径问题"
  - "LLM引导进化"
  - "数据生成器"
  - "渐进式微调"
  - "泛化能力"
---

# Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation

**会议**: AAAI 2026  
**arXiv**: [2511.10233](https://arxiv.org/abs/2511.10233)  
**作者**: Jianghan Zhu, Yaoxin Wu, Zhuoyi Lin, Zhengyuan Zhang, Haiyan Yin, Zhiguang Cao, Senthilnath Jayavelu, Xiaoli Li
**代码**: [GitHub](https://github.com/HenryZhu1029/EvoReal)  
**领域**: 优化  
**关键词**: 神经组合优化, 车辆路径问题, LLM引导进化, 数据生成器, 渐进式微调, 泛化能力

## 一句话总结

提出 EvoReal 框架，利用 LLM 驱动的进化搜索生成结构上接近真实世界的 VRP 合成实例，再通过两阶段渐进微调策略将预训练神经求解器适配到真实基准，在 TSPLib (1.05% gap) 和 CVRPLib (2.71% gap) 上大幅超越已有神经求解器。

## 研究背景与动机

### 问题背景
车辆路径问题 (VRP) 是经典 NP-hard 组合优化问题，广泛存在于物流调度和公共交通领域。近年来基于深度强化学习和注意力机制的神经组合优化 (NCO) 方法在合成 VRP 实例上取得了显著成功，代表性工作包括 POMO (Kwon et al. 2020) 和 LEHD (Luo et al. 2023)。然而这些方法的训练数据几乎全部来自均匀分布采样的合成实例，与真实世界中的路径问题存在巨大的分布偏移。

### 已有工作的不足

**合成到真实泛化鸿沟**：现有 NCO 模型在 TSPLib 和 CVRPLib 等真实基准上性能急剧下降。例如预训练 POMO 在大规模 (1000-5000节点) TSPLib 实例上的最优性 gap 高达 64.84%，几乎不可用。

**LLM 求解器的规模限制**：FunSearch、ReEvo 等 LLM 方法在启发式设计上表现出色，但受限于上下文长度，无法有效处理超过 50 节点的中大规模实例。

**直接微调效果有限**：简单地用真实基准实例直接微调预训练模型改善幅度有限，因为真实实例数量少且分布复杂，模型难以从中学到足够的结构先验。

**跨分布泛化研究不足**：现有工作主要聚焦于跨规模泛化 (小到大)，对跨分布泛化 (均匀到真实世界多样分布) 的关注较少。

### 核心动机
与其让 LLM 直接生成求解启发式或解决方案 (面临规模瓶颈)，不如利用 LLM 的代码生成和推理能力来进化数据生成器——生成结构上与真实世界对齐的合成训练数据。这是一个关键的视角转换：从 "LLM 作为求解器" 转向 "LLM 作为数据工厂设计师"，从而间接提升神经求解器在真实场景中的泛化能力。

## 方法详解

### 整体框架
EvoReal 包含两大核心组件：(1) LLM 驱动的生成器进化 和 (2) 两阶段渐进式微调。整体工作流程为：先从真实基准中划分验证集和测试集，按分布类型对验证集分组；然后用 LLM 进化模块为每种分布类型搜索最优数据生成器；最后用进化出的生成器产生合成数据分两阶段微调预训练模型。

### 生成器模块化设计
以 TSP 为例，从 TSPLib 的 70 个实例中选取 48 个作为验证集、22 个作为测试集。将 48 个验证实例按结构模式分为三类 (S1、S2、S3)，每类对应一种生成器。每个生成器由三部分描述：函数签名定义、生成代码实现、适应度分数。进化目标是最小化合成分布与真实分布之间的散度。

### LLM 驱动的进化搜索
基于 ReEvo 框架进行改进，具体流程为：

1. **种群初始化**：通过初始化提示词引导 LLM 生成 N 个初始生成器，提示词包含种子生成器蓝图和设计指导，为 LLM 提供关于目标分布的先验知识。
2. **种群扩展**：通过四种提示策略 (交叉、变异、短期反思、长期反思) 生成新的生成器个体。短期反思基于两个父代的相对性能差异进行针对性改进；长期反思将累积的短期反思蒸馏为更高层次的设计洞察。
3. **代理评估**：每个新生成器仅微调模型少量 epoch (如 LEHD 微调 10 个 epoch)，取验证集上的最佳平均 gap 作为适应度分数。这种低保真度评估策略大幅加速了进化搜索。
4. **排名选择**：基于适应度分数对种群排名并转化为被选概率。相比 ReEvo 引入了排名选择机制以提高精英生成器的存活率，并将选择阶段推迟到变异之后执行以平衡探索与利用。
5. **迭代终止**：当评估的生成器数量达到上限或连续 m 次迭代无法找到更好生成器时停止。

### 渐进式微调策略
进化完成后执行两阶段微调，核心思想是从容易 (合成数据) 到困难 (真实实例) 的渐进适配：

**第一阶段 (合成数据对齐)**：三类最优生成器协同生成大规模合成数据 (固定 100 节点)，用于微调预训练模型。目标是让模型初步适应多样化的类 TSPLib 风格节点分布模式以建立结构先验。训练 epoch 数远多于代理评估阶段，并持续在 48 个验证实例上监控性能。

**第二阶段 (真实实例适配)**：在第一阶段最优模型基础上直接用真实基准实例微调。由于模型已在第一阶段学习了丰富的分布结构，此时模型能更有效地吸收真实实例中的复杂模式，实现从合成到真实的平滑过渡。

这种两阶段设计不仅弥合了分布 gap，还同时解决了规模 gap，因为第一阶段的合成数据与第二阶段的真实实例在规模和分布上均存在梯度差异。

## 实验关键数据

### TSPLib 性能对比 (70 个实例)

| 方法 | [0,200) Gap | [200,500) Gap | [500,1000) Gap | [1000,5000) Gap | 整体 Gap |
|------|------------|--------------|---------------|----------------|---------|
| LKH-3 | 0.00% | 0.00% | 0.00% | 0.03% | 0.01% |
| ORTools | 1.68% | 3.31% | 3.69% | 4.41% | 3.06% |
| POMO (x8 aug) | 4.63% | 10.46% | 26.31% | 64.84% | 26.66% |
| LEHD (RRC-50) | 0.33% | 0.66% | 2.13% | 6.47% | 2.48% |
| ELG (x8 aug) | 1.15% | 3.98% | 8.73% | 11.36% | 5.62% |
| **POMO (ours, x8)** | **1.14%** | **2.16%** | **5.76%** | **32.43%** | **11.59%** |
| **LEHD (ours, RRC-50)** | **0.30%** | **0.35%** | **0.73%** | **2.54%** | **1.05%** |

LEHD (ours) 在所有规模区间上均优于全部神经求解器和 ORTools，整体 gap 从 2.48% 降至 1.05%。POMO (ours) 在大规模实例上的 gap 从 64.84% 降至 32.43%，降幅超过一半。

### CVRPLib 性能对比 (100 个 SetX 实例)

| 方法 | [0,200) Gap | [200,500) Gap | [500,1002) Gap | 整体 Gap |
|------|------------|--------------|---------------|---------|
| HGS-CVRP | 0.01% | 0.06% | 0.24% | 0.11% |
| ORTools | 2.14% | 4.15% | 5.02% | 4.01% |
| POMO (x8 aug) | 6.75% | 14.95% | 41.15% | 21.53% |
| LEHD (RRC-50) | 4.91% | 4.73% | 8.27% | 5.90% |
| ELG (x8 aug) | 4.51% | 5.52% | 7.80% | 6.03% |
| **POMO (ours, x8)** | **3.52%** | **4.60%** | **6.20%** | **4.87%** |
| **LEHD (ours, RRC-50)** | **2.56%** | **2.59%** | **3.00%** | **2.71%** |

LEHD (ours) 在 CVRPLib 上各规模区间的 gap 差异仅 0.44% (2.56%-3.00%)，说明渐进微调有效缩小了小规模与大规模实例之间的性能差距。

## 亮点与洞察

- **数据中心视角的创新**：不改模型架构不设计新启发式，而是用 LLM 进化数据生成器来弥合分布 gap，思路新颖且实用。EvoReal 可以作为即插即用模块应用于任意 NCO 求解器。
- **LLM 能力的巧妙利用**：绕过了 LLM 在大规模实例上的上下文限制，让 LLM 写生成器代码 (短文本) 而非直接求解实例 (长文本)，将 LLM 的优势最大化。
- **渐进微调优于直接微调**：消融实验表明跳过第一阶段直接用真实数据微调的效果远不如两阶段策略 (POMO 大规模 gap: 44.85% vs 36.48%)，合成数据提供的分布先验至关重要。
- **进化生成器优于朴素分布**：与 Beta、指数、高斯混合等朴素分布相比，进化生成器从一开始就表现更优且收敛到更低 gap，验证了 LLM 能捕捉真实数据的复杂结构模式。
- **推理时间几乎不变**：由于不修改模型架构微调后的推理时间与原始模型基本一致。

## 局限性

- **进化搜索成本较高**：生成器进化需要多次代理评估 (每次涉及短期微调)，总体计算开销不可忽视且依赖 o3 等高端 LLM
- **仅验证了两个骨干模型**：实验仅在 POMO 和 LEHD 上验证是否适用于 BQ、ELG 等其他架构未知
- **生成器分类依赖领域知识**：TSP 验证集按结构分为三类的方式需要人工设计
- **大规模实例仍有差距**：POMO (ours) 在 1000-5000 节点上 gap 仍达 32.43%，距传统求解器差距巨大
- **仅针对 VRP 系列问题**：尚未扩展到 MIS、bin packing 等其他组合优化问题
- **第二阶段存在过拟合风险**：LEHD (ours) 在 CVRP 小规模实例上的 gap 反而高于大规模可能因第二阶段过拟合

## 相关工作对比

- **POMO / LEHD (原始)**：在均匀分布上预训练的 NCO 模型泛化到真实基准时性能大幅下降。EvoReal 在不改动架构前提下将两者性能提升 50% 以上
- **ELG (Gao et al. 2024)**：通过集成可迁移局部策略提升泛化 TSPLib 整体 gap 5.62%。EvoReal-LEHD 以 1.05% gap 大幅领先
- **CNF (Zhou et al. 2024)**：多分布多任务联合训练 TSPLib 整体 gap 23.42%，说明仅靠训练策略调整不足以弥合真实分布 gap
- **ReEvo (Ye et al. 2024)**：LLM 进化启发式求解器受限于实例规模。EvoReal 继承其进化框架并改进转向进化生成器而非求解器
- **FunSearch (Romera-Paredes et al. 2024)**：LLM 进化搜索框架先驱关注函数发现而非数据生成

## 评分

- 新颖性: ⭐⭐⭐⭐ — "LLM 进化数据生成器" 的视角切换是核心贡献，渐进微调策略属标准领域适配思路
- 实验充分度: ⭐⭐⭐⭐ — TSPLib 和 CVRPLib 双基准覆盖消融实验完备但仅两个骨干模型
- 写作质量: ⭐⭐⭐⭐ — 结构清晰动机明确图示直观
- 价值: ⭐⭐⭐⭐ — 为 NCO 模型真实世界部署提供了实用且通用的适配方案代码已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering](instance_generation_for_meta-black-box_optimization_through_latent_space_reverse.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](../../NeurIPS2025/optimization/rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[AAAI 2026\] PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing](peoat_personalization-guided_evolutionary_question_assembly_for_one-shot_adaptiv.md)
- [\[AAAI 2026\] Parametrized Multi-Agent Routing via Deep Attention Models](parametrized_multi-agent_routing_via_deep_attention_models.md)
- [\[NeurIPS 2025\] Learning to Insert for Constructive Neural Vehicle Routing Solver](../../NeurIPS2025/optimization/learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)

</div>

<!-- RELATED:END -->
