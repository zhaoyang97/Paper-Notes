---
title: >-
  [论文解读] Meta-World+: An Improved, Standardized, RL Benchmark
description: >-
  [NeurIPS 2025][强化学习][多任务强化学习] 本文系统揭示 Meta-World 基准在不同版本间因奖励函数不一致导致的算法比较失真问题，并发布标准化新版本 Meta-World+，明确保留 V1/V2 两套奖励函数，新增 MT25/ML25 任务集，升级至 Gymnasium API，实现完全可复现的多任务和元强化学习评估。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "多任务强化学习"
  - "元强化学习"
  - "benchmark"
  - "奖励函数"
  - "可复现性"
---

# Meta-World+: An Improved, Standardized, RL Benchmark

**会议**: NeurIPS 2025  
**arXiv**: [2505.11289](https://arxiv.org/abs/2505.11289)  
**代码**: [GitHub](https://github.com/Farama-Foundation/Metaworld/)  
**领域**: 强化学习  
**关键词**: 多任务强化学习, 元强化学习, benchmark, 奖励函数, 可复现性

## 一句话总结

本文系统揭示 Meta-World 基准在不同版本间因奖励函数不一致导致的算法比较失真问题，并发布标准化新版本 Meta-World+，明确保留 V1/V2 两套奖励函数，新增 MT25/ML25 任务集，升级至 Gymnasium API，实现完全可复现的多任务和元强化学习评估。

## 研究背景与动机

Meta-World 是多任务 RL 和元 RL 领域使用最广泛的基准之一，包含 50 个机器人操作任务。然而自最初发布以来，其内部奖励函数经历了未文档化的重大修改，导致以下严重问题：

**版本混乱**：原始的 V1 奖励函数在某个时间节点被 V2 奖励函数覆盖，但未有明确版本记录。V1 的 pick-place 任务奖励范围约为负值到 1200，而 V2 奖励范围为 (0, 10)，两者的量级和设计哲学完全不同。

**比较失真**：不同时期发表的论文使用了不同版本的奖励函数，直接引用先前论文的数值进行比较本身就是不公平的。例如 PaCo 在 V1 上 MT10 仅 26.2% 成功率，但在 V2 上达到 73.6%。

**缺乏标准化**：Meta-World 依赖已停止维护的 OpenAI Gym 和 Mujoco-Py 包，不利于长期科研使用。

作者的核心动机在于消除这些混淆、建立标准化评估平台，并从经验层面为未来基准设计提供指导。

## 方法详解

### 整体框架

Meta-World+ 是一个工程驱动的基准改进工作而非算法创新，其核心是对已有基准进行重新工程化。框架包含三个层面：(1) 奖励函数版本管理——明确保留 V1 和 V2 两套奖励函数作为可选配置；(2) 任务集扩展——新增 MT25/ML25 中间规模任务集；(3) 现代化升级——兼容最新的 Gymnasium API 和 MuJoCo Python 绑定。

### 关键设计

1. **V1 与 V2 奖励函数对比分析**：V1 奖励以 pick-place 模板为基础修改生成各任务奖励，奖励范围大、跨任务差异显著；V2 奖励引入模糊约束，将所有任务的奖励统一到 (0, 10) 范围，使得各任务回报分布更均匀。作者通过 Q 函数损失分析揭示：V2 的统一尺度使 Q 函数更容易建模状态-动作值，从而提升整体成功率。这与 PopArt 的发现一致——**跨任务奖励的一致缩放是多任务学习有效性的关键**。

2. **新增任务集 MT25/ML25**：在原有 MT10/MT50 和 ML10/ML45 之间插入中间规模任务集。MT25 的计算成本约为 MT50 的一半（~12小时 vs ~25小时，A100 GPU），但比 MT10（~6小时）提供更充分的评估。同时支持用户自定义任意大小和组合的任务集，便于研究者设计控制实验。

3. **Gymnasium 集成与现代化**：将 Meta-World 的自定义环境实现对齐 Gymnasium 标准 API，移除对已弃用的 OpenAI Gym 和 Mujoco-Py 的依赖。用户可直接使用 Gymnasium 生态系统的全套工具和基础设施。

### 评估协议

遵循 Agarwal et al. (2021) 的统计建议，在 10 个随机种子上报告结果，使用四分位均值 (IQM) 进行比较。多任务学习每个任务评估 50 个 episode（对应 50 个目标位置），元学习评估在 10 个适应 episode 后进行 3 个 episode 评估。所有方法基于 JAX 重新实现。

## 实验关键数据

### 主实验

**多任务 RL 结果：V1 vs V2 奖励函数对比**

| 算法 | 论文报告 MT10 | MT10 V1 | MT10 V2 | 论文报告 MT50 | MT50 V1 | MT50 V2 |
|------|-------------|---------|---------|-------------|---------|---------|
| SM | 71.8 | 71.4 | **84.9** | 61.0 | 60.6 | **65.8** |
| PaCo | 85.4 | 26.2 | **73.6** | 57.3 | 18.6 | **58.4** |
| MOORE | 88.7 | 61.4 | **83.2** | 72.9 | 61.2 | **72.0** |

**元 RL 结果（ML10/ML45）**

| 算法 | ML10 V1 | ML10 V2 | ML45 V1 | ML45 V2 |
|------|---------|---------|---------|---------|
| MAML | ~35% | ~35% | ~25% | ~25% |
| RL2 | ~15% | ~35% | ~10% | ~25% |

### 消融实验

| 配置 | MT10 成功率 | MT25 成功率 | MT50 成功率 | 说明 |
|------|----------|----------|----------|------|
| MTMHSAC (V2) | ~75% | ~65% | ~60% | 任务数增加导致性能下降（容量问题） |
| MAML ML10 | ~35% | ~35% (ML25) | ~35% (ML45) | 元 RL 对任务集规模不敏感 |

### 关键发现

- **所有多任务 RL 算法在 V2 上表现均优于 V1**：PCGrad 和 SM 在两种奖励下都是最优方法，这与它们分别通过梯度投影和软模块化缓解梯度冲突的机制一致。
- **元 RL 对奖励版本不敏感（RL2 除外）**：MAML 在 V1 和 V2 上无统计差异，因为其基于策略梯度而非 Q 学习。RL2 在 V1 上性能骤降是因为原始奖励直接作为观测输入且未归一化。
- **PaCo 的"虚假优势"被揭露**：PaCo 论文报告 MT10 85.4%，但实际使用的是 V2 奖励；在 V1 上仅 26.2%，表明先前的跨版本比较完全不可靠。

## 亮点与洞察

- 这是一篇非常扎实的基准修正工作，明确指出了社区中存在的"复制数字而非重跑实验"的不良习惯。
- V2 奖励设计的经验启示：**跨任务奖励的一致缩放**对多任务 RL 至关重要，直接影响 Q 函数学习质量。
- MT25/ML25 作为算力友好的中间选项具有实际价值，可用于初筛算法后再做 MT50 全面评估。

## 局限与展望

- 仅关注 Sawyer 单臂操作环境，未涉及跨形态迁移（cross-embodiment）。
- 虽揭示 V1/V2 差异但未设计新的更优奖励函数（V3），仅做保留式兼容处理。
- 元 RL 基线只测试了 MAML 和 RL2 两个经典方法，未测试更新的方法如 AMAGO-2。
- 任务多样性仍局限于桌面操作，缺乏导航、多阶段任务等更复杂场景。

## 相关工作与启发

- 与 RLBench、MANISKILL3 等操作基准相比，Meta-World 独特之处在于所有任务共享相同的状态和动作空间，使多任务/元学习成为可能。
- PopArt (Hessel et al., 2019) 的奖励缩放原则在此得到经验验证。
- 本文的教训可推广到所有 RL 基准：**版本管理和奖励函数设计应该被视为基准设计的一等公民**。

## 补充说明

- 基线代码库基于 JAX 实现，开源于 GitHub，便于社区复用和扩展。
- 论文附录包含所有 50 个任务的可视化、V1/V2 奖励函数的完整设计理由、MT25/ML25 任务集的构成细节。
- 本文对"从论文复制数字 vs 自己跑实验"的讨论值得所有基准论文和实验论文的作者关注。

## 评分

- 新颖性: ⭐⭐⭐ 工程贡献为主，方法创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 跨版本对比极为系统，10 种子 IQM 统计严格
- 写作质量: ⭐⭐⭐⭐ 条理清晰，问题阐述透彻
- 价值: ⭐⭐⭐⭐ 对社区的修正意义显著，标准化基准释出有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization](metabox-v2_a_unified_benchmark_platform_for_meta-black-box_optimization.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [\[NeurIPS 2025\] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](../../ICLR2026/reinforcement_learning/virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)

</div>

<!-- RELATED:END -->
