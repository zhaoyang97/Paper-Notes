---
title: >-
  [论文解读] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling
description: >-
  [ICML 2025][机器人][机器人规划] 将 LLM 的自精炼规划过程建模为不动点问题（深度均衡模型），通过隐式微分实现端到端监督训练，无需额外验证器或 RL，并设计嵌套均衡求解实现闭环长程机器人规划。 领域现状 领域现状：LLM 在机器人任务规划中展现潜力，但受限于单向依赖（无法回顾已生成的 token）、缺乏错误纠…
tags:
  - "ICML 2025"
  - "机器人"
  - "机器人规划"
  - "自精炼"
  - "深度均衡模型"
  - "长程规划"
  - "推理时计算"
---

# Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling

**会议**: ICML 2025  
**arXiv**: [2410.01440](https://arxiv.org/abs/2410.01440)  
**代码**: [https://github.com/Singularity0104/equilibrium-planner](https://github.com/Singularity0104/equilibrium-planner)  
**领域**: 机器人  
**关键词**: 机器人规划, 自精炼, 深度均衡模型, 长程规划, 推理时计算

## 一句话总结
将 LLM 的自精炼规划过程建模为不动点问题（深度均衡模型），通过隐式微分实现端到端监督训练，无需额外验证器或 RL，并设计嵌套均衡求解实现闭环长程机器人规划。

## 研究背景与动机

### 领域现状

**领域现状**：LLM 在机器人任务规划中展现潜力，但受限于单向依赖（无法回顾已生成的 token）、缺乏错误纠正、固定计算量无法动态分配。

**现有痛点**：自精炼（self-refinement）策略可解决上述问题（引入双向依赖+动态纠错），但训练困难——需要通过无限自精炼步骤做反向传播，或构建复杂的 RL/验证器管道。

**核心矛盾**：自精炼的训练如何简单高效地实现？

**本文目标**：用简单的监督学习训练自精炼的 LLM 规划器。

**切入角度**：将自精炼视为不动点迭代 $x_{t+1} = f_\theta(x_t, c)$，理想计划是均衡点 $x^* = f_\theta(x^*, c)$。

**核心 idea**：用深度均衡模型的隐式微分绕过无限步反向传播，实现端到端监督训练。

## 方法详解

### 整体框架
1. 将 LLM 规划器定义为不动点映射 $f_\theta$
2. 前向推理：用 Anderson/Broyden 方法求解均衡点 $x^* = f_\theta(x^*, c)$
3. 反向传播：用隐式函数定理计算梯度（无需展开所有迭代步）
4. 嵌套均衡：内循环精炼计划，外循环收集环境反馈

### 关键设计

1. **均衡序列建模**:

    - 功能：将 LLM 自精炼建模为不动点问题
    - 核心思路：理想计划是精炼过程的不动点——再怎么精炼也不会改变
    - Jacobian-free 近似简化梯度计算
    - 设计动机：避免展开无限步的反向传播，用隐式微分实现 O(1) 内存训练

2. **嵌套均衡求解**:

    - 功能：内循环精炼计划（固定反馈），外循环更新反馈（与环境交互）
    - 核心思路：重用前一个均衡解作为下一轮的初始化，加速收敛
    - 设计动机：高效整合闭环环境反馈

3. **世界模型辅助**:

    - 功能：在无法与真实环境交互时用世界模型估计反馈
    - 核心思路：训练一个小型世界模型预测行动后果
    - 设计动机：减少真实环境交互次数

### 损失函数 / 训练策略
- 纯监督学习（无 RL、无验证器）
- 损失：均衡点与真实计划的交叉熵
- 推理时可动态增加迭代次数来提升质量

## 实验关键数据

### 主实验
VirtualHome-Env 基准：

| 方法 | 成功率 | 可执行率 |
|------|--------|---------|
| ReAct (LLM) | 42.3% | 65.1% |
| Tree-of-Thought | 51.7% | 72.4% |
| **均衡规划器** | **58.9%** | **78.2%** |

### 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| 单次生成（无精炼） | 38.5% | 基线 |
| 固定 3 步精炼 | 52.1% | 改进但不自适应 |
| 均衡精炼（动态步数） | **58.9%** | 自适应分配计算 |
| 无世界模型 | 53.2% | 反馈不足 |
| +世界模型 | **58.9%** | 完整方法 |

### 关键发现
- 推理时计算量与规划质量正相关——更多迭代 = 更好的计划
- 均衡模型比 tree search 方法更高效（不需要枚举分支）
- 嵌套均衡的初始化复用使收敛速度提升 2-3×
- 简单监督学习就能训练出有效的自精炼规划器

## 亮点与洞察
- **均衡模型 × LLM 规划**的结合非常优雅——将 deep equilibrium models 从视觉/图像生成扩展到序列规划
- 隐式微分实现了"无限深度的精炼过程可用有限内存训练"
- 推理时计算扩展（inference-time scaling）是当前热点，本文提供了非 tree search 的新路径

## 局限与展望
- 不动点存在性和唯一性没有理论保证
- VirtualHome 环境相对简单，真实机器人场景待验证
- 世界模型的准确性是瓶颈

## 相关工作与启发
- **vs Tree-of-Thought**: Tree search 枚举分支，均衡模型迭代精炼，后者更高效
- **vs DeepSeek R1**: 用 RL 训练推理，本文用监督学习+均衡模型更简单
- **vs Reflexion/Self-Refine**: 依赖提示工程，本文端到端训练

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 均衡模型用于规划，非常新颖
- 实验充分度: ⭐⭐⭐⭐ 充分消融和扩展性分析
- 写作质量: ⭐⭐⭐⭐⭐ 数学优雅，动机清晰
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 规划和推理时计算有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](../../ICML2026/robotics/drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)
- [\[ICCV 2025\] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation](../../ICCV2025/robotics/pasg_a_closed-loop_framework_for_automated_geometric_primitive_extraction_and_se.md)
- [\[ICML 2025\] Efficient Robotic Policy Learning via Latent Space Backward Planning](efficient_robotic_policy_learning_via_latent_space_backward_planning.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[ICML 2026\] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](../../ICML2026/robotics/hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)

</div>

<!-- RELATED:END -->
