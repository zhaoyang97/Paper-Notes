---
title: >-
  [论文解读] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
description: >-
  [CVPR 2025][机器人][长程导航] 定义长程视觉语言导航（LH-VLN）任务，构建 NavGen 自动生成平台和 LHPR-VLN 基准（3260 个多阶段任务，平均 150 步），提出 MGDM 方法通过短期记忆模糊+长期记忆检索+CoT反馈实现多阶段导航，在 ISR 指标上超越 NaviLLM 23%。
tags:
  - "CVPR 2025"
  - "机器人"
  - "长程导航"
  - "多阶段任务"
  - "视觉语言导航"
  - "记忆机制"
  - "基准评估"
---

# Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method

**会议**: CVPR 2025  
**arXiv**: [2412.09082](https://arxiv.org/abs/2412.09082)  
**代码**: [https://hcplab-sysu.github.io/LH-VLN](https://hcplab-sysu.github.io/LH-VLN)  
**领域**: 机器人 / 视觉语言导航  
**关键词**: 长程导航, 多阶段任务, 视觉语言导航, 记忆机制, 基准评估

## 一句话总结

定义长程视觉语言导航（LH-VLN）任务，构建 NavGen 自动生成平台和 LHPR-VLN 基准（3260 个多阶段任务，平均 150 步），提出 MGDM 方法通过短期记忆模糊+长期记忆检索+CoT反馈实现多阶段导航，在 ISR 指标上超越 NaviLLM 23%。

## 研究背景与动机

**领域现状**：视觉语言导航（VLN）让智能体按自然语言指令在3D环境中导航。现有基准（如 R2R、VLN-CE）平均路径只有 55 步，指令涉及单一目标——远低于真实场景需求。

**现有痛点**：真实场景中的导航通常是多阶段的（"先去厨房拿杯子，再去客厅放到桌上"），涉及 150+ 步的长程规划。现有方法和基准都无法评估这种多阶段长程能力。

**核心矛盾**：长程多阶段导航需要处理子任务间的依赖关系（先完成 A 才能开始 B），但现有评估指标（如 SR、SPL）只看最终结果，无法衡量中间阶段的正确性。

**切入角度**：定义三个新指标（ISR/CSR/CGT）分别评估独立子任务成功率、条件子任务成功率和路径难度加权成功率。用 GPT-4 + NavGen 平台自动生成大规模多阶段任务。

**核心 idea**：新任务定义 + 新评估指标 + 大规模自动生成基准 = 长程多阶段 VLN。

## 方法详解

### 关键设计

1. **NavGen 自动生成平台**:

    - 功能：从 3D 场景自动生成多阶段导航任务
    - 核心思路：GPT-4 给定场景中的对象列表和拓扑结构，自动生成包含 2-4 个子任务的导航指令。每个子任务有独立的起点/终点和成功判定条件

2. **MGDM（记忆引导的决策模型）**:

    - 功能：处理长程导航中的记忆管理
    - 核心思路：短期记忆模糊（将近期观察压缩为摘要避免信息过载）+ 长期记忆检索（从历史中检索相关经验）+ Chain-of-Thought 反馈（用 CoT 分析当前状态并决定下一步行动）

3. **三个新评估指标**:

    - ISR (独立子任务成功率)：每个子任务独立评估
    - CSR (条件子任务成功率)：考虑子任务间依赖（前一个失败→后续全算失败）
    - CGT：CSR 加路径难度权重

### 损失函数 / 训练策略

模仿学习 + 交叉熵损失。LHPR-VLN 包含 3260 个任务，覆盖 216 个 HM3D 场景，39% 两子任务 / 52.4% 三子任务 / 8.6% 四子任务。

## 实验关键数据

### 主实验

| 方法 | ISR | CSR | CGT |
|------|-----|-----|-----|
| NaviLLM (微调) | 3.81% | 1.67% | 2.54% |
| **MGDM** | **4.69%** | **3.30%** | **5.83%** |

所有基线在 2-3 子任务上成功率接近 0%，说明长程导航极具挑战性。

### 关键发现
- **长程导航对所有现有方法都极具挑战**：最优方法仅 ~5% 成功率
- **子任务依赖严重影响性能**：CSR 远低于 ISR，说明错误级联是主要失败原因
- **记忆机制对长程关键**：无记忆管理的方法完全失败

## 亮点与洞察
- **任务定义和基准是核心贡献**——揭示了现有 VLN 方法在长程场景下的根本性不足
- **极低的绝对性能**——5% 成功率意味着长程 VLN 是一个远未解决的开放问题

## 局限与展望
- 基线方法性能极低，方法改进空间有限难以展示
- 仅限 Habitat 仿真器
- 记忆机制为特定任务设计，通用性待验证

## 评分
- 新颖性: ⭐⭐⭐⭐ 新任务+新指标+新基准的贡献体系完整
- 实验充分度: ⭐⭐⭐⭐ 多基线但绝对性能太低
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 为 VLN 社区指明了长程方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[CVPR 2026\] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](../../CVPR2026/robotics/recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)
- [\[CVPR 2026\] NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction](../../CVPR2026/robotics/navforesee_a_unified_vision-language_world_model_for_hierarchical_planning_and_d.md)
- [\[ICML 2025\] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](../../ICML2025/robotics/closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [\[NeurIPS 2025\] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](../../NeurIPS2025/robotics/rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)

</div>

<!-- RELATED:END -->
