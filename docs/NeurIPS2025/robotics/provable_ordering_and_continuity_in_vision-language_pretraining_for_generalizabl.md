---
title: >-
  [论文解读] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents
description: >-
  [NeurIPS 2025][机器人][视觉语言预训练] 提出 AcTOL，通过视觉-语言排序损失和布朗桥约束来学习有序且连续的视觉-语言表征，无需刚性目标到达假设，在模拟和真实机器人操作任务上显著提升下游表现。 领域现状 领域现状：利用人类动作视频预训练视觉-语言表征以减少机器人专家演示依赖是一个有前景的方向…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "视觉语言预训练"
  - "体化智能"
  - "模仿学习"
  - "时序一致性"
  - "布朗桥"
---

# Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents

**会议**: NeurIPS 2025  
**arXiv**: [2502.01218](https://arxiv.org/abs/2502.01218)  
**代码**: [https://actol-pretrain.github.io/](https://actol-pretrain.github.io/)  
**领域**: 强化学习  
**关键词**: 视觉语言预训练, 体化智能, 模仿学习, 时序一致性, 布朗桥

## 一句话总结
提出 AcTOL，通过视觉-语言排序损失和布朗桥约束来学习有序且连续的视觉-语言表征，无需刚性目标到达假设，在模拟和真实机器人操作任务上显著提升下游表现。

## 研究背景与动机

### 领域现状

**领域现状**：利用人类动作视频预训练视觉-语言表征以减少机器人专家演示依赖是一个有前景的方向。R3M、LIV、DecisionNCE 等方法使用时间对比学习

**现有痛点**：现有方法基于"目标到达"假设——假设视频中语言指令的语义与越靠后的帧对齐越好。但实际视频中动作可能提前终止或包含不相关后续内容，导致错误的视觉-语言关联

**核心矛盾**：真实人类动作视频标注粗糙、充满噪声，刚性假设不成立

**核心 idea**：利用视频内在的时间一致性，让表征满足**有序性**（时间更近的帧语义差异更小）和**连续性**（相邻帧的表征平滑过渡）

## 方法详解

### 关键设计

1. **视觉-语言排序 (VLO) 损失**：

    - 核心思路：对锚帧 $o_i$ 和任意帧对 $(o_j, o_k)$，定义语义对齐差分 $\mathfrak{R}(\mathbf{v}_i, \mathbf{v}_j, \mathbf{l}) = -\|\text{sim}(\mathbf{v}_i, \mathbf{l}) - \text{sim}(\mathbf{v}_j, \mathbf{l})\|_2$
    - 负样本集合 $\mathcal{N}_{i,j}$ 选择时间距离**更远**的帧，用 InfoNCE 风格损失对比
    - 理论保证：当 $\mathcal{L}_{VLO}$ 接近下界 $\mathcal{L}^*$ 时，表征满足 VLO 性质

2. **布朗桥约束**：

    - 将视频帧间隔建模为布朗桥过程：均值线性插值，方差中间最大
    - 损失：$\mathcal{L}_{BB} = \frac{1}{T}\sum_{t} \frac{1}{2\text{Var}[\mathbf{B}(t)]}\|\mathbf{v}_t - \mathbb{E}[\mathbf{B}(t)]\|^2$
    - 保证视觉表征局部平滑

3. **语言鲁棒性**：理论证明表征对语言扰动 $\|\mathbf{l} - \mathbf{l}'\| \leq \delta_l$ 的语义对齐变化 $\leq 2C\delta_l$

## 实验关键数据

### 主实验 — 模拟环境成功率 (15 demos)

| 方法 | Franka Kitchen | Metaworld |
|------|---------------|-----------|
| CLIP | 27.47 | 60.33 |
| R3M | 42.20 | 56.50 |
| LIV | 42.73 | 64.33 |
| DecisionNCE | 43.20 | 59.08 |
| AcTOL w/o BB | 54.20 | 70.83 |
| **AcTOL** | **61.80 (+43%)** | **74.13 (+15%)** |

### 真实机器人 — Unitree D1

| 方法 | Pick Cup | Open Drawer | Close Drawer |
|------|---------|------------|-------------|
| DecisionNCE | 20% | 40% | 60% |
| **AcTOL** | **50%** | **80%** | **90%** |

### 关键发现
- 布朗桥约束贡献显著（AcTOL vs w/o BB: +7.6% Franka Kitchen）
- AcTOL 在语言扰动下性能几乎不下降，而 LIV 下降 11.9%
- 少量 5 个演示时 AcTOL 甚至超过其他方法用 15-25 个演示的表现

## 亮点与洞察
- **不假设终帧是目标**是关键创新——只用帧间的相对时间距离来约束表征，更鲁棒
- **布朗桥作为连续性正则化器**的想法优雅：自然地将不确定性建模引入时间表征

## 局限与展望
- 对循环/重复动作（如搅拌）可能不适用，因为时间排序假设不成立
- 预训练数据集仅 EPIC-KITCHEN-100，未验证在更大数据集上的表现

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ VLO + 布朗桥的组合新颖且有理论保证
- 实验充分度: ⭐⭐⭐⭐⭐ 模拟+真实/鲁棒性/消融/微调全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析深入，实验设计精心
- 价值: ⭐⭐⭐⭐⭐ 显著推进了体化预训练的前沿

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [\[CVPR 2025\] g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks](../../CVPR2025/robotics/g3d-lf_generalizable_3d-language_feature_fields_for_embodied_tasks.md)

</div>

<!-- RELATED:END -->
