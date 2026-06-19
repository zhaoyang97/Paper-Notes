---
title: >-
  [论文解读] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data
description: >-
  [NeurIPS 2025][机器人][操作概念] 提出自监督框架从无标注多模态机器人演示中学习层级操作概念，通过跨模态相关性网络和多时域子目标预测器组织表示，增强模仿学习策略在新物体、新障碍和新环境下的泛化能力。 1. 泛化瓶颈：当前机器人操作策略在训练分布内表现良好，但遇到未见障碍物、新物体外观或新环境时常常失败（如训练…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "操作概念"
  - "层级表示"
  - "跨模态相关性"
  - "多时间尺度子目标"
  - "自监督学习"
---

# HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data

**会议**: NeurIPS 2025  
**arXiv**: [2510.11321](https://arxiv.org/abs/2510.11321)  
**代码**: [HiMaCon](https://github.com/zrllrz/HiMaCon)  
**领域**: 机器人操作 / 表示学习 / 模仿学习  
**关键词**: 操作概念, 层级表示, 跨模态相关性, 多时间尺度子目标, 自监督学习

## 一句话总结
提出自监督框架从无标注多模态机器人演示中学习层级操作概念，通过跨模态相关性网络和多时域子目标预测器组织表示，增强模仿学习策略在新物体、新障碍和新环境下的泛化能力。

## 背景与动机

1. **泛化瓶颈**：当前机器人操作策略在训练分布内表现良好，但遇到未见障碍物、新物体外观或新环境时常常失败（如训练时无障碍→测试时有障碍）。
2. **表示学习的不足**：
    - 单模态方法（纯视觉或纯本体感觉）无法捕捉跨模态的功能不变性
    - 时间表示学习方法忽视了操作任务的层级时间结构
    - 跨模态对齐方法仅做特征拼接/对齐，未建模模态间的相关性
3. **核心假设**：操作概念（如"将物体放入容器"）编码了跨物体、跨环境持续存在的不变关系模式。通过同时建模跨模态相关性和多时间尺度子目标，可以学到可迁移的层级操作概念。

## 核心问题
如何从无标注的多模态机器人演示中自动发现层级化的操作概念，使其既编码跨模态功能不变性又组织为多时间尺度子目标，从而增强策略泛化？

## 方法详解

### 整体框架（两阶段）
**Stage 1: 概念发现** — 概念编码器 $\mathcal{E}$ 将多模态观测映射为概念 latent，通过两个目标函数训练：
- 跨模态相关性网络 (CMCN) $\mathcal{C}$
- 多时域未来预测器 (MHFP) $\mathcal{F}$

**Stage 2: 策略增强** — 将学到的概念通过联合预测头集成到模仿学习中。

### 概念编码器

给定轨迹 $\tau_i = \{(\mathbf{o}_i^t, a_i^t)\}_{t=1}^{T_i}$，多模态观测 $\mathbf{o}_i^t = \{o_i^{1,t},...,o_i^{M,t}\}$，编码器输出概念序列：

$$\mathbf{z}_i \leftarrow \mathcal{E}(\mathbf{o}_i;\Theta_\mathcal{E})$$

使用 Transformer 捕获时间依赖，每个时间步得到 $z_i^t \in \mathbb{R}^Z$。

### 跨模态相关性学习

核心思想：操作概念应捕获跨模态的相关性（视觉+本体感觉+力反馈之间的持续模式），而非简单拼接特征。最大化条件互信息：

$$\max_\mathbf{Z}\sum_{S\subsetneq[M], S\neq\emptyset} \mathbb{I}(\mathbf{O}_S : \mathbf{O}_{[M]\setminus S} \mid \mathbf{Z})$$

通过 mask-and-predict 策略实现：随机遮蔽部分模态，用未遮蔽模态和概念重构全部观测：

$$\mathcal{L}_\text{mm}(t, \tau_i) = \mathbb{E}_S \|\mathcal{C}(o_i^{[M]\setminus S,t}, z_i^t;\Theta_c) - o_i^t\|$$

### 多时域子目标表示

用球面距离量化概念相似性：$\text{dist}(z,u) = \frac{1}{\pi}\arccos\langle\frac{z}{\|z\|_2}, \frac{u}{\|u\|_2}\rangle$

相干性阈值 $\epsilon$ 决定子过程粒度：小 $\epsilon$ → 短时域细粒度子目标；大 $\epsilon$ → 长时域粗粒度目标。子过程由概念 latent 一致性自动划分。

多时域预测器学习预测每个子过程的终止状态观测：

$$\mathcal{L}_\text{mh}(t, \tau_i) = \mathbb{E}_\epsilon\|\mathcal{F}(\mathbf{o}_i^t, z_i^t, \epsilon;\Theta_f) - \mathbf{o}_i^{g(t;\mathbf{z}_i,\epsilon)}\|$$

### 总训练目标

$$\mathcal{L}_z(t, \tau_i) = \lambda_\text{mm}\mathcal{L}_\text{mm}(t, \tau_i) + \lambda_\text{mh}\mathcal{L}_\text{mh}(t, \tau_i)$$

### 策略增强（Stage 2）

将概念预测作为正则化集成到模仿学习：

$$\mathcal{L}_\pi(t, \tau_i, \ell_i) = \|\hat{a}_i^t - a_i^t\| + \lambda_\text{mc}\|\hat{z}_i^t - z_i^t\|$$

策略包含共享骨干 $\pi_h$、概念预测头 $\pi_z$ 和动作解码头 $\pi_a$，兼容 ACT 和 Diffusion Policy。

## 实验关键数据

### LIBERO 基准（概念仅在 L90 训练）

| 设置 | 策略 | Plain | XSkill | RPT | **HiMaCon** |
|------|------|-------|--------|-----|-------------|
| L90 (原任务) | ACT | 46.6 | 73.4 | 68.8 | **74.8** |
| L90 (原任务) | DP | 75.1 | 87.7 | 84.3 | **89.6** |
| L-LONG (长时域迁移) | ACT | 54.0 | 55.0 | 59.0 | **63.0** |
| L-LONG (长时域迁移) | DP | 34.1 | 73.0 | 61.3 | **89.0** |
| L-GOAL (新环境泛化) | ACT | 57.0 | 77.0 | 75.0 | **81.0** |
| L-GOAL (新环境泛化) | DP | 90.7 | 93.0 | 91.5 | **95.7** |

关键发现：
- 在长时域迁移（L-LONG + DP）上比 Plain 提升 **54.9 个百分点**（34.1→89.0）
- 在新环境泛化上持续领先，证明概念的迁移能力
- 与 11 种基线方法对比均为最佳或次佳

### 真实机器人实验
论文在真实机器人上部署验证，概念增强策略在面对未见障碍物时成功适应（如放杯子时绕过障碍），而无概念策略直接失败。

## 亮点
- **理论动机扎实**：从认知科学（多模态相关性驱动概念形成）和运动控制（层级目标组织）中汲取灵感
- **自监督设计优雅**：mask-and-predict 同时实现跨模态相关性学习和信息压缩
- **$\epsilon$ 控制层级**：单一连续参数自然生成从短到长的子目标层级，无需预设层级数
- **架构无关**：概念增强通过联合预测头实现，兼容 ACT、Diffusion Policy 等不同策略架构
- **概念可解释**：学到的概念自动聚成类似人类理解的操作原语（抓取、放置、对齐等）

## 局限性
- 概念编码器需要在演示数据上预训练（Stage 1），额外增加了训练流水线复杂度
- 球面距离 + $\epsilon$ 阈值的子过程划分对概念 latent 空间的几何结构有隐式假设
- LIBERO 中的任务相对简单，在更复杂的双手操作或接触丰富任务上的效果未验证
- 真实机器人实验规模有限，统计显著性不强

## 评分
- 新颖性: ⭐⭐⭐⭐ — 跨模态+多时域子目标的层级概念发现是新颖组合
- 实验充分度: ⭐⭐⭐⭐ — 11 种基线、3 种评估设置、2 种策略、真实机器人
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、动机和设计之间的联系紧密
- 综合价值: ⭐⭐⭐⭐⭐ — 对机器人操作表示学习有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](../../ACL2025/robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)
- [\[CVPR 2026\] AURA: Multi-modal Shared Autonomy for Urban Navigation](../../CVPR2026/robotics/aura_multi-modal_shared_autonomy_for_urban_navigation.md)
- [\[CVPR 2026\] FloVerse: Floor Plan-Guided Multi-Modal Navigation](../../CVPR2026/robotics/floverse_floor_plan-guided_multi-modal_navigation.md)
- [\[NeurIPS 2025\] Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms](operation_veja_fixing_fundamental_concepts_missing_from_modern_roleplaying_train.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)

</div>

<!-- RELATED:END -->
