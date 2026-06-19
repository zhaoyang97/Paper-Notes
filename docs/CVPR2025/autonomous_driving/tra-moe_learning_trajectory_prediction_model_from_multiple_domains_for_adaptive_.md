---
title: >-
  [论文解读] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning
description: >-
  [CVPR 2025][自动驾驶][轨迹预测] 本文提出 Tra-MoE，利用稀疏门控混合专家(MoE)架构训练轨迹预测模型，有效融合大规模域外无动作视频数据与小规模域内机器人演示数据，并设计自适应策略条件化技术将 2D 轨迹与视觉观测显式对齐，在仿真和真实场景均显著提升机器人操控成功率。 领域现状：在机器人学习中…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "轨迹预测"
  - "混合专家"
  - "跨域学习"
  - "策略条件化"
  - "机器人操控"
---

# Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning

**会议**: CVPR 2025  
**arXiv**: [2411.14519](https://arxiv.org/abs/2411.14519)  
**代码**: [https://github.com/MCG-NJU/Tra-MoE](https://github.com/MCG-NJU/Tra-MoE)  
**领域**: 自动驾驶/机器人  
**关键词**: 轨迹预测, 混合专家, 跨域学习, 策略条件化, 机器人操控

## 一句话总结

本文提出 Tra-MoE，利用稀疏门控混合专家(MoE)架构训练轨迹预测模型，有效融合大规模域外无动作视频数据与小规模域内机器人演示数据，并设计自适应策略条件化技术将 2D 轨迹与视觉观测显式对齐，在仿真和真实场景均显著提升机器人操控成功率。

## 研究背景与动机

**领域现状**：在机器人学习中，一种可扩展的范式是先从无动作视频数据中学习轨迹预测模型，再用少量带动作标签的演示数据训练轨迹引导的策略模型。ATM 等工作在此方向取得了初步成功，但主要依赖域内数据训练轨迹模型。

**现有痛点**：如何有效利用大规模域外视频数据来联合训练轨迹模型仍未被充分探索。域外数据可能包含不同的环境、对象、技能和载体（embodiment），直接混合训练会导致优化冲突。实验表明，朴素地扩展域外训练数据反而导致域内任务性能下降 5.6 个百分点（57.6→52.0）。此外，如何有效地将预测的 2D 轨迹用于条件化策略模块也是一个未解决的挑战。

**核心矛盾**：域外数据能提供互补知识提升泛化能力，但不同域间数据分布差异巨大，统一 Transformer 模型难以在参数协作与专化之间取得平衡——协作要求共享参数学通用模式，专化要求不同子网络处理不同分布数据。

**本文目标**：(1) 设计能高效利用域外数据的轨迹模型架构；(2) 提出更有效的策略条件化方法，使 2D 轨迹更好地引导动作预测。

**切入角度**：作者观察到稀疏 MoE 架构天然具备参数协作（共享注意力层）与专化（不同专家处理不同输入）的双重能力，适合处理多域数据。同时，2D 轨迹直接映射到图像空间并编码为可学习嵌入，可以实现更灵活的空间对齐。

**核心 idea**：用稀疏门控 MoE（Top-1）替代 Transformer 中的部分 FFN 层构建 Tra-MoE，在保持恒定 FLOPs 的同时扩展模型容量以消化域外数据；再通过自适应轨迹掩码实现 2D 轨迹与图像的显式空间对齐。

## 方法详解

### 整体框架

Tra-MoE 的整体流程分两阶段：(1) **轨迹模型预训练**：在域外无动作视频 $\mathcal{D}_{ood}$ 与域内演示 $\mathcal{D}_{in}$ 上联合训练 MoE 轨迹预测模型，输入为图像观测 $o_t$、查询点集 $\mathbf{p}_t$ 和语言指令 $\ell$，输出未来 $H$ 步的任意点轨迹 $\mathbf{p}_{t:t+H} \in \mathbb{R}^{H \times K \times 2}$；(2) **策略训练**：冻结轨迹模型，仅在 $\mathcal{D}_{in}$ 上通过行为克隆训练策略模型，利用自适应条件化技术将轨迹信息融入视觉观测以预测机器人动作。

### 关键设计

1. **稀疏门控 MoE 轨迹模型**:

    - 功能：在保持恒定 FLOPs 的同时扩展模型容量，实现参数协作与专化的平衡
    - 核心思路：将 track transformer 中若干层的标准 FFN 替换为 MoE 块，每个 MoE 块包含 $N$ 个专家 FFN 和一个门控网络 $\mathcal{G}$。采用 Top-1 门控策略，即每个 token 仅激活一个专家：$\mathcal{G}(\mathbf{x}_s; \Theta)_i = \text{softmax}(\text{Top-1}(g(\mathbf{x}_s; \Theta), 1))_i$，最终输出为 $\mathcal{F}_{\text{sparse}}^{\text{MoE}} = \sum_{i=1}^{K} \mathcal{G}(\mathbf{x}_s)_i f_i(\mathbf{x}_s; \mathbf{W}_i)$。默认使用 4 个专家，替换第 1、5、8 层（数据更多时额外替换第 2、7 层）
    - 设计动机：MoE 的大部分参数（注意力层）被所有域数据共享训练，捕获通用互补模式实现协作；不同输入和不同 token 自然激活不同专家实现专化。实验证明即使将稠密模型扩展到与 MoE 相同参数量（宽度扩展 48.4/深度扩展 52.5 vs MoE 61.4），也无法获得等价效果

2. **MoE 训练策略（Router Z-Loss）**:

    - 功能：稳定稀疏 MoE 训练，避免门控网络输出过大 logits
    - 核心思路：作者系统研究了三种常用 MoE 辅助技术：(i) router z-loss $\mathcal{L}_z = \frac{1}{S}\sum_{k=1}^{S}(\log\sum_{i=1}^{N}e^{g_i^{(k)}})^2$ 惩罚门控 logits 过大，权重 $\lambda_z=10^{-4}$ 时提升 4.5 点；(ii) load-balancing loss $\mathcal{L}_{\text{lo-ba}} = N \sum_{i=1}^{N} \mathcal{Q}_i \mathcal{P}_i$ 平衡专家负载，但实验发现反而降低性能——因为数据分布不均时强制均衡会破坏专化优势；(iii) 向门控 logits 添加噪声也导致性能下降（56.9→55.8）
    - 设计动机：多域数据分布天然不均匀，强制均衡专家负载或加噪会抵消 MoE 的专化优势。仅使用 z-loss 稳定训练即可获得最佳效果

3. **自适应策略条件化**:

    - 功能：将 2D 轨迹显式对齐到图像空间，为策略模型提供更灵活的空间引导
    - 核心思路：构建额外的掩码通道，将 2D 轨迹点按空间位置填入掩码中，每个轨迹点设为可学习嵌入。将掩码与图像沿通道维度拼接为 $H \times W \times 4$ 的张量输入编码器。策略模型采用 early fusion（轨迹 token 与图像 token 在 fusion transformer 中交互）+ late fusion（融合特征与原始轨迹沿通道维度拼接）的双融合架构
    - 设计动机：轨迹不同位置的语义不同——起始点强调局部运动，终点关注全局趋势。手绘掩码（固定值128/255）虽有空间对齐但缺乏自适应性；可学习嵌入允许模型自动学习不同轨迹位置的最优表示。实验证明手绘掩码在 LIBERO-Goal 上大幅下降（81→58），而自适应掩码反而提升至 77

### 损失函数 / 训练策略

轨迹模型总损失 $\mathcal{L}_{\text{total}} = \lambda_{\text{tra}} \cdot \mathcal{L}_{\text{tra}} + \lambda_z \cdot \mathcal{L}_z$，其中 $\mathcal{L}_{\text{tra}}$ 为轨迹预测 MSE 损失，$\mathcal{L}_z$ 为 router z-loss（$\lambda_z = 10^{-4}$）。策略模型使用行为克隆的 MSE 损失训练。轨迹 GT 由 CoTracker 生成。

## 实验关键数据

### 主实验

仿真实验（LIBERO 基准，平均成功率 %）：

| 方法 | Spatial | Goal | Object | Long | Avg. |
|------|---------|------|--------|------|------|
| ATM (in-domain only) | 67.5 | 68.5 | 68.0 | 26.5 | 57.6 |
| ATM + OOD data | 49.5 | 67.0 | 56.5 | 35.0 | 52.0 |
| **Tra-MoE + OOD data** | **62.5** | **81.0** | **73.5** | **28.5** | **61.4** |
| + Adaptive mask | **69.5** | **77.0** | **88.0** | **30.5** | **66.3** |

真实世界实验（5 个任务，成功率 %）：

| 配置 | Pour | Push | Pick&Pass | Tissue | Fold | Avg. |
|------|------|------|-----------|--------|------|------|
| Baseline | 40.0 | 45.0 | 50.0 | 30.0 | 25.0 | 38.0 |
| + Human data | 45.0 | 35.0 | 50.0 | 25.0 | 35.0 | 38.0 |
| + Human + MoE | 40.0 | 50.0 | 60.0 | 35.0 | 45.0 | 46.0 |
| + Human + MoE + Adaptive | 60.0 | 70.0 | 65.0 | 35.0 | 50.0 | **56.0** |

### 消融实验

| 配置 | Avg. 成功率 | 说明 |
|------|-----------|------|
| Tra-MoE (4 experts, z-loss) | 61.4 | 完整模型 |
| w/o z-loss | 56.9 | 去掉 z-loss 掉 4.5 点 |
| + load-balancing loss (1e-3) | 52.6 | 负载均衡损失反而有害 |
| + noise to gating | 55.8 | 加噪也降低性能 |
| Dense 扩展 (width) | 48.4 | 同参数量稠密模型效果差 |
| Dense 扩展 (depth) | 52.5 | 深度扩展也不如 MoE |
| 2 experts | 56.0 | 少量专家已有效 |
| 3 experts | 55.1 | - |
| 4 experts | 56.9 | 更多专家趋势向好 |

### 关键发现

- MoE 架构对域外数据的利用至关重要：朴素扩展域外数据使稠密模型下降 5.6 点，但 Tra-MoE 反而提升 9.6 点（51.8→61.4）
- 将稠密模型扩展至与 MoE 相同参数量（宽度/深度扩展）完全无法匹配 MoE 的性能，证明了稀疏激活的结构优越性
- Load-balancing loss 在多域不均匀分布下有害，只有 z-loss 对稳定训练有帮助
- 跨 RLbench（不同物理引擎）数据训练时 Tra-MoE 仍优于 baseline 12.6 点
- 自适应掩码相比手绘掩码在 Goal 子任务上有 19 点优势，因其可学习嵌入能处理轨迹前后段重叠问题

## 亮点与洞察

- MoE 在机器人轨迹学习中的首次系统应用——从 NLP/CV 迁移 MoE 时发现 load-balancing loss 在多域机器人数据上反而有害，这一发现对后续工作有指导意义
- 自适应策略条件化将 2D 轨迹"画"到图像上作为额外通道，既简单又有效，核心洞察是轨迹不同位置的语义差异需要可学习表示来捕捉
- 域外数据 + MoE 的协同效应值得注意：单独加域外数据有害，单独加 MoE 也有害，但同时使用后产生了 9.6 点的大幅提升

## 局限与展望

- 实验规模相对有限——仿真仅在 LIBERO 上验证，真实世界仅 5 个任务各 50 条演示
- 域外数据与域内数据的比例（9:2）需要手动调整，缺乏自动化的数据混合策略
- 未探索更复杂的 MoE 路由策略（如 expert-choice、soft MoE 等）
- 未来方向：(1) 将方法扩展到更大规模的互联网视频数据；(2) 结合 3D 轨迹预测以获得更精确的动作映射；(3) 探索域外数据质量对模型性能的影响

## 相关工作与启发

- **vs ATM**：ATM 是本文的基线，仅用域内数据训练轨迹模型。Tra-MoE 在此基础上引入 MoE 和域外数据，将平均成功率从 57.6 提升到 66.3
- **vs RT-Trajectory**：RT-Trajectory 使用手绘轨迹掩码（固定值）条件化策略，本文的自适应可学习嵌入方式更灵活，特别是在轨迹重叠场景下表现更好
- **vs 大规模机器人基础模型（RT-2, Octo）**：这些工作采用预训练+微调范式且直接组合异构动作空间，可能导致次优解；Tra-MoE 通过无动作视频的轨迹空间统一了不同载体

## 评分

- **新颖性**: ⭐⭐⭐⭐ — MoE 在机器人轨迹学习中的应用新颖，但核心技术（MoE、轨迹条件化）各自已有先例
- **实验充分度**: ⭐⭐⭐⭐ — 仿真消融全面且深入，真实世界实验补充了实用性验证，但规模较小
- **写作质量**: ⭐⭐⭐⭐ — 研究问题清晰，消融设计逻辑严谨，finding 分析透彻
- **价值**: ⭐⭐⭐⭐ — MoE 在多域机器人学习中的发现（balancing loss 有害等）对社区有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[CVPR 2026\] W2W: Language-Model-Based Trajectory Prediction with Reinforcement Learning](../../CVPR2026/autonomous_driving/w2w_language-model-based_trajectory_prediction_with_reinforcement_learning.md)
- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](../../ICCV2025/autonomous_driving/donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)

</div>

<!-- RELATED:END -->
