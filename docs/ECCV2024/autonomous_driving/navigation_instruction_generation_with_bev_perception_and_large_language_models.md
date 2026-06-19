---
title: >-
  [论文解读] Navigation Instruction Generation with BEV Perception and Large Language Models
description: >-
  [ECCV 2024][自动驾驶][导航指令生成] 提出 BEVInstructor，将鸟瞰图 (BEV) 特征融合到多模态大语言模型中，通过 Perspective-BEV 融合编码器、参数高效的 Prompt Tuning 以及实例引导的迭代优化策略，在室内外导航指令生成任务上取得 SOTA。 导航指令生成要求具身智能体…
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "导航指令生成"
  - "鸟瞰图感知"
  - "多模态大语言模型"
  - "提示学习"
  - "迭代优化"
---

# Navigation Instruction Generation with BEV Perception and Large Language Models

**会议**: ECCV 2024  
**arXiv**: [2407.15087](https://arxiv.org/abs/2407.15087)  
**代码**: [有](https://github.com/FanScy/BEVInstructor)  
**领域**: 自动驾驶 / 具身智能  
**关键词**: 导航指令生成, 鸟瞰图感知, 多模态大语言模型, prompt tuning, 迭代优化

## 一句话总结

提出 BEVInstructor，将鸟瞰图 (BEV) 特征融合到多模态大语言模型中，通过 Perspective-BEV 融合编码器、参数高效的 Prompt Tuning 以及实例引导的迭代优化策略，在室内外导航指令生成任务上取得 SOTA。

## 研究背景与动机

导航指令生成要求具身智能体根据导航轨迹用自然语言描述路线，在机器人、人机交互等领域有重要价值，例如辅助视障人士导航、自主搜救报告等。

**现有方法的局限性**：

**缺乏 3D 感知**：现有方法 (如 CCC-speaker、Lana) 直接将 2D 透视观测映射为路线描述，忽略了 3D 环境的几何信息和物体语义，容易产生模糊的路径描述。

**MLLM 的领域差距**：多模态大语言模型 (GPT-4V、InstructBLIP 等) 主要在第三人称的独立图像-文本对上预训练，难以直接理解第一人称视角序列的空间上下文。零样本方式生成导航指令效果不佳。

**缺少逐步优化**：认知科学研究表明，人类描述路线时会先根据地标构思草稿再逐步完善，但现有方法缺乏这种迭代优化机制。

**核心动机**：引入 BEV 感知来编码 3D 空间语义和几何结构，结合 MLLM 的强大语言能力，并模拟人类"先标志物草稿 → 再完善描述"的过程来提升指令质量。

## 方法详解

### 整体框架

BEVInstructor 基于 LLaMA-7B，包含三大模块：(1) **Perspective-BEV Visual Encoder** 编码 3D 场景信息；(2) **Perspective-BEV Prompt Tuning** 实现参数高效的跨模态对齐；(3) **实例引导的迭代优化** 渐进式提升指令质量。

任务定义：给定导航路径的观测序列 $\mathcal{O} = \{O_t\}_{t=1}^T$ 和动作序列 $\mathcal{A} = \{a_t\}_{t=1}^T$，自回归生成指令 $\mathcal{X} = \{x_l\}_{l=1}^L$：

$$\max_\Theta \sum_{l=1}^L \log P_\Theta(x_l | x_{<l}, \mathcal{O}, \mathcal{A})$$

### 关键设计

1. **Perspective-BEV Visual Encoder**：编码 3D 环境语义和几何信息

    - **透视嵌入**：将多视角图像特征 $F_{t,k}$ 与方向角编码 $\delta_{t,k}$、时间步嵌入组合为 $p_{t,k} = \mathcal{E}^p(F_{t,k}) + \mathcal{E}^\delta(\delta_{t,k}) + E_t + E_o$
    - **BEV 嵌入**：通过 BEV 编码器 (6 层可变形注意力) 将多视角特征聚合到 $15 \times 15$ 的 BEV 网格，使用深度一致性权重 $w_{k,n}^c$ 区分不同深度的参考点投影。BEV 编码器在 3D 检测任务监督下预训练后冻结
    - **Perspective-BEV 融合**：用 Transformer 层融合 BEV 嵌入 $B_t$ 和透视嵌入 $[P_t, a_t]$，再通过轻量 Transformer $\mathcal{Q}$ 将 $H_b W_b$ 个 token 压缩为 $N_q = 10$ 个固定长度 token，避免输入 MLLM 时 token 过长
    - **设计动机**：2D 透视特征保留丰富视觉线索但缺乏 3D 几何，BEV 特征编码空间结构但缺乏纹理细节，两者互补融合实现全面场景理解

2. **Perspective-BEV Prompt Tuning**：参数高效地利用 MLLM 的跨模态能力

    - 在视觉嵌入 $O_{1:T}$ 中插入 $N_p$ 个可学习嵌入作为 Perspective-BEV Prompt：$O' = O_{1:T} \oplus E_v$
    - 在 LLaMA 最后 $N_a = 31$ 层引入 zero-initialized attention 和可学习 scale vector
    - **设计动机**：直接微调 MLLM 代价高且可能损害文本生成能力。通过仅更新 7.2% 的参数实现参数高效的场景-指令对齐

3. **实例引导的迭代优化**：模拟人类描述路线的认知过程

    - 第一阶段：BEVInstructor 先识别关键实例，生成地标 token $\mathcal{X}^I$
    - 第二阶段：基于地标草稿条件下生成完整指令：$\mathcal{O} \times \mathcal{A} \times \mathcal{X}^I \rightarrow \mathcal{X}$
    - **设计动机**：认知科学表明关键地标在人类路线描述中起核心作用，分阶段生成可逐步丰富指令中的物体语义

### 损失函数 / 训练策略

- **BEV 编码器预训练**：使用 $\ell_1$ loss + 交叉熵 loss 在 3D 检测任务上监督训练，然后冻结
- **指令生成训练**：自回归交叉熵损失，结合地标生成和指令生成的联合优化 (Eq. 11)
- 使用 AdamW 优化器，学习率 $1e^{-4}$，batch size 8，20K 迭代
- 冻结 LLaMA 大部分参数 (6.68B)，仅微调 <500M 参数

## 实验关键数据

### 主实验

在三个数据集上与 SOTA 比较：

| 数据集 | 指标 | BEVInstructor | 之前 SOTA (Lana) | 提升 |
|--------|------|---------------|-----------------|------|
| R2R val seen | SPICE | **0.220** | 0.201 | +1.9% |
| R2R val seen | CIDEr | **0.549** | 0.503 | +4.6% |
| R2R val unseen | SPICE | **0.208** | 0.194 | +1.4% |
| R2R val unseen | CIDEr | **0.449** | 0.419 | +3.0% |
| REVERIE val seen | CIDEr | **0.745** | 0.619 | **+12.6%** |
| REVERIE val unseen | CIDEr | **0.489** | 0.406 | **+8.3%** |
| UrbanWalk test | SPICE | **0.679** | 0.566 | +11.3% |
| UrbanWalk test | Rouge | **0.786** | 0.655 | +13.1% |

### 消融实验

R2R val unseen 上的组件消融：

| 配置 | SPICE | CIDEr | 说明 |
|------|-------|-------|------|
| 仅 Perspective | 0.154 | 0.209 | 基线 |
| 仅 BEV | 0.172 | 0.281 | BEV 单独已优于 Perspective |
| Perspective + BEV (concat) | 0.180 | 0.342 | 简单拼接两种特征 |
| + Fusion 模块 | 0.190 | 0.373 | Transformer 融合优于简单拼接 |
| + Iterative Refinement | 0.192 | 0.419 | 迭代优化带来 CIDEr +7.7% |
| **完整模型** | **0.208** | **0.449** | 所有模块互补叠加 |

融合方式对比 (R2R val unseen)：

| 融合方式 | SPICE | CIDEr | 说明 |
|---------|-------|-------|------|
| Addition | 0.185 | 0.366 | 简单相加 |
| Concat | 0.184 | 0.310 | 拼接 |
| **Ours (Transformer)** | **0.208** | **0.449** | Transformer 融合最优 |

### 关键发现

- BEV 特征单独已优于 Perspective 特征 (CIDEr 0.281 vs 0.209)，说明 3D 几何信息对指令生成至关重要
- 实例引导的迭代优化在所有设置下一致提升性能，一步优化效果最佳，进一步增加步数收益有限
- GPT-4V 零样本性能远低于微调方法 (SPICE 0.098 vs 0.208)，说明通用 MLLM 无法直接胜任导航指令生成

## 亮点与洞察

- 首次将 BEV 感知引入导航指令生成，实现 3D 空间理解与语言生成的有效桥接
- 参数高效设计 (仅 7.2% 参数可训练) 兼顾了性能和效率
- 在室内 (R2R, REVERIE) 和室外 (UrbanWalk) 场景均取得显著提升，验证了方法的通用性
- 生成的指令可实际用于指导导航 agent (HAMT/DUET)，验证了指令的实际可用性

## 局限与展望

- BEV 编码器需要 3D 检测任务的预训练数据，对新场景的适应性需要验证
- 当前仅在模拟器数据集上验证，真实物理环境的表现有待探索
- 可以探索将 BEV 感知与更强的 MLLM (如 LLaMA-2/3) 结合
- 迭代优化目前采用固定步数，可以考虑自适应停止策略

## 相关工作与启发

- 与 Lana (CVPR2023) 相比，BEVInstructor 引入 3D 几何先验，提升了场景理解的深度
- BEV 编码器的设计借鉴了自动驾驶领域的 BEVFormer，将其迁移到室内导航场景是新颖的应用
- 实例引导的迭代优化思想可以推广到其他需要分步推理的 vision-language 任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 BEV 感知与 MLLM 结合用于导航指令生成，设计合理
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个数据集 + 详细消融 + 下游 agent 评估，非常充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- **价值**: ⭐⭐⭐⭐ — 为具身智能的语言交互提供了新的技术路线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] H-V2X: A Large Scale Highway Dataset for BEV Perception](h-v2x_a_large_scale_highway_dataset_for_bev_perception.md)
- [\[CVPR 2026\] TrafficAlign: Aligning Large Language Models for Traffic Scenario Generation](../../CVPR2026/autonomous_driving/trafficalign_aligning_large_language_models_for_traffic_scenario_generation.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](../../CVPR2025/autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[ACL 2025\] Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)

</div>

<!-- RELATED:END -->
