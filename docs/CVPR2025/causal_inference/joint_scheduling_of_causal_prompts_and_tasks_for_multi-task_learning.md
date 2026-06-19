---
title: >-
  [论文解读] Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning
description: >-
  [CVPR 2025][因果推理][因果提示] 提出 JSCPT（Joint Scheduling of Causal Prompts and Tasks）框架，首先设计多任务视觉语言提示（MTVLP）并通过因果干预消除提示中的虚假相关特征，然后通过自适应任务调度器根据训练过程中任务关系的动态变化调整学习顺序和权重，在多个多任务视觉识别基准上取得显著提升。
tags:
  - "CVPR 2025"
  - "因果推理"
  - "因果提示"
  - "任务调度"
  - "多任务学习"
  - "VLM提示学习"
  - "虚假相关消除"
---

# Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning

**会议**: CVPR 2025  
**代码**: 未公开  
**领域**: 因果推理 / 多任务学习 / 视觉语言模型  
**关键词**: 因果提示, 任务调度, 多任务学习, VLM提示学习, 虚假相关消除

## 一句话总结
提出 JSCPT（Joint Scheduling of Causal Prompts and Tasks）框架，首先设计多任务视觉语言提示（MTVLP）并通过因果干预消除提示中的虚假相关特征，然后通过自适应任务调度器根据训练过程中任务关系的动态变化调整学习顺序和权重，在多个多任务视觉识别基准上取得显著提升。

## 研究背景与动机

**领域现状**：提示学习（Prompt Learning）是高效适配预训练视觉语言模型（VLM，如 CLIP）到下游任务的主流范式。通过学习少量提示向量即可适配新任务，无需微调整个模型。多任务提示学习进一步扩展到同时处理多个任务，用共享提示桥接多个下游视觉识别任务。

**现有痛点**：现有多任务提示学习方法存在两个被忽视的关键问题：(1) **虚假相关**——训练数据中存在与任务无关但统计上关联的特征（如"水"与"船"共现），提示可能学到这些虚假关联而非因果特征，导致在分布外数据上泛化性差；(2) **动态任务关系**——多个任务之间的梯度关系在训练过程中不断变化（如前期任务 A 和 B 互相促进，后期可能互相干扰），固定的联合训练策略（等权重、固定顺序）无法适应这种动态性。

**核心矛盾**：提示需要同时服务多个任务，但不同任务可能引入不同的虚假相关（偏置方向不同），且任务间的协同/冲突关系随训练阶段变化。这使得简单的共享提示在多任务场景下效果受限。

**本文目标**：(1) 在提示学习中引入因果推理，从因果而非相关的角度学习提示特征；(2) 建模任务间的动态关系，自适应调整多任务学习的调度策略。

**切入角度**：建立因果图（Causal Graph）描述提示特征与任务标签之间的因果和虚假路径，然后通过 do-calculus 进行因果干预，阻断虚假路径。

**核心 idea**：用因果干预净化提示特征（去虚假），用动态任务调度优化学习过程（去冲突），两者联合优化形成统一框架。

## 方法详解

### 整体框架
JSCPT 由三个核心组件组成：(1) **多任务视觉语言提示（MTVLP）**：为 CLIP 的文本和视觉编码器设计可学习的共享提示；(2) **因果提示学习模块**：基于因果图和反事实推理消除提示中的虚假相关特征；(3) **自适应任务调度器**：监测训练过程中各任务的梯度关系，动态调整任务权重和学习顺序。

### 关键设计

1. **多任务视觉语言提示（MTVLP）**：

    - 功能：为多个下游任务学习共享的提示向量
    - 核心思路：在 CLIP 的文本编码器和视觉编码器的输入层分别添加可学习的提示 token $\mathbf{P}_t \in \mathbb{R}^{N \times D}$ 和 $\mathbf{P}_v \in \mathbb{R}^{M \times D}$。所有任务共享同一组提示，通过任务特定的轻量适配层（线性投影）使每个任务获得针对性的提示表示 $\mathbf{P}_t^{(k)} = W_k \mathbf{P}_t$
    - 设计动机：共享提示减少参数量并鼓励跨任务知识迁移，任务特定投影保持各任务的区分性

2. **因果提示学习（Causal Prompt Learning）**：

    - 功能：消除提示特征中的虚假相关信号
    - 核心思路：构建因果图 $\mathcal{G}$，将提示特征 $Z$ 分解为因果特征 $Z_c$（与标签 $Y$ 有因果关系）和虚假特征 $Z_s$（与 $Y$ 仅统计相关）。通过因果干预 $P(Y | do(Z_c))$ 阻断混杂因子 $C$ 对预测的影响。利用特征解耦网络将提示特征分解为两部分，对因果特征部分施加后门调整公式 $P(Y | do(Z_c)) = \sum_c P(Y | Z_c, c) P(c)$，将混杂因子 $c$ 的影响积分掉。实践中通过对训练集的混杂因子分布进行采样近似求和
    - 设计动机：传统提示学习用标准交叉熵训练，会不加区分地利用所有相关特征（包括虚假的）。因果干预保证提示只编码与任务因果相关的特征

3. **自适应任务调度器（Adaptive Task Scheduler）**：

    - 功能：根据训练过程中任务间的动态关系调整学习策略
    - 核心思路：在每个训练 epoch，计算各任务对共享提示参数的梯度 $\nabla_{\mathbf{P}} \mathcal{L}_k$，通过梯度余弦相似度 $\cos(\nabla_k, \nabla_j)$ 衡量任务间的协同/冲突程度。当两个任务梯度方向相近时（协同），增大两者权重鼓励联合学习；当梯度方向相反时（冲突），降低冲突任务的权重或暂时搁置。调度策略还考虑各任务的当前损失值，优先关注学得较差的任务
    - 设计动机：多任务学习中"负迁移"（Negative Transfer）是核心挑战。自适应调度持续监控任务关系，实现"合则聚、斗则分"的灵活策略

### 损失函数 / 训练策略
总损失为因果交叉熵损失和任务调度权重的加权和：$\mathcal{L} = \sum_{k=1}^{K} w_k^{(t)} \cdot \mathcal{L}_{causal}^{(k)}$，其中 $w_k^{(t)}$ 是第 $t$ epoch 任务 $k$ 的动态权重。训练过程为交替优化：先固定调度权重更新提示参数，再固定提示更新调度器。

## 实验关键数据

### 主实验（多任务视觉识别）

| 方法 | Office-Home (Avg Acc) | DomainNet (Avg Acc) | VTAB (Avg Acc) | 平均提升 |
|------|----------------------|--------------------|----|----------|
| CoOp (单任务提示) | 72.8 | 54.3 | 68.1 | baseline |
| MaPLe (多任务提示) | 75.1 | 56.8 | 70.5 | +2.4 |
| TaskPrompter | 76.3 | 58.2 | 71.8 | +3.7 |
| **JSCPT (Ours)** | **79.6** | **61.7** | **74.2** | **+6.8** |

### 消融实验

| 配置 | Office-Home | DomainNet | 说明 |
|------|-------------|-----------|------|
| Full JSCPT | 79.6 | 61.7 | 完整模型 |
| w/o 因果干预 | 76.8 | 58.9 | 退化为标准提示 + 调度 |
| w/o 任务调度 | 77.4 | 59.5 | 因果提示 + 等权重训练 |
| w/o MTVLP（独立提示） | 75.2 | 56.4 | 不共享提示 |
| 固定任务权重 | 77.1 | 59.1 | 不动态调整权重 |

### 关键发现
- 因果干预贡献约 +2.8% 绝对精度提升（Office-Home），证实消除虚假相关的重要性
- 动态任务调度贡献约 +2.2%，在任务冲突明显的 DomainNet 上提升更大
- 两者联合效果优于各自单独使用之和（+6.8 > 2.8+2.2），存在正向协同
- 因果提示在分布外测试集上的优势更明显（+3.5% vs 分布内 +2.1%），符合因果推理提升泛化性的预期
- 共享提示（MTVLP）比独立提示高 4.4%，证实跨任务知识迁移的价值

## 亮点与洞察
- **因果推理 × 提示学习的首次结合**：在 VLM 提示学习中引入 do-calculus 是一个有理论深度的创新，为提示学习的鲁棒性研究开辟了新方向
- **"去虚假 + 去冲突"的双管齐下策略**：提示层面去虚假相关，任务层面去负迁移，两者相辅相成，覆盖了多任务提示学习的两大核心瓶颈
- **梯度信号作为任务关系的实时探针**：用梯度余弦相似度衡量任务协同/冲突是轻量且有效的方式

## 局限与展望
- 因果图中因果变量与虚假变量的分解依赖于先验假设，不同任务可能需要不同的因果图结构
- 任务数量较多时，梯度相似度矩阵的计算和调度策略的搜索空间快速增长
- 目前只在视觉分类任务上验证，对生成任务（如图像描述、VQA）的效果未知
- 因果干预中对混杂因子分布的采样近似可能引入估计偏差

## 相关工作与启发
- **vs CoOp/CoCoOp（提示学习）**: CoOp 学习单任务提示，CoCoOp 添加条件生成提示。但都未考虑虚假相关，JSCPT 从因果角度根本性解决这一问题
- **vs PCGrad/CAGrad（多任务梯度方法）**: PCGrad 通过梯度投影消除冲突，CAGrad 寻找平均梯度方向。JSCPT 的任务调度更灵活，不仅调整梯度方向还调整任务权重
- **vs CausalVLM（因果VLM）**: CausalVLM 在预训练阶段引入因果目标，本文在提示微调阶段引入，更轻量且易部署

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 因果推理与提示学习、动态任务调度的三方融合非常新颖
- 实验充分度: ⭐⭐⭐⭐ 多基准验证，消融充分，有分布外泛化分析
- 写作质量: ⭐⭐⭐⭐ 因果图的构建和理论推导清晰
- 价值: ⭐⭐⭐⭐ 为多任务 VLM 微调提供了理论驱动的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](../../ICML2026/causal_inference/tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICCV 2025\] Social Debiasing for Fair Multi-modal LLMs](../../ICCV2025/causal_inference/social_debiasing_for_fair_multi-modal_llms.md)
- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](../../ICLR2026/causal_inference/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [\[ICML 2025\] MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion](../../ICML2025/causal_inference/mpf_aligning_and_debiasing_language_models_post_deployment_via_multi_perspective.md)

</div>

<!-- RELATED:END -->
