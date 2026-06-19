---
title: >-
  [论文解读] Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing
description: >-
  [CVPR 2025][图学习][视频场景图] 提出 VISA 框架，从视觉（记忆引导序列建模 MGSM 降低特征方差）和语义（迭代关系生成器 IRG 引入层次上下文减少对偏置先验的依赖）双重角度对视频场景图生成进行去偏置，在 Action Genome 等数据集上大幅提升尾部类别性能。 领域现状：视频场景图生成（VidSG…
tags:
  - "CVPR 2025"
  - "图学习"
  - "视频场景图"
  - "去偏置"
  - "记忆引导"
  - "迭代关系生成"
  - "长尾分布"
---

# Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing

**会议**: CVPR 2025  
**arXiv**: [2503.00548](https://arxiv.org/abs/2503.00548)  
**代码**: 无  
**领域**: 图学习 / 场景图生成  
**关键词**: 视频场景图, 去偏置, 记忆引导, 迭代关系生成, 长尾分布

## 一句话总结

提出 VISA 框架，从视觉（记忆引导序列建模 MGSM 降低特征方差）和语义（迭代关系生成器 IRG 引入层次上下文减少对偏置先验的依赖）双重角度对视频场景图生成进行去偏置，在 Action Genome 等数据集上大幅提升尾部类别性能。

## 研究背景与动机

**领域现状**：视频场景图生成（VidSGG）旨在将视频内容结构化为 <主体-谓词-客体> 三元组，是视频理解的核心任务。现有方法（如 STTran、TEMPURA、FloCoDe）在常见关系上表现尚可，但在长尾分布下严重偏向高频类别。

**现有痛点**：偏置来源有两个被忽视的维度：（1）视觉偏置——视频中物体会因遮挡、模糊、尺度变化等导致特征方差大，Transformer 生成的视觉表示不够稳定，模型倾向于匹配高频实体；（2）语义偏置——仅靠视觉特征预测谓词时上下文不足，模型退化为依赖训练集中的频率先验。

**核心矛盾**：视觉不稳定性 + 语义上下文不足 → 模型无法区分细粒度关系（如"持有"vs"触摸"），只能退回到高频谓词的安全预测。

**本文目标** 从视觉和语义两个层面同时进行去偏置，让模型在尾部类别上也能做出准确预测。

**切入角度**：视觉方面用指数移动平均记忆平滑特征方差（理论证明方差从 $\Sigma$ 降至 $\frac{\lambda\Sigma}{2}$），语义方面用迭代生成器不断补充上下文信息，增大预测后验与偏置先验之间的 KL 散度。

**核心 idea**：记忆平滑视觉特征 + 迭代补充语义上下文 = 视觉语义双重去偏置。

## 方法详解

### 整体框架

输入视频帧经过物体检测器（Faster R-CNN + ResNet-101）提取物体区域特征，然后送入 VISA 框架的两个核心模块：MGSM 稳定视觉特征，IRG 迭代生成关系谓词。最终输出每帧的场景图三元组集合，包含注意力关系、空间关系和接触关系三类谓词。

### 关键设计

1. **记忆引导序列建模（MGSM）**:

    - 功能：稳定视频中物体的视觉特征表示，降低因遮挡/模糊等导致的特征方差
    - 核心思路：为每个物体维护一个指数移动平均记忆 $M_i^{t+1} = (1-\lambda)M_i^t + \lambda v_i^t$，理论证明记忆的方差为 $\text{Var}[M_i^t] = \frac{\lambda\Sigma}{2-\lambda} \approx \frac{\lambda\Sigma}{2}$（远小于原始方差 $\Sigma$）。然后通过自适应权重 $W_i^t = \sigma(\text{MLP}(v_i^t))$ 融合当前帧和前一帧特征，最后用双注意力机制以记忆为 Key、当前特征为 Value 进行增强
    - 设计动机：传统 Transformer 只关注帧间自注意力，忽略了物体特征的时序平滑性。EMA 记忆以几乎零额外成本提供了稳定的特征锚点，$\lambda$ 控制平滑强度（SGCLS 用 0.04，SGDET 用 0.06）

2. **迭代关系生成器（IRG）**:

    - 功能：通过迭代补充语义上下文，减少模型对偏置先验的依赖
    - 核心思路：基于信息论推导——额外上下文 $S$ 可降低条件熵 $H(r_{ij}|v_i,v_j,S) \leq H(r_{ij}|v_i,v_j)$，等价于增大预测后验与偏置先验的 KL 散度。第一次迭代用基础特征（视觉+空间+GloVe语义嵌入）预测初步场景图，后续迭代通过层次语义提取器（HSE）将已预测的三元组嵌入作为额外上下文反馈给关系生成器，逐步精化预测
    - 设计动机：单次预测时上下文不足，模型被迫依赖先验。迭代生成让模型利用已有的场景图预测作为额外线索（如"A 已被预测为在走路"可以帮助判断"A 是否在看某物"）

3. **层次语义提取器（HSE）**:

    - 功能：从已预测的三元组中提取多尺度语义信息用于下一次迭代
    - 核心思路：将复合特征分解为细粒度的主体/客体表示，用步长为 2 的卷积下采样后拼接，捕获多层次上下文
    - 设计动机：简单拼接无法有效融合视觉和语义信息，层次结构在消融中贡献 1.4-2.4% 的 mR@50 提升

### 损失函数 / 训练策略

总损失 $L_{\text{total}} = L_p + L_e + L_{\text{contra}}$，其中 $L_p$ 和 $L_e$ 分别是谓词和实体的交叉熵损失，$L_{\text{contra}}$ 是对比损失（沿用 TEMPURA）。优化器 AdamW，学习率 1e-5，单张 RTX 4090 训练 15 个 epoch。

## 实验关键数据

### 主实验

在 Action Genome 数据集上的 mR@K 对比（With Constraint）：

| 任务 | 指标 | TEMPURA | FloCoDe | **VISA** | 提升 |
|------|------|---------|---------|---------|------|
| PREDCLS | mR@10 | 42.9 | 44.8 | **46.9** | +2.1 |
| SGCLS | mR@10 | 34.0 | 37.4 | **40.8** | +3.4 |
| SGDET | mR@10 | 22.6 | 24.2 | **27.3** | +3.1 |

Semi Constraint（更接近实际应用）提升更大：

| 任务 | 指标 | TEMPURA | **VISA** | 提升 |
|------|------|---------|---------|------|
| PREDCLS | mR@20 | 44.5 | **56.3** | +11.8 |
| SGCLS | mR@20 | 39.5 | **52.6** | +13.1 |
| SGDET | mR@20 | 21.8 | **31.7** | +9.9 |

在 PVSG 和 4DPVSG 数据集上也有 7-8% 的提升。

### 消融实验

| 配置 | SGCLS mR@10 (Semi) | SGDET mR@10 (No Constr.) | 说明 |
|------|-------------------|-------------------------|------|
| Full VISA | 47.8 | 30.7 | 完整模型 |
| w/o MGSM | 45.6 | 27.9 | 视觉去偏置贡献 2-3% |
| w/o IRG | 34.0 | - | 语义去偏置贡献 13.8% |
| w/o HSE | -1.4~-2.4 mR@50 | - | 层次结构有效 |

### 关键发现
- **语义去偏置 (IRG) 贡献远大于视觉去偏置 (MGSM)**：IRG 移除后掉 13+ 个点 vs MGSM 移除后掉 2-3 个点，说明语义上下文不足是主要偏置来源
- **迭代次数 N 的收益快速饱和**：N=1→N=4 只提升约 0.8%，且 N≥2 训练时间翻倍，实用中 N=1 即可
- **尾部类别提升最显著**：SGDET No Constraint 下尾部类别提升 11.0%，验证了去偏置的有效性
- **$\lambda$ 对不同任务有不同最优值**：SGCLS（有GT框）用较小的 0.04，SGDET（从头检测）用较大的 0.06

## 亮点与洞察

- **理论与实践结合**：用高斯噪声模型推导 EMA 的方差缩减比，用信息论推导迭代生成器的去偏置原理，理论指导设计而非事后解释
- **双重去偏置的正交性**：视觉和语义去偏置解决的是不同层面的问题——前者改善特征质量，后者改善推理过程——两者可以独立贡献且叠加有效
- **EMA 记忆的轻量性**：只需要一个移动平均缓冲区和一层注意力，几乎不增加计算成本就能显著降低视觉特征方差

## 局限与展望

- **物体检测器是瓶颈**：小物体（如杯子）检测失败会跳过相关三元组，限制了整体性能
- **迭代自纠正能力有限**：N 增大后收益快速饱和，自生成的语义上下文无法突破模型本身的能力上限
- **数据集标注噪声**：Action Genome 存在错误标注和歧义标注（如"看着杯子"vs"靠近杯子"），影响评估公正性
- **仅在单数据集上验证主实验**：虽然补充了 PVSG/4DPVSG，但 Action Genome 是唯一的全面评估基准

## 相关工作与启发

- **vs TEMPURA**: TEMPURA 用对比学习辅助去偏置但不处理视觉噪声。VISA 在其基础上增加了 MGSM 视觉稳定和 IRG 语义迭代，Semi Constraint 下提升 11-13 个点
- **vs FloCoDe**: FloCoDe 关注流信息辅助谓词判断，但仍受视觉不稳定性影响。VISA 的 EMA 记忆从根本上改善了特征稳定性
- **vs 图像场景图方法**: 视频 SGG 的额外挑战在于时序不稳定性，MGSM 是专门针对这一问题的设计

## 评分
- 新颖性: ⭐⭐⭐⭐ 视觉语义双重去偏置的框架新颖，理论推导扎实
- 实验充分度: ⭐⭐⭐⭐ 三种约束设置、多数据集、详细消融，但主数据集单一
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但符号较多且部分公式可进一步简化
- 价值: ⭐⭐⭐⭐ 为 VidSGG 提供了新的去偏置思路，尾部类别提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Universal Scene Graph Generation](universal_scene_graph_generation.md)
- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](../../NeurIPS2025/graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[CVPR 2026\] Mixture-of-Experts based Feature Decoupling for Open Vocabulary Scene Graph Generation](../../CVPR2026/graph_learning/mixture-of-experts_based_feature_decoupling_for_open_vocabulary_scene_graph_gene.md)
- [\[CVPR 2026\] Robo-SGG: Exploiting Layout-Oriented Normalization and Restitution Can Improve Robust Scene Graph Generation](../../CVPR2026/graph_learning/robo-sgg_exploiting_layout-oriented_normalization_and_restitution_can_improve_ro.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](../../NeurIPS2025/graph_learning/duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)

</div>

<!-- RELATED:END -->
