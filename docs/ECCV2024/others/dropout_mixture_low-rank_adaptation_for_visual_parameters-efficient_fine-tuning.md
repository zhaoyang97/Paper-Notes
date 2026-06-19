---
title: >-
  [论文解读] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning
description: >-
  [ECCV 2024][参数高效微调] 本文提出 DMLoRA（Dropout-Mixture Low-Rank Adaptation），通过引入多分支上下投影结构并在训练过程中逐步dropout分支来平衡精度与正则化，配合两阶段学习缩放因子策略优化每层的缩放系数，在VTAB-1k和FGVC视觉微调基准上取得SOTA性能且推理无额外开销。
tags:
  - "ECCV 2024"
  - "参数高效微调"
  - "低秩适配"
  - "Dropout正则化"
  - "Transformer"
  - "VTAB-1k"
---

# Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 模型微调 / 参数高效微调  
**关键词**: 参数高效微调、低秩适配、Dropout正则化、视觉Transformer、VTAB-1k

## 一句话总结

本文提出 DMLoRA（Dropout-Mixture Low-Rank Adaptation），通过引入多分支上下投影结构并在训练过程中逐步dropout分支来平衡精度与正则化，配合两阶段学习缩放因子策略优化每层的缩放系数，在VTAB-1k和FGVC视觉微调基准上取得SOTA性能且推理无额外开销。

## 研究背景与动机

**领域现状**：参数高效微调（PEFT）已成为大模型适配下游任务的主流范式。LoRA（Low-Rank Adaptation）是其中最具代表性的方法之一，通过在原始权重旁添加低秩分解矩阵 $\Delta W = BA$（$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$）来实现参数高效更新。在NLP领域，LoRA已被广泛验证，近年来也开始被应用于视觉Transformer的微调。

**现有痛点**：将现有PEFT方法直接应用于不同的视觉任务时，性能波动显著。例如在VTAB-1k基准中，同一方法在自然图像、结构化任务和特殊图像三类任务上的表现差异很大。作者将这种性能不稳定性归因于现有PEFT方法的鲁棒性不足——单一的低秩适配路径可能在某些任务上找到好的梯度下降方向但在其他任务上陷入不好的局部最优。

**核心矛盾**：PEFT方法需要在模型容量和正则化之间取得平衡——增加适配参数可以提升模型容量但容易过拟合（尤其在小数据集上），减少参数虽然防止过拟合但可能欠拟合。另外，固定所有层使用相同的缩放因子忽略了不同层对下游任务的不同重要性。

**本文目标** (1) 如何为LoRA提供更鲁棒的梯度下降路径以提高在不同视觉任务上的稳定性？(2) 如何自适应地确定每层LoRA模块的最优缩放因子？

**切入角度**：作者从集成学习和Dropout正则化的角度出发。多个低秩分支可以看作多个弱学习器的集成，训练初期使用全部分支提供充足的模型容量，随着训练推进逐步dropout部分分支起到正则化效果。这种"先扩容后收缩"的动态训练策略可以在不同训练阶段自适应地平衡容量和正则化。

**核心 idea**：用多分支低秩适配+渐进式分支dropout实现动态的精度-正则化平衡，配合两阶段学习缩放因子最优化每层的适配强度。

## 方法详解

### 整体框架

DMLoRA的整体框架是一个针对预训练视觉Transformer（如ViT）的参数高效微调方法。在ViT的每个注意力层中，将标准LoRA的单分支结构替换为多分支结构——包含 $K$ 条并行的上投影和下投影路径。训练过程中，随着epoch增加，逐步以一定概率dropout部分分支，最终收敛到更少的分支。推理时将所有存活分支的权重合并到原始权重中，不引入额外计算开销。同时，采用两阶段学习缩放因子（2-Stage Learning Scalar）策略为每层确定最优的缩放系数。

### 关键设计

1. **多分支低秩适配（Multi-Branch Low-Rank Adaptation）**:

    - 功能：为模型提供多条梯度下降路径，增强训练的鲁棒性
    - 核心思路：将标准LoRA的单个上投影矩阵 $B$ 和下投影矩阵 $A$ 扩展为 $K$ 组并行分支 $\{(B_1, A_1), (B_2, A_2), ..., (B_K, A_K)\}$。每个分支的秩可以设计为 $r/K$ 以保持总参数量不变，也可以使用完整秩 $r$ 以增加模型容量。前向传播时，$\Delta W = \sum_{k=1}^{K} B_k A_k$。多分支结构使模型在参数空间中有更多的探索方向，不容易陷入单一的局部最优
    - 设计动机：类似于集成学习中多个弱分类器的组合优于单个强分类器，多个低秩分支的组合可以提供比单个分支更好的适配效果。同时多分支为后续的渐进式dropout提供了基础

2. **渐进式分支Dropout（Gradual Branch Dropout）**:

    - 功能：在训练过程中动态调节模型容量和正则化的平衡
    - 核心思路：训练初期保持所有 $K$ 个分支活跃，让模型充分学习下游任务的特征。随着训练推进，按照预定的调度函数（如线性或余弦调度）逐步增加分支被dropout的概率。被dropout的分支在该iteration的前向和反向传播中都不参与计算。训练后期，大部分分支被dropout，起到强正则化效果，防止过拟合。这个过程类似于从"宽搜索"过渡到"窄精调"
    - 设计动机：训练初期需要充足的容量来学习任务特征，此时正则化过强会导致欠拟合。训练后期模型已基本收敛，过多的参数会导致过拟合，此时需要增强正则化。渐进式dropout实现了这种"先松后紧"的自适应训练策略。推理时可以将所有分支合并为一个权重矩阵，因此不增加推理开销

3. **两阶段学习缩放因子（2-Stage Learning Scalar, 2S-LS）**:

    - 功能：自适应地为每层的DMLoRA模块确定最优的缩放因子 $\alpha$
    - 核心思路：LoRA中的缩放因子 $\alpha$ 控制着适配更新 $\Delta W$ 对原始权重的影响强度。传统方法对所有层使用相同的 $\alpha$，但不同层对不同任务的重要性不同。2S-LS策略分两阶段优化：第一阶段使用一个较大的统一缩放因子训练DMLoRA模块的权重参数；第二阶段冻结权重参数，将每层的缩放因子 $\alpha_l$ 设为可学习参数，通过较小的学习率进行优化。这样可以让模型自动发现哪些层需要更大的适配更新、哪些层保持原始权重更好
    - 设计动机：实验观察到不同视觉任务对ViT不同层的依赖程度不同——自然图像任务更依赖浅层（低级特征），结构化任务更依赖深层（高级语义）。固定缩放因子无法适应这种差异。两阶段方式避免了同时优化权重和缩放因子的优化困难

### 损失函数 / 训练策略

使用标准的交叉熵损失进行分类微调。训练策略分为两个阶段：阶段一使用全局统一的缩放因子训练DMLoRA的所有分支权重，同时执行渐进式分支dropout；阶段二冻结分支权重，仅优化每层的缩放因子 $\alpha_l$，学习率设为阶段一的十分之一。推理时，将各分支权重按缩放因子合并到原始预训练权重中：$W' = W + \frac{\alpha_l}{r} \sum_{k} B_k A_k$，不引入任何额外延迟。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 DMLoRA | 之前SOTA | 提升 |
|---|---|---|---|---|
| VTAB-1k (Natural, 7任务) | 平均准确率 | SOTA | 次优PEFT方法 | 提升 |
| VTAB-1k (Specialized, 4任务) | 平均准确率 | SOTA | 次优PEFT方法 | 提升 |
| VTAB-1k (Structured, 8任务) | 平均准确率 | SOTA | 次优PEFT方法 | 提升 |
| VTAB-1k (Overall, 19任务) | 平均准确率 | SOTA | 次优PEFT方法 | 一致领先 |
| FGVC (5个细粒度数据集) | 平均准确率 | SOTA | 次优PEFT方法 | 提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| 标准LoRA (单分支) | 基线性能 | 在部分任务上不稳定 |
| 多分支但不dropout | 相比单分支提升 | 多路径有帮助但可能过拟合 |
| 多分支+渐进dropout | 进一步提升 | dropout起到正则化作用 |
| 不使用2S-LS（统一缩放因子） | 低于完整模型 | 层间差异化缩放重要 |
| 完整DMLoRA (多分支+dropout+2S-LS) | 最佳 | 三个组件协同作用 |
| 分支数K的影响 | K=4或8较优 | 过少不够多样，过多增加训练开销 |

### 关键发现

- 多分支结构比单分支LoRA提供更鲁棒的梯度下降路径，减少了跨任务的性能波动
- 渐进式dropout是关键——始终保持全部分支或过早dropout都不如渐进式调度
- 不同层的最优缩放因子差异显著，验证了层间差异化适配的必要性
- DMLoRA在推理时可以完全合并到原始权重中，证明了"训练时结构化、推理时零开销"的可行性
- 在VTAB-1k的三类任务（Natural/Specialized/Structured）上都能稳定提升，验证了鲁棒性

## 亮点与洞察

- "训练时多分支、推理时合并"的范式优雅地解决了容量和效率的矛盾
- 渐进式dropout的思路很有创意——将Dropout正则化的应用粒度从神经元提升到结构分支
- 两阶段学习缩放因子的设计将层间差异化适配问题转化为简单的标量优化
- 整个方法推理零开销，对实际部署非常友好
- 实验设计全面：VTAB-1k覆盖19个不同类型的视觉任务

## 局限与展望

- 训练时多分支结构增加了训练显存和计算量，尽管推理无额外开销
- 渐进式dropout的调度函数（线性/余弦）是预设的，可以考虑基于验证性能的自适应调度
- 仅在ViT上验证，可以扩展到CNN、Swin Transformer等其他架构
- 分支的初始化策略可能影响最终效果，未深入分析
- 可以考虑将渐进dropout与结构化剪枝结合，在训练结束后永久移除不重要的分支
- 在更大规模的模型（如ViT-H、ViT-G）上的效果有待验证
- 未与其他PEFT方法（如Prefix Tuning、Adapter）的组合进行实验

## 相关工作与启发

- **LoRA**：低秩适配的开创性工作，本文在此基础上引入多分支和渐进dropout
- **AdaLoRA**：自适应调整LoRA的秩分配，与本文的层间缩放因子优化思路有相似之处
- **DoRA**：将LoRA分解为方向和大小两个分量分别适配
- **VPT (Visual Prompt Tuning)**：通过添加可学习prompt实现视觉微调
- **SSF (Scale & Shift)**：通过缩放和偏移实现参数高效微调
- **Dropout**：经典的正则化技术，本文将其从元素级推广到分支级
- 启发：渐进式结构dropout的思路可以推广到其他多分支架构中，如Mixture-of-Experts（在训练过程中渐进减少专家数量）

## 评分

- 新颖性: ⭐⭐⭐⭐ 多分支LoRA+渐进dropout的组合有一定新意，但各组件并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ VTAB-1k 19个任务和FGVC 5个数据集的全面评估很充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机论证合理
- 价值: ⭐⭐⭐⭐ 为LoRA提供了一种鲁棒性更强的变体，对视觉PEFT社区有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Functional Transform-Based Low-Rank Tensor Factorization for Multi-Dimensional Data Recovery](functional_transform-based_low-rank_tensor_factorization_for_multi-dimensional_d.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[CVPR 2026\] Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation](../../CVPR2026/others/basis-oriented_low-rank_transfer_for_few-shot_and_test-time_adaptation.md)
- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](attnzero_efficient_attention_discovery_for_vision_transformers.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)

</div>

<!-- RELATED:END -->
