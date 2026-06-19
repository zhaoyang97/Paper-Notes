---
title: >-
  [论文解读] ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration
description: >-
  [ECCV 2024][模型压缩][事件驱动推理] 提出基于行稀疏性探索的事件抑制方法ELSE，利用激活图中相邻行的空间相关性来减少非零激活（事件）数量，在目标检测和姿态估计任务上实现3.14~6.49倍的计算节省，且可与现有事件抑制方法互补。 领域现状：类脑计算架构（brain-inspired architecture）…
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "事件驱动推理"
  - "行稀疏性"
  - "激活图压缩"
  - "嵌入式AI"
  - "神经形态计算"
---

# ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration

**会议**: ECCV 2024  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 事件驱动推理, 行稀疏性, 激活图压缩, 嵌入式AI, 神经形态计算

## 一句话总结

提出基于行稀疏性探索的事件抑制方法ELSE，利用激活图中相邻行的空间相关性来减少非零激活（事件）数量，在目标检测和姿态估计任务上实现3.14~6.49倍的计算节省，且可与现有事件抑制方法互补。

## 研究背景与动机

**领域现状**：类脑计算架构（brain-inspired architecture）通过事件驱动（event-driven）的方式实现低功耗、低延迟的深度神经网络推理，非常适合嵌入式AI应用。在这类架构中，硬件性能直接取决于推理过程中非零激活（即事件）的数量——事件越少，计算越高效。

**现有痛点**：现有的事件抑制方法主要分为两类：空间抑制（spatial suppression）和时间抑制（temporal suppression）。空间抑制方法通过阈值化等方式直接减少单帧的事件数量，但忽略了激活图内部的空间冗余结构。时间抑制方法利用帧间差异只处理变化的像素，但需要存储完整的前一帧状态，导致状态存储开销（state memory footprint）很大，在资源受限的嵌入式平台上往往超出内存约束。

**核心矛盾**：激活图中存在大量的空间冗余——相邻行之间往往具有很高的相似性——但现有方法并未系统地利用这种行级别的空间相关性来进行事件压缩。

**本文目标** (1) 如何在不显著损失精度的前提下大幅减少事件驱动推理中的事件数量？(2) 如何与现有的空间/时间抑制方法形成互补？(3) 如何降低时间抑制方法的状态存储开销？

**切入角度**：作者观察到，卷积网络的激活图（feature map）中相邻行之间具有天然的空间相关性（spatial correlation），这是由卷积操作的局部感受野和图像的空间连续性决定的。如果相邻两行的激活值高度相似，则可以"跳过"冗余行的计算，用已有行的结果替代。

**核心 idea**：利用激活图中相邻行的空间相关性，通过行级别的稀疏化探索来抑制冗余事件，从而大幅减少事件驱动推理的计算量。

## 方法详解

### 整体框架

ELSE方法的核心思路是在事件驱动推理的pipeline中引入一个行级别的事件抑制模块。对于每一层网络的激活图，ELSE会逐行扫描并比较相邻行之间的差异。如果两行之间的差异低于某个阈值，则将其中一行标记为"可跳过"的冗余行，从而减少触发的事件计算。整个pipeline的输入是标准的卷积网络模型，输出是经过行稀疏化优化的事件流，可直接在事件驱动的硬件加速器上运行。

### 关键设计

1. **行间空间相关性分析（Line-based Spatial Correlation Analysis）**:

    - 功能：检测激活图中相邻行之间的冗余程度
    - 核心思路：对于激活图中的每一对相邻行，计算它们之间的差异度量（如逐元素差值的绝对值之和或均方差）。当差异低于预设阈值时，认为这两行具有高度冗余，后一行可以被前一行的值近似替代。这种检测是逐层进行的，因为不同层的激活特性不同（浅层空间相关性更强，深层更弱）。
    - 设计动机：卷积操作天然引入局部相关性，使得相邻行在大多数情况下非常相似。利用这种先验知识可以在几乎不损失信息的情况下大幅减少需要处理的事件数量。

2. **行级事件抑制机制（Line-level Event Suppression）**:

    - 功能：基于相关性分析结果，决定哪些行的事件可以被安全抑制
    - 核心思路：对于被标记为冗余的行，ELSE会将其所有非零激活都抑制为零（或复制参考行的值），从而将这些位置的事件从计算队列中移除。阈值的选择需要在压缩率和精度之间做平衡——阈值越大，越多行被抑制，但精度损失也可能越大。作者通过实验确定了不同网络和任务上的最优阈值范围。
    - 设计动机：行级别的操作粒度比逐像素操作更粗但比通道级操作更细，在压缩效率和精度保持之间取得了良好的平衡。

3. **与现有方法的互补组合策略（Complementary Combination Strategy）**:

    - 功能：将ELSE与现有的空间/时间抑制方法联合使用，实现更大的性能增益
    - 核心思路：ELSE可以作为一个"即插即用"的模块，与空间抑制方法串联使用时，先进行空间抑制再进行行稀疏化探索，两者协同进一步降低事件数量。与时间抑制方法配合时，由于ELSE减少了每帧的事件数量，时间抑制方法需要存储的状态信息（前一帧的激活值）也大幅减少，从而将状态存储开销降低超过2倍。
    - 设计动机：不同的事件抑制策略捕捉不同维度的冗余——空间抑制关注幅度冗余，时间抑制关注帧间冗余，而ELSE关注行间结构冗余。三者正交互补。

### 损失函数 / 训练策略

ELSE是一种推理时的事件抑制方法，不涉及额外的训练过程或损失函数修改。其核心参数是行间差异的阈值，可通过在验证集上搜索最优精度-效率权衡点来确定。对于不同的任务和网络架构，阈值需要分别调优。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | ELSE事件减少比 | 说明 |
|:---:|:---:|:---:|:---:|
| 目标检测（多种架构） | 事件触发计算减少 | 3.14~6.49× | 相比传统处理方式 |
| 姿态估计（多种架构） | 事件触发计算减少 | 2.43~5.75× | 相比传统处理方式 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|:---:|:---:|:---:|
| ELSE单独使用 | 3~6×事件减少 | 仅利用行间空间相关性 |
| ELSE + 空间抑制 | 显著增强计算节省 | 两种方法互补叠加 |
| ELSE + 时间抑制 | 状态内存减少>2× | 解决嵌入式平台内存约束问题 |

### 关键发现

- ELSE在目标检测任务上比姿态估计任务的压缩比更高（6.49× vs 5.75×），这与目标检测中大量背景区域的行间相关性更高有关
- ELSE与空间抑制方法组合可以达到"乘法级"的压缩效果
- 与时间抑制方法组合的最大贡献不在于多减少了多少事件，而在于大幅降低了状态存储需求，使得时间抑制方法可以部署在资源受限的嵌入式平台上
- 不同网络架构上ELSE的压缩效果有所不同，表明行间相关性与网络结构有关

## 亮点与洞察

- 行级别稀疏性是一个被忽视但效果显著的维度，与现有的像素级和通道级方法形成互补
- 方法的"即插即用"特性使其具有很强的实用价值——不需要重新训练模型
- 通过降低时间抑制方法的状态存储开销来解决嵌入式部署瓶颈，这一贡献在工程角度非常重要
- 从"脑启发计算"的角度，行级别的稀疏性探索也有一定的生物学合理性

## 局限与展望

- ECVA页面信息有限，具体的精度损失数据不完整——行抑制必然引入一定的精度降低，需要关注极端情况
- 阈值的自动选择策略（如基于目标精度约束的自适应阈值）值得探索
- 仅在目标检测和姿态估计两类任务上验证，其他视觉任务（如语义分割、图像分类）的效果未知
- 行级别的粒度是否是最优的有待进一步分析——也许可以探索块级别（block-based）或对角线级别的相关性
- 与结构化剪枝等模型压缩方法的联合使用值得研究

## 相关工作与启发

- 时间抑制方法（如delta network）利用帧间差异减少事件，本文与之互补
- 空间抑制方法通过阈值化或量化减少单帧事件数量
- 事件驱动硬件（如Loihi、SpiNNaker）的性能直接受益于更稀疏的事件流
- 启发：在其他计算范式中（如transformer的注意力计算），是否也存在类似的"行间冗余"可以利用？

## 评分

- 新颖性: ⭐⭐⭐⭐ 行级别稀疏性是一个简单但有效的新视角，虽然不是革命性创新
- 实验充分度: ⭐⭐⭐⭐ 覆盖了检测和姿态估计两类任务，但缺少更多任务和详细精度对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 对嵌入式AI部署有实际工程价值，且方法即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](../../ACL2025/model_compression/cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](../../ACL2025/model_compression/iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/model_compression/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[CVPR 2026\] Decompose, Mix, Adapt: A Unified Framework for Parameter-Efficient Neural Network Recombination and Compression](../../CVPR2026/model_compression/decompose_mix_adapt_a_unified_framework_for_parameter-efficient_neural_network_r.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](../../ICML2025/model_compression/efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)

</div>

<!-- RELATED:END -->
