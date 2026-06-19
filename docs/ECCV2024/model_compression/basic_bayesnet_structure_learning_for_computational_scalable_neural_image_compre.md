---
title: >-
  [论文解读] BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression
description: >-
  [ECCV 2024][模型压缩][神经图像压缩] 本文提出 BaSIC 框架，通过学习神经图像压缩（NIC）系统的贝叶斯网络结构，同时控制骨干网络复杂度和自回归单元的并行计算能力，首次实现了对 NIC 全流程的计算可扩展性控制。 领域现状：神经图像压缩（Neural Image Compression…
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "神经图像压缩"
  - "贝叶斯网络"
  - "计算可扩展性"
  - "自回归模型"
  - "结构学习"
---

# BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression

**会议**: ECCV 2024  
**代码**: [https://github.com/worldlife123/cbench_BaSIC](https://github.com/worldlife123/cbench_BaSIC)  
**领域**: 模型压缩 / 图像压缩  
**关键词**: 神经图像压缩, 贝叶斯网络, 计算可扩展性, 自回归模型, 结构学习

## 一句话总结

本文提出 BaSIC 框架，通过学习神经图像压缩（NIC）系统的贝叶斯网络结构，同时控制骨干网络复杂度和自回归单元的并行计算能力，首次实现了对 NIC 全流程的计算可扩展性控制。

## 研究背景与动机

**领域现状**：神经图像压缩（Neural Image Compression, NIC）在率失真（Rate-Distortion）性能上已超越传统编解码器（如 JPEG、BPG、VVC intra），但其巨大的计算开销严重阻碍了实际部署。一个典型的 NIC 系统包含编码器骨干（encoder backbone）、解码器骨干（decoder backbone）和自回归熵编码模块（autoregressive entropy model），三者共同决定了整体的计算复杂度。

**现有痛点**：现有的 NIC 加速工作大多聚焦于某个单一模块的加速（如只加速自回归模块或只压缩骨干网络），无法对整体计算复杂度进行精确控制。例如，某些方法只能将自回归步骤从 serial 变为 partial parallel，但无法控制骨干网络的计算量；另一些方法通过剪枝减小骨干，但忽略了自回归模块的加速。缺乏一个统一框架来同时控制所有模块的计算复杂度。

**核心矛盾**：NIC 系统的计算复杂度由多个相互关联的组件共同决定——骨干网络的层数/通道数决定了特征提取能力，自回归模块的依赖结构决定了并行度和压缩率。这两者之间存在复杂的耦合关系：简化骨干可能需要更强的自回归模型来补偿，反之亦然。目前缺乏一个理论框架来统一建模和优化这种耦合。

**本文目标** (1) 如何在一个统一框架下同时控制 NIC 的骨干网络复杂度和自回归并行度？(2) 如何在给定计算预算下，自动找到最优的复杂度分配方案？(3) 如何保证在计算约束下仍维持竞争性的压缩性能？

**切入角度**：作者将 NIC 系统建模为一个贝叶斯网络（BayesNet），其中节点代表潜在变量（latent variables），边代表依赖关系。控制 NIC 的计算复杂度等价于学习这个贝叶斯网络的结构——通过调整边的连接方式来同时控制骨干复杂度和自回归并行度。

**核心 idea**：将 NIC 系统建模为贝叶斯网络，通过结构学习将骨干网络复杂度控制和自回归并行度优化统一为贝叶斯网络的两个子问题求解。

## 方法详解

### 整体框架

BaSIC 将 NIC 的计算可扩展性问题分解为两个贝叶斯网络结构学习子问题：(1) **节点间结构学习**（Inter-Node Structure Learning）——通过异质二部贝叶斯网络控制骨干网络复杂度；(2) **节点内结构学习**（Intra-Node Structure Learning）——通过多部贝叶斯网络优化自回归单元的并行计算。输入为图像数据和计算预算约束，输出为满足约束的最优 NIC 模型配置。

### 关键设计

1. **异质二部贝叶斯网络（Heterogeneous Bipartite BayesNet）用于骨干复杂度控制**:

    - 功能：调控编码器-解码器骨干网络的计算复杂度
    - 核心思路：将 NIC 骨干网络的层间连接建模为一个二部图（bipartite graph），其中一侧节点代表编码器层的输出特征，另一侧节点代表解码器层的输入特征。图中的边代表层间数据依赖。通过学习这个二部图的结构（即决定哪些边保留、哪些边删除），可以等效地实现对骨干网络的通道剪枝和层跳跃。异质性体现在不同层的节点可以有不同的维度（通道数），允许非均匀的剪枝策略。结构学习的优化目标为在满足计算预算约束的条件下最大化率失真性能。
    - 设计动机：传统剪枝方法通常对所有层施加统一的剪枝比例，忽略了不同层对性能的不同贡献。贝叶斯网络结构学习框架天然支持非均匀、自适应的复杂度分配，且提供了概率论视角下的理论保证。

2. **多部贝叶斯网络（Multipartite BayesNet）用于自回归并行化**:

    - 功能：优化自回归熵编码模块的并行计算结构
    - 核心思路：标准自回归模型中，每个潜在变量依赖于其所有前驱变量，导致必须严格串行解码，计算速度极慢。本文将潜在变量组织为多部图结构：将变量分为多个组（groups），同一组内的变量互相独立（可并行），组间保持必要的依赖关系。这样即实现了分组并行解码。多部贝叶斯网络的结构学习目标是找到最优的分组方案，使得在给定并行度（即组数）下，组间依赖能捕获尽可能多的统计相关性，从而最小化压缩率的损失。算法通过贪心搜索的方式确定分组和组间依赖的拓扑排序。
    - 设计动机：现有的自回归并行化方法（如 Checkerboard、Channel-wise 分组）使用固定的人工设计分组方案，无法根据数据特性自适应调整。贝叶斯网络结构学习提供了数据驱动的最优分组方案。

3. **联合优化与计算预算控制**:

    - 功能：在给定总计算预算下，自动分配骨干和自回归的复杂度
    - 核心思路：定义全局计算预算 $C_{total} = C_{backbone} + C_{AR}$，其中骨干复杂度由二部 BayesNet 结构决定，自回归复杂度由多部 BayesNet 的组数决定。通过在不同的 $(C_{backbone}, C_{AR})$ 分配下分别求解两个子问题，然后选择率失真性能最优的分配方案。实际中采用多级（multi-level）Slimmable Network 技术，单次训练即可得到多个计算点的模型。
    - 设计动机：解耦两个子问题的求解大大降低了联合优化的复杂度，同时 Slimmable Network 的使用避免了为每个计算点单独训练模型的巨大开销。

### 损失函数 / 训练策略

训练损失为标准的率失真损失 $L = R + \lambda D$，其中 $R$ 为比特率，$D$ 为失真（MSE 或 MS-SSIM），$\lambda$ 控制率失真权衡。训练时采用 Slimmable Network 策略，即在每个 mini-batch 中随机采样不同的计算配置进行训练，使得单个模型支持多种计算级别。贝叶斯网络结构学习使用贪心搜索算法，在训练完成后执行，不增加训练阶段的计算开销。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (BaSIC) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| Kodak | BD-Rate (相同计算量) | 最优 | Slimmable NIC | 更准确的复杂度控制 + 更好的 RD 性能 |
| ImageNet 子集 | PSNR @ 同等 MACs | 竞争性 | 固定架构 NIC | 实现全计算可扩展性 |
| Kodak | 复杂度控制精度 | 高精度 | Slimmable baseline | 控制误差更小 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅骨干控制（无自回归优化） | BD-Rate 中等 | 缺少自回归并行化导致解码慢 |
| 仅自回归控制（固定骨干） | BD-Rate 中等 | 无法减少骨干计算量 |
| 均匀通道剪枝 | 较差 | 忽略层间差异导致性能损失大 |
| BaSIC 完整方案 | 最优 | 两个子问题联合优化效果最好 |
| 不同自回归分组方案 | RD 性能对比 | 学习到的分组优于手工 Checkerboard 分组 |

### 关键发现

- 通过贝叶斯网络结构学习发现的复杂度分配方案，比均匀剪枝在相同计算预算下提升了显著的 PSNR
- 自回归模块的贪心分组搜索发现的分组方案优于所有手工设计的分组策略
- BaSIC 能在非常宽的计算范围内（从极低到完整）提供连续的 RD 性能曲线
- 骨干和自回归的最优计算分配比例随码率变化：低码率下自回归更重要，高码率下骨干更重要

## 亮点与洞察

- **统一视角**：首次将 NIC 的骨干复杂度和自回归并行度控制统一在贝叶斯网络结构学习框架下，理论优雅
- **全计算可扩展性**：不同于只能加速某个模块的方法，BaSIC 实现了对 NIC 全流程的精确计算控制
- **数据驱动的结构发现**：自回归分组不再依赖人工设计，而是通过结构学习自动发现最优方案
- **实用的多级部署**：Slimmable Network + 结构学习的组合，使得单次训练即可支持多种部署场景

## 局限与展望

- 贝叶斯网络结构学习目前采用贪心搜索，可能陷入局部最优
- 两个子问题的解耦求解虽然降低了复杂度，但可能丢失骨干-自回归之间的耦合信息
- 当前主要在 Joint Autoregressive 和 Hyperprior 两种 NIC 框架上验证，对更新的 NIC 架构（如 Transformer-based）的适用性未验证
- Slimmable Network 训练中不同计算配置间可能存在性能干扰，极端配置下性能可能下降
- 缺少与硬件实际延迟的对比，MACs 不完全等同于真实推理速度

## 相关工作与启发

- **Slimmable Networks**（Yu & Huang）: 可切换宽度的网络训练策略，BaSIC 在此基础上加入结构学习
- **FSAR**（Finite State Autoregressive Entropy Coding）: 自回归加速方法，BaSIC 的代码基于此开发
- **Checkerboard Context Model**: 固定棋盘分组的自回归并行化，BaSIC 证明了学习分组更优
- **NIC 可扩展性综述**: 该方向的工作大多关注单模块加速，BaSIC 首次提出全局可扩展性框架

## 评分

- 新颖性: ⭐⭐⭐⭐ 贝叶斯网络视角建模 NIC 可扩展性问题非常新颖
- 实验充分度: ⭐⭐⭐ 消融实验较完整，但主实验的数据集和对比方法可以更丰富
- 写作质量: ⭐⭐⭐⭐ 数学建模清晰，两个子问题的分解逻辑通顺
- 价值: ⭐⭐⭐⭐ 全计算可扩展 NIC 对实际部署有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model](bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model.md)
- [\[ACL 2025\] Basic Reading Distillation](../../ACL2025/model_compression/basic_reading_distillation.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[ICML 2025\] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams](../../ICML2025/model_compression/lego_sketch_a_scalable_memory-augmented_neural_network_for_sketching_data_stream.md)

</div>

<!-- RELATED:END -->
