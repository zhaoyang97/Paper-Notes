---
title: >-
  [论文解读] MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition
description: >-
  [ICML2025][视频理解][Mamba] 提出 MoMa 框架，通过序列调制操作 (SeqMod) 将 Mamba 的线性复杂度 SSM 以 scale-bias 方式注入冻结的 CLIP Transformer，实现高效全时空动态建模，在多个视频识别基准上以更少计算量达到 SOTA 水平。 - 核心问题：如何高效地将…
tags:
  - "ICML2025"
  - "视频理解"
  - "Mamba"
  - "PEFT"
  - "图像基础模型适配"
  - "状态空间模型"
  - "时空建模"
---

# MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition

**会议**: ICML2025  
**arXiv**: [2506.23283](https://arxiv.org/abs/2506.23283)  
**作者**: Yuhuan Yang, Chaofan Ma, Zhenjie Mao, Jiangchao Yao, Ya Zhang, Yanfeng Wang
**机构**: 上海交通大学
**代码**: 待确认  
**领域**: 视频理解  
**关键词**: Mamba, PEFT, 视频理解, 图像基础模型适配, 状态空间模型, 时空建模

## 一句话总结

提出 MoMa 框架，通过序列调制操作 (SeqMod) 将 Mamba 的线性复杂度 SSM 以 scale-bias 方式注入冻结的 CLIP Transformer，实现高效全时空动态建模，在多个视频识别基准上以更少计算量达到 SOTA 水平。

## 研究背景与动机

- **核心问题**：如何高效地将图像基础模型 (IFM) 适配到视频理解任务？现有 PEFT 方法（AIM、DualPath、DiST 等）将空间和时间信息分开处理，无法显式捕获完整的时空动态关系。
- **朴素方案的问题**：对整个时空序列做 full attention 复杂度为 $O((HWT)^2)$，不可扩展；直接在冻结的 Transformer 中插入轻量 Mamba 模块会干扰预训练特征（实验证实效果不佳）。
- **动机**：Mamba 的 SSM 具有线性序列建模复杂度，适合处理视频中的长时空序列。但需要一种"不破坏预训练权重"的融合方式，从而兼顾 IFM 的强表征和 Mamba 的高效序列建模。

## 方法详解

### 整体框架：Divide-and-Modulate

MoMa 在 CLIP 的每个 Transformer 层中引入两个阶段：

1. **Divide 阶段**：降低注意力计算开销
2. **Modulate 阶段**：通过 Mamba SSM 注入全时空动态信息

修改后的前向过程为：

$$\mathbf{V}^{i+1} = \text{FFN}(\text{Modulate}(\text{Divide}(\mathbf{V}^i)))$$

其中 $\mathbf{V}^i \in \mathbb{R}^{HWT \times C}$ 为第 $i$ 层的视频特征。

### Divide 阶段

将每帧划分为 $w \times w$ 的非重叠窗口，在窗口内独立做 self-attention：

$$\mathbf{V}^i \rightarrow [\mathbf{s}_1^i, \mathbf{s}_2^i, \ldots, \mathbf{s}_N^i], \quad \mathbf{s}_n^i \in \mathbb{R}^{w^2 \times C}, \quad N = \frac{HWT}{w^2}$$

$$\mathbf{s}_n^{i\prime} = \text{Attention}(\mathbf{s}_n^i)$$

复杂度从逐帧 attention 的 $O((HW)^2 T)$ 降到 $O(w^2 \cdot HWT)$，实现线性时间复杂度。实验中 $w=8$。

### Modulate 阶段与 SeqMod 操作

**SSM 前向层**：对 Divide 输出 $\mathbf{x}^i$ 通过一个修改的 Mamba SSM 层，输出投影通道翻倍后按通道拆成两个序列：

$$\mathbf{y}_1^i, \mathbf{y}_2^i = \text{SSM}(\mathbf{x}^i)$$

SSM 内部采用多次双向扫描（空间+时间维度），参考 VideoMamba 的设计。

**SeqMod 操作**：受自适应归一化 (AdaN/FiLM) 启发，但将标量 scale/bias 扩展为与输入等长的张量序列，实现细粒度的序列到序列调制：

$$\text{SeqMod}(\mathbf{x}, \mathbf{y}_1, \mathbf{y}_2) = \underbrace{\mathbf{y}_1}_{\text{scale}} \odot \mathbf{x} + \underbrace{\mathbf{y}_2}_{\text{bias}} + \underbrace{\mathbf{x}}_{\text{skip}}$$

- $\mathbf{y}_1$ 作为序列级 scale，$\mathbf{y}_2$ 作为序列级 bias，$\odot$ 为逐元素乘法
- 保留跳跃连接，确保不破坏 CLIP 原始特征
- 与 AdaN 的关键区别：AdaN 使用全局标量调制，SeqMod 使用与序列等长的张量调制，保留细粒度时空信息

最终输出送入 CLIP 的冻结 FFN 层：

$$\mathbf{V}^{i+1} = \text{FFN}(\text{SeqMod}(\mathbf{x}^i, \mathbf{y}_1^i, \mathbf{y}_2^i))$$

### 训练过程

- 对最终层输出做平均池化得到视频表征：$\hat{\mathbf{y}}_o = \text{Average}(\mathbf{V}^L)$
- **冻结所有 CLIP 参数**，仅训练新引入的 SSM 层
- 损失函数 = 分类损失 + CLIP 蒸馏损失（保持零样本理解能力，防止特征空间偏移过大）
- 端到端训练

## 训练与实验设置

| 配置项 | 值 |
|---|---|
| 基础模型 | CLIP ViT-B/16 / ViT-L/14 |
| 窗口大小 $w$ | 8 |
| SSM 隐藏状态 | 16 |
| SSM 隐藏维度 | 384 |
| 激活函数 | GELU |
| 优化器 | AdamW, lr=3e-4, wd=0.05 |
| 训练设备 | 8x Tesla V100, fp16 |
| 训练时长 | ~12h (K400, 30 epochs) |
| 可训练参数 | 11M (ViT-B/16) / 39M (ViT-L/14) |

## 主要结果

### Kinetics-400

| 方法 | Backbone | GFLOPs | 可训练参数 | Top-1 |
|---|---|---|---|---|
| AIM | ViT-B/16 | 1214 | 11M | 84.5 |
| DiST | ViT-B/16 | 986 | 26M | 84.4 |
| **MoMa** | **ViT-B/16** | **902** | **11M** | **84.8** |
| AIM | ViT-L/14 | 5604 | 38M | 87.3 |
| DiST | ViT-L/14 | 4534 | 40M | 87.6 |
| **MoMa** | **ViT-L/14** | **4152** | **39M** | **87.8** |

### Something-Something V2（强时序建模需求）

| 方法 | Backbone | GFLOPs | Top-1 |
|---|---|---|---|
| AIM | ViT-B/16 | 2496 | 69.1 |
| DiST | ViT-B/16 | 1972 | 70.9 |
| **MoMa** | **ViT-B/16** | **1804** | **71.5** |
| DiST | ViT-L/14 | 9068 | 73.1 |
| **MoMa** | **ViT-L/14** | **8304** | **73.8** |

### 长视频识别（Breakfast / COIN）

| 方法 | Breakfast | COIN |
|---|---|---|
| VideoMamba (64帧) | 95.8 | 89.5 |
| **MoMa (64帧)** | **96.9** | **90.0** |

### 零样本迁移（HMDB51 / UCF101）

训练于 K400 后直接评估，性能优于 DiST，表明 CLIP 蒸馏损失有效保留了零样本泛化能力。

## 亮点与洞察

1. **融合策略的巧妙设计**：不直接用 Mamba 特征替换/拼接 Transformer 特征，而是通过 scale-bias 调制注入，最大限度保持预训练权重的稳定性。这一设计灵感来自 FiLM/AdaIN/DiT，但创新地将标量调制推广到序列级调制。
2. **效率优势显著**：在 K400 上相比 AIM 减少 25.6% FLOPs，相比 DiST 减少 8.5% FLOPs，且可训练参数更少（ViT-B/16 仅 11M）。
3. **窗口化注意力 + SSM 的互补**：Divide 阶段用窗口局部注意力捕获短程空间依赖，Modulate 阶段用 SSM 捕获全局长程时空依赖，分工明确。
4. **长视频场景表现突出**：在 Breakfast 和 COIN 数据集上不仅超越传统方法，也优于专为长视频设计的 VideoMamba。

## 局限与展望

1. **SSv2 上 ViT-L/14 的 8帧设置低于 DiST**（72.2 vs 73.1），需要 32 帧才能超过，说明在帧数受限时 Mamba 的优势减弱。
2. **K400 上 ViT-L/14 的 8帧设置也低于竞品**（86.7 vs 87.3），MoMa 对帧数的依赖较大。
3. **窗口大小 $w$ 为固定超参**，未探索自适应窗口或多尺度窗口策略。
4. **仅在分类任务上验证**，未涉及视频问答、视频描述生成等更复杂的视频理解任务。
5. **双向多次扫描**的具体设计（几次扫描、扫描顺序）对性能的影响未充分分析。
6. **CLIP 蒸馏损失的权重**如何设定缺少细致讨论。

## 可复现性要点

- 基于 CLIP 预训练模型，公开可获取
- 超参数完整披露（lr、wd、SSM 参数、窗口大小）
- 8x V100 + fp16，硬件门槛中等
- 使用 ActionCLIP 的 prompt 模板
- K400 训练 30 epochs 约 12 小时，复现成本可控
- 代码暂未公开，但方法描述足够详细（SSM 层结构、SeqMod 公式明确）

## 相关工作与启发

- **AIM / DiST / EVL / ST-Adapter**：同属 CLIP 适配视频理解的 PEFT 路线，MoMa 的核心区别在于用 SSM 替代额外的 attention/encoder 来做时序建模
- **VideoMamba**：从头训练的纯 Mamba 视频模型，MoMa 则是将 Mamba 嵌入预训练 Transformer
- **FiLM / AdaIN / DiT**：AdaN 技术的源头，启发了 SeqMod 的设计
- **Jamba / MambaVision**：Mamba-Attention 混合架构的先驱，但都是从头训练，MoMa 面临的约束不同

## 评分
- 新颖性: 4/5 - SeqMod 操作将 AdaN 推广到序列级调制用于 Mamba-Transformer 融合，思路新颖
- 实验充分度: 4/5 - 覆盖 K400、SSv2、长视频、零样本，消融实验分析了多种融合方式
- 写作质量: 4/5 - 结构清晰，公式推导完整，动机阐述充分
- 价值: 4/5 - 为 Mamba 嵌入预训练 Transformer 提供了一种通用的非破坏性融合范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](../../CVPR2025/video_understanding/pave_patching_and_adapting_video_large_language_models.md)
- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](../../CVPR2025/video_understanding/efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[CVPR 2026\] Gamba: Mamba-based Graph Convolutional Network with Dynamic Graph Topology Learning for Action Recognition](../../CVPR2026/video_understanding/gamba_mamba-based_graph_convolutional_network_with_dynamic_graph_topology_learni.md)
- [\[NeurIPS 2025\] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models](../../NeurIPS2025/video_understanding/mimeqa_towards_socially-intelligent_nonverbal_foundation_models.md)
- [\[CVPR 2026\] UniVBench: Towards Unified Evaluation for Video Foundation Models](../../CVPR2026/video_understanding/univbench_towards_unified_evaluation_for_video_foundation_models.md)

</div>

<!-- RELATED:END -->
