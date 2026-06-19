---
title: >-
  [论文解读] WAVE: Weight Templates for Adaptive Initialization of Variable-sized Models
description: >-
  [CVPR 2025][模型压缩][权重模板] 提出 WAVE，将变尺寸模型初始化重新定义为多任务学习问题，通过共享的尺寸无关权重模板和轻量级尺寸特定的权重缩放器（via Kronecker 积）实现高效初始化，仅需 3.3% 预训练参数即可在 10 个 epoch 内超越 150 epoch 训练的模型。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "权重模板"
  - "模型初始化"
  - "Learngene"
  - "Kronecker积"
  - "变尺寸模型"
---

# WAVE: Weight Templates for Adaptive Initialization of Variable-sized Models

**会议**: CVPR 2025  
**arXiv**: [2406.17503](https://arxiv.org/abs/2406.17503)  
**代码**: [GitHub](https://github.com/fu-feng/WAVE)  
**领域**: 模型压缩 / 模型初始化 (Model Compression / Model Initialization)  
**关键词**: 权重模板, 模型初始化, Learngene, Kronecker积, 变尺寸模型

## 一句话总结

提出 WAVE，将变尺寸模型初始化重新定义为多任务学习问题，通过共享的尺寸无关权重模板和轻量级尺寸特定的权重缩放器（via Kronecker 积）实现高效初始化，仅需 3.3% 预训练参数即可在 10 个 epoch 内超越 150 epoch 训练的模型。

## 研究背景与动机

随着模型参数量指数级增长，从头训练的成本越来越不可接受。微调预训练模型成为主流范式，但实际部署中常需要**不同大小**的模型（受内存、算力、延迟等约束）。现有预训练模型通常只有有限的几种尺寸（如 ViT-B 12 层），目标模型与预训练尺寸不匹配时需要重新预训练。

现有方法（如 Weight Selection、LiGO）通过选择或变换预训练权重矩阵来初始化不同大小的模型，但往往破坏原始模型中的结构化知识，或引入过多随机参数，效果受限。

WAVE 的核心洞察是将**变尺寸模型初始化**类比为**多任务学习**：每种尺寸的初始化是一个独立"任务"，需要一个通用的任务无关骨干（权重模板）加上少量任务特定的适配参数（权重缩放器）。在 Learngene 框架下，权重模板封装了尺寸无关的预训练知识，通过 Kronecker 积与尺寸特定的缩放器组合，可以灵活初始化任意深度和宽度的模型。

## 方法详解

### 整体框架

WAVE 分为两阶段：（1）**知识整合**——通过蒸馏将预训练模型知识整合到结构化的权重模板中（一次性过程，150 epoch on ImageNet）；（2）**模型初始化**——冻结权重模板，仅训练轻量级权重缩放器来确定模板的组合方式，适配目标模型尺寸（少量数据即可）。

### 关键设计

1. **权重模板与 Kronecker 积重构（Weight Templates + Kronecker Reconstruction）**:
    - 功能：将权重矩阵表示为共享模板的加权组合
    - 核心思路：对于 ViT 第 $l$ 层的权重矩阵 $W_\star^{(l)} \in \mathbb{R}^{m_1 \times m_2}$（$\star \in \{qkv, o, in, out\}$），用 $N_\star$ 个共享权重模板 $T_\star^{(t)} \in \mathbb{R}^{w_1 \times w_2}$ 和对应的缩放器 $S_\star^{(l,t)} \in \mathbb{R}^{\frac{m_1}{w_1} \times \frac{m_2}{w_2}}$ 通过 Kronecker 积重构：$W_\star^{(l)} = \sum_{t=1}^{N_\star} T_\star^{(t)} \otimes S_\star^{(l,t)}$。缩放器仅包含几千个参数，极其轻量
    - 设计动机：已有证据表明预训练 ViT 的不同层权重间存在显著的结构性关联（Mimetic Init、TLEG），权重模板可以捕获这种共享结构

2. **通过蒸馏整合预训练知识**:
    - 功能：将预训练模型的知识注入权重模板
    - 核心思路：构建一个辅助模型 $f_{aux}$，其参数 $\theta_{aux} = \mathcal{T} \otimes \mathcal{S}_{aux}$ 完全由权重模板和辅助缩放器通过 Kronecker 积生成。使用 KL 散度蒸馏 + 交叉熵损失 $\mathcal{L} = \text{KL}(z_{pre} \| z_{aux}) + \text{CE}(z_{aux}, y)$ 训练，梯度回传到模板和缩放器。辅助模型充当知识传递媒介和非结构化知识的过滤瓶颈
    - 设计动机：权重模板本身无法独立学习数据，需要辅助模型作为桥梁；Kronecker 积约束确保模板保持结构化

3. **轻量级缩放器适配（Lightweight Scaler Adaptation）**:
    - 功能：通过极少参数实现对任意尺寸模型的初始化
    - 核心思路：冻结权重模板 $\mathcal{T}$，根据目标模型尺寸构造对应的缩放器 $\mathcal{S}_{tar}$，用少量数据训练缩放器学习模板的组合规则。缩放器维度由目标权重矩阵和模板大小决定：$S_{\star,tar}^{(l,t)} \in \mathbb{R}^{\frac{m_1}{w_1} \times \frac{m_2}{w_2}}$。当目标模型的深度或宽度变化时，仅缩放器的层数或维度变化，模板始终共享
    - 设计动机：类似 LoRA 的思路——共享的大型表征（模板）+ 少量可训练参数（缩放器）实现高效适配

### 损失函数 / 训练策略

- **知识整合阶段**: KL 蒸馏 + CE 分类损失，150 epoch on ImageNet-1K
- **初始化阶段**: 仅训练缩放器（模板冻结），少量数据
- **教师模型**: 预训练 ViT-B 等作为 ancestry model
- **辅助模型**: 参数完全由模板+缩放器的 Kronecker 积构成

## 实验关键数据

### 主实验

ImageNet-1K 上不同尺寸模型初始化后训练 10 epoch 的 Top-1 准确率：

| 方法 | 传递参数占比 | 6层ViT | 8层ViT | 12层ViT | 跨宽度 |
|------|-----------|--------|--------|---------|-------|
| 从头训练 150 epoch | — | X% | Y% | Z% | — |
| Weight Selection | 100% | — | — | — | 仅支持小→大 |
| LiGO | 100% | — | — | — | 仅支持小→大 |
| TLEG | ~50% | — | — | — | 仅支持深度 |
| **WAVE (10 epoch)** | **3.3%** | **超越** | **超越** | **超越** | **支持深度+宽度** |

计算节省：对 $n$ 种尺寸的模型，WAVE 节省 $15n\times$ 计算量。

### 消融实验

| 配置 | 说明 |
|------|------|
| 不同模板数量 | 更多模板容量更大但收益递减 |
| 无蒸馏（随机模板） | 性能显著下降，证明知识整合必要性 |
| 固定缩放器（不训练） | 初始化质量差，说明缩放器学习组合规则重要 |
| 仅 KL loss | 比 KL+CE 略差 |
| 跨数据集迁移 | 模板知识是任务无关的，可迁移到多个下游数据集 |

### 关键发现

- 仅传递预训练模型 3.3% 的参数，WAVE 初始化 + 10 epoch 训练即超越从头 150 epoch 训练
- 首个同时支持**深度和宽度**变化的 Learngene 方法
- 权重模板的可视化揭示了高度结构化的知识模式，与预训练模型中观察到的模式一致
- 权重模板具有任务无关性，可在 CIFAR-100、Flowers 等多个下游数据集上直接使用

## 亮点与洞察

- **多任务学习视角的引入极具创造性**：将"初始化不同大小的模型"类比为"适配不同任务"，自然引出共享骨干（模板）+ 轻量适配器（缩放器）的架构，逻辑简洁有力
- **Kronecker 积作为模板组合机制非常优雅**：它自然地将模板的局部结构放大到任意目标大小，且数学上可以证明之前的层级 Learngene 方法（Heur-LG、Auto-LG、TLEG）都是 WAVE 的特例

## 局限与展望

- 权重模板的一次性构建仍需 150 epoch 训练，对于非常大的预训练模型可能开销较大
- 目前仅在 ViT 架构上验证，对 CNN、MLP-Mixer 等其他架构的适用性未知
- Kronecker 积的结构约束可能限制了对完全不规则权重模式的表达能力
- 缩放器的初始化策略对最终效果有影响，目前使用简单规则

## 相关工作与启发

- **vs Weight Selection**: 从大模型选择权重给小模型，直接传递但可能破坏结构；WAVE 通过模板保持结构化，且传递参数量仅 3.3%
- **vs LiGO**: 将小模型线性扩展为大模型，方向固定（小→大）；WAVE 双向适配
- **vs TLEG**: TLEG 将每层表征为两个基层的线性组合，是 WAVE 的特例；WAVE 支持更灵活的多模板组合和宽度变化

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 多任务视角 + Kronecker 积模板组合，统一了多种 Learngene 方法
- 实验充分度: ⭐⭐⭐⭐ 多种尺寸×多数据集×多 baseline 对比，首个 Learngene 综合基准
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 3.3% 参数传递实现 15n× 计算节省，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics](../../ICML2025/model_compression/beyond_zero_initialization_investigating_the_impact_of_non-zero_initialization_o.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](../../ICML2025/model_compression/random_initialization_of_gated_sparse_adapters.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[CVPR 2025\] HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis](hyperlora_parameter-efficient_adaptive_generation_for_portrait_synthesis.md)

</div>

<!-- RELATED:END -->
