---
title: >-
  [论文解读] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations
description: >-
  [AAAI 2026][可解释性][概念瓶颈模型] 提出 PCBM-ReD，通过从预训练视觉编码器中自动提取概念、MLLM 标注/过滤、重建引导选择，再利用 CLIP 视觉-文本对齐将图像表示稀疏分解为概念嵌入的线性组合，构建事后概念瓶颈模型，在 11 个分类任务上达到 SOTA 精度且保持可解释性。
tags:
  - "AAAI 2026"
  - "可解释性"
  - "概念瓶颈模型"
  - "CLIP"
  - "稀疏分解"
  - "表示学习"
---

# Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations

**会议**: AAAI 2026  
**arXiv**: [2601.12303](https://arxiv.org/abs/2601.12303)  
**代码**: [GitHub](https://github.com/peterant330/PCBM_ReD)  
**领域**: 可解释性  
**关键词**: 概念瓶颈模型, 可解释性, CLIP, 稀疏分解, 表示学习

## 一句话总结

提出 PCBM-ReD，通过从预训练视觉编码器中自动提取概念、MLLM 标注/过滤、重建引导选择，再利用 CLIP 视觉-文本对齐将图像表示稀疏分解为概念嵌入的线性组合，构建事后概念瓶颈模型，在 11 个分类任务上达到 SOTA 精度且保持可解释性。

## 研究背景与动机

深度学习模型的"黑箱"特性限制了其在医学影像、自动驾驶等关键领域的部署。概念瓶颈模型（CBM）通过人类可理解的概念中间层来解释预测，但现有方法存在多个问题：

**事后方法**：提取的概念不保证反映网络真实推理过程，概念与目标之间缺乏因果关系

**手工概念**（原始 CBM）：耗时且覆盖不全

**LLM 生成概念**（LaBo、Label-free CBM）：包含非视觉特征（如食物味道、鸟类行为），且与数据分布和编码器能力无关

**概念独立性**：现有方法不保证概念间的线性无关，影响干预效果

**精度-可解释性权衡**：现有 CBM 与端到端模型相比精度损失明显

**核心洞察**：概念应该从预训练编码器的表示中提取（数据驱动），而非独立于模型/数据地设计，这样才能最大化利用编码器的表示能力。

## 方法详解

### 整体框架

PCBM-ReD 三阶段流水线：

**阶段1：数据驱动概念发现**
- 用稀疏自编码器（SAE）对 CLIP 视觉编码器的隐空间进行字典学习：$\mathbf{I}_i \approx \mathbf{V}\mathbf{u}_i$
- 每列 $\mathbf{V}$ 代表一个概念，$\mathbf{u}_i$ 中的值反映概念在图像中的重要性
- 对每个概念，选取激活值最高的 Top-K 图像，用 MLLM（Llama-3.2-11B-Vision）描述视觉特征
- 用 LLM（DeepSeek-V3）汇总描述并生成候选概念，然后评分过滤：只保留视觉可识别、有判别力、无捷径的高质量概念

**阶段2：重建引导概念选择**
- 从候选概念中选择一个独立子集，使其嵌入能最大化重建图像表示空间
- 贪心算法（Algorithm 1）逐步选择使重建误差最小的概念，同时保证新概念与已选概念线性无关

**阶段3：事后类别-概念关联**
- 利用 CLIP 视觉-文本对齐，将图像嵌入稀疏分解为概念文本嵌入的加权和
- 在重建嵌入上训练线性层预测类别

### 关键设计

#### 重建引导概念选择算法

优化目标：

$$\min_{\mathcal{C}} \sum_{i=1}^N \min_{\beta_i(\mathcal{C})} \|\mathbf{I}_i - \mathbf{R}(\mathcal{C})^T \beta_i(\mathcal{C})\|_F^2$$

贪心策略避免离散优化的组合爆炸，关键效率技巧：
- 利用投影矩阵 $\mathbf{P}$ 增量更新，避免每次从头求解
- 检查线性依赖性（$z=0$ 时跳过），确保概念独立
- 算法完全无监督，适用于零样本/少样本场景

#### 稀疏分解与概念评分

利用正交匹配追踪（OMP）进行稀疏编码：

$$\mathbf{I}_i = \hat{\mathbf{I}}_i + \epsilon_i = \sum_{j=1}^m w_j^i \mathbf{c}_j + \epsilon_i$$

其中只有 $n < m$ 个 $w_j^i$ 非零，确保高可解释性（每张图只用少量关键概念解释）。

丢弃残差 $\epsilon_i$，仅保留拟合表示 $\hat{\mathbf{I}}_i$ 用于分类，满足 CBM 抽象。

#### 权重矩阵初始化

用 "This is a photo of [cls]" 的文本嵌入初始化分类器权重 $\mathbf{W}$，继承 CLIP 的零样本能力。

### 损失函数 / 训练策略

- 使用 Adam 优化器训练线性头，batch size 64，学习率 $5 \times 10^{-5}$
- 视觉编码器冻结，仅训练线性分类层
- 默认使用 CLIP ViT-L/14，瓶颈大小约 300 即饱和

## 实验关键数据

### 主实验

**表1：11个数据集测试精度（CLIP ViT-L/14），全监督设置**

| 方法 | 可解释 | ImageNet | CIFAR10 | CIFAR100 | Food | Aircraft | Flower | CUB | 平均 |
|---|---|---|---|---|---|---|---|---|---|
| Linear Probe | ✗ | 83.90 | 98.10 | 87.48 | 93.17 | 64.03 | 99.45 | 84.54 | 87.38 |
| LaBo | ✓ | 83.97 | 97.75 | 86.04 | 92.45 | 61.42 | 99.35 | 81.90 | 85.72 |
| Res-CBM | ✓ | 82.98 | 97.77 | 83.01 | 90.17 | 54.67 | 97.85 | 79.27 | 83.39 |
| **PCBM-ReD** | ✓ | **84.48** | 98.05 | **87.27** | **93.16** | **62.95** | 99.39 | **84.80** | **86.97** |

**表2：CLIP RN50 上与其他 CBM 的比较**

| 方法 | CIFAR10 | CIFAR100 | CUB | 平均 |
|---|---|---|---|---|
| Linear Probe | 88.80 | 70.10 | 72.14 | 77.01 |
| PCBM | 84.50 | 56.00 | 63.63 | 68.04 |
| Label-free CBM | 86.40 | 65.13 | 62.40 | 71.31 |
| VLG-CBM | 88.63 | 66.48 | 66.03 | 73.71 |
| **PCBM-ReD** | **88.61** | **70.03** | **72.01** | **76.88** |

### 消融实验

1. **瓶颈大小**：50 个概念即能达到合理精度，300 个概念基本饱和；所需概念数与类别数无关
2. **概念创建方式**：数据驱动方案 > LLM 生成概念 > WordNet 核心概念
3. **概念选择方法**：重建引导 > K-means > 随机采样，在小瓶颈时差异更显著
4. **概念评分关联**：稀疏分解 >> CLIP 相似度评分（精度差距显著）
5. **概念来源**：编码器不匹配时性能下降，说明概念需与编码器对齐

### 关键发现

1. **PCBM-ReD 与 Linear Probe 的差距仅 0.41%**（11 个数据集平均），首次将可解释 CBM 精度提升到接近端到端模型
2. **零样本能力保持**：重建嵌入 $\hat{\mathbf{I}}_i$ 继承原始嵌入的零样本能力，零样本精度与 CLIP 几乎相同
3. **少样本性能一致优于 LaBo**：平均超出 5.01%
4. **人类评估**（39 名志愿者）：PCBM-ReD 在视觉可识别性、描述忠实性、因果关系三个维度均优于 LLM 概念基线

## 亮点与洞察

- **"从表示中提取概念"**的核心思路打通了端到端模型和可解释 CBM 之间的壁垒：不再强加外部概念，而是发现编码器已学到的概念
- **三阶段流水线设计精巧**：SAE 提取 → MLLM 标注 → 重建引导选择 → 稀疏分解，每步都有清晰的目标
- **概念选择算法的无监督特性**使其天然适用于零样本/少样本场景，这是对现有 CBM 的重要拓展
- 利用 CLIP 多模态对齐性质将视觉嵌入分解为文本概念嵌入的线性组合，理论基础扎实（有 Gandelsman 等人的前序工作支持）

## 局限与展望

1. **依赖通用 MLLM 的描述能力**：对领域特定图像（如皮肤病变）描述不精确，导致 HAM 数据集上性能下降
2. **残差丢弃带来信息损失**：虽然残差项影响较小，但理论上不为零
3. **概念数量与质量依赖采样**：有限的探测图像可能导致概念覆盖不全
4. **SAE 的训练质量影响下游效果**：不同字典学习方法的影响值得探究
5. **可扩展到视频理解或医学影像的领域特定 MLLM**

## 相关工作与启发

- **原始 CBM（Koh et al., 2020）**：手工概念+手工标注，PCBM-ReD 完全自动化
- **LaBo（Yang et al., 2023）**：用 LLM 生成概念，但与数据/模型无关
- **Res-CBM（Shang et al., 2024）**：通过增量添加概念近似残差连接
- **Gandelsman et al.**：证明 CLIP 图像嵌入可分解为文本嵌入的加权和，为本文提供理论基础
- 启发：SAE + MLLM 的概念发现流水线可以应用于任何多模态模型的可解释性分析

## 评分

| 维度 | 分数 (1-5) |
|---|---|
| 新颖性 | 4.0 |
| 技术深度 | 4.5 |
| 实验充分性 | 4.5 |
| 写作质量 | 4.0 |
| 实用价值 | 4.0 |
| **总评** | **4.2** |

## 与相关工作的对比

| 方法 | 概念来源 | 数据驱动 | 模型相关 | 零样本 | 概念独立性 | 精度（ViT-L avg） |
|---|---|---|---|---|---|---|
| 原始 CBM (Koh 2020) | 手工设计+标注 | ✗ | ✗ | ✗ | ✗ | - |
| CompDL (Yun 2022) | 手工设计+CLIP | ✗ | ✗ | ✗ | ✗ | - |
| PCBM (Yuksekgonul 2022) | 手工+残差连接 | ✗ | ✗ | ✗ | ✗ | - |
| LaBo (Yang 2023) | LLM 生成 | ✗ | ✗ | ✗ | ✗ | 85.72 |
| Label-free CBM (Oikarinen 2023) | LLM 生成 | ✗ | ✗ | ✗ | ✗ | - |
| Res-CBM (Shang 2024) | LLM+增量残差 | 部分 | ✗ | ✗ | 部分 | 83.39 |
| **PCBM-ReD (本文)** | **SAE+MLLM** | **✓** | **✓** | **✓** | **✓** | **86.97** |

关键差异点：
- **概念来源**：本文首次从编码器表示中提取概念（SAE），再用 MLLM 标注，实现数据驱动+模型感知
- **概念独立性保障**：重建引导选择算法显式检查线性无关性，其他方法均无此保障
- **零样本/少样本能力**：因为概念选择无监督，且保留了 CLIP 嵌入的对齐特性，其他 CBM 方法不具备
- **精度提升来源**：从表示空间内部挖掘概念，最大化利用编码器能力，而非从外部强加概念

## 启发与关联

1. **SAE 作为概念发现工具**：稀疏自编码器在 mechanistic interpretability 中已被广泛用于分析 LLM 内部表示（Anthropic 的工作），本文将其延伸到视觉编码器，说明 SAE 是跨模态的通用概念发现范式
2. **"从模型中来，到模型中去"的可解释性范式**：不再假设概念是外部先验，而是承认概念本身就编码在模型表示中，只需发现和命名——这一思路可推广到任何基础模型
3. **MLLM 作为概念标注器**：利用多模态 LLM 为无监督发现的概念赋予人类可理解的语义标签，这是一种低成本的"人机协作"标注方案
4. **稀疏分解的双重作用**：既提供可解释性（每张图仅依赖少量概念），又保持精度（重建嵌入近似原始嵌入），是一个优雅的设计
5. **对下游应用的启示**：该框架可直接用于医学影像诊断的可解释性需求，只需替换领域特定的 MLLM 即可提升概念质量

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](partially_shared_concept_bottleneck_models.md)
- [\[AAAI 2026\] Flexible Concept Bottleneck Model](flexible_concept_bottleneck_model.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)

</div>

<!-- RELATED:END -->
