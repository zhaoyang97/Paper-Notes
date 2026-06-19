---
title: >-
  [论文解读] Learning Conformational Ensembles of Proteins Based on Backbone Geometry
description: >-
  [NeurIPS 2025][计算生物][蛋白质构象集合] 提出 BBFlow，一种基于蛋白质骨架几何信息的流匹配生成模型，用于蛋白质构象集合采样，无需进化序列信息或预训练折叠模型，推理速度比 AlphaFlow 快一个数量级以上，且可扩展到多链蛋白质。 蛋白质的功能依赖于其结构动力学，即在热力学平衡下蛋白质可访问的构象集合…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "蛋白质构象集合"
  - "流匹配"
  - "骨架几何"
  - "分子动力学模拟"
---

# Learning Conformational Ensembles of Proteins Based on Backbone Geometry

**会议**: NeurIPS 2025  
**arXiv**: [2503.05738](https://arxiv.org/abs/2503.05738)  
**代码**: [GitHub](https://github.com/graeter-group/bbflow)  
**领域**: 医学图像 / 计算生物学  
**关键词**: 蛋白质构象集合, 流匹配, 骨架几何, 分子动力学模拟

## 一句话总结

提出 BBFlow，一种基于蛋白质骨架几何信息的流匹配生成模型，用于蛋白质构象集合采样，无需进化序列信息或预训练折叠模型，推理速度比 AlphaFlow 快一个数量级以上，且可扩展到多链蛋白质。

## 研究背景与动机

蛋白质的功能依赖于其结构动力学，即在热力学平衡下蛋白质可访问的构象集合（Boltzmann 分布）。传统上通过分子动力学（MD）模拟来采样这些构象，但 MD 需要极长的模拟时间来克服局部自由能极小值，计算开销极其昂贵。

近年来，深度生成模型被提出作为 MD 的替代方案。当前 SOTA 方法如 AlphaFlow 依赖于对 AlphaFold 2 等预训练折叠模型的微调，并需要进化序列信息（MSA 或蛋白质语言模型权重）。这带来了三个核心问题：

**效率瓶颈**：依赖大型预训练折叠模型，每次推理都需要从序列预测整体折叠结构

**信息偏差**：进化信息（MSA）对于 de novo 设计的蛋白质不可用或非常稀缺，导致建模偏差

**适用范围有限**：现有方法仅限于单链蛋白质，无法处理多链蛋白质复合物

本文的核心 idea 是：将构象集合生成任务与结构预测任务解耦，仅基于蛋白质骨架的几何信息（而非序列信息）来学习构象分布。通过条件化平衡结构的几何编码，既消除了对进化信息的需求，又大幅提高了推理效率。

## 方法详解

### 整体框架

BBFlow 将蛋白质构象集合预测建模为一个条件结构生成任务：给定蛋白质的平衡态骨架结构 $x_{\text{eq}}$，学习条件概率分布 $p(x|x_{\text{eq}})$。具体采用 SE(3) 流匹配模型，将蛋白质骨架表示为一系列欧几里得帧 $x = (r, z) \in \text{SE}(3)$（旋转 + 平移），在 $\text{SE}(3)^N$ 流形上学习条件流向量场。

### 关键设计

#### 1. 平衡结构的几何编码

为了将平衡态结构作为条件输入，BBFlow 设计了两种互补的编码方式：

**距离编码**：将残基对之间的欧几里得距离离散化为 bin 特征：

$$s_{ij} = \text{bin}(\|z_i - z_j\|_2)$$

在 0 到 20Å 之间均匀分为 22 个 bin，作为初始边特征。这种编码类似于进化信息提供的接触图。

**方向编码**：对于距离小于 5Å 的残基对，计算等变的成对方向向量：

$$e_{ij} = r_i^{-1}\left(\frac{z_i - z_j}{\|z_i - z_j\|_2}\right)$$

通过将方向向量变换到残基 $i$ 的局部坐标系，使特征分量成为不变量，可与距离编码一起作为边特征。

**设计动机**：距离编码捕获全局的空间临近关系（类似接触图），而方向编码则提供更精细的局部几何结构信息。这两者共同替代了传统方法中进化信息的角色。

#### 2. 条件先验分布

与传统流匹配使用无条件先验不同，BBFlow 提出了一种条件先验分布 $p_0(x|x_{\text{eq}})$。通过在无条件先验样本和平衡结构之间进行测地线插值来生成先验样本：

$$x_0 = \gamma(x_{\text{uncond}}, x_{\text{eq}}, \xi)$$

其中 $\gamma$ 是 $x_{\text{uncond}}$ 和 $x_{\text{eq}}$ 之间的测地线，$\xi = 0.2$ 控制先验样本与平衡结构的接近程度。这可以看作将扩散模型中的 partial denoising 方法推广到流匹配框架。

**设计动机**：条件先验使初始噪声分布已经包含了目标蛋白质的粗略结构信息，降低了流匹配的学习难度，从而用更少的时间步即可生成高质量构象。

#### 3. 模型架构

采用 GAFL 架构（FrameDiff 的扩展），核心是 SE(3) 等变图神经网络，使用 Clifford Frame Attention (CFA) 机制。关键创新：

- **去除残基索引编码**：不使用残基在链中的位置索引作为输入特征，因为这一信息已经隐含在平衡结构的距离矩阵中。这样做减少了记忆效应，并使模型可以自然地从单链蛋白质迁移到多链蛋白质
- **氨基酸类型编码**：通过 one-hot 编码 + 线性层映射为 128 维嵌入，提供局部自由度信息
- 6 层消息传递块，逐步更新帧与节点/边特征

### 损失函数 / 训练策略

流匹配损失为条件流向量场的均方误差：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}\left[\|v - \hat{v}(x_t, t, x_{\text{eq}})\|^2_{\text{SE}(3)}\right]$$

其中度量定义为 $\|v\|^2_{\text{SE}(3)} = \text{Tr}(v_r v_r^T)/2 + \|v_z\|^2_2$，同时使用辅助损失。

训练在 ATLAS 数据集上进行（1265 个蛋白质用于训练），从零开始训练仅需 2 块 A100 GPU 3 天，时间步数设为 20。

## 实验关键数据

### 主实验（ATLAS 基准测试）

| 方法 | RMSF $r$↑ | RMSF MAE↓ | Pw-RMSD MAE↓ | DCCM $r$↑ | PCA $\mathcal{W}_2$↓ | 推理时间(s)↓ |
|------|-----------|-----------|--------------|-----------|---------------------|-------------|
| AlphaFlow | 0.86 | 0.59 | 1.35 | 0.86 | 1.47 | 32.0 |
| AlphaFlow-T | **0.92** | 0.41 | 0.91 | **0.89** | 1.28 | 32.6 |
| AlphaFlow-T_dist | 0.92 | 0.68 | 1.41 | 0.88 | 1.43 | 3.3 |
| ConfDiff | 0.88 | 0.62 | 1.45 | 0.86 | 1.41 | 20.2 |
| **BBFlow** | 0.90 | **0.42** | **0.77** | 0.87 | **1.33** | **0.8** |

BBFlow 在保持与 AlphaFlow-T 相当精度的同时，推理速度快约 40 倍。

### 消融实验

| 配置 | RMSF MAE↓ | Pw-RMSD MAE↓ | DCCM $r$↑ | 说明 |
|------|-----------|--------------|-----------|------|
| BBFlow（完整） | **0.42** | **0.77** | **0.87** | 所有组件 |
| 无方向编码 | 0.52 | 1.15 | 0.85 | 仅距离编码 |
| 无条件先验 | 0.48 | 0.90 | 0.86 | 使用无条件先验 |
| 有残基索引 | 0.42 | 0.82 | 0.88 | 加入残基位置 |
| 无氨基酸编码 | 0.54 | 0.93 | 0.85 | 仅纯几何 |
| 无距离编码 | 5.88 | 7.08 | 0.55 | 完全失败 |

### 关键发现

1. **De novo 蛋白质**：BBFlow 在 de novo 蛋白质上表现稳健（RMSF MAE=0.26），而依赖进化信息的 AlphaFlow（无模板）严重失败（MAE=4.76）
2. **多链蛋白质**：BBFlow 是首个可应用于多链蛋白质的构象集合生成模型，虽然仅在单链数据上训练，却可成功捕获链间和链内运动相关性
3. **速度-精度权衡**：BBFlow 在所有基线中实现了最佳的速度-精度平衡，300 残基蛋白质每个构象仅需 0.8 秒

## 亮点与洞察

- **几何信息替代进化信息**：证明了蛋白质构象采样不一定需要进化序列信息，仅凭骨架几何结构即可达到 SOTA 水平，这对计算生物学领域有深远启示
- **条件先验是关键创新**：将 partial denoising 推广到流匹配框架的条件先验，是一个通用的技术贡献
- **去除残基索引实现跨链迁移**：通过不使用位置索引而改用结构编码，优雅地解决了单链到多链的迁移问题

## 局限与展望

- 作为 MD 模拟器，BBFlow 无法预测远离平衡态的构象（如替代折叠状态），除非在相应长 MD 模拟数据上训练
- 瞬态接触预测精度不如使用 MSA 的方法，表明进化信息对预测稀有事件仍有价值
- 目前仅生成骨架构象，不包含侧链构象集合或蛋白质-配体相互作用

## 相关工作与启发

- AlphaFlow 系列通过微调 AlphaFold 2 实现构象采样，但受限于预训练模型
- FrameFlow / GAFL 在蛋白质设计中使用 SE(3) 流匹配，BBFlow 将其适配到条件生成
- 条件先验的思路可推广到其他条件生成任务中

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 条件先验 + 几何编码替代进化信息，idea 新颖且验证充分
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖自然蛋白质、de novo 蛋白质、多链蛋白质，有完整消融实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数学严谨，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 40 倍加速 + 无需进化信息 + 多链扩展，实际应用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)
- [\[ICML 2026\] Viral Proteins Reveal Geometry of Protein Language Models](../../ICML2026/computational_biology/viral_proteins_reveal_geometry_of_protein_language_models.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](learning_repetition-invariant_representations_for_polymer_informatics.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](gflownets_for_learning_better_drug-drug_interaction_representations.md)
- [\[NeurIPS 2025\] Self Iterative Label Refinement via Robust Unlabeled Learning](self_iterative_label_refinement_via_robust_unlabeled_learning.md)

</div>

<!-- RELATED:END -->
