---
title: >-
  [论文解读] Closing the Modality Gap Aligns Group-Wise Semantics
description: >-
  [ICLR2026][多模态VLM][modality gap] 证明 CLIP 中的 modality gap 对实例级任务（检索）无关紧要但严重损害群组级任务（聚类），并提出由 Align True Pairs loss + Centroid Uniformity loss 组成的新目标函数，在双模态和三模态设置中将 gap 几乎降为零，大幅提升聚类 V-Measure（+10-17 分），同时保持检索性能。
tags:
  - "ICLR2026"
  - "多模态VLM"
  - "modality gap"
  - "对比学习"
  - "CLIP"
  - "clustering"
  - "多模态"
---

# Closing the Modality Gap Aligns Group-Wise Semantics

**会议**: ICLR2026  
**arXiv**: [2601.18525](https://arxiv.org/abs/2601.18525)  
**代码**: [https://github.com/ispamm/ModGap](https://github.com/ispamm/ModGap)  
**领域**: 多模态VLM  
**关键词**: modality gap, contrastive learning, CLIP, clustering, multimodal alignment

## 一句话总结
证明 CLIP 中的 modality gap 对实例级任务（检索）无关紧要但严重损害群组级任务（聚类），并提出由 Align True Pairs loss + Centroid Uniformity loss 组成的新目标函数，在双模态和三模态设置中将 gap 几乎降为零，大幅提升聚类 V-Measure（+10-17 分），同时保持检索性能。

## 研究背景与动机

**领域现状**：CLIP 及其变体通过 InfoNCE 损失学习跨模态共享空间，但不同模态的嵌入会形成各自的聚类——即"modality gap"。现有工作对此问题态度分裂：有人认为缩小 gap 改善检索，有人认为 gap 和下游性能正相关。

**现有痛点**：(a) 现有研究只关注 gap 对检索（实例级任务）的影响，结论互相矛盾；(b) 所有方法只研究双模态（图像+文本），不涉及三模态及以上；(c) gap 的存在导致潜在空间"按模态聚类"而非"按语义聚类"，但这个后果未被系统分析。

**核心矛盾**：InfoNCE 优化的是正负对的相对排序（是否最相似），而非绝对距离（是否真正接近）。只要相对排序正确，检索就成功——即使正对的绝对余弦相似度只有 0.34。但聚类依赖绝对距离，gap 使类内散度膨胀 $\|\boldsymbol{\delta}\|^2$。

**本文目标** (a) 理论上阐明 gap 对实例级 vs 群组级任务的不同影响；(b) 提出有效缩小 gap 的方法；(c) 扩展到三模态。

**切入角度**：从 within-class scatter 的数学分解出发——gap 向量 $\boldsymbol{\delta}$ 与语义正交，因此等量膨胀所有聚类的散度，这对检索无关但对聚类致命。

**核心 idea**：modality gap 是检索的无害伪影，但是聚类的系统性障碍——用 true pair 对齐 + 质心均匀性两个损失函数可以同时消除 gap 和改善语义聚类。

## 方法详解

### 整体框架
这篇论文要解决的是 CLIP 类模型里"两个模态各自扎堆、跨模态语义对不齐"的 modality gap，而且要在不破坏检索的前提下消掉它。它的逻辑分两步：先从数学上回答"gap 到底该不该消"——证明 gap 是一个与语义正交的常数偏移，对只看相对排序的实例级任务（检索）隐形，却会均匀膨胀每个语义簇的类内散度、直接拖垮依赖绝对距离的群组级任务（聚类），所以确实该消；既然该消，再给出一个轻量的训练目标去消它。做法是保留标准的双向 InfoNCE 对比项不动，在它之上再挂两个显式的几何约束——一个把真正配对的样本往一起拽（$\mathcal{L}_{\text{ATP}}$），另一个把不同语义样本的跨模态质心在超球面上推开（$\mathcal{L}_{\text{CU}}$）。两者合起来记作 $\mathcal{L}_{\text{gap}}=\mathcal{L}_{\text{ATP}}+\mathcal{L}_{\text{CU}}$，最终训练目标是

$$\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{gap}} + \tfrac{1}{2}\big(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)}\big).$$

整个设计与模态数无关，从图像+文本的双模态直接推广到 $M$ 个模态（如音频+视频+文本）只需把求和范围扩到所有模态，不改任何架构。下面先讲两个损失各自怎么发力、为什么必须配对使用，再回到那段把"只缩 gap 不伤检索"讲透的理论分析。

### 关键设计

**1. Align True Pairs Loss（$\mathcal{L}_{\text{ATP}}$）：把"排序对了但实际还很远"的正对真正拽到一起**

gap 的根源在于 InfoNCE 只在乎相对排序——只要正对比所有负对更相似检索就成立，至于正对的绝对距离有多近它并不管，于是训练完成后匹配对的余弦相似度可能低到 0.34（隔着一条 gap）。$\mathcal{L}_{\text{ATP}}$ 直接补上这一刀，以某个锚模态 $a$ 为参照、最小化每个样本在其余模态与锚模态嵌入之间的欧氏距离：

$$\mathcal{L}_{\text{ATP}} = \frac{1}{M-1}\sum_{m\neq a}\frac{1}{N}\sum_i \big\|\mathbf{z}_i^m - \mathbf{z}_i^a\big\|_2^2.$$

这一项把正对的绝对距离按下去，gap 随之收缩。但它单独用会失控——所有点都被往一起拉，最后整个空间坍缩成一个点，语义结构荡然无存，所以必须配一个反向的张力项。

**2. Centroid Uniformity Loss（$\mathcal{L}_{\text{CU}}$）：在质心层面撑开空间，防止对齐变坍缩**

反向张力来自一个均匀性约束，但施加的对象很讲究：不是直接在单模态嵌入上推开，而是先算出每个样本的跨模态质心 $\boldsymbol{\mu}_k = \frac{1}{M}\sum_m \mathbf{z}_k^m$，再让不同语义样本的质心在单位超球面上尽量分散：

$$\mathcal{L}_{\text{CU}} = \log\frac{1}{N}\sum_i\sum_{j\neq i}\exp\big(-2\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2^2\big).$$

选质心而非单模态嵌入，是为了保住已经学到的跨模态对齐——推开的是"不同样本"，不会把同一样本的各模态又拆散。式中的 RBF 核（高斯核）形式与超球面上的均匀分布存在已知联系，最小化它等价于让质心铺满整个球面，从而给 $\mathcal{L}_{\text{ATP}}$ 的收缩提供恰好的排斥力。两者一拉一撑，gap 被压到接近零而空间不塌。

**3. 理论分析：为什么同一个 gap 对检索无害、对聚类致命**

这套损失之所以敢"只缩 gap 不怕伤检索"，背后有一个干净的数学解释。检索的成败只取决于相对排序——只要 $\text{sim}(\mathbf{z}_i^m, \mathbf{z}_i^n) > \max_{j\neq i}\text{sim}(\mathbf{z}_i^m, \mathbf{z}_j^n)$ 成立就检索正确，而 gap 是一个对所有样本一致的偏移，不改变任何一对的相对大小，因此检索完全感觉不到它。聚类则相反，它吃的是绝对距离：把类内散度（within-class scatter）做分解会得到

$$\mathbb{E}\big[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^\delta\|^2\big] \approx \mathbb{E}\big[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^0\|^2\big] + \|\boldsymbol{\delta}\|^2,$$

也就是 gap 向量 $\boldsymbol{\delta}$ 给每一个聚类的散度都加上同样大小的 $\|\boldsymbol{\delta}\|^2$，把本该按语义抱团的样本硬生生撑散。关键在于 gap 向量 $\boldsymbol{\delta}$ 被证明与语义方向正交（Zhang et al., 2023），所以它表现得就像一个常数偏移：排序不受影响，绝对距离却被整体抬高——这正是它"对实例级任务隐形、对群组级任务有害"的根本原因。

### 损失函数 / 训练策略
完整目标把上面三块拼起来：$\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{ATP}} + \mathcal{L}_{\text{CU}} + \frac{1}{2}(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)})$。训练中还观察到一个有意思的自洽现象：随着 gap 被压缩，非匹配对的梯度反而增大、变成更有信息量的 hard negatives，而已经拉近的匹配对梯度减小——优化的重心自然从"把模态对齐"转向"精细打磨语义结构"。

## 实验关键数据

### 主实验（Gap 值 vs 检索 vs 聚类）

| 方法 | 数据集 | Gap ↓ | CM R@1 | V-Measure ↑ | kNN ↑ |
|------|--------|-------|--------|-------------|-------|
| CLIP (LT) | MSCOCO (2-modal) | 0.47 | 74.6 | 12.98 | 26.3 |
| CLIP (FT) | MSCOCO | 0.12 | 73.2 | 12.99 | 31.0 |
| **Ours** | MSCOCO | **0.03** | 70.3 | **23.63** | **36.4** |
| CLIP (LT) | MSR-VTT (3-modal) | 0.29 | 34.2/10.3 | 23.3 | 52.9 |
| **Ours** | MSR-VTT | **0.07** | 32.8/11.8 | **32.1** | **58.0** |
| CLIP (LT) | AV-MNIST (3-modal) | 0.20 | 87.1/84.2 | 77.6 | 87.0 |
| **Ours** | AV-MNIST | **0.09** | 88.7/89.1 | **82.7** | **89.2** |

### 消融实验（Cos True Pairs 提升）

| 方法 | MSCOCO Gap | Cos TP ↑ | MM R@1 | CIDEr (captioning) |
|------|-----------|---------|--------|---------------------|
| CLIP (LT) | 0.47 | 0.34 | 72.5 | 153.2 |
| CLIP (FT) | 0.12 | 0.63 | 73.8 | 155.0 |
| Ours | **0.03** | **0.77** | 76.2 | **158.2** |

### 关键发现
- **检索几乎不受 gap 影响**：CLIP (LT) gap=0.47 和 Ours gap=0.03，MSCOCO 检索 R@1 只差 4.3 分，但 V-Measure 差 10.65 分——证实理论预测
- **聚类与 gap 强相关**：MSR-VTT 上人工控制 gap 从 0.3→0，V-Measure 从 23.3→+17.5 提升
- **正对余弦相似度从 0.34 到 0.77**：CLIP 训练后正对相似度竟然只有 0.34（很远！），说明 InfoNCE 只保证排序不保证接近
- **三模态也有效**：AV-MNIST 和 MSR-VTT（音频+视频+文本）三模态场景均成功缩小 gap 并提升聚类
- **captioning 也受益**：更好的对齐空间让解码器生成更准确的 caption（CIDEr +5）

## 亮点与洞察
- **重新定义 modality gap 的意义**：从"要不要缩 gap"的争论，转向"gap 对什么任务有影响"的精确分析——这个 insight 可以终结该领域的矛盾结论
- **$\mathcal{L}_{\text{ATP}}$ + $\mathcal{L}_{\text{CU}}$ 的互补设计**：拉近匹配对（可能坍缩）+ 在质心层面推开非匹配对（防坍缩）——比直接在嵌入层面做 alignment+uniformity 更优雅
- **梯度视角的机制解释**：gap 缩小 → 非匹配对变成更有效的 hard negatives → 梯度更集中于精细化语义结构——这个发现对理解对比学习动态有重要意义
- **可直接扩展到 N 模态**：方法设计天然支持任意数量模态，不需要架构修改

## 局限与展望
- **检索轻微下降**：MSCOCO 上检索从 74.6 降到 70.3——虽然理论上应该无关，但实际有微小 trade-off。可能是 $\mathcal{L}_{\text{ATP}}$ 的拉近操作轻微扰乱了排序
- **实验规模受限**：最大实验是 MSCOCO（EVA-CLIP ViT-G），未在 LAION-5B 预训练规模验证。大规模对比学习中 gap 的行为可能不同
- **只验证对比学习 pipeline**：CLIP-like 方法。SigLIP、BLIP-2 等非对比方法的 gap 行为未探索
- **改进方向**：将 $\mathcal{L}_{\text{gap}}$ 集成到 VLM 预训练中，可能改善下游群组级理解能力（如 zero-shot 分类、视觉常识推理等）

## 相关工作与启发
- **vs Liang et al. (2022)**：他们首次发现 modality gap 并提出 post-hoc translation 修复。本文提供了更优雅的训练时解决方案，且给出了理论解释"为什么要缩 gap"
- **vs Yaras et al. (2025) 固定温度方案**：固定温度可以部分缩小 gap（0.12），但不如本方法彻底（0.03），且没有理论解释为什么有用
- **vs NotAGap (Fahim et al.)**：NotAGap 缩小 gap 但 CosTP 反而下降（0.11 vs 0.34），说明只缩 gap 不够——还需要同时对齐正对

## 评分
- 新颖性: ⭐⭐⭐⭐ "gap 伤害聚类但不伤害检索"的洞察是核心贡献，loss 设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 4 数据集、2/3 模态、4 类下游任务的全面验证，但缺少大规模预训练实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论→实验→可视化的叙事一气呵成，数学推导清晰
- 价值: ⭐⭐⭐⭐ 为理解和改善多模态潜在空间提供了重要的理论和实践工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Is the Modality Gap a Bug or a Feature? A Robustness Perspective](../../CVPR2026/multimodal_vlm/is_the_modality_gap_a_bug_or_a_feature_a_robustness_perspective.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](../../NeurIPS2025/multimodal_vlm/hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)
- [\[CVPR 2026\] Bridging the Modality Gap in Compositional Zero-Shot Learning via Sparse Alignment and Unimodal Memory Bank](../../CVPR2026/multimodal_vlm/bridging_the_modality_gap_in_compositional_zero-shot_learning_via_sparse_alignme.md)
- [\[ICLR 2026\] MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition](mora_missing_modality_low-rank_adaptation_for_visual_recognition.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](../../AAAI2026/multimodal_vlm/aligning_the_true_semantics_constrained_decoupling_and_distr.md)

</div>

<!-- RELATED:END -->
