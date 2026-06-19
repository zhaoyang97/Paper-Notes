---
title: >-
  [论文解读] Few-Shot Learner Generalizes Across AI-Generated Image Detection
description: >-
  [ICML 2025][目标检测][AI-generated image detection] 首次将 AI 生成图像检测重新定义为少样本分类任务，提出 FSD (Few-Shot Detector) 基于原型网络学习度量空间，仅用 10 个来自未见生成模型的样本，在 GenImage 数据集上平均准确率达 84.1%，超越此前 SOTA (LARE2) +11.6%。
tags:
  - "ICML 2025"
  - "目标检测"
  - "AI-generated image detection"
  - "few-shot learning"
  - "prototypical network"
  - "domain generalization"
  - "deepfake detection"
---

# Few-Shot Learner Generalizes Across AI-Generated Image Detection

**会议**: ICML 2025  
**arXiv**: [2501.08763](https://arxiv.org/abs/2501.08763)  
**代码**: [GitHub](https://github.com/teheperinko541/Few-Shot-AIGI-Detector)  
**领域**: 目标检测  
**关键词**: AI-generated image detection, few-shot learning, prototypical network, domain generalization, deepfake detection

## 一句话总结

首次将 AI 生成图像检测重新定义为少样本分类任务，提出 FSD (Few-Shot Detector) 基于原型网络学习度量空间，仅用 10 个来自未见生成模型的样本，在 GenImage 数据集上平均准确率达 84.1%，超越此前 SOTA (LARE2) +11.6%。

## 研究背景与动机

**领域现状**：随着 Stable Diffusion、Midjourney 等扩散模型的快速发展，生成高度逼真的图像变得极为容易。现有 AI 生成图像检测方法分为两类：(1) 模型无关方法——通过检测合成图像中的通用伪影（频率特征、纹理模式等）；(2) 扩散模型重建方法——利用扩散模型的重建误差区分真假图像。

**现有痛点**：两类方法面临共同问题。模型无关方法在 CNN 时代发现的通用伪影（如频谱异常）在扩散模型生成的图像中不再存在，导致跨模型泛化急剧下降。扩散模型重建方法（如 DIRE、LARE2）依赖特定扩散模型的重建能力，对训练中未见过的模型（尤其是使用不同架构如 VQDM 的模型）检测效果差。更根本的困难是，从闭源模型（DALL-E、Midjourney）收集大量训练数据昂贵或不可行。

**核心矛盾**：追求"万能检测指标"（一个分类器检测所有生成模型）忽略了不同生成模型产生的伪影特征差异巨大的事实。但为每个新模型重新训练分类器又需要大量数据——形成"通用性不足 vs 数据需求过大"的两难。

**本文目标** 利用少量（如 10 张）来自未见生成模型的样本，快速适配检测器以识别该模型的生成图像——将问题从不可能的"万能检测"转化为可行的"少样本适配"。

**切入角度**：作者观察到在实际场景中，获取少量未见模型的样本通常是可行的（从 API 生成几张图像成本极低）。关键洞察是将不同生成模型视为不同类别而非统一为"假"类——这样可以学习到各模型特有的伪影特征，而非追求不存在的"通用伪影"。

**核心 idea**：将 AI 生成图像检测重新定义为 N-way K-shot 分类任务，用原型网络在度量空间中实现无需重训练的快速适配。

## 方法详解

### 整体框架

FSD 基于原型网络 (Prototypical Network)。训练阶段：从已知生成模型的数据集中，通过 episode 训练学习一个度量空间——使同类图像在空间中聚集、异类图像远离。测试阶段：给定少量（K 个）未见模型的样本作为 support set，计算每类的原型表示（嵌入向量均值），然后对测试图像按最近邻原则分类。

### 关键设计

1. **多类而非二分类的检测范式**:

    - 功能：将不同生成模型的图像分为不同类别，真实图像为单独一类
    - 核心思路：传统方法将所有生成图像归为一个"假"类做二分类。FSD 将每个生成模型（Midjourney、GLIDE、ADM、Stable Diffusion 等）视为独立类别。训练时将 GenImage 数据集的 8 个子集按来源分为 7 个类（其中 SD v1.4、v1.5 和 Wukong 因共享相同架构合并为一类 SD），每个 episode 随机采样 $N_c$ 个类、每类 $N_s$ 个 support 样本和 $N_q$ 个 query 样本
    - 设计动机：t-SNE 可视化实验清楚表明，FSD 的多类训练使不同来源的图像在特征空间中形成独特的聚类（包括未见类），而二分类器的特征空间中未见类样本高度分散。这说明多类训练迫使模型学习类内共性特征，有助于对未见类的泛化

2. **原型网络的度量学习**:

    - 功能：学习一个度量空间使得分类可以通过简单的距离计算完成
    - 核心思路：使用 ResNet-50 (ImageNet 预训练) 作为 backbone $f_\phi: \mathbb{R}^D \to \mathbb{R}^{1024}$，将图像映射到 1024 维特征空间。每类的原型表示 $\mathbf{c}_i = \frac{1}{|S_i|} \sum_{\mathbf{x}_j \in S_i} f_\phi(\mathbf{x}_j)$。分类概率通过 softmax 归一化的负平方欧氏距离计算：$p(y=i|\mathbf{x}_q) = \text{Softmax}_{1 \leq i \leq N}(-d(f_\phi(\mathbf{x}_q), \mathbf{c}_i))$。训练目标是最小化 query 样本的负对数概率 $J(\phi) = -\frac{1}{N_c N_q} \sum_k \sum_{\mathbf{x}_j \in Q_k} \log \text{Softmax}(-d(f_\phi(\mathbf{x}_j), \mathbf{c}_k))$
    - 设计动机：原型网络的均值嵌入自然抑制单个样本的噪声——只要生成模型的伪影是系统性的（频率特征、纹理模式等），少量样本的均值就能有效捕捉类内共性

3. **零样本检测的元数据向量**:

    - 功能：在完全没有测试类样本时仍能进行检测
    - 核心思路：在训练集中，从每个类随机选取 1024 个样本计算其原型表示作为"元数据向量"。测试时，如果最近的原型属于某个生成模型类，则图像被判为假。这种零样本方法不需要任何未见类的样本
    - 设计动机：提供了无需任何适配开销的基线能力——即使在完全没有新模型样本的情况下，FSD 仍然可以工作（零样本平均准确率 77.1%）

### 损失函数

训练使用标准的 prototypical loss——即 query 样本到其正确原型的负对数 softmax 距离。优化器为 Adam，学习率 $10^{-4}$，StepLR 调度器（$\gamma=0.5$，step=80000），训练 200K 步，batch size 16。每步采样 3-way 5-shot 5-query 的 episode。

## 实验关键数据

### GenImage 基准 6 类检测准确率 (%)

| 方法 | Midjourney | GLIDE | ADM | SD | VQDM | BigGAN | 平均 |
|------|-----------|-------|-----|-----|------|--------|------|
| Spec | 50.0 | 64.7 | 52.8 | 56.1 | 56.5 | 63.0 | 57.2 |
| CNNSpot | 52.8 | 73.3 | 55.0 | 55.9 | 54.4 | 66.2 | 59.6 |
| DIRE | 57.9 | 68.2 | 57.3 | 58.2 | 59.6 | 50.8 | 58.7 |
| LARE2 | 62.7 | 80.2 | 63.5 | 79.6 | 76.9 | 72.0 | 72.5 |
| FSD (zero-shot) | 75.1 | 93.9 | 74.1 | 88.0 | 69.1 | 62.1 | 77.1 |
| **FSD (10-shot)** | **80.9** | **97.1** | **79.2** | **88.8** | 76.2 | **82.2** | **84.1** |

### 跨生成器 10-shot 分类 (Accuracy/AP, %)

| 排除子集 | Midjourney | GLIDE | ADM | SD | VQDM | BigGAN |
|---------|-----------|-------|-----|-----|------|--------|
| Midjourney | 80.9/84.6 | 99.9/99.9 | 98.5/99.3 | 97.1/98.7 | 99.5/99.9 | 88.0/92.9 |
| GLIDE | 86.8/89.9 | 97.1/98.0 | 97.9/98.9 | 97.1/98.8 | 99.2/99.7 | 91.9/97.1 |
| ADM | 87.6/91.8 | 99.8/99.9 | 79.2/83.8 | 94.8/97.2 | 98.8/99.4 | 91.0/96.1 |
| SD | 86.1/89.7 | 99.9/99.9 | 97.4/98.8 | 88.8/92.5 | 96.6/98.5 | 89.5/95.4 |
| VQDM | 82.4/85.9 | 99.9/99.9 | 97.3/98.6 | 95.6/98.0 | 76.2/79.4 | 83.5/89.1 |
| BigGAN | 88.9/91.6 | 99.9/99.9 | 98.3/99.3 | 98.1/99.3 | 96.4/98.3 | 82.2/86.8 |

### 关键发现

1. FSD 10-shot 平均准确率 84.1% 大幅超过 LARE2 的 72.5%（+11.6%），且仅需 10 个样本而非完整训练集
2. 从 1-shot 到 10-shot 提升巨大（如 ADM 从 62.6% 到 79.2%，+16.6%），但 10-shot 到 200-shot 仅提升 2.5%——10-shot 是性能-成本最优点
3. 零样本 FSD (77.1%) 已超越所有需要完整训练集的传统方法（LARE2 72.5%），说明多类度量学习本身已学到强大的特征表示
4. 对角线（未见类）准确率普遍 >76%，而训练中见过的类准确率通常 >95%——泛化 gap 存在但可控
5. VQDM 类在被排除时检测最困难（76.2%），因其图像量化机制与其他扩散模型差异最大
6. t-SNE 可视化证实未见类在 FSD 特征空间中仍形成紧致聚类，而在二分类器中高度分散

## 亮点与洞察

1. **问题重定义切中要害**：从"不可能的万能检测"转为"可行的少样本适配"，这一视角转换本身就是重要贡献
2. **方法极简效果极好**：原型网络是最基础的度量学习方法之一，但在此问题上效果惊人——说明问题定义比方法复杂度更重要
3. **10-shot 在实际中完全可行**：从 Midjourney、DALL-E 等 API 生成 10 张图像的成本几乎为零
4. **多类 vs 二分类的洞察**：分源分类迫使模型关注类内共性而非类间对比，学到了更结构化的特征表示

## 局限性

- 需要知道或假设测试图来自哪个特定生成模型才能构建准确的 support set——在完全未知来源的场景下退化为零样本
- 对 VQDM 等使用根本不同技术路线的模型泛化有限（76.2% vs 其他类 >80%），说明当模型差异过大时少样本适配仍有挑战
- 原型网络通过均值计算原型表示，对 support set 中的异常样本敏感
- 未评估对抗性扰动（故意逃避检测）下的鲁棒性
- 仅在 GenImage 基准上验证，缺少更多测试集的交叉验证

## 相关工作与启发

- 与 UnivFD (Ojha et al., 2023) 使用 CLIP 语义特征的方法互补——FSD 基于 ResNet 提取低层次伪影特征
- 与 LARE2 (Luo et al., 2024) 依赖扩散模型重建的方向不同——FSD 完全数据驱动，不依赖任何特定模型
- 启发：元学习框架可替换 backbone（如用 CLIP 或 DINOv2 提取特征）进一步提升性能

## 评分

⭐⭐⭐⭐⭐ 问题重定义精准（首次将 AIGI 检测转化为少样本分类）、方法优雅（原型网络 + 多类训练）、实验充分（+11.6% SOTA）、实用性强（10-shot 成本近零）。缺少更多 benchmark 和对抗鲁棒性评估是小遗憾，但不影响整体的高质量。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] BlueGlass: A Framework for Composite AI Safety](blueglass_a_framework_for_composite_ai_safety.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](../../ICLR2026/object_detection/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/object_detection/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[ECCV 2024\] OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](../../ECCV2024/object_detection/openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/object_detection/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)

</div>

<!-- RELATED:END -->
