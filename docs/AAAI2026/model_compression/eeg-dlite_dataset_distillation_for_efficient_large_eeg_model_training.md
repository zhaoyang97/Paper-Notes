---
title: >-
  [论文解读] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training
description: >-
  [AAAI 2026][模型压缩][数据蒸馏] 提出 EEG-DLite 数据蒸馏框架，通过自监督编码+异常值过滤+多样性采样，将2500小时 EEG 数据集压缩至仅5%即可达到甚至超越全数据集预训练的基础模型性能，GPU预训练时间从30小时降至2小时。 领域现状 大规模 EEG 基础模型（如 LaBraM）通过在大量无标签…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "数据蒸馏"
  - "EEG基础模型"
  - "自监督学习"
  - "核心集选择"
  - "预训练效率"
---

# EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training

**会议**: AAAI 2026  
**arXiv**: [2512.12210](https://arxiv.org/abs/2512.12210)  
**代码**: [github](https://github.com/t170815518/EEG-DLite)  
**领域**: 模型压缩  
**关键词**: 数据蒸馏, EEG基础模型, 自监督学习, 核心集选择, 预训练效率

## 一句话总结
提出 EEG-DLite 数据蒸馏框架，通过自监督编码+异常值过滤+多样性采样，将2500小时 EEG 数据集压缩至仅5%即可达到甚至超越全数据集预训练的基础模型性能，GPU预训练时间从30小时降至2小时。

## 研究背景与动机

### 领域现状
大规模 EEG 基础模型（如 LaBraM）通过在大量无标签 EEG 数据上进行自监督预训练，已在情感识别、运动想象、临床分类等多种下游任务上展现出强大的泛化能力。这类模型通常基于 Transformer 架构，参数量可超过4亿，预训练数据量达2500小时以上。

### 现有痛点

**训练代价极高**：预训练需要大量 GPU 时间（30小时/4×RTX 4090）和存储资源，限制了参数优化和架构搜索的可行性

**EEG 信号特殊性**：EEG 信号具有极低的信噪比（SNR），容易被眼动、肌电等伪影干扰；且时间上相邻片段高度冗余

**数据质量被忽视**：现有工作聚焦架构创新和迁移学习，很少研究预训练数据的组成和质量如何影响模型泛化

### 核心矛盾
大规模 EEG 数据中存在大量噪声样本和冗余样本，但现有基础模型仍需在全量数据上预训练，导致资源严重浪费。

### 切入角度
受CV领域数据蒸馏方法启发，但针对 EEG 信号的低信噪比和高冗余特性设计专用的蒸馏策略。核心 idea 是：先用自监督自编码器将高维 EEG 压缩到低维潜在空间，再在该空间中高效地去噪和去冗余。

## 方法详解

### 整体框架
EEG-DLite 是一个三阶段的数据蒸馏 pipeline，与具体的基础模型架构解耦：
1. **多视角神经压缩器**：用自监督自编码器将 EEG 片段编码为紧凑的低维表示
2. **异常值去除**：基于 HBOS 方法在潜在空间过滤噪声/伪影样本
3. **多样性采样**：用 k-center 贪心算法选择最具代表性的子集

### 关键设计

1. **多视角自监督自编码器（Multi-view Neural Compressor）**:

    - **功能**：将高维 EEG 信号 $X \in \mathbb{R}^{C \times T}$ 压缩为64维潜在表示 $\mathbf{z}$
    - **核心思路**：对每个 EEG 片段同时计算原始信号、FFT幅值和FFT相位三个视角，将三个视角拼接后通过 CNN 提取 patch-level token，再输入 Transformer 编码器捕获全局依赖
    - **编码器**：6层自注意力层，8个注意力头；**解码器**：2层 Transformer + MLP
    - **设计动机**：直接在原始 EEG 空间做样本选择对噪声过于敏感，且维度太高计算量大；而频谱视角可提供互补的信号质量信息

2. **自监督训练目标**:

    - **重构损失**：$\mathcal{L}_{Rec} = \sum_{i=1}^{L}(\mathbf{x}'_i - \mathbf{x}_i)^2$，确保编码器能准确编码神经信号内容
    - **实例间判别损失（IDC）**：$\mathcal{L}_{IDC}$ 惩罚批次内不同样本 token 之间的余弦相似度过高，鼓励特征多样性
    - **联合目标**：$\mathcal{L} = \mathcal{L}_{Rec} + \beta \cdot \mathcal{L}_{IDC}$，其中 $\beta=0.0001$
    - **设计动机**：IDC 损失使得编码器学到更具区分力的表示，便于后续的异常值检测和多样性采样

3. **异常值去除（Outlier Sample Removal）**:

    - **功能**：基于 HBOS 方法计算每个样本的 OOD 得分，去除得分最高的 $\tau\%$ 样本
    - **OOD 得分**：$\text{OOD}(X) = \sum_{i=1}^{d} \log \frac{1}{p_i(x_i) + \alpha}$，其中 $p_i$ 是第 $i$ 个特征维度的直方图概率
    - **设计动机**：EEG 数据中常含有伪影污染的低质量片段，这些片段会降低后续多样性采样的质量；在潜在空间做异常检测比在原始空间更稳健

4. **多样性采样（Diversity Sampling）**:

    - **功能**：从去噪后的数据中选取 $\eta\%$ 最具代表性的样本
    - **核心公式**：$\min_{\boldsymbol{\mu} \subset \mathcal{Z}'} \max_{\mathbf{z} \in \mathcal{Z}'} \min_{k \in \mathcal{K}} \|\mathbf{z} - \boldsymbol{\mu}_k\|_2^2$（k-center 问题）
    - **实现**：采用贪心近似算法，迭代选择与已选点距离最远的样本，时间复杂度 $\mathcal{O}(k \times N \times d)$
    - **设计动机**：确保选出的子集能覆盖原始数据的分布多样性，而非简单的随机采样

### 训练策略
- 压缩器训练：Adam 优化器，学习率0.001，50 epochs，梯度裁剪 max norm=5.0
- 学习率调度：每10个 epoch 衰减0.5倍
- EEG 片段分割为20个不重叠 patch

## 实验关键数据

### 主实验

实验基于 LaBraM-base 架构，在4个下游任务上评估不同蒸馏方法和蒸馏比例：

| 数据集 | 指标 | EEG-DLite (5%) | Random (5%) | Full (100%) | 说明 |
|--------|------|----------------|-------------|-------------|------|
| SEED-V | Accuracy | **38.6** | 34.6 | 41.0 | 5分类情感识别 |
| SEED-V | F1 | **38.9** | 34.9 | 41.2 | EEG-DLite远超Random |
| MoBI | PCC | **0.550** | 0.530 | 0.538 | 回归任务，超越全数据 |
| MoBI | R² | **0.283** | 0.260 | 0.288 | 接近全数据性能 |
| TUEV | Balanced Acc | **62.9** | 62.3 | 64.1 | 6分类EEG事件检测 |
| TUEV | F1 | **80.7** | 79.3 | 83.1 | 仅用5%数据 |
| TUAB | Balanced Acc | **80.7** | 80.7 | 81.4 | 二分类正常/异常EEG |
| TUAB | AUROC | **90.3** | 90.0 | 90.2 | 超越全数据集 |

**核心发现**：仅用5%数据蒸馏子集预训练即可达到接近甚至超越全量数据的性能。

### 消融实验

| 配置 | SEED-V Acc (η=5%) | MoBI PCC (η=5%) | 说明 |
|------|-------------------|-----------------|------|
| Random baseline | 34.6 | 0.530 | 随机采样下界 |
| PCA + Diversity Sampling | 31.0 | 0.534 | PCA降维不如SSL |
| M3D (生成式) | 26.9 | 0.465 | 生成式方法效果差 |
| EEG-DLite (τ=0, 无OOD去除) | 未报告 | 未报告 | OOD去除有帮助 |
| **EEG-DLite (完整)** | **38.6** | **0.550** | 全部组件最优 |

**OOD去除消融**（SEED数据集，EEGNet监督学习）：

| 配置 | η(%) | Acc | F1 | κ |
|------|-------|-----|-----|---|
| 完整方法 τ=0 | 25 | 54.6 | 55.1 | 29.1 |
| 完整方法 τ=1% | 25 | **55.3** | **55.7** | **31.3** |
| Full Data | 100 | 54.6 | 55.4 | 30.8 |

### 关键发现
1. **5%即够**：仅用5%数据即可达到全数据集的性能水平，说明大规模 EEG 数据中存在大量冗余
2. **生成式方法失败**：M3D 生成的 EEG 合成样本质量极差（出现不自然的平台、方块状模式），在所有比例下均不如随机采样
3. **SSL优于PCA**：自监督学习生成的表示比PCA降维更稳定、更具判别力
4. **蒸馏后甚至更优**：在 TUEV、MoBI、TUAB 等数据集上，蒸馏子集训练的模型**超越**全量数据训练的模型
5. **受试者方差增大**：多样性采样后，不同受试者的样本贡献比例出现显著差异，反映了 EEG 信号的个体差异性
6. **时间大幅缩短**：GPU 预训练时间从30小时降至2小时（4×RTX 4090）

## 亮点与洞察
1. **首个 EEG 基础模型数据蒸馏工作**：填补了生理信号预训练数据效率的研究空白
2. **框架设计优雅**：三阶段设计（压缩→去噪→去冗余）逻辑清晰，且与下游基础模型架构解耦
3. **多视角编码**：同时利用时域和频域信息，充分利用 EEG 信号的频谱特性
4. **实用价值高**：5%数据 → 15x训练加速，对资源受限场景意义重大
5. **反直觉发现**：精心选择的小数据集可以比大数据集训练出更好的模型

## 局限与展望
1. 仅在 LaBraM 一个基础模型上验证，需扩展到其他 EEG 架构
2. 蒸馏过程本身也需要在全量数据上训练自编码器，存在一定的前期开销
3. 受试者感知的采样策略尚未被探索，可能进一步提升跨受试者泛化
4. 未探索不同预训练目标（如对比学习 vs 掩码预测）对蒸馏子集选择的影响
5. 生成式方法在 EEG 上的失败值得更深入的分析和专门设计

## 相关工作与启发
- **核心集选择**（Sener & Savarese 2018）：k-center 贪心算法是经典方法，本文将其适配到 EEG 潜在空间
- **M3D**（Zhang et al. 2024）：CV领域的轻量生成式蒸馏方法，但在 EEG 上完全失败，说明生理信号与视觉数据本质不同
- **LaBraM**（Jiang et al. 2024）：本文的基线基础模型，该工作首次展示了大规模 EEG 预训练的有效性
- **启发**：该框架可推广到其他生理信号（EMG, ECG）的基础模型预训练数据优化

## 评分
- 新颖性: ⭐⭐⭐⭐ （首个 EEG 数据蒸馏工作，但方法组件本身并不全新）
- 实验充分度: ⭐⭐⭐⭐⭐ （4个下游任务，多种蒸馏比例，充分的消融和对比）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富）
- 价值: ⭐⭐⭐⭐⭐ （实用意义大，15x训练加速且不损性能）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](post_training_quantization_for_efficient_dataset_condensation.md)
- [\[NeurIPS 2025\] BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG](../../NeurIPS2025/model_compression/barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[ICLR 2026\] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](../../ICLR2026/model_compression/paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)
- [\[ICML 2026\] Unifying Dataset Pruning and Distillation for Efficient Large-scale Compression](../../ICML2026/model_compression/unifying_dataset_pruning_and_distillation_for_efficient_large-scale_compression.md)

</div>

<!-- RELATED:END -->
