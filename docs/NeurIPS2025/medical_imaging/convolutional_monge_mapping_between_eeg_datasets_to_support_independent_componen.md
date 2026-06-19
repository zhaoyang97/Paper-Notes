---
title: >-
  [论文解读] Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling
description: >-
  [NeurIPS 2025][医学图像][EEG] 本文扩展 CMMN（Convolutional Monge Mapping Normalization）方法，提出通道平均 PSD + $\ell_1$ 归一化质心和 subject-to-subject 匹配两种策略，生成单一时域滤波器实现不同通道数的 EEG 数据集间域适应，在独立成分（IC）脑/非脑分类中 F1 从 0.77 提升至 0.84，超越 ICLabel（0.88→0.91）。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "EEG"
  - "域适应"
  - "最优传输"
  - "Convolutional Monge Mapping"
  - "独立成分分类"
---

# Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling

**会议**: NeurIPS 2025  
**arXiv**: [2509.01721](https://arxiv.org/abs/2509.01721)  
**代码**: [https://github.com/cniel-ud/ICWaves](https://github.com/cniel-ud/ICWaves)  
**领域**: EEG信号处理 / 域适应  
**关键词**: EEG, 域适应, 最优传输, Convolutional Monge Mapping, 独立成分分类

## 一句话总结
本文扩展 CMMN（Convolutional Monge Mapping Normalization）方法，提出通道平均 PSD + $\ell_1$ 归一化质心和 subject-to-subject 匹配两种策略，生成单一时域滤波器实现不同通道数的 EEG 数据集间域适应，在独立成分（IC）脑/非脑分类中 F1 从 0.77 提升至 0.84，超越 ICLabel（0.88→0.91）。

## 研究背景与动机

**领域现状**：EEG 记录包含丰富的神经活动信息，广泛用于癫痫和精神疾病诊断。独立成分分析（ICA）+ 自动 IC 标注是伪迹去除的主流方法，ICLabel 是目前最流行的 IC 分类器。

**现有痛点**：不同 EEG 采集系统（电极、放大器、模拟/数字滤波器、电网干扰）导致记录的频谱特性差异极大——如美国数据有 60Hz 工频噪声、欧洲数据有 50Hz 噪声。这种域差异严重影响跨数据集的 IC 分类性能。且不同数据集通道数不同（134~235 vs 64），传统 CMMN 为每个通道计算独立滤波器，无法跨通道数适配。

**核心矛盾**：原始 CMMN 为每个通道设计独立滤波器，但 IC 是通道的线性混合——对不同通道施加不同滤波器会改变 IC 的特性。更根本的问题是，源域和目标域通道数不同时，原始 CMMN 根本无法使用。

**本文目标** 设计一种单一滤波器的 CMMN 变体，(a) 适配不同通道数的 EEG 数据集，(b) 保持 IC 特性不变（因为所有通道共享同一滤波器）。

**切入角度**：用通道平均 PSD 代替逐通道 PSD，生成每个被试一个公共滤波器。结合 $\ell_1$ 归一化消除信号幅度差异（阻抗、电极差异），使频谱形状对齐。

**核心 idea**：通道平均 PSD + $\ell_1$ 归一化质心的 CMMN 滤波器实现跨通道数、跨采集系统的 EEG 域适应。

## 方法详解

### 整体框架
**输入**：源域 EEG（训练集，$I$ 个被试，各有 $C^S$ 通道）+ 目标域 EEG（测试集，$C^T$ 通道，$C^S \neq C^T$）。**处理**：为每个目标被试计算一个 CMMN 归一化滤波器 $h[n]$，对所有通道统一滤波。**输出**：频谱对齐后的目标 EEG，直接送入源域训练的分类器。

### 关键设计

1. **通道平均 PSD 计算（Step 1）**：

    - 功能：将多通道 EEG 的频谱信息压缩为一个单一 PSD
    - 核心思路：对每个被试的所有通道分别用 Welch 方法计算 PSD $\mathbf{p}_c$，然后取通道平均 $\bar{\mathbf{p}} = \frac{1}{C} \sum_{c=1}^{C} \mathbf{p}_c$。这样无论源域和目标域有多少通道，都产生相同维度的 PSD 向量
    - 设计动机：原始 CMMN 为每个通道设计独立滤波器，通道数不同时无法对应。通道平均使得单一滤波器可应用于任意通道数，且不会差异化改变各 IC

2. **$\ell_1$ 归一化质心（Step 2a）**：

    - 功能：计算源域被试的参考频谱，消除幅度偏差
    - 核心思路：对每个源域被试的通道平均 PSD 做 $\ell_1$ 归一化 $\tilde{\mathbf{p}}_i^S = \bar{\mathbf{p}}_i^S / \|\bar{\mathbf{p}}_i^S\|_1$，然后取均值 $\tilde{\mathbf{p}}_S = \frac{1}{I} \sum_{i=1}^I \tilde{\mathbf{p}}_i^S$ 作为质心。$\ell_1$ 归一化后 PSD 成为概率质量函数，每个被试贡献等权
    - 设计动机：PSD 单位是幅度平方，不归一化时高阻抗被试会主导均值。$\ell_1$ 归一化确保频谱形状相同的被试贡献相等

3. **Subject-to-Subject 匹配（Step 2b）**：

    - 功能：为每个目标被试找到频谱最接近的源域被试作为映射目标
    - 核心思路：计算目标被试和每个源域被试的 $\ell_1$ 归一化 PSD 之间的 Hellinger 距离 $d_{\text{He}}(\tilde{\mathbf{p}}_i^S, \tilde{\mathbf{p}}^T) = \frac{1}{\sqrt{2}} \|\sqrt{\tilde{\mathbf{p}}_i^S} - \sqrt{\tilde{\mathbf{p}}^T}\|_2$，选择最近邻 $i^* = \arg\min_i d_{\text{He}}$。Hellinger 距离等价于方差归一化 Gaussian 过程间的 Wasserstein-2 距离
    - 设计动机：质心方案对所有被试映射到同一参考，可能损失个体特异性。Subject-to-subject 映射保留更多源-目标匹配的细节信息

4. **归一化滤波器构建（Step 3）**：

    - 功能：计算将目标频谱映射到源参考频谱的线性滤波器
    - 核心思路：频率响应为 PSD 比值的平方根 $H[n] = \sqrt{\bar{p}^S[n] / \bar{p}^T[n]}$，时域冲激响应通过 IRFFT 获得：$\mathbf{h} = \text{IRFFT}_M(\mathbf{H})$。这是零相位线性滤波器，解决了源、目标高斯分布间的最优传输问题
    - 设计动机：滤波器直接均衡通道平均 PSD，使滤波后的目标信号频谱与源域参考对齐。滤波在 ICA 解混前后均可应用（因为是所有通道共用的时域滤波器）

### 损失函数 / 训练策略
- 该方法无需训练——滤波器直接从统计量（PSD）闭式计算
- 下游分类器（随机森林）用 PSD + 自相关特征训练，超参数通过 leave-one-subject-out 交叉验证选择
- 分段长度 $l_{\text{train}}$ 作为超参数，验证/测试使用 5 分钟和 50 分钟两种长度

## 实验关键数据

### 主实验 — 跨数据集 IC 分类 (Brain class F1)

| 分类器 | 段长 | 无滤波 | Barycenter | $\ell_1$ 归一化 Bary. | Subj-to-subj | p-value |
|--------|------|--------|------------|----------------------|--------------|---------|
| PSD/Autocorr | 5min | 0.77±0.09 | 0.78±0.12 | **0.84±0.07** | 0.79±0.17 | 0.0046 |
| ICLabel | 5min | 0.88±0.06 | — | — | — | — |
| PSD/Autocorr | 50min | 0.83±0.09 | 0.86±0.09 | **0.86±0.08** | 0.85±0.17 | 0.1696 |
| ICLabel | 50min | 0.89±0.05 | — | — | — | — |

### 消融实验 — 域内性能

| 分类器 | 5min | 50min | 说明 |
|--------|------|-------|------|
| PSD/Autocorr | **0.93±0.05** | **0.96±0.05** | 域内最优 |
| ICLabel | 0.88±0.05 | 0.89±0.07 | 通用基线 |

### 不同 CMMN 方案的对比

| 方案 | 5min F1 | 50min F1 | 说明 |
|------|---------|----------|------|
| 无滤波（baseline） | 0.77 | 0.83 | 域偏移严重影响性能 |
| 标准 Barycenter | 0.78 | 0.86 | 未归一化质心，幅度偏差影响 |
| $\ell_1$-norm Barycenter | **0.84** | **0.86** | 最佳方案，统计显著改善 |
| Subj-to-subj | 0.79 | 0.85 | 个体匹配，方差较大 |

### 关键发现
- $\ell_1$ 归一化质心是最稳定的方案：5min 段 F1 从 0.77→0.84（p=0.0046，Wilcoxon 检验显著）
- 学到的滤波器直觉上合理：将 50Hz 噪声衰减、60Hz 噪声放大（欧洲→美国映射）
- 通道平均 CMMN + PSD/Autocorr 分类器（F1=0.91 域内，0.84-0.86 跨域）在数据量有限时优于 ICLabel（0.88-0.89）
- Subject-to-subject 方案方差较大（±0.17），说明匹配质量不稳定，质心方案更鲁棒
- 50min 段上改善不显著（p=0.17），说明足够长的数据能部分缓解域偏移

## 亮点与洞察
- **通道平均的关键作用**：一个极其简单的修改（通道平均而非逐通道）解决了跨通道数域适应的根本障碍，且保证了 IC 特性不变——简洁优雅
- **$\ell_1$ 归一化消除幅度偏差**：PSD 是幅度平方，不归一化时异常值主导质心。$\ell_1$ 归一化将 PSD 转化为 PMF，使 Hellinger 距离等价于 Wasserstein-2 距离，理论优美
- **零训练域适应**：无需任何训练过程，滤波器从 PSD 闭式计算，适合临床快速部署
- **可迁移**：通道平均 CMMN 思路可推广到任何多通道生理信号（EMG、MEG）的跨设备域适应

## 局限与展望
- 仅在二分类（脑 vs 非脑 IC）上验证，未测试更细粒度的多类 IC 分类
- 实验规模较小：源域 27+7 被试，目标域仅 12 被试
- 通道平均假设所有通道频谱可比——对于空间分布差异极大的少通道蒙太奇可能不成立
- 仅用简单随机森林分类器，未探索与深度学习模型结合的效果
- Subj-to-subj 方案方差大，可探索 top-K 加权匹配或聚类匹配降低不稳定性

## 相关工作与启发
- **vs 原始 CMMN (Gnassounou 2023)**：原版为每通道建独立滤波器、用于睡眠分期；本文通道平均+$\ell_1$ 归一化，扩展到跨通道数场景+IC 分类
- **vs ICLabel**：ICLabel 在海量数据上训练，使用空间+频谱特征；本文仅用时序特征+CMMN 域适应，在小数据场景下超越 ICLabel
- **与最优传输域适应的关系**：CMMN 本质上是用 Wasserstein-2 barycenter 做频谱域的最优传输映射，为 1D 时序信号的域适应提供了优雅的理论框架

## 评分
- 新颖性: ⭐⭐⭐ 是对已有 CMMN 方法的增量扩展，通道平均和 $\ell_1$ 归一化思路简单
- 实验充分度: ⭐⭐⭐ 实验规模小（2 个数据集、44 被试），统计检验仅 5min 段显著
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导完整，但作为 workshop paper 篇幅受限
- 价值: ⭐⭐⭐⭐ 解决了跨通道数 EEG 域适应的实际问题，临床部署价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] NOIR: Neural Operator Mapping for Implicit Representations](../../CVPR2025/medical_imaging/noir_neural_operator_mapping_for_implicit_representations.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[ICLR 2026\] HEEGNet: Hyperbolic Embeddings for EEG](../../ICLR2026/medical_imaging/heegnet_hyperbolic_embeddings_for_eeg.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](../../ICCV2025/medical_imaging/radgpt_constructing_3d_image-text_tumor_datasets.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)

</div>

<!-- RELATED:END -->
