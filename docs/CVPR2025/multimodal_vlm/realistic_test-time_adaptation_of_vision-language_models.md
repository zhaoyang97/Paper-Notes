---
title: >-
  [论文解读] Realistic Test-Time Adaptation of Vision-Language Models
description: >-
  [CVPR 2025][多模态VLM][测试时适应] 本文揭示现有VLM测试时适应（TTA）/转导方法在realistic场景下（有效类数可变、非i.i.d.数据流）会严重损害CLIP的零样本鲁棒性，并提出StatA方法，通过在高斯聚类模型参数上引入基于文本编码器知识的KL散度正则化（统计锚），在所有部署场景中保持稳定提升。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "测试时适应"
  - "视觉语言模型"
  - "转导学习"
  - "统计锚"
  - "零样本分类"
---

# Realistic Test-Time Adaptation of Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2501.03729](https://arxiv.org/abs/2501.03729)  
**代码**: [https://github.com/MaxZanella/StatA](https://github.com/MaxZanella/StatA)  
**领域**: 多模态VLM  
**关键词**: 测试时适应, 视觉语言模型, 转导学习, 统计锚, 零样本分类

## 一句话总结

本文揭示现有VLM测试时适应（TTA）/转导方法在realistic场景下（有效类数可变、非i.i.d.数据流）会严重损害CLIP的零样本鲁棒性，并提出StatA方法，通过在高斯聚类模型参数上引入基于文本编码器知识的KL散度正则化（统计锚），在所有部署场景中保持稳定提升。

## 研究背景与动机

VLM（如CLIP）的零样本能力使其无需标注数据即可做分类，近年的TTA方法（转导推理、在线适应）进一步提升了性能。然而，现有方法都建立在**不现实的假设**之上：(1) 批次中包含所有类别且均匀分布；(2) 数据流是i.i.d.的。而真实部署中，卫星图像的patch可能只包含少数类别，视频帧中的样本高度相关。核心矛盾：**现有方法在有利假设下的性能增益，是以牺牲其他场景的零样本鲁棒性为代价的**。TransCLIP在Very Low有效类数场景下掉26.3%，ZLaP掉37.8%。本文切入角度：同时约束赋值变量和统计参数（而非只约束赋值），利用文本编码器知识做"锚定"。核心idea：用文本嵌入作为统计锚，在低数据量时保持模型参数接近文本先验。

## 方法详解

### 整体框架

StatA属于软概率聚类方法家族。给定CLIP的视觉特征 $\mathbf{f}_i$ 和文本嵌入 $\mathbf{t}_k$，方法交替优化两组变量：(1) 赋值向量 $\mathbf{z}_i$（样本属于各类的概率）；(2) 每个类的多元高斯模型参数 $(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。关键在于在标准MLE目标基础上增加一个StatA正则项，用KL散度约束模型参数不偏离文本编码器给出的先验锚。

### 关键设计

1. **Statistical Anchor (StatA) 正则项**:
    - 功能：防止模型参数在少数类别或低数据情况下偏离文本先验
    - 核心思路：为每个类别 $k$ 构建锚分布 $\mathcal{N}'_k = \mathcal{N}(\boldsymbol{\mu}'_k, \boldsymbol{\Sigma}')$，其中均值锚 $\boldsymbol{\mu}'_k = \mathbf{t}_k$（文本嵌入），协方差锚 $\boldsymbol{\Sigma}'$ 由零样本预测加权的视觉特征方差算出。然后通过 $\text{KL}(\mathcal{N}'_k || \mathcal{N}_k)$ 惩罚模型参数偏离锚
    - 设计动机：现有方法（PADDLE、Dirichlet、TransCLIP）只正则化赋值变量 $\mathbf{z}$，不约束模型参数 $\mathbf{M}$。但VLM的文本编码器天然提供了各类别原型的先验知识，不利用是浪费

2. **自适应凸组合更新**:
    - 功能：闭式更新 $\boldsymbol{\mu}_k$ 和 $\boldsymbol{\Sigma}_k$，自动在MLE估计和文本锚之间权衡
    - 核心思路：$\boldsymbol{\mu}_k = \beta_k \mathbf{v}_k + (1-\beta_k) \boldsymbol{\mu}'_k$，其中 $\beta_k = \frac{n_k}{n_k + \alpha}$，$n_k$ 是类 $k$ 的预测样本数。样本越多越信任MLE，样本越少越信任文本锚
    - 设计动机：当某类只有极少甚至零个样本时，MLE估计不可靠，此时应回退到文本先验。$\alpha=1$ 在所有实验中无需调参即可工作

3. **硬分配的 $\beta_k$ 计算**:
    - 功能：更鲁棒地估计每个类的预测样本数
    - 核心思路：将 $\beta_k$ 中的软赋值 $z_{i,k}$ 替换为硬分配 $\mathbb{1}[k = \arg\max_r z_{i,r}]$，避免软概率中残差分量引入的噪声
    - 设计动机：实验发现硬分配版本在各场景下更稳定，特别是在高有效类数场景

### 损失函数 / 训练策略

总目标函数：$\mathcal{L}_\mathcal{A}(\mathbf{z}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \mathcal{L}_{\text{MLE}}(\mathbf{z}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) + \alpha \sum_{k=1}^K \text{KL}(\mathcal{N}'_k || \mathcal{N}_k)$

其中 $\mathcal{L}_{\text{MLE}}$ 是标准的高斯混合模型对数似然目标。优化采用块坐标下降：先固定参数更新 $\mathbf{z}$，再固定 $\mathbf{z}$ 闭式更新参数。初始化用零样本softmax预测。StatA是训练无关的（training-free），只需在测试时运行几轮迭代。

## 实验关键数据

### 主实验（Batch Size=64, 11个数据集平均）

| 方法 | Very Low (1-4类) | Low (2-10类) | Medium (5-25类) | All |
|------|-----------------|-------------|----------------|-----|
| CLIP (zero-shot) | 65.2 | 65.2 | 65.2 | 65.2 |
| Dirichlet | **68.5** (+3.3) | **70.3** (+5.1) | 67.5 (+2.2) | 59.2 (-6.0) |
| ZLaP | 27.5 (-37.8) | 35.2 (-30.0) | 44.7 (-20.6) | 65.5 (+0.3) |
| TransCLIP | 38.9 (-26.3) | 40.4 (-24.8) | 42.7 (-22.5) | 66.1 (+0.9) |
| **StatA** | **70.4** (+5.1) | 69.3 (+4.1) | **67.4** (+2.2) | **66.5** (+1.3) |

StatA是**唯一在所有场景下都稳定正增益**的方法。

### 在线适应实验

| 方法 | Low相关性 | High相关性 | 类别分离 |
|------|----------|-----------|---------|
| CLIP | 65.2 | 65.2 | 65.2 |
| MTA | +1.3 | +1.3 | +1.3 |
| TDA | +1.7 | -0.3 | -1.3 |
| DMN-ZS | +2.3 | +0.2 | -2.9 |
| **StatA** | **+3.7** | **+2.9** | **+2.6** |

### 关键发现

- 现有转导方法（ZLaP、TransCLIP）在低有效类数场景下**崩溃式下降**（-20~-38%），本质是因为MLE的类平衡偏差
- Dirichlet在低类数场景强但在All类数场景下降6%，因为其MDL正则偏向少类
- StatA的 $\alpha=1$ 无需调参，是唯一一个在Very Low到All全部场景下都正增益的方法
- StatA处理数千样本仅需几秒，计算效率极高

## 亮点与洞察

- **抓住了真实部署的核心痛点**：现有TTA方法在理想分布假设下表现亮眼，但换个部署场景就崩溃，这是之前被忽视的重要问题
- **优雅的数学框架**：StatA的凸组合更新有清晰的直觉解释——"数据多就信数据，数据少就信文本先验"
- **Black-box兼容**：只需访问特征空间，不需模型内部参数，可通过API部署

## 局限与展望

- 实验仅限图像分类任务，未扩展到检测、分割等其他vision任务
- 锚分布使用共享对角协方差矩阵，可能限制了表达能力
- 在All类场景下提升有限（+1.3%），说明当数据充足时锚的约束作用减弱
- 未考虑标签空间随时间变化的开放世界场景

## 相关工作与启发

- **vs TransCLIP**: TransCLIP用KL散度约束赋值变量，StatA用KL约束模型参数——从"正则化预测"转向"正则化统计量"
- **vs EM-Dirichlet**: Dirichlet的MDL正则偏向少类，在All场景崩溃。StatA通过自适应 $\beta_k$ 避免了此偏差
- **vs TDA/DMN-ZS**: 这些在线方法构建memory bank，依赖均匀数据流，非i.i.d.时性能急剧下降

## 评分

- 新颖性: ⭐⭐⭐⭐ 从"正则化赋值"到"正则化模型参数"的视角转换新颖，StatA设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 6种有效类数x batch/online/stream设置x11数据集，评估极其全面
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，但LaTeX符号密集需要仔细阅读
- 价值: ⭐⭐⭐⭐ 揭示了TTA方法的"虚假繁荣"，为realistic部署提供了可靠基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM](free_on_the_fly_enhancing_flexibility_in_test-time_adaptation_with_online_em.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](../../NeurIPS2025/multimodal_vlm/mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)

</div>

<!-- RELATED:END -->
