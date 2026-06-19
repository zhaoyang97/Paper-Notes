---
title: >-
  [论文解读] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts
description: >-
  [ICCV 2025][目标检测][长尾异常检测] 本文提出长尾在线异常检测（LTOAD）新任务和benchmark，核心创新是用可学习的"类无关概念集"替代传统的类标签依赖，结合Concept VQ-VAE和综合prompt学习框架，在不需要类标签的情况下于offline和online场景下均达到SOTA。
tags:
  - "ICCV 2025"
  - "目标检测"
  - "长尾异常检测"
  - "在线学习"
  - "类无关概念"
  - "VQ-VAE"
  - "提示学习"
---

# Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts

**会议**: ICCV 2025  
**arXiv**: [2507.16946](https://arxiv.org/abs/2507.16946)  
**代码**: [https://doi.org/10.5281/zenodo.16283852](https://doi.org/10.5281/zenodo.16283852) (benchmark)  
**领域**: 医学图像 / 异常检测  
**关键词**: 长尾异常检测, 在线学习, 类无关概念, VQ-VAE, prompt learning

## 一句话总结
本文提出长尾在线异常检测（LTOAD）新任务和benchmark，核心创新是用可学习的"类无关概念集"替代传统的类标签依赖，结合Concept VQ-VAE和综合prompt学习框架，在不需要类标签的情况下于offline和online场景下均达到SOTA。

## 研究背景与动机

**领域现状**：异常检测（AD）旨在识别图像中的缺陷区域，广泛应用于工业制造和医学诊断。近年来研究了多种设置：无监督one-class AD、统一AD（一个模型处理所有类别）、长尾分布AD（LTAD）、以及在线AD。

**现有痛点**：LTAD方法是**类感知**的（class-aware），需要输入图像的类标签来选择对应的类专用模块，但在线流式数据场景中类标签通常不可用。现有在线AD方法没有考虑长尾分布。LTAD的架构没有利用最新的VQ-VAE技术，且prompt learning设计不完善。

**核心矛盾**：长尾分布 + 在线学习 + 无类标签，三者同时存在时现有方法无法工作。类感知方法用硬切换（hard switching），遇到未见类别时完全失败。

**本文目标** 定义LTOAD任务，设计类无关框架使LTAD方法能在在线场景中工作，并提出异常自适应在线学习算法。

**切入角度**：关键观察——一个图像的类别信息可以被多个"概念"（concepts）的组合来表示。例如"晶体管"可由"半导体"和"电路"两个概念组成。用可学习概念集 $\widehat{\mathcal{C}}$ 替代显式类标签 $\mathcal{C}$，软加权代替硬切换。

**核心 idea**：用可学习的类无关概念集替代类标签依赖，配合Concept VQ-VAE和异常自适应在线学习，实现无类标签的长尾在线异常检测。

## 方法详解

### 整体框架
模型采用两分支pipeline：(1) 重建分支R——基于Concept VQ-VAE重建视觉特征，通过输入特征与重建特征的差异检测异常；(2) 语义分支S——利用VLM的文本-图像对齐能力，比较视觉特征与正常/异常prompt特征的相似度来检测异常。两个分支的输出通过加权调和平均融合为最终预测。所有模块都用概念分数 $\mathbf{p}$ 做软加权而非类标签硬切换。

### 关键设计

1. **类无关概念集（Class-Agnostic Concept Set）**:

    - 功能：用学习到的概念集 $\widehat{\mathcal{C}}$ 替代显式类标签，为每张图像分配软标签 $\mathbf{p} \in [0,1]^{\hat{K}}$
    - 核心思路：给定训练数据和词汇表 $\mathcal{V}$，先用CLIP计算每张图像与词汇的相似度，通过投票选出top-$\hat{K}$个词汇初始化概念集。概念嵌入设为可学习。推理时 $\mathbf{p} = \text{SoftMax}(\{\langle \mathbf{f}^f, \mathbf{t}_{\hat{c}} \rangle\})$
    - 设计动机：概念集大小不需匹配实际类别数，灵活高效；软加权机制在遇到未见类别时仍可工作，而硬切换方法会直接失败

2. **Concept VQ-VAE**:

    - 功能：重建分支中的特征重建模块，检测异常区域
    - 核心思路：基于HVQ改造，用概念替代类别。每层每个概念构建独立的量化模块 $Q_{l,\hat{c}}$，其codebook用对应概念嵌入的采样初始化。训练时各量化模块由概念分数软加权。异常预测为 $\hat{Y}^R = \frac{1}{2}(1 - \langle F^i, F^r \rangle)$
    - 设计动机：传统HVQ需类标签做硬切换，在线场景不可用。Concept VQ通过软加权自然适配无标签场景

3. **综合Prompt学习框架**:

    - 功能：为语义分支构建正常和异常的prompt集合
    - 核心思路：正常prompt用模板"a normal $\hat{c}$"初始化；异常prompt通过询问AI"对于 $\hat{c}$，最可能观察到的5种异常是什么？"初始化。所有prompt特征设为可学习。语义分支通过比较正常相似度 $S^n = \sum_{\hat{c}} p_{\hat{c}} S_{\hat{c}}^n$ 和异常相似度 $S^a$ 的差异做出预测
    - 设计动机：之前的工作要么没有类特异异常prompt，要么不支持概念级别的prompt

4. **异常自适应在线学习算法 $\mathcal{A}^{AA}$**:

    - 功能：在线场景中利用异常样本信息更新模型
    - 核心思路：对每个在线batch，先用当前模型预测伪异常图 $\hat{M} = \mathcal{T}(\hat{Y})$。对可能是异常的样本（$r(\hat{Y}) \geq \tau$）赋予更大的梯度权重 $\beta$，因为异常样本在离线训练中未见过，更有信息量。最后用EMA更新参数保证稳定性：$\theta_t = \gamma \theta_{t-1} + (1-\gamma)\tilde{\theta}_t$
    - 设计动机：朴素在线学习直接复用离线损失函数，无法利用在线数据中的异常样本信息

### 损失函数 / 训练策略
- 整体损失为重建分支损失和语义分支损失的组合
- 最终预测通过加权调和平均融合：$\hat{Y} = (\alpha(\hat{Y}^R)^{-1} + (1-\alpha)(\hat{Y}^S)^{-1})^{-1}$
- 在线阶段使用EMA更新保证稳定性

## 实验关键数据

### 主实验（Offline MVTec LTAD）

| 方法 | 类无关 | exp100-Det | exp100-Seg | step100-Det | step100-Seg |
|------|-------|-----------|-----------|------------|------------|
| LTAD | ✗ | 88.86 | 94.46 | 87.36 | 93.83 |
| HVQ | ✗ | 87.43 | 95.25 | 85.39 | 94.17 |
| LTOAD* (无类标签版) | ✗ | 93.12 | 95.01 | 92.02 | 94.72 |
| **LTOAD** | **✓** | **93.42** | **95.21** | **92.33** | **95.11** |

### 消融实验

| 配置 | MVTec Det. | MVTec Seg. | 说明 |
|------|-----------|-----------|------|
| Full LTOAD | 93.42 | 95.21 | 完整模型 |
| w/o Concept VQ (用原始HVQ) | 87.43 | 95.25 | VQ架构升级贡献约6% Det. |
| w/o 软切换 (硬切换) | ~90 | ~94.2 | 软切换比硬切换更灵活有效 |
| w/o 异常prompt | ~91 | ~94.5 | 异常prompt对检测帮助明显 |

### 关键发现
- 类无关方法不仅没有降低性能，反而在大多数设置下超越需要类标签的LTAD方法（+4.63% image-AUROC on MVTec）
- Concept VQ-VAE相比原始HVQ贡献最大，验证了概念级量化比类级量化更有效
- 在最具挑战性的长尾在线设置中，LTOAD实现+0.53% image-AUROC提升
- 方法在跨数据集设置下也展现了良好泛化能力
- 8种不同的在线stream配置全面评估了方法的鲁棒性

## 亮点与洞察
- **概念集替代类标签**：非常优雅的设计——用可学习的软概念替代硬类标签，不仅解决了在线场景无类标签问题，还因为概念的组合性天然支持未见类别。可迁移到任何需要从class-aware转向class-agnostic的场景
- **完整的LTOAD benchmark**：8种stream配置（blurry/disjoint × head-first/tail-first/mixed）系统地研究了长尾+在线的各种挑战
- **异常样本自适应加权**：在线学习中通过模型自身的预测分数识别并加权利用异常样本，简洁有效

## 局限与展望
- 概念集大小 $\hat{K}$ 需手动设定，不同数据集可能需要调参
- 异常prompt通过LLM生成，质量依赖于LLM对特定领域的知识
- EMA衰减系数 $\gamma$ 对在线学习稳定性敏感
- 每个概念需要独立的VQ模块，概念数增多时参数量线性增长
- 医学异常检测（如Uni-Medical）的结果可以进一步深入分析

## 相关工作与启发
- **vs LTAD**: LTAD是class-aware的，需要类标签做硬切换，在线场景不可用。LTOAD通过概念集完全移除了类标签依赖且性能更好
- **vs HVQ**: HVQ用class-specific的codebook。LTOAD的Concept VQ用概念替代类别更灵活
- **vs UniAD**: UniAD是统一AD方法但也需要类信息。LTOAD在类无关前提下超越了它

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ LTOAD新任务定义+概念集替代类标签，思路非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4个数据集+8种在线配置+offline/online全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ LTOAD benchmark对社区有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](../../CVPR2025/object_detection/simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)
- [\[ECCV 2024\] Rectify the Regression Bias in Long-Tailed Object Detection](../../ECCV2024/object_detection/rectify_the_regression_bias_in_long-tailed_object_detection.md)
- [\[ICLR 2026\] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](../../ICLR2026/object_detection/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)
- [\[ICLR 2026\] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting](../../ICLR2026/object_detection/bootstrapping_mllm_for_weaklysupervised_classagnostic_object_counting.md)
- [\[ICML 2025\] CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering](../../ICML2025/object_detection/costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering.md)

</div>

<!-- RELATED:END -->
