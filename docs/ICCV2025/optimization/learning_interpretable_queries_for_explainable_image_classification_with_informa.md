---
title: >-
  [论文解读] Learning Interpretable Queries for Explainable Image Classification with Information Pursuit
description: >-
  [ICCV 2025][优化/理论][可解释分类] 在CLIP语义嵌入空间中将信息追踪（Information Pursuit）的查询字典参数化为可学习向量，通过交替优化算法学习任务充分的可解释查询字典，缩小了可解释分类器与黑盒分类器的性能差距。 Information Pursuit（IP）是一种可解释性内建（interp…
tags:
  - "ICCV 2025"
  - "优化/理论"
  - "可解释分类"
  - "信息追踪"
  - "稀疏字典学习"
  - "CLIP"
  - "查询字典优化"
---

# Learning Interpretable Queries for Explainable Image Classification with Information Pursuit

**会议**: ICCV 2025  
**arXiv**: [2312.11548](https://arxiv.org/abs/2312.11548)  
**代码**: 无  
**领域**: 可解释人工智能 / 图像分类  
**关键词**: 可解释分类, 信息追踪, 稀疏字典学习, CLIP, 查询字典优化

## 一句话总结

在CLIP语义嵌入空间中将信息追踪（Information Pursuit）的查询字典参数化为可学习向量，通过交替优化算法学习任务充分的可解释查询字典，缩小了可解释分类器与黑盒分类器的性能差距。

## 研究背景与动机

Information Pursuit（IP）是一种可解释性内建（interpretable-by-design）的分类框架：给定一组预定义的语义查询字典，IP按信息增益顺序选择最有信息量的查询子集，基于查询-回答对做出预测。然而IP面临关键限制：

**查询字典依赖人工**：之前的方法使用人工标注的概念（如CUB-200的鸟类属性）或LLM提示生成查询，质量取决于专家经验

**LLM生成的查询次优**：依赖提示工程启发式方法，产生的查询集可能冗余、不相关或不足

**性能差距**：使用手工字典的IP与黑盒分类器之间存在显著精度差距

核心问题：**如何学习一个对任务充分的查询字典？**

## 方法详解

### 整体框架

利用CLIP的语义嵌入空间，将每个查询参数化为该空间中的可学习向量 $\theta_i$，通过最近邻映射 $q^{(\theta_i)} = \arg\min_{q \in \mathcal{U}} \|\theta_i - q\|_2^2$ 保持可解释性（每个学到的查询始终对应查询宇宙中的自然语言概念）。采用交替优化算法：固定字典更新V-IP网络 → 固定V-IP网络更新字典。

### 关键设计

1. **CLIP空间中的查询参数化**: 查询宇宙 $\mathcal{U} = \{E_T(c) | c \in \mathcal{T}\}$ 由约30万个CLIP文本嵌入组成（来自多个LLM提示和COCO数据集标题）。K个可学习嵌入 $\theta = \{\theta_i\}_{i=1}^K$ 通过最近邻投影保证可解释性。使用STE直通估计器实现 $\arg\min$ 的反向传播。字典增强的V-IP目标为：$\arg\min_{\theta,\psi,\eta} J_{Q_\theta}(\psi, \eta)$。

2. **交替优化算法（Algorithm 1）**: 直接联合优化三者（$\theta$, $\psi$, $\eta$）会导致问题：字典更新后查询语义改变，原有querier策略失效。因此采用交替方式：每$t=4$步V-IP网络更新搭配1步字典更新。V-IP更新时冻结字典训练querier和classifier；字典更新时冻结V-IP网络只更新 $\theta$。

3. **与稀疏字典学习的联系**: 方法与K-SVD等经典稀疏字典学习算法存在深层联系：(a) IP选择查询子集 ≈ OMP稀疏编码（选择最有信息量的原子）；(b) V-IP更新 ≈ 稀疏编码步骤（计算语义编码）；(c) 字典更新 ≈ 字典原子更新（减少分类误差而非重建误差）。Proposition 1证明在偏差采样下，最优字典参数同时最小化所有查询预算下的KL散度之和。

4. **查询回答机制**: 使用CLIP ViT-L/14计算软回答：归一化点积 $\hat{q}^{(\theta_i)}(X) = (\langle q^{(\theta_i)}/\|q\|, E_I(X)/\|E_I(X)\| \rangle - m_\theta) / (M_\theta - m_\theta)$，通过阈值0.5量化为硬回答（二值化保证可解释性）。

### 损失函数 / 训练策略

- V-IP损失：$J_{Q_\theta}(\psi, \eta) = \mathbb{E}_{X,S}[D_{KL}(P(Y|X) \| P_\psi(Y|S, A_\eta(X,S)))]$
- Querier和Classifier均为两层MLP，使用掩码处理变长输入
- Adam优化器，V-IP更新和字典更新交替进行
- 基于验证集准确率AUC调参

## 实验关键数据

### 主实验 — 查询字典学习提升V-IP准确率

在6个数据集上，K-Learned vs K-LLM（固定查询预算 $\tau$）：

| 数据集 | 查询预算 $\tau$ | K-LLM | K-Learned (best init) | Black-Box |
|--------|-----------|-------|----------------------|-----------|
| RIVAL-10 | 10 | ~96% | **~98.7%** | ~99% |
| CIFAR-10 | 10 | ~90% | **~95.1%** | ~97% |
| CIFAR-100 | 50 | ~70% | **~75.2%** | ~82% |
| ImageNet-100 | 50 | ~79% | **~84.0%** | ~91% |
| CUB-200 | 100 | ~69% | **~74.5%** | ~82% |
| Stanford-Cars | 100 | ~77% | **~82.4%** | ~87% |

K-Learned在所有数据集上显著超越K-LLM，并大幅缩小了与黑盒模型的差距。

### 消融实验

交替优化 vs 联合优化：

| 数据集 | 查询预算 | 交替优化 | 联合优化 |
|--------|---------|---------|---------|
| RIVAL-10 | 10 | **98.73%** | 98.26% |
| CIFAR-10 | 10 | **95.12%** | 87.00% |
| CUB-200 | 100 | **74.52%** | 72.14% |
| Stanford-Cars | 100 | **82.39%** | 79.18% |

交替优化一致优于联合优化，在CIFAR-10上差距高达8%。

与4种SOTA CBM的对比（使用RN50 CLIP + 软回答）：

| 数据集 | K-Learned | PCBM | LaBo | Label-free | Res-CBM |
|--------|-----------|------|------|------------|---------|
| CIFAR-10 | **88.55%** | 82.08% | 87.52% | 86.77% | 88.03% |
| CIFAR-100 | **68.02%** | 56.00% | 67.36% | 67.45% | 67.91% |

K-Learned优于或可比4种SOTA概念瓶颈模型。

### 关键发现

- 三种初始化方式（K-LLM、K-Random、K-Medoids）都能从学习中受益，性能差异在5个百分点内
- 量化（硬回答+最近邻投影）降低了性能但保证了可解释性
- 在jellyfish分类案例中，V-IP通过8个查询（如"Wings? No", "Swims? Yes", "UFO-like? Yes"）逐步缩小后验熵，提供了透明的决策过程
- CLIP作为查询回答机制存在噪声（如将jellyfish回答为"anemone? Yes"）

## 亮点与洞察

- **将字典学习从信号处理引入可解释AI**：建立了IP查询选择与OMP稀疏编码的形式化联系（Proposition 1）
- **可解释性约束内建于参数化**：通过最近邻投影到查询宇宙，保证学到的查询始终可用自然语言表达
- **交替优化的必要性**：揭示了querier与字典之间的耦合问题，联合优化会导致语义漂移
- **渐进式解释**：IP的决策过程像"20个问题"游戏，每步都能观察后验分布变化，比CBM的静态解释更直观

## 局限与展望

- 严重依赖CLIP的查询回答质量，CLIP的噪声回答限制了最终性能
- 查询宇宙需要预先构建（约30万个查询），构建质量影响学到的字典
- 硬回答的量化丢失信息，但去掉量化又损害可解释性，二者难以兼得
- 未探索如何将方法扩展到更大规模的分类任务（如完整的ImageNet-1K）
- 查询预算$\tau$对性能影响大但需要手动设定

## 相关工作与启发

- 与Res-CBM的区别：Res-CBM通过残差模块弥补不完整字典，而本文直接学习足够的字典
- Sparse CLIP (SPLICE)将图像分解为概念的稀疏线性组合，思路类似但面向不同任务
- 可启发其他需要可解释中间表示的任务，如可解释VQA、医疗诊断

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 稀疏字典学习与信息追踪的桥接非常优雅
- **实验充分度**: ⭐⭐⭐⭐ 6个数据集，多种初始化和优化策略对比
- **写作质量**: ⭐⭐⭐⭐⭐ 理论推导严谨，与经典方法的联系深入
- **综合价值**: ⭐⭐⭐⭐ 为可解释分类器的查询设计提供了原则性的学习方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](../../NeurIPS2025/optimization/chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [\[NeurIPS 2025\] From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD](../../NeurIPS2025/optimization/from_information_to_generative_exponent_learning_rate_induces_phase_transitions_.md)
- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](../../NeurIPS2025/optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[ICML 2025\] Synonymous Variational Inference for Perceptual Image Compression](../../ICML2025/optimization/synonymous_variational_inference_for_perceptual_image_compression.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)

</div>

<!-- RELATED:END -->
