---
title: >-
  [论文解读] FACE: Faithful Automatic Concept Extraction
description: >-
  [NeurIPS 2025][概念解释] 提出 FACE 框架，在非负矩阵分解 (NMF) 中加入 KL 散度正则项，约束概念重建后的激活值保持与原始模型预测一致，从而提取真正忠实于模型决策过程的概念解释，在 ImageNet/COCO/CelebA 上全面超越 CRAFT 和 ICE。 领域现状：概念型解释方法（如 TCA…
tags:
  - "NeurIPS 2025"
  - "概念解释"
  - "NMF"
  - "KL散度"
  - "忠实性"
  - "非负矩阵分解"
  - "可解释性"
---

# FACE: Faithful Automatic Concept Extraction

**会议**: NeurIPS 2025  
**arXiv**: [2510.11675](https://arxiv.org/abs/2510.11675)  
**代码**: [GitHub](https://github.com/dipkamal/FACE)  
**领域**: 可解释 AI / 概念发现  
**关键词**: 概念解释, NMF, KL散度, 忠实性, 非负矩阵分解, 可解释性

## 一句话总结
提出 FACE 框架，在非负矩阵分解 (NMF) 中加入 KL 散度正则项，约束概念重建后的激活值保持与原始模型预测一致，从而提取真正忠实于模型决策过程的概念解释，在 ImageNet/COCO/CelebA 上全面超越 CRAFT 和 ICE。

## 研究背景与动机

**领域现状**：概念型解释方法（如 TCAV、ACE、CRAFT、ICE）试图用人类可理解的高层概念（如"毛发"、"耳朵"）解释模型决策，比像素级归因更直观。

**现有痛点**：(a) TCAV 需手动标注概念数据集，扩展性差；(b) ACE/ICE/CRAFT 等无监督方法通过对编码器激活做聚类或 NMF 发现概念，但仅关注重建误差，忽略分类器行为；(c) 标准 NMF 倾向捕获高方差方向而非类判别方向，可能导致重建后预测完全改变。

**核心矛盾**：低重建误差 $\|\mathbf{A} - \mathbf{UW}^\top\|_F^2$ 小 ≠ 预测保真。即使激活空间误差极小，经过分类器头 $h$ 和 softmax 的非线性放大后，预测分布可能大幅偏离。

**本文解决方案**：在 NMF 目标中引入 KL 散度正则项，直接约束原始激活和重建激活在分类器输出分布上的一致性。

## 方法详解

### 框架设置
将分类器 $f$ 分解为编码器 $g: \mathcal{X} \to \mathcal{G}$ 和分类头 $h: \mathcal{G} \to \mathcal{Y}$，$f(\mathbf{x}) = h(g(\mathbf{x}))$。对 $n$ 个样本的激活矩阵 $\mathbf{A} = g(\mathbf{X}) \in \mathbb{R}^{n \times p}_{+}$ 进行 NMF 分解为 $\mathbf{U} \in \mathbb{R}^{n \times r}_{+}$ 和 $\mathbf{W} \in \mathbb{R}^{p \times r}_{+}$。

### 核心优化目标
标准 NMF 仅最小化重建误差：

$$\min_{\mathbf{U}\geq 0, \mathbf{W}\geq 0} \frac{1}{2}\|\mathbf{A} - \mathbf{UW}^\top\|_F^2$$

FACE 增加 KL 散度正则项：

$$\min_{\mathbf{U}\geq 0, \mathbf{W}\geq 0} \frac{1}{2}\|\mathbf{A} - \mathbf{UW}^\top\|_F^2 + \lambda \cdot \text{KL}(h(\mathbf{A}) \| h(\mathbf{UW}^\top))$$

其中 $\lambda > 0$ 控制重建保真与预测对齐的权衡。KL 散度对 softmax 归一化后的 logits 计算。

### 理论保证
利用 Pinsker 不等式，当 KL 散度受限于 $\varepsilon$ 时，预测分布的总变差距离有界：

$$\|p - q\|_1 \leq \sqrt{2 \cdot \text{KL}(p \| q)} \leq \sqrt{2\varepsilon}$$

这意味着 KL 正则化直接控制了概念替换造成的预测偏差，而单纯最小化重建误差无法提供类似保证。

### 局部线性性
KL 正则化确保 softmax 在 $\mathbf{UW}^\top$ 附近其非线性可忽略，使概念空间中的操作（如删除/插入）对预测的影响是线性且可预测的。

### 优化方法
- 采用投影梯度下降交替更新 $\mathbf{U}$ 和 $\mathbf{W}$，投影到非负约束
- 初始化使用 NNDSVD (Non-negative Double SVD)
- 概念重要性使用 Sobol 索引量化

### 评估指标
- **C-Del (概念删除)**：逐步删除最重要概念后准确率下降幅度（面积），越高越忠实
- **C-Ins (概念插入)**：逐步插入最重要概念后准确率恢复速度，越高越忠实
- **C-Gini (基尼稀疏性)**：概念重要性分布的稀疏度，越高解释越简洁

## 实验关键数据

### 矩阵分解质量 (ResNet-34)

| 数据集 | 方法 | MSE ↓ | $D_\text{KL}$ ↓ |
|--------|------|-------|---------|
| ImageNet | ICE | 0.296 | 0.359 |
| ImageNet | CRAFT | 0.451 | 0.240 |
| ImageNet | **FACE** | 0.497 | **0.220** |
| COCO | ICE | 0.308 | 0.596 |
| COCO | CRAFT | 0.457 | 0.600 |
| COCO | **FACE** | 0.462 | **0.458** |
| CelebA | ICE | 0.148 | 0.212 |
| CelebA | CRAFT | 0.498 | 0.110 |
| CelebA | **FACE** | 0.375 | **0.021** |

### 忠实性与复杂度 (ResNet-34)

| 数据集 | 方法 | C-Ins ↑ | C-Del ↑ | C-Gini ↑ |
|--------|------|---------|---------|----------|
| ImageNet | ICE | 0.908 | 0.484 | 0.537 |
| ImageNet | CRAFT | 0.932 | 0.752 | 0.835 |
| ImageNet | **FACE** | **0.969** | **0.891** | **0.895** |
| COCO | ICE | 0.883 | 0.632 | 0.623 |
| COCO | CRAFT | 0.861 | 0.691 | 0.874 |
| COCO | **FACE** | **0.971** | **0.894** | **0.947** |
| CelebA | ICE | 0.910 | 0.365 | 0.662 |
| CelebA | CRAFT | 0.953 | 0.604 | 0.901 |
| CelebA | **FACE** | **0.971** | **0.635** | **0.928** |

### 重建后预测准确率
FACE 在所有类别上保持 100% 重建后 top-1 准确率，而 CRAFT 在 "Train" 类仅 40%。

### 消融实验 (λ 影响)
- ImageNet/COCO：小 λ（如 $10^{-5}$）即明显提升忠实性，过大 λ（$\geq 10^3$）性能下降
- CelebA（4 类）：可承受更大 λ（至 $10^5$），因低维分布对齐更容易
- 分解秩 $r=25$ 时性能趋于饱和

## 亮点与洞察
- **核心创新**：首次在 NMF 概念发现中引入预测对齐约束，将"看起来合理但对模型不忠实"的解释问题形式化并解决
- **理论与实验一致**：Pinsker 不等式提供了忠实性的定量保证，实验验证 FACE 在所有数据集/模型上 KL 散度最低
- **重要洞察**：模型可能不使用人类直觉认为的特征（如 FACE 发现兔子分类依赖"毛发"而非"头部"），忠实解释比直觉解释更重要
- **计算开销轻量**：优化以小矩阵乘法和线性头为主，低资源硬件可行

## 局限与展望
- 仅支持 CNN 架构，直接应用于 ViT 等 Transformer 架构需额外适配
- 类级别全局解释，不支持实例级概念发现
- $\lambda$ 需数据集特定调参，无自适应选择策略
- 缺乏人类评估研究验证 KL 约束是否真正改善人类理解
- 仅在 ResNet-34 和 MobileNetV2 上验证，未测试更大模型

## 相关工作对比
- **vs TCAV**: TCAV 需手工标注概念数据集；FACE 全自动发现概念
- **vs ACE**: ACE 超像素聚类易引入伪影；FACE 基于 NMF 空间更连续
- **vs CRAFT**: CRAFT 仅优化重建误差，发现的概念可能与模型决策不一致；FACE 通过 KL 约束确保忠实性
- **vs ICE**: ICE 卷积核级 NMF 仅捕获局部概念；FACE 在倒数第二层捕获高层语义
- **vs CRP/RelMax**: 基于反向传播的方法，不做激活分解；FACE 基于分解可量化概念重要性

## 评分
- 新颖性: ⭐⭐⭐⭐ KL散度正则NMF思路简洁有效，理论保证完整
- 实验充分度: ⭐⭐⭐⭐ 3数据集×2模型，消融全面（λ、秩r），多指标评估
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，理论推导严谨
- 价值: ⭐⭐⭐⭐ 解决概念解释忠实性的基础问题，可广泛适用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Faithful Group Shapley Value](faithful_group_shapley_value.md)
- [\[ACL 2025\] Graph-Structured Trajectory Extraction from Travelogues](../../ACL2025/others/graph-structured_trajectory_extraction_from_travelogues.md)
- [\[NeurIPS 2025\] Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure](obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[ICML 2025\] Efficient Network Automatic Relevance Determination](../../ICML2025/others/efficient_network_automatic_relevance_determination.md)

</div>

<!-- RELATED:END -->
