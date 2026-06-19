---
title: >-
  [论文解读] Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping
description: >-
  [CVPR 2025][自监督学习][类增量学习] 提出 GDDSG，用图着色理论将类按相似度分组——同组内类别尽量不相似（减少干扰），每组独立用 NCM 分类器+LoRA 适配器学习，在 CIFAR-100 10-step 上达到 94.00% 准确率和仅 0.78% 遗忘率（前 SOTA RanPAC 90.50%/3.49%）。
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "类增量学习"
  - "图着色"
  - "顺序鲁棒"
  - "相似度分组"
  - "NCM分类器"
---

# Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping

**会议**: CVPR 2025  
**arXiv**: [2502.20032](https://arxiv.org/abs/2502.20032)  
**代码**: [https://github.com/AIGNLAI/GDDSG](https://github.com/AIGNLAI/GDDSG)  
**领域**: LLM安全  
**关键词**: 类增量学习, 图着色, 顺序鲁棒, 相似度分组, NCM分类器

## 一句话总结

提出 GDDSG，用图着色理论将类按相似度分组——同组内类别尽量不相似（减少干扰），每组独立用 NCM 分类器+LoRA 适配器学习，在 CIFAR-100 10-step 上达到 94.00% 准确率和仅 0.78% 遗忘率（前 SOTA RanPAC 90.50%/3.49%）。

## 研究背景与动机

### 领域现状

**领域现状**：类增量学习（CIL）要求模型在不断学习新类的同时不遗忘旧类。现有方法的性能高度依赖类的到达顺序——在某些顺序下表现好，换一种顺序就大幅下降。

**现有痛点**：顺序敏感性（order sensitivity）是 CIL 的核心挑战但被大多研究忽视。理论分析（Corollary 1）证明：类间相似度越高→遗忘越多+顺序敏感性越强。

**核心矛盾**：连续到来的相似类（如先学"猫"再学"虎"）会导致严重的表示混淆和遗忘，但类的到达顺序不可控。

**切入角度**：不改变类的到达顺序，而是在学习时将相似类分到不同组——每组内的类尽量不相似。用图着色理论保证分组最优性。

**核心 idea**：构建相似度图 → Welsh-Powell 着色 → 异组内类不相似 → 独立 NCM + 适配器 = 顺序无关的增量学习。

### 解决思路

**本文目标**：### 关键设计

1. **SimGraph + 图着色**：用预训练特征的 L2 距离构建类间相似度图，自适应阈值 $\eta_{i,j}$ 决定是否连边。


## 方法详解

### 关键设计

1. **SimGraph + 图着色**：用预训练特征的 L2 距离构建类间相似度图，自适应阈值 $\eta_{i,j}$ 决定是否连边。Welsh-Powell 着色算法 $O(|V|^2)$ 将类分为颜色组——同色组内类不相似

2. **组内独立学习**：每组用独立的 NCM 分类器 + LoRA 适配器。互不干扰——消除了组内的类间干扰

3. **元特征推理**：推理时先用元分类器（RandomForest+KNN+LightGBM 集成）识别属于哪个组，再用组内 NCM 分类

### 损失函数 / 训练策略

正则化最小二乘回归 $\mathcal{L} = \|Y - H\Theta\|_F^2 + \lambda\|\Theta\|_F^2$，$\lambda$ 通过校准集交叉验证。backbone 冻结（ViT-B/16）。

## 实验关键数据

| 数据集 (10-step) | GDDSG | RanPAC | 遗忘率 |
|-----------------|-------|--------|--------|
| CIFAR-100 | **94.00%** | 90.50% | **0.78%** |
| CUB-200 | **91.64%** | 89.23% | 1.92% |
| Dogs | **92.64%** | 85.37% | 1.42% |

### 消融实验
- 无分组→准确率降到 91-92%——分组贡献 2-3%
- Brooks 定理满足概率在 N>35 时 >0.99
- 图着色产生的分组在多项式时间内最优

### 关键发现
- **顺序鲁棒性**：GDDSG 的性能方差远小于基线——不同顺序下性能一致
- **遗忘率极低**：0.78% vs RanPAC 3.49%——组内不相似保证了更少的干扰
- **理论保证**：Corollary 1 证明降低组内相似度同时降低期望遗忘和顺序敏感性

## 亮点与洞察
- **图着色 × CIL 的跨领域创新**——用组合优化的经典工具解决深度学习中的顺序敏感性
- **理论驱动设计**——从 Corollary 1 直接推导出方法

## 局限与展望
- 依赖冻结 ViT-B/16 backbone
- 元分类器增加推理复杂度
- Gram 矩阵和原型矩阵随类数增长

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 图着色+CIL 的创新组合，理论驱动
- 实验充分度: ⭐⭐⭐⭐ 4 数据集，顺序敏感性分析
- 写作质量: ⭐⭐⭐⭐ 理论清晰
- 价值: ⭐⭐⭐⭐⭐ 有望成为 CIL 的新默认框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)
- [\[CVPR 2026\] Geometry-driven OOD Detectors Are Class-Incremental Learners](../../CVPR2026/self_supervised/geometry-driven_ood_detectors_are_class-incremental_learners.md)
- [\[CVPR 2025\] SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning](sec-promptsemantic_complementary_prompting_for_few-shot_class-incremental_learni.md)
- [\[ACL 2025\] AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](../../ACL2025/self_supervised/analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)
- [\[CVPR 2026\] Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models](../../CVPR2026/self_supervised/representation-steered_incremental_adapter-tuning_for_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
