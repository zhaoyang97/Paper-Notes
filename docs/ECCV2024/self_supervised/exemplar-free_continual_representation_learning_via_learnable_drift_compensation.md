---
title: >-
  [论文解读] Exemplar-Free Continual Representation Learning via Learnable Drift Compensation
description: >-
  [ECCV 2024][自监督学习][持续学习] 提出可学习漂移补偿(LDC)，通过训练一个前向投影器将旧特征空间映射到新特征空间，在无需存储旧样本的情况下有效补偿类原型的语义漂移，首次实现了无样本半监督持续学习。 持续学习（Continual Learning）要求模型在非平稳数据流上依次学习多个任务，而不遗忘旧知识…
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "持续学习"
  - "类原型"
  - "语义漂移补偿"
  - "无样本"
  - "半监督学习"
---

# Exemplar-Free Continual Representation Learning via Learnable Drift Compensation

**会议**: ECCV 2024  
**arXiv**: [2407.08536](https://arxiv.org/abs/2407.08536)  
**代码**: [https://github.com/alviur/ldc](https://github.com/alviur/ldc)  
**领域**: 自监督  
**关键词**: 持续学习, 类原型, 语义漂移补偿, 无样本, 半监督学习

## 一句话总结

提出可学习漂移补偿(LDC)，通过训练一个前向投影器将旧特征空间映射到新特征空间，在无需存储旧样本的情况下有效补偿类原型的语义漂移，首次实现了无样本半监督持续学习。

## 研究背景与动机

持续学习（Continual Learning）要求模型在非平稳数据流上依次学习多个任务，而不遗忘旧知识。现有无样本（exemplar-free）方法面临一个核心难题：当backbone在新任务上更新时，旧类的特征表示会发生**语义漂移（semantic drift）**，导致之前存储的类原型（class prototype）在新特征空间中位置失效，造成灾难性遗忘。

**现有方法的痛点**：
- 大多数无样本方法只在"热启动"（Warm Start）设置下评估，即从大的首任务或预训练模型开始，回避了严重的特征漂移问题
- 现有漂移补偿方法如SDC假设漂移可以用局部平移近似，但实际漂移可能包含旋转、缩放等复杂变换
- 冻结backbone的方法（如FeTrIL、FeCAM）在冷启动设置下因首任务较小而表征能力不足

**核心发现与切入角度**：作者通过oracle实验发现了一个关键洞察——当用oracle完美补偿原型漂移后，性能可以大幅恢复。这说明backbone的判别能力并未因增量学习而显著下降，遗忘主要是原型位置偏移所致。这一发现直接激励了"与其防止遗忘，不如补偿漂移"的思路。

**核心idea**：学习一个简单的前向投影器，利用当前任务数据将旧特征空间的原型映射到新特征空间，无需标签、无需存储旧数据。

## 方法详解

### 整体框架

LDC在每个任务训练结束后应用。给定旧特征提取器 $f_\theta^{t-1}$ 和新特征提取器 $f_\theta^t$，学习一个前向投影器 $p_F^t$ 将旧特征空间映射到新特征空间。用当前任务数据 $D_t$ 同时通过两个frozen的提取器得到配对特征，训练投影器后更新所有旧类原型。最终使用最近类均值（NCM）分类器进行推理。

### 关键设计

1. **前向投影器（Forward Projector）**：投影器 $p_F^t$ 将旧特征映射到新特征空间。训练时最小化MSE损失：

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(p_F^t(f_\theta^{t-1}(x_i)) - f_\theta^t(x_i))^2$$

其中 $f_\theta^t$ 和 $f_\theta^{t-1}$ 均为冻结状态。**设计动机**：不同于SDC的局部平移假设，LDC通过可学习映射可以捕获任意形式的漂移（包括旋转、缩放）。实验发现简单的线性层效果最佳，说明特征空间之间的映射关系具有近线性特性。

2. **无标签原型更新**：训练完投影器后，所有旧类原型通过前向投影更新：

$$P_t^c = p_F^t(P_{t-1}^c)$$

**关键特点**：此过程不需要类标签，仅需要存储的旧原型和当前任务数据。这使得LDC天然适用于无监督和半监督场景。

3. **NCM分类器**：使用最近类均值分类器进行推理，将测试样本分配给距离最近的类原型：

$$y^* = \arg\min_{y=1,...,Y^t} \|f_\theta^t(x) - P_t^y\|$$

### 训练策略

- **监督CL**：可与LwF等正则化方法结合，在每个任务结束后额外训练投影器。LwF使用知识蒸馏损失 $\mathcal{L} = \mathcal{L}_{ce}(h_t(x), y) + \lambda \mathcal{L}_{ce}(h_{t-1}(x), h_t(x))$
- **半监督CL**：与自监督CL方法（PFR、CaSSLe、POCON）结合，利用全部无标签数据训练投影器，仅用少量标签计算原型。这是首个无样本半监督持续学习方法。
- 投影器用Adam优化器训练，监督设置下lr=0.001训练20个epoch，半监督设置下lr=5e-3训练100个epoch

## 实验关键数据

### 主实验（监督设置）

| 数据集 | 指标 | LwF+LDC | EFC (之前SOTA) | LwF+SDC | 提升 |
|--------|------|---------|----------------|---------|------|
| CIFAR-100 | $A_{last}$ | **45.4** | 43.6 | 40.6 | +1.8 vs EFC |
| CIFAR-100 | $A_{inc}$ | **59.5** | 58.6 | 56.2 | +0.9 vs EFC |
| Tiny-ImageNet | $A_{last}$ | **34.2** | 34.1 | 29.5 | +0.1 vs EFC |
| ImageNet100 | $A_{last}$ | **51.4** | 47.3 | 42.6 | +4.1 vs EFC |
| ImageNet100 | $A_{inc}$ | **69.4** | 59.9 | 65.3 | +9.5 vs EFC |
| Stanford Cars (ViT) | $A_{last}$ t10 | **62.9** | - | 53.3 | +9.6 vs SDC |

### 半监督设置（CIFAR-100, 10 tasks）

| 方法 | 无样本 | 0.8%标签 | 5%标签 | 25%标签 |
|------|--------|---------|--------|---------|
| CaSSLe+naive | ✓ | 19.4 | 23.2 | 23.6 |
| CaSSLe+SDC | ✓ | 22.1 | 25.8 | 26.5 |
| CaSSLe+LDC | ✓ | **27.0** | **32.8** | **35.0** |
| NNCSL (500 exemplars) | ✗ | 27.4 | 31.4 | 35.3 |

### 消融实验

| 投影器架构 | LwF+LDC (CIFAR-100) | CaSSLe+LDC (25%) |
|-----------|---------------------|-------------------|
| Linear (无bias) | **45.4** | **35.0** |
| Linear + bias | 45.2 | 34.3 |
| Linear + ReLU | 41.2 | 29.1 |
| MLP (两层) | 43.7 | 34.9 |

### 关键发现

1. **遗忘主要来自原型漂移而非特征退化**：Oracle实验显示完美漂移补偿可恢复大部分性能，说明backbone的判别力并未显著受损
2. **LDC在冷启动下远优于SDC**：在CIFAR-100上LDC比SDC高4.8%，因为SDC的局部平移假设在高漂移场景下失效
3. **LDC无需样本即可媲美NME**：LDC的性能与存储20个样本/类的NME相当
4. **半监督设置下LDC首次无需样本即可达到exemplar-based方法的水平**：5%和25%标签设置下与NNCSL持平

## 亮点与洞察

- 核心洞察极具价值："遗忘≠特征退化"，大部分遗忘可通过原型位置修正来恢复
- 方法极其简洁——仅需一个线性层做投影，训练开销极小（20个epoch）
- 统一框架：同一个LDC方法可无缝应用于监督、自监督和半监督CL
- 首个无样本半监督持续学习方法，打开了新的研究方向

## 局限与展望

- 投影器仅在当前任务数据上训练，可能因数据偏差导致部分旧类原型更新不精确（corrected与oracle之间仍有差距）
- 随着任务增多，原型经过多次投影的累积误差可能逐渐增大
- 只存储类均值原型，丢失了类内分布信息（如协方差）
- 未探索更复杂的backbone架构（如大规模ViT）和更多任务划分方式

## 相关工作与启发

- SDC [Yu et al.] 提出的漂移补偿思路是本文的直接前驱，但其平移假设限制了适用性
- FeCAM [Goswami et al.] 使用Mahalanobis距离替代欧氏距离，本文的LDC可与之结合
- MEA [Moon et al.] 也学习映射但需要标签和大量存储特征，不适用于无监督场景
- 可考虑将LDC与adapter-based方法或prompt-based方法结合

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心洞察（遗忘≠退化）和由此推导的方法思路清晰有力
- 实验充分度: ⭐⭐⭐⭐⭐ 三个设置（监督/自监督/半监督）×多数据集×详细消融
- 写作质量: ⭐⭐⭐⭐ 动机分析到方法推导逻辑流畅，图示清晰
- 价值: ⭐⭐⭐⭐ 方法简洁实用，开创了无样本半监督CL新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Exemplar-Free Continual Learning for State Space Models](../../CVPR2026/self_supervised/exemplar-free_continual_learning_for_state_space_models.md)
- [\[ECCV 2024\] Revisiting Supervision for Continual Representation Learning](revisiting_supervision_for_continual_representation_learning.md)
- [\[AAAI 2026\] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](../../AAAI2026/self_supervised/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)
- [\[CVPR 2026\] Dual-Estimator: Decoupling Global and Local Semantic Shift for Drift Compensation in Class-Incremental Learning](../../CVPR2026/self_supervised/dual-estimator_decoupling_global_and_local_semantic_shift_for_drift_compensation.md)
- [\[ECCV 2024\] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)

</div>

<!-- RELATED:END -->
