---
title: >-
  [论文解读] ReSA: Clustering Properties of Self-Supervised Learning
description: >-
  [ICML 2025][自监督学习][聚类性质] 系统分析了 JEA-based SSL 中各组件的聚类性质，发现 encoding 比 embedding 和 projector 隐层具有更优更稳定的聚类能力，据此提出 ReSA（Representation Self-Assignment）利用 encoding 的聚类信息引导 embedding 学习，形成正反馈 SSL 框架，在多个标准基准上大幅超越 SOTA。
tags:
  - "ICML 2025"
  - "自监督学习"
  - "聚类性质"
  - "ReSA"
  - "positive feedback"
  - "Sinkhorn-Knopp"
---

# ReSA: Clustering Properties of Self-Supervised Learning

**会议**: ICML 2025  
**arXiv**: [2501.18452](https://arxiv.org/abs/2501.18452)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: self-supervised learning, 聚类性质, ReSA, positive feedback, Sinkhorn-Knopp

## 一句话总结
系统分析了 JEA-based SSL 中各组件的聚类性质，发现 encoding 比 embedding 和 projector 隐层具有更优更稳定的聚类能力，据此提出 ReSA（Representation Self-Assignment）利用 encoding 的聚类信息引导 embedding 学习，形成正反馈 SSL 框架，在多个标准基准上大幅超越 SOTA。

## 研究背景与动机

### 领域现状
**领域现状**：自监督学习（SSL）通过联合嵌入架构（JEA）在无标签监督下学习语义丰富的表示，已在视觉表示学习中超越监督学习。JEA 包含共享编码器 $E_{\theta_e}$ 和投影器 $G_{\theta_g}$，分别输出 encoding $H$ 和 embedding $Z$。已有研究（Ben-Shaul et al., 2023）发现 SSL 表示具有层次聚类性质——增强样本级、语义类级和超类级三层聚类结构。

### 现有痛点与挑战
**现有痛点**：(1) 虽然已知 SSL 表示具有聚类性质，但**几乎没有方法利用这些性质来改进 SSL 本身**——丰富的聚类信息被白白浪费；(2) 不清楚 encoding 和 embedding 哪个具有更好的聚类性质——projector 的优化动态和信息流仍是开放问题；(3) 现有 SwAV 等在线聚类方法使用可学习原型（prototype）映射 embedding 到聚类空间，但原型需要额外参数且聚类在 embedding 上进行（信息已损失）。

**核心矛盾**：SSL 表示具有丰富的聚类性质但未被利用来改进 SSL 自身，且聚类信息在 JEA 的不同组件中分布不均。

### 研究目标与方案
**本文目标**：(1) 回答"哪里提取聚类性质最好"；(2) 回答"如何利用聚类性质"；(3) 回答"正反馈是否促进更好的聚类"。

**核心 idea**：Encoding 具有最优聚类性质 → 在 encoding 上做在线自聚类 → 用聚类分配矩阵引导 embedding 的交叉熵损失 → 正反馈循环提升表示质量。

## 方法详解

### 整体框架
ReSA 在标准 JEA（编码器+投影器）基础上，从编码器输出 $H$ 提取聚类信息，通过 Sinkhorn-Knopp 算法生成在线聚类分配矩阵 $A_H$，用其引导投影器输出 $Z, Z'$ 之间的交叉熵损失，形成闭环正反馈：更好的 encoding → 更好的聚类分配 → 更好的训练信号 → 更好的 encoding。

### 关键设计

1. **Encoding 聚类性质优势的实证发现**：

    - 功能：确定最优的聚类信息来源
    - 核心思路：使用 Silhouette Coefficient（SC_mean 度量局部聚类能力、SC_std 度量稳定性）和 Adjusted Rand Index（ARI 度量全局聚类能力）在 CIFAR-10/100 上评估 SimCLR、VICReg、SwAV、BYOL 等方法的 encoding $H$、embedding $Z$ 和 projector 隐层 $P_0, P_1$ 的聚类性质。发现：(a) encoding 在几乎所有方法上 SC_mean 更高、SC_std 更低、ARI 更高；(b) 训练过程中 encoding 的聚类指标持续改善而 embedding 在后期退化；(c) projector 隐层虽然线性评估准确率接近 encoding，但聚类性质明显较差
    - 设计动机：确认 encoding 是最佳的聚类信息源，为后续设计提供基础

2. **在线自聚类机制（Online Self-Clustering）**：

    - 功能：从 encoding 中提取聚类信息并生成软分配矩阵
    - 核心思路：不使用可学习原型，而是将 mini-batch 中的 encoding 样本同时作为被聚类的点和聚类锚点。计算余弦自相似度矩阵 $S_H = H^\top H$（L2 归一化后），然后通过 Sinkhorn-Knopp 算法（3 次迭代，正则参数 $\epsilon=0.05$）将 $\exp(S_H/\epsilon)$ 转换为双随机矩阵 $A_H$ 作为聚类分配
    - 设计动机：与 SwAV 使用可学习原型不同，ReSA 无需额外参数且在 encoding 空间操作——直接利用 encoding 的优势聚类性质。Sinkhorn-Knopp 不涉及梯度传播，在 GPU 上高效实现

3. **聚类引导的交叉熵损失**：

    - 功能：利用聚类分配引导 embedding 学习
    - 核心思路：ReSA 损失定义为 $\ell_{\text{ReSA}} = -\frac{1}{2m}(\sum_{i,j} A_H \circ \log \mathcal{D}(Z^\top Z') + \sum_{i,j} A_H^\top \circ \log \mathcal{D}(Z'^\top Z))$，其中 $\mathcal{D}$ 为 softmax 温度归一化、$\circ$ 为 Hadamard 积。$A_H$ 来自 encoding 的聚类信息，引导 embedding 空间中相似样本的对齐
    - 设计动机：与 SwAV 的"交换预测"机制对比——SwAV 在 embedding 上做 Sinkhorn 然后交换预测，ReSA 在 encoding 上做 Sinkhorn 然后引导 embedding，利用了 encoding 更优的聚类性质

### 损失函数 / 训练策略
总损失为 ReSA 交叉熵损失，温度超参数 $\tau$ 控制分布锐度。无需额外的对比负样本或动量编码器，Sinkhorn-Knopp 的正则化自然防止表示坍塌。

## 实验关键数据

### 主实验：ImageNet 线性评估

| 方法 | Backbone | Epochs | Top-1 Acc. |
|------|----------|--------|------------|
| SimCLR | ResNet-50 | 200 | 66.5% |
| BYOL | ResNet-50 | 200 | 70.6% |
| SwAV | ResNet-50 | 200 | 71.8% |
| VICReg | ResNet-50 | 200 | 68.6% |
| **ReSA** | ResNet-50 | 200 | **73.2%** |

### 消融实验：聚类信息来源

| 聚类来源 | SC_mean ↑ | SC_std ↓ | ARI ↑ | 训练稳定性 |
|---------|-----------|----------|-------|-----------|
| Embedding $Z$ | 较低 | 高 | 较低 | 后期退化 |
| Projector 隐层 $P_0$ | 中 | 中 | 中 | 不稳定 |
| **Encoding $H$** | **最高** | **最低** | **最高** | **持续改善** |

### 训练效率对比

| 方法 | 达到 70% Top-1 所需 Epochs | 说明 |
|------|---------------------------|------|
| SimCLR | 未达到 | — |
| BYOL | ~200 | — |
| SwAV | ~180 | — |
| **ReSA** | **~150** | 收敛更快 |

### 关键发现
- Encoding 在几乎所有 SSL 方法中具有最优聚类性质——这是普遍现象而非个别方法的特例
- ReSA 的正反馈机制不仅提升性能还加速收敛——更好的聚类信号带来更高效的训练
- ReSA 同时改善了细粒度和粗粒度聚类性质

## 亮点与洞察
- **正反馈 SSL 的新范式**：聚类性质 → 训练信号 → 更好的表示 → 更好的聚类性质——这种自增强循环是 SSL 方法论的概念性贡献
- **Encoding vs Embedding 的系统分析**：首次严格量化了 JEA 各组件的聚类性质差异，为 projector 的作用机制提供了新视角
- **无原型的在线聚类**：与 SwAV/DINOv2 的可学习原型不同，ReSA 直接在样本间自聚类——更简洁且无需额外参数

## 局限与展望
- **Sinkhorn-Knopp 在大 batch 下的计算开销**：自相似度矩阵 $S_H$ 为 $m \times m$，大 batch 时内存和计算成本增加
- **仅在视觉 SSL 上验证**：NLP 和多模态 SSL 中的聚类性质可能有不同表现
- **与知识蒸馏方法的关系**：ReSA 可视为一种自蒸馏——encoding 作为"教师"引导 embedding——与 DINO/iBOT 的关系值得深入探讨
- **聚类数量的隐式假设**：Sinkhorn-Knopp 不显式设定聚类数，但 batch 大小隐式限制了可发现的聚类结构

## 相关工作与启发
- **vs SwAV (Caron et al., 2020)**：SwAV 在 embedding 上用可学习原型做聚类——ReSA 在 encoding 上无原型自聚类，利用了更优的信息源
- **vs DINOv2 (Oquab et al., 2023)**：也使用 Sinkhorn-Knopp 但采用可学习原型——ReSA 的无原型设计更简洁
- **vs Ben-Shaul et al. (2023)**：他们发现了 SSL 的层次聚类性质——本文首次利用这些性质来改进 SSL 本身
- **vs Ma et al. (2023)**：利用 encoding 的增强鲁棒性重新加权正对对齐——但忽略了聚类信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 正反馈 SSL 是概念性贡献，encoding 聚类性质的系统分析有重要价值
- 实验充分度: ⭐⭐⭐⭐ 多方法多数据集的聚类分析 + ImageNet 大规模实验
- 写作质量: ⭐⭐⭐⭐ 三个递进问题组织结构清晰
- 价值: ⭐⭐⭐⭐ 为 SSL 社区提供了新的方法论范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](../../NeurIPS2025/self_supervised/t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](../../NeurIPS2025/self_supervised/understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
