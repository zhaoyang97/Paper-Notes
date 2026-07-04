---
title: >-
  [论文解读] Towards Flexible Perception with Visual Memory
description: >-
  [ICML2025][可解释性][visual memory] 将深度视觉模型的知识表示从"刻在权重里"转变为"存在外部数据库里"，用预训练编码器 + kNN 检索构建灵活的 Visual Memory，实现数据的即插即拔（添加/删除/扩展）和可解释分类，ImageNet 上达到 88.5% top-1 准确率。
tags:
  - "ICML2025"
  - "可解释性"
  - "visual memory"
  - "kNN classification"
  - "retrieval-based inference"
  - "machine unlearning"
  - "data attribution"
  - "scalable search"
---

# Towards Flexible Perception with Visual Memory

**会议**: ICML2025  
**arXiv**: [2408.08172](https://arxiv.org/abs/2408.08172)  
**作者**: Robert Geirhos, Priyank Jaini, Austin Stone, Sourabh Medapati, Xi Yi, George Toderici, Abhijit Ogale, Jonathon Shlens (Google DeepMind)
**代码**: 未开源  
**领域**: 可解释性  
**关键词**: visual memory, kNN classification, retrieval-based inference, machine unlearning, data attribution, scalable search

## 一句话总结

将深度视觉模型的知识表示从"刻在权重里"转变为"存在外部数据库里"，用预训练编码器 + kNN 检索构建灵活的 Visual Memory，实现数据的即插即拔（添加/删除/扩展）和可解释分类，ImageNet 上达到 88.5% top-1 准确率。

## 研究背景与动机

### 核心问题

当前深度学习模型将知识**静态地编码在数百万/数十亿参数**中，导致以下痛点：

**数据更新困难**：新数据可用或旧数据需撤回（隐私、公平性）时，必须重新训练或微调

**机器遗忘（unlearning）几乎不可能**：要从模型中移除某些训练样本的影响极其困难

**决策不可解释**：无法追溯一个预测是由哪些训练数据驱动的

**概念漂移（concept drift）**：现实世界不断变化，静态模型迅速过时

### 动机

作者主张应将 **"表示"（representation）** 和 **"知识"（memory）** 彻底解耦：用预训练模型负责学习通用特征表示，用外部可编辑数据库存储分类所需的知识。这一思想与心理学中的**样例理论（exemplar theory）**一致——人类通过与记忆中的已有样例进行比较来识别物体。

## 方法详解

### 整体架构

系统由两个模块组成：

1. **特征编码器 $\Phi$**：固定的预训练模型（如 DinoV2、CLIP），从图像提取特征向量
2. **Visual Memory 数据库**：存储 $(z_i, y_i)$ 对，其中 $z_i = \Phi(x_i)$ 为特征向量，$y_i$ 为标签

### 构建 Visual Memory

给定训练集 $\mathcal{D}_{\text{train}} = \{(x_1, y_1), \ldots, (x_N, y_N)\}$ 和预训练编码器 $\Phi$：

$$\text{VisualMemory} = \{(\Phi(x_1), y_1), (\Phi(x_2), y_2), \ldots, (\Phi(x_N), y_N)\}$$

仅存储特征向量（约占原始数据集 1-3% 的空间），不存储图像本身。

### 检索式分类

对测试图像 $\tilde{x}$，提取特征 $\tilde{z} = \Phi(\tilde{x})$，用余弦距离在 Visual Memory 中检索 $k$ 个最近邻 $\{(z_{[1]}, y_{[1]}), \ldots, (z_{[k]}, y_{[k]})\}$，满足：

$$\text{dist}(\tilde{z}, z_{[i]}) \leq \text{dist}(\tilde{z}, z_{[j]}), \quad \forall i \leq j$$

然后对邻居加权投票得到最终分类。

### 聚合策略：RankVoting（核心贡献）

作者发现现有的聚合方法（Plurality、Distance、Softmax voting）随着 $k$ 增大，准确率反而**下降**——它们对远距离邻居赋予了过多权重。作者提出 **RankVoting**：

$$w_i = \frac{1}{\alpha + \text{rank}_i}$$

其中 $\alpha = 2.0$ 为偏移量。这一幂函数权重方案使得准确率随 $k$ **单调上升**直至饱和，且无需调参。

| 聚合方法 | ImageNet Top-1 (DinoV2 ViT-L14) | 随 k 变化趋势 |
|---------|------|------|
| Plurality Voting | ~82.5% | 先升后降 |
| Distance Voting | ~83.0% | 先升后降 |
| Softmax Voting ($\tau=0.07$) | 83.6% | 先升后降 |
| **RankVoting** | **83.6%** | **单调上升至平稳** |
| + Gemini re-ranking | **88.5%** | — |

### Gemini Re-ranking

将 50 个最近邻及其标签放入 Gemini 1.5 Flash 的上下文中进行 in-context learning re-ranking，将准确率从 83.5% 提升至 **88.5%**（超过 DinoV2 ViT-L14 linear probing 的 86.3%）。Gemini 无邻居信息时仅 69.6%，说明性能主要由检索样本驱动。

### 检索加速

- 小规模（ImageNet）：GPU/TPU 上矩阵乘法 + argmax
- 大规模（10亿级）：使用 **ScaNN** 近似最近邻搜索，1M 特征上约 2ms/query（500-600 QPS），支持 CPU 部署

## 实验设置与主要结果

### 七大能力验证

#### 1. 灵活终身学习：添加 OOD 新类

在 ImageNet 1K 类的 Visual Memory 中直接插入 NINCO 数据集的 64 个 OOD 新类，**无需任何训练**：

| 场景 | IN-val Acc | NINCO Acc |
|------|-----------|-----------|
| 仅 IN-train | 83.6% | — |
| IN-train + NINCO | 83.6% (-0.02%) | **87.4%** |

新类加入对原始类性能几乎无影响（无灾难性遗忘），OOD 类即达 87%+ 准确率。

#### 2. 计算-记忆灵活权衡

不同模型在不同 memory 大小下的表现呈现规律性：小模型 + 大 memory ≈ 大模型 + 小 memory。例如 DinoV2 S/14 在 1.28M memory 下 ≈ L/14 在 ~70K memory。

#### 3. 十亿级数据无训练扩展

使用 JFT-3B 子集（ViT-22B 生成伪标签）构建十亿级 Visual Memory，ImageNet 验证误差**持续下降**，在 log-log 空间呈对数线性趋势。

#### 4. OOD 泛化提升

| 数据集 | Linear Probe | IN Memory | JFT Memory | + Gemini |
|--------|-------------|-----------|------------|---------|
| IN-A | 71.3 | 58.8 | 61.1 | 69.6 |
| IN-R | 74.4 | 62.8 | 73.7 | 81.4 |
| IN-Sketch | 59.3 | 61.5 | 68.0 | 75.0 |
| IN-V2 | 78.0 | 75.6 | 77.6 | 82.3 |
| IN-ReaL | 89.5 | 87.1 | 88.2 | 90.5 |

JFT memory 在 IN-R、IN-Sketch 上大幅超越 linear probe；Gemini re-ranking 全面领先。

#### 5. 机器遗忘：完美保证

从 Visual Memory 删除数据点即可实现**完美遗忘**（<20ms/sample），三项核心指标全优：

- **效率**：删除一个样本 <20ms vs. 重训练数小时
- **模型效用**：保留数据性能不受任何影响（by design）
- **遗忘质量**：100% 完全遗忘（by design）

前提条件：编码器需在安全的通用数据集上训练，待遗忘数据仅存在于 Visual Memory 中。

#### 6. 细粒度数据渐进添加（iNaturalist）

在 iNaturalist21（10,000 物种）上模拟新物种发现：仅加入 5-10 张新物种图片即可显著提升 species-level 准确率，且改善效果向上传播至 genus、family 等上层分类。

#### 7. Memory Pruning 与可解释决策

支持通过删除低质量/冗余样本进行 memory 压缩，以及追溯每个预测的数据来源（data attribution），实现可解释分类。

## 局限与展望

1. **表示与知识未完全解耦**：如果编码器本身在需要遗忘的数据上训练过，则"删除 memory"并不能完全消除影响
2. **仅验证分类任务**：检测、分割、生成等其他视觉任务的 Visual Memory 方案未探索
3. **Gemini re-ranking 成本高**：88.5% 的最佳结果依赖 VLM API 调用，推理成本和延迟不实用
4. **特征空间固定**：编码器冻结意味着无法通过 memory 数据改进表示质量本身
5. **检索瓶颈**：十亿级 memory 的存储和检索开销在边缘设备上仍不现实
6. **伪标签噪声**：JFT 十亿级实验依赖 ViT-22B 伪标签，标签质量直接限制上界
7. **未讨论多模态扩展**：仅用图像特征，未尝试融合文本或其他模态的 memory

## 相关工作与启发

- **样例理论 → Visual Memory**：从认知科学的 ALCOVE 模型到 kNN 分类的技术桥梁
- **检索增强学习**：与 kNN-LM (Khandelwal et al., 2019) 在 NLP 中的成功形成呼应
- **ScaNN 近似搜索**：十亿级检索的工程支撑
- **启发**：RAG 思想从 NLP 迁移到视觉的系统性探索，为"模型参数之外的知识表示"提供了有力论据

## 个人点评

这篇论文的核心价值在于**系统性**——并非某个单一技术突破，而是用一个极简框架（预训练编码器 + kNN + 数据库）统一回答了终身学习、机器遗忘、OOD 泛化、可解释性等多个深度学习的"老大难"问题。RankVoting 虽然简单但有效，发现了现有投票策略的过度自信问题。Gemini re-ranking 的结果令人印象深刻（88.5% 超过 linear probe），但实用性存疑。

最大的遗憾是这套方案**完全依赖预训练编码器的质量**——如果 DinoV2 的表示不足以区分某些类别，再多的 memory 也无济于事。此外，仅验证了分类任务，距离"通用 Visual Memory"还有很远。

论文写作极佳，故事讲得好，实验覆盖全面且细致。

## 评分
- 新颖性: ⭐⭐⭐ (思想不新但系统性论证有价值)
- 实验充分度: ⭐⭐⭐⭐⭐ (七大能力、十亿级规模、多数据集验证)
- 写作质量: ⭐⭐⭐⭐⭐ (叙事流畅，动机清晰)
- 价值: ⭐⭐⭐⭐ (对知识表示范式有启发意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)
- [\[CVPR 2025\] KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception](../../CVPR2025/interpretability/kvq_boosting_video_quality_assessment_via_saliency-guided_local_perception.md)
- [\[NeurIPS 2025\] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](../../NeurIPS2025/interpretability/dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](../../NeurIPS2025/interpretability/steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](../../NeurIPS2025/interpretability/model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)

</div>

<!-- RELATED:END -->
