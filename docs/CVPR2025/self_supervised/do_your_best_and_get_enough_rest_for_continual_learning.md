---
title: >-
  [论文解读] Do Your Best and Get Enough Rest for Continual Learning
description: >-
  [CVPR 2025][自监督学习][遗忘曲线] 受Ebbinghaus遗忘曲线理论启发，提出View-Batch Model(VBM)——通过将batch中多个不同样本替换为同一样本的多个增强视图（replay），延长回忆间隔V倍至最优范围，同时用one-to-many KL散度自监督损失从单样本中学习更多知识（do your best），作为drop-in替代方案在多种持续学习方法上一致提升性能。
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "遗忘曲线"
  - "间隔效应"
  - "View-Batch"
  - "回忆间隔优化"
  - "即插即用"
---

# Do Your Best and Get Enough Rest for Continual Learning

**会议**: CVPR 2025  
**arXiv**: [2503.18371](https://arxiv.org/abs/2503.18371)  
**代码**: [https://github.com/hankyul2/ViewBatchModel](https://github.com/hankyul2/ViewBatchModel)  
**领域**: 自监督学习 / 持续学习  
**关键词**: 遗忘曲线, 间隔效应, View-Batch, 回忆间隔优化, 即插即用

## 一句话总结

受Ebbinghaus遗忘曲线理论启发，提出View-Batch Model(VBM)——通过将batch中多个不同样本替换为同一样本的多个增强视图（replay），延长回忆间隔V倍至最优范围，同时用one-to-many KL散度自监督损失从单样本中学习更多知识（do your best），作为drop-in替代方案在多种持续学习方法上一致提升性能。

## 研究背景与动机

**领域现状**：持续学习的核心问题是灾难性遗忘。rehearsal方法（ER、DER++、iCaRL）使用记忆缓冲区重放旧样本，但回忆间隔（同一样本两次被训练之间的间隔）未被优化。

**现有痛点**：当前方法的回忆间隔=batch大小×训练步数=数据集大小，通常过短——模型短间隔内重复训练同一样本，根据遗忘曲线理论，这是低效的。最优回忆间隔需要足够长（但不能太长），以实现"间隔效应"(spacing effect)增强长期记忆保持。

**核心矛盾**：延长回忆间隔意味着每个样本被训练的次数减少——如何在延长间隔的同时从每个样本中提取更多知识？

**核心idea**：(1) 用view-batch（同一样本的V个增强视图）替代sample-batch，回忆间隔自动延长V倍；(2) 用自监督损失（weak vs strong增强视图的KL散度）从每个样本中学习更多。总epoch减少V倍保持总计算量不变。

## 方法详解

### 整体框架

原始scheduler：$\mathcal{A} = [\mathcal{B}_1^I, ..., \mathcal{B}_T^I, \mathcal{B}_1^I, ...]$，回忆间隔=$B \times T$
VBM scheduler：$\mathcal{A} = [\mathcal{B}_1^V, ..., \mathcal{B}_T^V, \mathcal{B}_1^V, ...]$，回忆间隔=$B \times T \times V$
其中 $\mathcal{V}_i = \{I_i\}_{j=1}^V$（同一样本的V个增强视图）。总epoch减少V倍。

### 关键设计

1. **View-Batch Replay**：

    - 功能：延长回忆间隔至最优范围
    - 核心思路：batch大小不变 $B$，但每个slot放同一样本的不同增强视图而非不同样本。实际唯一样本数减少到 $B/V$，回忆间隔延长V倍
    - 第一个视图用弱增强（水平翻转），其余V-1个用强增强（AutoAugment）
    - 实验验证：V=4时回忆间隔处于最优区间，遗忘程度（memory retention decay）最缓

2. **One-to-Many自监督损失**：

    - 功能：从单样本的多视图中学习更多知识
    - 核心思路：$L_{ssl} = \frac{1}{B \cdot (V-1)} \sum_{i=1}^B \sum_{j=2}^V D_{KL}(p_i^1 \| p_i^j)$，即弱增强视图的logit分布作为target，最小化与强增强视图之间的KL散度
    - 设计动机：不需要额外架构（teacher network等），仅在logit层面做一致性约束，task-agnostic的知识对遗忘更鲁棒

3. **Drop-in替代设计**：

    - 仅修改数据加载和loss函数，不改模型架构、不增加训练epoch
    - 总前向计算次数=原始方法（epoch减少V倍 × 每step多V个view = 持平）
    - 可与任何rehearsal/rehearsal-free方法组合

## 实验关键数据

### 主实验：VBM加持各CL方法（S-CIFAR-10, buffer=200）

| 方法 | 原始CIL/TIL Avg | VBM CIL/TIL Avg | ΔAvg |
|------|:-:|:-:|:-:|
| LwF (buffer=0) | -/62.0=62.0 | -/77.5=77.5 | **+15.6** |
| ER | 50.3/91.7=71.0 | 52.6/93.6=73.1 | +2.1 |
| iCaRL | 64.1/90.2=77.2 | 69.7/92.8=81.3 | **+4.1** |
| DER++ | 61.7/90.6=76.1 | 67.0/94.3=80.7 | **+4.5** |

### 消融实验：遗忘曲线验证

Fig.4实证验证了遗忘曲线理论在神经网络中的适用性：
- V=1（短回忆间隔）：memory retention decay陡峭，遗忘严重
- V=4（最优回忆间隔）：decay最缓，长期记忆保持最好
- V=16（过长回忆间隔）：虽然decay温和，但初始遗忘太多，整体性能下降

### 关键发现

- **全面一致提升**：Fig.2展示了在不同step size、buffer size、baseline方法、预训练模型、benchmark、protocol上VBM都有正向贡献，无negative case
- **对rehearsal-free方法提升最大**：LwF +15.6%（无buffer，回忆间隔优化收益最大）
- **与预训练模型兼容**：在使用预训练ViT的方法上也有效（CODA-Prompt等）
- **Ebbinghaus理论在神经网络中成立**：Fig.4的遗忘曲线形状与人类心理学研究一致

## 亮点与洞察

- **理论与实践的优雅结合**：将120多年前的心理学理论（Ebbinghaus遗忘曲线/间隔效应）成功迁移到神经网络持续学习，实证验证了其在deep learning中的适用性
- **极简但有效**：不需要新架构、新优化器、新损失函数设计——仅通过调整数据schedule和加一个KL散度loss
- **"Do your best AND get enough rest"的直觉**：学生（模型）应该每次学习时尽可能深入（SSL），但也需要足够的间隔来巩固记忆——这个类比非常贴切

## 局限与展望

- V的最优值需要根据数据集和任务调整，缺乏理论指导
- 自监督损失使用KL散度较简单，可探索更复杂的一致性约束
- 回忆间隔的"最优"定义是经验的，缺乏在神经网络上的形式化分析
- 仅验证分类任务，未扩展到检测/分割等持续学习场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 遗忘曲线理论在CL中的应用新颖，view-batch replay简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ Fig.2涵盖6个维度（step/buffer/method/pretrain/benchmark/protocol），验证极其全面
- 写作质量: ⭐⭐⭐⭐ 理论动机清晰，Fig.1遗忘曲线可视化直观
- 价值: ⭐⭐⭐⭐⭐ Drop-in替代、零额外开销、一致提升——CL从业者可立即受益

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)
- [\[CVPR 2026\] How Much 3D Do Video Foundation Models Encode?](../../CVPR2026/self_supervised/how_much_3d_do_video_foundation_models_encode.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](../../ICML2025/self_supervised/update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [\[NeurIPS 2025\] Continuous Subspace Optimization for Continual Learning (CoSO)](../../NeurIPS2025/self_supervised/continuous_subspace_optimization_for_continual_learning.md)

</div>

<!-- RELATED:END -->
